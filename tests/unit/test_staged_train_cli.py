"""Fail-closed staged trainer protocol and phase-aware optimizer resumption."""

from argparse import Namespace
import copy
import json
from pathlib import Path
import random

import pytest
import torch
from torch import nn


@pytest.mark.parametrize(
    "mode,requested,bad,expected",
    [
        ("pilot", True, 4, False),
        ("pilot", True, 5, True),
        ("pilot", True, 6, True),
        ("pilot", False, 5, False),
        ("full", False, 5, True),
        ("full", True, 4, False),
        ("long", True, 5, True),
        ("smoke", False, 5, False),
    ],
)
def test_explicit_early_stopping_supports_additional_training(
    mode, requested, bad, expected
):
    from scripts.train_refiner_staged import early_stopping_reached

    assert early_stopping_reached(mode, requested, bad, 5) is expected


def test_early_stopping_cli_flag_is_opt_in():
    from scripts.train_refiner_staged import parser

    argv = [
        "--config",
        "cfg",
        "--mode",
        "pilot",
        "--method",
        "staged",
        "--input-manifest",
        "train",
        "--monitor-manifest",
        "monitor",
        "--out-dir",
        "out",
    ]
    assert parser().parse_args(argv).early_stop is False
    assert parser().parse_args([*argv, "--early-stop"]).early_stop is True


@pytest.mark.parametrize("best", [float("inf"), 0.12])
def test_status_progress_reports_actual_phase_counters(best):
    from scripts.train_refiner_staged import status_progress

    progress = dict(
        phase="finetune",
        block_step=10,
        ft_step=20,
        block_processed=320,
        ft_processed=640,
        bad=2,
        best=best,
        best_checkpoint="student.pt",
        snapshots=[],
        alignment_evidence={"common_unchanged": True},
    )
    row = status_progress(progress)
    assert row["ft_processed"] == row["processed_clips"] == 640
    assert row["ft_step"] == row["step"] == 20
    assert row["bad"] == 2 and row["block_processed"] == 320
    assert row["best_monitor_gt_loss"] == (None if best == float("inf") else best)
    json.dumps(row, allow_nan=False)


def test_cli_requires_method_and_explicit_manifests():
    from scripts.train_refiner_staged import parser

    with pytest.raises(SystemExit):
        parser().parse_args(["--config", "cfg", "--mode", "smoke", "--out-dir", "out"])
    args = parser().parse_args(
        [
            "--config",
            "cfg",
            "--mode",
            "smoke",
            "--method",
            "staged",
            "--input-manifest",
            "train",
            "--monitor-manifest",
            "monitor",
            "--out-dir",
            "out",
        ]
    )
    assert args.method == "staged" and args.seed == 42
    with pytest.raises(SystemExit):
        parser().parse_args(
            [
                "--config",
                "cfg",
                "--mode",
                "smoke",
                "--method",
                "gt",
                "--architecture",
                "vssd_local_global_128x2_v2",
                "--input-manifest",
                "train",
                "--monitor-manifest",
                "monitor",
                "--out-dir",
                "out",
            ]
        )


def test_long_initialization_cannot_be_implicit():
    from scripts.train_refiner_staged import validate_options

    args = Namespace(
        mode="long",
        initialize_from=None,
        initialize_sha256=None,
        resume=None,
        window=None,
    )
    with pytest.raises(ValueError, match="pinned"):
        validate_options(args)
    args.initialize_from, args.initialize_sha256 = Path("student.pt"), "a" * 64
    validate_options(args)
    args.mode = "pilot"
    with pytest.raises(ValueError, match="long"):
        validate_options(args)


def manifests(tmp_path, count=2):
    split = {
        "train": [f"raw/adt/train_{i}.npz" for i in range(count)],
        "validation": ["raw/adt/monitor.npz"],
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    cfg = dict(
        schema_version=2,
        data=dict(split=str(split_path), split_sha256="split", window=16),
        improvement=dict(
            query_policy="strict_visible_gt_reanchor_v1", frontend_precision="fp32"
        ),
    )
    base = dict(
        status="complete",
        config_sha256="cfg",
        split_sha256="split",
        window=16,
        query_policy="strict_visible_gt_reanchor_v1",
        frontend_precision="fp32",
        query_seed=42,
        window_seed=42,
        preprocessing_sha256="a" * 64,
    )
    train = base | dict(
        partition="train", records=[dict(clip=p) for p in split["train"]]
    )
    monitor = base | dict(
        partition="validation", records=[dict(clip=p) for p in split["validation"]]
    )
    return cfg, train, monitor


def test_smoke_membership_is_subset_but_full_requires_exact_split(tmp_path):
    from scripts.train_refiner_staged import validate_manifests

    cfg, train, monitor = manifests(tmp_path)
    validate_manifests(cfg, train, monitor, "cfg", "smoke")
    with pytest.raises(ValueError, match="1027|15|exact"):
        validate_manifests(cfg, train, monitor, "cfg", "pilot")
    train["records"].append(dict(clip="raw/adt/not-in-split.npz"))
    with pytest.raises(ValueError, match="membership|split"):
        validate_manifests(cfg, train, monitor, "cfg", "smoke")


@pytest.mark.parametrize(
    "field,value",
    [
        ("query_policy", "legacy_anchor_filter_v1"),
        ("frontend_precision", "bf16"),
        ("window", 8),
        ("config_sha256", "wrong"),
        ("split_sha256", "wrong"),
        ("status", "running"),
    ],
)
def test_input_protocol_changes_are_rejected(tmp_path, field, value):
    from scripts.train_refiner_staged import validate_manifests

    cfg, train, monitor = manifests(tmp_path)
    train[field] = value
    with pytest.raises(ValueError, match="provenance|manifest|window|precision|policy"):
        validate_manifests(cfg, train, monitor, "cfg", "smoke")


def test_duplicate_clips_are_rejected(tmp_path):
    from scripts.train_refiner_staged import validate_manifests

    cfg, train, monitor = manifests(tmp_path)
    train["records"].append(copy.deepcopy(train["records"][0]))
    with pytest.raises(ValueError, match="duplicate|membership"):
        validate_manifests(cfg, train, monitor, "cfg", "smoke")


def test_full_split_accepts_every_required_clip_and_rejects_a_missing_monitor(tmp_path):
    from scripts.train_refiner_staged import validate_manifests

    cfg, train, monitor = manifests(tmp_path, count=1027)
    values = json.loads(Path(cfg["data"]["split"]).read_text())
    values["validation"] = [f"raw/adt/monitor_{i}.npz" for i in range(15)]
    Path(cfg["data"]["split"]).write_text(json.dumps(values))
    monitor["records"] = [dict(clip=p) for p in values["validation"]]
    validate_manifests(cfg, train, monitor, "cfg", "pilot")
    monitor["records"].pop()
    with pytest.raises(ValueError, match="exact"):
        validate_manifests(cfg, train, monitor, "cfg", "pilot")


def test_stage_budget_and_sampling_are_comparable():
    from scripts.train_refiner_staged import phase_budgets, sampling_rng

    cfg = dict(
        training=dict(
            smoke_steps=20,
            overfit_steps=32,
            pilot_steps=200,
            full_steps=800,
            long_steps=200,
        ),
        improvement=dict(block_steps=100),
    )
    assert phase_budgets(cfg, "smoke", "staged") == (10, 20)
    assert phase_budgets(cfg, "pilot", "staged") == (100, 200)
    assert phase_budgets(cfg, "pilot", "gt") == (0, 200)
    assert phase_budgets(cfg, "long", "staged") == (0, 200)
    assert sampling_rng(42, "finetune").getstate() == random.Random(42).getstate()
    assert sampling_rng(42, "block").getstate() == random.Random(42 + 187).getstate()


class ToyWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.student = nn.Linear(2, 1)
        self.teacher = nn.Linear(2, 1).requires_grad_(False)


def checkpoint_fixture():
    from scripts.train_refiner_staged import make_resume_state

    torch.manual_seed(1)
    wrapper = ToyWrapper()
    optimizer = torch.optim.AdamW(wrapper.student.parameters(), lr=0.01)
    wrapper.student(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    progress = dict(
        phase="finetune",
        block_step=10,
        ft_step=10,
        block_processed=320,
        ft_processed=320,
        best=0.2,
        patience_best=0.2,
        bad=0,
        best_checkpoint="public.pt",
        snapshots=[],
        alignment_evidence=dict(
            common_initial_sha256="a" * 64,
            common_unchanged=True,
            block_peak_vram_bytes=0,
        ),
    )
    identity = {
        "method": "staged",
        "architecture": "train",
        "source_sha256": "source",
        "effective_batch": 32,
        "processed_clip_limit": 20000,
    }
    rng = sampling_state = random.Random(42)
    state = make_resume_state(
        wrapper,
        optimizer,
        identity,
        progress,
        rng,
        {k: True for k, _ in wrapper.student.named_parameters()},
        cuda_rng=[],
    )
    assert sampling_state is rng
    return wrapper, optimizer, identity, state


def test_resume_is_phase_and_source_strict():
    from scripts.train_refiner_staged import validate_resume_state

    _, _, identity, state = checkpoint_fixture()
    validate_resume_state(state, identity, 10, 20)
    with pytest.raises(ValueError, match="identity"):
        validate_resume_state(state, identity | {"source_sha256": "changed"}, 10, 20)
    with pytest.raises(ValueError, match="phase|block"):
        validate_resume_state(state | {"block_step": 0}, identity, 10, 20)
    with pytest.raises(ValueError, match="schema|training"):
        validate_resume_state(
            dict(schema_version=1, architecture="train", model={}), identity, 10, 20
        )


def test_optimizer_resume_restores_student_and_omits_exact_teacher():
    from scripts.train_refiner_staged import restore_trainable_state

    wrapper, optimizer, _, state = checkpoint_fixture()
    fresh = ToyWrapper()
    teacher_before = {k: v.clone() for k, v in fresh.teacher.state_dict().items()}
    fresh_optimizer = torch.optim.AdamW(fresh.student.parameters(), lr=0.7)
    restore_trainable_state(fresh, fresh_optimizer, state)
    assert all(
        torch.equal(v, fresh.student.state_dict()[k])
        for k, v in wrapper.student.state_dict().items()
    )
    assert all(
        torch.equal(v, fresh.teacher.state_dict()[k]) for k, v in teacher_before.items()
    )
    assert fresh_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    corrupted = copy.deepcopy(state)
    del corrupted["trainable_state"]["student.bias"]
    with pytest.raises(ValueError, match="exactly|teacher"):
        restore_trainable_state(fresh, fresh_optimizer, corrupted)


def test_cpu_resumption_matches_uninterrupted_next_optimizer_step():
    from scripts.train_refiner_staged import restore_trainable_state

    wrapper, optimizer, _, state = checkpoint_fixture()
    fresh = ToyWrapper()
    fresh_optimizer = torch.optim.AdamW(fresh.student.parameters(), lr=0.7)
    restore_trainable_state(fresh, fresh_optimizer, state)
    for model, opt in ((wrapper, optimizer), (fresh, fresh_optimizer)):
        opt.zero_grad()
        model.student(torch.tensor([[0.2, 0.7]])).square().sum().backward()
        opt.step()
    assert all(
        torch.equal(v, fresh.student.state_dict()[k])
        for k, v in wrapper.student.state_dict().items()
    )


def test_block_resume_cannot_claim_finetune_progress():
    from scripts.train_refiner_staged import validate_resume_state

    _, _, identity, state = checkpoint_fixture()
    block = state | dict(
        phase="block",
        block_step=5,
        block_processed=160,
        ft_step=0,
        ft_processed=0,
        alignment_evidence=state["alignment_evidence"]
        | dict(common_unchanged=None, block_peak_vram_bytes=None),
    )
    validate_resume_state(block, identity, 10, 20)
    with pytest.raises(ValueError, match="phase"):
        validate_resume_state(block | {"ft_step": 1}, identity, 10, 20)


def test_gt_and_nonstaged_resume_have_no_alignment_phase():
    from scripts.train_refiner_staged import validate_resume_state

    _, _, identity, state = checkpoint_fixture()
    no_block = state | dict(
        block_step=0,
        block_processed=0,
        alignment_evidence=state["alignment_evidence"]
        | dict(common_unchanged=None, block_peak_vram_bytes=None),
    )
    validate_resume_state(no_block, identity, 0, 20)
    with pytest.raises(ValueError, match="phase"):
        validate_resume_state(
            no_block | dict(phase="block", ft_step=0), identity, 0, 20
        )


def strict_input_fixture(tmp_path):
    from mamba3_tracker.deployment.checkpoint import sha256

    raw = tmp_path / "raw" / "adt" / "clip.npz"
    depth = tmp_path / "depth" / "adt" / "clip.npz"
    raw.parent.mkdir(parents=True)
    depth.parent.mkdir(parents=True)
    raw.write_bytes(b"raw fixture")
    depth.write_bytes(b"depth fixture")
    teacher_cfg = tmp_path / "teacher.json"
    teacher_cfg.write_text(
        json.dumps(dict(data=dict(da3_depth_root=str(tmp_path / "depth"))))
    )
    manifest = dict(
        config_sha256="cfg",
        query_policy="strict_visible_gt_reanchor_v1",
        frontend_precision="fp32",
        preprocessing_sha256="p" * 64,
        query_seed=42,
        window_seed=42,
        window=16,
    )
    xyz = torch.tensor([[[[0.0, 0.0, 1.0]]]])
    query = torch.tensor([[[10.0, 5.0, 0.0]]])
    visible = torch.ones(1, 1, 1, dtype=torch.bool)
    K = torch.tensor([[[10.0, 0, 10.0], [0, 10.0, 5.0], [0, 0, 1.0]]])
    payload = {k: v for k, v in manifest.items() if k != "window"} | dict(
        clip=str(raw),
        requested_window=16,
        raw_sha256=sha256(raw),
        depth_sha256=sha256(depth),
        queries_xyt=query,
        anchor_audit=dict(anchor_valid_count=1, anchor_reprojection_max_px=0.0),
        inputs=dict(
            ray=torch.zeros(1, 1, 1, 2),
            z_raw=torch.ones(1, 1, 1),
            visibility=visible.float(),
            uv=query[..., :2].unsqueeze(1),
            depth_map=torch.ones(1, 1, 5, 5),
            dino_features=torch.zeros(1, 1, 384, 28, 28),
            intrinsics=K,
        ),
        target=dict(xyz=xyz, visible=visible, valid=visible, scale=torch.ones(1)),
    )
    artifact = tmp_path / "input.pt"
    torch.save(payload, artifact)
    record = dict(
        path=str(artifact),
        sha256=sha256(artifact),
        bytes=artifact.stat().st_size,
        clip=str(raw),
        frames=1,
        tracks=1,
        depth_hw=[5, 5],
        anchor_valid_count=1,
        anchor_reprojection_max_px=0.0,
    )
    cfg = dict(teacher=dict(run_config=str(teacher_cfg)))
    return cfg, manifest, record, payload, raw, depth


def test_strict_inputs_verify_anchor_and_raw_depth_identity(tmp_path):
    from scripts.train_refiner_staged import StrictInputs

    cfg, manifest, record, payload, _, _ = strict_input_fixture(tmp_path)
    inputs = StrictInputs([record], manifest, cfg)
    torch.testing.assert_close(inputs.get(0)["queries_xyt"], payload["queries_xyt"])
    assert len(inputs.verified_sources) == 2


@pytest.mark.parametrize("source", ["raw", "depth"])
def test_strict_inputs_refuse_changed_raw_or_depth(tmp_path, source):
    from scripts.train_refiner_staged import StrictInputs

    cfg, manifest, record, _, raw, depth = strict_input_fixture(tmp_path)
    (raw if source == "raw" else depth).write_bytes(b"changed source")
    with pytest.raises(ValueError, match="source hash"):
        StrictInputs([record], manifest, cfg).get(0)


def test_phase_resume_requires_preserved_common_freeze_evidence():
    from scripts.train_refiner_staged import validate_resume_state

    _, _, identity, state = checkpoint_fixture()
    bad = state | {
        "alignment_evidence": state["alignment_evidence"] | {"common_unchanged": None}
    }
    with pytest.raises(ValueError, match="evidence"):
        validate_resume_state(bad, identity, 10, 20)


@pytest.mark.parametrize(
    "changes",
    [
        {"block_processed": 319},
        {"ft_processed": 319},
        {"ft_step": 11},
        {"bad": 11},
    ],
)
def test_resume_processed_counts_match_optimizer_steps(changes):
    from scripts.train_refiner_staged import validate_resume_state

    _, _, identity, state = checkpoint_fixture()
    with pytest.raises(ValueError, match="progress|count|processed"):
        validate_resume_state(state | changes, identity, 10, 20)


@pytest.mark.parametrize(
    "key,value",
    [
        ("effective_batch", None),
        ("effective_batch", True),
        ("effective_batch", 0),
        ("processed_clip_limit", None),
        ("processed_clip_limit", True),
        ("processed_clip_limit", -1),
    ],
)
def test_resume_requires_explicit_positive_integer_budgets(key, value):
    from scripts.train_refiner_staged import validate_resume_state

    _, _, identity, state = checkpoint_fixture()
    identity = identity | {key: value}
    state = state | {"identity": identity}
    with pytest.raises(ValueError, match="budget|batch|limit"):
        validate_resume_state(state, identity, 10, 20)


def test_resume_partial_final_batch_is_valid_but_phantom_steps_are_not():
    from scripts.train_refiner_staged import validate_resume_state

    _, _, identity, state = checkpoint_fixture()
    identity = identity | {"processed_clip_limit": 321}
    state = state | {"identity": identity, "ft_step": 11, "ft_processed": 321}
    validate_resume_state(state, identity, 10, 20)
    with pytest.raises(ValueError, match="progress|count|processed|limit"):
        validate_resume_state(state | {"ft_step": 12}, identity, 10, 20)
    with pytest.raises(ValueError, match="progress|count|processed|limit"):
        validate_resume_state(state | {"ft_processed": 322}, identity, 10, 20)


@pytest.mark.parametrize("changes", [{"ft_processed": 32}, {"bad": 1}])
def test_block_phase_cannot_claim_finetune_clips_or_monitor_patience(changes):
    from scripts.train_refiner_staged import validate_resume_state

    _, _, identity, state = checkpoint_fixture()
    state = state | dict(
        phase="block",
        block_step=5,
        block_processed=160,
        ft_step=0,
        ft_processed=0,
        alignment_evidence=state["alignment_evidence"]
        | dict(common_unchanged=None, block_peak_vram_bytes=None),
    )
    validate_resume_state(state, identity, 10, 20)
    with pytest.raises(ValueError, match="phase|progress|count|processed"):
        validate_resume_state(state | changes, identity, 10, 20)


@pytest.mark.parametrize(
    "changes",
    [
        {"common_initial_sha256": "z" * 64},
        {"block_peak_vram_bytes": None},
        {"block_peak_vram_bytes": -1},
        {"block_peak_vram_bytes": True},
    ],
)
def test_finetune_alignment_proof_has_valid_hash_and_peak(changes):
    from scripts.train_refiner_staged import validate_resume_state

    _, _, identity, state = checkpoint_fixture()
    state = state | {"alignment_evidence": state["alignment_evidence"] | changes}
    with pytest.raises(ValueError, match="evidence"):
        validate_resume_state(state, identity, 10, 20)


def test_runtime_source_hashes_cover_imported_native_kernels_and_lockfile():
    from mamba3_tracker.deployment.checkpoint import sha256
    from scripts.train_refiner_staged import ROOT, runtime_source_paths, source_hashes

    modules = runtime_source_paths()
    expected_modules = {
        "mamba3_tracker.model.official_mamba3",
        "visionmamba3",
        "visionmamba3.projections",
        "visionmamba3.mask",
        "visionmamba3.cross_attention",
        "mamba_ssm.modules.mamba3",
        "mamba_ssm.ops.triton.layernorm_gated",
        "mamba_ssm.ops.triton.mamba3.mamba3_siso_combined",
        "mamba_ssm.ops.triton.mamba3.mamba3_siso_fwd",
        "mamba_ssm.ops.triton.mamba3.mamba3_siso_bwd",
        "mamba_ssm.ops.triton.mamba3.angle_dt",
        "mamba_ssm.ops.triton.mamba3.utils",
    }
    assert set(modules) == expected_modules
    hashes = source_hashes()
    assert hashes["uv.lock"] == sha256(ROOT / "uv.lock")
    for path in modules.values():
        assert path.is_file() and path.suffix == ".py"
        assert hashes[str(path.relative_to(ROOT))] == sha256(path)


def test_runtime_source_resolution_rejects_nonrepository_import(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from scripts import train_refiner_staged as module

    external_source = tmp_path / "external.py"
    external_source.write_text("# synthetic external runtime source\n")
    original = module.importlib.import_module
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(__file__=str(external_source))
            if name == "mamba_ssm.modules.mamba3"
            else original(name)
        ),
    )
    with pytest.raises(ValueError, match="source|repository"):
        module.runtime_source_paths()


def test_native_kernel_change_changes_resume_identity(monkeypatch):
    from scripts import train_refiner_staged as module

    before = module.source_hashes()
    kernel = "third_party/visionMamba3/third_party/mamba-ssm/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py"
    assert kernel in before
    original = module.sha256
    monkeypatch.setattr(
        module,
        "sha256",
        lambda path: "b" * 64 if Path(path) == module.ROOT / kernel else original(path),
    )
    after = module.source_hashes()
    assert before[kernel] != after[kernel]
    assert all(before[key] == value for key, value in after.items() if key != kernel)
    _, _, identity, state = checkpoint_fixture()
    identity = identity | {"source_sha256": before}
    state = state | {"identity": identity}
    with pytest.raises(ValueError, match="identity"):
        module.validate_resume_state(state, identity | {"source_sha256": after}, 10, 20)
