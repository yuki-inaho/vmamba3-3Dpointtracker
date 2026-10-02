"""Strict cache identity, provenance and disk-budget tests."""

import pytest
import torch


def context():
    return dict(
        teacher_sha256="a" * 64,
        split="train",
        window_start=0,
        window_length=8,
        track_ids=[1, 2],
        batch_order=["clip1"],
        precision="fp32",
        frontend_provenance="fixed",
        query_anchors=[0, 0],
        schema=1,
    )


def test_key_changes_with_window_tracks_zref():
    from mamba3_tracker.data.distillation_cache import cache_key

    c, x = context(), {"z_ref": torch.tensor(2.0), "mask": torch.ones(2)}
    key = cache_key(c, x)
    assert key != cache_key({**c, "window_start": 1}, x)
    assert key != cache_key({**c, "track_ids": [2, 1]}, x)
    assert key != cache_key(c, {**x, "z_ref": torch.tensor(2.00001)})


def test_batch_context_invalidates_key():
    from mamba3_tracker.data.distillation_cache import cache_key

    x = {"z_ref": torch.tensor(2.0), "mask": torch.ones(2)}
    assert cache_key(context(), x) != cache_key(
        {**context(), "batch_order": ["clip2"]}, x
    )


def test_wrong_split_rejected(tmp_path):
    from mamba3_tracker.data.distillation_cache import DistillationCache

    cache = DistillationCache(tmp_path, max_bytes=100000, reserve_bytes=0)
    with pytest.raises(ValueError, match="train"):
        cache.put(
            {**context(), "split": "minival"},
            {"z_ref": torch.tensor(2.0)},
            {"xyz": torch.ones(1)},
        )


def test_corrupt_hash_rejected(tmp_path):
    from mamba3_tracker.data.distillation_cache import DistillationCache

    cache = DistillationCache(tmp_path, max_bytes=100000, reserve_bytes=0)
    x, label = {"z_ref": torch.tensor(2.0)}, {"xyz": torch.ones(1)}
    key = cache.put(context(), x, label)
    torch.testing.assert_close(cache.get(context(), x)["xyz"], label["xyz"])
    (tmp_path / (key + ".pt")).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        cache.get(context(), x)


def test_budget_prevents_write(tmp_path):
    from mamba3_tracker.data.distillation_cache import DistillationCache

    cache = DistillationCache(tmp_path, max_bytes=1, reserve_bytes=0)
    with pytest.raises(ValueError, match="budget"):
        cache.put(context(), {"z_ref": torch.tensor(2.0)}, {"xyz": torch.ones(1)})
    assert not list(tmp_path.iterdir())


def test_prepare_cli_rejects_test_as_train():
    from scripts.prepare_refiner_kd import parser

    with pytest.raises(SystemExit):
        parser().parse_args(
            [
                "--config",
                "unused",
                "--mode",
                "pilot",
                "--partition",
                "minival",
                "--out-dir",
                "unused",
            ]
        )


def test_long_preparation_requires_explicit_new_frontend(tmp_path):
    from argparse import Namespace
    from scripts.prepare_refiner_kd import validate_preparation_mode

    args = Namespace(
        window=16, compute_frontend=False, new_input_root=None, mode="cache"
    )
    with pytest.raises(ValueError, match="explicit"):
        validate_preparation_mode(args)
    args.compute_frontend = True
    args.new_input_root = tmp_path
    validate_preparation_mode(args)
    args.new_input_root = tmp_path / "tapvid3d_frozen_cache"
    with pytest.raises(ValueError, match="existing"):
        validate_preparation_mode(args)


def test_selected_config_rejects_changed_base_or_nonbest(tmp_path, monkeypatch):
    import hashlib
    import json
    import yaml
    from scripts import prepare_refiner_kd as prepare

    base = yaml.safe_load(open("configs/distill_refiner_vssd.yaml"))
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base))
    pilot_path = tmp_path / "pilots.json"
    pilot_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "completed": [
                    {"arm": f"A{i}", "best_monitor_gt_loss": i / 10}
                    for i in range(1, 5)
                ],
            }
        )
    )
    actual = prepare.sha256
    # External teacher/split/protocol reads are out of scope of this unit test.
    fixed = {
        str(base["teacher"]["checkpoint"]): base["teacher"]["sha256"],
        str(base["data"]["split"]): base["data"]["split_sha256"],
        str(base["protocol"]): base["protocol_sha256"],
    }
    monkeypatch.setattr(
        prepare, "sha256", lambda p: fixed[str(p)] if str(p) in fixed else actual(p)
    )
    cfg = {
        **base,
        "selected_ablation": "A1",
        "selection": {
            "base_config": str(base_path),
            "base_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
            "pilot_report": str(pilot_path),
            "pilot_sha256": actual(pilot_path),
        },
    }
    selected = tmp_path / "selected.yaml"
    selected.write_text(yaml.safe_dump(cfg))
    assert prepare.read_config(selected)["selected_ablation"] == "A1"
    cfg["selected_ablation"] = "A4"
    selected.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="monitor-only"):
        prepare.read_config(selected)
    cfg["selected_ablation"] = "A1"
    cfg["data"] = {**cfg["data"], "window": 32}
    selected.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="may not change"):
        prepare.read_config(selected)
