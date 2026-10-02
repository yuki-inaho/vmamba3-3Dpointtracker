"""Strict GT reanchoring and independently identifiable preparation inputs."""

from argparse import Namespace
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml


def patch_clip(monkeypatch, xyz, visible):
    from mamba3_tracker.data import dataset as module

    frames, tracks = xyz.shape[:2]
    clip = SimpleNamespace(
        images=torch.zeros(frames, 3, 10, 20),
        tracks_XYZ=xyz,
        visibility=visible,
        queries_xyt=torch.tensor([[19.0, 9.0, 50.0]]).repeat(tracks, 1),
        K=torch.tensor([[10.0, 0, 10.0], [0, 10.0, 5.0], [0, 0, 1.0]]),
        H=10,
        W=20,
        N_q=tracks,
        clip_id="fixture",
        subset="adt",
    )
    monkeypatch.setattr(module, "peek_clip_F", lambda _: frames)
    monkeypatch.setattr(module, "load_clip", lambda *_, **__: clip)
    return module.TAPVid3DDataset


def strict_item(monkeypatch, xyz, visible):
    dataset = patch_clip(monkeypatch, xyz, visible)
    return dataset(
        [Path("adt/fixture.npz")],
        window_size=None,
        image_size=100,
        reanchor_window=True,
        strict_reanchor_window=True,
    )[0]


def test_reanchor_uses_first_valid_gt_frame_and_scales_xy(monkeypatch):
    xyz = torch.tensor(
        [
            [[float("nan"), 0, 1], [3.0, 0, 1], [0, 0, -1]],
            [[0.2, 0.1, 1], [0.0, 0, 1], [0, 0, 1]],
            [[0.4, 0.1, 1], [0.2, 0, 1], [0, 0, 1]],
        ]
    )
    visible = torch.tensor(
        [[True, True, True], [True, False, False], [True, True, False]]
    )
    item = strict_item(monkeypatch, xyz, visible)
    torch.testing.assert_close(item["query_idx"], torch.tensor([0, 1]))
    # Original (12,6) at frame1 and (12,5) at frame2, scaled x5/y10.
    torch.testing.assert_close(
        item["queries_xyt"], torch.tensor([[60.0, 60.0, 1.0], [60.0, 50.0, 2.0]])
    )
    torch.testing.assert_close(item["tracks_XYZ"], xyz[:, :2], equal_nan=True)
    assert item["visibility"][1, 0] and item["visibility"][2, 1]


@pytest.mark.parametrize(
    "invalid", ["invisible", "nonfinite", "negative", "zero", "bounds"]
)
def test_no_valid_anchor_is_explicit_error(monkeypatch, invalid):
    xyz = torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]])
    visible = torch.ones(2, 1, dtype=torch.bool)
    if invalid == "invisible":
        visible.zero_()
    elif invalid == "nonfinite":
        xyz[..., 0] = float("inf")
    elif invalid == "negative":
        xyz[..., 2] = -1
    elif invalid == "zero":
        xyz[..., 2] = 1e-6
    else:
        xyz[..., 0] = 2  # projected x30 outside original width20.
    with pytest.raises(ValueError, match="valid.*anchor"):
        strict_item(monkeypatch, xyz, visible)


def test_empty_fixed_window_selects_deterministic_valid_window_within_clip():
    from mamba3_tracker.data.dataset import choose_anchor_valid_window

    xyz = torch.tensor(
        [
            [[0.0, 0.0, -1.0]],
            [[0.0, 0.0, -1.0]],
            [[0.0, 0.0, 1.0]],
            [[0.1, 0.0, 1.0]],
            [[0.2, 0.0, 1.0]],
        ]
    )
    visible = torch.ones(5, 1, dtype=torch.bool)
    K = torch.tensor([[10.0, 0, 10.0], [0, 10.0, 5.0], [0, 0, 1.0]])
    first = choose_anchor_valid_window(xyz, visible, K, 10, 20, 2, 0)
    again = choose_anchor_valid_window(xyz, visible, K, 10, 20, 2, 0)
    assert first == again == 1


def test_valid_fixed_window_is_never_changed():
    from mamba3_tracker.data.dataset import choose_anchor_valid_window

    xyz = torch.tensor([[[0.0, 0.0, 1.0]], [[0.1, 0.0, 1.0]]])
    visible = torch.ones(2, 1, dtype=torch.bool)
    K = torch.tensor([[10.0, 0, 10.0], [0, 10.0, 5.0], [0, 0, 1.0]])
    assert choose_anchor_valid_window(xyz, visible, K, 10, 20, 2, 0) == 0


def test_window_search_still_fails_if_clip_has_no_valid_anchor():
    from mamba3_tracker.data.dataset import choose_anchor_valid_window

    xyz = torch.zeros(4, 1, 3)
    visible = torch.zeros(4, 1, dtype=torch.bool)
    K = torch.tensor([[10.0, 0, 10.0], [0, 10.0, 5.0], [0, 0, 1.0]])
    with pytest.raises(ValueError, match="any valid visible GT anchor"):
        choose_anchor_valid_window(xyz, visible, K, 10, 20, 2, 1)


def test_dataset_reanchors_in_same_clip_after_empty_fixed_window(monkeypatch):
    from mamba3_tracker.data import dataset as module

    xyz = torch.tensor(
        [
            [[0.0, 0.0, -1.0]],
            [[0.0, 0.0, -1.0]],
            [[0.0, 0.0, 1.0]],
            [[0.1, 0.0, 1.0]],
            [[0.2, 0.0, 1.0]],
        ]
    )
    visible = torch.ones(5, 1, dtype=torch.bool)
    queries = torch.tensor([[10.0, 5.0, 0.0]])
    intrinsics = torch.tensor([[10.0, 0, 10.0], [0, 10.0, 5.0], [0, 0, 1.0]])
    loaded_windows = []

    def load_window(path, frames):
        start, end = frames
        loaded_windows.append(frames)
        return SimpleNamespace(
            images=torch.zeros(end - start, 3, 10, 20),
            tracks_XYZ=xyz[start:end],
            visibility=visible[start:end],
            queries_xyt=queries,
            K=intrinsics,
            H=10,
            W=20,
            N_q=1,
            clip_id="fixture",
            subset="adt",
        )

    class AlwaysFirstWindow:
        def __init__(self, *_args):
            pass

        @staticmethod
        def randint(start, _end):
            return start

    monkeypatch.setattr(module.random, "Random", AlwaysFirstWindow)
    monkeypatch.setattr(module, "peek_clip_F", lambda _path: 5)
    monkeypatch.setattr(module, "load_clip", load_window)
    monkeypatch.setattr(module, "load_tracking_labels", lambda _path: (xyz, visible))
    item = module.TAPVid3DDataset(
        [Path("adt/fixture.npz")],
        window_size=2,
        image_size=20,
        reanchor_window=True,
        strict_reanchor_window=True,
    )[0]
    assert loaded_windows == [(0, 2), (1, 3)]
    assert item["frame_start"] == 1
    assert item["window_fallback"] is True
    torch.testing.assert_close(item["query_idx"], torch.tensor([0]))
    torch.testing.assert_close(item["tracks_XYZ"], xyz[1:3])
    torch.testing.assert_close(item["queries_xyt"], torch.tensor([[10.0, 10.0, 1.0]]))


def test_strict_reanchor_requires_reanchor_opt_in():
    from mamba3_tracker.data.dataset import TAPVid3DDataset

    with pytest.raises(ValueError, match="reanchor_window"):
        TAPVid3DDataset([], reanchor_window=False, strict_reanchor_window=True)


def test_legacy_fallback_keeps_its_existing_coordinates(monkeypatch):
    xyz = torch.ones(2, 1, 3)
    dataset = patch_clip(monkeypatch, xyz, torch.ones(2, 1, dtype=torch.bool))
    item = dataset(
        [Path("adt/fixture.npz")],
        window_size=None,
        image_size=100,
        reanchor_window=False,
    )[0]
    torch.testing.assert_close(item["queries_xyt"], torch.tensor([[95.0, 90.0, 0.0]]))


def test_strict_reanchor_time_is_relative_to_nonzero_window(monkeypatch):
    xyz = torch.tensor([[[0.0, 0.0, 1.0]], [[0.2, 0.1, 1.0]]])
    dataset = patch_clip(monkeypatch, xyz, torch.tensor([[False], [True]]))
    from mamba3_tracker.data import dataset as module

    monkeypatch.setattr(module, "peek_clip_F", lambda _: 10)
    item = dataset(
        [Path("adt/fixture.npz")],
        window_size=2,
        seed=0,
        image_size=100,
        reanchor_window=True,
        strict_reanchor_window=True,
    )[0]
    assert item["frame_start"] == 6
    torch.testing.assert_close(item["queries_xyt"], torch.tensor([[60.0, 60.0, 1.0]]))


def test_strict_bounds_exclude_right_bottom_edges(monkeypatch):
    xyz = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 0.5, 1.0], [-1.0, -0.5, 1.0]]])
    item = strict_item(monkeypatch, xyz, torch.ones(1, 3, dtype=torch.bool))
    torch.testing.assert_close(item["query_idx"], torch.tensor([2]))
    torch.testing.assert_close(item["queries_xyt"], torch.tensor([[0.0, 0.0, 0.0]]))


def test_query_subsampling_preserves_original_column_ids(monkeypatch):
    xyz = torch.zeros(1, 4, 3)
    xyz[..., 2] = 1
    dataset = patch_clip(monkeypatch, xyz, torch.ones(1, 4, dtype=torch.bool))
    item = dataset(
        [Path("adt/fixture.npz")],
        window_size=None,
        seed=0,
        max_queries=2,
        image_size=100,
        reanchor_window=True,
        strict_reanchor_window=True,
    )[0]
    torch.testing.assert_close(item["query_idx"], torch.tensor([1, 3]))


def preparation_args(tmp_path, **updates):
    values = dict(
        window=8,
        compute_frontend=True,
        new_input_root=tmp_path,
        mode="cache",
        reanchor_window=True,
    )
    return Namespace(**(values | updates))


def test_new_window8_reanchor_requires_explicit_compute_namespace(tmp_path):
    from scripts.prepare_refiner_kd import validate_preparation_mode

    validate_preparation_mode(preparation_args(tmp_path))
    with pytest.raises(ValueError, match="explicit"):
        validate_preparation_mode(preparation_args(tmp_path, compute_frontend=False))
    with pytest.raises(ValueError, match="existing"):
        validate_preparation_mode(preparation_args(Path("/workspace/vmamba3_data/new")))


def test_cli_has_explicit_reanchor_switch():
    from scripts.prepare_refiner_kd import parser

    args = parser().parse_args(
        [
            "--config",
            "unused",
            "--mode",
            "cache",
            "--partition",
            "train",
            "--out-dir",
            "unused",
            "--reanchor-window",
        ]
    )
    assert args.reanchor_window


def test_prepared_reuse_rejects_policy_and_precision_changes():
    from scripts.prepare_refiner_kd import validate_prepared_reuse

    report = dict(
        config_sha256="config",
        window=16,
        query_policy="strict_visible_gt_reanchor_v1",
        frontend_precision="fp32",
        preprocessing_sha256="preprocessing",
    )
    payload = dict(
        config_sha256="config",
        clip="clip",
        requested_window=16,
        query_policy=report["query_policy"],
        frontend_precision="fp32",
        preprocessing_sha256="preprocessing",
    )
    validate_prepared_reuse(payload, report, Path("clip"))
    for field, changed in (
        ("query_policy", "legacy_anchor_filter_v1"),
        ("frontend_precision", "bf16"),
        ("preprocessing_sha256", "changed"),
    ):
        with pytest.raises(ValueError, match="provenance|policy|precision"):
            validate_prepared_reuse(payload | {field: changed}, report, Path("clip"))


def test_legacy_prepared_artifact_reuses_exact_legacy_defaults():
    from scripts.prepare_refiner_kd import validate_prepared_reuse

    report = dict(
        config_sha256="config",
        window=8,
        query_policy="legacy_anchor_filter_v1",
        frontend_precision="bf16",
        preprocessing_sha256="new-audit",
    )
    validate_prepared_reuse(
        dict(config_sha256="config", clip="clip"), report, Path("clip")
    )
    with pytest.raises(ValueError, match="policy|query_policy"):
        validate_prepared_reuse(
            dict(config_sha256="config", clip="clip"),
            report | {"query_policy": "strict_visible_gt_reanchor_v1"},
            Path("clip"),
        )


def improved_config(tmp_path, monkeypatch):
    from scripts import prepare_refiner_kd as module

    cfg = yaml.safe_load(Path("configs/distill_refiner_vssd.yaml").read_text())
    cfg.update(
        schema_version=2,
        architecture="vssd_local_global_128x2_v2_train",
        improvement=dict(
            query_policy="strict_visible_gt_reanchor_v1",
            frontend_precision="fp32",
            block_steps=100,
            block_lr=0.0003,
            kd_gate="teacher_better",
            gradient_target_ratio=0.1,
            gradient_max_scale=100,
            gradient_direction_gate=True,
        ),
    )
    cfg["data"]["window"] = 16
    pinned = {
        str(cfg["teacher"]["checkpoint"]): cfg["teacher"]["sha256"],
        str(cfg["data"]["split"]): cfg["data"]["split_sha256"],
        str(cfg["protocol"]): cfg["protocol_sha256"],
    }
    monkeypatch.setattr(module, "sha256", lambda path: pinned[str(path)])
    path = tmp_path / "config.yaml"
    return module, cfg, path


def test_improved_config_accepts_strict_settings(tmp_path, monkeypatch):
    module, cfg, path = improved_config(tmp_path, monkeypatch)
    path.write_text(yaml.safe_dump(cfg))
    assert module.read_config(path)["improvement"] == cfg["improvement"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("frontend_precision", "bf16"),
        ("query_policy", "legacy"),
        ("block_steps", 0),
        ("gradient_target_ratio", -1),
        ("gradient_direction_gate", "true"),
        ("extra", 0),
    ],
)
def test_improved_config_rejects_invalid_settings(tmp_path, monkeypatch, field, value):
    module, cfg, path = improved_config(tmp_path, monkeypatch)
    cfg = copy.deepcopy(cfg)
    cfg["improvement"][field] = value
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="improvement|precision|policy"):
        module.read_config(path)


def test_anchor_audit_rejects_shifted_xy():
    from scripts.prepare_refiner_kd import audit_reanchored_queries

    xyz = torch.tensor([[[[0.0, 0.0, 1.0]], [[0.1, 0.0, 1.0]]]])
    visible = torch.ones(1, 2, 1, dtype=torch.bool)
    K = torch.tensor([[[10.0, 0, 10.0], [0, 10.0, 5.0], [0, 0, 1.0]]])
    query = torch.tensor([[[11.0, 5.0, 1.0]]])
    audit = audit_reanchored_queries(query, xyz, visible, K, 20)
    assert audit["anchor_valid_count"] == 1
    assert audit["anchor_reprojection_max_px"] == 0
    with pytest.raises(ValueError, match="projection"):
        audit_reanchored_queries(
            query + torch.tensor([1.0, 0.0, 0.0]), xyz, visible, K, 20
        )
