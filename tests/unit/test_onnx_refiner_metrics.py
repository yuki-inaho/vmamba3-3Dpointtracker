"""Regression tests for the original metric failures, loading and acceptance gates."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from mamba3_tracker.deployment import metrics
from mamba3_tracker.deployment.checkpoint import load_native_state, resolve_flow_config


def fixture():
    xyz = torch.tensor([[[0., 0., 3.], [1., 1., 4.]], [[0., 0., 3.], [1., 1., 4.]]])
    clip = SimpleNamespace(K=torch.tensor([[896., 0., 448.], [0., 896., 448.], [0., 0., 1.]]),
                           tracks_XYZ=xyz, visibility=torch.ones(2, 2),
                           images=torch.empty(2, 3, 1280, 1920, device="meta"), clip_id="fixture")
    return clip, xyz.numpy().transpose(1, 0, 2), np.ones((2, 2), np.float32)


def test_score_contains_absolute_metric_aj():
    result = metrics.score(*fixture())
    assert result["metric_average_jaccard"] == pytest.approx(1.)
    assert result["average_jaccard"] == pytest.approx(1.)


def test_intrinsics_match_official_256_scaling(monkeypatch):
    captured = []
    original = metrics.compute_clip_metrics_official

    def capture(*args):
        captured.append(args[-1])
        return original(*args)

    monkeypatch.setattr(metrics, "compute_clip_metrics_official", capture)
    metrics.score(*fixture())
    np.testing.assert_allclose(captured[0], np.array([896, 896, 448, 448]) * 256 / 1280)


def test_paired_visibility_uses_same_flow_mask(monkeypatch):
    clip, tracks, mask = fixture()
    captured = []
    original = metrics.score

    def capture(clip, values, visibility):
        captured.append(visibility)
        return original(clip, values, visibility)

    monkeypatch.setattr(metrics, "score", capture)
    native, onnx = metrics.paired_scores(clip, tracks, tracks * 1.02, mask)
    assert captured[0] is captured[1] is mask
    assert native["occlusion_accuracy"] == onnx["occlusion_accuracy"]


def test_score_rejects_truncated_frames():
    clip, tracks, mask = fixture()
    with pytest.raises(ValueError, match="all frames"):
        metrics.score(clip, tracks[:, :1], mask[:, :1])


def test_score_rejects_nonfinite_predictions():
    clip, tracks, mask = fixture()
    tracks = tracks.copy()
    tracks[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        metrics.score(clip, tracks, mask)


class TinyNative(nn.Module):
    def __init__(self):
        super().__init__()
        self.dino = nn.Module()
        self.dino.backbone = nn.Linear(2, 2)
        self.dino.backbone.requires_grad_(False)
        self.dino.register_buffer("imagenet_mean", torch.zeros(1, 3, 1, 1))
        self.dino.register_buffer("imagenet_std", torch.ones(1, 3, 1, 1))
        self.feat_proj = nn.Linear(2, 2)


def public_state(model):
    return {"model": {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("dino.backbone.")},
            "export": {"excluded_prefixes": ["dino.backbone."]}}


def test_public_checkpoint_only_omits_frozen_backbone():
    model = TinyNative()
    result = load_native_state(model, public_state(model))
    assert result["missing_frozen_backbone_tensors"] == 2
    assert result["all_supplied_weights_bitwise_equal"]


@pytest.mark.parametrize("fault", ["missing_tracker", "missing_buffer", "unexpected", "shape", "dtype", "nan", "trainable", "partial_backbone", "unmarked"])
def test_checkpoint_rejects_unapproved_mismatch(fault):
    model = TinyNative()
    state = public_state(model)
    if fault == "missing_tracker":
        del state["model"]["feat_proj.weight"]
    elif fault == "missing_buffer":
        del state["model"]["dino.imagenet_mean"]
    elif fault == "unexpected":
        state["model"]["unrelated.weight"] = torch.ones(1)
    elif fault == "shape":
        state["model"]["feat_proj.weight"] = torch.ones(3, 3)
    elif fault == "dtype":
        state["model"]["feat_proj.weight"] = state["model"]["feat_proj.weight"].half()
    elif fault == "nan":
        state["model"]["feat_proj.weight"][0, 0] = float("nan")
    elif fault == "trainable":
        model.dino.backbone.requires_grad_(True)
    elif fault == "partial_backbone":
        state["model"]["dino.backbone.weight"] = model.dino.backbone.weight.clone()
    elif fault == "unmarked":
        del state["export"]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ValueError):
        load_native_state(model, state)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())


def test_full_checkpoint_is_strict():
    model = TinyNative()
    result = load_native_state(model, {"model": model.state_dict()})
    assert result["missing_frozen_backbone_tensors"] == 0


def test_public_pt_requires_explicit_flow_config():
    state = {"cfg": {"model": {}}}
    with pytest.raises(ValueError, match="--run-config"):
        resolve_flow_config(state, None)
    flow = {"source": "waft_live", "scale": 0, "iters": 12, "fb_alpha": 0.05, "fb_beta": 1.}
    assert resolve_flow_config(state, {"flow": flow}) == flow


def gate_fixture():
    base = {"average_jaccard": .2, "metric_average_jaccard": .3, "occlusion_accuracy": .7}
    row = {"overall": base.copy(), "per_subset": {s: base.copy() for s in ("adt", "drivetrack", "pstudio")}}
    members = {(s, f"{i}.npz") for s in row["per_subset"] for i in range(3)}
    return {"native": copy.deepcopy(row), "onnx": copy.deepcopy(row)}, row, members


@pytest.mark.parametrize("fault", [None, "subset_degradation", "baseline", "oa", "nan", "missing", "frames", "failure"])
def test_acceptance_gate_is_fail_closed(fault):
    results, baseline, members = gate_fixture()
    complete = set(members)
    if fault == "subset_degradation":
        results["onnx"]["per_subset"]["drivetrack"]["average_jaccard"] -= .002
    elif fault == "baseline":
        results["native"]["overall"]["metric_average_jaccard"] += .00001
    elif fault == "oa":
        results["onnx"]["overall"]["occlusion_accuracy"] += 1e-8
    elif fault == "nan":
        results["onnx"]["overall"]["average_jaccard"] = float("nan")
    elif fault == "missing":
        complete.pop()
    result = metrics.evaluate_gate(results, baseline, completed=complete, expected=members,
                                   failures=int(fault == "failure"), frames=1484 if fault == "frames" else 1485,
                                   expected_frames=1485)
    assert result["passed"] == (fault is None)


def test_visible_error_distribution():
    a = np.zeros((1, 2, 3), np.float32)
    b = np.array([[[3., 4., 0.], [100., 0., 0.]]], np.float32)
    result = metrics.error_distribution(a, b, np.array([[1, 0]]))
    assert result["count"] == 1
    assert result["mean"] == result["p95"] == result["max"] == 5.
