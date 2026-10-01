"""Shared paired-evaluation scoring and fixed acceptance gates; no CUDA imports."""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from torch import Tensor

from mamba3_tracker.eval.tapvid3d_eval import (
    compute_clip_metrics_absolute,
    compute_clip_metrics_official,
)


class ClipLike(Protocol):
    K: Tensor
    images: Tensor
    tracks_XYZ: Tensor
    visibility: Tensor


def intrinsics_256(clip: ClipLike) -> np.ndarray:
    k = clip.K.detach().cpu().numpy()
    height, width = clip.images.shape[-2:]
    if min(height, width) < 1:
        raise ValueError("Original image dimensions must be positive")
    return np.array([k[0, 0], k[1, 1], k[0, 2], k[1, 2]]) * (256.0 / min(height, width))


def score(clip: ClipLike, tracks: np.ndarray, visibility: np.ndarray) -> dict[str, float]:
    gt = clip.tracks_XYZ.detach().cpu().numpy()
    visible = clip.visibility.detach().float().cpu().numpy()
    expected = (gt.shape[1], gt.shape[0])
    if tracks.shape != (*expected, 3) or visibility.shape != expected:
        raise ValueError("Paired scoring requires all frames and tracks, without truncation")
    if not np.isfinite(tracks).all() or not np.isfinite(visibility).all():
        raise ValueError("Nonfinite predictions must not be scored")
    intrinsics = intrinsics_256(clip)
    metrics = compute_clip_metrics_official(gt, visible, tracks, visibility, intrinsics)
    absolute = compute_clip_metrics_absolute(gt, visible, tracks, visibility, intrinsics)
    if abs(metrics["occlusion_accuracy"] - absolute["occlusion_accuracy"]) > 1e-10:
        raise RuntimeError("Median and absolute metrics disagree on the visibility mask")
    metrics.update(absolute)
    mask = (visible.T > 0.5) & np.isfinite(gt.transpose(1, 0, 2)).all(-1)
    if not mask.any():
        raise ValueError("No finite ground-truth-visible points to score")
    distance = np.linalg.norm((tracks - gt.transpose(1, 0, 2))[mask], axis=-1)
    metrics.update(metric_err_mean_m=float(distance.mean()), metric_err_median_m=float(np.median(distance)))
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("Nonfinite task metrics must not pass the acceptance gate")
    return metrics


def paired_scores(clip: ClipLike, native: np.ndarray, onnx: np.ndarray,
                  flow_visibility: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    # Deliberately one mask argument: neither oracle visibility nor vis_head is substituted.
    return score(clip, native, flow_visibility), score(clip, onnx, flow_visibility)


def error_distribution(native: np.ndarray, other: np.ndarray,
                       visible: np.ndarray | None = None) -> dict[str, float | int]:
    if native.shape != other.shape or not np.isfinite(native).all() or not np.isfinite(other).all():
        raise ValueError("Output differences require matching, finite arrays")
    delta = np.abs(other.astype(np.float64) - native.astype(np.float64))
    norms = np.linalg.norm(delta, axis=-1)
    if visible is not None:
        if visible.shape != norms.shape:
            raise ValueError("Visibility shape does not match the vector output")
        norms = norms[visible > 0.5]
    if norms.size == 0:
        raise ValueError("No visible values in the requested error distribution")
    return {"count": int(norms.size), "mean": float(norms.mean()),
            "p50": float(np.percentile(norms, 50)), "p95": float(np.percentile(norms, 95)),
            "max": float(norms.max()), "max_abs_component_all": float(delta.max())}


def evaluate_gate(results: dict[str, Any], baseline: dict[str, Any], *,
                  completed: set[tuple[str, str]], expected: set[tuple[str, str]],
                  failures: int, frames: int, expected_frames: int) -> dict[str, Any]:
    """Thresholds are fixed before evaluation; incomplete evidence always fails."""
    violations: list[str] = []
    if completed != expected or len(expected) != 9 or failures != 0:
        violations.append("membership_or_failures")
    if frames != expected_frames:
        violations.append("full_frame_coverage")
    scopes = ["overall", *sorted({subset for subset, _ in expected})]
    for scope in scopes:
        try:
            native = results["native"]["overall"] if scope == "overall" else results["native"]["per_subset"][scope]
            onnx = results["onnx"]["overall"] if scope == "overall" else results["onnx"]["per_subset"][scope]
            prior = baseline["overall"] if scope == "overall" else baseline["per_subset"][scope]
            for key in ("average_jaccard", "metric_average_jaccard"):
                values = [float(row[key]) for row in (native, onnx, prior)]
                if not all(np.isfinite(values)):
                    violations.append(f"{scope}.{key}.nonfinite")
                    continue
                if abs(values[0] - values[2]) > 1e-6:
                    violations.append(f"{scope}.{key}.baseline")
                if values[0] - values[1] > 0.001:
                    violations.append(f"{scope}.{key}.degradation")
            oa = [float(row["occlusion_accuracy"]) for row in (native, onnx)]
            if not all(np.isfinite(oa)) or abs(oa[0] - oa[1]) > 1e-10:
                violations.append(f"{scope}.visibility")
        except (KeyError, TypeError, ValueError):
            violations.append(f"{scope}.missing_metrics")
    return {"passed": not violations, "violations": violations,
            "baseline_atol": 1e-6, "maximum_degradation": 0.001, "oa_atol": 1e-10}
