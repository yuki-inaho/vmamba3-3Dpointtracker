"""Host contract and track-only chunking tests (no model or CUDA download)."""
from __future__ import annotations

import numpy as np
import pytest

from mamba3_tracker.deployment.runtime import (
    reference_depth_array,
    run_chunked,
    validate_feed,
)


def fixture():
    raw = np.array([1, 4, 9, 10, 20], np.float32).reshape(1, 1, 5)
    return {"ray": np.zeros((1, 1, 5, 2), np.float32), "z_raw": raw,
            "visibility": np.ones((1, 1, 5), np.float32), "uv": np.ones((1, 1, 5, 2), np.float32),
            "depth_map": np.ones((1, 1, 3, 4), np.float32),
            "dino_features": np.zeros((1, 1, 384, 28, 28), np.float32),
            "intrinsics": np.eye(3, dtype=np.float32)[None], "z_ref": reference_depth_array(raw)}


class FakeSession:
    def __init__(self):
        self.refs = []

    def run(self, names, feed):
        self.refs.append(feed["z_ref"].copy())
        scalar = feed["z_raw"] / feed["z_ref"]
        return [np.repeat(scalar[..., None], 3, -1), feed["uv"].copy(),
                scalar, np.zeros_like(feed["uv"])]


def test_chunked_inference_keeps_global_z_ref():
    feed = fixture()
    session = FakeSession()
    full = run_chunked(session, feed, track_chunk=5)
    chunked = run_chunked(session, feed, track_chunk=2)
    for a, b in zip(full, chunked, strict=True):
        np.testing.assert_array_equal(a, b)
    assert len(session.refs) == 4
    assert all(np.array_equal(x, feed["z_ref"]) for x in session.refs)


def test_reference_depth_is_lower_median_not_numpy_median():
    raw = np.array([1, 4, 9, 10], np.float32).reshape(1, 2, 2)
    assert float(reference_depth_array(raw)) == pytest.approx(4. + 1e-6)


@pytest.mark.parametrize("fault", ["dtype", "missing", "nan", "batch", "empty", "features", "median", "focal", "mask"])
def test_input_contract_rejects_invalid_inputs(fault):
    feed = fixture()
    if fault == "dtype":
        feed["ray"] = feed["ray"].astype(np.float64)
    elif fault == "missing":
        del feed["ray"]
    elif fault == "nan":
        feed["depth_map"][0, 0, 0, 0] = np.nan
    elif fault == "batch":
        feed["intrinsics"] = np.repeat(feed["intrinsics"], 2, axis=0)
    elif fault == "empty":
        feed["ray"] = feed["ray"][:, :, :0]
    elif fault == "features":
        feed["dino_features"] = feed["dino_features"][:, :, :383]
    elif fault == "median":
        feed["z_ref"] = np.array(10., np.float32)
    elif fault == "focal":
        feed["intrinsics"][0, 0, 0] = 0
    elif fault == "mask":
        feed["visibility"][0, 0, 0] = .5
    with pytest.raises((ValueError, TypeError)):
        validate_feed(feed)


def test_memory_guard_does_not_silently_shorten_frames():
    with pytest.raises(ValueError):
        run_chunked(FakeSession(), fixture(), track_chunk=0)
