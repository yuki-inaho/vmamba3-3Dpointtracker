"""Real-video CUDA checks must bind full coverage to identical input tensors."""

import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "gpu_video", SCRIPTS / "verify_refiner_gpu_video.py"
)
assert spec is not None and spec.loader is not None
video = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video)


def feed():
    return {
        "ray": np.zeros((1, 2, 3, 2), np.float32),
        "z_raw": np.ones((1, 2, 3), np.float32),
        "visibility": np.ones((1, 2, 3), np.float32),
        "uv": np.zeros((1, 2, 3, 2), np.float32),
        "depth_map": np.ones((1, 2, 4, 5), np.float32),
        "dino_features": np.zeros((1, 2, 384, 28, 28), np.float32),
        "intrinsics": np.eye(3, dtype=np.float32)[None],
        "z_ref": np.asarray(np.float32(1) + np.float32(1e-6)),
    }


def test_feed_signature_binds_shape_dtype_and_content():
    values = feed()
    signature = video.feed_signature(values)
    assert signature["ray"]["shape"] == [1, 2, 3, 2]
    assert signature["ray"]["dtype"] == "float32"
    video.verify_feed(values, signature, 2, 3)
    values["ray"][0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="signature"):
        video.verify_feed(values, signature, 2, 3)


@pytest.mark.parametrize("frames,tracks", [(1, 3), (2, 2)])
def test_feed_rejects_incomplete_coverage(frames, tracks):
    values = feed()
    with pytest.raises(ValueError, match="coverage"):
        video.verify_feed(values, video.feed_signature(values), frames, tracks)


def test_feed_rejects_implicit_dtype_conversion():
    values = feed()
    values["uv"] = values["uv"].astype(np.float64)
    with pytest.raises(TypeError, match="float32"):
        video.verify_feed(values, video.feed_signature(values), 2, 3)


def test_membership_rejects_missing_or_duplicate_clip():
    expected = [
        {"subset": s, "frames": f, "tracks": n}
        for s, f, n in (
            ("pstudio", 150, 460),
            ("drivetrack", 31, 256),
            ("adt", 300, 900),
        )
    ]
    video.check_membership(expected)
    with pytest.raises(ValueError, match="membership"):
        video.check_membership(expected[:2])
    with pytest.raises(ValueError, match="membership"):
        video.check_membership([expected[0], expected[0], expected[2]])
    expected[2]["frames"] = 299
    with pytest.raises(ValueError, match="coverage"):
        video.check_membership(expected)


@pytest.mark.parametrize("fails", [False, True])
def test_builder_cwd_is_explicit_and_always_restored(tmp_path, fails):
    initial = Path.cwd()

    def callback():
        assert Path.cwd() == tmp_path
        if fails:
            raise ValueError("builder failed")
        return "complete"

    if fails:
        with pytest.raises(ValueError, match="builder failed"):
            video.invoke_from(tmp_path, callback)
    else:
        assert video.invoke_from(tmp_path, callback) == "complete"
    assert Path.cwd() == initial
    assert Path(os.getcwd()) == initial
