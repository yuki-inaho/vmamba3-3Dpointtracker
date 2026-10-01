"""Strict, NumPy-only ONNX Runtime CPU inference for the best200 refiner."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from beartype import beartype
from jaxtyping import Float32, jaxtyped

INPUT_NAMES = ("ray", "z_raw", "visibility", "uv", "depth_map", "dino_features", "intrinsics", "z_ref")
OUTPUT_NAMES = ("xyz", "uv_refined", "vis_logits", "delta_uv")
TRACK_INPUT_NAMES = ("ray", "z_raw", "visibility", "uv")


@jaxtyped(typechecker=beartype)
def reference_depth_array(z_raw: Float32[np.ndarray, "batch frames tracks"]) -> np.ndarray:
    if not z_raw.size or not np.isfinite(z_raw).all():
        raise ValueError("z_raw must be nonempty and finite")
    flat = z_raw.ravel()
    lower = np.partition(flat, (flat.size - 1) // 2)[(flat.size - 1) // 2]
    return np.asarray(lower + np.float32(1e-6), dtype=np.float32)


def validate_feed(feed: dict[str, np.ndarray], *, check_reference: bool = True) -> tuple[int, int, int]:
    if set(feed) != set(INPUT_NAMES):
        raise ValueError(f"Expected exactly these inputs: {INPUT_NAMES}")
    for name, value in feed.items():
        if not isinstance(value, np.ndarray) or value.dtype != np.float32:
            raise TypeError(f"{name} must be a float32 ndarray; no implicit conversion")
        if not value.size or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain nonempty, finite values")
    ray = feed["ray"]
    if ray.ndim != 4 or ray.shape[-1] != 2 or min(ray.shape[:3]) < 1:
        raise ValueError("ray must have shape (B,F,N,2), with nonzero dimensions")
    b, f, n = ray.shape[:3]
    shapes = {"z_raw": (b, f, n), "visibility": (b, f, n), "uv": (b, f, n, 2),
              "dino_features": (b, f, 384, 28, 28), "intrinsics": (b, 3, 3), "z_ref": ()}
    for name, shape in shapes.items():
        if feed[name].shape != shape:
            raise ValueError(f"{name}: expected {shape}, got {feed[name].shape}")
    depth = feed["depth_map"]
    if depth.ndim != 4 or depth.shape[:2] != (b, f) or min(depth.shape[2:]) < 1:
        raise ValueError("depth_map must have shape (B,F,Hd,Wd)")
    if np.any(feed["z_raw"] < 0) or np.any(depth < 0) or float(feed["z_ref"]) <= 0:
        raise ValueError("Depth must be nonnegative and z_ref must be positive")
    vis = feed["visibility"]
    if np.any((vis != 0) & (vis != 1)):
        raise ValueError("visibility must be the binary flow mask (0 or 1)")
    k = feed["intrinsics"]
    if np.any(k[:, 0, 0] <= 0) or np.any(k[:, 1, 1] <= 0):
        raise ValueError("Camera focal lengths must be positive")
    if np.any(k[:, 0, 1] != 0) or np.any(k[:, 1, 0] != 0) or not np.all(k[:, 2] == [0, 0, 1]):
        raise ValueError("Only unskewed pinhole intrinsics are supported")
    if check_reference and not np.array_equal(feed["z_ref"], reference_depth_array(feed["z_raw"])):
        raise ValueError("z_ref must be the lower median of the full B/F/N input plus 1e-6")
    return b, f, n


def session_for(path: Path, threads: int = 4) -> ort.InferenceSession:
    if threads < 1:
        raise ValueError("threads must be positive")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    session.disable_fallback()
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("This deployment path only validates CPUExecutionProvider")
    if [x.name for x in session.get_inputs()] != list(INPUT_NAMES):
        raise ValueError("Unexpected ONNX input schema")
    if [x.name for x in session.get_outputs()] != list(OUTPUT_NAMES):
        raise ValueError("Unexpected ONNX output schema")
    if any(x.type != "tensor(float)" for x in (*session.get_inputs(), *session.get_outputs())):
        raise TypeError("All ONNX inputs and outputs must be float32")
    return session


def _run_validated(session: ort.InferenceSession, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    shape = feed["ray"].shape[:3]
    expected = [(*shape, 3), (*shape, 2), shape, (*shape, 2)]
    for name, item, want in zip(OUTPUT_NAMES, session.run(list(OUTPUT_NAMES), feed), expected, strict=True):
        if not isinstance(item, np.ndarray) or item.dtype != np.float32 or item.shape != want:
            raise RuntimeError(f"Invalid output schema for {name}")
        if not np.isfinite(item).all():
            raise RuntimeError(f"Nonfinite output for {name}")
        arrays.append(item)
    return arrays


def run_feed(session: ort.InferenceSession, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
    validate_feed(feed)
    return _run_validated(session, feed)


def run_chunked(session: ort.InferenceSession, feed: dict[str, np.ndarray], track_chunk: int = 32,
                memory_budget_mib: int = 1024) -> list[np.ndarray]:
    """Split tracks only; never reset the temporal state or recompute z_ref.

    The budget is a conservative attention-workspace estimate, not a bound on
    total RSS (DINO input features and runtime/model allocations are additional).
    """
    if track_chunk < 1 or memory_budget_mib < 1:
        raise ValueError("track_chunk and memory_budget_mib must be positive")
    b, f, n = validate_feed(feed)
    estimate = b * min(n, track_chunk) * 24 * f * f * 4 * 8
    if estimate > memory_budget_mib * 1024**2:
        raise MemoryError("Quadratic attention workspace exceeds budget; reduce --track-chunk (not frames)")
    groups: list[list[np.ndarray]] = [[] for _ in OUTPUT_NAMES]
    for start in range(0, n, track_chunk):
        chunk = dict(feed)
        for name in TRACK_INPUT_NAMES:
            chunk[name] = np.ascontiguousarray(feed[name][:, :, start:start + track_chunk])
        for group, value in zip(groups, _run_validated(session, chunk), strict=True):
            group.append(value)
    return [np.concatenate(group, axis=2) for group in groups]
