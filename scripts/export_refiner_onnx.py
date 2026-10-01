"""Export unchanged best200 refiner weights and verify FP32 ONNX, including long sequences."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch import Tensor

from mamba3_tracker.deployment.checkpoint import load_portable, read_checkpoint, sha256
from mamba3_tracker.deployment.io import save_npz
from mamba3_tracker.deployment.runtime import (
    INPUT_NAMES,
    OUTPUT_NAMES,
    run_chunked,
    run_feed,
    session_for,
)
from mamba3_tracker.model.onnx_refiner import reference_depth


def synthetic_inputs(batch: int, frames: int, tracks: int, dino_dim: int,
                     image_size: float, depth_shape: tuple[int, int] = (32, 48),
                     seed: int = 0) -> tuple[Tensor, ...]:
    rng = torch.Generator().manual_seed(seed)
    uv = torch.rand(batch, frames, tracks, 2, generator=rng) * image_size
    depth = 3 + torch.rand(batch, frames, *depth_shape, generator=rng) * 5
    features = torch.randn(batch, frames, dino_dim, 28, 28, generator=rng) * 0.3
    vis = (torch.rand(batch, frames, tracks, generator=rng) > 0.2).float()
    K = torch.tensor([[image_size, 0, image_size / 2],
                      [0, image_size, image_size / 2], [0, 0, 1]], dtype=torch.float32)
    K = K.unsqueeze(0).repeat(batch, 1, 1)
    ray = (uv - image_size / 2) / image_size
    z = F.grid_sample(depth.reshape(batch * frames, 1, *depth_shape),
                      (2 * uv / image_size - 1).reshape(batch * frames, 1, tracks, 2),
                      padding_mode="border", align_corners=False).reshape(batch, frames, tracks)
    return ray, z, vis, uv, depth, features, K, reference_depth(z)



def as_feed(inputs: tuple[Tensor, ...]) -> dict[str, np.ndarray]:
    return {name: value.detach().cpu().numpy() for name, value in zip(INPUT_NAMES, inputs, strict=True)}


def run_session(session: ort.InferenceSession, inputs: tuple[Tensor, ...]) -> list[np.ndarray]:
    return run_feed(session, as_feed(inputs))


def compare(expected, observed) -> dict[str, float]:
    errors = {}
    for name, ref, actual in zip(OUTPUT_NAMES, expected, observed, strict=True):
        reference = ref.detach().cpu().numpy() if isinstance(ref, Tensor) else ref
        if not np.isfinite(reference).all() or not np.isfinite(actual).all():
            raise RuntimeError(f"Nonfinite output: {name}")
        np.testing.assert_allclose(actual, reference, rtol=2e-4, atol=5e-5)
        errors[name] = float(np.max(np.abs(actual - reference)))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("result/onnx_best200"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--long-frames", type=int, default=128)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--fixture-dir", type=Path, help="Write a small actual-weight regression fixture for C++")
    args = parser.parse_args()
    if args.threads < 1 or args.long_frames < 128:
        parser.error("threads must be positive and --long-frames must be >=128")
    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "export_report.json"
    report_path.write_text(json.dumps({"success": False, "status": "started"}) + "\n")
    state = read_checkpoint(args.ckpt, args.expected_sha256)
    model = load_portable(state)
    dim = model.feat_proj.in_features
    example = synthetic_inputs(2, 8, 4, dim, model.image_size)
    step = state["step"]
    stem = "tracker_mamba3_daoff_best200" if step == 200 else f"tracker_mamba3_step{step}"
    path = args.out_dir / f"{stem}.onnx"
    dynamic_axes = {name: {0: "batch", 1: "frames", 2: "tracks"}
                    for name in ("ray", "z_raw", "visibility", "uv")}
    dynamic_axes.update({"depth_map": {0: "batch", 1: "frames", 2: "depth_height", 3: "depth_width"},
                         "dino_features": {0: "batch", 1: "frames"}, "intrinsics": {0: "batch"}})
    dynamic_axes.update({name: {0: "batch", 1: "frames", 2: "tracks"} for name in OUTPUT_NAMES})
    start = time.perf_counter()
    with torch.inference_mode():
        torch.onnx.export(model, example, str(path), input_names=list(INPUT_NAMES),
                          output_names=list(OUTPUT_NAMES), dynamic_axes=dynamic_axes,
                          opset_version=18, dynamo=False, external_data=False)
    graph = onnx.load(str(path))
    onnx.helper.set_model_props(graph, {
        "source_checkpoint_step": str(step), "source_checkpoint_sha256": sha256(args.ckpt),
        "scope": "v35 refiner; external DINO features, depth and flow; not RGB end-to-end",
        "precision": "FP32; standard ONNX operators; not validated against CUDA/BF16 task metrics",
        "model_config": json.dumps(state["cfg"]["model"], sort_keys=True),
        "dino_revision": str(state.get("export", {}).get("dino_revision", "not_recorded")),
        "upstream_mamba_revision": "e9594ce1c732d97440f0332fdc43170a2294dbfa",
        "z_ref": "lower median of all z_raw values in the full input batch + 1e-6",
    })
    onnx.checker.check_model(graph, full_check=True)
    if {node.domain for node in graph.graph.node} != {""}:
        raise RuntimeError("Custom operator domains are not allowed")
    if any(x.data_location == onnx.TensorProto.EXTERNAL for x in graph.graph.initializer):
        raise RuntimeError("A single-file ONNX artifact is required")
    onnx.save(graph, str(path))
    session = session_for(path, args.threads)
    schema = [{"name": item.name, "type": item.type, "shape": item.shape} for item in session.get_inputs()]
    for item in session.get_inputs():
        if item.name != "z_ref" and not isinstance(item.shape[0], str):
            raise RuntimeError("Batch dimension is not dynamic")
        if item.name not in ("intrinsics", "z_ref") and not isinstance(item.shape[1], str):
            raise RuntimeError("Frame dimension is not dynamic")
        if item.name in ("ray", "z_raw", "visibility", "uv") and not isinstance(item.shape[2], str):
            raise RuntimeError("Track dimension is not dynamic")
    verification = []
    cases = [(1, 1, 1, (24, 32)), (1, 8, 7, (32, 48)), (2, 13, 3, (48, 32)),
             (1, args.long_frames, 2, (17, 23))]
    for batch, frames, tracks, depth_shape in cases:
        inputs = synthetic_inputs(batch, frames, tracks, dim, model.image_size, depth_shape, seed=frames)
        with torch.inference_mode():
            expected = model(*inputs)
        feed = as_feed(inputs)
        observed = run_feed(session, feed)
        errors = compare(expected, observed)
        chunked = run_chunked(session, feed, track_chunk=1)
        chunk_errors = compare(observed, chunked)
        verification.append({"batch": batch, "frames": frames, "tracks": tracks,
                             "depth_shape": depth_shape, "max_abs_error": errors,
                             "chunk_1_max_abs_error": chunk_errors, "all_outputs_finite": True})
        print(f"[onnx] B={batch} F={frames} N={tracks}: {errors}; chunk={chunk_errors}", flush=True)
        del inputs, feed, expected, observed, chunked
    if args.fixture_dir:
        args.fixture_dir.mkdir(parents=True, exist_ok=True)
        inputs = synthetic_inputs(1, 3, 5, dim, model.image_size, (11, 19), seed=200)
        feed = as_feed(inputs)
        arrays = run_feed(session, feed)
        for name, value in feed.items():
            np.save(args.fixture_dir / f"{name}.npy", value, allow_pickle=False)
        for name, value in zip(OUTPUT_NAMES, arrays, strict=True):
            np.save(args.fixture_dir / f"expected_{name}.npy", value, allow_pickle=False)
        save_npz(args.fixture_dir / "inputs.npz", feed)
    report = {"success": True, "checkpoint_step": step, "checkpoint_sha256": sha256(args.ckpt),
              "file": path.name, "onnx_bytes": path.stat().st_size, "onnx_sha256": sha256(path),
              "opset": 18, "node_count": len(graph.graph.node), "operator_domains": [""],
              "single_file": True, "checker_full_check": True, "input_schema": schema,
              "tracker_tensors_loaded": len(model.state_dict()),
              "tracker_parameters": sum(p.numel() for p in model.parameters()),
              "all_tracker_weights_bitwise_equal": True, "runtime": ort.__version__,
              "providers": session.get_providers(), "torch": torch.__version__,
              "python": platform.python_version(), "numpy": np.__version__,
              "verification": verification, "elapsed_s": time.perf_counter() - start,
              "rtol": 2e-4, "atol": 5e-5, "task_accuracy_verified": False,
              "scope": "Actual checkpoint weights, synthetic inputs: portable FP32 vs ONNX FP32 only"}
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.out_dir / "SHA256SUMS").write_text(f"{report['onnx_sha256']}  {path.name}\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
