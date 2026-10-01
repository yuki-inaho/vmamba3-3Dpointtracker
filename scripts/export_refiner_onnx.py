"""Export the selected official-Mamba-3 refiner; verify with ONNX Runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch import Tensor

from mamba3_tracker.model.onnx_refiner import OnnxV35Refiner, reference_depth


INPUT_NAMES = ("ray", "z_raw", "visibility", "uv", "depth_map", "dino_features", "intrinsics", "z_ref")
OUTPUT_NAMES = ("xyz", "uv_refined", "vis_logits", "delta_uv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def session_for(path: Path, threads: int = 8) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def run_session(session: ort.InferenceSession, inputs: tuple[Tensor, ...]) -> list[np.ndarray]:
    feed = {name: value.detach().cpu().float().numpy() for name, value in zip(INPUT_NAMES, inputs)}
    return run_feed(session, feed)


def run_feed(session: ort.InferenceSession, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
    outputs = []
    for value in session.run(list(OUTPUT_NAMES), feed):
        if not isinstance(value, np.ndarray):
            raise TypeError("The refiner must return dense ndarray outputs")
        outputs.append(value)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("result/onnx_best200"))
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    weights = {k: v for k, v in state["model"].items() if not k.startswith("dino.")}
    dim = weights["feat_proj.weight"].shape[1]
    model = OnnxV35Refiner(state["cfg"]["model"], dino_dim=dim).float().eval()
    model.load_state_dict(weights, strict=True)
    assert all(torch.equal(value, weights[name]) for name, value in model.state_dict().items())
    example = synthetic_inputs(2, 8, 4, dim, model.image_size)
    path = args.out_dir / "tracker_mamba3_daoff_best200.onnx"
    dynamic_axes = {name: {0: "batch", 1: "frames", 2: "tracks"}
                    for name in ("ray", "z_raw", "visibility", "uv")}
    dynamic_axes.update({"depth_map": {0: "batch", 1: "frames", 2: "depth_height", 3: "depth_width"},
                         "dino_features": {0: "batch", 1: "frames"}, "intrinsics": {0: "batch"}})
    dynamic_axes.update({name: {0: "batch", 1: "frames", 2: "tracks"} for name in OUTPUT_NAMES})
    start = time.perf_counter()
    with torch.inference_mode():
        # A pure ATen graph also works with the legacy exporter used by the
        # linked VMamba_onnx reference; there are no custom scan operators.
        torch.onnx.export(model, example, str(path), input_names=list(INPUT_NAMES),
                          output_names=list(OUTPUT_NAMES), dynamic_axes=dynamic_axes,
                          opset_version=18, dynamo=False, external_data=False)
    graph = onnx.load(str(path))
    onnx.helper.set_model_props(graph, {
        "source_checkpoint_step": str(state["step"]),
        "source_checkpoint_sha256": sha256(args.ckpt),
        "scope": "v35 refiner with external frozen DINO features, depth and flow tracks",
        "precision": "FP32; standard ONNX operators",
        "model_config": json.dumps(state["cfg"]["model"], sort_keys=True),
        "z_ref": "lower median of all z_raw values in the full input batch + 1e-6",
    })
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, str(path))
    session = session_for(path, args.threads)
    verification = []
    for batch, frames, tracks, depth_shape in ((1, 1, 1, (24, 32)), (1, 8, 7, (32, 48)),
                                               (2, 13, 3, (48, 32))):
        inputs = synthetic_inputs(batch, frames, tracks, dim, model.image_size, depth_shape, seed=frames)
        with torch.inference_mode():
            expected = model(*inputs)
        observed = run_session(session, inputs)
        errors = {}
        for name, ref, actual in zip(OUTPUT_NAMES, expected, observed):
            np.testing.assert_allclose(actual, ref.numpy(), rtol=2e-4, atol=5e-5)
            errors[name] = float(np.max(np.abs(actual - ref.numpy())))
        verification.append({"batch": batch, "frames": frames, "tracks": tracks,
                             "depth_shape": depth_shape, "max_abs_error": errors})
        print(f"[onnx] verified dynamic shape B={batch} F={frames} N={tracks}: {errors}", flush=True)
    report = {"success": True, "checkpoint_step": state["step"],
              "checkpoint_sha256": sha256(args.ckpt), "file": path.name,
              "onnx_bytes": path.stat().st_size, "onnx_sha256": sha256(path),
              "opset": 18, "node_count": len(graph.graph.node),
              "operator_domains": sorted({node.domain for node in graph.graph.node}),
              "tracker_tensors_loaded": len(weights),
              "tracker_parameters": sum(p.numel() for p in model.parameters()),
              "all_tracker_weights_bitwise_equal": True,
              "runtime": ort.__version__, "providers": session.get_providers(),
              "verification": verification, "elapsed_s": time.perf_counter() - start,
              "scope": "Synthetic ONNX-vs-portable-FP32 verification; task metrics are evaluated separately."}
    (args.out_dir / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out_dir / "SHA256SUMS").write_text(f"{report['onnx_sha256']}  {path.name}\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
