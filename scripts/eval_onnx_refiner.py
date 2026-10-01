"""Compare ONNX and native best200 on identical TAPVid-3D front-end outputs."""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path
from typing import TypedDict
from unittest.mock import patch

import numpy as np
import torch
from torch import Tensor, nn

from eval_metric3d import _infer, _metric_err
from export_refiner_onnx import INPUT_NAMES, OUTPUT_NAMES, run_feed, session_for, sha256
from train_depth_refined_tracker import _build_waft_flow

from mamba3_tracker.data.tapvid3d import load_clip
from mamba3_tracker.eval.tapvid3d_eval import aggregate, compute_clip_metrics
from mamba3_tracker.model.depth_refined_tracker import Mamba3V35Refiner
from mamba3_tracker.model.heads import TrackerOutputs
from mamba3_tracker.model.onnx_refiner import reference_depth


class VariantResults(TypedDict):
    per_subset: dict[str, dict[str, float]]
    overall: dict[str, float]


class ClipComparison(TypedDict):
    subset: str
    clip_id: str
    max_abs_output_error: dict[str, float]
    average_jaccard_delta: float
    metric_average_jaccard_delta: float


class PairedOnnxRefiner(nn.Module):
    """DINO runs once; native and ONNX receive identical frozen features."""

    def __init__(self, native: Mamba3V35Refiner, onnx_path: Path,
                 track_chunk: int, threads: int) -> None:
        super().__init__()
        self.native = native
        self.session = session_for(onnx_path, threads)
        self.track_chunk = track_chunk
        self.native_output: TrackerOutputs | None = None
        self.last_errors: dict[str, float] = {}

    @torch.no_grad()
    def forward(self, ray: Tensor, z_raw: Tensor, vis: Tensor, uv: Tensor,
                depth_map: Tensor, images: Tensor, K: Tensor) -> TrackerOutputs:
        captured: list[Tensor] = []
        forward_video = self.native.dino.forward_video

        def capture(video: Tensor) -> list[Tensor]:
            features = forward_video(video)
            captured.append(features[0])
            return features

        with patch.object(self.native.dino, "forward_video", side_effect=capture):
            reference = self.native(ray, z_raw, vis, uv, depth_map, images, K)
        if len(captured) != 1:
            raise RuntimeError("Expected exactly one DINO feature extraction")
        inputs = (ray, z_raw, vis, uv, depth_map, captured[0], K, reference_depth(z_raw))
        # Keep the normalization of the WHOLE point set when chunking tracks.
        feed = {name: value.detach().cpu().float().numpy()
                for name, value in zip(INPUT_NAMES, inputs)}
        pieces: list[list[np.ndarray]] = [[] for _ in OUTPUT_NAMES]
        for start in range(0, ray.shape[2], self.track_chunk):
            chunk = dict(feed)
            for name in ("ray", "z_raw", "visibility", "uv"):
                chunk[name] = feed[name][:, :, start:start + self.track_chunk]
            outputs = run_feed(self.session, chunk)
            for group, output in zip(pieces, outputs):
                group.append(output)
        arrays = [np.concatenate(group, axis=2) for group in pieces]
        native_outputs = (reference.xyz, reference.uv, reference.vis_logits, reference.delta_uv)
        self.last_errors = {}
        for name, expected, actual in zip(OUTPUT_NAMES, native_outputs, arrays):
            if expected is None or not np.isfinite(actual).all():
                raise RuntimeError(f"Invalid {name} output")
            self.last_errors[name] = float(np.max(np.abs(expected.detach().float().cpu().numpy() - actual)))
        self.native_output = reference
        xyz, refined_uv, logits, delta = (torch.from_numpy(array).to(ray.device) for array in arrays)
        return TrackerOutputs(xyz=xyz, uv=refined_uv, vis_logits=logits,
                              spawn_logits=logits, delta_uv=delta)


def score(clip, tracks: np.ndarray, visibility: np.ndarray) -> dict:
    K = clip.K.numpy()
    intrinsics = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
    gt = clip.tracks_XYZ.numpy()
    visible = clip.visibility.float().numpy()
    metrics = compute_clip_metrics(gt, visible, tracks, visibility, intrinsics)
    mean, median = _metric_err(tracks, gt.transpose(1, 0, 2), visible.T)
    metrics.update(metric_err_mean_m=mean, metric_err_median_m=median, clip_id=clip.clip_id)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("configs/v64_metric_reference_minival.json"))
    parser.add_argument("--data-root", type=Path, default=Path("~/data/tapvid3d"))
    parser.add_argument("--depth-root", type=Path, default=Path("~/data/tapvid3d_da3"))
    parser.add_argument("--out-dir", type=Path, default=Path("result/onnx_best200/paired_reference"))
    parser.add_argument("--track-chunk", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if min(args.track_chunk, args.threads) < 1:
        parser.error("chunk and threads must be positive")
    if not torch.cuda.is_available():
        parser.error("native Triton comparison requires CUDA")
    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = state["cfg"]["model"]
    keys = inspect.signature(Mamba3V35Refiner).parameters
    native = Mamba3V35Refiner(**{key: value for key, value in config.items() if key in keys})
    native.load_state_dict(state["model"], strict=True)
    native = native.cuda().eval()
    paired = PairedOnnxRefiner(native, args.onnx, args.track_chunk, args.threads).eval()
    flow_cfg = state["cfg"]["flow"]
    if flow_cfg["source"] != "waft_live":
        raise ValueError("Selected best200 comparison expects WAFT live front-end")
    flow = _build_waft_flow(torch.device("cuda"), scale=flow_cfg.get("scale"), iters=flow_cfg.get("iters"))
    manifest = json.loads(args.manifest.read_text())
    native_rows: dict[str, list[dict]] = {}
    onnx_rows: dict[str, list[dict]] = {}
    differences: list[ClipComparison] = []
    started = time.perf_counter()
    for subset, files in manifest["files_by_subset"].items():
        native_rows[subset], onnx_rows[subset] = [], []
        for filename in files:
            clip = load_clip(args.data_root.expanduser() / subset / filename)
            tracks, visibility = _infer(
                method="v35", model=paired, flow_model=flow, clip=clip, device=torch.device("cuda"),
                image_size=int(config["image_size"]), fb_alpha=float(flow_cfg.get("fb_alpha", 0.05)),
                fb_beta=float(flow_cfg.get("fb_beta", 1.0)), da3_depth_root=args.depth_root.expanduser(),
                max_frames=0)
            reference = paired.native_output
            if reference is None:
                raise RuntimeError("Native output was not captured")
            ref_tracks = reference.xyz[0].transpose(0, 1).detach().float().cpu().numpy()
            ref_metrics, metrics = score(clip, ref_tracks, visibility), score(clip, tracks, visibility)
            native_rows[subset].append(ref_metrics)
            onnx_rows[subset].append(metrics)
            delta: ClipComparison = {"subset": subset, "clip_id": clip.clip_id,
                     "max_abs_output_error": paired.last_errors,
                     "average_jaccard_delta": metrics["average_jaccard"] - ref_metrics["average_jaccard"],
                     "metric_average_jaccard_delta": metrics["metric_average_jaccard"] - ref_metrics["metric_average_jaccard"]}
            differences.append(delta)
            (args.out_dir / "clip_comparison.json").write_text(json.dumps(differences, indent=2) + "\n")
            print(f"[paired] {subset}/{clip.clip_id}: AJ native={ref_metrics['average_jaccard']:.6f}"
                  f" ONNX={metrics['average_jaccard']:.6f}; metric-AJ native={ref_metrics['metric_average_jaccard']:.6f}"
                  f" ONNX={metrics['metric_average_jaccard']:.6f}; xyz max diff={paired.last_errors['xyz']:.6g}m",
                  flush=True)
    results: dict[str, VariantResults] = {}
    for label, rows in (("native", native_rows), ("onnx", onnx_rows)):
        per_subset = {subset: aggregate(values) for subset, values in rows.items()}
        results[label] = {"per_subset": per_subset, "overall": aggregate(list(per_subset.values()))}
        (args.out_dir / f"{label}_per_clip.json").write_text(json.dumps(rows, indent=2) + "\n")
    overall_delta = {key: results["onnx"]["overall"][key] - results["native"]["overall"][key]
                     for key in results["native"]["overall"]}
    report = {"checkpoint_step": state["step"], "checkpoint_sha256": sha256(args.ckpt),
              "onnx_sha256": sha256(args.onnx), "scope": "same fixed 9/150 minival clips; monitoring reference",
              "clip_manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest),
              "clips": len(differences), "failures": 0, "visibility_source": "same WAFT flow mask",
              "front_end_shared_between_native_and_onnx": True, "track_chunk": args.track_chunk,
              "results": results, "overall_delta": overall_delta,
              "max_abs_output_error": {name: max(row["max_abs_output_error"][name] for row in differences)
                                       for name in OUTPUT_NAMES},
              "elapsed_s": time.perf_counter() - started}
    (args.out_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
