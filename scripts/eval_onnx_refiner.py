"""Fail-closed native BF16 versus ONNX FP32 comparison on the fixed nine clips.

The refiner shares a single WAFT mask, DA3 map, and DINO feature extraction.
Run --preflight-only to inventory prerequisites without importing CUDA modules.
"""
from __future__ import annotations

import argparse
import inspect
import json
import platform
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import numpy as np
import torch
from torch import Tensor, nn

from mamba3_tracker.deployment.checkpoint import (
    PUBLIC_BEST200_SHA256,
    load_native_state,
    read_checkpoint,
    resolve_flow_config,
    sha256,
)
from mamba3_tracker.deployment.metrics import (
    error_distribution,
    evaluate_gate,
    paired_scores,
)
from mamba3_tracker.deployment.protocol import (
    EXPECTED_FRAMES,
    fixed_manifest,
    preflight,
    resolve_dino_revision,
)
from mamba3_tracker.deployment.runtime import (
    INPUT_NAMES,
    OUTPUT_NAMES,
    run_chunked,
    session_for,
)
from mamba3_tracker.model.heads import TrackerOutputs
from mamba3_tracker.model.onnx_refiner import reference_depth

if TYPE_CHECKING:
    from mamba3_tracker.model.depth_refined_tracker import Mamba3V35Refiner


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class PairedOnnxRefiner(nn.Module):
    """Native DINO runs once; ORT consumes exactly the captured frozen features."""

    def __init__(self, native: Mamba3V35Refiner, onnx_path: Path,
                 track_chunk: int, threads: int, memory_budget_mib: int) -> None:
        super().__init__()
        self.native = native
        self.session = session_for(onnx_path, threads)
        self.track_chunk = track_chunk
        self.memory_budget_mib = memory_budget_mib
        self.native_output: TrackerOutputs | None = None
        self.last_native: list[np.ndarray] = []
        self.last_onnx: list[np.ndarray] = []
        self.last_errors: dict[str, float] = {}

    @torch.no_grad()
    def forward(self, ray: Tensor, z_raw: Tensor, vis: Tensor, uv: Tensor,
                depth_map: Tensor, images: Tensor, K: Tensor) -> TrackerOutputs:
        self.native_output = None
        self.last_native, self.last_onnx, self.last_errors = [], [], {}
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
        feed = {name: value.detach().cpu().float().numpy()
                for name, value in zip(INPUT_NAMES, inputs, strict=True)}
        arrays = run_chunked(self.session, feed, self.track_chunk, self.memory_budget_mib)
        expected = (reference.xyz, reference.uv, reference.vis_logits, reference.delta_uv)
        native_arrays: list[np.ndarray] = []
        for name, ref, actual in zip(OUTPUT_NAMES, expected, arrays, strict=True):
            if ref is None:
                raise RuntimeError(f"Native output {name} is missing")
            value = ref.detach().float().cpu().numpy()
            if value.shape != actual.shape or not np.isfinite(value).all():
                raise RuntimeError(f"Invalid native output: {name}")
            native_arrays.append(value)
            self.last_errors[name] = float(np.max(np.abs(value - actual)))
        self.native_output, self.last_native, self.last_onnx = reference, native_arrays, arrays
        xyz, refined_uv, logits, delta = (torch.from_numpy(array).to(ray.device) for array in arrays)
        return TrackerOutputs(xyz=xyz, uv=refined_uv, vis_logits=logits,
                              spawn_logits=logits, delta_uv=delta)


def load_run_config(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    from mamba3_tracker.train.config import load_config
    return load_config(path)


def run_evaluation(args: argparse.Namespace, state: dict[str, Any], files: dict[str, list[str]],
                   flow_cfg: dict[str, Any], revision: str, report: dict[str, Any]) -> bool:
    # Heavy imports are intentionally after preflight, never on the CPU inventory path.
    from eval_metric3d import _infer
    from train_depth_refined_tracker import _build_waft_flow

    from mamba3_tracker.data.frozen_cache import module_fingerprint
    from mamba3_tracker.data.tapvid3d import load_clip
    from mamba3_tracker.eval.tapvid3d_eval import aggregate
    from mamba3_tracker.model.depth_refined_tracker import Mamba3V35Refiner

    config = state["cfg"]["model"]
    keys = inspect.signature(Mamba3V35Refiner).parameters
    kwargs = {key: value for key, value in config.items() if key in keys}
    kwargs["dino_revision"] = revision
    native = Mamba3V35Refiner(**kwargs)
    report["weight_loading"] = load_native_state(native, state)
    actual_revision = getattr(getattr(native.dino.backbone, "config", None), "_commit_hash", None)
    if actual_revision != revision:
        raise RuntimeError("Loaded DINO commit does not match the immutable requested revision")
    report["dino"] = {"model": config["dino_model"], "revision": actual_revision,
                      "fingerprint": module_fingerprint(native.dino.backbone)}
    native = native.cuda().eval()
    paired = PairedOnnxRefiner(native, args.onnx, args.track_chunk, args.threads, args.memory_budget_mib).eval()
    flow = _build_waft_flow(torch.device("cuda"), scale=flow_cfg["scale"], iters=flow_cfg["iters"])
    waft_root = Path(__file__).resolve().parents[1] / "third_party/WAFT"
    report["waft_checkpoint_sha256"] = sha256(waft_root / "ckpts/waft_a1_recommended.pth")
    native_rows: dict[str, list[dict[str, float]]] = {s: [] for s in files}
    onnx_rows: dict[str, list[dict[str, float]]] = {s: [] for s in files}
    differences: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    completed: set[tuple[str, str]] = set()
    frames = 0
    visible_errors: dict[str, dict[str, list[np.ndarray]]] = {s: {"xyz_m": [], "uv_px_896": []} for s in files}
    for subset, names in files.items():
        for filename in names:
            try:
                clip = load_clip(args.data_root / subset / filename)
                tracks, visibility = _infer(
                    method="v35", model=paired, flow_model=flow, clip=clip, device=torch.device("cuda"),
                    image_size=int(config["image_size"]), fb_alpha=float(flow_cfg["fb_alpha"]),
                    fb_beta=float(flow_cfg["fb_beta"]), da3_depth_root=args.depth_root,
                    max_frames=0, vis_source="flow")
                reference = paired.native_output
                if reference is None:
                    raise RuntimeError("Native output was not captured")
                ref_tracks = reference.xyz[0].transpose(0, 1).detach().float().cpu().numpy()
                ref_metrics, metrics = paired_scores(clip, ref_tracks, tracks, visibility)
                native_rows[subset].append(ref_metrics)
                onnx_rows[subset].append(metrics)
                visible = clip.visibility.detach().cpu().numpy()[None]
                distributions = {}
                for index, label in ((0, "xyz_m"), (1, "uv_px_896")):
                    first, second = paired.last_native[index], paired.last_onnx[index]
                    distributions[label] = error_distribution(first, second, visible)
                    norm = np.linalg.norm(second.astype(np.float64) - first.astype(np.float64), axis=-1)
                    visible_errors[subset][label].append(norm[visible > 0.5])
                differences.append({"subset": subset, "clip_id": clip.clip_id, "filename": filename,
                                    "frames": clip.F, "tracks": clip.N_q, "native": ref_metrics, "onnx": metrics,
                                    "max_abs_output_error": paired.last_errors,
                                    "gt_visible_output_difference": distributions,
                                    "average_jaccard_delta": metrics["average_jaccard"] - ref_metrics["average_jaccard"],
                                    "metric_average_jaccard_delta": metrics["metric_average_jaccard"] - ref_metrics["metric_average_jaccard"]})
                completed.add((subset, filename))
                frames += clip.F
                print(f"[paired] {subset}/{filename}: native metric-AJ={ref_metrics['metric_average_jaccard']:.8f}"
                      f" ONNX={metrics['metric_average_jaccard']:.8f}", flush=True)
            except Exception as error:  # noqa: BLE001 -- record every failure; the acceptance gate rejects it.
                # Do not serialize arbitrary exception text (HTTP responses may contain secrets).
                failures.append({"clip": f"{subset}/{filename}", "error_type": type(error).__name__})
                print(f"[paired] FAILED {subset}/{filename}: {type(error).__name__}", flush=True)
            write_json(args.out_dir / "clip_comparison.json", differences)
            write_json(args.out_dir / "failures.json", failures)
    results = {}
    for label, rows in (("native", native_rows), ("onnx", onnx_rows)):
        per_subset = {subset: aggregate(values) for subset, values in rows.items() if values}
        results[label] = {"per_subset": per_subset, "overall": aggregate(list(per_subset.values())) if per_subset else {}}
    baseline = json.loads(args.baseline.read_text())["reference"]
    gate = evaluate_gate(results, baseline, completed=completed,
                         expected={(subset, name) for subset, names in files.items() for name in names},
                         failures=len(failures), frames=frames, expected_frames=EXPECTED_FRAMES)
    distributions = {}
    for subset, groups in visible_errors.items():
        distributions[subset] = {}
        for label, values in groups.items():
            if values:
                combined = np.concatenate(values)
                distributions[subset][label] = {"count": int(combined.size), "mean": float(combined.mean()),
                                               "p50": float(np.percentile(combined, 50)),
                                               "p95": float(np.percentile(combined, 95)), "max": float(combined.max())}
    report.update(results=results, clips=len(completed), frames=frames, failures=len(failures),
                  error_distributions=distributions, acceptance_gate=gate,
                  task_accuracy_verified=gate["passed"], success=gate["passed"])
    return bool(gate["passed"])


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--run-config", type=Path)
    parser.add_argument("--manifest", type=Path, default=root / "configs/v64_metric_reference_minival.json")
    parser.add_argument("--baseline", type=Path, default=root / "doc/training_result_20261001.json")
    parser.add_argument("--data-root", type=Path, default=Path("~/data/tapvid3d"))
    parser.add_argument("--depth-root", type=Path, default=Path("~/data/tapvid3d_da3"))
    parser.add_argument("--out-dir", type=Path, default=root / "result/onnx_best200/paired_reference")
    parser.add_argument("--track-chunk", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-budget-mib", type=int, default=1024)
    parser.add_argument("--dino-revision")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if min(args.track_chunk, args.threads, args.memory_budget_mib) < 1:
        parser.error("chunk, threads and memory budget must be positive")
    for key in ("ckpt", "onnx", "manifest", "baseline", "data_root", "depth_root", "out_dir"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report: dict[str, Any] = {"success": False, "task_accuracy_verified": False, "status": "started",
                              "clips": 0, "frames": 0, "scope": "fixed 9-clip monitoring reference, not full 150-clip minival",
                              "visibility_source": "same WAFT flow mask", "front_end_shared": True,
                              "precision": {"native": "original CUDA/BF16 mixer", "onnx": "FP32 CPU"}}
    report_path = args.out_dir / "comparison.json"
    write_json(report_path, report)
    try:
        torch.set_num_threads(args.threads)
        state = read_checkpoint(args.ckpt)
        if state["step"] != 200 or state.get("weights_mode") != "eval":
            raise ValueError("This protocol requires the selected best200 eval weights")
        source_hash = sha256(args.ckpt)
        if state.get("export", {}).get("excluded_prefixes") and source_hash != PUBLIC_BEST200_SHA256:
            raise ValueError("Exported PT differs from the published best200 artifact")
        flow_cfg = resolve_flow_config(state, load_run_config(args.run_config))
        revision = resolve_dino_revision(state, args.dino_revision)
        files = fixed_manifest(args.manifest)
        session = session_for(args.onnx, args.threads)
        metadata = session.get_modelmeta().custom_metadata_map
        if metadata.get("source_checkpoint_sha256") != source_hash:
            raise ValueError("ONNX and native checkpoint provenance do not match")
        if metadata.get("dino_revision") != revision or json.loads(metadata["model_config"]) != state["cfg"]["model"]:
            raise ValueError("ONNX model configuration/DINO provenance does not match")
        del session
        inventory = preflight(files, args.data_root, args.depth_root, root, torch.cuda.is_available())
        report.update(checkpoint_sha256=source_hash, checkpoint_step=200, onnx_sha256=sha256(args.onnx),
                      manifest_sha256=sha256(args.manifest), baseline_sha256=sha256(args.baseline),
                      flow_config=flow_cfg, dino_revision=revision, preflight=inventory,
                      track_chunk=args.track_chunk, python=platform.python_version(), torch=torch.__version__,
                      cuda_available=torch.cuda.is_available(), gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        write_json(args.out_dir / "preflight.json", inventory)
        if not inventory["ready"] or args.preflight_only:
            report["status"] = "preflight_ready" if inventory["ready"] else "blocked_prerequisites"
            report["elapsed_s"] = time.perf_counter() - started
            write_json(report_path, report)
            print(json.dumps(inventory, indent=2), flush=True)
            return 0 if inventory["ready"] and args.preflight_only else 2
        passed = run_evaluation(args, state, files, flow_cfg, revision, report)
        report["status"] = "passed" if passed else "failed_acceptance_gate"
        report["elapsed_s"] = time.perf_counter() - started
        write_json(report_path, report)
        return 0 if passed else 1
    except Exception as error:
        report.update(status="failed", failure_type=type(error).__name__, elapsed_s=time.perf_counter() - started)
        write_json(report_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
