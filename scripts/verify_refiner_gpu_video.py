"""Replay complete, preselected real-video frontend inputs on ORT CUDA EP.

prepare/score use the root PyTorch environment; verify is NumPy/ORT-only and
runs in the isolated GPU environment. Original Desktop comparisons are read-only.
"""

from __future__ import annotations

import argparse
from contextlib import chdir
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

import numpy as np

from mamba3_tracker.deployment.runtime import OUTPUT_NAMES, run_chunked, validate_feed
from verify_refiner_gpu import check_metadata, compare, sha256, summarize_profile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "tracking_comparison_20261002"
DESTINATION = ROOT / "result/onnx_gpu_video_20261002"
COVERAGE = {"pstudio": (150, 460), "drivetrack": (31, 256), "adt": (300, 900)}
PREDICTION_NAMES = ("xyz", "uv896", "vis_logits", "delta_uv")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def feed_signature(feed: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
        for key, value in feed.items()
    }


def verify_feed(feed, signature, frames, tracks) -> None:
    if validate_feed(feed) != (1, frames, tracks):
        raise ValueError("Incomplete frame/track coverage")
    if feed_signature(feed) != signature:
        raise ValueError("Input tensor signature changed")


def check_membership(cases: list[dict[str, Any]]) -> None:
    if len(cases) != 3 or {c["subset"] for c in cases} != set(COVERAGE):
        raise ValueError("Expected fixed three-subset membership")
    for case in cases:
        if (case["frames"], case["tracks"]) != COVERAGE[case["subset"]]:
            raise ValueError("Incomplete frame/track coverage")


def checked_json(path: Path, digest: str) -> dict[str, Any]:
    if sha256(path) != digest:
        raise ValueError("JSON provenance changed")
    return json.loads(path.read_text())


def load_arrays(path: Path, digest: str) -> dict[str, np.ndarray]:
    if sha256(path) != digest:
        raise ValueError("NPZ provenance changed")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def prediction_outputs(arrays, method: str) -> list[np.ndarray]:
    values = []
    for name in PREDICTION_NAMES:
        value = arrays[f"{method}_{name}"]
        values.append(
            value.T[None] if value.ndim == 2 else value.transpose(1, 0, 2)[None]
        )
    return values


def invoke_from(directory: Path, callback, *args, **kwargs):
    """WAFT initialization uses relative weights even after its first import."""
    with chdir(directory):
        return callback(*args, **kwargs)


def prepare(args) -> None:
    import torch
    import compare_release_tracking as comparison
    import eval_onnx_refiner as paired
    import train_depth_refined_tracker as training

    manifest = comparison.read_manifest(args.source)
    check_membership(manifest["clips"])
    # Preserve the exact original manifest bytes; compare.infer binds reports to it.
    (args.out_dir / "manifest.json").write_bytes(
        (args.source / "manifest.json").read_bytes()
    )
    fixtures: dict[str, Any] = {
        "schema_version": 1,
        "ready": False,
        "scope": manifest["scope"],
        "checkpoint_step": 80,
        "checkpoint_sha256": manifest["new_sha256"],
        "onnx_sha256": manifest["onnx_sha256"],
        "source_manifest_sha256": sha256(args.source / "manifest.json"),
        "onnx": manifest["onnx"],
        "track_chunk": manifest["onnx_track_chunk"],
        "memory_budget_mib": manifest["onnx_memory_budget_mib"],
        "cases": [],
    }
    fixture_path = args.out_dir / "fixture_manifest.json"
    write_json(fixture_path, fixtures)
    for clip in manifest["clips"]:
        subset = clip["subset"]
        input_file = args.out_dir / f"{subset}_inputs.npz"
        captured = []
        original = paired.run_chunked

        def capture(session, feed, track_chunk, memory_budget_mib):
            if captured:
                raise RuntimeError("Expected one complete refiner feed per clip")
            signature = feed_signature(feed)
            verify_feed(feed, signature, clip["frames"], clip["tracks"])
            np.savez_compressed(input_file, **feed)
            captured.append(signature)
            return original(session, feed, track_chunk, memory_budget_mib)

        original_builder = training._build_waft_flow

        def build_flow(*values, **kwargs):
            return invoke_from(
                ROOT / "third_party/WAFT", original_builder, *values, **kwargs
            )

        with (
            patch.object(paired, "run_chunked", side_effect=capture),
            patch.object(training, "_build_waft_flow", side_effect=build_flow),
        ):
            comparison.infer(args.out_dir, manifest, subset)
        if len(captured) != 1:
            raise RuntimeError("Complete real-video input was not captured")
        report_file = args.out_dir / "reports" / f"{subset}.json"
        report = json.loads(report_file.read_text())
        prediction_file = args.out_dir / "predictions" / f"{subset}.npz"
        reference = load_arrays(prediction_file, report["prediction_sha256"])
        feed = load_arrays(input_file, sha256(input_file))
        if not np.array_equal(feed["visibility"][0].T, reference["flow_visibility"]):
            raise ValueError("Refiner and scoring flow visibility differ")
        old_report = json.loads(
            (args.source / "reports" / f"{subset}.json").read_text()
        )
        aliases = {
            "visibility": "flow_visibility",
            "depth_map": "depth",
            "dino_features": "dino",
            "intrinsics": "K",
        }
        old_matches = {
            key: value == old_report["shared_frontend_inputs"][aliases.get(key, key)]
            for key, value in captured[0].items()
            if key != "z_ref"
        }
        fixtures["cases"].append(
            {
                **clip,
                "inputs": input_file.name,
                "inputs_sha256": sha256(input_file),
                "input_signature": captured[0],
                "cpu_prediction": str(prediction_file.relative_to(args.out_dir)),
                "cpu_prediction_sha256": sha256(prediction_file),
                "prepare_report": str(report_file.relative_to(args.out_dir)),
                "prepare_report_sha256": sha256(report_file),
                "previous_frontend_tensor_matches": old_matches,
                "previous_report_sha256": sha256(
                    args.source / "reports" / f"{subset}.json"
                ),
            }
        )
        write_json(fixture_path, fixtures)
        print(
            f"[prepare] {subset}: full input saved; old frontend matches={all(old_matches.values())}",
            flush=True,
        )
        del feed, reference
        gc.collect()
        torch.cuda.empty_cache()
    check_membership(fixtures["cases"])
    fixtures["ready"] = True
    write_json(fixture_path, fixtures)


def read_fixtures(args) -> dict[str, Any]:
    fixture = json.loads((args.out_dir / "fixture_manifest.json").read_text())
    check_membership(fixture["cases"])
    if not fixture["ready"] or sha256(ROOT / fixture["onnx"]) != fixture["onnx_sha256"]:
        raise ValueError("Incomplete fixtures or changed ONNX")
    if sha256(args.out_dir / "manifest.json") != fixture["source_manifest_sha256"]:
        raise ValueError("Source manifest changed")
    return fixture


def verify(args) -> None:
    import onnxruntime as ort

    report_path = args.out_dir / "gpu_report.json"
    write_json(report_path, {"success": False, "status": "started"})
    fixture = read_fixtures(args)
    fixture_sha = sha256(args.out_dir / "fixture_manifest.json")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is absent")
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.enable_profiling = True
    options.profile_file_prefix = str(args.out_dir / "cuda_video_profile")
    session = ort.InferenceSession(
        str(ROOT / fixture["onnx"]),
        sess_options=options,
        providers=[
            ("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0}),
            "CPUExecutionProvider",
        ],
    )
    session.disable_fallback()
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("CUDA initialization failed; refusing CPU fallback")
    check_metadata(session, fixture)
    cases: list[dict[str, Any]] = []
    try:
        for clip in fixture["cases"]:
            feed = load_arrays(args.out_dir / clip["inputs"], clip["inputs_sha256"])
            verify_feed(feed, clip["input_signature"], clip["frames"], clip["tracks"])
            reference = load_arrays(
                args.out_dir / clip["cpu_prediction"], clip["cpu_prediction_sha256"]
            )
            checked_json(
                args.out_dir / clip["prepare_report"], clip["prepare_report_sha256"]
            )
            if not np.array_equal(
                feed["visibility"][0].T, reference["flow_visibility"]
            ):
                raise ValueError("Scoring mask differs from the prepared input")
            started = time.perf_counter()
            gpu = run_chunked(
                session, feed, fixture["track_chunk"], fixture["memory_budget_mib"]
            )
            elapsed = time.perf_counter() - started
            errors = compare(prediction_outputs(reference, "onnx"), gpu, OUTPUT_NAMES)
            prediction = args.out_dir / f"{clip['subset']}_gpu.npz"
            np.savez_compressed(prediction, **dict(zip(OUTPUT_NAMES, gpu, strict=True)))
            cases.append(
                {
                    "subset": clip["subset"],
                    "frames": clip["frames"],
                    "tracks": clip["tracks"],
                    "gpu_vs_cpu_max_abs_error": errors,
                    "elapsed_s_with_profiling": elapsed,
                    "prediction": prediction.name,
                    "prediction_sha256": sha256(prediction),
                    "inputs_sha256": clip["inputs_sha256"],
                }
            )
            print(
                f"[cuda] {clip['subset']} {clip['frames']}F/{clip['tracks']}N: {errors}",
                flush=True,
            )
            del feed, reference, gpu
    except Exception as error:
        write_json(
            report_path,
            {
                "success": False,
                "failure_type": type(error).__name__,
                "completed_cases": cases,
            },
        )
        raise
    finally:
        profile = Path(session.end_profiling())
    profile_summary = summarize_profile(json.loads(profile.read_text()))
    check_membership(cases)
    if sha256(args.out_dir / "fixture_manifest.json") != fixture_sha:
        raise ValueError("Fixture manifest changed during verification")
    read_fixtures(args)
    report = {
        "success": True,
        "scope": fixture["scope"],
        "real_video_gpu_inference_verified": True,
        "real_video_gpu_aj_evaluated": False,
        "official_acceptance_gate_passed": False,
        "checkpoint_step": 80,
        "checkpoint_sha256": fixture["checkpoint_sha256"],
        "onnx_sha256": fixture["onnx_sha256"],
        "fixture_manifest_sha256": fixture_sha,
        "onnxruntime_gpu_version": ort.__version__,
        "providers": session.get_providers(),
        "cuda_provider_options": session.get_provider_options()[
            "CUDAExecutionProvider"
        ],
        "rtol": 2e-4,
        "atol": 5e-5,
        "track_chunk": fixture["track_chunk"],
        "frames": sum(c["frames"] for c in cases),
        "cases": cases,
        "profile": profile.name,
        "profile_sha256": sha256(profile),
        "profile_summary": profile_summary,
    }
    write_json(report_path, report)


def score(args) -> None:
    from mamba3_tracker.data.tapvid3d import load_clip
    from mamba3_tracker.deployment.metrics import (
        error_distribution,
        score as score_clip,
    )

    write_json(args.out_dir / "summary.json", {"success": False, "status": "started"})
    fixture = read_fixtures(args)
    gpu_report = json.loads((args.out_dir / "gpu_report.json").read_text())
    if not gpu_report["success"] or gpu_report["fixture_manifest_sha256"] != sha256(
        args.out_dir / "fixture_manifest.json"
    ):
        raise ValueError("GPU verification is incomplete or stale")
    check_membership(gpu_report["cases"])
    if sha256(args.out_dir / gpu_report["profile"]) != gpu_report["profile_sha256"]:
        raise ValueError("GPU profile changed")
    rows: list[dict[str, Any]] = []
    for entry in fixture["cases"]:
        subset = entry["subset"]
        source = Path(entry["path"])
        if sha256(source) != entry["raw_sha256"]:
            raise ValueError("Raw clip changed")
        clip = load_clip(source)
        reference = load_arrays(
            args.out_dir / entry["cpu_prediction"], entry["cpu_prediction_sha256"]
        )
        prep = checked_json(
            args.out_dir / entry["prepare_report"], entry["prepare_report_sha256"]
        )
        gpu_entry = next(c for c in gpu_report["cases"] if c["subset"] == subset)
        observed = load_arrays(
            args.out_dir / gpu_entry["prediction"], gpu_entry["prediction_sha256"]
        )
        if set(observed) != set(OUTPUT_NAMES):
            raise ValueError("GPU prediction schema changed")
        gpu = [observed[key] for key in OUTPUT_NAMES]
        compare(prediction_outputs(reference, "onnx"), gpu, OUTPUT_NAMES)
        mask = reference["flow_visibility"]
        scores = {
            key: score_clip(clip, reference[f"{key}_xyz"], mask)
            for key in ("new", "onnx")
        }
        if scores != {key: prep["scores"][key] for key in ("new", "onnx")}:
            raise ValueError("Native/CPU metrics failed to reproduce")
        scores = {
            "native": scores["new"],
            "cpu": scores["onnx"],
            "gpu": score_clip(clip, gpu[0][0].transpose(1, 0, 2), mask),
        }
        differences = {
            name: error_distribution(first[0], second[0], reference["gt_visibility"].T)
            for name, first, second in zip(
                ("xyz_m", "uv_px_896"),
                prediction_outputs(reference, "onnx")[:2],
                gpu[:2],
                strict=True,
            )
        }
        deltas = {key: scores["gpu"][key] - scores["cpu"][key] for key in scores["gpu"]}
        rows.append(
            {
                "subset": subset,
                "frames": clip.F,
                "tracks": clip.N_q,
                "scores": scores,
                "gpu_minus_cpu_metrics": deltas,
                "gpu_vs_cpu_visible_difference": differences,
                "gpu_vs_cpu_max_abs_error": gpu_entry["gpu_vs_cpu_max_abs_error"],
            }
        )
        print(
            f"[score] {subset}: GPU metric-AJ={scores['gpu']['metric_average_jaccard']:.9f}, 3D-AJ={scores['gpu']['average_jaccard']:.9f}",
            flush=True,
        )
    check_membership(rows)
    means = {
        method: {
            key: float(np.mean([row["scores"][method][key] for row in rows]))
            for key in rows[0]["scores"][method]
        }
        for method in ("native", "cpu", "gpu")
    }
    write_json(
        args.out_dir / "summary.json",
        {
            "success": True,
            "scope": fixture["scope"],
            "real_video_gpu_aj_evaluated": True,
            "official_fixed9_evaluated": False,
            "official_full_minival_evaluated": False,
            "independent_test_set": False,
            "frames": sum(r["frames"] for r in rows),
            "clips": len(rows),
            "visibility_source": "same binary WAFT flow mask",
            "rtol": 2e-4,
            "atol": 5e-5,
            "onnx_sha256": fixture["onnx_sha256"],
            "checkpoint_sha256": fixture["checkpoint_sha256"],
            "gpu_report_sha256": sha256(args.out_dir / "gpu_report.json"),
            "per_clip": rows,
            "equal_three_clip_mean": means,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "verify", "score"), required=True
    )
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DESTINATION)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    args.source, args.out_dir = args.source.resolve(), args.out_dir.resolve()
    if args.source == args.out_dir:
        parser.error("Original Desktop comparison must not be overwritten")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "verify": verify, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
