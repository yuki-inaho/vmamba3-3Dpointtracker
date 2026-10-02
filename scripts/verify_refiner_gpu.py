"""Verify actual refiner weights on ORT CUDA EP, without replacing CPU deployment.

Prepare portable-FP32 fixtures in the root environment, then verify them in an
isolated onnxruntime-gpu environment. See docs/onnx_export_design_20261002.md.
These synthetic checks do not certify real-video AJ or native BF16 equivalence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


MAJOR_COMPUTE = {
    "MatMul",
    "FusedMatMul",
    "Gemm",
    "GridSample",
    "LayerNormalization",
    "SimplifiedLayerNormalization",
    "SkipLayerNormalization",
    "Conv",
    "Sin",
    "Cos",
    "Exp",
    "ReduceMean",
}


def summarize_profile(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Reject CPU-only sessions and major compute assigned to CPU."""
    providers: Counter[str] = Counter()
    cuda_ops: Counter[str] = Counter()
    cpu_ops: Counter[str] = Counter()
    cpu_types: Counter[str] = Counter()
    cpu_unknown_types = 0
    for item in events:
        args = item.get("args", {})
        provider, op = args.get("provider"), args.get("op_name")
        if item.get("cat") != "Node" or not provider or not op:
            continue
        providers[provider] += 1
        if provider == "CUDAExecutionProvider":
            cuda_ops[op] += 1
        elif provider == "CPUExecutionProvider":
            cpu_ops[op] += 1
            tensor_types = [
                dtype
                for tensor in (
                    *args.get("input_type_shape", []),
                    *args.get("output_type_shape", []),
                )
                for dtype in tensor
            ]
            cpu_types.update(tensor_types)
            if not tensor_types:
                cpu_unknown_types += 1
            if any(
                dtype.startswith(("float", "bfloat", "double"))
                for dtype in tensor_types
            ):
                raise RuntimeError(f"CPU compute uses floating tensors: {op}")
    if not (set(cuda_ops) & MAJOR_COMPUTE):
        raise RuntimeError("No executed CUDA compute nodes in the ORT profile")
    if set(cpu_ops) & MAJOR_COMPUTE:
        raise RuntimeError(f"Major CPU compute nodes: {dict(cpu_ops)}")
    return {
        "cuda_compute_verified": True,
        "whole_graph_gpu_only": not bool(cpu_ops),
        "node_events_by_provider": dict(providers),
        "cuda_ops": dict(sorted(cuda_ops.items())),
        "cpu_ops": dict(sorted(cpu_ops.items())),
        "cpu_tensor_types": dict(cpu_types),
        "cpu_nodes_have_only_integer_tensors": bool(cpu_types)
        and not cpu_unknown_types
        and all(dtype.startswith(("int", "uint")) for dtype in cpu_types),
        "major_compute_on_cpu": False,
    }


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def compare(expected, observed, output_names) -> dict[str, float]:
    errors = {}
    for name, ref, actual in zip(output_names, expected, observed, strict=True):
        if ref.shape != actual.shape or ref.dtype != actual.dtype:
            raise RuntimeError(f"Shape/dtype mismatch: {name}")
        if not np.isfinite(ref).all() or not np.isfinite(actual).all():
            raise RuntimeError(f"Nonfinite output: {name}")
        np.testing.assert_allclose(actual, ref, rtol=2e-4, atol=5e-5)
        errors[name] = float(np.max(np.abs(actual - ref)))
    return errors


def check_metadata(session, manifest) -> None:
    metadata = session.get_modelmeta().custom_metadata_map
    if metadata.get("source_checkpoint_sha256") != manifest["checkpoint_sha256"]:
        raise ValueError("ONNX checkpoint provenance mismatch")
    if metadata.get("source_checkpoint_step") != str(manifest["checkpoint_step"]):
        raise ValueError("ONNX checkpoint step mismatch")


def prepare(args) -> None:
    import torch
    from export_refiner_onnx import as_feed, synthetic_inputs

    from mamba3_tracker.deployment.checkpoint import load_portable, read_checkpoint
    from mamba3_tracker.deployment.runtime import OUTPUT_NAMES, run_feed, session_for

    torch.set_num_threads(args.threads)
    state = read_checkpoint(args.ckpt)
    model = load_portable(state)
    session = session_for(args.onnx, args.threads)
    manifest = {
        "schema_version": 1,
        "checkpoint_step": state["step"],
        "checkpoint_sha256": sha256(args.ckpt),
        "onnx_sha256": sha256(args.onnx),
        "scope": "actual checkpoint weights; synthetic inputs; not task accuracy",
        "cases": [],
    }
    check_metadata(session, manifest)
    for batch, frames, tracks, depth in (
        (1, 1, 1, (24, 32)),
        (1, 8, 7, (32, 48)),
        (2, 13, 3, (48, 32)),
        (1, 128, 2, (17, 23)),
    ):
        inputs = synthetic_inputs(
            batch,
            frames,
            tracks,
            384,
            model.image_size,
            depth,
            seed=frames,
        )
        with torch.inference_mode():
            expected = [x.numpy() for x in model(*inputs)]
        feed = as_feed(inputs)
        cpu = run_feed(session, feed)
        error = compare(expected, cpu, OUTPUT_NAMES)
        name = f"b{batch}_f{frames}_n{tracks}"
        input_path, expected_path = (
            args.out_dir / f"{name}_inputs.npz",
            args.out_dir / f"{name}_expected.npz",
        )
        np.savez_compressed(input_path, **feed)
        np.savez_compressed(
            expected_path, **dict(zip(OUTPUT_NAMES, expected, strict=True))
        )
        manifest["cases"].append(
            {
                "name": name,
                "batch": batch,
                "frames": frames,
                "tracks": tracks,
                "depth_shape": list(depth),
                "inputs": input_path.name,
                "expected": expected_path.name,
                "inputs_sha256": sha256(input_path),
                "expected_sha256": sha256(expected_path),
                "portable_vs_cpu_max_abs_error": error,
            }
        )
        print(f"[prepare] step{state['step']} {name}: portable/CPU passed", flush=True)
    write_json(args.out_dir / "fixture_manifest.json", manifest)


def verify(args) -> None:
    import onnxruntime as ort

    from mamba3_tracker.deployment.runtime import (
        OUTPUT_NAMES,
        run_chunked,
        run_feed,
        session_for,
    )

    report_path = args.out_dir / "gpu_report.json"
    write_json(report_path, {"success": False, "status": "started"})
    manifest = json.loads((args.out_dir / "fixture_manifest.json").read_text())
    if sha256(args.onnx) != manifest["onnx_sha256"]:
        raise ValueError("ONNX SHA256 differs from the prepared fixtures")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("Install onnxruntime-gpu; CUDAExecutionProvider is absent")
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.enable_profiling = True
    options.profile_file_prefix = str(args.out_dir / "cuda_profile")
    session = ort.InferenceSession(
        str(args.onnx),
        sess_options=options,
        providers=[
            ("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0}),
            "CPUExecutionProvider",
        ],
    )
    session.disable_fallback()
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("CUDA EP initialization failed; refusing CPU fallback")
    check_metadata(session, manifest)
    cpu_session = session_for(args.onnx, args.threads)
    cases = []
    try:
        for entry in manifest["cases"]:
            inputs, expected_file = (
                args.out_dir / entry["inputs"],
                args.out_dir / entry["expected"],
            )
            if (
                sha256(inputs) != entry["inputs_sha256"]
                or sha256(expected_file) != entry["expected_sha256"]
            ):
                raise ValueError("Prepared fixture SHA256 mismatch")
            with np.load(inputs, allow_pickle=False) as archive:
                feed = {name: archive[name] for name in archive.files}
            with np.load(expected_file, allow_pickle=False) as archive:
                expected = [archive[name] for name in OUTPUT_NAMES]
            gpu = run_feed(session, feed)
            cpu = run_feed(cpu_session, feed)
            chunked = run_chunked(session, feed, track_chunk=1)
            cases.append(
                {
                    "name": entry["name"],
                    "batch": entry["batch"],
                    "frames": entry["frames"],
                    "tracks": entry["tracks"],
                    "depth_shape": entry["depth_shape"],
                    "gpu_vs_portable_fp32_max_abs_error": compare(
                        expected, gpu, OUTPUT_NAMES
                    ),
                    "gpu_vs_cpu_ort_max_abs_error": compare(cpu, gpu, OUTPUT_NAMES),
                    "gpu_chunk1_vs_full_max_abs_error": compare(
                        gpu, chunked, OUTPUT_NAMES
                    ),
                }
            )
            print(
                f"[cuda] step{manifest['checkpoint_step']} {entry['name']}: all comparisons passed",
                flush=True,
            )
    finally:
        profile = Path(session.end_profiling())
    summary = summarize_profile(json.loads(profile.read_text()))
    if sha256(args.onnx) != manifest["onnx_sha256"]:
        raise ValueError("ONNX changed during verification")
    report = {
        "success": True,
        "scope": manifest["scope"],
        "task_accuracy_verified": False,
        "real_video_gpu_aj_evaluated": False,
        "checkpoint_step": manifest["checkpoint_step"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "onnx_sha256": manifest["onnx_sha256"],
        "onnx_unchanged": True,
        "onnxruntime_gpu_version": ort.__version__,
        "providers": session.get_providers(),
        "cuda_provider_options": session.get_provider_options()[
            "CUDAExecutionProvider"
        ],
        "rtol": 2e-4,
        "atol": 5e-5,
        "cases": cases,
        "profile": profile.name,
        "profile_sha256": sha256(profile),
        "profile_summary": summary,
    }
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "success": True,
                "checkpoint_step": manifest["checkpoint_step"],
                **summary,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "verify"), required=True)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or (args.stage == "prepare" and args.ckpt is None):
        parser.error("threads must be positive; prepare requires --ckpt")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        prepare(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
