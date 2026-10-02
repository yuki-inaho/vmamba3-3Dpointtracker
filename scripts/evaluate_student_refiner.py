"""Student ONNX CUDA proof and reproducible timing; NumPy-only replay path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from mamba3_tracker.deployment.runtime import (
    INPUT_NAMES,
    OUTPUT_NAMES,
    session_for,
    validate_feed,
)
from mamba3_tracker.deployment.student_runtime import run_student_chunked

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require_complete(observed, expected) -> None:
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise ValueError("Duplicate, unexpected or missing evaluation clips")


def summarize_timings(values, warmup: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if warmup < 0 or len(values) <= warmup or not np.isfinite(values).all():
        raise ValueError("Invalid timing samples")
    return {
        "p50_ms": float(np.median(values[warmup:])),
        "p95_ms": float(np.percentile(values[warmup:], 95)),
        "trials": len(values) - warmup,
        "warmup": warmup,
    }


def fixture_specs():
    shapes = [
        (1, 1, 1),
        (2, 8, 7),
        (1, 31, 32),
        (1, 128, 7),
        (1, 257, 1),
        (1, 300, 7),
        (1, 600, 32),
        (1, 600, 900),
    ]
    return [dict(shape=s, invisible=False, benchmark=s[2] <= 32) for s in shapes] + [
        dict(shape=(1, 31, 7), invisible=True, benchmark=False)
    ]


def prepare_fixtures(args):
    import torch
    from scripts.export_refiner_onnx import synthetic_inputs, as_feed, compare
    from mamba3_tracker.deployment.student_checkpoint import load_student

    torch.set_num_threads(4)
    model = load_student(args.student_dir / "student_refiner_deploy.pt")
    onnx_path = args.student_dir / "student_refiner_fp32.onnx"
    session = session_for(onnx_path)
    manifest: dict = {
        "onnx_sha256": digest(onnx_path),
        "checkpoint_sha256": digest(args.student_dir / "student_refiner_deploy.pt"),
        "architecture": model.architecture,
        "status": "preparing",
        "cases": [],
    }

    def persist(name, inputs, *, kind, benchmark_case):
        with torch.no_grad():
            parts = [
                model(
                    *(
                        value[:, :, i : i + 32] if j < 4 else value
                        for j, value in enumerate(inputs)
                    )
                )
                for i in range(0, inputs[0].shape[2], 32)
            ]
            expected = [torch.cat([p[j] for p in parts], dim=2) for j in range(4)]
        feed = as_feed(inputs)
        error = compare(expected, run_student_chunked(session, feed, 32))
        input_path, expected_path = (
            args.out_dir / (name + "_inputs.npz"),
            args.out_dir / (name + "_expected.npz"),
        )
        np.savez(input_path, **feed)
        np.savez(
            expected_path,
            **{
                name: value.numpy()
                for name, value in zip(OUTPUT_NAMES, expected, strict=True)
            },
        )
        manifest["cases"].append(
            {
                "name": name,
                "inputs": input_path.name,
                "expected": expected_path.name,
                "input_sha256": digest(input_path),
                "expected_sha256": digest(expected_path),
                "cpu_error": error,
                "kind": kind,
                "benchmark": benchmark_case,
            }
        )
        write_json(args.out_dir / "fixture_manifest.json", manifest)
        print(json.dumps({"prepared_gpu_fixture": name}), flush=True)

    for spec in fixture_specs():
        b, f, n = spec["shape"]
        name = f"b{b}_f{f}_n{n}" + ("_invisible" if spec["invisible"] else "")
        inputs = synthetic_inputs(b, f, n, 384, 896, depth_shape=(17, 21))
        if spec["invisible"]:
            inputs[2].zero_()
        persist(name, inputs, kind="synthetic", benchmark_case=spec["benchmark"])
    if args.evaluation_inputs is not None:
        source = json.loads(args.evaluation_inputs.read_text())
        if (
            source["status"] != "complete"
            or source["identity"]["partition"] != "monitor"
        ):
            raise ValueError("Real GPU fixtures require complete monitor preparation")
        selected_subsets = set()
        for row in source["records"]:
            if row["subset"] in selected_subsets:
                continue
            path = Path(row["path"])
            if digest(path) != row["sha256"]:
                raise ValueError("Real input fixture hash mismatch")
            payload = torch.load(path, weights_only=True, map_location="cpu")
            inputs = tuple(payload["inputs"][name].float() for name in INPUT_NAMES)
            persist(
                "real_" + row["subset"],
                inputs,
                kind="real_monitor",
                benchmark_case=False,
            )
            manifest["cases"][-1]["source_sha256"] = row["sha256"]
            selected_subsets.add(row["subset"])
        if selected_subsets != {"adt", "drivetrack", "pstudio"}:
            raise ValueError("Every subset must have a real-video GPU fixture")
    manifest["status"] = "complete"
    write_json(args.out_dir / "fixture_manifest.json", manifest)


def cuda_session(path: Path, out_dir: Path, profile: bool = False):
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider unavailable")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.enable_profiling = profile
    options.profile_file_prefix = str(out_dir / "cuda_profile")
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=[
            ("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0}),
            "CPUExecutionProvider",
        ],
    )
    session.disable_fallback()
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("CUDA initialization failed, no CPU fallback allowed")
    return session


def gpu(args):
    from scripts.verify_refiner_gpu import summarize_profile, compare

    manifest = json.loads((args.out_dir / "fixture_manifest.json").read_text())
    path = args.student_dir / "student_refiner_fp32.onnx"
    if manifest["status"] != "complete" or digest(path) != manifest["onnx_sha256"]:
        raise ValueError("Fixture/ONNX provenance mismatch")
    session = cuda_session(path, args.out_dir, profile=True)
    report = {
        "success": False,
        "status": "running",
        "cases": [],
        "onnx_sha256": digest(path),
    }
    write_json(args.out_dir / "gpu_report.json", report)
    for case in manifest["cases"]:
        inp, exp = args.out_dir / case["inputs"], args.out_dir / case["expected"]
        if (
            digest(inp) != case["input_sha256"]
            or digest(exp) != case["expected_sha256"]
        ):
            raise ValueError("GPU fixture hash mismatch")
        with np.load(inp) as source:
            feed = {name: source[name] for name in INPUT_NAMES}
        with np.load(exp) as source:
            expected = [source[name] for name in OUTPUT_NAMES]
        observed = run_student_chunked(session, feed, 32)
        row = {
            "name": case["name"],
            "errors": compare(expected, observed, OUTPUT_NAMES),
        }
        if feed["ray"].shape[1] == 31:
            row["chunk1_errors"] = compare(
                expected, run_student_chunked(session, feed, 1), OUTPUT_NAMES
            )
            order = np.arange(feed["ray"].shape[2])[::-1]
            permuted = dict(feed)
            for name in INPUT_NAMES[:4]:
                permuted[name] = np.ascontiguousarray(feed[name][:, :, order])
            row["permutation_errors"] = compare(
                [x[:, :, order] for x in expected],
                run_student_chunked(session, permuted, 7),
                OUTPUT_NAMES,
            )
        report["cases"].append(row)
        print(json.dumps(row), flush=True)
    profile_path = Path(session.end_profiling())
    profile = summarize_profile(json.loads(profile_path.read_text()))
    if profile["cpu_ops"] and not profile["cpu_nodes_have_only_integer_tensors"]:
        raise RuntimeError("CPU node types are not proven integer-only")
    report.update(
        success=True,
        status="complete",
        profile=profile,
        profile_sha256=digest(profile_path),
        scope="Dynamic and declared real-monitor fixtures, student FP32 numerical equivalence; not task accuracy",
    )
    write_json(args.out_dir / "gpu_report.json", report)


def benchmark_chunks(feed, track_chunk):
    if track_chunk < 1:
        raise ValueError("track_chunk must be positive")
    _, _, tracks = validate_feed(feed)
    return [
        {
            name: np.ascontiguousarray(value[:, :, start : start + track_chunk])
            if name in INPUT_NAMES[:4]
            else value
            for name, value in feed.items()
        }
        for start in range(0, tracks, track_chunk)
    ]


def measure_benchmark(operation, provider):
    """Run timed inference and report own-process GPU VRAM when applicable."""
    if provider == "cpu":
        return operation(), {
            "applicable": False,
            "reason": "CPU provider has no GPU memory measurement",
        }
    if provider != "cuda":
        raise ValueError("provider must be cpu or cuda")
    from mamba3_tracker.deployment.gpu_memory import ProcessGpuMemorySampler

    with ProcessGpuMemorySampler(device_index=0, interval_seconds=0.01) as sampler:
        result = operation()
    memory = sampler.report()
    memory["includes_session_initialization_peak"] = False
    return result, memory


def _benchmark_cases(args, fixture_dir, manifest, session, is_teacher):
    import onnxruntime as ort
    import resource
    from scripts.verify_refiner_gpu import compare

    results = []
    for case in manifest["cases"]:
        if not case.get("benchmark", True):
            continue
        input_path = fixture_dir / case["inputs"]
        if digest(input_path) != case["input_sha256"]:
            raise ValueError("Benchmark fixture hash mismatch")
        with np.load(input_path) as source:
            feed = {name: source[name] for name in INPUT_NAMES}
        expected_path = fixture_dir / case["expected"]
        if digest(expected_path) != case["expected_sha256"]:
            raise ValueError("Benchmark expected-output hash mismatch")
        with np.load(expected_path) as source:
            expected = [source[name] for name in OUTPUT_NAMES]
        chunks = benchmark_chunks(feed, args.track_chunk)

        def host_run():
            parts = [session.run(list(OUTPUT_NAMES), part) for part in chunks]
            return [
                np.concatenate([part[i] for part in parts], axis=2) for i in range(4)
            ]

        observed = host_run()
        if is_teacher:
            # Teacher numerical fidelity is independently verified, not against student outputs.
            expected = observed
        compare(expected, observed, OUTPUT_NAMES)
        timings = []
        for _ in range(120):
            start = time.perf_counter()
            host_run()
            timings.append((time.perf_counter() - start) * 1000)
        row = {
            "case": case["name"],
            "host_io_inclusive": summarize_timings(timings, 20),
            "shape": list(feed["ray"].shape[:3]),
            "track_chunk": args.track_chunk,
        }
        if args.provider == "cuda":
            shared = {
                name: ort.OrtValue.ortvalue_from_numpy(value, "cuda", 0)
                for name, value in feed.items()
                if name not in INPUT_NAMES[:4]
            }
            bindings, retained = [], []
            for part in chunks:
                binding = session.io_binding()
                values = {
                    **shared,
                    **{
                        name: ort.OrtValue.ortvalue_from_numpy(part[name], "cuda", 0)
                        for name in INPUT_NAMES[:4]
                    },
                }
                retained.append(values)
                for name, value in values.items():
                    binding.bind_ortvalue_input(name, value)
                for name in OUTPUT_NAMES:
                    binding.bind_output(name, "cuda", 0)
                bindings.append(binding)
            timings = []
            for _ in range(120):
                start = time.perf_counter()
                for binding in bindings:
                    session.run_with_iobinding(binding)
                    binding.synchronize_outputs()
                timings.append((time.perf_counter() - start) * 1000)
            row["model_only"] = summarize_timings(timings, 20)
            parts = [binding.copy_outputs_to_cpu() for binding in bindings]
            compare(
                expected,
                [np.concatenate([part[i] for part in parts], axis=2) for i in range(4)],
                OUTPUT_NAMES,
            )
            del bindings, retained, shared, values, binding
        else:
            row["model_only"] = row["host_io_inclusive"]
        results.append(row)
        row["process_peak_rss_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
    return results


def benchmark(args):
    fixture_dir = args.fixture_dir
    if fixture_dir is None:
        raise ValueError("benchmark requires --fixture-dir with verified inputs")
    manifest = json.loads((fixture_dir / "fixture_manifest.json").read_text())
    path = args.student_dir / "student_refiner_fp32.onnx"
    if digest(path) != manifest["onnx_sha256"]:
        raise ValueError("Benchmark ONNX provenance mismatch")
    is_teacher = args.teacher_onnx is not None
    if is_teacher:
        if not args.teacher_sha256 or digest(args.teacher_onnx) != args.teacher_sha256:
            raise ValueError(
                "Teacher benchmark requires an explicitly pinned ONNX hash"
            )
        path = args.teacher_onnx
    gpu_exclusivity = None
    if args.provider == "cuda":
        from mamba3_tracker.deployment.gpu_memory import require_exclusive_compute

        gpu_exclusivity = require_exclusive_compute(device_index=0)
    session = (
        cuda_session(path, args.out_dir)
        if args.provider == "cuda"
        else session_for(path, 4)
    )
    results, gpu_memory = measure_benchmark(
        lambda: _benchmark_cases(args, fixture_dir, manifest, session, is_teacher),
        args.provider,
    )
    write_json(
        args.out_dir / ("benchmark_teacher.json" if is_teacher else "benchmark.json"),
        {
            "provider": args.provider,
            "threads": 4,
            "onnx_sha256": digest(path),
            "model": "teacher" if is_teacher else "student",
            "onnx_bytes": path.stat().st_size,
            "track_chunk": args.track_chunk,
            "gpu_peak_memory": gpu_memory,
            "gpu_exclusivity": gpu_exclusivity,
            "results": results,
            "excluded_cases": [
                c["name"] for c in manifest["cases"] if not c.get("benchmark", True)
            ],
            "scope": "Seven declared synthetic B/F/N cases; extreme and real cases are correctness-only",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--student-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["gpu", "benchmark", "monitor", "accuracy"], required=True
    )
    parser.add_argument("--provider", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prepare-fixtures", action="store_true")
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--evaluation-inputs", type=Path)
    parser.add_argument(
        "--teacher-onnx",
        type=Path,
        help="Benchmark the pinned teacher on the identical fixtures instead",
    )
    parser.add_argument("--teacher-sha256")
    parser.add_argument(
        "--track-chunk",
        type=int,
        default=1,
        help="Identical track chunk for student/teacher benchmark",
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="Public student checkpoint, monitor only"
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.prepare_fixtures:
        if args.mode != "gpu":
            raise ValueError("Fixture preparation requires gpu mode")
        prepare_fixtures(args)
    elif args.mode == "gpu":
        if args.provider != "cuda":
            raise ValueError("GPU verification must use CUDA")
        gpu(args)
    elif args.mode == "benchmark":
        benchmark(args)
    else:
        from scripts.evaluate_student_accuracy import evaluate

        evaluate(args)


if __name__ == "__main__":
    main()
