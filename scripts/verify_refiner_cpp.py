"""Build-independent C++ subprocess tests against actual-weight FP32 and ORT outputs."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from export_refiner_onnx import as_feed, compare, synthetic_inputs

from mamba3_tracker.deployment.checkpoint import load_portable, read_checkpoint, sha256
from mamba3_tracker.deployment.runtime import OUTPUT_NAMES, run_chunked, session_for


def invoke(binary: Path, model: Path, inputs: Path, outputs: Path, chunk: int,
           threads: int, budget: int = 1024) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(binary.resolve()), str(model.resolve()), str(inputs), str(outputs),
                           str(chunk), str(threads), str(budget)], text=True, capture_output=True,
                          timeout=180, check=False)


def write_inputs(directory: Path, feed: dict[str, np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, array in feed.items():
        np.save(directory / f"{name}.npy", array, allow_pickle=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('{"success": false, "status": "started"}\n')
    torch.set_num_threads(args.threads)
    model = load_portable(read_checkpoint(args.ckpt))
    session = session_for(args.onnx, args.threads)
    rows, rejected = [], []
    with tempfile.TemporaryDirectory(prefix="refiner-cpp-") as temporary:
        root = Path(temporary)
        for batch, frames, tracks, depth in [(1, 1, 1, (11, 17)), (1, 8, 7, (32, 48)),
                                            (2, 13, 3, (19, 23)), (1, 128, 2, (17, 23))]:
            tensors = synthetic_inputs(batch, frames, tracks, 384, 896, depth, seed=frames + 200)
            feed = as_feed(tensors)
            with torch.inference_mode():
                reference = model(*tensors)
            ort = run_chunked(session, feed, track_chunk=1)
            write_inputs(root / "inputs", feed)
            for chunk in sorted({1, 2, tracks}):
                process = invoke(args.binary, args.onnx, root / "inputs", root / "outputs", chunk, args.threads)
                if process.returncode:
                    raise RuntimeError(process.stderr)
                result = [np.load(root / "outputs" / f"{name}.npy", allow_pickle=False) for name in OUTPUT_NAMES]
                native_error = compare(reference, result)
                python_error = compare(ort, result)
                rows.append({"B": batch, "F": frames, "N": tracks, "depth_shape": depth,
                             "chunk": chunk, "max_abs_error_vs_portable_fp32": native_error,
                             "max_abs_error_vs_python_ort": python_error, "all_finite": True,
                             "process": json.loads(process.stdout)})
                print(f"[C++] B={batch} F={frames} N={tracks} chunk={chunk}: {native_error}", flush=True)
            del tensors, feed, reference, ort, result
        good = as_feed(synthetic_inputs(1, 3, 5, 384, 896, (11, 19), seed=200))
        faults = ("missing_input", "nan", "wrong_z_ref", "float64", "fortran", "nonbinary_vis",
                  "truncated", "wrong_shape", "zero_chunk", "zero_threads", "bad_intrinsics")
        for fault in faults:
            input_dir = root / "bad_inputs"
            shutil.rmtree(input_dir, ignore_errors=True)
            write_inputs(input_dir, good)
            chunk, threads = 2, args.threads
            if fault == "missing_input":
                (input_dir / "depth_map.npy").unlink()
            elif fault == "nan":
                value = good["ray"].copy()
                value.flat[0] = np.nan
                np.save(input_dir / "ray.npy", value)
            elif fault == "wrong_z_ref":
                np.save(input_dir / "z_ref.npy", good["z_ref"] + np.float32(0.1))
            elif fault == "float64":
                np.save(input_dir / "z_raw.npy", good["z_raw"].astype(np.float64))
            elif fault == "fortran":
                np.save(input_dir / "ray.npy", np.asfortranarray(good["ray"]))
            elif fault == "nonbinary_vis":
                np.save(input_dir / "visibility.npy", np.full_like(good["visibility"], 0.5))
            elif fault == "truncated":
                path = input_dir / "uv.npy"
                path.write_bytes(path.read_bytes()[:-4])
            elif fault == "wrong_shape":
                np.save(input_dir / "uv.npy", good["uv"][:, :, :1])
            elif fault == "zero_chunk":
                chunk = 0
            elif fault == "zero_threads":
                threads = 0
            elif fault == "bad_intrinsics":
                value = good["intrinsics"].copy()
                value[0, 0, 0] = 0
                np.save(input_dir / "intrinsics.npy", value)
            process = invoke(args.binary, args.onnx, input_dir, root / "bad_outputs", chunk, threads)
            if process.returncode == 0:
                raise RuntimeError(f"C++ incorrectly accepted {fault}")
            rejected.append({"case": fault, "exit_code": process.returncode,
                             "stderr": process.stderr.strip()})
        # The long-sequence guard must fail before execution, not silently shorten F.
        feed = as_feed(synthetic_inputs(1, 128, 2, 384, 896, (7, 11)))
        write_inputs(root / "budget_inputs", feed)
        process = invoke(args.binary, args.onnx, root / "budget_inputs", root / "budget_outputs", 2, args.threads, 1)
        if process.returncode == 0 or "workspace" not in process.stderr:
            raise RuntimeError("C++ workspace guard did not reject the oversized request")
        rejected.append({"case": "workspace_guard", "exit_code": process.returncode,
                         "stderr": process.stderr.strip()})
    report = {"success": True, "source_checkpoint_sha256": sha256(args.ckpt),
              "onnx_sha256": sha256(args.onnx), "binary_sha256": sha256(args.binary),
              "provider": "CPUExecutionProvider", "scope": "actual trained refiner; synthetic inputs; not 9-clip task accuracy",
              "positive_cases": rows, "rejected_invalid_cases": rejected,
              "rtol": 2e-4, "atol": 5e-5, "task_accuracy_verified": False}
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"C++ verification passed: {len(rows)} positive and {len(rejected)} invalid-input cases.")


if __name__ == "__main__":
    main()
