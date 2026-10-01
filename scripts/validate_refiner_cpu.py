"""Reproduce all CPU quality/export/Python/C++ checks in the locked uv environments.

This command never runs DA training or claims nine-clip native CUDA accuracy.
The caller supplies the original checkpoint and the official ONNX Runtime SDK.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "result/onnx_best200_cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    args.ckpt, args.sdk_root, args.out_dir = (p.expanduser().resolve() for p in (args.ckpt, args.sdk_root, args.out_dir))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logs = args.out_dir / "logs"
    logs.mkdir(exist_ok=True)
    report_path = args.out_dir / "quality.json"
    report: dict[str, Any] = {"success": False, "scope": "CPU quality and refiner portability only",
                              "task_accuracy_verified": False, "full_dod_satisfied": False, "stages": []}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    started = time.perf_counter()
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS=str(args.threads))
    root_uv = ["uv", "run", "--locked", "--extra", "onnx"]
    lean_uv = ["uv", "run", "--project", "deploy/onnx_cpu", "--locked"]

    def run(label: str, command: list[str], accepted: tuple[int, ...] = (0,)) -> str:
        print(f"[CPU validation] {label}", flush=True)
        result = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        text = (result.stdout + "\n[stderr]\n" + result.stderr).replace(str(ROOT), "<workspace>")
        (logs / f"{label}.txt").write_text(text)
        report["stages"].append({"stage": label, "returncode": result.returncode,
                                  "passed": result.returncode in accepted})
        if result.returncode not in accepted:
            raise RuntimeError(f"{label} failed (exit {result.returncode}); see its log")
        return result.stdout

    try:
        source_hash = digest(args.ckpt)
        expected = "bcddf045311b76a91bc772e35901acbdb99f397b752ee0ba4beacb3fb1ded5f4"
        if source_hash != expected:
            raise ValueError("The release verification command requires the original public best200")
        root_lock_hash = digest(ROOT / "uv.lock")
        report.update(checkpoint_sha256=source_hash, host=platform.platform(),
                      root_lock_sha256=root_lock_hash)
        run("quality_tools", ["sh", "scripts/check_training_quality.sh"])
        run("lean_lock", ["uv", "lock", "--project", "deploy/onnx_cpu", "--check"])
        versions = run("versions", root_uv + ["python", "-c",
            "import json,sys,torch,numpy,onnx,onnxruntime; "
            "print(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=numpy.__version__,"
            "onnx=onnx.__version__,onnxruntime=onnxruntime.__version__,cuda_available=torch.cuda.is_available())))"])
        report["versions"] = json.loads(versions)
        run("export", root_uv + ["python", "scripts/export_refiner_onnx.py", "--ckpt", str(args.ckpt),
                                  "--out-dir", str(args.out_dir), "--threads", str(args.threads),
                                  "--expected-sha256", source_hash, "--fixture-dir", str(args.out_dir / "fixture")])
        model = args.out_dir / "tracker_mamba3_daoff_best200.onnx"
        build, install = args.out_dir.parent / "cpp_build", args.out_dir.parent / "cpp_runtime"
        run("cmake_configure", ["cmake", "-S", str(ROOT / "cpp/refiner"), "-B", str(build),
                                "-DCMAKE_BUILD_TYPE=Release", f"-DONNXRUNTIME_ROOT={args.sdk_root}",
                                f"-DCMAKE_INSTALL_PREFIX={install}"])
        run("cmake_build", ["cmake", "--build", str(build), "-j2"])
        run("cmake_install", ["cmake", "--install", str(build)])
        run("cpp_verify", root_uv + ["python", "scripts/verify_refiner_cpp.py", "--binary", str(install / "bin/refiner_runner"),
                                      "--onnx", str(model), "--ckpt", str(args.ckpt), "--threads", str(args.threads),
                                      "--report", str(args.out_dir / "cpp_report.json")])
        run("python_no_torch", lean_uv + ["python", "-c",
            "import importlib.util; assert importlib.util.find_spec('torch') is None; print('PyTorch is not installed in the inference environment')"])
        run("lean_inference", lean_uv + ["python", "scripts/run_refiner_onnx.py", "--onnx", str(model),
                                        "--inputs", str(args.out_dir / "fixture/inputs.npz"),
                                        "--outputs", str(args.out_dir / "lean_outputs.npz"),
                                        "--track-chunk", "2", "--threads", str(args.threads)])
        # Direct comparison in another lean process; no exporter or Torch import.
        comparison = (
            "from pathlib import Path; import sys,numpy as np; "
            "p=Path(sys.argv[1]); d=np.load(p/'lean_outputs.npz',allow_pickle=False); "
            "names=('xyz','uv_refined','vis_logits','delta_uv'); "
            "[np.testing.assert_allclose(d[n],np.load(p/'fixture'/('expected_'+n+'.npy'),allow_pickle=False),"
            "rtol=2e-4,atol=5e-5) for n in names]; "
            "assert 'torch' not in sys.modules; print('Independent CPU process: all four outputs match; no Torch import')"
        )
        run("independent_runtime", lean_uv + ["python", "-c", comparison, str(args.out_dir)])
        run("paired_preflight", root_uv + ["python", "scripts/eval_onnx_refiner.py", "--ckpt", str(args.ckpt),
                                          "--onnx", str(model), "--run-config", "configs/v64_official_mamba3_cache_phase.yaml",
                                          "--out-dir", str(args.out_dir / "paired_reference"), "--preflight-only"], (0, 2))
        if digest(args.ckpt) != source_hash or digest(ROOT / "uv.lock") != root_lock_hash:
            raise RuntimeError("Checkpoint or root lock changed during verification")
        report.update(success=True, elapsed_s=time.perf_counter() - started,
                      onnx_sha256=digest(model), lean_lock_sha256=digest(ROOT / "deploy/onnx_cpu/uv.lock"),
                      checkpoint_unchanged=True, root_lock_unchanged=True,
                      cpp_positive_cases=len(json.loads((args.out_dir / "cpp_report.json").read_text())["positive_cases"]),
                      cpp_rejected_invalid_cases=len(json.loads((args.out_dir / "cpp_report.json").read_text())["rejected_invalid_cases"]))
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        evidence = ROOT / "docs/evidence/onnx_best200_cpu"
        evidence.mkdir(parents=True, exist_ok=True)
        for filename in ("quality.json", "export_report.json", "cpp_report.json"):
            shutil.copyfile(args.out_dir / filename, evidence / filename)
        shutil.copyfile(args.out_dir / "paired_reference/preflight.json", evidence / "paired_preflight.json")
        shutil.copyfile(logs / "quality_tools.txt", evidence / "quality_tools.txt")
        shutil.copyfile(logs / "independent_runtime.txt", evidence / "independent_runtime.txt")
        print("CPU checks passed. Fixed-nine CUDA/BF16 task accuracy remains unverified.", flush=True)
        return 0
    except Exception as error:
        report.update(success=False, failure_type=type(error).__name__, elapsed_s=time.perf_counter() - started)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
