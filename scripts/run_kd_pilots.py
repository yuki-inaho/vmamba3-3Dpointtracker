"""Wait for the confirmed input producer, then run five matched pilot arms."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--prepare-pid", type=int, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
os.chdir(root)
out = root / "result/refiner_kd_20261002"
manifest_path = out / "inputs_train/input_manifest.json"
watched = [
    Path(p)
    for p in (
        "configs/distill_refiner_vssd.yaml",
        "scripts/train_refiner_kd.py",
        "scripts/prepare_refiner_kd.py",
        "src/mamba3_tracker/model/student_refiner.py",
        "src/mamba3_tracker/model/onnx_refiner.py",
        "src/mamba3_tracker/train/distillation.py",
        "src/mamba3_tracker/deployment/student_checkpoint.py",
    )
]
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
status = {
    "status": "waiting_for_inputs",
    "source_hashes": hashes,
    "completed": [],
    "prepare_pid": args.prepare_pid,
}


def save():
    temporary = out / "pilot_chain_status.partial"
    temporary.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
    temporary.replace(out / "pilot_chain_status.json")


try:
    while True:
        manifest = json.loads(manifest_path.read_text())
        status["prepared"] = len(manifest["records"])
        save()
        if manifest["status"] == "complete":
            if len(manifest["records"]) != 1027:
                raise ValueError("Incomplete train partition")
            break
        process = Path(f"/proc/{args.prepare_pid}/cmdline")
        if (
            not process.is_file()
            or b"scripts/prepare_refiner_kd.py" not in process.read_bytes()
        ):
            raise RuntimeError("Input producer stopped before completing its manifest")
        print(
            json.dumps(
                {
                    "status": "waiting_for_live_input_producer",
                    "prepared": status["prepared"],
                }
            ),
            flush=True,
        )
        time.sleep(20)
    for arm in ("A0", "A1", "A2", "A3", "A4"):
        if any(
            hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p)]
            for p in watched
        ):
            raise RuntimeError("Training source changed between pilot arms")
        run = out / f"pilot_{arm}_s42"
        status.update(status="running", current_arm=arm)
        save()
        subprocess.run(
            [
                "uv",
                "run",
                "--no-sync",
                "python",
                "scripts/train_refiner_kd.py",
                "--config",
                "configs/distill_refiner_vssd.yaml",
                "--mode",
                "pilot",
                "--ablation",
                arm,
                "--seed",
                "42",
                "--microbatch",
                "32",
                "--out-dir",
                str(run),
            ],
            check=True,
        )
        result = json.loads((run / "training_status.json").read_text())
        if result["status"] != "complete" or not result["teacher_unchanged"]:
            raise RuntimeError("Pilot did not complete cleanly")
        status["completed"].append(
            {
                "arm": arm,
                "best_monitor_gt_loss": result["best_monitor_gt_loss"],
                "steps": result["step"],
                "best_checkpoint": result["best_checkpoint"],
                "identity": result["identity"],
            }
        )
        save()
    status.update(
        status="complete",
        next_action="Review paired pilot results and choose full candidate; no automatic test-based selection",
    )
    save()
except Exception as error:
    status.update(status="failed", error_type=type(error).__name__, error=str(error))
    save()
    raise
