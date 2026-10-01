"""Run DA-off cached refiner training, reference scoring, then DA-on tuning.

The subprocess boundary keeps each phase independently resumable. If phase one
fails or is interrupted, phase two is never launched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from mamba3_tracker.train.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def _best_checkpoint(out_dir: Path) -> Path:
    manifest = out_dir / "checkpoints.json"
    if not manifest.is_file():
        raise RuntimeError(f"phase one did not write {manifest}")
    best = json.loads(manifest.read_text()).get("best", [])
    if not best or not best[0].get("path"):
        raise RuntimeError(f"phase one has no held-out best in {manifest}")
    path = out_dir / best[0]["path"]
    if not path.is_file():
        raise RuntimeError(f"phase one best checkpoint is missing: {path}")
    return path


def _run(cmd: list[str]) -> None:
    print("[staged] $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("~/data"))
    ap.add_argument("--cache-config", type=Path,
                    default=Path("configs/v64_amuse_cache_phase.yaml"))
    ap.add_argument("--finetune-config", type=Path,
                    default=Path("configs/v64_amuse_da_finetune.yaml"))
    ap.add_argument("--reference-manifest", type=Path,
                    default=Path("configs/v64_metric_reference_minival.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.data_root = args.data_root.expanduser()

    cache_cfg = load_config(args.cache_config)
    fine_cfg = load_config(args.finetune_config)
    if cache_cfg["data"].get("photometric_augment", True):
        raise ValueError("cache phase requires data.photometric_augment: false")
    if not cache_cfg.get("frozen_cache", {}).get("enabled", False):
        raise ValueError("cache phase requires frozen_cache.enabled: true")
    if fine_cfg["data"].get("photometric_augment", False) is not True:
        raise ValueError("fine-tune phase requires data.photometric_augment: true")
    if fine_cfg.get("frozen_cache", {}).get("enabled", False):
        raise ValueError("fine-tune phase must not use frozen cache with DA")
    if not fine_cfg["train"].get("init_best_from"):
        raise ValueError("fine-tune phase requires train.init_best_from")
    phase_one = Path(cache_cfg["train"]["out_dir"]).expanduser()
    direct = Path(cache_cfg["train"]["init_ckpt"]).expanduser()
    if not direct.is_file():
        raise FileNotFoundError(f"cache phase warm-start is missing: {direct}")
    if not args.reference_manifest.is_file():
        raise FileNotFoundError(f"reference manifest is missing: {args.reference_manifest}")

    plan = {
        "phase_1": {"config": str(args.cache_config), "out_dir": str(phase_one),
                    "warm_start": str(direct), "photometric_augment": False},
        "reference": str(args.reference_manifest),
        "phase_2": {"config": str(args.finetune_config),
                    "best_from": str(fine_cfg["train"]["init_best_from"]),
                    "photometric_augment": True},
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return

    train = [sys.executable, "scripts/train_depth_refined_tracker.py"]
    _run(train + ["--config", str(args.cache_config), "--data-root", str(args.data_root)])
    best = _best_checkpoint(phase_one)
    ref_dir = phase_one / "reference_eval" / f"best_{best.stem.removeprefix('best_')}"
    _run([sys.executable, "scripts/eval_metric3d.py", "--method", "v35",
          "--ckpt", str(best), "--depth", "da3l", "--split", "minival",
          "--clip-manifest", str(args.reference_manifest), "--out-dir", str(ref_dir),
          "--data-root", str(args.data_root)])
    _run([sys.executable, "scripts/summarize_reference_metric.py", "--metrics",
          str(ref_dir / "metrics.json"), "--manifest", str(args.reference_manifest),
          "--out", str(ref_dir / "reference_progress.json")])
    _run(train + ["--config", str(args.finetune_config), "--data-root", str(args.data_root)])


if __name__ == "__main__":
    main()
