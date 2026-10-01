"""Run DA-off cached refiner training, reference scoring, then DA-on tuning.

The subprocess boundary keeps each phase independently resumable. If phase one
fails or is interrupted, phase two is never launched.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path

from mamba3_tracker.train.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def _best_checkpoint(out_dir: Path) -> Path:
    status = json.loads((out_dir / "training_status.json").read_text())
    if status.get("reason") not in {"early_stopping", "clip_budget", "max_steps"}:
        raise RuntimeError(f"phase is not normally finished: {out_dir}: {status.get('reason')}")
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
    child = subprocess.Popen(cmd, cwd=ROOT, start_new_session=True)
    try:
        result = child.wait()
        if result:
            raise subprocess.CalledProcessError(result, cmd)
    except BaseException:
        # Include DataLoader workers when interrupted; do not leave a GPU job
        # running behind a stopped runner or accidentally launch phase two.
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        child.wait()
        raise


def _reference(best: Path, manifest: Path, data_root: Path) -> None:
    ref_dir = best.parent / "reference_eval" / best.stem
    _run([sys.executable, "scripts/eval_metric3d.py", "--method", "v35",
          "--ckpt", str(best), "--depth", "da3l", "--split", "minival",
          "--clip-manifest", str(manifest), "--out-dir", str(ref_dir),
          "--data-root", str(data_root)])
    _verify_reference(ref_dir, manifest)
    _run([sys.executable, "scripts/summarize_reference_metric.py", "--metrics",
          str(ref_dir / "metrics.json"), "--manifest", str(manifest),
          "--out", str(ref_dir / "reference_progress.json")])


def _verify_reference(ref_dir: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    metrics = json.loads((ref_dir / "metrics.json").read_text())
    if metrics.get("failures") != 0:
        raise RuntimeError("reference evaluation has failed clips; next phase is blocked")
    for subset, names in manifest["files_by_subset"].items():
        rows = json.loads((ref_dir / "metric_results" / f"{subset}.json").read_text())
        expected = {Path(name).stem for name in names}
        if len(rows) != len(names) or {r["clip_id"] for r in rows} != expected:
            raise RuntimeError(f"reference clip membership differs for {subset}")
    for metric in ("average_jaccard", "metric_average_jaccard"):
        value = float(metrics["overall"][metric])
        if not math.isfinite(value):
            raise RuntimeError(f"reference evaluation produced non-finite {metric}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("~/data"))
    ap.add_argument("--cache-config", type=Path,
                    default=Path("configs/v64_official_mamba3_cache_phase.yaml"))
    ap.add_argument("--finetune-config", type=Path,
                    default=Path("configs/v64_official_mamba3_da_finetune.yaml"))
    ap.add_argument("--reference-manifest", type=Path,
                    default=Path("configs/v64_metric_reference_minival.json"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cache-only", action="store_true",
                    help="Finish the DA-off phase and its reference evaluation, then exit")
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
    if Path(fine_cfg["train"]["init_best_from"]).expanduser().resolve() != phase_one.resolve():
        raise ValueError("fine-tune init_best_from must refer to this cache phase")
    if fine_cfg["train"]["out_dir"] == str(phase_one):
        raise ValueError("the two stages need different output directories")
    for name in ("dim", "num_layers", "temporal_mixer", "two_pool"):
        if cache_cfg["model"].get(name) != fine_cfg["model"].get(name):
            raise ValueError(f"stage architectures differ: model.{name}")
    direct = Path(cache_cfg["train"]["init_ckpt"]).expanduser()
    if not direct.is_file():
        raise FileNotFoundError(f"cache phase warm-start is missing: {direct}")
    if not args.reference_manifest.is_file():
        raise FileNotFoundError(f"reference manifest is missing: {args.reference_manifest}")

    plan = {
        "phase_1": {"config": str(args.cache_config), "out_dir": str(phase_one),
                    "warm_start": str(direct), "photometric_augment": False},
        "reference": str(args.reference_manifest),
        "cache_only": args.cache_only,
        "phase_2": {"config": str(args.finetune_config),
                    "best_from": str(fine_cfg["train"]["init_best_from"]),
                    "photometric_augment": True},
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return

    def terminate(signum, frame):
        raise KeyboardInterrupt("staged training terminated")

    signal.signal(signal.SIGTERM, terminate)
    train = [sys.executable, "scripts/train_depth_refined_tracker.py"]
    _run(train + ["--config", str(args.cache_config), "--data-root", str(args.data_root)])
    best = _best_checkpoint(phase_one)
    _reference(best, args.reference_manifest, args.data_root)
    if args.cache_only:
        return
    _run(train + ["--config", str(args.finetune_config), "--data-root", str(args.data_root)])
    _reference(_best_checkpoint(Path(fine_cfg["train"]["out_dir"])),
               args.reference_manifest, args.data_root)


if __name__ == "__main__":
    main()
