"""Deterministic TAPVid-3D partitions from README's HF mirror, with a raw disk cap.

The official minival is never sampled into train. Tarballs are streamed across
all shards, not retained. Resume is at validated NPZ boundaries; an interrupted
gzip shard is rescanned while complete clips are skipped.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib.util
import json
import math
from pathlib import Path
import random
import shutil
import tarfile
import threading
import time
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "tapvid3d_splits", ROOT / "src/mamba3_tracker/data/tapvid3d_splits.py"
)
SPLITS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SPLITS)
STORAGE_SPEC = importlib.util.spec_from_file_location(
    "data_storage", ROOT / "src/mamba3_tracker/data/storage.py"
)
STORAGE = importlib.util.module_from_spec(STORAGE_SPEC)
STORAGE_SPEC.loader.exec_module(STORAGE)
HF = "https://huggingface.co/datasets/ZhengGuangze/TAPVid-3D/resolve/main"
ARCHIVES = {
    "pstudio": ["pstudio.tar.gz"],
    "adt": [f"adt_batch_{i}.tar.gz" for i in range(10)],
    "drivetrack": [f"drivetrack_batch_{i}.tar.gz" for i in range(13)],
}
REQUIRED = {
    "images_jpeg_bytes.npy",
    "fx_fy_cx_cy.npy",
    "tracks_XYZ.npy",
    "visibility.npy",
    "queries_xyt.npy",
}


def log(**fields):
    print(
        json.dumps(
            {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields}
        ),
        flush=True,
    )


def valid(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return REQUIRED <= set(archive.namelist()) and archive.testzip() is None
    except (OSError, zipfile.BadZipFile, EOFError):
        return False


def atomic_json(path: Path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(content, indent=2) + "\n")
    temporary.replace(path)


def build_manifest(seed: int, fraction: float):
    selected = {}
    for subset in sorted(SPLITS.FULL_EVAL_FILES):
        files = sorted(SPLITS.FULL_EVAL_FILES[subset])
        random.Random(f"{seed}:{subset}").shuffle(files)
        selected[subset] = sorted(
            files[: max(1, math.floor(len(files) * fraction + 0.5))]
        )
        assert set(selected[subset]).isdisjoint(SPLITS.MINIVAL_FILES[subset])
    return {
        "schema_version": 1,
        "split": "full_eval_subset",
        "seed": seed,
        "fraction": fraction,
        "source": "ZhengGuangze/TAPVid-3D",
        "official_train_counts": {s: len(v) for s, v in SPLITS.FULL_EVAL_FILES.items()},
        "files_by_subset": selected,
        "train_files": [
            {"subset": s, "filename": n} for s, names in selected.items() for n in names
        ],
        "evaluation_files_by_subset": SPLITS.MINIVAL_FILES,
        "note": "A sampled training run; full 4419-clip official reproduction remains incomplete.",
    }


class Downloader:
    def __init__(self, args, manifest):
        self.args, self.manifest = args, manifest
        self.lock = threading.Lock()
        for partial in args.data_root.rglob("*.npz.part"):
            partial.unlink()
        self.used = STORAGE.tree_bytes(args.data_root)
        self.cap = int(args.raw_budget_gb * 1e9)
        self.reserve = int(args.reserve_gb * 1e9)
        self.searched = set()
        self.archive_members = {}
        if args.state_json.exists():
            old = json.loads(args.state_json.read_text())
            if (
                old.get("seed") == args.seed
                and old.get("fraction") == args.train_fraction
                and old.get("split") == args.split
            ):
                self.searched = set(old.get("searched_archives", []))
                self.archive_members = old.get("selected_archive_members", {})
        self.blocked = {}

    def names(self, subset):
        train = (
            self.manifest["files_by_subset"][subset]
            if self.args.split != "minival"
            else []
        )
        evaluation = SPLITS.MINIVAL_FILES[subset] if self.args.split != "train" else []
        return set(train + evaluation)

    def status(self):
        with self.lock:
            result = {}
            for subset in self.args.subsets:
                wanted = self.names(subset)
                # Each final name was validated before atomic rename. Startup scan
                # performs full CRC validation; status avoids repeatedly reading GBs.
                present = {
                    n for n in wanted if (self.args.data_root / subset / n).exists()
                }
                result[subset] = {
                    "required": len(wanted),
                    "downloaded": len(present),
                    "missing": sorted(wanted - present),
                }
            atomic_json(
                self.args.state_json,
                {
                    "seed": self.args.seed,
                    "fraction": self.args.train_fraction,
                    "split": self.args.split,
                    "data_root": str(self.args.data_root),
                    "raw_budget_bytes": self.cap,
                    "raw_bytes_accounted": self.used,
                    "searched_archives": sorted(self.searched),
                    "selected_archive_members": self.archive_members,
                    "budget_blocked_files": self.blocked,
                    "subsets": result,
                },
            )

    def subset(self, subset):
        outdir = self.args.data_root / subset
        outdir.mkdir(parents=True, exist_ok=True)
        wanted = {n for n in self.names(subset) if not valid(outdir / n)}
        for archive in ARCHIVES[subset]:
            if not wanted:
                break
            key = f"{subset}/{archive}"
            if (
                key in self.searched
                and key in self.archive_members
                and not wanted.intersection(self.archive_members[key])
            ):
                continue
            for attempt in range(1, self.args.attempts + 1):
                try:
                    log(
                        subset=subset,
                        archive=archive,
                        attempt=attempt,
                        missing=len(wanted),
                        status="streaming",
                    )
                    with requests.get(
                        f"{HF}/{archive}?download=true", stream=True, timeout=(30, 120)
                    ) as response:
                        response.raise_for_status()
                        if response.status_code != 200:
                            raise RuntimeError(
                                f"Expected full gzip stream, HTTP {response.status_code}"
                            )
                        with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
                            for member in tar:
                                name = Path(member.name).name
                                if member.isfile() and name in self.names(subset):
                                    with self.lock:
                                        known = self.archive_members.setdefault(key, [])
                                        if name not in known:
                                            known.append(name)
                                if not member.isfile() or name not in wanted:
                                    continue
                                # Reserve a complete member atomically before writing.
                                with self.lock:
                                    data_used = STORAGE.tree_bytes(
                                        self.args.data_root.parent
                                    )
                                    if (
                                        data_used + member.size
                                        > self.args.total_data_budget_gb * 1e9 - 1e9
                                    ):
                                        self.blocked[f"{subset}/{name}"] = member.size
                                        log(
                                            subset=subset,
                                            file=name,
                                            status="total_data_budget_blocked",
                                        )
                                        continue
                                    if self.used + member.size > self.cap:
                                        self.blocked[f"{subset}/{name}"] = member.size
                                        log(
                                            subset=subset,
                                            file=name,
                                            bytes=member.size,
                                            status="raw_budget_blocked",
                                        )
                                        continue
                                    if (
                                        shutil.disk_usage(self.args.data_root).free
                                        - member.size
                                        < self.reserve
                                    ):
                                        raise RuntimeError(
                                            "Free disk reserve would be exceeded"
                                        )
                                    self.used += member.size
                                part = (outdir / name).with_suffix(".npz.part")
                                try:
                                    source = tar.extractfile(member)
                                    if source is None:
                                        raise RuntimeError("Missing tar member stream")
                                    with part.open("wb") as fh:
                                        shutil.copyfileobj(source, fh, 1 << 20)
                                    if part.stat().st_size != member.size or not valid(
                                        part
                                    ):
                                        raise RuntimeError(
                                            f"NPZ CRC or required keys failed: {name}"
                                        )
                                    part.replace(outdir / name)
                                except Exception:
                                    part.unlink(missing_ok=True)
                                    with self.lock:
                                        self.used -= member.size
                                    raise
                                wanted.remove(name)
                                log(
                                    subset=subset,
                                    file=name,
                                    bytes=member.size,
                                    missing=len(wanted),
                                    archive=archive,
                                    status="verified",
                                )
                                self.status()
                                if not wanted:
                                    break
                    with self.lock:
                        self.searched.add(key)
                    self.status()
                    log(
                        subset=subset,
                        archive=archive,
                        missing=len(wanted),
                        status="searched",
                    )
                    break
                except Exception as exc:
                    log(
                        subset=subset,
                        archive=archive,
                        attempt=attempt,
                        status="retry",
                        error=str(exc),
                    )
                    if attempt == self.args.attempts:
                        raise
        if wanted:
            raise RuntimeError(
                f"{subset}: {len(wanted)} requested clips missing; see state JSON for budget blocks"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "data/tapvid3d")
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=ROOT / "temp/tapvid3d_train_subset_manifest.json",
    )
    parser.add_argument(
        "--state-json",
        type=Path,
        default=ROOT / "temp/tapvid3d_subset_download_status.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.15)
    parser.add_argument("--raw-budget-gb", type=float, default=30)
    parser.add_argument("--total-data-budget-gb", type=float, default=60)
    parser.add_argument("--reserve-gb", type=float, default=30)
    parser.add_argument(
        "--subsets", nargs="+", choices=list(ARCHIVES), default=list(ARCHIVES)
    )
    parser.add_argument("--split", choices=["train", "minival", "all"], default="all")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 < args.train_fraction <= 1:
        parser.error("--train-fraction must be in (0, 1]")
    if args.raw_budget_gb <= 0 or args.reserve_gb < 0:
        parser.error("Invalid disk budget")
    manifest = build_manifest(args.seed, args.train_fraction)
    if (
        args.train_manifest.exists()
        and json.loads(args.train_manifest.read_text()) != manifest
    ):
        parser.error(
            "Existing manifest differs; use a separate manifest path to change seed/fraction"
        )
    atomic_json(args.train_manifest, manifest)
    log(
        status="manifest",
        path=str(args.train_manifest),
        train_counts={s: len(v) for s, v in manifest["files_by_subset"].items()},
        raw_budget_gb=args.raw_budget_gb,
    )
    if args.dry_run:
        return
    args.data_root.mkdir(parents=True, exist_ok=True)
    download = Downloader(args, manifest)
    try:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(download.subset, s) for s in args.subsets]
            for job in cf.as_completed(jobs):
                job.result()
    finally:
        download.status()


if __name__ == "__main__":
    main()
