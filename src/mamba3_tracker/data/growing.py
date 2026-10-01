"""Append only completed clips while keeping validation membership immutable."""

import hashlib
import json
from pathlib import Path
import random

from .tapvid3d_splits import FULL_EVAL_FILES, MINIVAL_FILES


class PoolNotReady(RuntimeError):
    pass


class GrowingClipPool:
    def __init__(
        self,
        manifest,
        raw_root,
        depth_root,
        state_path,
        subsets,
        val_per_subset=5,
        seed=42,
    ):
        self.root = Path(raw_root).expanduser()
        self.depth = Path(depth_root).expanduser()
        self.state_path = Path(state_path)
        self.subsets = list(subsets)
        self.val_per_subset, self.seed = val_per_subset, seed
        content = Path(manifest).expanduser().read_bytes()
        self.selection_hash = hashlib.sha256(content).hexdigest()
        self.names = json.loads(content)["files_by_subset"]
        for subset in subsets:
            names = self.names[subset]
            if len(names) != len(set(names)) or not set(names) <= set(
                FULL_EVAL_FILES[subset]
            ):
                raise ValueError(f"invalid official training selection: {subset}")
            if set(names) & set(MINIVAL_FILES[subset]):
                raise ValueError("minival leakage in selection")
        self.val = None
        self.last_train = []
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            if (
                state["selection_hash"] != self.selection_hash
                or state["val_per_subset"] != val_per_subset
                or state["subsets"] != self.subsets
            ):
                raise ValueError(
                    "growing pool selection/validation differs from existing run"
                )
            self.val = [Path(p) for p in state["validation"]]
            self.last_train = [Path(p) for p in state["train"]]

    def ready_paths(self):
        ready = {}
        for subset in self.subsets:
            ready[subset] = []
            for name in sorted(self.names[subset]):
                raw, depth = self.root / subset / name, self.depth / subset / name
                marker = self.depth / subset / (name + ".ready.json")
                if not (raw.is_file() and depth.is_file() and marker.is_file()):
                    continue
                meta = json.loads(marker.read_text())
                if (
                    raw.stat().st_size == meta["raw_bytes"]
                    and depth.stat().st_size == meta["depth_bytes"]
                ):
                    ready[subset].append(raw)
        return ready

    def snapshot(self):
        ready = self.ready_paths()
        all_ready = {p for paths in ready.values() for p in paths}
        if self.val is None:
            deficient = {
                s: len(paths)
                for s, paths in ready.items()
                if len(paths) <= self.val_per_subset
            }
            if deficient:
                raise PoolNotReady(
                    f"ready clips need >{self.val_per_subset}/subset: {deficient}"
                )
            self.val = []
            for subset, paths in ready.items():
                self.val.extend(
                    random.Random(f"{self.seed}:{subset}").sample(
                        paths, self.val_per_subset
                    )
                )
        if not set(self.val) <= all_ready or not set(self.last_train) <= all_ready:
            raise ValueError(
                "previously admitted train/validation clips are no longer ready"
            )
        train = sorted(all_ready - set(self.val))
        self.last_train = train
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {
                    "selection_hash": self.selection_hash,
                    "val_per_subset": self.val_per_subset,
                    "subsets": self.subsets,
                    "validation": [str(p) for p in self.val],
                    "train": [str(p) for p in train],
                },
                indent=2,
            )
        )
        tmp.replace(self.state_path)
        return train, list(self.val)
