"""Finite photometric variants and resumable batches with a bounded working set."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch.utils.data import DataLoader, Dataset, Sampler


@dataclass(frozen=True)
class PhotometricPattern:
    name: str
    brightness: float
    contrast: float

    def __post_init__(self) -> None:
        if not self.name or not all(math.isfinite(v) and v > 0
                                    for v in (self.brightness, self.contrast)):
            raise ValueError("DA patterns need a name and finite positive coefficients")


@jaxtyped(typechecker=beartype)
def apply_pattern(
    images: Float[torch.Tensor, "frames 3 height width"],
    pattern: PhotometricPattern,
) -> Float[torch.Tensor, "frames 3 height width"]:
    """Same deterministic coefficients for every frame; geometry stays intact."""
    out = images * pattern.brightness
    mean = out.mean(dim=(-1, -2, -3), keepdim=True)
    return ((out - mean) * pattern.contrast + mean).clamp(0.0, 1.0)


@beartype
def parse_patterns(records: list[dict]) -> tuple[PhotometricPattern, ...]:
    patterns = tuple(PhotometricPattern(str(r["name"]), float(r["brightness"]),
                                        float(r["contrast"])) for r in records)
    if not patterns or len({p.name for p in patterns}) != len(patterns):
        raise ValueError("Fixed DA requires nonempty, uniquely named patterns")
    if len({(p.brightness, p.contrast) for p in patterns}) != len(patterns):
        raise ValueError("Fixed DA coefficients must be distinct")
    return patterns


@dataclass(frozen=True)
class DAIndex:
    clip: int
    pattern: int
    block: tuple[int, ...]
    epoch: int = 0
    position: int = 0


@beartype
def prewarm_loader(dataset: Dataset, indices: list[DAIndex], batch_size: int,
                   training_workers: int) -> DataLoader:
    """Use the training decode context, including CPU interpolation kernels.

    PyTorch's single-thread and multithread CPU resize paths can differ by an
    ULP. Hashing those images requires warming with the same worker context.
    """
    from .bucket_batch import worker_loader_options
    from .dataset import collate_tracking

    return DataLoader(dataset, batch_size=batch_size, sampler=indices,
                      collate_fn=collate_tracking,
                      **worker_loader_options(dataset, training_workers))


class ClipDataset(Protocol):
    clip_paths: list[Path]
    da3_depth_root: Path | None


class FixedDABatchSampler(Sampler[list[DAIndex]]):
    """Warm each depth-compatible block once, then reuse all variants locally.

    Only acknowledgements from the training loop advance the cursor. DataLoader
    prefetch may request batches ahead of that cursor without corrupting resume.
    """

    def __init__(self, dataset: ClipDataset, patterns: int, batch_size: int,
                 block_clips: int = 32, repeats: int = 2, seed: int = 0,
                 recipe_signature: str = "") -> None:
        if min(patterns, batch_size, block_clips, repeats) < 1:
            raise ValueError("Fixed DA batch limits must be positive")
        if dataset.da3_depth_root is None:
            raise ValueError("Fixed DA blocks require ready depth markers")
        self.dataset = dataset
        self.patterns, self.batch_size = patterns, batch_size
        self.block_clips, self.repeats, self.seed = block_clips, repeats, seed
        self.epoch, self.offset = 0, 0
        self._plan: list[list[DAIndex]] | None = None
        signature = [[str(p) for p in dataset.clip_paths], recipe_signature,
                     "brightness-contrast-v1"]
        self.signature = hashlib.sha256(json.dumps(signature).encode()).hexdigest()

    def _batches(self) -> list[list[DAIndex]]:
        if self._plan is not None:
            return self._plan
        depth_root = self.dataset.da3_depth_root
        assert depth_root is not None
        groups: dict[tuple[int, ...], list[int]] = {}
        for index, path in enumerate(self.dataset.clip_paths):
            marker = depth_root / path.parent.name / (path.name + ".ready.json")
            shape = tuple(int(v) for v in json.loads(marker.read_text())["shape"][1:])
            groups.setdefault(shape, []).append(index)
        rng = random.Random(self.seed + self.epoch)
        blocks: list[tuple[int, ...]] = []
        for indices in groups.values():
            rng.shuffle(indices)
            blocks.extend(tuple(indices[i:i + self.block_clips])
                          for i in range(0, len(indices), self.block_clips))
        rng.shuffle(blocks)
        plan: list[list[DAIndex]] = []
        for block in blocks:
            for _ in range(self.repeats):
                pairs = [(i, p) for i in block for p in range(self.patterns)]
                rng.shuffle(pairs)
                for start in range(0, len(pairs), self.batch_size):
                    position = len(plan)
                    plan.append([DAIndex(i, p, block, self.epoch, position)
                                 for i, p in pairs[start:start + self.batch_size]])
        self._plan = plan
        return plan

    def __iter__(self) -> Iterator[list[DAIndex]]:
        if self.offset == len(self._batches()):
            self.epoch += 1
            self.offset = 0
            self._plan = None
        # Freeze this iterator's start: acknowledgements can change offset.
        yield from self._batches()[self.offset:]

    def __len__(self) -> int:
        return len(self._batches()) - self.offset

    @property
    def covered_once(self) -> bool:
        return self.epoch > 0 or self.offset == len(self._batches())

    @beartype
    def acknowledge(self, epoch: int, position: int) -> None:
        if epoch != self.epoch or position != self.offset:
            raise ValueError("Fixed DA consumed batch differs from resumable cursor")
        self.offset += 1

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "offset": self.offset, "signature": self.signature,
                "patterns": self.patterns, "batch_size": self.batch_size,
                "block_clips": self.block_clips, "repeats": self.repeats, "seed": self.seed}

    def restore(self, state: dict) -> None:
        for name in ("signature", "patterns", "batch_size", "block_clips", "repeats", "seed"):
            if state[name] != getattr(self, name):
                raise ValueError(f"Fixed DA resume differs: {name}")
        self.epoch, self.offset = int(state["epoch"]), int(state["offset"])
        self._plan = None
        if not 0 <= self.offset <= len(self._batches()):
            raise ValueError("Invalid fixed DA resume offset")
