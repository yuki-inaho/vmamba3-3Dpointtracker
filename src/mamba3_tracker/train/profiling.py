"""Opt-in synchronized stage timings and runtime tensor shape checks."""

from collections import defaultdict
from contextlib import contextmanager
from statistics import median
from time import perf_counter
from typing import Iterator

from beartype import beartype
from jaxtyping import Float, jaxtyped
import torch
from torch import Tensor


@jaxtyped(typechecker=beartype)
def check_training_inputs(
    images: Float[Tensor, "batch frames 3 height width"],
    ray: Float[Tensor, "batch frames points 2"],
    depth: Float[Tensor, "batch frames depth_height depth_width"],
    target: Float[Tensor, "batch frames points 3"],
) -> None:
    """Reject incompatible batch/frame/point dimensions before timing training."""


class StageTimer:
    @beartype
    def __init__(self, device: torch.device, warmup: int = 1, enabled: bool = True):
        if warmup < 0:
            raise ValueError("warmup must be non-negative")
        self.device = device
        self.warmup = warmup
        self.enabled = enabled
        self.samples: dict[str, list[float]] = defaultdict(list)

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self, name: str, step: int) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._sync()
        start = perf_counter()
        with torch.profiler.record_function(name):
            yield
        self._sync()
        if step > self.warmup:
            self.samples[name].append(perf_counter() - start)

    def report(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {"count": len(values), "median_seconds": median(values),
                   "mean_seconds": sum(values) / len(values)}
            for name, values in self.samples.items() if values
        }
