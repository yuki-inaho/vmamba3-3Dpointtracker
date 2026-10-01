"""Bounded GPU batches with reserved headroom and retry after OOM."""

import torch


class AdaptiveBatchSize:
    def __init__(self, initial=16, maximum=128, reserve_gb=8, adaptive=True):
        if initial < 1 or maximum < 1 or reserve_gb < 0:
            raise ValueError("Positive batch limits and nonnegative reserve required")
        self.current = min(initial, maximum)
        self.maximum = maximum
        self.reserve_bytes = reserve_gb * 1e9
        self.adaptive = adaptive
        self.bytes_per_item = None
        self.retries = 0

    def size(self, remaining, free_bytes=None):
        size = min(self.current, remaining)
        if self.adaptive and free_bytes is not None and self.bytes_per_item:
            capacity = max(1, int((free_bytes - self.reserve_bytes) / self.bytes_per_item))
            size = min(size, capacity)
        return size

    def success(self, count, peak_extra_bytes=0):
        if peak_extra_bytes > 0:
            estimate = peak_extra_bytes / count * 1.25
            self.bytes_per_item = max(self.bytes_per_item or 0, estimate)
        if self.adaptive and count == self.current:
            self.current = min(self.maximum, self.current * 2)

    def oom(self, count):
        if count <= 1:
            raise torch.cuda.OutOfMemoryError("GPU batch of one does not fit")
        self.current = max(1, count // 2)
        # Hold the reduced size; growing back immediately would repeat the OOM.
        self.maximum = min(self.maximum, self.current)
        self.retries += 1
