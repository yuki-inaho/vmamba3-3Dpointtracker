"""GPU flow batches across clips, keeping only the current batch on CPU."""

import torch

from .gpu_batch import AdaptiveBatchSize


class BatchedFlow:
    def __init__(self, model, initial=16, maximum=64, reserve_gb=4):
        self.model, self.device = model, model.device
        self.batch = AdaptiveBatchSize(initial, maximum, reserve_gb)
        self.values = {}
        self.images = None
        self.last_max_batch = 0
        self.last_peak_allocated = 0

    @staticmethod
    def _key(first, second):
        return (first.data_ptr(), second.data_ptr(), tuple(first.shape), first.dtype, first.device)

    @torch.no_grad()
    def prefetch_windows(self, images):
        batch, frames = images.shape[:2]
        flat = images.flatten(0, 1)
        pairs = [(b * frames + i, b * frames + i + 1)
                 for b in range(batch) for i in range(frames - 1)]
        self._prefetch(flat, pairs)

    def prefetch_clip(self, images):
        pairs = [(i, i + 1) for i in range(len(images) - 1)]
        if all(self._key(images[i:i+1], images[j:j+1]) in self.values for i, j in pairs):
            return
        self._prefetch(images, pairs)

    @torch.no_grad()
    def _prefetch(self, images, forward):
        self.values.clear()
        self.last_max_batch = 0
        self.last_peak_allocated = 0
        self.images = images  # Prevent pointer reuse while these keys are live.
        pairs = forward + [(j, i) for i, j in forward]
        offset = 0
        while offset < len(pairs):
            cuda = images.device.type == "cuda"
            free = torch.cuda.mem_get_info(images.device)[0] if cuda else None
            if cuda:
                # This process can reuse its allocator cache without consuming
                # another process's reserved memory.
                free += torch.cuda.memory_reserved(images.device) - torch.cuda.memory_allocated(images.device)
            count = self.batch.size(len(pairs) - offset, free)
            pending = pairs[offset:offset + count]
            baseline = torch.cuda.memory_allocated(images.device) if cuda else 0
            if cuda:
                torch.cuda.reset_peak_memory_stats(images.device)
            try:
                output = self.model.flow(images[[i for i, _ in pending]], images[[j for _, j in pending]])
            except torch.cuda.OutOfMemoryError:
                self.batch.oom(count)
                torch.cuda.empty_cache()
                continue
            peak = torch.cuda.max_memory_allocated(images.device) if cuda else 0
            self.last_peak_allocated = max(self.last_peak_allocated, peak)
            self.batch.success(count, peak - baseline)
            self.last_max_batch = max(self.last_max_batch, count)
            cpu = output.cpu()
            del output
            for k, (i, j) in enumerate(pending):
                self.values[self._key(images[i:i+1], images[j:j+1])] = cpu[k:k+1]
            offset += count

    def consecutive_flows(self, images):
        self.prefetch_clip(images)
        fwd = [self.values[self._key(images[i:i+1], images[i+1:i+2])] for i in range(len(images)-1)]
        bwd = [self.values[self._key(images[i+1:i+2], images[i:i+1])] for i in range(len(images)-1)]
        return fwd, bwd

    @torch.no_grad()
    def flow(self, first, second):
        key = self._key(first, second)
        if key in self.values:
            return self.values[key].to(first.device)
        return self.model.flow(first, second)
