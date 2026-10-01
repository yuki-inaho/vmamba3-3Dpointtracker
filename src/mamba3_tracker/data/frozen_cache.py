"""Content-addressed, bounded storage for frozen tensor computations."""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from .gpu_batch import AdaptiveBatchSize
from .storage import tree_bytes


def module_fingerprint(module):
    digest = hashlib.sha256()
    config = getattr(module, "config", None)
    if config is not None:
        digest.update(json.dumps(config.to_dict(), sort_keys=True).encode())
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(json.dumps([str(value.dtype), list(value.shape)]).encode())
        digest.update(
            value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        )
    return digest.hexdigest()


class FrozenTensorCache:
    def __init__(self, root, namespace, max_bytes, data_root=None, total_bytes=60e9):
        self.group_root = Path(root).expanduser()
        self.root = (
            self.group_root
            / hashlib.sha256(namespace.encode()).hexdigest()[:16]
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self.data_root = Path(data_root).expanduser() if data_root else None
        self.total_bytes = int(total_bytes)
        self.hits = self.misses = 0

    def key(self, *tensors):
        digest = hashlib.sha256()
        for tensor in tensors:
            tensor = tensor.detach().contiguous()
            digest.update(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        if torch.is_autocast_enabled("cuda"):
            digest.update(str(torch.get_autocast_dtype("cuda")).encode())
        return digest.hexdigest()

    def get(self, key, device):
        path = self.root / (key + ".pt")
        if not path.is_file():
            self.misses += 1
            return None
        tensors = torch.load(path, map_location=device, weights_only=True)
        os.utime(path, None)
        self.hits += 1
        return tensors

    def put(self, key, tensors):
        self.put_many([(key, tensors)])

    def put_many(self, items):
        # Account/evict once per GPU batch, rather than walking the entire data
        # directory for every frame. Deduplicate repeated frames in the batch.
        temporary = {}
        for key, tensors in dict(items).items():
            path = self.root / (key + ".pt.partial")
            torch.save(tuple(t.detach().cpu().clone() for t in tensors), path)
            temporary[key] = path
        incoming = sum(path.stat().st_size for path in temporary.values())
        existing = sorted(
            self.group_root.rglob("*.pt"), key=lambda path: path.stat().st_mtime_ns
        )
        own_size = sum(path.stat().st_size for path in existing)
        total_size = (
            tree_bytes(self.data_root)
            if self.data_root
            else own_size + incoming
        )
        while existing and (
            own_size + incoming > self.max_bytes or total_size > self.total_bytes - 1e9
        ):
            old = existing.pop(0)
            size = old.stat().st_size
            old.unlink()
            own_size -= size
            total_size -= size
        if own_size + incoming > self.max_bytes or (
            self.data_root and total_size > self.total_bytes - 1e9
        ):
            for path in temporary.values():
                path.unlink()
            return
        for key, path in temporary.items():
            path.replace(self.root / (key + ".pt"))


class CachedFlow:
    def __init__(self, model, cache, batch_size=1, max_batch_size=8, reserve_gb=8):
        self.model, self.cache, self.device = model, cache, model.device
        self.batch = AdaptiveBatchSize(batch_size, max_batch_size, reserve_gb)
        self.prefetch_enabled = batch_size > 1
        self.warm_values = {}
        self.last_peak_allocated = 0

    @torch.no_grad()
    def prefetch_clip(self, images):
        """Warm consecutive pairs in GPU batches, with the same single-pair keys.

        Both tracking directions are independent image-pair computations. Cache
        hits need no GPU inference; training still updates the refiner normally.
        """
        if not self.prefetch_enabled or len(images) < 2:
            return
        self.warm_values.clear()
        self.last_peak_allocated = 0
        pairs = [(i, i + 1) for i in range(len(images) - 1)]
        pairs += [(j, i) for i, j in pairs]
        pending = []
        for first, second in pairs:
            key = self.cache.key(images[first:first + 1], images[second:second + 1])
            if not (self.cache.root / (key + ".pt")).is_file():
                pending.append((first, second, key))
        writer = ThreadPoolExecutor(max_workers=1)
        try:
            self._prefetch_batches(images, pending, writer)
        finally:
            writer.shutdown(wait=True)

    def _prefetch_batches(self, images, pending, writer):
        offset = 0
        writing = None
        while offset < len(pending):
            cuda = images.device.type == "cuda"
            free = torch.cuda.mem_get_info(images.device)[0] if cuda else None
            if cuda:
                free += torch.cuda.memory_reserved(images.device) - torch.cuda.memory_allocated(images.device)
            count = self.batch.size(len(pending) - offset, free)
            entries = pending[offset:offset + count]
            baseline = torch.cuda.memory_allocated(images.device) if cuda else 0
            if cuda:
                torch.cuda.reset_peak_memory_stats(images.device)
            try:
                output = self.model.flow(
                    images[[p[0] for p in entries]], images[[p[1] for p in entries]]
                )
            except torch.cuda.OutOfMemoryError:
                self.batch.oom(count)
                torch.cuda.empty_cache()
                continue
            peak = torch.cuda.max_memory_allocated(images.device) if cuda else 0
            self.last_peak_allocated = max(self.last_peak_allocated, peak)
            self.batch.success(count, peak - baseline)
            values = output.cpu()
            del output
            items = []
            for i, (_, _, key) in enumerate(entries):
                value = values[i:i + 1]
                self.warm_values[key] = value
                items.append((key, (value,)))
            if writing is not None:
                writing.result()  # At most one CPU write batch in flight.
            writing = writer.submit(self.cache.put_many, items)
            self.cache.misses += count
            offset += count
        if writing is not None:
            writing.result()

    @torch.no_grad()
    def flow(self, first, second):
        key = self.cache.key(first, second)
        if key in self.warm_values:
            self.cache.hits += 1
            return self.warm_values[key].to(first.device)
        found = self.cache.get(key, first.device)
        if found is not None:
            return found[0]
        result = self.model.flow(first, second)
        self.cache.put(key, (result,))
        return result
