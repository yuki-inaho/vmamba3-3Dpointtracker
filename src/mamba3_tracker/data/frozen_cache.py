"""Content-addressed, bounded storage for frozen tensor computations."""

import hashlib
import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from beartype import beartype

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
    def __init__(self, root, namespace, max_bytes, data_root=None, total_bytes=60e9, ram_bytes=0):
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
        self.ram_hits = 0
        self._ram: OrderedDict[str, tuple[torch.Tensor, ...]] = OrderedDict()
        self.ram_size = 0
        self.configure_ram(int(ram_bytes))

    @beartype
    def configure_ram(self, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError("RAM cache budget must be non-negative")
        self.ram_limit = max_bytes
        while self._ram and self.ram_size > self.ram_limit:
            _, values = self._ram.popitem(last=False)
            self.ram_size -= sum(t.numel() * t.element_size() for t in values)

    def _remember(self, key: str, tensors: tuple[torch.Tensor, ...]) -> None:
        size = sum(t.numel() * t.element_size() for t in tensors)
        if not self.ram_limit or size > self.ram_limit:
            return
        previous = self._ram.pop(key, None)
        if previous is not None:
            self.ram_size -= sum(t.numel() * t.element_size() for t in previous)
        while self._ram and self.ram_size + size > self.ram_limit:
            _, values = self._ram.popitem(last=False)
            self.ram_size -= sum(t.numel() * t.element_size() for t in values)
        # Clone slices: a one-frame view must not retain a whole GPU/CPU batch.
        self._ram[key] = tuple(t.detach().cpu().clone() for t in tensors)
        self.ram_size += size

    def key(self, *tensors):
        digest = hashlib.sha256()
        for tensor in tensors:
            tensor = tensor.detach().contiguous()
            digest.update(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        if torch.is_autocast_enabled("cuda"):
            digest.update(str(torch.get_autocast_dtype("cuda")).encode())
        return digest.hexdigest()

    def get(self, key: str, device: torch.device | str) -> tuple[torch.Tensor, ...] | None:
        if key in self._ram:
            values = self._ram.pop(key)
            self._ram[key] = values
            self.hits += 1
            self.ram_hits += 1
            return tuple(t.to(device) for t in values)
        path = self.root / (key + ".pt")
        if not path.is_file():
            self.misses += 1
            return None
        tensors = torch.load(path, map_location="cpu", weights_only=True)
        self._remember(key, tensors)
        os.utime(path, None)
        self.hits += 1
        return tuple(t.to(device) for t in tensors)

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
            self._remember(key, tuple(dict(items)[key]))


class CachedFlow:
    def __init__(self, model, cache, batch_size=1, max_batch_size=8, reserve_gb=8):
        self.model, self.cache, self.device = model, cache, model.device
        self.batch = AdaptiveBatchSize(batch_size, max_batch_size, reserve_gb)
        self.prefetch_enabled = batch_size > 1
        self.warm_values = {}
        self.last_peak_allocated = 0
        self._prefetched_clips = set()
        self._batch_keys = {}
        self._pair_aliases = OrderedDict()
        self._held_images = None

    @staticmethod
    def _view_token(tensor):
        return (tensor.data_ptr(), tuple(tensor.shape), tensor.stride(), tensor._version)

    @torch.no_grad()
    def prefetch_windows(self, windows):
        """Batch cache misses across clips without forming cross-clip pairs."""
        self._prefetched_clips.clear()
        frames = windows.shape[1]
        pairs = []
        for clip in range(len(windows)):
            self._prefetched_clips.add((windows[clip].data_ptr(), tuple(windows[clip].shape)))
            offset = clip * frames
            pairs.extend((offset + i, offset + i + 1) for i in range(frames - 1))
            pairs.extend((offset + i + 1, offset + i) for i in range(frames - 1))
        self._prefetch_pairs(windows.flatten(0, 1), pairs)

    def release_prefetch(self):
        self.warm_values.clear()
        self._prefetched_clips.clear()
        self._batch_keys.clear()
        self._held_images = None

    @torch.no_grad()
    def prefetch_clip(self, images):
        """Warm consecutive pairs in GPU batches, with the same single-pair keys.

        Both tracking directions are independent image-pair computations. Cache
        hits need no GPU inference; training still updates the refiner normally.
        """
        if not self.prefetch_enabled or len(images) < 2:
            return
        if (images.data_ptr(), tuple(images.shape)) in self._prefetched_clips:
            return
        self._prefetched_clips.clear()
        pairs = [(i, i + 1) for i in range(len(images) - 1)]
        pairs += [(j, i) for i, j in pairs]
        self._prefetch_pairs(images, pairs)

    def _prefetch_pairs(self, images, pairs):
        self.warm_values.clear()
        self.last_peak_allocated = 0
        pending = []
        # Hash each CPU frame once. Keep legacy pair keys, so disk caches stay usable.
        cpu_images = images.detach().cpu()
        frame_keys = [self.cache.key(frame.unsqueeze(0)) for frame in cpu_images]
        self._batch_keys.clear()
        self._held_images = images  # Hold storage until release; pointer keys cannot be recycled.
        for first, second in pairs:
            alias = (frame_keys[first], frame_keys[second])
            key = self._pair_aliases.get(alias)
            if key is None:
                key = self.cache.key(cpu_images[first:first + 1], cpu_images[second:second + 1])
            else:
                self._pair_aliases.move_to_end(alias)
            self._pair_aliases[alias] = key
            if len(self._pair_aliases) > 16384:
                self._pair_aliases.popitem(last=False)
            pointer_pair = (self._view_token(images[first:first + 1]),
                            self._view_token(images[second:second + 1]))
            self._batch_keys[pointer_pair] = key
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
                assert free is not None
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
        return self._flow(first, second, first.device)

    def _flow(self, first, second, device):
        key = self._batch_keys.get((self._view_token(first), self._view_token(second))) if len(first) == len(second) == 1 else None
        if key is None:
            key = self.cache.key(first, second)
        if key in self.warm_values:
            self.cache.hits += 1
            return self.warm_values[key].to(device)
        found = self.cache.get(key, device)
        if found is not None:
            return found[0]
        result = self.model.flow(first, second)
        self.cache.put(key, (result,))
        return result.to(device)

    def consecutive_flows(self, images):
        """Return CPU flow for the tracker without a GPU round trip on disk hits."""
        self.prefetch_clip(images)
        fwd = [self._flow(images[i:i + 1], images[i + 1:i + 2], "cpu") for i in range(len(images) - 1)]
        bwd = [self._flow(images[i + 1:i + 2], images[i:i + 1], "cpu") for i in range(len(images) - 1)]
        return fwd, bwd
