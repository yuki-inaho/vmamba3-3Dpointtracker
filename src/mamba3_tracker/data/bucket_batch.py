"""Batch ready clips with the same depth grid without resampling depth."""

import json
from collections import defaultdict

import torch
from torch.utils.data import Sampler, get_worker_info


class DepthBucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset, self.batch_size = dataset, batch_size

    def _groups(self):
        groups = defaultdict(list)
        for i, path in enumerate(self.dataset.clip_paths):
            marker = self.dataset.da3_depth_root / path.parent.name / (path.name + ".ready.json")
            shape = json.loads(marker.read_text())["shape"]
            groups[tuple(shape[1:])].append(i)
        return groups

    def __iter__(self):
        batches = []
        for indices in self._groups().values():
            shuffled = [indices[i] for i in torch.randperm(len(indices)).tolist()]
            batches.extend(shuffled[i:i + self.batch_size]
                           for i in range(0, len(shuffled), self.batch_size))
        for i in torch.randperm(len(batches)).tolist():
            yield batches[i]

    def __len__(self):
        return sum((len(v) + self.batch_size - 1) // self.batch_size for v in self._groups().values())


def seed_tracking_worker(worker_id):
    import random
    worker = get_worker_info()
    worker.dataset._rng = random.Random(worker.seed)
