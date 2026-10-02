"""Batch ready clips with the same depth grid without resampling depth."""

import json
import multiprocessing
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
            marker = (
                self.dataset.da3_depth_root
                / path.parent.name
                / (path.name + ".ready.json")
            )
            shape = json.loads(marker.read_text())["shape"]
            groups[tuple(shape[1:])].append(i)
        return groups

    def __iter__(self):
        batches = []
        for indices in self._groups().values():
            shuffled = [indices[i] for i in torch.randperm(len(indices)).tolist()]
            batches.extend(
                shuffled[i : i + self.batch_size]
                for i in range(0, len(shuffled), self.batch_size)
            )
        for i in torch.randperm(len(batches)).tolist():
            yield batches[i]

    def __len__(self):
        return sum(
            (len(v) + self.batch_size - 1) // self.batch_size
            for v in self._groups().values()
        )


def seed_tracking_worker(worker_id):
    import random

    worker = get_worker_info()
    threads = getattr(worker.dataset, "worker_cpu_threads", None)
    if threads is not None:
        torch.set_num_threads(threads)
    worker.dataset._rng = random.Random(worker.seed)


def validate_worker_configuration(threads, context):
    if threads is not None and (type(threads) is not int or threads < 1):
        raise ValueError("worker_cpu_threads must be a positive integer")
    if context is not None and context not in multiprocessing.get_all_start_methods():
        raise ValueError(
            "worker_multiprocessing_context must name an available start method"
        )
    if threads is not None and threads > 1 and context != "spawn":
        raise ValueError(
            "multithread workers require worker_multiprocessing_context: spawn"
        )


def worker_loader_options(dataset, nworkers, prefetch_factor=1):
    """Share the decode context between training and frozen-cache prewarming."""
    threads = getattr(dataset, "worker_cpu_threads", None)
    context = getattr(dataset, "worker_multiprocessing_context", None)
    validate_worker_configuration(threads, context)
    if type(nworkers) is not int or nworkers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if type(prefetch_factor) is not int or prefetch_factor < 1:
        raise ValueError("prefetch_factor must be a positive integer")
    options = dict(
        num_workers=nworkers,
        worker_init_fn=seed_tracking_worker,
        persistent_workers=False,
    )
    if nworkers:
        options["prefetch_factor"] = prefetch_factor
        if context is not None:
            options["multiprocessing_context"] = context
    return options
