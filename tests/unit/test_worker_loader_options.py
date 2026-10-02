from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from mamba3_tracker.data.bucket_batch import worker_loader_options
from mamba3_tracker.data.dataset import TAPVid3DDataset, collate_tracking
from mamba3_tracker.data.fixed_da import DAIndex, prewarm_loader
from mamba3_tracker.data.frozen_cache import FrozenTensorCache


class ResizeDataset(Dataset):
    worker_cpu_threads = 8
    worker_multiprocessing_context = "spawn"

    def __len__(self):
        return 1

    def __getitem__(self, index):
        native = torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(7))
        images = torch.nn.functional.interpolate(
            native, size=(256, 256), mode="bilinear", align_corners=False
        )
        return dict(
            images=images,
            queries_xyt=torch.tensor([[4.0, 4.0, 0.0]]),
            tracks_XYZ=torch.ones(2, 1, 3),
            visibility=torch.ones(2, 1, dtype=torch.bool),
            K=torch.eye(3),
            clip_id="clip",
            subset="adt",
            frame_start=0,
            query_idx=torch.arange(1),
            da_index=None,
        )


def test_spawn_thread_context_matches_main_and_prewarm():
    dataset = ResizeDataset()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(8)
    try:
        direct = dataset[0]["images"].unsqueeze(0)
        options = worker_loader_options(dataset, 1)
        training = DataLoader(
            dataset, batch_size=1, collate_fn=collate_tracking, timeout=30, **options
        )
        warm = prewarm_loader(dataset, [DAIndex(0, 0, (0,))], 1, training_workers=1)
        warm.timeout = 30
        assert warm.multiprocessing_context.get_start_method() == "spawn"
        assert warm.worker_init_fn is training.worker_init_fn
        assert warm.prefetch_factor == training.prefetch_factor == 1
        assert warm.num_workers == training.num_workers == 1
        a, b = next(iter(training)).images, next(iter(warm)).images
        assert torch.equal(a, direct) and torch.equal(b, direct)
        assert FrozenTensorCache.key(None, a) == FrozenTensorCache.key(None, b)
    finally:
        torch.set_num_threads(old_threads)


def test_default_and_no_worker_options_preserve_existing_behavior():
    dataset = TAPVid3DDataset([])
    assert dataset.worker_cpu_threads is None
    assert dataset.worker_multiprocessing_context is None
    options = worker_loader_options(dataset, 2, prefetch_factor=3)
    assert "multiprocessing_context" not in options
    assert options["prefetch_factor"] == 3
    assert options["num_workers"] == 2 and not options["persistent_workers"]
    direct = worker_loader_options(dataset, 0)
    assert "multiprocessing_context" not in direct and "prefetch_factor" not in direct


@pytest.mark.parametrize(
    "threads,context",
    [
        (0, None),
        (-1, None),
        (True, "spawn"),
        (1.5, "spawn"),
        (8, None),
        (8, "fork"),
        (1, "invalid"),
    ],
)
def test_invalid_worker_configuration_rejected(threads, context):
    with pytest.raises(ValueError):
        TAPVid3DDataset(
            [], worker_cpu_threads=threads, worker_multiprocessing_context=context
        )
    with pytest.raises(ValueError):
        worker_loader_options(
            SimpleNamespace(
                worker_cpu_threads=threads, worker_multiprocessing_context=context
            ),
            1,
        )
