import pytest
import torch

from mamba3_tracker.data.gpu_batch import AdaptiveBatchSize
from mamba3_tracker.data.frozen_cache import CachedFlow, FrozenTensorCache
from mamba3_tracker.data.storage import tree_bytes


def test_adaptive_reserves_memory_and_oom_does_not_regrow():
    batch = AdaptiveBatchSize(8, 64, reserve_gb=2)
    batch.success(8, 4e9)
    assert batch.current == 16
    assert batch.size(100, 4e9) == 3
    batch.oom(8)
    batch.success(4, 2e9)
    assert batch.current == 4
    assert batch.retries == 1
    with pytest.raises(torch.cuda.OutOfMemoryError):
        batch.oom(1)


def test_fixed_batch_and_invalid_limits():
    batch = AdaptiveBatchSize(16, 128, adaptive=False)
    batch.success(16, 5e9)
    assert batch.current == 16 and batch.size(4) == 4
    with pytest.raises(ValueError):
        AdaptiveBatchSize(0)


def test_disk_accounting_survives_concurrent_eviction(tmp_path):
    (tmp_path / "kept").write_bytes(b"123")

    class Disappeared:
        def stat(self):
            raise FileNotFoundError()

    class Root:
        def rglob(self, pattern):
            return [tmp_path / "kept", Disappeared()]

    assert tree_bytes(Root()) == 3


def test_pair_prefetch_batches_and_reuses_after_oom(tmp_path):
    class Model:
        device = "cpu"

        def __init__(self):
            self.calls = []

        def flow(self, first, second):
            self.calls.append(len(first))
            if len(first) > 2:
                raise torch.cuda.OutOfMemoryError("simulated small GPU")
            return (second - first)[:, :2]

    model = Model()
    cache = FrozenTensorCache(tmp_path, "flow", 1_000_000)
    wrapper = CachedFlow(model, cache, batch_size=4)
    images = torch.arange(5.0).view(5, 1, 1, 1).expand(5, 3, 4, 4).clone()
    wrapper.prefetch_clip(images)
    assert model.calls == [4, 2, 2, 2, 2]
    before = len(model.calls)
    for i in range(4):
        torch.testing.assert_close(wrapper.flow(images[i:i+1], images[i+1:i+2]),
                                   torch.ones(1, 2, 4, 4))
        torch.testing.assert_close(wrapper.flow(images[i+1:i+2], images[i:i+1]),
                                   -torch.ones(1, 2, 4, 4))
    wrapper.prefetch_clip(images)
    assert len(model.calls) == before
    assert wrapper.batch.retries == 1


def test_cached_batch_never_pairs_different_clips_and_reuses_bank(tmp_path):
    class Model:
        device = "cpu"

        def __init__(self):
            self.calls = 0

        def flow(self, first, second):
            self.calls += 1
            assert ((second - first).abs() < 10).all()
            return (second - first)[:, :2]

    model = Model()
    wrapper = CachedFlow(model, FrozenTensorCache(tmp_path, "flow", 1_000_000), batch_size=4)
    windows = torch.tensor([[0., 1., 2.], [100., 103., 106.]]).view(2, 3, 1, 1, 1)
    windows = windows.expand(2, 3, 3, 4, 4).clone()
    wrapper.prefetch_windows(windows)
    calls = model.calls
    for images in windows:
        wrapper.prefetch_clip(images)  # Must preserve the bank, not recompute it.
        for i in range(2):
            value = wrapper.flow(images[i:i+1], images[i+1:i+2])
            torch.testing.assert_close(value, (images[i+1:i+2]-images[i:i+1])[:, :2])
    assert model.calls == calls
    wrapper.release_prefetch()
    wrapper.prefetch_windows(windows)
    assert model.calls == calls
