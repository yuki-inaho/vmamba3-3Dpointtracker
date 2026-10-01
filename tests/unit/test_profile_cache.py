import pytest
import torch
from jaxtyping import TypeCheckError

from mamba3_tracker.data.frozen_cache import CachedFlow, FrozenTensorCache
from mamba3_tracker.data.synthetic import synthetic_batch
from mamba3_tracker.train.profiling import StageTimer, check_training_inputs


def test_runtime_shapes_reject_mismatched_frames_and_integer_images():
    batch = synthetic_batch(batch_size=2)
    values = [batch[k] for k in ("images", "ray", "depth_map", "target")]
    check_training_inputs(*values)
    with pytest.raises(TypeCheckError):
        check_training_inputs(values[0][:, :2], *values[1:])
    with pytest.raises(TypeCheckError):
        check_training_inputs(values[0].long(), *values[1:])


def test_ram_lru_bounds_storage_and_survives_disk_miss(tmp_path):
    cache = FrozenTensorCache(tmp_path, "ram", 100_000, ram_bytes=16)
    bank = torch.arange(10_000.)
    for offset in (0, 4):
        value = bank[offset:offset + 4]
        cache.put(cache.key(value), (value,))
    assert cache.ram_size == 16
    assert all(t.untyped_storage().nbytes() == 16 for values in cache._ram.values() for t in values)
    key = cache.key(bank[4:8])
    (cache.root / (key + ".pt")).unlink()
    found = cache.get(key, "cpu")
    assert found is not None
    torch.testing.assert_close(found[0], bank[4:8])
    assert cache.ram_hits == 1
    cache.configure_ram(0)
    assert cache.ram_size == 0 and not cache._ram
    with pytest.raises(ValueError):
        cache.configure_ram(-1)


def test_flow_legacy_keys_cpu_results_and_mutated_window(tmp_path):
    class Flow:
        device = "cpu"

        def flow(self, first, second):
            return (second - first)[:, :2]

    cache = FrozenTensorCache(tmp_path, "flow", 100_000, ram_bytes=100_000)
    wrapper = CachedFlow(Flow(), cache, batch_size=4)
    images = torch.arange(3.).view(3, 1, 1, 1).expand(3, 3, 4, 4).clone()
    legacy_key = cache.key(images[:1], images[1:2])
    wrapper.prefetch_windows(images.unsqueeze(0))
    assert (cache.root / (legacy_key + ".pt")).is_file()
    fwd, bwd = wrapper.consecutive_flows(images)
    torch.testing.assert_close(fwd[0], torch.ones(1, 2, 4, 4))
    torch.testing.assert_close(bwd[0], -fwd[0])
    images[1].add_(3)
    torch.testing.assert_close(wrapper.flow(images[:1], images[1:2]), torch.full((1, 2, 4, 4), 4.))
    wrapper.release_prefetch()
    assert not wrapper._batch_keys and wrapper._held_images is None


def test_profile_warmup_is_excluded():
    timer = StageTimer(torch.device("cpu"), warmup=1)
    for step in (1, 2):
        with timer.measure("work", step):
            sum(range(100))
    assert timer.report()["work"]["count"] == 1
