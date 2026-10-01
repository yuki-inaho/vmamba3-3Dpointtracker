import torch

from mamba3_tracker.data.frozen_cache import FrozenTensorCache, CachedFlow, module_fingerprint
from mamba3_tracker.model.dino_encoder import DINOv2Encoder


def test_key_and_reuse(tmp_path):
    cache = FrozenTensorCache(tmp_path, "test", 100_000)
    first = torch.arange(12.0).reshape(3, 4)
    key = cache.key(first)
    assert key != cache.key(first + 1)
    assert key != cache.key(first.reshape(4, 3))
    assert key != cache.key(first.half())
    assert cache.get(key, "cpu") is None
    cache.put(key, (first,))
    torch.testing.assert_close(cache.get(key, "cpu")[0], first)
    assert cache.hits == cache.misses == 1
    assert FrozenTensorCache(tmp_path, "other model", 100_000).get(key, "cpu") is None


def test_lru_bound(tmp_path):
    cache = FrozenTensorCache(tmp_path, "test", 12_000)
    for i in range(8):
        value = torch.full((1500,), float(i))
        cache.put(cache.key(value), (value,))
    assert sum(p.stat().st_size for p in cache.root.glob("*.pt")) <= 12_000
    assert not list(cache.root.glob("*.partial"))


def test_changed_model_shares_storage_budget(tmp_path):
    for i in range(8):
        cache = FrozenTensorCache(tmp_path, f"model revision {i}", 12_000)
        value = torch.full((1500,), float(i))
        cache.put(cache.key(value), (value,))
    assert sum(p.stat().st_size for p in tmp_path.rglob("*.pt")) <= 12_000


def test_cpu_batch_slice_saves_only_its_own_storage(tmp_path):
    cache = FrozenTensorCache(tmp_path, "flow", 12_000)
    value = torch.arange(20_000.0)[:1500]
    key = cache.key(value)
    cache.put(key, (value,))
    assert (cache.root / (key + ".pt")).stat().st_size < 12_000
    torch.testing.assert_close(cache.get(key, "cpu")[0], value)


def test_flow_hit_skips_computation(tmp_path):
    class Model:
        device = "cpu"
        calls = 0

        def flow(self, first, second):
            self.calls += 1
            return second - first

    model = Model()
    wrapper = CachedFlow(model, FrozenTensorCache(tmp_path, "flow", 100_000))
    first, second = torch.ones(1, 3, 4, 4), torch.zeros(1, 3, 4, 4)
    torch.testing.assert_close(wrapper.flow(first, second), wrapper.flow(first, second))
    assert model.calls == 1
    wrapper.flow(second, first)
    assert model.calls == 2


def test_fingerprint_includes_model_config():
    class Config:
        setting = 1

        def to_dict(self):
            return {"setting": self.setting}

    model = torch.nn.Linear(2, 2)
    model.config = Config()
    before = module_fingerprint(model)
    model.config.setting = 2
    assert module_fingerprint(model) != before
    model.config.setting = 1
    assert module_fingerprint(model) == before


def test_scalar_state_fingerprint():
    model = torch.nn.BatchNorm2d(2)
    before = module_fingerprint(model)
    model.num_batches_tracked.add_(1)
    assert module_fingerprint(model) != before


def test_frozen_backbone_stays_eval_with_trainable_fusion():
    # Build without fetching weights; exercise recursive parent train() calls.
    encoder = DINOv2Encoder.__new__(DINOv2Encoder)
    torch.nn.Module.__init__(encoder)
    encoder.backbone = torch.nn.Sequential(torch.nn.Dropout(0.5))
    encoder.fuse_proj = torch.nn.Linear(2, 2)
    parent = torch.nn.Sequential(encoder)
    parent.train()
    assert encoder.training and encoder.fuse_proj.training
    assert not encoder.backbone.training
    parent.eval()
    assert not encoder.training and not encoder.fuse_proj.training
