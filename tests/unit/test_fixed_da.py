import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from jaxtyping import TypeCheckError

from mamba3_tracker.data.dataset import TAPVid3DDataset
from mamba3_tracker.data.dataset import collate_tracking
from mamba3_tracker.data.fixed_da import (
    DAIndex, FixedDABatchSampler, PhotometricPattern, apply_pattern, parse_patterns, prewarm_loader,
)
from mamba3_tracker.data.frozen_cache import CachedFlow, FrozenTensorCache
from mamba3_tracker.train.config import load_config


PATTERNS = tuple(PhotometricPattern(n, b, c) for n, b, c in
                 [('dark', .8, 1.), ('bright', 1.2, 1.),
                  ('soft', 1., .8), ('hard', 1., 1.2)])


def test_fixed_da_reproduces_images_and_preserves_geometry(monkeypatch):
    import mamba3_tracker.data.dataset as module
    base = torch.linspace(.1, .7, 3 * 16 * 16).reshape(1, 3, 16, 16).repeat(8, 1, 1, 1)

    def clip(path, frames):
        return SimpleNamespace(images=base.clone(), tracks_XYZ=torch.ones(8, 2, 3),
                               visibility=torch.ones(8, 2, dtype=torch.bool), H=16, W=16,
                               N_q=2, queries_xyt=torch.tensor([[4., 4., float(frames[0])]]).repeat(2, 1),
                               K=torch.eye(3), clip_id='clip', subset='adt')

    monkeypatch.setattr(module, 'peek_clip_F', lambda _: 80)
    monkeypatch.setattr(module, 'load_clip', clip)
    ds = TAPVid3DDataset([Path('adt/clip.npz')], window_size=8, augment=True,
                        fixed_window_seed=42, fixed_patterns=PATTERNS, image_size=16)
    items = []
    for p in range(4):
        ds._rng.seed(2)
        first = ds[DAIndex(0, p, (0,))]
        ds._rng.seed(999)
        repeated = ds[DAIndex(0, p, (0,))]
        assert torch.equal(first['images'], repeated['images'])
        items.append(first)
    assert len({FrozenTensorCache.key(None, x['images']) for x in items}) == 4
    for item in items:
        for key in ('frame_start', 'queries_xyt', 'tracks_XYZ', 'visibility', 'K'):
            if isinstance(item[key], torch.Tensor):
                assert torch.equal(item[key], items[0][key])
            else:
                assert item[key] == items[0][key]
    with pytest.raises(ValueError, match='explicit pattern'):
        ds[0]
    with pytest.raises(TypeCheckError):
        apply_pattern(base.long(), PATTERNS[0])


def make_sampler(tmp_path, **kwargs):
    root = tmp_path / 'depth' / 'adt'
    root.mkdir(parents=True, exist_ok=True)
    paths = [tmp_path / 'adt' / f'{i}.npz' for i in range(9)]
    for i, p in enumerate(paths):
        (root / (p.name + '.ready.json')).write_text(json.dumps({'shape': [8, 16 if i < 5 else 8, 16]}))
    ds = SimpleNamespace(clip_paths=paths, da3_depth_root=root.parent)
    return FixedDABatchSampler(ds, 4, 3, block_clips=3, repeats=2, seed=7, **kwargs)


def test_blocks_cover_every_variant_and_resume_ignores_prefetch(tmp_path):
    sampler = make_sampler(tmp_path)
    plan = list(sampler)
    assert sampler.offset == 0  # Merely prefetching must not advance checkpoints.
    counts = Counter((x.clip, x.pattern) for b in plan for x in b)
    assert len(counts) == 9 * 4 and set(counts.values()) == {2}
    assert all(len({x.clip < 5 for x in batch}) == 1 for batch in plan)
    assert max(len(batch[0].block) for batch in plan) == 3
    for batch in plan[:3]:
        sampler.acknowledge(batch[0].epoch, batch[0].position)
    restored = make_sampler(tmp_path)
    restored.restore(sampler.state_dict())
    assert list(restored) == plan[3:]
    assert not restored.covered_once
    for batch in plan[3:]:
        restored.acknowledge(batch[0].epoch, batch[0].position)
    assert restored.covered_once
    second = list(restored)
    assert second[0][0].epoch == 1
    assert Counter((x.clip, x.pattern) for b in second for x in b) == counts
    assert second != plan
    changed = make_sampler(tmp_path, recipe_signature='different brightness')
    with pytest.raises(ValueError, match='signature'):
        changed.restore(sampler.state_dict())


def test_da_flow_cache_reuse_has_no_extra_inference(tmp_path):
    class Flow:
        device = 'cpu'
        calls = 0

        def flow(self, first, second):
            self.calls += 1
            return (second - first)[:, :2]

    base = torch.linspace(.1, .6, 3 * 3 * 16 * 16).reshape(3, 3, 16, 16)
    variants = torch.stack([apply_pattern(base, p) for p in PATTERNS])
    flow = Flow()
    cache = FrozenTensorCache(tmp_path, 'da-flow', 1_000_000)
    wrapper = CachedFlow(flow, cache, batch_size=4, max_batch_size=8)
    wrapper.prefetch_windows(variants)
    wrapper.release_prefetch()
    calls = flow.calls
    wrapper.prefetch_windows(variants)
    for images in variants:
        forward, _ = wrapper.consecutive_flows(images)
        for i, value in enumerate(forward):
            torch.testing.assert_close(value, (images[i+1:i+2] - images[i:i+1])[:, :2])
    assert flow.calls == calls and cache.misses == 16


def test_da4_recipe_is_a_new_warm_start_with_bounded_cache():
    cfg = load_config('configs/v64_official_mamba3_da4_cached.yaml')
    assert len(parse_patterns(cfg['data']['fixed_da']['patterns'])) == 4
    assert cfg['data']['fixed_window_seed'] == 42
    assert not cfg['train']['init_ckpt']
    assert cfg['train']['init_best_from'] == 'result/v64_official_mamba3_cache_phase'
    assert not cfg['model']['pretrained_mamba3']
    assert cfg['frozen_cache']['total_data_budget_gb'] == 120
    assert cfg['frozen_cache']['block_prewarm'] and not cfg['frozen_cache']['prewarm']


def test_prewarm_matches_worker_resize_pixels_and_keys(monkeypatch):
    import mamba3_tracker.data.dataset as module
    from torch.utils.data import DataLoader
    from mamba3_tracker.data.bucket_batch import seed_tracking_worker

    native = torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(7))

    def clip(path, frames):
        return SimpleNamespace(images=native.clone(), tracks_XYZ=torch.ones(2, 2, 3),
                               visibility=torch.ones(2, 2, dtype=torch.bool), H=64, W=64,
                               N_q=2, queries_xyt=torch.tensor([[4., 4., 0.]]).repeat(2, 1),
                               K=torch.eye(3), clip_id='clip', subset='adt')

    monkeypatch.setattr(module, 'peek_clip_F', lambda _: 2)
    monkeypatch.setattr(module, 'load_clip', clip)
    ds = TAPVid3DDataset([Path('adt/clip.npz')], window_size=2, augment=True,
                        fixed_window_seed=42, fixed_patterns=PATTERNS, image_size=256)
    indices = [DAIndex(0, p, (0,)) for p in range(4)]
    old_threads = torch.get_num_threads()
    torch.set_num_threads(8)
    try:
        training = DataLoader(ds, batch_size=4, sampler=indices, num_workers=1,
                              collate_fn=collate_tracking, worker_init_fn=seed_tracking_worker)
        warm = prewarm_loader(ds, indices, 4, training_workers=1)
        a, b = next(iter(training)).images, next(iter(warm)).images
        assert torch.equal(a, b)
        assert FrozenTensorCache.key(None, a) == FrozenTensorCache.key(None, b)
        # The main-process context must also match when training has no workers.
        direct = torch.stack([ds[i]['images'] for i in indices])
        no_workers = next(iter(prewarm_loader(ds, indices, 4, training_workers=0))).images
        assert torch.equal(direct, no_workers)
        assert warm.num_workers == 1 and warm.prefetch_factor == 1
    finally:
        torch.set_num_threads(old_threads)
