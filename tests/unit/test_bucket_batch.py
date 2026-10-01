import json
from types import SimpleNamespace

import torch

from mamba3_tracker.data.bucket_batch import DepthBucketBatchSampler
from mamba3_tracker.data.batched_flow import BatchedFlow
from searaft_flow import track_clip


def test_buckets_keep_native_depth_shapes_and_admit_new_clip(tmp_path):
    paths = []
    shapes = [(504, 504)] * 5 + [(280, 504)] * 4
    depth = tmp_path / "depth"
    (depth / "subset").mkdir(parents=True)
    for i, shape in enumerate(shapes):
        path = tmp_path / "subset" / f"clip{i}.npz"
        paths.append(path)
        (depth / "subset" / (path.name + ".ready.json")).write_text(json.dumps({"shape": [8, *shape]}))
    dataset = SimpleNamespace(clip_paths=paths, da3_depth_root=depth)
    sampler = DepthBucketBatchSampler(dataset, 3)
    torch.manual_seed(17)
    first = list(sampler)
    torch.manual_seed(17)
    assert list(sampler) == first
    assert sorted(i for batch in first for i in batch) == list(range(9))
    assert all(len({shapes[i] for i in batch}) == 1 for batch in first)
    assert len(sampler) == 4
    dataset.clip_paths.append(tmp_path / "subset/new.npz")
    (depth / "subset/new.npz.ready.json").write_text(json.dumps({"shape": [8, 504, 504]}))
    assert sorted(i for batch in sampler for i in batch) == list(range(10))


def test_cross_clip_batched_flow_matches_tracking_without_extra_calls():
    class Flow:
        device = "cpu"
        calls = 0

        def flow(self, first, second):
            self.calls += 1
            return (second - first)[:, :2]

    images = torch.arange(8.0).reshape(2, 4, 1, 1, 1).expand(2, 4, 3, 16, 16).clone()
    plain = Flow()
    queries = torch.tensor([[8., 8.]])
    anchor = torch.tensor([0])
    reference = [track_clip(plain, clip, queries, anchor, 16) for clip in images]
    flow = Flow()
    batched = BatchedFlow(flow, initial=4, maximum=8)
    batched.prefetch_windows(images)
    calls = flow.calls
    actual = [track_clip(batched, clip, queries, anchor, 16) for clip in images]
    assert flow.calls == calls and calls < plain.calls
    for (uv, vis), (ref_uv, ref_vis) in zip(actual, reference):
        torch.testing.assert_close(uv, ref_uv)
        torch.testing.assert_close(vis, ref_vis)


def test_gpu_peak_estimate_uses_each_call_not_the_larger_previous_tail(monkeypatch):
    class FakeCudaTensor(torch.Tensor):
        @property
        def device(self):
            return torch.device("cuda")

    peak = [100]
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (10000, 10000))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 100)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 100)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: peak[0])
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: peak.__setitem__(0, 100))

    class Flow:
        device = "cuda"

        def flow(self, first, second):
            peak[0] = max(peak[0], 100 + len(first) * 10)
            return torch.zeros(len(first), 2, 2, 2)

    images = torch.zeros(1, 4, 3, 2, 2).as_subclass(FakeCudaTensor)
    flow = BatchedFlow(Flow(), initial=4, maximum=4, reserve_gb=0)
    flow.prefetch_windows(images)  # Four pairs followed by a smaller tail of two.
    assert flow.batch.bytes_per_item == 12.5
    assert flow.last_peak_allocated == 140
    assert flow.last_max_batch == 4
