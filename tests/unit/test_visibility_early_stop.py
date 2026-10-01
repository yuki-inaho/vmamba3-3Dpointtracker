"""Exercise training, early-stop checkpoint persistence and stopped-run resume."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn


def test_visibility_training_stops_and_resume_preserves_completed_step(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/train_flow_vis_head.py"
    spec = importlib.util.spec_from_file_location("visibility_training", path)
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)

    class TinyHead(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(()))

        def forward(self, first, second):
            return self.bias.expand(first.shape[:-1])

    monkeypatch.setattr(training, "FlowVisHead", TinyHead)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cache = tmp_path / "cache"
    for split in ["train", "heldout"]:
        folder = cache / split / "adt"
        folder.mkdir(parents=True)
        np.savez(folder / "one.npz", flow_fwd=np.zeros((3, 4, 2), np.float32),
                 flow_bwd=np.zeros((3, 4, 2), np.float32),
                 vis_gt=np.zeros((3, 4), np.float32), vis_flow=np.zeros((3, 4), np.float32))
    (cache / "splits.json").write_text(json.dumps({
        split: {"adt": ["one.npz"]} for split in ["train", "heldout"]
    }))
    out = tmp_path / "output"
    cfg = {"model": dict(dim=4, state_dim=4, num_heads=1, num_layers=1, bidirectional=False),
           "data": dict(cache_dir=str(cache), subsets=["adt"], max_frames=3,
                        clip_cache=1, num_points=4),
           "train": dict(out_dir=str(out), seed=0, pos_weight=1, lr=0,
                         steps=6, warmup=0, decay=0, grad_clip=1, log_every=1,
                         val_every=1, ckpt_every=1, early_stop_patience=1,
                         early_stop_min_delta=0.001)}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, "argv", [str(path), str(config)])
    training.main()
    initial = torch.load(out / "latest.pt", weights_only=False)
    assert initial["step"] == 2
    assert initial["extra"]["early_stop"]["since"] == 1
    status = json.loads((out / "training_status.json").read_text())
    assert status["reason"] == "early_stopping"
    assert len(json.loads((out / "checkpoints.json").read_text())["best"]) == 2
    training.main()
    resumed = torch.load(out / "latest.pt", weights_only=False)
    assert resumed["step"] == 2
    assert resumed["history"] == initial["history"]
    assert torch.equal(resumed["model"]["bias"], initial["model"]["bias"])
