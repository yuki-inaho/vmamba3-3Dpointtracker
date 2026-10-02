import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


spec = importlib.util.spec_from_file_location(
    "training_motion",
    Path(__file__).resolve().parents[2] / "scripts/train_depth_refined_tracker.py",
)
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def test_motion_check_uses_configured_depth_root(tmp_path, monkeypatch):
    depth_root = tmp_path / "custom_depth"
    (depth_root / "adt").mkdir(parents=True)
    np.savez(depth_root / "adt" / "clip.npz", depth=np.full((2, 4, 4), 2.0))
    xyz = torch.tensor([[[1.0, 1.0, 1.0]], [[2.0, 1.0, 1.0]]])
    clip = SimpleNamespace(
        F=2,
        W=4,
        H=4,
        images=torch.zeros(2, 3, 4, 4),
        queries_xyt=torch.tensor([[1.0, 1.0, 0.0]]),
        K=torch.eye(3),
        subset="adt",
        clip_id="clip",
        tracks_XYZ=xyz,
        visibility=torch.ones(2, 1),
    )
    monkeypatch.setattr(training, "load_clip", lambda path: clip)
    monkeypatch.setattr(
        training, "track_clip", lambda *args: (xyz[..., :2], torch.ones(2, 1))
    )

    def forward(*args, **kwargs):
        torch.testing.assert_close(args[3], torch.full((1, 2, 1), 2.0))
        return SimpleNamespace(xyz=xyz.unsqueeze(0))

    monkeypatch.setattr(training, "_model_forward", forward)
    model = torch.nn.Linear(1, 1)
    result = training._motion_check(
        model,
        "v35",
        None,
        [tmp_path / "adt" / "clip.npz"],
        torch.device("cpu"),
        torch.bfloat16,
        4,
        0.05,
        1.0,
        da3_depth_root=depth_root,
    )
    assert result["adt"] == pytest.approx(1.0)
    assert model.training
