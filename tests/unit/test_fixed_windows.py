from pathlib import Path

import pytest

from mamba3_tracker.data.dataset import TAPVid3DDataset


def test_fixed_frame_window_is_worker_rng_independent(monkeypatch):
    import mamba3_tracker.data.dataset as module

    def capture_window(path, frames):
        raise RuntimeError(str(frames))

    monkeypatch.setattr(module, "peek_clip_F", lambda _: 300)
    monkeypatch.setattr(module, "load_clip", capture_window)
    dataset = TAPVid3DDataset([Path("adt/clip.npz")], window_size=8, fixed_window_seed=42)
    windows = []
    for seed in (1, 10, 100):
        dataset._rng.seed(seed)
        with pytest.raises(RuntimeError) as captured:
            dataset[0]
        windows.append(str(captured.value))
    assert len(set(windows)) == 1
    with pytest.raises(ValueError, match="augmentation off"):
        TAPVid3DDataset([], augment=True, fixed_window_seed=42)


def test_extends_preserves_raw_weights_and_rejects_cycles(tmp_path):
    from mamba3_tracker.train.config import load_config
    base = tmp_path / "base.yaml"
    base.write_text("version: v35\nmodel: {dim: 128}\nloss: {weights: {pos_3D: 1.0, reg_uv: 0.1}}\n")
    child = tmp_path / "child.yaml"
    child.write_text("extends: base.yaml\nmodel: {temporal_mixer: official_mamba3}\n")
    cfg = load_config(child)
    assert cfg["model"]["dim"] == 128
    assert cfg["loss"]["weights_raw"] == {"pos_3D": 1.0, "reg_uv": 0.1}
    assert sum(cfg["loss"]["weights"].values()) == pytest.approx(1.)
    base.write_text("extends: child.yaml\n")
    with pytest.raises(ValueError, match="cyclic"):
        load_config(child)
