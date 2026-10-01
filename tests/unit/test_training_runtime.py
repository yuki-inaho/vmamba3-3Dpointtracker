import copy
import json

import pytest
import torch
from torch import nn

from mamba3_tracker.train.runtime import (
    ClipBudget,
    CheckpointManager,
    build_optimizer,
    evaluation_weights,
    restore_checkpoint,
    manifest_train_paths,
)


def test_clip_budget_persists_actual_examples_and_disables_at_zero():
    disabled = ClipBudget()
    assert not disabled.enabled and not disabled.exhausted and disabled.remaining is None
    budget = ClipBudget(20_000, 4_989)
    budget.consume(32)
    assert budget.remaining == 14_979
    restored = ClipBudget(**budget.state_dict())
    restored.consume(14_979)
    assert restored.exhausted and restored.remaining == 0
    with pytest.raises(ValueError, match="at least one"):
        restored.consume(0)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(3, 4)
        self.hidden = nn.Linear(4, 4)
        self.head = nn.Linear(4, 1)
        self.frozen = nn.Parameter(torch.ones(4, 4), requires_grad=False)

    def forward(self, x):
        return self.head(self.hidden(self.embed(x)).tanh())


CFG = {
    "optimizer": "amuse",
    "lr": 0.001,
    "warmup": 2,
    "amuse": {"muon_lr": 0.02, "beta1": 0.8, "rho": 0.3},
}


def update(model, opt, x):
    opt.train()
    opt.zero_grad()
    model(x).square().mean().backward()
    opt.step()


def test_amuse_groups_and_eval_roundtrip():
    model = TinyModel()
    opt = build_optimizer(model, CFG)
    muon = {id(p) for g in opt.param_groups if g["use_muon"] for p in g["params"]}
    assert id(model.hidden.weight) in muon
    assert id(model.head.weight) not in muon
    assert id(model.embed.weight) not in muon
    assert all(id(model.frozen) != id(p) for g in opt.param_groups for p in g["params"])
    x = torch.randn(3, 3)
    for _ in range(4):
        update(model, opt, x)
    before = copy.deepcopy(model.state_dict())
    with evaluation_weights(opt):
        assert not opt.train_mode
        assert not torch.equal(before["hidden.weight"], model.hidden.weight)
    assert opt.train_mode
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=1e-6, rtol=1e-5)
    with pytest.raises(RuntimeError), evaluation_weights(opt):
        raise RuntimeError("validation failure")
    assert opt.train_mode


@pytest.mark.parametrize(
    "mode,scores,want",
    [
        ("min", [3.0, 1.0, 4.0, 2.0], [1.0, 2.0]),
        ("max", [3.0, 1.0, 4.0, 2.0], [4.0, 3.0]),
    ],
)
def test_retention_and_restart(tmp_path, mode, scores, want):
    model = TinyModel()
    opt = build_optimizer(model, CFG)
    mgr = CheckpointManager(tmp_path, k=2, mode=mode)
    for step, score in enumerate(scores, 1):
        update(model, opt, torch.randn(3, 3))
        mgr.save(step, model, opt, cfg={"train": CFG}, score=score)
    assert len(list(tmp_path.glob("*.pt"))) == 3
    rows = json.loads((tmp_path / "checkpoints.json").read_text())["best"]
    assert [row["score"] for row in rows] == want
    restarted = CheckpointManager(tmp_path, k=2, mode=mode)
    restarted.save(5, model, opt, cfg={"train": CFG}, score=float("nan"))
    assert len(list(tmp_path.glob("*.pt"))) == 3
    assert [r["score"] for r in restarted.best] == want


def test_amuse_saved_x_resume_matches_next_update(tmp_path):
    model = TinyModel()
    opt = build_optimizer(model, CFG)
    x = torch.randn(3, 3)
    for _ in range(4):
        update(model, opt, x)
    mgr = CheckpointManager(tmp_path, k=3)
    mgr.save(4, model, opt, cfg={"train": CFG})
    saved = torch.load(tmp_path / "latest.pt", weights_only=False)
    with evaluation_weights(opt):
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, saved["model"][name])
    update(model, opt, x)
    resumed = TinyModel()
    ropt = build_optimizer(resumed, CFG)
    state = restore_checkpoint(tmp_path / "latest.pt", resumed, ropt)
    assert state["step"] == 4
    update(resumed, ropt, x)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(
            value, resumed.state_dict()[name], atol=1e-6, rtol=1e-5
        )


def test_reject_optimizer_mismatch(tmp_path):
    model = TinyModel()
    opt = build_optimizer(model, CFG)
    CheckpointManager(tmp_path).save(0, model, opt, cfg={"train": CFG})
    with pytest.raises(ValueError, match="optimizer"):
        restore_checkpoint(
            tmp_path / "latest.pt", model, torch.optim.AdamW(model.parameters())
        )


def test_subset_manifest_requires_official_complete_selection(tmp_path):
    from mamba3_tracker.data.tapvid3d_splits import FULL_EVAL_FILES, MINIVAL_FILES

    manifest = tmp_path / "manifest.json"
    filename = FULL_EVAL_FILES["pstudio"][0]
    manifest.write_text(json.dumps({"files_by_subset": {"pstudio": [filename]}}))
    with pytest.raises(FileNotFoundError, match="missing"):
        manifest_train_paths(manifest, tmp_path, ["pstudio"])
    clip = tmp_path / "tapvid3d" / "pstudio" / filename
    clip.parent.mkdir(parents=True)
    clip.touch()
    assert manifest_train_paths(manifest, tmp_path, ["pstudio"]) == [clip]
    manifest.write_text(
        json.dumps({"files_by_subset": {"pstudio": [MINIVAL_FILES["pstudio"][0]]}})
    )
    with pytest.raises(ValueError, match="non-official"):
        manifest_train_paths(manifest, tmp_path, ["pstudio"])
