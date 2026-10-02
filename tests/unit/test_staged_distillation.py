"""GT-protecting regression KD and teacher-forced block alignment."""

import pytest
import torch
from torch import nn


def _prediction(value):
    xyz = torch.full((1, 3, 2, 3), value, requires_grad=True)
    return {"xyz": xyz, "duv": xyz[..., :2], "dlog": xyz[..., 2]}


def _target(valid=True):
    return {
        "xyz": torch.zeros(1, 3, 2, 3),
        "visible": torch.ones(1, 3, 2, dtype=torch.bool),
        "valid": torch.full((1, 3, 2), valid, dtype=torch.bool),
        "scale": torch.ones(1),
    }


def test_teacher_worse_is_not_imitated():
    from mamba3_tracker.train.staged_distillation import SelectiveDistillationLoss

    criterion = SelectiveDistillationLoss()
    s, t = _prediction(0.1), _prediction(0.2)
    out = criterion(s, t, _target(), 100)
    assert out["kd"].item() == 0
    assert out["gate_rate"].item() == 0
    assert out["gt"].item() > 0
    out["total"].backward()
    assert s["xyz"].grad.abs().sum() > 0
    assert t["xyz"].grad is None


def test_better_teacher_is_detached_and_confidence_has_no_floor():
    from mamba3_tracker.train.staged_distillation import SelectiveDistillationLoss

    s, t = _prediction(2.0), _prediction(1.0)
    out = SelectiveDistillationLoss()(s, t, _target(), 100)
    assert out["gate_rate"].item() == 1
    assert 0 < out["confidence_mean"].item() < 0.05
    assert out["kd"].item() > 0
    out["total"].backward()
    assert t["xyz"].grad is None


def test_selective_all_masked_zero_and_nan_fails():
    from mamba3_tracker.train.staged_distillation import SelectiveDistillationLoss

    s, t = _prediction(0.2), _prediction(0.1)
    out = SelectiveDistillationLoss()(s, t, _target(False), 100)
    assert out["total"].item() == 0
    out["total"].backward()
    assert s["xyz"].grad.count_nonzero() == 0
    t["xyz"] = t["xyz"] * float("nan")
    with pytest.raises(ValueError, match="finite"):
        SelectiveDistillationLoss()(s, t, _target(False), 100)


def test_gradient_gate_conflict_and_bounded_ratio():
    from mamba3_tracker.train.staged_distillation import gradient_balance

    p = nn.Parameter(torch.tensor([1.0, 2.0]))
    main = p.square().sum()
    opposed = -0.001 * main
    info = gradient_balance(main, opposed, (p,), target_ratio=0.1, max_scale=100)
    assert info["cosine"] == pytest.approx(-1)
    assert info["scale"] == 0
    aligned = 0.001 * main
    info = gradient_balance(main, aligned, (p,), target_ratio=0.1, max_scale=50)
    assert info["scale"] == 50
    assert info["effective_ratio"] == pytest.approx(0.05)
    assert p.grad is None
    zero = main * 0
    info = gradient_balance(main, zero, (p,), target_ratio=0.1, max_scale=100)
    assert info["scale"] == 0
    assert info["cosine"] == 0


class ToyRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.common = nn.Linear(4, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

    def _mix(self, layer, x):
        return layer(x)

    def forward_train(self, x):
        x = self.common(x)
        for layer in self.layers:
            x = layer(x)
        return {"xyz": x}


def test_teacher_forced_blocks_freeze_common_and_remove_hooks():
    from mamba3_tracker.train.staged_distillation import (
        block_alignment_loss,
        freeze_for_alignment,
        restore_trainability,
    )

    torch.manual_seed(2)
    s, t = ToyRefiner(), ToyRefiner().eval().requires_grad_(False)
    common_before = {k: v.clone() for k, v in s.common.state_dict().items()}
    teacher_before = {k: v.clone() for k, v in t.state_dict().items()}
    flags = freeze_for_alignment(s)
    assert not any(p.requires_grad for p in s.common.parameters())
    assert all(p.requires_grad for p in s.layers.parameters())
    opt = torch.optim.SGD(s.layers.parameters(), lr=0.02)
    x = torch.randn(2, 3, 4)
    initial = block_alignment_loss(s, t, (x,), _target())["total"].item()
    for _ in range(15):
        opt.zero_grad()
        out = block_alignment_loss(s, t, (x,), _target())
        out["total"].backward()
        opt.step()
    assert block_alignment_loss(s, t, (x,), _target())["total"].item() < initial
    assert all(
        torch.equal(v, s.common.state_dict()[k]) for k, v in common_before.items()
    )
    assert all(torch.equal(v, t.state_dict()[k]) for k, v in teacher_before.items())
    assert all(p.grad is None for p in t.parameters())
    assert not any(
        layer._forward_hooks or layer._forward_pre_hooks for layer in t.layers
    )
    restore_trainability(s, flags)
    assert all(p.requires_grad for p in s.parameters())


def test_alignment_zero_mask_and_teacher_must_be_frozen():
    from mamba3_tracker.train.staged_distillation import block_alignment_loss

    s, t = ToyRefiner(), ToyRefiner().eval().requires_grad_(False)
    x = torch.randn(2, 3, 4)
    value = block_alignment_loss(s, t, (x,), _target(False))["total"]
    assert value.item() == 0
    value.backward()
    assert all(p.grad is None or not p.grad.count_nonzero() for p in s.parameters())
    t.layers[0].weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        block_alignment_loss(s, t, (x,), _target())
