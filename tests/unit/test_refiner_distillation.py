"""Student size/geometry and explicit KD invariants; no downloaded fixtures."""

import pytest
import torch


def inputs(frames=3, tracks=4, batch=1):
    rng = torch.Generator().manual_seed(12)
    uv = torch.rand(batch, frames, tracks, 2, generator=rng) * 896
    depth = torch.rand(batch, frames, 11, 13, generator=rng) + 2
    z = torch.full((batch, frames, tracks), 2.5)
    k = torch.tensor([[500.0, 0, 448], [0, 500, 448], [0, 0, 1]]).repeat(batch, 1, 1)
    return (
        (uv - 448) / 500,
        z,
        torch.ones_like(z),
        uv,
        depth,
        torch.randn(batch, frames, 384, 28, 28, generator=rng),
        k,
        z.median() + 1e-6,
    )


def test_vssd_parameter_budget():
    from mamba3_tracker.model.student_refiner import StudentRefiner

    student = StudentRefiner()
    assert sum(p.numel() for p in student.parameters()) == 620502
    assert all(p.requires_grad for p in student.parameters())


def test_aux_absent_from_deploy():
    from mamba3_tracker.model.student_refiner import StudentRefiner

    student = StudentRefiner()
    assert not any(
        any(s in key for s in ("aux", "teacher", "dino."))
        for key in student.state_dict()
    )
    out = student(*inputs())
    assert [tuple(x.shape) for x in out] == [
        (1, 3, 4, 3),
        (1, 3, 4, 2),
        (1, 3, 4),
        (1, 3, 4, 2),
    ]


def test_features_match_inference_and_are_differentiable():
    from mamba3_tracker.model.student_refiner import StudentRefiner

    student, args = StudentRefiner(), inputs()
    train = student.forward_train(*args)
    for name, output in zip(("xyz", "uv_refined", "vis_logits", "duv"), student(*args)):
        torch.testing.assert_close(train[name], output, rtol=0, atol=0)
    assert len(train["hidden"]) == 2
    assert all(x.shape == (1, 3, 4, 128) for x in train["hidden"])
    train["xyz"].sum().backward()
    assert student.get_parameter("layers.0.q_proj.proj.weight").grad is not None


def test_same_batch_context():
    from mamba3_tracker.model.student_refiner import StudentRefiner

    student, args = StudentRefiner().eval(), inputs(batch=2)
    full = student(*args)
    parts = [student(*(x[b : b + 1] if x.ndim else x for x in args)) for b in range(2)]
    for j in range(4):
        torch.testing.assert_close(
            full[j], torch.cat([p[j] for p in parts]), rtol=1e-5, atol=1e-6
        )


def prediction(value=0.1):
    xyz = torch.full((1, 3, 4, 3), value, requires_grad=True)
    return {
        "xyz": xyz,
        "duv": xyz[..., :2],
        "dlog": xyz[..., 2],
        "vis_logits": xyz[..., 2],
        "hidden": (xyz.repeat(1, 1, 1, 43)[..., :128],) * 2,
    }


def loss_case(mask=True):
    from mamba3_tracker.train.distillation import RefinerDistillationLoss

    student, teacher = prediction(), prediction(0.2)
    target = {
        "xyz": torch.zeros(1, 3, 4, 3),
        "visible": torch.ones(1, 3, 4, dtype=torch.bool),
        "valid": torch.full((1, 3, 4), mask, dtype=torch.bool),
        "scale": torch.ones(1),
    }
    return RefinerDistillationLoss("A1"), student, teacher, target


def test_teacher_vis_is_not_distilled():
    loss, s, t, gt = loss_case()
    first = loss(s, t, gt, step=100)["total"]
    t["vis_logits"] = torch.full_like(t["vis_logits"], 999)
    torch.testing.assert_close(first, loss(s, t, gt, step=100)["total"], rtol=0, atol=0)


def test_all_masked_zero_grad():
    loss, s, t, gt = loss_case(False)
    value = loss(s, t, gt, step=100)["total"]
    assert value.item() == 0
    value.backward()
    assert torch.count_nonzero(s["xyz"].grad) == 0
    assert t["xyz"].grad is None


def test_nonfinite_fails():
    loss, s, t, gt = loss_case(False)
    s["xyz"] = s["xyz"] * float("nan")
    with pytest.raises(ValueError, match="finite"):
        loss(s, t, gt, step=100)


def test_confidence_does_not_cancel():
    from mamba3_tracker.train.distillation import masked_mean

    value, mask = torch.ones(2, 3), torch.ones(2, 3, dtype=torch.bool)
    assert masked_mean(
        value, mask, torch.full_like(value, 0.05)
    ).item() == pytest.approx(0.05)


def test_occlusion_invisible_is_supervised():
    from mamba3_tracker.train.distillation import occlusion_loss

    logits = torch.zeros(2, requires_grad=True)
    loss = occlusion_loss(
        logits, torch.tensor([True, False]), torch.ones(2, dtype=torch.bool)
    )
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([-0.25, 0.25]))


def test_teacher_is_frozen():
    from mamba3_tracker.model.student_refiner import StudentRefiner
    from mamba3_tracker.train.distillation import DistillationWrapper

    teacher = StudentRefiner()
    before = {k: v.clone() for k, v in teacher.state_dict().items()}
    wrapper = DistillationWrapper(StudentRefiner(), teacher).train()
    assert not teacher.training
    assert not any(p.requires_grad for p in teacher.parameters())
    s, t = wrapper(*inputs())
    s["xyz"].sum().backward()
    assert not t["xyz"].requires_grad
    assert all(p.grad is None for p in teacher.parameters())
    assert all(torch.equal(before[k], v) for k, v in teacher.state_dict().items())


def test_ablation_only_changes_loss_flags():
    from pathlib import Path
    import yaml
    from mamba3_tracker.train.distillation import ablation_terms

    assert ablation_terms("A0") == ()
    assert ablation_terms("A1") == ("residual", "xyz")
    assert ablation_terms("A4") == ("residual", "xyz", "feature", "temporal", "aux")
    with pytest.raises(ValueError):
        ablation_terms("unknown")
    cfg = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2] / "configs/distill_refiner_vssd.yaml"
        ).read_text()
    )
    assert all(
        tuple(terms) == ablation_terms(name) for name, terms in cfg["ablations"].items()
    )
    assert cfg["data"]["photometric_augment"] is False


def test_train_cli_strict_resume():
    from scripts.train_refiner_kd import check_resume

    identity = {"teacher": "a", "split": "b", "context": "c", "seed": 42}
    check_resume({"identity": identity}, identity)
    with pytest.raises(ValueError, match="identity"):
        check_resume({"identity": identity}, {**identity, "context": "changed"})


def test_a4_all_terms_finite_and_gt_only_matches_v35():
    from mamba3_tracker.model.student_refiner import StudentRefiner
    from mamba3_tracker.train.distillation import (
        DistillationWrapper,
        RefinerDistillationLoss,
    )
    from mamba3_tracker.model.heads import TrackerOutputs
    from mamba3_tracker.train.loss import TrackingLossV35

    wrapper = DistillationWrapper(StudentRefiner(), StudentRefiner())
    args = inputs()
    s, t = wrapper(*args)
    gt = {
        "xyz": s["xyz"].detach() + 0.2,
        "visible": args[2].bool(),
        "valid": torch.ones_like(args[2], dtype=torch.bool),
        "scale": torch.ones(1),
    }
    result = RefinerDistillationLoss("A4")(s, t, gt, 100)
    assert all(torch.isfinite(v) for v in result.values())
    result["total"].backward()
    assert wrapper.occlusion_head.weight.grad is not None
    # metre_space fixes the scale to 1 for an independent GT-loss reference.
    original = TrackingLossV35(
        {"pos_3D": 10 / 11, "reg_uv": 1 / 11, "metre_space": True}
    )
    reference = original(
        TrackerOutputs(
            xyz=s["xyz"],
            vis_logits=s["vis_logits"],
            spawn_logits=torch.zeros_like(s["vis_logits"]),
            delta_uv=s["duv"],
        ),
        gt["xyz"],
        gt["visible"],
        gt["valid"][:, 0],
        torch.zeros(1, 4, dtype=torch.long),
        args[6],
    )
    torch.testing.assert_close(result["gt"], reference.total)


def test_long_training_rejects_short_inputs():
    from scripts.train_refiner_kd import validate_long_inputs
    import pytest

    with pytest.raises(ValueError, match="window"):
        validate_long_inputs({"window": 8}, {"window": 16}, 16)
    with pytest.raises(ValueError, match="window"):
        validate_long_inputs({"window": 16}, {"window": 8}, 16)
    validate_long_inputs({"window": 16}, {"window": 16}, 16)
