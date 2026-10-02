"""Training-only block alignment and GT-protecting regression distillation.

Teacher-forced alignment follows the block-stage idea of MOHAWK, not its
Transformer matrix fitting. The teacher-better gate is a regression upper-bound
variant; bounded gradient balancing is an experiment, not a quality guarantee.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .distillation import RefinerDistillationLoss, _finite_prediction, masked_mean


class SelectiveDistillationLoss(nn.Module):
    terms = ("residual", "xyz")

    def __init__(self, warmup_steps: int = 100) -> None:
        super().__init__()
        self.gt_criterion = RefinerDistillationLoss("A0", warmup_steps)
        self.warmup_steps = max(1, warmup_steps)

    def forward(
        self,
        student: dict[str, Any],
        teacher: dict[str, Any],
        target: dict[str, Tensor],
        step: int,
    ) -> dict[str, Tensor]:
        result = self.gt_criterion(student, teacher, target, step)
        xyz, teacher_xyz, gt = (
            student["xyz"].float(),
            teacher["xyz"].detach().float(),
            target["xyz"].float(),
        )
        finite = torch.isfinite(gt).all(-1)
        mask = target["valid"].bool() & target["visible"].bool() & finite
        gt = torch.where(finite[..., None], gt, torch.zeros_like(gt))
        scale = target["scale"].detach().float().reshape(-1, 1, 1, 1).clamp_min(1e-3)
        teacher_error = ((teacher_xyz - gt) / scale).norm(dim=-1)
        student_error = ((xyz.detach() - gt) / scale).norm(dim=-1)
        better = (teacher_error < student_error).detach()
        confidence = (torch.exp(-teacher_error / 0.1) * better).detach()

        def huber(diff: Tensor) -> Tensor:
            error = F.huber_loss(
                diff, torch.zeros_like(diff), delta=0.1, reduction="none"
            )
            if error.ndim == mask.ndim + 1:
                error = error.mean(-1)
            return masked_mean(error, mask, confidence)

        result["residual"] = huber(
            (student["dlog"].float() - teacher["dlog"].detach().float()) / 2
        )
        result["residual"] += huber(
            (student["duv"].float() - teacher["duv"].detach().float()) / 2
        )
        result["xyz"] = huber((xyz - teacher_xyz) / scale)
        result["kd"] = 0.5 * result["residual"] + 0.25 * result["xyz"]
        result["gate_rate"] = masked_mean(better.float(), mask)
        result["confidence_mean"] = masked_mean(confidence, mask)
        result["teacher_normalized_error"] = masked_mean(teacher_error, mask)
        result["student_normalized_error"] = masked_mean(student_error, mask)
        ramp = min(1.0, max(0.0, step / self.warmup_steps))
        result["total"] = result["gt"] + ramp * result["kd"]
        return result


def freeze_for_alignment(student: nn.Module) -> dict[str, bool]:
    previous = {key: p.requires_grad for key, p in student.named_parameters()}
    for key, parameter in student.named_parameters():
        parameter.requires_grad_(key.startswith("layers."))
        parameter.grad = None
    return previous


def restore_trainability(student: nn.Module, flags: dict[str, bool]) -> None:
    if set(flags) != {name for name, _ in student.named_parameters()}:
        raise ValueError("Trainability restoration requires unchanged parameter names")
    for name, parameter in student.named_parameters():
        parameter.requires_grad_(flags[name])
        parameter.grad = None


def block_alignment_loss(
    student: Any, teacher: Any, inputs: Sequence[Tensor], target: dict[str, Tensor]
) -> dict[str, Tensor]:
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError("Alignment requires an eval, frozen teacher")
    if len(student.layers) != len(teacher.layers):
        raise ValueError("Alignment requires corresponding blocks")
    captured_inputs: list[Tensor | None] = [None] * len(teacher.layers)
    captured_outputs: list[Tensor | None] = [None] * len(teacher.layers)
    handles = []
    for index, layer in enumerate(teacher.layers):

        def before(_module, args, index=index):
            captured_inputs[index] = args[0].detach()

        def after(_module, _args, output, index=index):
            if not isinstance(output, Tensor):
                raise ValueError("Teacher mixer must produce a tensor")
            captured_outputs[index] = output.detach()

        handles.append(layer.register_forward_pre_hook(before))
        handles.append(layer.register_forward_hook(after))
    try:
        with torch.no_grad():
            _finite_prediction(teacher.forward_train(*inputs))
    finally:
        for handle in handles:
            handle.remove()
    mask = (
        target["visible"].bool()
        & target["valid"].bool()
        & torch.isfinite(target["xyz"]).all(-1)
    )
    mask = mask.permute(0, 2, 1).flatten(0, 1)
    terms = []
    for layer, x, expected in zip(
        student.layers, captured_inputs, captured_outputs, strict=True
    ):
        if x is None or expected is None:
            raise ValueError("Teacher did not execute every corresponding mixer")
        observed = student._mix(layer, x)
        if observed.shape != expected.shape or observed.shape[:-1] != mask.shape:
            raise ValueError("Block/query geometry mismatch")
        if not torch.isfinite(observed).all() or not torch.isfinite(expected).all():
            raise ValueError("Nonfinite block alignment output")
        # A detached teacher RMS retains relative output amplitude as well as direction.
        rms = expected.float().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-3)
        error = ((observed.float() - expected.float()) / rms).square().mean(-1)
        terms.append(masked_mean(error, mask))
    return {
        "total": torch.stack(terms).mean(),
        **{f"block_{i}": term for i, term in enumerate(terms)},
    }


def gradient_balance(
    main: Tensor,
    auxiliary: Tensor,
    parameters: Sequence[nn.Parameter],
    *,
    target_ratio: float,
    max_scale: float,
    direction_gate: bool = True,
) -> dict[str, float]:
    if not (0 < target_ratio <= 1 and math.isfinite(max_scale) and max_scale > 0):
        raise ValueError("Gradient ratio/scale bounds must be finite and positive")
    parameters = tuple(p for p in parameters if p.requires_grad)
    if not parameters:
        raise ValueError("No trainable parameters for gradient diagnostics")
    gm = torch.autograd.grad(main, parameters, retain_graph=True, allow_unused=True)
    ga = torch.autograd.grad(
        auxiliary, parameters, retain_graph=True, allow_unused=True
    )
    reference = main.detach().float().new_zeros(())
    squared_main, squared_aux, dot = (
        reference.clone(),
        reference.clone(),
        reference.clone(),
    )
    for m, a in zip(gm, ga, strict=True):
        if m is not None:
            if not torch.isfinite(m).all():
                raise ValueError("Nonfinite main gradient")
            squared_main += m.detach().float().square().sum()
        if a is not None:
            if not torch.isfinite(a).all():
                raise ValueError("Nonfinite auxiliary gradient")
            squared_aux += a.detach().float().square().sum()
        if m is not None and a is not None:
            dot += (m.detach().float() * a.detach().float()).sum()
    nm, na = float(squared_main.sqrt()), float(squared_aux.sqrt())
    cosine = float(dot) / (nm * na) if nm > 0 and na > 0 else 0.0
    cosine = min(1.0, max(-1.0, cosine))
    scale = min(max_scale, target_ratio * nm / na) if nm > 0 and na > 0 else 0.0
    if direction_gate:
        scale *= max(0.0, cosine)
    return {
        "gt_norm": nm,
        "kd_norm": na,
        "cosine": cosine,
        "scale": scale,
        "effective_ratio": scale * na / nm if nm > 0 else 0.0,
    }
