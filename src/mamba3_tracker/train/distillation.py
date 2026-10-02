"""Frozen teacher, disposable heads and confidence-weighted regression KD."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from mamba3_tracker.model.student_refiner import StudentRefiner


def ablation_terms(name: str) -> tuple[str, ...]:
    terms = ("residual", "xyz", "feature", "temporal", "aux")
    counts = {"A0": 0, "A1": 2, "A2": 3, "A3": 4, "A4": 5}
    if name not in counts:
        raise ValueError(f"Unknown ablation: {name}")
    return terms[: counts[name]]


def masked_mean(
    value: Tensor, mask: Tensor, confidence: Tensor | None = None
) -> Tensor:
    weight = mask.to(value.dtype)
    numerator = value * weight
    if confidence is not None:
        numerator = numerator * confidence.detach()
    return numerator.sum() / weight.sum().clamp_min(1)


def occlusion_loss(logits: Tensor, visible: Tensor, valid: Tensor) -> Tensor:
    return masked_mean(
        F.binary_cross_entropy_with_logits(logits, visible.float(), reduction="none"),
        valid,
    )


class DistillationWrapper(nn.Module):
    """The deployable student owns neither the teacher nor auxiliary weights."""

    def __init__(self, student: StudentRefiner, teacher: StudentRefiner) -> None:
        super().__init__()
        self.student = student
        self.teacher = teacher.eval().requires_grad_(False)
        self.projectors = nn.ModuleList([nn.Linear(128, 128) for _ in range(2)])
        self.aux_heads = nn.ModuleList([nn.Linear(128, 3) for _ in range(2)])
        self.occlusion_head = nn.Linear(128, 1)

    def train(self, mode: bool = True) -> DistillationWrapper:
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, *inputs: Tensor) -> tuple[dict[str, Any], dict[str, Any]]:
        # Both calls receive the identical tensors, including full-microbatch z_ref.
        with torch.no_grad():
            teacher = self.teacher.forward_train(*inputs)
        student = self.student.forward_train(*inputs)
        student["projected_hidden"] = [
            p(h) for p, h in zip(self.projectors, student["hidden"])
        ]
        auxiliary = [head(h) for head, h in zip(self.aux_heads, student["hidden"])]
        student["aux"] = [
            {"dlog": a[..., 0].clamp(-2, 2), "duv": 2 * a[..., 1:].tanh()}
            for a in auxiliary
        ]
        student["occlusion"] = self.occlusion_head(student["final_hidden"]).squeeze(-1)
        return student, teacher


def _finite_prediction(prediction: dict[str, Any]) -> None:
    for name, value in prediction.items():
        if isinstance(value, Tensor) and not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite model output: {name}")
        if isinstance(value, dict):
            _finite_prediction(value)
        if isinstance(value, (tuple, list)):
            for item in value:
                _finite_prediction({"item": item})


class RefinerDistillationLoss(nn.Module):
    def __init__(self, ablation: str, warmup_steps: int = 100) -> None:
        super().__init__()
        self.terms = ablation_terms(ablation)
        self.warmup_steps = max(1, warmup_steps)

    def forward(
        self,
        student: dict[str, Any],
        teacher: dict[str, Any],
        target: dict[str, Tensor],
        step: int,
    ) -> dict[str, Tensor]:
        _finite_prediction(student)
        _finite_prediction(teacher)
        xyz, gt = student["xyz"].float(), target["xyz"].float()
        finite = torch.isfinite(gt).all(-1)
        valid = target["valid"].bool() & finite
        mask = valid & target["visible"].bool()
        gt = torch.where(finite[..., None], gt, torch.zeros_like(gt))
        scale = target["scale"].detach().float().reshape(-1, 1, 1, 1).clamp_min(1e-3)
        if not torch.isfinite(scale).all():
            raise ValueError("Nonfinite GT depth scale")
        pos = masked_mean(((xyz - gt) / scale).abs().sum(-1), mask)
        uv_reg = masked_mean(student["duv"].float().square().sum(-1), mask)
        result = {"gt": (10 * pos + uv_reg) / 11, "pos_3d": pos, "uv_reg": uv_reg}
        teacher_xyz = teacher["xyz"].detach().float()
        confidence = (
            torch.exp(-((teacher_xyz - gt) / scale).norm(dim=-1) / 0.1)
            .clamp(0.05, 1)
            .detach()
        )

        def huber(
            diff: Tensor, use_mask: Tensor = mask, weight: Tensor = confidence
        ) -> Tensor:
            error = F.huber_loss(
                diff, torch.zeros_like(diff), delta=0.1, reduction="none"
            )
            if error.ndim == use_mask.ndim + 1:
                error = error.mean(-1)
            return masked_mean(error, use_mask, weight)

        if "residual" in self.terms:
            result["residual"] = huber(
                (student["dlog"].float() - teacher["dlog"].detach().float()) / 2
            )
            result["residual"] = result["residual"] + huber(
                (student["duv"].float() - teacher["duv"].detach().float()) / 2
            )
            result["xyz"] = huber((xyz - teacher_xyz) / scale)
        if "feature" in self.terms:
            if len(student["projected_hidden"]) != 2 or len(teacher["hidden"]) != 2:
                raise ValueError("Feature KD requires exactly two corresponding taps")
            result["feature"] = torch.stack(
                [
                    masked_mean(
                        (
                            F.layer_norm(s.float(), (128,))
                            - F.layer_norm(t.detach().float(), (128,))
                        )
                        .square()
                        .mean(-1),
                        mask,
                    )
                    for s, t in zip(
                        student["projected_hidden"], teacher["hidden"], strict=True
                    )
                ]
            ).mean()
        if "temporal" in self.terms:
            # Raw XYZ cancels because teacher and student use the same raw inputs.
            correction_difference = (xyz - teacher_xyz) / scale
            result["temporal"] = huber(
                correction_difference[:, 1:] - correction_difference[:, :-1],
                mask[:, 1:] & mask[:, :-1],
                torch.minimum(confidence[:, 1:], confidence[:, :-1]),
            )
        if "aux" in self.terms:
            if len(student["aux"]) != 2:
                raise ValueError("Auxiliary KD requires exactly two correction heads")
            result["aux"] = torch.stack(
                [
                    huber((a["dlog"] - teacher["dlog"].detach()) / 2)
                    + huber((a["duv"] - teacher["duv"].detach()) / 2)
                    for a in student["aux"]
                ]
            ).mean()
            result["occlusion"] = occlusion_loss(
                student["occlusion"], target["visible"], valid
            )
            result["aux"] = result["aux"] + 0.1 * result["occlusion"]
        coefficients = dict(
            residual=0.5, xyz=0.25, feature=0.05, temporal=0.05, aux=0.1
        )
        ramp = min(1.0, max(0.0, step / self.warmup_steps))
        result["total"] = result["gt"] + ramp * sum(
            coefficients[k] * result[k] for k in self.terms
        )
        return result
