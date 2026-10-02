"""Audit real staged GPU smoke and exact phase resume without changing models."""

import argparse
import json
import math
from pathlib import Path

import torch


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(equal(v, b[k]) for k, v in a.items())
        )
    if isinstance(a, (list, tuple)):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(equal(x, y) for x, y in zip(a, b, strict=True))
        )
    return type(a) is type(b) and a == b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--continuous", type=Path, required=True)
    p.add_argument("--resumed", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    dirs = [args.continuous, args.resumed]
    statuses = [json.loads((d / "training_status.json").read_text()) for d in dirs]
    checkpoints = [
        torch.load(d / "last.pt", weights_only=True, map_location="cpu") for d in dirs
    ]
    configs = [json.loads((d / "run_config.json").read_text()) for d in dirs]
    for s in statuses:
        assert s["status"] == "complete" and s["teacher_unchanged"] is True
        assert s["block_common_unchanged"] is True and s["step"] == 20
        assert s["processed_clips"] == 640 and s["block_processed"] == 320
    keys = (
        "identity",
        "trainable_state",
        "optimizer",
        "python_rng",
        "torch_rng",
        "cuda_rng",
        "student_trainability",
        "phase",
        "block_step",
        "ft_step",
        "block_processed",
        "ft_processed",
    )
    for key in keys:
        assert equal(checkpoints[0][key], checkpoints[1][key]), (
            f"Resume mismatch: {key}"
        )
    assert configs[0]["teacher_tensor_sha256"] == configs[1]["teacher_tensor_sha256"]
    metrics = [
        json.loads(line)
        for line in (args.continuous / "metrics.jsonl").read_text().splitlines()
    ]
    blocks = [r for r in metrics if r["phase"] == "block"]
    ft = [r for r in metrics if r["phase"] == "finetune"]
    assert blocks[-1]["losses"]["total"] < blocks[0]["losses"]["total"]
    assert (
        statuses[0]["final_fixed_train_probe"]["gt"]
        < statuses[0]["initial_fixed_train_probe"]["gt"]
    )
    for r in metrics:
        for v in r["losses"].values():
            assert math.isfinite(v)
        assert math.isfinite(r["grad_norm"])
    report = dict(
        success=True,
        resume_bitwise_keys=list(keys),
        resumed_tensor_count=len(checkpoints[0]["trainable_state"]),
        common_frozen_during_alignment=True,
        teacher_unchanged=True,
        block_loss_first=blocks[0]["losses"]["total"],
        block_loss_last=blocks[-1]["losses"]["total"],
        fixed_gt_first=statuses[0]["initial_fixed_train_probe"]["gt"],
        fixed_gt_last=statuses[0]["final_fixed_train_probe"]["gt"],
        monitor_gt_first=statuses[0]["initial_monitor_gt_loss"],
        monitor_gt_last=statuses[0]["best_monitor_gt_loss"],
        final_gate_rate=ft[-1]["losses"]["gate_rate"],
        final_cosine=ft[-1]["gradient_diagnostics"]["cosine"],
        final_applied_kd_gt_micro_mean_ratio=ft[-1]["ramp"]
        * ft[-1]["gradient_diagnostics"]["effective_ratio"],
        source_sha256=checkpoints[0]["identity"]["source_sha256"],
        scope="Two training clips, 15 short-window monitor clips, update/resume proof only; not full-frame accuracy",
    )
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
