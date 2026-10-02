"""Strict staged KD on correctly reanchored FP32 frontend inputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import importlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
from torch import nn

from mamba3_tracker.deployment.checkpoint import sha256
from mamba3_tracker.deployment.runtime import INPUT_NAMES
from mamba3_tracker.deployment.student_checkpoint import (
    create_student,
    load_native_teacher,
    load_student,
    save_student,
    tensor_digest,
)
from mamba3_tracker.train.distillation import RefinerDistillationLoss
from mamba3_tracker.train.staged_distillation import (
    SelectiveDistillationLoss,
    block_alignment_loss,
    freeze_for_alignment,
    gradient_balance,
    restore_trainability,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_refiner_kd import (  # noqa: E402
    atomic_json,
    audit_reanchored_queries,
    read_config,
)
from scripts.train_refiner_kd import (  # noqa: E402
    Inputs,
    atomic_checkpoint,
    check_resume,
    evaluate,
    fixed_probe,
    profile_batches,
    read_manifest,
)


TRAIN_ARCHITECTURES = ("vssd_two_pool_128x2_v1", "vssd_local_global_128x2_v2_train")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--mode", choices=["smoke", "overfit", "pilot", "full", "long"], required=True
    )
    p.add_argument("--architecture", choices=TRAIN_ARCHITECTURES)
    p.add_argument(
        "--method", choices=["gt", "selective", "balanced", "staged"], required=True
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--input-manifest", type=Path, required=True)
    p.add_argument("--monitor-manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--microbatch", type=int, choices=[1, 4, 8, 16, 32])
    p.add_argument(
        "--early-stop",
        action="store_true",
        help="Enable validation-patience stopping outside full mode as well.",
    )
    p.add_argument("--initialize-from", type=Path)
    p.add_argument("--initialize-sha256")
    p.add_argument("--window", type=int, choices=[8, 16, 32])
    return p


def early_stopping_reached(mode: str, requested: bool, bad: int, patience: int) -> bool:
    return (mode == "full" or requested) and bad >= patience


def validate_options(args) -> None:
    if args.mode == "long":
        if args.initialize_from is None or args.initialize_sha256 is None:
            raise ValueError(
                "Long adaptation requires pinned initialization checkpoint and SHA"
            )
    elif args.initialize_from is not None or args.initialize_sha256 is not None:
        raise ValueError("Initialization checkpoint is valid only for long mode")


def validate_manifests(
    cfg: dict, train: dict, monitor: dict, config_sha: str, mode: str
) -> None:
    if cfg.get("schema_version") != 2:
        raise ValueError("Staged trainer requires schema2 improvement configuration")
    split = json.loads(Path(cfg["data"]["split"]).read_text())
    for manifest, partition in ((train, "train"), (monitor, "validation")):
        expected = {
            "status": "complete",
            "partition": partition,
            "config_sha256": config_sha,
            "split_sha256": cfg["data"]["split_sha256"],
            "window": cfg["data"]["window"],
            "query_policy": "strict_visible_gt_reanchor_v1",
            "frontend_precision": "fp32",
            "query_seed": 42,
            "window_seed": 42,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Input manifest provenance/window/precision/query policy mismatch"
            )
        if (
            not isinstance(manifest.get("preprocessing_sha256"), str)
            or len(manifest["preprocessing_sha256"]) != 64
        ):
            raise ValueError("Input manifest missing preprocessing provenance")
        clips = [r["clip"] for r in manifest.get("records", [])]
        required = split[partition]
        if (
            not clips
            or len(clips) != len(set(clips))
            or len(required) != len(set(required))
            or not set(clips) <= set(required)
        ):
            raise ValueError("Input clip duplicate/membership differs from fixed split")
        if mode in ("pilot", "full", "long"):
            count = 1027 if partition == "train" else 15
            if len(clips) != count or set(clips) != set(required):
                raise ValueError(
                    "Full training requires exact train1027/monitor15 split membership"
                )


def phase_budgets(cfg: dict, mode: str, method: str) -> tuple[int, int]:
    block = (
        (
            min(10, cfg["improvement"]["block_steps"])
            if mode in ("smoke", "overfit")
            else cfg["improvement"]["block_steps"]
        )
        if method == "staged" and mode != "long"
        else 0
    )
    return block, cfg["training"][mode + "_steps"]


def sampling_rng(seed: int, phase: str) -> random.Random:
    if phase not in ("block", "finetune"):
        raise ValueError("Unknown sampling phase")
    return random.Random(seed + 187 if phase == "block" else seed)


def make_resume_state(
    wrapper, optimizer, identity, progress, rng, trainability, *, cuda_rng=None
) -> dict:
    return {
        "schema_version": 2,
        "identity": copy.deepcopy(identity),
        **copy.deepcopy(progress),
        "trainable_state": {
            k: v.detach().cpu().clone()
            for k, v in wrapper.state_dict().items()
            if not k.startswith("teacher.")
        },
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "python_rng": rng.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if cuda_rng is None else cuda_rng,
        "student_trainability": copy.deepcopy(trainability),
    }


def validate_resume_state(
    state: dict, identity: dict, block_budget: int, ft_budget: int
) -> None:
    required = {
        "schema_version",
        "identity",
        "phase",
        "block_step",
        "ft_step",
        "block_processed",
        "ft_processed",
        "best",
        "patience_best",
        "bad",
        "best_checkpoint",
        "snapshots",
        "trainable_state",
        "optimizer",
        "python_rng",
        "torch_rng",
        "cuda_rng",
        "student_trainability",
        "alignment_evidence",
    }
    if (
        not isinstance(state, dict)
        or set(state) != required
        or state["schema_version"] != 2
    ):
        raise ValueError(
            "Resume requires the complete training schema, not a public model"
        )
    check_resume(state, identity)
    effective_batch = identity.get("effective_batch")
    clip_limit = identity.get("processed_clip_limit")
    if (
        type(effective_batch) is not int
        or effective_batch < 1
        or type(clip_limit) is not int
        or clip_limit < 1
    ):
        raise ValueError(
            "Resume requires positive integer batch and clip-limit budgets"
        )
    phase, block, ft = state["phase"], state["block_step"], state["ft_step"]
    if (
        phase not in ("block", "finetune")
        or type(block) is not int
        or type(ft) is not int
        or not (0 <= block <= block_budget and 0 <= ft <= ft_budget)
        or (phase == "block" and (block_budget == 0 or ft != 0))
        or (phase == "finetune" and block != block_budget)
    ):
        raise ValueError("Invalid resume phase/block/finetune progress")
    for key in ("block_processed", "ft_processed", "bad"):
        if type(state[key]) is not int or state[key] < 0:
            raise ValueError("Invalid training progress in resume")
    if (
        state["block_processed"] != block * effective_batch
        or state["ft_processed"] != min(ft * effective_batch, clip_limit)
        or ft > (clip_limit + effective_batch - 1) // effective_batch
        or state["bad"] > ft
        or (phase == "block" and (state["ft_processed"] != 0 or state["bad"] != 0))
    ):
        raise ValueError("Resume processed counts differ from phase/optimizer progress")
    evidence = state["alignment_evidence"]
    if (
        not isinstance(evidence, dict)
        or set(evidence)
        != {"common_initial_sha256", "common_unchanged", "block_peak_vram_bytes"}
        or not isinstance(evidence["common_initial_sha256"], str)
        or len(evidence["common_initial_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in evidence["common_initial_sha256"])
        or (
            phase == "finetune"
            and block_budget > 0
            and (
                evidence["common_unchanged"] is not True
                or type(evidence["block_peak_vram_bytes"]) is not int
                or evidence["block_peak_vram_bytes"] < 0
            )
        )
        or (
            (phase == "block" or block_budget == 0)
            and (
                evidence["common_unchanged"] is not None
                or evidence["block_peak_vram_bytes"] is not None
            )
        )
    ):
        raise ValueError("Invalid or missing block-alignment evidence in phase resume")


def restore_trainable_state(wrapper, optimizer, state: dict) -> None:
    expected = {k for k in wrapper.state_dict() if not k.startswith("teacher.")}
    if set(state["trainable_state"]) != expected:
        raise ValueError("Resume must omit exactly the frozen teacher keys")
    for key, value in state["trainable_state"].items():
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != wrapper.state_dict()[key].shape
            or value.dtype != wrapper.state_dict()[key].dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Invalid training resume tensor: {key}")
    missing, unexpected = wrapper.load_state_dict(
        state["trainable_state"], strict=False
    )
    if unexpected or set(missing) != {
        k for k in wrapper.state_dict() if k.startswith("teacher.")
    }:
        raise ValueError("Resume must omit exactly the frozen teacher keys")
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))


class TeacherStudentPair(nn.Module):
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher.eval().requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, *inputs):
        with torch.no_grad():
            teacher = self.teacher.forward_train(*inputs)
        return self.student.forward_train(*inputs), teacher


class StrictInputs(Inputs):
    def __init__(self, records, manifest, cfg):
        super().__init__(records)
        self.manifest = manifest
        teacher_config = json.loads(Path(cfg["teacher"]["run_config"]).read_text())
        self.depth_root = Path(teacher_config["data"]["da3_depth_root"])
        self.verified_sources: set[tuple[str, str]] = set()

    def get(self, index):
        payload = super().get(index)
        record = self.records[index]
        for key in (
            "config_sha256",
            "query_policy",
            "frontend_precision",
            "preprocessing_sha256",
            "query_seed",
            "window_seed",
        ):
            if payload.get(key) != self.manifest[key]:
                raise ValueError(f"Input payload provenance mismatch: {key}")
        if (
            payload.get("requested_window") != self.manifest["window"]
            or payload.get("clip") != record["clip"]
        ):
            raise ValueError("Input payload window/clip mismatch")
        inputs = payload["inputs"]
        if (
            set(inputs) != set(INPUT_NAMES[:-1])
            or inputs["dino_features"].dtype != torch.float32
            or tuple(inputs["dino_features"].shape[2:]) != (384, 28, 28)
            or list(inputs["ray"].shape[1:3]) != [record["frames"], record["tracks"]]
        ):
            raise ValueError("Prepared input shape/FP32 frontend contract mismatch")
        audit = audit_reanchored_queries(
            payload["queries_xyt"],
            payload["target"]["xyz"],
            payload["target"]["visible"],
            inputs["intrinsics"],
            896,
        )
        if payload.get("anchor_audit") != audit or any(
            record.get(k) != v for k, v in audit.items()
        ):
            raise ValueError("Prepared anchor audit mismatch")
        valid = payload["target"]["valid"].bool()
        xyz = payload["target"]["xyz"]
        if bool((valid & (~torch.isfinite(xyz).all(-1) | (xyz[..., 2] <= 1e-6))).any()):
            raise ValueError("Invalid GT admitted by prepared validity mask")
        raw = Path(payload["clip"])
        depth = self.depth_root / raw.parent.name / raw.name
        for path, expected in (
            (raw, payload["raw_sha256"]),
            (depth, payload["depth_sha256"]),
        ):
            identity = (str(path), expected)
            if identity not in self.verified_sources:
                if sha256(path) != expected:
                    raise ValueError(f"Raw/depth source hash mismatch: {path}")
                self.verified_sources.add(identity)
        return payload


def runtime_source_paths() -> dict[str, Path]:
    """Resolve only the native SISO and v1 arithmetic used by this trainer.

    These imports are already loaded by the teacher/student constructors. Hash
    their real source files, not merely a guessed vendored filename; a shadowed
    pip installation or non-source module fails closed. Unused MIMO, CUTE and
    autoregressive-step implementations are deliberately outside this scope.
    """
    modules = (
        "mamba3_tracker.model.official_mamba3",
        "visionmamba3",
        "visionmamba3.projections",
        "visionmamba3.mask",
        "visionmamba3.cross_attention",
        "mamba_ssm.modules.mamba3",
        "mamba_ssm.ops.triton.layernorm_gated",
        "mamba_ssm.ops.triton.mamba3.mamba3_siso_combined",
        "mamba_ssm.ops.triton.mamba3.mamba3_siso_fwd",
        "mamba_ssm.ops.triton.mamba3.mamba3_siso_bwd",
        "mamba_ssm.ops.triton.mamba3.angle_dt",
        "mamba_ssm.ops.triton.mamba3.utils",
    )
    resolved = {}
    for name in modules:
        module = importlib.import_module(name)
        try:
            source_file = module.__file__
            if not isinstance(source_file, str):
                raise ValueError("Expected an imported source filename")
            path = Path(source_file).resolve(strict=True)
            path.relative_to(ROOT)
            if not path.is_file() or path.suffix != ".py":
                raise ValueError("Expected a Python source file")
        except (TypeError, AttributeError, OSError, ValueError) as error:
            raise ValueError(
                f"Runtime source must be inside the repository: {name}"
            ) from error
        resolved[name] = path
    return resolved


def source_hashes() -> dict[str, str]:
    paths = [
        "scripts/train_refiner_staged.py",
        "scripts/train_refiner_kd.py",
        "scripts/prepare_refiner_kd.py",
        "src/mamba3_tracker/data/dataset.py",
        "src/mamba3_tracker/train/staged_distillation.py",
        "src/mamba3_tracker/train/distillation.py",
        "src/mamba3_tracker/model/student_refiner.py",
        "src/mamba3_tracker/model/onnx_refiner.py",
        "src/mamba3_tracker/model/local_global_refiner.py",
        "src/mamba3_tracker/model/compact_vssd.py",
        "src/mamba3_tracker/model/rep_temporal.py",
        "src/mamba3_tracker/deployment/student_checkpoint.py",
        "uv.lock",
    ]
    paths.extend(
        str(path.relative_to(ROOT)) for path in runtime_source_paths().values()
    )
    return {path: sha256(ROOT / path) for path in paths}


def optimizer_for(student, cfg, phase):
    lr = cfg["improvement"]["block_lr"] if phase == "block" else cfg["optimizer"]["lr"]
    return torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=cfg["optimizer"]["weight_decay"],
    )


def profile_alignment_batches(student, teacher, dataset, candidates, cfg):
    """Profile teacher-forced gradients independently from finetune, without steps."""
    flags = freeze_for_alignment(student)
    index = max(range(len(dataset.records)), key=lambda i: dataset.records[i]["tracks"])
    reports, safe = [], []
    try:
        for count in candidates:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                inputs, target = dataset.batch([index] * count)
                out = block_alignment_loss(student, teacher, inputs, target)
                out["total"].backward()
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                free, total = torch.cuda.mem_get_info()
                ok = (
                    peak <= total - 2_000_000_000
                    and free
                    + torch.cuda.memory_reserved()
                    - torch.cuda.memory_allocated()
                    >= 2_000_000_000
                )
                reports.append(
                    dict(phase="block", microbatch=count, peak_bytes=peak, safe=ok)
                )
                if ok:
                    safe.append(count)
                del inputs, target, out
            except torch.cuda.OutOfMemoryError:
                reports.append(
                    dict(phase="block", microbatch=count, safe=False, oom=True)
                )
            finally:
                student.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
    finally:
        restore_trainability(student, flags)
    if not safe:
        raise RuntimeError("No safe block-alignment microbatch")
    return max(safe), reports


def batches(records, rng, microbatch, effective):
    buckets = defaultdict(list)
    for i, record in enumerate(records):
        buckets[(record["frames"], record["tracks"], *record["depth_hw"])].append(i)
    groups = list(buckets.values())
    weights = [len(group) for group in groups]
    completed = 0
    while completed < effective:
        group = rng.choices(groups, weights=weights, k=1)[0]
        indices = rng.sample(group, min(microbatch, effective - completed, len(group)))
        yield indices
        completed += len(indices)


def status_progress(progress: dict[str, Any]) -> dict[str, Any]:
    """Keep public status phase counters aligned with optimizer checkpoints."""
    return dict(
        phase=progress["phase"],
        block_step=progress["block_step"],
        ft_step=progress["ft_step"],
        step=progress["ft_step"],
        block_processed=progress["block_processed"],
        ft_processed=progress["ft_processed"],
        processed_clips=progress["ft_processed"],
        bad=progress["bad"],
        best_checkpoint=progress["best_checkpoint"],
        best_monitor_gt_loss=progress["best"]
        if math.isfinite(progress["best"])
        else None,
        snapshots=progress["snapshots"],
        alignment_evidence=progress["alignment_evidence"],
    )


def main() -> None:
    args = parser().parse_args()
    validate_options(args)
    cfg = read_config(args.config)
    if cfg["schema_version"] != 2:
        raise ValueError("Staged trainer requires schema2")
    architecture = args.architecture or cfg["architecture"]
    if args.window is not None and args.window != cfg["data"]["window"]:
        raise ValueError("CLI/config window mismatch")
    train_manifest = read_manifest(args.input_manifest, "train")
    monitor_manifest = read_manifest(args.monitor_manifest, "validation")
    validate_manifests(
        cfg, train_manifest, monitor_manifest, sha256(args.config), args.mode
    )
    if args.out_dir.exists() and args.resume is None:
        raise FileExistsError(f"Refusing to overwrite training run: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = (
        train_manifest["records"][:4]
        if args.mode == "overfit"
        else train_manifest["records"]
    )
    train_data = StrictInputs(records, train_manifest, cfg)
    monitor_data = StrictInputs(monitor_manifest["records"], monitor_manifest, cfg)
    block_budget, ft_budget = phase_budgets(cfg, args.mode, args.method)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    teacher = load_native_teacher(
        Path(cfg["teacher"]["checkpoint"]), cfg["teacher"]["sha256"]
    ).cuda()
    teacher_digest = tensor_digest(teacher.state_dict())
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    student = create_student(architecture)
    student.initialize_common(teacher.state_dict())
    if args.initialize_from is not None:
        initialized = load_student(args.initialize_from, args.initialize_sha256)
        if initialized.architecture != architecture:
            raise ValueError(
                "Long initialization must retain unfused training architecture"
            )
        student.load_state_dict(initialized.state_dict(), strict=True)
        del initialized
    pair = TeacherStudentPair(student.cuda(), teacher).cuda().train()
    initial_digest = tensor_digest(student.state_dict())
    criterion = (
        RefinerDistillationLoss("A0")
        if args.method == "gt"
        else SelectiveDistillationLoss(cfg["training"]["warmup_steps"])
    )
    monitor_criterion = RefinerDistillationLoss("A0")
    candidates = (
        [args.microbatch]
        if args.microbatch
        else cfg["training"]["microbatch_candidates"]
    )
    microbatch, profile = profile_batches(pair, train_data, criterion, candidates)
    block_profile = []
    if block_budget:
        _, block_profile = profile_alignment_batches(
            student,
            teacher,
            train_data,
            [r["microbatch"] for r in profile if r.get("safe")],
            cfg,
        )
        common_safe = {r["microbatch"] for r in profile if r.get("safe")} & {
            r["microbatch"] for r in block_profile if r.get("safe")
        }
        if not common_safe:
            raise RuntimeError("No microbatch safe in both block and finetune phases")
        microbatch = max(common_safe)
    identity = dict(
        teacher=cfg["teacher"]["sha256"],
        split=cfg["data"]["split_sha256"],
        config=sha256(args.config),
        input_manifest=sha256(args.input_manifest),
        monitor_manifest=sha256(args.monitor_manifest),
        seed=args.seed,
        mode=args.mode,
        method=args.method,
        early_stopping_enabled=args.mode == "full" or args.early_stop,
        architecture=architecture,
        microbatch=microbatch,
        effective_batch=cfg["training"]["effective_batch"],
        processed_clip_limit=cfg["training"]["processed_clip_limit"],
        window=cfg["data"]["window"],
        query_policy=train_manifest["query_policy"],
        frontend_precision="fp32",
        initial_student=initial_digest,
        source_sha256=source_hashes(),
        initialization_sha256=args.initialize_sha256,
        context="exact_shape_no_padding_full_microbatch_z_ref",
    )
    flags = {key: p.requires_grad for key, p in student.named_parameters()}
    progress: dict[str, Any] = dict(
        phase="block" if block_budget else "finetune",
        block_step=0,
        ft_step=0,
        block_processed=0,
        ft_processed=0,
        best=float("inf"),
        patience_best=float("inf"),
        bad=0,
        best_checkpoint=None,
        snapshots=[],
        alignment_evidence=dict(
            common_initial_sha256=tensor_digest(
                {
                    k: v
                    for k, v in student.state_dict().items()
                    if not k.startswith("layers.")
                }
            ),
            common_unchanged=None,
            block_peak_vram_bytes=None,
        ),
    )
    resumed = None
    if args.resume is not None:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=True)
        validate_resume_state(resumed, identity, block_budget, ft_budget)
        if resumed["student_trainability"] != flags:
            raise ValueError(
                "Resume trainability flags differ from original architecture"
            )
        progress.update({key: resumed[key] for key in progress})
    if progress["phase"] == "block":
        freeze_for_alignment(student)
    optimizer = optimizer_for(student, cfg, progress["phase"])
    rng = sampling_rng(args.seed, progress["phase"])
    if resumed is not None:
        restore_trainable_state(pair, optimizer, resumed)
        rng.setstate(resumed["python_rng"])
        torch.set_rng_state(resumed["torch_rng"])
        torch.cuda.set_rng_state_all(resumed["cuda_rng"])
    atomic_json(
        args.out_dir / "run_config.json",
        dict(
            config=cfg,
            identity=identity,
            microbatch_profile=profile,
            finetune_microbatch_profile=profile,
            block_microbatch_profile=block_profile,
            teacher_tensor_sha256=teacher_digest,
            block_budget=block_budget,
            finetune_budget=ft_budget,
        ),
    )
    start = time.monotonic()
    status: dict[str, Any] = dict(
        status="running",
        identity=identity,
        **progress,
        initial_student_sha256=initial_digest,
        initial_monitor_gt_loss=evaluate(pair, monitor_data, monitor_criterion),
    )
    # JSON status must never serialize the checkpoint's initial infinity sentinels.
    status.pop("best")
    status.pop("patience_best")
    status["initial_fixed_train_probe"] = fixed_probe(pair, train_data, criterion)
    status["block_common_unchanged"] = progress["alignment_evidence"][
        "common_unchanged"
    ]
    atomic_json(args.out_dir / "training_status.json", status)
    torch.cuda.reset_peak_memory_stats()

    def checkpoint(name: str) -> None:
        state = make_resume_state(pair, optimizer, identity, progress, rng, flags)
        atomic_checkpoint(args.out_dir / "last.pt", state)
        if name != "last.pt":
            atomic_checkpoint(args.out_dir / name, state)

    def log(row: dict) -> None:
        row["elapsed_seconds"] = time.monotonic() - start
        row["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
        with (args.out_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)
        status.update(status_progress(progress))
        atomic_json(args.out_dir / "training_status.json", status)

    try:
        if progress["phase"] == "block":
            common_before = progress["alignment_evidence"]["common_initial_sha256"]
            while progress["block_step"] < block_budget:
                optimizer.zero_grad(set_to_none=True)
                sums = defaultdict(float)
                actual = []
                effective = cfg["training"]["effective_batch"]
                for indices in batches(records, rng, microbatch, effective):
                    inputs, target = train_data.batch(indices)
                    losses = block_alignment_loss(student, teacher, inputs, target)
                    (losses["total"] * len(indices) / effective).backward()
                    for key, value in losses.items():
                        sums[key] += float(value.detach()) * len(indices) / effective
                    actual.append(len(indices))
                norm = torch.nn.utils.clip_grad_norm_(
                    student.layers.parameters(),
                    cfg["optimizer"]["grad_clip"],
                    error_if_nonfinite=True,
                )
                optimizer.step()
                progress["block_step"] += 1
                progress["block_processed"] += effective
                if (
                    progress["block_step"] % cfg["training"]["monitor_every"] == 0
                    or progress["block_step"] == block_budget
                ):
                    checkpoint(f"block_step{progress['block_step']}.pt")
                log(
                    dict(
                        phase="block",
                        block_step=progress["block_step"],
                        losses=dict(sums),
                        lr=cfg["improvement"]["block_lr"],
                        grad_norm=float(norm),
                        actual_microbatches=actual,
                        block_processed=progress["block_processed"],
                        finetune_processed=0,
                    )
                )
            if (
                tensor_digest(
                    {
                        k: v
                        for k, v in student.state_dict().items()
                        if not k.startswith("layers.")
                    }
                )
                != common_before
            ):
                raise RuntimeError(
                    "Frozen common geometry changed during block alignment"
                )
            restore_trainability(student, flags)
            progress["alignment_evidence"]["common_unchanged"] = True
            progress["alignment_evidence"]["block_peak_vram_bytes"] = (
                torch.cuda.max_memory_allocated()
            )
            progress["phase"] = "finetune"
            optimizer = optimizer_for(student, cfg, "finetune")
            rng = sampling_rng(args.seed, "finetune")
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            status["block_common_unchanged"] = True
            status["post_alignment_monitor_gt_loss"] = evaluate(
                pair, monitor_data, monitor_criterion
            )
            checkpoint("finetune_start.pt")
            torch.cuda.reset_peak_memory_stats()

        exit_reason = (
            "early_stop"
            if early_stopping_reached(
                args.mode, args.early_stop, progress["bad"], cfg["training"]["patience"]
            )
            else "budget_complete"
        )
        while (
            progress["ft_step"] < ft_budget
            and progress["ft_processed"] < cfg["training"]["processed_clip_limit"]
            and exit_reason != "early_stop"
        ):
            optimizer.zero_grad(set_to_none=True)
            next_step = progress["ft_step"] + 1
            lr = (
                cfg["training"]["long_lr"]
                if args.mode == "long"
                else cfg["optimizer"]["lr"]
            ) * min(1.0, next_step / cfg["training"]["warmup_steps"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            effective = min(
                cfg["training"]["effective_batch"],
                cfg["training"]["processed_clip_limit"] - progress["ft_processed"],
            )
            sums, diagnostics = defaultdict(float), defaultdict(float)
            actual = []
            diagnosed_clip_count = 0
            ramp = min(1.0, next_step / cfg["training"]["warmup_steps"])
            for indices in batches(records, rng, microbatch, effective):
                inputs, target = train_data.batch(indices)
                s, t = pair(*inputs)
                losses = criterion(s, t, target, next_step)
                scale = 1.0
                if args.method in ("balanced", "staged") or (
                    not actual
                    and (
                        next_step == 1
                        or next_step % cfg["training"]["monitor_every"] == 0
                    )
                ):
                    info = gradient_balance(
                        losses["gt"],
                        losses.get("kd", losses["gt"] * 0),
                        tuple(student.layers.parameters()),
                        target_ratio=cfg["improvement"]["gradient_target_ratio"],
                        max_scale=cfg["improvement"]["gradient_max_scale"],
                        direction_gate=cfg["improvement"]["gradient_direction_gate"],
                    )
                    if args.method in ("balanced", "staged"):
                        scale = info["scale"]
                    diagnostic_weight = (
                        len(indices) / effective
                        if args.method in ("balanced", "staged")
                        else 1.0
                    )
                    for key, value in info.items():
                        diagnostics[key] += value * diagnostic_weight
                    diagnosed_clip_count += len(indices)
                if args.method != "gt":
                    losses["total"] = losses["gt"] + ramp * scale * losses["kd"]
                losses["applied_kd_scale"] = losses["gt"].new_tensor(
                    scale if args.method != "gt" else 0.0
                )
                (losses["total"] * len(indices) / effective).backward()
                for key, value in losses.items():
                    sums[key] += float(value.detach()) * len(indices) / effective
                actual.append(len(indices))
            norm = torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                cfg["optimizer"]["grad_clip"],
                error_if_nonfinite=True,
            )
            optimizer.step()
            progress["ft_step"] = next_step
            progress["ft_processed"] += effective
            row = dict(
                phase="finetune",
                step=next_step,
                losses=dict(sums),
                gradient_diagnostics=dict(diagnostics),
                gradient_diagnostics_scope="all_microbatch_weighted_mean"
                if args.method in ("balanced", "staged")
                else "first_microbatch",
                gradient_parameter_scope="student.layers",
                diagnosed_clip_count=diagnosed_clip_count,
                ramp=ramp,
                lr=lr,
                grad_norm=float(norm),
                actual_microbatches=actual,
                processed_clips=progress["ft_processed"],
                block_processed=progress["block_processed"],
            )
            validate = (
                next_step % cfg["training"]["monitor_every"] == 0
                or next_step == ft_budget
                or progress["ft_processed"] >= cfg["training"]["processed_clip_limit"]
            )
            if validate:
                score = evaluate(pair, monitor_data, monitor_criterion)
                public = args.out_dir / f"student_step{next_step}.pt"
                save_student(student, public)
                progress["snapshots"].append(
                    dict(
                        step=next_step,
                        path=str(public.resolve()),
                        sha256=sha256(public),
                        monitor_gt_loss=score,
                    )
                )
                row["monitor_gt_loss"] = score
                if score < progress["best"]:
                    progress["best"], progress["best_checkpoint"] = (
                        score,
                        str(public.resolve()),
                    )
                if score < progress["patience_best"] - cfg["training"]["min_delta"]:
                    progress["patience_best"], progress["bad"] = score, 0
                else:
                    progress["bad"] += 1
                checkpoint(f"step{next_step}.pt" if args.mode == "smoke" else "last.pt")
            log(row)
            if early_stopping_reached(
                args.mode, args.early_stop, progress["bad"], cfg["training"]["patience"]
            ):
                exit_reason = "early_stop"
                break
        if (
            progress["ft_processed"] >= cfg["training"]["processed_clip_limit"]
            and progress["ft_step"] < ft_budget
        ):
            exit_reason = "processed_clip_limit"
        checkpoint("last.pt")
        if tensor_digest(teacher.state_dict()) != teacher_digest:
            raise RuntimeError("Frozen teacher changed during staged distillation")
        final_digest = tensor_digest(student.state_dict())
        if final_digest == initial_digest:
            raise RuntimeError("Student did not update")
        status.update(
            status="complete",
            teacher_unchanged=True,
            exit_reason=exit_reason,
            final_student_sha256=final_digest,
            finetune_peak_vram_bytes=torch.cuda.max_memory_allocated(),
            block_peak_vram_bytes=progress["alignment_evidence"][
                "block_peak_vram_bytes"
            ],
            block_common_unchanged=progress["alignment_evidence"]["common_unchanged"],
            peak_vram_bytes=max(
                torch.cuda.max_memory_allocated(),
                progress["alignment_evidence"]["block_peak_vram_bytes"] or 0,
            ),
            elapsed_seconds=time.monotonic() - start,
            final_fixed_train_probe=fixed_probe(pair, train_data, criterion),
        )
        atomic_json(args.out_dir / "training_status.json", status)
    except Exception as error:
        status.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            elapsed_seconds=time.monotonic() - start,
        )
        atomic_json(args.out_dir / "training_status.json", status)
        raise


if __name__ == "__main__":
    main()
