"""Deterministic online-teacher KD with exact-shape batches and strict resume."""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
import json
from pathlib import Path
import random
import sys
import time

import torch

from mamba3_tracker.deployment.checkpoint import sha256
from mamba3_tracker.deployment.runtime import INPUT_NAMES
from mamba3_tracker.deployment.student_checkpoint import (
    load_student,
    load_native_teacher,
    save_student,
    tensor_digest,
)
from mamba3_tracker.model.onnx_refiner import reference_depth
from mamba3_tracker.model.student_refiner import StudentRefiner
from mamba3_tracker.train.distillation import (
    DistillationWrapper,
    RefinerDistillationLoss,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_resume(state: dict, identity: dict) -> None:
    if state.get("identity") != identity:
        raise ValueError(
            "Resume identity mismatch: teacher/split/input/config/context/seed"
        )


def validate_long_inputs(
    train_manifest: dict, monitor_manifest: dict, window: int
) -> None:
    if window not in (16, 32, 64) or any(
        m.get("window") != window for m in (train_manifest, monitor_manifest)
    ):
        raise ValueError(
            "Long-window training requires matching train and monitor window inputs"
        )


class Inputs:
    def __init__(self, records: list[dict], limit_bytes: int = 4_000_000_000) -> None:
        self.records = records
        self.cache = OrderedDict()
        self.used = 0
        self.limit = limit_bytes

    def get(self, index: int) -> dict:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        record = self.records[index]
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Input artifact SHA mismatch: {path}")
        value = torch.load(path, weights_only=True, map_location="cpu")
        while self.cache and self.used + record["bytes"] > self.limit:
            previous, _ = self.cache.popitem(last=False)
            self.used -= self.records[previous]["bytes"]
        self.cache[index] = value
        self.used += record["bytes"]
        return value

    def batch(self, indices: list[int]):
        payloads = [self.get(i) for i in indices]
        values = [
            torch.cat([p["inputs"][name] for p in payloads]).cuda().float()
            for name in INPUT_NAMES[:-1]
        ]
        values.append(reference_depth(values[1]))
        target = {
            key: torch.cat([p["target"][key] for p in payloads]).cuda()
            for key in payloads[0]["target"]
        }
        return values, target


def read_manifest(path: Path, partition: str) -> dict:
    manifest = json.loads(path.read_text())
    if (
        manifest.get("status") != "complete"
        or manifest.get("partition") != partition
        or not manifest.get("records")
    ):
        raise ValueError(f"Incomplete or wrong-partition input manifest: {path}")
    return manifest


def atomic_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".partial")
    torch.save(state, temporary)
    temporary.replace(path)


@torch.no_grad()
def evaluate(wrapper, dataset, criterion):
    wrapper.eval()
    losses = []
    for i in range(len(dataset.records)):
        inputs, target = dataset.batch([i])
        student = wrapper.student.forward_train(*inputs)
        # A0 ignores teacher values; shared finite outputs avoid needless teacher evaluation.
        losses.append(float(criterion(student, student, target, 0)["gt"]))
    wrapper.train()
    return sum(losses) / len(losses)


def profile_batches(wrapper, dataset, criterion, candidates):
    index = max(range(len(dataset.records)), key=lambda i: dataset.records[i]["tracks"])
    measurements = []
    safe = []
    for count in candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            inputs, target = dataset.batch([index] * count)
            s, t = wrapper(*inputs)
            criterion(s, t, target, 100)["total"].backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            free, total = torch.cuda.mem_get_info()
            ok = (
                peak <= total - 2_000_000_000
                and free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
                >= 2_000_000_000
            )
            measurements.append(
                {
                    "microbatch": count,
                    "tracks": dataset.records[index]["tracks"],
                    "peak_bytes": peak,
                    "safe": ok,
                }
            )
            if ok:
                safe.append(count)
            del inputs, target, s, t
        except torch.cuda.OutOfMemoryError:
            measurements.append({"microbatch": count, "oom": True, "safe": False})
        finally:
            wrapper.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    if not safe:
        raise RuntimeError("No safe microbatch in the configured candidates")
    return max(safe), measurements


@torch.no_grad()
def fixed_probe(wrapper, dataset, criterion):
    totals = defaultdict(float)
    for index in range(min(4, len(dataset.records))):
        inputs, target = dataset.batch([index])
        student, teacher = wrapper(*inputs)
        for key, value in criterion(student, teacher, target, 100).items():
            totals[key] += float(value) / min(4, len(dataset.records))
    return dict(totals)


def main() -> None:
    from scripts.prepare_refiner_kd import atomic_json, read_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["smoke", "overfit", "pilot", "full", "long"], required=True
    )
    parser.add_argument("--ablation", choices=["A0", "A1", "A2", "A3", "A4"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Public student from preceding phase; resets optimizer",
    )
    parser.add_argument("--initialize-sha256")
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument(
        "--monitor-manifest",
        type=Path,
        default=Path(
            "result/refiner_kd_20261002/inputs_validation/input_manifest.json"
        ),
    )
    parser.add_argument("--microbatch", type=int, choices=[1, 4, 8, 16, 32])
    parser.add_argument("--window", type=int, choices=[16, 32, 64])
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.ablation is None:
        args.ablation = cfg.get("selected_ablation")
    if args.ablation is None:
        parser.error("ablation must be explicit or selected in config")
    if args.mode == "long":
        if (
            args.window is None
            or args.initialize_from is None
            or args.initialize_sha256 is None
            or args.input_manifest is None
        ):
            parser.error(
                "long requires window, explicit input manifest and pinned initialization checkpoint"
            )
    elif args.initialize_from is not None or args.initialize_sha256 is not None:
        parser.error("phase initialization is only valid for long mode")
    elif args.window is not None:
        parser.error("window may change only in long mode")
    if args.out_dir.exists() and args.resume is None:
        raise FileExistsError(f"Refusing to overwrite run: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.input_manifest = args.input_manifest or Path(
        "result/refiner_kd_20261002/"
        + ("cache_pilot" if args.mode in ("smoke", "overfit") else "inputs_train")
        + "/input_manifest.json"
    )
    train_manifest = read_manifest(args.input_manifest, "train")
    monitor_manifest = read_manifest(args.monitor_manifest, "validation")
    if args.mode == "long":
        validate_long_inputs(train_manifest, monitor_manifest, args.window)
    for manifest in (train_manifest, monitor_manifest):
        if (
            manifest["config_sha256"]
            != cfg.get("selection", {}).get("base_sha256", sha256(args.config))
            or manifest["split_sha256"] != cfg["data"]["split_sha256"]
        ):
            raise ValueError("Input/config provenance mismatch")
    records = (
        train_manifest["records"][:4]
        if args.mode == "overfit"
        else train_manifest["records"]
    )
    if args.mode in ("pilot", "full", "long") and len(records) != 1027:
        raise ValueError("Full train partition requires all 1027 prepared clips")
    train_data, monitor_data = Inputs(records), Inputs(monitor_manifest["records"])
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    teacher = load_native_teacher(
        Path(cfg["teacher"]["checkpoint"]), cfg["teacher"]["sha256"]
    ).cuda()
    teacher_digest = tensor_digest(teacher.state_dict())
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    student = StudentRefiner()
    student.initialize_common(teacher.state_dict())
    if args.initialize_from is not None:
        initialized = load_student(args.initialize_from, args.initialize_sha256)
        student.load_state_dict(initialized.state_dict(), strict=True)
        del initialized
    wrapper = DistillationWrapper(student.cuda(), teacher).cuda().train()
    initial_digest = tensor_digest(student.state_dict())
    criterion = RefinerDistillationLoss(args.ablation, cfg["training"]["warmup_steps"])
    monitor_criterion = RefinerDistillationLoss("A0")
    microbatch, profile = profile_batches(
        wrapper,
        train_data,
        criterion,
        [args.microbatch]
        if args.microbatch
        else cfg["training"]["microbatch_candidates"],
    )
    identity = {
        "teacher": cfg["teacher"]["sha256"],
        "split": cfg["data"]["split_sha256"],
        "config": sha256(args.config),
        "input_manifest": sha256(args.input_manifest),
        "monitor_manifest": sha256(args.monitor_manifest),
        "seed": args.seed,
        "mode": args.mode,
        "ablation": args.ablation,
        "microbatch": microbatch,
        "context": "exact_shape_no_padding_full_microbatch_z_ref",
        "initial_student": initial_digest,
    }
    if args.mode == "long":
        identity["window"] = args.window
        identity["initialization_sha256"] = args.initialize_sha256
    optimizer = torch.optim.AdamW(
        [p for p in wrapper.parameters() if p.requires_grad],
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )
    rng = random.Random(args.seed)
    step, processed, best, patience_best, bad = 0, 0, float("inf"), float("inf"), 0
    best_path = None
    if args.resume:
        state = torch.load(args.resume, weights_only=True, map_location="cpu")
        check_resume(state, identity)
        missing, unexpected = wrapper.load_state_dict(
            state["trainable_state"], strict=False
        )
        if unexpected or set(missing) != {
            k for k in wrapper.state_dict() if k.startswith("teacher.")
        }:
            raise ValueError("Resume must omit exactly the frozen teacher tensors")
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        step, processed, best, patience_best, bad = (
            state[k] for k in ("step", "processed", "best", "patience_best", "bad")
        )
        best_path = state["best_checkpoint"]
    atomic_json(
        args.out_dir / "run_config.json",
        {
            "config": cfg,
            "identity": identity,
            "microbatch_profile": profile,
            "teacher_tensor_sha256": teacher_digest,
        },
    )
    buckets = defaultdict(list)
    for i, record in enumerate(records):
        buckets[(record["frames"], record["tracks"], *record["depth_hw"])].append(i)
    groups = list(buckets.values())
    weights = [len(g) for g in groups]
    total_steps = cfg["training"][args.mode + "_steps"]
    initial_monitor = evaluate(wrapper, monitor_data, monitor_criterion)
    status = {
        "status": "running",
        "initial_monitor_gt_loss": initial_monitor,
        "identity": identity,
        "step": step,
        "initial_student_sha256": initial_digest,
    }
    status["initial_fixed_train_probe"] = fixed_probe(wrapper, train_data, criterion)
    atomic_json(args.out_dir / "training_status.json", status)
    start_time = time.monotonic()
    while step < total_steps and processed < cfg["training"]["processed_clip_limit"]:
        optimizer.zero_grad(set_to_none=True)
        lr = (
            cfg["training"]["long_lr"]
            if args.mode == "long"
            else cfg["optimizer"]["lr"]
        ) * min(1.0, (step + 1) / cfg["training"]["warmup_steps"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        effective = min(
            cfg["training"]["effective_batch"],
            cfg["training"]["processed_clip_limit"] - processed,
        )
        completed, sums = 0, defaultdict(float)
        term_grad_norms = {}
        clamp_rates = {}
        while completed < effective:
            bucket = rng.choices(groups, weights=weights, k=1)[0]
            indices = rng.sample(
                bucket, min(microbatch, effective - completed, len(bucket))
            )
            inputs, target = train_data.batch(indices)
            s, t = wrapper(*inputs)
            losses = criterion(s, t, target, step + 1)
            if completed == 0 and (
                step == 0 or (step + 1) % cfg["training"]["monitor_every"] == 0
            ):
                for term in ("gt", *criterion.terms):
                    gradients = torch.autograd.grad(
                        losses[term],
                        tuple(student.parameters()),
                        retain_graph=True,
                        allow_unused=True,
                    )
                    term_grad_norms[term] = float(
                        torch.stack(
                            [
                                g.detach().float().square().sum()
                                for g in gradients
                                if g is not None
                            ]
                        )
                        .sum()
                        .sqrt()
                    )
                clamp_rates = {
                    "dlog": float((s["dlog"].detach().abs() >= 2).float().mean()),
                    "duv": float((s["duv"].detach().abs() >= 1.99).float().mean()),
                }
            (losses["total"] * len(indices) / effective).backward()
            for key, value in losses.items():
                sums[key] += float(value.detach()) * len(indices) / effective
            completed += len(indices)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in wrapper.parameters() if p.requires_grad],
            cfg["optimizer"]["grad_clip"],
            error_if_nonfinite=True,
        )
        optimizer.step()
        step += 1
        processed += completed
        row = dict(
            step=step,
            processed_clips=processed,
            losses=dict(sums),
            lr=lr,
            grad_norm=float(grad_norm),
            elapsed_seconds=time.monotonic() - start_time,
        )
        row["term_grad_norms"] = term_grad_norms
        row["clamp_rates"] = clamp_rates
        validate = step % cfg["training"]["monitor_every"] == 0 or step == total_steps
        if validate:
            score = evaluate(wrapper, monitor_data, monitor_criterion)
            row["monitor_gt_loss"] = score
            if score < best:
                best = score
                best_path = str(
                    (args.out_dir / f"best_monitor_step{step}.pt").resolve()
                )
                save_student(student, Path(best_path))
            if score < patience_best - cfg["training"]["min_delta"]:
                patience_best, bad = score, 0
            else:
                bad += 1
            state = dict(
                schema_version=1,
                identity=identity,
                step=step,
                processed=processed,
                best=best,
                patience_best=patience_best,
                bad=bad,
                best_checkpoint=best_path,
                trainable_state={
                    k: v.detach().cpu()
                    for k, v in wrapper.state_dict().items()
                    if not k.startswith("teacher.")
                },
                optimizer=optimizer.state_dict(),
                python_rng=rng.getstate(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),
            )
            atomic_checkpoint(args.out_dir / "last.pt", state)
            if args.mode == "smoke":
                atomic_checkpoint(args.out_dir / f"step{step}.pt", state)
        with (args.out_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)
        status.update(
            step=step,
            processed_clips=processed,
            best_monitor_gt_loss=best if best_path else None,
            best_checkpoint=best_path,
        )
        atomic_json(args.out_dir / "training_status.json", status)
        if args.mode in ("pilot", "full") and bad >= cfg["training"]["patience"]:
            break
    if tensor_digest(teacher.state_dict()) != teacher_digest:
        raise RuntimeError("Teacher mutated during KD")
    if tensor_digest(student.state_dict()) == initial_digest:
        raise RuntimeError("Student did not update")
    status.update(
        status="complete",
        teacher_unchanged=True,
        exit_reason="early_stop"
        if bad >= cfg["training"]["patience"] and args.mode in ("pilot", "full")
        else "budget_complete",
        peak_vram_bytes=torch.cuda.max_memory_allocated(),
        elapsed_seconds=time.monotonic() - start_time,
        final_student_sha256=tensor_digest(student.state_dict()),
    )
    status["final_fixed_train_probe"] = fixed_probe(wrapper, train_data, criterion)
    atomic_json(args.out_dir / "training_status.json", status)


if __name__ == "__main__":
    main()
