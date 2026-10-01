"""Smoke the real refiner + visibility head with analytic artificial data.

uv run python scripts/smoke_train_synthetic.py --steps 6
uv run python scripts/smoke_train_synthetic.py --steps 8 --resume
Frozen WAFT/DA3 measurements are generated analytically; DINOv3 is real and frozen.
This checks training plumbing only and produces no official benchmark scores.
"""

import argparse
import cProfile
import json
import pstats
from pathlib import Path

from huggingface_hub import snapshot_download
import torch
from torch.utils.tensorboard import SummaryWriter

from mamba3_tracker.data.synthetic import synthetic_batch
from mamba3_tracker.data.frozen_cache import FrozenTensorCache, module_fingerprint
from mamba3_tracker.model.depth_refined_tracker import Mamba3V35Refiner
from mamba3_tracker.model.flow_vis_head import FlowVisHead
from mamba3_tracker.model.dino_encoder import DINOv2Encoder
from mamba3_tracker.train.profiling import StageTimer, check_training_inputs
from mamba3_tracker.train.runtime import (
    CheckpointManager,
    build_optimizer,
    evaluation_weights,
    restore_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--out-dir", type=Path, default=Path("result/synthetic_smoke"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--frozen-cache", action="store_true",
                        help="Reuse frozen DINO features on disk; check cached/uncached equality")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--points", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--dino-image-size", type=int, default=64)
    parser.add_argument("--skip-visibility", action="store_true")
    parser.add_argument("--profile-flow-cache", action="store_true",
                        help="Measure real tracking/cache plumbing using analytic frozen flow")
    parser.add_argument("--cache-ram-gb", type=float, default=0)
    parser.add_argument("--fixed-da-config", type=Path,
                        help="Prewarm and train on the finite DA patterns from this recipe")
    parser.add_argument("--temporal-mixer", choices=["vssd_cross", "official_mamba3"],
                        default="vssd_cross")
    parser.add_argument("--mamba3-checkpoint", type=Path,
                        default=Path("weights/mamba3-siso-187m/pytorch_model.bin"))
    parser.add_argument(
        "--dino-revision", default="114c1379950215c8b35dfcd4e90a5c251dde0d32"
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if min(args.batch_size, args.frames, args.points, args.image_size, args.dino_image_size) < 1:
        parser.error("batch, frame, point and image dimensions must be positive")
    if args.profile and not 0 <= args.profile_warmup < args.steps:
        parser.error("profile warmup must leave at least one measured step")
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    # Resolve local snapshot so a gated HTTP request/token is unnecessary on smoke runs.
    dino = snapshot_download(
        "facebook/dinov3-vits16-pretrain-lvd1689m",
        revision=args.dino_revision,
        local_files_only=True,
    )
    refiner = Mamba3V35Refiner(
        two_pool=args.temporal_mixer == "vssd_cross", temporal_mixer=args.temporal_mixer,
        dino_model=dino, dino_image_size=args.dino_image_size, image_size=args.image_size
    ).to(args.device)
    pretrained_report = None
    dino_encoder = refiner.dino
    assert isinstance(dino_encoder, DINOv2Encoder)
    if args.temporal_mixer == "official_mamba3":
        from mamba3_tracker.model.official_mamba3 import load_pretrained_mixers
        pretrained_report = load_pretrained_mixers(refiner.layers, args.mamba3_checkpoint)
        print(f"[smoke] loaded {pretrained_report['loaded_parameters']:,} official parameters", flush=True)
    head = FlowVisHead().to(args.device)
    batch_options = dict(device=args.device, frames=args.frames, points=args.points,
                         image_size=args.image_size, batch_size=args.batch_size)
    train_batch = synthetic_batch(args.seed, **batch_options)
    train_variants = [train_batch]
    pattern_names = []
    if args.fixed_da_config:
        from mamba3_tracker.data.fixed_da import apply_pattern, parse_patterns
        from mamba3_tracker.train.config import load_config
        patterns = parse_patterns(load_config(args.fixed_da_config)["data"]["fixed_da"]["patterns"])
        train_variants = [{**train_batch, "images": torch.stack([
            apply_pattern(images, p) for images in train_batch["images"]
        ])} for p in patterns]
        pattern_names = [p.name for p in patterns]
    val_batch = synthetic_batch(args.seed + 1, **batch_options)
    check_training_inputs(train_batch["images"], train_batch["ray"],
                          train_batch["depth_map"], train_batch["target"])
    timer = StageTimer(torch.device(args.device), args.profile_warmup, args.profile)
    cpu_profile = cProfile.Profile()
    feature_cache = None
    cache_equal = None
    if args.frozen_cache:
        # Keep this tiny artificial-data cache separate from real-data caches.
        feature_cache = FrozenTensorCache(
            args.out_dir / "frozen_cache",
            "dino:" + module_fingerprint(dino_encoder.backbone),
            max_bytes=2_000_000_000 if args.profile else 32_000_000,
            ram_bytes=int(args.cache_ram_gb * 1e9),
        )
        images = train_batch["images"].flatten(0, 1)
        refiner.eval()
        reference = dino_encoder._forward_one_image_batch(images)
        dino_encoder.configure_cache(feature_cache)
        first = dino_encoder._forward_one_image_batch(images)
        repeated = dino_encoder._forward_one_image_batch(images)
        cache_equal = all(torch.equal(a, b) and torch.equal(a, c)
                          for a, b, c in zip(reference, first, repeated))
        if not cache_equal:
            raise AssertionError("Artificial DINO cache changed frozen outputs")
        for variant in train_variants:
            images = variant["images"].flatten(0, 1)
            uncached = dino_encoder._forward_one_image_batch_uncached(images)
            first = dino_encoder._forward_one_image_batch(images)
            repeated = dino_encoder._forward_one_image_batch(images)
            if not all(torch.equal(a, b) and torch.equal(a, c)
                       for a, b, c in zip(uncached, first, repeated)):
                raise AssertionError("Fixed DA cached DINO features differ from live features")
    results = {}
    flow_probe = None
    if args.profile_flow_cache:
        from mamba3_tracker.data.dataset import TrackingBatch
        from mamba3_tracker.data.frozen_cache import CachedFlow
        from train_depth_refined_tracker import _run_flow_batch

        class AnalyticFlow:
            device = torch.device(args.device)

            def flow(self, first, second):
                out = first.new_zeros((len(first), 2, *first.shape[-2:]))
                return out

        probe_cache = FrozenTensorCache(args.out_dir / "flow_cache", "analytic-profile-v1", 4e9)
        # This option becomes active when the bounded RAM cache is available.
        if args.cache_ram_gb:
            probe_cache.configure_ram(int(args.cache_ram_gb * 1e9))
        probe_flow = CachedFlow(AnalyticFlow(), probe_cache, 32, 64, 4)
        queries = torch.cat([train_batch["uv"][:, 0],
                             torch.zeros(args.batch_size, args.points, 1, device=args.device)], -1)
        tracking_batch = TrackingBatch(
            images=train_batch["images"], queries_xyt=queries,
            tracks_XYZ=train_batch["target"], visibility=train_batch["vis"].bool(),
            query_mask=torch.ones(args.batch_size, args.points, device=args.device, dtype=torch.bool),
            K=train_batch["K"], clip_ids=[str(i) for i in range(args.batch_size)],
            subsets=["synthetic"] * args.batch_size, depth=train_batch["depth_map"],
        )
        _run_flow_batch(probe_flow, tracking_batch, torch.device(args.device), args.image_size, .05, 1)
        for variant in train_variants:
            probe_flow.prefetch_windows(variant["images"] * 255.0)
            probe_flow.release_prefetch()
        flow_misses_after_prewarm = probe_cache.misses

        def flow_probe():
            tracking_batch.images = active_batch["images"]
            return _run_flow_batch(probe_flow, tracking_batch, torch.device(args.device),
                                   args.image_size, .05, 1)

    models = [("refiner", refiner)]
    if not args.skip_visibility:
        models.append(("visibility", head))
    for name, model in models:
        root = args.out_dir / name
        if (root / "latest.pt").exists() and not args.resume:
            raise ValueError(
                "Existing smoke run: pass --resume or choose another --out-dir"
            )
        cfg = {
            "optimizer": "amuse",
            "lr": 0.001,
            "warmup": 2,
            "amuse": {"muon_lr": 0.02, "beta1": 0.8, "rho": 0.3},
        }
        optimizer = build_optimizer(model, cfg)
        manager = CheckpointManager(root, k=3)
        writer = SummaryWriter(str(root / "tensorboard"))
        start = 0
        history = []
        if args.resume:
            st = restore_checkpoint(
                root / "latest.pt", model, optimizer, device=args.device
            )
            start, history = st["step"], st["history"]

        def loss(batch):
            if name == "visibility":
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    model(batch["flow_fwd"], batch["flow_bwd"]), batch["vis"]
                )
            pred = model(
                *(
                    batch[k]
                    for k in ("ray", "z_raw", "vis", "uv", "depth_map", "images", "K")
                )
            )
            return (pred.xyz - batch["target"]).abs().mean()

        with torch.no_grad(), evaluation_weights(optimizer):
            baseline = float(loss(val_batch))
        val_loss = baseline
        writer.add_scalar("val/loss", baseline, start)
        for step in range(start + 1, args.steps + 1):
            active_batch = train_variants[(step - 1) % len(train_variants)]
            if args.profile:
                cpu_profile.enable()
            model.train()
            optimizer.train()
            optimizer.zero_grad(set_to_none=True)
            if flow_probe is not None and name == "refiner":
                with timer.measure("flow_cache_tracking", step):
                    flow_probe()
            with timer.measure(name + "/forward", step):
                train_loss = loss(active_batch)
            if not torch.isfinite(train_loss):
                raise FloatingPointError("non-finite artificial-data loss")
            with timer.measure(name + "/backward", step):
                train_loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gn):
                raise FloatingPointError("non-finite artificial-data gradient")
            with timer.measure(name + "/optimizer", step):
                optimizer.step()
            with timer.measure(name + "/validation", step):
                with torch.no_grad(), evaluation_weights(optimizer):
                    model.eval()
                    val_loss = float(loss(val_batch))
            row = {
                "step": step,
                "train_loss": float(train_loss.detach()),
                "val_loss": val_loss,
            }
            history.append(row)
            writer.add_scalar("train/loss", row["train_loss"], step)
            writer.add_scalar("train/grad_norm", float(gn), step)
            writer.add_scalar("val/loss", val_loss, step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
            if feature_cache is not None and name == "refiner":
                writer.add_scalar("cache/dino_hits", feature_cache.hits, step)
                writer.add_scalar("cache/dino_misses", feature_cache.misses, step)
            with timer.measure(name + "/checkpoint", step):
                manager.save(
                    step, model, optimizer, history=history,
                    cfg={"train": cfg, "synthetic": True}, score=val_loss,
                )
            if args.profile:
                cpu_profile.disable()
            print(json.dumps({"model": name, **row}), flush=True)
        writer.close()
        results[name] = {
            "start_step": start,
            "completed_step": args.steps,
            "baseline_val_loss": baseline,
            "final_val_loss": val_loss,
            "checkpoints": len(list(root.glob("*.pt"))),
        }
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {"synthetic": True, "official_metric": None, "results": results,
             "cache": None if feature_cache is None else {
                 "equal": cache_equal, "hits": feature_cache.hits,
                 "misses": feature_cache.misses,
             }, "pretrained_mamba3": pretrained_report,
             "fixed_da_patterns": pattern_names,
             "flow_training_misses": probe_cache.misses - flow_misses_after_prewarm
             if args.profile_flow_cache else None}, indent=2
        )
    )
    if args.profile:
        cpu_profile.dump_stats(str(args.out_dir / "cpu_profile.pstats"))
        with (args.out_dir / "cpu_profile.txt").open("w") as out:
            pstats.Stats(cpu_profile, stream=out).sort_stats("cumulative").print_stats(40)
        (args.out_dir / "timing.json").write_text(json.dumps({
            "synthetic": True, "warmup_steps": args.profile_warmup,
            "dimensions": {k: getattr(args, k) for k in
                           ("batch_size", "frames", "points", "image_size", "dino_image_size")},
            "stages": timer.report(),
        }, indent=2))


if __name__ == "__main__":
    main()
