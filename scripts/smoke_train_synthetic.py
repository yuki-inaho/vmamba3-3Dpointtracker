"""Smoke the real refiner + visibility head with analytic artificial data.

uv run python scripts/smoke_train_synthetic.py --steps 6
uv run python scripts/smoke_train_synthetic.py --steps 8 --resume
Frozen WAFT/DA3 measurements are generated analytically; DINOv3 is real and frozen.
This checks training plumbing only and produces no official benchmark scores.
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download
import torch
from torch.utils.tensorboard import SummaryWriter

from mamba3_tracker.data.synthetic import synthetic_batch
from mamba3_tracker.data.frozen_cache import FrozenTensorCache, module_fingerprint
from mamba3_tracker.model.depth_refined_tracker import Mamba3V35Refiner
from mamba3_tracker.model.flow_vis_head import FlowVisHead
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
        dino_model=dino, dino_image_size=64, image_size=64
    ).to(args.device)
    pretrained_report = None
    if args.temporal_mixer == "official_mamba3":
        from mamba3_tracker.model.official_mamba3 import load_pretrained_mixers
        pretrained_report = load_pretrained_mixers(refiner.layers, args.mamba3_checkpoint)
        print(f"[smoke] loaded {pretrained_report['loaded_parameters']:,} official parameters", flush=True)
    head = FlowVisHead().to(args.device)
    train_batch = synthetic_batch(args.seed, device=args.device)
    val_batch = synthetic_batch(args.seed + 1, device=args.device)
    feature_cache = None
    cache_equal = None
    if args.frozen_cache:
        # Keep this tiny artificial-data cache separate from real-data caches.
        feature_cache = FrozenTensorCache(
            args.out_dir / "frozen_cache",
            "dino:" + module_fingerprint(refiner.dino.backbone),
            max_bytes=32_000_000,
        )
        images = train_batch["images"].flatten(0, 1)
        refiner.eval()
        reference = refiner.dino._forward_one_image_batch(images)
        refiner.dino.configure_cache(feature_cache)
        first = refiner.dino._forward_one_image_batch(images)
        repeated = refiner.dino._forward_one_image_batch(images)
        cache_equal = all(torch.equal(a, b) and torch.equal(a, c)
                          for a, b, c in zip(reference, first, repeated))
        if not cache_equal:
            raise AssertionError("Artificial DINO cache changed frozen outputs")
    results = {}
    for name, model in [("refiner", refiner), ("visibility", head)]:
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
            model.train()
            optimizer.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss = loss(train_batch)
            if not torch.isfinite(train_loss):
                raise FloatingPointError("non-finite artificial-data loss")
            train_loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gn):
                raise FloatingPointError("non-finite artificial-data gradient")
            optimizer.step()
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
            manager.save(
                step,
                model,
                optimizer,
                history=history,
                cfg={"train": cfg, "synthetic": True},
                score=val_loss,
            )
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
             }, "pretrained_mamba3": pretrained_report}, indent=2
        )
    )


if __name__ == "__main__":
    main()
