"""Prepare full-frame, full-query evaluation inputs, without running/scoring refiners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import torch

from mamba3_tracker.data.tapvid3d import load_clip
from mamba3_tracker.data.tapvid3d_splits import MINIVAL_FILES
from mamba3_tracker.deployment.checkpoint import sha256
from mamba3_tracker.deployment.runtime import INPUT_NAMES
from mamba3_tracker.model.dino_encoder import DINOv2Encoder
from mamba3_tracker.model.onnx_refiner import reference_depth

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def expected_paths(cfg: dict, partition: str) -> list[Path]:
    if partition == "monitor":
        values = json.loads(Path(cfg["data"]["split"]).read_text())["validation"]
        if len(values) != 15 or len(set(values)) != 15:
            raise ValueError("Monitor must contain all 15 distinct fixed clips")
        return [Path(p) for p in values]
    if partition != "minival":
        raise ValueError("Only monitor or canonical minival may be prepared")
    return [
        Path(cfg["data"]["minival_root"]) / subset / name
        for subset, names in MINIVAL_FILES.items()
        for name in names
    ]


class Capture:
    def __init__(self, encoder):
        self.encoder = encoder
        self.inputs = None

    @torch.no_grad()
    def __call__(self, ray, z_raw, visibility, uv, depth, images, intrinsics):
        features = self.encoder.forward_video(images)[0].float()
        values = (
            ray,
            z_raw,
            visibility,
            uv,
            depth,
            features,
            intrinsics,
            reference_depth(z_raw),
        )
        self.inputs = {
            name: value.detach().cpu().contiguous()
            for name, value in zip(INPUT_NAMES, values, strict=True)
        }
        # _infer's return is discarded; no refiner prediction or metric is made.
        return SimpleNamespace(
            xyz=torch.zeros((*ray.shape[:-1], 3), device=ray.device),
            vis_logits=visibility,
        )


def main():
    from scripts.eval_metric3d import _infer
    from scripts.prepare_refiner_kd import atomic_json, read_config, sequence_id
    from scripts.train_depth_refined_tracker import _build_waft_flow

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--partition", choices=["monitor", "minival"], required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        help="Preparation diagnostic only; partial manifest never scores",
    )
    parser.add_argument("--budget-gb", type=float, default=200)
    parser.add_argument("--reserve-gb", type=float, default=100)
    parser.add_argument(
        "--available-only",
        action="store_true",
        help="Prepare only ready files; missing files remain explicitly partial",
    )
    args = parser.parse_args()
    cfg = read_config(args.config)
    paths = expected_paths(cfg, args.partition)
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    weight = ROOT / "third_party/WAFT/ckpts/waft_a1_recommended.pth"
    expected_flow = "9f4b24f48b3937eca690a12b73bc3190effde6d4d4c87db01998fe63d846397f"
    if sha256(weight) != expected_flow:
        raise ValueError("WAFT checkpoint differs from fixed training frontend")
    identity: dict = dict(
        schema_version=1,
        partition=args.partition,
        config_sha256=sha256(args.config),
        split_sha256=cfg["data"]["split_sha256"],
        image_size=896,
        dino_model="facebook/dinov3-vits16-pretrain-lvd1689m",
        dino_revision=cfg["data"]["dino_revision"],
        dino_image_size=448,
        dino_precision="fp32",
        waft_sha256=expected_flow,
        flow_scale=-1,
        flow_iters=4,
        fb_alpha=0.05,
        fb_beta=1.0,
        visibility="flow",
        frame_scope="all",
        query_scope="all",
    )
    manifest_path = args.out_dir / "evaluation_inputs.json"
    prior = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if prior is not None and prior["identity"] != identity:
        raise ValueError("Evaluation input identity mismatch; use a new namespace")
    manifest: dict = dict(
        identity=identity,
        expected=[str(p) for p in paths],
        records=[],
        status="preparing",
        missing=[],
    )
    old = {r["raw_path"]: r for r in prior["records"]} if prior else {}
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = None
    flow = None
    for path in paths[: args.limit]:
        depth_path = args.depth_root / path.parent.name / path.name
        ready = path.is_file() and depth_path.is_file()
        if args.partition == "minival":
            ready = ready and depth_path.with_suffix(".npz.ready.json").is_file()
        if not ready and args.available_only:
            manifest["missing"].append(str(path))
            continue
        if not ready:
            raise FileNotFoundError(f"Missing full-frame input: {path} / {depth_path}")
        raw_sha, depth_sha = sha256(path), sha256(depth_path)
        if str(path) in old:
            row = old[str(path)]
            if (row["raw_sha256"], row["depth_sha256"]) != (
                raw_sha,
                depth_sha,
            ) or sha256(Path(row["path"])) != row["sha256"]:
                raise ValueError("Cached full-frame inputs changed")
            manifest["records"].append(row)
            atomic_json(manifest_path, manifest)
            continue
        used = sum(p.stat().st_size for p in args.out_dir.glob("*/*.pt"))
        if (
            used >= args.budget_gb * 1e9
            or shutil.disk_usage(args.out_dir).free < args.reserve_gb * 1e9
        ):
            raise RuntimeError("Full-frame input disk budget/reserve reached")
        if encoder is None:
            encoder = (
                DINOv2Encoder(
                    identity["dino_model"],
                    image_size=448,
                    revision=identity["dino_revision"],
                )
                .cuda()
                .eval()
            )
            encoder.ENC_CHUNK = 8
            flow = _build_waft_flow("cuda", scale=-1, iters=4)
        started = time.monotonic()
        clip = load_clip(path)
        capture = Capture(encoder)
        _infer("v35", flow, capture, clip, 896, 0.05, 1.0, args.depth_root, 0, "cuda")
        values = capture.inputs
        if values is None or values["ray"].shape[1:3] != clip.tracks_XYZ.shape[:2]:
            raise ValueError("Full frame/query coverage mismatch")
        if not all(torch.isfinite(v).all() for v in values.values()):
            raise ValueError("Nonfinite prepared evaluation inputs")
        destination = args.out_dir / path.parent.name / (path.stem + ".pt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(
                f"Uncommitted input artifact requires audit: {destination}"
            )
        payload = dict(
            inputs=values,
            gt=clip.tracks_XYZ,
            visible=clip.visibility,
            K=clip.K,
            image_hw=list(clip.images.shape[-2:]),
            queries=clip.queries_xyt,
        )
        estimated = sum(v.numel() * v.element_size() for v in values.values())
        if (
            used + estimated > args.budget_gb * 1e9
            or shutil.disk_usage(args.out_dir).free - estimated < args.reserve_gb * 1e9
        ):
            raise RuntimeError("Next evaluation artifact exceeds disk budget/reserve")
        temporary = destination.with_suffix(".partial")
        torch.save(payload, temporary)
        temporary.replace(destination)
        row = dict(
            path=str(destination.resolve()),
            sha256=sha256(destination),
            bytes=destination.stat().st_size,
            raw_path=str(path),
            raw_sha256=raw_sha,
            depth_path=str(depth_path),
            depth_sha256=depth_sha,
            subset=path.parent.name,
            filename=path.name,
            sequence=sequence_id(path),
            frames=int(clip.F),
            tracks=int(clip.tracks_XYZ.shape[1]),
            seconds=time.monotonic() - started,
        )
        manifest["records"].append(row)
        atomic_json(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "prepared": len(manifest["records"]),
                    "expected": len(paths),
                    "clip": path.name,
                    "seconds": row["seconds"],
                }
            ),
            flush=True,
        )
        del clip, capture, values, payload
        torch.cuda.empty_cache()
    manifest["status"] = (
        "complete" if len(manifest["records"]) == len(paths) else "partial"
    )
    atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
