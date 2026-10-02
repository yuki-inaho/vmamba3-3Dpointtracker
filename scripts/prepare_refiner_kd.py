"""Audit KD data or prepare exact external inputs from read-only frozen caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from mamba3_tracker.data.dataset import TAPVid3DDataset, collate_tracking
from mamba3_tracker.data.frozen_cache import CachedFlow, FrozenTensorCache
from mamba3_tracker.data.tapvid3d_splits import MINIVAL_FILES
from mamba3_tracker.deployment.checkpoint import sha256
from mamba3_tracker.deployment.runtime import INPUT_NAMES
from mamba3_tracker.deployment.student_checkpoint import (
    load_native_teacher,
    tensor_digest,
)
from mamba3_tracker.model.onnx_refiner import reference_depth
from mamba3_tracker.train.loss import _per_clip_anchor_depth_scale
from searaft_flow import track_clip


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--mode", choices=["audit", "pilot", "cache"], required=True)
    p.add_argument("--partition", choices=["train", "validation"], default="train")
    p.add_argument("--limit", type=int)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--window", type=int, choices=[8, 16, 32], default=8)
    p.add_argument("--compute-frontend", action="store_true")
    p.add_argument("--reanchor-window", action="store_true")
    p.add_argument("--frontend-precision", choices=["fp32", "bf16"])
    p.add_argument("--new-input-root", type=Path)
    p.add_argument("--new-input-budget-gb", type=float, default=150)
    return p


def validate_preparation_mode(args) -> None:
    reanchor = getattr(args, "reanchor_window", False)
    if args.window == 8 and not reanchor:
        if args.compute_frontend or args.new_input_root is not None:
            raise ValueError("Existing 8-frame preparation must remain read-only")
        return
    if not args.compute_frontend or args.new_input_root is None or args.mode != "cache":
        raise ValueError(
            "New windows/reanchoring require explicit frontend computation and a new input root"
        )
    root = args.new_input_root.resolve()
    if (
        root.is_relative_to(Path("/workspace/vmamba3_data"))
        or "tapvid3d_frozen_cache" in root.parts
    ):
        raise ValueError("Must not write into existing training caches")


def read_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    base_keys = {
        "schema_version",
        "architecture",
        "teacher",
        "data",
        "optimizer",
        "training",
        "ablations",
        "protocol",
        "protocol_sha256",
    }
    schema = cfg.get("schema_version")
    if schema == 2:
        if set(cfg) != base_keys | {"improvement"}:
            raise ValueError("Unknown or missing improvement config keys")
        if cfg.get("architecture") != "vssd_local_global_128x2_v2_train":
            raise ValueError("Unsupported improvement architecture")
        improvement = cfg["improvement"]
        expected = {
            "query_policy",
            "frontend_precision",
            "block_steps",
            "block_lr",
            "kd_gate",
            "gradient_target_ratio",
            "gradient_max_scale",
            "gradient_direction_gate",
        }
        if not isinstance(improvement, dict) or set(improvement) != expected:
            raise ValueError("Unknown or missing improvement settings")
        if (
            improvement["query_policy"] != "strict_visible_gt_reanchor_v1"
            or improvement["frontend_precision"] != "fp32"
            or improvement["kd_gate"] != "teacher_better"
            or type(improvement["gradient_direction_gate"]) is not bool
            or type(improvement["block_steps"]) is not int
            or improvement["block_steps"] < 1
        ):
            raise ValueError("Invalid improvement policy/precision/curriculum")
        for key in ("block_lr", "gradient_target_ratio", "gradient_max_scale"):
            value = improvement[key]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Invalid improvement setting: {key}")
        if (
            improvement["gradient_target_ratio"] > 1
            or improvement["gradient_max_scale"] < 1
            or cfg["data"]["window"] not in (8, 16, 32)
            or cfg["data"]["fixed_window_seed"] != 42
            or cfg["data"]["image_size"] != 896
            or cfg["data"]["dino_image_size"] != 448
        ):
            raise ValueError("Invalid improvement data or gradient settings")
    elif set(cfg) not in (base_keys, base_keys | {"selected_ablation", "selection"}):
        raise ValueError("Unknown or missing KD config keys")
    if "selection" in cfg:
        selection = cfg["selection"]
        if set(selection) != {
            "base_config",
            "base_sha256",
            "pilot_report",
            "pilot_sha256",
        }:
            raise ValueError("Unknown or missing selection provenance")
        base_path = Path(selection["base_config"])
        if sha256(base_path) != selection["base_sha256"]:
            raise ValueError("Selected base config hash mismatch")
        base = yaml.safe_load(base_path.read_text())
        if set(base) != base_keys or base != {key: cfg[key] for key in base_keys}:
            raise ValueError("Selection may not change input or training configuration")
        report_path = Path(selection["pilot_report"])
        if sha256(report_path) != selection["pilot_sha256"]:
            raise ValueError("Pilot report hash mismatch")
        report = json.loads(report_path.read_text())
        if report.get("status") != "complete":
            raise ValueError("Pilot selection requires completed report")
        candidates = [
            r for r in report["completed"] if r["arm"] in ("A1", "A2", "A3", "A4")
        ]
        if {r["arm"] for r in candidates} != {"A1", "A2", "A3", "A4"}:
            raise ValueError("Pilot selection requires every KD arm")
        best = min(candidates, key=lambda r: (r["best_monitor_gt_loss"], r["arm"]))
        if cfg["selected_ablation"] != best["arm"]:
            raise ValueError("Selected ablation differs from monitor-only selection")
    if schema != 2 and (schema != 1 or cfg["architecture"] != "vssd_two_pool_128x2_v1"):
        raise ValueError("Unsupported KD config")
    if (
        cfg["data"]["photometric_augment"] is not False
        or cfg["data"]["label_mode"] != "online"
    ):
        raise ValueError("Requires DA-off and explicit online teacher")
    for path_value, expected in [
        (cfg["teacher"]["checkpoint"], cfg["teacher"]["sha256"]),
        (cfg["data"]["split"], cfg["data"]["split_sha256"]),
        (cfg["protocol"], cfg["protocol_sha256"]),
    ]:
        if sha256(Path(path_value)) != expected:
            raise ValueError(f"Configuration provenance hash mismatch: {path_value}")
    return cfg


def validate_prepared_reuse(payload: dict, report: dict, path: Path) -> None:
    if (
        payload.get("config_sha256") != report["config_sha256"]
        or payload.get("clip") != str(path)
        or payload.get("requested_window", 8) != report["window"]
    ):
        raise ValueError("Prepared input provenance differs; refusing reuse")
    # Historical artifacts omitted these fields; only their exact legacy semantics
    # may inherit the defaults. Strict/new-precision artifacts cannot reuse them.
    for key, default in (
        ("query_policy", "legacy_anchor_filter_v1"),
        ("frontend_precision", "bf16"),
    ):
        if payload.get(key, default) != report[key]:
            raise ValueError(f"Prepared input {key} differs; refusing reuse")
    if (
        report["query_policy"] != "legacy_anchor_filter_v1"
        or report["frontend_precision"] != "bf16"
    ):
        if payload.get("preprocessing_sha256") != report["preprocessing_sha256"]:
            raise ValueError(
                "Prepared preprocessing provenance differs; refusing reuse"
            )


def audit_reanchored_queries(
    queries, xyz, visible, intrinsics, image_size: int
) -> dict:
    """Prove that every emitted query names a visible, finite, in-frame GT point."""
    if (
        queries.ndim != 3
        or xyz.ndim != 4
        or queries.shape[-1] != 3
        or xyz.shape[-1] != 3
        or queries.shape[:2] != (xyz.shape[0], xyz.shape[2])
        or xyz.shape[1] < 1
        or xyz.shape[2] < 1
        or visible.shape != xyz.shape[:-1]
        or intrinsics.shape != (xyz.shape[0], 3, 3)
    ):
        raise ValueError("Invalid strict query/GT shapes")
    if not bool(torch.isfinite(intrinsics).all()) or not bool(
        ((intrinsics[:, 0, 0] > 0) & (intrinsics[:, 1, 1] > 0)).all()
    ):
        raise ValueError("Strict anchor audit requires finite positive intrinsics")
    times = queries[..., 2]
    if not bool(torch.isfinite(queries).all()) or not bool(
        ((times >= 0) & (times < xyz.shape[1]) & (times == times.floor())).all()
    ):
        raise ValueError("Invalid strict query anchor times")
    anchors = times.long()
    anchor_xyz = xyz.gather(1, anchors[:, None, :, None].expand(-1, 1, -1, 3)).squeeze(
        1
    )
    anchor_visible = visible.gather(1, anchors[:, None, :]).squeeze(1)
    if not bool(
        (
            anchor_visible
            & torch.isfinite(anchor_xyz).all(-1)
            & (anchor_xyz[..., 2] > 1e-6)
        ).all()
    ):
        raise ValueError("Strict query lacks a valid visible GT anchor")
    z = anchor_xyz[..., 2]
    fx, fy, cx, cy = [
        intrinsics[:, i, j, None] for i, j in ((0, 0), (1, 1), (0, 2), (1, 2))
    ]
    projected = torch.stack(
        (fx * anchor_xyz[..., 0] / z + cx, fy * anchor_xyz[..., 1] / z + cy), -1
    )
    xy = queries[..., :2]
    if not bool(((xy >= 0) & (xy < image_size)).all()):
        raise ValueError("Strict query outside resized image bounds")
    error = (projected - xy).norm(dim=-1)
    if not bool(torch.isfinite(error).all()) or bool((error > 1e-3).any()):
        raise ValueError("Strict query projection differs from same-frame GT")
    return {
        "anchor_valid_count": anchors.numel(),
        "anchor_reprojection_max_px": float(error.max()),
    }


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sequence_id(path: Path) -> str:
    subset = path.parent.name
    name = (
        path.stem.split("_")[1]
        if subset == "drivetrack"
        else path.stem.rsplit("_", 1)[0]
    )
    return subset + "/" + name


def audit(cfg: dict) -> dict:
    split = json.loads(Path(cfg["data"]["split"]).read_text())
    paths = {key: [Path(p) for p in split[key]] for key in ("train", "validation")}
    paths["minival"] = [
        Path(cfg["data"]["minival_root"]) / subset / n
        for subset, names in MINIVAL_FILES.items()
        for n in names
    ]
    report = {
        "split_sha256": cfg["data"]["split_sha256"],
        "blind_test": False,
        "scope": "Official minival with known upstream development exposure and conservative sequence overlaps",
        "splits": {},
        "overlaps": {},
    }
    for name, values in paths.items():
        report["splits"][name] = {
            "count": len(values),
            "missing": [str(p) for p in values if not p.is_file()],
            "sequences": sorted({sequence_id(p) for p in values}),
        }
    for left, right in [
        ("train", "validation"),
        ("train", "minival"),
        ("validation", "minival"),
    ]:
        lclip = {p.parent.name + "/" + p.name for p in paths[left]}
        rclip = {p.parent.name + "/" + p.name for p in paths[right]}
        report["overlaps"][left + ":" + right] = {
            "clips": sorted(lclip & rclip),
            "sequences": sorted(
                {sequence_id(p) for p in paths[left]}
                & {sequence_id(p) for p in paths[right]}
            ),
        }
        if lclip & rclip:
            raise ValueError("Clip leakage between splits")
    return report


class ReadOnlyCache:
    """No mtime changes, allocation, eviction, or implicit recomputation."""

    key = FrozenTensorCache.key

    def __init__(self, root: Path) -> None:
        self.root, self.hits, self.misses = root, 0, 0
        if not root.is_dir():
            raise FileNotFoundError(root)

    def get(self, key: str, device: str | torch.device):
        path = self.root / (key + ".pt")
        if not path.is_file():
            self.misses += 1
            raise FileNotFoundError(f"Strict frozen-cache miss: {path}")
        values = torch.load(path, weights_only=True, map_location="cpu")
        if not isinstance(values, tuple) or not all(
            isinstance(v, torch.Tensor) and torch.isfinite(v).all() for v in values
        ):
            raise ValueError(f"Invalid frozen tensors: {path}")
        self.hits += 1
        return tuple(value.to(device) for value in values)

    def put_many(self, items):
        raise RuntimeError("Writes to pre-existing frozen caches are forbidden")


class NoComputeFlow:
    device = torch.device("cuda")

    def flow(self, first, second):
        raise FileNotFoundError(
            "Missing flow cache; prepare a new explicit namespace before retrying"
        )


def prepare(cfg: dict, args: argparse.Namespace) -> None:
    validate_preparation_mode(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required to reproduce cached preprocessing keys")
    torch.set_num_threads(8)
    data = cfg["data"]
    reanchor = getattr(args, "reanchor_window", False)
    improvement = cfg.get("improvement", {})
    if improvement and (not reanchor or args.window != data["window"]):
        raise ValueError(
            "Improvement preparation requires explicit reanchoring and matching config window"
        )
    precision = getattr(args, "frontend_precision", None) or improvement.get(
        "frontend_precision", "fp32" if reanchor else "bf16"
    )
    if improvement and precision != improvement["frontend_precision"]:
        raise ValueError("Frontend precision differs from improvement configuration")
    if not args.compute_frontend and (data["window"] != 8 or precision != "bf16"):
        raise ValueError(
            "Read-only cache preparation currently requires the prewarmed 8-frame windows"
        )
    source_cfg = json.loads(Path(cfg["teacher"]["run_config"]).read_text())
    provenance_path = Path("/workspace/vmamba3_data/manifests/cache_provenance.json")
    provenance = json.loads(provenance_path.read_text())
    if provenance["dinov3"]["revision"] != data["dino_revision"]:
        raise ValueError("DINO provenance differs")
    flow_cache = dino_cache = encoder = None
    if args.compute_frontend:
        import sys

        root_path = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root_path))
        sys.path.insert(0, str(root_path / "scripts"))
        from scripts.train_depth_refined_tracker import _build_waft_flow
        from mamba3_tracker.model.dino_encoder import DINOv2Encoder

        weight = Path("third_party/WAFT/ckpts/waft_a1_recommended.pth")
        if (
            sha256(weight)
            != "9f4b24f48b3937eca690a12b73bc3190effde6d4d4c87db01998fe63d846397f"
        ):
            raise ValueError("WAFT provenance mismatch")
        flow = _build_waft_flow("cuda", scale=-1, iters=4)
        encoder = (
            DINOv2Encoder(
                "facebook/dinov3-vits16-pretrain-lvd1689m",
                image_size=448,
                revision=data["dino_revision"],
            )
            .cuda()
            .eval()
        )
        encoder.ENC_CHUNK = 8
    else:
        cache_root = Path(source_cfg["frozen_cache"]["root"])
        flow_cache = ReadOnlyCache(
            cache_root / "flow" / provenance["waft_flow"]["cache_namespace_sha256"][:16]
        )
        dino_cache = ReadOnlyCache(
            cache_root / "dino" / provenance["dinov3"]["cache_namespace_sha256"][:16]
        )
        flow = CachedFlow(NoComputeFlow(), flow_cache)
    paths = [
        Path(p) for p in json.loads(Path(data["split"]).read_text())[args.partition]
    ]
    if args.mode == "pilot" and args.partition != "train":
        raise ValueError("Teacher pilot is train-only")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        paths = paths[: args.limit]
    dataset = TAPVid3DDataset(
        paths,
        window_size=args.window,
        seed=42,
        max_queries=data["num_tracks"],
        augment=False,
        image_size=896,
        da3_depth_root=None,
        reanchor_window=reanchor,
        strict_reanchor_window=reanchor,
        fixed_window_seed=42,
    )
    budget_root = (
        args.new_input_root if args.compute_frontend else Path(data["input_cache_root"])
    )
    root = budget_root / f"window{args.window}" / args.partition
    root.mkdir(parents=True, exist_ok=True)
    used = sum(p.stat().st_size for p in budget_root.rglob("*.pt"))
    teacher = (
        load_native_teacher(
            Path(cfg["teacher"]["checkpoint"]), cfg["teacher"]["sha256"]
        ).cuda()
        if args.mode == "pilot"
        else None
    )
    before_teacher = (
        tensor_digest(teacher.state_dict()) if teacher is not None else None
    )
    report = {
        "status": "running",
        "mode": args.mode,
        "partition": args.partition,
        "records": [],
        "config_sha256": sha256(args.config),
        "frontend_provenance_sha256": sha256(provenance_path),
        "split_sha256": data["split_sha256"],
        "teacher_sha256": cfg["teacher"]["sha256"],
        "label_mode": "online",
        "window": args.window,
        "query_policy": "strict_visible_gt_reanchor_v1"
        if reanchor
        else "legacy_anchor_filter_v1",
        "query_seed": 42,
        "window_seed": 42,
        "frontend_precision": precision,
        "frontend_mode": f"explicit_compute_{precision}_dino"
        if args.compute_frontend
        else "read_only_existing_cache",
    }
    preprocessing = {
        key: report[key]
        for key in (
            "window",
            "query_policy",
            "query_seed",
            "window_seed",
            "frontend_precision",
            "frontend_provenance_sha256",
        )
    }
    preprocessing.update(
        num_tracks=data["num_tracks"],
        image_size=896,
        dino_image_size=448,
        dino_revision=data["dino_revision"],
    )
    report["preprocessing_sha256"] = hashlib.sha256(
        json.dumps(preprocessing, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any]
    for index, path in enumerate(paths):
        start_time = time.monotonic()
        destination = root / (path.parent.name + "__" + path.stem + ".pt")
        if destination.exists():
            payload = torch.load(destination, weights_only=True, map_location="cpu")
            validate_prepared_reuse(payload, report, path)
        else:
            # Clip-local query RNG makes interrupted preparations independent of order.
            dataset._rng = random.Random(
                hashlib.sha256(
                    f"42:{path.parent.name}/{path.name}".encode()
                ).hexdigest()
            )
            item = dataset[index]
            if item["clip_id"] != path.stem:
                raise ValueError("Dataset silently substituted a different clip")
            depth_path = (
                Path(source_cfg["data"]["da3_depth_root"])
                / path.parent.name
                / path.name
            )
            begin, end = item["frame_start"], item["frame_start"] + len(item["images"])
            with np.load(depth_path) as depth_file:
                if "depth_q" in depth_file:
                    minimum, maximum = (
                        float(depth_file["d_min"]),
                        float(depth_file["d_max"]),
                    )
                    depth = minimum + depth_file["depth_q"][begin:end].astype(
                        np.float32
                    ) * (max(maximum - minimum, 1e-6) / 65535.0)
                else:
                    depth = depth_file["depth"][begin:end].astype(np.float32)
            item["depth"] = torch.from_numpy(depth)
            batch = collate_tracking([item])
            anchor_audit = (
                audit_reanchored_queries(
                    batch.queries_xyt, batch.tracks_XYZ, batch.visibility, batch.K, 896
                )
                if reanchor
                else None
            )
            images = batch.images.cuda()
            images255 = images[0] * 255.0
            with torch.no_grad():
                uv, vis = track_clip(
                    flow,
                    images255,
                    batch.queries_xyt[0, :, :2].cuda(),
                    batch.queries_xyt[0, :, 2].long().cuda(),
                    896,
                    0.05,
                    1.0,
                )
                if isinstance(flow, CachedFlow):
                    flow.release_prefetch()
                uv, vis = uv[None].cuda(), vis[None].cuda()
                k = batch.K.cuda()
                ray = torch.stack(
                    (
                        (uv[..., 0] - k[:, None, None, 0, 2]) / k[:, None, None, 0, 0],
                        (uv[..., 1] - k[:, None, None, 1, 2]) / k[:, None, None, 1, 1],
                    ),
                    -1,
                )
                if batch.depth is None:
                    raise ValueError("Metric depth input is required")
                depth_tensor = batch.depth.cuda()
                frames, tracks = uv.shape[1:3]
                z = F.grid_sample(
                    depth_tensor.reshape(frames, 1, *depth_tensor.shape[-2:]),
                    (2 * uv / 896 - 1).reshape(frames, 1, tracks, 2),
                    padding_mode="border",
                    align_corners=False,
                ).reshape(1, frames, tracks)
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=precision == "bf16"
                ):
                    resized = F.interpolate(
                        images.flatten(0, 1),
                        size=(448, 448),
                        mode="bilinear",
                        align_corners=False,
                    )
                    if encoder is not None:
                        features = encoder.forward_video(images)[0].cpu()
                    else:
                        if dino_cache is None:
                            raise RuntimeError("Missing explicit DINO backend")
                        features = torch.stack(
                            [
                                dino_cache.get(dino_cache.key(frame), "cpu")[0]
                                for frame in resized
                            ]
                        ).unsqueeze(0)
                anchors = batch.queries_xyt[..., 2].long().clamp(0, frames - 1)
                initial = batch.tracks_XYZ.gather(
                    1, anchors[:, None, :, None].expand(1, 1, tracks, 3)
                ).squeeze(1)
                payload = {
                    "inputs": dict(
                        zip(
                            INPUT_NAMES[:-1],
                            (
                                ray.cpu(),
                                z.cpu(),
                                vis.cpu(),
                                uv.cpu(),
                                depth_tensor.cpu(),
                                features,
                                k.cpu(),
                            ),
                            strict=True,
                        )
                    ),
                    "target": {
                        "xyz": batch.tracks_XYZ,
                        "visible": batch.visibility,
                        "valid": (
                            batch.query_mask[:, None].expand(1, frames, tracks)
                            & torch.isfinite(batch.tracks_XYZ).all(-1)
                            & (batch.tracks_XYZ[..., 2] > 1e-6)
                        )
                        if reanchor
                        else batch.query_mask[:, None].expand(1, frames, tracks),
                        "scale": _per_clip_anchor_depth_scale(
                            initial, batch.query_mask
                        ),
                    },
                    "clip": str(path),
                    "sequence": sequence_id(path),
                    "raw_sha256": sha256(path),
                    "depth_sha256": sha256(depth_path),
                    "config_sha256": report["config_sha256"],
                    "frontend_provenance_sha256": report["frontend_provenance_sha256"],
                    "window_start": begin,
                    "window_fallback": bool(item.get("window_fallback", False)),
                    "requested_window": args.window,
                    "query_policy": report["query_policy"],
                    "query_seed": report["query_seed"],
                    "window_seed": report["window_seed"],
                    "frontend_precision": precision,
                    "preprocessing_sha256": report["preprocessing_sha256"],
                    "queries_xyt": batch.queries_xyt,
                    "anchor_audit": anchor_audit,
                    "query_idx": batch.query_idx[0],
                    "anchors": anchors,
                }
            required = (
                sum(v.numel() * v.element_size() for v in payload["inputs"].values())
                + 1_000_000
            )
            if used + required > (
                args.new_input_budget_gb * 1e9
                if args.compute_frontend
                else data["input_cache_max_bytes"]
            ) or shutil.disk_usage(root).free - required < (
                100_000_000_000 if args.compute_frontend else 30_000_000_000
            ):
                raise ValueError("Prepared input cache budget exceeded")
            temporary = destination.with_suffix(".partial")
            torch.save(payload, temporary)
            temporary.replace(destination)
            used += destination.stat().st_size
        record = {
            "path": str(destination),
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
            "clip": str(path),
            "frames": payload["inputs"]["ray"].shape[1],
            "tracks": payload["inputs"]["ray"].shape[2],
            "depth_hw": list(payload["inputs"]["depth_map"].shape[-2:]),
            "window_start": int(payload.get("window_start", 0)),
            "window_fallback": bool(payload.get("window_fallback", False)),
        }
        if reanchor:
            anchor_audit = audit_reanchored_queries(
                payload["queries_xyt"],
                payload["target"]["xyz"],
                payload["target"]["visible"],
                payload["inputs"]["intrinsics"],
                896,
            )
            if payload.get("anchor_audit") != anchor_audit:
                raise ValueError("Prepared anchor audit differs; refusing reuse")
            record.update(anchor_audit)
            record["valid_gt_count"] = int(payload["target"]["valid"].sum())
        if teacher is not None:
            values = [
                payload["inputs"][name].cuda().float() for name in INPUT_NAMES[:-1]
            ]
            values.append(reference_depth(values[1]))
            with torch.no_grad():
                prediction = teacher.forward_train(*values)
                repeated = teacher.forward_train(*values)
            for name in ("xyz", "duv", "dlog"):
                torch.testing.assert_close(
                    prediction[name], repeated[name], rtol=0, atol=0
                )
                if not torch.isfinite(prediction[name]).all():
                    raise ValueError("Nonfinite teacher pilot")
            record["teacher_exact_repeat"] = True
            record["teacher_label_tensor_bytes"] = sum(
                v.numel() * v.element_size()
                for k, v in prediction.items()
                if isinstance(v, torch.Tensor)
            )
        record["seconds"] = time.monotonic() - start_time
        report["records"].append(record)
        atomic_json(args.out_dir / "input_manifest.json", report)
        print(
            json.dumps(
                {
                    "prepared": index + 1,
                    "total": len(paths),
                    "clip": path.name,
                    "seconds": record["seconds"],
                }
            ),
            flush=True,
        )
    if teacher is not None and tensor_digest(teacher.state_dict()) != before_teacher:
        raise ValueError("Frozen teacher mutated during pilot")
    report.update(
        status="complete",
        flow_cache_hits=flow_cache.hits if flow_cache is not None else 0,
        dino_cache_hits=dino_cache.hits if dino_cache is not None else 0,
        teacher_unchanged=teacher is not None,
        teacher_tensor_sha256=before_teacher,
    )
    atomic_json(args.out_dir / "input_manifest.json", report)


def main() -> None:
    args = parser().parse_args()
    cfg = read_config(args.config)
    if args.mode == "audit":
        atomic_json(args.out_dir / "split_audit.json", audit(cfg))
    else:
        prepare(cfg, args)


if __name__ == "__main__":
    main()
