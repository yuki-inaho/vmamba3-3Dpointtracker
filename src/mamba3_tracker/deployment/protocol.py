"""Fixed best200 paired protocol and CPU-safe preflight; no model downloads."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from mamba3_tracker.deployment.checkpoint import sha256

MANIFEST_SHA256 = "bf3552d14d04d6a19492fe91d0401903d6f7457902c9d92c79c788626b9af4db"
EXPECTED_FRAMES = 1485
DINO_REVISION = "114c1379950215c8b35dfcd4e90a5c251dde0d32"


def fixed_manifest(path: Path) -> dict[str, list[str]]:
    if sha256(path) != MANIFEST_SHA256:
        raise ValueError("The paired acceptance gate requires the unchanged fixed-nine manifest")
    manifest = json.loads(path.read_text())
    files = manifest["files_by_subset"]
    if set(files) != {"adt", "drivetrack", "pstudio"} or any(len(v) != 3 for v in files.values()):
        raise ValueError("Expected exactly three clips in each fixed subset")
    return files


def resolve_dino_revision(state: dict[str, Any], explicit: str | None = None) -> str:
    stored = state.get("export", {}).get("dino_revision")
    if stored and explicit and stored != explicit:
        raise ValueError("DINO revision differs from checkpoint provenance")
    revision = explicit or stored
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("An immutable 40-digit DINO revision is required; never use main")
    if revision != DINO_REVISION:
        raise ValueError("The fixed best200 protocol requires its recorded DINO revision")
    return revision


def preflight(files: dict[str, list[str]], data_root: Path, depth_root: Path,
              repo_root: Path, cuda_available: bool) -> dict[str, Any]:
    """Absence or mismatched frame coverage is a blocker, never a task pass."""
    blockers: list[str] = []
    if not cuda_available:
        blockers.append("CUDA unavailable: the unchanged native Triton/BF16 path requires CUDA")
    for relative in ("third_party/visionMamba3/src/visionmamba3/__init__.py",
                     "third_party/visionMamba3/third_party/mamba-ssm/mamba_ssm/modules/mamba3.py",
                     "third_party/WAFT/model.py", "third_party/WAFT/ckpts/waft_a1_recommended.pth"):
        if not (repo_root / relative).is_file():
            blockers.append(f"Missing native dependency: {relative}")
    entries, total_frames = [], 0
    for subset, names in files.items():
        for filename in names:
            relative = f"{subset}/{filename}"
            data, depth = data_root / relative, depth_root / relative
            row: dict[str, Any] = {"clip": relative, "data_present": data.is_file(), "depth_present": depth.is_file()}
            for label, path in (("data", data), ("depth", depth)):
                if not path.is_file():
                    blockers.append(f"Missing {label}: {relative}")
            if data.is_file() and depth.is_file():
                try:
                    with np.load(data, allow_pickle=False) as sample, np.load(depth, allow_pickle=False) as maps:
                        if "images_jpeg_bytes" not in sample.files:
                            raise ValueError("Clip has no RGB frames")
                        xyz, visible = sample["tracks_XYZ"], sample["visibility"]
                        if xyz.ndim != 3 or xyz.shape[-1] != 3 or visible.shape != xyz.shape[:2]:
                            raise ValueError("Ground-truth shape mismatch")
                        values = maps["depth_q"] if "depth_q" in maps else maps["depth"]
                        if values.ndim != 3 or values.shape[0] != xyz.shape[0] or not np.isfinite(values).all():
                            raise ValueError("Depth must be finite and cover every frame")
                        row.update(frames=int(xyz.shape[0]), tracks=int(xyz.shape[1]),
                                   data_sha256=sha256(data), depth_sha256=sha256(depth))
                        total_frames += int(xyz.shape[0])
                except (KeyError, ValueError, OSError) as error:
                    blockers.append(f"Invalid input {relative}: {type(error).__name__}")
            entries.append(row)
    if total_frames != EXPECTED_FRAMES:
        blockers.append(f"Full input frame coverage is {total_frames}, expected {EXPECTED_FRAMES}")
    return {"ready": not blockers, "blockers": blockers, "inputs": entries,
            "available_valid_frames": total_frames, "expected_frames": EXPECTED_FRAMES,
            "task_accuracy_verified": False}
