"""Strict, train-only teacher labels; no cache eviction or implicit recompute."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch
from torch import Tensor

REQUIRED_CONTEXT = frozenset(
    (
        "teacher_sha256",
        "split",
        "window_start",
        "window_length",
        "track_ids",
        "batch_order",
        "precision",
        "frontend_provenance",
        "query_anchors",
        "schema",
    )
)


def cache_key(context: dict[str, Any], inputs: dict[str, Tensor]) -> str:
    if not REQUIRED_CONTEXT <= context.keys():
        raise ValueError(f"Missing cache context: {REQUIRED_CONTEXT - context.keys()}")
    z_ref = inputs.get("z_ref")
    if z_ref is None or z_ref.dtype != torch.float32 or z_ref.ndim != 0:
        raise ValueError("Exact scalar FP32 z_ref is required")
    digest = hashlib.sha256(
        json.dumps(context, sort_keys=True, allow_nan=False).encode()
    )
    for name, tensor in sorted(inputs.items()):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite cache input: {name}")
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class DistillationCache:
    """Single-writer namespace. Publish data before the commit-marker manifest."""

    def __init__(
        self,
        root: Path,
        max_bytes: int = 20_000_000_000,
        reserve_bytes: int = 10_000_000_000,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes, self.reserve_bytes = max_bytes, reserve_bytes
        if max_bytes <= 0 or reserve_bytes < 0:
            raise ValueError("Invalid cache budget")

    @staticmethod
    def _check_split(context: dict[str, Any]) -> None:
        if context.get("split") != "train":
            raise ValueError("Teacher-label cache is train-only")

    def get(
        self, context: dict[str, Any], inputs: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        self._check_split(context)
        key = cache_key(context, inputs)
        manifest = json.loads((self.root / f"{key}.json").read_text())
        raw = (self.root / f"{key}.pt").read_bytes()
        if (
            manifest.get("key") != key
            or manifest.get("sha256") != hashlib.sha256(raw).hexdigest()
        ):
            raise ValueError("Teacher cache hash mismatch")
        result = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        if not isinstance(result, dict) or not all(
            isinstance(v, Tensor) and torch.isfinite(v).all() for v in result.values()
        ):
            raise ValueError("Invalid teacher cache payload")
        return result

    def put(
        self,
        context: dict[str, Any],
        inputs: dict[str, Tensor],
        labels: dict[str, Tensor],
    ) -> str:
        self._check_split(context)
        key = cache_key(context, inputs)
        if (self.root / f"{key}.json").exists():
            existing = self.get(context, inputs)
            if existing.keys() != labels.keys() or any(
                not torch.equal(existing[k], v.detach().cpu())
                for k, v in labels.items()
            ):
                raise ValueError("Existing teacher cache differs; refusing overwrite")
            return key
        if not all(torch.isfinite(v).all() for v in labels.values()):
            raise ValueError("Nonfinite teacher labels")
        stream = io.BytesIO()
        torch.save({k: v.detach().cpu() for k, v in labels.items()}, stream)
        raw = stream.getvalue()
        metadata = json.dumps(
            {"key": key, "sha256": hashlib.sha256(raw).hexdigest(), "context": context},
            sort_keys=True,
            allow_nan=False,
        ).encode()
        used = sum(p.stat().st_size for p in self.root.iterdir() if p.is_file())
        required = len(raw) + len(metadata)
        if (
            used + required > self.max_bytes
            or shutil.disk_usage(self.root).free - required < self.reserve_bytes
        ):
            raise ValueError("Teacher cache disk budget exceeded")
        for suffix, payload in (("pt", raw), ("json", metadata)):
            with tempfile.NamedTemporaryFile(
                dir=self.root, suffix=".partial", delete=False
            ) as output:
                temporary = Path(output.name)
                try:
                    output.write(payload)
                    output.flush()
                    temporary.replace(self.root / f"{key}.{suffix}")
                finally:
                    temporary.unlink(missing_ok=True)
        return key
