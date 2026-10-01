"""Fail-closed loading and provenance for full and public refiner checkpoints."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

BACKBONE_PREFIX = "dino.backbone."
PUBLIC_BEST200_SHA256 = "bcddf045311b76a91bc772e35901acbdb99f397b752ee0ba4beacb3fb1ded5f4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_checkpoint(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    """Never opt into arbitrary pickle execution to load a checkpoint."""
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError("Checkpoint SHA256 does not match the expected source")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not isinstance(state.get("cfg"), dict):
        raise TypeError("Expected a checkpoint with cfg, model and step")
    if not isinstance(state.get("step"), int) or not isinstance(state.get("model"), dict):
        raise TypeError("Checkpoint model/step schema is invalid")
    if not isinstance(state["cfg"].get("model"), dict) or not state["model"]:
        raise ValueError("Checkpoint model config/weights must not be empty")
    for name, value in state["model"].items():
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise TypeError("Every model entry must be a named tensor")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Checkpoint contains nonfinite values: {name}")
    return state


def _validate_supplied(target: dict[str, Tensor], supplied: dict[str, Tensor]) -> None:
    unexpected = sorted(set(supplied) - set(target))
    if unexpected:
        raise ValueError(f"Unexpected checkpoint keys: {unexpected}")
    for name, value in supplied.items():
        expected = target[name]
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(f"Shape/dtype mismatch for {name}: {value.shape}/{value.dtype}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite checkpoint tensor: {name}")


def load_native_state(model: nn.Module, state: dict[str, Any]) -> dict[str, Any]:
    """Only an explicitly exported, frozen DINO backbone may be absent.

    Validation happens before load_state_dict, so invalid input does not partly
    overwrite a live module. Full checkpoints retain strict loading semantics.
    """
    target = model.state_dict()
    supplied = state["model"]
    _validate_supplied(target, supplied)
    missing = sorted(set(target) - set(supplied))
    exported = state.get("export", {}).get("excluded_prefixes") == [BACKBONE_PREFIX]
    if missing and (not exported or any(not k.startswith(BACKBONE_PREFIX) for k in missing)):
        raise ValueError(f"Unapproved missing checkpoint keys: {missing}")
    if missing:
        backbone_keys = {key for key in target if key.startswith(BACKBONE_PREFIX)}
        if set(missing) != backbone_keys:
            raise ValueError("A public checkpoint must omit the entire frozen backbone, not a partial one")
        parameters = [(k, p) for k, p in model.named_parameters() if k.startswith(BACKBONE_PREFIX)]
        if not parameters or any(p.requires_grad for _, p in parameters):
            raise ValueError("The restored DINO backbone must be frozen")
    model.load_state_dict(supplied, strict=not bool(missing))
    actual = model.state_dict()
    if not all(torch.equal(actual[name].detach().cpu(), value.detach().cpu())
               for name, value in supplied.items()):
        raise RuntimeError("Loaded native checkpoint tensors are not bitwise equal")
    return {"loaded_tensors": len(supplied), "missing_frozen_backbone_tensors": len(missing),
            "all_supplied_weights_bitwise_equal": True, "public_export": exported}


def load_portable(state: dict[str, Any]):
    from mamba3_tracker.model.onnx_refiner import OnnxV35Refiner

    weights = {k: v for k, v in state["model"].items() if not k.startswith("dino.")}
    if "feat_proj.weight" not in weights:
        raise ValueError("Checkpoint is missing feat_proj.weight")
    model = OnnxV35Refiner(state["cfg"]["model"], dino_dim=weights["feat_proj.weight"].shape[1]).float().eval()
    _validate_supplied(model.state_dict(), weights)
    model.load_state_dict(weights, strict=True)
    if not all(torch.equal(value, weights[name]) for name, value in model.state_dict().items()):
        raise RuntimeError("Portable weights are not bitwise equal to the source")
    return model


def resolve_flow_config(state: dict[str, Any], run_config: dict[str, Any] | None) -> dict[str, Any]:
    """Public PT only has model config. Require an explicit run config for flow."""
    stored = state["cfg"].get("flow")
    supplied = None if run_config is None else run_config.get("flow")
    if stored is None and supplied is None:
        raise ValueError("Checkpoint has no flow config; supply --run-config explicitly")
    if stored is not None and supplied is not None and stored != supplied:
        raise ValueError("Run-config flow differs from the checkpoint")
    selected = stored if stored is not None else supplied
    if not isinstance(selected, dict):
        raise TypeError("Flow config must be a mapping")
    flow = dict(selected)
    if flow.get("source") != "waft_live":
        raise ValueError("The best200 paired protocol requires waft_live, without fallback")
    for name in ("scale", "iters", "fb_alpha", "fb_beta"):
        if name not in flow:
            raise ValueError(f"Explicit WAFT config is missing {name}")
    return flow
