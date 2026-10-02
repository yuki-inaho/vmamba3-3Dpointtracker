"""Fail-closed portable student format, independent of teacher checkpoint schema."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor, nn

from mamba3_tracker.model.student_refiner import StudentRefiner
from .checkpoint import read_checkpoint, sha256
from .student_runtime import workspace_bytes as workspace_bytes


def create_student(architecture: str) -> StudentRefiner:
    if architecture == StudentRefiner.architecture:
        return StudentRefiner()
    if architecture in (
        "vssd_local_global_128x2_v2_train",
        "vssd_local_global_128x2_v2",
    ):
        from mamba3_tracker.model.local_global_refiner import LocalGlobalStudentRefiner

        return LocalGlobalStudentRefiner(
            deploy=architecture == "vssd_local_global_128x2_v2"
        )
    raise ValueError("Unsupported student architecture")


def tensor_digest(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_student(model: nn.Module) -> StudentRefiner:
    """Check the registered source contract, including nonserialized semantics.

    State shapes alone do not identify a function: pooling normalization, Conv
    padding, activation choice and geometry bounds also affect its outputs.
    Compare to a fresh registered structure without consuming the caller RNG.
    Training/eval flags are deliberately excluded because both may be saved.
    """
    architecture = getattr(model, "architecture", None)
    if not isinstance(architecture, str):
        raise ValueError(
            "Student contract requires a registered architecture; no wrapper"
        )
    with torch.random.fork_rng(devices=[]):
        expected = create_student(architecture)
    if type(model) is not type(expected):
        raise ValueError("Student contract rejects wrapper/teacher/aux model types")
    assert isinstance(model, StudentRefiner)
    source_modules, target_modules = (
        dict(model.named_modules()),
        dict(expected.named_modules()),
    )
    if source_modules.keys() != target_modules.keys():
        raise ValueError("Student structure differs from registered contract")
    for name, module in source_modules.items():
        target = target_modules[name]
        if type(module) is not type(target):
            raise ValueError(f"Student structure differs at {name}")
        # All public attributes of these registered modules are simple semantic
        # configuration. Weights/buffers/children live in nn.Module's private
        # dictionaries and are checked separately below.
        attrs = {
            key: value
            for key, value in vars(module).items()
            if not key.startswith("_") and key != "training"
        }
        target_attrs = {
            key: value
            for key, value in vars(target).items()
            if not key.startswith("_") and key != "training"
        }
        if attrs != target_attrs:
            raise ValueError(f"Student semantic contract differs at {name}")
    state, target_state = model.state_dict(), expected.state_dict()
    if state.keys() != target_state.keys():
        raise ValueError("Unexpected or missing student tensors")
    for key, value in state.items():
        if (
            value.shape != target_state[key].shape
            or value.dtype != torch.float32
            or not torch.isfinite(value).all()
        ):
            raise ValueError(
                f"Invalid student tensor contract: {key}; finite FP32 required"
            )
    return model


def save_student(model: nn.Module, path: Path) -> None:
    model = validate_student(model)
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if not all(
        v.dtype == torch.float32 and torch.isfinite(v).all() for v in state.values()
    ):
        raise ValueError("Public student must be finite FP32")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(
        dict(
            schema_version=1,
            architecture=model.architecture,
            model=state,
            model_sha256=tensor_digest(state),
        ),
        temporary,
    )
    temporary.replace(path)


def load_student(path: Path, expected_sha256: str | None = None) -> StudentRefiner:
    if expected_sha256 is not None and sha256(Path(path)) != expected_sha256:
        raise ValueError("Student file SHA256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "architecture",
        "model",
        "model_sha256",
    }:
        raise ValueError("Invalid public student schema")
    if payload["schema_version"] != 1:
        raise ValueError("Unsupported student schema/architecture")
    model = create_student(payload["architecture"]).float().eval()
    state, target = payload["model"], model.state_dict()
    if not isinstance(state, dict) or state.keys() != target.keys():
        raise ValueError("Unexpected or missing student tensors")
    for key, value in state.items():
        if (
            not isinstance(value, Tensor)
            or value.shape != target[key].shape
            or value.dtype != torch.float32
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Invalid student tensor: {key}")
    if tensor_digest(state) != payload["model_sha256"]:
        raise ValueError("Student tensor SHA256 mismatch")
    model.load_state_dict(state, strict=True)
    if tensor_digest(model.state_dict()) != payload["model_sha256"]:
        raise RuntimeError("Loaded student is not bitwise equal")
    return model


def load_native_teacher(path: Path, expected_sha256: str) -> StudentRefiner:
    """External-feature native teacher, preserving official BF16 mixer execution."""
    from mamba3_tracker.model.official_mamba3 import OfficialMamba3Adapter

    class NativeTeacher(StudentRefiner):
        architecture = "official_mamba3_teacher_external_features"

    state = read_checkpoint(path, expected_sha256)
    cfg = state["cfg"]["model"]
    if cfg.get("temporal_mixer") != "official_mamba3" or cfg.get("two_pool", False):
        raise ValueError("KD native teacher requires the fixed official teacher")
    # Allocate just the shared geometry and native layers, no DINO model.
    model = NativeTeacher.__new__(NativeTeacher)
    nn.Module.__init__(model)
    model._init_geometry(cfg)
    model.layers = nn.ModuleList(
        [OfficialMamba3Adapter(128, True, 128) for _ in range(2)]
    )
    supplied = {k: v for k, v in state["model"].items() if not k.startswith("dino.")}
    model.load_state_dict(supplied, strict=True)
    if tensor_digest(model.state_dict()) != tensor_digest(supplied):
        raise RuntimeError("Native teacher weights changed during load")
    return model.eval().requires_grad_(False)
