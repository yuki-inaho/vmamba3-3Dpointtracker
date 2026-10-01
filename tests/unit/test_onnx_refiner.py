"""Check the export math against the independent upstream recurrent reference."""

import ast
import math
from pathlib import Path
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F
from einops import repeat

from mamba3_tracker.model.onnx_refiner import OnnxV35Refiner, reference_depth, siso_attention


@pytest.fixture(scope="module")
def upstream_recurrence():
    root = Path(__file__).resolve().parents[2]
    source = root / "third_party/visionMamba3/third_party/mamba-ssm/tests/ops/triton/test_mamba3_siso.py"
    # Load the upstream reference alone; CPU tests do not import CUDA kernels.
    tree = ast.parse(source.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "mamba3_siso_step_ref")
    namespace = {"torch": torch, "F": F, "math": math, "repeat": repeat,
                 "Optional": Optional, "Tuple": Tuple}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["mamba3_siso_step_ref"]


def inputs(frames):
    rng = torch.Generator().manual_seed(37)
    q, k = (torch.randn(2, frames, 3, 16, generator=rng) for _ in range(2))
    v, z = (torch.randn(2, frames, 3, 7, generator=rng) for _ in range(2))
    dt = torch.rand(2, frames, 3, generator=rng) * 0.1
    adt = -torch.rand(2, frames, 3, generator=rng) * dt
    trap = torch.randn(2, frames, 3, generator=rng)
    angles = torch.randn(2, frames, 3, 4, generator=rng)
    d = torch.rand(3, generator=rng)
    return q, k, v, adt, dt, trap, angles, d, z


@pytest.mark.parametrize("frames", [1, 8, 31])
def test_attention_matches_official_recurrence(upstream_recurrence, frames):
    q, k, v, adt, dt, trap, angles, d, z = inputs(frames)
    actual = siso_attention(q, k, v, adt, dt, trap, angles, d, z)
    expected, _ = upstream_recurrence(
        q, k, v, adt.transpose(1, 2), dt.transpose(1, 2), trap.transpose(1, 2),
        torch.zeros(3, 16), torch.zeros(3, 16), angles, D=d, Z=z)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_attention_is_causal_despite_shifted_trapezoid():
    args = inputs(8)
    full = siso_attention(*args)
    prefix_args = tuple(value if index == 7 else value[:, :3] for index, value in enumerate(args))
    prefix = siso_attention(*prefix_args)
    torch.testing.assert_close(full[:, :3], prefix, atol=2e-5, rtol=2e-5)


def test_reference_depth_uses_lower_median():
    depth = torch.tensor([1.0, 4.0, 9.0, 10.0]).reshape(1, 2, 2)
    assert reference_depth(depth).item() == pytest.approx(4.0 + 1e-6)


def test_export_rejects_different_mixer():
    with pytest.raises(ValueError, match="official_mamba3"):
        OnnxV35Refiner({"temporal_mixer": "vssd_cross"})
