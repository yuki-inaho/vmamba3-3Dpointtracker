"""A listed CUDA provider alone must not count as GPU execution evidence."""

import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "gpu_verification",
    Path(__file__).resolve().parents[2] / "scripts/verify_refiner_gpu.py",
)
assert spec is not None and spec.loader is not None
verification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verification)


def event(provider, op):
    return {"cat": "Node", "args": {"provider": provider, "op_name": op}}


def test_profile_requires_executed_cuda_compute_and_records_cpu_shape_nodes():
    result = verification.summarize_profile(
        [
            event("CUDAExecutionProvider", "MatMul"),
            event("CPUExecutionProvider", "Shape"),
        ]
    )
    assert result["cuda_compute_verified"]
    assert result["node_events_by_provider"]["CUDAExecutionProvider"] == 1
    assert result["cpu_ops"] == {"Shape": 1}


@pytest.mark.parametrize(
    "events",
    [
        [],
        [event("CPUExecutionProvider", "MatMul")],
        [event("CUDAExecutionProvider", "MemcpyFromHost")],
    ],
)
def test_profile_rejects_missing_cuda_compute(events):
    with pytest.raises(RuntimeError, match="CUDA compute"):
        verification.summarize_profile(events)


def test_profile_rejects_major_cpu_compute_even_with_cuda_available():
    with pytest.raises(RuntimeError, match="CPU compute"):
        verification.summarize_profile(
            [
                event("CUDAExecutionProvider", "MatMul"),
                event("CPUExecutionProvider", "GridSample"),
            ]
        )


def test_profile_rejects_floating_point_cpu_arithmetic():
    cpu = event("CPUExecutionProvider", "Mul")
    cpu["args"]["input_type_shape"] = [{"float": [2, 3]}]
    with pytest.raises(RuntimeError, match="CPU compute"):
        verification.summarize_profile([event("CUDAExecutionProvider", "MatMul"), cpu])
