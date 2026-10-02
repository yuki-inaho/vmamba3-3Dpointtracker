"""Deploy schema, graph contract and nonquadratic chunk accounting."""

from pathlib import Path
import pytest
import torch


def test_strict_student_schema(tmp_path):
    from mamba3_tracker.deployment.student_checkpoint import save_student, load_student
    from mamba3_tracker.model.student_refiner import StudentRefiner

    model, path = StudentRefiner(), tmp_path / "student.pt"
    save_student(model, path)
    loaded = load_student(path)
    assert all(
        torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items()
    )
    payload = torch.load(path, weights_only=True)
    payload["architecture"] = "unknown"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="architecture"):
        load_student(path)


def test_no_teacher_aux_in_public(tmp_path):
    from mamba3_tracker.deployment.student_checkpoint import save_student, load_student
    from mamba3_tracker.model.student_refiner import StudentRefiner

    path = tmp_path / "student.pt"
    save_student(StudentRefiner(), path)
    payload = torch.load(path, weights_only=True)
    assert set(payload) == {"schema_version", "architecture", "model", "model_sha256"}
    payload["model"]["aux.bad"] = torch.ones(1)
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_student(path)


def test_linear_workspace_not_teacher_quadratic():
    from mamba3_tracker.deployment.student_checkpoint import workspace_bytes

    assert workspace_bytes(1, 600, 32) == 2 * workspace_bytes(1, 300, 32)
    with pytest.raises(ValueError):
        workspace_bytes(1, 0, 32)


def test_legacy_teacher_unchanged():
    from mamba3_tracker.deployment.checkpoint import read_checkpoint, load_portable

    root = Path(__file__).resolve().parents[2]
    state = read_checkpoint(
        root / "weights/mamba3-preview-20261002-best80/tracker_mamba3_daoff_best80.pt",
        "c022511872c2d59a39bc1c39a3c62f44c142ee8ff9ca7dc7859881a78ae490b7",
    )
    model = load_portable(state)
    assert sum(p.numel() for p in model.parameters()) == 8117988
    assert all(torch.equal(v, state["model"][k]) for k, v in model.state_dict().items())


def test_track_chunk_invariance():
    from mamba3_tracker.model.student_refiner import StudentRefiner
    from scripts.export_refiner_onnx import synthetic_inputs

    model = StudentRefiner().eval()
    args = synthetic_inputs(1, 31, 7, 384, 896)
    with torch.no_grad():
        ref = model(*args)
        parts = [
            model(*(a[:, :, i : i + 1] if j < 4 else a for j, a in enumerate(args)))
            for i in range(7)
        ]
    for j in range(4):
        torch.testing.assert_close(
            ref[j], torch.cat([p[j] for p in parts], dim=2), rtol=2e-4, atol=5e-5
        )


def test_export_cli_requires_exactly_one_source():
    from scripts.export_student_refiner_onnx import parser

    with pytest.raises(SystemExit):
        parser().parse_args(["--out-dir", "unused"])
    with pytest.raises(SystemExit):
        parser().parse_args(
            ["--out-dir", "unused", "--ckpt", "a", "--init-config", "b"]
        )


def test_dynamic_axes(tmp_path):
    from scripts.export_student_refiner_onnx import export_graph
    from scripts.export_refiner_onnx import synthetic_inputs, compare, run_session
    from mamba3_tracker.deployment.runtime import session_for
    from mamba3_tracker.model.student_refiner import StudentRefiner

    model = StudentRefiner().eval()
    with torch.no_grad():
        model.get_parameter("dz_head.2.weight").normal_(std=0.01)
        model.get_parameter("duv_head.2.weight").normal_(std=0.01)
    path = tmp_path / "student.onnx"
    export_graph(model, path)
    session = session_for(path)
    for b, f, n in [(1, 1, 1), (2, 31, 7)]:
        args = synthetic_inputs(b, f, n, 384, 896, depth_shape=(17, 21))
        with torch.no_grad():
            compare(model(*args), run_session(session, args))


def test_cuda_profile_rejects_float_cpu():
    from scripts.verify_refiner_gpu import summarize_profile

    with pytest.raises(RuntimeError):
        summarize_profile(
            [
                {
                    "cat": "Node",
                    "name": "cpu_kernel_time",
                    "args": {
                        "provider": "CPUExecutionProvider",
                        "op_name": "MatMul",
                        "input_type_shape": [{"float": [1, 2]}],
                        "output_type_shape": [{"float": [1, 2]}],
                    },
                }
            ]
        )


def test_evaluator_rejects_missing_clip():
    from scripts.evaluate_student_refiner import require_complete

    with pytest.raises(ValueError, match="missing"):
        require_complete(["a"], ["a", "b"])


def test_partial_is_not_full_accuracy():
    from scripts.evaluate_student_refiner import require_complete

    with pytest.raises(ValueError):
        require_complete(["a", "a"], ["a", "b"])


def test_benchmark_excludes_warmup():
    from scripts.evaluate_student_refiner import summarize_timings

    result = summarize_timings([999, 999, 1, 2, 3], warmup=2)
    assert result["p50_ms"] == 2
    assert result["trials"] == 3


def test_benchmark_cpu_memory_is_explicitly_inapplicable():
    from scripts.evaluate_student_refiner import measure_benchmark

    results, memory = measure_benchmark(lambda: ["measured"], "cpu")
    assert results == ["measured"] and memory["applicable"] is False


def test_benchmark_cuda_memory_closes_before_reporting(monkeypatch):
    from mamba3_tracker.deployment import gpu_memory
    from scripts.evaluate_student_refiner import measure_benchmark

    events = []

    class Sampler:
        def __init__(self, **kwargs):
            assert kwargs == {"device_index": 0, "interval_seconds": 0.01}

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *args):
            events.append("exit")

        def report(self):
            events.append("report")
            return dict(peak_bytes=123, lower_bound=True)

    monkeypatch.setattr(gpu_memory, "ProcessGpuMemorySampler", Sampler)

    def operation():
        events.append("work")
        return ["measured"]

    results, memory = measure_benchmark(operation, "cuda")
    assert results == ["measured"] and memory["peak_bytes"] == 123
    assert memory["includes_session_initialization_peak"] is False
    assert events == ["enter", "work", "exit", "report"]


def test_benchmark_measurement_failure_is_not_success(monkeypatch):
    from mamba3_tracker.deployment import gpu_memory
    from scripts.evaluate_student_refiner import measure_benchmark

    class Sampler:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            raise RuntimeError("NVML missing PID")

    monkeypatch.setattr(gpu_memory, "ProcessGpuMemorySampler", Sampler)
    with pytest.raises(RuntimeError, match="missing PID"):
        measure_benchmark(lambda: [], "cuda")


def test_gpu_fixture_shapes_cover_extremes():
    from scripts.evaluate_student_refiner import fixture_specs

    cases = fixture_specs()
    assert any(c["shape"] == (1, 600, 900) for c in cases)
    assert any(c["invisible"] for c in cases)
    assert any(c["shape"][0] == 2 for c in cases)


def test_benchmark_chunks_preserve_full_reference_and_time():
    from scripts.evaluate_student_refiner import benchmark_chunks
    from scripts.export_refiner_onnx import synthetic_inputs, as_feed

    feed = as_feed(synthetic_inputs(1, 8, 7, 384, 896, depth_shape=(17, 21)))
    chunks = benchmark_chunks(feed, 3)
    assert [part["ray"].shape[2] for part in chunks] == [3, 3, 1]
    assert all(part["ray"].shape[1] == 8 for part in chunks)
    assert all(part["z_ref"] is feed["z_ref"] for part in chunks)
    assert all(part["dino_features"] is feed["dino_features"] for part in chunks)
    with pytest.raises(ValueError):
        benchmark_chunks(feed, 0)
