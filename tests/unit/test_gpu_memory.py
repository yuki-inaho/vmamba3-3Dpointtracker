"""Pure-CPU NVML mocks; sampling must never turn missing data into zero."""

import importlib
import json
import os
import threading
from types import SimpleNamespace

import pytest


def _process(memory, pid=None):
    return SimpleNamespace(
        pid=os.getpid() if pid is None else pid, usedGpuMemory=memory
    )


class FakeNvml:
    NVML_VALUE_NOT_AVAILABLE = 2**64 - 1

    def __init__(self, responses=None, *, init_error=None, shutdown_error=None):
        self.responses = responses if responses is not None else [[_process(100)]]
        self.init_error, self.shutdown_error = init_error, shutdown_error
        self.init_calls = self.shutdown_calls = self.query_calls = 0
        self.device_indices = []
        self.second_query = threading.Event()

    def nvmlInit(self):
        self.init_calls += 1
        if self.init_error is not None:
            raise self.init_error

    def nvmlShutdown(self):
        self.shutdown_calls += 1
        if self.shutdown_error is not None:
            raise self.shutdown_error

    def nvmlDeviceGetHandleByIndex(self, index):
        self.device_indices.append(index)
        return index

    def nvmlDeviceGetComputeRunningProcesses(self, _handle):
        response = self.responses[min(self.query_calls, len(self.responses) - 1)]
        self.query_calls += 1
        if self.query_calls == 2:
            self.second_query.set()
        if isinstance(response, BaseException):
            raise response
        return response


def _install(monkeypatch, fake):
    module = importlib.import_module("mamba3_tracker.deployment.gpu_memory")
    original = module.importlib.import_module
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name: fake if name == "pynvml" else original(name),
    )
    return module.ProcessGpuMemorySampler


def test_own_pid_threaded_peak_start_end_and_json_report(monkeypatch):
    # Other PIDs, including their missing/huge memory, must not affect this PID.
    unrelated = [_process(10**13, os.getpid() + 1), _process(None, os.getpid() + 2)]
    fake = FakeNvml(
        [
            [_process(100), *unrelated],
            [_process(700), *unrelated],
            [_process(300), *unrelated],
        ]
    )
    cls = _install(monkeypatch, fake)
    sampler = cls(device_index=2, interval_seconds=0.01)
    with sampler as entered:
        assert entered is sampler
        assert fake.second_query.wait(1)
        with pytest.raises(RuntimeError, match="finished"):
            sampler.report()
    report = sampler.report()
    assert report["process_pid"] == os.getpid()
    assert report["device_index"] == 2
    assert report["start_bytes"] == 100
    assert report["peak_bytes"] == 700
    assert report["end_bytes"] == 300
    assert report["samples"] >= 3
    assert report["interval_seconds"] == 0.01
    assert report["duration_seconds"] > 0
    assert report["observed_max_gap_seconds"] > 0
    assert report["lower_bound"] is True
    assert report["peak_is_instantaneous_allocator_peak"] is False
    assert "NVML" in report["method"]
    assert "physical" in report["device_index_kind"]
    json.dumps(report, allow_nan=False)
    assert fake.init_calls == fake.shutdown_calls == 1
    assert fake.device_indices == [2]


def test_import_and_constructor_are_lazy_and_missing_dependency_fails(monkeypatch):
    module = importlib.import_module("mamba3_tracker.deployment.gpu_memory")
    original = module.importlib.import_module
    calls = []

    def missing(name):
        if name == "pynvml":
            calls.append(name)
            raise ModuleNotFoundError("pynvml missing")
        return original(name)

    monkeypatch.setattr(module.importlib, "import_module", missing)
    sampler = module.ProcessGpuMemorySampler()
    assert not calls
    with pytest.raises(RuntimeError, match="nvidia-ml-py"):
        with sampler:
            pass
    assert calls == ["pynvml"]


@pytest.mark.parametrize("memory", [None, -1, 2**64 - 1, 0.5, "100", True])
def test_missing_invalid_and_sentinel_memory_fail_and_shutdown(monkeypatch, memory):
    fake = FakeNvml([[_process(memory)]])
    sampler = _install(monkeypatch, fake)()
    with pytest.raises(RuntimeError, match="memory"):
        with sampler:
            pass
    assert fake.init_calls == fake.shutdown_calls == 1
    with pytest.raises(RuntimeError):
        sampler.report()


def test_missing_own_pid_is_not_zero(monkeypatch):
    fake = FakeNvml([[_process(100, os.getpid() + 1)]])
    with pytest.raises(RuntimeError, match="PID"):
        with _install(monkeypatch, fake)():
            pass
    assert fake.shutdown_calls == 1


def test_duplicate_own_pid_is_ambiguous_and_not_summed(monkeypatch):
    fake = FakeNvml([[_process(100), _process(300)]])
    with pytest.raises(RuntimeError, match="multiple|duplicate|ambiguous"):
        with _install(monkeypatch, fake)():
            pass
    assert fake.shutdown_calls == 1


def test_present_zero_memory_is_valid(monkeypatch):
    fake = FakeNvml([[_process(0)]])
    with _install(monkeypatch, fake)(interval_seconds=1) as sampler:
        pass
    report = sampler.report()
    assert report["start_bytes"] == report["peak_bytes"] == report["end_bytes"] == 0
    assert report["samples"] == 2


@pytest.mark.parametrize("stage", ["initial", "background", "end"])
def test_query_error_at_each_stage_fails_the_measurement(monkeypatch, stage):
    responses = (
        [RuntimeError("query denied")]
        if stage == "initial"
        else [[_process(100)], RuntimeError("query denied")]
    )
    fake = FakeNvml(responses)
    sampler = _install(monkeypatch, fake)(
        interval_seconds=0.01 if stage == "background" else 1
    )
    with pytest.raises(RuntimeError, match="query denied"):
        with sampler:
            if stage == "background":
                assert fake.second_query.wait(1)
    assert fake.init_calls == fake.shutdown_calls == 1
    with pytest.raises(RuntimeError):
        sampler.report()


def test_failed_init_does_not_shutdown_unacquired_reference(monkeypatch):
    fake = FakeNvml(init_error=RuntimeError("driver unavailable"))
    with pytest.raises(RuntimeError, match="driver unavailable"):
        with _install(monkeypatch, fake)():
            pass
    assert fake.init_calls == 1 and fake.shutdown_calls == 0


def test_shutdown_error_is_not_reported_as_success(monkeypatch):
    fake = FakeNvml(shutdown_error=RuntimeError("shutdown denied"))
    sampler = _install(monkeypatch, fake)(interval_seconds=1)
    with pytest.raises(RuntimeError, match="shutdown denied"):
        with sampler:
            pass
    assert fake.shutdown_calls == 1
    with pytest.raises(RuntimeError):
        sampler.report()


def test_body_failure_preserves_original_error_and_cleans_up(monkeypatch):
    fake = FakeNvml()
    sampler = _install(monkeypatch, fake)(interval_seconds=1)
    with pytest.raises(ValueError, match="benchmark failed"):
        with sampler:
            raise ValueError("benchmark failed")
    assert fake.shutdown_calls == 1
    with pytest.raises(RuntimeError, match="failed"):
        sampler.report()


def test_report_requires_finished_context_and_sampler_is_single_use(monkeypatch):
    fake = FakeNvml()
    sampler = _install(monkeypatch, fake)(interval_seconds=1)
    with pytest.raises(RuntimeError, match="finished"):
        sampler.report()
    with sampler:
        pass
    first = sampler.report()
    with pytest.raises(RuntimeError, match="single-use"):
        with sampler:
            pass
    assert sampler.report() == first
    assert fake.init_calls == fake.shutdown_calls == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device_index": -1},
        {"device_index": True},
        {"device_index": 0.5},
        {"interval_seconds": 0},
        {"interval_seconds": -0.1},
        {"interval_seconds": float("nan")},
        {"interval_seconds": float("inf")},
        {"interval_seconds": True},
    ],
)
def test_invalid_sampler_options_are_rejected(monkeypatch, kwargs):
    cls = _install(monkeypatch, FakeNvml())
    with pytest.raises(ValueError):
        cls(**kwargs)


@pytest.mark.parametrize("stage", ["handle", "initial", "end", "start", "join"])
@pytest.mark.parametrize("exception_class", [KeyboardInterrupt, SystemExit])
def test_interrupt_preserves_exception_and_releases_nvml(
    monkeypatch, stage, exception_class
):
    responses = (
        (
            [exception_class("cancelled")]
            if stage == "initial"
            else [[_process(100)], exception_class("cancelled")]
        )
        if stage in ("initial", "end")
        else None
    )
    fake = FakeNvml(responses)
    sampler = _install(monkeypatch, fake)(interval_seconds=1)
    if stage == "handle":

        def interrupted_handle(_index):
            raise exception_class("cancelled")

        monkeypatch.setattr(fake, "nvmlDeviceGetHandleByIndex", interrupted_handle)
    if stage in ("start", "join"):
        original = getattr(threading.Thread, stage)

        def interrupted_thread(thread, *args, **kwargs):
            original(thread, *args, **kwargs)
            raise exception_class("cancelled")

        monkeypatch.setattr(threading.Thread, stage, interrupted_thread)
    with pytest.raises(exception_class, match="cancelled"):
        with sampler:
            pass
    assert fake.init_calls == fake.shutdown_calls == 1
    if sampler._thread is not None:
        assert not sampler._thread.is_alive()
    with pytest.raises(RuntimeError):
        sampler.report()


@pytest.mark.parametrize("own_present", [False, True])
def test_no_other_compute_pids_allows_no_context_or_own_pid(monkeypatch, own_present):
    fake = FakeNvml([[_process(None)] if own_present else []])
    _install(monkeypatch, fake)
    module = importlib.import_module("mamba3_tracker.deployment.gpu_memory")
    report = module.require_exclusive_compute(device_index=2)
    assert report["no_other_compute_pids"] is True
    assert report["other_compute_pids"] == []
    assert report["compute_pids"] == ([os.getpid()] if own_present else [])
    assert "graphics" in report["scope"]
    assert "point-in-time" in report["scope"]
    json.dumps(report, allow_nan=False)
    assert fake.init_calls == fake.shutdown_calls == 1
    assert fake.device_indices == [2]


def test_no_other_compute_pids_rejects_only_safe_pid_list(monkeypatch):
    other = os.getpid() + 1000
    fake = FakeNvml([[_process(None), _process(10**13, other)]])
    _install(monkeypatch, fake)
    module = importlib.import_module("mamba3_tracker.deployment.gpu_memory")
    with pytest.raises(RuntimeError, match=str(other)) as caught:
        module.require_exclusive_compute()
    assert str(10**13) not in str(caught.value)
    assert fake.shutdown_calls == 1


@pytest.mark.parametrize("stage", ["init", "query", "shutdown"])
def test_no_other_compute_pids_nvml_failures_are_explicit(monkeypatch, stage):
    error = RuntimeError("NVML denied")
    fake = FakeNvml(
        [error] if stage == "query" else [[]],
        init_error=error if stage == "init" else None,
        shutdown_error=error if stage == "shutdown" else None,
    )
    _install(monkeypatch, fake)
    module = importlib.import_module("mamba3_tracker.deployment.gpu_memory")
    with pytest.raises(RuntimeError, match="NVML denied"):
        module.require_exclusive_compute()
    assert fake.shutdown_calls == (0 if stage == "init" else 1)


def test_no_other_compute_pids_wraps_nvml_library_errors(monkeypatch):
    class LibraryError(Exception):
        pass

    fake = FakeNvml([LibraryError("library unavailable")])
    _install(monkeypatch, fake)
    module = importlib.import_module("mamba3_tracker.deployment.gpu_memory")
    with pytest.raises(RuntimeError, match="library unavailable"):
        module.require_exclusive_compute()
    assert fake.shutdown_calls == 1


def test_cleanup_interrupt_is_not_masked_by_initial_query_error(monkeypatch):
    fake = FakeNvml(
        [RuntimeError("query denied")],
        shutdown_error=KeyboardInterrupt("cancel cleanup"),
    )
    sampler = _install(monkeypatch, fake)()
    with pytest.raises(KeyboardInterrupt, match="cancel cleanup"):
        with sampler:
            pass
    assert fake.shutdown_calls == 1
