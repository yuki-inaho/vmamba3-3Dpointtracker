"""Optional NVML own-process VRAM sampling, not a CUDA allocator peak.

The module itself imports neither CUDA libraries nor pynvml. Enter this context
only for CUDA benchmarks, after creating a CUDA context on the selected NVML
physical device. Synchronize benchmark work before leaving it. The requested
10 ms wait is not a real-time frequency guarantee: query time and scheduling
can miss short peaks, so the largest observed value is explicitly a lower bound.

NVML reports absolute process memory, including existing contexts/arenas and
other work in the same process. Other PIDs are excluded, never subtracted from
device-global memory. Missing PID, MPS attribution, ambiguous MIG process rows,
unavailable values, or driver/API errors are failures rather than zero usage.
"""

from __future__ import annotations

import importlib
import math
import operator
import os
import threading
import time
from numbers import Real
from types import TracebackType
from typing import Any


def _validate_device_index(device_index: int) -> None:
    if (
        isinstance(device_index, bool)
        or not isinstance(device_index, int)
        or device_index < 0
    ):
        raise ValueError("device_index must be a nonnegative NVML physical index")


def _load_nvml() -> Any:
    try:
        return importlib.import_module("pynvml")
    except ImportError as error:
        raise RuntimeError(
            "CUDA VRAM sampling requires optional nvidia-ml-py"
        ) from error


def require_exclusive_compute(device_index: int = 0) -> dict[str, Any]:
    """Read-only point-in-time check for other compute PIDs, not a reservation.

    This deliberately allows no own-PID CUDA context yet. It neither checks
    graphics processes nor claims exclusive use of all GPU hardware/memory.
    NVML's physical ordinal need not match CUDA_VISIBLE_DEVICES ordinals.
    """
    _validate_device_index(device_index)
    backend = _load_nvml()
    initialized = False
    failure: BaseException | None = None
    try:
        backend.nvmlInit()
        initialized = True
        handle = backend.nvmlDeviceGetHandleByIndex(device_index)
        processes = backend.nvmlDeviceGetComputeRunningProcesses(handle)
        pids = sorted({operator.index(process.pid) for process in processes})
        own_pid = os.getpid()
        other_pids = [pid for pid in pids if pid != own_pid]
        if other_pids:
            raise RuntimeError(
                f"Other compute PIDs on physical GPU {device_index}: {other_pids}"
            )
        return {
            "process_pid": own_pid,
            "device_index": device_index,
            "device_index_kind": "NVML physical device ordinal, not CUDA-visible ordinal",
            "compute_pids": pids,
            "other_compute_pids": [],
            "no_other_compute_pids": True,
            "scope": "Read-only point-in-time compute-PID check; graphics processes not checked; no reservation or continued-exclusivity guarantee",
        }
    except BaseException as error:
        if isinstance(error, Exception):
            failure = RuntimeError(
                f"NVML compute-PID exclusivity check failed: {error}"
            )
            raise failure from error
        failure = error
        raise
    finally:
        if initialized:
            try:
                backend.nvmlShutdown()
            except BaseException as error:
                if failure is None:
                    if isinstance(error, Exception):
                        raise RuntimeError(
                            f"NVML exclusivity-check shutdown failed: {error}"
                        ) from error
                    raise
                failure.add_note(f"NVML shutdown also failed: {error}")


class ProcessGpuMemorySampler:
    """Single-use context manager sampling NVML usedGpuMemory for os.getpid()."""

    def __init__(self, device_index: int = 0, interval_seconds: float = 0.01) -> None:
        _validate_device_index(device_index)
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, Real)
            or not math.isfinite(interval_seconds)
            or not 0 < interval_seconds <= threading.TIMEOUT_MAX
        ):
            raise ValueError("interval_seconds must be finite, positive, and supported")
        self.device_index = device_index
        self.interval_seconds = float(interval_seconds)
        self._pid = os.getpid()
        self._nvml: Any = None
        self._handle: Any = None
        self._initialized = False
        self._used = self._closed = False
        self._failure: BaseException | None = None
        self._stop = threading.Event()
        self._worker_done = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[tuple[float, int]] = []

    def _sample(self) -> None:
        try:
            processes = self._nvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
            matches = [process for process in processes if process.pid == self._pid]
            if not matches:
                raise RuntimeError(
                    f"Own PID {self._pid} is absent from NVML compute processes; "
                    "check active CUDA context, physical device, PID namespace or MPS"
                )
            if len(matches) != 1:
                raise RuntimeError(
                    "Ambiguous duplicate own-PID NVML process rows (MIG unsupported)"
                )
            raw_memory = matches[0].usedGpuMemory
            if raw_memory is None or isinstance(raw_memory, bool):
                raise RuntimeError("NVML process memory unavailable or invalid")
            try:
                memory = operator.index(raw_memory)
            except TypeError as error:
                raise RuntimeError(
                    "NVML process memory must be integer bytes"
                ) from error
            if not 0 <= memory < 2**64 - 1:
                raise RuntimeError(
                    "NVML process memory invalid/NVML_VALUE_NOT_AVAILABLE"
                )
            self._samples.append((time.monotonic(), memory))
        except Exception as error:
            raise RuntimeError(f"NVML memory sampling failed: {error}") from error

    def _sampling_loop(self) -> None:
        try:
            while not self._stop.wait(self.interval_seconds):
                self._sample()
        except BaseException as error:
            self._failure = error
            self._stop.set()
        finally:
            self._worker_done.set()

    def _remember_failure(self, error: BaseException) -> None:
        if self._failure is None:
            self._failure = error
        elif not isinstance(error, Exception) and isinstance(self._failure, Exception):
            error.add_note(f"GPU memory measurement also failed: {self._failure}")
            self._failure = error
        else:
            self._failure.add_note(f"GPU memory cleanup also failed: {error}")

    def _stop_worker(self) -> None:
        self._stop.set()
        if self._thread is None or self._thread.ident is None:
            return
        interruption: BaseException | None = None
        # Wait for our finally marker rather than relying only on is_alive():
        # an interrupted CPython join can mark a thread stopped prematurely.
        while not self._worker_done.is_set():
            try:
                self._worker_done.wait()
            except BaseException as error:
                if interruption is None:
                    interruption = error
                else:
                    interruption.add_note(f"Another interrupt during cleanup: {error}")
        try:
            self._thread.join()
        except BaseException as error:
            if interruption is None:
                interruption = error
            else:
                interruption.add_note(f"Thread join also interrupted: {error}")
        if interruption is not None:
            raise interruption

    def _shutdown(self) -> None:
        if self._initialized:
            # Exactly one shutdown attempt per acquired NVML reference.
            self._initialized = False
            try:
                self._nvml.nvmlShutdown()
            except BaseException as error:
                self._remember_failure(error)

    def __enter__(self) -> ProcessGpuMemorySampler:
        if self._used:
            raise RuntimeError("GPU memory sampler is single-use")
        self._used = True
        self._pid = os.getpid()
        try:
            self._nvml = _load_nvml()
            self._nvml.nvmlInit()
            self._initialized = True
            self._handle = self._nvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._sample()
            self._thread = threading.Thread(
                target=self._sampling_loop,
                name=f"nvml-vram-pid-{self._pid}",
                daemon=True,
            )
            self._thread.start()
        except BaseException as error:
            self._failure = error
            try:
                self._stop_worker()
            except BaseException as cleanup_error:
                self._remember_failure(cleanup_error)
            finally:
                self._shutdown()
                self._closed = True
            if not isinstance(error, Exception):
                raise
            if self._failure is not None and not isinstance(self._failure, Exception):
                raise self._failure
            raise RuntimeError(
                f"NVML GPU memory context initialization failed: {error}"
            ) from error
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        try:
            self._stop_worker()
            # Never race a query against the final sample or NVML shutdown.
            if self._failure is None:
                self._sample()
        except BaseException as error:
            self._remember_failure(error)
        finally:
            self._shutdown()
            self._closed = True
        if exception is not None:
            if self._failure is not None:
                exception.add_note(
                    f"GPU memory measurement also failed: {self._failure}"
                )
            else:
                self._failure = RuntimeError(
                    "GPU memory measurement context body failed"
                )
            return False
        if self._failure is not None:
            if not isinstance(self._failure, Exception):
                raise self._failure
            raise RuntimeError(
                f"NVML GPU memory measurement failed: {self._failure}"
            ) from self._failure
        return False

    def report(self) -> dict[str, Any]:
        """JSON-safe report after successful close; no partial-success report."""
        if not self._closed:
            raise RuntimeError("GPU memory context must be finished before reporting")
        if self._failure is not None:
            raise RuntimeError(
                f"GPU memory measurement failed: {self._failure}"
            ) from self._failure
        times, memory = zip(*self._samples, strict=True)
        return {
            "method": "NVML own-PID compute-process usedGpuMemory threaded sampling",
            "process_pid": self._pid,
            "device_index": self.device_index,
            "device_index_kind": "NVML physical device ordinal, not CUDA-visible ordinal",
            "start_bytes": memory[0],
            "end_bytes": memory[-1],
            "peak_bytes": max(memory),
            "samples": len(memory),
            "interval_seconds": self.interval_seconds,
            "observed_max_gap_seconds": max(
                right - left for left, right in zip(times, times[1:])
            ),
            "duration_seconds": times[-1] - times[0],
            "lower_bound": True,
            "peak_is_instantaneous_allocator_peak": False,
            "scope": "Sampled absolute current-process memory; other PIDs excluded",
            "note": "Short peaks may be missed; synchronize CUDA before context exit. Existing arenas and other work in this process are included.",
        }
