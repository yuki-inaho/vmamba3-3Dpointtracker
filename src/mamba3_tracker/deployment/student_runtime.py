"""NumPy-only student runner; full temporal context with track-only chunks."""

from __future__ import annotations

import numpy as np

from .runtime import OUTPUT_NAMES, TRACK_INPUT_NAMES, _run_validated, validate_feed

ARCHITECTURE = "vssd_two_pool_128x2_v1"
ARCHITECTURES = frozenset((ARCHITECTURE, "vssd_local_global_128x2_v2"))


def workspace_bytes(batch: int, frames: int, tracks: int) -> int:
    if min(batch, frames, tracks) < 1:
        raise ValueError("Workspace dimensions must be positive")
    return batch * frames * tracks * 128 * 4 * 64


def run_student_chunked(
    session,
    feed: dict[str, np.ndarray],
    track_chunk: int = 32,
    memory_budget_mib: int = 1024,
) -> list[np.ndarray]:
    if (
        session.get_modelmeta().custom_metadata_map.get("architecture")
        not in ARCHITECTURES
    ):
        raise ValueError("Student runner refuses unknown/teacher architecture")
    if track_chunk < 1 or memory_budget_mib < 1:
        raise ValueError("Invalid chunk or memory budget")
    b, f, n = validate_feed(feed)
    if workspace_bytes(b, f, min(n, track_chunk)) > memory_budget_mib * 1024**2:
        raise MemoryError(
            "Linear student workspace exceeds budget; reduce tracks, not frames"
        )
    groups: list[list[np.ndarray]] = [[] for _ in OUTPUT_NAMES]
    for start in range(0, n, track_chunk):
        chunk = dict(feed)
        for name in TRACK_INPUT_NAMES:
            chunk[name] = np.ascontiguousarray(
                feed[name][:, :, start : start + track_chunk]
            )
        for group, value in zip(groups, _run_validated(session, chunk), strict=True):
            group.append(value)
    return [np.concatenate(group, axis=2) for group in groups]
