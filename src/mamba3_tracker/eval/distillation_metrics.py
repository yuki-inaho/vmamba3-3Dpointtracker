"""Paired, sequence-clustered non-inferiority gates for fixed TAPVid-3D sets."""

from collections import Counter

import numpy as np

SUBSETS = ("adt", "drivetrack", "pstudio")
METRICS = ("average_jaccard", "metric_average_jaccard")


def paired_acceptance(
    rows: list[dict],
    expected: set[tuple[str, str]],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
    macro_margin: float = 0.003,
    subset_margin: float = 0.005,
) -> dict:
    """Positive differences mean student degradation; resample whole sequences.

    Within each subset, each draw samples the original number of sequences with
    replacement. All clips of a sampled sequence remain together, and each draw
    uses the clip-weighted mean. The macro is an equal-weight subset average.
    These are one-sided percentile bounds, not evidence of unseen-data status.
    """
    keys = [(r["subset"], r["filename"]) for r in rows]
    if set(keys) != expected or any(n != 1 for n in Counter(keys).values()):
        raise ValueError("Evaluation must contain every expected clip exactly once")
    if {s for s, _ in expected} != set(SUBSETS):
        raise ValueError("All three official subsets are required")
    if bootstrap_samples < 1 or macro_margin < 0 or subset_margin < 0:
        raise ValueError("Invalid bootstrap count or margins")
    for row in rows:
        if not isinstance(row["sequence"], str) or not row["sequence"]:
            raise ValueError("Each clip requires an audited sequence ID")
        for source in ("teacher", "student"):
            for metric in METRICS:
                value = float(row[source][metric])
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("AJ must be finite and on the [0, 1] scale")
    rng = np.random.default_rng(seed)
    result: dict = {
        "passed": True,
        "metrics": {},
        "violations": [],
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "unit": "sequence_within_subset",
        "clips": len(rows),
        "macro_margin": macro_margin,
        "subset_margin": subset_margin,
    }
    # Same sequence draws for both metrics preserve the paired experiment.
    samples: dict[str, np.ndarray] = {}
    points: dict[str, np.ndarray] = {}
    counts: dict[str, tuple[int, int]] = {}
    for subset in SUBSETS:
        members = sorted(
            (r for r in rows if r["subset"] == subset),
            key=lambda r: r["filename"],
        )
        groups = sorted({r["sequence"] for r in members})
        sums = np.zeros((len(groups), len(METRICS)), dtype=np.float64)
        sizes = np.zeros(len(groups), dtype=np.int64)
        for i, group in enumerate(groups):
            for row in members:
                if row["sequence"] == group:
                    sizes[i] += 1
                    sums[i] += [
                        float(row["teacher"][m]) - float(row["student"][m])
                        for m in METRICS
                    ]
        draws = rng.integers(0, len(groups), (bootstrap_samples, len(groups)))
        samples[subset] = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)[:, None]
        points[subset] = sums.sum(axis=0) / sizes.sum()
        counts[subset] = (len(members), len(groups))
    macro_samples = np.stack(list(samples.values())).mean(axis=0)
    macro_point = np.stack(list(points.values())).mean(axis=0)
    for index, metric in enumerate(METRICS):
        report = {}
        for subset in (*SUBSETS, "macro"):
            point = macro_point[index] if subset == "macro" else points[subset][index]
            distribution = (
                macro_samples[:, index]
                if subset == "macro"
                else samples[subset][:, index]
            )
            upper = float(np.quantile(distribution, 0.95))
            margin = macro_margin if subset == "macro" else subset_margin
            passed = bool(point <= margin and upper <= margin)
            report[subset] = {
                "teacher_minus_student": float(point),
                "upper95": upper,
                "margin": margin,
                "passed": passed,
            }
            if subset != "macro":
                report[subset]["clips"], report[subset]["sequences"] = counts[subset]
            if not passed:
                result["violations"].append(f"{subset}.{metric}")
        result["metrics"][metric] = report
    result["passed"] = not result["violations"]
    return result
