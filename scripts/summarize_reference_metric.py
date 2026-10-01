"""State how a partial official-metric reference relates to the paper headline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    metrics, manifest = json.loads(args.metrics.read_text()), json.loads(args.manifest.read_text())
    observed = float(metrics["overall"]["metric_average_jaccard"])
    target = float(manifest["paper_reference"]["metric_average_jaccard"])
    result = {
        "scope": manifest["scope"],
        "reference_clips": manifest["clip_count"],
        "official_full_minival_clips": manifest["paper_reference"]["full_minival_clips"],
        "checkpoint": metrics.get("ckpt"),
        "official_metric_implementation": "TAPVid-3D median-scaled and absolute metric evaluators",
        "reference_metric_average_jaccard": observed,
        "paper_full_minival_metric_average_jaccard": target,
        "difference_from_paper": observed - target,
        "fraction_of_paper": observed / target if target else None,
        "not_directly_comparable": manifest["comparability"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
