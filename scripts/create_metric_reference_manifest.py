"""Create a fixed, ready-data TAPVid-3D minival reference manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mamba3_tracker.eval.reference import ready_minival_manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("~/data"))
    ap.add_argument("--depth-root", type=Path, default=Path("~/data/tapvid3d_da3"))
    ap.add_argument("--per-subset", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    manifest = ready_minival_manifest(args.data_root, args.depth_root, args.per_subset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[reference] wrote {args.out}: {manifest['clip_count']} fixed minival clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
