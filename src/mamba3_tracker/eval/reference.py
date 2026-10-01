"""Fixed, inspectable partial-minival reference sets for training monitoring."""

from __future__ import annotations

import json
from pathlib import Path

from mamba3_tracker.data.tapvid3d_splits import FULL_EVAL_FILES, MINIVAL_FILES


def _root(data_root: str | Path) -> Path:
    root = Path(data_root).expanduser()
    return root if root.name == "tapvid3d" else root / "tapvid3d"


def ready_minival_manifest(data_root, depth_root, per_subset: int) -> dict:
    """Select the first complete official-minival clips, preserving split order."""
    if per_subset < 1:
        raise ValueError("per_subset must be positive")
    raw_root, depth_root = _root(data_root), Path(depth_root).expanduser()
    selected = {}
    for subset, names in MINIVAL_FILES.items():
        ready = [
            name
            for name in names
            if (raw_root / subset / name).is_file()
            and (depth_root / subset / f"{Path(name).stem}.npz").is_file()
            and (depth_root / subset / f"{Path(name).stem}.npz.ready.json").is_file()
        ]
        if len(ready) < per_subset:
            raise FileNotFoundError(
                f"{subset}: need {per_subset} raw+ready-depth minival clips, found {len(ready)}"
            )
        selected[subset] = ready[:per_subset]
    return {
        "scope": "partial_minival_reference",
        "split": "minival",
        "files_by_subset": selected,
        "clip_count": sum(map(len, selected.values())),
        "paper_reference": {
            "metric_average_jaccard": 0.256,
            "full_minival_clips": 150,
            "meaning": "paper headline absolute metric-AJ on the complete minival protocol",
        },
        "comparability": (
            "This fixed partial minival set uses official metric machinery but is a monitoring "
            "reference only. It is not comparable to the 150-clip paper headline."
        ),
    }


def manifest_paths(path, data_root, split, subsets) -> dict[str, list[Path]]:
    """Load exact manifest members and reject split leakage or missing raw clips."""
    manifest = json.loads(Path(path).expanduser().read_text())
    allowed = {"minival": MINIVAL_FILES, "full_eval": FULL_EVAL_FILES}.get(split)
    if allowed is None:
        raise ValueError("clip manifests require split=minival or split=full_eval")
    selection = manifest.get("files_by_subset")
    if not isinstance(selection, dict):
        raise ValueError("clip manifest must contain files_by_subset")
    root = _root(data_root)
    result = {}
    for subset in subsets:
        names = selection.get(subset, [])
        if not names or len(names) != len(set(names)):
            raise ValueError(f"{subset}: manifest selection must be nonempty and unique")
        leaked = set(names) - set(allowed[subset])
        if leaked:
            raise ValueError(f"{subset}: manifest contains clips outside official {split}: {sorted(leaked)[0]}")
        paths = [root / subset / name for name in names]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{subset}: manifest raw clip missing: {missing[0]}")
        result[subset] = paths
    return result
