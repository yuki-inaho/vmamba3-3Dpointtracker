"""Read-only inventory with conservative source-sequence grouping."""

import hashlib
import json
from pathlib import Path
from mamba3_tracker.data.tapvid3d_splits import MINIVAL_FILES

split_path = Path(
    "/workspace/vmamba3_data/result/v64_daoff_unused100gb_train_20261002/growing_pool.json"
)
split = json.loads(split_path.read_text())


def group(path):
    subset, stem = path.parent.name, path.stem
    if subset == "drivetrack":
        return subset + "/" + stem.split("_")[1]
    return subset + "/" + stem.rsplit("_", 1)[0]


paths = {key: [Path(p) for p in split[key]] for key in ("train", "validation")}
paths["minival"] = [
    Path("/workspace/vmamba3_eval/raw") / subset / name
    for subset, names in MINIVAL_FILES.items()
    for name in names
]
report = {
    "schema_version": 1,
    "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
    "group_rule": "ADT/PStudio: strip final clip suffix; DriveTrack: source segment numeric id. Conservative grouping, not proof of identity.",
    "teacher_history": "Current teacher used train/validation; earlier public teacher provenance is incomplete; minival was referenced by upstream development. No blind-test claim.",
    "splits": {
        key: {
            "count": len(value),
            "groups": len({group(p) for p in value}),
            "missing": [str(p) for p in value if not p.is_file()],
        }
        for key, value in paths.items()
    },
    "overlaps": {},
}
for a, b in [("train", "validation"), ("train", "minival"), ("validation", "minival")]:
    ga, gb = {group(p) for p in paths[a]}, {group(p) for p in paths[b]}
    report["overlaps"][a + ":" + b] = {
        "clips": sorted(
            {p.parent.name + "/" + p.name for p in paths[a]}
            & {p.parent.name + "/" + p.name for p in paths[b]}
        ),
        "groups": sorted(ga & gb),
        "right_clips_in_shared_groups": sum(group(p) in ga for p in paths[b]),
    }
out = Path("result/refiner_kd_20261002/preflight/inventory.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {
            "report": str(out),
            "counts": {k: v["count"] for k, v in report["splits"].items()},
        }
    )
)
