import json

import pytest

from mamba3_tracker.data.growing import GrowingClipPool
from mamba3_tracker.data.tapvid3d_splits import FULL_EVAL_FILES, MINIVAL_FILES


def setup_pool(tmp_path):
    manifest = tmp_path / "selection.json"
    manifest.write_text(
        json.dumps(
            {
                "files_by_subset": {
                    s: FULL_EVAL_FILES[s][:4] for s in ["adt", "drivetrack", "pstudio"]
                }
            }
        )
    )
    return manifest, tmp_path / "tapvid3d", tmp_path / "depth", tmp_path / "pool.json"


def ready(root, depth, subset, name):
    for parent in [root / subset, depth / subset]:
        parent.mkdir(parents=True, exist_ok=True)
    (root / subset / name).write_bytes(b"raw")
    (depth / subset / name).write_bytes(b"depth")
    (depth / subset / (name + ".ready.json")).write_text(
        json.dumps({"raw_bytes": 3, "depth_bytes": 5, "frames": 300})
    )


def test_append_and_fixed_validation(tmp_path):
    manifest, root, depth, state = setup_pool(tmp_path)
    subsets = ["adt", "drivetrack", "pstudio"]
    for subset in subsets:
        for name in FULL_EVAL_FILES[subset][:2]:
            ready(root, depth, subset, name)
    pool = GrowingClipPool(manifest, root, depth, state, subsets, val_per_subset=1)
    train, val = pool.snapshot()
    assert len(train) == len(val) == 3
    original_val = list(val)
    for subset in subsets:
        ready(root, depth, subset, FULL_EVAL_FILES[subset][2])
        ready(root, depth, subset, MINIVAL_FILES[subset][0])
    train, val = pool.snapshot()
    assert len(train) == 6 and val == original_val
    assert not set(train) & set(val)
    restarted = GrowingClipPool(manifest, root, depth, state, subsets, val_per_subset=1)
    assert restarted.snapshot() == (train, val)


def test_incomplete_depth_not_eligible(tmp_path):
    manifest, root, depth, state = setup_pool(tmp_path)
    for subset in ["adt", "drivetrack", "pstudio"]:
        for name in FULL_EVAL_FILES[subset][:2]:
            ready(root, depth, subset, name)
    (depth / "adt" / (FULL_EVAL_FILES["adt"][0] + ".ready.json")).unlink()
    pool = GrowingClipPool(
        manifest, root, depth, state, ["adt", "drivetrack", "pstudio"], val_per_subset=1
    )
    with pytest.raises(RuntimeError, match="ready"):
        pool.snapshot()


def test_immutable_selection_on_resume(tmp_path):
    manifest, root, depth, state = setup_pool(tmp_path)
    for name in FULL_EVAL_FILES["pstudio"][:2]:
        ready(root, depth, "pstudio", name)
    pool = GrowingClipPool(manifest, root, depth, state, ["pstudio"], val_per_subset=1)
    pool.snapshot()
    data = json.loads(manifest.read_text())
    data["files_by_subset"]["pstudio"] = FULL_EVAL_FILES["pstudio"][1:4]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="selection"):
        GrowingClipPool(manifest, root, depth, state, ["pstudio"], val_per_subset=1)
