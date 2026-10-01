import json

import pytest

from mamba3_tracker.data.tapvid3d_splits import MINIVAL_FILES
from mamba3_tracker.eval.reference import manifest_paths, ready_minival_manifest


def _ready(raw, depth, subset, name):
    (raw / subset).mkdir(parents=True, exist_ok=True)
    (depth / subset).mkdir(parents=True, exist_ok=True)
    (raw / subset / name).write_bytes(b"raw")
    stem = name.removesuffix(".npz")
    (depth / subset / f"{stem}.npz").write_bytes(b"depth")
    (depth / subset / f"{stem}.npz.ready.json").write_text("{}")


def test_fixed_reference_is_ready_official_minival_and_rejects_leakage(tmp_path):
    raw, depth = tmp_path / "tapvid3d", tmp_path / "depth"
    for subset, names in MINIVAL_FILES.items():
        for name in names[:2]:
            _ready(raw, depth, subset, name)
    manifest = ready_minival_manifest(raw, depth, per_subset=2)
    assert manifest["clip_count"] == 6
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(manifest))
    got = manifest_paths(path, raw, "minival", list(MINIVAL_FILES))
    assert {subset: len(paths) for subset, paths in got.items()} == {"adt": 2, "drivetrack": 2, "pstudio": 2}
    manifest["files_by_subset"]["adt"][0] = "not-official.npz"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="outside official"):
        manifest_paths(path, raw, "minival", list(MINIVAL_FILES))
