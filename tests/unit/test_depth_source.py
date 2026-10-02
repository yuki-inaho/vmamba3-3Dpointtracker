import json

import pytest

from mamba3_tracker.data.depth_source import resolve


@pytest.mark.parametrize("name", ["da3l", "da3g"])
def test_stamped_relocation_preserves_canonical_source(tmp_path, name):
    original = resolve(name)
    original.stamp(tmp_path, root=str(tmp_path))
    relocated = resolve(tmp_path)
    assert relocated.name == original.name
    assert relocated.root == tmp_path.resolve()
    assert relocated.model == original.model
    assert relocated.label == original.label
    assert relocated.verify(tmp_path) == f"verified {name}"
    assert resolve(name) == original


def test_unstamped_unknown_path_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="not a known depth source"):
        resolve(tmp_path)


@pytest.mark.parametrize(
    "override",
    [
        {"depth_source": "unknown"},
        {"depth_source": "large"},
        {"root": "/not/the/requested/root"},
        {"model": "DA3NESTED-GIANT-LARGE (1.4B)"},
        {"depth_source": ["da3l"]},
        {"root": None},
        {"model": None},
    ],
)
def test_invalid_relocation_stamp_is_rejected(tmp_path, override):
    resolve("da3l").stamp(tmp_path, root=str(tmp_path))
    stamp = tmp_path / "depth_source.json"
    payload = json.loads(stamp.read_text())
    payload.update(override)
    stamp.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="invalid relocation stamp"):
        resolve(tmp_path)


@pytest.mark.parametrize("content", ["{broken", "[]", "null", "{}"])
def test_malformed_relocation_stamp_is_rejected(tmp_path, content):
    (tmp_path / "depth_source.json").write_text(content)
    with pytest.raises(SystemExit, match="invalid relocation stamp"):
        resolve(tmp_path)
