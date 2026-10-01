"""Protocol, provenance and no-pickle persistence regressions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mamba3_tracker.deployment.io import save_npz
from mamba3_tracker.deployment.protocol import (
    DINO_REVISION,
    fixed_manifest,
    preflight,
    resolve_dino_revision,
)

ROOT = Path(__file__).resolve().parents[2]


def test_fixed_manifest_requires_the_original_nine(tmp_path):
    path = ROOT / "configs/v64_metric_reference_minival.json"
    assert sum(len(v) for v in fixed_manifest(path).values()) == 9
    changed = tmp_path / "manifest.json"
    changed.write_text(path.read_text().replace("basketball_5", "basketball_6"))
    with pytest.raises(ValueError, match="unchanged fixed-nine"):
        fixed_manifest(changed)


def test_dino_uses_immutable_recorded_revision():
    state = {"export": {"dino_revision": DINO_REVISION}}
    assert resolve_dino_revision(state) == DINO_REVISION
    assert resolve_dino_revision({}, DINO_REVISION) == DINO_REVISION
    with pytest.raises(ValueError, match="differs"):
        resolve_dino_revision(state, "f" * 40)


@pytest.mark.parametrize("revision", [None, "main", "f" * 40, "abc"])
def test_unknown_dino_revision_is_rejected(revision):
    with pytest.raises(ValueError):
        resolve_dino_revision({}, revision)


def test_missing_data_and_cuda_cannot_pass_preflight(tmp_path):
    manifest = fixed_manifest(ROOT / "configs/v64_metric_reference_minival.json")
    result = preflight(manifest, tmp_path / "data", tmp_path / "depth", tmp_path, False)
    assert not result["ready"] and not result["task_accuracy_verified"]
    assert len(result["inputs"]) == 9 and result["available_valid_frames"] == 0
    assert any("CUDA" in item for item in result["blockers"])
    assert sum(item.startswith("Missing data") for item in result["blockers"]) == 9
    assert sum(item.startswith("Missing depth") for item in result["blockers"]) == 9


def test_numeric_npz_roundtrip_and_object_rejection(tmp_path):
    path = tmp_path / "data.npz"
    expected = {"z_ref": np.array(2, np.float32), "ray": np.ones((1, 3, 2, 2), np.float32)}
    save_npz(path, expected)
    with np.load(path, allow_pickle=False) as actual:
        assert set(actual.files) == set(expected)
        for key, value in expected.items():
            np.testing.assert_array_equal(actual[key], value)
    with pytest.raises(TypeError):
        save_npz(path, {"bad": np.array([object()], dtype=object)})
    # Invalid input must not replace a valid existing output.
    with np.load(path, allow_pickle=False) as actual:
        np.testing.assert_array_equal(actual["ray"], expected["ray"])
    with pytest.raises(ValueError):
        save_npz(path, {"../bad": np.zeros(1)})


def test_quality_versions_ignore_stderr_warnings(tmp_path, monkeypatch):
    import importlib.util
    import json
    import subprocess
    import sys

    spec = importlib.util.spec_from_file_location("cpu_verification", ROOT / "scripts/validate_refiner_cpu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "digest", lambda _: "bcddf045311b76a91bc772e35901acbdb99f397b752ee0ba4beacb3fb1ded5f4")
    monkeypatch.setattr(sys, "argv", ["validate_refiner_cpu.py", "--ckpt", str(tmp_path / "model.pt"),
                                    "--sdk-root", str(tmp_path / "sdk")])

    def fake_run(command, **kwargs):
        if "scripts/export_refiner_onnx.py" in command:
            raise RuntimeError("stop after version parsing")
        is_versions = any("import json,sys,torch" in item for item in command)
        stdout = json.dumps({"python": "test"}) if is_versions else ""
        stderr = "ORT device warning\n" if is_versions else ""
        if kwargs.get("stderr") == subprocess.STDOUT:
            stdout, stderr = stderr + stdout, ""
        return subprocess.CompletedProcess(command, 0, stdout, stderr)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="stop after version parsing"):
        module.main()
    result = json.loads((tmp_path / "result/onnx_best200_cpu/quality.json").read_text())
    assert result["versions"] == {"python": "test"}
    assert result["success"] is False
    assert "ORT device warning" in (tmp_path / "result/onnx_best200_cpu/logs/versions.txt").read_text()
