"""The demo CLI exposes existing stages without loading models or writing outputs."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "scripts" / "demo_tracking_comparison.py"


def test_tracking_demo_help_lists_stages_without_running_inference():
    result = subprocess.run(
        [sys.executable, str(DEMO), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert all(stage in result.stdout for stage in ("infer", "render", "audit"))
    assert "--out-dir" in result.stdout


def test_tracking_demo_inference_requires_explicit_subset():
    result = subprocess.run(
        [sys.executable, str(DEMO), "--stage", "infer"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "requires an explicit --subset" in result.stderr
