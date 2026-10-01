import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "staged", Path(__file__).resolve().parents[2] / "scripts/train_v64_staged.py"
)
staged = importlib.util.module_from_spec(spec)
spec.loader.exec_module(staged)


def test_reference_gate_rejects_failures_missing_clips_and_nan(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files_by_subset": {"adt": ["a.npz", "b.npz"]}}))
    ref = tmp_path / "reference"
    records = ref / "metric_results"
    records.mkdir(parents=True)
    metrics = {"failures": 0, "overall": {"average_jaccard": .2, "metric_average_jaccard": .3}}
    (ref / "metrics.json").write_text(json.dumps(metrics))
    rows = [{"clip_id": "a"}, {"clip_id": "b"}]
    (records / "adt.json").write_text(json.dumps(rows))
    staged._verify_reference(ref, manifest)
    (records / "adt.json").write_text(json.dumps(rows[:1]))
    with pytest.raises(RuntimeError, match="membership"):
        staged._verify_reference(ref, manifest)
    (records / "adt.json").write_text(json.dumps(rows))
    metrics["failures"] = 1
    (ref / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(RuntimeError, match="failed"):
        staged._verify_reference(ref, manifest)
    metrics["failures"] = 0
    metrics["overall"]["metric_average_jaccard"] = float("nan")
    (ref / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(RuntimeError, match="non-finite"):
        staged._verify_reference(ref, manifest)


def test_running_phase_cannot_transition_to_finetune(tmp_path):
    (tmp_path / "training_status.json").write_text(json.dumps({"reason": "running"}))
    with pytest.raises(RuntimeError, match="not normally finished"):
        staged._best_checkpoint(tmp_path)
