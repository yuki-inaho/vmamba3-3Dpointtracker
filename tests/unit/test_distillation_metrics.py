"""The predeclared AJ gate must reject missing clips and subset regressions."""

import pytest


def rows(delta):
    return [
        dict(
            subset=s,
            filename=str(i),
            sequence=f"{s}/{i // 2}",
            teacher={"average_jaccard": 0.3, "metric_average_jaccard": 0.2},
            student={
                "average_jaccard": 0.3 - delta[s],
                "metric_average_jaccard": 0.2 - delta[s],
            },
        )
        for s in ("adt", "drivetrack", "pstudio")
        for i in range(4)
    ]


def gate(data):
    from mamba3_tracker.eval.distillation_metrics import paired_acceptance

    return paired_acceptance(
        data,
        {(s, str(i)) for s in ("adt", "drivetrack", "pstudio") for i in range(4)},
        bootstrap_samples=1000,
    )


def test_constant_paired_difference_has_exact_interval():
    result = gate(rows(dict(adt=0.002, drivetrack=0.002, pstudio=0.002)))
    assert result["passed"]
    assert result["metrics"]["average_jaccard"]["macro"]["upper95"] == pytest.approx(
        0.002
    )


def test_subset_degradation_cannot_hide_in_macro():
    result = gate(rows(dict(adt=0.006, drivetrack=0, pstudio=0)))
    assert not result["passed"]
    assert "adt.average_jaccard" in result["violations"]


def test_missing_or_duplicate_clip_rejected():
    data = rows(dict(adt=0, drivetrack=0, pstudio=0))
    with pytest.raises(ValueError):
        gate(data[:-1])
    with pytest.raises(ValueError):
        gate(data + [data[0]])


def test_nonfinite_and_wrong_scale_rejected():
    for value in (float("nan"), -0.1, 30):
        data = rows(dict(adt=0, drivetrack=0, pstudio=0))
        data[0]["student"]["average_jaccard"] = value
        with pytest.raises(ValueError, match="AJ"):
            gate(data)


def test_partial_is_not_full_accuracy(monkeypatch):
    from pathlib import Path
    from scripts import evaluate_student_accuracy as evaluator

    monkeypatch.setattr(
        evaluator, "expected_paths", lambda cfg, part: [Path("adt/a.npz")]
    )
    cfg = {"data": {"split_sha256": "fixed", "dino_revision": "pin"}}
    identity: dict = dict(
        partition="monitor",
        split_sha256="fixed",
        frame_scope="all",
        query_scope="all",
        visibility="flow",
        dino_revision="pin",
        image_size=896,
        dino_image_size=448,
        dino_model="facebook/dinov3-vits16-pretrain-lvd1689m",
        dino_precision="fp32",
        flow_scale=-1,
        flow_iters=4,
        fb_alpha=0.05,
        fb_beta=1.0,
        waft_sha256="9f4b24f48b3937eca690a12b73bc3190effde6d4d4c87db01998fe63d846397f",
    )
    manifest: dict = dict(
        status="partial",
        identity=identity,
        records=[dict(subset="adt", filename="a.npz")],
    )
    with pytest.raises(ValueError, match="Incomplete"):
        evaluator.validate_inputs(manifest, cfg, "monitor")
    manifest["status"] = "complete"
    assert evaluator.validate_inputs(manifest, cfg, "monitor") == {("adt", "a.npz")}
    identity["image_size"] = 448
    with pytest.raises(ValueError, match="frontend"):
        evaluator.validate_inputs(manifest, cfg, "monitor")
    identity["image_size"] = 896
    manifest["records"].append(manifest["records"][0])
    with pytest.raises(ValueError):
        evaluator.validate_inputs(manifest, cfg, "monitor")
