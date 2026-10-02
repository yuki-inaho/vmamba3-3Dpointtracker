"""Package only explicitly whitelisted public KD assets; never bundle caches/secrets."""

from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "refiner_kd_v2_best180_20261002"
OUT = ROOT / "result/refiner_kd_improved_20261002/release_R2_best180"
EXPORT = ROOT / "result/refiner_kd_improved_20261002/export_R2_best180_release"
TRAIN = ROOT / "result/refiner_kd_improved_20261002/pilot_R2_fullcache_s42"
GPU = ROOT / "result/refiner_kd_improved_20261002/gpu_R2_best180_release_fixtures"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite public package: {OUT}")
    export = json.loads((EXPORT / "export_report.json").read_text())
    gpu = json.loads((GPU / "gpu_report.json").read_text())
    status = json.loads((TRAIN / "training_status.json").read_text())
    accuracy = json.loads(
        (
            ROOT
            / "result/refiner_kd_improved_20261002/monitor_full_pilot_R2_s42/accuracy_report.json"
        ).read_text()
    )
    if not (
        export["success"]
        and gpu["success"]
        and status["status"] == "complete"
        and accuracy["status"] == "complete"
        and len(accuracy["rows"]) == 15
    ):
        raise ValueError("Incomplete export/GPU/training/monitor evidence")
    if (
        gpu["onnx_sha256"] != export["onnx_sha256"]
        or digest(Path(status["best_checkpoint"])) != export["source_sha256"]
    ):
        raise ValueError("Export, GPU and training model provenance mismatch")
    OUT.mkdir(parents=True)
    copies = {
        f"{PREFIX}.pt": EXPORT / "student_refiner_deploy.pt",
        f"{PREFIX}.onnx": EXPORT / "student_refiner_fp32.onnx",
        f"{PREFIX}_train.pt": Path(status["best_checkpoint"]),
        f"{PREFIX}_export_report.json": EXPORT / "export_report.json",
        f"{PREFIX}_gpu_report.json": GPU / "gpu_report.json",
    }
    for name, source in copies.items():
        shutil.copy2(source, OUT / name)
    metric_names = list(accuracy["rows"][0]["teacher"])
    means = {
        method: {
            name: sum(row[method][name] for row in accuracy["rows"]) / 15
            for name in metric_names
        }
        for method in ("teacher", "student")
    }
    metrics = dict(
        status="complete",
        scope="known_validation15_all_frames_all_queries_not_blind",
        subset_counts={
            subset: sum(row["subset"] == subset for row in accuracy["rows"])
            for subset in ("pstudio", "drivetrack", "adt")
        },
        averaging="mean of 5 clips per subset, then equally weighted 3 subsets",
        summary=means,
        per_subset=accuracy["summary"],
        rows=accuracy["rows"],
        evaluation_identity=accuracy["identity"],
        train_parameters=export["train_parameters"],
        deploy_parameters=export["deploy_parameters"],
        training=dict(
            block_steps=100,
            finetune_steps=status["ft_step"],
            best_step=180,
            best_monitor_gt_loss=status["best_monitor_gt_loss"],
            teacher_same_monitor_gt_loss=0.19620265004535517,
            early_stopping_enabled=status["identity"]["early_stopping_enabled"],
            exit_reason=status["exit_reason"],
            teacher_sha256=status["identity"]["teacher"],
            source_sha256=status["identity"]["source_sha256"],
        ),
        official_minival150_evaluated=False,
        pointodyssey_evaluated=False,
        pointodyssey_delta_avg_mte_survival="not evaluated",
        student_onnx_task_accuracy150_evaluated=False,
        export_sha256=digest(EXPORT / "export_report.json"),
        gpu_report_sha256=digest(GPU / "gpu_report.json"),
    )
    dump(OUT / f"{PREFIX}_metrics.json", metrics)
    source_paths = sorted((ROOT / "src/mamba3_tracker").rglob("*.py"))
    source_paths += sorted(
        (ROOT / "third_party/visionMamba3/src/visionmamba3").rglob("*.py")
    )
    source_paths += [
        ROOT / p
        for p in (
            "pyproject.toml",
            "uv.lock",
            "LICENSE",
            "scripts/export_student_refiner_onnx.py",
            "scripts/export_refiner_onnx.py",
            "scripts/evaluate_student_refiner.py",
            "scripts/verify_refiner_gpu.py",
            "scripts/prepare_refiner_kd.py",
            "scripts/train_refiner_staged.py",
            "scripts/train_refiner_kd.py",
            "configs/distill_refiner_local_global.yaml",
            "docs/kd_accuracy_research_20261002.md",
        )
    ]
    for license_path in (ROOT / "third_party/visionMamba3/LICENSE",):
        if license_path.is_file():
            source_paths.append(license_path)
    source_paths = sorted(set(source_paths))
    source_manifest = {
        str(path.relative_to(ROOT)): digest(path) for path in source_paths
    }
    archive = OUT / f"{PREFIX}_source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in source_paths:
            bundle.write(path, str(path.relative_to(ROOT)))
        bundle.writestr(
            "KD_SOURCE_MANIFEST.json", json.dumps(source_manifest, indent=2)
        )
    card = f"""# Experimental lightweight refiner — KD v2 best180 (2026-10-02)

This is an additional experimental asset, not a replacement for the recommended teacher best80.
Copy-fused deployment: **{export["deploy_parameters"]:,} parameters**; train form: {export["train_parameters"]:,}.
FP32 deployment PT: {export["pt_bytes"]:,} bytes; ONNX: {export["onnx_bytes"]:,} bytes.
Selected best finetune step180 after block-alignment100 + finetune200, seed42, DA off.
Early stopping enabled (validation every10, patience5, min_delta0.001); the200-step ceiling was reached.

## Artifacts

- `{PREFIX}.onnx`: standard-op opset18, single-file FP32 refiner, fused temporal convolution; portable preferred inference format.
- `{PREFIX}.pt`: strict deployment checkpoint, no optimizer, teacher, gated DINO backbone or training-only auxiliary heads.
- `{PREFIX}_train.pt`: original public best180 checkpoint for the four-column demo, NOT an optimizer/resume checkpoint.
- `{PREFIX}_source.zip`: explicit Python/source snapshot needed to load this new architecture; the historical Release tag does not contain this KD implementation. No pretrained frontend weights, raw data, local tokens or caches are bundled.
- Companion export/GPU reports, metrics JSON and SHA256SUMS.

## Measured results and limitations

Evaluation uses **15 known validation clips** (5 each ADT/DriveTrack/PStudio), all frames/queries, shared FP32 frozen frontend inputs and the same flow visibility. This is NOT independent-test or official150-clip minival evaluation.
Numbers are proportions in [0,1]; multiply by100 for percentage presentation.

| Metric | Teacher best80 | Lightweight best180 |
| :--- | ---: | ---: |
| Median-scaled 3D-AJ | {means["teacher"]["average_jaccard"]:.8f} | {means["student"]["average_jaccard"]:.8f} |
| Median-scaled APD3D | {means["teacher"]["average_pts_within_thresh"]:.8f} | {means["student"]["average_pts_within_thresh"]:.8f} |
| Absolute metric-AJ | {means["teacher"]["metric_average_jaccard"]:.8f} | {means["student"]["metric_average_jaccard"]:.8f} |
| Absolute metric-APD3D | {means["teacher"]["metric_average_pts_within_thresh"]:.8f} | {means["student"]["metric_average_pts_within_thresh"]:.8f} |
| Occlusion accuracy | {means["teacher"]["occlusion_accuracy"]:.8f} | {means["student"]["occlusion_accuracy"]:.8f} |

Scale-normalized tracking structure is close, **but absolute metric accuracy is worse**; accuracy-preserving compression is NOT established. OA is identical because the flow visibility is shared, not because the student visibility head improved. Teacher mixer uses native BF16; student uses FP32. Results above are the UNFUSED PyTorch student; no full-minival deployed-model task accuracy is claimed.
Student loss0.2924778565 vs teacher0.1962026500 uses the same strict16-frame monitor loss; it is not comparable to earlier Release training losses.

## ONNX numerical checks

Copy-fusion parity and dynamic CPU ONNX cases passed, including temporal lengths1/8/31/128/257/300/600 and track chunking up to900 tracks. CUDAExecutionProvider checks passed on {len(gpu["cases"])} cases including3 real monitor clips, permutation and chunk1. Major floating computation was verified on CUDA; CPU shape/index nodes contain only integer tensors. These prove numerical/graph correctness, NOT accuracy parity with the teacher or end-to-end RGB ONNX.

## Inference contract

Inputs in order: ray, z_raw, visibility, uv, depth_map, dino_features, intrinsics, z_ref.
Outputs: xyz, uv_refined, vis_logits, delta_uv. Keep the full temporal axis; chunk only tracks.
Compute z_ref as the full B/F/N lower median z_raw plus1e-6 BEFORE chunking; do not recompute per chunk.
DINOv3 ViT-S/16 FP32 features are [B,F,384,28,28]; image coordinates use896x896. WAFT, DA3 metric-large and authorized DINOv3 weights remain EXTERNAL frozen prerequisites. This is not an RGB-to-tracks ONNX graph.

For standalone ONNX use onnxruntime or onnxruntime-gpu with these named FP32 inputs. The included `student_runtime.py` preserves track independence and full temporal context. GPU execution uses CUDAExecutionProvider (use_tf32=0); do not allow silent provider fallback.
For PyTorch, use the included source snapshot and `load_student(path, expected_sha256)`; do not attempt to load these weights with the old teacher architecture.

## Benchmark scope

PointOdyssey itself and its2D delta_avg/MTE/Survival have NOT been evaluated. Our current APD3D/AJ are TAPVid-3D measures, not PointOdyssey pixel metrics. The original tracker paper uses official TAPVid-3D minival150, so its published0.256 metric-AJ is not directly comparable to this known-validation15 experiment.
Primary sources: https://arxiv.org/html/2307.15055v1#S5.SS2 and https://arxiv.org/html/2609.34035v1#S4.SS2.
"""
    (OUT / f"{PREFIX}_MODEL_CARD.md").write_text(card)
    assets = sorted(OUT.iterdir())
    sums = "".join(f"{digest(path)}  {path.name}\n" for path in assets)
    (OUT / f"{PREFIX}_SHA256SUMS").write_text(sums)
    dump(
        ROOT
        / "result/refiner_kd_improved_20261002/release_R2_best180_package_inventory.json",
        {
            "assets": [
                {"name": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
                for p in sorted(OUT.iterdir())
            ],
            "source_files": source_manifest,
        },
    )
    print(
        json.dumps(
            {
                "out_dir": str(OUT),
                "assets": len(list(OUT.iterdir())),
                "deploy_parameters": export["deploy_parameters"],
            }
        )
    )


if __name__ == "__main__":
    main()
