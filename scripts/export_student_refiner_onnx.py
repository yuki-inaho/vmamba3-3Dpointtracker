"""Export a strict VSSD student and exercise the dynamic FP32 ONNX contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import onnx
import torch
import yaml

from mamba3_tracker.deployment.checkpoint import read_checkpoint, sha256
from mamba3_tracker.deployment.student_checkpoint import (
    create_student,
    load_student,
    save_student,
    validate_student,
)
from mamba3_tracker.deployment.student_runtime import run_student_chunked
from mamba3_tracker.deployment.runtime import INPUT_NAMES, OUTPUT_NAMES, session_for
from mamba3_tracker.model.student_refiner import StudentRefiner

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--init-config", type=Path)
    source.add_argument("--ckpt", type=Path)
    source.add_argument("--selection-manifest", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    return p


def export_graph(model: StudentRefiner, path: Path) -> None:
    from scripts.export_refiner_onnx import synthetic_inputs

    validate_student(model)
    if model.architecture not in (
        "vssd_two_pool_128x2_v1",
        "vssd_local_global_128x2_v2",
    ):
        raise ValueError("Export requires an explicitly fused deployment architecture")
    axes = {
        name: {0: "batch", 1: "frames", 2: "tracks"}
        for name in INPUT_NAMES[:4] + OUTPUT_NAMES
    }
    axes.update(
        depth_map={0: "batch", 1: "frames", 2: "depth_height", 3: "depth_width"},
        dino_features={0: "batch", 1: "frames"},
        intrinsics={0: "batch"},
    )
    with torch.inference_mode():
        torch.onnx.export(
            model,
            synthetic_inputs(2, 8, 4, 384, 896),
            str(path),
            input_names=list(INPUT_NAMES),
            output_names=list(OUTPUT_NAMES),
            dynamic_axes=axes,
            opset_version=18,
            dynamo=False,
            external_data=False,
        )
    graph = onnx.load(path)
    onnx.helper.set_model_props(
        graph,
        {
            "architecture": model.architecture,
            "scope": "external-feature refiner only",
            "precision": "FP32",
            "z_ref": "lower median of full B/F/N input plus 1e-6",
            "auxiliary_heads": "removed",
        },
    )
    onnx.checker.check_model(graph, full_check=True)
    if any(node.domain for node in graph.graph.node):
        raise ValueError("Custom ONNX operators are forbidden")
    if any(
        t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer
    ):
        raise ValueError("External ONNX tensor storage is forbidden")
    onnx.save(graph, path)


def main() -> None:
    from scripts.export_refiner_onnx import (
        as_feed,
        compare,
        run_session,
        synthetic_inputs,
    )

    args = parser().parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite export: {args.out_dir}")
    source_hash = None
    if args.init_config:
        cfg = yaml.safe_load(args.init_config.read_text())
        teacher = read_checkpoint(
            Path(cfg["teacher"]["checkpoint"]), cfg["teacher"]["sha256"]
        )
        model = create_student(cfg["architecture"]).eval()
        model.initialize_common(teacher["model"])
        status = "untrained_export_probe"
        source_hash = sha256(args.init_config)
    else:
        expected = None
        if args.selection_manifest:
            selected = json.loads(args.selection_manifest.read_text())
            args.ckpt = Path(selected["checkpoint"])
            expected = selected["sha256"]
        model = load_student(args.ckpt, expected)
        source_hash = sha256(args.ckpt)
        status = "trained_checkpoint_export"
    args.out_dir.mkdir(parents=True)
    report_path = args.out_dir / "export_report.json"
    report: dict[str, Any] = dict(
        success=False, status="started", source_status=status, source_sha256=source_hash
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if model.architecture == "vssd_local_global_128x2_v2_train":
        from mamba3_tracker.model.local_global_refiner import LocalGlobalStudentRefiner
        from scripts.export_refiner_onnx import synthetic_inputs

        source_model = model
        if not isinstance(source_model, LocalGlobalStudentRefiner):
            raise ValueError("Training architecture must have the exact v2 model class")
        model = source_model.to_deploy().float().eval()
        fusion_checks = []
        with torch.inference_mode():
            for b, f, n in ((1, 1, 1), (2, 8, 7), (1, 31, 7), (1, 600, 1)):
                values = synthetic_inputs(b, f, n, 384, 896)
                original, fused = source_model(*values), model(*values)
                errors = []
                for left, right in zip(original, fused, strict=True):
                    torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
                    errors.append(float((left - right).abs().max()))
                fusion_checks.append(
                    {"shape": [b, f, n], "max_absolute_errors": errors}
                )
        report["fusion_checks"] = fusion_checks
        report["train_parameters"] = sum(p.numel() for p in source_model.parameters())
        report["source_architecture"] = source_model.architecture
        del source_model
    else:
        report["fusion_checks"] = []
    report["architecture"] = model.architecture
    report["source_code_sha256"] = {
        name: sha256(ROOT / name)
        for name in (
            "scripts/export_student_refiner_onnx.py",
            "src/mamba3_tracker/model/student_refiner.py",
            "src/mamba3_tracker/model/compact_vssd.py",
            "src/mamba3_tracker/model/rep_temporal.py",
            "src/mamba3_tracker/model/local_global_refiner.py",
            "src/mamba3_tracker/model/onnx_refiner.py",
        )
    }
    pt = args.out_dir / "student_refiner_deploy.pt"
    path = args.out_dir / "student_refiner_fp32.onnx"
    save_student(model, pt)
    model = load_student(pt)
    export_graph(model, path)
    session = session_for(path, 4)
    checks = []
    with torch.inference_mode():
        for b, f, n in [
            (1, 1, 1),
            (2, 8, 7),
            (1, 31, 32),
            (1, 128, 7),
            (1, 257, 1),
            (1, 300, 7),
            (1, 600, 32),
        ]:
            inputs = synthetic_inputs(b, f, n, 384, 896, depth_shape=(17, 21))
            errors = compare(model(*inputs), run_session(session, inputs))
            checks.append(dict(shape=[b, f, n], max_absolute_errors=errors))
            print(json.dumps(checks[-1]), flush=True)
        inputs = list(synthetic_inputs(1, 31, 7, 384, 896))
        inputs[2].zero_()
        checks.append(
            dict(
                all_invisible=True,
                max_absolute_errors=compare(
                    model(*inputs), run_session(session, tuple(inputs))
                ),
            )
        )
        inputs = synthetic_inputs(1, 600, 900, 384, 896)
        feed = as_feed(inputs)
        observed = run_student_chunked(session, feed, track_chunk=32)
        # Independent torch chunks share the original scalar and full time span.
        parts = [
            model(*(a[:, :, i : i + 32] if j < 4 else a for j, a in enumerate(inputs)))
            for i in range(0, 900, 32)
        ]
        expected = [torch.cat([part[j] for part in parts], dim=2) for j in range(4)]
        checks.append(
            dict(
                shape=[1, 600, 900],
                chunk=32,
                max_absolute_errors=compare(expected, observed),
            )
        )
    count = sum(p.numel() for p in model.parameters())
    if count > 650000:
        raise ValueError("Student exceeds parameter gate")
    report.update(
        success=True,
        status="complete",
        checks=checks,
        deploy_parameters=count,
        pt_bytes=pt.stat().st_size,
        onnx_bytes=path.stat().st_size,
        pt_sha256=sha256(pt),
        onnx_sha256=sha256(path),
        numerical_gate={"rtol": 2e-4, "atol": 5e-5},
        task_accuracy_evaluated=False,
        cuda_evaluated=False,
    )
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.out_dir / "deployment_manifest.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
