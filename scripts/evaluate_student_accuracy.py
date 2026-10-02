"""Full-set teacher/student evaluation over hash-verified shared frontend inputs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mamba3_tracker.deployment.checkpoint import sha256
from mamba3_tracker.deployment.metrics import paired_scores
from mamba3_tracker.deployment.runtime import INPUT_NAMES, session_for
from mamba3_tracker.deployment.student_checkpoint import (
    load_native_teacher,
    load_student,
)
from mamba3_tracker.deployment.student_runtime import run_student_chunked
from mamba3_tracker.eval.distillation_metrics import METRICS, SUBSETS, paired_acceptance
from scripts.prepare_refiner_kd import atomic_json, read_config
from scripts.prepare_student_evaluation import expected_paths


def validate_inputs(manifest, cfg, partition):
    identity = manifest["identity"]
    frontend = dict(
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
    if any(identity.get(key) != value for key, value in frontend.items()):
        raise ValueError("Evaluation frontend differs from the fixed protocol")
    expected = {(p.parent.name, p.name) for p in expected_paths(cfg, partition)}
    observed = [(r["subset"], r["filename"]) for r in manifest["records"]]
    if (
        manifest["status"] != "complete"
        or identity["partition"] != partition
        or identity["split_sha256"] != cfg["data"]["split_sha256"]
        or len(observed) != len(set(observed))
        or set(observed) != expected
        or identity["frame_scope"] != "all"
        or identity["query_scope"] != "all"
        or identity["visibility"] != "flow"
        or identity["dino_revision"] != cfg["data"]["dino_revision"]
    ):
        raise ValueError(
            "Incomplete, wrong-partition or incompatible evaluation inputs"
        )
    return expected


@torch.no_grad()
def torch_chunked(model, inputs, chunk=32):
    parts = []
    for start in range(0, inputs["ray"].shape[2], chunk):
        values = [
            inputs[name][:, :, start : start + chunk] if i < 4 else inputs[name]
            for i, name in enumerate(INPUT_NAMES)
        ]
        parts.append([v.cpu().numpy() for v in model(*values)])
    return [np.concatenate([p[i] for p in parts], axis=2) for i in range(4)]


def evaluate(args):
    from scripts.export_refiner_onnx import compare

    cfg = read_config(args.config)
    if args.evaluation_inputs is None:
        raise ValueError("Full-frame evaluation requires --evaluation-inputs")
    manifest = json.loads(args.evaluation_inputs.read_text())
    if manifest["identity"]["config_sha256"] != sha256(args.config):
        raise ValueError("Evaluation inputs/config hash mismatch")
    partition = "monitor" if args.mode == "monitor" else "minival"
    expected = validate_inputs(manifest, cfg, partition)
    checkpoint = args.checkpoint or args.student_dir / "student_refiner_deploy.pt"
    if args.mode == "accuracy" and args.checkpoint is not None:
        raise ValueError("Final accuracy must use the exported deployment checkpoint")
    student = load_student(checkpoint).cuda().eval()
    teacher = (
        load_native_teacher(
            Path(cfg["teacher"]["checkpoint"]), cfg["teacher"]["sha256"]
        )
        .cuda()
        .eval()
    )
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    session = None
    onnx_path = args.student_dir / "student_refiner_fp32.onnx"
    if args.mode == "accuracy":
        if args.provider != "cpu":
            raise ValueError(
                "Native teacher accuracy runs in root CPU-ORT environment; CUDA replay is separate"
            )
        deployment = json.loads(
            (args.student_dir / "deployment_manifest.json").read_text()
        )
        # Export manifest schema is checked by the exporter; the actual artifact
        # hashes below pin precisely what this run evaluates.
        if not deployment:
            raise ValueError("Missing deployment provenance")
        session = session_for(onnx_path)
    identity = dict(
        checkpoint_sha256=sha256(checkpoint),
        teacher_sha256=cfg["teacher"]["sha256"],
        inputs_sha256=sha256(args.evaluation_inputs),
        config_sha256=sha256(args.config),
        partition=partition,
        onnx_sha256=sha256(onnx_path) if session else None,
    )
    report_path = args.out_dir / "accuracy_report.json"
    report: dict = dict(
        identity=identity,
        status="running",
        rows=[],
        expected=len(expected),
        scope="known_sequence_overlap_not_blind",
        student_execution="onnx_cpu" if session else "torch_fp32",
    )
    if report_path.exists():
        prior = json.loads(report_path.read_text())
        if prior["identity"] != identity:
            raise ValueError(
                "Evaluation output belongs to a different checkpoint/input set"
            )
        report["rows"] = prior["rows"]
    done = {(r["subset"], r["filename"]) for r in report["rows"]}
    if len(done) != len(report["rows"]) or not done <= expected:
        raise ValueError("Corrupt resume evaluation membership")
    atomic_json(report_path, report)
    for row in manifest["records"]:
        if (row["subset"], row["filename"]) in done:
            continue
        path = Path(row["path"])
        if sha256(path) != row["sha256"]:
            raise ValueError("Full-frame input artifact hash mismatch")
        payload = torch.load(path, weights_only=True, map_location="cpu")
        inputs = payload["inputs"]
        if (
            list(inputs["ray"].shape[1:3]) != [row["frames"], row["tracks"]]
            or payload["gt"].shape[:2] != inputs["ray"].shape[1:3]
        ):
            raise ValueError("Truncated evaluation input or GT")
        cuda_inputs = {k: v.cuda().float() for k, v in inputs.items()}
        native = torch_chunked(teacher, cuda_inputs)
        predicted = torch_chunked(student, cuda_inputs)
        deployed = predicted
        errors = None
        if session:
            feed = {k: v.numpy() for k, v in inputs.items()}
            deployed = run_student_chunked(session, feed, 32)
            errors = compare([torch.from_numpy(v) for v in predicted], deployed)
        h, w = payload["image_hw"]
        clip = SimpleNamespace(
            K=payload["K"],
            images=torch.empty(0, 3, h, w),
            tracks_XYZ=payload["gt"],
            visibility=payload["visible"],
        )
        visible = inputs["visibility"][0].numpy().T
        t_score, s_score = paired_scores(
            clip,
            native[0][0].transpose(1, 0, 2),
            deployed[0][0].transpose(1, 0, 2),
            visible,
        )
        result = {
            key: row[key]
            for key in ("subset", "filename", "sequence", "frames", "tracks")
        }
        result.update(teacher=t_score, student=s_score, student_onnx_error=errors)
        predictions_path = args.out_dir / (
            row["subset"] + "_" + Path(row["filename"]).stem + "_predictions.npz"
        )
        np.savez_compressed(
            predictions_path,
            teacher_xyz=native[0],
            student_xyz=deployed[0],
            flow_visibility=visible,
        )
        result["predictions_sha256"] = sha256(predictions_path)
        report["rows"].append(result)
        atomic_json(report_path, report)
        print(
            json.dumps(
                {
                    "evaluated": len(report["rows"]),
                    "expected": len(expected),
                    "clip": row["filename"],
                    "student_metric_aj": s_score["metric_average_jaccard"],
                }
            ),
            flush=True,
        )
        del payload, inputs, cuda_inputs, native, predicted, deployed
        torch.cuda.empty_cache()
    if {(r["subset"], r["filename"]) for r in report["rows"]} != expected:
        raise ValueError("Incomplete evaluation cannot be reported as successful")
    report["summary"] = {}
    for source in ("teacher", "student"):
        per_subset = {
            s: {
                m: float(
                    np.mean([r[source][m] for r in report["rows"] if r["subset"] == s])
                )
                for m in METRICS
            }
            for s in SUBSETS
        }
        report["summary"][source] = {
            "per_subset": per_subset,
            "macro": {
                m: float(np.mean([per_subset[s][m] for s in SUBSETS])) for m in METRICS
            },
        }
    if partition == "minival":
        report["acceptance"] = paired_acceptance(report["rows"], expected)
    report["status"] = "complete"
    atomic_json(report_path, report)
