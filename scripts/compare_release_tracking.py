"""Desktop release/student comparison on a predeclared validation manifest.

This is not the fixed-nine acceptance protocol or the 150-clip official minival.
Stages separate GPU prediction, CPU rendering, and independent media auditing.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from mamba3_tracker.deployment.checkpoint import (
    load_native_state,
    read_checkpoint,
    resolve_flow_config,
    sha256,
)
from mamba3_tracker.deployment.metrics import error_distribution, score
from mamba3_tracker.viz.comparison import render_comparison

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT.parent / "tracking_comparison_20261002"
DEFAULT_STUDENT_OUTPUT = ROOT.parent / "tracking_comparison_kd_20261002"
METHODS = ("old", "new", "onnx")


def comparison_methods(manifest: dict[str, Any]) -> tuple[str, ...]:
    return (*METHODS, "student") if "student_checkpoint" in manifest else METHODS


def prepare_student_manifest(
    output: Path, base: Path, checkpoint: Path, digest: str | None
):
    if output == base:
        raise ValueError("Student mode requires a separate output directory")
    manifest = read_manifest(base)
    if "student_checkpoint" in manifest:
        raise ValueError("Base directory must contain the original release comparison")
    checkpoint = checkpoint.resolve(strict=True)
    actual = sha256(checkpoint)
    if digest is not None and digest != actual:
        raise ValueError("Student checkpoint SHA256 mismatch")
    from mamba3_tracker.deployment.student_checkpoint import load_student

    student = load_student(checkpoint, actual)
    manifest.update(
        comparison_mode="with-student",
        student_checkpoint=str(checkpoint),
        student_sha256=actual,
        student_architecture=student.architecture,
        student_parameters=sum(p.numel() for p in student.parameters()),
        base_manifest_sha256=sha256(base / "manifest.json"),
    )
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError(
            "Output belongs to another student/model/input; choose a new --out-dir"
        )
    if not path.exists():
        write_json(path, manifest)
    return manifest


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def read_manifest(output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "manifest.json").read_text())
    pool = json.loads(Path(manifest["pool"]).read_text())
    if len(manifest["clips"]) != 3 or {c["subset"] for c in manifest["clips"]} != {
        "pstudio",
        "drivetrack",
        "adt",
    }:
        raise ValueError("Expected exactly one fixed clip per subset")
    for clip in manifest["clips"]:
        subset, path = clip["subset"], clip["path"]
        first = next(p for p in pool["validation"] if Path(p).parent.name == subset)
        if path != first or path in pool["train"]:
            raise ValueError("Clip violates the prespecified validation selection")
        depth = Path(manifest["depth_root"]) / subset / Path(path).name
        if (
            sha256(Path(path)) != clip["raw_sha256"]
            or sha256(depth) != clip["depth_sha256"]
        ):
            raise ValueError("Raw/depth provenance changed")
    for key, digest in (
        ("old_checkpoint", "old_sha256"),
        ("new_checkpoint", "new_sha256"),
        ("onnx", "onnx_sha256"),
    ):
        if sha256(ROOT / manifest[key]) != manifest[digest]:
            raise ValueError(f"Model SHA256 mismatch: {key}")
    if (
        "student_checkpoint" in manifest
        and sha256(Path(manifest["student_checkpoint"])) != manifest["student_sha256"]
    ):
        raise ValueError("Student checkpoint SHA256 mismatch")
    return manifest


def build_models(manifest: dict[str, Any]):
    from eval_onnx_refiner import PairedOnnxRefiner

    from mamba3_tracker.data.frozen_cache import module_fingerprint
    from mamba3_tracker.model.depth_refined_tracker import Mamba3V35Refiner

    states = {
        key: read_checkpoint(
            ROOT / manifest[f"{key}_checkpoint"], manifest[f"{key}_sha256"]
        )
        for key in ("old", "new")
    }
    if states["old"]["step"] != 200 or states["new"]["step"] != 80:
        raise ValueError("Unexpected release checkpoint steps")
    if any(state.get("weights_mode") != "eval" for state in states.values()):
        raise ValueError("Only eval-mode release weights are allowed")
    config = states["new"]["cfg"]["model"]
    # The old artifact retains a training-time initialization record, which
    # Mamba3V35Refiner does not accept or read during inference. Do not ignore
    # any architectural fields; supplied checkpoint tensors are still strict.
    old_runtime_config = {
        key: value
        for key, value in states["old"]["cfg"]["model"].items()
        if key != "pretrained_mamba3"
    }
    new_runtime_config = {
        key: value for key, value in config.items() if key != "pretrained_mamba3"
    }
    if new_runtime_config != old_runtime_config:
        raise ValueError(
            "Architecture/frontend config differs; sharing would be invalid"
        )
    run_config = json.loads((ROOT / manifest["run_config"]).read_text())
    flow_config = resolve_flow_config(states["new"], run_config)
    if resolve_flow_config(states["old"], run_config) != flow_config:
        raise ValueError("Release flow settings differ")
    arguments = {
        key: value
        for key, value in config.items()
        if key in inspect.signature(Mamba3V35Refiner).parameters
    }
    arguments["dino_revision"] = manifest["dino_revision"]
    models = {}
    audit: dict[str, Any] = {
        "ignored_training_only_config_fields": ["pretrained_mamba3"]
    }
    for key in ("old", "new"):
        model = Mamba3V35Refiner(**arguments)
        audit[key] = load_native_state(model, states[key])
        revision = getattr(
            getattr(model.dino.backbone, "config", None), "_commit_hash", None
        )
        if revision != manifest["dino_revision"]:
            raise ValueError("Loaded DINO revision mismatch")
        models[key] = model
    old_dino, new_dino = (
        models["old"].dino.state_dict(),
        models["new"].dino.state_dict(),
    )
    if set(old_dino) != set(new_dino) or not all(
        torch.equal(old_dino[name], new_dino[name]) for name in old_dino
    ):
        raise ValueError("Old/new frozen frontend tensors differ")
    audit["shared_dino_fingerprint"] = module_fingerprint(models["new"].dino.backbone)
    models["old"].dino = models["new"].dino
    old, new = (models[key].cuda().eval() for key in ("old", "new"))
    paired = PairedOnnxRefiner(
        new,
        ROOT / manifest["onnx"],
        manifest["onnx_track_chunk"],
        manifest["threads"],
        manifest["onnx_memory_budget_mib"],
    ).eval()
    metadata = paired.session.get_modelmeta().custom_metadata_map
    if metadata.get("source_checkpoint_sha256") != manifest["new_sha256"]:
        raise ValueError("ONNX source checkpoint provenance mismatch")
    if (
        metadata.get("dino_revision") != manifest["dino_revision"]
        or json.loads(metadata["model_config"]) != config
    ):
        raise ValueError("ONNX config/DINO revision mismatch")
    audit["onnx_metadata_verified"] = True
    return old, paired, flow_config, audit


class ThreeWayRefiner(nn.Module):
    """Single set of frontend arguments; one DINO extraction; no independent masks."""

    def __init__(self, old, paired, student=None):
        super().__init__()
        self.old, self.paired = old, paired
        self.student = student
        self.outputs: dict[str, list[np.ndarray]] = {}
        self.inputs: dict[str, Any] = {}
        self.dino_calls = 0

    @torch.no_grad()
    def forward(self, ray, z_raw, vis, uv, depth_map, images, k):
        forward_video = self.paired.native.dino.forward_video
        features = []

        def cached(video):
            if not features:
                features.extend(forward_video(video))
                self.dino_calls += 1
            return features

        # Both native encoders are the same frozen module, checked bitwise above.
        with patch.object(self.paired.native.dino, "forward_video", side_effect=cached):
            onnx_output = self.paired(ray, z_raw, vis, uv, depth_map, images, k)
            old_output = self.old(ray, z_raw, vis, uv, depth_map, images, k)
        self.outputs = {
            "old": [
                value.detach().float().cpu().numpy()
                for value in (
                    old_output.xyz,
                    old_output.uv,
                    old_output.vis_logits,
                    old_output.delta_uv,
                )
            ],
            "new": self.paired.last_native,
            "onnx": self.paired.last_onnx,
        }
        if self.student is not None:
            from mamba3_tracker.model.onnx_refiner import reference_depth

            values = (
                ray,
                z_raw,
                vis,
                uv,
                depth_map,
                features[0],
                k,
                reference_depth(z_raw),
            )
            parts = []
            for start in range(0, ray.shape[2], 32):
                chunk = [
                    v[:, :, start : start + 32] if i < 4 else v
                    for i, v in enumerate(values)
                ]
                parts.append(
                    [v.detach().float().cpu().numpy() for v in self.student(*chunk)]
                )
            self.outputs["student"] = [
                np.concatenate([p[i] for p in parts], axis=2) for i in range(4)
            ]
        for name, value in zip(
            ("ray", "z_raw", "flow_visibility", "uv", "depth", "images", "K", "dino"),
            (ray, z_raw, vis, uv, depth_map, images, k, features[0]),
            strict=True,
        ):
            array = value.detach().float().cpu().numpy()
            self.inputs[name] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        if self.dino_calls != 1:
            raise RuntimeError("DINO must be computed once per clip")
        return onnx_output


def infer(output: Path, manifest: dict[str, Any], subset: str) -> None:
    from eval_metric3d import _infer
    from train_depth_refined_tracker import _build_waft_flow

    from mamba3_tracker.data.tapvid3d import load_clip

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the authentic native Mamba-3 mixer")
    started = time.perf_counter()
    torch.set_num_threads(manifest["threads"])
    entry = next(c for c in manifest["clips"] if c["subset"] == subset)
    old, paired, flow_config, audit = build_models(manifest)
    student = None
    if "student_checkpoint" in manifest:
        from mamba3_tracker.deployment.student_checkpoint import load_student

        student = (
            load_student(
                Path(manifest["student_checkpoint"]), manifest["student_sha256"]
            )
            .cuda()
            .eval()
        )
        audit["student"] = {
            "architecture": student.architecture,
            "parameters": sum(p.numel() for p in student.parameters()),
            "checkpoint_sha256": manifest["student_sha256"],
        }
    combined = ThreeWayRefiner(old, paired, student).eval()
    flow = _build_waft_flow(
        torch.device("cuda"), scale=flow_config["scale"], iters=flow_config["iters"]
    )
    clip = load_clip(entry["path"])
    if (clip.F, clip.N_q) != (entry["frames"], entry["tracks"]):
        raise ValueError("Actual frame/query count does not match manifest")
    print(
        f"[infer] {subset}: all {clip.F} frames, {clip.N_q} tracks, shared frontend",
        flush=True,
    )
    _, visibility = _infer(
        method="v35",
        flow_model=flow,
        model=combined,
        clip=clip,
        image_size=896,
        fb_alpha=flow_config["fb_alpha"],
        fb_beta=flow_config["fb_beta"],
        da3_depth_root=Path(manifest["depth_root"]),
        max_frames=0,
        device=torch.device("cuda"),
        vis_source="flow",
    )
    arrays = {
        "gt": clip.tracks_XYZ.transpose(0, 1).numpy(),
        "gt_visibility": clip.visibility.transpose(0, 1).numpy(),
        "flow_visibility": visibility,
        "queries_xyt": clip.queries_xyt.numpy(),
        "K": clip.K.numpy(),
    }
    scores = {}
    for key, values in combined.outputs.items():
        arrays[f"{key}_xyz"] = values[0][0].transpose(1, 0, 2)
        arrays[f"{key}_uv896"] = values[1][0].transpose(1, 0, 2)
        arrays[f"{key}_vis_logits"] = values[2][0].T
        arrays[f"{key}_delta_uv"] = values[3][0].transpose(1, 0, 2)
        scores[key] = score(clip, arrays[f"{key}_xyz"], visibility)
        print(
            f"[score] {key}: metric-AJ={scores[key]['metric_average_jaccard']:.8f}, 3D-AJ={scores[key]['average_jaccard']:.8f}",
            flush=True,
        )
    differences = {
        name: error_distribution(
            arrays[f"new_{name}"], arrays[f"onnx_{name}"], arrays["gt_visibility"]
        )
        for name in ("xyz", "uv896")
    }
    print(f"[difference] {json.dumps(differences)}", flush=True)
    prediction = output / "predictions" / f"{subset}.npz"
    prediction.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prediction, **arrays)
    report = {
        "status": "complete",
        "scope": manifest["scope"],
        "subset": subset,
        "clip_id": clip.clip_id,
        "frames": clip.F,
        "tracks": clip.N_q,
        "scores": scores,
        "native_onnx_difference": differences,
        "shared_frontend_inputs": combined.inputs,
        "dino_extractions": combined.dino_calls,
        "visibility_source": "same binary WAFT forward-backward mask for all methods",
        "precision": {
            "native": "CUDA BF16 Mamba-3 mixer, other operations FP32",
            "onnx": "CPU FP32 refiner",
        },
        "official_full_minival_evaluated": False,
        "depth_model": "DA3 metric-large",
        "weight_loading": audit,
        "flow_config": flow_config,
        "waft_checkpoint_sha256": sha256(
            ROOT / "third_party/WAFT/ckpts/waft_a1_recommended.pth"
        ),
        "manifest_sha256": sha256(output / "manifest.json"),
        "prediction_sha256": sha256(prediction),
        "old_sha256": manifest["old_sha256"],
        "new_sha256": manifest["new_sha256"],
        "onnx_sha256": manifest["onnx_sha256"],
        "elapsed_s": time.perf_counter() - started,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }
    if student is not None:
        report["student_sha256"] = manifest["student_sha256"]
        report["precision"]["student"] = "CUDA FP32 lightweight refiner"
    write_json(output / "reports" / f"{subset}.json", report)
    print(f"[infer] saved {prediction}; {report['elapsed_s']:.1f} s", flush=True)


def load_prediction(output: Path, subset: str):
    report = json.loads((output / "reports" / f"{subset}.json").read_text())
    prediction = output / "predictions" / f"{subset}.npz"
    if report["manifest_sha256"] != sha256(output / "manifest.json") or report[
        "prediction_sha256"
    ] != sha256(prediction):
        raise ValueError("Cached prediction or manifest provenance changed")
    manifest = json.loads((output / "manifest.json").read_text())
    if set(report["scores"]) != set(comparison_methods(manifest)):
        raise ValueError("Prediction methods differ from manifest; rerun infer")
    if (
        "student_checkpoint" in manifest
        and report.get("student_sha256") != manifest["student_sha256"]
    ):
        raise ValueError("Student prediction/checkpoint provenance mismatch")
    with np.load(prediction) as data:
        arrays = {key: data[key] for key in data.files}
    return arrays, report


def render(output: Path, manifest: dict[str, Any]) -> None:
    from PIL import Image, ImageOps

    from mamba3_tracker.data.tapvid3d import load_clip

    torch.set_num_threads(manifest["threads"])
    thumbnails = []
    files = []
    for entry in manifest["clips"]:
        subset = entry["subset"]
        arrays, report = load_prediction(output, subset)
        clip = load_clip(entry["path"])
        frames = (
            (clip.images.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).numpy()
        )
        path = output / f"comparison_{subset}.mp4"
        layout = render_comparison(
            frames,
            arrays,
            report,
            path,
            fps=manifest["render"]["fps"],
            count=manifest["render"]["display_tracks"],
            tail=manifest["render"]["tail_frames"],
        )
        write_json(output / "reports" / f"{subset}_layout.json", layout)
        thumbnail = Image.open(layout["snapshots"][1])
        thumbnails.append(ImageOps.contain(thumbnail, (1280, 720)))
        files.append(path)
    overview = Image.new("RGB", (1280, 2160), "#0f172a")
    for index, thumbnail in enumerate(thumbnails):
        overview.paste(thumbnail, (0, index * 720))
    overview.save(output / "comparison_overview.png")
    listing = output / "reports" / "concat.txt"
    listing.write_text("".join(f"file '{path.as_posix()}'\n" for path in files))
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output / "comparison_all.mp4"),
        ],
        check=True,
    )


def audit(output: Path, manifest: dict[str, Any]) -> None:
    rows = []
    for entry in manifest["clips"]:
        _, row = load_prediction(output, entry["subset"])
        rows.append(row)
    files = {f"comparison_{c['subset']}.mp4": c["frames"] for c in manifest["clips"]}
    files["comparison_all.mp4"] = sum(c["frames"] for c in manifest["clips"])
    media = []
    for filename, frames in files.items():
        path = output / filename
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        info = json.loads(result.stdout)
        stream = info["streams"][0]
        if (
            stream["codec_name"],
            stream["pix_fmt"],
            stream["width"],
            stream["height"],
            stream["r_frame_rate"],
            int(stream["nb_read_frames"]),
        ) != (
            "h264",
            "yuv420p",
            640 * len(comparison_methods(manifest)),
            1080,
            f"{manifest['render']['fps']}/1",
            frames,
        ):
            raise ValueError(f"Invalid media properties: {filename}")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
            check=True,
        )
        media.append(
            {
                "filename": filename,
                "expected_frames": frames,
                "ffprobe": info,
                "full_decode": "passed",
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    summary = {
        "scope": manifest["scope"],
        "clips": rows,
        "equal_subset_mean": {
            key: {
                metric: float(np.mean([row["scores"][key][metric] for row in rows]))
                for metric in rows[0]["scores"][key]
            }
            for key in comparison_methods(manifest)
        },
        "official_full_minival_evaluated": False,
        "media": media,
        "audit_passed": True,
    }
    write_json(output / "reports" / "summary.json", summary)
    lines = [
        "# 旧Release・新Release・ONNX 追跡比較\n\n",
        "まず `comparison_all.mp4` を開くと、PStudio → DriveTrack → ADT の順で3本を確認できます。\n",
        "\n## 表示形式\n\n",
        "論文 [3D Point Tracking with State Space Models](https://arxiv.org/html/2609.34035v1#S4.SS6) の図9を参考に、予測=実線、GT=破線、同一点=同色、共有カメラ・等軸メートル目盛りを採用しました。今回の左/中央/右は旧best200 / 新best80(native) / 新best80(ONNX CPU)です。DELTAとの比較ではありません。上段は実映像へ投影した追跡、下段は各フレームのカメラ座標系XYZです。固定世界座標の軌跡ではありません。\n",
        "色付き塗りつぶし点=予測可視、色付き中空点=フローによる予測遮蔽、GTは輪郭点。過去30フレームだけを表示し、非可視区間を線で結びません。描画の32点はGT可視フレーム数順（同順位はID順）で選び、3方法に同じIDを使います。数値は描画点だけでなく全クエリ・全フレームで評価しています。3D範囲はGTと3方法の全表示点を含む共有cubeで、予測を方法別に拡大縮小しません。再生は15fps指定で、元映像の速度を保証しません。\n",
        "各3D列にはGT中心の詳細窓もあります。表示GTの可視点だけの各軸10/90百分位の中央と最大幅×1.1で固定した等軸メートルcubeで、3方法が同じ範囲を使います。詳細窓だけ範囲外の点を切り、全体窓は全表示予測を含みます。数値評価にはこの切り取りも描画点の選択も影響しません。\n",
        "\n## 同一入力での結果\n\n",
        "AJは0〜1、高いほど良好。3D-AJは公式計算器による中央値スケーリング、metric-AJはスケーリングなし・固定メートル閾値です。\n\n",
        "| 動画 | フレーム / 点数 | 旧metric-AJ | 新metric-AJ | ONNX metric-AJ | 新→ONNX XYZ差p95 |\n",
        "| :--- | ---: | ---: | ---: | ---: | ---: |\n",
    ]
    for row in rows:
        lines.append(
            f"| {row['subset']} | {row['frames']} / {row['tracks']} | {row['scores']['old']['metric_average_jaccard']:.5f} | {row['scores']['new']['metric_average_jaccard']:.5f} | {row['scores']['onnx']['metric_average_jaccard']:.5f} | {row['native_onnx_difference']['xyz']['p95']:.6f} m |\n"
        )
    lines.extend(
        [
            "\nこの3本では新モデルの絶対metric-AJは全て上昇しましたが、PStudio・DriveTrackのスケール補正後3D-AJは下がっています。全ての指標で改善したという意味ではありません。\n",
            f"\n新nativeとONNXのAJ差（3D-AJ/metric-AJ両方）の最大絶対値は {max(abs(row['scores']['new'][key] - row['scores']['onnx'][key]) for row in rows for key in ('average_jaccard', 'metric_average_jaccard')):.8f}。GT可視点のXYZ差の最大値は {max(row['native_onnx_difference']['xyz']['max'] for row in rows):.6f} m で、完全一致ではありません。\n",
            "\n## 公平性と限界\n\n",
            "3本は新学習の検証15本から各subsetの先頭1本を推論前に固定しました。1027本の重み更新用集合とは重複しませんが、検証集合はbest80のモデル選択に利用済みであり、独立テスト集合ではありません。公式minival150本でも従来Releaseの9本監視集合でもありません。過去に公表した異なる集合のloss/AJや論文の0.256とは直接比較できません。今回の旧/新モデルの比較は、ここに掲載した同一3本に限定して可能です。\n",
            "旧・新は今回のofficial Mamba-3 adapter構成で、論文図9のVSSD-2poolモデルそのものではありません。WAFT、DA3 metric-large、DINOv3 ViT-S/16、クエリ、フロー可視性を完全共有しています。GTは評価/描画選択にのみ利用し推論には渡しません。ONNXはrefinerのみ（前段は外部共有）で、end-to-end ONNX化ではありません。nativeはmixerのみCUDA BF16、ONNXはCPU FP32のためビット一致は要求しません。個別のXYZ/UV差分とAJ差をreports各JSONで確認できます。\n",
            "\n## 再現と証跡\n\n",
            "`manifest.json` に固定入力・PT/ONNX/raw/depthのSHA、DINO revision、表示設定があります。`predictions/*.npz` はGT・3方法XYZ・UV・flow可視性を保持し、`reports/*.json` は全クエリ評価、前段入力fingerprint、メディア全デコード結果、表示ID/共有軸を保持します。`comparison_overview.png` は各動画中間フレームの一覧です。\n",
            "repo rootから（GPU推論時はbashrcを秘密非表示でsourceし、scripts/cudnn_env.shをsource）:\n\n```sh\nuv run python scripts/compare_release_tracking.py --stage infer --subset pstudio\nuv run python scripts/compare_release_tracking.py --stage infer --subset drivetrack\nuv run python scripts/compare_release_tracking.py --stage infer --subset adt\nuv run python scripts/compare_release_tracking.py --stage render\nuv run python scripts/compare_release_tracking.py --stage audit\n```\n",
        ]
    )
    if "student_checkpoint" in manifest:
        lines = [
            "# Releaseと軽量refinerの4列比較\n\n",
            "左から旧best200 / 新best80 native / 新best80 ONNX CPU / 軽量refiner CUDA FP32。予測=実線、GT=破線、全列で同一点ID・同色・同一前処理とflow visibility・共有メートル軸を使用。\n\n",
            f"軽量モデル: `{manifest['student_architecture']}`、{manifest['student_parameters']:,} parameters。checkpoint SHA256: `{manifest['student_sha256']}`。\n\n",
            "全フレーム・全クエリで評価する。validation選択動画であり、独立テスト/公式minivalではない。3D表示は各フレームのカメラ座標で、予測を列別に拡大縮小しない。軽量refinerはPyTorch FP32であり、ここでは軽量モデルのONNX数値一致を検証していない。\n\n",
            "| 動画 | フレーム/点数 | 旧metric-AJ | 新metric-AJ | ONNX metric-AJ | 軽量metric-AJ |\n",
            "| :--- | ---: | ---: | ---: | ---: | ---: |\n",
        ]
        for row in rows:
            scores = " | ".join(
                f"{row['scores'][key]['metric_average_jaccard']:.5f}"
                for key in comparison_methods(manifest)
            )
            lines.append(
                f"| {row['subset']} | {row['frames']} / {row['tracks']} | {scores} |\n"
            )
        lines.append(
            "\n`comparison_all.mp4`は今回audit対象のsubsetのみを連結。各subset動画、manifest、predictions、reports、共有表示軸とsnapshotも同じフォルダに保存。全3本を生成するにはREADME.mdのwith-student手順を実行する。\n"
        )
    (output / "README.md").write_text("".join(lines))
    print(
        f"[audit] {len(files)} H264/yuv420p videos, {sum(c['frames'] for c in manifest['clips'])} clip frames, full decoding PASSED",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--mode", choices=("release", "with-student"), default="release"
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--student-checkpoint", type=Path)
    parser.add_argument("--student-sha256")
    parser.add_argument("--stage", choices=("infer", "render", "audit"), required=True)
    parser.add_argument("--subset", choices=("pstudio", "drivetrack", "adt"))
    args = parser.parse_args()
    if args.stage == "infer" and args.subset is None:
        parser.error("infer requires an explicit --subset")
    output = (
        args.out_dir
        or (DEFAULT_STUDENT_OUTPUT if args.mode == "with-student" else DEFAULT_OUTPUT)
    ).resolve()
    if args.mode == "release" and (args.student_checkpoint or args.student_sha256):
        parser.error("student arguments require --mode with-student")
    if args.mode == "with-student" and args.student_checkpoint is not None:
        manifest = prepare_student_manifest(
            output,
            args.base_dir.resolve(),
            args.student_checkpoint,
            args.student_sha256,
        )
    else:
        if args.mode == "with-student" and not (output / "manifest.json").exists():
            parser.error("first student run requires --student-checkpoint")
        if args.student_sha256 is not None:
            parser.error("--student-sha256 requires --student-checkpoint")
        manifest = read_manifest(output)
    if ("student_checkpoint" in manifest) != (args.mode == "with-student"):
        parser.error("--mode differs from the output manifest")
    if args.subset is not None and args.stage != "infer":
        manifest = dict(
            manifest, clips=[c for c in manifest["clips"] if c["subset"] == args.subset]
        )
    if args.stage == "infer":
        infer(output, manifest, args.subset)
    elif args.stage == "render":
        render(output, manifest)
    else:
        audit(output, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
