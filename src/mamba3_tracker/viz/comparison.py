"""Fair, deterministic geometry and a paper-inspired three-column video layout."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np


def project_xyz(xyz: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Camera-frame metres to original-image pixels, with no fitted scaling."""
    xyz = np.asarray(xyz, dtype=np.float64)
    valid = np.isfinite(xyz).all(-1) & (xyz[..., 2] > 0)
    uv = np.full((*xyz.shape[:-1], 2), np.nan)
    uv[..., 0][valid] = k[0, 0] * xyz[..., 0][valid] / xyz[..., 2][valid] + k[0, 2]
    uv[..., 1][valid] = k[1, 1] * xyz[..., 1][valid] / xyz[..., 2][valid] + k[1, 2]
    return uv


def select_tracks(visibility: np.ndarray, count: int) -> np.ndarray:
    """GT visibility only, deterministic track-ID ties; never inspect predictions."""
    if visibility.ndim != 2 or visibility.shape[0] == 0 or count < 1:
        raise ValueError(
            "Nonempty N/F visibility and a positive track limit are required"
        )
    ids = np.arange(visibility.shape[0])
    order = np.lexsort((ids, -(visibility > 0.5).sum(axis=1)))
    return order[:count]


def shared_cube(arrays: list[np.ndarray]) -> np.ndarray:
    """One equal-metre cube containing all displayed predictions and GT."""
    points = np.concatenate([array.reshape(-1, 3) for array in arrays])
    points = points[np.isfinite(points).all(axis=1)]
    if not points.size:
        raise ValueError("Cannot frame an empty/nonfinite trajectory collection")
    low, high = points.min(0), points.max(0)
    centre = (low + high) / 2
    half = max(float((high - low).max()), 0.1) * 0.55
    return np.column_stack((centre - half, centre + half))


def gt_detail_cube(gt: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    """Fixed, prediction-independent detail volume from GT-visible 10/90 percentiles."""
    points = gt[(visibility > 0.5) & np.isfinite(gt).all(-1)]
    if not points.size:
        raise ValueError("A GT detail window requires finite, visible ground truth")
    low, high = np.quantile(points, [0.1, 0.9], axis=0)
    centre = (low + high) / 2
    half = max(float((high - low).max()), 0.1) * 0.55
    return np.column_stack((centre - half, centre + half))


def _clip_detail(points: np.ndarray, limits: np.ndarray) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64).copy()
    result[np.any((result < limits[:, 0]) | (result > limits[:, 1]), axis=-1)] = np.nan
    return result


def masked_path(
    xyz: np.ndarray, visibility: np.ndarray, frame: int, tail: int
) -> np.ndarray:
    """NaN-separated path, without joining across occlusion or showing future frames."""
    if tail < 1 or not 0 <= frame < len(xyz) or len(xyz) != len(visibility):
        raise ValueError("Invalid frame, tail, or visibility length")
    start = max(0, frame - tail + 1)
    points = np.asarray(xyz[start : frame + 1], dtype=np.float64).copy()
    points[visibility[start : frame + 1] <= 0.5] = np.nan
    return points


def set_point_3d(artist: Any, point: np.ndarray) -> None:
    """Matplotlib 3D's nonfinite-point path requires ndarray, not Python lists."""
    artist.set_data_3d(np.asarray(point, dtype=np.float64).reshape(3, 1))


def _dashed(draw, path: np.ndarray, colour: tuple[int, ...]) -> None:
    for a, b in zip(path[:-1], path[1:]):
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        length = float(np.linalg.norm(b - a))
        if length > 2000:
            continue
        for distance in np.arange(0.0, length, 9.0):
            end = min(distance + 4.0, length)
            if length > 0:
                p, q = a + (b - a) * distance / length, a + (b - a) * end / length
                draw.line([tuple(p), tuple(q)], fill=colour, width=2)


def _solid(draw, path: np.ndarray, colour: tuple[int, ...]) -> None:
    for a, b in zip(path[:-1], path[1:]):
        if (
            np.isfinite(a).all()
            and np.isfinite(b).all()
            and np.linalg.norm(b - a) < 2000
        ):
            draw.line([tuple(a), tuple(b)], fill=colour, width=2)


def render_comparison(
    frames: np.ndarray,
    arrays: dict[str, np.ndarray],
    report: dict[str, Any],
    output: Path,
    fps: int = 15,
    count: int = 32,
    tail: int = 30,
) -> dict[str, Any]:
    """Render full frames; selection and crop affect display only, not scoring."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    from PIL import Image, ImageDraw, ImageFont

    from mamba3_tracker.viz.track3d_axes import apply_equal_cube, apply_image_like_view

    methods = ("old", "new", "onnx") + (
        ("student",) if "student" in report["scores"] else ()
    )
    names = (
        "OLD RELEASE / best200",
        "NEW RELEASE / best80",
        "NEW RELEASE / best80 ONNX",
    ) + (("LIGHTWEIGHT / distilled refiner",) if "student" in methods else ())
    columns = len(methods)
    canvas_width = 640 * columns
    gt, vis = arrays["gt"], arrays["gt_visibility"]
    flow_vis, k = arrays["flow_visibility"], arrays["K"]
    selected = select_tracks(vis, count)
    lims = shared_cube(
        [gt[selected], *(arrays[f"{key}_xyz"][selected] for key in methods)]
    )
    detail_limits = gt_detail_cube(gt[selected], vis[selected])
    detail_span = float(detail_limits[0, 1] - detail_limits[0, 0])
    colours = [
        tuple(int(v * 255) for v in plt.get_cmap("hsv")(i / len(selected))[:3])
        for i in range(len(selected))
    ]
    colours_float = [tuple(v / 255 for v in c) for c in colours]
    height, width = frames.shape[1:3]
    scale = min(640 / width, 360 / height)
    size = (round(width * scale), round(height * scale))
    offset = np.array([(640 - size[0]) // 2, (360 - size[1]) // 2])
    uv = {key: project_xyz(arrays[f"{key}_xyz"], k) * scale + offset for key in methods}
    gt_uv = project_xyz(gt, k) * scale + offset
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 21)
    small = ImageFont.truetype(font_path, 18)
    title = ImageFont.truetype(font_path, 27)
    fig = plt.figure(figsize=(canvas_width / 100, 4.6), dpi=100, facecolor="#f8fafc")
    fig.subplots_adjust(left=0.01, right=0.98, bottom=0.08, top=0.88, wspace=0.02)
    lines, detail_lines = [], []

    def add_tracks(ax):
        own = []
        for colour in colours_float:
            (gt_line,) = ax.plot([], [], [], "--", color=colour, alpha=0.5, lw=1.3)
            (pred_line,) = ax.plot([], [], [], "-", color=colour, lw=1.8)
            (gt_dot,) = ax.plot([], [], [], "o", color=colour, mfc="none", ms=4)
            (pred_dot,) = ax.plot([], [], [], "o", color=colour, ms=4)
            own.append((gt_line, pred_line, gt_dot, pred_dot))
        return own

    for column, key in enumerate(methods):
        ax = cast(
            Axes3D,
            fig.add_axes(
                ((column + 0.309) / columns, 0.08, 0.654 / columns, 0.80),
                projection="3d",
            ),
        )
        apply_equal_cube(ax, lims)
        apply_image_like_view(ax)
        ax.set_facecolor("#f8fafc")
        ax.tick_params(labelsize=11, pad=0)
        for dimension, label in zip(
            (ax.xaxis, ax.yaxis, ax.zaxis), ("X [m]", "Y [m]", "Z [m]")
        ):
            dimension.label.set_text(label)
            dimension.label.set_fontsize(12)
        lines.append(add_tracks(ax))
        detail = fig.add_axes(
            ((column + 0.039) / columns, 0.22, 0.315 / columns, 0.50), projection="3d"
        )
        apply_equal_cube(detail, detail_limits)
        apply_image_like_view(detail)
        detail.set_axis_off()
        detail.set_facecolor("#edf2f7")
        detail_lines.append(add_tracks(detail))
        fig.text(
            (column + 0.192) / columns,
            0.15,
            f"GT-centred detail\nshared {detail_span:.2f} m cube\n(outside detail cropped)",
            ha="center",
            va="center",
            fontsize=9,
            color="#334155",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{canvas_width}x1080",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "8",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    snapshots = []
    error_log = output.with_suffix(".ffmpeg.log")
    with error_log.open("wb") as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
        try:
            assert process.stdin is not None
            for frame in range(len(frames)):
                canvas = Image.new("RGB", (canvas_width, 1080), "#0f172a")
                draw = ImageDraw.Draw(canvas)
                label = f"{report['subset'].upper()} | {report['clip_id']}"
                while title.getlength(label) > canvas_width - 470:
                    label = label[:-4] + "..."
                draw.text((22, 14), label, font=title, fill="#f8fafc")
                draw.text(
                    (canvas_width - 370, 18),
                    f"FRAME {frame + 1:03d}/{len(frames):03d} | {fps} fps",
                    font=font,
                    fill="#f8fafc",
                )
                source = Image.fromarray(frames[frame]).resize(
                    size, Image.Resampling.LANCZOS
                )
                for column, key in enumerate(methods):
                    x = column * 640
                    draw.text((x + 20, 68), names[column], font=font, fill="#7dd3fc")
                    draw.text(
                        (x + 20, 97),
                        "CUDA / FP32 lightweight"
                        if key == "student"
                        else (
                            "CPU / FP32 refiner"
                            if key == "onnx"
                            else "CUDA / BF16 mixer"
                        ),
                        font=small,
                        fill="#cbd5e1",
                    )
                    panel = Image.new("RGB", (640, 360), "black")
                    panel.paste(source, tuple(offset))
                    pd = ImageDraw.Draw(panel)
                    for index, point in enumerate(selected):
                        colour = colours[index]
                        _dashed(
                            pd,
                            masked_path(gt_uv[point], vis[point], frame, tail),
                            colour,
                        )
                        _solid(
                            pd,
                            masked_path(uv[key][point], flow_vis[point], frame, tail),
                            colour,
                        )
                        for point_uv, visible, radius, prediction in (
                            (gt_uv[point, frame], vis[point, frame], 5, False),
                            (uv[key][point, frame], flow_vis[point, frame], 3, True),
                        ):
                            if not np.isfinite(point_uv).all():
                                continue
                            u, v = point_uv
                            if not 0 <= u < 640 or not 0 <= v < 360:
                                continue
                            if not prediction and visible <= 0.5:
                                continue
                            box = (u - radius, v - radius, u + radius, v + radius)
                            pd.ellipse(
                                box,
                                fill=colour if prediction and visible > 0.5 else None,
                                outline=colour,
                                width=2,
                            )
                    canvas.paste(panel, (x, 120))
                    metrics = report["scores"][key]
                    draw.text(
                        (x + 20, 489),
                        f"metric-AJ {metrics['metric_average_jaccard']:.4f}  |  3D-AJ {metrics['average_jaccard']:.4f}",
                        font=font,
                        fill="white",
                    )
                    draw.text(
                        (x + 20, 521),
                        f"3D error: mean {metrics['metric_err_mean_m']:.3f} m / median {metrics['metric_err_median_m']:.3f} m",
                        font=small,
                        fill="#cbd5e1",
                    )
                    if key == "onnx":
                        difference = report["native_onnx_difference"]
                        note = f"vs native p95: {difference['xyz']['p95'] * 100:.3f} cm / {difference['uv896']['p95']:.5f} px (896)"
                    else:
                        note = (
                            "Full frames / all queries scored; no metric scale fitting"
                        )
                    draw.text((x + 20, 546), note, font=small, fill="#a5f3fc")
                    for group, limits in (
                        (lines[column], None),
                        (detail_lines[column], detail_limits),
                    ):
                        for index, point in enumerate(selected):
                            gl, pl, gd, pp = group[index]
                            for line, path in (
                                (gl, masked_path(gt[point], vis[point], frame, tail)),
                                (
                                    pl,
                                    masked_path(
                                        arrays[f"{key}_xyz"][point],
                                        flow_vis[point],
                                        frame,
                                        tail,
                                    ),
                                ),
                            ):
                                if limits is not None:
                                    path = _clip_detail(path, limits)
                                line.set_data_3d(path[:, 0], path[:, 1], path[:, 2])
                            g = (
                                gt[point, frame]
                                if vis[point, frame] > 0.5
                                else np.full(3, np.nan)
                            )
                            p = arrays[f"{key}_xyz"][point, frame]
                            if limits is not None:
                                g, p = _clip_detail(np.stack((g, p)), limits)
                            set_point_3d(gd, g)
                            set_point_3d(pp, p)
                            pp.set_markerfacecolor(
                                colours_float[index]
                                if flow_vis[point, frame] > 0.5
                                else "none"
                            )
                fig.canvas.draw()
                plot = np.asarray(cast(FigureCanvasAgg, fig.canvas).buffer_rgba())[
                    ..., :3
                ]
                canvas.paste(Image.fromarray(plot), (0, 570))
                draw.text(
                    (20, 1038),
                    "SOLID / filled = prediction  |  DASHED / ring = GT  |  hollow prediction = flow-occluded",
                    font=small,
                    fill="white",
                )
                draw.text(
                    (20, 1060),
                    f"Same {len(selected)} display IDs / {report['tracks']} scored tracks; per-frame camera XYZ; validation selection, NOT official minival",
                    font=small,
                    fill="#cbd5e1",
                )
                if frame in {0, len(frames) // 2, len(frames) - 1}:
                    snapshot = output.with_name(f"{output.stem}_frame{frame:03d}.png")
                    canvas.save(snapshot)
                    snapshots.append(str(snapshot))
                process.stdin.write(canvas.tobytes())
                if frame % 50 == 0:
                    print(
                        f"[render] {report['subset']} {frame}/{len(frames)}", flush=True
                    )
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f"ffmpeg failed; see {error_log}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            plt.close(fig)
    return {
        "methods": list(methods),
        "width": canvas_width,
        "height": 1080,
        "selected_track_ids": selected.tolist(),
        "shared_cube_m": lims.tolist(),
        "shared_gt_detail_cube_m": detail_limits.tolist(),
        "detail_selection": "GT-visible 10/90 percentiles, no prediction-based selection",
        "tail_frames": tail,
        "fps_display": fps,
        "snapshots": snapshots,
        "command": command,
        "method_specific_scaling": False,
        "scope": "per-frame camera coordinates, not reconstructed world coordinates",
    }
