"""Pre-compute DA3-Large metric-depth maps for every TAPVid-3D clip.

v31 uses these depths as a frozen scale source: at train/eval time the 2D
tracker emits (u, v) per (frame, query), the cached depth is bilinearly
sampled at that pixel, and the (X, Y, Z) is recovered via pinhole
unprojection with the clip's intrinsics.

Saving once per clip avoids ~2-3 s of DA3 forward in every training step
(would add ~24h to a 30k-step v31 run). One-time cost is ~45-90 min on
a single GPU; disk ~5 MB per clip × 4569 clips = ~22 GB total.

Output layout (one .npz per clip — uint16 quantized for ~4× disk savings
vs float32; per-clip min/max gives ~0.1 mm precision at 10 m range):
    ~/data/tapvid3d_da3/<subset>/<clip_stem>.npz
        depth_q      : (F, Hd, Wd) uint16   — quantized depth (0..65535)
        d_min        : float32              — per-clip min metres (skips Z=0)
        d_max        : float32              — per-clip max metres
        process_res  : int                  — DA3 working resolution (504)
        clip_path    : str                  — original .npz path
        h, w         : int, int             — original frame H, W

Decode: depth = d_min + (depth_q / 65535) * (d_max - d_min).

Run:
    uv run python scripts/precompute_da3_depths.py
    uv run python scripts/precompute_da3_depths.py --clips 3 --dry-run
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import zipfile
from itertools import zip_longest
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.modules.setdefault("moviepy.editor", types.ModuleType("moviepy.editor"))
sys.path.insert(0, "third_party/depth-anything-3/src")

from depth_anything_3.api import DepthAnything3  # noqa: E402

from mamba3_tracker.data.dataset import official_train_test_split  # noqa: E402
from mamba3_tracker.data.gpu_batch import AdaptiveBatchSize  # noqa: E402
from mamba3_tracker.data.storage import tree_bytes  # noqa: E402


def _which_subset(path: Path) -> str:
    for s in ("pstudio", "drivetrack", "adt"):
        if f"/{s}/" in str(path) or s in path.parts:
            return s
    return "unknown"


def _decode_frame(jpeg):
    with Image.open(io.BytesIO(bytes(jpeg))) as image:
        return np.asarray(image.convert("RGB"))


def _decode_all_frames(jpeg_bytes_arr: np.ndarray, workers=8) -> list[np.ndarray]:
    with ThreadPoolExecutor(max_workers=workers) as decoder:
        return list(decoder.map(_decode_frame, jpeg_bytes_arr))


def _write_depth(depth, clip_path, out_path, process_res, height, width):
    """CPU compression/CRC runs alongside the next GPU inference."""
    valid = depth[depth > 0]
    if valid.size == 0:
        raise ValueError("no positive depth")
    d_min, d_max = float(valid.min()), float(valid.max())
    scale = max(d_max - d_min, 1e-6)
    depth_q = np.clip(np.round((depth - d_min) / scale * 65535.0), 0, 65535).astype(np.uint16)
    temporary = out_path.with_suffix(".npz.partial")
    with temporary.open("wb") as output:
        np.savez_compressed(
            output, depth_q=depth_q, d_min=np.float32(d_min), d_max=np.float32(d_max),
            process_res=np.int32(process_res), clip_path=str(clip_path),
            h=np.int32(height), w=np.int32(width),
        )
    temporary.replace(out_path)
    _mark_ready(clip_path, out_path)
    return depth.shape


def _metric_depth_only(model, images, process_res):
    """Same vendored DA3 inference stages, omitting unused RGB visualization.

    No extrinsics, camera alignment or exports are requested by this worker.
    Depth conversion is the library's original output processor. Pinning lets
    its non-blocking CUDA transfer use page-locked host memory.
    """
    imgs_cpu, _, _ = model._preprocess_inputs(images, None, None, process_res,
                                            "upper_bound_resize")
    imgs_cpu = imgs_cpu.pin_memory()
    imgs, _, _ = model._prepare_model_inputs(imgs_cpu, None, None)
    output = model._run_model_forward(imgs, None, None, [], False, False)
    return model._convert_to_prediction(output)


def _infer_depth(model, rgbs, args, batch, device):
    depth_chunks = []
    start = 0
    while start < len(rgbs):
        cuda = device.type == "cuda"
        free = torch.cuda.mem_get_info(device)[0] if cuda else None
        count = batch.size(len(rgbs) - start, free)
        baseline = torch.cuda.memory_allocated(device) if cuda else 0
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        before = time.perf_counter()
        try:
            with torch.inference_mode():
                if args.auto_batch:
                    pred = _metric_depth_only(model, rgbs[start:start + count], args.process_res)
                else:
                    pred = model.inference(rgbs[start:start + count],
                                           process_res=args.process_res, export_format="mini_npz")
        except torch.cuda.OutOfMemoryError:
            if not args.auto_batch:
                raise
            batch.oom(count)
            torch.cuda.empty_cache()
            print(f"[da3-precompute] OOM retry frame={start} batch={batch.current}", flush=True)
            continue
        depth_chunks.append(np.asarray(pred.depth, dtype=np.float32))
        del pred
        peak = torch.cuda.max_memory_allocated(device) if cuda else 0
        batch.success(count, peak - baseline)
        row = {"time": time.time(), "frames": count, "seconds": time.perf_counter() - before,
               "peak_gpu_gb": peak / 1e9, "next_batch": batch.current, "oom_retries": batch.retries}
        row["frames_per_second"] = count / row["seconds"]
        print("[da3-precompute] GPU " + json.dumps(row), flush=True)
        if args.stats_json:
            args.stats_json.parent.mkdir(parents=True, exist_ok=True)
            with args.stats_json.open("a") as stats:
                stats.write(json.dumps(row) + "\n")
        start += count
    return np.concatenate(depth_chunks, axis=0)


def _mark_ready(raw_path, depth_path):
    with zipfile.ZipFile(depth_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"depth CRC failed: {bad}")
    with np.load(depth_path) as depth:
        if not {"depth_q", "d_min", "d_max"} <= set(depth.files):
            raise ValueError("depth quantization metadata missing")
        shape = depth["depth_q"].shape
    from mamba3_tracker.data.tapvid3d import peek_clip_F
    if shape[0] != peek_clip_F(raw_path):
        raise ValueError("depth cache frame count differs from raw clip")
    marker = depth_path.with_suffix(".npz.ready.json")
    tmp = marker.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            {
                "raw_bytes": raw_path.stat().st_size,
                "depth_bytes": depth_path.stat().st_size,
                "frames": shape[0],
                "shape": shape,
            }
        )
    )
    tmp.replace(marker)


def _work_queue(args):
    """Round-robin all subsets; new downloads are admitted on each watch scan."""
    while True:
        train, test = official_train_test_split(args.data_root, subsets=args.subsets)
        if args.train_manifest:
            selected = json.loads(args.train_manifest.read_text())["files_by_subset"]
            train = [p for p in train if p.name in selected[p.parent.name]]
        all_clips = test if args.minival else train + test
        # Prepare initial training clips in every subset before full-clip eval data.
        per_subset = []
        for subset in args.subsets:
            tr = sorted(p for p in train if p.parent.name == subset)
            rest = sorted(
                p for p in all_clips if p.parent.name == subset and p not in tr[:6]
            )
            per_subset.append((tr[:6] if not args.minival else []) + rest)
        pending = [p for row in zip_longest(*per_subset) for p in row if p is not None]
        if args.clips:
            pending = pending[: args.clips]
        if args.watch:
            pending = [
                p
                for p in pending
                if not (
                    args.out_root / p.parent.name / (p.name + ".ready.json")
                ).is_file()
            ]
        for path in pending:
            yield path
        if not args.watch:
            return
        print(
            f"[da3-precompute] WATCH: pass done; waiting {args.poll_seconds}s for clips",
            flush=True,
        )
        time.sleep(args.poll_seconds)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path.home() / "data")
    ap.add_argument(
        "--out-root", type=Path, default=Path.home() / "data" / "tapvid3d_da3"
    )
    ap.add_argument("--model-name", default="da3metric-large")
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--subsets", nargs="+", default=["pstudio", "drivetrack", "adt"])
    ap.add_argument(
        "--clips",
        type=int,
        default=0,
        help="Limit total clips for smoke testing (0 = all).",
    )
    ap.add_argument(
        "--chunk-frames",
        type=int,
        default=16,
        help="DA3-Large at 504² won't fit a full ADT clip (300 frames) in "
        "<=12 GiB. Process this many frames per DA3 forward and "
        "concatenate depth outputs along the frame axis.",
    )
    ap.add_argument("--minival", action="store_true", help="only the 150 minival clips")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--train-manifest", type=Path)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--poll-seconds", type=float, default=30)
    ap.add_argument("--auto-batch", action="store_true", help="Grow independent metric-depth GPU batches")
    ap.add_argument("--max-batch-frames", type=int, default=128)
    ap.add_argument("--gpu-reserve-gb", type=float, default=8,
                    help="Leave GPU memory for simultaneous training")
    ap.add_argument("--decode-workers", type=int, default=8)
    ap.add_argument("--cpu-threads", type=int, default=8)
    ap.add_argument("--stats-json", type=Path, help="Append batch throughput/VRAM/OOM statistics")
    ap.add_argument(
        "--total-data-budget-gb",
        type=float,
        default=60,
        help="Raw + depth + feature/flow cache cap under data-root",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.auto_batch and (device.type != "cuda" or not args.model_name.startswith("da3metric-")):
        ap.error("--auto-batch requires CUDA and an independent-frame da3metric model")
    if args.decode_workers < 1 or args.cpu_threads < 1:
        ap.error("--decode-workers and --cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    batch = AdaptiveBatchSize(args.chunk_frames, max(args.chunk_frames, args.max_batch_frames),
                              args.gpu_reserve_gb, adaptive=args.auto_batch)
    print(
        f"[da3-precompute] device={device}, model={args.model_name}, "
        f"process_res={args.process_res}",
        flush=True,
    )

    print("[da3-precompute] loading DA3 ...", flush=True)
    model = DepthAnything3.from_pretrained(f"depth-anything/{args.model_name}").to(
        device
    )
    model.device = device
    model.eval()
    print("[da3-precompute] loaded", flush=True)

    args.out_root.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    n_done = n_skipped = n_failed = 0
    writer = ThreadPoolExecutor(max_workers=1)
    pending = None
    pending_path = None

    def finish_write():
        nonlocal pending, n_done, n_failed
        if pending is None:
            return
        try:
            shape = pending.result()
            n_done += 1
            print(f"[da3-precompute] READY {pending_path} shape={shape} written={n_done} "
                  f"skipped={n_skipped} failed={n_failed}", flush=True)
        except Exception as error:
            n_failed += 1
            print(f"[da3-precompute] WRITE FAIL {pending_path}: {error}", flush=True)
        pending = None

    for i, clip_path in enumerate(_work_queue(args)):
        if pending is not None and (pending.done() or pending_path == f"{_which_subset(clip_path)}/{clip_path.name}"):
            finish_write()
        subset = _which_subset(clip_path)
        out_dir = args.out_root / subset
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (clip_path.stem + ".npz")
        if args.skip_existing and out_path.exists():
            try:
                _mark_ready(clip_path, out_path)
                n_skipped += 1
                continue
            except (OSError, ValueError, zipfile.BadZipFile, EOFError):
                print(
                    f"[da3-precompute] rebuilding corrupt cache {out_path}", flush=True
                )
        storage_root = (
            args.data_root
            if args.data_root.name != "tapvid3d"
            else args.data_root.parent
        )
        used = tree_bytes(storage_root)
        # Leave 1GB for concurrent downloader writes; raw downloader has a 30GB cap.
        if used > args.total_data_budget_gb * 1e9 - 1e9:
            print(
                f"[da3-precompute] BUDGET raw+cache={used / 1e9:.2f}GB; waiting",
                flush=True,
            )
            if not args.watch:
                return 2
            time.sleep(args.poll_seconds)
            continue
        try:
            with np.load(clip_path, allow_pickle=True) as d:
                if "images_jpeg_bytes" not in d.files:
                    print(
                        f"  [{i:5d}] {subset:<11} {clip_path.name}: SKIP labels-only",
                        flush=True,
                    )
                    n_skipped += 1
                    continue
                rgbs = _decode_all_frames(d["images_jpeg_bytes"], args.decode_workers)
            H, W = rgbs[0].shape[:2]
            depth = _infer_depth(model, rgbs, args, batch, device)
            del rgbs
            finish_write()  # Bound the queue to one clip; propagate prior save errors.
            pending_path = f"{subset}/{clip_path.name}"
            pending = writer.submit(_write_depth, depth, clip_path, out_path, args.process_res, H, W)
        except Exception as e:
            print(
                f"  [{i:5d}] {subset:<11} {clip_path.name}: FAIL ({type(e).__name__}: {e})",
                flush=True,
            )
            n_failed += 1

    finish_write()
    writer.shutdown()
    elapsed = time.time() - t_start
    print(
        f"\n[da3-precompute] DONE: {n_done} written, {n_skipped} skipped, {n_failed} failed "
        f"in {elapsed / 60:.1f} min",
        flush=True,
    )
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
