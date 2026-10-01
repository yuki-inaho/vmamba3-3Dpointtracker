"""Run refiner ONNX on real precomputed inputs; neither Torch nor CUDA is needed."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mamba3_tracker.deployment.io import save_npz
from mamba3_tracker.deployment.runtime import (
    INPUT_NAMES,
    OUTPUT_NAMES,
    reference_depth_array,
    run_chunked,
    session_for,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="NPZ with the named float32 tensors")
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--track-chunk", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-budget-mib", type=int, default=1024)
    parser.add_argument("--npy-directory", type=Path, help="Also save the validated inputs as C++ runner NPY files")
    args = parser.parse_args()
    with np.load(args.inputs, allow_pickle=False) as archive:
        feed = {name: archive[name] for name in archive.files}
    if "z_ref" not in feed and "z_raw" in feed:
        feed["z_ref"] = reference_depth_array(feed["z_raw"])
    outputs = run_chunked(session_for(args.onnx, args.threads), feed, args.track_chunk, args.memory_budget_mib)
    args.outputs.parent.mkdir(parents=True, exist_ok=True)
    save_npz(args.outputs, dict(zip(OUTPUT_NAMES, outputs, strict=True)))
    if args.npy_directory:
        args.npy_directory.mkdir(parents=True, exist_ok=True)
        for name in INPUT_NAMES:
            np.save(args.npy_directory / f"{name}.npy", feed[name], allow_pickle=False)
    print(f"refiner inference succeeded: {outputs[0].shape}; CPU, float32")


if __name__ == "__main__":
    main()
