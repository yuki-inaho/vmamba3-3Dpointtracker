"""Download the exact official Mamba-3 SISO checkpoint used by the tracker."""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from mamba3_tracker.model.official_mamba3 import MAMBA3_REPO, MAMBA3_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("weights/mamba3-siso-187m"))
    args = parser.parse_args()
    for name in ("config.json", "pytorch_model.bin"):
        path = hf_hub_download(MAMBA3_REPO, name, revision=MAMBA3_REVISION,
                               local_dir=args.out_dir.expanduser())
        print(path, flush=True)


if __name__ == "__main__":
    main()
