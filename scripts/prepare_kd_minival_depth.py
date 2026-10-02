"""Use pinned DA3 metric weights for all available minival clips, then watch."""

import hashlib
import importlib.util
import json
from pathlib import Path
import time

from mamba3_tracker.data.tapvid3d_splits import MINIVAL_FILES
from mamba3_tracker.data.storage import tree_bytes

ROOT = Path(__file__).resolve().parents[1]

snapshot = (
    Path.home()
    / ".cache/huggingface/hub/models--depth-anything--da3metric-large/snapshots/4010e39f3634a45bc60553321fb49fb760bd594e"
)
weight = snapshot / "model.safetensors"
with weight.open("rb") as stream:
    actual = hashlib.file_digest(stream, "sha256").hexdigest()
if actual != "bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776":
    raise ValueError("DA3 checkpoint differs from the teacher frontend")
spec = importlib.util.spec_from_file_location(
    "kd_depth_precompute", ROOT / "scripts/precompute_da3_depths.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original_loader = module.DepthAnything3.from_pretrained


def pinned_loader(requested):
    if requested != "depth-anything/da3metric-large":
        raise ValueError("Unexpected depth model")
    return original_loader(str(snapshot))


def work_queue(args):
    while True:
        ready = 0
        pending = []
        for subset, names in MINIVAL_FILES.items():
            for name in names:
                source = args.data_root / subset / name
                marker = args.out_root / subset / (name + ".ready.json")
                if marker.is_file():
                    ready += 1
                elif source.is_file():
                    pending.append(source)
        report = {
            "ready": ready,
            "expected": 150,
            "available_pending": len(pending),
            "status": "complete" if ready == 150 else "running",
            "model_sha256": actual,
            "revision": snapshot.name,
            "process_resolution": args.process_res,
        }
        status = args.out_root / "preparation_status.json"
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text(json.dumps(report, indent=2) + "\n")
        if ready == 150:
            return
        for source in pending:
            yield source
        print(
            json.dumps({"depth_ready_at_scan": ready, "pending_at_scan": len(pending)}),
            flush=True,
        )
        time.sleep(20)


module.DepthAnything3.from_pretrained = staticmethod(pinned_loader)
module._work_queue = work_queue
# Include raw, prepared refiner inputs and depth in the same 60GB evaluation cap.
module.tree_bytes = lambda unused: tree_bytes(Path("/workspace/vmamba3_eval"))
if __name__ == "__main__":
    raise SystemExit(module.main())
