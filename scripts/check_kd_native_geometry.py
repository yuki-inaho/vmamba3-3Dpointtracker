"""Compare the KD teacher with the original V35 forward at fixed features."""

import json
from pathlib import Path
import torch
from torch import nn
from mamba3_tracker.deployment.student_checkpoint import load_native_teacher
from mamba3_tracker.deployment.runtime import INPUT_NAMES
from mamba3_tracker.model.depth_refined_tracker import Mamba3V35Refiner
from mamba3_tracker.model.onnx_refiner import reference_depth

torch.set_num_threads(8)
teacher = load_native_teacher(
    Path("weights/mamba3-preview-20261002-best80/tracker_mamba3_daoff_best80.pt"),
    "c022511872c2d59a39bc1c39a3c62f44c142ee8ff9ca7dc7859881a78ae490b7",
).cuda()
manifest = json.loads(
    Path("result/refiner_kd_20261002/cache_pilot/input_manifest.json").read_text()
)
rows = []
for record in manifest["records"][:3]:
    payload = torch.load(record["path"], weights_only=True)
    values = [payload["inputs"][name].cuda().float() for name in INPUT_NAMES[:-1]]
    values.append(reference_depth(values[1]))

    class FixedDino(nn.Module):
        def forward_video(self, images):
            return [values[5]]

    original = Mamba3V35Refiner.__new__(Mamba3V35Refiner)
    nn.Module.__init__(original)
    for key, module in teacher.named_children():
        setattr(original, key, module)
    for name in (
        "dim",
        "image_size",
        "patch_size",
        "max_log_correction",
        "max_delta_uv",
        "gate_by_vis",
    ):
        setattr(original, name, getattr(teacher, name))
    original.dino = FixedDino()
    original.per_frame_scale = original.within_frame = original.pose_head = False
    original.eval()
    with torch.no_grad():
        expected = original(
            *values[:5], torch.zeros(1, device="cuda"), values[6], z_ref=values[7]
        )
        actual = teacher(*values)
    errors = {}
    for name, ref, value in zip(
        ("xyz", "uv", "vis", "duv"),
        (expected.xyz, expected.uv, expected.vis_logits, expected.delta_uv),
        actual,
    ):
        torch.testing.assert_close(ref, value, rtol=0, atol=0)
        errors[name] = float((ref - value).abs().max())
    rows.append({"clip": record["clip"], "max_absolute_error": errors})
output = Path("result/refiner_kd_20261002/native_geometry_parity.json")
output.write_text(
    json.dumps(
        {
            "success": True,
            "scope": "Original V35 vs external-feature native teacher; same FP32 feature maps; BF16 native mixers",
            "checks": rows,
        },
        indent=2,
    )
    + "\n"
)
print(output)
