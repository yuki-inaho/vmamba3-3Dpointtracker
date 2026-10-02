"""External-feature VSSD two-pool student; train-only heads live elsewhere."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from visionmamba3.cross_attention import Mamba3CrossAttention

from .onnx_refiner import OnnxV35Refiner


class StudentRefiner(OnnxV35Refiner):
    architecture = "vssd_two_pool_128x2_v1"

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self._init_geometry({})
        self.layers = nn.ModuleList(
            [
                Mamba3CrossAttention(
                    dim_q=128,
                    dim_kv=128,
                    num_heads=4,
                    state_dim=64,
                    two_pool=True,
                    bidirectional_mask=False,
                )
                for _ in range(2)
            ]
        )
        with torch.no_grad():
            for head in (self.dz_head, self.duv_head):
                linear = head[-1]
                assert isinstance(linear, nn.Linear) and linear.bias is not None
                linear.weight.zero_()
                linear.bias.zero_()

    def _mix(self, layer: nn.Module, x: Tensor) -> Tensor:
        return layer(x, x)

    def forward_train(
        self,
        ray: Tensor,
        z_raw: Tensor,
        vis: Tensor,
        uv: Tensor,
        depth_map: Tensor,
        dino_features: Tensor,
        K: Tensor,
        z_ref: Tensor,
    ) -> dict[str, Any]:
        x, depth, hidden = self._encode_inputs(
            ray, z_raw, vis, uv, depth_map, dino_features, z_ref
        )
        xyz, new_uv, logits, duv = self._readout(x, depth, uv, K)
        return {
            "xyz": xyz,
            "uv_refined": new_uv,
            "vis_logits": logits,
            "duv": duv,
            "dlog": self.dz_head(x).squeeze(-1).clamp(-2, 2),
            "hidden": hidden,
            "final_hidden": x,
        }

    def initialize_common(self, teacher_state: dict[str, Tensor]) -> list[str]:
        """Copy all and only shape-matched shared tensors; never reshape mixers."""
        own = self.state_dict()
        keys = [key for key in own if not key.startswith("layers.")]
        for key in keys:
            if key not in teacher_state or own[key].shape != teacher_state[key].shape:
                raise ValueError(
                    f"Missing or mismatched teacher geometry tensor: {key}"
                )
        with torch.no_grad():
            for key in keys:
                own[key].copy_(teacher_state[key])
        return keys
