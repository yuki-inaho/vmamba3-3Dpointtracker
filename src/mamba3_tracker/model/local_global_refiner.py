"""Track-independent local/global refiner with explicit deployment fusion.

Only the linear temporal branches are reparameterized. LayerNorm, normalized
global pooling, learned residual scales, and the nonlinear FFN remain explicit
standard operators. The existing geometry contract supplies eight inputs and
four outputs, with neither frozen feature extractors nor train-only heads.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from .compact_vssd import CompactCrossAttention
from .rep_temporal import RepTemporalDWConv
from .student_refiner import StudentRefiner


class LocalGlobalMixer(nn.Module):
    """Return a residual increment for each independently folded track.

    Time is axis 1 of (batch * tracks, frames, channels). A zero-padded temporal
    convolution preserves that ordering; global pools never materialize F x F.
    LayerNorm works on channels only, never across frames or tracks.
    """

    def __init__(self, *, deploy: bool = False) -> None:
        super().__init__()
        self.local = RepTemporalDWConv(128, deploy=deploy)
        self.global_mixer = CompactCrossAttention(
            dim=128, num_heads=4, state_dim=64, normalized=True
        )
        self.ffn = nn.Sequential(nn.Linear(128, 512), nn.GELU(), nn.Linear(512, 128))
        self.local_norm = nn.LayerNorm(128)
        self.global_norm = nn.LayerNorm(128)
        self.ffn_norm = nn.LayerNorm(128)
        self.local_scale = nn.Parameter(torch.full((128,), 0.01))
        self.global_scale = nn.Parameter(torch.full((128,), 0.01))
        self.ffn_scale = nn.Parameter(torch.full((128,), 0.01))

    def forward(self, x: Tensor) -> Tensor:
        local_delta = self.local_scale * self.local(self.local_norm(x))
        local_x = x + local_delta
        global_input = self.global_norm(local_x)
        global_delta = self.global_scale * self.global_mixer(global_input, global_input)
        combined_x = local_x + global_delta
        ffn_delta = self.ffn_scale * self.ffn(self.ffn_norm(combined_x))
        # Summing increments avoids subtracting a large skip from the result.
        return local_delta + global_delta + ffn_delta


class LocalGlobalStudentRefiner(StudentRefiner):
    """128-wide, two-layer normalized global + Rep temporal + FFN student.

    Geometry and training readout intentionally inherit the old student's
    implementations. The old v1 architecture and state-dict remain untouched.
    Train and deploy have distinct identities because their convolution keys
    differ; conversion returns an independent copy rather than mutating a run.
    """

    architecture = "vssd_local_global_128x2_v2"
    train_architecture = "vssd_local_global_128x2_v2_train"

    def __init__(self, *, deploy: bool = False) -> None:
        nn.Module.__init__(self)
        self._init_geometry({})
        self.deploy = deploy
        self.architecture = (
            type(self).architecture if deploy else self.train_architecture
        )
        self.layers = nn.ModuleList([LocalGlobalMixer(deploy=deploy) for _ in range(2)])
        with torch.no_grad():
            for head in (self.dz_head, self.duv_head):
                linear = head[-1]
                assert isinstance(linear, nn.Linear) and linear.bias is not None
                linear.weight.zero_()
                linear.bias.zero_()

    def _mix(self, layer: nn.Module, x: Tensor) -> Tensor:
        return layer(x)

    def to_deploy(self) -> LocalGlobalStudentRefiner:
        """Deep-copy finite tensors, then fold linear convolution branches only."""
        if not all(torch.isfinite(value).all() for value in self.state_dict().values()):
            raise ValueError("Deployment conversion requires finite parameters")
        result = copy.deepcopy(self)
        for layer in result.layers:
            assert isinstance(layer, LocalGlobalMixer)
            layer.local = layer.local.to_deploy()
        result.deploy = True
        result.architecture = type(self).architecture
        return result
