"""Linear temporal depthwise branches with exact algebraic deployment fusion.

The branch sum has no BatchNorm or activation. Any nonlinear channel mixing
belongs outside this module and cannot be folded into the deployment Conv1d.
Inputs are (independent tracks, frames, channels); only frames are convolved.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RepTemporalDWConv(nn.Module):
    def __init__(self, channels: int, *, deploy: bool = False) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels, self.deploy = channels, deploy
        if deploy:
            self.reparam = self._branch(7)
        else:
            self.branch7, self.branch3, self.branch1 = (
                self._branch(kernel) for kernel in (7, 3, 1)
            )

    def _branch(self, kernel: int) -> nn.Conv1d:
        return nn.Conv1d(
            self.channels,
            self.channels,
            kernel,
            padding=kernel // 2,
            groups=self.channels,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        temporal = x.transpose(1, 2)
        if self.deploy:
            y = self.reparam(temporal)
        else:
            y = (
                self.branch7(temporal)
                + self.branch3(temporal)
                + self.branch1(temporal)
                + temporal
            )
        return y.transpose(1, 2)

    def _validate_branch(self, branch: nn.Module, kernel: int) -> nn.Conv1d:
        if not isinstance(branch, nn.Conv1d):
            raise ValueError(
                "Fusion requires linear Conv1d branches without activations"
            )
        if (
            branch.in_channels != self.channels
            or branch.out_channels != self.channels
            or branch.groups != self.channels
            or branch.kernel_size != (kernel,)
            or branch.stride != (1,)
            or branch.dilation != (1,)
            or branch.padding != (kernel // 2,)
            or branch.padding_mode != "zeros"
            or branch.bias is None
        ):
            raise ValueError(
                "Fusion requires matching depthwise Conv1d stride/padding/groups"
            )
        return branch

    def equivalent_kernel_bias(self) -> tuple[Tensor, Tensor]:
        """Fuse finite same-grid branches, retaining autograd for diagnostics."""
        if not all(torch.isfinite(value).all() for value in self.state_dict().values()):
            raise ValueError("Fusion requires finite parameters")
        if self.deploy:
            branch = self._validate_branch(self.reparam, 7)
            assert branch.bias is not None
            return branch.weight, branch.bias
        branch7 = self._validate_branch(self.branch7, 7)
        branch3 = self._validate_branch(self.branch3, 3)
        branch1 = self._validate_branch(self.branch1, 1)
        identity = torch.zeros_like(branch7.weight)
        identity[:, 0, 3] = 1
        weight = (
            branch7.weight
            + F.pad(branch3.weight, (2, 2))
            + F.pad(branch1.weight, (3, 3))
            + identity
        )
        assert (
            branch7.bias is not None
            and branch3.bias is not None
            and branch1.bias is not None
        )
        bias = branch7.bias + branch3.bias + branch1.bias
        if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
            raise ValueError("Fusion produced non-finite parameters")
        return weight, bias

    def to_deploy(self) -> RepTemporalDWConv:
        """Return an independent single-convolution module, without mutating self."""
        weight, bias = self.equivalent_kernel_bias()
        if self.deploy:
            return copy.deepcopy(self)
        fused = RepTemporalDWConv(self.channels, deploy=True).to(
            device=weight.device, dtype=weight.dtype
        )
        assert fused.reparam.bias is not None
        with torch.no_grad():
            fused.reparam.weight.copy_(weight)
            fused.reparam.bias.copy_(bias)
        fused.train(self.training)
        return fused
