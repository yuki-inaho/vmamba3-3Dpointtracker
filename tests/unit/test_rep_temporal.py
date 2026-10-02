"""Train-time linear temporal branches must fuse without changing the function."""

import pytest
import torch


@pytest.mark.parametrize("frames", [1, 8, 31, 600])
def test_linear_branch_fusion_preserves_values_gradients_and_source(frames):
    from mamba3_tracker.model.rep_temporal import RepTemporalDWConv

    torch.manual_seed(26)
    model = RepTemporalDWConv(128).eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    deploy = model.to_deploy()
    assert deploy.deploy
    assert sum(p.numel() for p in deploy.parameters()) == 1024
    assert not any("branch" in key for key in deploy.state_dict())
    x = torch.randn(6, frames, 128, requires_grad=True)
    expected, actual = model(x), deploy(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for got, want in zip(
        torch.autograd.grad(actual.sum(), (x,)),
        torch.autograd.grad(expected.sum(), (x,)),
    ):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    assert all(
        torch.equal(value, model.state_dict()[key]) for key, value in before.items()
    )
    with torch.no_grad():
        deploy.reparam.weight.add_(1)
    assert all(
        torch.equal(value, model.state_dict()[key]) for key, value in before.items()
    )


def test_temporal_order_and_batch_independence():
    from mamba3_tracker.model.rep_temporal import RepTemporalDWConv

    model = RepTemporalDWConv(2, deploy=True)
    assert model.reparam.bias is not None
    with torch.no_grad():
        model.reparam.weight.zero_()
        model.reparam.bias.zero_()
        model.reparam.weight[:, 0, 2] = 1  # Previous frame only; no batch/track mixing.
    x = torch.arange(2 * 8 * 2, dtype=torch.float32).reshape(2, 8, 2)
    want = torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)
    torch.testing.assert_close(model(x), want, rtol=0, atol=0)
    torch.testing.assert_close(model(x.flip(0)).flip(0), want, rtol=0, atol=0)


def test_fusion_rejects_nonfinite_and_is_idempotent_copy():
    from mamba3_tracker.model.rep_temporal import RepTemporalDWConv

    model = RepTemporalDWConv(4)
    fused = model.to_deploy()
    again = fused.to_deploy()
    assert fused is not again
    assert all(
        torch.equal(value, again.state_dict()[key])
        for key, value in fused.state_dict().items()
    )
    with torch.no_grad():
        model.branch7.weight[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model.to_deploy()


def test_branch_nonlinearity_is_not_silently_fused():
    from mamba3_tracker.model.rep_temporal import RepTemporalDWConv

    model = RepTemporalDWConv(4)
    model.add_module("branch3", torch.nn.Sequential(model.branch3, torch.nn.ReLU()))
    with pytest.raises(ValueError, match="Conv1d"):
        model.to_deploy()
