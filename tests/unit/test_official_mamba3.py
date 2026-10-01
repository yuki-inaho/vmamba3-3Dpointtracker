import torch
import pytest

from mamba3_tracker.model.official_mamba3 import OfficialMamba3Adapter, load_pretrained_mixers


def test_exact_pretrained_mixer_transfer_and_rejection(tmp_path):
    layer = OfficialMamba3Adapter()
    state = {f"backbone.layers.3.mixer.{k}": v.detach().clone()
             for k, v in layer.mixer.state_dict().items()}
    state["backbone.layers.3.norm.weight"] = torch.full_like(layer.norm.weight, 2.)
    state["backbone.layers.3.mixer.D"] = torch.full_like(layer.mixer.D, 3.)
    state["lm_head.weight"] = torch.ones(2, 2)
    path = tmp_path / "pretrained.pt"
    torch.save(state, path)
    report = load_pretrained_mixers([layer], path, [3])
    assert report["mixer_coverage"] == 1.0 and report["loaded_tensors"] == 9
    assert report["loaded_parameters"] == 3_822_640
    assert torch.equal(layer.norm.weight, state["backbone.layers.3.norm.weight"])
    assert torch.equal(layer.mixer.D, state["backbone.layers.3.mixer.D"])
    with pytest.raises(ValueError, match="sha256"):
        load_pretrained_mixers([layer], path, [3], expected_sha256="incorrect hash")
    original = layer.mixer.in_proj.weight.detach().clone()
    state["backbone.layers.3.mixer.in_proj.weight"] = torch.ones(1, 1)
    torch.save(state, path)
    with pytest.raises(ValueError, match="incompatible"):
        load_pretrained_mixers([layer], path, [3])
    assert torch.equal(layer.mixer.in_proj.weight, original)
    with pytest.raises(ValueError, match="match"):
        load_pretrained_mixers([layer], path, [0, 1])
