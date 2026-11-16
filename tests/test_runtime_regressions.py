"""Cross-layer contracts, tested without model inference or device access."""
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from b70_ablator.editor import Editor
from b70_ablator.phases import ModelPhases, PhasedHF


def tiny_modules():
    model = nn.Module()
    model.config = SimpleNamespace(model_type='llama', hidden_size=4)
    model.model = nn.Module()
    block = nn.Module()
    block.self_attn, block.mlp = nn.Module(), nn.Module()
    for name, outputs, inputs in [('o_proj',4,4),('q_proj',4,4),('k_proj',2,4),('v_proj',2,4)]:
        setattr(block.self_attn, name, nn.Linear(inputs, outputs, bias=False))
    for name, outputs, inputs in [('down_proj',4,8),('gate_proj',8,4),('up_proj',8,4)]:
        setattr(block.mlp, name, nn.Linear(inputs, outputs, bias=False))
    model.model.layers = nn.ModuleList([block])
    return model


def test_floor_crosses_the_real_phased_wrapper_without_loading_a_model():
    seen = []
    def generate(prompt, min_tokens=None):
        seen.append((prompt, min_tokens))
        return {'content':'fixture', 'sequence':torch.tensor([[1,2]])}
    owner = ModelPhases()
    owner.current_target = SimpleNamespace(generate=generate)
    response = PhasedHF(owner).generate('fixture', min_tokens=150)
    assert seen == [('fixture',150)]
    assert response['sequence'].device.type == 'cpu'


@pytest.mark.parametrize('method', ['mpoa','leace'])
def test_full_surface_applies_each_method_in_its_own_coordinate_space(method):
    editor = Editor(tiny_modules(), 1, edit_targets='full')
    # Deliberately oblique LEACE factors, not an orthogonal special case.
    left = torch.tensor([[[1.],[.5],[0.],[0.]]])
    right = torch.tensor([[[.8,.4,0.,0.]]])
    basis = torch.tensor([[[1.,0.,0.,0.]]])
    plan = {'method':method, 'basis':basis,
            'row_normalization':'full' if method=='mpoa' else 'none'}
    if method == 'leace':
        plan.update(left=left, right=right)
    params = {g:{'strength':.7,'center':.5,'width':1.,'floor':1.} for g in editor.groups}
    originals = {a.source_key:a.base.weight.detach().clone() for *_,a in editor.targets}
    editor.apply(plan, params)
    for *_, adapter in editor.targets:
        original = originals[adapter.source_key]
        if method == 'leace':
            delta = (original @ left[0]) @ right[0] if adapter.side=='input' else left[0] @ (right[0] @ original)
            expected = original - .7 * delta
        else:
            d = basis[0]
            delta = (original @ d.T) @ d if adapter.side=='input' else d.T @ (d @ original)
            expected = original - .7 * delta
            expected *= original.norm(dim=-1,keepdim=True) / expected.norm(dim=-1,keepdim=True)
        assert torch.allclose(adapter.effective_weight(),expected,atol=1e-6), adapter.source_key
        assert torch.equal(adapter.base.weight,original)
    editor.reset()
    assert all(not a.enabled for *_,a in editor.targets)
