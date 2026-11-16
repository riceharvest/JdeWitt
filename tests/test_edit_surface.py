"""The edit surface: which projections are edited, and on which side."""
import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM
from b70_ablator.editor import AblatedLinear, Editor, MAX_STRENGTH, discover_targets


def llama():
    torch.manual_seed(11)
    return LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=64, eos_token_id=2, pad_token_id=0)).eval()


def test_input_side_projection_removes_the_component_the_module_reads():
    base = nn.Linear(8, 4, bias=False)
    w = base.weight.detach().clone()
    layer = AblatedLinear(base, rank=1, side='input')
    d = torch.zeros(1, 8)
    d[0, 0] = 1.
    x = torch.randn(3, 8)
    layer.set_directions(d, strength=1.)
    # dW = -s * (W d.T) d, so the input's component along d is scaled by (1-s).
    expected = nn.functional.linear(x - (x @ d.T) @ d, w)
    assert torch.allclose(layer(x), expected, atol=1e-6)
    assert torch.equal(base.weight, w)
    # At full strength the direction is gone from the input entirely, so an input
    # that is exactly d must produce the same output as a zero input.
    assert torch.allclose(layer(d.repeat(3, 1)), layer(torch.zeros(3, 8)), atol=1e-6)
    # Half strength scales that component rather than removing it.
    layer.set_directions(d, strength=.5)
    half = nn.functional.linear(x - .5 * (x @ d.T) @ d, w)
    assert torch.allclose(layer(x), half, atol=1e-6)


def test_input_side_rejects_a_direction_of_the_wrong_width():
    layer = AblatedLinear(nn.Linear(8, 4), rank=1, side='input')
    with pytest.raises(ValueError, match='Invalid projection'):
        layer.set_directions(torch.zeros(1, 4), strength=.5)


def test_input_side_full_normalization_preserves_edited_row_norms():
    layer = AblatedLinear(nn.Linear(8, 4), rank=1, side='input')
    d = torch.zeros(1, 8)
    d[0, 0] = 1.
    original = layer.base.weight.detach().clone()
    layer.set_directions(d, strength=.5, row_normalization='full')
    assert torch.allclose(layer.effective_weight().norm(dim=-1),original.norm(dim=-1),atol=1e-6)


def test_writeback_is_the_default_surface():
    model = llama()
    editor = Editor(model, 1)
    assert editor.edit_targets == 'writeback'
    assert editor.groups == ['attn', 'mlp']
    paths = [adapter.source_path for *_, adapter in editor.targets]
    assert paths == ['model.layers.0.self_attn.o_proj', 'model.layers.0.mlp.down_proj',
                     'model.layers.1.self_attn.o_proj', 'model.layers.1.mlp.down_proj']


def test_full_surface_covers_every_residual_read_and_write_projection():
    model = llama()
    editor = Editor(model, 1, edit_targets='full')
    assert editor.groups == ['attn', 'attn_in', 'mlp', 'mlp_in']
    assert editor.group_sides == {'attn': 'output', 'attn_in': 'input',
                                  'mlp': 'output', 'mlp_in': 'input'}
    paths = [adapter.source_path for *_, adapter in editor.targets]
    for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'):
        assert any(p.endswith(name) for p in paths), name
    # Every layer carries all seven, twice per layer.
    assert len(editor.targets) == 2 * 7
    assert editor.layers_with['attn_in'] == [0, 1]
    assert [len(editor.layers_with[g]) for g in editor.groups] == [2, 2, 2, 2]


def test_grouped_query_k_and_v_are_editable_on_the_input_side_only():
    model = llama()
    layers = discover_targets(model, 'full')[0]
    # hidden 32 with 4 heads of dim 8 and 2 kv heads: k/v project to 16, not 32,
    # so the write-back operator cannot express them. The read side can.
    assert layers[0].self_attn.k_proj.out_features == 16
    assert layers[0].self_attn.k_proj.in_features == 32
    editor = Editor(model, 1, edit_targets='full')
    by_path = {adapter.source_path: adapter for *_, adapter in editor.targets}
    assert by_path['model.layers.0.self_attn.k_proj'].side == 'input'
    assert by_path['model.layers.0.self_attn.o_proj'].side == 'output'
    # The direction is 32 wide on both sides even though k_proj writes only 16.
    assert by_path['model.layers.0.self_attn.k_proj'].direction_shape() == (1, 32)
    assert by_path['model.layers.0.self_attn.o_proj'].direction_shape() == (1, 32)
    assert by_path['model.layers.0.self_attn.k_proj'].A.shape == (1, 32)
    assert by_path['model.layers.0.self_attn.k_proj'].B.shape == (16, 1)


def test_apply_requires_a_parameter_block_for_every_group():
    model = llama()
    editor = Editor(model, 1, edit_targets='full')
    directions = torch.zeros(2, 1, 32)
    directions[:, :, 0] = 1
    full = {g: {'strength': .5, 'center': .5, 'width': 1., 'floor': 1.} for g in editor.groups}
    editor.apply(directions, full)
    assert all(adapter.enabled for *_, adapter in editor.targets)
    editor.reset()
    with pytest.raises(ValueError, match='exactly'):
        editor.apply(directions, {g: full[g] for g in ('attn', 'mlp')})


def test_read_and_write_groups_are_tuned_independently():
    model = llama()
    editor = Editor(model, 1, edit_targets='full')
    directions = torch.zeros(2, 1, 32)
    directions[:, :, 0] = 1
    params = {g: {'strength': 0., 'center': .5, 'width': 1., 'floor': 1.} for g in editor.groups}
    params['attn_in'] = {'strength': 1., 'center': .5, 'width': 1., 'floor': 1.}
    editor.apply(directions, params)
    for *_, adapter in editor.targets:
        if adapter.side == 'input' and adapter.source_path.endswith('q_proj'):
            assert adapter.enabled and adapter.B.abs().sum() > 0
        elif adapter.side == 'output':
            assert not adapter.enabled


@pytest.mark.parametrize('strength,ok', [(0., True), (2., True), (6., True), (6.01, False), (-.1, False)])
def test_strength_bound_is_the_hard_ceiling_not_the_search_default(strength, ok):
    layer = AblatedLinear(nn.Linear(8, 4), rank=1)
    d = torch.zeros(1, 4)
    d[0, 0] = 1.
    if ok:
        layer.set_directions(d, strength)
        assert layer.enabled == (strength != 0)
    else:
        with pytest.raises(ValueError, match='Invalid projection'):
            layer.set_directions(d, strength)
    assert MAX_STRENGTH == 6.


def test_editor_accepts_a_strength_the_old_ceiling_rejected():
    model = llama()
    editor = Editor(model, 1)
    directions = torch.zeros(2, 1, 32)
    directions[:, :, 0] = 1
    params = {g: {'strength': 4., 'center': .5, 'width': 1., 'floor': 1.} for g in editor.groups}
    editor.apply(directions, params)
    assert all(adapter.enabled for *_, adapter in editor.targets)


def test_unknown_edit_surface_is_rejected():
    model = llama()
    with pytest.raises(ValueError, match='edit_targets'):
        Editor(model, 1, edit_targets='everything')


@pytest.mark.xpu
def test_full_surface_survives_streamed_export_on_the_b70(tmp_path):
    """Read-side edits are a new device operator; prove them through a real export."""
    from transformers import LlamaConfig, LlamaForCausalLM
    from b70_ablator.export import stream_merge, verify_resident_export
    from b70_ablator.hf import require_b70
    import gc
    device = require_b70()
    torch.manual_seed(29)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=64, eos_token_id=2, pad_token_id=0)).to(device, dtype=torch.bfloat16).eval()
    source = tmp_path / 'source'
    model.save_pretrained(source)
    editor = Editor(model, 1, operator='merged-resident', edit_targets='full')
    assert len(editor.targets) == 14
    directions = torch.zeros(2, 1, 32, device=device)
    directions[:, :, 0] = 1
    params = {g: {'strength': .73, 'center': .5, 'width': 1., 'floor': 1.} for g in editor.groups}
    editor.apply(directions, params)
    ids = torch.tensor([[1, 7, 8, 9]], device=device)
    with torch.inference_mode():
        expected = model(ids, use_cache=False).logits.detach().clone()
    output = tmp_path / 'export'
    stream_merge(source, output, editor)
    audit = verify_resident_export(source, output, editor)
    assert audit['edited_tensors'] == 14
    del editor, model
    gc.collect()
    torch.xpu.empty_cache()
    loaded = LlamaForCausalLM.from_pretrained(output, dtype=torch.bfloat16, device_map={'': device}).eval()
    with torch.inference_mode():
        actual = loaded(ids, use_cache=False).logits
    assert torch.equal(expected, actual)
    assert all(p.device.type == 'xpu' for p in loaded.parameters())
    torch.xpu.synchronize()
