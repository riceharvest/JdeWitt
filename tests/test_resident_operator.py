"""Operator parity tests; tiny random models are mechanics, not quality evidence."""
import pytest
import torch
from torch import nn
from b70_ablator.editor import AblatedLinear, Editor
from b70_ablator.hf import require_b70


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_resident_operator_uses_export_rounding_and_keeps_base_immutable(dtype):
    torch.manual_seed(17)
    base = nn.Linear(64, 32, bias=True, dtype=dtype)
    adapter = AblatedLinear(base, 2, operator='merged-resident')
    direction = torch.linalg.qr(torch.randn(32, 2)).Q.T
    adapter.set_directions(direction, .73)
    original = base.weight.detach().clone()
    pointer = base.weight.data_ptr()
    x = torch.randn(3, 64, dtype=dtype)
    weight = (original.float() + adapter.B.float() @ adapter.A.float()).to(dtype)
    expected = nn.functional.linear(x, weight, base.bias)
    live = base(x) + nn.functional.linear(nn.functional.linear(x, adapter.A), adapter.B)
    assert not torch.equal(live, expected), 'Fixture must expose the rounding mismatch'
    assert torch.equal(adapter(x), expected)
    assert torch.equal(base.weight, original) and base.weight.data_ptr() == pointer
    adapter.reset()
    assert torch.equal(adapter(x), base(x))


def test_resident_precision_preflight_rejects_search_export_dtype_mismatch(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM
    from b70_ablator.export import require_resident_precision
    model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)).to(dtype=torch.float16)
    model.save_pretrained(tmp_path)
    editor = Editor(model, 1)
    assert require_resident_precision(tmp_path, editor) > 0
    model.to(dtype=torch.bfloat16)
    with pytest.raises(ValueError, match='precision'):
        require_resident_precision(tmp_path, editor)


def test_resident_operator_rejects_quantized_base_without_original_weights():
    base = nn.Linear(4, 3)
    base.weight.quant_state = object()
    with pytest.raises(ValueError, match='original.*precision'):
        AblatedLinear(base, 1, operator='merged-resident')


@pytest.mark.xpu
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_actual_b70_resident_operator_never_offloads_or_changes_weights(dtype, monkeypatch):
    device = require_b70()
    torch.manual_seed(17)
    base = nn.Linear(256, 128, bias=True, device=device, dtype=dtype).eval()
    adapter = AblatedLinear(base, 2, operator='merged-resident')
    d = torch.linalg.qr(torch.randn(128, 2)).Q.T.to(device)
    adapter.set_directions(d, .73)
    original = base.weight.detach().clone()
    pointer = base.weight.data_ptr()
    x = torch.randn(4, 256, device=device, dtype=dtype)
    expected = nn.functional.linear(x, (original.float() + adapter.B.float() @ adapter.A.float()).to(dtype), base.bias)
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU transfer in resident trial forward')
    monkeypatch.setattr(torch.Tensor, 'cpu', forbidden)
    with torch.inference_mode():
        for _ in range(3):
            assert torch.equal(adapter(x), expected)
    assert torch.equal(original, base.weight) and pointer == base.weight.data_ptr()
    assert all(p.device.type == 'xpu' for p in adapter.parameters())
    torch.xpu.synchronize()


@pytest.mark.xpu
def test_actual_b70_resident_logits_match_reloaded_streamed_export(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM
    from b70_ablator.export import stream_merge
    import gc
    device = require_b70()
    torch.manual_seed(23)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)).to(device, dtype=torch.bfloat16).eval()
    source = tmp_path / 'source'
    model.save_pretrained(source)
    editor = Editor(model, 1, operator='merged-resident')
    d = torch.zeros(2, 1, 32, device=device)
    d[:, :, 0] = 1
    params = {c: {'strength': .73, 'center': .5, 'width': 1., 'floor': 1.} for c in ('attn', 'mlp')}
    editor.apply(d, params)
    ids = torch.tensor([[1, 7, 8, 9]], device=device)
    with torch.inference_mode():
        expected = model(ids, use_cache=False).logits.detach().clone()
    output = tmp_path / 'export'
    stream_merge(source, output, editor)
    from b70_ablator.export import verify_resident_export
    audit = verify_resident_export(source, output, editor)
    assert audit['edited_tensors'] == 4 and audit['operator'] == 'merged-resident'
    from safetensors.torch import load_file, save_file
    import json
    index = json.loads((output/'model.safetensors.index.json').read_text())
    key = 'model.layers.0.self_attn.o_proj.weight'
    shard = output/index['weight_map'][key]
    original_shard = load_file(shard)
    changed = {**original_shard, key: original_shard[key] + 1}
    save_file(changed, shard)
    with pytest.raises(ValueError, match='export'):
        verify_resident_export(source, output, editor)
    save_file(original_shard, shard)
    del editor, model
    gc.collect()
    torch.xpu.empty_cache()
    loaded = LlamaForCausalLM.from_pretrained(output, dtype=torch.bfloat16, device_map={'': device}).eval()
    with torch.inference_mode():
        actual = loaded(ids, use_cache=False).logits
    assert torch.equal(expected, actual)
    assert all(p.device.type == 'xpu' for p in loaded.parameters())
    torch.xpu.synchronize()
