import json
import pytest
import torch
from safetensors.torch import load_file
from b70_ablator.export import stream_merge
from b70_ablator.editor import Editor, AblatedLinear
from transformers import LlamaConfig, LlamaForCausalLM


@pytest.mark.xpu
def test_stream_merge_preserves_original_and_does_not_reload_full_model(tmp_path):
    model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)).to('xpu',dtype=torch.float16).eval()
    source = tmp_path / 'source'
    source.mkdir()
    model.save_pretrained(source)
    original = load_file(source / 'model.safetensors')
    editor = Editor(model,1)
    d = torch.zeros(2,1,32,device='xpu')
    d[:,:,0]=1
    params = {c:{'strength':.5,'center':.5,'width':1.,'floor':1.} for c in ('attn','mlp')}
    editor.apply(d,params)
    output = tmp_path / 'merged'
    stream_merge(source,output,editor,max_shard_bytes=20000)
    index = json.loads((output/'model.safetensors.index.json').read_text())
    tensors={}
    for shard in set(index['weight_map'].values()):
        tensors.update(load_file(output/shard))
    key='model.layers.0.self_attn.o_proj.weight'
    a=editor.targets[0][-1]
    expected=original[key].float()+a.B.float().cpu() @ a.A.float().cpu()
    assert torch.allclose(tensors[key].float(),expected,atol=.001)
    assert torch.equal(load_file(source/'model.safetensors')[key],original[key])
    assert set(tensors)==set(original)
    assert all(p.device.type=='xpu' for p in model.parameters())


@pytest.mark.xpu
def test_nf4_projection_dequantizes_on_actual_b70():
    import bitsandbytes as bnb
    linear=bnb.nn.Linear4bit(64,32,bias=False,compute_dtype=torch.bfloat16,quant_type='nf4')
    linear=linear.to('xpu')
    adapter=AblatedLinear(linear,rank=1)
    d=torch.zeros(1,32,device='xpu')
    d[0,0]=1
    adapter.set_directions(d,.5)
    assert adapter.A.dtype==torch.bfloat16
    out=adapter(torch.randn(1,64,device='xpu',dtype=torch.bfloat16))
    assert torch.isfinite(out).all()
    assert out.device.type=='xpu'
    assert linear.weight.dtype==torch.uint8
