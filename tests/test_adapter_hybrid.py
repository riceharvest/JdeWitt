import torch
import pytest
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5ForConditionalGeneration
from jdewitt.editor import discover_targets, Editor
from jdewitt.hf import HFModel, require_b70


def make_model():
    text=Qwen3_5TextConfig(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,linear_key_head_dim=16,linear_value_head_dim=16,linear_num_key_heads=2,linear_num_value_heads=4,layer_types=['linear_attention','full_attention'],eos_token_id=2,pad_token_id=0)
    vision={'depth':1,'hidden_size':32,'intermediate_size':64,'num_heads':2,'out_hidden_size':64,'num_position_embeddings':16,'patch_size':2,'spatial_merge_size':2,'temporal_patch_size':1,'deepstack_visual_indexes':[]}
    return Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=text,vision_config=vision,image_token_id=120,video_token_id=121,vision_start_token_id=122,vision_end_token_id=123)).eval()


def test_real_hybrid_definition_is_not_silently_skipped():
    model=make_model()
    layers,targets=discover_targets(model)
    assert len(layers)==2
    assert len(targets)==4
    assert targets[0][3].endswith('linear_attn.out_proj')
    assert targets[2][3].endswith('self_attn.o_proj')
    assert not any('visual' in path for _,_,_,path,_ in targets)


@pytest.mark.xpu
def test_hybrid_b70_parity_generation_and_full_reply_kl():
    torch.manual_seed(9)
    model=make_model()
    ids=torch.tensor([[1,7,8,9]])
    with torch.inference_mode():
        expected=model(ids,use_cache=False).logits
    device=require_b70()
    model=model.to(device)
    with torch.inference_mode():
        actual=model(ids.to(device),use_cache=False).logits.cpu()
    assert torch.allclose(expected,actual,atol=1e-4,rtol=1e-4)
    editor=Editor(model,1)
    backend=HFModel.from_objects(model,None,editor,8)
    directions=torch.zeros(2,1,64,device=device)
    directions[:,:,0]=1
    params={c:{'strength':.5,'center':.5,'width':1.,'floor':1.} for c in ('attn','mlp')}
    editor.apply(directions,params)
    with torch.inference_mode():
        sequence=model.generate(ids.to(device),max_new_tokens=4,do_sample=False)
    metrics=backend.trajectory_kl(sequence,4,0,chunk_size=2)
    assert metrics['full_reply_kl']>0
    assert all(p.device.type=='xpu' for p in model.parameters())
