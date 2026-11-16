"""Negative controls with tiny CPU models, not evidence of ablation quality."""
import pytest
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from b70_ablator.editor import Editor


def fixture():
    torch.manual_seed(71)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,
        eos_token_id=2,pad_token_id=0)).eval()
    ids=torch.tensor([[1,7,8,9]])
    directions=torch.zeros(2,1,32)
    directions[:,:,0]=1
    params={c:{'strength':.73,'center':.5,'width':1.,'floor':1.} for c in ('attn','mlp')}
    return model,ids,directions,params


def test_zero_edit_and_reset_match_baseline_logits_tokens_and_original_storage():
    model,ids,directions,params=fixture()
    originals=[(p,p.detach().clone(),p.data_ptr()) for p in model.parameters()]
    with torch.inference_mode():
        baseline=model(ids,use_cache=False).logits
        tokens=model.generate(ids,max_new_tokens=4,do_sample=False)
    editor=Editor(model,1)
    zero={k:{**v,'strength':0.} for k,v in params.items()}
    for _ in range(3):
        editor.apply(directions,params)
        with torch.inference_mode():
            assert not torch.equal(model(ids,use_cache=False).logits,baseline)
        editor.apply(directions,zero)
        with torch.inference_mode():
            assert torch.equal(model(ids,use_cache=False).logits,baseline)
            assert torch.equal(model.generate(ids,max_new_tokens=4,do_sample=False),tokens)
        editor.reset()
    assert all(torch.equal(p,original) and p.data_ptr()==pointer for p,original,pointer in originals)


@pytest.mark.parametrize('change',['zero-width','nan-center','missing-component','nonfinite-direction','wrong-rank','nonorthogonal'])
def test_invalid_edits_fail_closed_without_partial_active_layers(change):
    model,_,directions,params=fixture()
    editor=Editor(model,1)
    if change=='zero-width':
        params['mlp']['width']=0.
    elif change=='nan-center':
        params['mlp']['center']=float('nan')
    elif change=='missing-component':
        del params['mlp']
    elif change=='nonfinite-direction':
        directions[-1,0,0]=float('nan')
    elif change=='wrong-rank':
        directions=torch.zeros(2,2,32)
    else:
        directions[-1,0,0]=2.
    with pytest.raises(ValueError):
        editor.apply(directions,params)
    assert not any(a.enabled for *_,a in editor.targets)
