import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM
from jdewitt.editor import Editor
from jdewitt.hf import HFModel, require_b70


@pytest.mark.xpu
def test_whole_reply_kl_does_not_reprefill_every_prefix():
    device=require_b70()
    model=LlamaForCausalLM(LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,pad_token_id=0,bos_token_id=1,eos_token_id=2)).to(device).eval()
    editor=Editor(model,1)
    backend=HFModel.from_objects(model,None,editor,8)
    sequence=torch.tensor([[1,7,8,9,10,11,12,13,14,15,16,17,18,19,20]],device=device)
    processed=[]
    def count(module,args,kwargs):
        ids=args[0] if args else kwargs['input_ids']
        processed.append(ids.shape[1])
    handle=model.register_forward_pre_hook(count,with_kwargs=True)
    try:
        result=backend.trajectory_kl(sequence,3,0,chunk_size=3)
    finally:
        handle.remove()
    assert result['tokens']==12
    assert result['full_reply_kl']<1e-7
    assert sum(processed)<=2*(sequence.shape[1]-1)
