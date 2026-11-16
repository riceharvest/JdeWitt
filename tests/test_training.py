"""Training tests use tiny random fixtures, not refusal-removal quality data."""
import pytest
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from b70_ablator.editor import Editor
from b70_ablator.training import completion_logprob,dpo_loss,train_adapters


def test_completion_logprob_masks_prompt_tokens_and_dpo_prefers_chosen():
    logits=torch.zeros(1,4,6)
    ids=torch.tensor([[1,2,3,4]])
    expected=torch.log_softmax(logits[:,:-1],-1)[0,1:,0].sum()
    assert torch.equal(completion_logprob(logits,ids,2),expected)
    good=dpo_loss(torch.tensor(2.),torch.tensor(0.),torch.tensor(0.),torch.tensor(0.),.2)
    bad=dpo_loss(torch.tensor(0.),torch.tensor(2.),torch.tensor(0.),torch.tensor(0.),.2)
    assert good<bad
    with pytest.raises(ValueError):
        completion_logprob(logits,ids,4)


@pytest.mark.parametrize('mode',['sft','dpo'])
def test_training_updates_only_edit_factors_and_real_loss(mode,monkeypatch):
    # Torch 2.14 checks accelerator graph health even for CPU parameters.
    # This fixture must not initialize the real B70.
    monkeypatch.setattr(torch.accelerator,'current_accelerator',lambda **kwargs:None)
    torch.manual_seed(7)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1)).eval()
    editor=Editor(model,2)
    before={name:p.detach().clone() for name,p in model.named_parameters()}
    row={'ids':torch.tensor([[1,5,6,7]]),'prompt_length':2}
    if mode=='dpo':
        row['rejected_ids']=torch.tensor([[1,5,9,10]])
    result=train_adapters(model,editor,[row],mode=mode,steps=3,learning_rate=.02,beta=.1,seed=7)
    assert len(result['losses'])==3 and all(torch.isfinite(torch.tensor(result['losses'])))
    assert any(a.B.abs().sum()>0 for *_,a in editor.targets)
    assert all(torch.equal(p,before[name]) for name,p in model.named_parameters())
    assert result['losses'][-1]<result['losses'][0]


@pytest.mark.parametrize('changed',[{'steps':0},{'rank':True},{'seed':-1},{'learning_rate':float('nan')},
                                    {'beta':0.},{'ctx_size':8},{'trust_remote_code':'yes'},{'mode':[]},{'dtype':[]}])
def test_training_rejects_invalid_settings_before_model_loading(changed):
    from b70_ablator.training import validate_training_config
    config=dict(mode='sft',model='/reviewed/local/model',revision=None,rank=1,ctx_size=64,max_tokens=8,
                dtype='bfloat16',trust_remote_code=False,data='train.jsonl',holdout='holdout.jsonl',
                steps=1,learning_rate=.01,beta=.1,seed=7,output='new-output')
    with pytest.raises(ValueError):
        validate_training_config(config|changed)
