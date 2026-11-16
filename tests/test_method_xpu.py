"""Actual B70 operators on small fixtures, NOT model refusal-quality evidence."""
import pytest
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from b70_ablator.editor import Editor
from b70_ablator.hf import require_b70
from b70_ablator.pipeline import configure_compiler_paths
from b70_ablator.methods import fit_ara
from b70_ablator.plans import fit_plan,move_plan
from b70_ablator.scoring import ConsistencyPolicy
from b70_ablator.training import train_adapters
from b70_ablator.export import stream_merge,verify_resident_export

pytestmark=pytest.mark.xpu


def model_and_editor():
    configure_compiler_paths()
    device=require_b70()
    torch.manual_seed(7)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1)).to(device=device,dtype=torch.bfloat16).eval()
    return model,Editor(model,1)


def examples():
    rows=[]
    for language in ('en','nl'):
        for family in range(6):
            good=torch.full((2,16),.1)+torch.arange(16).expand(2,-1)*(.001*(family+1))
            bad=good.clone()
            bad[:,0]+=2
            bad[:,1]+=.1*family
            for label,values in [('good',good),('bad',bad)]:
                rows.append(({'language':language,'family_id':str(family),'label':label},values))
    return rows


@pytest.mark.parametrize('method',['directional','projected','mpoa','svd','whitened-svd','som','leace'])
def test_b70_method_plan_forward_and_verified_stream_export(tmp_path,method):
    model,editor=model_and_editor()
    # Save originals, not the wrapped editor module state.
    source=tmp_path/'source'
    editor.merge()
    model.save_pretrained(source)
    editor=Editor(model,1)
    options={'ridge':.1} if method in {'leace','whitened-svd'} else {'som_steps':4,'som_learning_rate':.2} if method=='som' else {}
    rows=examples()
    plan,_=fit_plan(rows,1,ConsistencyPolicy(2,2,.3,2,2,.9,0),7,[t for t,_ in rows],method,options)
    ids=torch.tensor([[1,5,6]],device='xpu')
    with torch.no_grad():
        baseline=model(ids).logits
        params={k:{'strength':.5,'center':.5,'width':1.,'floor':1.} for k in ('attn','mlp')}
        editor.apply(move_plan(plan,'xpu'),params)
        edited=model(ids).logits
        assert torch.isfinite(edited).all() and not torch.equal(edited,baseline)
        stream_merge(source,tmp_path/'edited',editor)
        audit=verify_resident_export(source,tmp_path/'edited',editor)
        assert audit['edited_tensors']==len(editor.targets)
        editor.reset()
        assert torch.equal(model(ids).logits,baseline)


@pytest.mark.parametrize('mode',['sft','dpo'])
def test_b70_trains_real_edit_factors_and_verifies_export(tmp_path,mode):
    model,editor=model_and_editor()
    editor.merge()
    source=tmp_path/'source'
    model.save_pretrained(source)
    editor=Editor(model,1)
    row={'ids':torch.tensor([[1,5,6,7]]),'rejected_ids':torch.tensor([[1,5,9,10]]),'prompt_length':2}
    result=train_adapters(model,editor,[row],mode,2,.01,.1,7)
    assert all(torch.isfinite(torch.tensor(result['losses'])))
    assert any(bool(a.B.abs().sum()>0) for *_,a in editor.targets)
    stream_merge(source,tmp_path/'edited',editor)
    assert verify_resident_export(source,tmp_path/'edited',editor)['edited_tensors']==len(editor.targets)


def test_b70_ara_optimizer_updates_real_same_device_factors():
    configure_compiler_paths()
    device=require_b70()
    weight=torch.eye(3,device=device)
    good=torch.tensor([[1.,0.,0.],[1.,.1,0.]])
    bad=torch.tensor([[0.,1.,0.],[0.,1.,.1]])
    options={'steps':2,'learning_rate':.3,'neighbors':2,'preserve':1.,'steer':1.,'overcorrect':.1,'repulsion_cap':1.}
    a,b,report=fit_ara(weight,good,good,bad,bad,1,options,7)
    assert a.device.type==b.device.type=='xpu'
    assert b.abs().sum()>0 and report['final_loss']<report['initial_loss']
