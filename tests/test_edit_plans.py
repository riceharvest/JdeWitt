"""Drive real edit plans and their boundary/validation contracts."""
import pytest
import torch
from torch import nn
from transformers import LlamaConfig,LlamaForCausalLM
from jdewitt.editor import Editor
from jdewitt.methods import fit_directions
from jdewitt.plans import fit_plan,move_plan
from jdewitt.scoring import ConsistencyPolicy,extract_answer
from jdewitt.hf import BoundaryContract
from jdewitt.erasure import leace_factors


def data():
    g=torch.Generator().manual_seed(42)
    rows=[]
    for lang in ('en','nl'):
        for family in range(8):
            good=torch.randn(2,8,generator=g)
            bad=good+torch.tensor([[2.,.2,0,0,0,0,0,0]]).expand(2,-1)
            for label,value in [('good',good),('bad',bad)]:
                rows.append(({'language':lang,'family_id':str(family),'label':label},value))
    return rows


def test_leace_operator_erases_binary_mean_contrast_not_just_svd_direction():
    rows=data()
    good=torch.stack([x for t,x in rows if t['label']=='good'])
    bad=torch.stack([x for t,x in rows if t['label']=='bad'])
    left,right=leace_factors(good,bad,.1)
    p=torch.eye(8).expand(2,-1,-1)-left@right
    delta=(bad.mean(0)-good.mean(0)).unsqueeze(-1)
    assert torch.allclose(p@delta,torch.zeros_like(delta),atol=1e-5)
    assert torch.allclose(p@p,p,atol=1e-5)
    assert fit_directions(rows,1,'leace',{'ridge':.1},7).shape==(2,1,8)


def test_all_registered_architectures_have_explicit_boundary_contracts():
    from jdewitt.architectures import ARCHITECTURES
    for name in ARCHITECTURES:
        assert BoundaryContract.for_architecture(name).architecture==name


def test_harmony_final_channel_is_shared_by_scoring_and_residual_extraction():
    raw='<|channel|>analysis<|message|>thinking<|end|><|start|>assistant<|channel|>final<|message|>answer'
    marker='<|channel|>final<|message|>'
    class Tokenizer:
        def decode(self,ids,**kwargs):return raw[:len(ids)]
    boundary=BoundaryContract.for_architecture('gpt_oss').boundary(Tokenizer(),list(range(len(raw))),False,marker)
    assert raw[boundary:]=='answer'
    assert extract_answer(raw,'stop',final_marker=marker).text=='answer'


def test_ara_plan_factors_survive_edit_reset_and_reapplication():
    model=LlamaForCausalLM(LlamaConfig(vocab_size=16,hidden_size=8,intermediate_size=16,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1)).eval()
    editor=Editor(model,2)
    rows=data()
    io=[]
    for task,_ in rows:
        captures={}
        for *_,a in editor.targets:
            x=torch.zeros(a.A.shape[-1])
            x[0 if task['label']=='good' else 1]=1
            captures[a.source_key]=(x,nn.functional.linear(x,a.base.weight.detach(),a.base.bias))
        io.append((task,captures))
    policy=ConsistencyPolicy(2,2,.5,2,2,.9,0)
    options={'steps':1,'learning_rate':.2,'neighbors':1,'preserve':1.,'steer':1.,'overcorrect':.1,'repulsion_cap':1.}
    plan,_=fit_plan(rows,2,policy,7,[t for t,_ in rows],'ara',options,io,editor)
    params={k:{'strength':.5,'center':.5,'width':1.,'floor':1.} for k in ('attn','mlp')}
    ids=torch.tensor([[1,2,3]])
    editor.apply(move_plan(plan,'cpu'),params)
    with torch.no_grad():
        before=model(ids).logits
    editor.reset()
    editor.apply(move_plan(plan,'cpu'),params)
    with torch.no_grad():
        after=model(ids).logits
    assert torch.equal(before,after)


def test_ascii_harmony_and_unfinished_analysis_fail_closed():
    raw='<\u007cchannel\u007c>analysis<\u007cmessage\u007c>thinking<\u007cchannel\u007c>final<\u007cmessage\u007c>answer'
    class Tokenizer:
        def decode(self,ids,**kwargs):return raw[:len(ids)]
    contract=BoundaryContract.for_architecture('gpt_oss')
    boundary=contract.boundary(Tokenizer(),list(range(len(raw))),False)
    assert raw[boundary:]=='answer'
    raw='<\u007cchannel\u007c>analysis<\u007cmessage\u007c>unfinished'
    with pytest.raises(ValueError,match='final'):
        contract.boundary(Tokenizer(),list(range(len(raw))),False)


def test_hyperconnection_residuals_use_explicit_backbone_width():
    from jdewitt.hf import residual_vector
    from types import SimpleNamespace
    config=SimpleNamespace(model_type='qwen4_exp_text',hidden_size=4,hc_count=2)
    hidden=torch.arange(8.).reshape(1,1,8)
    assert torch.equal(residual_vector(hidden,config),torch.tensor([2.,3.,4.,5.]))
    with pytest.raises(ValueError,match='width'):
        residual_vector(hidden,SimpleNamespace(model_type='qwen3',hidden_size=4))


def test_unverified_ascii_analysis_channel_is_not_a_plain_answer():
    class Tokenizer:
        def decode(self,ids,**kwargs):
            return '<\u007cchannel\u007c>analysis<\u007cmessage\u007c>unfinished'
    with pytest.raises(ValueError,match='channel'):
        BoundaryContract.for_architecture('qwen3').boundary(Tokenizer(),[1],False)
