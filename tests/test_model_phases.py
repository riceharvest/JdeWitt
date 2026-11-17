"""Ownership/scheduling fixtures; not model compliance measurements."""
from contextlib import contextmanager
from types import SimpleNamespace
import pytest
import torch
from jdewitt.phases import ModelPhases, PhasedHF, PhasedJudge


class FixtureEditor:
    def __init__(self,events):
        self.events=events
        self.parameters=None
    def apply(self,directions,params):
        self.parameters=params
        self.events.append('apply-edit')
    def reset(self):
        self.parameters=None
        self.events.append('reset-edit')


class FixtureTarget:
    def __init__(self,events):
        self.events=events
        self.editor=FixtureEditor(events)
        self.device=torch.device('cpu')
        self.fingerprint={'fixture_target':'fixed'}
        self.tokenizer=object()
        events.append('load-target')
    def generate(self,prompt):
        return {'content':'fixture answer','finish_reason':'stop','thinking_open':False,
                'sequence':torch.tensor([[1,2,3]]),'prompt_length':1,'generated_ids':[2,3]}
    def residual_at_answer(self,prompt,response):
        return torch.ones(2,3)
    def release(self):
        self.events.append('release-target')


def factories(events):
    def target():
        assert events.count('load-judge')==events.count('release-judge')
        return FixtureTarget(events)
    @contextmanager
    def judge():
        assert events.count('load-target')==events.count('release-target')
        events.append('load-judge')
        try:
            yield SimpleNamespace(classify=lambda *args: 'fixture verdict')
        finally:
            events.append('release-judge')
    return target,judge


def test_phase_switch_releases_old_model_before_loading_next():
    events=[]
    target,judge=factories(events)
    with ModelPhases(target,judge) as phases:
        assert phases.target() is phases.target()
        assert phases.judge().classify()=='fixture verdict'
        phases.target()
    assert events==['load-target','release-target','load-judge','release-judge','load-target','release-target']


def test_reload_restores_edit_but_reset_does_not_reload_target():
    events=[]
    target,judge=factories(events)
    with ModelPhases(target,judge) as phases:
        backend=PhasedHF(phases)
        directions=torch.ones(2,1,3)
        parameters={'fixture_strength':0.5}
        backend.editor.apply(directions,parameters)
        phases.judge()
        backend.generate('fixture prompt')
        assert phases.target().editor.parameters==parameters
        phases.judge()
        before=events.count('load-target')
        backend.editor.reset()
        assert events.count('load-target')==before
        backend.generate('fixture prompt')
        assert phases.target().editor.parameters is None
    assert events.count('apply-edit')==2


def test_closed_phase_owner_cannot_restart_a_model():
    events=[]
    target,judge=factories(events)
    with ModelPhases(target,judge) as phases:
        phases.target()
    with pytest.raises(RuntimeError,match='closed'):
        phases.target()


def test_failure_while_loading_judge_leaves_target_released():
    events=[]
    target,_=factories(events)
    @contextmanager
    def broken():
        raise RuntimeError('fixture loading failure')
        yield
    with ModelPhases(target,broken) as phases:
        phases.target()
        with pytest.raises(RuntimeError,match='loading failure'):
            phases.judge()
        assert phases.current_target is None
        assert events==['load-target','release-target']


def test_metadata_does_not_reload_target_after_judge_takes_over():
    events=[]
    target,judge=factories(events)
    with ModelPhases(target,judge) as phases:
        backend=PhasedHF(phases)
        fingerprint=backend.fingerprint
        tokenizer=backend.tokenizer
        phases.judge()
        assert backend.device==torch.device('cpu')
        assert backend.fingerprint==fingerprint and backend.tokenizer is tokenizer
        assert events.count('load-target')==1


def test_judge_rubric_binding_does_not_load_any_model():
    events=[]
    target,judge=factories(events)
    records=[]
    binder=SimpleNamespace(bind_rubrics=lambda rows:records.extend(rows))
    with ModelPhases(target,judge) as phases:
        proxy=PhasedJudge(phases,binder)
        proxy.bind_rubrics([{'fixture':'rubric'}])
        assert records==[{'fixture':'rubric'}] and not events


@pytest.mark.xpu
def test_real_b70_weights_are_freed_between_phases():
    import gc
    from transformers import LlamaConfig,LlamaForCausalLM
    from jdewitt.hf import HFModel,require_b70
    from jdewitt.editor import Editor
    device=require_b70()
    # Remove prior tests' delayed Python references before taking the baseline.
    gc.collect()
    torch.xpu.synchronize()
    initial=torch.xpu.memory_allocated()
    snapshots=[]
    def target():
        model=LlamaForCausalLM(LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,
                                         num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)).to(device,dtype=torch.float16).eval()
        return HFModel.from_objects(model,None,Editor(model,1),8)
    @contextmanager
    def judge():
        torch.xpu.synchronize()
        snapshots.append(torch.xpu.memory_allocated())
        yield SimpleNamespace()
    with ModelPhases(target,judge) as phases:
        backend=PhasedHF(phases)
        directions=torch.zeros(2,1,32)
        directions[:,:,0]=1
        parameters={c:{'strength':.5,'center':.5,'width':1.,'floor':1.} for c in ('attn','mlp')}
        backend.editor.apply(directions,parameters)
        assert torch.xpu.memory_allocated()>initial
        phases.judge()
        assert snapshots==[initial]
        restored=phases.target()
        assert any(a.enabled for *_,a in restored.editor.targets)
    torch.xpu.synchronize()
    assert torch.xpu.memory_allocated()==initial
