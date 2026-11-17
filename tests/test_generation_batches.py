"""Budgeted batching contracts; all generators here are explicit fixtures."""
from types import SimpleNamespace
import pytest
import torch
from jdewitt.answers import AnswerStore, generate_answers
from jdewitt.hf import HFModel, BoundaryContract
from jdewitt.transport import ChatClient


def tasks():
    return [{'id':str(i),'kind':kind,'prompt':prompt} for i,kind,prompt in
            [(0,'refusal','long prompt words here'),(1,'math','two words'),
             (2,'refusal','short'),(3,'refusal','also short')]]


class BatchGenerator:
    max_tokens = 8
    def __init__(self):
        self.calls = []
    def prompt_tokens(self,prompt):
        return len(prompt.split())
    def generate(self,*args,**kwargs):
        raise AssertionError('batch policy must not use serial generate')
    def generate_batch(self,prompts,min_tokens=None):
        self.calls.append((list(prompts),min_tokens))
        for prompt in prompts:
            yield {'content':prompt,'finish_reason':'stop','floor_enforced':min_tokens is not None}


def test_budgeted_batches_keep_floor_groups_and_original_task_ids(tmp_path):
    suite = tasks()
    store = AnswerStore(tmp_path,suite,{'batch-fixture':1})
    gen = BatchGenerator()
    report = generate_answers(suite,gen,store,refusal_floor=4,
                              batch_policy={'max_sequences':3,'max_padded_tokens':20})
    assert report['generated']==4 and report['batches']==3
    assert any(prompts==['short','also short'] and floor==4 for prompts,floor in gen.calls)
    for prompts,floor in gen.calls:
        assert len(prompts)*(max(map(gen.prompt_tokens,prompts))+gen.max_tokens)<=20
        assert ('two words' not in prompts) if floor else prompts==['two words']
    assert all(store.get(t)['content']==t['prompt'] for t in suite)
    previous=list(gen.calls)
    report=generate_answers(suite,gen,store,refusal_floor=4,
                            batch_policy={'max_sequences':3,'max_padded_tokens':20})
    assert gen.calls==previous and report['reused']==4


def test_batch_interruption_preserves_each_yielded_answer(tmp_path):
    suite=tasks()[2:]
    store=AnswerStore(tmp_path,suite,{'batch-fixture':1})
    class Broken(BatchGenerator):
        def generate_batch(self,prompts,min_tokens=None):
            yield {'content':prompts[0],'finish_reason':'stop'}
            raise RuntimeError('fixture interrupted')
    policy={'max_sequences':2,'max_padded_tokens':40}
    with pytest.raises(RuntimeError,match='interrupted'):
        generate_answers(suite,Broken(),store,batch_policy=policy)
    assert store.status()['completed']==1
    gen=BatchGenerator()
    result=generate_answers(suite,gen,store,batch_policy=policy)
    assert result['generated']==1 and result['reused']==1
    assert gen.calls[0][0]==['short','also short'], 'resume must retain the original batch cohort'


@pytest.mark.parametrize('policy',[{'max_sequences':True,'max_padded_tokens':20},
                                   {'max_sequences':2}, {'max_sequences':2,'max_padded_tokens':1}])
def test_bad_or_too_small_batch_budget_fails_before_generation(tmp_path,policy):
    suite=tasks()
    gen=BatchGenerator()
    with pytest.raises(ValueError,match='batch|budget'):
        generate_answers(suite,gen,AnswerStore(tmp_path,suite,{'fixture':1}),batch_policy=policy)
    assert not gen.calls


def test_answer_progress_does_not_rescan_the_whole_ledger_per_prompt(tmp_path,monkeypatch):
    suite=tasks()
    store=AnswerStore(tmp_path,suite,{'fixture':1})
    old=store.get
    reads=[]
    def read(task):
        reads.append(task['id'])
        return old(task)
    monkeypatch.setattr(store,'get',read)
    class Serial:
        def generate(self,prompt):
            return {'content':prompt,'finish_reason':'stop'}
    generate_answers(suite,Serial(),store)
    assert len(reads)<=3*len(suite)


def fake_hf():
    target=HFModel.__new__(HFModel)
    target.device=torch.device('cpu')
    target.config=SimpleNamespace(model_type='llama',max_position_embeddings=64)
    target.ctx_size,target.max_tokens,target.final_marker=64,8,None
    target.boundary_contract=BoundaryContract.for_architecture('llama')
    class Tokens:
        pad_token_id=0
        eos_token_id=9
        def __call__(self,text,**kwargs):
            ids=[int(x) for x in text.split()]
            return {'input_ids':torch.tensor([ids]),'attention_mask':torch.ones(1,len(ids),dtype=torch.long)}
        def decode(self,ids,**kwargs):
            return ' '.join(map(str,ids))
    target.tokenizer=Tokens()
    target._render=lambda prompt:prompt
    calls=[]
    def generate(input_ids,attention_mask,**kwargs):
        calls.append((input_ids.clone(),attention_mask.clone(),kwargs))
        return torch.cat([input_ids,torch.tensor([[3,9,0],[4,5,9]])],dim=1)
    target.model=SimpleNamespace(generate=generate,generation_config=SimpleNamespace(eos_token_id=[9]))
    return target,calls


def test_hf_batches_left_padding_and_removes_padding_before_saved_trajectories():
    hf,calls=fake_hf()
    results=hf.generate_batch(['1 2','7'],min_tokens=1)
    ids,mask,kwargs=calls[0]
    assert ids.tolist()==[[1,2],[0,7]] and mask.tolist()==[[1,1],[0,1]]
    assert kwargs['min_new_tokens']==1 and kwargs['do_sample'] is False
    assert [r['prompt_length'] for r in results]==[2,1]
    assert [r['sequence'].tolist() for r in results]==[[[1,2,3,9]],[[7,4,5,9]]]
    assert [r['generated_ids'] for r in results]==[[3],[4,5]]
    assert all(r['finish_reason']=='stop' for r in results)


def test_hf_batch_preflights_all_rows_before_any_generate():
    hf,calls=fake_hf()
    hf.ctx_size=10
    with pytest.raises(ValueError,match='context'):
        hf.generate_batch(['1','1 2 3'])
    assert not calls


def test_loopback_batches_overlap_requests_but_restore_input_order(monkeypatch):
    import threading
    client=ChatClient('http://127.0.0.1:9/v1','fixture',8)
    barrier=threading.Barrier(2)
    def generate(prompt,min_tokens=None):
        barrier.wait(timeout=3)
        return {'content':prompt,'floor':min_tokens}
    monkeypatch.setattr(client,'generate',generate)
    assert list(client.generate_batch(['a','b'],min_tokens=4))==[
        {'content':'a','floor':4},{'content':'b','floor':4}]


def test_silent_backend_floor_loss_is_rejected_before_persisting(tmp_path):
    suite=tasks()[:1]
    store=AnswerStore(tmp_path,suite,{'fixture':1})
    class LostFloor:
        def generate(self,prompt,min_tokens=None):
            return {'content':'fixture','finish_reason':'stop'}
    with pytest.raises(ValueError,match='floor'):
        generate_answers(suite,LostFloor(),store,refusal_floor=4)
    assert store.status()['completed']==0


def test_server_parallelism_keeps_the_per_request_context_budget(tmp_path,monkeypatch):
    from jdewitt.release import ManagedGGUF
    monkeypatch.setattr('jdewitt.release.load_mode_flags',lambda _:['--load-mode','none'])
    manager=ManagedGGUF(tmp_path/'fixture.gguf','never-executed',128,32,tmp_path/'log',parallel=3)
    command=manager.command('Vulkan0',12345,'fixture')
    assert command[command.index('--parallel')+1]=='3'
    assert command[command.index('--ctx-size')+1]=='384'
    assert command[command.index('--gpu-layers')+1]=='999'


def test_batch_and_cache_run_settings_are_explicit_and_validated():
    import json
    from pathlib import Path
    from jdewitt.search import RunConfig
    raw=json.loads((Path(__file__).parents[1]/'examples/harsh-corpus/run.json').read_text())
    config=RunConfig.from_json({**raw,'generation_batch':{'max_sequences':3,'max_padded_tokens':8192},
                              'merged_cache_bytes':4096})
    assert config.generation_batch['max_sequences']==3 and config.merged_cache_bytes==4096
    for changed in ({'generation_batch':{'max_sequences':True,'max_padded_tokens':4}},
                    {'merged_cache_bytes':-1},{'merged_cache_bytes':True}):
        with pytest.raises(ValueError,match='batch|cache'):
            RunConfig.from_json({**raw,**changed})
