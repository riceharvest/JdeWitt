from dataclasses import dataclass, field
from types import SimpleNamespace
from b70_ablator import pipeline


def fixture_policy():
    return {'min_direction_examples':2,'min_direction_families':2,'min_direction_overlap':.9,
            'min_refusal_families':2,'min_variants_per_family':2,'confidence':.9,'max_task_regressions':0}


def test_one_command_can_own_and_close_fixed_judge(tmp_path,monkeypatch):
    import json
    (tmp_path/'config.json').write_text(json.dumps({'model_type':'llama'}))
    @dataclass(frozen=True)
    class Config:
        model:str=str(tmp_path)
        revision:str='local'
        judge_url:str='managed'
        judge_model:str='ignored-when-managed'
        judge_artifact:str='fixed-judge.gguf'
        llama_server:str='llama-server'
        ctx_size:int=4096
        judge_max_tokens:int=1024
        output:str=str(tmp_path)
        consistency:dict=field(default_factory=fixture_policy)
        quantization:str='none'
    seen=[]
    class Manager:
        def __init__(self,*args):
            seen.append(('created',args))
        def __enter__(self):
            return SimpleNamespace(url='http://127.0.0.1:9999/v1',model='verified-fixed-hash')
        def __exit__(self,*args):
            seen.append(('closed',))
    monkeypatch.setattr(pipeline,'ManagedGGUF',Manager)
    monkeypatch.setattr(pipeline,'verify_judge',lambda c: {'url':c.judge_url,'model':c.judge_model})
    monkeypatch.setattr(pipeline,'_run',lambda c,phases=None: phases.judge())
    result=pipeline.run(Config())
    assert result=={'url':'http://127.0.0.1:9999/v1','model':'verified-fixed-hash'}
    assert seen[-1]==('closed',)


def test_unowned_external_judge_is_rejected_before_starting_search(tmp_path,monkeypatch):
    import json
    import pytest
    (tmp_path/'config.json').write_text(json.dumps({'model_type':'llama'}))
    config=SimpleNamespace(model=str(tmp_path),revision='local',judge_url='http://127.0.0.1:1234/v1',
                           consistency=fixture_policy(),quantization='none')
    monkeypatch.setattr(pipeline,'_run',lambda *args,**kwargs:pytest.fail('Unowned GPU model must not start search'))
    with pytest.raises(ValueError,match='managed'):
        pipeline.run(config)


def test_unsupported_local_architecture_never_starts_expensive_judge(tmp_path,monkeypatch):
    import json
    import pytest
    (tmp_path/'config.json').write_text(json.dumps({'model_type':'unknown_future_architecture'}))
    config=SimpleNamespace(model=str(tmp_path),revision='local',judge_url='managed',judge_artifact='judge.gguf',
                           llama_server='server',ctx_size=4096,judge_max_tokens=2048,output=str(tmp_path),
                           consistency=fixture_policy(),quantization='none')
    def forbid(*args):
        raise AssertionError('Expensive judge started before architecture preflight')
    monkeypatch.setattr(pipeline,'ManagedGGUF',forbid)
    with pytest.raises(ValueError,match='Unsupported model architecture'):
        pipeline.run(config)
