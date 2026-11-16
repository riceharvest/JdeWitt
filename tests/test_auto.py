"""Automatic selection scheduling fixtures, never model-quality evidence."""
import copy
import json
from dataclasses import asdict
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
QUANTS=('Q4_K_M','Q5_K_M')


def settings(tmp_path):
    model=tmp_path/'model'
    model.mkdir()
    (model/'config.json').write_text('{"model_type":"llama"}')
    (model/'fixture.safetensors').write_bytes(b'NOT model weights: scheduling fixture')
    tool=tmp_path/'tool'
    tool.write_bytes(b'NOT executable: scheduling fixture')
    base=json.loads((ROOT/'examples/first-test/run-refusal-check.json').read_text())
    base.update(model=str(model),revision='fixture',output=str(tmp_path/'auto'),rank=1,
                converter=str(tool),quantizer=str(tool),converter_python=str(tool),llama_server=str(tool),
                trials=2,finalists=1,startup_trials=1,consistency={
                    'min_direction_examples':2,'min_direction_families':2,'min_direction_overlap':.8,
                    'min_refusal_families':2,'min_variants_per_family':2,'confidence':.9,'max_task_regressions':0})
    for name in ('directions','selection','holdout'):
        if name=='directions':
            rows=[dict(id=lang+str(f)+label,kind='direction',language=lang,prompt=lang+str(f)+label,
                       label=label,family_id='d'+str(f))
                  for lang in ('en','nl') for f in range(2) for label in ('good','bad')]
        else:
            rows=[dict(id=name+str(f)+str(v),kind='refusal',language='en',prompt=name+str(f)+str(v),family_id=name+str(f))
                  for f in range(2) for v in range(2)]+[
                  dict(id=name+'h',kind='harmless',language='en',prompt=name+'h'),
                  dict(id=name+'m',kind='math',language='en',prompt=name+'m',expected='4'),
                  dict(id=name+'c',kind='code',language='en',prompt=name+'c',tests='assert True')]
        path=tmp_path/(name+'.jsonl')
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        base[name]=str(path)
    return {'run':base,'total_trials':4,'candidates':[
        {'id':'mean','method':'directional','rank':1,'method_options':{},'trials':2},
        {'id':'subspace','method':'svd','rank':1,'method_options':{},'trials':2}]}


def result(rate):
    return {'scoring_mode':'refusal-check','refusal_check_failure_rate':rate,'invalid':0,
            'capability_pass_rate':1.,'by_language':{},'by_kind':{},
            'disclaimer_rate':None,'mean_preamble_chars':None,'paired':{'passes':True}}


def runner_fixture(events,interrupt=None,holdout_failure=False,changed_baseline=False):
    def runner(config,*,selection_only=False,expected_winner=None):
        from b70_ablator.answers import atomic_json
        from b70_ablator.errors import HoldoutRejected
        root=Path(config.output)
        name=root.name
        events.append((name,selection_only,expected_winner))
        if not selection_only:
            assert expected_winner=='0'
            if holdout_failure:
                raise HoldoutRejected('Fixture selected candidate failed sealed holdout')
            from b70_ablator.release import create_release
            artifacts=json.loads((root/'quantized-ranking.json').read_text())['candidates'][0]['artifacts']
            hf=root/'fixture-hf'
            hf.mkdir(exist_ok=True)
            (hf/'config.json').write_text('{"fixture":"NOT model weights"}')
            create_release(root/'release',hf,artifacts,{'fixture':True},'Scheduling fixture, NOT a trained model')
            return {'release':str(root/'release'),'winner':'0','holdout':{'fixture':True}}
        if interrupt==name and sum(e[0]==name for e in events)==1:
            raise ConnectionError('fixture transport interrupted')
        rate=.4 if name=='mean' else .1
        artifacts={}
        baseline_artifacts={}
        for q in QUANTS:
            file=root/(q+'.gguf')
            file.parent.mkdir(parents=True,exist_ok=True)
            file.write_bytes(('NOT GGUF: '+name+q).encode())
            from b70_ablator.storage import file_digest
            artifacts[q]={'path':str(file),'sha256':file_digest(file),'quant':q,'bytes':file.stat().st_size}
            baseline_artifacts[q]={'sha256':('different' if changed_baseline and name=='subspace' else 'same')+q}
        candidate={'id':'0','artifacts':artifacts,'quants':{q:result(rate) for q in QUANTS}}
        report={'baseline':{q:result(.8) for q in QUANTS},'baseline_artifacts':baseline_artifacts,
                'candidates':[candidate],'ranked_ids':['0']}
        atomic_json(root/'quantized-ranking.json',report)
        return {'status':'selection-complete',**report}
    return runner


@pytest.mark.parametrize('change',[{'total_trials':3},{'total_trials':True},
    {'candidates':[]},{'candidates':[{'id':'../escape','method':'svd','rank':1,'method_options':{},'trials':4}]}])
def test_auto_configuration_requires_an_explicit_valid_fixed_budget(tmp_path,change):
    from b70_ablator.auto import AutoConfig
    with pytest.raises(ValueError):
        AutoConfig.from_json(settings(tmp_path)|change)


def test_auto_runs_selection_for_all_methods_then_only_certifies_global_winner(tmp_path):
    from b70_ablator.auto import AutoConfig,run_auto
    events=[]
    config=AutoConfig.from_json(settings(tmp_path))
    result=run_auto(config,runner=runner_fixture(events))
    assert events==[('mean',True,None),('subspace',True,None),('subspace',False,'0')]
    assert result['winner']['method']=='svd'
    assert result['trial_budget']==4
    assert result['status']=='complete'
    assert not (Path(config.run.output)/'candidates'/'mean'/'release').exists()


def test_auto_resumes_completed_selection_without_rerunning_other_method(tmp_path):
    from b70_ablator.auto import AutoConfig,run_auto
    config=AutoConfig.from_json(settings(tmp_path))
    events=[]
    runner=runner_fixture(events,interrupt='subspace')
    with pytest.raises(ConnectionError):
        run_auto(config,runner=runner)
    result=run_auto(config,runner=runner)
    assert result['winner']['method']=='svd'
    assert sum(name=='mean' for name,_,_ in events)==1


def test_auto_holdout_failure_never_falls_back_to_another_method(tmp_path):
    from b70_ablator.auto import AutoConfig,run_auto
    from b70_ablator.errors import HoldoutRejected
    config=AutoConfig.from_json(settings(tmp_path))
    events=[]
    runner=runner_fixture(events,holdout_failure=True)
    with pytest.raises(HoldoutRejected):
        run_auto(config,runner=runner)
    before=list(events)
    with pytest.raises(HoldoutRejected):
        run_auto(config,runner=runner)
    assert events==before
    assert [name for name,selection,_ in events if not selection]==['subspace']


def test_auto_rejects_changed_inputs_before_reusing_selection(tmp_path):
    from b70_ablator.auto import AutoConfig,run_auto
    config=AutoConfig.from_json(settings(tmp_path))
    events=[]
    runner=runner_fixture(events,interrupt='subspace')
    with pytest.raises(ConnectionError):
        run_auto(config,runner=runner)
    Path(config.run.selection).write_text('changed in place')
    with pytest.raises(ValueError,match='identity|JSON|Expecting'):
        run_auto(config,runner=runner)
    assert len(events)==2


def test_auto_rejects_incomparable_quantized_baselines(tmp_path):
    from b70_ablator.auto import AutoConfig,run_auto
    config=AutoConfig.from_json(settings(tmp_path))
    events=[]
    with pytest.raises(ValueError,match='baseline'):
        run_auto(config,runner=runner_fixture(events,changed_baseline=True))
    assert all(selection for _,selection,_ in events)


def test_auto_rejects_corrupted_completed_selection_artifacts(tmp_path):
    from b70_ablator.auto import AutoConfig,run_auto
    config=AutoConfig.from_json(settings(tmp_path))
    events=[]
    runner=runner_fixture(events,interrupt='subspace')
    with pytest.raises(ConnectionError):
        run_auto(config,runner=runner)
    (Path(config.run.output)/'candidates'/'mean'/'Q4_K_M.gguf').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='integrity|changed'):
        run_auto(config,runner=runner)
    assert len(events)==2


def test_auto_configuration_does_not_mutate_the_callers_config(tmp_path):
    from b70_ablator.auto import AutoConfig
    raw=settings(tmp_path)
    before=copy.deepcopy(raw)
    parsed=AutoConfig.from_json(raw)
    assert raw==before
    assert [c.method for c in parsed.runs()]==['directional','svd']
    assert sum(c.trials for c in parsed.runs())==parsed.total_trials
    assert all(Path(c.output).parent.name=='candidates' for c in parsed.runs())
    assert asdict(parsed.run)['output']==raw['run']['output']


def test_completed_auto_run_is_reused_without_loading_any_model(tmp_path):
    from b70_ablator.auto import AutoConfig,run_auto
    config=AutoConfig.from_json(settings(tmp_path))
    events=[]
    runner=runner_fixture(events)
    first=run_auto(config,runner=runner)
    before=list(events)
    assert run_auto(config,runner=runner)==first
    assert events==before


def test_auto_resumes_the_sealed_winner_after_certification_interrupt(tmp_path):
    from b70_ablator.auto import AutoConfig,run_auto
    config=AutoConfig.from_json(settings(tmp_path))
    events=[]
    delegate=runner_fixture(events)
    def runner(child,**kwargs):
        if not kwargs.get('selection_only') and not any(not selection for _,selection,_ in events):
            events.append((Path(child.output).name,False,kwargs['expected_winner']))
            raise ConnectionError('fixture certification interrupted')
        return delegate(child,**kwargs)
    with pytest.raises(ConnectionError):
        run_auto(config,runner=runner)
    result=run_auto(config,runner=runner)
    assert result['winner']['candidate_id']=='subspace'
    assert events==[('mean',True,None),('subspace',True,None),('subspace',False,'0'),('subspace',False,'0')]


def test_auto_records_incompatible_methods_without_silently_spending_their_budget(tmp_path):
    from b70_ablator.auto import AutoConfig,run_auto
    from b70_ablator.errors import UnsupportedMethod
    config=AutoConfig.from_json(settings(tmp_path))
    events=[]
    delegate=runner_fixture(events)
    def runner(child,**kwargs):
        if Path(child.output).name=='mean':
            events.append(('mean',True,None))
            raise UnsupportedMethod('Fixture unsupported projection layout')
        return delegate(child,**kwargs)
    result=run_auto(config,runner=runner)
    assert result['winner']['outcomes'][0]['status']=='incompatible'
    assert result['trial_budget']==4
    assert len(events)==3


def test_auto_records_rejected_directions_without_hiding_runtime_failures(tmp_path):
    from b70_ablator.auto import AutoConfig,run_auto
    from b70_ablator.errors import DirectionRejected
    config=AutoConfig.from_json(settings(tmp_path))
    events=[]
    delegate=runner_fixture(events)
    def runner(child,**kwargs):
        if Path(child.output).name=='mean':
            raise DirectionRejected('Fixture unstable directions')
        return delegate(child,**kwargs)
    result=run_auto(config,runner=runner)
    assert result['winner']['outcomes'][0]['status']=='rejected'


def test_degenerate_direction_fitting_is_an_explicit_method_rejection():
    import torch
    from b70_ablator.numerics import directions_from_languages
    from b70_ablator.errors import DirectionRejected
    with pytest.raises(DirectionRejected,match='Degenerate'):
        directions_from_languages({'en':torch.zeros(1,3)},{'en':torch.zeros(1,3)},1)


def test_validate_auto_rejects_outputs_that_overlap_original_weights(tmp_path):
    from b70_ablator.auto import AutoConfig,validate_auto
    raw=settings(tmp_path)
    raw['run']['output']=str(Path(raw['run']['model'])/'experiment')
    with pytest.raises(ValueError,match='overlap'):
        validate_auto(AutoConfig.from_json(raw))
