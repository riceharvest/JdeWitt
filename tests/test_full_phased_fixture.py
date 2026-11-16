"""Full pipeline scheduling fixture. Exported fixture bytes are NOT model weights."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import optuna
import pytest
from b70_ablator import pipeline
from b70_ablator.release import sha256_file
from b70_ablator.scoring import Judgment
from b70_ablator.search import RunConfig


@pytest.mark.parametrize('scoring_mode,interrupt,corrupt_export,selection_only',[
    ('judge',False,False,False),('judge',True,False,False),('refusal-check',False,False,False),
    ('refusal-check',False,True,False),('refusal-check',False,False,True),('refusal-check',False,False,'mismatch')])
def test_full_fixture_run_never_overlaps_target_and_judge(tmp_path,monkeypatch,scoring_mode,interrupt,corrupt_export,selection_only):
    source=tmp_path/'fixture-source'
    source.mkdir()
    (source/'config.json').write_text(json.dumps({'model_type':'llama'}))
    (source/'fixture.safetensors').write_bytes(b'Explicit fixture bytes, NOT real model weights')
    judge_path=tmp_path/'fixture-judge.gguf'
    judge_path.write_bytes(b'Explicit fixture judge identity, NOT a real model')
    tool=tmp_path/'fixture-tool'
    tool.write_bytes(b'Fixture tool identity, never executed')
    tasks={}
    def write_suite(name,rows):
        path=tmp_path/(name+'.jsonl')
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        tasks.update({r['prompt']:r for r in rows})
        return str(path)
    directions=[{'id':lang+family+label,'kind':'direction','language':lang,'prompt':lang+' '+family+' '+label,
                 'label':label,'family_id':'direction-'+family,'requirements':['Fixture content']}
                for lang in ('en','nl') for family in ('a','b') for label in ('good','bad')]
    def suite(prefix):
        ordinary = [{'id':prefix+lang+kind,'kind':kind,'language':lang,'prompt':prefix+' '+lang+' '+kind,
                 **({'requirements':['Fixture content']} if kind=='refusal' else
                    {'expected':'4'} if kind=='math' else
                    {'tests':'assert f(3)==4'} if kind=='code' else {})}
                for lang in ('en','nl') for kind in ('code','math','harmless')]
        return ordinary + [{'id':prefix+lang+str(f)+str(v), 'kind':'refusal','language':lang,
                            'prompt':f'{prefix} {lang} refusal {f} {v}', 'family_id':f'{prefix}-family-{f}',
                            'requirements':['Fixture content']}
                           for lang in ('en','nl') for f in range(4) for v in range(2)]
    calibration=tmp_path/'fixture-calibration.json'
    calibration.write_text(json.dumps([{'language':'en'},{'language':'nl'}]))
    raw_config={'model':str(source),'revision':'fixture','model_license':'fixture-only-NOT-model-weights',
        'directions':write_suite('directions',directions),'selection':write_suite('selection',suite('s')),
        'holdout':write_suite('holdout',suite('h')),'calibration':str(calibration),
        'judge_url':'managed','judge_model':'fixture','judge_artifact':str(judge_path),
        'judge_max_tokens':32,'judge_min_confidence':0.,'judge_min_accuracy':1.,'max_tokens':32,'ctx_size':128,
        'trials':1,'startup_trials':1,'finalists':1,'rank':1,'seed':7,'max_full_reply_kl':.1,
        'max_capability_drop':0.,'vram_headroom_gib':.01,'converter':str(tool),'quantizer':str(tool),
        'converter_python':str(tool),'llama_server':str(tool),'sandbox_image':'python:3.12-slim',
        'output':str(tmp_path/'fixture-run'),'scoring_mode':scoring_mode,'refusal_floor':8,
        'consistency':{'min_direction_examples':2,'min_direction_families':2,'min_direction_overlap':.9,
                       'min_refusal_families':2,'min_variants_per_family':2,'confidence':.9,'max_task_regressions':0}}
    if scoring_mode=='refusal-check':
        raw_config['generation_batch']={'max_sequences':3,'max_padded_tokens':256}
        for key in list(raw_config):
            if key.startswith('judge_') or key=='calibration':
                del raw_config[key]
    config=RunConfig.from_json(raw_config)
    live={'target':False,'judge':False,'gguf':False}
    events=[]
    target_generations=[]
    calibration_calls=[]
    pending={'interrupt':interrupt}
    def response(prompt,edited=False):
        task=tasks[prompt]
        refusal='Sorry, I cannot assist with that request.'
        content=('fixture compliance' if edited else refusal) if task['kind']=='refusal' else (
            ('fixture good' if task['label']=='good' else refusal) if task['kind']=='direction' else
            'def f(x):\n    return x + 1' if task['kind']=='code' else '4' if task['kind']=='math' else 'fixture harmless answer')
        return {'content':content,'finish_reason':'stop','thinking_open':False,'completion_tokens':3,
                'sequence':torch.tensor([[1,2,3,4,5]]),'prompt_length':2,'generated_ids':[3,4,5]}
    class Editor:
        def __init__(self):
            self.edited=False
            self.targets=[('fixture',)]
            self.groups=['attn','mlp']
            self.edit_targets='writeback'
        def apply(self,*args):
            self.edited=True
        def reset(self):
            self.edited=False
    class Target:
        def __init__(self,*args):
            assert not any(live.values())
            live['target']=True
            events.append('load-target')
            self.device=torch.device('cpu')
            self.fingerprint={'architecture':'llama','rank':1,'dtype':'fixture','quantization':'none'}
            self.tokenizer=SimpleNamespace(decode=lambda ids,**kwargs:'fixture answer')
            self.editor=Editor()
            self.max_tokens=32
        def prompt_tokens(self,prompt):
            return len(prompt.split())
        def generate_batch(self,prompts,min_tokens=None):
            events.append('target-batch')
            return [self.generate(prompt,min_tokens=min_tokens) for prompt in prompts]
        def generate(self,prompt,min_tokens=None):
            assert live['target'] and not live['judge']
            assert min_tokens == (8 if tasks[prompt]['kind']=='refusal' else None)
            target_generations.append((prompt,self.editor.edited))
            return {**response(prompt,self.editor.edited),'floor_enforced':min_tokens is not None}
        def residual_at_answer(self,prompt,answer):
            return torch.ones(2,4)*(1 if tasks[prompt]['label']=='good' else 0)
        def trajectory_kl(self,sequence,*args):
            return {'full_reply_kl':.01,'answer_kl':.01,'tokens':3,'answer_tokens':3}
        def release(self):
            live['target']=False
            events.append('release-target')
    class Judge:
        client=SimpleNamespace(verify=lambda:None)
        def bind_rubrics(self,*args):
            pass
        def calibrate(self,*args):
            calibration_calls.append('fixture check')
            return {'total':2,'accuracy':1.}
        def classify(self,prompt,answer,language):
            assert live['judge'] and not live['target'] and not live['gguf']
            if answer=='fixture compliance' and pending['interrupt']:
                pending['interrupt']=False
                raise ConnectionError('Fixture trial judge disconnected')
            verdict='refusal' if answer=='Sorry, I cannot assist with that request.' else 'compliance'
            return Judgment(verdict,False,0,1.,'Explicit fixture judgment')
    class Manager:
        def __init__(self,artifact,*args,parallel=1):
            self.judge=Path(artifact)==judge_path
            self.edited='finalist-' in str(artifact)
            assert parallel==(1 if self.judge or scoring_mode=='judge' else 3)
        def __enter__(self):
            assert not any(live.values())
            kind='judge' if self.judge else 'gguf'
            live[kind]=True
            events.append('load-'+kind)
            def generate(prompt,min_tokens=None):
                assert min_tokens == (8 if tasks[prompt]['kind']=='refusal' else None)
                return {**response(prompt,self.edited),'floor_enforced':min_tokens is not None}
            def batch(prompts,min_tokens=None):
                events.append('gguf-batch')
                return [generate(prompt,min_tokens=min_tokens) for prompt in prompts]
            return SimpleNamespace(url='http://127.0.0.1:9999/v1',model='fixture',generate=generate,
                generate_batch=batch,prompt_tokens=lambda prompt:len(prompt.split()),max_tokens=32)
        def __exit__(self,*args):
            kind='judge' if self.judge else 'gguf'
            live[kind]=False
            events.append('release-'+kind)
    def merge(source,output,editor):
        assert live['target'] and editor.targets
        output.mkdir(parents=True)
        (output/'fixture.safetensors').write_bytes(b'Fixture export only, NOT model weights')
        return {'fixture':True}
    def convert(source,output,*args):
        output.mkdir(parents=True)
        result={}
        for quant in ('Q4_K_M','Q5_K_M'):
            p=output/(quant+'.gguf')
            p.write_bytes(b'Fixture export only, NOT a GGUF model')
            result[quant]={'path':str(p),'quant':quant,'sha256':sha256_file(p),'bytes':p.stat().st_size}
        return result
    monkeypatch.setattr(pipeline,'HFModel',Target)
    monkeypatch.setattr(pipeline,'ManagedGGUF',Manager)
    monkeypatch.setattr(pipeline,'verify_judge',lambda config:Judge())
    monkeypatch.setattr(pipeline,'require_b70',lambda:None)
    monkeypatch.setattr(pipeline,'preflight_hf',lambda *args:{'fixture':'explicit CPU scheduling, not real token counts'})
    # This is CPU-only scheduling coverage. Never initialize the actual GPU.
    monkeypatch.setattr(torch.xpu,'mem_get_info',lambda *args:(100*1024**3,100*1024**3))
    monkeypatch.setattr(pipeline,'stream_merge',merge)
    monkeypatch.setattr(pipeline,'require_resident_precision',lambda *args:0)
    def verify_export(*args):
        events.append('verify-resident-export')
        if corrupt_export:
            raise ValueError('Fixture export differs from resident operator')
        return {'operator':'merged-resident','edited_tensors':1,'shard_sha256':{'fixture':'not-real'}}
    monkeypatch.setattr(pipeline,'verify_resident_export',verify_export)
    monkeypatch.setattr(pipeline,'export_gguf',convert)
    if corrupt_export:
        with pytest.raises(ValueError,match='differs from resident'):
            pipeline.run(config)
        assert not (Path(config.output)/'release').exists()
        assert events.count('load-gguf')==0 and not any(live.values())
        return
    if selection_only=='mismatch':
        with pytest.raises(ValueError,match='Sealed winner changed'):
            pipeline.run(config,expected_winner='not-the-selected-trial')
        assert not (Path(config.output)/'baseline-quant-holdout').exists()
        assert not (Path(config.output)/'release').exists()
        assert not any(live.values())
        return
    if interrupt:
        with pytest.raises(ConnectionError,match='disconnected'):
            pipeline.run(config)
        assert not any(live.values())
        study=optuna.load_study(study_name='b70-ablator',storage='sqlite:///'+str(Path(config.output)/'study.db'))
        assert len(study.trials)==1 and study.trials[0].state==optuna.trial.TrialState.RUNNING
        original=list(target_generations)
    result=pipeline.run(config,selection_only=True) if selection_only else pipeline.run(config)
    if selection_only:
        assert result['status']=='selection-complete'
        assert result['ranked_ids']==['0']
        assert not (Path(config.output)/'release').exists()
        assert not (Path(config.output)/'baseline-quant-holdout').exists()
        assert not (Path(config.output)/'finalist-0'/'holdout').exists()
        assert events.count('load-gguf')==4 and not any(live.values())
        return
    if interrupt:
        assert target_generations[:len(original)]==original
        assert len(target_generations)-len(original)==len(suite('s'))
        assert calibration_calls==['fixture check']
    assert Path(result['release']).is_dir()
    pointer=json.loads((Path(config.output)/'directions.checkpoint.json').read_text())
    answers=json.loads((Path(config.output)/'trial-0'/'answers'/'manifest.json').read_text())
    bound=answers['identity']['edit_identity']
    assert bound['plan_sha256']==pointer['sha256']
    assert bound['derivation']['method']==config.method
    assert bound['derivation']['method_options']==(config.method_options or {})
    assert bound['derivation']['consistency']==config.consistency
    assert not any(live.values())
    assert events.count('load-target')==events.count('release-target')
    assert events.count('load-judge')==events.count('release-judge')
    assert events.count('load-gguf')==events.count('release-gguf')
    assert events.count('load-gguf')==8
    assert events.count('verify-resident-export')==1
    report=json.loads((Path(config.output)/'trial-0'/'trial-report.json').read_text())
    metadata=json.loads((Path(result['release'])/'evaluation.json').read_text())
    assert metadata['winner']['export_verification']['operator']=='merged-resident'
    for folder in ('baseline-quant-selection','baseline-quant-holdout','finalist-0/selection','finalist-0/holdout'):
        for quant in ('Q4_K_M','Q5_K_M'):
            manifest=json.loads((Path(config.output)/folder/quant/'answers/manifest.json').read_text())
            assert manifest['identity']['refusal_floor']==8
            assert manifest['identity']['generation_batch']==config.generation_batch
    assert all(r['paired']['passes'] for r in metadata['holdout'].values())
    if scoring_mode=='refusal-check':
        assert 'target-batch' in events and 'gguf-batch' in events
        assert not calibration_calls and events.count('load-judge')==0
        assert report['noncompliance_rate'] is None and report['refusal_check_failure_rate']==0.
        assert metadata['status']=='refusal-screened'
    else:
        assert report['noncompliance_rate']==0.
