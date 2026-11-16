"""Command-line framework tests use labeled fixtures, never live model claims."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from b70_ablator import cli
from b70_ablator.answers import AnswerStore, generate_answers


def fixture_suite(tmp_path):
    rows=[{'id':'m','kind':'math','language':'en','prompt':'Two plus two?','expected':'4'}]
    path=tmp_path/'suite.jsonl'
    path.write_text(json.dumps(rows[0])+'\n')
    return rows,path


@pytest.mark.parametrize('command',['generate','score','status'])
def test_real_console_commands_are_available(command):
    binary=Path(sys.executable).parent/'b70-ablator'
    result=subprocess.run([str(binary),command,'--help'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert command in result.stdout


def test_status_reads_answer_count_without_starting_a_model(tmp_path,monkeypatch,capsys):
    rows,path=fixture_suite(tmp_path)
    store=AnswerStore(tmp_path/'answers',rows,{'fixture':'status'})
    store.put(rows[0],{'content':'4','finish_reason':'stop'})
    monkeypatch.setattr(sys,'argv',['b70-ablator','status','--answers',str(store.directory)])
    cli.main()
    result=json.loads(capsys.readouterr().out)
    assert result['completed']==1 and result['missing']==0


def test_generate_closes_target_before_score_command_starts_judge(tmp_path,monkeypatch,capsys):
    from b70_ablator import job_cli
    rows,path=fixture_suite(tmp_path)
    model=tmp_path/'fixture-model.gguf'
    server=tmp_path/'fixture-server'
    judge=tmp_path/'fixture-judge.gguf'
    for p in (model,server,judge):
        p.write_text('Fixture identity bytes, not a model or executable')
    events=[]
    class Manager:
        def __init__(self,artifact,*args):
            self.kind='target' if Path(artifact)==model else 'judge'
        def __enter__(self):
            events.append('open-'+self.kind)
            return SimpleNamespace(url='http://127.0.0.1:9999/v1',model='fixture-model',
                                   generate=lambda prompt:{'content':'4','finish_reason':'stop'})
        def __exit__(self,*args):
            events.append('close-'+self.kind)
    class FixedJudge:
        def calibrate(self,*args):
            events.append('calibrated-judge')
            return {'accuracy':1.,'total':1}
    monkeypatch.setattr(job_cli,'ManagedGGUF',Manager)
    monkeypatch.setattr(job_cli,'verify_judge',lambda args:FixedJudge())
    monkeypatch.setattr(job_cli,'sandbox_identity',lambda image:'fixture-sandbox-image-id')
    monkeypatch.setattr(sys,'argv',['b70-ablator','generate','--model',str(model),'--server',str(server),
        '--suite',str(path),'--max-tokens','32','--ctx-size','128','--output',str(tmp_path/'answers')])
    cli.main()
    capsys.readouterr()
    calibration=tmp_path/'calibration.json'
    calibration.write_text(json.dumps([{'language':'en'}]))
    monkeypatch.setattr(sys,'argv',['b70-ablator','score','--answers',str(tmp_path/'answers'),
        '--suite',str(path),'--server',str(server),'--judge-artifact',str(judge),'--calibration',str(calibration),
        '--judge-max-tokens','32','--ctx-size','128','--judge-min-confidence','0',
        '--judge-min-accuracy','1','--sandbox-image','fixture-sandbox','--output',str(tmp_path/'scores')])
    cli.main()
    report=json.loads(capsys.readouterr().out)
    assert report['capability_pass_rate']==1
    assert events==['open-target','close-target','open-judge','calibrated-judge','close-judge']
    monkeypatch.setattr(job_cli,'ManagedGGUF',lambda *args:pytest.fail('Completed score job must not reload the judge'))
    cli.main()
    assert json.loads(capsys.readouterr().out)==report


def test_score_job_rejects_missing_answers_before_loading_judge(tmp_path,monkeypatch):
    from b70_ablator import job_cli
    rows,path=fixture_suite(tmp_path)
    store=AnswerStore(tmp_path/'answers',rows,{'fixture':'missing'})
    monkeypatch.setattr(job_cli,'ManagedGGUF',lambda *args:pytest.fail('Judge must not start'))
    args=SimpleNamespace(answers=store.directory,suite=path)
    with pytest.raises(ValueError,match='missing'):
        job_cli.score_job(args)


def test_answer_job_fixture_is_reused_without_generator(tmp_path):
    rows,_=fixture_suite(tmp_path)
    store=AnswerStore(tmp_path/'answers',rows,{'fixture':'already-saved'})
    store.put(rows[0],{'content':'4','finish_reason':'stop'})
    result=generate_answers(rows,None,store)
    assert result['generated']==0 and result['reused']==1


def test_generate_cli_forwards_floor_batch_policy_and_resume_identity(tmp_path,monkeypatch,capsys):
    from b70_ablator import job_cli
    rows=[{'id':'r','kind':'refusal','language':'en','prompt':'Fixture request.'}]
    path=tmp_path/'refusals.jsonl'
    path.write_text(json.dumps(rows[0])+'\n')
    model,server=tmp_path/'model.gguf',tmp_path/'server'
    model.write_text('fixture only')
    server.write_text('fixture only')
    seen=[]
    class Manager:
        def __init__(self,*args,parallel=1):
            assert parallel==2
        def __enter__(self):
            def generate(prompts,min_tokens=None):
                seen.append((prompts,min_tokens))
                return [{'content':'fixture answer','finish_reason':'stop','floor_enforced':True}]
            return SimpleNamespace(max_tokens=32,prompt_tokens=lambda p:4,generate_batch=generate)
        def __exit__(self,*args):
            pass
    monkeypatch.setattr(job_cli,'ManagedGGUF',Manager)
    monkeypatch.setattr(sys,'argv',['b70-ablator','generate','--model',str(model),'--server',str(server),
        '--suite',str(path),'--max-tokens','32','--ctx-size','128','--output',str(tmp_path/'answers'),
        '--refusal-floor','8','--batch-max-sequences','2','--batch-token-budget','128'])
    cli.main()
    assert json.loads(capsys.readouterr().out)['generated']==1
    assert seen==[(['Fixture request.'],8)]
    identity=json.loads((tmp_path/'answers/manifest.json').read_text())['identity']
    assert identity['refusal_floor']==8 and identity['generation_batch']['max_sequences']==2
    monkeypatch.setattr(job_cli,'ManagedGGUF',lambda *a,**k:pytest.fail('Completed jobs must not load'))
    cli.main()
    assert json.loads(capsys.readouterr().out)['reused']==1


def test_refusal_only_cli_needs_no_judge_arguments_or_model(tmp_path,monkeypatch,capsys):
    from b70_ablator import job_cli
    rows=[{'id':'r','kind':'refusal','language':'en','prompt':'Fixture request.'}]
    path=tmp_path/'suite.jsonl'
    path.write_text(json.dumps(rows[0])+'\n')
    store=AnswerStore(tmp_path/'answers',rows,{'fixture':'refusal-only'})
    store.put(rows[0],{'content':'I cannot assist with that request.','finish_reason':'stop'})
    monkeypatch.setattr(job_cli,'ManagedGGUF',lambda *args:pytest.fail('Fast check must never load a model'))
    monkeypatch.setattr(job_cli,'verify_judge',lambda *args:pytest.fail('Fast check must never verify a judge'))
    monkeypatch.setattr(sys,'argv',['b70-ablator','score','--mode','refusal-check',
        '--answers',str(store.directory),'--suite',str(path),'--output',str(tmp_path/'scores')])
    cli.main()
    result=json.loads(capsys.readouterr().out)
    assert result['refusals']==1 and result['noncompliance_rate'] is None
    assert not (tmp_path/'scores'/'judge-calibration.json').exists()
