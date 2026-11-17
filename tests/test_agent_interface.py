"""Agent operations must be machine-readable and must not load models."""
import json
import subprocess
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]


def invoke(*args):
    code="""import sys,importlib.abc
class NoModels(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in {'torch','transformers','bitsandbytes'}:
   raise AssertionError('Model runtime imported by metadata command: '+fullname)
sys.meta_path.insert(0,NoModels())
from jdewitt.cli import main
main()
"""
    return subprocess.run([sys.executable,'-c',code,*map(str,args)],cwd=ROOT,capture_output=True,text=True)


def test_validation_reports_bad_config_as_json_without_model_import(tmp_path):
    path=tmp_path/'config.json'
    path.write_text('{}')
    result=invoke('validate-run',path)
    assert result.returncode==1
    error=json.loads(result.stderr)
    assert error['error']['type']=='ValueError'
    assert 'configuration' in error['error']['message']


def test_argument_errors_are_json():
    result=invoke('not-a-command')
    assert result.returncode==2
    assert json.loads(result.stderr)['error']['type']=='ArgumentError'


def test_auto_cli_help_and_invalid_config_never_import_model_runtime(tmp_path):
    for command in ('auto','validate-auto'):
        help_result=invoke(command,'--help')
        assert help_result.returncode==0,help_result.stderr
        path=tmp_path/'invalid-auto.json'
        path.write_text('{}')
        result=invoke(command,path)
        assert result.returncode==1,result.stderr
        assert json.loads(result.stderr)['error']['type']=='ValueError'


def test_run_status_missing_directory_does_not_create_it(tmp_path):
    output=tmp_path/'absent'
    result=invoke('run-status',output)
    assert result.returncode==1
    assert not output.exists()
    assert json.loads(result.stderr)['error']['type']=='FileNotFoundError'


def test_run_status_reads_trials_without_mutating_database(tmp_path):
    from jdewitt.search import build_study
    from jdewitt.agent import run_status
    study=build_study(tmp_path/'study.db',{'fixture':True},7,1)
    trial=study.ask()
    trial.set_user_attr('feasible',True)
    study.tell(trial,.25)
    study.ask()
    from jdewitt.storage import file_digest
    before=file_digest(tmp_path/'study.db')
    status=run_status(tmp_path)
    assert status['trial_counts']=={'COMPLETE':1,'RUNNING':1}
    assert status['best_feasible']['number']==0
    assert status['best_feasible']['value']==.25
    assert status['release_verified'] is False
    assert before==file_digest(tmp_path/'study.db')


def test_validation_reuses_actual_suite_gates_and_never_creates_output(tmp_path):
    from jdewitt.search import RunConfig
    from jdewitt.validation import validate_run_inputs
    raw=json.loads((ROOT/'examples/first-test/run-refusal-check.json').read_text())
    raw['output']=str(tmp_path/'not-created')
    config=RunConfig.from_json(raw)
    with pytest.raises(ValueError,match='consistency'):
        validate_run_inputs(config)
    assert not Path(raw['output']).exists()


def test_catalog_exposes_recent_architecture_blockers_without_model_imports():
    result=invoke('architectures')
    assert result.returncode==0,result.stderr
    catalog=json.loads(result.stdout)
    inventory=catalog['recent_releases']
    assert inventory['since']=='2026-07-16'
    assert inventory['until']=='2026-10-16'
    for name,entry in inventory['architectures'].items():
        assert entry['adapter_registered']==(name in catalog['architectures'])
        assert not entry['full_checkpoint_b70_verified']
    assert inventory['architectures']['deepseek_v41']['status']=='not-supported'
    assert inventory['architectures']['llada2_moe']['limitation']


def test_validate_run_success_reads_local_metadata_only(tmp_path):
    # Deliberately invalid weight/tool bytes prove validation is metadata-only,
    # not a kernel, checkpoint-quality, converter or sandbox verification.
    source=tmp_path/'source'
    source.mkdir()
    (source/'config.json').write_text('{"model_type":"llama"}')
    (source/'model.safetensors').write_bytes(b'fixture, NOT a model')
    tool=tmp_path/'tool'
    tool.write_bytes(b'fixture, NOT an executable')
    raw=json.loads((ROOT/'examples/first-test/run-refusal-check.json').read_text())
    raw.update(model=str(source),output=str(tmp_path/'output'),converter=str(tool),quantizer=str(tool),
               converter_python=str(tool),llama_server=str(tool),rank=1,consistency={
                   'min_direction_examples':2,'min_direction_families':2,'min_direction_overlap':.9,
                   'min_refusal_families':2,'min_variants_per_family':2,'confidence':.9,'max_task_regressions':0})
    directions=[dict(id=lang+str(f)+label,kind='direction',language=lang,prompt=lang+str(f)+label,
                     label=label,family_id='d'+str(f))
                for lang in ('en','nl') for f in range(2) for label in ('good','bad')]
    for name in ('directions','selection','holdout'):
        rows=directions if name=='directions' else [dict(id=name+str(f)+str(v),kind='refusal',language='en',
              prompt=name+str(f)+str(v),family_id=name+str(f)) for f in range(2) for v in range(2)]+[
              dict(id=name+'h',kind='harmless',language='en',prompt=name+'h'),
              dict(id=name+'m',kind='math',language='en',prompt=name+'m',expected='4'),
              dict(id=name+'c',kind='code',language='en',prompt=name+'c',tests='assert True')]
        path=tmp_path/(name+'.jsonl')
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        raw[name]=str(path)
    config=tmp_path/'config.json'
    config.write_text(json.dumps(raw))
    result=invoke('validate-run',config)
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout)
    assert report['status']=='metadata-valid' and report['models_loaded'] is False
    assert report['suites']['directions']['tasks']==8
    assert report['direction_families']=={'en':2,'nl':2}
    assert not (tmp_path/'output').exists()
