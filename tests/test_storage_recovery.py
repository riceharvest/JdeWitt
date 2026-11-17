"""Filesystem fault injection. Fixtures are not model or quality measurements."""
import errno
import json
from types import SimpleNamespace
import pytest
import torch
from jdewitt import pipeline, answers


def test_pipeline_atomic_json_preserves_old_file_when_commit_fails(tmp_path, monkeypatch):
    path=tmp_path/'state.json'
    answers.atomic_json(path,{'value':'old'})
    before=path.read_bytes()
    def fail(*args,**kwargs):
        raise OSError(errno.ENOSPC,'Fixture disk full')
    monkeypatch.setattr(answers.os,'replace',fail)
    with pytest.raises(OSError):
        pipeline.atomic_json(path,{'value':'new'})
    assert path.read_bytes()==before
    assert not list(tmp_path.glob('*.new')) and not list(tmp_path.glob('.*.new'))


def test_tensor_checkpoint_commits_payload_and_audit_with_one_pointer(tmp_path):
    from jdewitt import storage
    path=tmp_path/'directions.pt'
    identity={'fixture':'fixed source and settings'}
    data=torch.arange(6).reshape(2,3)
    storage.save_checkpoint(path,data,identity,{'accepted':4})
    restored,audit=storage.load_checkpoint(path,identity)
    assert torch.equal(restored,data) and audit=={'accepted':4}
    with pytest.raises(ValueError,match='identity'):
        storage.load_checkpoint(path,{'fixture':'changed'})
    pointer=json.loads(path.with_suffix('.checkpoint.json').read_text())
    (tmp_path/pointer['payload']).write_bytes(b'Corrupt fixture payload')
    with pytest.raises(ValueError,match='integrity'):
        storage.load_checkpoint(path,identity)


def test_checkpoint_failed_pointer_commit_never_trusts_orphan_payload(tmp_path,monkeypatch):
    from jdewitt import storage
    path=tmp_path/'directions.pt'
    identity={'fixture':'same'}
    original=storage.atomic_json
    def fail_pointer(destination,value):
        if str(destination).endswith('.checkpoint.json'):
            raise OSError(errno.ENOSPC,'Fixture disk full before pointer commit')
        return original(destination,value)
    monkeypatch.setattr(storage,'atomic_json',fail_pointer)
    with pytest.raises(OSError):
        storage.save_checkpoint(path,torch.ones(2,3),identity,{'fixture':'audit'})
    assert storage.load_checkpoint(path,identity) is None
    monkeypatch.setattr(storage,'atomic_json',original)
    storage.save_checkpoint(path,torch.zeros(2,3),identity,{'fixture':'recovered'})
    data,audit=storage.load_checkpoint(path,identity)
    assert torch.equal(data,torch.zeros(2,3)) and audit['fixture']=='recovered'


def test_checkpoint_truncated_write_never_replaces_previous_valid_state(tmp_path,monkeypatch):
    from jdewitt import storage
    path=tmp_path/'references.pt'
    identity={'fixture':'same'}
    storage.save_checkpoint(path,torch.ones(2),identity,{})
    def partial(value,destination):
        destination.write(b'Partial fixture torch file')
        raise OSError(errno.ENOSPC,'Fixture disk full during payload write')
    monkeypatch.setattr(torch,'save',partial)
    with pytest.raises(OSError):
        storage.save_checkpoint(path,torch.zeros(2),identity,{})
    data,_=storage.load_checkpoint(path,identity)
    assert torch.equal(data,torch.ones(2))


def test_interrupted_invalid_score_retry_can_resume_without_old_report_conflict(tmp_path):
    tasks=[{'id':'r','kind':'refusal','language':'en','prompt':'Fixture answer request'}]
    store=answers.AnswerStore(tmp_path/'answers',tasks,{'fixture':'generator'})
    store.put(tasks[0],{'content':'Fixture answer','finish_reason':'stop'})
    class Invalid:
        def classify(self,*args):
            raise ValueError('Fixture uncertain judge')
    class Disconnected:
        def classify(self,*args):
            raise ConnectionError('Fixture retry disconnect')
    from jdewitt.scoring import Judgment
    valid=SimpleNamespace(classify=lambda *args:Judgment('compliance',False,0,1.,'Fixture evidence'))
    directory=tmp_path/'scores'
    identity={'fixture':'same judge'}
    answers.score_answers(tasks,store,Invalid(),directory,identity)
    with pytest.raises(ConnectionError):
        answers.score_answers(tasks,store,Disconnected(),directory,identity,retry_invalid=True)
    result=answers.score_answers(tasks,store,valid,directory,identity,retry_invalid=True)
    assert result['invalid']==0 and result['noncompliance_rate']==0
    assert list((directory/'history').glob('*report.json'))
