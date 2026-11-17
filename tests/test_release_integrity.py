"""Release file integrity fixtures, never proof of actual model quality."""
import errno
import json
import sys
from pathlib import Path
import subprocess
import pytest
from jdewitt import release


def inputs(tmp_path):
    hf=tmp_path/'fixture-hf'
    hf.mkdir()
    (hf/'model.safetensors').write_bytes(b'Explicit fixture, NOT model weights')
    (hf/'config.json').write_text('{"fixture":true}')
    artifacts={}
    for q in release.QUANTS:
        path=tmp_path/(q+'.gguf')
        path.write_bytes(b'Explicit fixture, NOT GGUF '+q.encode())
        artifacts[q]={'path':str(path),'sha256':release.sha256_file(path)}
    return hf,artifacts,{'purpose':'file integrity fixture, not a model release'}


def build(tmp_path):
    hf,artifacts,metadata=inputs(tmp_path)
    destination=tmp_path/'release'
    release.create_release(destination,hf,artifacts,metadata,'Fixture README, no quality claim')
    return destination,hf,artifacts,metadata


def test_release_files_are_independent_from_working_artifacts(tmp_path):
    destination,hf,artifacts,_=build(tmp_path)
    for source,target in [(hf/'model.safetensors',destination/'hf'/'model.safetensors')]+[
        (Path(a['path']),destination/('model-'+q+'.gguf')) for q,a in artifacts.items()]:
        assert source.stat().st_ino!=target.stat().st_ino
        before=target.read_bytes()
        source.write_bytes(b'Changed working fixture')
        assert target.read_bytes()==before
    assert release.verify_release(destination)['valid'] is True


@pytest.mark.parametrize('change',['altered','missing','extra','symlink','index-traversal'])
def test_verify_release_rejects_corruption_or_unexpected_files(tmp_path,change):
    destination,_,_,_=build(tmp_path)
    path=destination/'hf'/'model.safetensors'
    if change=='altered':
        path.write_bytes(b'Corrupted fixture')
    elif change=='missing':
        path.unlink()
    elif change=='extra':
        (destination/'unexpected.bin').write_bytes(b'Unexpected fixture')
    elif change=='symlink':
        outside=tmp_path/'outside'
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    else:
        index=destination/'files.json'
        data=json.loads(index.read_text())
        data['files']['../outside']={'sha256':'0'*64,'bytes':1}
        index.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        release.verify_release(destination)


def test_same_release_is_idempotent_but_different_identity_cannot_overwrite(tmp_path):
    destination,hf,artifacts,metadata=build(tmp_path)
    before=(destination/'files.json').read_bytes()
    release.create_release(destination,hf,artifacts,metadata,'Fixture README, no quality claim')
    assert (destination/'files.json').read_bytes()==before
    with pytest.raises(ValueError,match='identity'):
        release.create_release(destination,hf,artifacts,{'purpose':'different fixture'},'Fixture README')
    assert (destination/'files.json').read_bytes()==before


def test_release_recovers_after_partial_copy_without_publishing_partial_files(tmp_path,monkeypatch):
    hf,artifacts,metadata=inputs(tmp_path)
    destination=tmp_path/'release'
    original=release.copy_independent
    calls=[]
    def fail(source,target):
        calls.append(str(source))
        if len(calls)==2:
            raise OSError(errno.ENOSPC,'Fixture disk full during copy')
        return original(source,target)
    monkeypatch.setattr(release,'copy_independent',fail)
    with pytest.raises(OSError):
        release.create_release(destination,hf,artifacts,metadata,'Fixture README')
    assert not destination.exists()
    assert list(tmp_path.glob('release-staging-*'))
    monkeypatch.setattr(release,'copy_independent',original)
    release.create_release(destination,hf,artifacts,metadata,'Fixture README')
    assert release.verify_release(destination)['valid']


def test_actual_console_verify_release_reads_only_files(tmp_path):
    destination,_,_,_=build(tmp_path)
    command=[str(Path(sys.executable).parent/'jdewitt'),'verify-release',str(destination)]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    proof=json.loads(result.stdout)
    assert proof['valid'] and proof['files']==6
    (destination/'README.md').write_text('Changed fixture')
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode!=0 and 'mismatch' in result.stderr.lower()
