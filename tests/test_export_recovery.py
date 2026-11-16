"""Export interruption fixtures; placeholder bytes are explicitly NOT models."""
import errno
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from safetensors.torch import save_file
from b70_ablator import export, release


def test_interrupted_safetensors_write_never_commits_a_truncated_shard(tmp_path,monkeypatch):
    source=tmp_path/'source'
    source.mkdir()
    save_file({'fixture.weight':torch.arange(8)},source/'model.safetensors')
    (source/'config.json').write_text('{"model_type":"fixture"}')
    def partial(tensors,path,**kwargs):
        Path(path).write_bytes(b'Truncated fixture shard')
        raise OSError(errno.ENOSPC,'Fixture disk full')
    monkeypatch.setattr(export,'save_file',partial)
    destination=tmp_path/'export'
    with pytest.raises(OSError):
        export.stream_merge(source,destination,SimpleNamespace(targets=[]))
    assert not list(destination.glob('*.safetensors'))
    assert not (destination/'model.safetensors.index.json').exists()
    assert not list(destination.glob('.*.new'))


def test_quant_resume_reuses_verified_steps_and_rejects_changed_input(tmp_path,monkeypatch):
    hf=tmp_path/'hf'
    hf.mkdir()
    (hf/'fixture.safetensors').write_bytes(b'Explicit fixture source, NOT weights')
    tools=[]
    for name in ('converter','quantizer','python'):
        path=tmp_path/name
        path.write_bytes(b'Explicit fixture executable identity, never executed')
        tools.append(path)
    calls=[]
    fail={'q5':True}
    def run(command,log,**kwargs):
        calls.append([str(v) for v in command])
        target=Path(command[command.index('--outfile')+1]) if '--outfile' in command else Path(command[-2])
        target.write_bytes(b'Explicit fixture artifact, NOT GGUF')
        if command[-1]=='Q5_K_M' and fail['q5']:
            raise RuntimeError('Fixture interrupted Q5 conversion')
    def proof(path,quant):
        return {'path':str(Path(path).resolve()),'quant':quant,'sha256':release.sha256_file(path),'bytes':Path(path).stat().st_size,'fixture':True}
    monkeypatch.setattr(release,'run_logged',run)
    monkeypatch.setattr(release,'verify_quant',proof)
    monkeypatch.setattr(release,'verify_bf16',lambda path:proof(path,'BF16'))
    destination=tmp_path/'quants'
    with pytest.raises(RuntimeError,match='interrupted Q5'):
        release.export_gguf(hf,destination,*tools)
    assert (destination/'model-Q4_K_M.gguf').exists()
    assert not (destination/'model-Q5_K_M.gguf').exists()
    assert not list(destination.glob('.*.new'))
    fail['q5']=False
    result=release.export_gguf(hf,destination,*tools)
    assert len(calls)==4 and set(result)==set(release.QUANTS)
    again=release.export_gguf(hf,destination,*tools)
    assert again==result and len(calls)==4
    state=json.loads((destination/'conversion-state.json').read_text())
    assert state['complete']
    (hf/'fixture.safetensors').write_bytes(b'Changed fixture source')
    with pytest.raises(ValueError,match='identity'):
        release.export_gguf(hf,destination,*tools)
