# SPDX-License-Identifier: AGPL-3.0-or-later
"""Durable local files and immutable tensor payloads; no model/device operations."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import uuid
from .answers import atomic_json
from .data import digest


def sync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def atomic_destination(path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.new')
    try:
        yield temporary
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary,path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def file_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def save_checkpoint(path,value,identity,metadata):
    import torch
    path=Path(path)
    payload=path.with_name(path.stem+'.'+uuid.uuid4().hex+'.pt')
    with atomic_destination(payload) as temporary:
        with temporary.open('xb') as stream:
            torch.save(value,stream)
            stream.flush()
    pointer={'version':1,'identity_digest':digest(identity),'payload':payload.name,
             'sha256':file_digest(payload),'metadata':metadata}
    pointer['record_digest']=digest(pointer)
    # Only this final pointer makes the payload and its audit a committed pair.
    atomic_json(path.with_suffix('.checkpoint.json'),pointer)
    return pointer


def load_checkpoint(path,identity):
    import torch
    path=Path(path)
    pointer_path=path.with_suffix('.checkpoint.json')
    if not pointer_path.exists():
        if path.exists():
            raise ValueError('Legacy unbound tensor checkpoint requires a new output directory')
        return None
    pointer=json.loads(pointer_path.read_text())
    content={k:v for k,v in pointer.items() if k!='record_digest'}
    if set(pointer)!={'version','identity_digest','payload','sha256','metadata','record_digest'} or pointer['version']!=1 or pointer['record_digest']!=digest(content):
        raise ValueError('Tensor checkpoint pointer integrity mismatch')
    if pointer['identity_digest']!=digest(identity):
        raise ValueError('Tensor checkpoint identity mismatch')
    name=pointer['payload']
    if not isinstance(name,str) or Path(name).name!=name or name in {'.','..'}:
        raise ValueError('Invalid checkpoint payload path')
    payload=path.parent/name
    if payload.is_symlink() or not payload.is_file() or file_digest(payload)!=pointer['sha256']:
        raise ValueError('Tensor checkpoint payload integrity mismatch')
    data=torch.load(payload,map_location='cpu',weights_only=True)
    if file_digest(payload)!=pointer['sha256']:
        raise ValueError('Tensor checkpoint changed while loading')
    return data,pointer['metadata']
