# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stream original safetensors through B70 edits; never CPU-reload a whole model."""
import json
import shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from .compat import repair_export_metadata

from contextlib import ExitStack
from .release import sha256_file
from .storage import atomic_destination
from .answers import atomic_json



def stream_merge(source, output, editor, max_shard_bytes=1_000_000_000):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if max_shard_bytes <= 0:
        raise ValueError('Invalid shard budget')
    files = sorted(source.glob('*.safetensors'))
    if not files:
        raise ValueError('Streaming export requires original safetensors')
    weight_bytes = sum(p.stat().st_size for p in files)
    if shutil.disk_usage(source).free < weight_bytes * 2:
        raise RuntimeError('Insufficient disk for merged weights and conversion')
    output.mkdir(parents=True)
    edits = {}
    for *_,adapter in editor.targets:
        # Source key paths are recorded by discovery, not reconstructed from shapes.
        if not hasattr(adapter,'source_path'):
            raise ValueError('Missing source weight key on projection adapter')
        edits[getattr(adapter,'source_key',adapter.source_path+'.weight')] = adapter
    shards, shard, size, index, total = [], {}, 0, {}, 0
    seen_edits = set()
    def flush():
        nonlocal shard,size
        if not shard:
            return
        name=f'model-{len(shards)+1:05d}.safetensors'
        with atomic_destination(output/name) as temporary:
            save_file(shard,str(temporary),metadata={'format':'pt'})
        for key in shard:
            index[key]=name
        shards.append(name)
        shard={}
        size=0
    for file in files:
        with safe_open(str(file),framework='pt',device='cpu') as reader:
            for key in reader.keys():
                tensor=reader.get_tensor(key)
                if key in edits:
                    adapter=edits[key]
                    if tensor.shape != adapter.base.weight.shape:
                        raise ValueError(f'Weight layout changed: {key}')
                    # CPU is staging/storage only. Weight arithmetic occurs on B70.
                    device=adapter.A.device
                    if device.type not in ('xpu','cuda'):
                        raise ValueError('Streaming merge requires device-resident adapters (B70 primary or an allowed CUDA secondary)')
                    edited=tensor.to(device=device)
                    with torch.no_grad():
                        tensor=adapter.merge_source(edited).cpu().contiguous()
                    del edited
                    seen_edits.add(key)
                n=tensor.numel()*tensor.element_size()
                if size+n>max_shard_bytes:
                    flush()
                if key in index or key in shard:
                    raise ValueError(f'Duplicate tensor key: {key}')
                shard[key]=tensor.contiguous()
                size+=n
                total+=n
    flush()
    if seen_edits!=set(edits):
        raise ValueError('Some projection weights were not exported')
    atomic_json(output/'model.safetensors.index.json',{'metadata':{'total_size':total},'weight_map':index})
    # Preserve tokenizer, generation, processor, custom modeling and model-license files.
    for file in source.iterdir():
        if file.is_file() and file.suffix not in {'.safetensors','.bin','.pt','.pth','.gguf'} and not file.name.endswith('.index.json') and not file.name.startswith('.'):
            shutil.copy2(file,output/file.name)
    repairs = repair_export_metadata(output)
    return {'total_size':total,'shards':len(shards),'edited_tensors':len(seen_edits),'compatibility_fixes':repairs}


def require_resident_precision(source, editor):
    adapters = {getattr(a,'source_key',a.source_path+'.weight'): a for *_, a in editor.targets}
    seen = set()
    names = {torch.float16: 'F16', torch.bfloat16: 'BF16', torch.float32: 'F32'}
    workspace = 0
    for file in sorted(Path(source).glob('*.safetensors')):
        with safe_open(file, framework='pt', device='cpu') as reader:
            for key in reader.keys():
                if key not in adapters:
                    continue
                a = adapters[key]
                if a.operator != 'merged-resident' or getattr(a.base.weight, 'quant_state', None) is not None:
                    raise ValueError('Consistent resident export requires original-precision, unquantized trial weights')
                if reader.get_slice(key).get_dtype() != names.get(a.base.weight.dtype):
                    raise ValueError('Search dtype must match original export precision; no silent dtype conversion')
                seen.add(key)
                # Conservative per-projection FP32 operands/results plus rounded weight.
                workspace = max(workspace, a.base.weight.numel() * (12 + a.base.weight.element_size()))
    if seen != set(adapters):
        raise ValueError('Original source lacks some resident projection weights')
    return workspace


@torch.inference_mode()
def verify_resident_export(source, output, editor):
    """Prove stored tensors match the resident operator without reloading/offloading.

    Serialized tensors are streamed from storage one at a time. Original GPU
    weights keep their allocation and contents throughout this verification.
    """
    source, output = Path(source), Path(output)
    workspace = require_resident_precision(source, editor)
    index = json.loads((output/'model.safetensors.index.json').read_text())['weight_map']
    adapters = {getattr(a,'source_key',a.source_path+'.weight'): a for *_, a in editor.targets}
    seen = set()
    with ExitStack() as stack:
        outputs = {}
        for name in set(index.values()):
            path = (output/name).resolve()
            if not path.is_relative_to(output.resolve()):
                raise ValueError('Invalid export shard path')
            outputs[name] = stack.enter_context(safe_open(path, framework='pt', device='cpu'))
        if {k for r in outputs.values() for k in r.keys()} != set(index):
            raise ValueError('Stored export tensor set disagrees with index')
        for file in sorted(source.glob('*.safetensors')):
            with safe_open(file, framework='pt', device='cpu') as original:
                for key in original.keys():
                    if key not in index or key in seen:
                        raise ValueError('Source/export tensor identity mismatch')
                    seen.add(key)
                    stored = outputs[index[key]].get_tensor(key)
                    base = original.get_tensor(key)
                    if stored.dtype != base.dtype or stored.shape != base.shape:
                        raise ValueError(f'Stored export layout/precision mismatch: {key}')
                    if key in adapters:
                        a = adapters[key]
                        if a.base.weight.device.type not in ('xpu','cuda'):
                            raise ValueError('Export verification requires device-resident weights (B70 primary or an allowed CUDA secondary)')
                        if not torch.equal(base.to(a.base.weight.device), a.base.weight):
                            raise ValueError('Resident original weights differ from export source')
                        expected = a.merge_source(a.base.weight)
                        if not torch.equal(stored.to(expected.device), expected):
                            raise ValueError(f'Stored export differs from resident operator: {key}')
                    elif not torch.equal(stored, base):
                        raise ValueError(f'Untouched export tensor changed: {key}')
        if seen != set(index):
            raise ValueError('Stored export contains unexpected tensors')
    return {'operator': 'merged-resident', 'edited_tensors': len(adapters), 'unchanged_tensors': len(seen)-len(adapters),
            'workspace_bytes': workspace, 'shard_sha256': {name: sha256_file(output/name) for name in sorted(outputs)},
            'file_sha256': {p.name:sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()}}
