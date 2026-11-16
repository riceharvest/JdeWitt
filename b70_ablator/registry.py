# SPDX-License-Identifier: AGPL-3.0-or-later
"""Local fitted-plan registry: provenance-gated library of edit plans for warm starts.

Plans are the small fitted tensors (basis / LEACE factors / ARA factors), never
model weights. Entries are registered from a finished run directory, bind the
exact checkpoint payload by sha256 and identity digest, and carry gates: either
a supplied evidence file (verify-release / robustness-report output) or an
explicit unmeasured label. Lookup for transfer matches fit context
(method, options, suite digest) and verifies tensor geometry at load time;
a matching entry for a different checkpoint in the same family transfers, a
shape mismatch refuses loudly.
"""
import json
import shutil
from pathlib import Path

from .answers import atomic_json
from .clock import now_iso
from .storage import digest, file_digest

INDEX = 'plans.jsonl'


class RegistryError(ValueError):
    pass


def _read_json(path):
    return json.loads(Path(path).read_text())


def _index_path(registry):
    return Path(registry) / INDEX


def list_plans(registry):
    """Index entries; returns an error string per line only via load, never tensor data."""
    path = _index_path(registry)
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _entry_key(identity_digest, model, revision):
    return digest({'identity_digest': identity_digest, 'model': model, 'revision': revision})[:16]


def register_plan(registry, run_output, gates_path=None, allow_unmeasured=False):
    """Bind a run's fitted directions checkpoint into the registry.

    ``run_output`` must contain directions.pt plus its integrity pointer and a
    provenance.json written by the pipeline. Gates default to requiring an
    evidence file; ``allow_unmeasured`` labels the entry UNMEASURED forever.
    """
    run = Path(run_output)
    pointer_path = run / 'directions.checkpoint.json'
    provenance_path = run / 'provenance.json'
    missing = [str(p) for p in (pointer_path, provenance_path) if not p.is_file()]
    if missing:
        raise RegistryError(f'{run}: not a fitted run, missing ' + ', '.join(missing))
    pointer = _read_json(pointer_path)
    content = {k: v for k, v in pointer.items() if k != 'record_digest'}
    if pointer.get('record_digest') != digest(content):
        raise RegistryError('run checkpoint pointer integrity mismatch; refusing to register')
    name = pointer.get('payload')
    if not isinstance(name, str) or Path(name).name != name or name in {'.', '..'}:
        raise RegistryError('Invalid checkpoint payload path')
    payload = run / name
    if payload.is_symlink() or not payload.is_file():
        raise RegistryError(f'run checkpoint payload {payload} is missing; refusing to register')
    sha = file_digest(payload)
    if sha != pointer.get('sha256'):
        raise RegistryError('run checkpoint payload integrity mismatch; refusing to register a corrupted fit')
    provenance = _read_json(provenance_path)
    config = provenance.get('config') or {}
    if not config.get('model'):
        raise RegistryError('run provenance has no model identity; refusing to register')

    gates = None
    if gates_path is not None:
        gates_path = Path(gates_path)
        if not gates_path.is_file():
            raise FileNotFoundError(gates_path)
        gates = _read_json(gates_path)
        gates = {'source_file': str(gates_path), 'sha256': file_digest(gates_path), 'contents': gates}
    elif not allow_unmeasured:
        raise RegistryError('no gates supplied: pass --gates EVIDENCE.json (verify-release or '
                            'robustness-report output) or --allow-unmeasured to label this fit UNMEASURED')

    registry = Path(registry)
    key = _entry_key(pointer['identity_digest'], config['model'], provenance.get('resolved_revision'))
    directory = registry / 'plans' / key
    if (directory / 'meta.json').exists():
        raise RegistryError(f'plan {key} is already registered')
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(payload, directory / 'plan.pt')
    meta = {'version': 1, 'key': key, 'registered_at': now_iso(),
            'identity_digest': pointer['identity_digest'], 'payload_sha256': sha,
            'model': config['model'], 'revision': provenance.get('resolved_revision'),
            'method': config.get('method'), 'method_options': config.get('method_options') or {},
            'rank': config.get('rank'), 'seed': config.get('seed'),
            'generation_fingerprint': (pointer.get('metadata') or {}).get('generation_fingerprint'),
            'suite_digests': {'directions': provenance.get('directions_digest'),
                              'selection': provenance.get('selection_digest'),
                              'holdout': provenance.get('holdout_digest')},
            'source_files_sha256': provenance.get('source_files'),
            'gates': gates, 'measured': gates is not None}
    meta['meta_digest'] = digest({k: v for k, v in meta.items() if k != 'meta_digest'})
    atomic_json(directory / 'meta.json', meta)
    index_line = dict(meta, payload_file=str(directory / 'plan.pt'))
    index_line.pop('source_files_sha256', None)
    path = _index_path(registry)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(index_line) + '\n')
        stream.flush()
    return {'key': key, 'measured': meta['measured'], 'method': meta['method'],
            'model': meta['model'], 'note': 'plan registered; tensors verified against the run checkpoint sha256'}


def load_plan(registry, key, expect=None, device='cpu'):
    """Load a registered plan; verifies metadata and payload integrity every time.

    ``expect`` may name {'layers','rank','hidden'}; tensors whose geometry
    cannot carry the target layer/hidden sizes raise instead of silently
    transferring garbage.
    """
    import torch
    path = Path(registry) / 'plans' / key / 'meta.json'
    if not path.is_file():
        raise RegistryError(f'plan {key} is not registered')
    meta = _read_json(path)
    content = {k: v for k, v in meta.items() if k != 'meta_digest'}
    if meta.get('meta_digest') != digest(content):
        raise RegistryError(f'plan {key} metadata integrity mismatch')
    payload = Path(registry) / 'plans' / key / 'plan.pt'
    if payload.is_symlink() or not payload.is_file() or file_digest(payload) != meta['payload_sha256']:
        raise RegistryError(f'plan {key} payload integrity mismatch')
    value = torch.load(payload, map_location=device, weights_only=True)
    directions, audit = value
    basis = directions.get('basis') if isinstance(directions, dict) else None
    geometry = {}
    if isinstance(basis, torch.Tensor) and basis.ndim == 3:
        geometry = {'layers': int(basis.shape[0]), 'rank': int(basis.shape[1]), 'hidden': int(basis.shape[2])}
    if expect:
        for field, got in geometry.items():
            want = expect.get(field)
            if want is not None and want != got:
                raise RegistryError(f'plan {key} has {field}={got}, target needs {want}; refusing the transfer')
        for field in ('layers', 'rank', 'hidden'):
            if field in expect and geometry.get(field) is None:
                raise RegistryError(f'plan {key} has no verifiable 3D basis to match {field}; refusing the transfer')
    return {'plan': directions, 'audit': audit, 'meta': meta, 'geometry': geometry}


GEOMETRY_KEYS = ('architecture', 'layers', 'hidden_size', 'rank', 'dtype', 'quantization', 'edit_groups', 'boundary')


def warm_start_candidates(registry, method, method_options=None, directions_digest=None,
                          geometry=None, require_gates=False, same_model=None, same_revision=None):
    """Registered plans that fit the search context.

    A plan transfers only when the method, options, suite digest, and every
    given target geometry key (the checkpoint fingerprint minus model and
    revision) match. A plan with no recorded fingerprint is ineligible,
    never a wildcard; unmeasured entries stay labelled.
    """
    method_options = method_options or {}
    want = digest(method_options)
    out = []
    for entry in list_plans(registry):
        if entry.get('method') != method:
            continue
        if digest(entry.get('method_options') or {}) != want:
            continue
        if directions_digest and (entry.get('suite_digests') or {}).get('directions') != directions_digest:
            continue
        if require_gates and not entry.get('measured'):
            continue
        meta = _read_json(Path(registry) / 'plans' / entry['key'] / 'meta.json')
        fingerprint = meta.get('generation_fingerprint')
        reason = 'eligible'
        if geometry is not None:
            if not fingerprint:
                reason = 'no recorded generation fingerprint'
            else:
                mismatch = [k for k in GEOMETRY_KEYS if k in fingerprint and fingerprint.get(k) != geometry.get(k)]
                missing = sorted(set(geometry) - set(fingerprint))
                if mismatch:
                    reason = 'geometry mismatch: ' + ', '.join(mismatch)
                elif missing:
                    reason = 'target geometry not recorded: ' + ', '.join(missing)
        out.append({k: meta.get(k) for k in ('key', 'model', 'revision', 'method', 'rank', 'seed', 'measured', 'registered_at')} | {
            'eligible': reason == 'eligible', 'reason': reason,
            'same_model': same_model is None or meta.get('model') == same_model,
            'same_revision': same_revision is None or meta.get('revision') == same_revision})
    out.sort(key=lambda e: (not e['same_model'], not e['same_revision'], not e['measured'], -1 if e['measured'] else 0, e['registered_at']))
    return out
