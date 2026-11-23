# SPDX-License-Identifier: AGPL-3.0-or-later
"""Seed one Space-bank entry from a capture: fit -> register -> bundle -> entry.

The Space cannot fit an axis (it has no captures), so an entry is produced offline
once and then applied to a visitor's artifact by digest. This script is that
offline step for the first covered base, Qwen3-4B-Instruct-2507 at the measured
`f15s15-l935` recipe, and it refuses to guess anything it can read:

  1. fit the capture with the arm's own estimator (`spanfit.shared_axis`, the
     sign-aligned shared fit) and time it;
  2. bind the fit into a run directory the registry accepts (`directions.pt` +
     its integrity pointer + `provenance.json`), rank 1, basis `[layers, 1, hidden]`
     because a bank axis is `[layers, rank, hidden]` in bf16;
  3. `plan-register` (unmeasured: this refit has no wave of its own), then
     `plan-load --expect <layers> 1 <hidden>` so a wrong geometry stops here;
  4. `plan-bundle` into the bank and `direction-export` to bf16 so the axis is
     portable and inspectable;
  5. write the §5.1 entry beside the bundle (and the per-model `entry.json` copy),
     then resolve the artifact the entry claims to cover through `bank-check`.

Nothing here measures removal or retention. The entry is born `banked_unmeasured`;
the transfer card earns the label on an artifact class, not this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + '\n')


def fit(records_path, run_dir, model, revision, shape='shared'):
    """Fit the arm's axes from a capture and bind them as a rank-1 plan."""
    import torch

    from jdewitt.spanfit import load_records, per_layer_axes, shared_axis
    from jdewitt.storage import save_checkpoint

    started = time.monotonic()
    records = load_records(str(records_path))
    if shape == 'shared':
        axes, _direction = shared_axis(records)   # [L, W] broadcast of the shared direction
    else:
        axes = per_layer_axes(records)
    elapsed = time.monotonic() - started
    axes = axes.float().contiguous()
    if axes.ndim != 2:
        raise SystemExit(f'fit produced {tuple(axes.shape)}; a bank axis is [layers, hidden]')
    layers, hidden = axes.shape

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'axes.pt').parent.mkdir(parents=True, exist_ok=True)
    torch.save({'axes': axes}, run_dir / 'axes.pt')

    # A bank plan is [layers, rank, hidden]: the registry's geometry check reads
    # rank from the basis and `plan-load --expect L 1 H` must then pass.
    basis = axes.unsqueeze(1).contiguous()
    plan = {'method': 'directional', 'basis': basis, 'row_normalization': 'none'}
    identity = {'model': model, 'revision': revision,
                'records_sha256': _sha256(records_path),
                'shape': shape, 'estimator': 'sign_aligned_shared' if shape == 'shared' else 'contrast_mean'}
    save_checkpoint(run_dir / 'directions.pt', plan, identity,
                    {'generation_fingerprint': {'architecture': 'qwen3', 'layers': layers,
                                                'hidden_size': hidden, 'rank': 1,
                                                'dtype': 'float32'}})
    provenance = {
        'config': {'model': model, 'method': 'directional', 'rank': 1, 'seed': None,
                   'method_options': {'shape': shape, 'axis_space': 'output',
                                      'estimator': 'sign_aligned_shared' if shape == 'shared'
                                      else 'contrast_mean'}},
        'resolved_revision': revision,
        'directions_digest': identity['records_sha256'],
    }
    _write_json(run_dir / 'provenance.json', provenance)
    fit_record = {'axes': list(axes.shape), 'shape': shape, 'fit_seconds': round(elapsed, 2),
                  'axes_sha256': _sha256(run_dir / 'axes.pt'),
                  'records_sha256': identity['records_sha256'],
                  'estimator': identity['estimator']}
    _write_json(run_dir / 'fit.json', fit_record)
    return {'axes': axes, 'layers': layers, 'hidden': hidden, 'fit': fit_record}


def register(run_dir, registry, layers, hidden):
    from jdewitt.registry import load_plan, register_plan

    registered = register_plan(registry, run_dir, gates_path=None, allow_unmeasured=True)
    key = registered['key']
    loaded = load_plan(registry, key, {'layers': layers, 'rank': 1, 'hidden': hidden})
    return {'key': key, 'geometry': loaded['geometry'], 'measured': registered['measured']}


def entry_for(key, model, revision, layers, hidden, footprint, strength, bundle, sha256):
    first, last = footprint
    return {
        'key': key,
        'base': {'repo': model, 'revision': revision},
        'architecture': 'qwen3',
        'geometry': {'layers': layers, 'hidden': hidden, 'hidden_source': 'artifact'},
        'targets': {'attn': 'attn_output', 'mlp': 'ffn_down'},
        'footprint': {'first': first, 'last': last,
                      'fraction': [round(first / layers, 4), round(last / layers, 4)]},
        'strength': {'attn': strength, 'mlp': strength},
        'edit': {'method': 'directional', 'shape': 'shared', 'space': 'output',
                 'row_normalization': 'none', 'min_cos': 0.95,
                 'types': ['q6_k', 'q8_0', 'f16'], 'repair': 'none'},
        'sources': {'quant_types': ['q4_k', 'q5_k', 'q6_k', 'q8_0']},
        'measurement': {'state': 'banked_unmeasured', 'artifact_class': None,
                        'removal': None, 'retention': None, 'config': None},
        'bundle': {'file': bundle, 'sha256': sha256},
    }


def write_bank(bank_dir, entry, slug):
    """Put the entry in `bank.json` (what the loader reads) and in the §5.1 unit."""
    bank_dir = Path(bank_dir)
    bank_path = bank_dir / 'bank.json'
    if not bank_path.is_file():
        raise SystemExit(f'{bank_path} is missing; the bank document is the unit bank-check reads')
    document = json.loads(bank_path.read_text())
    if document.get('schema') != 'jdewitt-space-bank-v1':
        raise SystemExit(f'{bank_path} schema is {document.get("schema")!r}')
    entries = [row for row in document.get('entries', []) if row.get('key') != entry['key']]
    entries.append(entry)
    document['entries'] = entries
    _write_json(bank_path, document)
    _write_json(bank_dir / slug / 'entry.json', entry)
    return {'bank': str(bank_path), 'entries': [row['key'] for row in entries],
            'entry_copy': str(bank_dir / slug / 'entry.json')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True, help='capture .pt the fit reads')
    parser.add_argument('--run-dir', type=Path, required=True, help='where the fitted run is bound')
    parser.add_argument('--registry', type=Path, help='defaults to <run-dir>/registry')
    parser.add_argument('--bank', type=Path, default=Path('spaces/bank'))
    parser.add_argument('--key', default='qwen3-4b-instruct-2507/f15s15-l935')
    parser.add_argument('--slug', default='qwen3-4b-instruct-2507')
    parser.add_argument('--model', default='Qwen/Qwen3-4B-Instruct-2507')
    parser.add_argument('--revision', default='cdbee75f17c01a7cc42f958dc650907174af0554')
    parser.add_argument('--footprint', default='15-35', help='first-last, inclusive')
    parser.add_argument('--strength', type=float, default=1.5)
    parser.add_argument('--bundle-name', default='qwen3-4b-instruct-2507-f15s15-l935.tar')
    parser.add_argument('--direction-out', type=Path,
                        default=Path('verification/space/qwen3-4b-refit/direction-f15s15-l935.gguf'))
    parser.add_argument('--shape', choices=('shared', 'perlayer'), default='shared')
    parser.add_argument('--gguf', type=Path, help='artifact the entry must resolve (bank-check)')
    parser.add_argument('--report', type=Path, help='write the whole run report here')
    args = parser.parse_args(argv)

    registry = args.registry or (args.run_dir / 'registry')
    fitted = fit(args.records, args.run_dir, args.model, args.revision, args.shape)
    registered = register(args.run_dir, registry, fitted['layers'], fitted['hidden'])

    from jdewitt.registry import bundle_plan

    bank_dir = Path(args.bank)
    bank_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = bank_dir / args.bundle_name
    bundle = bundle_plan(registry, registered['key'], bundle_path)
    bundle_sha = _sha256(bundle_path)

    from jdewitt.agent import direction_export

    args.direction_out.parent.mkdir(parents=True, exist_ok=True)
    exported = direction_export(registry, registered['key'], args.direction_out, 'bf16')

    first, last = (int(part) for part in args.footprint.split('-'))
    if not 0 <= first <= last < fitted['layers']:
        raise SystemExit(f'footprint {args.footprint} does not fit {fitted["layers"]} layers')
    entry = entry_for(args.key, args.model, args.revision, fitted['layers'],
                      fitted['hidden'], (first, last), args.strength,
                      args.bundle_name, bundle_sha)
    written = write_bank(bank_dir, entry, args.slug)

    report = {'fit': fitted['fit'], 'plan': registered, 'bundle': {'file': args.bundle_name,
              'sha256': bundle_sha, 'bytes': bundle_path.stat().st_size, 'plan': bundle},
              'direction': {'file': str(args.direction_out), 'sha256': _sha256(args.direction_out),
                            'bytes': args.direction_out.stat().st_size, 'export': exported},
              'entry': entry, 'written': written}
    if args.gguf:
        from jdewitt.bank import bank_check

        resolved = bank_check(bank_dir, args.gguf)
        report['resolved'] = resolved
        report['gate'] = {'targets': len(resolved['target_names']),
                          'layers': resolved['layers'][0], 'last_layer': resolved['layers'][-1],
                          'source_types': resolved['artifact']['source_types']}
    if args.report:
        _write_json(args.report, report)
    summary = {k: report[k] for k in ('fit', 'plan', 'gate') if k in report}
    summary['bundle'] = {'file': report['bundle']['file'], 'sha256': report['bundle']['sha256'],
                         'bytes': report['bundle']['bytes']}
    summary['direction'] = {'file': report['direction']['file'],
                            'sha256': report['direction']['sha256']}
    summary['entry'] = {'key': entry['key'], 'path': written['entry_copy'],
                        'fraction': entry['footprint']['fraction']}
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
