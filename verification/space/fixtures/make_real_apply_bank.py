# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build a throwaway bank entry for a REAL artifact, to exercise `bank-apply` on it.

Run from the repo root with the project venv (needs torch for the plan payload):

    python verification/space/fixtures/make_real_apply_bank.py \\
        --gguf /path/to/artifact.gguf --bank /tmp/bank-qwen2 \\
        --layers 10 12 --seed 1

The bank that ships in `spaces/bank/bank.json` is empty on purpose (no fitted axis
is in this clone), so the real-artifact gate run needs an entry whose geometry is
the artifact's OWN — read from its header and tensor table here — and whose axis is
an explicitly SYNTHETIC, seeded unit direction. That is enough to prove the command
(the write path, the codec ladder, the per-tensor measurement, the untouched bytes);
it proves nothing about removal, and the entry says so: `measurement.state` is
`synthetic_axis`, and the report's `removal_measured` is false.

This is a verification fixture builder, never a way to seed the shipped bank.
"""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader

from jdewitt.answers import atomic_json
from jdewitt.data import digest
from jdewitt.ggufgeom import gguf_weight_shapes, inferred_hidden_width
from jdewitt.registry import bundle_plan

BUNDLE = 'synthetic-plan.tar'


def _declared(reader, suffix):
    for key, field in reader.fields.items():
        if key.endswith(f'.{suffix}') or key == f'general.{suffix}':
            return field.contents()
    return None


def write_plan(registry, key, layers, hidden, seed):
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((len(layers), 1, hidden)).astype(np.float32)
    basis /= np.linalg.norm(basis, axis=-1, keepdims=True)
    plan = {'basis': torch.from_numpy(basis), 'layers': list(layers),
            'method': 'directional', 'fixture': 'synthetic axis; not a fitted removal'}
    directory = Path(registry) / 'plans' / key
    directory.mkdir(parents=True)
    payload = directory / 'plan.pt'
    with payload.open('wb') as stream:
        torch.save(plan, stream)
    meta = {'version': 1, 'key': key, 'registered_at': '2026-11-23T00:00:00Z',
            'identity_digest': digest({'synthetic': key}),
            'payload_sha256': hashlib.sha256(payload.read_bytes()).hexdigest(),
            'model': 'synthetic-fixture', 'revision': 'synthetic',
            'method': 'directional', 'method_options': {}, 'rank': 1, 'seed': seed,
            'generation_fingerprint': {'layers': len(layers), 'hidden_size': hidden, 'rank': 1},
            'suite_digests': {'directions': None, 'selection': None, 'holdout': None},
            'gates': None, 'measured': False}
    meta['meta_digest'] = digest(meta)
    atomic_json(directory / 'meta.json', meta)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--entry-key', default='synthetic/f00l01')
    parser.add_argument('--layers', type=int, nargs=2, required=True, metavar=('FIRST', 'LAST'))
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--min-cos', type=float, default=0.95)
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--types', nargs='+', default=['q6_k', 'q8_0', 'f16'])
    args = parser.parse_args(argv)

    reader = GGUFReader(str(args.gguf))
    arch = str(_declared(reader, 'architecture'))
    layers = int(_declared(reader, 'block_count'))
    shapes = gguf_weight_shapes(args.gguf, reader=reader)
    hidden = inferred_hidden_width(shapes)
    first, last = args.layers
    if hidden is None:
        raise SystemExit(f'{args.gguf} carries no residual-write projection')
    if not (0 <= first <= last < layers):
        raise SystemExit(f'layers {first}..{last} are outside 0..{layers - 1}')

    footprint = list(range(first, last + 1))
    args.bank.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='jdewitt-synthetic-bank-') as registry:
        write_plan(registry, 'synthetic' + args.entry_key.replace('/', ''), footprint, hidden,
                   args.seed)
        bundle = args.bank / BUNDLE
        if bundle.exists():
            bundle.unlink()
        written = bundle_plan(registry, 'synthetic' + args.entry_key.replace('/', ''), bundle)

    entry = {
        'key': args.entry_key,
        'base': {'repo': str(args.gguf), 'revision': 'local-file'},
        'architecture': arch,
        'geometry': {'layers': layers, 'hidden': hidden, 'hidden_source': 'artifact'},
        'targets': {'attn': 'attn_output', 'mlp': 'ffn_down'},
        'footprint': {'first': first, 'last': last,
                      'fraction': [round(first / layers, 4), round(last / layers, 4)]},
        'strength': {'attn': args.strength, 'mlp': args.strength},
        'edit': {'method': 'directional', 'shape': 'shared', 'space': 'output',
                 'row_normalization': 'none', 'min_cos': args.min_cos,
                 'types': list(args.types), 'repair': 'none'},
        'sources': {'quant_types': ['q4_k', 'q6_k', 'q8_0', 'f16']},
        'measurement': {'state': 'synthetic_axis', 'artifact_class': None,
                        'removal': None, 'retention': None, 'config': None},
        'bundle': {'file': BUNDLE, 'sha256': written['bundle_sha256']},
    }
    (args.bank / 'bank.json').write_text(
        json.dumps({'schema': 'jdewitt-space-bank-v1', 'entries': [entry]}, indent=2) + '\n')
    print(json.dumps({'bank': str(args.bank), 'arch': arch, 'layers': layers,
                      'hidden': hidden, 'footprint': [first, last],
                      'bundle_sha256': written['bundle_sha256']}, indent=2))


if __name__ == '__main__':
    main()
