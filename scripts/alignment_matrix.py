#!/usr/bin/env python
"""Run the static alignment-matrix experiment and write the report.

    python scripts/alignment_matrix.py --base BASE/model.safetensors \
        --instruct INSTRUCT/model.safetensors --twin TWIN/model.safetensors \
        --provenance verification/alignment-matrix/provenance.json \
        --out verification/alignment-matrix/report.json --emit-deltas DIR

No model, no forward, no device inference: tensor algebra and hashing only.
``--emit-deltas`` writes the float32 rank-k delta for the k the pre-registered
static rule selects, named as GGUF tensors (``blk.N.attn_output`` /
``blk.N.ffn_down``), for the separate application step.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

# The board's CPU is shared; a 32-thread BLAS spends more time contending than
# computing. Set this before numpy is imported.
for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '4')

import numpy as np  # noqa: E402  (the thread cap must be set before numpy loads)

from jdewitt.alignment import (CAVEATS, DEFAULT_KS, SCHEMA, aggregate, choose_k,  # noqa: E402
                               load_matrix, tensor_curve, writeback_tensors)

GGUF_NAMES = {
    'self_attn.o_proj.weight': 'attn_output.weight',
    'mlp.down_proj.weight': 'ffn_down.weight',
}


def sha256(path, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--instruct', type=Path, required=True)
    parser.add_argument('--twin', type=Path, default=None)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--emit-deltas', type=Path, default=None)
    parser.add_argument('--ks', default=','.join(str(k) for k in DEFAULT_KS))
    parser.add_argument('--residual-floor', type=float, default=0.95)
    args = parser.parse_args()

    ks = tuple(int(value) for value in args.ks.split(',') if value.strip())
    provenance = json.loads(args.provenance.read_text())

    # Re-hash the inputs instead of trusting the fetch step.
    checks = {}
    for key, path in (('base', args.base), ('instruct', args.instruct), ('twin', args.twin)):
        if path is None:
            continue
        expected = provenance[key]['sha256']
        started = time.time()
        actual = sha256(path)
        checks[f'{key}_sha256'] = {'expected': expected, 'actual': actual,
                                   'ok': actual == expected,
                                   'seconds': round(time.time() - started, 2)}
        if actual != expected:
            raise SystemExit(f'{key}: sha256 {actual} != provenance {expected}')

    from safetensors import safe_open
    with safe_open(str(args.instruct), framework='pt') as handle:
        keys = list(handle.keys())
    tensors = writeback_tensors(keys)
    if not tensors:
        raise SystemExit('no writeback tensors found')

    started = time.time()
    rows = []
    kept = {}
    for index, (name, kind, layer) in enumerate(tensors, 1):
        base = load_matrix(args.base, name)
        instruct = load_matrix(args.instruct, name)
        twin = load_matrix(args.twin, name) if args.twin is not None else None
        curve, factors = tensor_curve(base, instruct, twin, ks=ks, return_factors=True)
        curve.update({'name': name, 'kind': kind, 'layer': layer})
        rows.append(curve)
        # Keep the top-k singular factors so the delta for the chosen k can be
        # emitted without a second SVD pass.
        kept[name] = factors
        del base, instruct, twin, factors
        print(f'[{index}/{len(tensors)}] {name} delta_fro={curve["delta_norm"]:.4f}', flush=True)

    all_rows = rows
    by_kind = {}
    for kind in ('attn', 'ffn'):
        subset = [row for row in rows if row['kind'] == kind]
        if subset:
            by_kind[kind] = aggregate(subset, ks=ks)
    report_aggregate = {'all': aggregate(all_rows, ks=ks), **by_kind}

    selection = choose_k(report_aggregate['all'], ks=ks, residual_floor=args.residual_floor)
    k = selection['k']

    if args.emit_deltas is not None:
        args.emit_deltas.mkdir(parents=True, exist_ok=True)
        for name, kind, layer in tensors:
            u, s, vt = kept[name]
            delta = ((u[:, :k] * s[:k]) @ vt[:k]).astype(np.float32)
            leaf = next(leaf for suffix, leaf in GGUF_NAMES.items() if name.endswith(suffix))
            gguf_name = f'blk.{layer}.{leaf}'
            np.save(args.emit_deltas / f'{gguf_name}.npy', delta)
        (args.emit_deltas / 'manifest.json').write_text(json.dumps(
            {'k': k, 'names': sorted(path.stem for path in args.emit_deltas.glob('*.npy')),
             'selection': selection}, indent=2))

    max_energy_error = max(row['checks']['max_removed_energy_error'] for row in rows)
    max_energy_rel_error = max(row['checks']['max_removed_energy_rel_error'] for row in rows)
    max_remain_error = max(row['checks']['max_delta_remain_error'] for row in rows)
    report = {
        'schema': SCHEMA,
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'removal_measured': False,
        'experiment': {
            'delta': 'W_instruct - W_base, HF [out, in] layout',
            'removal': 'W_k = W_instruct - U[:, :k] S[:k] Vt[:k] (rank-k truncation of the delta)',
            'writeback_tensors': [{'name': name, 'kind': kind, 'layer': layer}
                                  for name, kind, layer in tensors],
            'tensor_count': len(tensors),
            'layers': len({layer for _n, _kind, layer in tensors}),
            'ks': list(ks),
            'twin_proxy': 'W_twin - W_instruct; twin_capture is the share of the twin delta '
                          'inside the removed U_k / V_k subspace (static proxy, not a refusal measurement)',
        },
        'provenance': provenance,
        'inputs_sha256': checks,
        'aggregate': report_aggregate,
        'selection': selection,
        'checks': {
            'max_removed_energy_error': max_energy_error,
            'max_removed_energy_rel_error': max_energy_rel_error,
            'max_delta_remain_error': max_remain_error,
            'energy_identity_holds': max_energy_rel_error < 1e-9,
            'remain_identity_holds': max_remain_error < 1e-6,
            'singular_values_descending': all(
                all(a >= b for a, b in zip(row['singular_values'], row['singular_values'][1:]))
                for row in rows),
        },
        'per_tensor': rows,
        'caveats': list(CAVEATS),
        'seconds': round(time.time() - started, 2),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({'out': str(args.out), 'tensors': len(tensors), 'k_applied': k,
                      'selection': selection, 'seconds': report['seconds']}, indent=2))
    for name in ('all', 'attn', 'ffn'):
        if name in report_aggregate:
            print(name, json.dumps(report_aggregate[name], indent=2))


if __name__ == '__main__':
    main()
