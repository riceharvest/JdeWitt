#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Train one small top-k SAE per non-protective family at the compute plan's budget.

Budget (steps, features, k, batch, lr, layer) comes from
``docs/SAE-COMPUTE-PLAN.md``; the stop rule is priced in ``scripts/sae_budget.py``
and re-checked here before the first gradient step -- if a family's training is
priced above the artifact's own merge+cert cost the lane stops and says
``unmeasured`` instead of training.

Input is ``capture``'s records (``[layers, hidden]`` per row, answer locus). Only
one layer is used, so ``--dump-residuals`` writes the ``[rows, hidden]`` matrix
this script actually trains on, with the row metadata beside it; the record file
is never rewritten. Rows are split deterministically by sorted id (every 4th row
held out), so the same inputs always give the same split, and the training rows
of a family never contain a held-out row (asserted, not assumed).

Outputs, all under ``--out``: per-family checkpoint (.npz, fp16), config (.json),
the train report, the feature table (json + md), and a digests file. The run
generates ZERO tokens (R1): every number is device time on captured rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from jdewitt import sae_fit

FAMILIES = (
    'hb-std-sel-illegal',
    'hb-std-sel-cybercrime-intrusion',
    'sr-sel-non-violent-crimes',
    'sr-sel-violence',
)


def read_records(path: Path) -> tuple[list[tuple[dict, torch.Tensor]], dict]:
    payload = torch.load(path, map_location='cpu', weights_only=False)
    records = payload.get('records') if isinstance(payload, dict) else None
    if not records:
        raise SystemExit(f'{path}: no records list')
    capture = payload.get('capture', {})
    return records, capture


def split_rows(rows: list[dict], holdout_every: int = 4) -> tuple[list[dict], list[dict]]:
    """Deterministic split by sorted id: every ``holdout_every``-th row is held out."""
    ordered = sorted(rows, key=lambda row: row['id'])
    heldout = [row for index, row in enumerate(ordered) if index % holdout_every == holdout_every - 1]
    train = [row for index, row in enumerate(ordered) if index % holdout_every != holdout_every - 1]
    if not heldout or not train:
        raise SystemExit(f'{len(rows)} rows cannot be split at holdout_every={holdout_every}')
    return train, heldout


def group(records, layer: int) -> dict:
    """``{'refusal': {family: [row dict]}, 'compliant': [...]}`` from labelled records."""
    out = {'refusal': {}, 'compliant': []}
    for meta, rec in records:
        tensor = rec if torch.is_tensor(rec) else torch.as_tensor(rec)
        if tensor.ndim != 2:
            raise SystemExit(f'record {meta.get("id")} has shape {tuple(tensor.shape)}; expected [layers, hidden]')
        row = {'id': meta['id'], 'label': meta.get('label'), 'kind': meta.get('kind'),
               'family_id': meta.get('family_id'), 'prompt': meta.get('prompt'),
               'shape': meta.get('shape'), 'vector': tensor[layer].float()}
        label = meta.get('label')
        if label == 'bad':
            out['refusal'].setdefault(meta.get('family_id'), []).append(row)
        elif label == 'good':
            out['compliant'].append(row)
    return out


def price(seconds: float, merge_cert_seconds: int) -> dict:
    return {'gpu_seconds': round(seconds, 3),
            'merge_cert_seconds': merge_cert_seconds,
            'stop_rule_fires': seconds > merge_cert_seconds}


def dump_residuals(out: Path, grouped: dict, layer: int, capture: dict) -> dict:
    """Write the exact ``[rows, hidden]`` matrices the trainer reads, plus their metadata."""
    entries = []
    for family, rows in grouped['refusal'].items():
        for row in rows:
            entries.append(('refusal', family, row))
    for row in grouped['compliant']:
        entries.append(('compliant', row['family_id'], row))
    import numpy as np
    # float32, not fp16: the captured values are bf16, so widening to fp32 is exact.
    # An fp16 matrix is lossy at ~2e-4 per entry, which was enough to move a family's
    # held-out FVU across the baseline rule -- the committed matrix must be the one
    # the report trained on.
    matrix = torch.stack([row['vector'] for _, _, row in entries]).to(torch.float32)
    path = out / f'residuals-layer{layer}.npz'
    np.savez_compressed(path, matrix=matrix.numpy())
    meta_path = out / f'residuals-layer{layer}.meta.json'
    meta = {'layer': layer, 'shape': list(matrix.shape), 'dtype': 'float32 (captured bf16, widened)',
            'matrix_file': path.name, 'matrix_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'capture': {key: value for key, value in capture.items() if key not in ('targets', 'compute_backend')},
            'rows': [{'id': row['id'], 'role': role, 'family_id': family, 'prompt': row['prompt'],
                      'shape': row['shape']} for role, family, row in entries]}
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True) + '\n')
    return {'path': str(path), 'sha256': meta['matrix_sha256'], 'shape': list(matrix.shape),
            'dtype': 'float32', 'meta': str(meta_path),
            'meta_sha256': hashlib.sha256(meta_path.read_bytes()).hexdigest()}


def load_residuals(path: Path) -> tuple[dict, dict, int | None]:
    import numpy as np
    matrix = torch.as_tensor(np.load(path)['matrix']).float()
    meta = json.loads(path.with_suffix('.meta.json').read_text())
    if list(matrix.shape) != meta['shape']:
        raise SystemExit(f'{path}: matrix shape {list(matrix.shape)} != meta {meta["shape"]}')
    grouped = {'refusal': {}, 'compliant': []}
    for row_meta, vector in zip(meta['rows'], matrix):
        row = dict(row_meta, vector=vector)
        if row_meta['role'] == 'refusal':
            grouped['refusal'].setdefault(row_meta['family_id'], []).append(row)
        else:
            grouped['compliant'].append(row)
    return grouped, meta.get('capture', {}), meta.get('layer')


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_checkpoint(path: Path, model, standardiser, config: dict) -> dict:
    state = sae_fit.checkpoint_state(model, standardiser)
    npz = {name: tensor.numpy().astype('float16') for name, tensor in state.items()}
    import numpy as np
    np.savez_compressed(path, **npz)
    return {
        'path': str(path),
        'sha256': sha256_file(path),
        'bytes': path.stat().st_size,
        'weight_digests': {name: sae_fit.digest_tensor(tensor) for name, tensor in state.items()},
        'config': config,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Train per-family top-k SAEs on captured residuals')
    parser.add_argument('--records', type=Path, default=Path('verification/p5/sae/inputs/records.pt'))
    parser.add_argument('--residuals', type=Path, default=None,
                        help='derived [rows, hidden] matrix from --dump-residuals (skips the model-free record load)')
    parser.add_argument('--out', type=Path, default=Path('verification/p5/sae'))
    parser.add_argument('--layer', type=int, default=10)
    parser.add_argument('--features', type=int, default=256)
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=20261123,
                        help='first seed; --seeds overrides with a list')
    parser.add_argument('--seeds', default=None,
                        help='comma list; the published checkpoint is the seed whose held-out FVU is the '
                             'median (ties go to the lowest seed). Default: --seed only')
    parser.add_argument('--holdout-every', type=int, default=4)
    parser.add_argument('--top-features', type=int, default=12)
    parser.add_argument('--merge-cert-seconds', type=int, default=540)
    parser.add_argument('--device', default='xpu')
    parser.add_argument('--dump-residuals', action='store_true')
    parser.add_argument('--rerun-check', action='store_true',
                        help='train twice per family and compare weight digests (gate 2: the run re-runs)')
    args = parser.parse_args(argv)

    seeds = [int(seed) for seed in args.seeds.split(',')] if args.seeds else [args.seed]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / 'checkpoints').mkdir(exist_ok=True)

    if args.residuals:
        grouped, capture, layer = load_residuals(args.residuals)
        inputs = {'residuals': str(args.residuals), 'residuals_sha256': sha256_file(args.residuals)}
    else:
        records, capture = read_records(args.records)
        grouped = group(records, args.layer)
        layer = args.layer
        inputs = {'records': str(args.records), 'records_sha256': sha256_file(args.records)}
        if args.dump_residuals:
            inputs['residuals_dump'] = dump_residuals(out, grouped, args.layer, capture)
            print(f"wrote {inputs['residuals_dump']['path']} "
                  f"{tuple(inputs['residuals_dump']['shape'])} sha256 {inputs['residuals_dump']['sha256'][:16]}")

    compliant = grouped['compliant']
    families = [family for family in FAMILIES if family in grouped['refusal']]
    extra = sorted(set(grouped['refusal']) - set(FAMILIES))
    if extra:
        print(f'ignoring families outside the plan: {extra}')
    if len(compliant) < 10:
        print(f'only {len(compliant)} compliant rows; the contrast side will be noisy')

    device = torch.device(args.device)
    if device.type == 'xpu' and not torch.xpu.is_available():
        raise SystemExit('xpu requested but unavailable; pass --device cpu to price a CPU run')
    device_name = torch.xpu.get_device_name(0) if device.type == 'xpu' else str(device)

    # The stop rule is priced BEFORE the first real step: a short probe on the largest
    # family's train rows, extrapolated to the plan's step count. If any family is priced
    # above the artifact's own merge+cert cost, the lane stops unmeasured.
    probe_family = max(families, key=lambda name: len(grouped['refusal'][name]))
    probe_train, _ = split_rows(grouped['refusal'][probe_family], args.holdout_every)
    probe_steps = 40
    _, _, probe_history = sae_fit.train_sae(
        [row['vector'] for row in probe_train], features=args.features, k=args.k, steps=probe_steps,
        batch=args.batch, lr=args.lr, seed=args.seed, device=args.device)
    seconds_per_step = probe_history['seconds'] / probe_steps
    projected = seconds_per_step * args.steps
    report = {'plan': {'features': args.features, 'k': args.k, 'steps': args.steps, 'batch': args.batch,
                       'lr': args.lr, 'seeds': seeds, 'layer': layer, 'holdout_every': args.holdout_every,
                       'median_rule': 'published checkpoint is the median-FVU seed'},
              'device': device_name, 'device_type': device.type, 'inputs': inputs,
              'capture': {key: value for key, value in capture.items()
                          if key not in ('targets', 'compute_backend')},
              'stop_rule': {'probe_family': probe_family, 'probe_steps': probe_steps,
                            'measured_seconds_per_step': round(seconds_per_step, 6),
                            'projected_seconds_per_family': round(projected, 3),
                            'merge_cert_seconds': args.merge_cert_seconds,
                            'fires': projected > args.merge_cert_seconds},
              'generated_tokens': 0, 'families': []}
    if report['stop_rule']['fires']:
        print(f"STOP RULE FIRED: {projected:.1f} s per family at {args.steps} steps is above the "
              f"merge+cert cost {args.merge_cert_seconds} s; training is unmeasured, not run")
        (out / 'train-report.json').write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
        return 2
    gpu_seconds = probe_history['seconds']
    started = time.perf_counter()
    for family in families:
        rows = grouped['refusal'].get(family, [])
        train, heldout = split_rows(rows, args.holdout_every)
        train_ids = {row['id'] for row in train}
        if train_ids & {row['id'] for row in heldout}:
            raise SystemExit(f'{family}: split is not disjoint')
        train_vectors = [row['vector'] for row in train]
        heldout_vectors = [row['vector'] for row in heldout]
        runs = []
        for seed in seeds:
            model, standardiser, history = sae_fit.train_sae(
                train_vectors, features=args.features, k=args.k, steps=args.steps, batch=args.batch,
                lr=args.lr, seed=seed, device=args.device, record_every=max(1, args.steps // 6))
            scored = sae_fit.score_heldout(model, standardiser, heldout_vectors, train_vectors,
                                           device=args.device)
            gpu_seconds += history['seconds']
            runs.append({'seed': seed, 'model': model, 'standardiser': standardiser,
                         'history': history, 'heldout': scored})
        # The published checkpoint is the median-FVU seed: with 7-8 held-out rows a single
        # seed's FVU is noise, and publishing the best seed would be selection on the test set.
        runs.sort(key=lambda run: (run['heldout']['fvu'], run['seed']))
        chosen = runs[(len(runs) - 1) // 2]
        model, standardiser, history, scored = (chosen['model'], chosen['standardiser'],
                                                chosen['history'], chosen['heldout'])
        priced = price(sum(run['history']['seconds'] for run in runs), args.merge_cert_seconds)
        spread = [run['heldout']['fvu'] for run in runs]

        cross_family = families[(families.index(family) + 1) % len(families)]
        cross_rows = grouped['refusal'].get(cross_family, [])
        cross = None
        if cross_rows:
            cross_split, _ = split_rows(cross_rows, args.holdout_every)
            cross = sae_fit.score_heldout(model, standardiser, [row['vector'] for row in cross_split],
                                          train_vectors, device=args.device)

        passed = scored['fvu'] < 1.0 and scored['fvu'] < scored['pca_control_fvu']
        status = 'trained' if passed else 'stopped'
        config = {'family': family, 'layer': layer, 'features': args.features, 'k': args.k,
                  'steps': args.steps, 'batch': args.batch, 'lr': args.lr, 'seed': chosen['seed'],
                  'seeds': [run['seed'] for run in runs],
                  'median_rule': 'checkpoint is the seed whose held-out FVU is the median',
                  'rows_train': len(train), 'rows_heldout': len(heldout),
                  'train_ids_sha256': hashlib.sha256(','.join(sorted(train_ids)).encode()).hexdigest(),
                  'heldout_ids': [row['id'] for row in heldout]}
        checkpoint = save_checkpoint(out / 'checkpoints' / f'{family}.npz', model, standardiser, config)
        rerun = None
        if args.rerun_check:
            again, _, again_history = sae_fit.train_sae(
                train_vectors, features=args.features, k=args.k, steps=args.steps, batch=args.batch,
                lr=args.lr, seed=chosen['seed'], device=args.device)
            rerun = {'weight_digests_match': sae_fit.digest_tensor(again.W_dec) == sae_fit.digest_tensor(model.W_dec),
                     'gpu_seconds': round(again_history['seconds'], 3)}
            gpu_seconds += again_history['seconds']

        table = sae_fit.feature_table(model, standardiser, train_vectors, [row['vector'] for row in compliant],
                                      top_k=args.top_features, device=args.device)
        entry = {'family': family, 'status': status, 'rows_train': len(train), 'rows_heldout': len(heldout),
                 'seed_published': chosen['seed'],
                 'seed_spread': {'values': [round(value, 6) for value in spread],
                                 'min': round(min(spread), 6), 'max': round(max(spread), 6)},
                 'per_seed': [{'seed': run['seed'], 'fvu': round(run['heldout']['fvu'], 6)}
                              for run in sorted(runs, key=lambda run: run['seed'])],
                 'split': {'train_ids': sorted(train_ids), 'heldout_ids': [row['id'] for row in heldout]},
                 'heldout': scored, 'cross_family': {'family': cross_family, **(cross or {})},
                 'gpu': priced, 'loss_trace': history['loss_trace'], 'checkpoint': checkpoint,
                 'rerun': rerun, 'features': table}
        report['families'].append(entry)
        print(f"{family:36s} train={len(train):3d} heldout={len(heldout):3d} "
              f"FVU={scored['fvu']:.4f} ({min(spread):.4f}-{max(spread):.4f}) "
              f"pca={scored['pca_control_fvu']:.4f} "
              f"cross={cross['fvu'] if cross else float('nan'):.4f} "
              f"{status} seed={chosen['seed']} gpu={priced['gpu_seconds']:.2f}s")
        if rerun and not rerun['weight_digests_match']:
            print(f'  RERUN MISMATCH for {family}: the same seed did not reproduce the weights')

    wall = time.perf_counter() - started
    trained = [entry for entry in report['families'] if entry['status'] == 'trained']
    report['totals'] = {'families_trained': len(trained), 'families_stopped': len(report['families']) - len(trained),
                        'gpu_seconds': round(gpu_seconds, 3), 'wall_seconds': round(wall, 3),
                        'generated_tokens': 0,
                        'stop_rule': f'a family stops if its own training is priced above the '
                                     f'merge+cert cost ({args.merge_cert_seconds} s); '
                                     f'fvu must beat 1.0 (train mean) and the rank-k PCA control'}
    (out / 'train-report.json').write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    table_path = out / 'feature-table.json'
    table_path.write_text(json.dumps({entry['family']: entry['features'] for entry in report['families']},
                                    indent=1, sort_keys=True) + '\n')
    (out / 'feature-table.md').write_text(_markdown(report, args.top_features))
    digests = {'train_report_sha256': sha256_file(out / 'train-report.json'),
               'feature_table_sha256': sha256_file(table_path),
               'checkpoints': {entry['family']: entry['checkpoint']['sha256'] for entry in report['families']},
               'inputs': inputs}
    (out / 'digests.json').write_text(json.dumps(digests, indent=1, sort_keys=True) + '\n')

    print(f"\nfamilies trained {len(trained)}/{len(report['families'])}; "
          f"gpu {gpu_seconds:.2f}s; generated tokens 0")
    return 0 if len(trained) >= 2 else 1


def _markdown(report: dict, top_features: int) -> str:
    lines = ['# P5 SAE feature table', '',
             'Top features by |activation contrast| (refusal minus compliant), per family.',
             f'Layer {report["plan"]["layer"]}, features {report["plan"]["features"]}, '
             f'k={report["plan"]["k"]}, seeds {report["plan"]["seeds"]} (median published).', '',
             '| family | status | held-out FVU | PCA-k FVU | features |',
             '|---|---|---:|---:|---:|']
    for entry in report['families']:
        lines.append(f"| {entry['family']} | {entry['status']} | {entry['heldout']['fvu']:.4f} "
                     f"| {entry['heldout']['pca_control_fvu']:.4f} | {len(entry['features'])} |")
    for entry in report['families']:
        lines += ['', f"## {entry['family']} ({entry['status']})", '',
                  f"rows: {entry['rows_train']} train / {entry['rows_heldout']} held out; "
                  f"gpu {entry['gpu']['gpu_seconds']} s", '',
                  '| feature | mean refusal | mean compliant | contrast | active ref/compl | cos to axis |',
                  '|---:|---:|---:|---:|---:|---:|']
        for row in entry['features'][:top_features]:
            lines.append(f"| {row['feature']} | {row['mean_refusal']:.3f} | {row['mean_compliant']:.3f} "
                         f"| {row['contrast']:+.3f} | {row['active_refusal']:.2f}/{row['active_compliant']:.2f} "
                         f"| {row['cos_to_axis']:+.3f} |")
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    raise SystemExit(main())
