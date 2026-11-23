#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Capacity sweep: which dictionary size is identifiable at this row count?

The plan's capacity (dictionary size, active features) is chosen here, on the
same held-out rows the report uses, and the choice is recorded with its
alternatives: at 24 train rows per family a large over-complete dictionary cannot
be identified, so a plan that names 256 features by default would train something
whose FVU loses to a rank-k linear control of the same active rank. The rule is
the plan's: the held-out FVU of the median of three seeds must beat the rank-k
PVC control in at least three of the four families, and among the configurations
that clear it the lowest median FVU wins.

Writes one JSON: per configuration, per family, per-seed FVU, the median, and the
control. Costs device time only (generated tokens 0).
"""
from __future__ import annotations

import argparse
import json
import statistics
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

# (features, k): the plan's capacity candidates. k=2 is included because the row count,
# not the compute budget, is what binds: 24 train rows cannot identify more than a few
# directions per row, and a larger active set only gives the rank-k control more to match.
CONFIGS = (
    (32, 2), (64, 2), (64, 4), (64, 8), (128, 2), (128, 4), (256, 2), (256, 8), (512, 2),
)


def load_residuals(path: Path) -> tuple[dict, dict]:
    import numpy as np
    matrix = torch.as_tensor(np.load(path)['matrix']).float()
    meta = json.loads(path.with_suffix('.meta.json').read_text())
    grouped = {'refusal': {}, 'compliant': []}
    for row_meta, vector in zip(meta['rows'], matrix):
        row = dict(row_meta, vector=vector)
        if row_meta['role'] == 'refusal':
            grouped['refusal'].setdefault(row_meta['family_id'], []).append(row)
        else:
            grouped['compliant'].append(row)
    return grouped, meta


def split(rows: list[dict], holdout_every: int) -> tuple[list[dict], list[dict]]:
    ordered = sorted(rows, key=lambda row: row['id'])
    heldout = [row for index, row in enumerate(ordered) if index % holdout_every == holdout_every - 1]
    train = [row for index, row in enumerate(ordered) if index % holdout_every != holdout_every - 1]
    return train, heldout


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Sweep SAE capacity at the captured row count')
    parser.add_argument('--residuals', type=Path, default=Path('verification/p5/sae/residuals-layer10.npz'))
    parser.add_argument('--out', type=Path, default=Path('verification/p5/sae/capacity-sweep.json'))
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seeds', default='20261123,20261124,20261125')
    parser.add_argument('--holdout-every', type=int, default=4)
    parser.add_argument('--device', default='xpu')
    args = parser.parse_args(argv)

    seeds = [int(seed) for seed in args.seeds.split(',')]
    grouped, meta = load_residuals(args.residuals)
    families = [family for family in FAMILIES if family in grouped['refusal']]
    results = {}
    started = time.perf_counter()
    gpu_seconds = 0.0
    for features, k in CONFIGS:
        entry = {}
        wins = 0
        for family in families:
            train, heldout = split(grouped['refusal'][family], args.holdout_every)
            fvus = []
            for seed in seeds:
                model, standardiser, history = sae_fit.train_sae(
                    [row['vector'] for row in train], features=features, k=k, steps=args.steps,
                    batch=min(args.batch, len(train)), lr=args.lr, seed=seed, device=args.device)
                fvus.append(sae_fit.score_heldout(model, standardiser, [row['vector'] for row in heldout],
                                                  [row['vector'] for row in train], device=args.device)['fvu'])
                gpu_seconds += history['seconds']
            control = sae_fit.pca_control_fvu(torch.stack([row['vector'] for row in train]),
                                              torch.stack([row['vector'] for row in heldout]), k)
            median = statistics.median(fvus)
            wins += median < control
            entry[family] = {'fvu_per_seed': [round(value, 6) for value in fvus],
                             'fvu_median': round(median, 6), 'pca_control_fvu': round(control, 6),
                             'beats_control': median < control}
        results[f'features={features},k={k}'] = {'families': entry, 'wins': wins,
                                                 'families_total': len(families)}
    payload = {'layer': meta.get('layer'), 'rows': meta.get('shape', [None])[0],
               'configs': results, 'seeds': seeds, 'steps': args.steps, 'batch': args.batch,
               'lr': args.lr, 'holdout_every': args.holdout_every,
               'device_seconds': round(gpu_seconds, 3), 'wall_seconds': round(time.perf_counter() - started, 3),
               'generated_tokens': 0,
               'rule': 'median-of-seeds held-out FVU < rank-k PCA control; pick lowest median FVU '
                       'among configs that win in >= 3/4 families'}
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + '\n')
    for name, entry in results.items():
        print(f'{name:16s} wins {entry["wins"]}/{entry["families_total"]}  '
              + '  '.join(f'{family[:14]}:{value["fvu_median"]:.3f}/{value["pca_control_fvu"]:.3f}'
                          for family, value in entry['families'].items()))
    print(f'device seconds {gpu_seconds:.1f}; generated tokens 0; wrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
