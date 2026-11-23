#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The arithmetic of a per-family SAE train step, printed before anything trains.

Inputs are row counts, dictionary shapes and step counts; nothing here loads a
model, generates a token or touches a capture file. Output is the table
``docs/SAE-COMPUTE-PLAN.md`` quotes, in ``key = value`` form so
``--compare docs/SAE-COMPUTE-PLAN.md`` can re-read the doc and prove the two
agree (exit 1 on any mismatch).

The step FLOPs estimate is ``2 * batch * features * hidden * 3``: three dense
matmuls per row (encode, decode, both backward passes at ~2x the forward), which
is the top-k SAE's actual work because the active set is fixed at ``k`` and the
sparsity never removes a matmul.

Step rate: a GPU-hour derived from a rate that was not measured on this device is
labelled ``ASSUMPTION`` next to the number. ``--timing`` runs the real trainer
for ``--timing-steps`` steps on the target device and prints ``measured``.
R1: SAE training generates ZERO tokens -- the cost is device time, not inference
volume -- and the output says so on its own line.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ASSUMED_STEPS_PER_SECOND = 1000.0  # only ever printed with the ASSUMPTION label


def arithmetic(rows: int, families: int, layers: int, hidden: int, features: int, k: int,
               steps: int, batch: int, rate: float | None, merge_cert_seconds: int,
               rate_label: str | None = None) -> dict:
    activation_bytes = layers * hidden * 4
    train_matrix_bytes = features * hidden * 4
    step_flops = 2 * batch * features * hidden * 3
    measured = rate is not None and (rate_label or 'measured') == 'measured'
    rate = float(rate) if measured else ASSUMED_STEPS_PER_SECOND
    seconds_per_family = steps / rate
    seconds_total = seconds_per_family * families
    return {
        'rows': rows,
        'families': families,
        'layers': layers,
        'hidden': hidden,
        'features': features,
        'k': k,
        'steps': steps,
        'batch': batch,
        'activation_bytes_per_row': activation_bytes,
        'train_matrix_bytes': train_matrix_bytes,
        'step_flops': step_flops,
        'steps_per_second': round(rate, 3),
        'step_rate_source': 'measured' if measured else 'ASSUMPTION',
        'gpu_seconds_per_family': round(steps / rate, 3),
        'gpu_seconds_total': round(seconds_total, 3),
        'gpu_hours_total': round(seconds_total / 3600.0, 6),
        'merge_cert_seconds': merge_cert_seconds,
        'generated_tokens': 0,
        'stop_rule_gpu_seconds_per_family': merge_cert_seconds,
        'stop_rule_fires': bool(steps / rate > merge_cert_seconds),
    }


def timing(batch: int, features: int, hidden: int, k: int, steps: int, device: str) -> float:
    """Measured steps/second of the real trainer on this device (torch import stays local).

    A 20-step warm-up is paid first and excluded: on XPU the first kernels
    compile, so a raw short run under-reports the steady rate the plan is priced at.
    """
    import torch

    from jdewitt import sae_fit

    rows = torch.randn(batch, hidden)
    sae_fit.train_sae(rows, features=features, k=k, steps=20, batch=batch,
                      lr=1e-3, seed=20261123, device=device)
    _, _, history = sae_fit.train_sae(rows, features=features, k=k, steps=steps, batch=batch,
                                     lr=1e-3, seed=20261123, device=device)
    return steps / history['seconds']


def print_table(numbers: dict) -> None:
    print('SAE train budget (one row per number the doc quotes)')
    for key, value in numbers.items():
        print(f'  {key} = {value}')
    print("  note = generated tokens are 0: SAE training is device time, not inference volume (R1)")
    label = numbers['step_rate_source']
    print(f'  note = every GPU number above is {label} at steps_per_second={numbers["steps_per_second"]}')


DOC_BLOCK = re.compile(r'<!-- sae-budget:start -->(.*?)<!-- sae-budget:end -->', re.S)
DOC_LINE = re.compile(r'^\s*(\w+)\s*=\s*(.+?)\s*$')


def read_doc_table(path: Path) -> dict:
    text = path.read_text()
    match = DOC_BLOCK.search(text)
    if not match:
        raise SystemExit(f'{path}: no <!-- sae-budget:start --> block')
    table = {}
    for line in match.group(1).splitlines():
        hit = DOC_LINE.match(line)
        if hit:
            table[hit.group(1)] = hit.group(2)
    return table


def compare(numbers: dict, path: Path) -> int:
    table = read_doc_table(path)
    mismatches = []
    for key, value in numbers.items():
        if key not in table:
            mismatches.append(f'{key}: missing from {path}')
            continue
        doc_value = table[key]
        try:
            if isinstance(value, bool):
                same = doc_value.strip().lower() == str(value).lower()
            elif isinstance(value, int):
                same = int(doc_value) == value
            elif isinstance(value, float):
                same = abs(float(doc_value) - value) <= max(1e-6, abs(value) * 1e-9)
            else:
                same = doc_value == str(value)
        except ValueError:
            same = False
        if not same:
            mismatches.append(f'{key}: doc={doc_value} script={value}')
    extra = sorted(set(table) - set(numbers))
    if extra:
        mismatches.append(f'{path} carries keys the script does not produce: {extra}')
    if mismatches:
        print(f'doc numbers != script output ({len(mismatches)} mismatch(es))')
        for line in mismatches:
            print(f'  {line}')
        return 1
    print(f'doc numbers == script output ({len(numbers)} keys, {path})')
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='SAE train budget: bytes, FLOPs, GPU time, stop rule')
    parser.add_argument('--rows', type=int, default=168, help='captured rows in the suite')
    parser.add_argument('--families', type=int, default=4)
    parser.add_argument('--layers', type=int, default=28)
    parser.add_argument('--hidden', type=int, default=2048)
    parser.add_argument('--features', type=int, default=256)
    parser.add_argument('--k', type=int, default=16, help='active features per row (top-k SAE)')
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--merge-cert-seconds', type=int, default=540,
                        help='the artifact cost the stop rule compares against (ROADMAP phase 2: ~9 min)')
    parser.add_argument('--timing', action='store_true',
                        help='measure the step rate on the target device instead of assuming it')
    parser.add_argument('--rate', type=float, default=None,
                        help='a previously measured steps/second (so --compare reproduces the doc '
                             'without a fresh timing run); pass --rate-label ASSUMPTION if it was not measured')
    parser.add_argument('--rate-label', choices=('measured', 'ASSUMPTION'), default='measured',
                        help='how the --rate number was obtained')
    parser.add_argument('--timing-steps', type=int, default=100)
    parser.add_argument('--device', default='xpu', help='device for --timing (the plan\'s target GPU)')
    parser.add_argument('--compare', type=Path, default=None, help='re-read the plan doc and diff its table')
    parser.add_argument('--json', action='store_true', help='emit the numbers as JSON instead of a table')
    args = parser.parse_args(argv)

    if args.timing and args.rate:
        raise SystemExit('pass --timing or --rate, not both: one rate, one label')
    rate = args.rate
    if args.timing:
        rate = timing(args.batch, args.features, args.hidden, args.k, args.timing_steps, args.device)
    numbers = arithmetic(args.rows, args.families, args.layers, args.hidden, args.features, args.k,
                         args.steps, args.batch, rate, args.merge_cert_seconds,
                         args.rate_label if args.rate else None)
    if args.json:
        import json
        print(json.dumps(numbers, indent=1, sort_keys=True))
    else:
        print_table(numbers)
    if args.compare:
        return compare(numbers, args.compare)
    return 0


if __name__ == '__main__':
    sys.exit(main())
