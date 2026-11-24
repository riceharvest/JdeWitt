# SPDX-License-Identifier: AGPL-3.0-or-later
"""P7 q25: realized fidelity of the REFIT arm, measured on the shipped bytes.

The q3 arm's fidelity was a codec *proxy* because its artifact was deleted: seeded
unit axes at the arm's footprint, re-encoded through Q8_0, `arm_measurement=false`.
This lineage's arm exists (the refit merged into base-q8.gguf), so the honest
reading is the delivered one: `ggufio.compare_edit` takes dequant(arm) - dequant(base)
and compares it with the projection the writer was asked to apply at the strength
it was applied. A low cos here means the Q8_0 grid, not the fitted axis, decided
what ships.

    python scripts/p7_q25_fidelity.py \
        --base verification/p7-q25-15b/artifacts/base-q8.gguf \
        --arm  verification/p7-q25-15b/artifacts/arm-refit-q8.gguf \
        --axes verification/p7-q25-15b/records/refit-axes.pt \
        --out  verification/p7-q25-15b/fidelity.json [--dry] [--limit N]

`--dry` prints which tensors would be measured (discovered from the artifact, the
same discovery `spanfit build --gguf-targets auto` used) and exits without reading
tensor data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def plan(base: Path, axes_path: Path, limit: int | None) -> dict:
    """{tensor name: (axis row index, )} from the artifact's own tensor names."""
    import torch
    from jdewitt.gguf_adapters import discover_writeback_targets, gguf_weight_shapes

    axes = torch.load(axes_path, map_location='cpu', weights_only=False)['axes']
    shapes = gguf_weight_shapes(str(base))
    found = discover_writeback_targets(shapes, int(axes.shape[-1]))
    targets = {layer: names for layer, names in found['targets'].items()
               if 0 <= layer < int(axes.shape[0])}
    if not targets:
        raise SystemExit(f'{base} carries no residual-write projection at width {int(axes.shape[-1])}')
    pairs = [(name, layer) for layer, names in sorted(targets.items()) for name in names]
    if limit:
        pairs = pairs[:limit]
    return {'axes': axes, 'pairs': pairs, 'layers': sorted(targets),
            'unmapped': found['unmapped_writeback_width']}


def main(argv=None):
    parser = argparse.ArgumentParser(prog='p7_q25_fidelity')
    parser.add_argument('--base', type=Path, required=True, help='pristine Q8_0 artifact the merge started from')
    parser.add_argument('--arm', type=Path, required=True, help='served artifact (base + merged axis)')
    parser.add_argument('--axes', type=Path, required=True, help='spanfit axes .pt the merge applied')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scale', type=float, default=1.5)
    parser.add_argument('--space', choices=('output', 'input'), default='output')
    parser.add_argument('--limit', type=int, default=None, help='measure only the first N tensors (smoke)')
    parser.add_argument('--dry', action='store_true', help='print the plan, read no tensor data')
    args = parser.parse_args(argv)

    for path in (args.base, args.arm, args.axes):
        if not path.is_file():
            raise SystemExit(f'missing operand: {path}')
    if args.base == args.arm:
        raise SystemExit('--base and --arm are the same file: that would measure a zero edit')

    built = plan(args.base, args.axes, args.limit)
    names = [name for name, _layer in built['pairs']]
    if args.dry:
        print(json.dumps({'dry': True, 'base': str(args.base), 'arm': str(args.arm),
                          'axes': list(built['axes'].shape), 'scale': args.scale,
                          'space': args.space, 'tensors': names,
                          'count': len(names), 'layers': built['layers'],
                          'unmapped_writeback_width': built['unmapped']}, indent=1))
        return 0

    from jdewitt.ggufio import compare_edit

    edits = {name: (built['axes'][layer].numpy(), args.space) for name, layer in built['pairs']}
    report = compare_edit(str(args.base), str(args.arm), edits, scale=args.scale)
    if not report['tensors']:
        raise SystemExit('compare_edit measured no tensor: the plan and the artifact disagree')
    failures = [row for row in report['tensors'] if 'error' in row]
    if failures:
        raise SystemExit(f'compare_edit failed on {len(failures)} tensor(s): {failures[:2]}')
    # compare_edit writes edited_type/source_type; the packet's codec table reads
    # qtype (the name preview_edit uses). Carry it here so both writers are one shape.
    for row in report['tensors']:
        row.setdefault('qtype', row.get('edited_type') or row.get('source_type'))

    payload = {
        'schema_version': 1,
        'lineage': 'qwen25-1.5b',
        'label': 'refit-sas15',
        'measured': True,
        # The arm exists here, so this is the delivered edit, not a codec proxy.
        'arm_measurement': True,
        'method': ("ggufio.compare_edit: dequant(arm) - dequant(base) against the projection "
                   "spanfit build applied at the recorded scale, on the shipped Q8_0 bytes"),
        'artifact_sha256': sha256_file(args.arm),
        'artifact_bytes': args.arm.stat().st_size,
        'base_sha256': sha256_file(args.base),
        'axes': str(args.axes), 'scale': args.scale, 'axis_space': args.space,
        'tensors': report['tensors'],
        'sampled': report['sampled'],
        'min_cos': report['min_cos'],
        'mean_cos': report['mean_cos'],
        'min_snr': min((row['snr'] for row in report['tensors']), default=None),
        'note': 'a low cos here means the codec, not the axis, decides what ships',
        'scope': ('the served artifact itself: the refit axis merged into base-q8.gguf at scale '
                  f'{args.scale}, {report["sampled"]} edited tensors (o_proj + down_proj), '
                  'every layer that carries a residual-write projection'),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + '\n')
    print(json.dumps({key: payload[key] for key in
                      ('sampled', 'min_cos', 'mean_cos', 'min_snr', 'artifact_sha256')}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
