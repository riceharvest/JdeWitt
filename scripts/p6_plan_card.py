# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build the phase-6 plan card from the measured Spark 1.7B fit.

The p6 measurement card (``scripts/p6_runtime_equivalence.py``) fitted one plan
from a capture and built the adapter and the merged twin from it, but never
registered the plan: a plan-bundle could not be produced from a fit that only
existed as a pair of artifacts. This command closes that loop without inventing
numbers.

It re-derives the fit from the captured records with the same code the
measurement used (``spanfit.per_layer_axes``) and REFUSES unless the
reconstruction's ``directions_sha256`` equals the one recorded in the measured
JSON. It then writes the run-output shape ``registry.register_plan`` verifies
(directions payload + integrity pointer + provenance), registers the plan with
the measured record as its gates file, and archives it with
``registry.bundle_plan``.

Nothing here is a behaviour measurement: the gates file is the measured
``p6-runtime-equivalence`` record, and the card carries the fit's own
provenance (records sha256, directions sha256, layer footprint).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jdewitt.answers import atomic_json  # noqa: E402
from jdewitt.registry import bundle_plan, register_plan  # noqa: E402
from jdewitt.spanfit import load_records, per_layer_axes  # noqa: E402
from jdewitt.data import digest  # noqa: E402
from jdewitt.storage import file_digest  # noqa: E402


def fit(records_path, layers):
    """Per-layer unit axes for ``layers``, exactly as the measurement fitted them."""
    records = load_records(records_path)
    axes = per_layer_axes(records)
    directions = {}
    for layer in layers:
        axis = np.asarray(axes[layer].float().cpu(), dtype=np.float32)
        norm = float(np.linalg.norm(axis))
        if abs(norm - 1.0) > 1e-4:
            raise SystemExit(f'fitted axis for layer {layer} has norm {norm}, not 1')
        directions[layer] = axis
    return directions


def directions_sha(directions, layers):
    return hashlib.sha256(b''.join(directions[layer].tobytes() for layer in layers)).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument('--measured', type=Path, required=True,
                        help='measured p6-runtime-equivalence JSON that names the plan')
    parser.add_argument('--records', type=Path,
                        help='capture records (default: rows_source.records in --measured)')
    parser.add_argument('--work', type=Path, required=True, help='run-output directory to write')
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('-o', '--output', type=Path, required=True, help='plan-bundle tar')
    parser.add_argument('--model', default=None,
                        help='model identity recorded in the registry (default: capture.model)')
    parser.add_argument('--revision', default=None,
                        help='resolved revision (default: base sha256)')
    args = parser.parse_args(argv)

    measured = json.loads(args.measured.read_text())
    if measured.get('kind') != 'p6-runtime-equivalence':
        raise SystemExit(f'{args.measured} is not a p6-runtime-equivalence record')
    plan = measured.get('plan') or {}
    layers = [int(layer) for layer in plan.get('layers') or []]
    if not layers or layers != sorted(layers):
        raise SystemExit(f'{args.measured} records no sorted plan layer list')
    base = measured.get('base') or {}
    base_sha = str(base.get('sha256') or '')
    if len(base_sha) != 64:
        raise SystemExit(f'{args.measured} records no base artifact sha256')
    records_path = args.records or Path((measured.get('rows_source') or {}).get('records') or '')
    if not records_path.is_file():
        raise SystemExit(f'records {records_path} do not exist')
    recorded_records_sha = (measured.get('rows_source') or {}).get('records_sha256')
    if recorded_records_sha and recorded_records_sha != file_digest(records_path):
        raise SystemExit(f'{records_path} sha256 does not match the measured record; refusing')

    directions = fit(records_path, layers)
    sha = directions_sha(directions, layers)
    want = str(plan.get('directions_sha256') or '')
    if sha != want:
        raise SystemExit(f're-derived directions sha256 {sha} != measured {want}; the fit does '
                         f'not reproduce and no card may be cut from it')

    basis = torch.from_numpy(np.stack([directions[layer] for layer in layers]))[:, None, :].contiguous()
    if tuple(basis.shape) != (len(layers), int(plan.get('rank') or 1), int(plan.get('width') or 0)):
        raise SystemExit(f'basis shape {tuple(basis.shape)} disagrees with the measured plan')

    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    payload = work / 'directions.pt'
    torch.save(({'basis': basis, 'layers': layers, 'rank': int(plan.get('rank') or 1),
                 'footprint': plan.get('footprint')},
                {'source': plan.get('fit'), 'directions_sha256': sha,
                 'records': str(records_path), 'records_sha256': file_digest(records_path),
                 'note': 'per-layer unit axes, one row per layer in `layers`'}), payload)
    payload_sha = file_digest(payload)
    architecture = (measured.get('capture') or {}).get('architecture')
    fingerprint = {key: value for key, value in {
        'architecture': architecture,
        'layers': int((measured.get('capture') or {}).get('layers') or 0) or None,
        'hidden_size': int((measured.get('capture') or {}).get('hidden') or 0) or None,
        'rank': int(plan.get('rank') or 1),
        'dtype': (measured.get('capture') or {}).get('dtype'),
    }.items() if value is not None}
    identity_digest = digest({'base_sha256': base_sha, 'plan': plan})
    pointer = {'version': 1, 'payload': payload.name, 'sha256': payload_sha,
               'identity_digest': identity_digest, 'metadata': {'generation_fingerprint': fingerprint},
               'note': 'plan card payload for the p6 fitted plan; re-derived and sha-checked'}
    pointer['record_digest'] = digest(pointer)
    atomic_json(work / 'directions.checkpoint.json', pointer)
    model = args.model or (measured.get('capture') or {}).get('model') or 'Spark-X2.6-1.7B'
    provenance = {
        'version': 1,
        'config': {'model': model, 'method': 'perlayer', 'rank': int(plan.get('rank') or 1),
                   'seed': None,
                   'method_options': {'footprint': plan.get('footprint'), 'profile': plan.get('profile'),
                                      'basis': plan.get('basis'), 'layer_indices': layers,
                                      'fit': plan.get('fit')}},
        'resolved_revision': args.revision or base_sha,
        'directions_digest': sha,
        'source_files': {'records': file_digest(records_path),
                         'measured': file_digest(args.measured)},
        'note': 'registered from the measured fit so the plan card ships with the artifacts '
                'built from it; gates are the measured p6-runtime-equivalence record',
    }
    atomic_json(work / 'provenance.json', provenance)

    registered = register_plan(args.registry, work, gates_path=args.measured)
    bundling = bundle_plan(args.registry, registered['key'], args.output)
    result = {'plan_key': registered['key'], 'measured': registered['measured'],
              'directions_sha256': sha, 'payload_sha256': payload_sha,
              'bundle': str(args.output), 'bundle_sha256': bundling['bundle_sha256'],
              'meta_digest': bundling['meta_digest'], 'layers': layers,
              'basis_shape': list(basis.shape)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
