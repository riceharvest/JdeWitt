# SPDX-License-Identifier: AGPL-3.0-or-later
"""P7 q25: write the cert spec the analysis card runs `release-certify` against.

The spec is data, and every field in it is either a measured operand (from
``operands.json``) or a path this preparation produced. Two things it must get
right, because certify refuses the packet without them:

* the artifact identity digest -- computed with ``certify``'s own function over
  the artifact block, so the shape-audit entries carry exactly the digest the
  certifier will recompute (R8: an audit of another artifact is not evidence);
* the wording-inventory digests -- from the shipped checker code, so a removal
  count is bound to the pattern set that produced it.

    python scripts/p7_q25_spec.py [--out verification/p7-q25-15b/spec.json]

Nothing here measures: the waves, shape audits, drift and fidelity are written by
``run_waves.sh``. The spec can be written before they exist, which is the point --
it is the contract the wave run has to satisfy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from jdewitt.certify import _identity_digest, inventory_digests  # noqa: E402

OUT_DIR = REPO / 'verification/p7-q25-15b'
# Evidence paths are written repository-relative so the spec survives the merge out
# of the worktree; the certifier resolves them from the repo root. Artifact paths are
# absolute because the GGUF files live outside the tree.
REL = 'verification/p7-q25-15b'
PORT = 8141
PINNED = ('-m {artifact} -c 4096 --parallel 4 --jinja --no-webui --host 127.0.0.1 '
          f'--port {PORT}')
ARM_SCALE = 1.5


def edited_tensors(artifact: str, axes_path: Path) -> list[str]:
    """The tensors the merge visited, discovered from the artifact itself.

    Same discovery `spanfit build --gguf-targets auto` used; the spec's edited
    count is a read of the served file, not a number typed from a plan.
    """
    import torch
    from jdewitt.gguf_adapters import discover_writeback_targets, gguf_weight_shapes

    axes = torch.load(axes_path, map_location='cpu', weights_only=False)['axes']
    found = discover_writeback_targets(gguf_weight_shapes(artifact), int(axes.shape[-1]))
    return sorted(name for names in found['targets'].values() for name in names)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='p7_q25_spec')
    parser.add_argument('--out', type=Path, default=OUT_DIR / 'spec.json')
    parser.add_argument('--operands', type=Path, default=OUT_DIR / 'operands.json')
    args = parser.parse_args(argv)

    operands = json.loads(args.operands.read_text())
    by_role = {entry['role']: entry for entry in operands['artifacts']}
    arm, base = by_role['arm-refit'], by_role['base']
    published = by_role['published']
    edited = edited_tensors(arm['path'], Path(operands['axes']))

    argv = PINNED.format(artifact=arm['path'])
    spec = {
        'schema_version': 1,
        'lineage': 'qwen25-1.5b',
        'arm': {
            'label': 'refit-sas15',
            'refit': True,
            'edited_tensors': len(edited),
            'edited_tensor_footprint': ('blk.0-27 x {attn_output.weight, ffn_down.weight} '
                                        '(28 layers x 2 writeback projections)'),
            'chosen_type': arm['quant']['dominant'],
            'scale': ARM_SCALE,
            'byte_cost': arm['bytes'],
            'byte_cost_source': 'the shipped arm file itself (operands.json), not a cost model',
            'note': ('REFIT of the shared axis from a CPU capture of the base, merged into the base '
                     'Q8_0 bytes at scale 1.5. The original q25-v4-dir000.gguf adapter is not on '
                     'this box; this artifact is not that adapter and must never be labelled SAS@1.5.'),
        },
        'artifact': {
            'path': arm['path'],
            'sha256': arm['sha256'],
            'sha256_prefix_16': arm['sha256_prefix_16'],
            'bytes': arm['bytes'],
            'recipe': {
                'command': arm['recipe'],
                'base_sha256_prefix_16': base['sha256_prefix_16'],
                'edit': ('shared sign-aligned axis, output space, attn_output + ffn_down, '
                         'blk.0-27, scale 1.5, merged into the base Q8_0 bytes'),
                'note': ('all other bytes verbatim (spanfit edit_tensors); no 16-bit emit and no '
                         'whole-file requantize'),
            },
        },
        'base': {
            'path': base['path'],
            'sha256': base['sha256'],
            'sha256_prefix_16': base['sha256_prefix_16'],
            'bytes': base['bytes'],
            'quant_type': base['quant']['dominant'],
        },
        'suites': {
            'removal': {'path': f'{REL}/suites/harmful-40.jsonl', 'in_clone': True},
            'held_out': {'path': 'verification/screens/q3-holdout-wider.jsonl', 'in_clone': True},
            'retention': {'path': 'verification/capability/capability-run-en.jsonl',
                          'in_clone': True},
        },
        'configs': {
            # One argv per axis, and it is the same argv: the retention comparison is
            # only a comparison when all three artifacts were measured at one config.
            axis: {'argv': argv} for axis in ('removal', 'retention', 'held_out')
        },
        'pinned_cert_argv': argv,
        'inputs': {
            'removal': {
                'wave': f'{REL}/waves/removal-arm',
                'suite': 'harmful-40.jsonl',
                'base_wave': f'{REL}/waves/removal-base',
                'competitor_wave': f'{REL}/waves/removal-published',
                'competitor_label': 'published build (josiefied-q25-v1, Q8_0)',
                'note': ('both wording inventories are graded from this one wave: score_response '
                         'stores refusal_check and refusal_check_wide at write time'),
            },
            'held_out': {
                'wave': f'{REL}/waves/heldout120-arm',
                'suite': 'verification/screens/q3-holdout-wider.jsonl',
                'competitor_wave': f'{REL}/waves/heldout120-published',
                'competitor_label': 'published build (josiefied-q25-v1, Q8_0)',
                'base_wave': f'{REL}/waves/heldout120-base',
            },
            'retention': {
                'wave': f'{REL}/waves/retention-arm',
                'base_wave': f'{REL}/waves/retention-base',
                'competitor_wave': f'{REL}/waves/retention-published',
                'suite': 'verification/capability/capability-run-en.jsonl',
            },
            'shape_audit': [],
            'wording_inventory': inventory_digests(),
            'drift': f'{REL}/drift.json',
            'fidelity': f'{REL}/fidelity.json',
        },
        # Not read by certify: the screens it does not gate on, recorded beside the
        # gated ones so the 24 held-out families are not lost and not misreported.
        'side_screens': {
            'holdout_24_families': {
                'wave': f'{REL}/waves/holdout24-arm',
                'base_wave': f'{REL}/waves/holdout24-base',
                'suite': f'{REL}/suites/holdout-24f.jsonl',
                'rows': 24,
                'note': ('the 24 held-out families the card calls out; certify floors held_out at '
                         '120 rows, so this is reported beside the axis, never as the axis'),
            },
            'retention_published': {
                'wave': f'{REL}/waves/retention-published',
                'note': 'the published build at the same 1024-cap retention argv',
            },
        },
    }
    digest = _identity_digest(spec['artifact'])
    spec['artifact_identity_digest'] = digest
    spec['inputs']['shape_audit'] = [
        {'path': f'{REL}/shape-audit-removal.json', 'axis': 'removal',
         'artifact_identity_digest': digest},
        {'path': f'{REL}/shape-audit-heldout.json', 'axis': 'held_out',
         'artifact_identity_digest': digest},
    ]
    spec['published'] = {'path': published['path'], 'sha256': published['sha256'],
                         'sha256_prefix_16': published['sha256_prefix_16'],
                         'bytes': published['bytes'], 'quant_type': published['quant']['dominant']}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spec, indent=1) + '\n')
    print(json.dumps({'spec': str(args.out), 'artifact_identity_digest': digest,
                      'wording_inventory': spec['inputs']['wording_inventory'],
                      'pinned_cert_argv': argv}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
