# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shape saved P7 measurements into the exact inputs `release-certify` reads.

`release-certify` (jdewitt/certify.py) is deliberately strict about the shape of
each input: a fidelity file must carry ``measured``/``min_cos``/``mean_cos``/
``sampled``; a drift file must carry ``control_delta_ln_ppl`` and ``budget``; a
spec must name the artifact identity and every wave. The measuring tools
(`scripts/p4f_gate.py fidelity`, `scripts/refusal_ppl_screen.py`,
`scripts/refusal_shape_audit.py`) emit their own reports, which are richer and
differently named. This module is the single translation step, kept out of the
shell so it can be read and self-checked without a model:

    python verification/p7-minicpm/cert_inputs.py fidelity  --raw R --out O ...
    python verification/p7-minicpm/cert_inputs.py drift     --screen S --out O ...
    python verification/p7-minicpm/cert_inputs.py audit-identity --audit A --digest D
    python verification/p7-minicpm/cert_inputs.py spec      --out O [flags]
    python verification/p7-minicpm/cert_inputs.py --self-check

Nothing here measures a model: every number is lifted from a saved artifact and
labelled with where it came from. A derivation that cannot find its number exits
non-zero instead of writing a null the certifier would have to guess about.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LINEAGE = 'minicpm-v47'
MODEL_NAME = 'openbmb/MiniCPM-V-4.7-1B'


def read_json(path: Path) -> dict:
    path = Path(path).expanduser()
    if not path.is_file():
        raise SystemExit(f'{path} does not exist')
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise SystemExit(f'{path} is not JSON: {error}') from error


def write_json(path: Path, payload: dict) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1, sort_keys=True) + '\n'
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)


# ---------------------------------------------------------------- fidelity
def from_fidelity(raw_path: Path, artifact_sha: str, artifact_bytes: int,
                  label: str, scale: float) -> dict:
    """`p4f_gate.py fidelity` -> the certifier's realized-fidelity block.

    The gate's report is `compare_edit` over every discovered residual-write
    tensor: full-tensor decode of the base, the same tensor in the merged file,
    and the cosine between the delta the writer intended and the delta stored.
    That is a measurement of the arm's own bytes, so `arm_measurement` is true
    here (unlike a codec proxy run on a different artifact).
    """
    raw = read_json(raw_path)
    rows = [row for row in raw.get('per_tensor') or [] if row.get('cos') is not None]
    if not rows:
        raise SystemExit(f'{raw_path} holds no per-tensor cos readings; the fidelity '
                         'input would be empty and certify must refuse it')
    summary = raw.get('summary') or {}
    cos = [float(row['cos']) for row in rows]
    tensors = [{'name': row.get('name'), 'qtype': row.get('qtype'),
                'shape': row.get('shape'), 'cos': row.get('cos'),
                'norm_ratio': row.get('norm_ratio'), 'snr': row.get('snr'),
                'delta_norm': row.get('delta_norm'), 'noise_norm': row.get('noise_norm')}
               for row in rows]
    return {
        'schema_version': 1,
        'lineage': LINEAGE,
        'label': label,
        'measured': True,
        'arm_measurement': True,
        'method': ('ggufio.compare_edit over the residual-write tensors the merge '
                   'discovered: full-tensor decode of base and merged bytes, cosine '
                   'between the intended and the stored delta'),
        'artifact_sha256': artifact_sha,
        'artifact_bytes': artifact_bytes,
        'scale': scale,
        'sampled': len(rows),
        'min_cos': round(min(cos), 4),
        'mean_cos': round(sum(cos) / len(cos), 4),
        'max_cos': round(max(cos), 4),
        'min_snr': summary.get('min_snr'),
        'codec_event': summary.get('codec_event'),
        'min_edit_cos_floor': summary.get('min_edit_cos_floor'),
        'tensors': tensors,
        'scope': ('the merged artifact measured against its own base bytes (the file the '
                  'waves served), not a codec proxy and not an adapter'),
        'source': str(raw_path),
        'note': ('a tensor below the floor cannot hold this edit; that is a codec event, '
                 'labelled as one, not a reason to raise the scale'),
    }


# ---------------------------------------------------------------- drift
def from_drift(screen_path: Path, artifact: str, label: str, budget: float) -> dict:
    """`refusal_ppl_screen.py` -> the certifier's drift block.

    The screen reads three corpora by prefill only: refusal, delivery and a benign
    control. The control delta (merged minus base, ln PPL on the control corpus)
    is generic drift from quantization/tone; the budget is what the lineage has
    always used. A missing reading is a refusal, never a zero.
    """
    raw = read_json(screen_path)
    artifacts = raw.get('artifacts') or {}
    if artifact not in artifacts:
        raise SystemExit(f'{screen_path} has no artifact {artifact!r}; it holds '
                         f'{sorted(artifacts)}')
    row = artifacts[artifact]
    delta = row.get('delta_control')
    if delta is None:
        raise SystemExit(f'{screen_path} records no delta_control for {artifact!r}; '
                         'the control corpus did not produce a reading')
    refusal = (row.get('refusal') or {}).get('ln_ppl')
    delivery = (row.get('delivery') or {}).get('ln_ppl')
    return {
        'schema_version': 1,
        'lineage': LINEAGE,
        'label': label,
        'measured': True,
        'instrument': ('llama-perplexity, prefill only, no generated tokens; control '
                       'corpus drift of the merged bytes against the base, one runner '
                       'config (ctx 512, 16 threads)'),
        'control_delta_ln_ppl': delta,
        'control_ln_ppl': (row.get('control') or {}).get('ln_ppl'),
        'budget': budget,
        'inside_budget': abs(float(delta)) <= float(budget),
        'delta_BR': row.get('delta_BR'),
        'BR': row.get('BR'),
        'refusal_ln_ppl': refusal,
        'delivery_ln_ppl': delivery,
        'source': str(screen_path),
        'source_pointer': f'artifacts.{artifact}.delta_control',
        'note': ('the control drift is inside the budget, so a BR move is not generic '
                 'drift; a control reading outside the budget is reported as drift, '
                 'never subtracted away'),
    }


# ---------------------------------------------------------------- shape audit
def attach_identity(audit_path: Path, digest: str) -> dict:
    """Stamp the audit with the artifact identity it was run on (R8).

    `refusal_shape_audit.py` reads saved predictions and knows nothing about which
    file produced them. `release-certify` refuses an audit that does not declare
    the artifact it read, so the digest is written into the document itself rather
    than only into the spec: a stranger reading the audit file can see which bytes
    it describes.
    """
    document = read_json(audit_path)
    document['artifact_identity_digest'] = digest
    document['artifact_identity_digest_source'] = (
        'sha256 + bytes + spanfit recipe of the merged served artifact; the digest '
        'release-certify computes from the spec artifact block (R8: never an adapter)')
    write_json(audit_path, document)
    return document


# ---------------------------------------------------------------- spec
def build_spec(**kw) -> tuple[dict, str]:
    """Assemble a release-certify spec and the identity digest it binds.

    The identity is the digest `jdewitt.certify` computes over
    ``{sha256, sha256_prefix_16, bytes, recipe{command, base_sha256_prefix_16,
    edit}}``. It is computed with the shipped function so the spec and the packet
    cannot disagree about what was served.
    """
    sys.path.insert(0, str(REPO))
    from jdewitt.certify import inventory_digests  # noqa: E402
    try:
        from jdewitt.certify import _identity_digest as identity_digest  # noqa: E402
    except ImportError as error:  # pragma: no cover - the shipped module is the contract
        raise SystemExit(f'jdewitt.certify does not expose _identity_digest: {error}')

    artifact = {
        'path': kw['artifact'],
        'sha256': kw['artifact_sha'],
        'sha256_prefix_16': kw['artifact_sha'][:16],
        'bytes': kw['artifact_bytes'],
        'recipe': {'command': kw['recipe'],
                   'base_sha256_prefix_16': kw['base_sha'][:16],
                   'edit': kw['edit']},
    }
    base = {'path': kw['base'], 'sha256': kw['base_sha'],
            'sha256_prefix_16': kw['base_sha'][:16], 'bytes': kw['base_bytes']}
    digests = inventory_digests()
    spec = {
        'schema_version': 1,
        'lineage': LINEAGE,
        'model': {'name': MODEL_NAME, 'revision': kw['revision'],
                  'quant_type': 'Q4_K_M'},
        'arm': {
            'label': kw['arm_label'],
            'edited_tensors': kw['edited_tensors'],
            'edited_tensor_footprint': kw['footprint'],
            'chosen_type': 'Q4_K_M',
            'scale': kw['scale'],
            'axis': 'per-layer refit axes (spanfit axes --shape perlayer), output space',
            'byte_cost': kw['artifact_bytes'],
        },
        'artifact': artifact,
        'base': base,
        'suites': {
            'removal': {'path': kw['removal_suite'], 'in_clone': True},
            'held_out': {'path': kw['heldout_suite'], 'in_clone': True},
            'retention': {'path': kw['retention_suite'], 'in_clone': True},
        },
        'configs': {
            'removal': {'argv': kw['config_removal']},
            'held_out': {'argv': kw['config_heldout']},
            'retention': {'argv': kw['config_retention']},
        },
        'pinned_cert_argv': kw['pinned_argv'],
        'inputs': {
            'removal': {'wave': kw['removal_wave'], 'suite': kw['removal_suite']},
            'held_out': {'wave': kw['heldout_wave'], 'suite': kw['heldout_suite']},
            'retention': {'wave': kw['retention_wave'],
                          'base_wave': kw['retention_base_wave'],
                          'suite': kw['retention_suite']},
            'shape_audit': [
                {'path': kw['audit_removal'], 'axis': 'removal',
                 'artifact_identity_digest': identity_digest(artifact)},
                {'path': kw['audit_heldout'], 'axis': 'held_out',
                 'artifact_identity_digest': identity_digest(artifact)},
            ],
            'wording_inventory': {'shipped': digests['shipped'],
                                  'wide': digests['wide']},
            'drift': kw['drift'],
            'fidelity': kw['fidelity'],
        },
        'caps': {'removal': kw['cap_removal'], 'held_out': kw['cap_heldout'],
                 'retention': kw['cap_retention']},
        'note': ('one arm, both axes, one pinned serve config per axis; the 2048/2 '
                 'comparison lives in comparison.json beside this spec and is never '
                 'substituted for the 4096/4 removal number (R9.4)'),
    }
    spec_identity = identity_digest(artifact)
    spec['artifact_identity_digest'] = spec_identity
    return spec, spec_identity


def command_fidelity(args) -> int:
    payload = from_fidelity(args.raw, args.artifact_sha, args.artifact_bytes,
                            args.label, args.scale)
    write_json(args.out, payload)
    print(json.dumps({key: payload[key] for key in
                      ('sampled', 'min_cos', 'mean_cos', 'max_cos', 'min_snr')}))
    return 0


def command_drift(args) -> int:
    payload = from_drift(args.screen, args.artifact, args.label, args.budget)
    write_json(args.out, payload)
    print(json.dumps({key: payload[key] for key in
                      ('control_delta_ln_ppl', 'budget', 'inside_budget', 'delta_BR')}))
    return 0


def command_audit_identity(args) -> int:
    document = attach_identity(args.audit, args.digest)
    print(json.dumps({'audit': str(args.audit), 'artifact_identity_digest': args.digest,
                      'audits': len(document.get('audits') or [])}))
    return 0


def command_spec(args) -> int:
    spec, digest = build_spec(**vars(args))
    write_json(args.out, spec)
    print(json.dumps({'spec': str(args.out), 'artifact_identity_digest': digest}))
    return 0


def self_check() -> int:
    import tempfile

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw = root / 'fidelity-raw.json'
        write_json(raw, {'per_tensor': [
            {'name': 'blk.0.attn_output.weight', 'qtype': 'Q4_K_M', 'shape': [1024, 1024],
             'cos': 0.8, 'norm_ratio': 1.0, 'snr': 1.2, 'delta_norm': 0.1, 'noise_norm': 0.05},
            {'name': 'blk.0.ffn_down.weight', 'qtype': 'Q4_K_M', 'shape': [3584, 1024],
             'cos': 0.6, 'norm_ratio': 1.0, 'snr': 0.9, 'delta_norm': 0.1, 'noise_norm': 0.08},
            {'name': 'blk.1.ffn_down.weight', 'error': 'shape mismatch'}],
            'summary': {'min_cos': 0.6, 'mean_cos': 0.7, 'min_snr': 0.9,
                        'codec_event': True, 'min_edit_cos_floor': 0.9}})
        fid = from_fidelity(raw, 'a' * 64, 672004000, 'arm', 1.0)
        checks += [('fidelity skips the error row', fid['sampled'] == 2),
                   ('fidelity min_cos from the rows', fid['min_cos'] == 0.6),
                   ('fidelity is an arm measurement', fid['arm_measurement'] is True),
                   ('fidelity keeps the codec event', fid['codec_event'] is True)]
        screen = root / 'screen.json'
        write_json(screen, {'artifacts': {
            'base': {'control': {'ln_ppl': 0.2}},
            'merged': {'control': {'ln_ppl': 0.21}, 'delta_control': 0.01,
                       'delta_BR': -0.2, 'refusal': {'ln_ppl': 1.0},
                       'delivery': {'ln_ppl': 1.2}}}})
        drift = from_drift(screen, 'merged', 'arm', 0.02)
        checks += [('drift reads delta_control', drift['control_delta_ln_ppl'] == 0.01),
                   ('drift inside budget', drift['inside_budget'] is True),
                   ('drift reports its pointer',
                    drift['source_pointer'] == 'artifacts.merged.delta_control')]
        audit = root / 'audit.json'
        write_json(audit, {'audits': [{'label': 'removal', 'rows': 40, 'shapes': {}}]})
        attach_identity(audit, 'b' * 64)
        checks.append(('audit carries the identity digest',
                       read_json(audit)['artifact_identity_digest'] == 'b' * 64))
        spec, digest = build_spec(
            artifact=str(root / 'merged.gguf'), artifact_sha='c' * 64,
            artifact_bytes=123, base=str(root / 'base.gguf'), base_sha='d' * 64,
            base_bytes=456, recipe='python -m jdewitt.spanfit build ...',
            edit='per-layer axes, output space', arm_label='arm', scale=1.0,
            edited_tensors=48, footprint='blk.0-23 x {attn_output, ffn_down}',
            removal_suite='verification/screens/advbench-40.jsonl',
            heldout_suite='verification/p7-minicpm/heldout-24f.jsonl',
            retention_suite='verification/capability/capability-run-en.jsonl',
            config_removal='-m x -c 4096 --parallel 4', config_heldout='-m x -c 4096',
            config_retention='-m x -c 4096', pinned_argv='-m x -c 4096 --parallel 4',
            removal_wave='w/r', heldout_wave='w/h', retention_wave='w/t',
            retention_base_wave='w/b', audit_removal=str(audit),
            audit_heldout=str(audit), drift=str(screen), fidelity=str(raw),
            cap_removal=1024, cap_heldout=1024, cap_retention=1024,
            revision='286817f1132b998913a129a221c89b4402b4ab80', out=None)
        checks += [('spec binds the artifact sha', spec['artifact']['sha256'] == 'c' * 64),
                   ('spec names both inventories',
                    bool(spec['inputs']['wording_inventory']['shipped'])),
                   ('spec identity digest is 64 hex', len(digest) == 64),
                   ('spec audit entries carry the digest',
                    all(entry['artifact_identity_digest'] == digest
                        for entry in spec['inputs']['shape_audit']))]
    for label, ok in checks:
        print(f'{"PASS" if ok else "FAIL"} {label}')
    return 0 if all(ok for _, ok in checks) else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--self-check', action='store_true',
                        help='synthetic fixtures; reads and writes no repo files')
    sub = parser.add_subparsers(dest='command')

    fid = sub.add_parser('fidelity', help='p4f_gate fidelity report -> cert fidelity input')
    fid.add_argument('--raw', type=Path, required=True)
    fid.add_argument('--out', type=Path, required=True)
    fid.add_argument('--artifact-sha', required=True)
    fid.add_argument('--artifact-bytes', type=int, required=True)
    fid.add_argument('--label', required=True)
    fid.add_argument('--scale', type=float, required=True)
    fid.set_defaults(func=command_fidelity)

    drift = sub.add_parser('drift', help='ppl screen -> cert drift input')
    drift.add_argument('--screen', type=Path, required=True)
    drift.add_argument('--out', type=Path, required=True)
    drift.add_argument('--artifact', required=True, help='name used in --artifact NAME=PATH')
    drift.add_argument('--label', required=True)
    drift.add_argument('--budget', type=float, default=0.02)
    drift.set_defaults(func=command_drift)

    stamp = sub.add_parser('audit-identity', help='stamp a shape audit with the artifact digest')
    stamp.add_argument('--audit', type=Path, required=True)
    stamp.add_argument('--digest', required=True)
    stamp.set_defaults(func=command_audit_identity)

    spec = sub.add_parser('spec', help='write the release-certify spec')
    for flag in ('artifact', 'artifact-sha', 'base', 'base-sha', 'recipe', 'edit',
                 'arm-label', 'footprint', 'removal-suite', 'heldout-suite',
                 'retention-suite', 'config-removal', 'config-heldout',
                 'config-retention', 'pinned-argv', 'removal-wave', 'heldout-wave',
                 'retention-wave', 'retention-base-wave', 'audit-removal',
                 'audit-heldout', 'drift', 'fidelity', 'revision'):
        spec.add_argument(f'--{flag}', required=True)
    for flag in ('artifact-bytes', 'base-bytes', 'edited-tensors', 'cap-removal',
                 'cap-heldout', 'cap-retention'):
        spec.add_argument(f'--{flag}', required=True, type=int)
    spec.add_argument('--scale', required=True, type=float)
    spec.add_argument('--out', type=Path, required=True)
    spec.set_defaults(func=command_spec)

    args = parser.parse_args(argv)
    if args.self_check:
        return self_check()
    if not getattr(args, 'func', None):
        parser.error('pick a subcommand (fidelity, drift, audit-identity, spec) or --self-check')
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
