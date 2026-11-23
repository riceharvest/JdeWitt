# SPDX-License-Identifier: AGPL-3.0-or-later
"""Derive the release's label inputs from the measured Spark 1.7B record.

Two files are written, and neither invents a number:

* ``twin-record.json`` — the merged twin's measured record in the shape
  ``adapter-card`` labels (artifact sha, per-tensor delivered cos/SNR, edited
  counts, edit fingerprint, the behaviour pair). Every field is copied from the
  p6-runtime-equivalence JSON's ``twin``/``behaviour`` blocks, which name the
  shipped twin by sha256.
* ``evaluation.json`` — the behaviour record the release's identity binds: the
  R8 pair with its GATED status, the probe-level agreement, and the pinned
  serve argv of every arm.

The measured JSON's own sha256 is recorded in both, so a reader can re-run this
script and diff.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

TWIN_KIND = 'merged-twin-measured'


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def twin_record(measured: dict, measured_sha: str) -> dict:
    twin = measured['twin']
    plan = measured['plan']
    behaviour = measured['behaviour']
    rows = twin['delivered']['tensors']
    if not rows:
        raise SystemExit('the measured twin carries no delivered rows')
    edited = int(twin['edited_tensors'])
    if len(rows) != edited:
        raise SystemExit(f'{len(rows)} delivered rows for {edited} edited tensors; refusing')
    fingerprint = {
        'family': f'{plan["shape"]}|{plan["footprint"]}',
        'shape': plan['shape'],
        'layers': [plan['layers'][0], plan['layers'][-1]],
        'layer_indices': plan['layers'],
        'scale': plan['scale'],
        'rank': plan['rank'],
        'edited_tensors': edited,
        'base_artifact_sha256': measured['base']['sha256'],
        'directions_sha256': plan['directions_sha256'],
    }
    return {
        'version': 1,
        'kind': TWIN_KIND,
        'generated_by': 'scripts/p6_release_inputs.py from the measured p6 record',
        'artifact_sha256': twin['sha256'],
        'artifact_bytes': twin['bytes'],
        'base_artifact_sha256': measured['base']['sha256'],
        'arch': measured['base']['arch'],
        'tensor_count': measured['base']['tensor_count'],
        'edited_tensors': edited,
        'untouched_tensors': int(twin['untouched_tensors']),
        'edit_fingerprint': fingerprint,
        'delivered': {'tensors': rows, 'sampled': twin['delivered']['sampled'],
                      'min_cos': twin['delivered']['min_cos'],
                      'min_snr': twin['delivered']['min_snr'],
                      'instrument': twin['delivered']['instrument']},
        'behaviour': {'adapter_curve': behaviour['adapter_curve'],
                      'merged_bytes': behaviour['merged_bytes']},
        'adapter_delta_br': measured['adapter_delta_br'],
        'merged_delta_br': measured['merged_delta_br'],
        'adapter_to_merged_ratio': measured['adapter_to_merged_ratio'],
        'ranking_valid_on_this_artifact': measured['ranking_valid_on_this_artifact'],
        'ranking_validity_ratio_floor': measured['ranking_validity_ratio_floor'],
        'merged_effect_status': measured['merged_effect_status'],
        'curve_is_not_merged_bytes': True,
        'r8_note': measured['r8_note'],
        'source_record': {'file': 'spark-1.7b-q4km.json', 'sha256': measured_sha},
        'note': 'measured twin record: delivered cos/SNR are ggufio.compare_edit on the twin '
                'against the pristine base; behaviour numbers are the measured generation-scored '
                'pair, labelled by merged_effect_status',
    }


def evaluation(measured: dict, measured_sha: str) -> dict:
    behaviour = measured['behaviour']
    serve = measured['serve']
    arms = serve['arms']
    merged_argv = arms['merged']['argv']
    if not merged_argv or not all(isinstance(token, str) for token in merged_argv):
        raise SystemExit('the merged arm records no pinned serve argv')
    return {
        'version': 1,
        'kind': 'p6-evaluation',
        'generated_by': 'scripts/p6_release_inputs.py from the measured p6 record',
        'task': measured.get('task'),
        'status': 'GATED',
        'claim': measured.get('claim'),
        'claim_limit': measured.get('claim_limit'),
        'source_record': {'file': 'spark-1.7b-q4km.json', 'sha256': measured_sha},
        'base': measured['base'],
        'artifacts': {
            'adapter': {'sha256': measured['adapter']['sha256'], 'bytes': measured['adapter']['bytes'],
                        'rank': measured['adapter']['rank'], 'scale': measured['adapter']['scale']},
            'merged_twin': {'sha256': measured['twin']['sha256'], 'bytes': measured['twin']['bytes'],
                            'edited_tensors': measured['twin']['edited_tensors'],
                            'untouched_tensors': measured['twin']['untouched_tensors']},
        },
        'behaviour': {
            'measurement_class': 'behaviour-claim',
            'basis': 'generation-scored on the B70; probe-level',
            'status': 'GATED',
            'base_br': behaviour['base_br'],
            'adapter_delta_br': measured['adapter_delta_br'],
            'merged_delta_br': measured['merged_delta_br'],
            'adapter_to_merged_ratio': measured['adapter_to_merged_ratio'],
            'ranking_valid_on_this_artifact': measured['ranking_valid_on_this_artifact'],
            'ranking_validity_ratio_floor': measured['ranking_validity_ratio_floor'],
            'curve_is_not_merged_bytes': measured['curve_is_not_merged_bytes'],
            'delta_br_definition': behaviour['delta_br_definition'],
            'agreement': measured['agreement'],
            'ppl_readings': behaviour['readings'],
            'r8_note': measured['r8_note'],
        },
        'serve': {
            'config': serve['config'],
            'arms': {arm: {'artifact': block.get('artifact'), 'adapter': block.get('adapter'),
                           'argv': block.get('argv')}
                     for arm, block in arms.items()},
        },
        # release-assemble reads this path for the pinned argv it copies into twin.json.
        'serve_argv': merged_argv,
        'note': 'probe-level behaviour: 4 harmful rows, one per family, greedy 200 tokens. No '
                'certification wave and no vLLM/SGLang claim; the numbers are GATED.',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or '').splitlines()[0])
    parser.add_argument('--measured', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    measured_sha = file_sha256(args.measured)
    measured = json.loads(args.measured.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    record = twin_record(measured, measured_sha)
    evaluation_document = evaluation(measured, measured_sha)
    (args.out / 'twin-record.json').write_text(json.dumps(record, indent=2) + '\n')
    (args.out / 'evaluation.json').write_text(json.dumps(evaluation_document, indent=2) + '\n')
    summary = {
        'twin_record': str(args.out / 'twin-record.json'),
        'evaluation': str(args.out / 'evaluation.json'),
        'artifact_sha256': record['artifact_sha256'],
        'rows': len(record['delivered']['tensors']),
        'edited': record['edited_tensors'], 'untouched': record['untouched_tensors'],
        'min_cos': record['delivered']['min_cos'], 'min_snr': record['delivered']['min_snr'],
        'merged_effect_status': record['merged_effect_status'],
        'serve_argv_tokens': len(evaluation_document['serve_argv']),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
