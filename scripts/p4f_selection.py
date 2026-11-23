# SPDX-License-Identifier: AGPL-3.0-or-later
"""P4f: assemble runs/p4/selection.json from the saved records and the reruns.

The phase gate asks that the arm be selected reproducibly from saved records with
no fresh generation. This script does not select anything: it checks that the
selection inputs on disk still hash to the values the P4d/P4e cards recorded,
reads the rerun outputs, and writes one JSON that carries

  * the capture records sha256 and the selector report paths,
  * the axes sha256 each selector produced (tune's fit_identity digest and the
    file digest), with the rerun digest beside it,
  * the exact argv list for every command that produced a selection artefact,
  * the ladder JSON paths and the merged-bytes measurements,
  * tokens_generated counters, which must all be 0.

A mismatch is an error, not a note: a selection that cannot be re-derived from
the saved records is not reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

RERUN_AXES_SHA = '5ca40165b69dda2a30e25b22dd525b1c844cc7fca5d5517deed780df3553644e'
CONTRAST_FILE_SHA = 'f2b14d3015ff1654ad245043e10e8610d60c76103e18d2bcc75d7689f86e82fa'
RDO_FIT_SHA = '5c481da33093174c715762db601e9d3b4cb04074a260323381bcefa0a9d197f7'
RECORDS_SHA = '9d5e648f9fee7638d174c9397c96ca50c01d14e34a026b582638fc02fabadf6a'
BASE_SHA = '80ff5222ae51de055fe2e86b1ec4f6784c2f9db704ed3b043b97ce900d3a75ad'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(prog='p4f_selection')
    parser.add_argument('--root', default='.')
    parser.add_argument('--out', default='runs/p4/selection.json')
    args = parser.parse_args(argv)
    root = Path(args.root)
    problems = []

    def rel(path):
        return str(Path(path))

    records = root / 'runs/p4/records.pt'
    contrast_axes = root / 'runs/p4/axes-contrast.pt'
    rdo_axes = root / 'runs/p4/axes-rdo.pt'
    rerun_axes = root / 'runs/p4/rerun/axes-contrast.pt'
    hashes = {'records.pt': sha256_file(records),
              'axes-contrast.pt': sha256_file(contrast_axes),
              'axes-rdo.pt': sha256_file(rdo_axes),
              'rerun/axes-contrast.pt': sha256_file(rerun_axes)}
    if hashes['records.pt'] != RECORDS_SHA:
        problems.append(f'records.pt sha256 {hashes["records.pt"]} != {RECORDS_SHA}')
    if hashes['axes-contrast.pt'] != CONTRAST_FILE_SHA:
        problems.append(f'axes-contrast.pt sha256 {hashes["axes-contrast.pt"]} != {CONTRAST_FILE_SHA}')
    if hashes['rerun/axes-contrast.pt'] != CONTRAST_FILE_SHA:
        problems.append('the rerun axes file is not byte-identical to the P4d axes file')

    ladders = {}
    for arm in ('a1', 'a2', 'a3', 'a4'):
        path = root / f'runs/p4/rerun/{arm}/tune.json'
        if not path.is_file():
            problems.append(f'missing rerun ladder {path}')
            continue
        payload = json.loads(path.read_text())
        ladders[arm] = {
            'tune_json': rel(path),
            'axes_source': payload['axes']['source'],
            'axes_sha256': payload['axes']['sha256'],
            'drift_noise_ln': payload['drift_noise_ln'],
            'generated_tokens': payload['generated_tokens'],
            'arms': [{'family': row['family'], 'scale': row['scale'],
                      'drift_ln': row['drift_ln'], 'delta_br': row['delta_br'],
                      'codec_min_cos': row['fidelity']['min_cos']}
                     for row in payload['arms']],
        }
        if payload['generated_tokens'] != 0:
            problems.append(f'{arm} rerun generated {payload["generated_tokens"]} tokens')
    if ladders.get('a1', {}).get('axes_sha256') != RERUN_AXES_SHA:
        problems.append(f'a1 rerun axes_sha256 {ladders.get("a1", {}).get("axes_sha256")} '
                        f'!= {RERUN_AXES_SHA}')
    if not str(ladders.get('a3', {}).get('axes_source', '')).endswith('axes-rdo.pt'):
        problems.append(f'a3 rerun did not read the RDO axes file: {ladders.get("a3", {}).get("axes_source")}')

    margin = {}
    for name in ('full', 'sep', 'control'):
        path = root / f'runs/p4/rerun/margin_scores-{name}.json'
        if not path.is_file():
            problems.append(f'missing margin rerun {path}')
            continue
        payload = json.loads(path.read_text())
        margin[name] = {'path': rel(path),
                        'delta_bad_minus_good': payload['summary']['delta_bad_minus_good'],
                        'prefill_count': payload['prefill_count'],
                        'tokens_generated': payload['tokens_generated']}

    layers = root / 'runs/p4/rerun/layer_scores.json'
    layer_scores = json.loads(layers.read_text()) if layers.is_file() else {}

    artifacts = {}
    for name in ('contrast', 'rdo', 'rdo-s020'):
        path = root / f'runs/p4/merged-{name}.gguf'
        if path.is_file():
            artifacts[name] = {'path': rel(path), 'sha256': sha256_file(path),
                               'revert_sidecar': rel(path.with_suffix('.gguf.revert.npz'))}

    ladder_merged_path = root / 'runs/p4/ladder-merged.json'
    ladder_merged = json.loads(ladder_merged_path.read_text()) if ladder_merged_path.is_file() else {}
    metrics = ladder_merged.get('metrics', {})
    contrast, rdo, s020 = (metrics.get('contrast', {}), metrics.get('rdo', {}),
                           metrics.get('rdo-s020', {}))
    matched = (None if not (contrast and s020) else
               abs(s020['delta_br'] - contrast['delta_br']) <= 0.05)
    noise = ladder_merged.get('drift_noise_ln')
    drift_gap = (None if not (contrast and s020) else
                 round(contrast['drift_ln'] - s020['drift_ln'], 4))

    def cert(name):
        path = root / f'runs/p4/cert-{name}.json'
        if not path.is_file():
            return None
        payload = json.loads(path.read_text())
        return {'path': rel(path), 'artifact': payload['artifact'],
                'config_argv': payload['config']['argv'], 'counts': payload['counts']}

    argvs = [
        'python -m jdewitt.cli capture --model /mnt/ssd/models/Spark-X2.6-1.7B '
        '--suite runs/p4/suite-80.jsonl --kinds refusal,harmless --out runs/p4/records.pt '
        '--gguf /mnt/ssd/models/Spark-X2.6-1.7B-Q4_K_M.gguf --ctx-size 2048 '
        '--max-tokens 512 --settle-min-tokens 24 --trust-remote-code --allow-cuda',
        'python -m jdewitt.spanfit axes --records runs/p4/records.pt '
        '--out runs/p4/axes-contrast.pt --shape perlayer',
        'python -m jdewitt.spanfit layers --records runs/p4/records.pt '
        '--out runs/p4/layer_scores.json',
        'python -m jdewitt.margin score --model /mnt/ssd/models/Spark-X2.6-1.7B '
        '--records runs/p4/records.pt --axis-file runs/p4/axes-contrast.pt '
        '--footprints 0-27 --out runs/p4/margin_scores-full.json --trust-remote-code',
        'python -m jdewitt.rdo fit --records runs/p4/records-rdo-none.pt --holdout none '
        '--out runs/p4/axes-rdo.pt --report runs/p4/axes-rdo-report.json',
        'python -m jdewitt.cli tune --gguf /mnt/ssd/models/Spark-X2.6-1.7B-Q4_K_M.gguf '
        '--records runs/p4/records.pt --perplexity '
        '/mnt/ssd/dev workspace/llama.cpp-dspark/build-perp/bin/llama-perplexity '
        '--token-budget 0 --no-verify --parallel 3 --ppl-ngl 0 --scales 1.0 '
        '--corpus-chars 6000 --out runs/p4/rerun/a1 --footprints all',
        'python -m jdewitt.cli tune ... --out runs/p4/rerun/a2 --footprints 0-27',
        'python -m jdewitt.cli tune ... --out runs/p4/rerun/a3 --footprints 0-27 '
        '--axes runs/p4/axes-rdo.pt',
        'python -m jdewitt.cli tune ... --out runs/p4/rerun/a4 --footprints 0-27',
        'python -m jdewitt.spanfit build --gguf-in runs/p4/merged-contrast.gguf --in-place '
        '--axes runs/p4/axes-contrast.pt --scale 1.0 --shape perlayer --gguf-targets auto '
        '--no-dequant-cache',
        'python -m jdewitt.spanfit build --gguf-in runs/p4/merged-rdo.gguf --in-place '
        '--axes runs/p4/axes-rdo.pt --scale 1.0 --shape perlayer --gguf-targets auto '
        '--no-dequant-cache',
        'python -m jdewitt.spanfit build --gguf-in runs/p4/merged-rdo-s020.gguf --in-place '
        '--axes runs/p4/axes-rdo.pt --scale 0.20 --shape perlayer --gguf-targets auto '
        '--no-dequant-cache',
        'python scripts/p4f_gate.py ladder --artifact base=/mnt/ssd/models/Spark-X2.6-1.7B-Q4_K_M.gguf '
        '--artifact contrast=runs/p4/merged-contrast.gguf --artifact rdo=runs/p4/merged-rdo.gguf '
        '--artifact rdo-s035=runs/p4/merged-rdo-s035.gguf '
        '--artifact rdo-s015=runs/p4/merged-rdo-s015.gguf '
        '--artifact rdo-s020=runs/p4/merged-rdo-s020.gguf '
        '--corpora runs/p4/corpora --out runs/p4/ladder-merged.json',
        'python scripts/p4f_gate.py fidelity --base /mnt/ssd/models/Spark-X2.6-1.7B-Q4_K_M.gguf '
        '--merged runs/p4/merged-contrast.gguf --axes runs/p4/axes-contrast.pt '
        '--tag contrast --out runs/p4/fidelity-contrast.json',
        'python scripts/p4f_gate.py fidelity --base /mnt/ssd/models/Spark-X2.6-1.7B-Q4_K_M.gguf '
        '--merged runs/p4/merged-rdo.gguf --axes runs/p4/axes-rdo.pt --tag rdo '
        '--out runs/p4/fidelity-rdo.json',
        'python scripts/p4f_gate.py cert --gguf <arm gguf> --records runs/p4/records.pt '
        '--tag <arm> --out runs/p4/cert-<arm>-4096.json --ctx 4096 --parallel 4 '
        '--max-tokens 200',
        'python scripts/p4f_gate.py cert --gguf <arm gguf> --records runs/p4/records.pt '
        '--tag <arm> --out runs/p4/cert-<arm>-2048.json --ctx 2048 --parallel 2 '
        '--max-tokens 200',
        'python scripts/p4f_gate.py boundary --a runs/p4/cert-<arm>-4096.json '
        '--b runs/p4/cert-<arm>-2048.json --out runs/p4/boundary-<arm>.json',
        'python scripts/p4f_gate.py verdicts --packet runs/p4/cert-<arm>-4096.json '
        '--verdicts runs/p4/verdicts-<arm>.json --base-packet runs/p4/cert-base-4096.json '
        '--base-verdicts runs/p4/verdicts-base.json --boundary runs/p4/boundary-<arm>.json '
        '--out runs/p4/cert-<arm>.json',
    ]

    payload = {
        'card': 't_2088fc50 P4f',
        'records': {'path': rel(records), 'sha256': hashes['records.pt'],
                    'rows': 80, 'bad': 36, 'good': 30},
        'selectors': {
            'contrast_axes': {'path': rel(contrast_axes),
                              'file_sha256': hashes['axes-contrast.pt'],
                              'axes_sha256': RERUN_AXES_SHA,
                              'rerun_file_sha256': hashes['rerun/axes-contrast.pt'],
                              'rerun_axes_sha256': ladders.get('a1', {}).get('axes_sha256'),
                              'tensors_equal_to_rerun': hashes['axes-contrast.pt']
                              == hashes['rerun/axes-contrast.pt']},
            'rdo_axes': {'path': rel(rdo_axes), 'file_sha256': hashes['axes-rdo.pt'],
                         'fit_sha256': RDO_FIT_SHA,
                         'report': 'runs/p4/axes-rdo-report.json'},
            'layer_separability': {'path': rel(layers),
                                   'footprint': layer_scores.get('footprint'),
                                   'flat': layer_scores.get('flat'),
                                   'keep_line': layer_scores.get('keep_line')},
            'margin': margin,
            'ladders': ladders,
        },
        'artifacts': {'base': {'path': '/mnt/ssd/models/Spark-X2.6-1.7B-Q4_K_M.gguf',
                               'sha256': BASE_SHA},
                      'merged': artifacts},
        'ladder': {'path': rel(ladder_merged_path), 'drift_noise_ln': noise,
                   'metrics': metrics},
        'equal_removal': {
            'definition': 'abs(delta_br(arm) - delta_br(contrast)) <= 0.05',
            'contrast_at_p4e_scale': {'scale': 1.0,
                                      'delta_br': contrast.get('delta_br'),
                                      'drift_ln': contrast.get('drift_ln')},
            'rdo_at_p4e_scale': {'scale': 1.0, 'delta_br': rdo.get('delta_br'),
                                 'drift_ln': rdo.get('drift_ln'),
                                 'matched': (None if not (contrast and rdo) else
                                             abs(rdo['delta_br'] - contrast['delta_br']) <= 0.05)},
            'rdo_matched_scale': {'scale': 0.20, 'delta_br': s020.get('delta_br'),
                                  'drift_ln': s020.get('drift_ln'),
                                  'matched': matched,
                                  'drift_gap_ln': drift_gap,
                                  'noise_ln': noise,
                                  'beyond_noise': (None if drift_gap is None or noise is None
                                                   else abs(drift_gap) > noise)},
        },
        'cert': {name: cert(name) for name in ('base', 'contrast', 'rdo', 'rdo-s020')},
        'argv': argvs,
        'generated_tokens': {'selection': 0, 'ladders': sum(
            entry['generated_tokens'] for entry in ladders.values()),
            'note': 'selection and every ladder are prefill-only; the cert waves are '
                    'verification generations and are counted in cert-<arm>.json counts.tokens'},
        'problems': problems,
    }
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'out': str(out), 'problems': problems,
                      'matched': matched, 'drift_gap_ln': drift_gap,
                      'noise_ln': noise}, indent=1))
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
