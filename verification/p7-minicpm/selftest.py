# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end self-test of the P7 MiniCPM cert plumbing, with no model and no GPU.

The wave card cannot be run on a CPU card, but the *contract* between the wave
outputs, the spec and `release-certify` can be: this script builds a synthetic
measurement set from the real committed suites and drives the real commands over
it. It proves the shapes, the config slots, the both-inventory grading, the R8
audit binding and the R9.4 pair — everything that is decided by code rather than
by a served model.

It never writes inside the repo: every fixture goes under --out (a scratch dir),
the answers are obviously synthetic, and the packet it produces is labelled
`synthetic-selftest` so it can never be mistaken for a certified row.

    python verification/p7-minicpm/selftest.py --out /mnt/ssd/p7-minicpm-selftest
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('P7_REPO') or HERE.parents[1]).resolve()
sys.path.insert(0, str(REPO))

from jdewitt.refusal_check import RefusalChecker  # noqa: E402

REMOVAL_SUITE = REPO / 'verification/screens/advbench-40.jsonl'
HELDOUT_SUITE = REPO / 'verification/p7-minicpm/heldout-24f.jsonl'
RETENTION_SUITE = REPO / 'verification/capability/capability-run-en.jsonl'
SOLVER = {'tool': REPO / 'verification/p7-minicpm/cert_inputs.py'}

REFUSAL = ('I cannot help with that. Building or using this would cause real harm, '
           'and I will not provide instructions for it.')
COMPLY = ('Sure. Here is a general overview of the topic, written in plain prose, with '
          'enough detail to read as an answer and no procedural content.')


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def graded_row(task: dict, answer: str, passed: bool | None = None) -> dict:
    checker = RefusalChecker()
    language = task.get('language') or 'en'
    row = {'id': task['id'], 'kind': task['kind'], 'language': language,
           'prompt': task['prompt'], 'answer': answer, 'status': 'ok',
           'completion_tokens': 40, 'scoring_mode': 'refusal-check'}
    if task['kind'] == 'refusal':
        row['refusal_check'] = checker.check(answer, language, wide=False)
        row['refusal_check_wide'] = checker.check(answer, language, wide=True)
    if task.get('family_id'):
        row['family_id'] = task['family_id']
    if passed is not None:
        row['passed'] = passed
    return row


def write_wave(directory: Path, suite: Path, config: str, cap: int,
               predict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    tasks = rows(suite)
    out = [predict(task) for task in tasks]
    (directory / 'predictions.jsonl').write_text(
        ''.join(json.dumps(row, sort_keys=True) + '\n' for row in out))
    summary = {'label': directory.name, 'suite': str(suite), 'rows': len(out),
               'kinds': sorted({row['kind'] for row in out}), 'measured': True,
               'invalid': 0, 'serve_config': config, 'max_tokens': cap,
               'refusals_detected': sum(1 for row in out
                                        if (row.get('refusal_check') or {}).get('result') == 'refusal_detected'),
               'capability_passed': sum(1 for row in out if row.get('passed') is True)}
    (directory / 'wave.json').write_text(json.dumps(summary, indent=1, sort_keys=True) + '\n')
    return directory


def run(*argv, expect_ok: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run([str(a) for a in argv], cwd=REPO, capture_output=True, text=True)
    if expect_ok and result.returncode != 0:
        raise SystemExit(f'command failed ({result.returncode}): {" ".join(map(str, argv))}\n'
                         f'{result.stdout[-2000:]}\n{result.stderr[-2000:]}')
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.expanduser().resolve()
    if REPO in out.parents or out == REPO:
        raise SystemExit('--out must be outside the repo: this fixture is a test, not evidence')
    out.mkdir(parents=True, exist_ok=True)

    merged = out / 'synthetic-merged.gguf'
    merged.write_bytes(b'synthetic-selftest-artifact\n')
    import hashlib
    merged_sha = hashlib.sha256(merged.read_bytes()).hexdigest()
    merged_bytes = merged.stat().st_size
    base = out / 'synthetic-base.gguf'
    base.write_bytes(b'synthetic-base\n')
    base_sha = hashlib.sha256(base.read_bytes()).hexdigest()

    cfg_4096 = f'-m {merged} -c 4096 --parallel 4 --device none --jinja --no-webui --host 127.0.0.1 --port 8134'
    cfg_2048 = f'-m {merged} -c 2048 --parallel 2 --device none --jinja --no-webui --host 127.0.0.1 --port 8134'
    cfg_base = f'-m {base} -c 4096 --parallel 4 --device none --jinja --no-webui --host 127.0.0.1 --port 8134'

    # removal: base refuses everything; the arm complies on 37/40; the 2048/2
    # run flips three of them back to refusal -> boundary, robustly removed = 34
    flip = {'advbench-train-000007', 'advbench-train-000019', 'advbench-train-000031'}
    write_wave(out / 'waves/base-removal', REMOVAL_SUITE, cfg_base, 1024,
               lambda t: graded_row(t, REFUSAL))
    w_removal = write_wave(out / 'waves/removal-4096', REMOVAL_SUITE, cfg_4096, 1024,
                           lambda t: graded_row(t, COMPLY))
    w_removal_low = write_wave(out / 'waves/removal-2048', REMOVAL_SUITE, cfg_2048, 1024,
                               lambda t: graded_row(t, REFUSAL if t['id'] in flip else COMPLY))
    w_heldout = write_wave(out / 'waves/heldout', HELDOUT_SUITE, cfg_4096, 1024,
                           lambda t: graded_row(t, COMPLY))
    # retention: the arm loses 5 maths and 2 code rows; answers stay long enough
    def retention(task: dict) -> dict:
        index = int(task['id'].split('-')[-1]) if task['id'].split('-')[-1].isdigit() else 0
        passed = not (task['kind'] == 'math' and index % 12 == 0) and \
                 not (task['kind'] == 'code' and index % 31 == 0)
        answer = ('The answer is 42. ' if task['kind'] == 'math' else 'def solve():\n    return 42\n')
        answer += 'work ' * (8 if task['kind'] == 'math' else 4)
        if not passed:
            answer = 'I am not sure.' + ' ' * 2
        return graded_row(task, answer, passed=passed)
    w_retention = write_wave(out / 'waves/retention', RETENTION_SUITE, cfg_4096, 1024, retention)
    w_base_retention = write_wave(out / 'waves/base-retention', RETENTION_SUITE, cfg_base, 1024,
                                  lambda t: graded_row(t, ('The answer is 42. ' if t['kind'] == 'math'
                                                           else 'def solve():\n    return 42\n')
                                                       + 'work ' * (10 if t['kind'] == 'math' else 5),
                                                       passed=True))

    # fidelity: a synthetic compare_edit report over two tensors
    fidelity_raw = out / 'fidelity-raw.json'
    fidelity_raw.write_text(json.dumps({'per_tensor': [
        {'name': 'blk.0.attn_output.weight', 'qtype': 'Q4_K_M', 'shape': [1024, 1024],
         'cos': 0.91, 'norm_ratio': 1.01, 'snr': 2.1, 'delta_norm': 0.02, 'noise_norm': 0.01},
        {'name': 'blk.0.ffn_down.weight', 'qtype': 'Q4_K_M', 'shape': [3584, 1024],
         'cos': 0.73, 'norm_ratio': 1.02, 'snr': 1.1, 'delta_norm': 0.03, 'noise_norm': 0.02}],
        'summary': {'min_cos': 0.73, 'mean_cos': 0.82, 'min_snr': 1.1,
                    'codec_event': True, 'min_edit_cos_floor': 0.9}}))
    fidelity = out / 'fidelity.json'
    run(sys.executable, SOLVER['tool'], 'fidelity', '--raw', fidelity_raw, '--out', fidelity,
        '--artifact-sha', merged_sha, '--artifact-bytes', merged_bytes,
        '--label', 'synthetic-selftest', '--scale', 1.0)

    # drift: synthetic ppl screen, control delta inside budget
    screen = out / 'ppl-screen.json'
    screen.write_text(json.dumps({'artifacts': {
        'base': {'control': {'ln_ppl': 0.20}, 'refusal': {'ln_ppl': 0.80}, 'delivery': {'ln_ppl': 1.10}},
        'synthetic-selftest': {'control': {'ln_ppl': 0.205}, 'delta_control': 0.005,
                               'delta_BR': -0.31, 'BR': -0.5,
                               'refusal': {'ln_ppl': 1.11}, 'delivery': {'ln_ppl': 1.61}}}}))
    drift = out / 'drift.json'
    run(sys.executable, SOLVER['tool'], 'drift', '--screen', screen, '--out', drift,
        '--artifact', 'synthetic-selftest', '--label', 'synthetic-selftest', '--budget', 0.02)

    # shape audits off the synthetic waves
    audit_removal = out / 'shape-audit-removal.json'
    audit_heldout = out / 'shape-audit-heldout.json'
    run(sys.executable, REPO / 'scripts/refusal_shape_audit.py',
        f'removal={w_removal / "predictions.jsonl"}', '--out', audit_removal)
    run(sys.executable, REPO / 'scripts/refusal_shape_audit.py',
        f'held_out={w_heldout / "predictions.jsonl"}', '--out', audit_heldout)

    spec = out / 'spec.json'
    run(sys.executable, SOLVER['tool'], 'spec', '--out', spec,
        '--artifact', merged, '--artifact-sha', merged_sha, '--artifact-bytes', merged_bytes,
        '--base', base, '--base-sha', base_sha, '--base-bytes', base.stat().st_size,
        '--recipe', 'python -m jdewitt.spanfit build --gguf-targets auto --scale 1.0',
        '--edit', 'synthetic selftest axes', '--scale', 1.0, '--arm-label', 'synthetic-selftest',
        '--edited-tensors', 48, '--footprint', 'blk.0-23 x {attn_output.weight, ffn_down.weight}',
        '--removal-suite', REMOVAL_SUITE, '--heldout-suite', HELDOUT_SUITE,
        '--retention-suite', RETENTION_SUITE,
        '--config-removal', cfg_4096, '--config-heldout', cfg_4096,
        '--config-retention', cfg_4096, '--pinned-argv', cfg_4096,
        '--removal-wave', w_removal, '--heldout-wave', w_heldout,
        '--retention-wave', w_retention, '--retention-base-wave', w_base_retention,
        '--audit-removal', audit_removal, '--audit-heldout', audit_heldout,
        '--drift', drift, '--fidelity', fidelity,
        '--revision', '286817f1132b998913a129a221c89b4402b4ab80',
        '--cap-removal', 1024, '--cap-heldout', 1024, '--cap-retention', 1024)
    digest = json.loads(spec.read_text())['artifact_identity_digest']
    for audit in (audit_removal, audit_heldout):
        run(sys.executable, SOLVER['tool'], 'audit-identity', '--audit', audit, '--digest', digest)

    packet_path = out / 'packet.json'
    run('uv', 'run', 'jdewitt', 'release-certify', '--spec', spec, '--out', packet_path)
    packet = json.loads(packet_path.read_text())

    comparison = out / 'comparison.json'
    run(sys.executable, REPO / 'scripts/cert_config_pair.py',
        '--a', w_removal, '--b', w_removal_low,
        '--label-a', 'ctx 4096 / parallel 4 / cap 1024',
        '--label-b', 'ctx 2048 / parallel 2 / cap 1024', '--out', comparison)
    pair = json.loads(comparison.read_text())

    # A certifier that accepts everything proves nothing: an audit bound to another
    # artifact must be refused (R8), and it must write no packet.
    tampered = json.loads(audit_removal.read_text())
    tampered['artifact_identity_digest'] = '0' * 64
    tampered_path = out / 'shape-audit-tampered.json'
    tampered_path.write_text(json.dumps(tampered))
    tampered_spec = json.loads(spec.read_text())
    tampered_spec['inputs']['shape_audit'][0]['path'] = str(tampered_path)
    tampered_spec['inputs']['shape_audit'][0]['artifact_identity_digest'] = '0' * 64
    tampered_spec_path = out / 'spec-tampered.json'
    tampered_spec_path.write_text(json.dumps(tampered_spec, indent=1, sort_keys=True))
    refused_packet = out / 'packet-tampered.json'
    refusal = run('uv', 'run', 'jdewitt', 'release-certify', '--spec', tampered_spec_path,
                  '--out', refused_packet, expect_ok=False)

    checks = [
        ('packet emitted', packet_path.is_file()),
        ('packet is one arm, both axes',
         packet.get('measured') is True
         and packet['removal']['harmful40']['inventories']['wide']['n_m'] == '40/40'
         and packet['removal']['held_out_120']['inventories']['wide']['n_m'] == '120/120'),
        ('both inventories recorded with digests',
         set(packet['removal']['harmful40']['inventories']) == {'shipped', 'wide'}),
        ('held-out families counted', packet.get('held_out_families') == 24),
        ('retention paired both kinds',
         packet['retention']['math']['strict']['paired_rows'] > 0
         and packet['retention']['code']['strict']['paired_rows'] > 0),
        ('median answer length present',
         packet['retention']['math']['median_answer_chars'] > 0),
        ('pinned cert match all axes',
         all(packet['pinned_cert_match'].get(axis) is True
             for axis in ('removal', 'retention', 'held_out'))),
        ('drift inside budget', packet['drift']['inside_budget'] is True),
        ('fidelity is the arm measurement', packet['realized_fidelity']['arm_measurement'] is True),
        ('shape audit bound to the artifact digest',
         packet['shape']['artifact_identity_digest'] == digest),
        ('config pair: a 40/40, b 37/40',
         pair['n_m_a'] == '40/40' and pair['n_m_b'] == '37/40'),
        ('config pair: 3 boundary rows', len(pair['boundary']) == 3),
        ('config pair: robustly removed 37', pair['robustly_removed_a'] == 37),
        ('R8: an audit of another artifact is refused', refusal.returncode != 0),
        ('R8: refusal writes no packet', not refused_packet.exists()),
        ('R8: refusal names the mismatch', 'CERTIFY_ARTIFACT_MISMATCH' in
         (refusal.stdout + refusal.stderr)),
    ]
    failed = 0
    for label, ok in checks:
        print(f'{"PASS" if ok else "FAIL"} {label}')
        failed += 0 if ok else 1
    print(json.dumps({'packet': str(packet_path), 'comparison': str(comparison),
                      'artifact_identity_digest': digest,
                      'removal_4096': pair['n_m_a'], 'removal_2048': pair['n_m_b'],
                      'boundary': pair['boundary']}, indent=1))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
