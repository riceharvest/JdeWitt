# SPDX-License-Identifier: AGPL-3.0-or-later
"""Re-grade a saved refusal wave's stored labels from its own answer text.

A wave writes ``refusal_check`` / ``refusal_check_wide`` at measure time. Those
labels are evidence of what that build graded, and ``release-certify`` refuses to
certify a removal count when they disagree with the shipped ``RefusalChecker`` --
a count and a digest must describe the same grader (``CERTIFY_ONE_INVENTORY``).

One known way they disagree: ``jdewitt/evaluation.py`` replaces the shipped label
with a ``degenerate-<reason>`` refusal override when ``scoring.degenerate_output``
fires, and only for the narrow inventory. The override is a rule about forced
continuation, not an inventory reading; the wider inventory is graded from text
either way, and the certifier grades both from text. This tool closes that gap
*without touching a single answer*: it recomputes both inventories from the stored
answer text with the shipped checker, rewrites only the two label fields on rows
whose stored label differs, keeps the original file beside the new one, and leaves
a per-row audit of what changed and why.

    python verification/p7-q3-4b/regrade_wave.py verification/p7-q3-4b/waves/removal-arm

Refuses to run twice over the same wave: the pre-regrade copy is the original
evidence, and a second pass would make it a copy of a copy.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from jdewitt.answers import atomic_text  # noqa: E402
from jdewitt.refusal_check import RefusalChecker  # noqa: E402

LABEL_KEYS = ('refusal_check', 'refusal_check_wide')
RESULTS = ('refusal_detected', 'no_explicit_refusal', 'uncertain')


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stored_result(row: dict, key: str):
    value = row.get(key)
    if isinstance(value, dict) and value.get('result') in RESULTS:
        return value['result']
    return None


def regrade(directory: Path) -> dict:
    directory = directory.expanduser().resolve()
    predictions = directory / 'predictions.jsonl'
    wave_path = directory / 'wave.json'
    backup = directory / 'predictions.pre-regrade.jsonl'
    audit_path = directory / 'regrade-audit.json'
    if not predictions.is_file():
        raise SystemExit(f'{directory} has no predictions.jsonl')
    if backup.exists() or audit_path.exists():
        raise SystemExit(f'{directory} has already been regraded '
                         f'({backup.name}/{audit_path.name} exists); the pre-regrade '
                         'copy is the original evidence')
    original_sha = _sha(predictions)
    rows = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]

    checker = RefusalChecker()
    refusals_before = sum(1 for row in rows
                          if _stored_result(row, 'refusal_check') == 'refusal_detected')
    original_rows = copy.deepcopy(rows)
    changed = []
    for row in rows:
        if row.get('status') != 'ok' or row.get('kind') != 'refusal':
            continue
        language = row.get('language') or 'en'
        answer = row.get('answer') or ''
        for key, wide in (('refusal_check', False), ('refusal_check_wide', True)):
            fresh = checker.check(answer, language, wide=wide)
            before = _stored_result(row, key)
            if before == fresh['result']:
                continue
            changed.append({'id': row.get('id'), 'field': key,
                            'before': before, 'after': fresh['result'],
                            'before_method': (row.get(key) or {}).get('method'),
                            'after_method': fresh['method'],
                            'degenerate': row.get('degenerate'),
                            'answer_sha256': hashlib.sha256(answer.encode()).hexdigest()})
            row[key] = fresh

    atomic_text(backup, predictions.read_text())
    atomic_text(predictions, '\n'.join(json.dumps(row) for row in rows) + '\n')

    # Self-check: nothing but the two label fields may differ from the original
    # rows. A re-serialisation that quietly normalised a field would be an edit to
    # evidence, and this refuses rather than shipping one.
    written = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
    if len(written) != len(original_rows):
        raise SystemExit(f'{directory}: wrote {len(written)} rows for {len(original_rows)} read')
    for before_row, after_row in zip(original_rows, written):
        for key in set(before_row) | set(after_row):
            if key not in LABEL_KEYS and before_row.get(key) != after_row.get(key):
                raise SystemExit(f'{directory}: field {key!r} on {before_row.get("id")!r} '
                                 'changed outside the label fields; refusing to leave it')

    report = {
        'schema_version': 1,
        'kind': 'refusal-regrade',
        'directory': str(directory),
        'regraded_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'rows': len(rows),
        'rows_graded': sum(1 for row in rows
                           if row.get('status') == 'ok' and row.get('kind') == 'refusal'),
        'changed_fields': len(changed),
        'changed_rows': sorted({item['id'] for item in changed}),
        'changes': changed,
        'predictions_sha256_before': original_sha,
        'predictions_sha256_after': _sha(predictions),
        'answer_text_touched': False,
        'refusals_detected_before': refusals_before,
        'refusals_detected_after': sum(
            1 for row in rows if _stored_result(row, 'refusal_check') == 'refusal_detected'),
        'note': ('Both inventories recomputed from each stored answer with the shipped '
                 'RefusalChecker. Answers, statuses, tokens and every other field are '
                 'byte-identical; only the two label fields on the rows listed above '
                 'changed. A wave-time degenerate-override label is recorded in the row\'s '
                 '`degenerate` field and stays there.'),
    }
    if wave_path.is_file():
        wave = json.loads(wave_path.read_text())
        reported = wave.get('refusals_detected')
        wave['refusals_detected_at_measure_time'] = reported
        wave['refusals_detected'] = report['refusals_detected_after']
        wave['regrade'] = {'at': report['regraded_at'],
                           'script': 'verification/p7-q3-4b/regrade_wave.py',
                           'changed_fields': len(changed),
                           'changed_rows': report['changed_rows'],
                           'audit': str(audit_path)}
        atomic_text(wave_path, json.dumps(wave, indent=1, sort_keys=True) + '\n')
        report['wave_refusals_detected'] = {'at_measure_time': reported,
                                            'after_regrade': report['refusals_detected_after']}
    atomic_text(audit_path, json.dumps(report, indent=1) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('waves', nargs='+', type=Path, metavar='WAVE_DIR')
    args = parser.parse_args(argv)
    for directory in args.waves:
        report = regrade(directory)
        print(json.dumps({key: report[key] for key in
                          ('directory', 'rows', 'rows_graded', 'changed_fields',
                           'changed_rows', 'predictions_sha256_before',
                           'predictions_sha256_after')}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
