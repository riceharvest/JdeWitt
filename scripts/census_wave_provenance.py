# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prove a census wave resolves to the prompts, licence and revision it claims.

Usage:
    python scripts/census_wave_provenance.py verification/public-corpus-census/import-run/wave
    python scripts/census_wave_provenance.py <wave dir> --check

A wave is generation evidence: predictions.jsonl (one row per suite id) plus
wave.json (the serve argv and the token ledger). Neither file says where the
prompts came from. This tool joins them back to the import split and the census
inventory and writes ``provenance-check.json`` next to them, so a reader does not
have to trust the label on the directory:

  * every prediction id exists in the holdout split, and the holdout split is the
    one ``import-run/provenance.json`` recorded (task count *and* row digest, using
    the importer's own ``jdewitt.data.digest`` so the two cannot drift);
  * every holdout row carries the ``source`` and ``license`` the census declares,
    and the census entry pins a dataset and a 40-hex revision;
  * the census sha256 recorded in ``provenance.json`` still matches
    ``sources.json`` today — a licence that changed after the wave is a fact about
    the wave, not a detail to gloss over;
  * wave.json carries the serve argv and a token ledger that agrees with the
    per-row ``completion_tokens`` in predictions.jsonl.

It also sanitises the committed artifacts: absolute ``/mnt``, ``/home``,
``/Users`` and ``/root`` paths become repo-relative or ``<local>/<basename>``,
board task ids become ``<task>``, and every rewrite is listed in the check file.
Absolute paths inside model output are *not* rewritten (that would edit evidence);
they are reported and fail the run so a human can decide.

The check never says anything about harm or removal. A refusal-check result is a
screen over wording; the semantic read stays human. ``claims`` is written
explicitly so nothing downstream can read the counts as a removal claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from jdewitt.data import digest, load_suite  # noqa: E402
from jdewitt.sources import load_census  # noqa: E402

CHECK_FORMAT = 'jdewitt-census-wave-provenance/1'
EXPECTED_HOLDOUT = 'holdout'
ABSOLUTE_PATH = re.compile(r'(?<![\w<>])(/(?:mnt|home|Users|root)/[^\s"\';,)\]]*)')
TASK_ID = re.compile(r'\bt_[0-9a-f]{8}\b')
REVISION = re.compile(r'^[0-9a-f]{40}$')


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            hasher.update(block)
    return hasher.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f'{path.name}:{number} is not JSON: {error}') from error
        if not isinstance(row, dict):
            raise ValueError(f'{path.name}:{number} is not an object')
        rows.append(row)
    return rows


def sanitise_text(text: str, repo_root: Path, field: str, rewrites: list[dict]) -> str:
    """Make one string portable. Rewrites are recorded, originals are not."""

    def replace_path(match: re.Match) -> str:
        raw = match.group(1)
        candidate = Path(raw)
        try:
            relative = candidate.resolve().relative_to(repo_root.resolve()).as_posix()
        except (ValueError, OSError):
            target = '<local>/' + candidate.name
            rewrites.append({'field': field, 'kind': 'redacted-local', 'to': target})
            return target
        rewrites.append({'field': field, 'kind': 'repo-relative', 'to': relative})
        return relative

    text = ABSOLUTE_PATH.sub(replace_path, text)

    def replace_task(match: re.Match) -> str:
        rewrites.append({'field': field, 'kind': 'task-id', 'to': '<task>'})
        return '<task>'

    return TASK_ID.sub(replace_task, text)


def sanitise_value(value, field: str, repo_root: Path, rewrites: list[dict]):
    if isinstance(value, str):
        return sanitise_text(value, repo_root, field, rewrites)
    if isinstance(value, list):
        return [sanitise_value(item, f'{field}[{index}]', repo_root, rewrites)
                for index, item in enumerate(value)]
    if isinstance(value, dict):
        return {key: sanitise_value(item, f'{field}.{key}', repo_root, rewrites)
                for key, item in value.items()}
    return value


def leaks_in_predictions(rows: list[dict]) -> list[dict]:
    """Absolute paths and task ids inside saved answers. Reported, never rewritten."""
    found = []
    for row in rows:
        for key, value in row.items():
            if not isinstance(value, str):
                continue
            paths = ABSOLUTE_PATH.findall(value)
            tasks = TASK_ID.findall(value)
            if paths or tasks:
                found.append({'id': row.get('id'), 'field': key,
                              'absolute_paths': len(paths), 'task_ids': len(tasks)})
    return found


def build_check(wave_dir: Path, holdout_path: Path, provenance_path: Path,
                census_path: Path, repo_root: Path = REPO) -> dict:
    checks: list[dict] = []

    def record(check_id: str, ok: bool, detail: str) -> bool:
        checks.append({'id': check_id, 'ok': bool(ok), 'detail': detail})
        return bool(ok)

    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    holdout = load_suite(holdout_path)
    holdout_ids = {row['id'] for row in holdout}

    predictions_path = wave_dir / 'predictions.jsonl'
    wave_path = wave_dir / 'wave.json'
    predictions = read_jsonl(predictions_path) if predictions_path.is_file() else []
    wave = json.loads(wave_path.read_text(encoding='utf-8')) if wave_path.is_file() else None

    record('predictions_present', bool(predictions),
           f'{predictions_path.name}: {len(predictions)} rows')

    prediction_ids = [row.get('id') for row in predictions]
    unique = len(set(prediction_ids))
    record('prediction_ids_unique', unique == len(prediction_ids),
           f'{unique} distinct ids over {len(prediction_ids)} rows')

    missing = sorted(holdout_ids - set(prediction_ids))
    extra = sorted(set(prediction_ids) - holdout_ids)
    record('ids_resolve_to_holdout', not missing and not extra,
           f'missing from wave: {len(missing)} {missing[:5]}; not in holdout: {len(extra)} {extra[:5]}')

    record('wave_json_present', wave is not None, f'{wave_path.name} exists: {wave is not None}')
    if wave is not None:
        record('wave_rows_match', wave.get('rows') == len(predictions),
               f"wave.json rows={wave.get('rows')}, predictions={len(predictions)}")
    else:
        record('wave_rows_match', False, 'no wave.json to compare')

    split = (provenance.get('splits') or {}).get(EXPECTED_HOLDOUT) or {}
    holdout_digest = digest(holdout)
    record('holdout_matches_provenance',
           split.get('digest') == holdout_digest and split.get('tasks') == len(holdout),
           f"provenance digest={str(split.get('digest'))[:16]}..., tasks={split.get('tasks')}; "
           f"on disk digest={holdout_digest[:16]}..., tasks={len(holdout)}")

    census = load_census(census_path)
    census_matches = census['sha256'] == (provenance.get('census') or {}).get('sha256')
    record('census_digest_matches_provenance', census_matches,
           f"provenance census sha256={str((provenance.get('census') or {}).get('sha256'))[:16]}..., "
           f"on disk={census['sha256'][:16]}...")

    resolved: dict[str, dict] = {}
    unresolved: list[str] = []
    for row in holdout:
        source, licence = row.get('source'), row.get('license')
        entry = census['sources'].get(source) if isinstance(source, str) else None
        if entry is None or not isinstance(licence, str) or not licence.strip():
            unresolved.append(str(row.get('id')))
            continue
        aggregate = resolved.setdefault(source, {
            'id': source, 'dataset': entry.get('dataset'), 'revision': entry.get('revision'),
            'license': entry['license'], 'license_url': entry.get('license_url'),
            'rows': 0, 'row_license_values': set()})
        aggregate['rows'] += 1
        aggregate['row_license_values'].add(licence)

    for aggregate in resolved.values():
        aggregate['row_license_values'] = sorted(aggregate['row_license_values'])
    record('every_holdout_row_resolves', not unresolved,
           f'{len(holdout) - len(unresolved)}/{len(holdout)} rows carry a census source; '
           f'unresolved: {unresolved[:5]}')

    licence_ok = all(aggregate['row_license_values'] == [aggregate['license']]
                     and REVISION.match(str(aggregate['revision'] or ''))
                     and aggregate['dataset']
                     for aggregate in resolved.values())
    record('rows_carry_census_license_and_pinned_revision', licence_ok,
           '; '.join(f"{a['id']}->{a['dataset']}@{str(a['revision'])[:12]} {a['license']} "
                     f"rows_license={a['row_license_values']}" for a in resolved.values()) or 'no sources')

    prediction_licence_ok = True
    for row in predictions:
        if 'source' in row or 'license' in row:
            expected = resolved.get(row.get('source'))
            if expected is None or row.get('license') != expected['license']:
                prediction_licence_ok = False
                break
    record('predictions_do_not_contradict_census', prediction_licence_ok,
           'prediction rows carry no source/license, or agree with the holdout/census')

    argv = (wave or {}).get('serve_config') or ''
    record('wave_carries_argv', bool(argv.strip()) and bool(re.search(r'(^|\s)-m[ =]|--model', argv)),
           'serve_config names the served artifact' if argv else 'serve_config is empty')

    tokens = (wave or {}).get('completion_tokens_total')
    row_tokens = [row.get('completion_tokens') for row in predictions if row.get('completion_tokens') is not None]
    ledger_detail = f'wave ledger={tokens!r}'
    ledger_ok = isinstance(tokens, int) and tokens >= 0
    if ledger_ok and row_tokens:
        ledger_ok = sum(row_tokens) == tokens
        ledger_detail += f', sum of prediction rows={sum(row_tokens)}'
    record('token_ledger_agrees', ledger_ok, ledger_detail)

    invalid = sum(1 for row in predictions if row.get('status') not in (None, 'ok'))
    measured = (wave or {}).get('measured')
    record('wave_measured', measured is True and invalid < len(predictions) and bool(predictions),
           f'measured={measured!r}, invalid rows={invalid}/{len(predictions)}')

    rewrites: list[dict] = []
    sanitised_wave = sanitise_value(wave, 'wave', repo_root, rewrites) if wave is not None else None
    leaked = leaks_in_predictions(predictions)
    record('predictions_have_no_local_paths', not leaked,
           f'{len(leaked)} row/field leaks' + (f' {leaked[:3]}' if leaked else ''))

    refusal_rows = [row for row in predictions
                    if (row.get('refusal_check') or {}).get('result') == 'refusal_detected']
    no_refusal_rows = [row for row in predictions
                       if (row.get('refusal_check') or {}).get('result') == 'no_explicit_refusal']
    screen = {'rows': len(predictions),
              'refusals_detected': len(refusal_rows),
              'no_explicit_refusal': len(no_refusal_rows),
              'ungraded': len(predictions) - len(refusal_rows) - len(no_refusal_rows),
              'invalid': invalid,
              'completion_tokens_total': tokens,
              'scope': 'screen over refusal wording only',
              'removal_claimed': False}

    ok = all(check['ok'] for check in checks)
    return {
        'format': CHECK_FORMAT,
        'generated': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'ok': ok,
        'failed_checks': [check['id'] for check in checks if not check['ok']],
        'checks': checks,
        'wave': {
            'label': (wave or {}).get('label'),
            'rows': len(predictions),
            'predictions': 'predictions.jsonl',
            'predictions_sha256': file_sha256(predictions_path) if predictions_path.is_file() else None,
            'wave_json': 'wave.json',
            'wave_json_sha256': file_sha256(wave_path) if wave_path.is_file() else None,
        },
        'holdout': {'path': portable_path(holdout_path, repo_root), 'tasks': len(holdout),
                    'digest': holdout_digest,
                    'declared_digest': split.get('digest')},
        'census': {'path': portable_path(census_path, repo_root), 'sha256': census['sha256'],
                   'declared_sha256': (provenance.get('census') or {}).get('sha256'),
                   'matches': census_matches},
        'resolved_provenance': sorted(resolved.values(), key=lambda item: item['id']),
        'unresolved_rows': unresolved,
        'argv': {'serve_config': (sanitised_wave or {}).get('serve_config'),
                 'structured': (sanitised_wave or {}).get('serve_structured')},
        'screen': screen,
        'claims': {'removal': False, 'capability': False,
                   'note': 'a refusal-check pass is a screen over wording; it is not evidence that '
                           'the asked-for harm was or was not delivered, and nothing here claims a row '
                           'was removed from the model'},
        'sanitisation': {'rewrites': rewrites, 'predictions_leaks': leaked},
        'not_checked': ['that the answers are honest reads of the model',
                        'that refusing wording means refusing the request',
                        'that the model was the artifact the argv names',
                        'token budgets and model fit'],
    }


def portable_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except (ValueError, OSError):
        return '<local>/' + path.name


def comparable(report: dict) -> dict:
    """The parts of the check that can be recomputed from the inputs.

    ``generated`` is a run stamp, and ``sanitisation.rewrites`` records a
    one-time edit to wave.json that a re-run of the check cannot see again
    (the file it describes is already portable). Everything else — ids, digests,
    licences, argv, ledger, screen counts — must match exactly.
    """
    return {key: value for key, value in report.items() if key not in {'generated', 'sanitisation'}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('wave_dir', type=Path, help='directory holding predictions.jsonl and wave.json')
    parser.add_argument('--holdout', type=Path,
                        default=REPO / 'verification/public-corpus-census/import-run/holdout.jsonl')
    parser.add_argument('--provenance', type=Path,
                        default=REPO / 'verification/public-corpus-census/import-run/provenance.json')
    parser.add_argument('--census', type=Path,
                        default=REPO / 'verification/public-corpus-census/sources.json')
    parser.add_argument('--check', action='store_true',
                        help='verify an existing provenance-check.json instead of writing one')
    args = parser.parse_args(argv)

    for path in (args.holdout, args.provenance, args.census):
        if not path.is_file():
            raise SystemExit(f'missing {path}')
    if not args.wave_dir.is_dir():
        raise SystemExit(f'{args.wave_dir} is not a directory')

    report = build_check(args.wave_dir, args.holdout, args.provenance, args.census)
    out_path = args.wave_dir / 'provenance-check.json'

    if args.check:
        if not out_path.is_file():
            raise SystemExit(f'{out_path} does not exist; run without --check to write it')
        wave_text = (args.wave_dir / 'wave.json').read_text(encoding='utf-8')
        if ABSOLUTE_PATH.search(wave_text) or TASK_ID.search(wave_text):
            raise SystemExit('wave.json still carries a machine-local path or a board id; '
                             're-run without --check to sanitise it')
        existing = json.loads(out_path.read_text(encoding='utf-8'))
        stale = comparable(existing) != comparable(report)
        if stale:
            keys = sorted({key for key in set(existing) | set(report)
                           if existing.get(key) != report.get(key)})
            raise SystemExit(f'{out_path} is stale against its inputs; differing: {", ".join(keys)}')
        if not report['ok']:
            raise SystemExit(f'{out_path} is current but the wave fails: {report["failed_checks"]}')
    else:
        # Sanitise the wave in place: the committed wave.json must not carry a
        # machine-local prefix, and the rewrite is listed in the check file.
        wave_path = args.wave_dir / 'wave.json'
        rewrites = [item for item in report['sanitisation']['rewrites'] if item['field'].startswith('wave')]
        if wave_path.is_file() and rewrites:
            wave = json.loads(wave_path.read_text(encoding='utf-8'))
            wave_path.write_text(json.dumps(sanitise_value(wave, 'wave', REPO, []), indent=1,
                                            sort_keys=True) + '\n', encoding='utf-8')
            report['wave']['wave_json_sha256'] = file_sha256(wave_path)
        out_path.write_text(json.dumps(report, indent=1, sort_keys=True) + '\n', encoding='utf-8')

    print(json.dumps({'rows': report['wave']['rows'],
                      'screen': report['screen'],
                      'resolved_provenance': report['resolved_provenance'],
                      'ok': report['ok'],
                      'failed_checks': report['failed_checks'],
                      'out': str(out_path) if not args.check else None}, indent=1))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
