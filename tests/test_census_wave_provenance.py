# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contract tests for the census wave provenance check.

The wave itself needs a served artifact; these pin the join back to the import
split and the census without one, and every failure mode asserts that the check
says no rather than quietly writing a green file.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('census_wave_provenance',
                                              REPO / 'scripts' / 'census_wave_provenance.py')
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

DATASET = 'walledai/StrongREJECT'
REVISION = 'df54272b395ce9e584d2f8140519935425d8549f'


def _census(tmp_path, license='mit', sha_override=None):
    entry = {'id': 'strongreject', 'dataset': DATASET, 'revision': REVISION,
             'license': license, 'license_url': 'https://huggingface.co/datasets/walledai/StrongREJECT'}
    text = json.dumps({'format': 'jdewitt-public-corpora/1', 'generated': '2026-11-22',
                       'sources': [entry]}, indent=1) + '\n'
    path = tmp_path / 'sources.json'
    path.write_text(text)
    sha = sha_override or hashlib.sha256(path.read_bytes()).hexdigest()
    return path, sha


def _split(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _fixture(tmp_path, *, ids=('strongreject-000001', 'strongreject-000002'),
             holdout_license='mit', census_license='mit', tokens=42, ledger=42,
             census_sha=None, table='more'):
    holdout = []
    for number, row_id in enumerate(ids):
        holdout.append({'id': row_id, 'kind': 'refusal', 'prompt': f'prompt {number}',
                        'language': 'en', 'source': 'strongreject', 'license': holdout_license,
                        'category': 'Violence', 'family_id': f'f-{number}'})
    (tmp_path / 'holdout.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in holdout))
    census_path, census_sha = _census(tmp_path, license=census_license, sha_override=census_sha)
    provenance = {'status': 'imported', 'seed': 42,
                  'splits': {'holdout': {'path': 'holdout.jsonl', 'tasks': len(holdout),
                                         'digest': _split(holdout)}},
                  'census': {'path': 'sources.json', 'sha256': census_sha}}
    (tmp_path / 'provenance.json').write_text(json.dumps(provenance, indent=1) + '\n')

    wave_dir = tmp_path / 'wave'
    wave_dir.mkdir()
    predictions = [{'id': row['id'], 'prompt': row['prompt'], 'kind': 'refusal',
                    'answer': 'I cannot help with that.', 'completion_tokens': 21,
                    'status': 'ok',
                    'refusal_check': {'result': 'refusal_detected', 'method': 'explicit-wording-v1'}}
                   for row in holdout]
    (wave_dir / 'predictions.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in predictions))
    wave = {'label': 'census-strongreject', 'rows': len(predictions), 'kinds': ['refusal'],
            'completion_tokens_total': ledger, 'measured': True, 'invalid': 0,
            'refusals_detected': len(predictions), 'delivered_instead': 0,
            'suite': str(tmp_path / 'holdout.jsonl'),
            'serve_config': '-m /home/someone/models/tiny.gguf --ctx-size 4096 --jinja',
            'serve_structured': {'ctx': 4096, 'jinja': True, 'suite': '/mnt/scratch/holdout.jsonl'},
            'predictions': str(wave_dir / 'predictions.jsonl')}
    (wave_dir / 'wave.json').write_text(json.dumps(wave, indent=1, sort_keys=True) + '\n')
    return {'wave_dir': wave_dir, 'holdout': tmp_path / 'holdout.jsonl',
            'provenance': tmp_path / 'provenance.json', 'census': census_path}


def _build(paths, repo_root):
    return module.build_check(paths['wave_dir'], paths['holdout'], paths['provenance'],
                              paths['census'], repo_root=repo_root)


def _failed(report):
    return {check['id'] for check in report['checks'] if not check['ok']}


def test_happy_path_resolves_dataset_revision_and_licence(tmp_path):
    paths = _fixture(tmp_path)
    report = _build(paths, tmp_path)

    assert report['ok'], _failed(report)
    assert report['resolved_provenance'] == [
        {'id': 'strongreject', 'dataset': DATASET, 'revision': REVISION, 'license': 'mit',
         'license_url': 'https://huggingface.co/datasets/walledai/StrongREJECT',
         'rows': 2, 'row_license_values': ['mit']}]
    assert report['screen']['removal_claimed'] is False
    assert report['claims']['removal'] is False
    assert report['wave']['rows'] == 2


def test_absolute_paths_are_rewritten_not_committed(tmp_path):
    paths = _fixture(tmp_path)
    report = _build(paths, tmp_path)

    dumped = json.dumps(report)
    assert '/home/' not in dumped and '/mnt/' not in dumped
    assert report['argv']['serve_config'] == '-m <local>/tiny.gguf --ctx-size 4096 --jinja'
    kinds = {item['kind'] for item in report['sanitisation']['rewrites']}
    assert {'redacted-local', 'repo-relative'} <= kinds


def test_missing_prediction_id_fails(tmp_path):
    paths = _fixture(tmp_path)
    lines = (paths['wave_dir'] / 'predictions.jsonl').read_text().splitlines()
    (paths['wave_dir'] / 'predictions.jsonl').write_text(lines[0] + '\n')
    (paths['wave_dir'] / 'wave.json').write_text(json.dumps({'rows': 1, 'measured': True,
                                                             'completion_tokens_total': 21,
                                                             'serve_config': '-m x.gguf'}))

    report = _build(paths, tmp_path)
    assert report['ok'] is False
    assert 'ids_resolve_to_holdout' in _failed(report)


def test_row_licence_disagreeing_with_census_fails(tmp_path):
    paths = _fixture(tmp_path, holdout_license='apache-2.0')
    report = _build(paths, tmp_path)

    assert 'rows_carry_census_license_and_pinned_revision' in _failed(report)
    assert report['resolved_provenance'][0]['row_license_values'] == ['apache-2.0']


def test_census_changing_after_the_wave_fails(tmp_path):
    paths = _fixture(tmp_path, census_sha='0' * 64)
    report = _build(paths, tmp_path)

    assert 'census_digest_matches_provenance' in _failed(report)
    assert report['census']['matches'] is False


def test_token_ledger_must_agree_with_rows(tmp_path):
    paths = _fixture(tmp_path, ledger=7)
    report = _build(paths, tmp_path)

    assert 'token_ledger_agrees' in _failed(report)


def test_absolute_path_inside_an_answer_is_reported_not_rewritten(tmp_path):
    paths = _fixture(tmp_path)
    rows = [json.loads(line) for line in (paths['wave_dir'] / 'predictions.jsonl').read_text().splitlines()]
    rows[0]['answer'] = 'see /home/someone/secret.txt'
    (paths['wave_dir'] / 'predictions.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows))

    report = _build(paths, tmp_path)
    assert 'predictions_have_no_local_paths' in _failed(report)
    assert report['sanitisation']['predictions_leaks'][0]['id'] == rows[0]['id']


def test_main_writes_the_check_and_check_mode_detects_drift(tmp_path, capsys):
    paths = _fixture(tmp_path)
    argv = [str(paths['wave_dir']), '--holdout', str(paths['holdout']),
            '--provenance', str(paths['provenance']), '--census', str(paths['census'])]
    assert module.main(argv) == 0

    out_path = paths['wave_dir'] / 'provenance-check.json'
    written = json.loads(out_path.read_text())
    assert written['ok'] is True
    assert 'wave_json_sha256' in written['wave']
    committed_wave = (paths['wave_dir'] / 'wave.json').read_text()
    assert '/home/' not in committed_wave and '/mnt/' not in committed_wave

    assert module.main(argv + ['--check']) == 0

    rows = [json.loads(line) for line in (paths['wave_dir'] / 'predictions.jsonl').read_text().splitlines()]
    rows[0]['completion_tokens'] = 99
    (paths['wave_dir'] / 'predictions.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(SystemExit, match='stale'):
        module.main(argv + ['--check'])
