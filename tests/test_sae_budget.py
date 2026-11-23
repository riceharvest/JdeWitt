"""The budget arithmetic and its doc-diff, plus the deterministic suite builder."""
import json
from pathlib import Path

import pytest

import scripts.sae_budget as budget


def test_arithmetic_prints_the_bytes_flops_and_zero_tokens():
    numbers = budget.arithmetic(rows=168, families=4, layers=28, hidden=2048, features=256, k=16,
                                steps=1200, batch=16, rate=None, merge_cert_seconds=540)
    assert numbers['activation_bytes_per_row'] == 28 * 2048 * 4
    assert numbers['train_matrix_bytes'] == 256 * 2048 * 4
    assert numbers['step_flops'] == 2 * 16 * 256 * 2048 * 3
    assert numbers['generated_tokens'] == 0
    assert numbers['step_rate_source'] == 'ASSUMPTION'
    zeros = [key for key, value in numbers.items() if value == 0 and not isinstance(value, bool)]
    assert zeros == ['generated_tokens']


def test_a_measured_rate_is_labelled_measured_and_changes_the_gpu_numbers():
    assumed = budget.arithmetic(rows=168, families=4, layers=28, hidden=2048, features=256, k=16,
                                steps=1200, batch=16, rate=None, merge_cert_seconds=540)
    measured = budget.arithmetic(rows=168, families=4, layers=28, hidden=2048, features=256, k=16,
                                 steps=1200, batch=16, rate=10.0, merge_cert_seconds=540)
    assert measured['step_rate_source'] == 'measured'
    assert measured['gpu_seconds_per_family'] == pytest.approx(120.0)
    assert measured['gpu_seconds_total'] == pytest.approx(480.0)
    assert measured['gpu_hours_total'] == pytest.approx(480.0 / 3600.0, abs=1e-6)
    assert assumed['gpu_seconds_per_family'] < measured['gpu_seconds_per_family']


def test_stop_rule_fires_when_a_family_costs_more_than_the_merge_cert():
    numbers = budget.arithmetic(rows=168, families=4, layers=28, hidden=2048, features=256, k=16,
                                steps=1200, batch=16, rate=1.0, merge_cert_seconds=540)
    assert numbers['stop_rule_fires'] is True
    slow_ok = budget.arithmetic(rows=168, families=4, layers=28, hidden=2048, features=256, k=16,
                                steps=1200, batch=16, rate=10.0, merge_cert_seconds=540)
    assert slow_ok['stop_rule_fires'] is False


def _doc(tmp_path, extra=''):
    path = tmp_path / 'PLAN.md'
    path.write_text('# plan\n\n<!-- sae-budget:start -->\n'
                    'rows = 168\nfamilies = 4\nlayers = 28\nhidden = 2048\nfeatures = 256\nk = 16\n'
                    'steps = 1200\nbatch = 16\nactivation_bytes_per_row = 229376\n'
                    'train_matrix_bytes = 2097152\nstep_flops = 50331648\n'
                    'steps_per_second = 10.0\nstep_rate_source = measured\n'
                    'gpu_seconds_per_family = 120.0\ngpu_seconds_total = 480.0\n'
                    'gpu_hours_total = 0.133333\nmerge_cert_seconds = 540\ngenerated_tokens = 0\n'
                    'stop_rule_gpu_seconds_per_family = 540\nstop_rule_fires = False\n'
                    f'{extra}<!-- sae-budget:end -->\n')
    return path


def test_compare_passes_on_a_matching_doc_and_fails_on_a_moved_number(tmp_path, capsys):
    numbers = budget.arithmetic(rows=168, families=4, layers=28, hidden=2048, features=256, k=16,
                                steps=1200, batch=16, rate=10.0, merge_cert_seconds=540)
    assert budget.compare(numbers, _doc(tmp_path)) == 0
    assert 'doc numbers == script output' in capsys.readouterr().out
    assert budget.compare(numbers, _doc(tmp_path, extra='features = 512\n')) == 1
    assert 'doc numbers != script output' in capsys.readouterr().out


def test_read_doc_table_refuses_a_doc_without_the_block(tmp_path):
    path = tmp_path / 'EMPTY.md'
    path.write_text('# nothing here\n')
    with pytest.raises(SystemExit, match='sae-budget:start'):
        budget.read_doc_table(path)


def _suite_source(tmp_path, refusal_per_family=6, families=('fam-a', 'fam-b'), harmless=3):
    rows = []
    for family in families:
        for index in range(refusal_per_family):
            rows.append({'id': f'{family}-{index:03d}', 'kind': 'refusal', 'prompt': 'p',
                         'family_id': family})
    for index in range(harmless):
        rows.append({'id': f'good-{index:03d}', 'kind': 'harmless', 'prompt': 'q'})
    path = tmp_path / 'source.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return path


def test_suite_builder_is_deterministic_and_family_disjoint(tmp_path):
    import scripts.p5_sae_suite as suite
    source = _suite_source(tmp_path)
    first = suite.build(source, ['fam-a', 'fam-b'], per_family=4, harmless=2)
    second = suite.build(source, ['fam-a', 'fam-b'], per_family=4, harmless=2)
    assert [row['id'] for row in first['suite']] == [row['id'] for row in second['suite']]
    assert first['meta']['rows'] == 4 * 2 + 2
    train_families = {row['family_id'] for row in first['suite'] if row['kind'] == 'refusal'}
    good = [row for row in first['suite'] if row['kind'] == 'harmless']
    assert train_families == {'fam-a', 'fam-b'}
    assert all(row['family_id'] is None for row in good)
    # No row is used twice, and each family contributes exactly the asked count.
    ids = [row['id'] for row in first['suite']]
    assert len(ids) == len(set(ids))


def test_suite_builder_refuses_a_family_short_of_rows(tmp_path):
    import scripts.p5_sae_suite as suite
    source = _suite_source(tmp_path, refusal_per_family=2)
    with pytest.raises(SystemExit, match='short of 4'):
        suite.build(source, ['fam-a'], per_family=4, harmless=1)


def test_suite_builder_writes_a_digest_that_matches_the_file(tmp_path):
    import hashlib
    import scripts.p5_sae_suite as suite
    source = _suite_source(tmp_path)
    out = tmp_path / 'out.jsonl'
    args = ['--source', str(source), '--family', 'fam-a', '--per-family', '3',
            '--harmless', '2', '--out', str(out)]
    assert suite.main(args) == 0
    meta = json.loads(out.with_suffix('.meta.json').read_text())
    assert meta['suite_sha256'] == hashlib.sha256(Path(out).read_bytes()).hexdigest()
    assert meta['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
