# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pin the MiniCPM-V 4.7 trial example: metadata only, no model or GPU work."""
import json
from pathlib import Path

import pytest

from jdewitt.auto import AutoConfig, validate_auto
from jdewitt.data import load_suite
from jdewitt.search import RunConfig
from jdewitt.validation import validate_run_inputs

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples' / 'minicpm-v47'
MODEL = Path('/mnt/ssd/models/MiniCPM-V-4.7-1B')
requires_model = pytest.mark.skipif(not (MODEL / 'config.json').is_file(),
                                    reason='MiniCPM-V 4.7 trial model is not present locally')


def test_trial_suites_keep_splits_disjoint_and_family_annotated():
    directions = load_suite(EXAMPLES / 'directions.jsonl')
    selection = load_suite(EXAMPLES / 'selection.jsonl')
    holdout = load_suite(EXAMPLES / 'holdout.jsonl')
    assert len(directions) == 16 and len(selection) == 18 and len(holdout) == 18
    assert {r['language'] for r in directions} == {'en', 'nl'}
    for suite in (selection, holdout):
        assert {r['kind'] for r in suite} == {'refusal', 'harmless', 'math', 'code'}
        assert all(r.get('family_id') for r in suite if r['kind'] == 'refusal')
    direction_families = {r['family_id'] for r in directions}
    evaluated_families = {r['family_id'] for r in selection + holdout if r['kind'] == 'refusal'}
    assert not direction_families & evaluated_families


@requires_model
def test_trial_run_config_validates_without_loading_any_model():
    config = RunConfig.from_json(json.loads((EXAMPLES / 'run.json').read_text()))
    suites, report = validate_run_inputs(config)
    assert report['status'] == 'metadata-valid' and report['models_loaded'] is False
    assert report['output_created'] is False
    assert report['direction_families'] == {'en': 2, 'nl': 2}
    assert config.method == 'directional' and config.quantization == 'none'
    assert config.scoring_mode == 'refusal-check'
    assert {name: len(rows) for name, rows in suites.items()} == {
        'directions': 16, 'selection': 18, 'holdout': 18}
    assert not Path(config.output).exists()


@requires_model
def test_trial_auto_config_allocates_every_trial_and_creates_nothing():
    raw = json.loads((EXAMPLES / 'auto.json').read_text())
    config = AutoConfig.from_json(raw)
    assert [c['method'] for c in config.candidates] == ['directional', 'projected', 'mpoa', 'svd']
    assert sum(c['trials'] for c in config.candidates) == raw['total_trials'] == 8
    report = validate_auto(config)
    assert report['trial_budget'] == 8 and report['models_loaded'] is False
    assert 'no fallback after holdout failure' in report['holdout_policy'].lower()
    assert not Path(config.run.output).exists()
    for candidate in report['candidates']:
        assert candidate['status'] == 'metadata-valid'
