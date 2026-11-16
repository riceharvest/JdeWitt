# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pin the public-corpus example: spec checks offline, suite checks when imported."""
import json
from pathlib import Path

import pytest

from b70_ablator.data import load_suite
from b70_ablator.search import RunConfig
from b70_ablator.sources import import_suites, validate_import
from b70_ablator.validation import validate_run_inputs

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples' / 'public-corpus'
IMPORTED = Path(__file__).resolve().parents[1] / 'verification' / 'public-corpus-suites'


def test_example_import_spec_checks_offline(tmp_path):
    spec = json.loads((EXAMPLES / 'import.json').read_text())
    report = validate_import(spec)
    assert report['status'] == 'metadata-valid'
    assert report['downloads_performed'] is False and report['output_created'] is False
    assert report['remote_sources'] == 4 and report['sources'] == 6
    assert sorted(report['splits']) == ['holdout', 'selection']
    assert sum(entry['passthrough'] for entry in report['sources_checked']) == 2
    assert not list(tmp_path.iterdir())


def test_example_import_spec_declares_pinned_revisions_for_every_remote_source():
    spec = json.loads((EXAMPLES / 'import.json').read_text())
    remote = [source for sources in spec['splits'].values() for source in sources if 'dataset' in source]
    assert remote and all(len(source['revision']) == 40 for source in remote)
    assert all(source.get('column') for source in remote)
    assert all(source['split'] for source in remote)


def test_example_import_is_reproducible_from_the_spec(tmp_path):
    """The committed suites must be exactly what the spec produces, byte for byte."""
    if not (IMPORTED / 'selection.jsonl').is_file():
        pytest.skip('public-corpus suites have not been imported into this checkout')
    spec = json.loads((EXAMPLES / 'import.json').read_text())
    fresh = tmp_path / 'fresh'
    import_suites(spec, fresh)
    for split in ('selection', 'holdout'):
        assert (fresh / f'{split}.jsonl').read_bytes() == (IMPORTED / f'{split}.jsonl').read_bytes()
    committed = json.loads((IMPORTED / 'import-provenance.json').read_text())
    produced = json.loads((fresh / 'import-provenance.json').read_text())
    # Only the recorded output paths may differ; every other fact must be identical.
    def strip(report):
        return {**report, 'splits': {name: {k: v for k, v in entry.items() if k != 'path'}
                                     for name, entry in report['splits'].items()}}
    assert strip(produced) == strip(committed)
    assert committed['synthetic_family_sources'] == 2
    assert all(record['file_sha256'] for record in committed['sources'])


@pytest.mark.skipif(not (IMPORTED / 'selection.jsonl').is_file(),
                    reason='public-corpus suites have not been imported into this checkout')
def test_example_run_config_accepts_the_imported_suites():
    config = RunConfig.from_json(json.loads((EXAMPLES / 'run.json').read_text()))
    suites, report = validate_run_inputs(config)
    assert report['status'] == 'metadata-valid' and report['models_loaded'] is False
    assert {name: len(rows) for name, rows in suites.items()} == {'directions': 16, 'selection': 246,
                                                                  'holdout': 246}
    assert report['direction_families'] == {'en': 2, 'nl': 2}
    imported = load_suite(config.selection)
    refusal_families = {row['family_id'] for row in imported if row['kind'] == 'refusal'}
    assert len(refusal_families) == 60
    assert {row['kind'] for row in imported} == {'refusal', 'harmless', 'math', 'code'}
    assert not Path(config.output).exists()
