"""Census artifact contract: the untyped split, the quarterly stamp and the no-clobber write.

These exercise `hub_census` against a fake Hub inventory, so the shape and the
refusal behaviour are pinned without a network call.
"""
import json
import re
import shlex
from types import SimpleNamespace

import huggingface_hub
import pytest

from jdewitt.cli import build_parser
from jdewitt.hubcensus import census_next_steps, hub_census, quarter_of
from jdewitt.schemas import enforce, validate_artifact


class FakeApi:
    """Minimal HfApi stand-in: author -> models, newest first."""

    def __init__(self, models):
        self.models = models

    def list_models(self, author=None, **kwargs):
        assert kwargs['pipeline_tag'] == 'text-generation'
        return iter(self.models.get(author, []))


def model(repo, downloads, config, created_at='2026-11-01T00:00:00.000Z'):
    return SimpleNamespace(id=repo, downloads=downloads, config=config, created_at=created_at)


@pytest.fixture
def hub(monkeypatch):
    inventory = {
        'LiquidAI': [model('LiquidAI/LFM3-1.7B', 212_493, {'model_type': 'lfm3'})],
        'nvidia': [model('nvidia/NVIDIA-Nemotron-3.5-Bolt', 214_731, None),
                   model('nvidia/Dory-4B', 61_627, {'model_type': 'dory'})],
        'baidu': [model('baidu/ERNIE-5.5-Flash', 7_449, {'model_type': 'ernie5_5_moe'})],
        'Qwen': [model('Qwen/Qwen3-4B', 900_000, {'model_type': 'qwen3'}),
                 model('Qwen/Qwen3-gguf', 5_000, None),
                 model('Qwen/old', 99, {'model_type': 'qwen3'}, created_at='2026-01-01T00:00:00.000Z')],
    }
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda *a, **k: FakeApi(inventory))
    monkeypatch.setenv('JDEWITT_CLOCK', '2026-11-22T19:52:07Z')
    return inventory


def test_untyped_rows_leave_the_worklist_and_keep_their_downloads(hub):
    payload = hub_census(authors=list(hub), since='2026-08-01', per_author=10)
    assert [row['model_type'] for row in payload['adapter_worklist']] == ['lfm3', 'dory', 'ernie5_5_moe']
    assert payload['untyped_repositories'] == [
        {'label': 'no-model-type', 'downloads': 214_731,
         'example_models': ['nvidia/NVIDIA-Nemotron-3.5-Bolt']}]
    assert 'no-model-type' not in [row['model_type'] for row in payload['adapter_worklist']]
    # The derived artifact repository is counted, skipped, and absent from both lists.
    assert payload['derived_artifact_repositories'] == 1
    assert payload['repositories_checked'] == 6
    assert payload['window'] == {'since': '2026-08-01', 'authors': list(hub), 'per_author_cap': 10}
    assert payload['supported_model_types'] == ['qwen3']
    assert payload['unresolved_gaps'] and payload['author_errors'] == []
    enforce('hub-census', payload)


def test_census_stamps_utc_instant_and_quarter(hub):
    payload = hub_census(authors=['LiquidAI'], since='2026-08-01')
    assert payload['census_at'] == '2026-11-22T19:52:07Z'
    assert payload['quarter'] == '2026-Q4'
    assert quarter_of('2026-01-01T00:00:00Z') == '2026-Q1'
    assert quarter_of('2026-06-30T23:59:59Z') == '2026-Q2'
    assert quarter_of('2026-12-31T23:59:59Z') == '2026-Q4'


def test_out_persists_the_same_payload_and_refuses_to_clobber(hub, tmp_path):
    target = tmp_path / 'hub-census' / '2026-11-22.json'
    payload = hub_census(authors=['LiquidAI'], since='2026-08-01', out=target)
    assert json.loads(target.read_text()) == payload
    # The directory names the writing command, so verify-artifact needs no --name.
    checked = validate_artifact(target)
    assert checked['schema'] == 'hub-census.command.json' and checked['valid']
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        hub_census(authors=['LiquidAI'], since='2026-08-01', out=target)
    assert target.read_bytes() == before


def test_a_failing_author_is_an_error_row_not_a_lost_census(monkeypatch):
    class Broken(FakeApi):
        def list_models(self, author=None, **kwargs):
            raise RuntimeError('hub said no')

    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda *a, **k: Broken({}))
    monkeypatch.setenv('JDEWITT_CLOCK', '2026-11-22T19:52:07Z')
    payload = hub_census(authors=['ghost'], since='2026-08-01')
    assert payload['author_errors'] == [{'author': 'ghost', 'error': 'hub said no'}]
    assert payload['adapter_worklist'] == [] and payload['untyped_repositories'] == []


def test_next_is_a_capture_tune_pair_per_worklist_row_in_download_order(hub):
    payload = hub_census(authors=list(hub), since='2026-08-01', per_author=10)
    steps = payload['next']
    assert len(steps) == 2 * len(payload['adapter_worklist']) == 6
    assert steps[0].startswith('uv run jdewitt capture --model LiquidAI/LFM3-1.7B')
    assert 'records-lfm3.pt' in steps[0] and 'records-lfm3.pt' in steps[1]
    assert steps[1].startswith('uv run jdewitt tune')
    # Two architectures never share a records file, and the untyped row is not an
    # architecture, so it earns no adapter work at all.
    records = {re.search(r'records-(\S+)\.pt', step).group(1) for step in steps}
    assert records == {'lfm3', 'dory', 'ernie5_5_moe'}
    assert not any('no-model-type' in step for step in steps)


def test_next_follows_the_artifact_order():

    worklist = [{'model_type': 'lfm3', 'downloads': 10, 'example_models': ['a/LFM3']},
                {'model_type': 'dory', 'downloads': 5, 'example_models': ['b/Dory']}]
    steps = census_next_steps(worklist)
    assert [step.startswith('uv run jdewitt capture') for step in steps] == [True, False, True, False]
    assert 'records-dory.pt' in steps[3]


def test_every_next_line_is_a_command_the_cli_parses(hub):
    parser = build_parser()
    payload = hub_census(authors=list(hub), since='2026-08-01', per_author=10)
    for step in payload['next']:
        tokens = shlex.split(step)
        assert tokens[:3] == ['uv', 'run', 'jdewitt']
        parsed = parser.parse_args(tokens[3:])
        assert parsed.command in ('capture', 'tune')
        # Required arguments the worklist must name for the line to be runnable.
        if parsed.command == 'capture':
            assert parsed.model and parsed.suite and parsed.ctx_size and parsed.out
        else:
            assert parsed.gguf and parsed.records
