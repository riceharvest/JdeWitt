"""Census artifact contract: the untyped split, the quarterly stamp and the no-clobber write.

These exercise `hub_census` against a fake Hub inventory, so the shape and the
refusal behaviour are pinned without a network call.
"""
import json
from types import SimpleNamespace

import huggingface_hub
import pytest

from jdewitt.hubcensus import hub_census, quarter_of
from jdewitt.schemas import enforce


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
