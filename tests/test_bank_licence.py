"""Licence and access gate: two refusals, verbatim licence, bank-owned allow-list.

These exercise the gate against a fake Hub, so the refusal behaviour, the cache
key and the digest line are pinned without a network call.
"""
import hashlib
import json
from types import SimpleNamespace

import huggingface_hub
import pytest

from jdewitt import bank_licence
from jdewitt.bank_licence import (ALLOW_LIST_SCHEMA, DECISION_ALLOWED, DECISION_REFUSED, GATED_REPO,
                                  LICENCE_ACCEPTED, LICENCE_UNKNOWN, licence_gate, load_allow_list)

APACHE = {'apache-2.0': {'name': 'Apache License 2.0', 'used_by': ['qwen3-4b-instruct-2507/f15s15-l935']}}


class FakeApi:
    def __init__(self, table):
        self.table = table
        self.calls = []

    def model_info(self, repo, revision=None, files_metadata=None):
        self.calls.append((repo, revision, files_metadata))
        assert files_metadata is False
        return self.table[(repo, revision)]


def hub_info(gated: object = False, licence: object = 'apache-2.0', tags: list | None = None,
             sha: str = 'deadbeef'):
    card = SimpleNamespace(license=licence) if licence is not None else None
    if tags is None:
        tags = [f'license:{licence}'] if licence else []
    return SimpleNamespace(gated=gated, card_data=card, tags=tags, sha=sha)


def write_bank(directory, allow=None):
    directory.mkdir(parents=True, exist_ok=True)
    payload = {'schema': ALLOW_LIST_SCHEMA, 'allow': allow if allow is not None else APACHE}
    (directory / 'licences.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    digest = hashlib.sha256((directory / 'licences.json').read_bytes()).hexdigest()
    (directory / 'SHA256SUMS').write_text(f'{digest}  licences.json\n', encoding='utf-8')
    return directory


@pytest.fixture(autouse=True)
def clear_hub_cache():
    bank_licence.clear_cache()
    yield
    bank_licence.clear_cache()


def test_allowed_licence_is_returned_verbatim_and_cached_per_repo_revision(tmp_path):
    write_bank(tmp_path)
    api = FakeApi({('Qwen/Qwen3-4B-Instruct-2507', 'rev1'): hub_info(licence='Apache-2.0'),
                   ('Qwen/Qwen3-4B-Instruct-2507', 'rev2'): hub_info(licence='Apache-2.0')})
    first = licence_gate('Qwen/Qwen3-4B-Instruct-2507', 'rev1', bank_dir=tmp_path, api=api)
    assert first['decision'] == DECISION_ALLOWED and first['reason'] == LICENCE_ACCEPTED
    # Casefold matches the bank key, but the card's spelling is what goes on the output card.
    assert first['licence'] == 'Apache-2.0'
    assert first['licence_source'] == 'cardData.license'
    assert first['resolved_revision'] == 'deadbeef'
    # Same (repo, revision) is one Hub call; a new revision is a new call.
    licence_gate('Qwen/Qwen3-4B-Instruct-2507', 'rev1', bank_dir=tmp_path, api=api)
    licence_gate('Qwen/Qwen3-4B-Instruct-2507', 'rev2', bank_dir=tmp_path, api=api)
    assert api.calls == [('Qwen/Qwen3-4B-Instruct-2507', 'rev1', False),
                         ('Qwen/Qwen3-4B-Instruct-2507', 'rev2', False)]


def test_gated_repo_is_refused_without_needing_the_bank(tmp_path):
    missing_bank = tmp_path / 'not-created'
    api = FakeApi({('meta-llama/Llama-3.2-1B', 'rev'): hub_info(gated='manual', licence='llama3.2')})
    row = licence_gate('meta-llama/Llama-3.2-1B', 'rev', bank_dir=missing_bank, api=api)
    assert row['decision'] == DECISION_REFUSED and row['reason'] == GATED_REPO
    assert row['licence'] == 'llama3.2'
    assert 'anonymous Space cannot authenticate' in row['detail']
    assert 'allow_list_sha256' not in row


def test_gated_false_string_is_not_gated(tmp_path):
    write_bank(tmp_path)
    api = FakeApi({('Qwen/Qwen3-4B-Instruct-2507', 'rev'): hub_info(gated='false')})
    assert licence_gate('Qwen/Qwen3-4B-Instruct-2507', 'rev', bank_dir=tmp_path, api=api)['decision'] == DECISION_ALLOWED


def test_off_list_licence_is_unknown_and_names_what_was_found(tmp_path):
    write_bank(tmp_path)
    api = FakeApi({('bigscience/bloom-560m', 'rev'): hub_info(licence='bigscience-bloom-rail-1.0')})
    row = licence_gate('bigscience/bloom-560m', 'rev', bank_dir=tmp_path, api=api)
    assert row['decision'] == DECISION_REFUSED and row['reason'] == LICENCE_UNKNOWN
    assert row['licence'] == 'bigscience-bloom-rail-1.0'
    assert 'bigscience-bloom-rail-1.0' in row['detail'] and 'apache-2.0' in row['detail']


def test_missing_licence_is_unknown(tmp_path):
    write_bank(tmp_path)
    api = FakeApi({('owner/no-card', 'rev'): hub_info(licence=None, tags=[])})
    row = licence_gate('owner/no-card', 'rev', bank_dir=tmp_path, api=api)
    assert row['decision'] == DECISION_REFUSED and row['reason'] == LICENCE_UNKNOWN
    assert row['licence'] is None and row['licence_source'] == 'missing'
    assert 'no licence' in row['detail']


def test_licence_tag_is_used_when_card_data_is_absent(tmp_path):
    write_bank(tmp_path, {'mit': {'name': 'MIT License', 'used_by': ['fixture']}})
    api = FakeApi({('owner/tag-only', 'rev'): hub_info(licence=None, tags=['license:mit'])})
    row = licence_gate('owner/tag-only', 'rev', bank_dir=tmp_path, api=api)
    assert row['decision'] == DECISION_ALLOWED and row['licence'] == 'mit'
    assert row['licence_source'] == 'tags'


def test_ambiguous_licence_tags_are_unknown_and_named(tmp_path):
    write_bank(tmp_path)
    api = FakeApi({('owner/two-licences', 'rev'): hub_info(licence=None, tags=['license:mit', 'license:apache-2.0'])})
    row = licence_gate('owner/two-licences', 'rev', bank_dir=tmp_path, api=api)
    assert row['reason'] == LICENCE_UNKNOWN and row['licence'] is None
    assert 'mit' in row['detail'] and 'apache-2.0' in row['detail']


def test_no_download_function_is_imported_or_called(tmp_path, monkeypatch):
    write_bank(tmp_path)

    def refuse_download(*args, **kwargs):
        raise AssertionError('the licence gate must not download')

    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', refuse_download)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', refuse_download)
    api = FakeApi({('Qwen/Qwen3-4B-Instruct-2507', 'rev'): hub_info()})
    assert licence_gate('Qwen/Qwen3-4B-Instruct-2507', 'rev', bank_dir=tmp_path, api=api)['decision'] == DECISION_ALLOWED


def test_missing_allow_list_fails_closed(tmp_path):
    api = FakeApi({('Qwen/Qwen3-4B-Instruct-2507', 'rev'): hub_info()})
    with pytest.raises(ValueError, match='allow-list is missing'):
        licence_gate('Qwen/Qwen3-4B-Instruct-2507', 'rev', bank_dir=tmp_path, api=api)


def test_allow_list_digest_mismatch_fails_closed(tmp_path):
    write_bank(tmp_path)
    (tmp_path / 'licences.json').write_text(json.dumps({'schema': ALLOW_LIST_SCHEMA, 'allow': {'mit': {}}}),
                                           encoding='utf-8')
    with pytest.raises(ValueError, match='digest'):
        load_allow_list(tmp_path)


def test_repository_bank_allows_the_v1_base_and_reports_the_digest_line():
    table, digest, path = load_allow_list()
    assert table == {'apache-2.0': 'apache-2.0'}
    assert path.name == 'licences.json' and len(digest) == 64
    sums = path.with_name('SHA256SUMS').read_text(encoding='utf-8').split()
    assert sums == [digest, 'licences.json']
