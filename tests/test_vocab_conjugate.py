# SPDX-License-Identifier: AGPL-3.0-or-later
"""The opener conjugation: bijection, storage exactness, row identity and revert.

A tiny Q8_0 fixture with a hand-built byte-level BPE tokenizer. The point is not
scale: it is that the map is a transposition on the chosen ids, that the codec
stores the partner's row byte for byte, that exactly the mapped rows move, and
that the pre-edit bytes restore the file. No model is loaded anywhere.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from jdewitt.ggufio import revert_inplace, write_gguf
from jdewitt.quants import CODECS
from jdewitt.vocabconjugate import SCHEMA, load_pairs, resolve_pairs, vocab_conjugate
from jdewitt.vocabremap import _byte_level, read_tokenizer

WIDTH = 32  # exactly one Q8_0 block per row


def _merges(symbol):
    """Merge chain that builds one byte-level symbol out of its characters."""
    characters = list(symbol)
    chain, current = [], characters[0]
    for character in characters[1:]:
        chain.append(f'{current} {character}')
        current += character
    return chain


def _table(surfaces):
    """A pair table over the given (denial, compliance) surfaces."""
    return {'schema': 'jdewitt.vocab-conjugate-pairs/1',
            'pairs': [{'id': f'pair-{index}', 'language': 'en', 'denial': denial,
                       'compliance': compliance, 'kind': 'denial_opener'}
                      for index, (denial, compliance) in enumerate(surfaces)]}


def _fixture(tmp_path, surfaces=(('Sorry', 'Sure'), ('I', 'Here')), name='m.gguf'):
    # Merge ranks are the list order. Every leading-space chain comes before every
    # bare chain, so BPE can never merge 'S u' before 'Ġ S' and split the spaced
    # symbol into two tokens: the fixture must resolve each surface to one id in
    # both contexts, exactly like the real tokenizer does.
    spaced = [_byte_level(' ' + surface) for pair in surfaces for surface in pair]
    bare = [_byte_level(surface) for pair in surfaces for surface in pair]
    merges = []
    for symbol in spaced + bare:
        for merge in _merges(symbol):
            if merge not in merges:
                merges.append(merge)
    tokens = sorted(set(spaced + bare))
    rows = len(tokens)
    rng = np.random.RandomState(7)
    values = rng.randn(rows, WIDTH).astype(np.float32)
    block_bytes, block_els, quantize, _decode = CODECS['Q8_0']
    assert WIDTH == block_els
    raw = quantize(values.reshape(-1).copy())
    path = tmp_path / name
    write_gguf(path, {'general.architecture': 'qwen2', 'general.name': 'conjugate-fixture',
                      'tokenizer.ggml.model': 'gpt2', 'tokenizer.ggml.pre': 'qwen2',
                      'tokenizer.ggml.tokens': tokens, 'tokenizer.ggml.merges': merges},
               [('token_embd.weight', (WIDTH, rows), 8, raw)])
    return path, tokens


def _packaged_fixture(tmp_path):
    """The packaged table's four pairs, on a fixture whose tokenizer can spell them."""
    document, _raw = load_pairs()
    surfaces = tuple((pair['denial'], pair['compliance']) for pair in document['pairs'])
    return _fixture(tmp_path, surfaces=surfaces)


def test_the_table_is_a_transposition_on_exactly_two_tokens_per_pair(tmp_path):
    document, _raw = load_pairs()
    artifact, tokens = _packaged_fixture(tmp_path)
    tokenizer = read_tokenizer(artifact)
    resolution = resolve_pairs(document, tokenizer)
    assert resolution['problems'] == []
    assert resolution['checks']['chosen_ids'] == 16
    assert resolution['checks']['distinct_ids'] == 16
    assert resolution['checks']['injective'] is True
    assert resolution['checks']['involution'] is True
    assert resolution['checks']['fixed_points'] == 0
    assert resolution['checks']['unmapped_ids'] == len(tokens) - 16
    assert all(item['from'] != item['to'] for item in resolution['permutation'])


def test_dry_run_reports_the_pair_table_tied_flag_and_row_cosines(tmp_path):
    artifact, _tokens = _packaged_fixture(tmp_path)
    report = vocab_conjugate(artifact)
    assert report['schema'] == SCHEMA
    assert report['removal_measured'] is False
    assert report['discovery']['tied'] is True
    assert report['permutation']['checks']['chosen_ids'] == 16  # four pairs x two contexts
    assert report['verdict']['ok'] is True
    rows = report['edit']['rows']['token_embd.weight']
    assert len(rows) == 16
    assert {row['id'] for row in rows} == set(report['permutation']['ids'])
    # a transposition preserves the row multiset, so the codec stores it exactly
    assert report['dry_run_measurement']['min_cos'] == 1.0
    assert report['tensors'][0]['storage']['cos_touched_rows'] == 1.0
    for row in rows:
        assert row['cos_before_after'] == round(row['cos_before_after'], 6)
        assert 0.0 <= row['norm_before'] and 0.0 <= row['norm_after']


def test_apply_moves_exactly_the_mapped_rows_and_reverts_byte_exactly(tmp_path):
    artifact, _tokens = _packaged_fixture(tmp_path)
    out = tmp_path / 'conjugated.gguf'
    report = vocab_conjugate(artifact, out=out, apply=True)
    assert report['apply']['untouched']['differing_bytes_outside_edited_tensors'] == 0
    identity = report['apply']['row_identity']['token_embd.weight']
    assert identity['changed_rows_equal_mapped_rows'] is True
    assert identity['every_mapped_row_holds_its_partner'] is True
    assert identity['rows_changed'] == identity['rows_mapped'] == 16
    assert identity['unchanged_rows'] == identity['rows_total'] - 16
    assert report['apply']['shipped']['min_cos'] == 1.0
    assert report['verdict']['ok'] is True
    # revert restores the source, byte for byte
    sidecar = tmp_path / 'conjugated.gguf.revert.npz'
    reverted = revert_inplace(out, sidecar)
    assert reverted['byte_exact'] is True
    assert reverted['original_sha256'] == report['artifact']['sha256']
    assert out.read_bytes() == artifact.read_bytes()


def test_a_pair_that_is_not_one_token_is_refused_not_guessed(tmp_path):
    artifact, _tokens = _fixture(tmp_path)
    table = tmp_path / 'pairs.json'
    table.write_text(json.dumps(_table((('Sorry', 'Definitely'),))))
    with pytest.raises(ValueError, match='not expressible as a row transposition'):
        vocab_conjugate(artifact, table=table)


def test_a_table_that_reuses_an_id_is_refused_as_a_non_bijection(tmp_path):
    artifact, _tokens = _fixture(tmp_path)
    table = tmp_path / 'pairs.json'
    table.write_text(json.dumps(_table((('Sorry', 'Sure'), ('Sure', 'Here')))))
    with pytest.raises(ValueError, match='not expressible as a row transposition'):
        vocab_conjugate(artifact, table=table)


def test_apply_refuses_to_clobber_and_needs_an_out(tmp_path):
    artifact, _tokens = _fixture(tmp_path)
    out = tmp_path / 'conjugated.gguf'
    out.write_bytes(b'existing')
    with pytest.raises(ValueError, match='refuses to clobber'):
        vocab_conjugate(artifact, out=out, apply=True)
    with pytest.raises(ValueError, match='needs --out'):
        vocab_conjugate(artifact, apply=True)
