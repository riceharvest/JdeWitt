# SPDX-License-Identifier: AGPL-3.0-or-later
"""The centroid rewrite: group resolution, the three targets, storage and revert.

A tiny Q8_0 fixture with a hand-built byte-level BPE tokenizer. The point is not
scale: it is that only denial rows are written, that the compliance rows stay the
centroid and the control, that each mode's algebra is the one it reports, and that
the pre-edit bytes restore the file. No model is loaded anywhere.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from jdewitt.ggufio import revert_inplace, write_gguf
from jdewitt.quants import CODECS
from jdewitt.vocabconjugate import load_pairs
from jdewitt.vocabcentroid import MODES, SCHEMA, resolve_groups, vocab_centroid
from jdewitt.vocabremap import _byte_level, read_tokenizer

WIDTH = 32  # exactly one Q8_0 block per row


def _merges(symbol):
    characters = list(symbol)
    chain, current = [], characters[0]
    for character in characters[1:]:
        chain.append(f'{current} {character}')
        current += character
    return chain


def _fixture(tmp_path, surfaces=None):
    if surfaces is None:
        document, _raw = load_pairs()
        surfaces = [(pair['denial'], pair['compliance']) for pair in document['pairs']]
    spaced = [_byte_level(' ' + surface) for pair in surfaces for surface in pair]
    bare = [_byte_level(surface) for pair in surfaces for surface in pair]
    merges = []
    for symbol in spaced + bare:
        for merge in _merges(symbol):
            if merge not in merges:
                merges.append(merge)
    tokens = sorted(set(spaced + bare))
    rows = len(tokens)
    rng = np.random.RandomState(11)
    values = rng.randn(rows, WIDTH).astype(np.float32)
    block_bytes, block_els, quantize, _decode = CODECS['Q8_0']
    assert WIDTH == block_els
    raw = quantize(values.reshape(-1).copy())
    path = tmp_path / 'm.gguf'
    write_gguf(path, {'general.architecture': 'qwen2', 'general.name': 'centroid-fixture',
                      'tokenizer.ggml.model': 'gpt2', 'tokenizer.ggml.pre': 'qwen2',
                      'tokenizer.ggml.tokens': tokens, 'tokenizer.ggml.merges': merges},
               [('token_embd.weight', (WIDTH, rows), 8, raw)])
    return path, tokens


def test_the_groups_are_the_denial_targets_and_the_compliance_centroid(tmp_path):
    artifact, _tokens = _fixture(tmp_path)
    document, _raw = load_pairs()
    groups = resolve_groups(document, read_tokenizer(artifact))
    assert groups['ok'] is True
    assert groups['n_denial'] == groups['n_compliance'] == 8  # four pairs x two contexts
    assert not set(groups['denial_ids']) & set(groups['compliance_ids'])
    every = groups['denial_ids'] + groups['compliance_ids']
    assert len(every) == len(set(every)) == 16
    assert all(len(groups['rows'][token_id]) == 1 for token_id in groups['denial_ids'])


def test_centroid_mode_writes_the_group_mean_and_leaves_the_reference_rows(tmp_path):
    artifact, _tokens = _fixture(tmp_path)
    out = tmp_path / 'centroid.gguf'
    report = vocab_centroid(artifact, mode='centroid', out=out, apply=True)
    assert report['schema'] == SCHEMA
    assert report['removal_measured'] is False
    assert report['discovery']['tied'] is True
    assert report['verdict']['ok'] is True
    rows = report['edit']['rows']['token_embd.weight']
    assert {row['id'] for row in rows} == set(report['groups']['denial_ids'])
    for row in rows:
        assert row['cos_after_centroid'] == 1.0
        assert row['norm_after'] == report['tensors'][0]['groups']['centroid_norm']
        assert 0.0 <= row['norm_before']
    identity = report['apply']['row_identity']['token_embd.weight']
    assert identity['changed_rows_equal_mapped_rows'] is True
    assert identity['reference_rows_changed'] == []
    assert identity['differing_rows_outside_map'] == []
    assert identity['rows_changed'] == 8
    assert identity['realized_cos_min'] >= 0.95
    assert report['apply']['untouched']['differing_bytes_outside_edited_tensors'] == 0
    assert report['apply']['bytes_changed']['differing_bytes'] > 0
    stored = report['apply']['stored_cosines']['token_embd.weight']
    assert stored['min_centroid'] >= 0.99
    reverted = revert_inplace(out, tmp_path / 'centroid.gguf.revert.npz')
    assert reverted['byte_exact'] is True
    assert out.read_bytes() == artifact.read_bytes()


def test_orthogonal_modes_remove_the_denial_axis(tmp_path):
    artifact, _tokens = _fixture(tmp_path)
    for mode in ('centroid-orthogonal', 'row-orthogonal'):
        out = tmp_path / f'{mode}.gguf'
        report = vocab_centroid(artifact, mode=mode, out=out, apply=True)
        assert report['verdict']['ok'] is True
        rows = report['edit']['rows']['token_embd.weight']
        for row in rows:
            assert abs(row['cos_after_axis']) < 1e-5, (mode, row)
            assert row['norm_after'] <= row['norm_before'] + 1e-6
            if mode == 'row-orthogonal':
                # removing the row's own projection loses exactly |cos|*norm
                assert abs(row['delta_norm'] - abs(row['cos_before_axis']) * row['norm_before']) < 1e-3
        if mode == 'centroid-orthogonal':
            # one target: the centroid with its axis component removed
            assert len({round(row['norm_after'], 6) for row in rows}) == 1
            assert len({round(row['cos_after_centroid'], 6) for row in rows}) == 1
        stored = report['apply']['stored_cosines']['token_embd.weight']
        assert stored['max_abs_axis'] < 1e-2
        assert report['verdict']['ok'] is True
        revert_inplace(out, tmp_path / f'{mode}.gguf.revert.npz')
        assert out.read_bytes() == artifact.read_bytes()


def test_a_table_that_makes_an_id_both_target_and_reference_is_refused(tmp_path):
    artifact, _tokens = _fixture(tmp_path, surfaces=[('Sorry', 'Sure'), ('Sure', 'Here')])
    table = tmp_path / 'pairs.json'
    table.write_text(json.dumps({'schema': 'jdewitt.vocab-conjugate-pairs/1',
                                 'pairs': [{'id': 'a', 'language': 'en', 'denial': 'Sorry',
                                            'compliance': 'Sure'},
                                           {'id': 'b', 'language': 'en', 'denial': 'Sure',
                                            'compliance': 'Here'}]}))
    with pytest.raises(ValueError, match='not expressible'):
        vocab_centroid(artifact, table=table)


def test_an_unknown_mode_is_refused(tmp_path):
    artifact, _tokens = _fixture(tmp_path)
    with pytest.raises(ValueError, match='unknown --mode'):
        vocab_centroid(artifact, mode='mean', out=tmp_path / 'x.gguf', apply=True)
    assert set(MODES) == {'centroid', 'centroid-orthogonal', 'row-orthogonal'}


def test_apply_refuses_to_clobber_and_needs_an_out(tmp_path):
    artifact, _tokens = _fixture(tmp_path)
    out = tmp_path / 'centroid.gguf'
    out.write_bytes(b'existing')
    with pytest.raises(ValueError, match='refuses to clobber'):
        vocab_centroid(artifact, out=out, apply=True)
    with pytest.raises(ValueError, match='needs --out'):
        vocab_centroid(artifact, apply=True)
