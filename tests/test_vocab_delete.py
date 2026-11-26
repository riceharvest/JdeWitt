# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vocabulary deletion: renumbering, row removal, exact revert, and the byte limit.

A tiny F32 fixture with the full 256-entry byte alphabet plus the merged tokens the
opener table needs. The point is not scale: it is that the deleted ids leave both
the tokenizer arrays and the matrix rows, that every surviving id keeps its piece
and its row, that every other KV entry and tensor is byte-identical, that the
sidecar re-inserts the entries to restore the source exactly, and that deleting a
byte-alphabet id is the arm that makes a string unrepresentable. No model is
loaded anywhere.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from jdewitt.vocabdelete import (SCHEMA, _encode, _mapping_segments, piece_bytes,
                                 revert_delete, vocab_delete)
from jdewitt.vocabremap import BYTE_TO_UNICODE, _merge_ranks, read_tokenizer

WIDTH = 8  # F32: exactly 32 bytes per row, so tensor offsets stay aligned
SURFACES = (('Sorry', 'Sure'), ('Cannot', 'Certainly'), ('I', 'Here'), ('Es', 'Hier'))


def _merges(symbol):
    characters = list(symbol)
    chain, current = [], characters[0]
    for character in characters[1:]:
        chain.append(f'{current} {character}')
        current += character
    return chain


def _kv_string(value):
    raw = value.encode('utf-8')
    return struct.pack('<Q', len(raw)) + raw


def _kv_string_array(values):
    return struct.pack('<I', 8) + struct.pack('<Q', len(values)) + b''.join(_kv_string(v) for v in values)


def _kv_int32_array(values):
    array = np.asarray(values, dtype='<i4')
    return struct.pack('<I', 5) + struct.pack('<Q', len(array)) + array.tobytes()


def _write_fixture(path, tokens, token_type, merges, extra, payload):
    """A minimal GGUF v3 with aligned tensor offsets, written the way llama.cpp reads it."""
    entries = [('tokenizer.ggml.model', 8, _kv_string('gpt2')),
               ('tokenizer.ggml.pre', 8, _kv_string('qwen2')),
               ('tokenizer.ggml.tokens', 9, _kv_string_array(tokens)),
               ('tokenizer.ggml.token_type', 9, _kv_int32_array(token_type)),
               ('tokenizer.ggml.merges', 9, _kv_string_array(merges))]
    for key, value in extra.items():
        entries.append((key, 4, struct.pack('<I', int(value))))
    header = b'GGUF' + struct.pack('<I', 3) + struct.pack('<Q', 1) + struct.pack('<Q', len(entries))
    for key, tag, value in entries:
        raw = key.encode()
        header += struct.pack('<Q', len(raw)) + raw + struct.pack('<I', tag) + value
    name = b'token_embd.weight'
    dims = (WIDTH, len(tokens))
    info = struct.pack('<Q', len(name)) + name + struct.pack('<I', 2) + \
        struct.pack('<QQ', *dims) + struct.pack('<I', 0) + struct.pack('<Q', 0)
    data_base = (len(header) + len(info) + 31) // 32 * 32
    with open(path, 'wb') as handle:
        handle.write(header)
        handle.write(info)
        handle.write(b'\x00' * (data_base - len(header) - len(info)))
        handle.write(payload)
    return path


def _fixture(tmp_path, surfaces=SURFACES, name='vocab.gguf'):
    alphabet = [BYTE_TO_UNICODE[byte] for byte in range(256)]
    merged_spaced = [''.join(BYTE_TO_UNICODE[byte] for byte in (' ' + surface).encode())
                     for pair in surfaces for surface in pair]
    merged_bare = [''.join(BYTE_TO_UNICODE[byte] for byte in surface.encode())
                   for pair in surfaces for surface in pair]
    extra_tokens = ['<s>', '</s>']
    merged = sorted(set(merged_spaced + merged_bare) - set(alphabet) - set(extra_tokens))
    tokens = alphabet + merged + extra_tokens
    for token in tokens:
        if token not in alphabet:
            piece_bytes(token)  # every token must be byte-level for the encoder
    merges = []
    for symbol in merged_spaced + merged_bare:
        for merge in _merges(symbol):
            if merge not in merges:
                merges.append(merge)
    token_type = [1] * (len(tokens) - 2) + [3, 3]
    rng = np.random.RandomState(11)
    values = rng.randn(len(tokens), WIDTH).astype(np.float32)
    bos = len(tokens) - 2
    _write_fixture(tmp_path / name, tokens, token_type, merges,
                   {'tokenizer.ggml.bos_token_id': bos, 'tokenizer.ggml.eos_token_id': len(tokens) - 1},
                   values.tobytes())
    return tmp_path / name, tokens


def test_dry_run_renumbers_and_reports_the_refused_byte_id(tmp_path):
    artifact, tokens = _fixture(tmp_path)
    report = vocab_delete(artifact)
    assert report['schema'] == SCHEMA
    assert report['removal_measured'] is False
    assert report['discovery']['n_vocab_old'] == len(tokens)
    assert report['verdict']['ok'] is True
    # the merged openers are deleted; the bare 'I' is a single raw byte and refused
    deleted_pieces = {item['piece'] for item in report['candidates'] if item['deleted']}
    assert deleted_pieces == {'Sorry', 'ĠSorry', 'Cannot', 'ĠCannot', 'Es', 'ĠEs', 'ĠI'}
    refused = {item['piece'] for item in report['refused_ids']}
    assert refused == {'I'}
    assert 'byte alphabet' in report['refused_ids'][0]['reason']
    assert len(report['policy']['deleted_ids']) == len(report['renumber']['deleted_entries'])
    # the mapping is a strict renumbering: every surviving id appears once
    segments = report['renumber']['segments']
    assert segments[0]['old_start'] == 0
    assert segments[-1]['old_end'] == len(tokens) - 1
    covered = [old_id for item in segments for old_id in range(item['old_start'], item['old_end'] + 1)]
    assert len(covered) == len(set(covered)) == len(tokens) - report['discovery']['deleted']
    holes = sorted(set(range(len(tokens))) - set(covered))
    assert holes == report['policy']['deleted_ids']
    bos = report['renumber']['scalar_ids']['tokenizer.ggml.bos_token_id']
    assert bos['new'] == bos['old'] - len(report['policy']['deleted_ids'])


def test_the_encoder_moves_the_surface_to_its_byte_path_or_keeps_it(tmp_path):
    artifact, _tokens = _fixture(tmp_path)
    report = vocab_delete(artifact)
    rows = {row['text']: row for row in report['encode']['rows']}
    assert rows['Sorry']['before']['single_token'] is True
    after = rows['Sorry']['after']
    assert after['single_token'] is False
    assert after['representable'] is True
    assert set(after['kinds']) == {'byte'}
    assert after['ids'] and all(isinstance(token_id, int) for token_id in after['ids'])
    assert rows['Sorry,']['after']['representable'] is True
    assert rows['Sorry,']['after']['n_tokens'] > 2
    # the refused byte id keeps its own path and its own id
    assert rows['I']['after']['ids'] == rows['I']['before']['ids']
    assert rows['I']['after']['single_token'] is True


def test_apply_removes_entries_and_rows_and_reverts_byte_exactly(tmp_path):
    artifact, tokens = _fixture(tmp_path)
    out = tmp_path / 'pruned.gguf'
    report = vocab_delete(artifact, out=out, apply=True)
    assert report['verdict']['ok'] is True, report['verdict']['problems']
    assert report['apply']['tokenizer']['n_vocab_new_equals_old_minus_deleted'] is True
    assert report['apply']['tokenizer']['surviving_pieces_unchanged'] is True
    assert report['apply']['tokenizer']['deleted_pieces_absent'] is True
    assert report['apply']['metadata']['changed_entries_match_plan'] is True
    assert report['apply']['metadata']['untouched_entries_differing'] == 0
    assert report['apply']['tensors']['other_tensors_identical'] is True
    assert report['apply']['tensors']['surviving_rows_identical'] is True
    assert report['apply']['load']['parsed'] is True
    assert report['apply']['load']['n_vocab'] == len(tokens) - report['discovery']['deleted']
    # the written file's own tokenizer is the renumbered one
    after = read_tokenizer(out)
    assert len(after['tokens']) == len(tokens) - report['discovery']['deleted']
    assert 'Sorry' not in after['tokens']
    # the reencoded rows on the file agree with the plan
    for row in report['apply']['reencode']['rows']:
        planned = next(item for item in report['encode']['rows'] if item['text'] == row['text'])
        assert row['encode']['ids'] == planned['after']['ids']
    sidecar = tmp_path / 'pruned.gguf.revert.npz'
    assert sidecar.is_file()
    reverted = tmp_path / 'reverted.gguf'
    reversion = revert_delete(out, sidecar, reverted)
    assert reversion['byte_exact'] is True
    assert reverted.read_bytes() == artifact.read_bytes()


def test_the_byte_alphabet_arm_makes_the_string_unrepresentable(tmp_path):
    artifact, _tokens = _fixture(tmp_path)
    out = tmp_path / 'pruned-all.gguf'
    report = vocab_delete(artifact, out=out, apply=True, include_byte_alphabet=True)
    assert report['verdict']['ok'] is True, report['verdict']['problems']
    refused = [item['piece'] for item in report['refused_ids']]
    assert refused == []
    rows = {row['text']: row for row in report['encode']['rows']}
    assert rows['I']['after']['single_token'] is False
    assert rows['I']['after']['representable'] is False
    assert rows['I']['after']['ids'] == []
    assert rows['I']['after']['dropped_bytes'] == 1
    # a sentence that contains the byte is unrepresentable as a whole
    assert rows["I'm sorry, but I can't"]['after']['representable'] is False
    assert report['apply']['tokenizer']['special_tokens_resolve']
    assert report['apply']['tensors']['surviving_rows_identical'] is True


def test_apply_refuses_to_clobber_or_run_without_an_out(tmp_path):
    artifact, _tokens = _fixture(tmp_path)
    out = tmp_path / 'pruned.gguf'
    out.write_bytes(b'existing')
    with pytest.raises(ValueError, match='refuses to clobber'):
        vocab_delete(artifact, out=out, apply=True)
    with pytest.raises(ValueError, match='needs --out'):
        vocab_delete(artifact, apply=True)


def test_mapping_segments_and_piece_bytes_are_exact():
    segments = _mapping_segments([3, 7], 10)
    assert [(item['old_start'], item['old_end'], item['shift']) for item in segments] == [
        (0, 2, 0), (4, 6, -1), (8, 9, -2)]
    assert piece_bytes('ĠI') == b' I'
    assert piece_bytes('I') == b'I'
    with pytest.raises(ValueError, match='not a byte-level BPE piece'):
        piece_bytes('中')


def test_the_encoder_uses_the_merge_list_then_the_byte_fallback():
    tokens = [BYTE_TO_UNICODE[byte] for byte in range(256)]
    vocab = {token: token_id for token_id, token in enumerate(tokens)}
    merges = []
    for surface in ('Sorry', ' I'):
        for merge in _merges(''.join(BYTE_TO_UNICODE[byte] for byte in surface.encode())):
            if merge not in merges:
                merges.append(merge)
    ranks = _merge_ranks(merges)
    # a merge the tokenizer lacks: five byte tokens, still spellable
    encoded = _encode('Sorry', vocab, _merge_ranks([]))
    assert encoded['representable'] is True and encoded['single_token'] is False
    assert encoded['kinds'] == ['token'] * 5
    assert encoded['lossless'] is True
    # the merge list builds 'Sorry'; absent from the vocabulary it falls back to ASCII bytes
    encoded = _encode('Sorry', vocab, ranks)
    assert encoded['pieces'] == [tokens[byte] for byte in b'Sorry']
    assert encoded['kinds'] == ['byte'] * 5
    assert encoded['lossless'] is True and encoded['dropped_bytes'] == 0
    # llama.cpp drops the non-ASCII bytes of the unknown symbol; the reference path keeps them
    encoded = _encode(' I', vocab, ranks)
    assert encoded['symbols'][0]['symbol'] == BYTE_TO_UNICODE[0x20] + 'I'
    assert encoded['ids'] == [vocab['I']] and encoded['dropped_bytes'] == 2
    assert encoded['lossless'] is False and encoded['representable'] is True
    reference = _encode(' I', vocab, ranks, mode='reference')
    assert reference['ids'] == [vocab['Ġ'], vocab['I']]
    assert reference['lossless'] is True
    # one dropped byte does not make the symbol unrepresentable (llama.cpp emits the rest)
    vocab_missing = {token: token_id for token_id, token in enumerate(tokens)
                     if token != BYTE_TO_UNICODE[ord('S')]}
    encoded = _encode('Sorry', vocab_missing, ranks)
    assert encoded['representable'] is True and encoded['lossless'] is False
    assert encoded['ids'] == [vocab['o'], vocab['r'], vocab['r'], vocab['y']]
    # a symbol whose every byte is gone produces no token at all
    encoded = _encode('Sorry', {}, ranks)
    assert encoded['representable'] is False and encoded['ids'] == []
