# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Space bank loader: geometry from the artifact, typed refusals, no torch."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from jdewitt.bank import SCHEMA, _decodable, _projected_layers, bank_check, load_bank, resolve
from jdewitt.errors import BankRefused, error_code

FIXTURES = Path(__file__).resolve().parents[1] / 'verification' / 'space' / 'fixtures'
GGUF = FIXTURES / 'fixture-2layer-f16.gguf'
BANK = FIXTURES / 'bank'
BAD_DIGEST_BANK = FIXTURES / 'bank-bad-digest'


def _bank_copy(tmp_path, mutate=None, bundle='fixture-bank.tar'):
    """A writable copy of the fixture bank, optionally with one entry mutated."""
    target = tmp_path / 'bank'
    target.mkdir()
    document = json.loads((BANK / 'bank.json').read_text())
    if mutate is not None:
        mutate(document['entries'][0])
    (target / 'bank.json').write_text(json.dumps(document))
    shutil.copy(BANK / bundle, target / bundle)
    return target


def _code(bank, gguf=GGUF):
    with pytest.raises(BankRefused) as caught:
        resolve(bank, gguf)
    return caught.value.code, str(caught.value)


def test_fixture_resolves_target_names_layers_strength_and_edit_types():
    report = bank_check(BANK, GGUF)
    assert report['schema'] == SCHEMA
    assert report['artifact']['architecture'] == 'bankfixture'
    assert report['artifact']['layers'] == 2
    assert report['artifact']['hidden'] == 256
    assert report['layers'] == [0, 1]
    assert report['targets'] == {
        'attn': ['blk.0.attn_output.weight', 'blk.1.attn_output.weight'],
        'mlp': ['blk.0.ffn_down.weight', 'blk.1.ffn_down.weight'],
    }
    assert report['target_names'] == [
        'blk.0.attn_output.weight', 'blk.0.ffn_down.weight',
        'blk.1.attn_output.weight', 'blk.1.ffn_down.weight',
    ]
    assert report['strength'] == {'attn': 1.5, 'mlp': 1.5}
    assert report['edit_types'] == ['q8_0', 'f16']
    assert report['bundle']['verified'] is True
    assert report['refusal'] is None


def test_hidden_wide_query_projection_is_not_a_target():
    """A 256-wide attn_q is discovered as width-matching but never edited (§5.1)."""
    from jdewitt.ggufgeom import (discover_writeback_targets, gguf_weight_shapes,
                                  inferred_hidden_width)

    report = bank_check(BANK, GGUF)
    assert not any('attn_q' in name for name in report['target_names'])
    shapes = gguf_weight_shapes(GGUF)
    discovered = discover_writeback_targets(shapes, inferred_hidden_width(shapes))
    assert 'blk.0.attn_q.weight' in discovered['unmapped_writeback_width']


def test_bank_miss_names_the_architecture(tmp_path):
    (tmp_path / 'bank.json').write_text(json.dumps({'schema': SCHEMA, 'entries': []}))
    code, message = _code(tmp_path)
    assert code == 'BANK_MISS'
    assert 'bankfixture' in message


def test_geometry_hidden_mismatch(tmp_path):
    bank = _bank_copy(tmp_path, lambda entry: entry['geometry'].__setitem__('hidden', 512))
    code, message = _code(bank)
    assert code == 'GEOMETRY_MISMATCH'
    assert '512' in message and '256' in message


def test_geometry_mismatch_names_missing_layers(tmp_path):
    def widen(entry):
        entry['footprint'] = {'first': 0, 'last': 3, 'fraction': [0.0, 1.0]}
    bank = _bank_copy(tmp_path, widen)
    code, message = _code(bank)
    assert code == 'GEOMETRY_MISMATCH'
    assert 'layer 2' in message and 'layer 3' in message
    assert 'attn+mlp' in message


def test_unsupported_quant_names_the_type(tmp_path, monkeypatch):
    import jdewitt.bank as bank_module
    monkeypatch.setattr(bank_module, '_decodable', lambda name: False)
    code, message = _code(BANK)
    assert code == 'UNSUPPORTED_QUANT'
    assert 'F16' in message


def test_declared_unverified_types_are_not_decodable():
    assert _decodable('F16') is True
    assert _decodable('Q4_K') is True
    assert _decodable('NOT_A_REAL_TYPE') is False


def test_bad_bundle_digest_is_refused_before_the_axis(tmp_path):
    code, message = _code(BAD_DIGEST_BANK)
    assert code == 'BUNDLE_BAD_DIGEST'
    assert 'fixture-bank-corrupt.tar' in message


def test_missing_bundle_is_a_bad_digest(tmp_path):
    bank = _bank_copy(tmp_path)
    (bank / 'fixture-bank.tar').unlink()
    code, message = _code(bank)
    assert code == 'BUNDLE_BAD_DIGEST'
    assert 'missing' in message


def test_projected_footprint_reproduces_the_roadmap_example():
    entry = {'key': 'qwen3', 'geometry': {'layers': 36},
             'footprint': {'first': 15, 'last': 35, 'fraction': [0.4167, 0.9722]}}
    layers, projected = _projected_layers(entry, 36)
    assert layers == list(range(15, 36))
    assert projected is False
    layers, projected = _projected_layers(entry, 18)
    assert projected is True
    assert layers == list(range(round(0.4167 * 18), round(0.9722 * 18) + 1))


def test_projected_footprint_clamps_to_whole_layers():
    entry = {'key': 'k', 'geometry': {'layers': 36},
             'footprint': {'first': 15, 'last': 35, 'fraction': [0.95, 1.05]}}
    layers, projected = _projected_layers(entry, 4)
    assert projected is True
    assert layers == [3]


def test_bank_loads_and_is_empty_bank_is_valid(tmp_path):
    (tmp_path / 'bank.json').write_text(json.dumps({'schema': SCHEMA, 'entries': []}))
    assert load_bank(tmp_path)['entries'] == []
    with pytest.raises(ValueError, match='schema'):
        (tmp_path / 'bank.json').write_text(json.dumps({'schema': 'other', 'entries': []}))
        load_bank(tmp_path)


def test_bundle_file_must_be_a_plain_name(tmp_path):
    def absolute(entry):
        entry['bundle']['file'] = '/etc/passwd'
    bank = _bank_copy(tmp_path, absolute)
    with pytest.raises(ValueError, match='plain file name'):
        load_bank(bank)


def test_error_code_is_the_refusal_code():
    assert error_code(BankRefused('BANK_MISS', 'no entry')) == 'BANK_MISS'
    assert error_code(BankRefused('GEOMETRY_MISMATCH', 'x')) == 'GEOMETRY_MISMATCH'
    assert error_code(BankRefused('UNSUPPORTED_QUANT', 'x')) == 'UNSUPPORTED_QUANT'
    assert error_code(BankRefused('BUNDLE_BAD_DIGEST', 'x')) == 'BUNDLE_BAD_DIGEST'


def test_importing_the_bank_path_does_not_import_torch():
    result = subprocess.run(
        [sys.executable, '-c',
         "import jdewitt.bank, sys; assert 'torch' not in sys.modules, 'torch imported'"],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stderr
