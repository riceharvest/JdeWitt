# SPDX-License-Identifier: AGPL-3.0-or-later
"""bank-apply: their GGUF in, an edited GGUF out, one JSON, no model in the process."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from jdewitt.bank_apply import bank_apply
from jdewitt.errors import BankApplyRefused, BankRefused, error_code

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'verification' / 'space' / 'fixtures'
BANK = FIXTURES / 'bank-apply'
GGUF = BANK / 'fixture-2layer-q8_0.gguf'
ENTRY = 'applyfixture/f00l01-q8'
TARGETS = ('blk.0.attn_output.weight', 'blk.0.ffn_down.weight',
           'blk.1.attn_output.weight', 'blk.1.ffn_down.weight')
UNTOUCHED = ('blk.0.attn_q.weight', 'blk.1.attn_q.weight')


def _bank_copy(tmp_path, mutate=None):
    """A writable copy of the fixture bank, optionally with the entry mutated."""
    target = tmp_path / 'bank'
    target.mkdir()
    document = json.loads((BANK / 'bank.json').read_text())
    if mutate is not None:
        mutate(document['entries'][0])
    (target / 'bank.json').write_text(json.dumps(document))
    shutil.copy(BANK / 'fixture-apply-plan.tar', target / 'fixture-apply-plan.tar')
    return target


def _raw_tensor(path, name):
    """The stored bytes of one tensor, read from the file itself."""
    from gguf import GGUFReader

    tensor = next(item for item in GGUFReader(str(path)).tensors if item.name == name)
    with open(path, 'rb') as handle:
        handle.seek(int(tensor.data_offset))
        return handle.read(int(tensor.n_bytes))


def test_fixture_delivers_the_floor_and_reports_codec_fidelity_not_removal(tmp_path):
    out = tmp_path / 'edited.gguf'
    report = bank_apply(BANK, ENTRY, GGUF, out)
    assert out.is_file()
    assert report['command'] == 'bank-apply'
    assert report['entry_key'] == ENTRY
    assert report['bundle_sha256'] == json.loads((BANK / 'bank.json').read_text())[
        'entries'][0]['bundle']['sha256']
    assert report['base'] == {'repo': 'jdewitt/bank-apply-fixture', 'revision': 'fixture'}
    assert report['entry']['measurement']['state'] == 'banked_unmeasured'
    # q8_0 is probed first and misses the entry's 0.999 floor; the ladder steps to f16.
    assert [row['edit_type'] for row in report['edit_type']['readings']] == ['Q8_0', 'F16']
    assert report['edit_type']['chosen'] == 'F16'
    assert report['edit_type']['floor_cleared'] is True
    assert report['edited']['count'] == 4
    assert report['edited']['min_cos'] >= 0.999
    assert report['edited']['floor_met'] is True
    assert report['edited']['below_floor'] == []
    assert {row['name'] for row in report['edited']['tensors']} == set(TARGETS)
    # per-kind strength (§5.1): the delivered rows carry DIFFERENT intended strengths.
    strengths = {row['kind']: row['intended_strength'] for row in report['edited']['tensors']}
    assert strengths == {'attn': 1.5, 'mlp': 0.75}
    assert report['fingerprint']['scale_per_tensor']['blk.0.attn_output.weight'] == 1.5
    assert report['fingerprint']['scale_per_tensor']['blk.0.ffn_down.weight'] == 0.75
    assert report['untouched']['count'] == 2
    assert report['untouched']['verified'] is True
    assert report['untouched']['mismatched'] == []
    assert report['bytes_in'] > 0 and report['bytes_out'] > 0
    assert report['wall_seconds'] >= 0
    assert report['removal_measured'] is False
    assert 'not measured' in report['claims']['removal']


def test_untouched_tensors_are_byte_identical_in_the_written_file(tmp_path):
    out = tmp_path / 'edited.gguf'
    report = bank_apply(BANK, ENTRY, GGUF, out)
    assert report['untouched']['verified'] is True
    for name in UNTOUCHED:
        assert _raw_tensor(out, name) == _raw_tensor(GGUF, name), name
    for name in TARGETS:
        assert _raw_tensor(out, name) != _raw_tensor(GGUF, name), name


def test_second_run_writes_the_same_sha256(tmp_path):
    first = bank_apply(BANK, ENTRY, GGUF, tmp_path / 'a.gguf')
    second = bank_apply(BANK, ENTRY, GGUF, tmp_path / 'b.gguf')
    assert first['output_sha256'] == second['output_sha256']
    assert first['edited']['count'] == second['edited']['count']
    assert first['untouched']['count'] == second['untouched']['count']


def test_geometry_mismatch_refuses_before_writing_a_byte(tmp_path):
    bank = _bank_copy(tmp_path, lambda entry: entry['geometry'].__setitem__('hidden', 128))
    out = tmp_path / 'edited.gguf'
    with pytest.raises(BankRefused) as caught:
        bank_apply(bank, ENTRY, GGUF, out)
    assert caught.value.code == 'GEOMETRY_MISMATCH'
    assert not out.exists()


def test_footprint_outside_the_basis_rows_refuses_before_writing(tmp_path):
    def widen(entry):
        entry['footprint'] = {'first': 0, 'last': 3, 'fraction': [0.0, 1.0]}
    bank = _bank_copy(tmp_path, widen)
    out = tmp_path / 'edited.gguf'
    with pytest.raises(BankRefused) as caught:
        bank_apply(bank, ENTRY, GGUF, out)
    assert caught.value.code == 'GEOMETRY_MISMATCH'
    assert not out.exists()


def test_codec_floor_refuses_and_names_the_worst_tensor(tmp_path):
    def strip(entry):
        entry['edit']['types'] = ['q8_0']
        entry['edit']['min_cos'] = 1.0
    bank = _bank_copy(tmp_path, strip)
    out = tmp_path / 'edited.gguf'
    with pytest.raises(BankApplyRefused) as caught:
        bank_apply(bank, ENTRY, GGUF, out)
    assert caught.value.code == 'CODEC_FLOOR'
    assert 'blk.0.ffn_down.weight' in str(caught.value)
    assert error_code(caught.value) == 'CODEC_FLOOR'
    assert not out.exists()


def test_edit_type_outside_the_entry_is_refused(tmp_path):
    out = tmp_path / 'edited.gguf'
    with pytest.raises(BankApplyRefused) as caught:
        bank_apply(BANK, ENTRY, GGUF, out, edit_type='q6_k')
    assert caught.value.code == 'EDIT_TYPE_NOT_ALLOWED'
    assert not out.exists()


def test_explicit_type_inside_the_entry_is_written_without_a_probe(tmp_path):
    out = tmp_path / 'edited.gguf'
    report = bank_apply(BANK, ENTRY, GGUF, out, edit_type='q8_0')
    assert report['edit_type']['mode'] == 'explicit'
    assert report['edit_type']['chosen'] == 'Q8_0'
    assert report['edit_type']['readings'] == []
    assert report['edited']['count'] == 4
    assert out.is_file()


def test_unknown_entry_key_is_a_bank_miss(tmp_path):
    out = tmp_path / 'edited.gguf'
    with pytest.raises(BankRefused) as caught:
        bank_apply(BANK, 'applyfixture/not-there', GGUF, out)
    assert caught.value.code == 'BANK_MISS'
    assert 'applyfixture/f00l01-q8' in str(caught.value)
    assert not out.exists()


def test_bundle_digest_mismatch_refuses_before_writing(tmp_path):
    bank = _bank_copy(tmp_path)
    bundle = bank / 'fixture-apply-plan.tar'
    raw = bytearray(bundle.read_bytes())
    raw[-1] ^= 0xFF
    bundle.write_bytes(bytes(raw))
    out = tmp_path / 'edited.gguf'
    with pytest.raises(BankRefused) as caught:
        bank_apply(bank, ENTRY, GGUF, out)
    assert caught.value.code == 'BUNDLE_BAD_DIGEST'
    assert not out.exists()


def test_out_cannot_be_the_input(tmp_path):
    with pytest.raises(BankApplyRefused) as caught:
        bank_apply(BANK, ENTRY, GGUF, GGUF)
    assert caught.value.code == 'OUT_IS_INPUT'


def test_strength_scale_multiplies_the_entry_per_kind(tmp_path):
    out = tmp_path / 'edited.gguf'
    report = bank_apply(BANK, ENTRY, GGUF, out, strength_scale=0.5)
    assert report['entry']['strength'] == {'attn': 0.75, 'mlp': 0.375}
    assert {row['intended_strength'] for row in report['edited']['tensors']} == {0.75, 0.375}


def test_projected_footprint_is_placed_on_the_entry_rows_end_to_end():
    """§5.1: a projected footprint spans the entry's own footprint, endpoints kept."""
    from jdewitt.bank_apply import _row_by_layer

    # The roadmap example: a 36-layer entry footprint 15-35 on an 18-layer artifact.
    projected = _row_by_layer({'footprint': {'first': 15, 'last': 35},
                               'layers': list(range(8, 18)), 'entry': {'key': 'k'}})
    assert projected[8] == 15 and projected[17] == 35
    assert list(projected) == list(range(8, 18))
    # Same depth -> identity, no drift.
    identity = _row_by_layer({'footprint': {'first': 10, 'last': 12},
                              'layers': [10, 11, 12], 'entry': {'key': 'k'}})
    assert identity == {10: 10, 11: 11, 12: 12}
    # A single-layer footprint lands on the entry's first row, never outside it.
    single = _row_by_layer({'footprint': {'first': 34, 'last': 35},
                            'layers': [3], 'entry': {'key': 'k'}})
    assert single == {3: 34}


def test_reapplying_an_entry_to_its_own_output_is_a_byte_copy_no_op(tmp_path):
    """cos cannot see a double application, so the second run must not write one."""
    first = bank_apply(BANK, ENTRY, GGUF, tmp_path / 'a.gguf')
    again = bank_apply(BANK, ENTRY, tmp_path / 'a.gguf', tmp_path / 'b.gguf')
    assert again['applied'] is False
    assert again['already_applied'] is True
    assert again['edited']['count'] == 0
    assert again['output_sha256'] == first['output_sha256']
    assert again['output_sha256'] == again['input_sha256']
    assert (tmp_path / 'b.gguf').read_bytes() == (tmp_path / 'a.gguf').read_bytes()


def test_applying_the_entry_imports_no_tokenizer_or_transformers(tmp_path):
    """The Space runs this with no model server, no transformers and no tokenizer.

    gguf-py imports the ``sentencepiece`` PYTHON package on import (it is a vocab
    reader), but nothing here builds a tokenizer, loads a tokenizer model or touches
    transformers; the assertion is on the modules that would mean a real one is in
    the process.
    """
    script = (
        'import sys\n'
        'from pathlib import Path\n'
        'from jdewitt.bank_apply import bank_apply\n'
        f'bank_apply(Path({str(BANK)!r}), {ENTRY!r}, Path({str(GGUF)!r}), '
        f'Path({str(tmp_path / "out.gguf")!r}))\n'
        "for name in ('transformers', 'tokenizers'):\n"
        '    assert name not in sys.modules, name\n'
    )
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                            cwd=ROOT)
    assert result.returncode == 0, result.stderr
