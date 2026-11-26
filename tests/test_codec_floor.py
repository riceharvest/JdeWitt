"""Static checks for the codec-floor experiment: algebra, codecs, bytes.

The committed report is checked too, but these tests are the ones that would catch a
regression: the subspace must be the row space of the embedding rows, the delivered cos
must be the cos of the aligned block rather than of the whole tensor, a coarse grid must
be shown to be indistinguishable from a random subspace of the same rank, and a written
file must keep every untouched tensor byte-identical and revert byte-exactly.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from jdewitt.codec_floor import (GgufBpe, aggregate, alignment_curve, byte_level_alphabet,
                                 denial_id_sets, prepare_alignment, prepared_curve,
                                 quantization_error, random_subspace, row_block_ok, roundtrip,
                                 subspace, verify_written)
from jdewitt.ggufio import revert_inplace, save_whole_file_revert, write_gguf
from jdewitt.ggufwrite import parse_container, write_container
from jdewitt.quants import type_id_for_name

REPORT = Path(__file__).resolve().parents[1] / 'verification/codec-floor/report.json'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ vocabulary and ids
def test_byte_level_alphabet_roundtrips_every_byte():
    to_escaped, from_escaped = byte_level_alphabet()
    assert len(to_escaped) == 256 and len(from_escaped) == 256
    for value in range(256):
        assert from_escaped[to_escaped[value]] == value


def _tiny_tokenizer(words, pattern=r'\s?\w+|\s+'):
    """A vocabulary that contains every byte character plus each word's merge chain.

    A byte-level BPE vocabulary always carries all 256 byte tokens, and a word becomes a
    single id only if every intermediate merge exists. Both are built from the words the
    test needs, so the tokenizer behaves like the artifact's own rather than like a toy
    that cannot encode anything.
    """
    to_escaped, _from_escaped = byte_level_alphabet()
    tokens = [to_escaped[value] for value in range(256)]
    merges, seen = [], set()

    def chain(escaped):
        for cut in range(1, len(escaped)):
            pair = (escaped[:cut], escaped[cut:cut + 1])
            if pair not in seen:
                seen.add(pair)
                merges.append(pair)

    for word in words:
        escaped = ''.join(to_escaped[byte] for byte in word.encode('utf-8'))
        chain(escaped)
        if escaped not in tokens:
            tokens.append(escaped)
    return GgufBpe(tokens, [' '.join(pair) for pair in merges], pattern=pattern)


def test_bpe_uses_the_supplied_vocab_and_merges():
    tokenizer = _tiny_tokenizer(['ab', ' a', ' ab', 'I am'])
    assert tokenizer.decode(tokenizer.encode('ab')) == 'ab'
    assert tokenizer.decode(tokenizer.encode(' ab')) == ' ab'
    assert tokenizer.decode(tokenizer.encode('I am')) == 'I am'
    # a merge chain that exists in the vocabulary is one id, not four
    assert len(tokenizer.encode('ab')) == 1
    assert tokenizer.piece(tokenizer.encode(' a')[0]) == ' a'
    # and a string whose pieces have no chain still round-trips through several ids
    assert len(tokenizer.encode(' aba')) >= 1


def test_denial_id_sets_cover_every_surface_and_report_skips():
    tokenizer = _tiny_tokenizer(['I', ' cannot', ' sorry', ' do', ' not', ' promote',
                                 ' provide'])
    surfaces = [{'id': 'en-i', 'form': 'I', 'kind': 'refusal_opener'},
                {'id': 'en-cannot', 'form': ' cannot', 'kind': 'denial_word'},
                {'id': 'en-do-not', 'form': ' do not', 'kind': 'refusal_opener'},
                {'id': 'en-three', 'form': ' do not promote', 'kind': 'refusal_opener'}]
    sets = denial_id_sets(tokenizer, surfaces)
    by_surface = {row['id']: row for row in sets['surfaces']}
    assert by_surface['en-i']['single_token'] is True
    assert by_surface['en-cannot']['single_token'] is True
    assert by_surface['en-do-not']['single_token'] is False
    assert len(by_surface['en-do-not']['ids']) == 2
    for row in sets['surfaces']:
        assert set(row['ids']) <= set(sets['surface_span'])
        assert row['ids'][0] in sets['opener_span']
    # openers are deduplicated: two surfaces can open with the same token
    assert sets['opener_span'] == list(dict.fromkeys(row['ids'][0] for row in sets['surfaces']))
    assert sets['skipped_openers'] == []
    assert len(sets['surface_span']) == len(set(sets['surface_span']))


def test_subspace_is_the_row_space_and_is_orthonormal():
    generator = np.random.default_rng(3)
    rows = generator.normal(size=(6, 40))
    block = subspace(rows)
    v = block['V']
    assert v.shape == (40, 6)
    assert np.allclose(v.T @ v, np.eye(6), atol=1e-10)
    # every row lies in the span: no residual
    assert np.allclose(rows - (rows @ v) @ v.T, 0.0, atol=1e-9)
    outside = generator.normal(size=40)
    assert np.linalg.norm(outside - (outside @ v) @ v.T) > 0.1
    assert 0.0 < block['top1_energy_share'] <= 1.0
    assert block['rank'] == 6


def test_subspace_rank_drops_when_a_row_is_duplicated():
    generator = np.random.default_rng(5)
    rows = generator.normal(size=(4, 32))
    stacked = np.concatenate([rows, rows[:2]], axis=0)
    assert subspace(stacked)['rank'] == 4


def test_random_subspace_is_orthonormal_and_seeded():
    first = random_subspace(64, 5, seed=0)
    again = random_subspace(64, 5, seed=0)
    assert np.array_equal(first, again)
    assert np.allclose(first.T @ first, np.eye(5), atol=1e-12)
    assert np.linalg.norm(first.T @ random_subspace(64, 5, seed=1)) < 5.0


# ------------------------------------------------------------------------ the measurement
def test_aligned_curve_of_an_unchanged_tensor_is_the_identity():
    generator = np.random.default_rng(7)
    weight = generator.normal(size=(48, 96)).astype(np.float32)
    basis = random_subspace(48, 4, seed=2)
    curve = alignment_curve(weight, basis, weight, ks=(1, 2, 4))
    expected = float(np.sum((basis.T.astype(np.float64) @ weight) ** 2))
    total = float(np.sum(weight.astype(np.float64) ** 2))
    for k, row in curve['curves'].items():
        assert np.isclose(row['cos'], 1.0, atol=1e-6)
        assert np.isclose(row['surviving_energy_fraction'], 1.0, atol=1e-6)
        assert np.isclose(row['removed_energy_fraction'], 0.0, atol=1e-6)
        assert np.isclose(row['isotropic_fraction'], int(k) / 48)
    assert np.isclose(curve['curves']['4']['aligned_energy'], expected)
    assert np.isclose(curve['curves']['4']['aligned_energy_fraction'], expected / total)


def test_curve_is_a_prefix_energy_and_detects_a_deleted_component():
    generator = np.random.default_rng(11)
    coefficients = generator.normal(size=(3, 10))
    basis = random_subspace(20, 3, seed=4)
    weight = (basis @ coefficients).astype(np.float32)  # all energy inside the subspace
    gone = np.zeros_like(weight)
    curve = alignment_curve(weight, basis, gone, ks=(1, 2, 3))
    assert np.isclose(curve['curves']['3']['aligned_energy_fraction'], 1.0, atol=1e-6)
    assert np.isclose(curve['curves']['3']['cos'], 0.0, atol=1e-6)
    assert np.isclose(curve['curves']['3']['removed_energy_fraction'], 1.0, atol=1e-6)
    energies = [curve['curves'][str(k)]['aligned_energy'] for k in (1, 2, 3)]
    assert energies[0] < energies[1] < energies[2]


def test_prepared_alignment_splits_the_random_control():
    generator = np.random.default_rng(13)
    weight = generator.normal(size=(32, 64)).astype(np.float32)
    basis = random_subspace(32, 3, seed=5)
    control = random_subspace(32, 3, seed=6)
    prepared = prepare_alignment(weight, basis, control)
    assert prepared['rank'] == 3 and prepared['random_rank'] == 3
    measured, random_curves = prepared_curve(prepared, weight, ks=(1, 2))
    assert measured['rank'] == 3 and random_curves['rank'] == 3
    # the measured block equals a standalone measurement of the same basis
    standalone = alignment_curve(weight, basis, weight, ks=(1, 2, 3))
    assert np.isclose(measured['curves']['3']['aligned_energy'],
                      standalone['curves']['3']['aligned_energy'])


def test_quantization_error_is_the_energy_fraction_of_the_residual():
    weight = np.array([[1.0, -2.0], [0.5, 0.0]], dtype=np.float32)
    quantized = np.array([[1.0, -2.0], [0.5, 1.0]], dtype=np.float32)
    error = quantization_error(weight, quantized)
    assert np.isclose(error['noise_energy'], 1.0)
    assert np.isclose(error['relative_noise_energy'], 1.0 / 5.25)
    assert np.isclose(error['rms_ratio'], (1.0 / 5.25) ** 0.5)


def test_coarse_grids_lose_more_energy_than_fine_ones():
    generator = np.random.default_rng(17)
    weight = generator.normal(size=(4, 256)).astype(np.float32)
    errors = {}
    for type_name in ('q4_0', 'q2_k', 'q2_0'):
        quantized, payload = roundtrip(weight, type_name)
        assert payload.dtype == np.uint8
        assert quantized.shape == weight.shape
        errors[type_name] = quantization_error(weight, quantized)['relative_noise_energy']
    assert errors['q4_0'] < errors['q2_k'] < errors['q2_0']
    assert errors['q4_0'] < 0.02


def test_row_block_rule_matches_the_ggml_block_widths():
    assert row_block_ok(896, 'q4_0') == (True, 32)
    assert row_block_ok(896, 'q2_0') == (True, 64)
    assert row_block_ok(896, 'q2_k') == (False, 256)
    assert row_block_ok(4864, 'q2_k') == (True, 256)


def test_aggregate_weights_by_tensor_energy_and_keeps_medians():
    def row(energy, fraction, cos):
        return {'weight_energy': energy, 'codec': {'relative_noise_energy': 0.1,
                                                   'rms_ratio': 0.3, 'snr_db': 10.0},
                'selectivity': 1.0,
                'curves': {'2': {'aligned_energy_fraction': fraction,
                                 'surviving_energy_fraction': 1.0,
                                 'removed_energy_fraction': 1.0 - cos,
                                 'cos': cos, 'isotropic_fraction': 0.5}}}

    result = aggregate([row(100.0, 0.1, 0.9), row(300.0, 0.5, 1.0)], rank_key='2')
    assert np.isclose(result['curves']['2']['aligned_energy_fraction'], (10.0 + 150.0) / 400.0)
    assert np.isclose(result['curves']['2']['cos_median'], 0.95)


# --------------------------------------------------------------------------- file bytes
def _synthetic_gguf(path):
    generator = np.random.default_rng(19)
    tensors = []
    for name, dims in (('blk.0.attn_output.weight', (64, 2)),
                       ('blk.0.ffn_down.weight', (64, 1)),
                       ('blk.0.attn_q.weight', (64, 3))):
        logical = np.ascontiguousarray(generator.normal(size=tuple(reversed(dims))).astype(np.float32))
        tensors.append((name, dims, 0, logical.tobytes()))
    write_gguf(path, {'general.architecture': 'test', 'general.name': 'codec-floor',
                      'tokenizer.ggml.model': 'gpt2'}, tensors)
    return parse_container(path)


def test_written_file_keeps_untouched_tensors_and_reverts(tmp_path):
    source = tmp_path / 'source.gguf'
    header = _synthetic_gguf(source)
    header['source'] = str(source)
    original = _sha(source)
    from jdewitt.ggufio import decode_values

    tensor = next(t for t in header['tensors'] if t['name'] == 'blk.0.attn_output.weight')
    with open(source, 'rb') as handle:
        handle.seek(header['data_base'] + tensor['offset'])
        raw = np.frombuffer(handle.read(64 * 2 * 4), dtype=np.uint8)
    weight = decode_values(raw, 'F32').reshape(2, 64)
    quantized, payload = roundtrip(weight, 'q4_0')
    out = tmp_path / 'edited.gguf'
    write_container(out, header, {'blk.0.attn_output.weight': payload},
                    {'blk.0.attn_output.weight': type_id_for_name('q4_0')})
    checks = verify_written(source, out, {'blk.0.attn_output.weight': payload},
                            {'blk.0.attn_output.weight': type_id_for_name('q4_0')}, header)
    assert checks['loads'] is True
    assert checks['untouched_all_bit_identical'] is True
    assert checks['untouched_tensors_byte_identical'] == 2
    assert checks['edited_tensors_with_target_type'] == 1
    assert checks['kv_section_byte_identical'] is True
    assert checks['tensor_names_order_identical'] is True
    assert checks['problems'] == []
    assert _sha(out) != original

    sidecar = tmp_path / 'revert.npz'
    save_whole_file_revert(source, out, sidecar)
    revert = revert_inplace(out, sidecar)
    assert revert['byte_exact'] is True
    assert _sha(out) == original


def test_verify_written_catches_a_tampered_payload(tmp_path):
    source = tmp_path / 'source.gguf'
    header = _synthetic_gguf(source)
    header['source'] = str(source)
    from jdewitt.ggufio import decode_values

    tensor = next(t for t in header['tensors'] if t['name'] == 'blk.0.ffn_down.weight')
    with open(source, 'rb') as handle:
        handle.seek(header['data_base'] + tensor['offset'])
        raw = np.frombuffer(handle.read(64 * 4), dtype=np.uint8)
    weight = decode_values(raw, 'F32').reshape(1, 64)
    _quantized, payload = roundtrip(weight, 'q4_0')
    out = tmp_path / 'edited.gguf'
    write_container(out, header, {'blk.0.ffn_down.weight': payload},
                    {'blk.0.ffn_down.weight': type_id_for_name('q4_0')})
    tampered = payload.copy()
    tampered[0] ^= 0xFF
    checks = verify_written(source, out, {'blk.0.ffn_down.weight': tampered},
                            {'blk.0.ffn_down.weight': type_id_for_name('q4_0')}, header)
    assert checks['untouched_all_bit_identical'] is False
    assert checks['problems'], 'a payload that is not the measured bytes must be reported'


# ------------------------------------------------------------------------ committed report
def test_committed_report_is_static_and_checks_pass():
    report = json.loads(REPORT.read_text())
    assert report['removal_measured'] is False
    assert report['zero_forward']['gpu'] is False
    assert report['zero_forward']['serve'] is False
    for name, value in report['checks'].items():
        assert value is True, f'{name} is not true: {value!r}'
    for type_name, block in report['types'].items():
        artifact = block['artifact']
        assert artifact['revert']['byte_exact'] is True
        assert artifact['revert']['restored_sha256'] == report['artifact']['sha256']
        assert artifact['untouched_all_bit_identical'] is True
        assert artifact['kv_section_byte_identical'] is True
        assert artifact['loads'] is True
        assert artifact['tensors_decoded'] == report['artifact']['tensors']
        assert artifact['problems'] == []
        for row in block['per_tensor']:
            measured = row['surface_span']
            assert 0.0 <= measured['aligned_energy_fraction'] <= 1.0
            assert 0.0 <= measured['cos'] <= 1.0
            assert measured['rank'] == report['subspaces']['surface_span']['rank']
            assert row['mean_axis']['rank'] == 1
        if block['skipped']:
            # a skip is only ever the row-block rule, and it names the reason
            for skip in block['skipped']:
                assert 'blocks' in skip['reason']


def test_committed_report_is_isotropic_declared_and_typed():
    report = json.loads(REPORT.read_text())
    surface = report['subspaces']['surface_span']
    assert surface['rank'] >= 2
    assert abs(surface['orthonormality_error']) < 1e-6
    assert report['artifact']['sha256'].startswith('0b2e1747')
    # the aligned fraction of a writeback tensor is not an excess: it is ~rank/width
    for type_name, block in report['types'].items():
        aggregate_row = block['aggregate']['surface_span']['edit']
        assert 0.2 < aggregate_row['fraction_over_isotropic'] < 5.0
