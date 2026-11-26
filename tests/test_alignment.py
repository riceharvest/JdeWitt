"""Static checks for the alignment-matrix experiment: algebra and byte identity."""
import hashlib
import json
from pathlib import Path

import numpy as np

from jdewitt.alignment import (_svd, aggregate, choose_k, tensor_curve,
                               writeback_tensors)
from jdewitt.ggufio import edit_tensors_inplace, revert_inplace, write_gguf


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_writeback_selection_keeps_only_writeback_kinds():
    keys = [
        'model.layers.0.self_attn.o_proj.weight',
        'model.layers.0.mlp.down_proj.weight',
        'model.layers.0.self_attn.q_proj.weight',
        'model.layers.0.mlp.gate_proj.weight',
        'model.embed_tokens.weight',
        'model.layers.1.self_attn.o_proj.weight',
    ]
    assert writeback_tensors(keys) == [
        ('model.layers.0.self_attn.o_proj.weight', 'attn', 0),
        ('model.layers.0.mlp.down_proj.weight', 'ffn', 0),
        ('model.layers.1.self_attn.o_proj.weight', 'attn', 1),
    ]


def test_gram_svd_matches_lapack_both_orientations():
    rng = np.random.default_rng(7)
    for shape in ((12, 16), (16, 12)):
        matrix = rng.normal(size=shape)
        u, s, vt = _svd(matrix)
        ref_u, ref_s, ref_vt = np.linalg.svd(matrix, full_matrices=False)
        assert np.allclose(s, ref_s, rtol=1e-12, atol=1e-12)
        reconstructed = (u[:, :4] * s[:4]) @ vt[:4]
        expected = (ref_u[:, :4] * ref_s[:4]) @ ref_vt[:4]
        assert np.allclose(reconstructed, expected, rtol=1e-10, atol=1e-10)
        assert np.allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-10)


def test_removed_energy_is_the_sum_of_removed_singular_squares():
    rng = np.random.default_rng(11)
    base = rng.normal(size=(10, 14))
    u, _ = np.linalg.qr(rng.normal(size=(10, 3)))
    v, _ = np.linalg.qr(rng.normal(size=(14, 3)))
    delta = (u * np.array([5.0, 3.0, 1.0])) @ v.T
    instruct = base + delta
    curve = tensor_curve(base, instruct, None, ks=(1, 2, 3))
    assert curve['checks']['max_removed_energy_rel_error'] < 1e-12
    energies = {k: row['removed_energy'] for k, row in curve['curves'].items()}
    assert np.isclose(energies['1'], 25.0)
    assert np.isclose(energies['2'], 34.0)
    assert np.isclose(energies['3'], 35.0)
    assert np.isclose(curve['curves']['1']['removed_energy_fraction'], 25.0 / 35.0)
    # remaining delta is orthogonal to the removed part, so the norm identity holds
    assert np.isclose(curve['curves']['1']['delta_residual_norm_ratio'], (10.0 / 35.0) ** 0.5)
    # weight norm ratio is the surviving share of the instruct matrix, not the delta
    assert 0.0 < curve['curves']['3']['weight_norm_ratio'] < 1.0


def test_twin_capture_reads_the_removed_subspace():
    rng = np.random.default_rng(13)
    base = rng.normal(size=(10, 14))
    u, _ = np.linalg.qr(rng.normal(size=(10, 2)))
    v, _ = np.linalg.qr(rng.normal(size=(14, 2)))
    delta = (u * np.array([4.0, 2.0])) @ v.T
    instruct = base + delta
    twin = instruct + 0.5 * delta  # twin delta lives entirely inside the same subspace
    curve = tensor_curve(base, instruct, twin, ks=(1, 2))
    for row in curve['curves'].values():
        assert np.isclose(row['twin_left_capture'], row['removed_energy_fraction'], atol=1e-9)
        assert np.isclose(row['twin_right_capture'], row['removed_energy_fraction'], atol=1e-9)
        assert row['twin_left_overlap'] > 0.999


def test_choose_k_obeys_the_residual_floor():
    curves = {'1': {'weight_norm_ratio': 0.99, 'twin_left_capture': 0.2, 'twin_right_capture': 0.2},
              '4': {'weight_norm_ratio': 0.90, 'twin_left_capture': 0.9, 'twin_right_capture': 0.9}}
    assert choose_k(curves, ks=(1, 4), residual_floor=0.95)['k'] == 1
    assert choose_k(curves, ks=(1, 4), residual_floor=0.80)['k'] == 4
    infeasible = {'1': {'weight_norm_ratio': 0.10, 'twin_left_capture': 0.2,
                        'twin_right_capture': 0.2}}
    assert choose_k(infeasible, ks=(1,), residual_floor=0.95)['feasible'] is False


def test_aggregate_weights_by_energy():
    def row(energy, frac, twin_capture, instruct_norm):
        return {'delta_energy': energy, 'instruct_norm': instruct_norm,
                'twin_delta_norm': 1.0,
                'curves': {'2': {'removed_energy': energy * frac,
                                 'weight_norm_ratio': 0.5,
                                 'twin_left_capture': twin_capture,
                                 'twin_right_capture': twin_capture,
                                 'twin_left_overlap': twin_capture,
                                 'twin_right_overlap': twin_capture}}}
    rows = [row(100.0, 0.5, 0.0, 10.0), row(300.0, 0.9, 1.0, 10.0)]
    result = aggregate(rows, ks=(2,))['2']
    assert np.isclose(result['removed_energy_fraction'], (50.0 + 270.0) / 400.0)
    assert np.isclose(result['delta_residual_norm_ratio'], ((50.0 + 30.0) / 400.0) ** 0.5)
    assert np.isclose(result['weight_norm_ratio'], 0.5)
    assert np.isclose(result['twin_left_capture'], 0.5)


def test_inplace_edit_leaves_untouched_tensors_bit_identical_and_reverts(tmp_path):
    rng = np.random.default_rng(17)
    tensors = []
    for name, dims in (('blk.0.attn_output.weight', (3, 4)),
                       ('blk.0.ffn_down.weight', (6, 4)),
                       ('blk.9.attn_q.weight', (4, 4))):
        logical = np.ascontiguousarray(rng.normal(size=tuple(reversed(dims))).astype(np.float32))
        tensors.append((name, dims, 0, logical.tobytes()))
    source = tmp_path / 'source.gguf'
    write_gguf(source, {'general.architecture': 'test', 'general.name': 'pytest'}, tensors)
    original = _sha(source)
    deltas = {'blk.0.attn_output.weight': np.full((4, 3), 0.05, dtype=np.float32),
              'blk.0.ffn_down.weight': np.full((4, 6), 0.05, dtype=np.float32)}

    work = tmp_path / 'work.gguf'
    work.write_bytes(source.read_bytes())
    sidecar = tmp_path / 'revert.npz'
    report = edit_tensors_inplace(work, deltas, revert_path=sidecar)
    assert report['tensors_edited'] == 2
    assert _sha(work) != original

    from jdewitt.agent import gguf_tensor_diff
    diff = gguf_tensor_diff(source, work)
    assert diff['summary']['bit_identical'] == 1
    assert {row['name'] for row in diff['differing_tensors']} == set(deltas)

    restored = revert_inplace(work, sidecar)
    assert restored['restored_sha256'] == original
    assert _sha(work) == original


REPORT = Path(__file__).resolve().parents[1] / 'verification/alignment-matrix/report.json'
APPLY = Path(__file__).resolve().parents[1] / 'verification/alignment-matrix/apply.json'


def test_committed_report_energy_identity_and_flag():
    report = json.loads(REPORT.read_text())
    assert report['removal_measured'] is False
    assert report['checks']['energy_identity_holds'] is True
    assert report['checks']['remain_identity_holds'] is True
    assert report['checks']['singular_values_descending'] is True
    for row in report['per_tensor']:
        singular = row['singular_values']
        for k, curve in row['curves'].items():
            expected = sum(value ** 2 for value in singular[:int(k)])
            assert np.isclose(curve['removed_energy'], expected, rtol=1e-12, atol=0.0)
            assert 0.0 <= curve['removed_energy_fraction'] <= 1.0
            assert 0.0 <= curve['delta_residual_norm_ratio'] <= 1.0


def test_committed_apply_report_byte_checks():
    report = json.loads(APPLY.read_text())
    assert report['removal_measured'] is False
    inplace = report['arms']['inplace_preserve_codec']
    assert inplace['revert']['byte_exact'] is True
    assert inplace['revert']['restored_sha256'] == inplace['revert']['source_sha256']
    assert inplace['metadata_bytes_identical'] is True
    assert inplace['tokenizer_identical'] is True
    assert inplace['untouched']['untouched_all_bit_identical'] is True
    assert inplace['untouched']['differing_outside_edit_set'] == []
    reencode = report['arms']['out_of_place_f16']
    assert reencode['tokenizer_identical'] is True
    assert reencode['untouched']['untouched_all_bit_identical'] is True
    assert reencode['untouched']['differing_outside_edit_set'] == []
    assert reencode['fidelity']['cos_min'] >= 0.999
    # the in-place 3-bit arm must record that it did not cover every delta
    assert report['checks']['inplace_covers_all_deltas'] is False
    assert inplace['skipped_count'] > 0
