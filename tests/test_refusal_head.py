# SPDX-License-Identifier: AGPL-3.0-or-later
"""The head-geometry axis identities, on toys small enough to check by hand.

No model, no tokenizer behind these: a stub tokenizer and 4-wide arrays. If a
tensor edit ever stopped being a projection, or the margin identity stopped
being exact, these fail before any checkpoint is downloaded.
"""
from __future__ import annotations

import numpy as np
import pytest

from jdewitt.refusal_head import (COMPLIANCE_WORDS, DENIAL_WORDS, cosine, edit_stats,
                                  head_axis, head_channel_closed, margin_reduction,
                                  project_out, resolve_surfaces, surface_forms,
                                  table_ids, top_left_direction)


class StubTokenizer:
    """Maps whole surfaces to fixed ids; multi-token surfaces split on the space."""

    def __init__(self, table):
        self.table = table

    def encode(self, text, add_special_tokens=False):
        if text in self.table:
            return list(self.table[text])
        if text.startswith(' '):
            return self.encode(text[1:])
        return [abs(hash(text)) % 100000]


def test_surface_forms_cover_bare_spaced_and_lowercase():
    forms = surface_forms('No')
    assert forms == ('No', ' No', 'no', ' no')


def test_multi_token_surface_is_recorded_not_folded_into_the_mean():
    tokenizer = StubTokenizer({'No': [7, 9], ' No': [7], 'no': [8], ' no': [8],
                               'Unfortunately': [12], ' Unfortunately': [12],
                               'unfortunately': [11, 13], ' unfortunately': [11]})
    resolved = resolve_surfaces(tokenizer, ('No', 'Unfortunately'))
    assert sorted(resolved['ids']) == [7, 8, 11, 12]
    assert any(row['word'] == 'No' and row['form'] == 'No' for row in resolved['multi_token'])
    assert 13 not in resolved['ids']


def test_single_token_table_surface_contributes_and_fused_denial_resolves():
    tokenizer = StubTokenizer({'cannot': [20], ' cannot': [21], 'can not': [22, 23]})
    resolved = resolve_surfaces(tokenizer, (), extra_surfaces=('cannot', ' cannot', 'can not'))
    assert resolved['ids'] == [20, 21]
    assert [row['surface'] for row in resolved['table_single_token']] == ['cannot', ' cannot']


def test_table_ids_first_token_variant_is_the_first_id_of_each_surface():
    tokenizer = StubTokenizer({'cannot': [20], 'can not': [22, 23]})
    assert table_ids(tokenizer, ('cannot', 'can not')) == [20, 22]
    assert table_ids(tokenizer, ('cannot', 'can not'), first_token_only=False) == [20, 22, 23]


def _toy(head=None, seed=0):
    rng = np.random.default_rng(seed)
    if head is None:
        head = rng.normal(size=(6, 4))
    return head, rng


def test_axis_is_unit_and_points_from_compliance_to_denial():
    head, _ = _toy()
    out = head_axis(head, [0, 1], [2, 3])
    d = out['d']
    assert abs(float(np.linalg.norm(d)) - 1.0) < 1e-12
    raw = head[[0, 1]].mean(axis=0) - head[[2, 3]].mean(axis=0)
    assert np.allclose(d, raw / np.linalg.norm(raw))
    assert out['margin'] == pytest.approx(float(np.linalg.norm(raw)))


def test_axis_needs_both_sides():
    head, _ = _toy()
    with pytest.raises(ValueError):
        head_axis(head, [], [2])
    flat = np.ones((6, 4))
    with pytest.raises(ValueError):
        head_axis(flat, [0], [1])


def test_projection_residual_equals_projected_energy():
    head, rng = _toy()
    W = rng.normal(size=(5, 4))
    d = head_axis(head, [0, 1], [2, 3])['d']
    stats = edit_stats(W, d)
    assert stats['unit_ok']
    assert stats['residual_identity_ok']
    assert stats['delta_norm_fro'] == pytest.approx(stats['projected_energy_norm'], rel=1e-12)
    assert stats['delta_norm_fro'] > 0
    assert stats['output_along_d_max'] < 1e-12
    edited, delta = project_out(W, d)
    assert np.allclose(edited, W - np.outer(W @ d, d))
    assert np.allclose(edited @ d, 0.0, atol=1e-12)


def test_projected_energy_is_not_the_frobenius_norm():
    # Guard against a check that could pass by using ||W||_F on both sides.
    head, rng = _toy()
    W = np.eye(4) * 3.0
    d = np.array([1.0, 0.0, 0.0, 0.0])
    stats = edit_stats(W, d)
    assert stats['delta_norm_fro'] == pytest.approx(3.0)
    assert stats['norm_fro'] == pytest.approx(6.0)


def test_margin_reduction_is_exactly_the_projection_for_any_block_input():
    head, rng = _toy()
    axis = head_axis(head, [0, 1], [2, 3])
    d, g = axis['d'], axis['g']
    W = rng.normal(size=(5, 4))
    for _ in range(5):
        a = rng.normal(size=5)
        out = margin_reduction(g, d, W, a)
        assert out['identity_ok'], out
        assert out['reduction'] == pytest.approx(out['predicted'], rel=1e-10, abs=1e-12)


def test_head_channel_closes_exactly_when_d_is_projected_out():
    head, _ = _toy()
    out = head_axis(head, [0, 1], [2, 3])
    closed = head_channel_closed(out['g'], out['d'])
    assert closed['closed']
    assert closed['margin_per_unit_d'] == pytest.approx(out['margin'])
    assert closed['margin_after'] == pytest.approx(0.0, abs=1e-12)


def test_rank_one_delta_recovers_its_own_left_factor():
    rng = np.random.default_rng(3)
    u = rng.normal(size=6)
    u /= np.linalg.norm(u)
    v = rng.normal(size=8)
    delta = -2.5 * np.outer(u, v)
    out = top_left_direction(delta)
    assert abs(cosine(out['direction'], u)) == pytest.approx(1.0, abs=1e-10)
    assert out['effective_rank'] == pytest.approx(1.0, rel=1e-10)


def test_rank_two_delta_reports_energy_below_one():
    rng = np.random.default_rng(4)
    delta = np.outer(rng.normal(size=6), rng.normal(size=8)) + 0.1 * np.outer(
        rng.normal(size=6), rng.normal(size=8))
    out = top_left_direction(delta)
    assert out['energy_fraction_top'] > 0.9
    assert out['effective_rank'] > 1.0


def test_cosine_refuses_shape_and_zero_vectors():
    with pytest.raises(ValueError):
        cosine(np.ones(3), np.ones(4))
    with pytest.raises(ValueError):
        cosine(np.zeros(3), np.ones(3))


def test_vocabularies_are_the_task_sets():
    assert DENIAL_WORDS == ('I', 'Sorry', 'Cannot', 'Unfortunately', 'No')
    assert COMPLIANCE_WORDS == ('Sure', 'Certainly', 'Yes', 'Here', 'Okay', 'Absolutely')
