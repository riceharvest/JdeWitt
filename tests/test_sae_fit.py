"""The SAE fit: shapes, FVU identity, seed determinism, and the contrast table."""
import json
import math

import pytest
import torch

from jdewitt import sae_fit


def rows(n, hidden, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(hidden, generator=generator) for _ in range(n)]


def test_topk_encoder_keeps_exactly_k_nonzero_per_row():
    model = sae_fit.TopKSAE(hidden=8, features=16, k=3)
    acts = model.encode(torch.randn(5, 8))
    assert acts.shape == (5, 16)
    assert torch.equal((acts > 0).sum(dim=1), torch.full((5,), 3))
    assert (acts >= 0).all()


def test_decoder_rows_stay_unit_norm_through_training():
    model, _, _ = sae_fit.train_sae(rows(8, 8), features=16, k=3, steps=30, batch=4,
                                    lr=1e-2, seed=1)
    norms = model.decoder_directions().norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_k_outside_the_dictionary_is_refused():
    with pytest.raises(ValueError, match='k must be in'):
        sae_fit.TopKSAE(hidden=8, features=4, k=5)


def test_fvu_is_zero_for_an_exact_reconstruction_and_one_for_the_mean():
    x = torch.randn(6, 4, generator=torch.Generator().manual_seed(2))
    assert sae_fit.fvu(x, x.clone()) == 0.0
    assert math.isclose(sae_fit.fvu(x, x.mean(dim=0, keepdim=True).expand_as(x)), 1.0)


def test_fvu_refuses_a_zero_variance_evaluation():
    x = torch.ones(4, 3)
    with pytest.raises(ValueError, match='zero variance'):
        sae_fit.fvu(x, x + 1.0)


def test_training_is_deterministic_for_a_seed_and_moves_for_a_different_one():
    data = rows(12, 8, seed=3)
    first, _, _ = sae_fit.train_sae(data, features=16, k=4, steps=30, batch=6, lr=1e-2, seed=7)
    again, _, _ = sae_fit.train_sae(data, features=16, k=4, steps=30, batch=6, lr=1e-2, seed=7)
    other, _, _ = sae_fit.train_sae(data, features=16, k=4, steps=30, batch=6, lr=1e-2, seed=8)
    assert sae_fit.digest_tensor(first.W_dec) == sae_fit.digest_tensor(again.W_dec)
    assert sae_fit.digest_tensor(first.W_dec) != sae_fit.digest_tensor(other.W_dec)


def test_standardiser_is_fit_on_the_train_split_only():
    train = rows(6, 5, seed=4)
    heldout = [torch.full((5,), 100.0)] * 3
    matrix = sae_fit._as_matrix(train)
    standardiser = sae_fit.Standardiser(matrix)
    assert torch.allclose(standardiser.mean.flatten(), matrix.mean(dim=0))
    # A held-out row of a wildly different scale must not appear in the scaling.
    polluted = sae_fit.Standardiser(sae_fit._as_matrix(train + heldout))
    assert not torch.allclose(standardiser.mean.flatten(), polluted.mean.flatten())
    model, _, _ = sae_fit.train_sae(train, features=8, k=2, steps=5, batch=3, lr=1e-2, seed=1)
    before = standardiser.mean.clone()
    sae_fit.score_heldout(model, standardiser, heldout, train)
    assert torch.equal(standardiser.mean, before)


def test_rank_r_data_is_reconstructed_by_the_matching_pca_control():
    generator = torch.Generator().manual_seed(5)
    basis = torch.linalg.qr(torch.randn(6, 2, generator=generator))[0]
    coefficients = torch.randn(20, 2, generator=generator)
    data = coefficients @ basis.T
    train, heldout = data[:16], data[16:]
    assert sae_fit.pca_control_fvu(train, heldout, rank=2) < 1e-6
    # Same control at rank 0 would be the mean predictor, i.e. FVU 1.
    assert math.isclose(sae_fit.pca_control_fvu(train, heldout, rank=0), 1.0, abs_tol=1e-5)


def test_score_heldout_reports_the_baseline_rule_verbatim():
    data = rows(16, 8, seed=6)
    model, standardiser, history = sae_fit.train_sae(data[:10], features=16, k=4, steps=20,
                                                     batch=5, lr=1e-2, seed=2)
    scored = sae_fit.score_heldout(model, standardiser, data[10:], data[:10])
    assert scored['rows'] == 6
    assert scored['baseline_rule'] == 'fvu < 1.0 and fvu < pca_control_fvu'
    assert history['seconds'] >= 0.0 and history['steps'] == 20


def test_feature_table_contrast_is_the_difference_of_the_two_activation_means():
    refusal, compliant = rows(9, 8, seed=7), rows(11, 8, seed=8)
    model, standardiser, _ = sae_fit.train_sae(refusal, features=16, k=2, steps=20, batch=4,
                                               lr=1e-2, seed=3)
    table = sae_fit.feature_table(model, standardiser, refusal, compliant, top_k=4)
    assert len(table) == 4
    acts_r = model.encode(standardiser(sae_fit._as_matrix(refusal)))
    acts_c = model.encode(standardiser(sae_fit._as_matrix(compliant)))
    for row in table:
        feature = row['feature']
        assert math.isclose(row['mean_refusal'], float(acts_r[:, feature].mean()), abs_tol=1e-5)
        assert math.isclose(row['mean_compliant'], float(acts_c[:, feature].mean()), abs_tol=1e-5)
        assert math.isclose(row['contrast'], row['mean_refusal'] - row['mean_compliant'], abs_tol=1e-5)
        assert -1.0 - 1e-5 <= row['cos_to_axis'] <= 1.0 + 1e-5
    # Ordered by |contrast|, descending.
    magnitudes = [abs(row['contrast']) for row in table]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_checkpoint_state_and_config_json_are_round_trippable():
    model, standardiser, _ = sae_fit.train_sae(rows(6, 8), features=8, k=2, steps=5, batch=3,
                                               lr=1e-2, seed=4)
    state = sae_fit.checkpoint_state(model, standardiser)
    assert set(state) >= {'W_enc', 'W_dec', 'b_enc', 'b_dec', 'standardiser_mean', 'standardiser_scale'}
    config = json.loads(sae_fit.config_json({'features': 8, 'k': 2}))
    assert config == {'features': 8, 'k': 2}
    assert sae_fit.digest_tensor(state['W_dec']) == sae_fit.digest_tensor(model.W_dec)
