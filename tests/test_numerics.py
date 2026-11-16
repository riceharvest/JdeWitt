import pytest
import torch
from torch import nn
from b70_ablator.numerics import token_kl, full_reply_kl, directions_from_languages
from b70_ablator.editor import AblatedLinear, discover_targets


def test_kl_checks_late_tokens_not_only_first():
    base = torch.zeros(4, 3)
    candidate = base.clone()
    candidate[-1, 0] = 4
    assert token_kl(base[0:1], candidate[0:1]).item() == 0
    assert full_reply_kl(base, candidate, answer_mask=torch.ones(4, dtype=torch.bool)) > .1


def test_kl_mask_and_shape_mismatch():
    p = torch.zeros(4, 3)
    q = p.clone()
    q[0, 0] = 4
    assert full_reply_kl(p, q, answer_mask=torch.tensor([False, True, True, True])) == 0
    with pytest.raises(ValueError):
        full_reply_kl(p, q[:1], answer_mask=torch.ones(4, dtype=torch.bool))
    with pytest.raises(ValueError):
        full_reply_kl(p, q, answer_mask=torch.zeros(4, dtype=torch.bool))


def test_nonfinite_kl_fails_closed():
    with pytest.raises(ValueError):
        token_kl(torch.zeros(2, 3), torch.full((2, 3), float('nan')))


def test_multilingual_directions_balanced_and_orthonormal():
    good = {'en': torch.tensor([[1., 0., 0.]]), 'nl': torch.tensor([[0., 1., 0.]])}
    bad = {'en': torch.tensor([[1., 0., 1.]]), 'nl': torch.tensor([[0., 2., 0.]])}
    d = directions_from_languages(good, bad, rank=2)
    assert d.shape == (1, 2, 3)
    assert torch.allclose(d[0] @ d[0].T, torch.eye(2), atol=1e-5)


def test_degenerate_directions_fail():
    x = {'en': torch.zeros(1, 3)}
    with pytest.raises(ValueError):
        directions_from_languages(x, x, rank=1)


def test_rankk_delta_matches_projection_and_reset():
    base = nn.Linear(4, 3, bias=True)
    w = base.weight.detach().clone()
    b = base.bias.detach().clone()
    layer = AblatedLinear(base, rank=2)
    d = torch.eye(3)[:2]
    layer.set_directions(d, strength=.7)
    x = torch.randn(2, 4)
    expected = nn.functional.linear(x, w - .7 * d.T @ (d @ w), b)
    assert torch.allclose(layer(x), expected, atol=1e-6)
    assert torch.equal(base.weight, w)
    layer.reset()
    assert torch.equal(layer(x), base(x))


def test_merge_matches_adapter():
    layer = AblatedLinear(nn.Linear(4, 3), rank=1)
    layer.set_directions(torch.tensor([[1., 0., 0.]]), .5)
    x = torch.randn(2, 4)
    before = layer(x)
    merged = layer.merge()
    assert torch.allclose(before, merged(x), atol=1e-6)


def test_discovery_fails_on_unknown_and_empty_layers():
    from types import SimpleNamespace
    model = nn.Module()
    model.config = SimpleNamespace(model_type='future_unknown')
    with pytest.raises(ValueError, match='adapter'):
        discover_targets(model)


def test_mixed_dtype_projection_explicitly_aligned():
    layer = AblatedLinear(nn.Linear(4, 3, dtype=torch.bfloat16), rank=1)
    layer.set_directions(torch.tensor([[1., 0., 0.]], dtype=torch.float32), .5)
    assert layer.A.dtype == layer.base.weight.dtype
    assert layer.B.dtype == layer.base.weight.dtype
    assert torch.isfinite(layer(torch.randn(2, 4, dtype=torch.bfloat16))).all()
