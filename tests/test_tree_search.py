"""Algebra-only fixtures: no pretrained model or device initialization."""
from dataclasses import replace

import pytest
import torch


def api():
    from b70_ablator.tree_search import Action, TreeSettings, search_tree
    return Action, TreeSettings, search_tree


class Segments:
    state_bytes = 12
    n_layers = 3

    def __init__(self):
        self.calls = []

    def prepare(self):
        return torch.tensor([[[2., 2., 1.]]])

    def advance(self, h, start, stop):
        self.calls.append((start, stop))
        for layer in range(start, stop):
            # Deliberately in-place: sibling states must never alias.
            if layer == 1:
                h[..., 0] += .5 * h[..., 1]
            if layer == 2:
                h[..., 0] = (h[..., 0] - 2).square() + h[..., 1]
        return h


class Readout:
    def evaluate(self, h, baseline, checkpoint):
        x = float(h[0, -1, 0])
        damage = float(abs(h[0, -1, 2] - baseline[0, -1, 2]))
        return {'objective': x, 'refusal_margin': x, 'protected_kl': damage,
                'norm_change': 0., 'admissible': damage <= .01}


def setup(beam_width=2, **kwargs):
    Action, TreeSettings, search_tree = api()
    settings = TreeSettings(checkpoints=(0, 1, 2), beam_width=beam_width, finalists=1,
                            max_expansions=30, max_state_bytes=10000, strengths=(1.,), **kwargs)
    actions = [Action('x', {0: torch.tensor([[1., 0., 0.]]), 1: torch.tensor([[1., 0., 0.]])}),
               Action('y', {0: torch.tensor([[0., 1., 0.]]), 1: torch.tensor([[0., 1., 0.]])})]
    return search_tree, settings, actions


def test_tree_keeps_a_temporarily_worse_branch_that_wins_after_further_forward():
    run, settings, actions = setup()
    backend = Segments()
    result = run(backend, Readout(), actions, settings)
    assert result.finalists[0]['path'] == [
        {'checkpoint': 0, 'action': 'y', 'strength': 1.},
        {'checkpoint': 1, 'action': 'none', 'strength': 0.}]
    assert result.finalists[0]['metrics']['objective'] == 0.
    _, greedy, _ = setup(beam_width=1)
    assert run(Segments(), Readout(), actions, greedy).finalists[0]['metrics']['objective'] == 1.
    assert backend.calls.count((0, 1)) == 1, 'shared prefix must be computed once'
    assert result.expansions == 9
    assert result.baseline['objective'] == 3.
    assert all('checkpoint' in row and 'path' in row for row in result.trace)


def test_rejected_damage_and_nonfinite_branches_cannot_win():
    run, settings, actions = setup()
    actions.append(api()[0]('damage', {i: torch.tensor([[0., 0., 1.]]) for i in (0, 1)}))
    result = run(Segments(), Readout(), actions, settings)
    assert any(row.get('rejected') == 'proxy_guard' for row in result.trace)
    assert all(step['action'] != 'damage' for step in result.finalists[0]['path'])
    class Broken(Segments):
        def advance(self, h, start, stop):
            h = super().advance(h, start, stop)
            if start == 1 and float(h[0, -1, 1]) == 0:
                h.fill_(float('nan'))
            return h
    result = run(Broken(), Readout(), actions[:2], settings)
    assert any(row.get('rejected') == 'nonfinite_state' for row in result.trace)
    assert all(torch.isfinite(torch.tensor(r['metrics']['objective'])) for r in result.finalists)


def test_state_budget_rejected_before_prepare_or_partial_forward():
    run, settings, actions = setup()
    settings = replace(settings, max_state_bytes=1)
    backend = Segments()
    backend.prepare = lambda: pytest.fail('budget must be checked before any forward')
    with pytest.raises(ValueError, match='state.*budget'):
        run(backend, Readout(), actions, settings)
    assert not backend.calls


@pytest.mark.parametrize('field,value', [('beam_width', True), ('finalists', 3),
    ('checkpoints', (1, 0, 2)), ('checkpoints', (0, 1)),
    ('strengths', (float('nan'),)), ('strengths', (0.,)), ('max_expansions', 1)])
def test_invalid_settings_or_insufficient_complete_search_budget(field, value):
    run, settings, actions = setup()
    with pytest.raises(ValueError):
        run(Segments(), Readout(), actions, replace(settings, **{field: value}))


def test_projection_changes_only_last_token_and_never_mutates_parent():
    from b70_ablator.tree_search import project_last
    h = torch.tensor([[[1., 2.], [3., 4.]]])
    before = h.clone()
    edited = project_last(h, torch.tensor([[1., 0.]]), 1.)
    torch.testing.assert_close(edited, torch.tensor([[[1., 2.], [0., 4.]]]))
    torch.testing.assert_close(h, before)
    with pytest.raises(ValueError, match='orthonormal'):
        project_last(h, torch.tensor([[2., 0.]]), 1.)


def test_jspace_transport_norm_and_selected_logits_match_explicit_math():
    from b70_ablator.tree_search import JSpaceReadout
    class RMS(torch.nn.Module):
        def forward(self, h):
            return h / h.square().mean(-1, keepdim=True).add(1e-6).sqrt()
    j = torch.tensor([[2., 1.], [0., 3.]])
    head = torch.tensor([[1., 0.], [0., 1.], [1., -1.], [-1., 1.]])
    lens = JSpaceReadout({0: j}, RMS(), head, [0], [1], [2, 3],
                        max_protected_kl=.001, max_norm_change=.1, drift_weight=2.)
    h = torch.tensor([[[1., 2.]]])
    expected = RMS()(h[0, -1] @ j.T) @ head.T
    torch.testing.assert_close(lens.logits(h, 0), expected)
    score = lens.evaluate(h, h, 0)
    assert score['admissible'] and score['protected_kl'] == 0.
    changed = torch.tensor([[[2., 1.]]])
    assert not lens.evaluate(changed, h, 0)['admissible']
    # Directions are Jacobian pullbacks, not raw unembedding/logit-lens directions.
    action = lens.pullback_action('refusal', [0], [0])
    expected_direction = (head[0] - head[1]) @ j
    expected_direction /= expected_direction.norm()
    torch.testing.assert_close(action.bases[0], expected_direction.unsqueeze(0))


def test_jspace_rejects_missing_lens_and_degenerate_axes():
    from b70_ablator.tree_search import JSpaceReadout
    args = ({0: torch.eye(2)}, torch.nn.Identity(), torch.eye(4, 2))
    with pytest.raises(ValueError, match='disjoint'):
        JSpaceReadout(*args, [0], [0], [2, 3], max_protected_kl=1., max_norm_change=1., drift_weight=1.)
    lens = JSpaceReadout(*args, [0], [1], [2, 3], max_protected_kl=1., max_norm_change=1., drift_weight=1.)
    with pytest.raises(ValueError, match='checkpoint'):
        lens.logits(torch.ones(1, 1, 2), 1)
