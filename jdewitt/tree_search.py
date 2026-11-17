# SPDX-License-Identifier: AGPL-3.0-or-later
"""Experimental prefill intervention tree. Scores are proxies, never certification.

Checkpoints are ZERO-BASED block OUTPUTS, matching upstream jlens. Each child
really advances through the next block segment before it is scored. Only the
last prompt position is edited. No model weights, KV caches or decoder outputs
are stored in the tree. Runtime adapters must supply explicit same-device states.
"""
from dataclasses import dataclass, field
import math

import torch
from torch.nn import functional as F

from .architectures import MAX_STRENGTH
from .tree_config import TreeSettings as TreeSettings


@dataclass(frozen=True)
class Action:
    name: str
    bases: dict



@dataclass
class TreeResult:
    finalists: list
    baseline: dict
    trace: list
    expansions: int
    state_bound_bytes: int
    baseline_features: list | None = None
    feature_schema: dict | None = None


@dataclass
class _Node:
    hidden: torch.Tensor
    path: list
    metrics: dict
    history: list = field(default_factory=list)


def _basis_ok(basis, width, device):
    if (not isinstance(basis, torch.Tensor) or basis.device != device or basis.ndim != 2
            or basis.shape[1] != width or not 0 < basis.shape[0] <= width
            or not basis.is_floating_point() or not bool(torch.isfinite(basis).all())):
        raise ValueError('Tree direction needs a finite same-device [rank, hidden] basis')
    b = basis.float()
    if not torch.allclose(b @ b.T, torch.eye(len(b), device=device), atol=1e-4):
        raise ValueError('Tree direction basis must be orthonormal')


def project_last(hidden, basis, strength):
    if (hidden.ndim != 3 or hidden.shape[0] != 1 or not hidden.is_floating_point()
            or type(strength) not in (float, int) or not math.isfinite(strength)
            or not 0 <= strength <= MAX_STRENGTH):
        raise ValueError('Tree projection requires a single finite-budget residual sequence')
    _basis_ok(basis, hidden.shape[-1], hidden.device)
    result = hidden.clone()
    h, b = hidden[0, -1].float(), basis.float()
    result[0, -1] = (h - strength * ((h @ b.T) @ b)).to(hidden.dtype)
    return result


def _rank(node):
    # Stable ties prefer fewer interventions, then their exact named trajectory.
    return (node.metrics.get('ranker_priority', 0), node.metrics['objective'],
            sum(s['action'] != 'none' for s in node.path),
            tuple((s['action'], s['strength']) for s in node.path))


def _check_metrics(metrics):
    if (not isinstance(metrics, dict) or type(metrics.get('admissible')) is not bool
            or type(metrics.get('objective')) not in (int, float)
            or any(type(v) in (int, float) and not math.isfinite(v) for v in metrics.values())):
        raise ValueError('Invalid/nonfinite tree proxy metrics')


@torch.inference_mode()
def search_tree(backend, readout, actions, settings):
    """Bounded beam of real partial forwards, with an independent unedited path.

    max_state_bytes bounds saved branch tensors, not attention/MLP workspaces or
    total VRAM. Candidates are evaluated serially and only the top beam is kept.
    An insufficient expansion budget fails before any forward, never truncates
    the action set and quietly biases selection. No full answer is decoded here.
    """
    settings.validate(backend.n_layers, len(actions))
    names = [a.name for a in actions]
    if len(set(names)) != len(names) or any(not isinstance(n, str) or not n.strip() or n == 'none' for n in names):
        raise ValueError('Tree action names must be unique and cannot be none')
    # Old beam + new beam + baseline/working clones, plus transient final-vector
    # projections. This is intentionally conservative and independent of pruning.
    bound = (2 * settings.beam_width + 16) * backend.state_bytes
    if type(backend.state_bytes) is not int or backend.state_bytes < 1 or bound > settings.max_state_bytes:
        raise ValueError(f'Tree state budget too small; conservative saved-state bound is {bound} bytes')
    initial = backend.prepare()
    if (initial.ndim != 3 or initial.shape[0] != 1 or initial.shape[1] < 1
            or not initial.is_floating_point() or initial.numel() * initial.element_size() != backend.state_bytes):
        raise ValueError('Tree backend state shape/byte declaration mismatch')
    for action in actions:
        if set(action.bases) != set(settings.checkpoints[:-1]):
            raise ValueError('Every tree direction must cover exactly the intervention checkpoints')
        for basis in action.bases.values():
            _basis_ok(basis, initial.shape[-1], initial.device)
    baseline = backend.advance(initial, 0, settings.checkpoints[0] + 1)
    del initial
    if not bool(torch.isfinite(baseline).all()):
        raise ValueError('Nonfinite unedited tree prefix')
    trajectory_mode = callable(getattr(readout, 'snapshot', None))
    baseline_history = ([readout.snapshot(baseline, settings.checkpoints[0])] if trajectory_mode else [])
    beam = [_Node(baseline, [], {}, baseline_history)]
    trace, expansions = [], 0
    baseline_metrics = None
    for start_cp, stop_cp in zip(settings.checkpoints, settings.checkpoints[1:]):
        baseline = backend.advance(baseline.clone(), start_cp + 1, stop_cp + 1)
        if not bool(torch.isfinite(baseline).all()):
            raise ValueError('Nonfinite unedited tree baseline')
        if trajectory_mode:
            baseline_history = baseline_history + [readout.snapshot(baseline, stop_cp)]
            baseline_metrics = readout.evaluate_history(baseline, baseline, stop_cp,
                                                        baseline_history, baseline_history, is_baseline=True)
        else:
            baseline_metrics = readout.evaluate(baseline, baseline, stop_cp)
        _check_metrics(baseline_metrics)
        if not baseline_metrics['admissible']:
            raise ValueError('Tree proxy rejects its own unedited baseline')
        kept = []
        for parent in beam:
            options = [('none', 0., None)] + [(a.name, float(s), a.bases[start_cp])
                                               for a in actions for s in settings.strengths]
            for name, strength, basis in options:
                path = parent.path + [{'checkpoint': start_cp, 'action': name, 'strength': strength}]
                state = parent.hidden.clone() if basis is None else project_last(parent.hidden, basis, strength)
                state = backend.advance(state, start_cp + 1, stop_cp + 1)
                expansions += 1
                row = {'checkpoint': stop_cp, 'path': path}
                if not bool(torch.isfinite(state).all()):
                    row['rejected'] = 'nonfinite_state'
                else:
                    history = parent.history
                    if trajectory_mode:
                        history = history + [readout.snapshot(state, stop_cp)]
                        row['features'] = readout.encode_history(history, baseline_history)
                        metrics = readout.evaluate_history(state, baseline, stop_cp, history, baseline_history)
                    else:
                        metrics = readout.evaluate(state, baseline, stop_cp)
                    _check_metrics(metrics)
                    row['metrics'] = metrics
                    if not metrics['admissible']:
                        row['rejected'] = 'proxy_guard'
                    else:
                        kept.append(_Node(state, path, metrics, history))
                        kept.sort(key=_rank)
                        del kept[settings.beam_width:]
                trace.append(row)
                del state
        if not kept:
            # Do not silently relax a failed gate or manufacture a winner.
            raise ValueError(f'No admissible tree branches at checkpoint {stop_cp}')
        beam = kept
        del parent, kept
    finalists = [{'path': node.path, 'metrics': node.metrics} for node in beam[:settings.finalists]]
    baseline_features = readout.encode_history(baseline_history, baseline_history) if trajectory_mode else None
    schema = readout.schema if trajectory_mode else None
    return TreeResult(finalists, baseline_metrics, trace, expansions, bound, baseline_features, schema)


class JSpaceReadout:
    """Actual J_l transport + the model's final norm + selected unembedding rows.

    Refusal-vs-reference logit margin is only a ranking proxy. Protected-token KL
    and residual-norm guards detect some damage, NOT preserved task capability.
    The final checkpoint uses J=I (actual output readout), not a fitted surrogate.
    """
    def __init__(self, jacobians, norm, head_weight, refusal_ids, reference_ids, protected_ids,
                 *, max_protected_kl, max_norm_change, drift_weight):
        groups = [list(g) for g in (refusal_ids, reference_ids, protected_ids)]
        if (any(not g or len(set(g)) != len(g) for g in groups) or len(groups[2]) < 2
                or any(type(i) is not int or not 0 <= i < head_weight.shape[0] for g in groups for i in g)):
            raise ValueError('Readout needs valid distinct refusal/reference tokens and at least two protected tokens')
        if any(set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i)):
            raise ValueError('Readout token groups must be disjoint')
        for value in (max_protected_kl, max_norm_change, drift_weight):
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError('Readout guards/weight need explicit finite nonnegative values')
        if (not jacobians or any(type(k) is not int or k < 0 for k in jacobians)
                or any(j.shape != (head_weight.shape[1], head_weight.shape[1])
                       or j.device != head_weight.device or not j.is_floating_point()
                       or not bool(torch.isfinite(j).all()) for j in jacobians.values())):
            raise ValueError('Readout Jacobian shape/device/finite mismatch')
        self.jacobians, self.norm, self.head = jacobians, norm, head_weight
        self.groups = groups
        self.ids = sum(groups, [])
        indices = torch.tensor(self.ids, device=head_weight.device)
        self.rows = head_weight.detach().index_select(0, indices).float()
        if not bool(torch.isfinite(self.rows).all()):
            raise ValueError('Nonfinite unembedding readout')
        self.max_protected_kl, self.max_norm_change, self.drift_weight = max_protected_kl, max_norm_change, drift_weight

    def normalized(self, hidden, checkpoint):
        if checkpoint not in self.jacobians:
            raise ValueError(f'Missing Jacobian readout checkpoint {checkpoint}')
        if hidden.device != self.head.device:
            raise ValueError('Readout cannot offload residual states')
        transported = hidden[0, -1].float() @ self.jacobians[checkpoint].float().T
        dtype = getattr(getattr(self.norm, 'weight', None), 'dtype', self.head.dtype)
        return self.norm(transported.to(dtype)).float()

    def logits(self, hidden, checkpoint):
        return F.linear(self.normalized(hidden, checkpoint), self.rows)

    def evaluate(self, hidden, baseline, checkpoint):
        logits, base = self.logits(hidden, checkpoint), self.logits(baseline, checkpoint)
        r, q = len(self.groups[0]), len(self.groups[1])
        margin = logits[:r].mean() - logits[r:r+q].mean()
        base_margin = base[:r].mean() - base[r:r+q].mean()
        protected_kl = F.kl_div(F.log_softmax(logits[r+q:], dim=-1),
                                F.softmax(base[r+q:], dim=-1), reduction='sum').clamp_min(0)
        norm = hidden[0, -1].float().norm()
        base_norm = baseline[0, -1].float().norm()
        change = ((norm - base_norm).abs() / base_norm.clamp_min(1e-12))
        kl, change = float(protected_kl), float(change)
        objective = float(margin - base_margin) + self.drift_weight * kl
        return {'objective': objective, 'refusal_margin': float(margin),
                'baseline_refusal_margin': float(base_margin), 'protected_kl': kl, 'norm_change': change,
                'admissible': kl <= self.max_protected_kl and change <= self.max_norm_change}

    def pullback_action(self, name, token_ids, checkpoints):
        if (not token_ids or len(set(token_ids)) != len(token_ids)
                or any(type(i) is not int or not 0 <= i < len(self.head) for i in token_ids)):
            raise ValueError('Pullback action needs valid distinct token IDs')
        # A fixed local-linear candidate basis, not a claim that token narration
        # is the causal behavior. Independent names/groups can target perception.
        def mean_rows(ids):
            idx = torch.tensor(ids, device=self.head.device)
            return self.head.detach().index_select(0, idx).float().mean(0)
        w = mean_rows(token_ids) - mean_rows(self.groups[1])
        # norm(ones) recovers the effective RMS gain up to a positive common
        # scalar. This also handles Qwen3.5's zero-centered (1 + weight) gain.
        # A scalar cancels when the direction is normalized below.
        dtype = getattr(getattr(self.norm, 'weight', None), 'dtype', self.head.dtype)
        gain = self.norm(torch.ones_like(w, dtype=dtype)).detach().float()
        w = w * gain
        bases = {}
        for checkpoint in checkpoints:
            if checkpoint not in self.jacobians:
                raise ValueError(f'Missing pullback Jacobian checkpoint {checkpoint}')
            direction = w @ self.jacobians[checkpoint].float()
            size = direction.norm()
            if not bool(torch.isfinite(size)) or float(size) < 1e-12:
                raise ValueError(f'Degenerate pullback direction: {name} at {checkpoint}')
            bases[checkpoint] = (direction / size).unsqueeze(0)
        return Action(name, bases)
