# SPDX-License-Identifier: AGPL-3.0-or-later
import torch
import torch.nn.functional as F
from .data import digest
from .errors import DirectionRejected


def token_kl(base, candidate):
    if base.shape != candidate.shape or base.ndim != 2 or base.shape[0] == 0:
        raise ValueError('KL needs matching nonempty [tokens, vocabulary] logits')
    if not torch.isfinite(base).all() or not torch.isfinite(candidate).all():
        raise ValueError('Nonfinite logits')
    # Explicit dtype alignment avoids mixed-dtype oneDNN/XPU reinterpretation bugs.
    p = F.log_softmax(base.float(), dim=-1)
    q = F.log_softmax(candidate.float(), dim=-1)
    return (p.exp() * (p - q)).sum(dim=-1).clamp_min(0)


def full_reply_kl(base, candidate, answer_mask):
    if answer_mask.ndim != 1 or answer_mask.shape[0] != base.shape[0] or answer_mask.dtype != torch.bool or not answer_mask.any():
        raise ValueError('Missing or invalid answer-token mask')
    return float(token_kl(base, candidate)[answer_mask].mean().item())


def directions_from_languages(good, bad, rank):
    if not good or set(good) != set(bad) or rank < 1 or rank > len(good):
        raise ValueError('Need matched good/bad languages and sufficient rank')
    deltas = []
    shape = None
    for language in sorted(good):
        g, b = good[language].float(), bad[language].float()
        if g.shape != b.shape or g.ndim != 2 or not torch.isfinite(g).all() or not torch.isfinite(b).all():
            raise ValueError('Means must match [layers, residual width]')
        if shape is not None and shape != g.shape:
            raise ValueError('Language shapes differ')
        shape = g.shape
        delta = b - g
        norms = delta.norm(dim=-1, keepdim=True)
        if (norms <= 1e-8).any():
            raise DirectionRejected('Degenerate refusal direction')
        # One normalized vector per language: language dataset size cannot dominate.
        deltas.append(delta / norms)
    matrix = torch.stack(deltas, dim=1)  # layers, languages, width
    _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    if rank > vh.shape[1] or (singular[:, rank - 1] <= 1e-6).any():
        raise DirectionRejected('Requested directions are linearly dependent')
    return vh[:, :rank, :].contiguous()


def stable_directions(records, rank, policy, seed, expected_tasks=None, fitter=None):
    """Split whole matched task families, never split translations/paraphrases.

    Use principal-angle overlap, invariant to SVD signs or basis rotation.
    Returns small CPU statistics; no model-weight offload is involved.
    """
    records = list(records)
    if expected_tasks is not None:
        expected = {(t['language'], t['family_id'], t['label']) for t in expected_tasks}
        actual = {(t['language'], t['family_id'], t['label']) for t, _ in records}
        missing = expected - actual
        if missing:
            raise ValueError(f'Missing confirmed direction coverage: {sorted(missing)}')
    grouped = {}
    counts = {}
    for task, residual in records:
        language, label = task['language'], task['label']
        family = task.get('family_id')
        if not isinstance(family, str) or not family.strip() or label not in {'good', 'bad'}:
            raise ValueError('Direction coverage requires matched family_id and good/bad labels')
        grouped.setdefault(language, {}).setdefault(family, {}).setdefault(label, []).append(residual)
        counts.setdefault(language, {'good': 0, 'bad': 0})[label] += 1
    if len(grouped) < 2 or any(min(c.values()) < policy.min_direction_examples for c in counts.values()):
        raise ValueError('Insufficient confirmed direction sample coverage')
    if any(len(families) < policy.min_direction_families or any(set(v) != {'good', 'bad'} for v in families.values()) for families in grouped.values()):
        raise ValueError('Insufficient matched good/bad direction family coverage')
    # One global assignment so translations of a family cannot land in both halves.
    family_ids = sorted({f for families in grouped.values() for f in families}, key=lambda f: digest([seed, f]))
    halves = {f: i % 2 for i, f in enumerate(family_ids)}
    def fit(half=None):
        means = {'good': {}, 'bad': {}}
        for language, families in grouped.items():
            for label in means:
                # Equal family weight prevents a heavily paraphrased topic dominating.
                values = [torch.stack(labels[label]).mean(0) for f, labels in families.items() if half is None or halves[f] == half]
                if not values:
                    raise ValueError('Direction subset lacks language coverage; rebalance authored families')
                means[label][language] = torch.stack(values).mean(0)
        if fitter is not None:
            selected = [(task, residual) for task, residual in records
                        if half is None or halves[task['family_id']] == half]
            return fitter(selected, rank)
        return directions_from_languages(means['good'], means['bad'], rank)
    first, second = fit(0), fit(1)
    overlaps = torch.linalg.svdvals(first @ second.transpose(-1, -2)).clamp(0, 1)
    minimum = float(overlaps.min())
    report = {'accepted_counts': counts, 'family_counts': {k: len(v) for k, v in grouped.items()},
              'subset_assignment': halves, 'minimum_overlap': minimum, 'layer_principal_overlaps': overlaps.tolist(),
              'required_overlap': policy.min_direction_overlap, 'seed': seed}
    if minimum < policy.min_direction_overlap:
        raise DirectionRejected(f'Direction stability failed: minimum subspace overlap {minimum:.6f}')
    return fit(), report
