# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pure helpers for the short-decode stage-1 trial funnel.

No torch and no model imports: the semantics are hand-checkable. Stage 1 is a
heuristic prefilter, never a certification: its probe answers never enter the
answer store, the gates or any report a release cites.
"""
import math


def validate_search_stages(spec, max_tokens):
    """Normalize and validate the optional ``search_stages`` run field.

    The three core keys are the short-decode probe. ``confirm_fraction`` and
    ``confirm_keep_top`` are an optional second rung: a full-budget
    evaluation of a stratified selection subset, halved again before the
    trial pays for the entire selection suite. Both must be strictly below 1:
    a confirm rung that never prunes or never subsets just doubles cost.
    """
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError('search_stages must be an object')
    base = {'probe_fraction', 'probe_tokens', 'keep_top'}
    optional = {'confirm_fraction', 'confirm_keep_top'}
    keys = set(spec)
    if not base <= keys or keys - base - optional:
        raise ValueError('search_stages needs probe_fraction, probe_tokens and keep_top, optionally confirm_fraction and confirm_keep_top')
    if ('confirm_fraction' in keys) != ('confirm_keep_top' in keys):
        raise ValueError('search_stages.confirm_fraction and confirm_keep_top must be given together')
    pf, pt, kt = spec['probe_fraction'], spec['probe_tokens'], spec['keep_top']
    if type(pf) not in (int, float) or not math.isfinite(pf) or not 0 < pf <= 1:
        raise ValueError('search_stages.probe_fraction must be in (0, 1]')
    if type(pt) is not int or not 0 < pt <= max_tokens:
        raise ValueError(f'search_stages.probe_tokens must be a positive integer within max_tokens ({max_tokens})')
    if type(kt) not in (int, float) or not math.isfinite(kt) or not 0 < kt <= 1:
        raise ValueError('search_stages.keep_top must be in (0, 1]')
    out = {'probe_fraction': float(pf), 'probe_tokens': pt, 'keep_top': float(kt)}
    if 'confirm_fraction' in keys:
        cf, ck = spec['confirm_fraction'], spec['confirm_keep_top']
        if type(cf) not in (int, float) or not math.isfinite(cf) or not 0 < cf < 1:
            raise ValueError('search_stages.confirm_fraction must be in (0, 1) — a full-suite confirm evaluates the whole suite twice')
        if type(ck) not in (int, float) or not math.isfinite(ck) or not 0 < ck < 1:
            raise ValueError('search_stages.confirm_keep_top must be in (0, 1) — a confirm rung that never prunes only adds cost')
        out['confirm_fraction'] = float(cf)
        out['confirm_keep_top'] = float(ck)
    return out


def stratified_sample(rows, fraction):
    """Deterministic family-stratified subsample of suite rows.

    Stratifies on ``family_id`` where authored, otherwise on ``kind``; every
    non-empty stratum contributes at least one row and picks at
    ``round(fraction * size)`` (capped by the stratum size) at an even stride
    over suite order. No RNG is serialized: resume and rerun draw the same
    slice, and the result preserves suite order.
    """
    strata = {}
    for row in rows:
        strata.setdefault(row.get('family_id') or row['kind'], []).append(row)
    picked = set()
    for key in sorted(strata):
        group = strata[key]
        want = min(len(group), max(1, round(fraction * len(group))))
        step = len(group) / want
        for i in range(want):
            picked.add(group[round(i * step)]['id'])
    return [row for row in rows if row['id'] in picked]


def stage1_score(refused, total):
    """Fraction of probe rows still refusing inside the short window; lower is better."""
    if total <= 0:
        raise ValueError('Stage-1 score needs at least one probe row')
    return refused / total


def stage1_pruned(score, prior_scores, keep_top):
    """Deterministic successive-halving keep-line decision.

    ``prior_scores`` holds the stage-1 score of every earlier trial that
    actually reached stage 2 (full-budget evaluation). The current trial is
    pruned only once at least ``ceil(1/keep_top)`` (minimum 2) such priors
    exist, and only when its score is strictly worse than the cut line — a
    tie always survives, so pruning never rests on measurement noise alone.
    """
    needed = max(2, math.ceil(1 / keep_top))
    if len(prior_scores) < needed:
        return False
    keep = max(1, math.floor(keep_top * (len(prior_scores) + 1)))
    ordered = sorted(prior_scores)
    return score > ordered[keep - 1]


def validate_candidate_probe(spec, trials):
    """Normalize and validate the optional ``candidate_probe`` run field.

    The probe reuses the stage-1 short-decode signal, so a run that sets it
    must also set ``search_stages`` (checked by the caller that knows both).
    """
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) != {'samples', 'enqueue_top'}:
        raise ValueError('candidate_probe requires exactly samples and enqueue_top')
    samples, top = spec['samples'], spec['enqueue_top']
    if type(samples) is not int or not 1 <= samples <= 4000:
        raise ValueError('candidate_probe.samples must be an integer in [1, 4000]')
    if type(top) is not int or not 1 <= top <= min(samples, trials):
        raise ValueError(f'candidate_probe.enqueue_top must be an integer in [1, min(samples, trials={trials})]')
    return {'samples': samples, 'enqueue_top': top}
