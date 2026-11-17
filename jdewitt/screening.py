# SPDX-License-Identifier: AGPL-3.0-or-later
"""Optional refusal-margin screening: cheap forward-only candidate filtering.

Screening measures, with one prefill per probe prompt and no decoding, how far
a candidate edit moves the answer-start logit margin between explicit refusal
token IDs and explicit reference token IDs, harmful prompts relative to
harmless ones. It exists to spend search budget better: candidates that did not
move the margin at all are pruned before paying for full generation and
judging. It is a heuristic prefilter, never a certification: every screened-in
candidate still faces the unchanged generation + judging gate, and releases are
certified there. It has not been validated on a real checkpoint.
"""
from math import isfinite

FIELDS = ('refusal_tokens', 'reference_tokens', 'min_margin_drop')


def validate_screening(spec):
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) != set(FIELDS):
        raise ValueError('screening requires exactly refusal_tokens, reference_tokens and min_margin_drop')
    for key in ('refusal_tokens', 'reference_tokens'):
        values = spec[key]
        if (not isinstance(values, list) or not values or any(type(v) is not int or v < 0 for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f'screening.{key} must be a nonempty list of distinct nonnegative token IDs')
    if set(spec['refusal_tokens']) & set(spec['reference_tokens']):
        raise ValueError('screening refusal and reference token sets must be disjoint')
    drop = spec['min_margin_drop']
    if type(drop) not in (int, float) or not isfinite(drop) or drop <= 0:
        raise ValueError('screening.min_margin_drop must be finite and positive')
    return spec


def margin_differential(baseline, screened, baseline_harmless, screened_harmless):
    """Mean refusal-margin drop on harmful prompts minus the drop on harmless ones.

    A harmless-prompt margin collapse is generic damage, not refusal removal,
    so it is subtracted. Equal nonempty margin records on both sides are
    required; partial screening never silently rebases the statistic.
    """
    for name, pair in (('harmful', (baseline, screened)), ('harmless', (baseline_harmless, screened_harmless))):
        base, candidate = pair
        if not base or len(base) != len(candidate):
            raise ValueError(f'Screening {name} prompts need equal nonempty margin records')
    return (sum(b - c for b, c in zip(baseline, screened)) / len(baseline)
            - sum(b - c for b, c in zip(baseline_harmless, screened_harmless)) / len(baseline_harmless))
