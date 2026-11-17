# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pure statistics helpers for offline scoring reports. No model or device use."""
import math

_Z = {0.8: 1.2815515655446004, 0.9: 1.6448536269514722, 0.95: 1.959963984540054,
      0.98: 2.326347874043241, 0.99: 2.5758293035489004}


def wilson_interval(successes, total, confidence=0.95):
    """Wilson score interval for a binary pass rate; no normal-approximation at n<10."""
    if total < 1:
        raise ValueError('Wilson interval needs at least one trial')
    if not 0 <= successes <= total:
        raise ValueError('Wilson interval requires 0 <= successes <= total')
    if confidence not in _Z:
        raise ValueError(f'confidence must be one of {sorted(_Z)}')
    z = _Z[confidence]
    n = total
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {'point': p, 'lower': max(0.0, center - half), 'upper': min(1.0, center + half),
            'confidence': confidence, 'method': 'Wilson score interval'}


def cohens_kappa(labels_a, labels_b, confidence=0.95):
    """Cohen's kappa for paired nominal labels plus the Wilson interval on raw agreement.

    Raw percent agreement is misleading when one label dominates: two judges that answer
    "refusal" to everything agree 90% of the time and measure nothing. Kappa subtracts the
    chance-agreement baseline implied by the two marginal label rates. The interval is
    Fleiss' normal-approximation standard error, the standard paired-rater interval.
    A single shared label is degenerate and reported as None, never as perfect.
    """
    pairs = list(zip(labels_a, labels_b))
    total = len(pairs)
    base = {'common': total, 'confidence': confidence, 'method': "Cohen's kappa with Fleiss CI"}
    if total < 2:
        return {**base, 'raw_agreement': None, 'kappa': None, 'kappa_interval': None,
                'agreement_wilson': None, 'note': 'Fewer than two paired labels.'}
    observed = sum(1 for first, second in pairs if first == second) / total
    labels = set(labels_a) | set(labels_b)
    expected = sum((labels_a.count(label) / total) * (labels_b.count(label) / total) for label in labels)
    if 1 - expected <= 1e-12:
        return {**base, 'raw_agreement': round(observed, 4), 'kappa': None, 'kappa_interval': None,
                'agreement_wilson': wilson_interval(int(observed * total), total, confidence),
                'note': 'Degenerate: one label carries all rows on both sides, so chance agreement equals 1 and kappa is undefined. Consensus here proves nothing.'}
    kappa = (observed - expected) / (1 - expected)
    z = _Z[confidence]
    standard_error = math.sqrt(observed * (1 - observed)) / ((1 - expected) * math.sqrt(total))
    interval = [max(-1.0, round(kappa - z * standard_error, 4)), min(1.0, round(kappa + z * standard_error, 4))]
    return {**base, 'raw_agreement': round(observed, 4), 'chance_agreement': round(expected, 4),
            'kappa': round(kappa, 4), 'kappa_interval': interval,
            'agreement_wilson': wilson_interval(int(observed * total), total, confidence),
            'note': 'Kappa is chance-corrected agreement; report it alongside raw agreement.'}


def family_sign_test(improved, regressed):
    """Exact two-sided sign test over independent paired units (families)."""
    if type(improved) is not int or type(regressed) is not int or improved < 0 or regressed < 0:
        raise ValueError('sign test counts must be nonnegative integers')
    n = improved + regressed
    if n == 0:
        return {'improved': improved, 'regressed': regressed, 'informative_units': 0,
                'p_two_sided': 1.0, 'method': 'exact sign test on independent families (no informative units)'}
    tail = sum(math.comb(n, i) for i in range(max(improved, regressed), n + 1)) / 2 ** n
    return {'improved': improved, 'regressed': regressed, 'informative_units': n,
            'p_two_sided': min(1.0, 2 * tail), 'method': 'exact sign test on independent families'}


def family_bootstrap(pass_fractions, seed=0, resamples=10000, confidence=0.95):
    """Resample families with replacement and bootstrap the family-averaged pass rate.

    Families are the independent units; items inside a family are not. Deterministic
    for a given seed, so a release evidence file can be reproduced.
    """
    import random
    if not pass_fractions or any(not 0.0 <= f <= 1.0 for f in pass_fractions):
        raise ValueError('pass_fractions must be values in [0,1]')
    if type(seed) is not int or resamples < 2:
        raise ValueError('seed must be an int and resamples >= 2')
    alpha = 1 - confidence
    rng = random.Random(seed)
    k = len(pass_fractions)
    means = sorted(sum(rng.choice(pass_fractions) for _ in range(k)) / k for _ in range(resamples))
    lo = means[max(0, int(alpha / 2 * resamples))]
    hi = means[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return {'families': len(pass_fractions), 'resamples': resamples, 'seed': seed,
            'mean': sum(pass_fractions) / len(pass_fractions), 'lower': lo, 'upper': hi,
            'confidence': confidence, 'method': 'family-level nonparametric bootstrap'}


def paired_permutation_test(differences, seed=0, resamples=None, confidence=0.95):
    """Two-sided sign-flip permutation test on paired per-unit score differences.

    ``differences`` are after-minus-before metrics per independent unit. Exact for
    n <= 16 units, otherwise a seeded permutation of the same size used twice.
    """
    import random
    values = [float(d) for d in differences]
    n = len(values)
    if n < 2:
        return {'units': n, 'observed_mean_difference': (sum(values) / n) if n else None,
                'p_two_sided': 1.0, 'confidence': confidence,
                'method': 'sign-flip permutation (no power: fewer than two paired units)'}
    if not all(math.isfinite(v) for v in values):
        raise ValueError('paired permutation needs finite differences')
    observed = sum(values) / n
    if n <= 16:
        flips = [sign * abs(sum((-1 if mask & i else 1) * v for i, v in enumerate(values))) / n
                 for mask in range(1 << n) for sign in (1, -1)]
        p = (1 + sum(1 for s in flips if s >= abs(observed))) / len(flips)
        method = 'exact sign-flip permutation'
    else:
        if resamples is None:
            resamples = 10000
        if type(resamples) is not int or resamples < 100:
            raise ValueError('resamples must be an integer >= 100 for large samples')
        rng = random.Random(seed)
        hits = 0
        for _ in range(resamples):
            flags = [1 if rng.random() < 0.5 else -1 for _ in values]
            if abs(sum(f * v for f, v in zip(flags, values))) / n >= abs(observed):
                hits += 1
        p = (1 + hits) / resamples
        method = f'seeded sign-flip permutation ({resamples} resamples)'
    return {'units': n, 'observed_mean_difference': observed, 'p_two_sided': min(1.0, p),
            'confidence': confidence, 'method': method}


def cohens_h(successes_a, total_a, successes_b, total_b):
    """Cohen's h for two proportions: signed 2*asin(sqrt(p)) difference.

    Positive h means the second proportion is larger. Undefined (None) only
    when a denominator is zero; a zero-rate side uses the asin(0)=0 endpoint,
    which is the standard convention.
    """
    if total_a <= 0 or total_b <= 0:
        raise ValueError('cohens_h needs at least one trial per side')
    pa = successes_a / total_a
    pb = successes_b / total_b
    return 2 * math.asin(math.sqrt(pb)) - 2 * math.asin(math.sqrt(pa))


def holm_adjust(p_values):
    """Holm-Bonferroni step-down adjustment of a list of p-values.

    Returns adjusted p-values in the original order, each clipped to 1.0.
    A family of k tests is thus kept at family-wise alpha under the (valid
    but conservative) Holm procedure.
    """
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [1.0] * n
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted
