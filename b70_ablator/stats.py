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
