# SPDX-License-Identifier: AGPL-3.0-or-later
"""Static alignment-matrix experiment: SVD of ``W_instruct - W_base``.

This module only does weight algebra. It never loads a model, never runs a
forward pass, and never claims a behavioural effect: every report carries
``removal_measured: false``.

The alignment delta of a writeback tensor is

    dW = W_instruct - W_base

with ``W`` in the HF ``[out_features, in_features]`` layout. ``dW`` is
decomposed with an economy SVD ``dW = U S Vt``. Removing the top-``k``
directions means subtracting the rank-``k`` truncation from the instruct
weights:

    W_k = W_instruct - U[:, :k] S[:k] Vt[:k]

so the removed matrix is exactly the truncated part of ``dW`` and its squared
Frobenius norm is exactly ``sum(S[:k] ** 2)``. That identity is the atom of the
static check; it is recomputed from the full matrices, not assumed.

Two ratios are reported per ``k``:

  * ``removed_energy_fraction`` — how much of the alignment delta's energy the
    removal takes (higher = more removed);
  * ``delta_residual_norm_ratio`` — ``||dW - dW_k||_F / ||dW||_F``, the residual
    of the alignment delta (lower = more removed; equals ``sqrt(1 - removed)``);
  * ``weight_norm_ratio`` — ``||W_k||_F / ||W_instruct||_F``, the share of the
    instruct weight matrix that survives. It sits just above 1.0 because the
    alignment delta is tiny next to ``W``; it is reported because "instruction
    energy kept" is a weight-side question, not a delta-side one.

The "refusal-consistent" share is proxied statically by a public abliterated
twin: its own delta ``dW_twin = W_twin - W_instruct`` is a weight-space edit
that a third party already applied for refusal removal. ``twin_capture`` is the
share of that edit which lies inside the removed ``U_k`` / ``V_k`` subspace.
This is a weight-space proxy, not a measurement of refusal: the twin's delta
also carries its own fine-tuning and quantization, and the instructions' delta
also carries instruction-following. See ``CAVEATS``.
"""
from __future__ import annotations

import numpy as np

SCHEMA = 'jdewitt-alignment-matrix-v1'
WRITEBACK_SUFFIXES = (
    ('self_attn.o_proj.weight', 'attn'),
    ('mlp.down_proj.weight', 'ffn'),
)
DEFAULT_KS = (1, 2, 4, 8)

CAVEATS = (
    'removal_measured is false: nothing here observes what the model answers.',
    'The alignment delta also carries instruction-following, so a capability cost '
    'is expected and is NOT measured by this experiment.',
    'twin_capture reads a public abliterated checkpoint as a weight-space proxy for '
    'the refusal-consistent component; it is not a refusal measurement, and the '
    'twin delta also contains that checkpoint\'s own fine-tuning and quantization.',
    'Singular directions are statistical (largest delta energy), not causal; a '
    'direction being removed first does not make it a refusal direction.',
    'The curve is the result. Any k below is a displayed point on it, not a '
    'behaviourally validated winner.',
)


def _svd(matrix):
    """Economy SVD via the smaller Gram matrix.

    ``dW dW^T = U S^2 U^T`` gives the left singular vectors and squared singular
    values from a symmetric eigendecomposition; the right vectors follow from
    ``V = dW^T U / S``. For the writeback shapes here (896 x 896 and 896 x 4864)
    this is the same decomposition LAPACK would return, at a fraction of the
    cost on a loaded CPU. Small singular values are the least accurate (they are
    squares), which is why the curve only reads the top ``k <= 8``.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    m, n = matrix.shape
    if m <= n:
        gram = matrix @ matrix.T
        values, left = np.linalg.eigh(gram)
        order = np.argsort(values)[::-1]
        values, left = values[order], left[:, order]
        singular = np.sqrt(np.maximum(values, 0.0))
        safe = np.where(singular > 0, singular, 1.0)
        right = (matrix.T @ left) / safe
        return left, singular, right.T
    transposed = matrix.T
    left, singular, vt = _svd(transposed)
    return vt.T, singular, left.T


def writeback_tensors(keys):
    """Ordered ``(name, kind, layer)`` for every writeback tensor present."""
    found = []
    for name in keys:
        if not name.startswith('model.layers.'):
            continue
        for suffix, kind in WRITEBACK_SUFFIXES:
            if name.endswith(suffix):
                layer = int(name.split('.')[2])
                found.append((name, kind, layer))
    found.sort(key=lambda row: (row[2], row[1]))
    return found


def load_matrix(path, name, framework='pt'):
    """One tensor as float64 ``[out, in]``. bf16 is read through torch."""
    from safetensors import safe_open

    with safe_open(str(path), framework=framework) as handle:
        tensor = handle.get_tensor(name)
    if framework == 'pt':
        tensor = tensor.detach().to('cpu', dtype=None).double().numpy()
    else:
        tensor = np.asarray(tensor, dtype=np.float64)
    if tensor.ndim != 2:
        raise ValueError(f'{name} is {tensor.ndim}-D, expected a 2-D weight matrix')
    return tensor


def _subspace_overlap(left, right):
    """Mean squared cosine between the column spaces of two orthonormal bases."""
    if left.shape[1] != right.shape[1]:
        raise ValueError('overlap needs equally sized bases')
    cos = left.T @ right
    return float(np.square(cos).sum() / left.shape[1])


def tensor_curve(base, instruct, twin, ks=DEFAULT_KS, twin_top=None, return_factors=False):
    """Per-tensor SVD curve. ``twin`` may be None to skip the twin proxy.

    Returns a dict with the singular values and, for each k, the energy ratios
    and the twin-capture proxy. The algebra identities are returned under
    ``checks`` so the caller can verify them without re-deriving anything.
    With ``return_factors`` the economy SVD factors (``u, s, vt``) are returned
    as well, so a caller can emit a delta without a second decomposition.
    """
    if base.shape != instruct.shape:
        raise ValueError(f'shape mismatch: base {base.shape} vs instruct {instruct.shape}')
    ks = tuple(sorted({int(k) for k in ks}))
    if not ks or ks[0] < 1:
        raise ValueError('k values must be positive')
    delta = instruct - base
    u, s, vt = _svd(delta)
    if ks[-1] > s.size:
        raise ValueError(f'k={ks[-1]} exceeds rank {s.size}')

    total_energy = float(np.square(s).sum())
    instruct_norm = float(np.linalg.norm(instruct))
    delta_norm = float(np.linalg.norm(delta))

    twin_u = twin_vt = twin_delta = None
    twin_delta_norm = None
    if twin is not None:
        if twin.shape != instruct.shape:
            raise ValueError(f'twin shape {twin.shape} != instruct {instruct.shape}')
        twin_delta = twin - instruct
        twin_u, twin_s, twin_vt = _svd(twin_delta)
        twin_delta_norm = float(np.linalg.norm(twin_delta))
        twin_top = ks[-1] if twin_top is None else min(int(twin_top), twin_s.size)

    curves = {}
    max_energy_error = 0.0
    max_energy_rel_error = 0.0
    max_remain_error = 0.0
    for k in ks:
        removed = (u[:, :k] * s[:k]) @ vt[:k]
        edited = instruct - removed
        removed_energy = float(np.square(s[:k]).sum())
        measured_energy = float(np.square(np.linalg.norm(removed)))
        remain = delta - removed
        row = {
            'k': k,
            'removed_energy': removed_energy,
            'removed_energy_fraction': removed_energy / total_energy if total_energy else 0.0,
            'delta_residual_norm_ratio': float(np.linalg.norm(remain) / delta_norm) if delta_norm else 0.0,
            'weight_norm_ratio': float(np.linalg.norm(edited) / instruct_norm) if instruct_norm else 0.0,
        }
        # The two identities: removed energy is the squared removed singulars,
        # and the remaining delta norm is sqrt(1 - removed fraction).
        error = abs(measured_energy - removed_energy)
        max_energy_error = max(max_energy_error, error)
        if removed_energy:
            max_energy_rel_error = max(max_energy_rel_error, error / removed_energy)
        if delta_norm:
            expected = (1.0 - removed_energy / total_energy) ** 0.5
            max_remain_error = max(max_remain_error, abs(row['delta_residual_norm_ratio'] - expected))
        if twin is not None:
            top = min(k, twin_top)
            uk, vtk = u[:, :k], vt[:k]
            ut, vtt = twin_u[:, :top], twin_vt[:top]
            left_capture = float(np.square(np.linalg.norm(uk.T @ twin_delta)))
            right_capture = float(np.square(np.linalg.norm(twin_delta @ vtk.T)))
            row.update({
                'twin_left_capture': left_capture / (twin_delta_norm ** 2),
                'twin_right_capture': right_capture / (twin_delta_norm ** 2),
                'twin_left_overlap': _subspace_overlap(uk, ut),
                'twin_right_overlap': _subspace_overlap(vtk.T, vtt.T),
            })
        curves[str(k)] = row

    result = {
        'shape': list(instruct.shape),
        'instruct_norm': instruct_norm,
        'delta_norm': delta_norm,
        'delta_energy': total_energy,
        'twin_delta_norm': twin_delta_norm,
        'singular_values': [float(value) for value in s[:max(ks)]],
        'curves': curves,
        'checks': {'max_removed_energy_error': max_energy_error,
                   'max_removed_energy_rel_error': max_energy_rel_error,
                   'max_delta_remain_error': max_remain_error},
    }
    if return_factors:
        return result, (u[:, :max(ks)], s[:max(ks)], vt[:max(ks)])
    return result


def aggregate(rows, ks=DEFAULT_KS):
    """Energy-weighted aggregate over per-tensor curves.

    ``removed_energy_fraction`` weights each tensor by its delta energy, the
    residual ratio is the ratio of total squared norms, and ``twin_capture``
    weights by the twin delta energy. Tensors without a twin are skipped only
    for the twin fields.
    """
    ks = tuple(sorted({int(k) for k in ks}))
    total_delta = sum(row['delta_energy'] for row in rows)
    total_edited = {k: 0.0 for k in ks}
    total_instruct = sum(row['instruct_norm'] ** 2 for row in rows)
    twin_rows = [row for row in rows if row['twin_delta_norm'] is not None]
    total_twin = sum(row['twin_delta_norm'] ** 2 for row in twin_rows)
    out = {str(k): {} for k in ks}
    for k in ks:
        removed = 0.0
        remain = 0.0
        for row in rows:
            curve = row['curves'][str(k)]
            removed += curve['removed_energy']
            remain += row['delta_energy'] - curve['removed_energy']
            total_edited[k] += curve['weight_norm_ratio'] ** 2 * row['instruct_norm'] ** 2
        out[str(k)]['removed_energy_fraction'] = removed / total_delta if total_delta else 0.0
        out[str(k)]['removed_energy'] = removed
        out[str(k)]['delta_residual_norm_ratio'] = (remain / total_delta) ** 0.5 if total_delta else 0.0
        out[str(k)]['weight_norm_ratio'] = (total_edited[k] / total_instruct) ** 0.5 if total_instruct else 0.0
        for side in ('left', 'right'):
            weighted = sum(row['curves'][str(k)][f'twin_{side}_capture'] * row['twin_delta_norm'] ** 2
                           for row in twin_rows)
            out[str(k)][f'twin_{side}_capture'] = weighted / total_twin if total_twin else 0.0
            weighted_overlap = sum(row['curves'][str(k)][f'twin_{side}_overlap'] * row['twin_delta_norm'] ** 2
                                   for row in twin_rows)
            out[str(k)][f'twin_{side}_overlap'] = weighted_overlap / total_twin if total_twin else 0.0
        out[str(k)]['tensors'] = len(rows)
    return out


def choose_k(aggregate_curves, ks=DEFAULT_KS, residual_floor=0.95):
    """Pre-registered static rule: most twin capture at ``weight >= floor``.

    This reads the curve only; it does not consult any behaviour. The floor is
    the minimum share of the instruct weight norm that must survive. If no k
    meets the floor, the smallest k is returned with ``feasible: False``.
    """
    ks = tuple(sorted({int(k) for k in ks}))
    feasible = [k for k in ks if aggregate_curves[str(k)]['weight_norm_ratio'] >= residual_floor]
    if not feasible:
        return {'k': ks[0], 'feasible': False, 'residual_floor': residual_floor,
                'rule': 'max twin_capture subject to weight_norm_ratio >= floor; '
                        'no k met the floor, smallest k returned'}
    best = max(feasible, key=lambda k: (aggregate_curves[str(k)]['twin_left_capture']
                                        + aggregate_curves[str(k)]['twin_right_capture'], -k))
    return {'k': best, 'feasible': True, 'residual_floor': residual_floor,
            'rule': 'max twin_capture (left+right) subject to weight_norm_ratio >= floor',
            'twin_capture': (aggregate_curves[str(best)]['twin_left_capture']
                             + aggregate_curves[str(best)]['twin_right_capture']),
            'weight_norm_ratio': aggregate_curves[str(best)].get('weight_norm_ratio'),
            'removed_energy_fraction': aggregate_curves[str(best)].get('removed_energy_fraction'),
            'delta_residual_norm_ratio': aggregate_curves[str(best)].get('delta_residual_norm_ratio')}
