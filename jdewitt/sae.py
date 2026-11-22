# SPDX-License-Identifier: AGPL-3.0-or-later
"""SAE feature ablation as the rank-1 removal this repo already runs (P5, CODE contract).

CONTRACT. An SAE feature edit IS a rank-1 projection removal applied post hoc to a
weight. For an output-side projection ``W [out, in]`` and a feature direction
``d [out]`` (one column of ``W_dec``, i.e. the direction the feature writes into the
residual stream) the edit is

    W <- W - s * d (d^T W) / (d^T d)

which is exactly ``spanfit.removal_delta`` / the ``AblatedLinear`` A/B projection. A
feature is therefore only a NEW THING if its direction is not the contrast/oracle
axis: if the feature direction IS the oracle direction, this lane is the previous
lane with extra steps -- same math, same edit, one more fit to train and one more
checkpoint to keep. That is why an arm built on this lane must carry
``cos_to_axis(feature_direction, oracle_axis)`` in its card next to every number it
reports, and must not claim novelty on top of a removal the contrast fit already
performs. A cosine near 1 is a novelty claim already measured false; a cosine near 0
is the only case where the extra compute buys something, and then it has to explain
why a direction the contrast fit does not see carries the behaviour.

No model, no weights and no claims live here: shapes, refusals, and the
realized-vs-intended cosine of an edit that already happened. GQA is handled by the
side, not by a table: grouped-query projections write to ``heads * head_dim != hidden``,
so a residual-stream feature direction addresses their READ side (``side='input'``)
and never their output side.

The same quantity ``realized_cos`` measures is what the codec lane
(``ggufio.preview_edit``, ``--min-edit-cos``) reports on re-encoded bytes: an edit that
survives the arithmetic but not the storage is caught there, not here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch

SIDES = ('output', 'input')
SAE_TENSORS = ('W_enc', 'W_dec', 'b_enc', 'b_dec')


def _finite(tensor: torch.Tensor, what: str) -> None:
    """Refuse a tensor holding NaN/Inf, naming the tensor and how much of it is bad."""
    if not torch.isfinite(tensor).all():
        bad = int((~torch.isfinite(tensor)).sum())
        raise ValueError(f'{what} has nonfinite values ({bad} of {tensor.numel()} entries are '
                         f'NaN/Inf); refusing to compute a projection on it')


def _matrix(weight) -> torch.Tensor:
    if not torch.is_tensor(weight):
        raise TypeError(f'weight is {type(weight).__name__}, not a torch.Tensor')
    if weight.ndim != 2:
        raise ValueError(f'weight is {weight.ndim}-d; an ablation needs a 2-d [out, in] matrix')
    if not weight.is_floating_point():
        raise ValueError(f'weight has dtype {weight.dtype}; an ablation needs a floating dtype')
    _finite(weight, 'weight tensor')
    return weight


@dataclass(frozen=True)
class SAE:
    """Minimal SAE: ``W_enc [F, H]``, ``W_dec [H, F]``, ``b_enc [F]``, ``b_dec [H]``.

    Shapes and finiteness are checked once, at construction, so every later use
    (a direction, a feature set) can assume a consistent ``F``/``H`` and no NaN.
    Nothing here encodes or decodes activations: the lane needs the decoder
    directions, and an SAE that only stores four tensors is what a fit hands back.
    """

    W_enc: torch.Tensor
    W_dec: torch.Tensor
    b_enc: torch.Tensor
    b_dec: torch.Tensor

    def __post_init__(self) -> None:
        for name in SAE_TENSORS:
            tensor = getattr(self, name)
            if not torch.is_tensor(tensor):
                raise TypeError(f'SAE tensor {name} is {type(tensor).__name__}, not a torch.Tensor')
            if not tensor.is_floating_point():
                raise ValueError(f'SAE tensor {name} has dtype {tensor.dtype}; projections need floating point')
            _finite(tensor, f'SAE tensor {name}')
        if self.W_enc.ndim != 2:
            raise ValueError(f'W_enc is {tuple(self.W_enc.shape)}; it must be 2-d [F, H]')
        if self.W_dec.ndim != 2:
            raise ValueError(f'W_dec is {tuple(self.W_dec.shape)}; it must be 2-d [H, F]')
        features, width = int(self.W_enc.shape[0]), int(self.W_enc.shape[1])
        if features < 1 or width < 1:
            raise ValueError(f'SAE has an empty axis: W_enc {tuple(self.W_enc.shape)}')
        if tuple(self.W_dec.shape) != (width, features):
            raise ValueError(f'W_dec is {tuple(self.W_dec.shape)}, but W_enc {tuple(self.W_enc.shape)} '
                             f'requires [{width}, {features}]')
        if tuple(self.b_enc.shape) != (features,):
            raise ValueError(f'b_enc is {tuple(self.b_enc.shape)}; W_enc has {features} features')
        if tuple(self.b_dec.shape) != (width,):
            raise ValueError(f'b_dec is {tuple(self.b_dec.shape)}; W_enc has width {width}')

    @property
    def features(self) -> int:
        return int(self.W_enc.shape[0])

    @property
    def width(self) -> int:
        return int(self.W_enc.shape[1])

    def direction(self, feature: int) -> torch.Tensor:
        """The feature's residual-space direction: column ``feature`` of ``W_dec`` [H]."""
        if type(feature) is not int:
            raise TypeError(f'feature id is {type(feature).__name__}, not int')
        if not 0 <= feature < self.features:
            raise IndexError(f'feature {feature} is outside [0, {self.features})')
        d = self.W_dec[:, feature].detach().clone()
        if float(d.norm()) == 0.0:
            # A pruned / dead decoder column: there is no direction, so an ablation
            # along it would be a silent zero edit reported as a feature removal.
            raise ValueError(f'SAE feature {feature} has a zero-norm decoder column; '
                             f'there is no direction to ablate')
        return d

    def directions(self, features) -> torch.Tensor:
        """Stack directions for the selected features as ``[G, H]`` (not normalized)."""
        ids = [int(f) for f in features]
        if not ids:
            raise ValueError('no SAE features selected')
        if len(set(ids)) != len(ids):
            raise ValueError(f'duplicate feature ids in {ids}; a repeated direction is a '
                             f'rank deficit, not a wider removal')
        return torch.stack([self.direction(f) for f in ids])


def load(path) -> SAE:
    """Load a minimal SAE from a ``torch.save`` archive of its four tensors.

    Accepts ``{'W_enc','W_dec','b_enc','b_dec'}`` directly or under a
    ``'state_dict'`` key; extra keys (fit metadata, hyperparameters) are ignored.
    Weights-only loading, CPU, no model class and no code execution from the file.
    """
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(payload, dict) and isinstance(payload.get('state_dict'), dict):
        payload = payload['state_dict']
    if not isinstance(payload, dict):
        raise ValueError(f'{path}: SAE archive is {type(payload).__name__}, expected a dict of '
                         f'{SAE_TENSORS}')
    missing = [name for name in SAE_TENSORS if name not in payload]
    if missing:
        raise ValueError(f'{path}: SAE archive is missing {missing}; it has {sorted(payload)}')
    return SAE(**{name: payload[name] for name in SAE_TENSORS})


def feature_ablation(direction, weight, strength: float = 1.0, side: str = 'output') -> torch.Tensor:
    """Delta that removes one feature direction from a weight (``spanfit.removal_delta``).

    ``side='output'`` (the contract form): ``dW = -s d (d^T W) / (d^T d)`` with
    ``d`` in ``R^out`` -- the component the module WRITES into the residual stream.
    ``side='input'``: ``dW = -s (W d) d^T / (d^T d)`` with ``d`` in ``R^in`` -- the
    same removal on the read side, which is where a grouped-query projection with
    ``out != hidden`` can be addressed at all.

    Returns the DELTA, not the edited weight, so the caller keeps the pre-edit copy
    and ``realized_cos`` gets both ends of the comparison. The delta is computed in
    float64/float32 (fp32 for half/bfloat16 weights) and is not cast back.

    Refuses, by name: a nonfinite weight, a nonfinite direction, a direction whose
    length does not match the edited side, and a zero-norm direction (``d^T d == 0``
    divides the edit away -- a no-op that would look like a successful removal).
    """
    if side not in SIDES:
        raise ValueError(f'side must be one of {SIDES}, not {side!r}')
    if not isinstance(strength, (int, float)) or not math.isfinite(strength):
        raise ValueError(f'strength is {strength!r}; it must be a finite number')
    weight = _matrix(weight)
    d = torch.as_tensor(direction)
    if d.ndim > 2:
        raise ValueError(f'feature direction is {d.ndim}-d; a single feature is one vector')
    d = d.reshape(-1)
    if not d.is_floating_point():
        d = d.float()
    _finite(d, 'feature direction')
    want = int(weight.shape[0]) if side == 'output' else int(weight.shape[1])
    if d.numel() != want:
        raise ValueError(f'{side}-side direction has {d.numel()} entries; weight '
                         f'{tuple(weight.shape)} edits its {side} dim of size {want}')
    denom = float(torch.dot(d.double(), d.double()))
    if not denom > 0.0:
        raise ValueError(f'zero-norm feature direction (||d||^2 = {denom!r}); there is nothing '
                         f'to project out and an ablation along it is a silent no-op')
    dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
    a = d.to(dtype)
    w = weight.to(dtype)
    if side == 'output':
        projected = a.unsqueeze(1) @ (a.unsqueeze(0) @ w)
    else:
        projected = (w @ a.unsqueeze(1)) @ a.unsqueeze(0)
    return (-float(strength) * projected) / denom


def span_ablation(directions, weight, strength: float = 1.0, side: str = 'output') -> torch.Tensor:
    """Delta for a whole feature set: the sum of its rank-1 removals.

    For orthonormal rows this is the projection onto the span, ``-s D^T (D W)``, and k
    rank-1 removals equal it exactly (``spanfit.per_layer_subspace``). Non-orthogonal
    SAE directions are orthonormalized first, because removing ``d1`` and then ``d2``
    is not the same edit as removing the span of ``d1, d2`` once they overlap.
    """
    basis, _ = orthonormalize(directions)
    rows = list(basis)
    total = feature_ablation(rows[0], weight, strength, side)
    for row in rows[1:]:
        total = total + feature_ablation(row, weight, strength, side)
    return total


def orthonormalize(directions, tol: float = 1e-6) -> tuple[torch.Tensor, bool]:
    """Rows -> orthonormal basis of their span; ``(basis, changed)``.

    SAE decoder columns are not orthogonal, and both the resident adapter and the
    span formula assume unit rows with ``D D^T = I``. Rows are normalized, then QR'd
    when the Gram matrix is not already the identity; each basis row is sign-aligned
    to the direction it replaces so the sign keeps its meaning (removal, not
    addition). Linearly dependent features are refused: their "rank-k" removal is
    really rank-k', and reporting it as k features would overstate the edit.
    """
    d = torch.as_tensor(directions)
    if d.ndim == 1:
        d = d.unsqueeze(0)
    if d.ndim != 2:
        raise ValueError(f'directions are {d.ndim}-d; expected [features, width]')
    if not d.is_floating_point():
        d = d.float()
    _finite(d, 'feature directions')
    norms = d.norm(dim=1)
    zero = torch.nonzero(norms == 0, as_tuple=False).flatten()
    if zero.numel():
        raise ValueError(f'feature direction row {int(zero[0])} is zero-norm; refusing a '
                         f'silent no-op removal')
    unit = d / norms.unsqueeze(1)
    if unit.shape[0] == 1:
        return unit.contiguous(), False
    gram = unit @ unit.T
    identity = torch.eye(unit.shape[0], dtype=unit.dtype, device=unit.device)
    if torch.allclose(gram, identity, atol=tol):
        return unit.contiguous(), False
    q, r = torch.linalg.qr(unit.T)                     # q [width, G], r [G, G]
    deficit = torch.nonzero(r.diagonal().abs() <= tol, as_tuple=False).flatten()
    if deficit.numel():
        raise ValueError(f'feature directions are linearly dependent (rank deficit at row '
                         f'{int(deficit[0])}); their span is smaller than the feature count')
    basis = q.T.contiguous()
    signs = torch.sign((basis * unit).sum(dim=1))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return (basis * signs.unsqueeze(1)).contiguous(), True


def realized_cos(weight_before, intended_delta, weight_after) -> float:
    """cos(realized, intended): the delta the weight actually holds vs the one asked for.

    ``realized = weight_after - weight_before``. 1.0 means the edit landed exactly as
    computed, 0.0 means the change is orthogonal to the intent (the edit was overwritten,
    rounded away, or applied on the wrong side), negative means the weight moved the
    other way. The codec lane (``ggufio.preview_edit``, ``--min-edit-cos``) measures
    this same quantity on re-encoded bytes; this is the in-memory form of that number,
    so a resident edit and a merged artifact are reported on the same scale.

    A no-op intended delta whose realized delta is also zero is 1.0: nothing was asked
    for and nothing happened. Exactly one zero is 0.0 -- a delta that moved nothing
    cannot be said to have realized an edit. Both zeros reported as 1.0 is why callers
    must not read a perfect cosine as proof that an edit exists; the residual-projection
    check is what proves the direction left the weight.
    """
    before = torch.as_tensor(weight_before)
    intended = torch.as_tensor(intended_delta)
    after = torch.as_tensor(weight_after)
    if before.shape != after.shape or before.shape != intended.shape:
        raise ValueError(f'shapes differ: before {tuple(before.shape)}, intended '
                         f'{tuple(intended.shape)}, after {tuple(after.shape)}')
    intent = intended.reshape(-1).float()
    change = (after.float() - before.float()).reshape(-1)
    _finite(intent, 'intended delta')
    _finite(change, 'realized delta')
    intent_norm = float(intent.norm())
    change_norm = float(change.norm())
    if intent_norm == 0.0 or change_norm == 0.0:
        return 1.0 if intent_norm == change_norm else 0.0
    return max(-1.0, min(1.0, float(torch.dot(intent, change) / (intent_norm * change_norm))))


def cos_to_axis(direction, axis) -> float:
    """cos(feature direction, oracle/contrast axis) -- the number the arm card needs.

    This is the whole novelty question in one scalar: ~1 means the SAE lane is the
    rank-1 removal lane with extra steps, ~0 means the feature carries a direction the
    contrast fit does not see (the only case where the added compute is not just cost).
    """
    d = torch.as_tensor(direction).reshape(-1).float()
    a = torch.as_tensor(axis).reshape(-1).float()
    if d.numel() != a.numel():
        raise ValueError(f'direction has {d.numel()} entries; axis has {a.numel()}')
    _finite(d, 'feature direction')
    _finite(a, 'oracle axis')
    norms = (float(d.norm()), float(a.norm()))
    if not all(norms):
        raise ValueError(f'zero-norm input to cos_to_axis (direction {norms[0]}, axis {norms[1]})')
    return max(-1.0, min(1.0, float(torch.dot(d, a) / (norms[0] * norms[1]))))


class ResidentFeatureEdit:
    """Apply selected SAE features to a live model through ``editor.targets``, reversibly.

    ``editor.targets`` is the resident edit surface the ``Editor`` already built:
    ``(layer index, group, side, parent, name, adapter)``. Nothing here re-discovers
    the module tree or decides which projections matter -- that is the editor's job,
    and a second opinion would be a second source of truth about the edit surface.

    The edit goes through the adapter's own ``set_directions`` (the A/B factors the
    resident operator serves), never through ``merge()``: the base weights are not
    touched, so ``revert()`` drops the factors and the pre-edit bytes are back. A
    model that is already serving another edit is refused rather than silently
    overwritten.
    """

    def __init__(self, editor):
        self.editor = editor
        self._live = []          # [(target tuple, adapter, pre-edit weight clone)]
        self.records = []        # per-target cos / residual report

    def apply(self, features, scale: float = 1.0) -> list[dict]:
        """Apply ``features`` at strength ``scale``; returns the per-tensor records.

        ``features`` is ``[G, W]`` directions in the residual stream (build them with
        ``sae.directions([...])``) or a ``{group: [G, W]}`` mapping when groups need
        different feature sets. The resident adapter must have been built at rank
        ``G`` -- its A/B buffers are fixed-rank and padding them with dummy rows would
        remove directions nobody selected, so a mismatch is refused. Targets whose
        side width does not match ``W`` are an error, not a skip: a silent skip is how
        a lane reports an edit over the attention blocks while the FFN still carries
        the behaviour.
        """
        if not isinstance(scale, (int, float)) or not math.isfinite(scale):
            raise ValueError(f'scale is {scale!r}; it must be a finite number')
        if self._live:
            raise ValueError('this feature edit is already applied; call revert() first')
        self.records = []
        try:
            for index, group, side, _parent, _name, adapter in self.editor.targets:
                width = int(adapter.direction_width())
                chosen = features[group] if isinstance(features, Mapping) else features
                selected = _direction_matrix(chosen, width, group, adapter)
                basis, orthogonalized = orthonormalize(selected)
                rank = int(adapter.A.shape[0])
                if basis.shape[0] != rank:
                    # The resident operator's A/B buffers are fixed-rank and
                    # set_directions insists on an orthonormal (rank, width) basis:
                    # padding a 2-feature edit to rank 4 with dummy rows would remove
                    # directions nobody selected. Build the Editor at the feature count.
                    raise ValueError(f'group {group!r} selects {basis.shape[0]} features but '
                                     f'{adapter.source_key} was built at rank {rank}; construct '
                                     f'the resident Editor at rank {basis.shape[0]} for this '
                                     f'feature set')
                if getattr(adapter, 'enabled', False):
                    raise ValueError(f'{adapter.source_key} already carries a resident edit; '
                                     f'reset or merge it before applying SAE features')
                before = adapter.canonical_weight().detach().float().clone()
                if before.ndim != 2:
                    raise ValueError(f'{adapter.source_key} is a {before.ndim}-d fused weight; the '
                                     f'SAE contract measures a 2-d [out, in] projection, so this '
                                     f'target needs its per-expert form before it is claimable')
                adapter.set_directions(basis, scale)
                # Registered before the measurement so a failure below still reverts
                # this target: revert() is the only thing that can turn it off again.
                self._live.append(((index, group, side, _parent, _name, adapter), adapter, before))
                after = adapter.effective_weight().detach().float()
                intended = span_ablation(basis, before, scale, side)
                residual = _residual_projection(basis, after, side)
                want = _residual_projection(basis, before, side)
                self.records.append({
                    'layer': index, 'group': group, 'side': side,
                    'source_key': getattr(adapter, 'source_key', None),
                    'shape': list(before.shape), 'rank': int(basis.shape[0]),
                    'orthogonalized': bool(orthogonalized),
                    'cos': realized_cos(before, intended, after),
                    'max_abs_delta_err': float((intended.detach() - (after - before)).abs().max()),
                    'residual_projection': residual,
                    'residual_before': want,
                })
        except BaseException:
            self.revert()
            raise
        return self.records

    def revert(self) -> dict:
        """Drop every applied factor set; returns the byte-identity report.

        ``byte_identical`` is ``torch.equal`` of each adapter's canonical (served)
        weight against the float32 clone taken before ``set_directions``, and
        ``factors_cleared`` additionally proves the revert dropped the A/B factors,
        the row scale and the ``enabled`` flag rather than only looking equal.
        """
        report = {'targets': 0, 'byte_identical': True, 'factors_cleared': True,
                  'per_target': {}}
        for _target, adapter, saved in reversed(self._live):
            index, group = _target[0], _target[1]
            adapter.reset()
            now = adapter.canonical_weight().detach()
            same = torch.equal(now.float(), saved)
            cleared = (float(adapter.A.abs().max()) == 0.0
                       and float(adapter.B.abs().max()) == 0.0
                       and bool(torch.all(adapter.row_scale == 1))
                       and not getattr(adapter, 'enabled', False))
            key = getattr(adapter, 'source_key', f'layer{index}.{group}')
            report['per_target'][key] = {'bytes': same, 'factors': cleared}
            report['byte_identical'] &= same
            report['factors_cleared'] &= cleared
            report['targets'] += 1
        self._live.clear()
        return report


def _direction_matrix(value, width: int, group: str, adapter) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f'group {group!r} directions are {type(value).__name__}, not a torch.Tensor')
    d = value.float()
    if d.ndim == 1:
        d = d.unsqueeze(0)
    if d.ndim != 2:
        raise ValueError(f'group {group!r} directions are {d.ndim}-d; expected [features, width]')
    if int(d.shape[1]) != width:
        raise ValueError(f'group {group!r} at {getattr(adapter, "source_key", "?")} exposes '
                         f'{width} entries on its {adapter.side} side; the feature directions '
                         f'have {int(d.shape[1])}. An SAE feature is a residual-stream vector, '
                         f'so it has to match the side the target edits')
    return d


def _residual_projection(basis: torch.Tensor, weight: torch.Tensor, side: str) -> float:
    """||D along the edited side|| after the edit, absolute (what must be ~0)."""
    w = weight.float()
    d = basis.float()
    if side == 'output':
        return float((d @ w).norm())
    return float((w @ d.T).norm())


def apply_to_resident(editor, features, scale: float = 1.0) -> ResidentFeatureEdit:
    """Apply SAE features on the resident edit surface; returns the handle that reverts it.

    ``edit = apply_to_resident(editor, sae.directions([12, 401]), scale=1.0)``
    then ``edit.revert()``. The handle owns the revert because the pre-edit copies and
    the applied target list must travel with the edit; a bare ``revert()`` would have
    to guess which edit on a possibly-shared model it is undoing.

    The ``Editor`` must be built at rank ``len(features)``: the resident adapter's
    A/B buffers are fixed-rank, and extending an orthonormal basis to fill them would
    remove directions that were never selected.
    """
    edit = ResidentFeatureEdit(editor)
    edit.apply(features, scale)
    return edit
