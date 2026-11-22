# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-layer strength profiles: how the edit is spread over the footprint.

The searched strength is one scalar over the whole footprint. A profile gives
each layer in the footprint its own weight w_l, and the edit on layer l becomes
`w_l * scale * (d^T d W)` instead of `scale * (d^T d W)`.

  flat    every weight 1.0: today's behaviour, byte for byte.
  bell    full weight at the centre, tapering linearly to `min_weight` at
          `width_fraction` of the footprint away (Heretic-style curve).
  manual  explicit '10=1.0,11=0.8'; unlisted footprint layers get 1.0.

Weights are normalised so their mean over the footprint is 1.0 (`mean='one'`):
a bell then spends the same total edit as flat at the same scale, so a win is
the shape and not a stronger edit. `mean='keep'` leaves the weights as built.

The edit is quadratic in the direction (d^T d W), so one weight is applied as
d * sqrt(w): adapter factors and merged bytes both take their directions from
`profiled_axes`, so the two cannot disagree.
"""
from __future__ import annotations

import hashlib
import math

KINDS = ('flat', 'bell', 'manual')
MEANS = ('keep', 'one')


def parse_spec(spec):
    """'10=1.0,11=0.8' -> {10: 1.0, 11: 0.8}."""
    weights = {}
    for piece in (spec or '').split(','):
        if not piece.strip():
            continue
        layer, sep, value = piece.partition('=')
        if not sep:
            raise ValueError(f"--profile-spec entry {piece!r} is not LAYER=WEIGHT")
        weights[int(layer)] = float(value)
    return weights


def build_profile(kind, layers, params=None):
    """{layer: weight} over `layers` for one profile kind.

    params: centre_fraction, width_fraction, min_weight, spec, mean ('one'|'keep').
    """
    params = params or {}
    layers = sorted(int(layer) for layer in layers)
    if not layers:
        raise ValueError('a profile needs at least one layer')
    mean = params.get('mean', 'one')
    if kind not in KINDS:
        raise ValueError(f'unknown profile {kind!r}; choose from {KINDS}')
    if mean not in MEANS:
        raise ValueError(f'unknown --profile-mean {mean!r}; choose from {MEANS}')
    if kind == 'flat':
        return {layer: 1.0 for layer in layers}
    if kind == 'bell':
        centre = float(params.get('centre_fraction', 0.5))
        width = float(params.get('width_fraction', 0.5))
        floor = float(params.get('min_weight', 0.3))
        if not 0.0 <= centre <= 1.0:
            raise ValueError('--profile-centre must be in [0, 1]')
        if width <= 0.0:
            raise ValueError('--profile-width must be > 0')
        if not 0.0 < floor <= 1.0:
            raise ValueError('--profile-min must be in (0, 1]')
        last = len(layers) - 1
        raw = {}
        for index, layer in enumerate(layers):
            position = index / last if last else centre
            distance = min(1.0, abs(position - centre) / width)
            raw[layer] = 1.0 - (1.0 - floor) * distance
    else:
        listed = parse_spec(params.get('spec'))
        if not listed:
            raise ValueError("--profile manual needs --profile-spec '10=1.0,11=0.8'")
        # a spec written for the widest footprint is read through each narrower one
        if not set(listed) & set(layers):
            raise ValueError(f'--profile-spec names no layer inside the footprint '
                             f'{layers[0]}-{layers[-1]}')
        if any(value <= 0.0 for value in listed.values()):
            raise ValueError('--profile-spec weights must be > 0')
        raw = {layer: listed.get(layer, 1.0) for layer in layers}
    if mean == 'one':
        average = sum(raw.values()) / len(raw)
        raw = {layer: value / average for layer, value in raw.items()}
    return raw


def profile_name(kind, params=None, weights=None):
    """Short unique label for one profile: becomes part of the arm key."""
    params = params or {}
    if kind == 'flat':
        return 'flat'
    tail = '' if params.get('mean', 'one') == 'one' else '-keep'
    if kind == 'bell':
        return (f"bell-c{float(params.get('centre_fraction', 0.5)):g}"
                f"-w{float(params.get('width_fraction', 0.5)):g}"
                f"-m{float(params.get('min_weight', 0.3)):g}{tail}")
    digest = hashlib.sha1(repr(sorted((weights or {}).items())).encode()).hexdigest()[:6]
    return f'manual-{digest}{tail}'


def is_flat(weights):
    return all(value == 1.0 for value in weights.values())


def layer_axis(axis, weight):
    """One layer's direction(s) scaled so the edit carries `weight`. weight 1.0 is the SAME object."""
    if weight == 1.0:
        return axis
    return axis * math.sqrt(weight)


def profiled_axes(axes, weights):
    """axes [layers, ...] with each weighted layer scaled by sqrt(weight).

    A flat profile returns `axes` itself, so the flat path touches no array.
    """
    if not weights or is_flat(weights):
        return axes
    out = axes.clone()
    for layer, weight in weights.items():
        out[layer] = layer_axis(axes[layer], weight)
    return out


def self_check():
    """Offline identity check on a tiny tensor (no model, no GPU).

    (1) flat profile: adapter factors equal today's scale-only build exactly;
    (2) bell profile: B @ A equals weight * (the scale-only delta) per layer;
    (3) mean 'one' really averages to 1.0 and keeps the bell's shape.
    """
    import torch

    from .gguf_adapters import serve_factors

    torch.manual_seed(0)
    layers = [4, 5, 6, 7, 8]
    axes = torch.nn.functional.normalize(torch.randn(10, 6), dim=-1)
    served = {layer: torch.randn(6, 9) for layer in layers}

    flat = build_profile('flat', layers)
    assert profiled_axes(axes, flat) is axes
    for layer in layers:
        old = serve_factors(axes[layer].reshape(1, -1), served[layer], 1.0)
        new = serve_factors(layer_axis(axes[layer], flat[layer]).reshape(1, -1), served[layer], 1.0)
        assert all(torch.equal(x, y) for x, y in zip(old, new)), f'flat differs at layer {layer}'

    bell = build_profile('bell', layers, {'centre_fraction': 0.5, 'width_fraction': 0.5,
                                          'min_weight': 0.3})
    assert abs(sum(bell.values()) / len(bell) - 1.0) < 1e-9
    assert bell[6] > bell[5] > bell[4] and abs(bell[4] - bell[8]) < 1e-9
    tilted = profiled_axes(axes, bell)
    for layer in layers:
        a0, b0 = serve_factors(axes[layer].reshape(1, -1), served[layer], 1.0)
        a1, b1 = serve_factors(tilted[layer].reshape(1, -1), served[layer], 1.0)
        assert torch.allclose(b1 @ a1, bell[layer] * (b0 @ a0), atol=1e-5), f'bell off at {layer}'
    assert torch.equal(tilted[0], axes[0]), 'a layer outside the footprint moved'
    return {'flat_identical': True, 'bell_weights': {k: round(v, 4) for k, v in bell.items()}}
