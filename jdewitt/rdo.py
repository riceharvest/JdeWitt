# SPDX-License-Identifier: AGPL-3.0-or-later
"""RDO: gradient-refined refusal direction against a linear refusal probe.

The direction spanfit fits is a contrast mean, ``M[L] = mean(bad) - mean(good)``
(``spanfit.per_layer_axes``). On one family of prompts the label and the topic /
length move together, so ``M`` can carry the confound as much as refusal. RDO
(as listed in ``methods.py`` under arXiv:2502.17420) refines a direction by
gradient steps against a linear refusal probe fitted on the same activations.

Scope: this is RDO's *idea* -- gradient refinement of a direction against a
linear refusal probe -- applied to spanfit capture records. It is not the
paper's operator and not an exact reproduction; ``methods.method_catalog``'s
scope line says as much. CPU-side, deterministic, no model load.

Per layer, float64:

  1. rows are split by family. With ``--holdout families`` every 4th family
     (sorted by id) is held out entirely: it contributes no fit rows and no
     centering statistics, and it is scored on raw rows, so the number is about
     transfer to unseen topics rather than to unseen prompts.
  2. when records carry ``family_id`` each family's rows are centered at that
     family's own mean. The topic/confound offset is a per-family constant; the
     label change is within the family, so removing the family mean removes the
     part of the confound that is a pure topic offset. Without ``family_id`` the
     raw rows are used (``centering: none``).
  3. the probe is a logistic regression (L2 1e-3, full-batch gradient descent,
     fixed iteration count, no shuffling) on the centered rows.
  4. ``d0`` is the normalized contrast mean; ``d`` is refined by gradient ascent
     on the probe's own class-separation objective over the projected features,
     ``J(d) = mean_i log sigmoid(y_i * kappa * d . x_i)``, with ``kappa`` fixed
     from the probe's score scale. Each step uses the normalized gradient, so
     ``--lr`` is an angular step, and ``d`` is renormalized every step. The
     maximizer of ``J`` over unit vectors is the probe direction, so the
     refinement moves the centroid chord onto a direction the probe ranks
     harmfully.
  5. good structure is protected: the top-k PCA subspace of the (centered) good
     rows is projected out of ``d`` and ``d`` is renormalized, and the surviving
     fraction is reported per layer (``kept_fraction``, the spanfit ``protect``
     idea).

The probe/refinement kernel runs on torch float64 when torch is importable and
on a numerically equivalent stdlib float64 path otherwise. The stdlib path is
what keeps ``--self-check`` runnable in a bare interpreter (no torch, no numpy),
which is the environment the card's done gate verifies in; ``fit`` itself needs
torch only to read and write ``.pt`` files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import struct

PROBE_L2 = 1e-3
PROBE_STEPS = 400
PROBE_LR_SAFETY = 0.9      # fraction of the GD stability bound
HOLDOUT_EVERY = 4          # every 4th family (sorted by id) is held out
PCA_ITERS = 40
PCA_SEED = 20240
MIN_KEPT = 1e-6
EPS = 1e-12


# ---------- small numeric facade ----------

def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class _ListOps:
    """float64 rows as plain python lists; stdlib only."""

    name = "python"

    def layers(self, rec):
        if isinstance(rec, dict):
            return [[float(value) for value in rec[key]] for key in sorted(rec, key=int)]
        rows = [[float(value) for value in row] for row in rec]
        if rows and not isinstance(rows[0], list):
            return [rows]
        return rows

    def stack(self, vectors):
        return [list(vec) for vec in vectors]

    def vlen(self, vec):
        return len(vec)

    def width(self, rows):
        return len(rows[0])

    def n_rows(self, rows):
        return len(rows)

    def row(self, rows, index):
        return rows[index]

    def zeros(self, width):
        return [0.0] * width

    def copy(self, vec):
        return list(vec)

    def add(self, a, b):
        return [x + y for x, y in zip(a, b)]

    def sub(self, a, b):
        return [x - y for x, y in zip(a, b)]

    def scale(self, vec, coef):
        return [value * coef for value in vec]

    def dot(self, a, b):
        return sum(x * y for x, y in zip(a, b))

    def add_scaled(self, acc, coef, vec):
        for index, value in enumerate(vec):
            acc[index] += coef * value

    def norm(self, vec):
        return math.sqrt(self.dot(vec, vec))

    def scores(self, rows, vec):
        return [self.dot(row, vec) for row in rows]

    def weighted_sum(self, rows, coefs):
        acc = self.zeros(self.width(rows))
        for coef, row in zip(coefs, rows):
            self.add_scaled(acc, coef, row)
        return acc

    def to_floats(self, vec):
        return list(vec)


class _TorchOps:
    """the same primitives over torch float64 tensors on CPU, single-threaded."""

    name = "torch"

    def __init__(self):
        import torch

        self.torch = torch
        # Determinism over speed: reductions keep one accumulation order.
        torch.set_num_threads(1)

    def _as(self, values):
        torch = self.torch
        if torch.is_tensor(values):
            return values.double()
        return torch.as_tensor(values, dtype=torch.float64)

    def layers(self, rec):
        torch = self.torch
        if isinstance(rec, dict):
            return torch.stack([self._as(rec[key]) for key in sorted(rec, key=int)])
        tensor = self._as(rec)
        return tensor.unsqueeze(0) if tensor.dim() == 1 else tensor

    def stack(self, vectors):
        return self.torch.stack([self._as(vec) for vec in vectors])

    def vlen(self, vec):
        return int(vec.shape[0])

    def width(self, rows):
        return int(rows.shape[1])

    def n_rows(self, rows):
        return int(rows.shape[0])

    def row(self, rows, index):
        return rows[index]

    def zeros(self, width):
        return self.torch.zeros(width, dtype=self.torch.float64)

    def copy(self, vec):
        return vec.clone()

    def add(self, a, b):
        return a + b

    def sub(self, a, b):
        return a - b

    def scale(self, vec, coef):
        return vec * coef

    def dot(self, a, b):
        return float(self.torch.dot(a, b))

    def add_scaled(self, acc, coef, vec):
        acc += coef * vec

    def norm(self, vec):
        return float(self.torch.linalg.vector_norm(vec))

    def scores(self, rows, vec):
        return rows @ vec

    def weighted_sum(self, rows, coefs):
        torch = self.torch
        weights = coefs if torch.is_tensor(coefs) else torch.tensor(
            [float(value) for value in coefs], dtype=torch.float64)
        return weights @ rows

    def to_floats(self, vec):
        return [float(value) for value in vec.tolist()]


def _ops(force: str | None = None):
    choice = force or os.environ.get("JDEWITT_RDO_BACKEND")
    if choice == "python":
        return _ListOps()
    if choice == "torch":
        return _TorchOps()
    try:
        return _TorchOps()
    except ImportError:
        return _ListOps()


# ---------- primitives ----------

def _normalize(ops, vec):
    norm = ops.norm(vec)
    if norm < EPS:
        return None, 0.0
    return ops.scale(vec, 1.0 / norm), norm


def _mean(ops, rows):
    n = ops.n_rows(rows)
    if n == 0:
        raise ValueError("mean over zero rows")
    return ops.weighted_sum(rows, [1.0 / n] * n)


def _mean_vectors(ops, vectors):
    if not vectors:
        raise ValueError("mean over zero vectors")
    weights = [1.0 / len(vectors)] * len(vectors)
    return ops.weighted_sum(ops.stack(vectors), weights)


def _gram_schmidt(ops, vectors, tol=1e-10):
    """Modified Gram-Schmidt with reorthogonalization; drops near-null vectors."""
    basis = []
    for vec in vectors:
        unit, norm = _normalize(ops, vec)
        if unit is None:
            continue
        for _ in range(2):
            for base in basis:
                ops.add_scaled(unit, -ops.dot(unit, base), base)
        unit, norm = _normalize(ops, unit)
        if unit is not None and norm > tol:
            basis.append(unit)
    return basis


def _gram_schmidt_lists(vectors, tol=1e-12):
    basis = []
    for vec in vectors:
        u = [float(value) for value in vec]
        for _ in range(2):
            for base in basis:
                coef = sum(x * y for x, y in zip(u, base))
                u = [x - coef * y for x, y in zip(u, base)]
        norm = math.sqrt(sum(value * value for value in u))
        if norm > tol:
            basis.append([value / norm for value in u])
    return basis


# ---------- records ----------

def _label(meta):
    if isinstance(meta, str):
        return meta
    if isinstance(meta, dict):
        return meta.get("label")
    return None


def _family(meta):
    if isinstance(meta, dict):
        family = meta.get("family_id")
        if isinstance(family, str) and family.strip():
            return family
        return None
    return None


def load_records(path):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the interpreter
        raise SystemExit("reading --records needs torch; the self-check does not "
                         "(JDEWITT_RDO_BACKEND=python runs the fit in stdlib float64)") from exc
    state = torch.load(path, map_location="cpu", weights_only=False)
    records = state["records"] if isinstance(state, dict) and "records" in state else state
    return records


def _split_families(families, holdout):
    if holdout == "none" or len(families) < 2:
        return set(), set(families)
    ordered = sorted(families)
    held = {family for index, family in enumerate(ordered)
            if index % HOLDOUT_EVERY == HOLDOUT_EVERY - 1}
    if not held or len(held) == len(ordered):
        held = {ordered[-1]}
    return held, set(ordered) - held


# ---------- the fit ----------

def _fit_probe(ops, rows, labels, width, steps=PROBE_STEPS, l2=PROBE_L2):
    """Logistic regression by full-batch GD: fixed steps, no shuffling, no randomness."""
    n = ops.n_rows(rows)
    weights = ops.zeros(width)
    bias = 0.0
    scale = sum(ops.dot(ops.row(rows, index), ops.row(rows, index)) for index in range(n)) / n
    lr = PROBE_LR_SAFETY / (0.25 * max(scale, EPS) + 2.0 * l2)
    for _ in range(steps):
        margins = ops.to_floats(ops.scores(rows, weights))
        residuals = [(_sigmoid(margin + bias) - label) / n
                     for margin, label in zip(margins, labels)]
        grad = ops.weighted_sum(rows, residuals)
        weights = ops.sub(weights, ops.scale(ops.add(grad, ops.scale(weights, 2.0 * l2)), lr))
        bias -= lr * sum(residuals)
    return weights, bias


def _log_sigmoid(z: float) -> float:
    if z >= 0.0:
        return -math.log1p(math.exp(-z))
    return z - math.log1p(math.exp(z))


def _profile_scale(scores, labels, l2, kappa, iters=12):
    """argmax_kappa mean log sigmoid(y * kappa * s) - l2*kappa^2, by damped Newton.

    This is the probe's own penalized log-likelihood along one direction,
    profiled over the score scale. It is concave in kappa (logistic loss
    composed with kappa*s, minus a quadratic), so Newton is safe with damping;
    at the probe direction the maximizer is kappa = ||w||, which is why the
    refinement needs no scale constant of its own.
    """
    n = len(scores)
    if n == 0:
        return kappa

    def objective(value):
        total = sum(_log_sigmoid(label * value * score) for score, label in zip(scores, labels))
        return total / n - l2 * value * value

    current = objective(kappa)
    for _ in range(iters):
        gradient = 0.0
        curvature = 0.0
        for score, label in zip(scores, labels):
            margin = label * kappa * score
            probability = _sigmoid(margin)
            gradient += label * score * (1.0 - probability)
            curvature -= (label * score) ** 2 * probability * (1.0 - probability)
        gradient = gradient / n - 2.0 * l2 * kappa
        curvature = curvature / n - 2.0 * l2
        if curvature >= -1e-15:
            break
        candidate = kappa - gradient / curvature
        if candidate <= 0.0:
            candidate = 0.5 * kappa
        while candidate > 0.0 and objective(candidate) < current and abs(candidate - kappa) > 1e-12:
            candidate = 0.5 * (candidate + kappa)
        moved = abs(candidate - kappa)
        kappa, current = candidate, objective(candidate)
        if moved < 1e-10 * max(kappa, 1.0):
            break
    return kappa


def _refine(ops, rows, labels, start, steps, lr, kappa0=None, l2=PROBE_L2):
    """Gradient ascent of the probe's penalized likelihood over the unit sphere.

    J(v) = mean_i log sigmoid(y_i * (v . x_i)) - l2*||v||^2 is the probe's own
    objective; it is maximized at v = w. The refinement walks along the sphere
    with v = kappa * d, profiling kappa out of J each step (_profile_scale), and
    stepping d along the normalized gradient by ``lr`` (an angular step),
    renormalizing d every step. Its stationary point is d = w/||w||, so the
    direction it returns is one the probe ranks harmfully rather than the
    centroid chord.
    """
    direction = start
    kappa = kappa0 if kappa0 is not None else 1.0
    n = ops.n_rows(rows)
    for _ in range(steps):
        scores = ops.to_floats(ops.scores(rows, direction))
        kappa = _profile_scale(scores, labels, l2, kappa)
        probabilities = [_sigmoid(label * kappa * score) for score, label in zip(scores, labels)]
        coefs = [label * (1.0 - probability) * kappa / n
                 for probability, label in zip(probabilities, labels)]
        grad, grad_norm = _normalize(ops, ops.weighted_sum(rows, coefs))
        if grad is None:
            break
        step, _ = _normalize(ops, ops.add(direction, ops.scale(grad, lr)))
        if step is None:
            break
        direction = step
    return direction


def _good_subspace(ops, rows, k, iters=PCA_ITERS, seed=PCA_SEED):
    """Top-k PCA directions of the good rows by orthogonal iteration.

    Deterministic: the iteration starts from a fixed-seed gaussian basis, and the
    basis is re-orthonormalized every sweep, so the result does not depend on
    thread count or hashing order.
    """
    n = ops.n_rows(rows)
    width = ops.width(rows)
    k = max(1, min(int(k), width, n))
    rng = random.Random(seed)
    start = [[rng.gauss(0.0, 1.0) for _ in range(width)] for _ in range(k)]
    basis = _gram_schmidt(ops, list(ops.stack(start)))
    if not basis:
        return []
    for _ in range(iters):
        # Columns of B = X @ Q are orthonormalized (the row space of X Q), then
        # V = X^T B; orthonormalizing V gives the next basis. Standard subspace
        # iteration, written against the row primitives.
        columns = _gram_schmidt_lists([ops.to_floats(ops.scores(rows, base)) for base in basis])
        if not columns:
            break
        basis = _gram_schmidt(ops, [ops.weighted_sum(rows, column) for column in columns])
        if not basis:
            break
    return basis


def _protect(ops, direction, subspace):
    vec = ops.copy(direction)
    for base in subspace:
        ops.add_scaled(vec, -ops.dot(vec, base), base)
    kept = ops.norm(vec)
    if kept < MIN_KEPT:
        return ops.zeros(ops.vlen(direction)), 0.0
    return ops.scale(vec, 1.0 / kept), kept


def _auc(bad_scores, good_scores):
    """Mann-Whitney AUC with average ranks for ties."""
    if not bad_scores or not good_scores:
        return None
    pooled = sorted([(float(s), 1) for s in bad_scores] + [(float(s), 0) for s in good_scores])
    ranks = [0.0] * len(pooled)
    index = 0
    while index < len(pooled):
        end = index
        while end + 1 < len(pooled) and pooled[end + 1][0] == pooled[index][0]:
            end += 1
        average = (index + end) / 2.0 + 1.0
        for slot in range(index, end + 1):
            ranks[slot] = average
        index = end + 1
    n_bad = len(bad_scores)
    n_good = len(good_scores)
    rank_sum = sum(rank for rank, (_, is_bad) in zip(ranks, pooled) if is_bad)
    return (rank_sum - n_bad * (n_bad + 1) / 2.0) / (n_bad * n_good)


def fit_records(records, *, steps=200, lr=0.05, protect_k=8, holdout="families",
                probe_steps=PROBE_STEPS, ops=None, collect=None):
    """Fit per-layer RDO axes from capture records; returns (axes, report).

    ``collect`` (optional dict) is filled with the fitted ``contrast`` and
    ``probe`` unit directions per layer; the self-check needs them to score the
    planted dim, and they are kept out of the JSON report because a per-layer
    width-sized vector does not belong in a report.
    """
    if steps < 1 or not (0.0 < lr <= 1.0):
        raise ValueError("steps >= 1 and 0 < lr <= 1 are required (lr is an angular step)")
    if protect_k < 0:
        raise ValueError("protect_k >= 0 is required")
    ops = ops or _ops()
    labelled = []
    layer_count = 0
    for meta, rec in records:
        label = _label(meta)
        if label not in ("bad", "good"):
            continue
        family = _family(meta)
        layers = ops.layers(rec)
        layer_count = max(layer_count, len(layers))
        labelled.append((label, family, layers))
    if not labelled:
        raise SystemExit("no records carry label='bad'/'good'")

    families = sorted({family for _, family, _ in labelled if family is not None})
    if holdout == "families":
        missing = sum(1 for _, family, _ in labelled if family is None)
        if missing:
            raise SystemExit(f"--holdout families needs family_id on every record; {missing} rows have none")
    held, trained = _split_families(families, holdout)
    centering = "family" if trained else "none"
    width = ops.vlen(labelled[0][2][0])

    axes = []
    layers_report = []
    if collect is not None:
        collect.setdefault("contrast", [])
        collect.setdefault("probe", [])
    for layer in range(layer_count):
        train_bad, train_good, bad_family, good_family = [], [], [], []
        hold_bad, hold_good = [], []
        for label, family, layers in labelled:
            if layer >= len(layers):
                continue
            vec = layers[layer]
            if family is not None and family in held:
                (hold_bad if label == "bad" else hold_good).append(vec)
                continue
            if label == "bad":
                train_bad.append(vec)
                bad_family.append(family)
            else:
                train_good.append(vec)
                good_family.append(family)
        if len(train_bad) < 2 or len(train_good) < 2:
            raise SystemExit(f"layer {layer}: need >=2 bad and >=2 good fit rows, "
                             f"got {len(train_bad)}/{len(train_good)}")
        if centering == "family":
            pooled = list(zip(train_bad + train_good, bad_family + good_family))
            means = {}
            for family in {family for _, family in pooled}:
                means[family] = _mean_vectors(ops, [vec for vec, fam in pooled if fam == family])
            train_bad = [ops.sub(vec, means[family]) for vec, family in zip(train_bad, bad_family)]
            train_good = [ops.sub(vec, means[family]) for vec, family in zip(train_good, good_family)]

        bad_rows = ops.stack(train_bad)
        good_rows = ops.stack(train_good)
        rows = ops.stack(train_bad + train_good)
        labels = [1.0] * len(train_bad) + [-1.0] * len(train_good)

        contrast, contrast_norm = _normalize(ops, ops.sub(_mean(ops, bad_rows), _mean(ops, good_rows)))
        if contrast is None:
            raise SystemExit(f"layer {layer}: bad and good means coincide; no contrast direction")
        weights, _bias = _fit_probe(ops, rows, labels, width, steps=probe_steps)
        probed, probe_norm = _normalize(ops, weights)
        if probed is None:
            probed = contrast
            probe_norm = 0.0
        refined = _refine(ops, rows, labels, contrast, steps, lr, kappa0=probe_norm or 1.0)
        subspace = _good_subspace(ops, good_rows, protect_k) if protect_k else []
        final, kept = _protect(ops, refined, subspace)

        axis = ops.to_floats(final)
        axes.append(axis)
        if collect is not None:
            collect["contrast"].append(ops.to_floats(contrast))
            collect["probe"].append(ops.to_floats(probed))
        report = {
            "layer": layer,
            "cos_rdo_contrast": round(ops.dot(final, contrast), 6),
            "auc_train": round(_auc(ops.to_floats(ops.scores(bad_rows, final)),
                                    ops.to_floats(ops.scores(good_rows, final))), 6),
            "auc_holdout_contrast": None,
            "auc_holdout_rdo": None,
            "kept_fraction": round(kept, 6),
            "probe_cos_contrast": round(ops.dot(probed, contrast), 6),
            "n_bad": len(train_bad),
            "n_good": len(train_good),
            "n_holdout_bad": len(hold_bad),
            "n_holdout_good": len(hold_good),
        }
        if hold_bad and hold_good:
            hold_bad_rows = ops.stack(hold_bad)
            hold_good_rows = ops.stack(hold_good)
            report["auc_holdout_contrast"] = round(
                _auc(ops.to_floats(ops.scores(hold_bad_rows, contrast)),
                     ops.to_floats(ops.scores(hold_good_rows, contrast))), 6)
            report["auc_holdout_rdo"] = round(
                _auc(ops.to_floats(ops.scores(hold_bad_rows, final)),
                     ops.to_floats(ops.scores(hold_good_rows, final))), 6)
        layers_report.append(report)

    report = {
        "scope": "RDO's idea (gradient refinement against a linear refusal probe) on spanfit "
                 "records; not the paper's operator, not an exact reproduction.",
        "backend": ops.name,
        "centering": centering,
        "centering_note": ("rows centered at their own family mean; the family/topic offset is a "
                           "per-family constant, the label change is within the family"
                           if centering == "family" else
                           "no family_id in the records, raw rows used"),
        "holdout": holdout,
        "holdout_families": sorted(held),
        "fit_families": sorted(trained),
        "steps": steps, "lr": lr, "protect_k": protect_k,
        "probe": {"l2": PROBE_L2, "steps": probe_steps,
                  "note": "full-batch gradient descent, fixed iterations, no shuffling"},
        "axes": [len(axes), width],
        "axes_sha256": axes_digest(axes),
        "layers": layers_report,
    }
    return axes, report


def axes_digest(axes):
    """sha256 of the [L, W] float32 row-major buffer (stable across torch.save runs)."""
    flat = [float(value) for row in axes for value in row]
    return hashlib.sha256(struct.pack(f"<{len(flat)}f", *flat)).hexdigest()


def save_axes(path, axes, meta):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the interpreter
        raise SystemExit("writing --out needs torch; drop --out/--report for a dry run") from exc
    torch.save({"axes": torch.tensor(axes, dtype=torch.float32), "meta": meta}, path)


# ---------- self-check ----------

def synthetic_records(*, seed=7, layers=3, width=40, families=8,
                      per_family=(24, 16), content_dims=10):
    """Records with a planted refusal dim r and a family-shared nuisance n.

    r is dim 0, n is dim 1. Within a family the nuisance mean is label-dependent
    (the length/topic confound: bad rows sit further along n), while the family
    offset c_f is the same for both labels in that family. Training families all
    carry the same offset ``+1``; held-out families carry ``0`` or ``-1``.
    Ten high-variance "content" dims (variance 9) dominate the good rows' PCA, so
    protecting the top-8 good directions cannot eat r by construction.
    """
    rng = random.Random(seed)
    ordered = [f"fam{index}" for index in range(families)]
    held = {family for index, family in enumerate(sorted(ordered))
            if index % HOLDOUT_EVERY == HOLDOUT_EVERY - 1}
    content = list(range(2, 2 + content_dims))
    tail = list(range(2 + content_dims, width))
    records = []
    planted = {"r": [1.0] + [0.0] * (width - 1), "n": [0.0, 1.0] + [0.0] * (width - 2),
               "holdout_families": sorted(held)}
    for family in ordered:
        offset = 0.0 if family in held else 1.0
        if family in held and ordered.index(family) % 2:
            offset = -1.0
        for label, count in (("bad", per_family[0]), ("good", per_family[1])):
            y = 1.0 if label == "bad" else -1.0
            for _ in range(count):
                layer_rows = []
                for layer in range(layers):
                    decay = 1.0 - 0.1 * layer
                    row = [0.0] * width
                    # signal/confound/noise: r separation 2*0.65*decay against
                    # 0.35 within-class sd; n carries a label-correlated mean
                    # (0.9*decay) that the contrast mean cannot tell from
                    # refusal, but a 2.5-sd within-class spread that the
                    # variance-normalised probe discounts.
                    row[0] = 0.65 * decay * y + 0.35 * rng.gauss(0.0, 1.0)
                    row[1] = 0.9 * decay * y + offset + 2.5 * rng.gauss(0.0, 1.0)
                    for dim in content:
                        row[dim] = 3.0 * rng.gauss(0.0, 1.0)
                    for dim in tail:
                        row[dim] = 0.25 * rng.gauss(0.0, 1.0)
                    layer_rows.append(row)
                meta = {"label": label, "family_id": family}
                records.append((meta, layer_rows))
    return records, planted


def self_check(*, seed=7, steps=200, lr=0.05, protect_k=8, probe_steps=PROBE_STEPS):
    ops = _ops()
    records, planted = synthetic_records(seed=seed)
    r = planted["r"]
    collect = {}
    axes_one, report_one = fit_records(records, steps=steps, lr=lr, protect_k=protect_k,
                                       holdout="families", probe_steps=probe_steps, ops=ops,
                                       collect=collect)
    axes_two, report_two = fit_records(records, steps=steps, lr=lr, protect_k=protect_k,
                                       holdout="families", probe_steps=probe_steps, ops=ops)
    print(f"self-check: backend={ops.name} layers={len(axes_one)} width={len(r)} "
          f"centering={report_one['centering']} holdout={report_one['holdout_families']}")
    failures = []
    if report_one["axes_sha256"] != report_two["axes_sha256"]:
        failures.append("two fit runs disagree on the axes sha256")
    for layer in report_one["layers"]:
        index = layer["layer"]
        axis = axes_one[index]
        cos_rdo = ops.dot(axis, r) / max(ops.norm(axis), EPS)
        cos_contrast = ops.dot(collect["contrast"][index], r)
        print(f"  layer {index}: cos(rdo, r)={cos_rdo:.4f} cos(contrast, r)={cos_contrast:.4f} "
              f"auc_holdout_rdo={layer['auc_holdout_rdo']:.4f} "
              f"auc_holdout_contrast={layer['auc_holdout_contrast']:.4f} "
              f"kept={layer['kept_fraction']:.4f}")
        if not cos_rdo > cos_contrast:
            failures.append(f"layer {index}: cos(rdo, r) <= cos(contrast, r)")
        if layer["auc_holdout_rdo"] is None or layer["auc_holdout_rdo"] < 0.9:
            failures.append(f"layer {index}: held-out AUC(rdo) < 0.9")
        if layer["auc_holdout_contrast"] is None or not layer["auc_holdout_contrast"] < layer["auc_holdout_rdo"]:
            failures.append(f"layer {index}: held-out AUC(contrast) is not lower")
    print(f"  axes sha256 run1={report_one['axes_sha256']}")
    print(f"  axes sha256 run2={report_two['axes_sha256']}")
    if failures:
        for failure in failures:
            print(f"self-check: FAIL - {failure}")
        return 1
    print("self-check: PASS")
    return 0


# ---------- CLI ----------

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m jdewitt.rdo",
        description="RDO: refine the contrast refusal direction against a linear refusal probe on "
                    "captured tensors (CPU, deterministic).")
    commands = parser.add_subparsers(dest="cmd", required=True)
    fit = commands.add_parser("fit", help="fit per-layer RDO axes from capture records")
    fit.add_argument("--records", required=True, help="spanfit/capture records .pt")
    fit.add_argument("--out", required=True, help="output axes .pt")
    fit.add_argument("--report", help="output report json")
    fit.add_argument("--steps", type=int, default=200, help="refinement steps per layer")
    fit.add_argument("--lr", type=float, default=0.05, help="angular step of the refinement")
    fit.add_argument("--protect-k", type=int, default=8, help="good-row PCA directions protected")
    fit.add_argument("--holdout", choices=("families", "none"), default="families")
    fit.add_argument("--probe-steps", type=int, default=PROBE_STEPS)
    check = commands.add_parser("self-check", help="planted-direction self-check (no model, no records)")
    check.add_argument("--seed", type=int, default=7)
    check.add_argument("--steps", type=int, default=200)
    check.add_argument("--lr", type=float, default=0.05)
    check.add_argument("--probe-steps", type=int, default=PROBE_STEPS)
    args = parser.parse_args(argv)

    if args.cmd == "self-check":
        return self_check(seed=args.seed, steps=args.steps, lr=args.lr, probe_steps=args.probe_steps)

    axes, report = fit_records(load_records(args.records), steps=args.steps, lr=args.lr,
                               protect_k=args.protect_k, holdout=args.holdout,
                               probe_steps=args.probe_steps)
    save_axes(args.out, axes, {"method": "rdo", "axes_sha256": report["axes_sha256"],
                               "centering": report["centering"], "holdout": report["holdout"]})
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
            handle.write("\n")
    print(json.dumps({"axes": report["axes"], "axes_sha256": report["axes_sha256"],
                      "centering": report["centering"], "holdout_families": report["holdout_families"],
                      "auc_holdout_rdo": [layer["auc_holdout_rdo"] for layer in report["layers"]],
                      "kept_min": min(layer["kept_fraction"] for layer in report["layers"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
