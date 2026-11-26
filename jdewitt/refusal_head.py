# SPDX-License-Identifier: AGPL-3.0-or-later
"""Refusal axis from output-head geometry alone — no forward pass, no activations.

The idea is the J-space pullback (docs/non-heretic-methods.md) with the Jacobian
replaced by the one matrix that is known without a capture: the output head. A
refusal is narrated by a small set of tokens; if the head is tied (or not), the
rows it scores those tokens with are vectors in residual space, and a contrast of
their means names a direction without ever running the model.

    u   = mean(E_denial) - mean(E_compliance)      # vocabulary-index weights
    d   = u / ||u||                                # = normalize(W_out^T u)

`u` is the logit-space direction: raising a token's margin means moving `h` along
its head row, so the direction that raises the denial-minus-compliance margin is
the normalised mean row difference. The writeback edit that removes it is

    W' = W (I - d d^T)                             # tensors in GGUF order [in, out]

which zeroes every output component along d. This module is the algebra only: no
model is built, no token is generated, nothing is served. Every function here is
batch matrix arithmetic over arrays a caller has already loaded, so a test can
check the identities on a 4-wide toy and the report can carry `removal_measured:
false` with a clear conscience.

The three identities the report re-checks per tensor (they are exact, so the
tolerance is float32 noise and not a threshold):

  * d is unit:                     ||d|| = 1
  * projection residual:           ||W(I - d d^T) - W||_F = ||W d||_2
  * head margin reduction:         m - m' = (u^T W_out d) (d^T W^T a)
                                    = ||W_out^T u|| (W d)^T a   for any a

where m = u^T W_out W^T a is the denial-minus-compliance margin contributed by
that tensor for a block input a. Nothing in this file reads a tokenizer or a
checkpoint; the caller supplies both.
"""
from __future__ import annotations

import numpy as np

# The task names the two vocabularies. Kept as the primary inventory so the axis
# is reproducible from prose alone; the repo's denial table is an addition the
# caller opts into (see resolve_surfaces), never a replacement.
DENIAL_WORDS = ('I', 'Sorry', 'Cannot', 'Unfortunately', 'No')
COMPLIANCE_WORDS = ('Sure', 'Certainly', 'Yes', 'Here', 'Okay', 'Absolutely')


def surface_forms(word: str) -> tuple[str, ...]:
    """Every surface a word takes at a token boundary, in both cases.

    A word starts an answer bare ('I cannot') and appears mid-answer spaced
    (' I cannot'); the leading-space token and the bare token are different
    rows in the head, so a direction built from one of them is half an axis.
    """
    seen, forms = set(), []
    for text in (word, word.lower()):
        for form in (text, ' ' + text):
            if form not in seen:
                seen.add(form)
                forms.append(form)
    return tuple(forms)


def _encode(tokenizer, text: str) -> list[int]:
    return [int(i) for i in tokenizer.encode(text, add_special_tokens=False)]


def resolve_surfaces(tokenizer, words, *, extra_surfaces=()):
    """Resolve words (and whole surfaces) to head-row ids, one token at a time.

    A surface that tokenizes to more than one id is recorded as `multi_token`
    and NOT admitted: a subword row ('fortunately', 'ann') does not carry the
    denial word, so folding those rows into the mean would put noise in the axis
    and hide it behind an average. The only exception is a caller-supplied
    surface that resolves to a single token (' cannot' -> 4157), which is a real
    head row for the whole surface and is why the repo's table can contribute.
    """
    ids, by_word, multi, surfaces = [], {}, [], []
    for word in words:
        rows = []
        for form in surface_forms(word):
            resolved = _encode(tokenizer, form)
            if len(resolved) == 1:
                rows.append({'form': form, 'id': resolved[0]})
                if resolved[0] not in ids:
                    ids.append(resolved[0])
            else:
                multi.append({'word': word, 'form': form, 'ids': resolved,
                              'reason': 'multi-token surface: a subword row is not the word'})
        by_word[word] = rows
    for surface in extra_surfaces:
        resolved = _encode(tokenizer, surface)
        if len(resolved) == 1:
            surfaces.append({'surface': surface, 'id': resolved[0]})
            if resolved[0] not in ids:
                ids.append(resolved[0])
        else:
            multi.append({'word': surface, 'form': surface, 'ids': resolved,
                          'reason': 'multi-token table surface: admitted only in the all-ids variant'})
    return {'ids': ids, 'by_word': by_word, 'multi_token': multi, 'table_single_token': surfaces}


def table_ids(tokenizer, surfaces, *, first_token_only: bool = True):
    """Ids named by whole table surfaces (used only for the sensitivity variant)."""
    ids = []
    for surface in surfaces:
        resolved = _encode(tokenizer, surface)
        chosen = resolved[:1] if first_token_only else resolved
        for token in chosen:
            if token not in ids:
                ids.append(token)
    return ids


def head_axis(head: np.ndarray, denial_ids, compliance_ids):
    """d = normalize(mean(E_denial) - mean(E_compliance)) for head rows E.

    `head` is [vocab, hidden] in HF order. The denial/compliance margin of a
    residual vector h is h . g with g = W_out^T u, so ||g|| is the margin per
    unit step along d and is reported beside it: a direction with a tiny margin
    is arithmetic, not a usable axis.
    """
    if not len(denial_ids) or not len(compliance_ids):
        raise ValueError('both vocabularies must resolve to at least one head row')
    rows_d = np.asarray(head[np.asarray(denial_ids, dtype=np.int64)], dtype=np.float64)
    rows_c = np.asarray(head[np.asarray(compliance_ids, dtype=np.int64)], dtype=np.float64)
    g = rows_d.mean(axis=0) - rows_c.mean(axis=0)
    norm = float(np.linalg.norm(g))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError('denial and compliance head means coincide: no axis exists')
    d = (g / norm).astype(np.float64)
    return {
        'd': d,
        'g': g,                               # unnormalised contrast, hidden space
        'margin': norm,                       # ||W_out^T u|| = ||g|| in float64
        'row_norms': {
            'denial_mean': float(np.linalg.norm(rows_d.mean(axis=0))),
            'compliance_mean': float(np.linalg.norm(rows_c.mean(axis=0))),
            'denial_rows_mean_l2': float(np.linalg.norm(rows_d, axis=1).mean()),
            'compliance_rows_mean_l2': float(np.linalg.norm(rows_c, axis=1).mean()),
        },
        'n_denial': len(denial_ids), 'n_compliance': len(compliance_ids),
    }


def project_out(W: np.ndarray, d: np.ndarray):
    """W' = W (I - d d^T) for W in [in, out] order, in float64.

    Returns (edited, delta) so a caller can hash/report the edit rather than
    re-derive it. No in-place write: the caller decides if anything is stored.
    """
    W = np.asarray(W, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    if W.ndim != 2 or d.ndim != 1 or W.shape[1] != d.shape[0]:
        raise ValueError(f'project_out expects W [in,out] with out=={d.shape[0]}, got {W.shape}')
    delta = np.outer(W @ d, d)               # W d d^T, [in, out]
    return W - delta, delta


def edit_stats(W: np.ndarray, d: np.ndarray, *, identity_tol: float = 1e-9):
    """Per-tensor edit norm plus the two exact identities, checked not assumed.

    The residual identity is compared against `||W d||_2` computed WITHOUT
    forming the projection, so a bug in `project_out` cannot satisfy its own
    check by construction.
    """
    W64 = np.asarray(W, dtype=np.float64)
    d64 = np.asarray(d, dtype=np.float64)
    edited, delta = project_out(W64, d64)
    fro_W = float(np.linalg.norm(W64))
    fro_delta = float(np.linalg.norm(delta))
    projected = float(np.linalg.norm(W64 @ d64))
    headroom = max(fro_delta, projected, 1.0)
    return {
        'shape_in_out': [int(W64.shape[0]), int(W64.shape[1])],
        'norm_fro': fro_W,
        'delta_norm_fro': fro_delta,
        'relative_edit_norm': fro_delta / fro_W if fro_W else None,
        'projected_energy_norm': projected,
        'residual_identity_abs': abs(fro_delta - projected) / headroom,
        'residual_identity_ok': abs(fro_delta - projected) <= identity_tol * headroom,
        'output_along_d_max': float(np.abs(edited @ d64).max()) if edited.size else 0.0,
        'unit_ok': abs(float(np.linalg.norm(d64)) - 1.0) <= 1e-12,
    }


def margin_reduction(g: np.ndarray, d: np.ndarray, W: np.ndarray, a: np.ndarray,
                     *, identity_tol: float = 1e-9):
    """The head margin lost when this tensor's d-component is projected away.

    m(a)  = u^T W_out W^T a        (denial-minus-compliance margin, this tensor)
    m'(a) = u^T W_out W'^T a       (after W' = W(I - dd^T))
    m - m' = (u^T W_out d)(d^T W^T a) = ||W_out^T u|| (W d) . a

    Checked against the independent right-hand side, so the 'exactly the
    projection' claim is a measurement of two matrix products, not a restatement.
    """
    W64 = np.asarray(W, dtype=np.float64)
    d64 = np.asarray(d, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)      # W_out^T u, the unnormalised contrast
    edited, _ = project_out(W64, d64)
    before = float(g @ (W64.T @ a))
    after = float(g @ (edited.T @ a))
    predicted = float(np.linalg.norm(g)) * float((W64 @ d64) @ a)
    observed = before - after
    headroom = max(abs(before), abs(observed), abs(predicted), 1e-6)
    return {
        'before': before, 'after': after, 'reduction': observed,
        'predicted': predicted,
        'identity_abs': abs(observed - predicted) / headroom,
        'identity_ok': abs(observed - predicted) <= identity_tol * headroom,
    }


def head_channel_closed(g: np.ndarray, d: np.ndarray, *, identity_tol: float = 1e-9):
    """Does projecting d out of the head's input close the d-margin channel exactly?

    M = u^T W_out d = ||W_out^T u||; with W_out (I - dd^T) the same reading is
    zero by construction. Reported because it is the cleanest statement of what
    the writeback edit buys: the head can no longer score any d-component.
    """
    g = np.asarray(g, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    closed = float(g @ (np.eye(d.shape[0]) - np.outer(d, d)) @ d)
    return {'margin_per_unit_d': float(np.linalg.norm(g)),
            'margin_after': closed,
            'closed': abs(closed) <= identity_tol * max(float(np.linalg.norm(g)), 1.0)}


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f'cosine needs equal shapes, got {a.shape} and {b.shape}')
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        raise ValueError('cosine against a zero vector is undefined')
    return float(a @ b / (na * nb))


def top_left_direction(delta: np.ndarray):
    """Dominant residual-space direction of a weight delta, and its rank-1 share.

    delta is [in, out] in GGUF order; a rank-1 abliteration delta is -s u v^T,
    so the left factor (the direction removed from the residual stream) is the
    top eigenvector of delta delta^T. Fraction of energy in it says how far the
    delta is from rank-1; it is reported, never assumed.
    """
    delta = np.asarray(delta, dtype=np.float64)
    gram = delta @ delta.T
    values, vectors = np.linalg.eigh(gram)
    top = int(np.argmax(values))
    sigma_sq = max(float(values[top]), 0.0)
    total = float(np.sum(np.abs(values)))
    u = vectors[:, top]
    if u[int(np.argmax(np.abs(u)))] < 0:
        u = -u
    return {'direction': u,
            'sigma_sq': sigma_sq,
            'energy_fraction_top': (sigma_sq / total) if total else None,
            'effective_rank': (total / sigma_sq) if sigma_sq > 0 else None}
