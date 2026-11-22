"""Answer-span per-layer rank-1 removal: the cross-lineage fit (METHODOLOGY R4/R7).

Label-only procedure (no oracle, no per-model constants):
  1. capture.py-style hooks record each decoder layer's OUTPUT projection input at
     the mean of the answer span, for harmful rows (bad) and harmless rows (good);
  2. M[L] = mean(bad)[L] - mean(good)[L] is that layer's refusal direction;
  3. for every output-side projection W of layer L (the last matrices that touch the
     residual stream before it continues -- o_proj/out_proj/down_proj/w2, found from
     the module tree, not a constant table), subtract W's component along M[L]:
         delta = -(a^T (a W)),  a = M[L]/||M[L]||.
The same math runs on a merged gguf (deployment artifact, R8) or in-memory on HF
weights (serve space == edit space, for lineages with no local llama.cpp support).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import TYPE_CHECKING, Any

from .errors import NoVerifiedAdapter


class _LazyModule:
    """Module proxy that imports on first attribute access.

    `python -m jdewitt.spanfit build --help` must answer on a machine where the
    numeric stack is not installed -- same reason scripts/merge_bench.py keeps
    `import torch` inside its functions. Every real code path touches a tensor
    immediately, so the first `np.x`/`torch.x` loads the module once and the
    proxy costs nothing after that. Annotations stay strings (`from __future__
    import annotations`), so importing this module needs only the stdlib.
    """

    __slots__ = ("_name", "_module")

    def __init__(self, name: str):
        self._name = name
        self._module: Any = None

    def __getattr__(self, attr: str) -> Any:
        module = self._module
        if module is None:
            import importlib
            module = self._module = importlib.import_module(self._name)
        return getattr(module, attr)

    def __repr__(self) -> str:
        return f"<lazy module {self._name!r}>"


if TYPE_CHECKING:  # annotation-only imports; the runtime objects below are lazy
    import numpy as np
    import torch
else:
    np = _LazyModule("numpy")
    torch = _LazyModule("torch")


# ---------- fit ----------

def _label(meta) -> str | None:
    if isinstance(meta, str):
        return meta
    if isinstance(meta, dict):
        return meta.get("label")
    return None


def group_means(records: list[tuple], label: str) -> torch.Tensor:
    """Stack per-layer contrast vectors [N,L,W] for rows whose meta label == label."""
    arrs = []
    for meta, rec in records:
        if _label(meta) != label:
            continue
        if torch.is_tensor(rec):
            arrs.append(rec.double())
        elif isinstance(rec, dict):
            arrs.append(torch.stack([torch.as_tensor(np.asarray(rec[k])).double()
                                     for k in sorted(rec, key=int)]))
    if not arrs:
        raise SystemExit(f"no records with label={label!r} in {len(records)} records")
    return torch.stack(arrs)


def per_layer_axes(records: list[tuple], label_bad: str = "bad",
                    label_good: str = "good") -> torch.Tensor:
    """Normalized per-layer refusal axis M[L,W] from (meta, rec) pairs."""
    bad = group_means(records, label_bad)
    good = group_means(records, label_good)
    M = (bad.mean(0) - good.mean(0))
    M = M / M.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return M.float()


def load_records(path: str) -> list[tuple]:
    st = torch.load(path, map_location="cpu", weights_only=False)
    return st["records"] if isinstance(st, dict) and "records" in st else st


# ---------- separability: which layers carry the harmful/harmless split ----------
#
# A footprint is a band of layers an edit is applied to, and `tune --footprints` tests
# bands (all, 10-27, 16-27) with nothing behind the choice. The capture already stores
# (meta, [L, W]) rows per label, so a layer's separation is a closed-form read of a file
# that exists: how far apart the two label means point (cosine) and how big that gap is
# next to the spread of each label's own rows (snr). A layer gains snr only when the
# harmful rows and the harmless rows sit in different places at this width; a layer
# where both labels cover the same cloud carries nothing to remove, however large the
# raw activation is.

MIN_ROWS_PER_LABEL = 3
MIN_LAYERS = 8
DEFAULT_KEEP = 0.5


def _footprint_refusal(detail: str) -> None:
    """Exit non-zero on a capture that cannot support a footprint, always in the same words.

    ``"a footprint over two rows is noise"`` is the fixed sentence so a caller can tell a
    capture that is too small (or too flat) from a crash: no band, no half-band, no
    silent zero.
    """
    raise SystemExit(f"layers: {detail}; a footprint over two rows is noise")


def _residual_array(rec) -> np.ndarray:
    """One stored row as a [L, W] float64 array (capture's tensor, or a {layer: vec} dict)."""
    if isinstance(rec, dict):
        arr = np.stack([np.asarray(rec[k], dtype=np.float64) for k in sorted(rec, key=int)])
    elif torch.is_tensor(rec):
        arr = rec.detach().cpu().double().numpy()
    else:
        arr = np.asarray(rec, dtype=np.float64)
    if arr.ndim != 2:
        raise SystemExit(
            f"layers: a residual row is {arr.ndim}-D {tuple(arr.shape)}; capture stores one "
            f"[L, W] tensor per row, so a bare vector cannot say which layer it came from")
    return arr


def _label_rows(records: list[tuple], label: str) -> np.ndarray:
    """Stack every row whose meta label is ``label`` into an [N, L, W] float64 array."""
    rows = [_residual_array(rec) for meta, rec in records if _label(meta) == label]
    if not rows:
        return np.zeros((0, 0, 0), dtype=np.float64)
    shapes = {row.shape for row in rows}
    if len(shapes) != 1:
        raise SystemExit(
            f"layers: the {label!r} rows disagree on shape {sorted(shapes)}; rows from more "
            f"than one capture or layer set cannot be scored as one footprint")
    return np.stack(rows)


def layer_separability(records: list[tuple], keep: float = DEFAULT_KEEP,
                       label_bad: str = "bad", label_good: str = "good") -> dict:
    """Per-layer harmful-vs-harmless separability from saved capture rows (COSMIC-style).

    For each layer L (float64, ascending, no sampling):

    * ``mu_bad``/``mu_good`` are the label means over that layer's [W] residual;
    * ``cosine = cos(mu_bad, mu_good)`` -- anti-aligned means are what a removal axis
      can exploit; two means pointing the same way are a topic/length confound, not an
      edit site;
    * ``snr = ||mu_bad - mu_good|| / (mean_bad_dist + mean_good_dist)`` -- the gap in
      units of each label's own mean distance to its centre, so a wide harmless cloud
      cannot buy a layer a high score just by being wide;
    * the footprint is ``min-max`` over the layers whose snr clears ``keep`` times the
      strongest layer's snr, i.e. exactly the ``a-b`` string ``tune --footprints`` takes;
    * a uniform score sets ``flat: true``: the band then covers the whole capture and
      is not a target, so the flat flag exists instead of pretending the band located
      something.

    The returned payload holds no timestamps and nothing sampled, so the same records
    always produce the same bytes and the same sha256.
    """
    if not 0.0 < keep <= 1.0:
        raise SystemExit(f"layers: --keep {keep} is outside (0, 1]; the keep line is a "
                         f"fraction of the strongest layer's snr")
    bad = _label_rows(records, label_bad)
    good = _label_rows(records, label_good)
    for label, rows in ((label_bad, bad), (label_good, good)):
        if len(rows) < MIN_ROWS_PER_LABEL:
            _footprint_refusal(
                f"{len(rows)} row(s) labelled {label!r} (need >= {MIN_ROWS_PER_LABEL})")
    if bad.shape[1:] != good.shape[1:]:
        raise SystemExit(f"layers: label {label_bad!r} is [L,W] {bad.shape[1:]}, label "
                         f"{label_good!r} is {good.shape[1:]}; the two labels were not "
                         f"captured over the same layers and width")
    n_layers, width = int(bad.shape[1]), int(bad.shape[2])
    if n_layers < MIN_LAYERS:
        _footprint_refusal(f"the capture holds {n_layers} layers (need >= {MIN_LAYERS})")

    mu_bad, mu_good = bad.mean(axis=0), good.mean(axis=0)
    scored = []
    for index in range(n_layers):
        b, g = mu_bad[index], mu_good[index]
        norm_b, norm_g = float(np.linalg.norm(b)), float(np.linalg.norm(g))
        cosine = (float(np.clip(b @ g / (norm_b * norm_g), -1.0, 1.0))
                  if norm_b and norm_g else 0.0)
        bad_dist = float(np.linalg.norm(bad[:, index, :] - b, axis=1).mean())
        good_dist = float(np.linalg.norm(good[:, index, :] - g, axis=1).mean())
        gap = float(np.linalg.norm(b - g))
        spread = bad_dist + good_dist
        scored.append({"layer": index, "snr": gap / spread if spread else 0.0,
                       "cosine": cosine, "gap": gap,
                       "bad_dist": bad_dist, "good_dist": good_dist})

    snrs = np.array([row["snr"] for row in scored], dtype=np.float64)
    peak = float(snrs.max())
    if peak <= 0.0:
        _footprint_refusal(
            f"every layer scores snr 0 over {n_layers} layers: the two label means are the "
            f"same point at each label's own spread, so no layer separates them")

    keep_line = keep * peak
    in_band = [row["layer"] for row in scored if row["snr"] >= keep_line]
    ranked = sorted(range(n_layers), key=lambda index: (-snrs[index], index))
    for rank, index in enumerate(ranked, start=1):
        scored[index]["rank"] = rank
    return {
        "rows": {label_bad: int(len(bad)), label_good: int(len(good))},
        "n_layers": n_layers,
        "width": width,
        "keep": float(keep),
        "max_snr": peak,
        "min_snr": float(snrs.min()),
        "spread": peak - float(snrs.min()),
        "keep_line": keep_line,
        "footprint": f"{min(in_band)}-{max(in_band)}",
        "flat": bool(np.allclose(snrs, snrs[0], rtol=1e-9, atol=1e-15)),
        "ranked": ranked,
        "layers": scored,
    }


def _print_layer_scores(payload: dict, digest: str, out: str | None = None) -> None:
    """The human read: the band, every layer's snr/cosine/rank, and the payload sha256."""
    note = ("  (flat: the score is uniform, so the band is the whole capture, not a target)"
            if payload["flat"] else "")
    print(f"footprint {payload['footprint']}{note}")
    print(f"{'layer':>5} {'snr':>10} {'cosine':>8} {'rank':>5}  in-band")
    for row in payload["layers"]:
        print(f"{row['layer']:>5} {row['snr']:>10.6f} {row['cosine']:>8.4f} "
              f"{row['rank']:>5}  {'yes' if row['snr'] >= payload['keep_line'] else 'no'}")
    rows = payload["rows"]
    print(f"rows bad {rows['bad']} good {rows['good']}; layers {payload['n_layers']} "
          f"width {payload['width']}; keep {payload['keep']:g} x max snr "
          f"{payload['max_snr']:.6f} = keep line {payload['keep_line']:.6f}")
    print(f"snr min {payload['min_snr']:.6f} max {payload['max_snr']:.6f} "
          f"spread {payload['spread']:.6f}; ranked {payload['ranked']}")
    print(f"sha256 {digest}" + (f"  (payload written to {out})" if out else ""))


def _synthetic_capture(amps: list[float], n_layers: int = 24, width: int = 16,
                       n_rows: int = 6, spread: float = 0.01,
                       label_bad: str = "bad", label_good: str = "good") -> list[tuple]:
    """Deterministic capture with a known answer: ``amps[L]`` pushes the two label means
    apart along layer L's own unit axis, and every row carries the same symmetric spread.

    No RNG. The per-row offsets are symmetric halves, so their mean is exactly zero and
    the spread term cancels out of each label mean: snr[L] is exactly
    ``amps[L] / (mean|offset| * spread)``, which makes a self-check failure a math
    failure rather than a dice roll.
    """
    offsets = np.arange(n_rows, dtype=np.float64) - (n_rows - 1) / 2.0
    spread_axis = np.arange(1.0, width + 1.0)
    spread_axis /= np.linalg.norm(spread_axis)
    records = []
    for label, sign in ((label_bad, 1.0), (label_good, -1.0)):
        for row_index, offset in enumerate(offsets):
            row = np.zeros((n_layers, width), dtype=np.float64)
            for layer, amp in enumerate(amps):
                direction = np.zeros(width, dtype=np.float64)
                direction[layer % width] = 1.0
                row[layer] = sign * amp * direction + offset * spread * spread_axis
            records.append(({"label": label, "row": row_index}, row))
    return records


def self_check() -> None:
    """Two synthetic captures with known answers; no records file read, no model, no GPU.

    (a) bad/good means anti-aligned only on layers 5-7: the footprint must be exactly
        ``5-7`` and every other layer must sit below the keep line;
    (b) equal snr on every layer: the band is the whole capture (``0-<last>``) and
        ``flat`` is true, so a uniform score cannot masquerade as a located band.

    A third check scores (a) twice and compares the payloads, because the score is a
    closed-form read: two runs over the same rows must be byte-identical.
    """
    n_layers = 24
    planted = [1.0 if 5 <= layer <= 7 else 0.0 for layer in range(n_layers)]
    report = layer_separability(_synthetic_capture(planted), keep=DEFAULT_KEEP)
    above = [row["layer"] for row in report["layers"] if row["snr"] >= report["keep_line"]]
    off_band = [row["snr"] for row in report["layers"] if not 5 <= row["layer"] <= 7]
    repeat = layer_separability(_synthetic_capture(planted), keep=DEFAULT_KEEP)
    checks = [
        ("planted layers 5-7 are the only ones over the keep line", above == [5, 6, 7]),
        ("footprint is exactly 5-7", report["footprint"] == "5-7"),
        ("every other layer is below the keep line", max(off_band) < report["keep_line"]),
        ("a planted band is not flagged flat", report["flat"] is False),
        ("two runs over the same rows produce the same payload",
         json.dumps(report, sort_keys=True) == json.dumps(repeat, sort_keys=True)),
    ]
    uniform = layer_separability(_synthetic_capture([1.0] * n_layers), keep=DEFAULT_KEEP)
    checks += [
        ("a uniform score covers the whole capture",
         uniform["footprint"] == f"0-{n_layers - 1}"),
        ("a uniform score is flagged flat", uniform["flat"] is True),
    ]
    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"self-check {'PASS' if ok else 'FAIL'} {name}")
    if failed:
        raise SystemExit(f"layers --self-check: {len(failed)} check(s) failed: {failed}")
    print(f"self-check PASS planted footprint {report['footprint']} "
          f"(keep line {report['keep_line']:.6f}, other layers 0.0), "
          f"uniform footprint {uniform['footprint']} flat={uniform['flat']}")


# ---------- edit ----------

def removal_delta(weight: torch.Tensor, axis: torch.Tensor,
                  strength: float = 1.0, renormalize: bool = False) -> torch.Tensor:
    """-(a (a^T W)); axis acts on dim 0 of W ([out, in]), i.e. the layer's
    residual-write side. Caller supplies matching layouts (torch params or
    ggufio logical [out, in] dequantized arrays).

    ``renormalize`` preserves each row's magnitude (Heretic's ``full`` mode): the
    projection shortens rows, and a shortened row changes the module's output scale,
    which shows up as perplexity damage that has nothing to do with refusal. Row
    magnitudes are rescaled to their original values, so the direction is removed and
    the scale is kept.
    """
    a = axis.reshape(-1, 1).to(weight.dtype)
    delta = -(strength * (a @ (a.T @ weight)))
    if not renormalize:
        return delta
    edited = weight + delta
    before = weight.norm(dim=1, keepdim=True).clamp_min(1e-12)
    after = edited.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return edited * (before / after) - weight


def shared_axis(records: list[tuple], label_bad: str = "bad",
                label_good: str = "good") -> torch.Tensor:
    """Sign-aligned shared refusal axis broadcast over layers (the SAS fit that
    certified q25 at 40/40; on mc47 the per-layer fit beat it 90 vs 65 - the
    searcher certifies which SHAPE wins, per R7)."""
    bad = group_means(records, label_bad)
    good = group_means(records, label_good)
    D = bad.shape[-1]
    M = bad.mean(0) - good.mean(0)            # [L, D]
    R = M / M.norm(dim=1, keepdim=True).clamp_min(1e-12)  # rows onto the sphere
    G = torch.zeros(D, D, dtype=bad.dtype)
    for r in R:
        G += torch.outer(r, r)
    evals, evecs = torch.linalg.eigh(G)
    a = evecs[:, -1]                           # top eigenvector = shared direction
    if float(a @ M.mean(0)) < 0:
        a = -a
    return (a.unsqueeze(0).expand(M.shape[0], D).float().contiguous(), a.float())


def per_layer_subspace(records: list[tuple], rank: int = 2,
                       label_bad: str = "bad", label_good: str = "good") -> torch.Tensor:
    """[L, rank, W] orthonormal directions per layer, best first.

    Rank-1 removal leaves a residual: refusal is not one vector on every model, and a
    second direction can carry what the first left behind. The subspace is the top-k
    right singular vectors of the bad-minus-good contrast matrix, each sign-aligned to
    the layer's mean difference so the direction keeps its meaning (removing it must
    reduce the refusal side). Rows are orthonormal, so projection out of the subspace
    is `W - D^T D W` and k stacked rank-1 removals equal that exactly.
    """
    bad = group_means(records, label_bad)                  # [N_bad, L, W]
    good = group_means(records, label_good)                # [N_good, L, W]
    layers = bad.shape[1]
    width = bad.shape[2]
    rank = max(1, min(int(rank), int(bad.shape[0]), width))
    out = torch.zeros(layers, rank, width)
    for layer in range(layers):
        matrix = (bad[:, layer, :] - good[:, layer, :].mean(0)).float()
        _u, _s, vh = torch.linalg.svd(matrix, full_matrices=False)
        basis = vh[:rank].clone()
        mean = matrix.mean(0)
        for index in range(basis.shape[0]):
            if float(basis[index] @ mean) < 0:
                basis[index] = -basis[index]
        if float(basis.norm(dim=1).min()) < 1e-8:
            raise ValueError(f'layer {layer} has a degenerate contrast subspace')
        out[layer] = basis / basis.norm(dim=1, keepdim=True)
    return out.contiguous()


def protection_subspace(records: list[tuple], label: str, k: int = 8) -> torch.Tensor:
    """Top-k principal directions [L, k, D] of mean-centered answer-span
    activations for rows with the given label - the directions "good behavior"
    uses in the residual write-space that removal edits must not destroy."""
    X = group_means(records, label)                    # [N, L, D] double
    U = torch.zeros(X.shape[1], k, X.shape[2], dtype=torch.float64)
    for lay in range(X.shape[1]):
        Xl = X[:, lay, :] - X[:, lay, :].mean(0, keepdim=True)
        _, _, Vh = torch.linalg.svd(Xl, full_matrices=False)
        U[lay] = Vh[:k]
    return U.float()


def protect_axes(axes: torch.Tensor, subspace: torch.Tensor,
                 min_norm: float = 0.15) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove from each layer axis the components inside the protection
    subspace, renormalize; return (protected axes [L,D], kept fraction [L]).
    If the axis is nearly all protected (norm < min_norm) the layer is skipped
    (axis zeroed) rather than removing an un-protected fraction of it."""
    a = axes.clone()
    kept = torch.ones(a.shape[0])
    for lay in range(a.shape[0]):
        U = subspace[lay].to(a.dtype)
        a[lay] = a[lay] - (U.T @ (U @ a[lay]))
        n = float(a[lay].norm())
        n0 = float(axes[lay].norm())
        kept[lay] = n / max(n0, 1e-12)
        if n < min_norm * n0:
            a[lay] = 0.0
    a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return a, kept


def mixed_axes(perlayer: torch.Tensor, shared: torch.Tensor, alpha: float) -> torch.Tensor:
    """Interpolate each layer's axis toward the shared axis and renormalize:
    a_l = norm(alpha * s + (1-alpha) * a_l). alpha=0 -> perlayer, 1 -> shared.
    `shared` may be [D] or [L,D]."""
    m = (1.0 - alpha) * perlayer + alpha * shared
    return m / m.norm(dim=1, keepdim=True).clamp_min(1e-12)


def build_merged(gguf_in: str, gguf_out: str, axes: torch.Tensor,
                 source_keys: list[str], *, name_map=None, scale: float = 1.0,
                 axis_space: str = "input", in_place: bool = False,
                 renormalize: bool = False, dequant_cache=None) -> dict:
    """Merge per-layer rank-1 removals into `gguf_in`, write `gguf_out` (R8 path).

    source_keys are HF-style names ('model.layers.<L>.<suffix>') or, through
    `name_map`, the artifact's own tensor names; each must resolve to a gguf
    tensor whose input dim matches axes.shape[1]. Layout follows the ggufio
    convention: file order = reversed logical dims, i.e. ggml [in,out].

    in_place patches `gguf_in`'s own bytes (no copy) and leaves a revert sidecar
    beside it; that is the path an agent should use on a model it already serves,
    because the artifact is quantized in place and never re-quantized as a whole.

    Both paths hand ``(axis, space)`` to the writer instead of precomputing a
    float32 delta, so each tensor is dequantized ONCE per arm. ``dequant_cache``
    defaults to ``<gguf_in>.dequant/`` and serves a base from a previous arm
    without unpacking it again; pass ``False`` to disable it.
    """
    from .gguf_adapters import gguf_weight_shapes, resolve_gguf_names
    from .ggufio import DequantBases, edit_tensors, project_tensors_inplace

    t0 = time.monotonic()
    shapes = gguf_weight_shapes(gguf_in)
    names = resolve_gguf_names(source_keys, shapes) if name_map is None else name_map
    if not names:
        raise SystemExit("no target tensors resolved from source keys")
    cache = None
    if dequant_cache is not False:
        cache = (dequant_cache if isinstance(dequant_cache, DequantBases)
                 else DequantBases(str(gguf_in) + '.dequant' if dequant_cache is None
                                   else dequant_cache))

    projections = {}
    skipped = []
    for key, base in names.items():
        layer = int([p for p in key.split(".") if p.isdigit()][0])
        axis = axes[layer]
        # A zero-length axis means 'do not touch this layer'. Skipping it is not
        # an optimisation: an identity delta still goes through dequant-requant,
        # so a merge scoped to blk.9+ can still rewrite blk.0-8 and the resulting
        # artifact is no longer byte-comparable to its base in the layers the
        # source edit never touched.
        if float(axis.norm()) == 0.0:
            skipped.append(f'{base}:zero-axis')
            continue
        dims = shapes.get(base)
        if dims is None or len(dims) != 2:
            skipped.append(f'{base}:not-a-2d-weight')
            continue
        # gguf shape is (in, out); the axis lives on the side the caller declared,
        # and checking it here means a wrong side is refused before any byte moves.
        want = int(dims[1]) if axis_space == 'output' else int(dims[0])
        if want != axis.shape[0]:
            skipped.append(f'{base}:axis-dim-{axis.shape[0]}-want-{want}')
            continue
        # The writer computes the delta from the weight it is already dequantizing,
        # so the axis is the only thing that crosses over. Renormalization is a
        # writer mode for the same reason (out-of-place needs whole rows, which a
        # precomputed delta pass also had to dequantize for).
        projections[base] = (axis.detach().cpu().numpy().astype(np.float32), axis_space)
    if in_place:
        # One file, byte-level: the artifact keeps its quantization, its size and
        # every offset it had; only the changed tiles move, and the sidecar keeps
        # the pre-patch bytes so this is reversible.
        report = project_tensors_inplace(gguf_in, projections, scale=scale,
                                         revert_path=str(gguf_in) + '.revert.npz',
                                         renormalize=renormalize, dequant_cache=cache)
    else:
        report = edit_tensors(gguf_in, gguf_out, projections=projections, scale=scale,
                              renormalize=renormalize, dequant_cache=cache)
    report["skipped"] = skipped
    report["elapsed_s"] = round(time.monotonic() - t0, 1)
    return report


def gguf_targets(gguf_in: str, max_probe_layers: int = 512) -> dict:
    """Which output-side projections this artifact actually carries, and the exact
    build flags that use them.

    Reads tensor names straight off the GGUF; no model load, no GPU. The point is
    that source_key is otherwise typed by hand from the architecture notes, and a
    wrong or unmapped key is the class of mistake that merges successfully while
    editing nothing. The wildcard keys and --layers this returns are verified
    against the file itself, and the layer count is reported per suffix so a
    patched or partial artifact shows up as a mismatch instead of a silent skip.
    """
    from .gguf_adapters import (HF_TO_LLAMA, discover_writeback_targets,
                                gguf_weight_shapes, inferred_hidden_width)
    shapes = gguf_weight_shapes(gguf_in)
    # The generic view: which residual-write projections this artifact carries at
    # its own hidden width. This is what `build --gguf-targets auto` uses, so an
    # agent can look before it merges instead of reading architecture notes.
    hidden = inferred_hidden_width(shapes)
    auto = discover_writeback_targets(shapes, hidden) if hidden else {'targets': {}, 'unmapped_writeback_width': []}
    auto_view = {'hidden': hidden, 'layers': len(auto['targets']),
                 'targets_per_layer': [len(names) for _, names in sorted(auto['targets'].items())][:4],
                 'unmapped_width_tensors': auto['unmapped_writeback_width'][:8]}
    found, suffixes = {}, {}
    for candidate, suffix in HF_TO_LLAMA:
        name = f"blk.0.{suffix}.weight"
        if name not in shapes:
            continue
        count, dims = 0, shapes[name]
        while f"blk.{count}.{suffix}.weight" in shapes:
            count += 1
        found[suffix] = {"layers": count,
                         "gguf_name": f"blk.*.{suffix}.weight",
                         "source_key": f"model.layers.*.{candidate}.weight",
                         "gguf_dims": [int(d) for d in dims]}
        suffixes[suffix] = count
    if not found:
        return {"gguf": gguf_in, "targets": {}, "auto": auto_view,
                "error": "no output-side projection this tool knows how to edit "
                         f"(looked for {', '.join(f'blk.N.{s}.weight' for _, s in HF_TO_LLAMA)})"}
    layers = min(suffixes.values())
    return {"gguf": gguf_in, "targets": found, "auto": auto_view,
            "source_key": [v["source_key"] for v in found.values()],
            "layers": layers,
            "build_line": " ".join(f"--source-key {found[s]['source_key']}"
                                   for s in sorted(found)) + f" --layers {layers}",
            "note": "layers is the smallest suffix count: if two suffixes differ the "
                    "artifact is patched or mixed, so merge scope is decided here rather "
                    "than silently. Axis length is checked against the resolved side at "
                    "merge time (--axis-space); nothing is skipped quietly."}


def parse_layer_ids(spec: str) -> list[int]:
    """'9-35' -> [9..35], '9-20,30' -> [9..20, 30]; used by --only-layers."""
    ids = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = part.split('-', 1)
            ids += list(range(int(start), int(end) + 1))
        else:
            ids.append(int(part))
    if not ids:
        raise SystemExit(f"--only-layers {spec!r} names no layers")
    return sorted(set(ids))


def target_source_keys(n_layers: int, out_key_suffixes: tuple[str, ...]) -> list[str]:
    """HF source keys for every output-side projection found under decoder layers."""
    keys = []
    for L in range(n_layers):
        for suf in out_key_suffixes:
            keys.append(f"model.layers.{L}.{suf}")
    return keys


def discover_out_keys(model: torch.nn.Module, layer_path: str = "") -> list[str]:
    """Structural discovery of output-side projections: the final matrix on each
    branch that writes back into the residual stream. Name vocabulary, not a table:
    matches *o_proj / *out_proj / *w2 / *down_proj under decoder layers."""
    base = model
    if layer_path:
        for hop in layer_path.split("."):
            base = getattr(base, hop)
    found = {}
    for L, block in enumerate(base):
        for pname, _ in block.named_parameters():
            tail = pname.split(".")[-1]
            stem = pname[: -len(tail)]
            if tail.endswith("_weight"):
                stem = stem[:-len("_weight")]
            if (stem.endswith("o_proj") or stem.endswith("out_proj")
                    or stem.endswith("down_proj") or stem in ("w2", "fc2")):
                found[L] = sorted(found.get(L, []) + [pname])
    suffixes = sorted({p for names in found.values() for p in names})
    return suffixes


# ---------- in-memory apply (non-gguf lineages) ----------

class InPlaceRemoval:
    """Apply/revert per-layer rank-1 removals directly on live HF weights."""

    def __init__(self, model: torch.nn.Module, axes: torch.Tensor,
                 key_format: str = "model.layers.{L}.{suffix}"):
        self.model, self.axes, self.key_format = model, axes, key_format
        self.saved: dict[str, torch.Tensor] = {}

    def _targets(self):
        out = []
        for L in range(self.axes.shape[0]):
            for suffix in self.suffixes:
                key = self.key_format.format(L=L, suffix=suffix)
                try:
                    p = self.model.get_parameter(key)
                except AttributeError:
                    continue
                out.append((key, p, L, suffix))
        return out

    @property
    def suffixes(self):  # filled by caller/CLI from discovery or args
        return self._suffixes

    def apply(self, suffixes: list[str]):
        self._suffixes = suffixes
        for key, p, L, _ in self._targets():
            w = p.detach().float()
            axis = self.axes[L]
            if w.shape[0] != axis.shape[0]:  # axis lives on the out dim
                continue
            self.saved.setdefault(key, p.detach().to("cpu").clone())
            delta = axis.reshape(-1, 1) @ (axis.reshape(1, -1) @ w)
            with torch.no_grad():
                p.copy_((w - delta).to(p.dtype))
        return len(self.saved)

    def revert(self):
        for key, saved in self.saved.items():
            p = self.model.get_parameter(key)
            with torch.no_grad():
                p.copy_(saved.to(p.device))
        self.saved.clear()


# ---------- CLI ----------

def main(argv=None):
    """CLI entry: expected adapter refusals print the same envelope as the main CLI.

    spanfit is its own entry point (`python -m jdewitt.spanfit`), so without this
    a NoVerifiedAdapter would surface as a bare traceback/exit text while
    `python -m jdewitt edit-quantized` reports the typed code. One code, one
    shape, whichever entry point hit it.
    """
    try:
        return _main(argv)
    except NoVerifiedAdapter as error:
        import sys
        words = [a for a in (sys.argv[1:] if argv is None else argv) if not a.startswith('-')]
        print(json.dumps({'error': {'code': 'NO_VERIFIED_ADAPTER',
                                    'type': type(error).__name__,
                                    'message': str(error),
                                    'command': words[0] if words else 'spanfit'}}),
              file=sys.stderr)
        raise SystemExit(1) from error


def _main(argv=None):
    ap = argparse.ArgumentParser(prog="jdewitt.spanfit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("build", help="fit axes from records, write merged gguf")
    a.add_argument("--records", help="residual records to fit from (omit with --axes)")
    a.add_argument("--gguf-in", required=True)
    a.add_argument("--out", help="output gguf; omit with --in-place to patch --gguf-in itself")
    a.add_argument("--in-place", action="store_true",
                   help="patch --gguf-in's own bytes instead of writing a copy: the artifact you "
                        "already serve is the artifact that changes, no new file, no requantization "
                        "of the model. A .revert.npz sidecar records the pre-patch bytes of every "
                        "tile written, and `gguf-revert` restores them")
    a.add_argument("--gguf-targets", choices=("auto",), default=None,
                   help="discover this artifact's residual-write projections from its own tensor "
                        "names and hidden width; replaces --source-key (no per-model table, no "
                        "hand-typed glob, works on architectures this repo has never seen)")
    a.add_argument("--axes", help="precomputed axes [D] or [L,D] (torch.save {'axes': ...}) "
                                  "instead of fitting from records — how an externally "
                                  "recovered direction gets merged")
    a.add_argument("--axes-key", default="axes",
                   help="key inside the --axes file to read, for files that hold more "
                        "than one recovered direction (e.g. axes_attn / axes_ffn)")
    a.add_argument("--source-key", action="append", default=[],
                   help="repeatable; '*' expands 0..NL-1 (e.g. model.layers.*.self_attn.o_proj.weight). "
                        "Omit with --gguf-targets auto")
    a.add_argument("--layers", type=int, default=0, help="NL for '*' expansion")
    a.add_argument("--only-layers",
                   help="restrict the '*' expansion to these layers: '9-35' or "
                        "'9-20,30'. The published oracle leaves blk.0-8 untouched; "
                        "whether the layers below an edit matter is a build flag, not "
                        "a guess, and this makes it testable")
    a.add_argument("--shape", choices=("perlayer", "shared"), default="perlayer")
    a.add_argument("--axis-space", choices=("input", "output"), default="output",
                   help="output (default): axis lives in the residual dim W writes "
                        "into, dW = -s u (u^T W); input: axis lives on the module "
                        "input dim and is applied transposed")
    a.add_argument("--scale", type=float, default=1.0,
                   help="projection multiplier; 1.0 removes the axis completely, "
                        "1.5 is the certified q25 SAS scale (the curve is knife-edge, METHODOLOGY R8)")
    a.add_argument("--dequant-cache", metavar="DIR",
                   help="content-addressed dequantized-base cache (one .npy per tensor, keyed on "
                        "sha256 of that tensor's raw blocks + name + quant type); default "
                        "<gguf-in>.dequant/. A warm merge reads the base memmap instead of "
                        "unpacking the blocks again")
    a.add_argument("--no-dequant-cache", action="store_true",
                   help="do not read or write a dequantized-base cache")
    g = sub.add_parser("targets", help="list the output-side projections an artifact has")
    g.add_argument("--gguf-in", required=True)
    m = sub.add_parser("axes", help="fit axes and save the [L,W] tensor")
    m.add_argument("--records", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--cos-to", help="report max cosine distance vs another axes tensor")
    m.add_argument("--shape", choices=("perlayer", "shared"), default="perlayer")
    r = sub.add_parser("protect", help="project fitted axes out of a protection subspace")
    r.add_argument("--axes", required=True, help="axes file to protect")
    r.add_argument("--axes-key", default="axes")
    r.add_argument("--records", required=True, help="capture records with a task label")
    r.add_argument("--label", required=True, help="label of the behavior to protect")
    r.add_argument("--k", type=int, default=8, help="protection subspace rank per layer")
    r.add_argument("--min-norm", type=float, default=0.15,
                   help="zero a layer's axis if less than this fraction survives")
    r.add_argument("--out", required=True)
    s = sub.add_parser("layers",
                       help="score harmful-vs-harmless separability per layer from saved "
                            "capture records; emits the band `tune --footprints` can take")
    s.add_argument("--records", help="saved capture records (spanfit record format)")
    s.add_argument("--out", help="write the scored payload here (JSON); the sha256 printed "
                                 "is of exactly these bytes")
    s.add_argument("--keep", type=float, default=DEFAULT_KEEP,
                   help=f"keep line as a fraction of the strongest layer's snr "
                        f"(default {DEFAULT_KEEP:g})")
    s.add_argument("--self-check", action="store_true",
                   help="synthetic captures with known answers: a planted 5-7 band must be "
                        "found, and a uniform score must come back flat")
    for p in (a, m, r):
        p.add_argument("--time-log", help="JSONL ledger; logs fit/merge phases")
    args = ap.parse_args(argv)

    if args.cmd == "layers":
        if args.self_check:
            return self_check()
        if not args.records:
            raise SystemExit(
                "layers needs --records PATH (a saved capture) or --self-check: the score is a "
                "read of records already on disk, not a model run")
        payload = layer_separability(load_records(args.records), keep=args.keep)
        blob = (json.dumps(payload, indent=1) + "\n").encode()
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "wb") as handle:
                handle.write(blob)
        _print_layer_scores(payload, hashlib.sha256(blob).hexdigest(), args.out)
        return

    if args.cmd == "targets":
        print(json.dumps(gguf_targets(args.gguf_in), indent=1))
        return

    if args.cmd == "build":
        if args.gguf_targets and args.source_key:
            raise SystemExit("--gguf-targets auto replaces --source-key; pass one of them")
        if not args.gguf_targets and not args.source_key:
            raise SystemExit("build needs --gguf-targets auto or at least one --source-key; "
                             "`python -m jdewitt.spanfit targets --gguf-in <file>` lists what an "
                             "artifact carries")
        if not args.in_place and not args.out:
            raise SystemExit("build writes a new file: pass --out PATH, or --in-place to patch "
                             "--gguf-in itself")

    t0 = time.monotonic()
    if args.cmd in ("build", "protect") and args.axes:
        if args.cmd == "build" and args.records:
            raise SystemExit("--axes and --records are alternatives; pick one")
        payload = torch.load(args.axes, map_location="cpu", weights_only=False)
        if args.axes_key not in payload:
            raise SystemExit(f"--axes file has keys {sorted(payload)}, not {args.axes_key!r}")
        axes = payload[args.axes_key]
    else:
        if not args.records:
            raise SystemExit("build needs --records or --axes")
        recs = load_records(args.records)
        axes = (shared_axis(recs)[0] if args.shape == "shared"
                else per_layer_axes(recs))
    if args.time_log:
        from .timing import record
        record(args.time_log, lineage=os.path.basename(args.records or args.axes),
               phase='fit', seconds=time.monotonic() - t0, kind='cpu')
    if args.cmd == "protect":
        recs = load_records(args.records)
        space = protection_subspace(recs, args.label, args.k)
        if axes.dim() == 1:
            axes = axes.unsqueeze(0).repeat(space.shape[0], 1)
        protected, kept = protect_axes(axes, space, args.min_norm)
        torch.save({"axes": protected, "kept": kept, "subspace_rank": args.k,
                    "protected_label": args.label}, args.out)
        print(json.dumps({
            "axes": list(protected.shape), "protected_label": args.label,
            "subspace_rank": args.k,
            "kept_mean": round(float(kept.mean()), 4),
            "kept_min": round(float(kept.min()), 4),
            "layers_zeroed": int((kept < args.min_norm).sum()),
            "note": "kept fraction is how much of the axis survives outside the "
                    "protected subspace; a layer that drops below min-norm is zeroed "
                    "rather than edited along a mostly-protected direction"}))
        return

    if args.cmd == "axes":
        torch.save({"axes": axes}, args.out)
        rep = {"axes": list(axes.shape), "elapsed_s": round(time.monotonic() - t0, 1)}
        if args.cos_to:
            other = torch.load(args.cos_to, map_location="cpu")["axes"]
            cos = torch.nn.functional.cosine_similarity(axes, other, dim=1)
            rep["min_cos_vs"] = float(cos.min())
            rep["mean_cos_vs"] = float(cos.mean())
        print(json.dumps(rep))
        return

    layer_count = args.layers or (int(axes.shape[0]) if args.gguf_targets else args.layers)
    if args.only_layers:
        ids = parse_layer_ids(args.only_layers)
        if ids and any(i >= layer_count for i in ids):
            raise SystemExit(f"--only-layers {ids} reaches past --layers {layer_count}")
    else:
        ids = list(range(layer_count))
    name_map = None
    if args.gguf_targets == "auto":
        # Discovery, not a table: the artifact reports which residual-write
        # projections it carries at this width. A layer with no target is a
        # capture/artifact mismatch (wrong checkpoint, converted with a different
        # hidden size, sharded file), and a merge that silently edits the layers
        # it happens to find is a partial edit reported as a complete one.
        from .gguf_adapters import discover_writeback_targets, gguf_weight_shapes, inferred_hidden_width
        shapes = gguf_weight_shapes(args.gguf_in)
        found = discover_writeback_targets(shapes, int(axes.shape[-1]))
        targets = {L: names for L, names in found['targets'].items() if L in set(ids)}
        missing = sorted(set(ids) - set(targets))
        if missing:
            width = int(axes.shape[-1])
            unmapped = found['unmapped_writeback_width'][:8]
            raise NoVerifiedAdapter(
                f"--gguf-targets auto: this artifact has no residual-write projection at width "
                f"{width} for layers {missing[:8]}"
                f"{'...' if len(missing) > 8 else ''} ({len(missing)} of {len(ids)} layers); "
                f"found residual-write layers {sorted(targets)} with hidden width "
                f"{inferred_hidden_width(shapes)}. The axes and the artifact disagree, so a merge "
                f"here would edit part of the model and look complete. Width-matching tensors it "
                f"would not edit: {unmapped or 'none'}. List what the file carries with "
                f"`python -m jdewitt.spanfit targets --gguf-in <artifact>`; do not rename a tensor "
                f"to make this fit — discovery proves a tensor NAME, and certifying a layout is "
                f"the export smoke, not this merge",
                width=width, missing_layers=missing[:8], missing_layers_total=len(missing),
                found_layers=sorted(targets), unmapped_writeback_width=unmapped,
                artifact=str(args.gguf_in))
        keys = sorted(name for names in targets.values() for name in names)
        name_map = {name: name for name in keys}
    else:
        keys = []
        for k in args.source_key:
            if "*" in k:
                keys += [k.replace("*", str(L)) for L in ids]
            else:
                keys.append(k)
    if not keys:
        raise SystemExit("no target tensors: --source-key expanded to nothing; run "
                         "`python -m jdewitt.spanfit targets --gguf-in <file>` to see "
                         "the keys this artifact actually carries (a merge that edits "
                         "0 tensors still exits 0)")
    out_path = args.gguf_in if args.in_place else args.out
    cache = False if args.no_dequant_cache else args.dequant_cache
    rep = build_merged(args.gguf_in, out_path, axes, keys, name_map=name_map,
                       scale=args.scale, axis_space=args.axis_space, in_place=args.in_place,
                       dequant_cache=cache)
    # `tensors_edited` counts tensors the merge visited, not tensors whose bytes moved:
    # a scale-0.0 run reports 36 tensors edited with changed_blocks 0 in every one of
    # them and exits 0. What a no-op actually looks like is zero changed blocks, so that
    # is the number checked, and it is put in the report so nothing downstream has to
    # infer it from the tensor list.
    changed = sum(r['changed_blocks'] for r in rep['tensors'])
    rep['tensors_changed'] = sum(1 for r in rep['tensors'] if r['changed_blocks'])
    rep['changed_blocks_total'] = int(changed)
    if rep['tensors_changed'] == 0:
        if rep.get('revert_sidecar'):
            os.unlink(rep['revert_sidecar'])
        elif args.out and os.path.exists(args.out):
            os.unlink(args.out)
        raise SystemExit(
            f"build moved 0 bytes across {len(rep['tensors'])} visited tensors "
            f"(scale={args.scale}, axis_space={args.axis_space}, shape={args.shape}, "
            f"axes={list(axes.shape)}): nothing was written. That is not a result to "
            f"print — the axes file, the side, or the layer plan does not match this "
            f"artifact.")
    rep["elapsed_s"] = rep.get("elapsed_s", round(time.monotonic() - t0, 1))
    if args.time_log:
        from .timing import record
        record(args.time_log, lineage=os.path.basename(out_path),
               phase='merge', seconds=rep['elapsed_s'], kind='cpu')
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
