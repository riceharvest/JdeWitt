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
import json
import os
import time

import numpy as np
import torch


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


# ---------- edit ----------

def removal_delta(weight: torch.Tensor, axis: torch.Tensor,
                  strength: float = 1.0) -> torch.Tensor:
    """-(a (a^T W)); axis acts on dim 0 of W ([out, in]), i.e. the layer's
    residual-write side. Caller supplies matching layouts (torch params or
    ggufio logical [out, in] dequantized arrays)."""
    a = axis.reshape(-1, 1).to(weight.dtype)
    return -(strength * (a @ (a.T @ weight)))


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
                 axis_space: str = "input", in_place: bool = False) -> dict:
    """Merge per-layer rank-1 removals into `gguf_in`, write `gguf_out` (R8 path).

    source_keys are HF-style names ('model.layers.<L>.<suffix>') or, through
    `name_map`, the artifact's own tensor names; each must resolve to a gguf
    tensor whose input dim matches axes.shape[1]. Layout follows the ggufio
    convention: file order = reversed logical dims, i.e. ggml [in,out].

    in_place patches `gguf_in`'s own bytes (no copy) and leaves a revert sidecar
    beside it; that is the path an agent should use on a model it already serves,
    because the artifact is quantized in place and never re-quantized as a whole.
    """
    from .gguf_adapters import gguf_weight_shapes, resolve_gguf_names
    from .ggufio import dequantize_tensor, edit_tensors, project_tensors_inplace

    t0 = time.monotonic()
    shapes = gguf_weight_shapes(gguf_in)
    names = resolve_gguf_names(source_keys, shapes) if name_map is None else name_map
    reader = None
    try:
        from gguf import GGUFReader  # provided by dspark gguf-py on sys.path
        reader = GGUFReader(gguf_in)
    except Exception:
        pass
    if not names:
        raise SystemExit("no target tensors resolved from source keys")

    projections = {}
    deltas = {}
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
        if in_place:
            # The in-place writer computes the delta from the weight it is already
            # dequantizing, so the axis is the only thing that crosses over.
            projections[base] = (axis.detach().cpu().numpy().astype(np.float32), axis_space)
            continue
        w = dequantize_tensor(gguf_in, base, reader=reader)  # ggml [in, out]
        a = axis.reshape(-1, 1).numpy()
        if axis_space == "output":
            # axis lives in the space W writes into (the residual stream):
            # dW = -s * u (u^T W). This is the form a residual-stream capture
            # gives you, and the only one usable when the module's input dim
            # differs from hidden (qwen3-4b o_proj: 4096 in, 2560 out).
            deltas[base] = -(scale * (a @ (a.T @ w))).astype(np.float32)
        else:
            wt = w.T
            deltas[base] = -(scale * (a @ (a.T @ wt))).T.astype(np.float32)
        del w
    if in_place:
        # One file, byte-level: the artifact keeps its quantization, its size and
        # every offset it had; only the changed tiles move, and the sidecar keeps
        # the pre-patch bytes so this is reversible.
        report = project_tensors_inplace(gguf_in, projections, scale=scale,
                                         revert_path=str(gguf_in) + '.revert.npz')
    else:
        report = edit_tensors(gguf_in, gguf_out, deltas)
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
    from .gguf_adapters import HF_TO_LLAMA, WRITEBACK_TENSOR, discover_writeback_targets, gguf_weight_shapes
    shapes = gguf_weight_shapes(gguf_in)
    # The generic view: which residual-write projections this artifact carries at
    # its own hidden width. This is what `build --gguf-targets auto` uses, so an
    # agent can look before it merges instead of reading architecture notes.
    widths = {}
    for name, dims in shapes.items():
        suffix = name.rsplit('.', 2)[-2] if name.endswith('.weight') else ''
        if len(dims) == 2 and WRITEBACK_TENSOR.match(suffix):
            widths[int(dims[1])] = widths.get(int(dims[1]), 0) + 1
    hidden = max(widths.items(), key=lambda item: item[1])[0] if widths else None
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
    for p in (a, m, r):
        p.add_argument("--time-log", help="JSONL ledger; logs fit/merge phases")
    args = ap.parse_args(argv)

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
        from .gguf_adapters import discover_writeback_targets, gguf_weight_shapes
        found = discover_writeback_targets(gguf_weight_shapes(args.gguf_in), int(axes.shape[-1]))
        targets = {L: names for L, names in found['targets'].items() if L in set(ids)}
        missing = sorted(set(ids) - set(targets))
        if missing:
            raise SystemExit(
                f"--gguf-targets auto: this artifact has no residual-write projection at width "
                f"{int(axes.shape[-1])} for layers {missing[:12]}"
                f"{'...' if len(missing) > 12 else ''}; found layers {sorted(targets)}. The axes "
                f"and the artifact disagree, so a merge here would edit part of the model and "
                f"look complete. Width-matching tensors it would not edit: "
                f"{found['unmapped_writeback_width'][:6]}")
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
    rep = build_merged(args.gguf_in, out_path, axes, keys, name_map=name_map,
                       scale=args.scale, axis_space=args.axis_space, in_place=args.in_place)
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
