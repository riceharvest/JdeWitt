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
                 source_keys: list[str], *, name_map=None, scale: float = 1.0) -> dict:
    """Merge per-layer rank-1 removals into `gguf_in`, write `gguf_out` (R8 path).

    source_keys are HF-style names ('model.layers.<L>.<suffix>'); each must resolve
    to a gguf tensor whose input dim matches axes.shape[1]. Layout follows the
    ggufio convention: file order = reversed logical dims, i.e. ggml [in,out].
    """
    from .gguf_adapters import gguf_weight_shapes, resolve_gguf_names
    from .ggufio import dequantize_tensor, edit_tensors

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

    deltas = {}
    skipped = []
    for key, base in names.items():
        layer = int([p for p in key.split(".") if p.isdigit()][0])
        w = dequantize_tensor(gguf_in, base, reader=reader)  # ggml [in, out]
        axis = axes[layer]
        if w.shape[0] != axis.shape[0]:
            skipped.append(base)
            continue
        deltas[base] = -scale * (axis.reshape(-1, 1).numpy() @
                            (axis.reshape(1, -1).numpy() @ w)).astype(np.float32)
    report = edit_tensors(gguf_in, gguf_out, deltas)
    report["skipped"] = skipped
    report["elapsed_s"] = round(time.monotonic() - t0, 1)
    return report


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
    a.add_argument("--records", required=True)
    a.add_argument("--gguf-in", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--source-key", action="append", required=True,
                   help="repeatable; '*' expands 0..NL-1 (e.g. model.layers.*.self_attn.o_proj.weight)")
    a.add_argument("--layers", type=int, default=0, help="NL for '*' expansion")
    a.add_argument("--shape", choices=("perlayer", "shared"), default="perlayer")
    a.add_argument("--scale", type=float, default=1.0,
                   help="projection multiplier; 1.0 removes the axis completely, "
                        "1.5 is the certified q25 SAS scale (the curve is knife-edge, METHODOLOGY R8)")
    m = sub.add_parser("axes", help="fit axes and save the [L,W] tensor")
    m.add_argument("--records", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--cos-to", help="report max cosine distance vs another axes tensor")
    m.add_argument("--shape", choices=("perlayer", "shared"), default="perlayer")
    for p in (a, m):
        p.add_argument("--time-log", help="JSONL ledger; logs fit/merge phases")
    args = ap.parse_args(argv)

    t0 = time.monotonic()
    recs = load_records(args.records)
    axes = (shared_axis(recs)[0] if args.shape == "shared"
            else per_layer_axes(recs))
    if args.time_log:
        from .timing import record
        record(args.time_log, lineage=os.path.basename(args.records),
               phase='fit', seconds=time.monotonic() - t0, kind='cpu')
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

    keys = []
    for k in args.source_key:
        if "*" in k:
            keys += [k.replace("*", str(L)) for L in range(args.layers)]
        else:
            keys.append(k)
    rep = build_merged(args.gguf_in, args.out, axes, keys, scale=args.scale)
    rep["elapsed_s"] = rep.get("elapsed_s", round(time.monotonic() - t0, 1))
    if args.time_log:
        from .timing import record
        record(args.time_log, lineage=os.path.basename(args.out),
               phase='merge', seconds=rep['elapsed_s'], kind='cpu')
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
