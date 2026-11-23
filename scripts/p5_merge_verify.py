# SPDX-License-Identifier: AGPL-3.0-or-later
"""Merge integrity for a P5 in-place arm: revert exactness, untouched bytes, codec cos.

Three separate claims, each measured rather than inherited from the writer's report:

1. ``untouched``: every tensor the merge did not target is byte-identical between
   the base artifact and the patched one (raw block bytes compared tensor by
   tensor), and the set of tensors that DO differ is exactly the edited footprint.
2. ``preview_edit``: per edited tensor, cos(realized, intended) from
   ``ggufio.preview_edit`` -- the codec proxy, measured against the quant grid the
   artifact actually stores, on the base file (the edit is simulated, not read back).
3. ``revert``: an in-place patch is only honest if it is reversible. The caller
   passes a scratch copy of the patched file and its sidecar; the copy is reverted
   and its sha256 must equal the base artifact's, byte for byte.

Writes one JSON packet; exit 1 if any claim fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(path: Path) -> dict[str, bytes]:
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    out = {}
    with open(path, "rb") as handle:
        for tensor in reader.tensors:
            handle.seek(int(tensor.data_offset))
            out[tensor.name] = handle.read(int(tensor.n_bytes))
    return out


def main(argv=None) -> int:
    import numpy as np
    import torch
    from jdewitt.ggufio import preview_edit

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--axes", required=True)
    ap.add_argument("--axes-key", default="axes")
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--axis-space", default="output")
    ap.add_argument("--layers", default="15-35")
    ap.add_argument("--sides", default="attn_output,ffn_down",
                    help="gguf tensor suffix after blk.N.")
    ap.add_argument("--revert-copy", help="scratch copy of --artifact to revert; required for claim 3")
    ap.add_argument("--revert-sidecar", help="sidecar written by the in-place merge")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    layers = []
    for part in args.layers.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            layers += list(range(int(a), int(b) + 1))
        elif part.strip():
            layers.append(int(part))
    sides = [s.strip() for s in args.sides.split(",") if s.strip()]
    edited_names = [f"blk.{L}.{side}.weight" for L in layers for side in sides]

    base_sha, art_sha = sha256(args.base), sha256(args.artifact)
    base_t, art_t = tensor_bytes(Path(args.base)), tensor_bytes(Path(args.artifact))
    missing = [n for n in edited_names if n not in base_t or n not in art_t]
    if missing:
        raise SystemExit(f"edited tensors missing from a file: {missing[:5]}")

    differing = sorted(n for n in base_t if base_t[n] != art_t.get(n))
    unexpected = sorted(set(differing) - set(edited_names))
    unchanged_edited = sorted(set(edited_names) - set(differing))

    axes = torch.load(args.axes, map_location="cpu", weights_only=False)[args.axes_key]
    edits = {name: (axes[int(name.split(".")[1])].numpy().astype("float32"), args.axis_space)
             for name in edited_names}
    preview = preview_edit(args.base, edits, names=edited_names, scale=args.scale)
    rows = preview.get("tensors") if isinstance(preview, dict) else preview
    cos_values = [float(row.get("cos", row.get("realized_cos", 0.0))) for row in rows]

    packet = {
        "schema": "jdewitt.p5-merge-integrity/1",
        "base": {"path": args.base, "sha256": base_sha, "bytes": Path(args.base).stat().st_size},
        "artifact": {"path": args.artifact, "sha256": art_sha,
                     "bytes": Path(args.artifact).stat().st_size},
        "edited": {"layers": args.layers, "sides": sides, "tensors": len(edited_names),
                   "scale": args.scale, "axis_space": args.axis_space,
                   "axes": {"path": args.axes, "key": args.axes_key}},
        "1_untouched_bytes": {
            "identical": not unexpected and not unchanged_edited,
            "tensors_total": len(base_t),
            "tensors_differing": differing,
            "unexpected_differences": unexpected,
            "edited_but_unchanged": unchanged_edited,
            "note": "raw block bytes compared per tensor; an edited tensor whose bytes are "
                    "identical would mean the merge did not land there",
        },
        "2_preview_edit": {
            "kind": "codec-proxy",
            "per_tensor": rows,
            "cos_mean": round(sum(cos_values) / len(cos_values), 6) if cos_values else None,
            "cos_min": round(min(cos_values), 6) if cos_values else None,
            "cos_max": round(max(cos_values), 6) if cos_values else None,
            "scale": args.scale,
        },
    }

    if args.revert_copy and args.revert_sidecar:
        from jdewitt.ggufio import revert_inplace

        copy = Path(args.revert_copy)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.artifact, copy)
        pre = sha256(copy)
        result = revert_inplace(copy, args.revert_sidecar)
        restored = sha256(copy)
        packet["3_revert"] = {
            **result,
            "scratch_copy": str(copy),
            "scratch_sha256_before_revert": pre,
            "restored_sha256_recomputed": restored,
            "base_sha256": base_sha,
            "exact": restored == base_sha,
            "note": "reverted a copy, not the served artifact",
        }
        copy.unlink()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(packet, indent=1) + "\n")
    print(json.dumps({k: packet[k] for k in packet if k.startswith(("1_", "2_", "3_"))},
                     indent=1)[:4000])
    ok = packet["1_untouched_bytes"]["identical"]
    if "3_revert" in packet:
        ok = ok and packet["3_revert"]["exact"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
