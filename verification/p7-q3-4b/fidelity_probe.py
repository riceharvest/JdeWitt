#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure the realized fidelity of the FITTED arm on the artifact that serves it.

The banked packet carried a codec proxy (`arm_measurement: false`) because the arm
file had been deleted. This run rebuilds the arm, so the reading can be the arm's
own: each edited tensor is re-encoded through the shipped Q8_0 codec at the arm's
real fitted direction, and the realized change is compared with the change asked
for. cos == 1.0 would mean the codec stored the edit exactly.

    uv run python verification/p7-q3-4b/fidelity_probe.py \
        --gguf /mnt/ssd/hermes/models/q3-4b-f15s15-l935-refit.gguf \
        --axes /mnt/ssd/hermes/models/_fit/q3-shared-axes-refit.pt \
        --scale 1.5 --out verification/p7-q3-4b/fidelity.json

`min_cos` here is the release matrix's `fidelity_min_cos` reading (tune's
--min-edit-cos default is 0.9): it says whether the codec can store this edit, not
whether the direction removes refusal. The shape audit and the waves answer that.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from jdewitt.answers import atomic_json  # noqa: E402
from jdewitt.ggufio import preview_edit  # noqa: E402

ATTN = "blk.{layer}.attn_output.weight"
FFN = "blk.{layer}.ffn_down.weight"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--axes", type=Path, required=True,
                        help="spanfit axes file (torch.save {'axes': [L, W]})")
    parser.add_argument("--axes-key", default="axes")
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--layers", default="15,25,35",
                        help="comma list of layers to sample (edit band is 15-35)")
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--label", default="f15s15-l935-refit")
    parser.add_argument("--lineage", default="qwen3-4b-instruct-2507")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    import torch

    payload = torch.load(args.axes, map_location="cpu", weights_only=False)
    if args.axes_key not in payload:
        raise SystemExit(f"{args.axes} has keys {sorted(payload)}, not {args.axes_key!r}")
    axes = payload[args.axes_key]
    layers = [int(part) for part in args.layers.split(",") if part.strip()]
    if axes.dim() != 2:
        raise SystemExit(f"axes is {tuple(axes.shape)}, expected [L, W] (spanfit axes)")
    for layer in layers:
        if layer >= axes.shape[0]:
            raise SystemExit(f"layer {layer} is outside an axes tensor of {axes.shape[0]} layers")

    edits = {}
    for layer in layers:
        axis = axes[layer].float().numpy()
        edits[ATTN.format(layer=layer)] = (axis, "output")
        edits[FFN.format(layer=layer)] = (axis, "output")
    report = preview_edit(str(args.gguf), edits, scale=args.scale,
                          windows=args.windows, blocks=args.blocks)
    if not report:
        raise SystemExit("preview_edit measured no tensor; the names or the artifact are wrong")
    cosmos = [float(row["cos"]) for row in report]
    document = {
        "schema_version": 1,
        "lineage": args.lineage,
        "label": args.label,
        "measured": True,
        "arm_measurement": True,
        "arm_refit": True,
        "method": (f"ggufio.preview_edit on the served Q8_0 codec at the FITTED refit "
                   f"direction, scale {args.scale}, {args.windows} windows x {args.blocks} "
                   "blocks per tensor"),
        "artifact_sha256": sha256(args.gguf),
        "artifact_bytes": args.gguf.stat().st_size,
        "axes_sha256": sha256(args.axes),
        "scale": args.scale,
        "tensors": report,
        "sampled": len(report),
        "min_cos": round(min(cosmos), 4),
        "mean_cos": round(sum(cosmos) / len(cosmos), 4),
        "min_snr": min(float(row["snr"]) for row in report),
        "note": ("a low cos here means the codec, not the axis, decides what ships; this is "
                 "the arm's own fitted direction, unlike a codec proxy"),
        "scope": ("realized codec fidelity of the refit arm artifact: each sampled edited "
                  "tensor is re-encoded through Q8_0 with the fitted axis applied at the "
                  "certified scale. It measures what the codec stores, not the behavioural "
                  "effect of the edit"),
    }
    atomic_json(args.out, document)
    print(json.dumps({key: document[key] for key in
                      ("sampled", "min_cos", "mean_cos", "min_snr", "artifact_sha256")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
