#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure the realized fidelity of the FITTED arm on the artifact that serves it.

The banked packet carried a codec proxy (`arm_measurement: false`) because the arm
file had been deleted. This run rebuilds the arm, so the reading can be the arm's
own. Two readings, both per edited tensor:

* `preview_edit` re-encodes the edited tile through the shipped Q8_0 codec and asks
  whether the codec would store the projection (`--min-edit-cos` is this reading);
* `compare_edit` dequantizes the base and the shipped file and compares the stored
  change with the intended one, so it binds the bytes that are actually served.

`min_cos`/`mean_cos` in the document are the DELIVERED (compare_edit) numbers: the
stricter, file-level reading. The codec simulation is kept beside them.

    uv run python verification/p7-q3-4b/fidelity_probe.py \
        --gguf /mnt/ssd/hermes/models/q3-4b-f15s15-l935-refit.gguf \
        --base /mnt/ssd/hermes/models/q3-4b-base-q8.gguf \
        --axes /mnt/ssd/hermes/models/_fit/q3-shared-axes-refit.pt \
        --scale 1.5 --out verification/p7-q3-4b/fidelity.json
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
from jdewitt.ggufio import compare_edit, preview_edit  # noqa: E402

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
    parser.add_argument("--base", type=Path, required=True,
                        help="pristine artifact the arm was built from (compare_edit source)")
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

    simulated = preview_edit(str(args.gguf), edits, scale=args.scale,
                             windows=args.windows, blocks=args.blocks)
    if not simulated.get("tensors"):
        raise SystemExit("preview_edit measured no tensor; the names or the artifact are wrong")
    delivered = compare_edit(str(args.base), str(args.gguf), edits, scale=args.scale)
    if delivered.get("min_cos") is None:
        raise SystemExit("compare_edit measured no tensor with a cos; base and arm disagree")
    document = {
        "schema_version": 1,
        "lineage": args.lineage,
        "label": args.label,
        "measured": True,
        "arm_measurement": True,
        "arm_refit": True,
        "method": (f"ggufio.compare_edit (delivered bytes) + ggufio.preview_edit (codec "
                   f"simulation) at the FITTED refit direction, scale {args.scale}, "
                   f"{args.windows} windows x {args.blocks} blocks per tensor"),
        "artifact_sha256": sha256(args.gguf),
        "artifact_bytes": args.gguf.stat().st_size,
        "base_sha256": sha256(args.base),
        "axes_sha256": sha256(args.axes),
        "scale": args.scale,
        "tensors": simulated["tensors"],
        "codec_simulation": {key: simulated[key] for key in
                             ("sampled", "min_cos", "mean_cos", "min_snr")},
        "delivered": {key: delivered[key] for key in ("sampled", "min_cos", "mean_cos")},
        "delivered_tensors": delivered["tensors"],
        "sampled": delivered["sampled"],
        "min_cos": delivered["min_cos"],
        "mean_cos": delivered["mean_cos"],
        "min_snr": simulated.get("min_snr"),
        "note": ("min_cos/mean_cos are the DELIVERED reading (stored change vs intended "
                 "projection on the served bytes); codec_simulation is the same question "
                 "asked of the Q8_0 codec alone. A low cos means the codec, not the axis, "
                 "decides what ships"),
        "scope": ("realized codec fidelity of the refit arm artifact: every sampled edited "
                  "tensor is compared between the pristine base and the shipped Q8_0 arm at "
                  "the certified scale. It measures what the file stores, not the "
                  "behavioural effect of the edit"),
    }
    atomic_json(args.out, document)
    print(json.dumps({key: document[key] for key in
                      ("sampled", "min_cos", "mean_cos", "artifact_sha256")}, indent=1))
    print(json.dumps({"codec_simulation": document["codec_simulation"],
                      "delivered": document["delivered"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
