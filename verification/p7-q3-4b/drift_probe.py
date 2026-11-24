#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Control-corpus drift for one arm: llama-perplexity on base and arm, ln-PPL delta.

Drift is the one axis that is not a wave: it is a prefill-only perplexity reading
over benign control text (`verification/ppl-probe/control.txt`), the same instrument
and corpus the banked screen used (ctx 512, 16 threads). The packet needs a measured
delta and the budget it is judged against; this writes exactly that document.

    uv run python verification/p7-q3-4b/drift_probe.py \
        --base /mnt/ssd/hermes/models/q3-4b-base-q8.gguf \
        --arm  /mnt/ssd/hermes/models/q3-4b-f15s15-l935-refit.gguf \
        --out verification/p7-q3-4b/drift.json

A move here is generic degeneration, not refusal removal (that is the shape audit's
job); a delta inside the budget means the edit did not cost the model its grip on
ordinary text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from jdewitt.answers import atomic_json  # noqa: E402

DEFAULT_BINARY = "/home/dario/llama.cpp-spark/build-cpu/bin/llama-perplexity"
DEFAULT_CONTROL = REPO / "verification/ppl-probe/control.txt"
FINAL = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(binary: str, model: Path, corpus: Path, ctx: int, threads: int) -> dict:
    argv = [binary, "-m", str(model), "-f", str(corpus), "-c", str(ctx), "-t", str(threads)]
    started = time.monotonic()
    process = subprocess.run(argv, capture_output=True, text=True, timeout=3600)
    seconds = round(time.monotonic() - started, 1)
    if process.returncode != 0:
        raise SystemExit(f"llama-perplexity exited {process.returncode} for {model}:\n"
                         f"{process.stderr[-1500:]}")
    matches = FINAL.findall(process.stdout + process.stderr)
    if not matches:
        raise SystemExit(f"llama-perplexity printed no Final estimate for {model}:\n"
                         f"{(process.stdout + process.stderr)[-1500:]}")
    ppl, uncertainty = (float(value) for value in matches[-1])
    return {"ln_ppl": round(math.log(ppl), 4), "ppl": ppl,
            "uncertainty": uncertainty, "seconds": seconds, "argv": argv}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--binary", default=DEFAULT_BINARY)
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--budget", type=float, default=0.02)
    parser.add_argument("--label", default="f15s15-l935-refit")
    parser.add_argument("--lineage", default="qwen3-4b-instruct-2507")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not Path(args.binary).is_file():
        raise SystemExit(f"llama-perplexity not found at {args.binary}; pass --binary")
    for path in (args.base, args.arm, args.control):
        if not path.is_file():
            raise SystemExit(f"{path} does not exist")

    base = run(args.binary, args.base, args.control, args.ctx, args.threads)
    arm = run(args.binary, args.arm, args.control, args.ctx, args.threads)
    delta = round(arm["ln_ppl"] - base["ln_ppl"], 4)
    document = {
        "schema_version": 1,
        "lineage": args.lineage,
        "label": args.label,
        "measured": True,
        "arm_refit": True,
        "instrument": f"llama-perplexity, ctx {args.ctx}, {args.threads} threads; "
                      "control corpus, prefill only",
        "control_delta_ln_ppl": delta,
        "budget": args.budget,
        "inside_budget": abs(delta) <= args.budget,
        "artifacts": {"base": {"path": str(args.base), "sha256": sha256(args.base),
                               **{key: base[key] for key in ("ln_ppl", "ppl", "uncertainty",
                                                              "seconds")}},
                      "arm": {"path": str(args.arm), "sha256": sha256(args.arm),
                              **{key: arm[key] for key in ("ln_ppl", "ppl", "uncertainty",
                                                             "seconds")}}},
        "control_corpus": {"path": str(args.control), "sha256": sha256(args.control),
                           "bytes": args.control.stat().st_size},
        "runner": {"binary": args.binary, "ctx": args.ctx, "threads": args.threads},
        "note": ("the control-corpus drift is a generic-degeneration reading, not a "
                 "delivery or removal claim; delta is ln(ppl_arm) - ln(ppl_base)"),
    }
    atomic_json(args.out, document)
    print(json.dumps({key: document[key] for key in
                      ("control_delta_ln_ppl", "budget", "inside_budget")}, indent=1))
    return 0 if document["inside_budget"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
