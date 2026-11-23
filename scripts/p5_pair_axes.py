# SPDX-License-Identifier: AGPL-3.0-or-later
"""Save the pair-diff delivery axis as a mergeable axes file (zero tokens).

``scripts/pair_axis_check.py`` fits this axis and prints the cos against the banked
oracle, but keeps the fitted tensor in memory: the check decides whether a merge is
justified, it does not hand the merge its input. This script fits the *same* rows
with the *same* call (``spanfit.per_layer_axes(..., pairs=True)`` over the vetted
pairs, relabelled from the agent verdicts) and writes ``{"axes": [L,W]}`` in the
shape ``spanfit build --axes`` reads, so the artifact is provably the axis the gate
put at mean |cos| <= 0.3 and not a re-fit.

Inputs must be the gate packet's own files; the packet's sha256s are checked and
recorded so a merge from different rows cannot claim the gate's decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from pair_axis_check import fit_records_for_vetted_pairs, pair_report  # noqa: E402


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    import torch
    from jdewitt.delivery_pairs import read_pairs
    from jdewitt.spanfit import load_records, per_layer_axes

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records", required=True, help="capture records .pt (spanfit format)")
    ap.add_argument("--verdicts", required=True, help="delivery-pair JSONL with agent verdicts")
    ap.add_argument("--gate-packet", default=None,
                    help="pair-axis-check packet the merge relies on; its recorded "
                         "sha256s are enforced when given")
    ap.add_argument("--out", required=True, help="axes .pt to write")
    ap.add_argument("--meta", default=None, help="optional JSON provenance beside the .pt")
    args = ap.parse_args(argv)

    records_sha, pairs_sha = sha256(args.records), sha256(args.verdicts)
    gate = None
    if args.gate_packet:
        gate = json.loads(Path(args.gate_packet).read_text())
        if gate.get("decision") != "orthogonal":
            raise SystemExit(f"gate packet decision is {gate.get('decision')!r}, not "
                             f"'orthogonal': no merge is justified by {args.gate_packet}")
        for label, mine, theirs in (
                ("records", records_sha, (gate.get("records") or {}).get("sha256")),
                ("verdicts", pairs_sha, (gate.get("verdicts") or {}).get("sha256"))):
            if theirs and theirs != mine:
                raise SystemExit(f"{label} sha256 {mine[:16]} != gate packet's "
                                 f"{theirs[:16]}: these are not the gate's rows")

    records = load_records(args.records)
    pairs = read_pairs(args.verdicts)
    report = pair_report(pairs)
    if report["n_vetted_pairs"] < 10:
        raise SystemExit(f"insufficient: {report['n_vetted_pairs']} vetted delivery pairs "
                         f"(< 10); the gate would not have run")
    rows = fit_records_for_vetted_pairs(records, pairs)
    axes = per_layer_axes(rows, "bad", "deliver", pairs=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"axes": axes}, out)

    meta = {
        "schema": "jdewitt.p5-pair-axes/1",
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": "spanfit.per_layer_axes(records, 'bad', 'deliver', pairs=True) over "
                  "vetted pairs relabelled from committed agent verdicts",
        "records": {"path": str(args.records), "sha256": records_sha},
        "verdicts": {"path": str(args.verdicts), "sha256": pairs_sha,
                     "n_vetted_pairs": report["n_vetted_pairs"],
                     "by_verdict": report["by_verdict"]},
        "gate_packet": args.gate_packet,
        "gate_sha256_by_key": None if gate is None else
            {k: gate.get(k) for k in ("mean_cos", "mean_abs_cos", "decision")},
        "axes": {"path": str(out), "sha256": sha256(out),
                 "shape": [int(x) for x in axes.shape], "dtype": str(axes.dtype)},
        "rows_used": len(rows),
    }
    if args.meta:
        Path(args.meta).parent.mkdir(parents=True, exist_ok=True)
        Path(args.meta).write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
