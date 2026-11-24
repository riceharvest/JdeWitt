#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build the deterministic capture pool for the labelled f15s15 refit.

The banked arm (``q3-shared-axes.pt`` fitted from a 190-family pool) died with
``/mnt/ssd/hermes/models`` and ``/mnt/ssd/hermes/cache/scratch/probes``. Neither
the axes file nor the 190-pool capture records exist on this host any more, so
the plan cannot be restored byte-for-byte; the P7 spec allows a refit but says it
must never be called the same plan. This script makes that refit reproducible and
auditable instead of ad hoc: one committed suite, fixed strides, no randomness.

The pool is the P4d capture recipe applied to q3-4b (the same selection suite and
the same strides p4d used for Spark), which keeps the refit off the two decks the
cert measures:

  * refusal rows come from ``verification/harsh-suites/selection.jsonl`` -- the
    pool the q3 direction was fitted from, not ``advbench-40`` (the cert removal
    deck) and not ``q3-holdout-wider`` (the cert held-out deck);
  * harmless rows come from the same file, so the "good" side of the contrast is
    the base model complying with a benign prompt, exactly the contrast
    ``spanfit.shared_axis`` expects (bad = declines a harmful ask, good = answers
    a harmless one).

Strides mirror ``verification/p4d/README.md``: refusal ``[::25][:40]``, harmless
``[::8][:40]``. Outputs ``pool.jsonl`` plus a provenance JSON carrying the source
sha256, the strides and the family counts, so a different pool cannot be passed
off as this one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SUITE = REPO / "verification/harsh-suites/selection.jsonl"
REFUSAL_STRIDE, REFUSAL_KEEP = 25, 40
HARMLESS_STRIDE, HARMLESS_KEEP = 8, 40
MIN_PER_KIND = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stride_pick(rows: list[dict], stride: int, keep: int) -> list[dict]:
    picked = [row for index, row in enumerate(rows) if index % stride == 0][:keep]
    return picked


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--out", type=Path, required=True, help="pool.jsonl to write")
    parser.add_argument("--meta", type=Path, default=None,
                        help="provenance JSON (default: <out>.meta.json)")
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in args.suite.read_text().splitlines() if line.strip()]
    refusal = stride_pick([r for r in rows if r.get("kind") == "refusal"],
                          REFUSAL_STRIDE, REFUSAL_KEEP)
    harmless = stride_pick([r for r in rows if r.get("kind") == "harmless"],
                           HARMLESS_STRIDE, HARMLESS_KEEP)
    if len(refusal) < MIN_PER_KIND or len(harmless) < MIN_PER_KIND:
        raise SystemExit(
            f"pool too thin: {len(refusal)} refusal / {len(harmless)} harmless rows "
            f"(need >= {MIN_PER_KIND} of each); a shared-axis fit on fewer rows cannot "
            "carry a label mean")
    pool = refusal + harmless
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in pool))
    meta = {
        "schema": "jdewitt.p7-refit-pool/1",
        "source": str(args.suite),
        "source_sha256": sha256(args.suite),
        "pool": str(args.out),
        "pool_sha256": sha256(args.out),
        "rows": len(pool),
        "kinds": {"refusal": len(refusal), "harmless": len(harmless)},
        "families": len({row.get("family_id") or row["id"] for row in pool}),
        "strides": {"refusal": f"[::{REFUSAL_STRIDE}][:{REFUSAL_KEEP}]",
                    "harmless": f"[::{HARMLESS_STRIDE}][:{HARMLESS_KEEP}]"},
        "recipe_from": "verification/p4d/README.md (same suite, same strides, applied to q3-4b)",
        "labels": "the capture's own shape labels: bad = the base declined a harmful ask, "
                  "good = the base answered a harmless one; nothing is labelled by wording",
        "note": "REFIT pool. The banked q3-shared-axes.pt pool (300 candidates over 190 "
                "families) and its records were deleted with /mnt/ssd/hermes; this pool "
                "reproduces the fit recipe, not the fit.",
    }
    meta_path = args.meta or args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(json.dumps(meta, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
