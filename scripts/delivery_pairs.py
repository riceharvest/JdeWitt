# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read instrument for delivery pairs: list candidates, record verdicts, count.

The schema lives in ``jdewitt/delivery_pairs.py`` and is shared by both sources
(``capture --pairs-out`` and the public response-mirror importer), so this one
script reads either corpus. A pair's delivery side is a shape CANDIDATE until an
agent opens it; ``vet`` is the only way a verdict enters the file, and every
verdict carries the reason it was reached and who read it (METHODOLOGY R2).

  python scripts/delivery_pairs.py list  pairs.jsonl
  python scripts/delivery_pairs.py show  pairs.jsonl --pair PAIR_ID
  python scripts/delivery_pairs.py vet   pairs.jsonl --pair PAIR_ID \
         --verdict pivot --reason "lecture, no procedure" --reader hermes#2372
  python scripts/delivery_pairs.py summary pairs.jsonl --min-vetted 10

``summary`` exits 1 when fewer than ``--min-vetted`` real deliveries are
recorded, so a card can cite the number instead of assuming the gate passed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jdewitt.delivery_pairs import VERDICTS, read_pairs, summarize, vet, write_pairs


def _find(pairs, pair_id):
    for pair in pairs:
        if pair["pair_id"] == pair_id:
            return pair
    raise SystemExit(f"no pair {pair_id!r}; list the file to see the ids")


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="one line per pair: id, frames, shape, verdict")
    listing.add_argument("pairs", type=Path)
    show = sub.add_parser("show", help="the full record, answers included")
    show.add_argument("pairs", type=Path)
    show.add_argument("--pair", required=True)
    record = sub.add_parser("vet", help="record an agent read of one candidate")
    record.add_argument("pairs", type=Path)
    record.add_argument("--pair", required=True)
    record.add_argument("--verdict", required=True, choices=VERDICTS)
    record.add_argument("--reason", required=True)
    record.add_argument("--reader", required=True)
    summary = sub.add_parser("summary", help="counts; exits 1 below --min-vetted deliveries")
    summary.add_argument("pairs", type=Path)
    summary.add_argument("--min-vetted", type=int, default=None)
    args = ap.parse_args(argv)

    pairs = read_pairs(args.pairs)
    if args.command == "list":
        for pair in pairs:
            delivery = pair.get("delivery") or {}
            refusal = pair.get("refusal") or {}
            print(json.dumps({"pair_id": pair["pair_id"], "kind": pair.get("kind"),
                              "refusal_frame": refusal.get("frame"),
                              "delivery_frame": delivery.get("frame"),
                              "delivery_shape": delivery.get("shape"),
                              "verdict": pair.get("verdict")}))
        return 0
    if args.command == "show":
        print(json.dumps(_find(pairs, args.pair), ensure_ascii=False, indent=2))
        return 0
    if args.command == "vet":
        vet(_find(pairs, args.pair), args.verdict, args.reason, args.reader)
        write_pairs(args.pairs, pairs)
        print(json.dumps(summarize(pairs), indent=2))
        return 0
    result = summarize(pairs, args.min_vetted)
    print(json.dumps(result, indent=2))
    if args.min_vetted is not None and not result["sufficient_for_fit"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
