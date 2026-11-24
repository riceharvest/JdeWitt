#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Config sensitivity (METHODOLOGY R9.4): which verdicts flip between two serve configs.

R9 pins certification at `-c 4096 --parallel 4`. The banked numbers were taken at
8192; a row whose verdict depends on the context is `boundary`, not evidence, so the
pinned number must be reported with the flip count beside it and rows that flip are
excluded from `robustly_removed`.

    uv run python verification/p7-q3-4b/compare_configs.py \
        --pinned verification/p7-q3-4b/waves/removal-arm \
        --other  verification/p7-q3-4b/sensitivity-8192/waves/removal-arm \
        --out    verification/p7-q3-4b/sensitivity-removal.json

Reads saved waves only (no model, no GPU). Both sides are graded with the shipped
RefusalChecker on the saved answer text under BOTH inventories, exactly as
`release-certify` does, so the comparison cannot disagree with the packet.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from jdewitt.answers import atomic_json  # noqa: E402
from jdewitt.refusal_check import RefusalChecker  # noqa: E402


def load(directory: Path) -> tuple[dict, dict[str, dict]]:
    wave_path = directory / "wave.json"
    predictions = directory / "predictions.jsonl"
    if not wave_path.is_file() or not predictions.is_file():
        raise SystemExit(f"{directory} is not a wave (needs wave.json + predictions.jsonl)")
    rows = {row["id"]: row for row in
            (json.loads(line) for line in predictions.read_text().splitlines() if line.strip())}
    return json.loads(wave_path.read_text()), rows


def grade(rows: dict[str, dict], wide: bool) -> dict[str, bool]:
    """True = REMOVED (the answer carries no refusal wording under this inventory).

    Same direction as `release-certify`: a row counts as removed when the shipped
    checker finds no refusal in the saved answer, so the counts line up with the
    packet instead of reading inverted.
    """
    checker = RefusalChecker()
    return {pid: checker.check(row.get("answer") or "", row.get("language") or "en",
                               wide=wide)["result"] != "refusal_detected"
            for pid, row in rows.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pinned", type=Path, required=True)
    parser.add_argument("--other", type=Path, required=True)
    parser.add_argument("--axis", default="removal")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    pinned_wave, pinned_rows = load(args.pinned)
    other_wave, other_rows = load(args.other)
    shared = sorted(set(pinned_rows) & set(other_rows))
    if not shared:
        raise SystemExit(f"{args.pinned} and {args.other} share no prompt id; not comparable")

    block = {}
    for wide, name in ((False, "shipped"), (True, "wide")):
        left, right = grade(pinned_rows, wide), grade(other_rows, wide)
        flipped = [pid for pid in shared if left[pid] != right[pid]]
        block[name] = {
            "pinned_removed": sum(1 for pid in shared if left[pid]),
            "other_removed": sum(1 for pid in shared if right[pid]),
            "paired": len(shared),
            "flipped": len(flipped),
            "flipped_ids": flipped,
            "robustly_removed": sum(1 for pid in shared if left[pid] and not right[pid]),
            "boundary_ids": flipped,
            "flip_detail": [
                {"id": pid, "pinned_verdict": "removed" if left[pid] else "not_removed",
                 "other_verdict": "removed" if right[pid] else "not_removed",
                 "pinned_answer_chars": len(pinned_rows[pid].get("answer") or ""),
                 "other_answer_chars": len(other_rows[pid].get("answer") or "")}
                for pid in flipped],
        }
    document = {
        "schema": "jdewitt.p7-config-sensitivity/1",
        "axis": args.axis,
        "pinned": {"wave": str(args.pinned), "serve_config": pinned_wave.get("serve_config"),
                   "cap": pinned_wave.get("max_tokens"), "rows": pinned_wave.get("rows")},
        "other": {"wave": str(args.other), "serve_config": other_wave.get("serve_config"),
                  "cap": other_wave.get("max_tokens"), "rows": other_wave.get("rows")},
        "inventories": block,
        "reading": ("a verdict that flips between the two configs is `boundary`: the row is "
                    "excluded from robustly_removed and the pinned number is reported with "
                    "the flip count beside it (R9.4). Same-ctx runs must not be compared here"),
        "note": "saved-wave analysis only; both sides graded with the shipped RefusalChecker",
    }
    atomic_json(args.out, document)
    print(json.dumps({name: {key: block[name][key] for key in
                             ("pinned_removed", "other_removed", "flipped", "robustly_removed")}
                      for name in block}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
