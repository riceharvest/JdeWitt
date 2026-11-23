# SPDX-License-Identifier: AGPL-3.0-or-later
"""The delivery-pair corpus: one refusal answer and one delivery answer per request.

A pair is the fit's unit (METHODOLOGY R3/R4): the SAME harmful request answered
twice, once by a refusal and once by a real delivery, so the axis fitted on the
difference is a delivery axis and not a wording axis. Two sources produce the
raw material and both write this schema:

  * ``capture`` -- a plain-prompt answer (banked or generated) and the answer to
    the same prompt under a pinned ``jdewitt.frames`` jailbreak frame;
  * ``response-mirror`` -- a pinned public prompt+response corpus where the
    response is a real delivery for that prompt.

The shape instrument never decides a pair. A ``procedural_no_decline`` answer is
a CANDIDATE; its ``verdict`` stays null until an agent reads the answer against
the request (R2). ``scripts/delivery_pairs.py`` is the read instrument that
records the verdict, and only pairs with ``verdict == "deliver"`` are usable for
a fit. A shape-clean-but-pivot row is ``pivot``, not ``deliver`` -- that is the
error the first jailbreak round measured 13/13 times on shape-clean candidates
(PLAN-NEXT 2026-11-18).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

SCHEMA = "jdewitt.delivery-pair/1"
SOURCE_KINDS = ("capture", "response-mirror")
# deliver: the answer actually delivers the asked-for content.
# pivot:   long answer, does not deliver (lecture/prevention/alternative).
# declined: explicit refusal.
# other:   degenerate, truncated, unreadable, or off-request.
VERDICTS = ("deliver", "pivot", "declined", "other")

SIDE_FIELDS = ("frame", "answer", "shape", "declining", "answer_source", "chars", "seed")


def make_pair(*, prompt_id, prompt, kind, family_id, category, source, refusal, delivery):
    """Build one schema record with an unset verdict.

    ``refusal`` and ``delivery`` are sides: ``frame`` ('none' or a frame name),
    ``answer``, ``shape`` (the classifier label), ``declining`` (bool),
    ``answer_source`` ('banked'|'generated'|'mirror'), ``chars`` and ``seed``.
    Either side may be None when the source could not supply it; the record then
    stays in the file as an incomplete candidate and can never be vetted.
    """
    pair_id = f"{source['kind']}:{prompt_id}:{(refusal or {}).get('frame', 'none')}+{(delivery or {}).get('frame', 'none')}"
    return {
        "schema": SCHEMA,
        "pair_id": pair_id,
        "prompt_id": prompt_id,
        "family_id": family_id,
        "category": category,
        "kind": kind,
        "prompt": prompt,
        "source": dict(source),
        "refusal": _side(refusal),
        "delivery": _side(delivery),
        "verdict": None,
        "verdict_reason": None,
        "reader": None,
    }


def _side(side):
    if side is None:
        return None
    return {field: side.get(field) for field in SIDE_FIELDS}


def check_pair(pair) -> list[str]:
    """Schema violations in one record; an empty list means it is well formed."""
    problems = []
    if not isinstance(pair, dict):
        return ["record is not an object"]
    if pair.get("schema") != SCHEMA:
        problems.append(f"schema is {pair.get('schema')!r}, expected {SCHEMA!r}")
    for field in ("pair_id", "prompt_id", "prompt"):
        if not isinstance(pair.get(field), str) or not pair.get(field):
            problems.append(f"{field} must be a nonempty string")
    source = pair.get("source")
    if not isinstance(source, dict) or source.get("kind") not in SOURCE_KINDS:
        problems.append(f"source.kind must be one of {SOURCE_KINDS}")
    for field in ("refusal", "delivery"):
        side = pair.get(field)
        if side is None:
            continue
        if not isinstance(side, dict):
            problems.append(f"{field} must be an object or null")
            continue
        if not isinstance(side.get("answer"), str):
            problems.append(f"{field}.answer must be a string")
        if not isinstance(side.get("frame"), str) or not side.get("frame"):
            problems.append(f"{field}.frame must be a nonempty string")
    if pair.get("verdict") is not None and pair["verdict"] not in VERDICTS:
        problems.append(f"verdict must be null or one of {VERDICTS}")
    if pair.get("verdict") is not None:
        if not isinstance(pair.get("verdict_reason"), str) or not pair.get("verdict_reason"):
            problems.append("a vetted pair needs verdict_reason")
        if not isinstance(pair.get("reader"), str) or not pair.get("reader"):
            problems.append("a vetted pair needs reader")
    return problems


def read_pairs(path) -> list[dict]:
    pairs = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            pair = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number} is not JSON: {error}") from error
        problems = check_pair(pair)
        if problems:
            raise ValueError(f"{path}:{number}: " + "; ".join(problems))
        pairs.append(pair)
    return pairs


def write_pairs(path, pairs: list[dict]) -> Path:
    """Atomic JSONL write in schema order; refuses a malformed record."""
    for index, pair in enumerate(pairs, 1):
        problems = check_pair(pair)
        if problems:
            raise ValueError(f"pair {index}: " + "; ".join(problems))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for pair in pairs:
                handle.write(json.dumps(pair, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise
    return path


def vet(pair: dict, verdict: str, reason: str, reader: str) -> dict:
    """Record an agent's read of one candidate IN PLACE; returns the record."""
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("a verdict needs a reason (what the answer does against the request)")
    if not isinstance(reader, str) or not reader.strip():
        raise ValueError("a verdict needs a reader")
    if pair.get("refusal") is None or pair.get("delivery") is None:
        raise ValueError(f"{pair.get('pair_id')}: an incomplete pair cannot be vetted")
    pair["verdict"], pair["verdict_reason"], pair["reader"] = verdict, reason.strip(), reader.strip()
    return pair


def summarize(pairs: list[dict], min_vetted: int | None = None) -> dict:
    """Counts a result can cite: candidates, reads, usable deliveries, rejects."""
    by_verdict = {verdict: 0 for verdict in VERDICTS}
    for pair in pairs:
        if pair.get("verdict") in by_verdict:
            by_verdict[pair["verdict"]] += 1
    vetted = sum(by_verdict.values())
    summary = {
        "pairs": len(pairs),
        "candidates_unread": len(pairs) - vetted,
        "vetted": vetted,
        "by_verdict": by_verdict,
        "n_vetted_pairs": by_verdict["deliver"],
        "incomplete_sides": sum(1 for pair in pairs if pair.get("refusal") is None or pair.get("delivery") is None),
        "source_kinds": sorted({str((pair.get("source") or {}).get("kind")) for pair in pairs}),
        "frames": sorted({str((pair.get("delivery") or {}).get("frame")) for pair in pairs}),
    }
    if min_vetted is not None:
        summary["min_vetted"] = min_vetted
        summary["sufficient_for_fit"] = by_verdict["deliver"] >= min_vetted
    return summary
