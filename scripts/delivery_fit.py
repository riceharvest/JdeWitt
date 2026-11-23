# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build the fit corpus from the hand-read delivery pairs, with the gate set excluded.

The fit unit is a pair (METHODOLOGY R3/R4): the SAME harmful request answered once
by a refusal and once by a real delivery. This script does not generate or judge
anything. It imports the deliveries a working agent already read and recorded --
``verification/ppl-probe/`` (PLAN-NEXT 2026-11-18: 10 hand-read pairs; the read
list lives in ``scripts/build_ppl_probe.py`` as ``VERIFIED``) -- into the
``jdewitt.delivery-pair/1`` schema, then writes the two ``train``/``holdout``
suite files ``jdewitt train`` consumes.

Two hard guards, neither of which a flag can disable:

* **gate-set exclusion** -- any prompt id that appears in
  ``verification/q3-4b-frames/suite-24.jsonl`` (the 24-family content read) is
  refused, because training on the gate set would make the content number a
  memory read (the leakage rule from the pair-corpus contract). The refusal is
  reported with the ids, never silently dropped.
* **family split** -- train and holdout are cut on ``family_id``, never on rows,
  and the family sets are proved disjoint before the files are written.

The delivery side is the framed capture answer (frame recorded); the refusal side
is the same model's PLAIN answer to the same prompt, taken from the committed
prediction wave, not from the truncated probe text. Both sides are scored against
the plain prompt: the frame only explains how the delivery was produced.

    python scripts/delivery_fit.py build --source-dir verification/ppl-probe \
        --out verification/q3-4b-fit-pairs --holdout-family advbench-o0-g000003
    python scripts/delivery_fit.py check verification/q3-4b-fit-pairs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from jdewitt.delivery_pairs import (  # noqa: E402
    SCHEMA, make_pair, read_pairs, summarize as summarize_pairs, vet, write_pairs)
from jdewitt.refusal_shape import classify  # noqa: E402
from build_ppl_probe import VERIFIED  # noqa: E402

GATE_SET = REPO / "verification/q3-4b-frames/suite-24.jsonl"
# The three fit-split families the capture deliveries came from
# (verification/public-corpus-suites/selection.jsonl). Anything outside the fit
# split -- the 24-family gate set and the 120 wider holdout -- is excluded.
FIT_FAMILIES = {
    "advbench-train-000007": "advbench-o0-g000003",
    "advbench-train-000021": "advbench-o0-g000010",
    "advbench-train-000025": "advbench-o0-g000012",
}
# Plain-prompt refusal side (committed wave; the same model that produced the
# framed delivery). capture ids only: oracle/wider rows belong to a held-out screen.
REFUSAL_WAVES = (
    "verification/q3-4b-harmful40/q3base/predictions.jsonl",
)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gate_ids():
    return {json.loads(line)["id"] for line in GATE_SET.read_text().splitlines() if line.strip()}


def prompts_by_id():
    prompts = {}
    for rel in REFUSAL_WAVES + ("verification/public-corpus-suites/selection.jsonl",
                                "verification/q3-4b-holdout/hold-base/predictions.jsonl",
                                "verification/q3-4b-holdout-wider/wider-base/predictions.jsonl"):
        path = REPO / rel
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id") and row.get("prompt"):
                prompts.setdefault(row["id"], row["prompt"])
    return prompts


def docs(path):
    """Split a probe corpus into docs (prompt line + answer) by known anchors.

    ``build_ppl_probe`` joins docs with a blank line, but answers contain blank
    lines themselves, so a blank-line split over-counts. Each doc starts with a
    known prompt, so we cut at the next prompt's offset instead.
    """
    text = path.read_text()
    return text


def anchored_docs(text, prompts):
    """Cut the corpus at each prompt occurrence, in file order."""
    starts = []
    for prompt in prompts:
        index = text.find(prompt)
        if index >= 0:
            starts.append((index, prompt))
    starts.sort()
    if not starts:
        return []
    docs = []
    for number, (index, prompt) in enumerate(starts):
        end = starts[number + 1][0] if number + 1 < len(starts) else len(text)
        body = text[index:end].strip()
        answer = body[len(prompt):].strip()
        docs.append((prompt, answer))
    return docs


def build(args):
    ids = gate_ids()
    usable = []
    refused = []
    for source, prompt_id, frame in VERIFIED:
        if source != "capture":
            refused.append((prompt_id, f"source {source!r} is a held-out screen, not the fit split"))
            continue
        if prompt_id in ids:
            refused.append((prompt_id, "in the 24-family gate set (content axis)"))
            continue
        if prompt_id not in FIT_FAMILIES:
            refused.append((prompt_id, "not in the fit-split family map"))
            continue
        usable.append((prompt_id, frame))

    prompts = prompts_by_id()
    text = anchored_docs(docs(args.source_dir / "delivery.txt"),
                         [prompts[prompt_id] for prompt_id, _ in usable
                          if prompt_id in prompts])
    by_prompt = {prompt: answer for prompt, answer in text}
    refusals = {}
    for rel in REFUSAL_WAVES:
        for line in (REPO / rel).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("id") in FIT_FAMILIES:
                    refusals[row["id"]] = row

    pairs = []
    for prompt_id, frame in usable:
        prompt = prompts.get(prompt_id)
        if prompt is None:
            raise SystemExit(f"no prompt text for {prompt_id}")
        delivery = by_prompt.get(prompt)
        base = refusals.get(prompt_id)
        if not delivery:
            raise SystemExit(f"no delivery doc for {prompt_id} in {args.source_dir}/delivery.txt")
        if not base or not (base.get("answer") or "").strip():
            raise SystemExit(f"no committed plain refusal for {prompt_id}")
        shape_d = classify(delivery)
        shape_r = classify(base["answer"])
        source = {
            "kind": "capture",
            "model": "Qwen/Qwen3-4B-Instruct-2507",
            "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
            "capture": "2026-11-18 jailbreak round (scratch records lost); delivery text is the "
                       "doc committed in verification/ppl-probe/delivery.txt",
            "delivery_sha256": hashlib.sha256(delivery.encode()).hexdigest(),
            "refusal_wave": "verification/q3-4b-harmful40/q3base/predictions.jsonl",
            "refusal_sha256": hashlib.sha256((base.get("answer") or "").encode()).hexdigest(),
            "read": "scripts/build_ppl_probe.py VERIFIED (PLAN-NEXT 2026-11-18 hand read)",
        }
        pair = make_pair(
            prompt_id=prompt_id, prompt=prompt, kind="refusal",
            family_id=FIT_FAMILIES[prompt_id], category=None, source=source,
            refusal={"frame": "none", "answer": base["answer"], "shape": shape_r["shape"],
                     "declining": shape_r["declining"], "answer_source": "banked",
                     "chars": len(base["answer"]), "seed": None},
            delivery={"frame": frame, "answer": delivery, "shape": shape_d["shape"],
                      "declining": shape_d["declining"], "answer_source": "mirror",
                      "chars": len(delivery), "seed": None})
        # The 2026-11-18 agent read is the verdict: only hand-verified delivery
        # ids entered VERIFIED. Recorded, not re-judged here.
        vet(pair, "deliver",
            "hand-read as a real delivery of the asked-for content on 2026-11-18 "
            "(build_ppl_probe.VERIFIED); imported, not re-judged",
            "ppl-probe read 2026-11-18 (imported by delivery_fit build)")
        pairs.append(pair)

    out = Path(args.out)
    write_pairs(out / "pairs.jsonl", pairs)

    holdout_family = args.holdout_family
    families = {pair["family_id"] for pair in pairs}
    if holdout_family not in families:
        raise SystemExit(f"--holdout-family {holdout_family} is not in the corpus families {sorted(families)}")
    train = [pair for pair in pairs if pair["family_id"] != holdout_family]
    holdout = [pair for pair in pairs if pair["family_id"] == holdout_family]
    if not train or not holdout:
        raise SystemExit("both split sides must be nonempty")

    def suite_row(pair):
        return {"id": pair["prompt_id"], "prompt": pair["prompt"], "kind": "refusal",
                "language": "en", "family_id": pair["family_id"],
                "chosen": pair["delivery"]["answer"], "rejected": pair["refusal"]["answer"],
                # SFT reads this field; DPO reads chosen/rejected. Same text, so one
                # corpus serves both modes without a second copy.
                "completion": pair["delivery"]["answer"]}

    def write_suite(name, rows):
        path = out / name
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        return path

    train_path, holdout_path = write_suite("train.jsonl", [suite_row(p) for p in train]), \
        write_suite("holdout.jsonl", [suite_row(p) for p in holdout])

    from jdewitt.data import load_suite, validate_splits  # noqa: E402
    train_rows, holdout_rows = load_suite(train_path), load_suite(holdout_path)
    validate_splits({"train": train_rows, "holdout": holdout_rows})

    provenance = {
        "schema": SCHEMA,
        "source": "verification/ppl-probe (hand-read deliveries)",
        "gate_set": str(GATE_SET.relative_to(REPO)),
        "gate_set_sha256": sha256_file(GATE_SET),
        "families": {"train": sorted({p["family_id"] for p in train}),
                     "holdout": [holdout_family]},
        "refused": [{"id": pid, "reason": why} for pid, why in refused],
        "counts": summarize_pairs(pairs),
        "files": {name: {"sha256": sha256_file(out / name),
                         "rows": len(rows)} for name, rows in
                  (("pairs.jsonl", pairs), ("train.jsonl", train_rows),
                   ("holdout.jsonl", holdout_rows))},
        "note": ("The fit corpus is the fit split only: the 24-family content read and the "
                 "120 wider held-out families are refused, so a number taken on them is a "
                 "generalization measurement, not a memory read. Corpus size is reported "
                 "against the pair contract's fit-ready floor of 10."),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    print(json.dumps(provenance, indent=1))
    return 0


def check(path):
    path = Path(path)
    pairs = read_pairs(path / "pairs.jsonl")
    summary = summarize_pairs(pairs, 10)
    ids = gate_ids()
    leaked = [pair["prompt_id"] for pair in pairs if pair["prompt_id"] in ids]
    if leaked:
        raise SystemExit(f"gate-set leak: {leaked}")
    from jdewitt.data import load_suite, validate_splits  # noqa: E402
    validate_splits({"train": load_suite(path / "train.jsonl"),
                     "holdout": load_suite(path / "holdout.jsonl")})
    print(json.dumps({**summary, "gate_set_leak": 0}, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="write pairs.jsonl + train/holdout suites")
    b.add_argument("--source-dir", type=Path, default=REPO / "verification/ppl-probe")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--holdout-family", required=True)
    c = sub.add_parser("check", help="re-validate a built corpus (gate set, splits)")
    c.add_argument("path", type=Path)
    args = ap.parse_args(argv)
    if args.command == "build":
        return build(args)
    return check(args.path)


if __name__ == "__main__":
    sys.exit(main())
