# SPDX-License-Identifier: AGPL-3.0-or-later
"""Zero-token decision: does the pair-diff delivery axis point where the oracle does?

PLAN-NEXT 2026-11-18 states the branch in one line: alignment near 1 kills the
rank-1 delivery hypothesis (an axis that reproduces the oracle is already known to
deliver 0/20 on the wider sample); orthogonality justifies exactly one merge test.
This script spends no tokens and loads no model -- it fits the axis from a capture
file plus committed agent verdicts and compares it to the banked oracle direction.

The fit is ``spanfit.per_layer_axes(records, 'bad', 'deliver', pairs=True)``: the
delivery-minus-refusal difference is taken inside one request (``meta['prompt_id']``)
before pooling, so a frame's wording or a different mix of requests on each side
cannot become the axis.

Inputs:
  --records   spanfit capture records (torch.save({'records': [(meta, [L,W])]}))
  --verdicts  delivery-pair JSONL; only ``verdict == "deliver"`` pairs are usable
  --oracle    banked oracle axes: a .pt file or a directory holding exactly one .pt
              (keys tried: --oracle-key, then axes_attn, axes, axis)

Exit codes: 0 a decision was printed; 2 bad/missing input (a deliver record without
a verdict is named); 3 fewer than --min-vetted usable pairs (the capture cannot
support a fit -- no axis is fitted and no cos is computed).

Decision (on the layers the oracle actually edited, matched to the capture's range):
  mean |cos| >= 0.9  -> dead: the rank-1 delivery hypothesis is dead; expect the
                        same read as the oracle-shaped arms
  mean |cos| <= 0.3  -> orthogonal: one merge test is justified
  otherwise          -> inconclusive: report the number, do NOT spend a merge
|cos| is used because an axis and its negation are the same merge target; the
signed mean and min are reported beside it so the sign structure stays visible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

DEAD_MEAN_ABS_COS = 0.9
ORTHOGONAL_MEAN_ABS_COS = 0.3
MIN_VETTED_PAIRS = 10
ORACLE_KEYS = ("axes_attn", "axes", "axis")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fail(message: str, code: int = 2) -> NoReturn:
    """Exit with ``message`` and ``code``; never silently."""
    print(message, file=sys.stderr, flush=True)
    raise SystemExit(code)


def find_oracle_file(path: str | Path) -> Path:
    """A .pt file, or the one .pt inside a directory (the banked packet's home)."""
    root = Path(path)
    if not root.exists():
        _fail(f"oracle axes not found: {root}")
    if root.is_file():
        return root
    candidates = sorted(p for p in root.glob("*.pt") if p.is_file())
    if not candidates:
        _fail(f"oracle directory {root} holds no .pt axes file "
              f"(bank it there, or pass the .pt path directly)")
    if len(candidates) != 1:
        _fail(f"oracle directory {root} holds {len(candidates)} .pt files "
              f"({', '.join(p.name for p in candidates)}); pass the axes file with --oracle")
    return candidates[0]


def load_oracle_axes(path: str | Path, key: str | None = None):
    """[L,W] oracle axis tensor and the payload key it came from."""
    import torch

    file = find_oracle_file(path)
    payload = torch.load(file, map_location="cpu", weights_only=False)
    if torch.is_tensor(payload):
        axes, used = payload, None
    elif isinstance(payload, dict):
        tried = ([key] if key else []) + [k for k in ORACLE_KEYS if k != key]
        used = next((k for k in tried if k in payload), None)
        if used is None:
            _fail(f"oracle file {file} has keys {sorted(payload)}, none of "
                  f"{([key] if key else []) + list(ORACLE_KEYS)}")
        axes = payload[used]
    else:
        _fail(f"oracle file {file} holds {type(payload).__name__}, not an axes dict")
    if not (hasattr(axes, "dim") and axes.dim() == 2):
        _fail(f"oracle axes {file}:{used} is not a [L,W] tensor "
              f"(shape {getattr(axes, 'shape', None)})")
    return axes.detach().cpu().double(), used, file


def _label(meta) -> str | None:
    return meta.get("label") if isinstance(meta, dict) else meta


def _prompt_id(meta):
    if isinstance(meta, dict):
        return meta.get("prompt_id") if meta.get("prompt_id") is not None else meta.get("id")
    return None


def _frame(meta) -> str:
    return (meta.get("frame") or "none") if isinstance(meta, dict) else "none"


def _record_key(meta) -> tuple:
    return (_prompt_id(meta), _frame(meta))


def record_report(records) -> dict:
    labels: dict[str, int] = {}
    for meta, _ in records:
        label = _label(meta) or "unlabelled"
        labels[label] = labels.get(label, 0) + 1
    shape = getattr(records[0][1], "shape", None) if records else None
    return {"rows": len(records), "labels": labels,
            "layers": int(shape[0]) if shape else None,
            "width": int(shape[1]) if shape else None}


def pair_report(pairs) -> dict:
    by_verdict = {"deliver": 0, "pivot": 0, "declined": 0, "other": 0}
    unread = 0
    for pair in pairs:
        verdict = pair.get("verdict")
        if verdict in by_verdict:
            by_verdict[verdict] += 1
        elif verdict is None:
            unread += 1
    return {"pairs": len(pairs), "by_verdict": by_verdict, "unread": unread,
            "vetted": sum(by_verdict.values()), "n_vetted_pairs": by_verdict["deliver"]}


def unread_deliver_records(records, pairs) -> list[str]:
    """Capture rows labelled ``deliver`` that no committed verdict covers, by name."""
    pair_index = {}
    for pair in pairs:
        side = pair.get("delivery") or {}
        pair_index.setdefault((pair.get("prompt_id"), side.get("frame")), pair)
    missing = []
    for meta, _ in records:
        if _label(meta) != "deliver":
            continue
        pair = pair_index.get(_record_key(meta))
        if pair is None or pair.get("verdict") is None:
            missing.append(f"{_prompt_id(meta)}[frame={_frame(meta)}]")
    return missing


def fit_records_for_vetted_pairs(records, pairs) -> list[tuple]:
    """Rows of the vetted pairs, relabelled from the agent read (verdict wins).

    A capture label is the shape screen's guess; the verdict is the read. A pair
    that was vetted ``deliver`` is fitted as ``deliver`` even when the shape screen
    had said ``ambiguous``, and its refusal side is fitted as ``bad``.
    """
    index: dict[tuple, list] = {}
    for meta, rec in records:
        index.setdefault(_record_key(meta), []).append((meta, rec))
    chosen: dict[tuple, tuple] = {}
    problems = []
    for pair in pairs:
        if pair.get("verdict") != "deliver":
            continue
        for label, side_name in (("bad", "refusal"), ("deliver", "delivery")):
            side = pair.get(side_name) or {}
            key = (pair.get("prompt_id"), side.get("frame") or "none")
            rows = index.get(key, [])
            if len(rows) != 1:
                problems.append(f"{pair.get('pair_id')}: {side_name} record "
                                f"{key[0]}[frame={key[1]}] has {len(rows)} rows, need 1")
                continue
            meta, rec = rows[0]
            chosen[(pair.get("pair_id"), label)] = (dict(meta, label=label, pair_id=pair.get("pair_id")), rec)
    if problems:
        _fail("vetted pair without its capture rows:\n  " + "\n  ".join(problems))
    return [chosen[key] for key in chosen]


def cos_vector(pair_axes, oracle_axes) -> tuple[list[dict], list[int]]:
    """Per-layer cos on the oracle's edit footprint, intersected with the capture range."""
    import torch

    oracle_norm = oracle_axes.norm(dim=1)
    footprint = [layer for layer in range(oracle_axes.shape[0]) if float(oracle_norm[layer]) > 0]
    layers = [layer for layer in footprint if layer < pair_axes.shape[0]]
    if not layers:
        _fail(f"oracle edits layers {footprint} but the fitted axis has only "
              f"{pair_axes.shape[0]} rows; nothing to compare")
    rows = []
    for layer in layers:
        mine = pair_axes[layer].double()
        theirs = oracle_axes[layer]
        denom = float(mine.norm() * theirs.norm())
        cos = float(torch.dot(mine, theirs) / denom) if denom > 0 else 0.0
        rows.append({"layer": layer, "cos": round(cos, 6),
                     "oracle_norm": round(float(theirs.norm()), 6),
                     "pair_norm": round(float(mine.norm()), 6)})
    return rows, layers


def decide(cosines: list[dict]) -> dict:
    """One of dead / orthogonal / inconclusive, with the numbers that chose it."""
    values = [row["cos"] for row in cosines]
    mean_signed = sum(values) / len(values)
    mean_abs = sum(abs(value) for value in values) / len(values)
    min_signed = min(values)
    min_abs = min(abs(value) for value in values)
    if mean_abs >= DEAD_MEAN_ABS_COS:
        decision = "dead"
        line = (f"decision: DEAD - mean |cos| {mean_abs:.4f} >= {DEAD_MEAN_ABS_COS}: "
                f"rank-1 delivery hypothesis dead (expect the same read)")
    elif mean_abs <= ORTHOGONAL_MEAN_ABS_COS:
        decision = "orthogonal"
        line = (f"decision: ORTHOGONAL - mean |cos| {mean_abs:.4f} <= {ORTHOGONAL_MEAN_ABS_COS}: "
                f"orthogonal, one merge test justified")
    else:
        decision = "inconclusive"
        line = (f"decision: INCONCLUSIVE - mean |cos| {mean_abs:.4f} lies between "
                f"{ORTHOGONAL_MEAN_ABS_COS} and {DEAD_MEAN_ABS_COS}: report the number, "
                f"do NOT spend a merge")
    return {"decision": decision, "decision_line": line,
            "mean_cos": round(mean_signed, 6), "mean_abs_cos": round(mean_abs, 6),
            "min_cos": round(min_signed, 6), "min_abs_cos": round(min_abs, 6),
            "layers": [row["layer"] for row in cosines]}


def print_table(cosines: list[dict]) -> None:
    print(f"{'layer':>5}  {'oracle_norm':>11}  {'pair_norm':>9}  {'cos':>9}")
    for row in cosines:
        print(f"{row['layer']:>5}  {row['oracle_norm']:>11.6f}  {row['pair_norm']:>9.6f}  "
              f"{row['cos']:>+9.6f}")


def write_packet(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise
    return path


def run(args) -> int:
    from jdewitt.delivery_pairs import read_pairs
    from jdewitt.spanfit import load_records, per_layer_axes

    started = time.monotonic()
    records_path, verdicts_path = Path(args.records), Path(args.verdicts)
    if not records_path.is_file():
        _fail(f"records not found: {records_path}")
    if not verdicts_path.is_file():
        _fail(f"verdicts not found: {verdicts_path}")
    try:
        records = load_records(str(records_path))
        pairs = read_pairs(verdicts_path)
    except (ValueError, OSError, RuntimeError) as error:
        _fail(f"cannot read inputs: {error}")
    report, verdicts = record_report(records), pair_report(pairs)

    header = (f"pair-axis check  records={records_path} sha256={sha256(records_path)[:16]}  "
              f"rows={report['rows']} layers={report['layers']} width={report['width']}")
    print(header, flush=True)
    print(f"verdicts: pairs={verdicts['pairs']} vetted_deliver={verdicts['n_vetted_pairs']} "
          f"pivot={verdicts['by_verdict']['pivot']} declined={verdicts['by_verdict']['declined']} "
          f"other={verdicts['by_verdict']['other']} unread={verdicts['unread']}", flush=True)

    missing = unread_deliver_records(records, pairs)
    if missing:
        _fail("deliver record without a verdict: " + "; ".join(missing))

    base_packet = {
        "schema": "jdewitt.pair-axis-check/1",
        "scope": ("Zero-token decision: cos(pair-diff delivery axis, banked oracle axis). "
                  "No model loaded, no merge spent, no completion generated."),
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "records": dict(report, path=str(records_path), sha256=sha256(records_path)),
        "verdicts": dict(verdicts, path=str(verdicts_path), sha256=sha256(verdicts_path)),
        "thresholds": {"dead_mean_abs_cos": DEAD_MEAN_ABS_COS,
                       "orthogonal_mean_abs_cos": ORTHOGONAL_MEAN_ABS_COS,
                       "min_vetted_pairs": args.min_vetted},
        "oracle_arg": str(args.oracle),
        "oracle_key_arg": args.oracle_key,
    }
    if verdicts["n_vetted_pairs"] < args.min_vetted:
        line = (f"insufficient: {verdicts['n_vetted_pairs']} vetted delivery pairs "
                f"(< {args.min_vetted}) - no axis fitted, no cos computed, no merge spent")
        packet = dict(base_packet, decision="insufficient", decision_line=line, cos=[],
                      mean_cos=None, mean_abs_cos=None, min_cos=None, min_abs_cos=None,
                      elapsed_s=round(time.monotonic() - started, 3))
        out = write_packet(args.out, packet)
        print(line, flush=True)
        print(f"wrote {out}", flush=True)
        return 3

    fit_rows = fit_records_for_vetted_pairs(records, pairs)
    oracle_axes, oracle_key, oracle_file = load_oracle_axes(args.oracle, args.oracle_key)
    if oracle_axes.shape[1] != report["width"]:
        _fail(f"oracle width {oracle_axes.shape[1]} != capture width {report['width']}; "
              f"the axes are not in this capture's space")
    pair_axes = per_layer_axes(fit_rows, "bad", "deliver", pairs=True).double()
    cosines, layers = cos_vector(pair_axes, oracle_axes)
    verdict = decide(cosines)
    print_table(cosines)
    print(f"mean cos {verdict['mean_cos']:+.6f}  mean |cos| {verdict['mean_abs_cos']:.6f}  "
          f"min cos {verdict['min_cos']:+.6f}  min |cos| {verdict['min_abs_cos']:.6f}  "
          f"layers {len(layers)}", flush=True)
    print(verdict["decision_line"], flush=True)

    packet = dict(base_packet, oracle=dict(path=str(oracle_file), arg=str(args.oracle),
                                           key=oracle_key, sha256=sha256(oracle_file),
                                           footprint_layers=layers),
                  fit={"method": "spanfit.per_layer_axes(..., pairs=True)", "pairs_used": len(fit_rows) // 2,
                       "rows_used": len(fit_rows), "layers": int(pair_axes.shape[0]),
                       "width": int(pair_axes.shape[1])},
                  cos=cosines, decision=verdict["decision"], decision_line=verdict["decision_line"],
                  mean_cos=verdict["mean_cos"], mean_abs_cos=verdict["mean_abs_cos"],
                  min_cos=verdict["min_cos"], min_abs_cos=verdict["min_abs_cos"],
                  elapsed_s=round(time.monotonic() - started, 3))
    out = write_packet(args.out, packet)
    print(f"wrote {out}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a pair-diff delivery axis with the banked oracle axis (zero tokens).")
    parser.add_argument("--records", required=True, help="capture records .pt (spanfit format)")
    parser.add_argument("--verdicts", required=True, help="delivery-pair JSONL with agent verdicts")
    parser.add_argument("--oracle", required=True, help="oracle axes .pt, or the directory holding it")
    parser.add_argument("--oracle-key", default=None,
                        help=f"payload key (default: first of {', '.join(ORACLE_KEYS)})")
    parser.add_argument("--out", default="verification/p5/pair-axis-check.json")
    parser.add_argument("--min-vetted", type=int, default=MIN_VETTED_PAIRS,
                        help=f"usable pairs a fit needs (default {MIN_VETTED_PAIRS})")
    return parser


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
