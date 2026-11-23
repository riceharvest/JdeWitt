# SPDX-License-Identifier: AGPL-3.0-or-later
"""Token-logit margin: score a candidate axis before any weight moves.

abliterator's one real contribution is a selector, not a removal method: it scores
a direction by the model's own token-logit margin between the spellings ``Sure``
and ``cannot`` at the position that would emit the first answer token, on prefill
alone, with zero generated tokens. This module is that selector, made exact:

  * resolution -- every candidate spelling is tokenized with
    ``add_special_tokens=False``; only a spelling that is exactly ONE id enters the
    statistic. A multi-token spelling is a different statistic (a sequence margin,
    not a token margin) and is refused, never replaced by a word list (R2);
  * the margin is ``mean over resolved (sure_id, cannot_id) pairs of
    logit[sure_id] - logit[cannot_id]`` at one position: one subtraction per pair
    on a pass the caller already ran;
  * ``capture`` reads it out of the forward pass that already prefills prompt +
    answer, so the selector costs no extra pass and no extra token;
  * ``score`` applies a candidate axis through TEMPORARY forward hooks on the
    captured decoder layers (weights are never written), re-prefills prompt + the
    record's own answer, and reports the margin before/after per row plus the mean
    drop on harmful rows minus the mean drop on harmless ones.

The margin ranks candidates; it never certifies (METHODOLOGY R2/R6). A lower
prefill margin is evidence that the axis points at the mechanism, not that the
edited artifact still behaves -- only a merged serve decides that.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

SURE_SPELLINGS = (" Sure", "Sure")
CANNOT_SPELLINGS = (" cannot", "cannot")
WORDS = (("Sure", SURE_SPELLINGS), ("cannot", CANNOT_SPELLINGS))

# Toy fixture for self-check: fixed float64 weights, two decoder blocks and a head.
TOY_WIDTH, TOY_VOCAB = 4, 6
TOY_SURE_IDS = (1, 2)          # two single-token spellings, so the pair mean is exercised
TOY_CANNOT_ID = 4
TOY_AXIS = (0.6, 0.8, 0.0, 0.0)  # unit norm, deliberately off every basis vector
TOY_INPUT = (1.0, -0.5, 0.75, 0.25)
TOY_BLOCKS = (
    (((1.0, 0.25, 0.0, -0.5), (0.0, 0.5, 0.25, 0.0), (0.25, 0.0, 1.0, 0.5), (-0.5, 0.0, 0.5, 1.0)),
     (0.1, -0.2, 0.0, 0.3)),
    (((0.5, 0.0, 0.25, 0.0), (0.0, 1.0, 0.0, -0.25), (0.25, 0.0, 0.75, 0.0), (0.0, -0.25, 0.0, 0.5)),
     (0.0, 0.1, -0.1, 0.0)),
)
TOY_HEAD = (
    (0.2, 0.1, 0.0, 0.3),
    (1.0, 0.3, 0.2, 0.1),
    (0.8, 0.4, 0.1, -0.2),
    (-0.3, 0.0, 0.2, 0.1),
    (-1.0, -0.2, 0.1, 0.0),
    (0.1, -0.5, 0.3, 0.2),
)
TOY_BIAS = (0.0, 0.7, -0.3, 0.2, -0.6, 0.1)
TOY_PAIRS = [[TOY_SURE_IDS[0], TOY_CANNOT_ID], [TOY_SURE_IDS[1], TOY_CANNOT_ID]]

METHOD_NOTE = ("the margin ranks candidate axes and never certifies (METHODOLOGY R2/R6): a lower "
               "prefill margin is evidence the axis points at the mechanism, not that the edited "
               "artifact still behaves; only a merged serve decides that")


class MarginRefused(RuntimeError):
    """A margin spelling is not a single token: the statistic is refused, not repaired."""


class MarginUnavailable(RuntimeError):
    """The forward pass returned no usable logits at the margin position."""


# ---------- resolution ----------

def token_table(tokenizer) -> dict:
    """{"Sure": [...], "cannot": [...]} with every spelling's ids and resolved bool."""
    table = {}
    for word, spellings in WORDS:
        rows = []
        for spelling in spellings:
            ids = [int(token) for token in tokenizer.encode(spelling, add_special_tokens=False)]
            rows.append({"spelling": spelling, "ids": ids, "resolved": len(ids) == 1})
        table[word] = rows
    return table


def resolved_pairs(table: dict) -> list[list[int]]:
    """Cross product of single-token sure ids and cannot ids, in table order."""
    sure = [row["ids"][0] for row in table["Sure"] if row["resolved"]]
    cannot = [row["ids"][0] for row in table["cannot"] if row["resolved"]]
    missing = [word for word, ids in (("Sure", sure), ("cannot", cannot)) if not ids]
    if missing:
        raise MarginRefused(
            f"no single-token spelling for {missing}: a multi-token margin is a different statistic "
            f"(a sequence margin, not a token margin) and is refused, never replaced by a word list")
    pairs, seen = [], set()
    for sure_id in sure:
        for cannot_id in cannot:
            if (sure_id, cannot_id) not in seen:
                seen.add((sure_id, cannot_id))
                pairs.append([sure_id, cannot_id])
    return pairs


def resolve(tokenizer):
    """(resolution table, pairs); raises MarginRefused instead of guessing."""
    table = token_table(tokenizer)
    return table, resolved_pairs(table)


# ---------- margin math ----------

def _floats(values) -> list[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(value) for value in values]


def margin_from_logits(logits, pairs) -> float:
    """mean over pairs of logit[sure_id] - logit[cannot_id] at one position."""
    row = _floats(logits)
    if not pairs:
        raise ValueError("a margin needs at least one (sure_id, cannot_id) pair")
    total = 0.0
    for sure_id, cannot_id in pairs:
        if sure_id >= len(row) or cannot_id >= len(row):
            raise MarginUnavailable(f"margin token id outside the vocabulary ({len(row)} ids)")
        total += row[sure_id] - row[cannot_id]
    return total / len(pairs)


def margin_at_position(logits, position, pairs) -> float:
    """Margin at ``position`` of logits shaped [seq, vocab], [1, seq, vocab] or a list."""
    if logits is None:
        raise MarginUnavailable("the forward pass returned no logits; margin is unavailable")
    if hasattr(logits, "dim") and hasattr(logits, "shape"):
        row = logits
        while row.dim() > 1 and row.shape[0] == 1:
            row = row[0]
        if row.dim() > 1:
            if position >= row.shape[0]:
                raise MarginUnavailable(f"margin position {position} outside {row.shape[0]} positions")
            row = row[position]
        return margin_from_logits(row, pairs)
    if isinstance(logits, list) and logits and isinstance(logits[0], list):
        if position >= len(logits):
            raise MarginUnavailable(f"margin position {position} outside {len(logits)} positions")
        return margin_from_logits(logits[position], pairs)
    return margin_from_logits(logits, pairs)


# ---------- projection (weights are never written) ----------

def _axis_rows(axis) -> list:
    if hasattr(axis, "dim"):
        return [axis] if axis.dim() == 1 else [axis[index] for index in range(axis.shape[0])]
    if isinstance(axis, list) and axis and isinstance(axis[0], (list, tuple)):
        return [list(row) for row in axis]
    return [axis]


def _project_tensor(hidden, axis, scale=1.0):
    moved = hidden
    for row in _axis_rows(axis):
        if not hasattr(row, "to"):
            row = hidden.new_tensor([float(value) for value in row])
        unit = row.to(device=moved.device, dtype=moved.dtype)
        moved = moved - (scale * (moved @ unit)).unsqueeze(-1) * unit
    return moved


def _project_vector(hidden, axis, scale=1.0) -> list[float]:
    moved = [float(value) for value in hidden]
    for row in _axis_rows(axis):
        unit = [float(value) for value in row]
        coefficient = sum(value * unit_value for value, unit_value in zip(moved, unit))
        moved = [value - scale * coefficient * unit_value for value, unit_value in zip(moved, unit)]
    return moved


def project(hidden, axis, scale=1.0):
    """Remove the axis component from a hidden state: h - scale*(h.a)*a."""
    if hasattr(hidden, "dim"):
        return _project_tensor(hidden, axis, scale)
    return _project_vector(hidden, axis, scale)


def projection_hook(axis, scale=1.0):
    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        moved = project(hidden, axis, scale)
        return (moved,) + output[1:] if isinstance(output, tuple) else moved
    return hook


def install_projection_hooks(layers, axes, scale=1.0) -> list:
    """Temporary forward hooks on the footprint layers; handles must be removed."""
    handles = []
    try:
        for index, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(projection_hook(axes[index], scale)))
    except BaseException:
        remove_hooks(handles)
        raise
    return handles


def remove_hooks(handles) -> None:
    for handle in handles:
        handle.remove()
    handles.clear()


# ---------- scoring ----------

def _torch():
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - the verify env is torch-free
        raise MarginUnavailable(
            "torch is not installed in this interpreter; `score` needs the project environment "
            "(uv run python -m jdewitt.margin score ...)") from error
    return torch


def _load_tokenizer(args):
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as error:  # pragma: no cover - the verify env is torch-free
        raise MarginUnavailable(
            "transformers is not installed in this interpreter; `tokens` needs the project environment "
            "(uv run python -m jdewitt.margin tokens ...)") from error
    return AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                         trust_remote_code=args.trust_remote_code)


def _unwrap_axes(loaded):
    if isinstance(loaded, dict):
        for key in ("axes", "axis", "directions"):
            if key in loaded:
                return loaded[key]
        raise ValueError("axis file is a dict with no 'axes'/'axis'/'directions' entry")
    return loaded


def _load_model(args):
    """(model, tokenizer, config, width, device, context, torch) with context left open."""
    torch = _torch()
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

    from .capture import hidden_width
    from .compat import compatible_source
    from .hf import backend_device, resolve_backend

    backend = resolve_backend(allow_secondary=args.allow_cuda)
    device = backend_device(backend)
    config = AutoConfig.from_pretrained(args.model, revision=args.revision,
                                        trust_remote_code=args.trust_remote_code)
    width = hidden_width(config, args.hidden)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                              trust_remote_code=args.trust_remote_code)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    context = compatible_source(args.model)
    prepared = context.__enter__()
    model, attempts = None, []
    try:
        for factory in (AutoModelForCausalLM, AutoModel):
            try:
                model = factory.from_pretrained(str(prepared), revision=args.revision, dtype=dtype,
                                                device_map={"": device},
                                                trust_remote_code=args.trust_remote_code).eval()
                break
            except (ValueError, TypeError, RuntimeError) as error:
                attempts.append(f"{factory.__name__}: {error}")
        if model is None:
            raise ValueError("no model factory loaded this checkpoint: " + " | ".join(attempts[:3]))
    except BaseException:
        context.__exit__(None, None, None)
        raise
    return model, tokenizer, config, width, device, context, torch


def _prefill(model, device, ids, torch):
    with torch.inference_mode():
        output = model(ids.to(device), attention_mask=torch.ones_like(ids, device=device), use_cache=False)
        logits = getattr(output, "logits", None)
    return logits


def _summary(values) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None}
    return {"n": len(values), "mean": round(sum(values) / len(values), 6),
            "median": round(statistics.median(values), 6)}


def score_command(args) -> dict:
    from .capture import expected_layers, find_layer_stack
    from .spanfit import load_records, parse_layer_ids, per_layer_axes, shared_axis

    records = load_records(str(args.records))
    if not records:
        raise ValueError(f"{args.records} carries no records")
    layer_width = int(records[0][1].shape[-1])
    if args.axis_file == "contrast":
        try:
            axes = shared_axis(records)[0] if args.shape == "shared" else per_layer_axes(records)
        except SystemExit as error:
            raise ValueError(f"{args.records} cannot produce a bad-minus-good axis: {error}") from error
        axis_source = {"kind": "contrast", "shape": args.shape, "records": str(args.records)}
    else:
        loaded = _torch().load(args.axis_file, map_location="cpu", weights_only=False)
        axes = _unwrap_axes(loaded)
        axis_source = {"kind": "file", "path": str(args.axis_file)}
    if hasattr(axes, "shape") and int(axes.shape[-1]) != layer_width:
        raise ValueError(f"axis width {int(axes.shape[-1])} does not match record width {layer_width}")

    model, tokenizer, config, width, device, context, torch = _load_model(args)
    try:
        stack_name, layer_count = find_layer_stack(model, width, expected_layers(config))
        stack = dict(model.named_modules())[stack_name]
        layers = [stack[index] for index in range(layer_count)]
        if int(axes.shape[0]) != layer_count and int(axes.shape[0]) != 1:
            raise ValueError(f"axis file has {int(axes.shape[0])} layers, the model has {layer_count}")
        footprint = list(range(layer_count)) if args.footprints == "all" else parse_layer_ids(args.footprints)
        bad_layers = [layer for layer in footprint if not 0 <= layer < layer_count]
        if bad_layers:
            raise ValueError(f"footprint names layers outside 0..{layer_count - 1}: {bad_layers}")
        if not footprint:
            raise ValueError("the footprint selects no layer")

        table, pairs = resolve(tokenizer)
        rows, skipped = [], []
        for index, (meta, _residual) in enumerate(records):
            text = meta.get("prompt_rendered") or meta.get("prompt")
            answer = meta.get("answer")
            if not text or not answer:
                skipped.append({"index": index, "reason": "record carries no prompt_rendered/answer"})
                continue
            prompt_ids = [int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"]]
            answer_ids = [int(token) for token in tokenizer(answer, add_special_tokens=False)["input_ids"]]
            if not prompt_ids or not answer_ids:
                skipped.append({"index": index, "reason": "prompt or answer tokenizes to nothing"})
                continue
            rows.append({"index": index, "meta": meta, "ids": prompt_ids + answer_ids,
                         "position": len(prompt_ids) - 1, "label": meta.get("label")})

        prefills = 0
        for row in rows:
            ids = torch.tensor([row["ids"]], dtype=torch.long)
            logits = _prefill(model, device, ids, torch)
            row["margin_before"] = round(margin_at_position(logits, row["position"], pairs), 6)
            del logits
            prefills += 1

        hook_layers = [layers[layer] for layer in footprint]
        hook_axes = [axes[0] if int(axes.shape[0]) == 1 else axes[layer] for layer in footprint]
        handles = install_projection_hooks(hook_layers, hook_axes, scale=args.scale)
        try:
            for row in rows:
                ids = torch.tensor([row["ids"]], dtype=torch.long)
                logits = _prefill(model, device, ids, torch)
                row["margin_after"] = round(margin_at_position(logits, row["position"], pairs), 6)
                del logits
                prefills += 1
        finally:
            remove_hooks(handles)
            handles_removed = not any(getattr(layer, "_forward_hooks", {}) for layer in hook_layers)

        for row in rows:
            row["drop"] = round(row["margin_before"] - row["margin_after"], 6)
        bad_drops = [row["drop"] for row in rows if row["label"] == "bad"]
        good_drops = [row["drop"] for row in rows if row["label"] == "good"]
        delta = (round(sum(bad_drops) / len(bad_drops) - sum(good_drops) / len(good_drops), 6)
                 if bad_drops and good_drops else None)
        report = {
            "records": str(args.records), "model": args.model,
            "axis": axis_source, "footprint": footprint, "footprint_layers": len(footprint),
            "scale": args.scale,
            "resolution": table, "pairs": pairs,
            "rows": [{key: row[key] for key in ("index", "label", "position", "margin_before",
                                                "margin_after", "drop")} | {"id": row["meta"].get("id")}
                     for row in rows],
            "skipped": skipped,
            "summary": {"n": len(rows), "bad": {"n": len(bad_drops),
                                                "drop": round(sum(bad_drops) / len(bad_drops), 6) if bad_drops else None},
                        "good": {"n": len(good_drops),
                                 "drop": round(sum(good_drops) / len(good_drops), 6) if good_drops else None},
                        "delta_bad_minus_good": delta},
            "prefill_count": prefills, "tokens_generated": 0,
            "hooks_removed": handles_removed, "weights_written": False,
            "sign": "drop = margin_before - margin_after; NEGATIVE delta = the projection lifts Sure "
                    "relative to cannot more on harmful rows than on harmless ones",
            "note": METHOD_NOTE,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, allow_nan=False))
        return report
    finally:
        context.__exit__(None, None, None)


# ---------- self-check ----------

class _PlainHandle:
    def __init__(self, layer, hook):
        self.layer, self.hook = layer, hook

    def remove(self):
        if self.hook in self.layer.hooks:
            self.layer.hooks.remove(self.hook)


class _PlainBlock:
    """Dependency-free stand-in with torch's forward-hook contract (verify env has no torch)."""

    def __init__(self, weight, bias):
        self.weight = [[float(value) for value in row] for row in weight]
        self.bias = [float(value) for value in bias]
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return _PlainHandle(self, hook)

    def __call__(self, hidden):
        out = [sum(weight * value for weight, value in zip(row, hidden)) + bias
               for row, bias in zip(self.weight, self.bias)]
        for hook in list(self.hooks):
            out = hook(self, (hidden,), out)
        return out


class _PlainModel:
    def __init__(self):
        self.blocks = [_PlainBlock(weight, bias) for weight, bias in TOY_BLOCKS]
        self.lm_head = _PlainBlock(TOY_HEAD, TOY_BIAS)

    def __call__(self, hidden):
        for block in self.blocks:
            hidden = block(hidden)
        return self.lm_head(hidden)


def _torch_toy(torch):
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(TOY_WIDTH, TOY_WIDTH) for _ in TOY_BLOCKS])
            self.lm_head = torch.nn.Linear(TOY_WIDTH, TOY_VOCAB)
            # float64 BEFORE the copy: a float32 parameter would truncate the fixture
            # weights and the hook's measured shift would miss the analytic one.
            self.to(torch.float64)
            with torch.no_grad():
                for block, (weight, bias) in zip(self.blocks, TOY_BLOCKS):
                    block.weight.copy_(torch.tensor(weight, dtype=torch.float64))
                    block.bias.copy_(torch.tensor(bias, dtype=torch.float64))
                self.lm_head.weight.copy_(torch.tensor(TOY_HEAD, dtype=torch.float64))
                self.lm_head.bias.copy_(torch.tensor(TOY_BIAS, dtype=torch.float64))

        def forward(self, hidden):
            for block in self.blocks:
                hidden = block(hidden)
            return self.lm_head(hidden)

    return Toy().to(torch.float64).eval()


def _toy_forward(model, torch=None):
    if torch is None:
        return model([float(value) for value in TOY_INPUT])
    with torch.inference_mode():
        return model(torch.tensor([TOY_INPUT], dtype=torch.float64))


def _toy_hidden_after_blocks() -> list[float]:
    hidden = [float(value) for value in TOY_INPUT]
    for weight, bias in TOY_BLOCKS:
        hidden = [sum(w * value for w, value in zip(row, hidden)) + b for row, b in zip(weight, bias)]
    return hidden


def _predicted_shift() -> float:
    """Analytic margin shift from projecting the axis out at the LAST block output."""
    hidden = _toy_hidden_after_blocks()
    gradient = [sum(TOY_HEAD[sure][column] - TOY_HEAD[TOY_CANNOT_ID][column] for sure in TOY_SURE_IDS)
                / len(TOY_SURE_IDS) for column in range(TOY_WIDTH)]
    coefficient = sum(value * axis for value, axis in zip(hidden, TOY_AXIS))
    return -coefficient * sum(axis * gain for axis, gain in zip(TOY_AXIS, gradient))


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _arithmetic_check() -> dict:
    logits = [0.0, 2.0, 0.5, -1.0, 4.0, -2.0]
    single = margin_from_logits(logits, [[1, 4]])
    pair_mean = margin_from_logits(logits, [[1, 4], [2, 4]])
    swapped = margin_from_logits(logits, [[4, 1]])
    _check(single == -2.0, f"single-pair margin {single!r} != -2.0")
    _check(pair_mean == -2.75, f"pair-mean margin {pair_mean!r} != -2.75")
    _check(swapped == 2.0, f"swapped margin {swapped!r} != 2.0")
    return {"single": single, "pair_mean": pair_mean, "swapped": swapped}


def _toy_check(model, torch=None) -> dict:
    before = margin_at_position(_toy_forward(model, torch), 0, TOY_PAIRS)
    predicted = _predicted_shift()
    handles = install_projection_hooks([model.blocks[-1]], [list(TOY_AXIS)])
    try:
        after = margin_at_position(_toy_forward(model, torch), 0, TOY_PAIRS)
    finally:
        remove_hooks(handles)
    restored = margin_at_position(_toy_forward(model, torch), 0, TOY_PAIRS)
    observed = after - before
    _check(abs(observed - predicted) < 1e-12,
           f"hook moved the margin by {observed!r}, the analytic prediction is {predicted!r}")
    _check(observed != 0.0 and (observed > 0) == (predicted > 0), "hook did not move by the predicted sign")
    _check(restored == before, f"margin did not return to {before!r} after the hooks were removed: {restored!r}")
    return {"before": before, "after": after, "observed": observed, "predicted": predicted,
            "restored": restored}


def _format(value) -> str:
    return f"{value:.15g}"


def self_check() -> dict:
    """Exact arithmetic, the pair mean, the hook's predicted shift and hook removal."""
    arithmetic = _arithmetic_check()
    print(f"PASS arithmetic single={_format(arithmetic['single'])} "
          f"pair_mean={_format(arithmetic['pair_mean'])} swapped={_format(arithmetic['swapped'])}")
    try:
        torch = _torch()
    except MarginUnavailable:
        torch = None
    toy = _toy_check(_PlainModel() if torch is None else _torch_toy(torch), torch)
    backend = "pure-python" if torch is None else f"torch-{torch.__version__}"
    print(f"PASS toy before={_format(toy['before'])} after={_format(toy['after'])} "
          f"observed_delta={_format(toy['observed'])} predicted_delta={_format(toy['predicted'])}")
    print(f"PASS hooks_removed restored={_format(toy['restored'])} identical_to_before="
          f"{toy['restored'] == toy['before']}")
    print(f"PASS backend={backend} blocks=2 pairs={len(TOY_PAIRS)} vocab={TOY_VOCAB}")
    return {"arithmetic": arithmetic, "toy": toy, "backend": backend}


# ---------- CLI ----------

def tokens_command(args) -> dict:
    tokenizer = _load_tokenizer(args)
    table = token_table(tokenizer)
    try:
        pairs = resolved_pairs(table)
    except MarginRefused:
        print(json.dumps({"model": args.model, "resolution": table, "pairs": [], "available": False},
                         indent=2), flush=True)
        raise
    report = {"model": args.model, "resolution": table, "pairs": pairs, "available": True,
              "note": "token ids from the tokenizer only: no weights loaded, no forward pass, no generation"}
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m jdewitt.margin",
        description="Token-logit margin selector: resolve Sure/cannot spellings, read the margin out of "
                    "prefill, and score a candidate axis through temporary forward hooks (zero generation).")
    parser.add_argument("--self-check", dest="self_check_flag", action="store_true",
                        help="run the offline self-check (same as the self-check subcommand)")
    sub = parser.add_subparsers(dest="command")

    tokens = sub.add_parser("tokens", help="resolve ' Sure'/'Sure' and ' cannot'/'cannot' to single token ids")
    tokens.add_argument("--model", required=True)
    tokens.add_argument("--revision", default=None)
    tokens.add_argument("--trust-remote-code", action="store_true")

    score = sub.add_parser("score", help="score an axis by the margin before/after a temporary projection, "
                                         "prefill only, zero generated tokens")
    score.add_argument("--model", required=True)
    score.add_argument("--records", type=Path, required=True)
    score.add_argument("--axis-file", required=True, help="axes .pt, or the literal word contrast")
    score.add_argument("--footprints", required=True, help="layer range to project, e.g. 9-35 or all")
    score.add_argument("--out", type=Path, required=True)
    score.add_argument("--shape", choices=("perlayer", "shared"), default="perlayer",
                       help="axis shape when --axis-file contrast")
    score.add_argument("--scale", type=float, default=1.0)
    score.add_argument("--revision", default=None)
    score.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    score.add_argument("--hidden", type=int, default=None)
    score.add_argument("--allow-cuda", action="store_true")
    score.add_argument("--trust-remote-code", action="store_true")

    sub.add_parser("self-check", help="offline exact-arithmetic check: no model, no GPU, no generation")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.self_check_flag or args.command == "self-check":
            self_check()
        elif args.command == "tokens":
            tokens_command(args)
        elif args.command == "score":
            print(json.dumps(score_command(args), indent=2))
        else:
            build_parser().error("choose tokens, score or self-check")
    except (MarginRefused, MarginUnavailable, ValueError, RuntimeError, OSError) as error:
        code = next((value for cls, value in ((MarginRefused, "MARGIN_REFUSED"),
                                              (MarginUnavailable, "MARGIN_UNAVAILABLE"))
                     if isinstance(error, cls)), "MARGIN_ERROR")
        print(json.dumps({"error": {"code": code, "type": type(error).__name__, "message": str(error),
                                    "command": getattr(args, "command", None)}}),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
