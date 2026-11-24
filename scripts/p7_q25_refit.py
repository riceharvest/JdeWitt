# SPDX-License-Identifier: AGPL-3.0-or-later
"""P7 qwen2.5-1.5b prep: capture fit records on CPU so the SAS axis can be refit.

Why this exists: the q25 SAS@1.5 arm was a llama.cpp LoRA adapter
(`q25-v4-dir000.gguf`) served at scale 1.5 on the unmodified BF16 base, and that
file is not on this box. A Phase 7 row needs the merged bytes (R8), and the spec
for this lineage says a missing arm must be refit from a capture of the base and
labelled a refit, never "SAS@1.5".

`jdewitt.capture` is the one capture path (label by answer SHAPE, hooks on the
discovered decoder stack, spanfit record format), but its CLI deliberately
refuses CPU ("CPU fallback is forbidden") because a capture is normally a
GPU-side step. This card is the CPU prep card: it holds no GPU while it codes or
fits. So this script drives capture's own helpers on an explicit CPU device and
writes the same spanfit records format; everything after the capture (axis fit,
merge) runs through the shipped commands:

    python -m jdewitt.spanfit axes --records RECORDS.pt --shape shared --out AXES.pt
    python -m jdewitt.spanfit build --gguf-in BASE.gguf --out ARM.gguf --axes AXES.pt ...

The run is resumable: every finished row is appended to ``OUT.partial`` and a
rerun skips the ids already there, because a CPU capture of 80 rows is tens of
minutes and this box's runs have been killed mid-turn before.

Nothing here certifies and nothing here is called SAS@1.5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from jdewitt import capture as cap  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_partial(path: Path) -> dict:
    """{row id: (meta, residual)} from a previous, interrupted run."""
    if not path.is_file():
        return {}
    try:
        payload = cap.torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        return {}
    return {meta.get('id'): (meta, residual) for meta, residual in payload.get('records') or []}


def load_seed(path: Path) -> dict:
    """{row id: (meta, residual)} from an earlier FINISHED run of this same capture.

    The partial file only exists while a run is unfinished, and a run that ends
    early (``--limit``) finishes and deletes it. Without this seed, an incremental
    capture (the only kind this box allows: a 4 GiB worker cgroup cannot hold a
    float32 1.5B) would re-generate every row it already has. Rows are keyed by
    suite id, and the report records the suite digest, so a changed suite shows up
    in the report rather than silently mixing two pools.
    """
    if not path.is_file():
        return {}
    try:
        payload = cap.torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        return {}
    return {meta.get('id'): (meta, residual) for meta, residual in payload.get('records') or []}


def save_partial(path: Path, records: list) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    cap.torch.save({'records': records}, tmp)
    tmp.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='p7_q25_refit')
    parser.add_argument('--model', required=True, help='local HF checkpoint to capture from (the base)')
    parser.add_argument('--suite', type=Path, required=True, help='jsonl with id/kind/prompt fit rows')
    parser.add_argument('--out', type=Path, required=True, help='spanfit records .pt to write')
    parser.add_argument('--report', type=Path, required=True, help='JSON capture report')
    parser.add_argument('--kinds', default='refusal,harmless')
    parser.add_argument('--locus', choices=('answer', 'first-answer', 'last-prompt'), default='answer',
                        help='R4 locus; answer = mean of the answer span (same reading the q25 SAS fit used)')
    parser.add_argument('--max-tokens', type=int, default=40,
                        help='CPU generation cap per row; the label only needs the opener')
    parser.add_argument('--settle-min-tokens', type=int, default=20)
    parser.add_argument('--ctx-size', type=int, default=4096)
    parser.add_argument('--dtype', choices=('float32', 'bfloat16'), default='float32',
                        help='float32 is the fast CPU path (bf16 matmul falls back to emulation here)')
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--system', default='You are a helpful assistant.')
    parser.add_argument('--raw-prompt', action='store_true')
    parser.add_argument('--no-settle', dest='settle', action='store_false')
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    device = torch.device('cpu')
    kinds = tuple(part.strip() for part in args.kinds.split(',') if part.strip())
    rows, filtered = cap.read_rows(args.suite, kinds, args.limit)
    if not rows:
        raise SystemExit(f'{args.suite} has no rows for kinds {list(kinds)}')

    started = time.monotonic()
    config = AutoConfig.from_pretrained(args.model)
    width = cap.hidden_width(config)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[args.dtype]
    print(f'[p7] loading {args.model} on cpu ({args.dtype}, {args.threads} threads)', flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).eval()
    stack_name, layer_count = cap.find_layer_stack(model, width, cap.expected_layers(config))
    stack = dict(model.named_modules())[stack_name]
    layers = [stack[index] for index in range(layer_count)]
    targets = {index: cap.writeback_targets(layers[index], width) for index in range(layer_count)}
    empty = [index for index, found in targets.items() if not found]
    if empty:
        raise SystemExit(f'blocks {empty[:8]} carry no residual-write projection at width {width}')

    partial_path = args.out.with_suffix(args.out.suffix + '.partial')
    done = {**load_seed(args.out), **load_partial(partial_path)}
    records = list(done.values())
    labels, discarded, tokens = {}, {}, {'prompt': 0, 'answer': 0}
    settled = 0
    for meta, _residual in records:  # rows banked by an earlier chunk count, not just new ones
        labels[meta['label']] = labels.get(meta['label'], 0) + 1
        tokens['prompt'] += meta['prompt_tokens']
        tokens['answer'] += meta['answer_tokens']
        settled += 1 if meta.get('settled_at') else 0
    for position, row in enumerate(rows, 1):
        row_id = row.get('id')
        if row_id in done:
            print(f'[p7] {position}/{len(rows)} {row_id} already captured', flush=True)
            continue
        t0 = time.monotonic()
        try:
            captured, reason = cap.capture_one(
                model, tokenizer, row, layers, width, locus=args.locus,
                max_tokens=args.max_tokens, ctx_size=args.ctx_size, system=args.system,
                raw=args.raw_prompt, device=device, settle=args.settle,
                settle_min_tokens=args.settle_min_tokens, answer_text=None, margin_pairs=[])
        except (ValueError, RuntimeError) as error:
            discarded[str(error)[:200]] = discarded.get(str(error)[:200], 0) + 1
            print(f'[p7] {position}/{len(rows)} {row_id} discarded: {error}', flush=True)
            continue
        if captured is None:
            key = (reason or {}).get('reason', 'unknown')
            discarded[key] = discarded.get(key, 0) + 1
            print(f'[p7] {position}/{len(rows)} {row_id} discarded: {key}', flush=True)
            continue
        meta, residual = captured
        records.append((meta, residual))
        labels[meta['label']] = labels.get(meta['label'], 0) + 1
        tokens['prompt'] += meta['prompt_tokens']
        tokens['answer'] += meta['answer_tokens']
        settled += 1 if meta.get('settled_at') else 0
        save_partial(partial_path, records)
        print(f'[p7] {position}/{len(rows)} {row_id} {meta["label"]:<8} {meta["shape"]:<24} '
              f'{meta["answer_tokens"]:>3} tok {time.monotonic() - t0:5.1f}s', flush=True)

    if float(labels.get('bad', 0)) == 0 or float(labels.get('good', 0)) == 0:
        raise SystemExit(f'capture has no usable contrast: labels {labels}; spanfit needs bad and good rows')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'records': records}, args.out)
    identity = {
        'model': args.model, 'revision': getattr(config, '_commit_hash', None),
        'architecture': getattr(config, 'model_type', None), 'hidden': width,
        'layers': layer_count, 'layer_path': stack_name, 'locus': args.locus,
        'dtype': args.dtype, 'max_tokens': args.max_tokens, 'ctx_size': args.ctx_size,
        'kinds': list(kinds), 'suite': str(args.suite),
        'suite_sha256': sha256_file(args.suite),
        'targets': {index: [name for name, _ in found] for index, found in targets.items()},
        'compute_backend': {'type': 'cpu', 'rank': 'prep-only', 'name': 'torch cpu'},
        'refit': True,
    }
    lengths = [meta['answer_tokens'] for meta, _ in records]
    report = {
        'scope': 'CPU capture of the base for the p7 q25 SAS refit (prep card; no GPU, no wave).',
        'records': str(args.out), 'records_bytes': args.out.stat().st_size,
        'records_sha256': sha256_file(args.out), 'identity': identity,
        'rows_read': len(rows), 'records_written': len(records),
        'labels': labels, 'filtered_kinds': filtered, 'discarded': discarded,
        'tokens': tokens, 'settling': {'enabled': bool(args.settle),
                                       'min_tokens': args.settle_min_tokens,
                                       'rows_settled': settled},
        'answer_tokens': {'median': int(statistics.median(lengths)) if lengths else 0,
                          'max': max(lengths) if lengths else 0},
        'device': 'cpu', 'threads': args.threads,
        'elapsed_s': round(time.monotonic() - started, 1),
        'next': [
            (f'uv run python -m jdewitt.spanfit axes --records {args.out} --shape shared '
             f'--out {args.out.with_name("refit-axes.pt")}'),
        ],
        'note': ('This is a REFIT of the shared axis from a capture of the base. It is not the '
                 'deleted q25-v4-dir000.gguf adapter and must never be labelled SAS@1.5.'),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1) + '\n')
    partial_path.unlink(missing_ok=True)
    print(json.dumps({k: report[k] for k in ('rows_read', 'records_written', 'labels', 'elapsed_s')}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
