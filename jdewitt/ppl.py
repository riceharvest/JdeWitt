# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prefill-only perplexity: rank an edit without generating a single token.

llama-perplexity scores a corpus in one prefill pass, so a strength or shape
ladder costs zero output tokens (METHODOLOGY R1: inference volume is the cost).
Two-sided reading, as measured in this repo:
  BR      = ln PPL(refusal corpus) - ln PPL(delivery corpus)
            how much better the model predicts its own refusal continuations
            than delivery text on the same harmful prompts; a removal that
            actually damages the refusal branch moves this down.
  control = ln PPL(control corpus) drift, so a BR move can be attributed to the
            branch instead of to the whole distribution moving.
One-sided refusal-only readings were falsified as a branch screen (style
distance dominates), so callers must supply both corpora to interpret BR.

LoRA adapters are the cheap arm affordance: an arm is a small adapter file plus
`--lora-scaled FILE:SCALE`, so a strength ladder is seconds per arm and no
artifact builds. That curve is a SCREENING signal, not a shipped edit (R8: on
tensors the server graph transforms, the adapter curve is not the export delta),
which is why the caller re-measures the winner on the merged file.
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

PPL_RE = re.compile(r'PPL = ([0-9.]+)(?: \+\/- ([0-9.]+))?')

# Default CPU build of the dspark fork used by every banked probe in this repo.
DEFAULT_BINARY = '/mnt/ssd/dev workspace/llama.cpp-dspark/build-perp/bin/llama-perplexity'


def perplexity(binary, model, corpus, *, ctx=512, threads=16, lora=None,
               lora_scale=None, loras=None, ngl=0, timeout=1800):
    """PPL of `model` over `corpus`, optionally with one LoRA adapter applied."""
    path = Path(model)
    if not path.is_file():
        raise ValueError(f'{model} does not exist')
    if not Path(corpus).is_file():
        raise ValueError(f'corpus {corpus} does not exist')
    if not Path(binary).exists():
        raise ValueError(f'perplexity binary {binary} does not exist; pass --perplexity')
    # -ngl is a parameter, not a constant: the ladder is compute-bound and a GPU reads
    # the same numbers ~10x faster, so refusing GPU perplexity would make the free part
    # of the search the slow part.
    command = [str(binary), '-m', str(path), '-f', str(corpus), '-ngl', str(int(ngl)),
               '-c', str(ctx), '-t', str(threads)]
    if loras:
        # Repeatable --lora-scaled is deprecated in this fork: only the LAST value is
        # used, so a multi-adapter arm (per-band strength profile) must arrive as ONE
        # comma-separated argument. Mixing the two spellings silently drops adapters.
        command += ['--lora-scaled',
                    ','.join(f'{path}:{float(scale):g}' for path, scale in loras)]
    elif lora:
        if lora_scale is None:
            command += ['--lora', str(lora)]
        else:
            command += ['--lora-scaled', f'{lora}:{float(lora_scale):g}']
    started = time.monotonic()
    finished = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    output = finished.stdout + finished.stderr
    match = PPL_RE.search(output)
    if match is None:
        return {'error': 'no PPL line', 'returncode': finished.returncode,
                'tail': output[-600:]}
    return {'ppl': float(match.group(1)),
            'uncertainty': float(match.group(2)) if match.group(2) else None,
            'seconds': round(time.monotonic() - started, 1)}


def ln(value):
    import math
    if value is None or value <= 0:
        raise ValueError('perplexity must be positive to take a log')
    return math.log(value)


def ladder(binary, model, jobs, threads=16, parallel=3, ngl=0, timeout=1800):
    """Run many perplexity readings CONCURRENTLY: {key: reading}.

    ``jobs`` are dicts with key/corpus/lora/scale/ctx (and optionally ngl, which
    overrides the ladder default for one reading). Each reading is its own
    llama-perplexity process, so a shape x strength x corpus ladder costs about one
    sequential run instead of one per arm: measured 3.7 s per reading on a 1.7B
    Q4_K_M at ctx 128, and 3-way concurrency turns a 17-reading ladder into ~7 s
    of wall clock while every reading still carries its own thread count.
    """
    from concurrent.futures import ThreadPoolExecutor

    per_job_threads = max(1, int(threads) // max(1, int(parallel)))

    def one(job):
        reading = perplexity(binary, model, job['corpus'], ctx=job.get('ctx', 512),
                             threads=per_job_threads, lora=job.get('lora'),
                             lora_scale=job.get('scale'), loras=job.get('loras'),
                             ngl=job.get('ngl', ngl), timeout=timeout)
        return job['key'], reading

    results = {}
    if not jobs:
        return results
    with ThreadPoolExecutor(max_workers=max(1, int(parallel))) as pool:
        for key, reading in pool.map(one, jobs):
            results[key] = reading
    return results
