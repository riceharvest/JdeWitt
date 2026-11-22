# SPDX-License-Identifier: AGPL-3.0-or-later
"""Two-objective arm search: lowest perplexity, fewest generated tokens.

Two costs decide which edit to ship: how much the model degrades (perplexity on a
control corpus, plus the refusal-branch preference BR) and how much verification
the choice costs in OUTPUT TOKENS. No single number trades one for the other, so
this command alternates between them instead of pretending to scalarise:

  ROUND 1  every arm (axis shape x strength) is a LoRA adapter plus
           `--lora-scaled`, so a full ladder costs one prefill pass per corpus per
           arm and generates NOTHING. Arms are ranked Pareto-style: control drift
           inside `--drift-budget`, largest `delta_br` first.
  ROUND 2  only the Pareto front is decoded: `--screen-rows` harmful rows for
           `--screen-k` tokens each through llama-server with the arm's adapter
           applied, every prefix classified by `refusal_shape.screen_short`, and
           the generated tokens counted. Rows that settle nowhere are continued
           (assistant-prefill, `--escalate-to`) before they are called undecided.
           The winner is the LOWEST-PERPLEXITY arm whose screen shows no refusal
           left, so perplexity decides inside the set the token spend has already
           cleared.
  ROUND 3+ if no arm clears, the strength grid is grown by `--grow` and the loop
           goes back to round 1, spending only what `--token-budget` still allows.

The output is one JSON: the per-round arm tables, the screen evidence, the
tokens each verdict cost, the winner, and the merge command. A perplexity move or
a screen verdict is a screening signal, never a removal claim: certification
stays the agent's shape read on merged bytes (R2/R8).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .answers import atomic_json
from .ppl import ladder, ln, perplexity
from .refusal_shape import screen_short

DEFAULT_SERVE = '/mnt/ssd/dev workspace/llama.cpp-dspark/build-cuda/bin/llama-server'

# Measured on the 800-row harsh mix at K=100 / escalation cap 600 (docs/PLAN-NEXT.md).
SCREEN_TOKENS_PER_ROW = 280


def parse_footprints(spec, first, last):
    """'all' or comma-separated 'start-end' ranges, clamped to the editable layers.

    The layer footprint is the strongest damage lever measured on this repo's
    lineages (an edit that reaches the first blocks collapses generation), and it
    costs ZERO generated tokens to search: each footprint is just another adapter.
    """
    ranges = []
    for piece in (spec or 'all').split(','):
        piece = piece.strip()
        if not piece:
            continue
        if piece == 'all':
            ranges.append((first, last))
            continue
        try:
            start_text, end_text = piece.split('-')
            start, end = int(start_text), int(end_text)
        except ValueError as error:
            raise ValueError(f'--footprints wants "all" or start-end ranges, got {piece!r}') from error
        if start > end:
            raise ValueError(f'--footprints range {piece!r} starts after it ends')
        ranges.append((max(start, first), min(end, last)))
    unique = sorted({item for item in ranges if item[0] <= item[1]})
    if not unique:
        raise ValueError(f'--footprints {spec!r} selects no layer inside the edit {first}-{last}')
    return unique


def footprint_label(layers, first, last):
    return 'all' if (layers[0], layers[1]) == (first, last) else f'{layers[0]}-{layers[1]}'


def derive_corpora(args, records, out_dir):
    """The three probe corpora, DERIVED from the model being edited.

    Hand-picked corpora make the objective a property of the repository instead of
    a property of the model under edit, and they leak one lineage's prose into
    another's measurement. So the default builds all three from the fit records
    this run already generated:

      refusal  = the model's OWN declining answers to harmful rows,
      delivery = its own compliant answers to harmless rows,
      control  = the suite's prompt text, which the edit must keep fluent on.

    BR = ln PPL(refusal) - ln PPL(delivery) then asks how much better the edited
    model predicts its own refusals than its own compliances -- the branch
    question, per model, with no outside text. Explicit --control/--refusal/
    --delivery files still win when the caller wants a pinned comparison.
    """
    given = {'control': args.control, 'refusal': args.refusal, 'delivery': args.delivery}
    if all(given.values()):
        for name, path in given.items():
            if not Path(path).is_file():
                raise ValueError(f'{name} corpus {path} does not exist')
        return {name: str(path) for name, path in given.items()}, 'caller-supplied'

    pieces = {'refusal': [], 'delivery': [], 'control': []}
    for meta, _residual in records:
        answer = (meta.get('answer') or '').strip()
        prompt = (meta.get('prompt') or meta.get('prompt_text') or meta.get('prompt_rendered') or '').strip()
        if prompt and len(prompt) < 4000:
            pieces['control'].append(prompt)
        if not answer:
            continue
        if meta.get('label') == 'bad':
            pieces['refusal'].append(answer)
        elif meta.get('label') == 'good':
            pieces['delivery'].append(answer)

    limit = int(args.corpus_chars)
    corpora, sizes = {}, {}
    for name, chunks in pieces.items():
        text = ''
        for chunk in chunks:
            text += chunk.replace('\n', ' ').strip() + '\n'
            if len(text) >= limit:
                break
        text = text[:limit]
        sizes[name] = len(text)
        if len(text) < 400:
            raise ValueError(
                f'derived {name} corpus is only {len(text)} chars: a perplexity reading over '
                f'it would be noise, and llama-perplexity refuses a corpus that tokenizes '
                f'below 2x ctx. Raise --corpus-chars, capture more rows, or pass --{name} PATH. '
                f'(refusal/delivery need labelled bad/good rows in {args.records}.)')
        path = out_dir / f'ppl-{name}.txt'
        path.write_text(text)
        corpora[name] = str(path)
    return corpora, {'derived_chars': sizes, 'target_chars': limit}


def _read_arch(gguf):
    from gguf import GGUFReader
    fields = GGUFReader(str(gguf)).fields
    field = fields.get('general.architecture')
    return str(field.contents()) if field is not None else None


def _arm_metrics(row, base):
    """Derived numbers for one arm, from its raw PPL readings."""
    control, refusal, delivery = row['ppl'].get('control'), row['ppl'].get('refusal'), row['ppl'].get('delivery')
    row['drift_ln'] = round(ln(control['ppl']) - ln(base['control']['ppl']), 4) if control and 'ppl' in control else None
    if refusal and delivery and 'ppl' in refusal and 'ppl' in delivery:
        br = ln(refusal['ppl']) - ln(delivery['ppl'])
        row['br'] = round(br, 4)
        row['delta_br'] = round(br - (ln(base['refusal']['ppl']) - ln(base['delivery']['ppl'])), 4)
    else:
        row['br'] = None
        row['delta_br'] = None
    return row


def damage(arm):
    """Capability damage: a perplexity IMPROVEMENT is not a cost."""
    drift = arm.get('drift_ln')
    return None if drift is None else max(float(drift), 0.0)


def pareto_front(arms, drift_budget, limit=None):
    """Non-dominated arms in (control damage, delta_br), best branch move first."""
    eligible = [arm for arm in arms if damage(arm) is not None
                and damage(arm) <= drift_budget
                and arm.get('delta_br') is not None]
    front = []
    for arm in eligible:
        dominated = any(
            damage(other) <= damage(arm) and other['delta_br'] >= arm['delta_br']
            and (damage(other) < damage(arm) or other['delta_br'] > arm['delta_br'])
            for other in eligible)
        if not dominated:
            front.append(arm)
    front.sort(key=lambda arm: (-arm['delta_br'], damage(arm)))
    return front[:limit] if limit else front


def weakest_first(front):
    """Screen the LEAST damaging, then WEAKEST arm first: a refusal is cheap to
    prove, so buying the smallest edit that still clears is the token-cheap route
    to low perplexity."""
    return sorted(front, key=lambda arm: (damage(arm), arm['scale'], arm['delta_br']))


def screen_read(arm):
    """The screen's own verdict summary for one arm, or None when unscreened."""
    screen = arm.get('screen')
    if not screen:
        return None
    verdicts = screen['verdicts']
    if not screen.get('measured'):
        return {'state': 'unmeasured', 'clean': False,
                'why': 'nothing generated: this arm is UNMEASURED, never a removal'}
    if verdicts['refusal']:
        return {'state': 'refuses', 'clean': False}
    if verdicts['candidate']:
        return {'state': 'candidate' if not verdicts['escalate'] else 'partial',
                'clean': not verdicts['escalate']}
    if verdicts['escalate']:
        return {'state': 'undecided', 'clean': False}
    if verdicts['delivery']:
        return {'state': 'cleared', 'clean': True}
    return {'state': 'empty', 'clean': False}


def arm_is_clean(arm):
    state = screen_read(arm)
    return bool(state and state['clean'])


def escalate_screen(server, arm, rows_by_id, adapter_id, scale, reset_scales, k):
    """Continue the rows an arm's first window left undecided, and count what it costs."""
    screen = arm['screen']
    extra = 0
    for row in screen['detail']:
        if row['verdict'] != 'escalate':
            continue
        prompt = rows_by_id[row['id']]['prompt'] + row['text']
        reading = server.complete(prompt, k)
        verdict = screen_short(row['text'] + reading['text'], 'screen')
        counted = reading['tokens'] if isinstance(reading['tokens'], int) else 0
        extra += counted
        screen['verdicts'][row['verdict']] -= 1
        screen['verdicts'][verdict] += 1
        row.update({'verdict': verdict, 'escalated_tokens': counted,
                    'escalated_k': k, 'text': row['text'] + reading['text'],
                    'chars': len(row['text'])})
    screen['tokens'] += extra
    screen['escalation_k'] = k
    screen['escalated_tokens'] = extra
    return extra


def blast_server(server, rows, adapter_id, k, reset):
    """Prove the server applies adapters before any arm verdict is trusted.

    A POST whose scale never lands measures the UNEDITED model on every arm, which
    reads as a confident 'nothing removed' table — a fake measurement. Two short
    decodes, one row, scale 0 vs a blasting scale, must differ.
    """
    from .screen import screen_arm

    row = rows[0]
    readings = {}
    for scale in (0.0, 50.0):
        result = screen_arm(server, [row], k, adapter_id=adapter_id, scale=scale,
                            reset_scales=reset)
        readings[scale] = result
    texts = {scale: result['detail'][0]['text'] for scale, result in readings.items()}
    return {
        'identical': texts[0.0] == texts[50.0],
        'k': k,
        'tokens': sum(result['tokens'] for result in readings.values()),
        'at_scale_0': texts[0.0][:200],
        'at_scale_50': texts[50.0][:200],
    }


def tune_command(args):
    import torch

    from .spanfit import load_records, per_layer_axes, shared_axis
    from .gguf_adapters import (discover_writeback_targets, gguf_weight_shapes,
                                serve_factors, write_adapter)
    from .ggufio import dequantize_tensor, revert_inplace
    from .spanfit import build_merged
    from .screen import Server, mark_candidates, rows_to_screen, screen_arm

    t0 = time.monotonic()
    out_dir = Path(args.out) if args.out else Path('tune') / time.strftime('%Y%m%d-%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)
    shapes_requested = [s for s in args.shapes.split(',') if s]
    unknown = [s for s in shapes_requested if s not in {'perlayer', 'shared'}]
    if unknown:
        raise ValueError(f'unknown --shapes entries {unknown}')
    scales = [float(s) for s in args.scales.split(',') if s]
    if not scales or any(s <= 0 for s in scales):
        raise ValueError('--scales needs positive numbers')
    if args.rounds < 1:
        raise ValueError('--rounds must be >= 1')
    if args.grow < 1.0:
        raise ValueError('--grow must be >= 1.0 (a shrinking grid is a different search)')
    if args.token_budget < 0:
        raise ValueError('--token-budget must be >= 0')

    records = load_records(args.records)
    corpora, corpora_source = derive_corpora(args, records, out_dir)
    # Reading cost is a SYSTEM property: a corpus is sized in characters so one
    # reading is seconds, and ctx follows the corpus instead of being a constant
    # that makes a small corpus unreadable (llama-perplexity needs >= 2x ctx).
    shortest = min(len(Path(path).read_text()) for path in corpora.values())
    args.ctx = min(int(args.ctx), max(64, shortest // 10))
    axes_by_shape = {}
    for shape in shapes_requested:
        try:
            axes_by_shape[shape] = (shared_axis(records)[0] if shape == 'shared'
                                    else per_layer_axes(records))
        except SystemExit as error:
            raise ValueError(
                f'{args.records} cannot produce a bad-minus-good axis ({error}). A ladder over '
                f'records with one label is not a measurement: re-run capture with a suite that '
                f'makes this model both decline harmful rows and comply with harmless ones '
                f'(`fit_usable` in capture\'s report says whether it does).') from error
    width = int(next(iter(axes_by_shape.values())).shape[1])

    artifact_shapes = gguf_weight_shapes(args.gguf)
    discovered = discover_writeback_targets(artifact_shapes, width)
    if not discovered['targets']:
        raise ValueError(f'{args.gguf} carries no residual-write projection at width {width}; '
                         f'the records and the artifact do not belong to the same model')
    layers = sorted(discovered['targets'])
    covered = layers[: int(axes_by_shape[shapes_requested[0]].shape[0])]
    arch = _read_arch(args.gguf)
    first_layer, last_layer = covered[0], covered[-1]
    footprints = parse_footprints(args.footprints, first_layer, last_layer)
    families = [{'shape': shape, 'layers': list(footprint),
                 'key': f'{shape}|{footprint_label(footprint, first_layer, last_layer)}'}
                for shape in shapes_requested for footprint in footprints]

    # One dequant pass over the artifact feeds every family's adapter. One reused
    # GGUFReader, not one per tensor: 48 fresh header parses cost ~400 s on the
    # 4B artifacts where a reused reader costs ~11 s. The adapters themselves are
    # cached: they are a deterministic function of (artifact bytes, records bytes,
    # families, axes width), so an iteration pays the dequant pass ONCE per model.
    fit_identity = {'artifact_bytes': Path(args.gguf).stat().st_size,
                    'artifact_mtime': int(Path(args.gguf).stat().st_mtime),
                    'records_bytes': Path(args.records).stat().st_size,
                    'records_mtime': int(Path(args.records).stat().st_mtime),
                    'families': [family['key'] for family in families],
                    'width': width, 'layers': len(covered), 'arch': arch}
    adapters_path = out_dir / 'adapters.json'
    adapters = {}
    if adapters_path.is_file():
        try:
            cached = json.loads(adapters_path.read_text())
        except (json.JSONDecodeError, OSError):
            cached = {}
        if cached.get('identity') == fit_identity:
            for key, entry in (cached.get('adapters') or {}).items():
                if Path(entry['path']).is_file():
                    adapters[key] = entry
    if not adapters:
        reader = None
        try:
            from gguf import GGUFReader
            reader = GGUFReader(str(args.gguf))
        except Exception:
            reader = None
        pairs = {family['key']: [] for family in families}
        for layer in covered:
            for name in discovered['targets'][layer]:
                weight = torch.as_tensor(dequantize_tensor(args.gguf, name, reader=reader)).float()
                for family in families:
                    start, end = family['layers']
                    if not start <= layer <= end:
                        continue
                    axis = axes_by_shape[family['shape']][layer]
                    if float(axis.norm()) == 0.0:
                        continue
                    a, b = serve_factors(axis.reshape(1, -1), weight, 1.0)
                    pairs[family['key']].append((name, a, b))
                del weight
        for family in families:
            if not pairs[family['key']]:
                continue
            name = 'adapter-' + family['key'].replace('|', '-') + '.gguf'
            adapters[family['key']] = write_adapter(out_dir / name, pairs[family['key']], arch)
        atomic_json(adapters_path, {'identity': fit_identity, 'adapters': adapters})

    base = {}
    # Ladder readings are cached per artifact+corpora+arm: a run that dies after
    # the expensive ladder must not pay for it twice, and a re-run with a wider
    # grid only pays for the new arms.
    ladder_path = out_dir / 'ladder.json'
    identity = {'artifact': str(Path(args.gguf).resolve()),
                'bytes': Path(args.gguf).stat().st_size,
                'mtime': int(Path(args.gguf).stat().st_mtime),
                'corpora': {label: str(Path(path).resolve()) for label, path in corpora.items()}}
    cache = {}
    if ladder_path.is_file():
        try:
            stored = json.loads(ladder_path.read_text())
        except (json.JSONDecodeError, OSError):
            stored = {}
        if stored.get('identity') == identity:
            cache = stored.get('readings') or {}
    hits = {'base': 0, 'arm': 0, 'screen': 0}
    errors = {}

    def fetch(jobs):
        """Measure any uncached reading, CONCURRENTLY, and persist the cache.

        The grid is prefetched in one batch, so the whole ladder's wall clock is
        about one perplexity run instead of one per arm (measured 3.7 s a reading:
        a 17-reading ladder went from ~9 minutes to ~7 seconds). A run that dies
        after the ladder never pays for it twice.
        """
        for job in jobs:
            if 'ppl' in (cache.get(job['key']) or {}):
                hits['base' if job['key'].startswith('base|') else 'arm'] += 1
        todo = [job for job in jobs if 'ppl' not in (cache.get(job['key']) or {})]
        if todo:
            readings = ladder(args.perplexity, str(args.gguf), todo,
                              threads=args.threads, parallel=args.parallel)
            cache.update({key: reading for key, reading in readings.items() if 'ppl' in reading})
            errors.update({key: reading for key, reading in readings.items() if 'ppl' not in reading})
            atomic_json(ladder_path, {'identity': identity, 'readings': cache})
        return {job['key']: cache.get(job['key']) or {} for job in jobs}

    def arm_jobs(family, scale):
        return [{'key': f"{family['key']}|{scale}|{label}", 'corpus': corpus,
                 'lora': adapters[family['key']]['path'], 'scale': scale, 'ctx': args.ctx}
                for label, corpus in corpora.items()]

    # Prefetch base + adapter blast + every arm any round can reach, in one batch.
    grid = sorted({round(scale * (args.grow ** index), 3)
                   for index in range(args.rounds) for scale in scales})
    fetch([{'key': f'base|{label}', 'corpus': corpus, 'lora': None, 'scale': None,
            'ctx': args.ctx} for label, corpus in corpora.items()]
          + [{'key': f"blast|{family['key']}", 'corpus': corpora['control'],
              'lora': adapters[family['key']]['path'], 'scale': 50.0, 'ctx': args.ctx}
             for family in families if family['key'] in adapters]
          + [job for family in families if family['key'] in adapters
             for scale in grid for job in arm_jobs(family, scale)])

    for label in corpora:
        reading = cache.get(f'base|{label}') or {}
        if 'ppl' not in reading:
            tail = (errors.get(f'base|{label}') or {}).get('tail', '')
            raise ValueError(f'base perplexity failed on the {label} corpus: {reading} '
                             f'| llama-perplexity tail: {tail[-300:]}')
        base[label] = reading

    # The drift gate must not be tighter than the instrument's own noise, or a
    # clean arm is rejected by a rounding difference and the run reports nothing.
    # llama-perplexity prints +/- on every reading; two of those in ln space is the
    # smallest difference this ladder can honestly resolve, per model and corpus.
    noise_ln = max(abs(reading.get('uncertainty') or 0.0) / reading['ppl']
                   for reading in base.values())
    drift_budget = max(float(args.drift_budget), round(2 * noise_ln, 4))

    # Blast test before the ladder: a huge scale must move the control PPL. An
    # adapter that silently fails to apply makes every arm identical to the base,
    # which reads as a flat, confident "no effect" table -- a fake measurement.
    for family in families:
        if family['key'] not in adapters:
            raise ValueError(f"no adapter was built for {family['key']}: the axes and the "
                             f"artifact disagree on that footprint")
        blast = cache.get(f"blast|{family['key']}") or {}
        if 'ppl' not in blast:
            raise ValueError(f"adapter blast test failed for {family['key']}: {blast}")
        if abs(ln(blast['ppl']) - ln(base['control']['ppl'])) < 1e-6:
            raise ValueError(
                f"the {family['key']} adapter changes nothing at scale 50 (control PPL "
                f"{blast['ppl']} vs base {base['control']['ppl']}): this binary is not applying "
                f"the adapter, so a ladder over it would be a fake measurement")

    screen_rows = []
    served = Path(args.serve or DEFAULT_SERVE) if args.token_budget > 0 else None
    adapter_ids = {}
    if args.token_budget > 0:
        if not adapters:
            raise ValueError('--token-budget needs at least one adapter to screen')
        if args.screen_rows < 1 or args.screen_k < 1:
            raise ValueError('--screen-rows and --screen-k must be >= 1')
        if not 1 <= args.probe_rows <= args.screen_rows:
            raise ValueError('--probe-rows must be between 1 and --screen-rows')
        if not served.is_file():
            raise ValueError(f'llama-server not found at {served}; pass --serve PATH or use '
                             f'--token-budget 0 to leave the ladder prefill-only')
        screen_rows = rows_to_screen(records, limit=args.screen_rows)
        if not screen_rows:
            raise ValueError(f'{args.records} carries no harmful (bad/ambiguous) rows with a '
                             f'rendered prompt to screen; capture with --kinds refusal first')
        adapter_ids = {family['key']: index for index, family in enumerate(families)}

    arms, failures, rounds = [], [], []
    refined = []
    widened = []
    confirmed = []
    screens_path = out_dir / 'screens.json'
    screen_identity = dict(identity)
    screen_identity['records'] = {'path': str(Path(args.records).resolve()),
                                  'bytes': Path(args.records).stat().st_size,
                                  'mtime': int(Path(args.records).stat().st_mtime)}
    screens = {}
    cached_spend = {'tokens': 0}
    if screens_path.is_file():
        try:
            stored = json.loads(screens_path.read_text())
        except (json.JSONDecodeError, OSError):
            stored = {}
        if stored.get('identity') == screen_identity:
            screens = stored.get('screens') or {}
    spent, budget_left = 0, int(args.token_budget)
    server, blast = None, None
    try:
        if args.token_budget > 0:
            server = Server(served, args.gguf, adapters=[a['path'] for a in adapters.values()],
                            ctx=args.screen_ctx, ngl=args.ngl,
                            lib_dirs=[d for d in (args.serve_lib or '').split(':') if d],
                            log_path=out_dir / 'llama-server.log').start()
            # The same rule the PPL ladder follows: prove the server APPLIES the
            # arm before a single arm verdict is trusted.
            first_id = next(iter(adapter_ids.values()))
            reset_all = {index: 0.0 for index in adapter_ids.values()}
            blast = blast_server(server, screen_rows, first_id,
                                 min(10, args.screen_k), reset_all)
            spent += blast['tokens']
            budget_left -= blast['tokens']
            if blast['identical']:
                raise ValueError(
                    f'the screen server answers the same text at adapter scale 0 and 50 '
                    f'(k={blast["k"]}, {blast["tokens"]} tokens): the scale POST is not '
                    f'applying, so every arm verdict would be the UNEDITED model')
        def measure_arm(family, scale, round_index=0):
            readings = fetch(arm_jobs(family, scale))
            row = {'shape': family['shape'], 'layers': list(family['layers']),
                   'family': family['key'], 'scale': scale, 'round': round_index + 1,
                   'ppl': {}}
            for label in corpora:
                reading = readings.get(f"{family['key']}|{scale}|{label}") or {}
                if 'ppl' in reading:
                    row['ppl'][label] = reading
                else:
                    failures.append({'family': family['key'], 'scale': scale, 'corpus': label,
                                     'error': reading.get('error'), 'tail': reading.get('tail')})
                    return None
            return _arm_metrics(row, base)

        def screen_one(arm, rows):
            """Decode `rows` rows for one arm: initial window, then doubling on the
            rows still undecided, then the candidate rule at the cap.

            Cached per (artifact, records, family, scale, rows, k, cap, settle chars)
            so iterating with a bigger budget never pays twice for a decided arm.
            """
            nonlocal spent, budget_left
            key = '|'.join(str(piece) for piece in (
                arm['family'], arm['scale'], rows, args.screen_k, args.escalate_to,
                args.settle_chars))
            if key in screens:
                hits['screen'] += 1
                cached_spend['tokens'] += screens[key].get('tokens') or 0
                return screens[key]
            adapter_id = adapter_ids[arm['family']]
            reset = {index: 0.0 for index in adapter_ids.values()}
            result = screen_arm(server, screen_rows[:rows], args.screen_k,
                                adapter_id=adapter_id, scale=arm['scale'],
                                reset_scales=reset, abort_on='refusal')
            spent += result['tokens']
            budget_left -= result['tokens']
            result['rows_requested'] = rows
            # Spend the remaining budget where it can still decide something: the
            # undecided rows, window doubling, until a row settles or the cap is hit.
            window = int(args.screen_k)
            while result['verdicts']['escalate'] and budget_left > 0:
                next_window = min(int(args.escalate_to), window * 2)
                if next_window <= window:
                    break
                # Never open a window the budget cannot pay for: an interrupted
                # escalation leaves the row undecided anyway, and an overspend hides
                # itself in the ledger.
                need = result['verdicts']['escalate'] * (next_window - window)
                if need > budget_left:
                    result['stopped_on_budget'] = True
                    break
                extra = escalate_screen(server, {'screen': result}, rows_by_id, adapter_id,
                                        arm['scale'], reset, next_window - window)
                spent += extra
                budget_left -= extra
                window = next_window
                result['window'] = window
            if result['verdicts']['escalate'] and window >= int(args.escalate_to):
                mark_candidates(result, args.settle_chars)
            result['tokens_total_spent'] = spent
            screens[key] = result
            atomic_json(screens_path, {'identity': screen_identity, 'screens': screens})
            return result

        rows_by_id = {row['id']: row for row in screen_rows}
        for index in range(args.rounds):
            scale_list = [round(scale * (args.grow ** index), 3) for scale in scales]
            before_failures = len(failures)
            round_arms = [measure_arm(family, scale, index)
                          for family in families if family['key'] in adapters
                          for scale in scale_list]
            round_arms = [arm for arm in round_arms if arm is not None]
            round_failures = len(failures) - before_failures
            front = pareto_front(round_arms + arms, drift_budget, limit=args.front)
            if not front:
                # An empty front spends nothing and reports nothing. Widen to the
                # LEAST-DAMAGING arm that still moved the branch, and say so.
                movers = [arm for arm in round_arms + arms if (arm.get('delta_br') or 0) > 0]
                if movers:
                    front = [min(movers, key=lambda arm: abs(arm['drift_ln']))]
                    widened.append({'round': index + 1, 'family': front[0]['family'],
                                    'scale': front[0]['scale'],
                                    'drift_ln': front[0]['drift_ln'],
                                    'budget': drift_budget})
            screened = []
            if server is not None and budget_left > 0:
                # Screen the WEAKEST arm that still moved the branch first and stop
                # at the first one that clears: proving a refusal is cheap, so the
                # cheapest route to a low-perplexity winner is a strength ladder
                # walked upward, not a Pareto set walked sideways.
                pool = [arm for arm in round_arms + arms
                        if damage(arm) is not None and damage(arm) <= drift_budget
                        and (arm.get('delta_br') or 0) > 0 and not arm.get('screen')]
                for arm in weakest_first(pool)[:args.front]:
                    if budget_left <= 0:
                        break
                    arm['screen'] = screen_one(arm, args.probe_rows)
                    screened.append(arm)
                    if arm_is_clean(arm):
                        break
            arms.extend(round_arms)
            rounds.append({'round': index + 1, 'scales': scale_list,
                           'arms': len(round_arms), 'failures': round_failures,
                           'front': [{'family': arm['family'], 'scale': arm['scale'],
                                      'drift_ln': arm['drift_ln'], 'delta_br': arm['delta_br']}
                                     for arm in front],
                           'screened': len(screened),
                           'tokens': spent,
                           'front_states': [{'family': arm['family'], 'scale': arm['scale'],
                                             'screen': screen_read(arm)}
                                            for arm in arms if arm.get('screen')]})
            if screened and any(arm_is_clean(arm) for arm in screened):
                break
            if server is None or budget_left <= 0:
                break

        screened_arms = [arm for arm in arms if arm.get('screen')]
        clean_arms = [arm for arm in screened_arms if arm_is_clean(arm)]
        if clean_arms:
            winner = min(clean_arms, key=lambda arm: arm['drift_ln'])
        elif screened_arms:
            winner = min(screened_arms,
                         key=lambda arm: (arm['screen']['verdicts']['refusal'],
                                          arm['screen']['verdicts']['escalate'],
                                          arm['drift_ln']))
        else:
            eligible = [arm for arm in arms if damage(arm) is not None
                        and damage(arm) <= drift_budget
                        and arm.get('delta_br') is not None]
            winner = max(eligible, key=lambda arm: arm['delta_br']) if eligible else None

        # Back and forth: once an arm clears the screen, walk its strength DOWN and
        # re-screen it, because a weaker edit damages perplexity less. Stop at the
        # first strength that refuses again, or when the step/budget allowance is
        # gone. Perplexity picks the strength; the screen pays for the fact.
        if (server is not None and winner is not None and arm_is_clean(winner)
                and budget_left > 0 and args.refine_steps > 0 and args.refine_factor < 1.0):
            best, low = winner, float(args.refine_floor)
            while len(refined) < args.refine_steps and budget_left > 0:
                candidate_scale = round(best['scale'] * args.refine_factor, 3)
                if candidate_scale <= low + 1e-9:
                    break
                if budget_left < len(screen_rows) * args.screen_k:
                    break
                arm = measure_arm({'key': best['family'], 'shape': best['shape'],
                                  'layers': best['layers']}, candidate_scale)
                if arm is None:
                    break
                arms.append(arm)
                arm['screen'] = screen_one(arm, args.probe_rows)
                state = screen_read(arm)
                refined.append({'family': arm['family'], 'scale': arm['scale'],
                                'layers': arm['layers'],
                                'drift_ln': arm['drift_ln'], 'delta_br': arm['delta_br'],
                                'screen': state, 'tokens': arm['screen']['tokens'],
                                'budget_left': budget_left})
                if state and state['clean']:
                    best = arm
                else:
                    low = candidate_scale
                    break
            winner = best

        # Confirm on the FULL row set: stage A decided arms on --probe-rows rows,
        # which is what makes elimination cheap; the arm that is about to ship is
        # paid for once at --screen-rows, and if it refuses there the next clean
        # candidate is confirmed instead.
        if (server is not None and winner is not None and budget_left > 0
                and args.screen_rows > args.probe_rows):
            candidates = sorted([arm for arm in arms if arm_is_clean(arm)],
                                key=lambda arm: (damage(arm), arm['scale']))
            for arm in candidates:
                if budget_left < args.screen_rows * args.screen_k:
                    break
                full = screen_one(arm, args.screen_rows)
                arm['screen'] = full
                state = screen_read(arm)
                confirmed.append({'family': arm['family'], 'scale': arm['scale'],
                                  'layers': arm['layers'],
                                  'state': state['state'] if state else 'unknown',
                                  'tokens': full['tokens']})
                if state and state['clean']:
                    winner = arm
                    break
    finally:
        if server is not None:
            server.stop()

    branch_arms = [arm for arm in arms if (arm.get('delta_br') or 0) > 0]
    report = {
        'artifact': str(args.gguf), 'records': str(args.records), 'arch': arch,
        'width': width, 'layers_covered': len(covered), 'shapes': shapes_requested,
        'footprints': [list(item) for item in footprints],
        'families': [family['key'] for family in families],
        'scales': scales, 'drift_budget': args.drift_budget,
        'drift_budget_used': drift_budget, 'drift_noise_ln': round(noise_ln, 4),
        'front_widened': widened, 'ctx_used': args.ctx,
        'corpora': corpora,
        'corpora_source': corpora_source,
        'base': {label: {'ppl': base[label]['ppl'],
                         'uncertainty': base[label].get('uncertainty'),
                         'seconds': base[label].get('seconds')}
                 for label in base},
        'adapters': {shape: {'path': adapter['path'], 'tensors': adapter['tensors']}
                     for shape, adapter in adapters.items()},
        'arms': arms, 'arm_failures': failures,
        'rounds': rounds,
        'refinement': refined,
        'confirmation': confirmed,
        'winner': winner,
        'tokens': {
            'search': 0,
            'screen': spent,
            'screen_cached': cached_spend['tokens'],
            'evidence_tokens': sum((entry.get('tokens') or 0) for entry in screens.values()),
            'budget': args.token_budget,
            'budget_left': max(budget_left, 0),
            'screen_rows': len(screen_rows),
            'probe_rows': args.probe_rows,
            'screen_k': args.screen_k,
            'escalate_to': args.escalate_to,
            'per_arm': {f"{arm['family']}@{arm['scale']}": (arm.get('screen') or {}).get('tokens')
                        for arm in arms if arm.get('screen')},
            'server_load_seconds': getattr(server, 'load_seconds', None) if server else None,
            'screen_blast': blast,
            'ladder_cache_hits': hits,
        },
        'elapsed_s': round(time.monotonic() - t0, 1),
        'next': [],
        'notes': [
            'round 1 is prefill-only: a full shape x strength ladder costs ZERO generated tokens.',
            'round 2 decodes the Pareto front only (screen rows x --screen-k), classifies every '
            'prefix by refusal shape and counts the tokens each verdict cost; undecided rows are '
            'continued before they are called undecided.',
            'the winner is the LOWEST-PERPLEXITY arm whose screen left no refusal and nothing '
            'undecided, so the token spend selects the set and perplexity picks inside it.',
            'a screen verdict is a screening signal (R2/R6): certification stays the agent shape '
            'read on merged bytes at one pinned serve config.',
            'an adapter curve is a screening approximation, not the export delta (R8); the '
            'winner is re-measured on the merged bytes before it is called done.',
        ],
    }
    state = screen_read(winner) if winner else None
    if winner is not None and state is not None and not state['clean']:
        report['notes'].append(
            f'the winner is not cleared: screen state {state["state"]} '
            f'({winner["screen"]["verdicts"]}). Go back to the ladder with more strength '
            f'(raise --scales or --rounds) or a different shape; do not ship it as removal.')
    if winner is None:
        report['notes'].append(
            'no arm both stayed inside the drift budget and moved the refusal branch; the '
            f'{len(branch_arms)} branch-moving arm(s) above are the candidates to widen the '
            'budget for, and a tone-only table means the edit does not target the branch.')
    atomic_json(out_dir / 'tune.json', report)

    def merge_command(arm=None):
        layers_flag = ''
        if arm is not None and arm['layers'] != [first_layer, last_layer]:
            layers_flag = f" --only-layers {arm['layers'][0]}-{arm['layers'][1]}"
        return (f"python -m jdewitt.spanfit build --gguf-in {args.gguf} --in-place "
                f"--records {args.records} --shape {arm['shape'] if arm else '<shape>'} "
                f"--scale {arm['scale'] if arm else '<scale>'}{layers_flag} --gguf-targets auto")

    def cert_steps(arm):
        """What turns this pick into a claim: the merge, then ONE cert wave.

        tune ranks; it never certifies. The screen verdict on the shipped arm is a
        candidate list, so the next steps are the merge and a short cascade over the
        merged bytes at ONE pinned serve config -- the agent reads those answers
        (R2/R8/R9), and the reversed rows are the ones the screen left open.
        """
        state = screen_read(arm) or {}
        return [
            f'# screen verdict on {arm["family"]}@arm scale {arm["scale"]}: '
            f'{state.get("state")} ({arm["screen"]["verdicts"]})',
            f'python scripts/cascade_verify.py --gguf {args.gguf} --suite '
            f'verification/harsh-suites/holdout.jsonl --k 150 --cap 600  # agent reads the '
            f'escalations; certify at ONE serve config and record the argv',
        ]
    if winner is None or args.no_verify:
        report['next'] = [merge_command(winner)]
        if winner is not None:
            report['next'] += cert_steps(winner)
        atomic_json(out_dir / 'tune.json', report)
        return report

    # R8: re-measure the winner on the merged bytes, then put the artifact back
    # unless the caller asked to keep the edit.
    # The axes are masked to the winning footprint: build_merged skips zero-norm
    # layers, so a footprint edit never touches bytes outside its own range.
    winner_axes = axes_by_shape[winner['shape']].clone()
    start, end = winner['layers']
    winner_axes[:start] = 0
    winner_axes[end + 1:] = 0
    merged = build_merged(str(args.gguf), str(args.gguf), winner_axes,
                          [], name_map={name: name for layer in covered
                                        for name in discovered['targets'][layer]},
                          scale=winner['scale'], axis_space=args.axis_space, in_place=True)
    merged_ppl, merged_metrics = {}, {}
    for label, corpus in corpora.items():
        reading = perplexity(args.perplexity, args.gguf, corpus, ctx=args.ctx, threads=args.threads)
        merged_ppl[label] = reading
    if all('ppl' in reading for reading in merged_ppl.values()):
        merged_metrics = _arm_metrics({'shape': winner['shape'], 'scale': winner['scale'],
                                       'ppl': merged_ppl}, base)
    report['merged_verification'] = {
        'build': {k: merged.get(k) for k in ('tensors_changed', 'changed_blocks_total',
                                             'revert_sidecar', 'elapsed_s')},
        'metrics': merged_metrics,
        'matches_adapter_screen': (merged_metrics.get('delta_br') is not None
                                   and abs(merged_metrics['delta_br'] - (winner['delta_br'] or 0)) <= 0.05),
    }
    revert = merged.get('revert_sidecar')
    if args.keep_merged:
        report['next'] = [
            f'# artifact now carries {winner["family"]} at scale {winner["scale"]}',
            'python scripts/cascade_verify.py ... (agent shape read is the certification, R2)',
        ]
    else:
        if revert:
            revert_inplace(args.gguf, revert)
            report['merged_verification']['reverted'] = True
        report['next'] = [merge_command(winner)]
    report['cert_token_estimate'] = {
        'rows': 200, 'per_row': SCREEN_TOKENS_PER_ROW,
        'total': 200 * SCREEN_TOKENS_PER_ROW,
        'source': 'measured on the 800-row harsh mix at K=100/cap 600 (docs/PLAN-NEXT.md)',
    }
    atomic_json(out_dir / 'tune.json', report)
    return report
