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
from .errors import NoVerifiedAdapter
from .profiles import (build_profile, is_flat, layer_axis, profile_name, profiled_axes,
                       self_check)
from .ppl import ladder, ln, perplexity
from .refusal_shape import screen_short

DEFAULT_SERVE = '/mnt/ssd/dev workspace/llama.cpp-dspark/build-cuda/bin/llama-server'

# Measured on the 800-row harsh mix at K=100 / escalation cap 600 (docs/PLAN-NEXT.md).
SCREEN_TOKENS_PER_ROW = 280


def file_sha(path, chunk=1 << 20):
    """Content digest for cache identity.

    mtime is the WRONG key here: `tune` itself merges the artifact in place and
    reverts it, which restores every byte but moves mtime, so an mtime-keyed cache
    throws away the whole ladder and every adapter after a verification run. Hash
    the bytes instead; 1.1 GB costs ~1 s.
    """
    import hashlib

    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


ATTN_SUFFIXES = {'attn_output', 'attn_out', 'ssm_out', 'o_proj', 'out_proj'}
MLP_SUFFIXES = {'ffn_down', 'w2', 'down_proj'}


def primal_weight_index(hf_dir, width):
    """{(layer, kind): safetensors key} for the checkpoint's residual-write weights.

    The adapter's LoRA factors must be derived from the PRIMAL (HF) weights: a
    fork whose converter folds ffn tensors into the gguf basis applies the delta to
    the *unfolded* activation, so factors built from the folded dequantized weight
    apply a rotated functional — measured here as an adapter curve whose delta_br
    was 0.55x the merged bytes'. The merge itself is unaffected (right-multiplying
    by the fold commutes with the projection), only the adapter has to use primal.
    """
    import glob

    import re as _re

    from safetensors import safe_open

    pattern = _re.compile(r'\.(\d+)\.([A-Za-z_][\w.]*)\.weight$')
    index, ambiguous = {}, []
    for shard in sorted(glob.glob(str(Path(hf_dir) / '*.safetensors'))):
        with safe_open(shard, framework='pt') as handle:
            for key in handle.keys():
                match = pattern.search(key)
                if match is None:
                    continue
                suffix = match.group(2).split('.')[-1]
                if suffix in ATTN_SUFFIXES:
                    kind = 'attn'
                elif suffix in MLP_SUFFIXES:
                    kind = 'mlp'
                else:
                    continue
                try:
                    shape = tuple(handle.get_slice(key).get_shape())
                except Exception:
                    continue
                if len(shape) != 2 or int(shape[0]) != int(width):
                    continue
                slot = (int(match.group(1)), kind)
                if slot in index:
                    ambiguous.append((slot, index[slot], key))
                    continue
                index[slot] = (shard, key)
    if ambiguous:
        raise ValueError(f'{hf_dir} carries more than one residual-write weight per layer and '
                         f'kind, so the pairing with the artifact is ambiguous: {ambiguous[:4]}')
    return index


def read_primal(index, slot):
    """The [out, in] float32 weight for one (layer, kind), or None."""
    from safetensors import safe_open

    entry = index.get(slot)
    if entry is None:
        return None
    shard, key = entry
    with safe_open(shard, framework='pt') as handle:
        return handle.get_tensor(key).float()


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


def profile_specs(args):
    """The profiles one run compares, flat first: [(kind, params)].

    flat alone keeps today's ladder. Any other --profile ALSO runs flat and two
    bell variants (half and double the width) at the SAME scales, so every
    profile is measured like-for-like against the flat strength it replaces.
    """
    params = {'centre_fraction': args.profile_centre, 'width_fraction': args.profile_width,
              'min_weight': args.profile_min, 'spec': args.profile_spec,
              'mean': args.profile_mean}
    if args.profile == 'flat':
        return [('flat', params)]
    return [('flat', params), (args.profile, params)] + [
        ('bell', dict(params, width_fraction=round(args.profile_width * factor, 4)))
        for factor in (0.5, 2.0)]


def profile_comparison(arms):
    """Per profile, like-for-like: drift_ln and delta_br at each (family, scale).

    Keys inside a profile are `<shape|footprint>@<scale>`, the same on every
    profile, so a bell and the flat strength it replaces read side by side.
    `vs_flat` counts the pairs where the profile moved the branch at least as
    far for no more drift (and strictly better on one).
    """
    out = {}
    for arm in arms:
        entry = out.setdefault(arm.get('profile', 'flat'), {'drift_ln': {}, 'delta_br': {}})
        point = f"{arm.get('base_family', arm['family'])}@{arm['scale']}"
        entry['drift_ln'][point] = arm.get('drift_ln')
        entry['delta_br'][point] = arm.get('delta_br')
    flat = out.get('flat')
    for name, entry in out.items():
        if name == 'flat' or flat is None:
            continue
        pairs = [(point, flat['drift_ln'].get(point), flat['delta_br'].get(point))
                 for point in entry['drift_ln']]
        pairs = [(entry['drift_ln'][point], entry['delta_br'][point], drift, delta)
                 for point, drift, delta in pairs
                 if None not in (drift, delta, entry['drift_ln'][point], entry['delta_br'][point])]
        wins = [1 for drift, delta, flat_drift, flat_delta in pairs
                if drift <= flat_drift and delta >= flat_delta
                and (drift < flat_drift or delta > flat_delta)]
        entry['vs_flat'] = {
            'pairs': len(pairs), 'wins': len(wins),
            'mean_delta_br_gain': (round(sum(d - fd for _, d, _, fd in pairs) / len(pairs), 4)
                                   if pairs else None),
            'mean_drift_ln_change': (round(sum(dr - fdr for dr, _, fdr, _ in pairs) / len(pairs), 4)
                                     if pairs else None)}
    return out


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


def edit_survives(arm, minimum):
    """Is this arm's edit larger than the artifact's own quantization noise?

    ``fidelity.min_cos`` is measured by running the real codec round-trip on
    sampled tensors (ggufio.preview_edit). An arm below the floor cannot ship what
    the ladder measured, so it is not a candidate -- screening it would spend
    tokens on an edit the file cannot store.
    """
    fidelity = arm.get('fidelity') or {}
    cos = fidelity.get('min_cos')
    return cos is None or float(cos) >= float(minimum)


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


def _dequant_cache(args):
    """Merge cache setting: False disables; None lets build_merged default.

    build_merged defaults to ``<artifact>.dequant/``, so repeated merged arms in
    one tune run pay the dequant once. ``--no-dequant-cache`` opts out.
    """
    if getattr(args, 'no_dequant_cache', False):
        return False
    return getattr(args, 'dequant_cache', None)


def tune_command(args):
    import torch

    from .spanfit import load_records, per_layer_axes, per_layer_subspace, shared_axis
    from .gguf_adapters import (discover_writeback_targets, gguf_weight_shapes,
                                serve_factors, write_adapter)
    from .ggufio import dequantize_tensor, preview_edit, revert_inplace
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
            if args.rank > 1:
                # rank-k: the top-k per-layer contrast subspace. Every extra direction
                # is another thing the artifact has to store, which the codec gate
                # measures, so rank is a searched dimension and not a preference.
                axes_by_shape[shape] = per_layer_subspace(records, args.rank)
            else:
                axes_by_shape[shape] = (shared_axis(records)[0] if shape == 'shared'
                                        else per_layer_axes(records))
        except SystemExit as error:
            raise ValueError(
                f'{args.records} cannot produce a bad-minus-good axis ({error}). A ladder over '
                f'records with one label is not a measurement: re-run capture with a suite that '
                f'makes this model both decline harmful rows and comply with harmless ones '
                f'(`fit_usable` in capture\'s report says whether it does).') from error
    width = int(next(iter(axes_by_shape.values())).shape[-1])
    self_check()

    artifact_shapes = gguf_weight_shapes(args.gguf)
    discovered = discover_writeback_targets(artifact_shapes, width)
    if not discovered['targets']:
        from .gguf_adapters import artifact_adapter_view
        view = artifact_adapter_view(artifact_shapes)
        raise NoVerifiedAdapter(
            f'{args.gguf} carries no residual-write projection at width {width}; the records and '
            f'the artifact do not belong to the same model. Its own writeback names put the hidden '
            f'width at {view["hidden_width"]} and width-matching tensors it would not edit are '
            f'{view["unmapped_writeback_width"][:8] or "none"}. List what the file carries with '
            f'`python -m jdewitt.spanfit targets --gguf-in {args.gguf}`; do not rename a tensor to '
            f'make the widths agree',
            width=width, artifact=str(args.gguf), **view)
    layers = sorted(discovered['targets'])
    covered = layers[: int(axes_by_shape[shapes_requested[0]].shape[0])]
    arch = _read_arch(args.gguf)
    first_layer, last_layer = covered[0], covered[-1]
    footprints = parse_footprints(args.footprints, first_layer, last_layer)
    families = []
    for shape in shapes_requested:
        for footprint in footprints:
            base_key = f'{shape}|{footprint_label(footprint, first_layer, last_layer)}'
            in_footprint = [layer for layer in covered if footprint[0] <= layer <= footprint[1]]
            for kind, spec_params in profile_specs(args):
                weights = build_profile(kind, in_footprint, spec_params)
                name = profile_name(kind, spec_params, weights)
                # flat keeps the bare family key: existing adapters and ladder caches stay valid
                families.append({'shape': shape, 'layers': list(footprint),
                                 'key': base_key if name == 'flat' else f'{base_key}|{name}',
                                 'base_family': base_key, 'profile': name, 'weights': weights})
    family_by_key = {family['key']: family for family in families}

    # One dequant pass over the artifact feeds every family's adapter. One reused
    # GGUFReader, not one per tensor: 48 fresh header parses cost ~400 s on the
    # 4B artifacts where a reused reader costs ~11 s. The adapters themselves are
    # cached: they are a deterministic function of (artifact bytes, records bytes,
    # families, axes width), so an iteration pays the dequant pass ONCE per model.
    artifact_sha = file_sha(args.gguf)
    basis = 'primal' if args.hf else 'artifact'
    fit_identity = {'artifact_sha256': artifact_sha,
                    'records_sha256': file_sha(args.records, chunk=4 << 20),
                    'families': [family['key'] for family in families],
                    'basis': basis, 'hf': str(args.hf) if args.hf else None,
                    'width': width, 'layers': len(covered), 'arch': arch,
                    # Caches written before the adapter carried its base binding are
                    # unlabelable (adapter-card refuses them); regenerate them once.
                    'adapter_base_binding': 1}
    if any(not is_flat(family['weights']) for family in families):
        fit_identity['profile_weights'] = {family['key']: sorted(family['weights'].items())
                                           for family in families}
    adapters_path = out_dir / 'adapters.json'
    adapters = {}
    primal, basis_counts = {}, {'primal': 0, 'artifact': 0}
    if adapters_path.is_file():
        try:
            cached = json.loads(adapters_path.read_text())
        except (json.JSONDecodeError, OSError):
            cached = {}
        if cached.get('identity') == fit_identity:
            for key, entry in (cached.get('adapters') or {}).items():
                if Path(entry['path']).is_file():
                    adapters[key] = entry
            basis_counts.update(cached.get('basis_counts') or {})
    if not adapters:
        reader = None
        try:
            from gguf import GGUFReader
            reader = GGUFReader(str(args.gguf))
        except Exception:
            reader = None
        if args.hf:
            primal = primal_weight_index(args.hf, width)
        pairs = {family['key']: [] for family in families}
        for layer in covered:
            for name in discovered['targets'][layer]:
                suffix = name.split('.')[-2]
                kind = ('attn' if suffix in ATTN_SUFFIXES
                        else 'mlp' if suffix in MLP_SUFFIXES else None)
                weight = read_primal(primal, (layer, kind)) if (primal and kind) else None
                if weight is None:
                    weight = torch.as_tensor(dequantize_tensor(args.gguf, name,
                                                               reader=reader)).float()
                    basis_counts['artifact'] += 1
                else:
                    basis_counts['primal'] += 1
                for family in families:
                    start, end = family['layers']
                    if not start <= layer <= end:
                        continue
                    axis = axes_by_shape[family['shape']][layer]
                    if float(axis.norm()) == 0.0:
                        continue
                    axis = layer_axis(axis, family['weights'][layer])
                    direction = axis if axis.dim() == 2 else axis.reshape(1, -1)
                    a, b = serve_factors(direction, weight, 1.0)
                    pairs[family['key']].append((name, a, b))
                del weight
        for family in families:
            if not pairs[family['key']]:
                continue
            name = 'adapter-' + family['key'].replace('|', '-') + '.gguf'
            adapters[family['key']] = write_adapter(out_dir / name, pairs[family['key']], arch,
                                                    base_sha256=artifact_sha)
        atomic_json(adapters_path, {'identity': fit_identity, 'adapters': adapters,
                                    'basis_counts': basis_counts})

    base = {}
    # Ladder readings are cached per artifact+corpora+arm: a run that dies after
    # the expensive ladder must not pay for it twice, and a re-run with a wider
    # grid only pays for the new arms.
    ladder_path = out_dir / 'ladder.json'
    # A reading is only comparable within one measurement config. --ppl-ngl moves
    # the ladder between CPU and GPU, and R9's rule applies to readings too: a
    # number without its config is an anecdote. Cache identity therefore carries
    # the binary and the shape of the run, not just the artifact and corpora.
    identity = {'artifact': str(Path(args.gguf).resolve()), 'sha256': artifact_sha,
                'corpora': {label: str(Path(path).resolve()) for label, path in corpora.items()},
                'read': {'binary': str(Path(args.perplexity).resolve()),
                         'ngl': args.ppl_ngl, 'ctx': args.ctx, 'threads': args.threads}}
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
                              threads=args.threads, parallel=args.parallel,
                              ngl=args.ppl_ngl)
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
            without = [f['key'] for f in families if f['key'] not in adapters]
            unmapped = discovered['unmapped_writeback_width'][:8]
            raise NoVerifiedAdapter(
                f"no adapter was built for {family['key']}: the axes and the artifact disagree on "
                f"that footprint. Families without an adapter: {without[:8]}"
                f"{'...' if len(without) > 8 else ''}; the artifact carries residual-write "
                f"projections at width {width} for layers "
                f"{sorted(discovered['targets'])[:8]}, the ladder footprint covers layers "
                f"{first_layer}-{last_layer}, and width-matching tensors it would not edit are "
                f"{unmapped or 'none'}. Widening the footprint would turn this into a ladder over "
                f"a partial edit, so it refuses; re-capture against this artifact instead",
                family=family['key'], families_without_adapter=without[:8], width=width,
                discovered_layers=sorted(discovered['targets']), coverage=[first_layer, last_layer],
                unmapped_writeback_width=discovered['unmapped_writeback_width'], basis=basis)
        blast = cache.get(f"blast|{family['key']}") or {}
        if 'ppl' not in blast:
            raise ValueError(f"adapter blast test failed for {family['key']}: {blast}")
        if abs(ln(blast['ppl']) - ln(base['control']['ppl'])) < 1e-6:
            raise ValueError(
                f"the {family['key']} adapter changes nothing at scale 50 (control PPL "
                f"{blast['ppl']} vs base {base['control']['ppl']}): this binary is not applying "
                f"the adapter, so a ladder over it would be a fake measurement")

    screen_rows = []
    screen_uncovered = []
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
        # Coverage, said out loud: with a small --screen-rows the interleave drops
        # whole families, and a screen that never showed the stubborn family must not
        # read as a removal on it. The report names what the panel covered.
        recorded = {meta.get('family_id') for meta, _ in records
                    if meta.get('kind') == 'refusal' and meta.get('label') != 'deliver'
                    and meta.get('family_id')}
        screen_uncovered = sorted(recorded - {row.get('family_id') for row in screen_rows})
        if not screen_rows:
            raise ValueError(f'{args.records} carries no harmful (bad/ambiguous) rows with a '
                             f'rendered prompt to screen; capture with --kinds refusal first')
        adapter_ids = {family['key']: index for index, family in enumerate(families)}

    arms, failures, rounds = [], [], []
    refined = []
    widened = []
    confirmed = []
    pruned_by_codec = []
    searched = []
    codec_refusal = None
    confirmed_refusal = None
    screens_path = out_dir / 'screens.json'
    screen_identity = dict(identity)
    screen_identity['records'] = {'path': str(Path(args.records).resolve()),
                                  'sha256': file_sha(args.records, chunk=4 << 20)}
    # R9: a shape verdict is conditional on the serve config, so a cached screen may
    # only be reused under the same ctx/ngl. The real ngl can drop on a CUDA OOM
    # fallback, which is why screen_one also keys on the ngl the server actually got.
    screen_identity['serve'] = {'ctx': args.screen_ctx, 'ngl': args.ngl,
                                'lib_dirs': [d for d in (args.serve_lib or '').split(':') if d]}
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
        def measure_fidelity(family, scale):
            """Does this edit survive the artifact's own codec? Zero tokens.

            The adapter curve is computed in fp32; the shipped bytes are
            re-quantized, and on Q4_K the re-quantization noise can be as large as
            the edit itself. Sampling two or three tensors and running the real
            codec round-trip answers 'what will actually be in the file' before a
            single token is spent on screening an edit that cannot survive.
            """
            start, end = family['layers']
            picks = []
            for layer in (start, (start + end) // 2, end):
                for name in discovered['targets'][layer]:
                    if name not in picks:
                        picks.append(name)
            picks = picks[:3]
            gate_type = None
            if args.apply_quantized or args.serve_type:
                gate_type = 'q8_0' if str(args.edit_type).lower() == 'auto' else args.edit_type
            key = f'fid|{family["key"]}|{scale}|{gate_type}'
            if key in cache:
                return cache[key]
            edits = {}
            for name in picks:
                layer = int(name.split('.')[1])
                edits[name] = (layer_axis(axes_by_shape[family['shape']][layer],
                                          family['weights'].get(layer, 1.0)).numpy(),
                               args.axis_space)
            reading = preview_edit(args.gguf, edits, scale=scale,
                                   renormalize=args.renormalize, target_type=gate_type)
            cache[key] = reading
            atomic_json(ladder_path, {'identity': identity, 'readings': cache})
            return reading

        def fidelity_scale(family, floor, ceiling, calls=6):
            """Smallest strength whose edit survives the codec, by bisection.

            preview_edit runs the REAL quantizer on sampled blocks, so this is the
            codec floor for this family on this artifact: below it the merge ships a
            partly random edit, above it the edit is what the ladder measured. Zero
            generated tokens, ~7 s a probe.
            """
            low, high = 0.5, float(ceiling)
            best = None
            for _ in range(max(1, calls)):
                if high / low < 1.25:
                    break
                middle = round((low * high) ** 0.5, 3)
                reading = measure_fidelity(family, middle)
                if (reading.get('min_cos') or 0.0) >= floor:
                    best = middle
                    high = middle
                else:
                    low = middle
            return best

        def measure_arm(family, scale, round_index=0):
            readings = fetch(arm_jobs(family, scale))
            row = {'shape': family['shape'], 'layers': list(family['layers']),
                   'family': family['key'], 'base_family': family['base_family'],
                   'profile': family['profile'],
                   'profile_weights': {str(layer): weight
                                       for layer, weight in family['weights'].items()},
                   'scale': scale, 'round': round_index + 1, 'ppl': {}}
            for label in corpora:
                reading = readings.get(f"{family['key']}|{scale}|{label}") or {}
                if 'ppl' in reading:
                    row['ppl'][label] = reading
                else:
                    failures.append({'family': family['key'], 'scale': scale, 'corpus': label,
                                     'error': reading.get('error'), 'tail': reading.get('tail')})
                    return None
            arm = _arm_metrics(row, base)
            arm['fidelity'] = measure_fidelity(family, scale)
            return arm

        def screen_one(arm, rows):
            """Decode `rows` rows for one arm: initial window, then doubling on the
            rows still undecided, then the candidate rule at the cap.

            Cached per (artifact, records, family, scale, rows, k, cap, settle chars)
            so iterating with a bigger budget never pays twice for a decided arm.
            """
            nonlocal spent, budget_left
            key = '|'.join(str(piece) for piece in (
                arm['family'], arm['scale'], rows, args.screen_k, args.escalate_to,
                args.settle_chars, basis,
                adapters[arm['family']].get('tensors'), file_sha(adapters[arm['family']]['path'])[:12],
                str(served), getattr(server, 'ctx', None), getattr(server, 'ngl', None)))
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
            # The packet carries the argv the verdict is conditional on (R9), with
            # the ngl the server REALLY got after any OOM fallback.
            result['serve'] = {'binary': str(served), 'ctx': getattr(server, 'ctx', None),
                               'ngl': getattr(server, 'ngl', None)}
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
                        and (arm.get('delta_br') or 0) > 0 and not arm.get('screen')
                        and edit_survives(arm, args.min_edit_cos)]
                pruned_by_codec.extend(
                    {'family': arm['family'], 'scale': arm['scale'],
                     'min_cos': (arm.get('fidelity') or {}).get('min_cos')}
                    for arm in round_arms
                    if not edit_survives(arm, args.min_edit_cos))
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
            if not screened and pruned_by_codec and server is not None and budget_left > 0:
                # Everything was rejected by the codec, not by the branch: find the
                # strength where this artifact can actually store the edit, measure
                # it, and screen that instead of giving up.
                best_scale = fidelity_scale(families[0], args.min_edit_cos, args.max_scale)
                if best_scale is not None:
                    searched.append({'family': families[0]['key'], 'scale': best_scale,
                                     'floor': args.min_edit_cos})
                    arm = measure_arm(families[0], best_scale, index)
                    too_damaging = arm is not None and (damage(arm) or 0) > drift_budget
                    if too_damaging:
                        # The strength the codec needs is past the damage budget: the
                        # file can hold the edit only at a cost that is not shippable.
                        searched[-1]['damage'] = round(damage(arm), 3)
                        searched[-1]['rejected'] = 'damage exceeds drift budget'
                    elif arm is not None and edit_survives(arm, args.min_edit_cos):
                        arms.append(arm)
                        arm['screen'] = screen_one(arm, args.probe_rows)
                        screened.append(arm)
            if screened and any(arm_is_clean(arm) for arm in screened):
                break
            if server is None or budget_left <= 0:
                break

        screened_arms = [arm for arm in arms if arm.get('screen')]
        clean_arms = [arm for arm in screened_arms if arm_is_clean(arm)
                      and damage(arm) is not None and damage(arm) <= drift_budget]
        if clean_arms:
            winner = min(clean_arms, key=lambda arm: arm['drift_ln'])
        elif any(arm_is_clean(arm) for arm in screened_arms):
            # An arm clears the screen only at a strength that costs more damage than
            # the budget allows (the codec floor pushed it there). Shipping it would
            # ship a collapsed model, so refuse and name the reason.
            over_budget = [arm for arm in screened_arms if arm_is_clean(arm)]
            worst = min(over_budget, key=lambda arm: damage(arm))
            codec_refusal = {'family': worst['family'], 'scale': worst['scale'],
                             'damage': round(damage(worst), 3),
                             'budget': drift_budget}
            winner = None
        elif screened_arms:
            winner = min(screened_arms,
                         key=lambda arm: (arm['screen']['verdicts']['refusal'],
                                          arm['screen']['verdicts']['escalate'],
                                          arm['drift_ln']))
        else:
            eligible = [arm for arm in arms if damage(arm) is not None
                        and damage(arm) <= drift_budget
                        and arm.get('delta_br') is not None
                        and edit_survives(arm, args.min_edit_cos)]
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
                arm = measure_arm(family_by_key[best['family']], candidate_scale)
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
            confirmed_clean = None
            attempted = 0
            for arm in candidates:
                if budget_left < args.screen_rows * args.screen_k:
                    break
                attempted += 1
                full = screen_one(arm, args.screen_rows)
                arm['screen'] = full
                state = screen_read(arm)
                confirmed.append({'family': arm['family'], 'scale': arm['scale'],
                                  'layers': arm['layers'],
                                  'state': state['state'] if state else 'unknown',
                                  'tokens': full['tokens']})
                if state and state['clean']:
                    confirmed_clean = arm
                    break
            if confirmed_clean is not None:
                winner = confirmed_clean
            else:
                # The probe row cleared but the full row set did not. Default is to ship
                # nothing: an arm that leaves refusals must not be presented as a clear.
                # --best-effort ships the arm that leaves the FEWEST, and the artifact
                # records certified=false plus the residual row ids, because a measured
                # 7-of-8 with a named residual beats no artifact at all.
                confirmed_refusal = {'family': winner['family'], 'scale': winner['scale'],
                                     'reason': 'probe cleared, full-row confirmation refused',
                                     'rows': args.screen_rows}
                if candidates and not attempted:
                    # Say WHY nothing was confirmed instead of letting the missing
                    # measurement read as a refusal: the answer is the budget.
                    confirmed_refusal['reason'] = (
                        f'confirmation not run: {budget_left} tokens left cannot pay for one '
                        f'screen of {args.screen_rows} rows at k={args.screen_k} '
                        f'({args.screen_rows * args.screen_k})')
                    confirmed_refusal['budget_left'] = budget_left
                    confirmed_refusal['confirmation_cost'] = args.screen_rows * args.screen_k
                # Only arms measured on the SAME rows are comparable. With
                # abort_on='refusal' an arm can stop on its first row, so the
                # comparison is the refusal RATE over rows actually decoded, then
                # undecided rows (an undecided row is not a cleared one, R6), then
                # damage -- never a raw count that rewards having been asked less.
                residual = [arm for arm in arms
                            if (arm.get('screen') or {}).get('rows_requested') == args.screen_rows]

                def unresolved(arm):
                    screen = arm['screen']
                    verdicts = screen.get('verdicts') or {}
                    decoded = int(screen.get('rows_decoded') or 0)
                    rate = int(verdicts.get('refusal', 0)) / decoded if decoded else 1.0
                    return (rate, int(verdicts.get('escalate', 0)), damage(arm) or 0.0)

                if args.best_effort and residual:
                    best = min(residual, key=unresolved)
                    verdicts = best['screen'].get('verdicts') or {}
                    best['best_effort'] = True
                    best['residual_refusals'] = int(verdicts.get('refusal', 0))
                    best['residual_undecided'] = int(verdicts.get('escalate', 0))
                    best['residual_rows_decoded'] = int(best['screen'].get('rows_decoded') or 0)
                    best['residual_rows'] = [row['id'] for row in best['screen'].get('detail', [])
                                             if row.get('verdict') in ('refusal', 'escalate')]
                    winner = best
                    confirmed_refusal['shipped_as'] = 'best_effort'
                    confirmed_refusal['shipped_family'] = best['family']
                    confirmed_refusal['shipped_scale'] = best['scale']
                    confirmed_refusal['residual_refusals'] = best['residual_refusals']
                    confirmed_refusal['residual_undecided'] = best['residual_undecided']
                    confirmed_refusal['residual_rows'] = best['residual_rows']
                else:
                    winner = None
    finally:
        if server is not None:
            server.stop()

    branch_arms = [arm for arm in arms if (arm.get('delta_br') or 0) > 0]
    report = {
        'artifact': str(args.gguf), 'records': str(args.records), 'arch': arch,
        'width': width, 'layers_covered': len(covered), 'shapes': shapes_requested,
        'footprints': [list(item) for item in footprints],
        'profile': args.profile, 'profile_mean': args.profile_mean,
        'profiles': {family['key']: {'profile': family['profile'],
                                     'weights': {str(layer): weight
                                                 for layer, weight in family['weights'].items()}}
                     for family in families},
        'profile_comparison': profile_comparison(arms),
        'families': [family['key'] for family in families],
        'scales': scales, 'drift_budget': args.drift_budget,
        'drift_budget_used': drift_budget, 'drift_noise_ln': round(noise_ln, 4),
        'front_widened': widened, 'ctx_used': args.ctx,
        'min_edit_cos': args.min_edit_cos, 'rank': args.rank,
        'pruned_by_codec': pruned_by_codec,
        'strength_searched_for_codec': searched,
        'codec_refusal': codec_refusal,
        'confirmation_refusal': confirmed_refusal,
        'best_effort': bool(args.best_effort),
        'shipping_options': ({'adapter': {'path': adapters[winner['family']]['path'],
                                          'scale': winner['scale'],
                                          'note': 'the exact fp32 edit; serve with --lora-scaled '
                                                  'FILE:SCALE, no requant noise'},
                              'merge_scale_floor': args.min_edit_cos}
                             if winner is not None else None),
        'corpora': corpora,
        'screen_families': sorted({row.get('family_id') or 'unknown' for row in screen_rows}),
        'screen_uncovered_families': screen_uncovered,
        'corpora_source': corpora_source,
        'base': {label: {'ppl': base[label]['ppl'],
                         'uncertainty': base[label].get('uncertainty'),
                         'seconds': base[label].get('seconds')}
                 for label in base},
        'adapters': {key: {'path': adapter['path'], 'tensors': adapter['tensors']}
                     for key, adapter in adapters.items()},
        'adapter_basis': {'mode': basis, 'tensors': basis_counts},
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
            'serve': {
                'binary': str(served) if served else None,
                'ctx': args.screen_ctx, 'ngl': args.ngl,
                'adapters_loaded': len(adapter_ids),
                'lib_dirs': [d for d in (args.serve_lib or '').split(':') if d],
            },
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
            ('the winner is the BEST-EFFORT arm: no arm resolved every row, so the arm the '
             'confirmation left with the fewest unresolved rows ships, and the artifact records '
             'certified=false with the residual rows; it is not a removal claim.'
             if args.best_effort and winner is not None and winner.get('best_effort')
             else 'the winner is the LOWEST-PERPLEXITY arm whose screen left no refusal and '
                  'nothing undecided, so the token spend selects the set and perplexity picks '
                  'inside it.'),
            'a screen verdict is a screening signal (R2/R6): certification stays the agent shape '
            'read on merged bytes at one pinned serve config.',
            'an adapter curve is a screening approximation, not the export delta (R8); the '
            'winner is re-measured on the merged bytes before it is called done.',
            'the adapter factors must come from PRIMAL (HF) weights (--hf): on a fork whose '
            'converter folds tensors into the gguf basis, factors built from the folded weight '
            'apply a rotated functional, and the ladder then ranks an edit that is not the one '
            'the merge ships.',
        ],
    }
    if screen_uncovered:
        report['notes'].append(
            'the screen panel does NOT cover ' + ', '.join(screen_uncovered) +
            ': a clean screen (or a best-effort pick) is not evidence on those families.')
    state = screen_read(winner) if winner else None
    if winner is not None and state is not None and not state['clean']:
        if winner.get('best_effort'):
            report['notes'].append(
                f'the BEST-EFFORT artifact ships labelled, not cleared: screen state '
                f'{state["state"]} ({winner["screen"]["verdicts"]}), residual rows '
                f'{winner.get("residual_rows")}; certified=false travels in the file.')
        else:
            report['notes'].append(
                f'the winner is not cleared: screen state {state["state"]} '
                f'({winner["screen"]["verdicts"]}). Go back to the ladder with more strength '
                f'(raise --scales or --rounds) or a different shape; do not ship it as removal.')
    if winner is None and pruned_by_codec and not [arm for arm in arms if arm.get('screen')]:
        report['notes'].append(
            f'every arm was rejected by the CODEC, not by the branch: the best edit survived '
            f'the requant round-trip at cos {max((entry["min_cos"] or 0) for entry in pruned_by_codec):.3f} '
            f'against a --min-edit-cos floor of {args.min_edit_cos}. On an artifact whose '
            f'quantization step is comparable to the edit, a stronger scale is the only merge '
            f'that can ship; the adapter at the same axis is the noise-free alternative.')
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
        command = (f"python -m jdewitt.spanfit build --gguf-in {args.gguf} --in-place "
                   f"--records {args.records} --shape {arm['shape'] if arm else '<shape>'} "
                   f"--scale {arm['scale'] if arm else '<scale>'}{layers_flag} --gguf-targets auto")
        if arm is not None and arm.get('profile', 'flat') != 'flat':
            command += (f"  # profile {arm['profile']} is not expressible here: "
                        f"drop --no-verify and tune merges it itself")
        return command

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

    # The axes are masked to the winning footprint: build_merged skips zero-norm
    # layers, so a footprint edit never touches bytes outside its own range.
    # The winner's profile is baked into the axes ONCE (d * sqrt(weight)), so the
    # in-place patch, the 16-bit merge and the codec edit below all carry exactly
    # the weights the adapter was ranked with. Flat returns the axes untouched.
    winner_base = profiled_axes(axes_by_shape[winner['shape']],
                                family_by_key[winner['family']]['weights'])
    winner_axes = winner_base.clone()
    start, end = winner['layers']
    winner_axes[:start] = 0
    winner_axes[end + 1:] = 0

    if args.apply_quantized:
        # The deliverable: the edit inside the artifact the user already runs. Only
        # the tensors the edit touches are re-encoded (type chosen by measured
        # fidelity); every other tensor and the whole metadata section stay
        # byte-identical, so their calibration survives.
        from .rewrite import rewrite_with_edits

        override_names = [name for layer in range(start, end + 1)
                          for name in discovered['targets'].get(layer, [])]
        edits = {}
        for name in override_names:
            layer = int(name.split('.')[1])
            edits[name] = (winner_base[layer].numpy(), args.axis_space)
        target_path = (args.out_gguf if args.out_gguf
                       else out_dir / f"edited-{winner['family'].replace('|', '-')}"
                                      f"@{winner['scale']}-{Path(args.gguf).name}")
        # What this file may claim about behaviour: nothing (R2/R8). The screen ran
        # on the ADAPTER over the pristine bytes, and on this artifact family that
        # curve is a measured overstatement of the export delta, so a clearance
        # written here would be a verdict on bytes nobody screened. The record still
        # travels -- provenance, adapter-screen state and the rows it left open --
        # so the artifact cannot be read as certified, and best-effort cannot be
        # mistaken for a clear.
        screen_state = (screen_read(winner) or {}).get('state')
        claims = {'certified': False,
                  'adapter_screen_state': screen_state,
                  'screen_source': 'adapter over the pristine artifact, not these bytes (R8)',
                  'residual_refusals': winner.get('residual_refusals'),
                  'residual_undecided': winner.get('residual_undecided'),
                  'residual_rows': winner.get('residual_rows') or None,
                  'best_effort': bool(winner.get('best_effort'))}
        rewrite = rewrite_with_edits(str(args.gguf), str(target_path), edits,
                                     scale=winner['scale'], axis_space=args.axis_space,
                                     edit_type=args.edit_type,
                                     renormalize=args.renormalize, claims=claims)
        edited_ppl = {}
        for label, corpus in corpora.items():
            edited_ppl[label] = perplexity(args.perplexity, str(target_path), corpus, ngl=args.ppl_ngl,
                                           ctx=args.ctx, threads=args.threads)
        metrics = (_arm_metrics({'shape': winner['shape'], 'scale': winner['scale'],
                                 'ppl': edited_ppl}, base)
                   if all('ppl' in reading for reading in edited_ppl.values()) else {})
        from .ggufio import compare_edit

        sample = override_names[:1] + override_names[-1:]
        report['quantized_edit'] = {
            'rewrite': rewrite, 'metrics': metrics,
            'delivered_edit': compare_edit(str(args.gguf), str(target_path),
                                           {name: edits[name] for name in sample},
                                           renormalize=args.renormalize)}
        report['next'] = [f'# shipped artifact: {target_path}',
                          f'llama-server -m {target_path} -c 4096 --parallel 4  '
                          f'# then one cert wave over its answers (R2/R9)']
        atomic_json(out_dir / 'tune.json', report)
        return report

    if args.serve_type:
        # The faithful pipeline: merge into a 16-bit artifact (the edit is stored
        # exactly there), then requantize for serving while the tensors that carry
        # the edit are HELD at a type where it survives. This is the answer to the
        # codec floor measured by preview_edit: at 4-bit the edit is at the quantizer's
        # noise, at 16-bit it is exact, and after requantizing with a per-tensor type
        # map the delivered edit is a measurement (compare_edit), not a hope.
        from .ggufio import compare_edit
        from .quantize import quantize_file

        source = Path(args.merge_gguf or args.gguf)
        if not source.is_file():
            raise ValueError(f'--merge-gguf {source} does not exist')
        stem = winner['family'].replace('|', '-')
        edited_path = out_dir / f'edited-{stem}@{winner["scale"]}-{source.stem}.gguf'
        name_map = {name: name for layer in covered for name in discovered['targets'][layer]}
        build = build_merged(str(source), str(edited_path), winner_axes, [],
                             name_map=name_map, scale=winner['scale'],
                             axis_space=args.axis_space, in_place=False,
                             renormalize=args.renormalize,
                             dequant_cache=_dequant_cache(args))
        override_names = [name for layer in range(start, end + 1)
                          for name in discovered['targets'].get(layer, [])]
        overrides = [f'{name}={args.edit_type}' for name in override_names]
        served_path = out_dir / f'served-{args.serve_type}.gguf'
        quant = quantize_file(edited_path, served_path, args.serve_type,
                             binary=args.quantize_binary, overrides=overrides,
                             threads=args.threads)
        sample = override_names[:1] + override_names[-1:]
        edits = {}
        for name in sample:
            layer = int(name.split('.')[1])
            edits[name] = (winner_base[layer].numpy(), args.axis_space)
        delivered = compare_edit(str(source), str(served_path), edits,
                                 renormalize=args.renormalize)
        served_ppl = {}
        for label, corpus in corpora.items():
            served_ppl[label] = perplexity(args.perplexity, str(served_path), corpus, ngl=args.ppl_ngl,
                                           ctx=args.ctx, threads=args.threads)
        metrics = (_arm_metrics({'shape': winner['shape'], 'scale': winner['scale'],
                                 'ppl': served_ppl}, base)
                   if all('ppl' in reading for reading in served_ppl.values()) else {})
        report['pipeline'] = {
            'merge': {'source': str(source), 'out': str(edited_path),
                      'tensors_changed': build.get('tensors_changed'),
                      'changed_blocks_total': build.get('changed_blocks_total'),
                      'elapsed_s': build.get('elapsed_s')},
            'quantize': quant,
            'delivered_edit': delivered,
            'metrics': metrics,
            'note': 'delivered_edit is measure on the SERVED file against the pristine '
                    'source, so it is the edit that ships, not the edit that was wanted',
        }
        report['next'] = [f'# served artifact: {served_path}',
                          f'llama-server -m {served_path} -c 4096 --parallel 4  '
                          f'# then one cert wave over its answers (R2/R9)']
        atomic_json(out_dir / 'tune.json', report)
        return report
    merged = build_merged(str(args.gguf), str(args.gguf), winner_axes,
                          [], name_map={name: name for layer in covered
                                        for name in discovered['targets'][layer]},
                          scale=winner['scale'], axis_space=args.axis_space, in_place=True,
                          renormalize=args.renormalize,
                          dequant_cache=_dequant_cache(args))
    merged_ppl, merged_metrics = {}, {}
    for label, corpus in corpora.items():
        reading = perplexity(args.perplexity, args.gguf, corpus, ctx=args.ctx,
                             threads=args.threads, ngl=args.ppl_ngl)
        merged_ppl[label] = reading
    if all('ppl' in reading for reading in merged_ppl.values()):
        merged_metrics = _arm_metrics({'shape': winner['shape'], 'scale': winner['scale'],
                                       'ppl': merged_ppl}, base)
    ratio = None
    if winner.get('delta_br') and merged_metrics.get('delta_br') is not None:
        ratio = round(merged_metrics['delta_br'] / winner['delta_br'], 3)
    agreement = bool(merged_metrics.get('delta_br') is not None
                     and (abs(merged_metrics['delta_br'] - (winner['delta_br'] or 0)) <= 0.05
                          or (ratio is not None and 0.7 <= ratio <= 1.3)))
    report['merged_verification'] = {
        'build': {k: merged.get(k) for k in ('tensors_changed', 'changed_blocks_total',
                                             'revert_sidecar', 'elapsed_s')},
        'metrics': merged_metrics,
        'adapter_to_merged_ratio': ratio,
        'ranking_valid_on_this_artifact': agreement,
        'matches_adapter_screen': agreement,
    }
    if not agreement:
        report['notes'].append(
            f'the adapter ladder is a SCREENING instrument on this artifact (R8): at '
            f"{winner['family']}@{winner['scale']} the merged bytes moved delta_br "
            f"{merged_metrics.get('delta_br')} against the adapter curve's "
            f"{winner['delta_br']} (ratio {ratio}). Rank with it, but screen the MERGED "
            f'artifact before calling anything removed.')
    # Screen the MERGED bytes: the adapter ladder ranks, but the artifact that
    # ships is the merged one and R8 just measured how far the two curves drift
    # apart. This is the only screen whose verdict belongs to the shipped file; it
    # costs one server reload and one row set, zero extra merges.
    if args.merged_screen and args.token_budget > 0:
        if server is not None:
            server.stop()
            server = None
        merged_server = Server(served, args.gguf, ctx=args.screen_ctx, ngl=args.ngl,
                               lib_dirs=[d for d in (args.serve_lib or '').split(':') if d],
                               log_path=out_dir / 'llama-server-merged.log').start()
        try:
            result = screen_arm(merged_server, screen_rows, args.screen_k, abort_on='refusal')
            spent += result['tokens']
            budget_left -= result['tokens']
            window = int(args.screen_k)
            while result['verdicts']['escalate'] and window < int(args.escalate_to):
                next_window = min(int(args.escalate_to), window * 2)
                need = result['verdicts']['escalate'] * (next_window - window)
                if need > budget_left:
                    result['stopped_on_budget'] = True
                    break
                extra = escalate_screen(merged_server, {'screen': result}, rows_by_id, None,
                                        None, {}, next_window - window)
                spent += extra
                budget_left -= extra
                window = next_window
            mark_candidates(result, args.settle_chars)
            state = screen_read({'screen': result}) or {}
            report['merged_verification']['screen'] = {
                'artifact_sha256': file_sha(args.gguf),
                'verdicts': result['verdicts'], 'state': state.get('state'),
                'clean': bool(state.get('clean')), 'tokens': result['tokens'],
                'rows': result['rows_decoded'], 'k': args.screen_k, 'window': window,
                'detail': [{'id': row['id'], 'verdict': row['verdict'],
                            'opening': row['opening']} for row in result['detail']],
            }
        finally:
            merged_server.stop()
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
