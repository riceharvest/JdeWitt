# SPDX-License-Identifier: AGPL-3.0-or-later
"""P6 GATED: adapter-vs-merged runtime equivalence on one fitted plan.

One fitted plan (a per-layer rank-1 footprint), two artifacts built from it on
the same basis and the same base bytes:

  * the GGUF LoRA adapter   -- gguf_adapters.write_adapter
  * the merged twin         -- gguf_adapters.served_deltas(projections=True)
                               -> ggufio.edit_tensors

then three serves at ONE pinned config (base, adapter-on-base at scale 1.0, and
the merged twin), the same harmful probe rows, greedy decoding, and per-row
refusal-SHAPE verdicts. The adapter load is confirmed through GET
/lora-adapters and a blast check (scale 0 vs 50 must change the text).

METHODOLOGY R8 is the reason this command exists: on this artifact the adapter
curve is not the merged bytes. The JSON records both delta_br values with their
measurements, the merged/adapter ratio, and the ranking-validity flag. The
adapter curve is a screening number; it is never the edit's effect.

METHODOLOGY R9 is why every arm's full llama-server argv is in the record. The
shape verdicts are probe-level (4 rows) unless the caller passes a wider bank;
the record says which. Nothing here claims a non-GGUF runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from jdewitt.answers import atomic_json
from jdewitt.board import coherence_flags
from jdewitt.gguf_adapters import (discover_writeback_targets, gguf_weight_shapes,
                                   inferred_hidden_width, serve_factors, served_deltas,
                                   write_adapter)
from jdewitt.ggufio import DequantBases, compare_edit, dequantize_tensor, edit_tensors
from jdewitt.ppl import ln
from jdewitt.refusal_shape import classify
from jdewitt.screen import Server, screen_arm
from jdewitt.spanfit import load_records, per_layer_axes
from jdewitt.tune import blast_server

DEFAULT_ARTIFACT = '/mnt/ssd/models/Spark-X2.6-1.7B-Q4_K_M.gguf'
DEFAULT_RECORDS = '/mnt/ssd/hermes/cache/scratch/capture-spark-mixed.pt'
DEFAULT_SUITE = '/mnt/ssd/hermes/cache/scratch/capture-spark-mixed.jsonl'
DEFAULT_SERVE = '/mnt/ssd/b70-code/verify-release/llama-b70-vulkan-ubuntu24-x64/llama-server'
DEFAULT_PPL = '/mnt/ssd/b70-code/verify-release/llama-b70-vulkan-ubuntu24-x64/llama-perplexity'
RANKING_RATIO_FLOOR = 0.7
PPL_RE = re.compile(r'PPL = ([0-9.]+)(?: \+/- ([0-9.]+))?')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def say(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', file=sys.stderr, flush=True)


def probe_rows(records, suite, families=4):
    """One harmful row per family, in sorted family order -- the board's probe shape."""
    prompts = {}
    for line in Path(suite).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            prompts[row['id']] = row
    grouped = {}
    for meta, _residual in records:
        if meta.get('kind') != 'refusal' or meta.get('label') == 'deliver':
            continue
        family = str(meta.get('family_id') or 'unfamilied')
        grouped.setdefault(family, []).append(meta['id'])
    eligible = sorted(name for name, ids in grouped.items() if ids)
    if len(eligible) < families:
        raise ValueError(f'records hold {len(eligible)} harmful families, probe asked for {families}')
    rows = []
    for family in eligible[:families]:
        row_id = grouped[family][0]
        if row_id not in prompts:
            raise ValueError(f'{row_id} is in the records but not in {suite}')
        rows.append({'id': row_id, 'family': family, 'prompt': prompts[row_id]['prompt']})
    return rows


def fit_axes(records, layers):
    axes = per_layer_axes(records)
    directions = {layer: np.asarray(axes[layer].float().cpu(), dtype=np.float32) for layer in layers}
    for layer, axis in directions.items():
        norm = float(np.linalg.norm(axis))
        if abs(norm - 1.0) > 1e-4:
            raise ValueError(f'fitted axis for layer {layer} has norm {norm}, not 1: it is not a direction')
    return directions


def build_adapter(artifact, directions, names_by_layer, out_path, base_sha, cache, arch):
    from gguf import GGUFReader

    reader = GGUFReader(str(artifact))
    pairs = []
    for layer in sorted(names_by_layer):
        for name in names_by_layer[layer]:
            weight = torch.as_tensor(dequantize_tensor(artifact, name, reader=reader,
                                                       dequant_cache=cache)).float()
            direction = torch.as_tensor(directions[layer]).reshape(1, -1)
            a, b = serve_factors(direction, weight, 1.0)
            pairs.append((name, a, b))
            del weight
    report = write_adapter(out_path, pairs, arch, base_sha256=base_sha)
    return report


def build_twin(artifact, directions, names_by_layer, out_path, cache):
    projections = {}
    for layer in sorted(names_by_layer):
        names = names_by_layer[layer]
        projections.update(served_deltas(artifact, directions[layer], names, 1.0,
                                         as_projections=True))
    report = edit_tensors(artifact, out_path, projections=projections, scale=1.0,
                          dequant_cache=cache)
    report['projections'] = {name: {'axis_width': int(axis.shape[-1]), 'space': space}
                             for name, (axis, space) in projections.items()}
    return report


def tensor_diff(base, twin):
    """Which tensors the merge changed, and proof nothing else moved."""
    from gguf import GGUFReader

    left = GGUFReader(str(base))
    right = GGUFReader(str(twin))
    by_name = {tensor.name: tensor for tensor in right.tensors}
    changed, missing, shape_changed = [], [], []
    for tensor in left.tensors:
        other = by_name.get(tensor.name)
        if other is None:
            missing.append(tensor.name)
            continue
        if tuple(tensor.shape) != tuple(other.shape) or tensor.tensor_type != other.tensor_type:
            shape_changed.append(tensor.name)
            continue
        with open(base, 'rb') as handle:
            handle.seek(int(tensor.data_offset))
            first = handle.read(int(tensor.n_bytes))
        with open(twin, 'rb') as handle:
            handle.seek(int(other.data_offset))
            second = handle.read(int(other.n_bytes))
        if first != second:
            changed.append(tensor.name)
    extra = sorted(set(by_name) - {tensor.name for tensor in left.tensors})
    return {'shared': len(by_name) - len(extra), 'changed': sorted(changed),
            'changed_count': len(changed), 'unchanged_count': len(by_name) - len(extra) - len(changed),
            'missing_in_twin': missing, 'shape_or_type_changed': shape_changed,
            'extra_in_twin': extra}


def read_ppl(binary, model, corpus, ctx, threads, ngl, device, lora=None, scale=None, timeout=1800):
    argv = [str(binary), '-m', str(model), '-f', str(corpus), '-ngl', str(ngl),
            '--device', device, '-c', str(ctx), '-t', str(threads)]
    if lora:
        argv += ['--lora-scaled', f'{lora}:{float(scale):g}']
    started = time.monotonic()
    done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    output = done.stdout + done.stderr
    match = PPL_RE.search(output)
    if match is None:
        return {'error': 'no PPL line', 'returncode': done.returncode,
                'tail': output[-800:], 'argv': argv}
    return {'ppl': float(match.group(1)),
            'uncertainty': float(match.group(2)) if match.group(2) else None,
            'seconds': round(time.monotonic() - started, 1), 'argv': argv}


def derive_corpora(records, suite, out_dir, chars=4000):
    pieces = {'refusal': [], 'delivery': [], 'control': []}
    suite_prompts = {}
    for line in Path(suite).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            suite_prompts[row['id']] = row['prompt']
    for meta, _residual in records:
        prompt = suite_prompts.get(meta.get('id'))
        if prompt:
            pieces['control'].append(prompt)
        answer = (meta.get('answer') or '').strip()
        if not answer:
            continue
        if meta.get('label') == 'bad':
            pieces['refusal'].append(answer)
        elif meta.get('label') == 'good':
            pieces['delivery'].append(answer)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpora = {}
    for name, chunks in pieces.items():
        text = ''
        for chunk in chunks:
            text += chunk.replace('\n', ' ').strip() + '\n'
            if len(text) >= chars:
                break
        text = text[:chars]
        if len(text) < 400:
            raise ValueError(f'derived {name} corpus is only {len(text)} chars: perplexity would be noise')
        path = out_dir / f'ppl-{name}.txt'
        path.write_text(text)
        corpora[name] = {'path': str(path), 'chars': len(text), 'sha256': sha256_file(path)}
    return corpora


class PinnedServer(Server):
    """A screen.Server with the device pinned and no OOM downgrade.

    ``Server.start`` halves ngl on a CUDA OOM and would then quietly score an
    offloaded-poorly artifact. A runtime-equivalence card cannot degrade its own
    config, so an OOM here is a hard failure.
    """

    def __init__(self, *args, device='Vulkan1', verbosity=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.verbosity = verbosity

    def _command(self):
        base = super()._command()
        extra = ['--device', self.device, '--split-mode', 'none', '--jinja', '-lv', str(self.verbosity)]
        marker = base.index('--lora') if '--lora' in base else len(base)
        return base[:marker] + extra + base[marker:]

    def start(self):
        return self._start_once()

    def proof(self):
        text = Path(self.log_path).read_text(errors='replace') if self.log_path else ''
        return {
            'device_line': next((line.strip() for line in text.splitlines()
                                 if 'using device' in line and re.search(r'Vulkan\d+', line)), None),
            'offload_lines': sorted(set(re.findall(r'offloaded \d+/\d+ layers to GPU', text))),
            'lora_buffer_lines': sorted(set(line.strip() for line in text.splitlines()
                                            if 'LoRA buffer size' in line))[:4],
            'vulkan_model_buffer': next((line.strip() for line in text.splitlines()
                                         if 'model buffer size' in line and 'Vulkan' in line), None),
            'props': self.props,
            'argv': self._command(),
            'load_seconds': self.load_seconds,
        }


def render_prompt(server, text, system='You are a helpful assistant.'):
    payload = {'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': text}],
               'add_generation_prompt': True, 'chat_template_kwargs': {'enable_thinking': False}}
    response = server._post('/apply-template', payload, timeout=120)
    rendered = response.get('prompt') if isinstance(response, dict) else None
    if not isinstance(rendered, str) or not rendered.strip():
        raise RuntimeError('llama-server /apply-template returned no prompt')
    return rendered


def arm_rows(server, rows, k, adapter_id=None, scale=None):
    result = screen_arm(server, rows, k, adapter_id=adapter_id, scale=scale)
    detail = []
    for row in result['detail']:
        shape = classify(row['text'])
        detail.append({'id': row['id'], 'tokens': row['tokens'], 'verdict': row['verdict'],
                       'shape': shape['shape'], 'declining': shape['declining'],
                       'procedural': shape['procedural'], 'steps': shape['steps'],
                       'structure': shape['structure'], 'chars': len(row['text']),
                       'coherence_flags': coherence_flags(row['text']),
                       'text': row['text']})
    return {'verdicts': result['verdicts'], 'tokens': result['tokens'], 'rows': detail}


def serve_arm(artifact, serve, device, ctx, ngl, k, rows, adapter=None, adapter_scale=1.0,
              log_path=None, blast=False):
    with PinnedServer(serve, artifact, adapters=[adapter] if adapter else (), ctx=ctx, ngl=ngl,
                      lib_dirs=(), log_path=log_path) as server:
        load = None
        rendered = [dict(row) for row in rows]
        if adapter:
            server.set_scales({0: adapter_scale})
            load = server._get('/lora-adapters')
            entry = next((item for item in load or [] if item.get('id') == 0), None)
            if entry is None or abs(float(entry.get('scale', 0.0)) - adapter_scale) > 1e-6:
                raise RuntimeError(f'server does not apply adapter 0 at scale {adapter_scale}: {load}')
        for row in rendered:
            row['rendered'] = render_prompt(server, row['prompt'])
            row['prompt'] = row['rendered']
        arm = arm_rows(server, rendered, k)
        rendered_sha256 = [sha256_bytes(row['rendered'].encode()) for row in rendered]
        proof = server.proof()
        blast_report = None
        if blast:
            blast_report = blast_server(server, rendered, 0, 32, {})
            # Restore the declared scale before the row pass is reported.
            server.set_scales({0: adapter_scale})
            load = server._get('/lora-adapters')
    if blast_report is not None and blast_report['identical']:
        raise RuntimeError('adapter blast check: scale 0 and scale 50 produced identical text; '
                           'the adapter is not applied and no arm verdict is trustworthy')
    return {'load': load, 'arm': arm, 'proof': proof, 'blast': blast_report,
            'rendered_sha256': rendered_sha256}


def counts_by_verdict(rows, arm):
    counts = {}
    for row in rows:
        verdict = ((row.get(arm) or {}).get('agent_read') or {}).get('verdict')
        if verdict:
            counts[verdict] = counts.get(verdict, 0) + 1
    return counts


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', default=DEFAULT_ARTIFACT)
    parser.add_argument('--records', default=DEFAULT_RECORDS)
    parser.add_argument('--suite', default=DEFAULT_SUITE)
    parser.add_argument('--serve', default=DEFAULT_SERVE)
    parser.add_argument('--ppl', default=DEFAULT_PPL)
    parser.add_argument('--out', required=True)
    parser.add_argument('--workdir', required=True)
    parser.add_argument('--layer-first', type=int, default=10)
    parser.add_argument('--layer-last', type=int, default=27)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--rank', type=int, default=1)
    parser.add_argument('--families', type=int, default=4)
    parser.add_argument('--tokens', type=int, default=200)
    parser.add_argument('--ctx', type=int, default=4096)
    parser.add_argument('--ngl', type=int, default=999)
    parser.add_argument('--device', default='Vulkan1')
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--ppl-ctx', type=int, default=256)
    parser.add_argument('--agent-verdicts', default=None,
                        help='JSON {row_id: {verdict, note}} authored by the agent after reading the '
                             'generations; embeds them and computes the agreement field')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.rank != 1:
        raise SystemExit('this card pins the rank-1 plan; pass --rank 1')
    for path in (args.artifact, args.records, args.suite, args.serve, args.ppl):
        if not Path(path).is_file():
            raise SystemExit(f'missing input: {path}')
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    base_sha = sha256_file(args.artifact)
    raw_records = torch.load(args.records, map_location='cpu', weights_only=False)
    capture = raw_records.get('capture', {}) if isinstance(raw_records, dict) else {}
    records = load_records(args.records)
    records_sha = sha256_file(args.records)
    suite_sha = sha256_file(args.suite)
    recorded_suite_sha = capture.get('suite_sha256')
    if recorded_suite_sha and recorded_suite_sha != suite_sha:
        raise SystemExit(f'{args.suite} has sha256 {suite_sha}, but the capture recorded '
                         f'suite_sha256 {recorded_suite_sha}: the rows are not the captured rows')
    rows = probe_rows(records, args.suite, args.families)

    shapes = gguf_weight_shapes(args.artifact)
    width = inferred_hidden_width(shapes)
    discovered = discover_writeback_targets(shapes, width)
    layers = [layer for layer in sorted(discovered['targets'])
              if args.layer_first <= layer <= args.layer_last]
    names_by_layer = {layer: discovered['targets'][layer] for layer in layers}
    directions = fit_axes(records, layers)
    direction_sha = sha256_bytes(b''.join(directions[layer].tobytes() for layer in layers))
    names = [name for layer in layers for name in names_by_layer[layer]]

    from gguf import GGUFReader
    reader = GGUFReader(str(args.artifact))
    arch = str(reader.fields['general.architecture'].contents())
    tensor_count = len(reader.tensors)
    cache = DequantBases(work / 'dequant')

    plan = {'shape': 'perlayer', 'footprint': f'{args.layer_first}-{args.layer_last}',
            'rank': args.rank, 'scale': args.scale, 'basis': 'artifact',
            'layers': layers, 'tensors': len(names), 'tensor_names': names,
            'width': width, 'directions_sha256': direction_sha,
            'records': str(args.records), 'records_sha256': records_sha,
            'fit': 'spanfit.per_layer_axes over the capture records',
            'profile': 'flat'}
    say(f'plan perlayer|{args.layer_first}-{args.layer_last} rank {args.rank}: {len(names)} tensors, '
        f'directions {direction_sha[:12]}')

    adapter_path = work / 'adapter-rank1.gguf'
    adapter_meta = work / 'adapter-meta.json'
    adapter_identity = {'base_sha256': base_sha, 'plan': plan}
    if adapter_path.is_file() and adapter_meta.is_file() and json.loads(adapter_meta.read_text()) == adapter_identity:
        adapter_report = {'path': str(adapter_path), 'bytes': adapter_path.stat().st_size,
                          'tensors': len(names), 'ranks': [1], 'cached': True}
    else:
        say('building adapter from dequantized served weights ...')
        adapter_report = build_adapter(args.artifact, directions, names_by_layer, adapter_path,
                                       base_sha, cache, arch)
        atomic_json(adapter_meta, adapter_identity)
    adapter_sha = sha256_file(adapter_path)
    say(f'adapter {adapter_sha[:12]} {adapter_report["bytes"]} bytes')

    twin_path = work / 'merged-twin.gguf'
    twin_meta = work / 'twin-meta.json'
    twin_identity = {'base_sha256': base_sha, 'adapter_sha256': adapter_sha, 'plan': plan}
    if not (twin_path.is_file() and twin_meta.is_file()
            and json.loads(twin_meta.read_text()) == twin_identity):
        if twin_path.is_file():
            twin_path.unlink()
        say('building merged twin (served_deltas projections -> edit_tensors) ...')
        build_twin(args.artifact, directions, names_by_layer, twin_path, cache)
        atomic_json(twin_meta, twin_identity)
    twin_sha = sha256_file(twin_path)
    say(f'twin {twin_sha[:12]} {twin_path.stat().st_size} bytes')
    diff = tensor_diff(args.artifact, twin_path)
    if diff['changed_count'] != len(names):
        raise RuntimeError(f'twin changed {diff["changed_count"]} tensors, plan named {len(names)}')

    edits = {layer_name: (directions[layer], 'output')
             for layer in layers for layer_name in names_by_layer[layer]}
    fidelity_cache = work / 'fidelity.json'
    if fidelity_cache.is_file() and json.loads(fidelity_cache.read_text()).get('twin_sha256') == twin_sha:
        fidelity = json.loads(fidelity_cache.read_text())['fidelity']
    else:
        say('measuring per-tensor delivered cos/SNR ...')
        fidelity = compare_edit(args.artifact, twin_path, edits, names=names)
        atomic_json(fidelity_cache, {'twin_sha256': twin_sha, 'fidelity': fidelity})
    cosines = sorted(row['cos'] for row in fidelity['tensors'])
    snrs = sorted(row['snr'] for row in fidelity['tensors'])
    delivered = {'tensors': fidelity['tensors'], 'sampled': fidelity['sampled'],
                 'min_cos': cosines[0], 'median_cos': statistics.median(cosines), 'max_cos': cosines[-1],
                 'min_snr': snrs[0], 'median_snr': statistics.median(snrs), 'max_snr': snrs[-1],
                 'instrument': 'ggufio.compare_edit on the merged twin vs the pristine base (full-tensor decode)'}

    corpora = derive_corpora(records, args.suite, work / 'corpora')
    ppl_identity = {'base_sha256': base_sha, 'adapter_sha256': adapter_sha, 'twin_sha256': twin_sha,
                    'corpora': {name: item['sha256'] for name, item in corpora.items()},
                    'config': {'binary': str(args.ppl), 'ctx': args.ppl_ctx, 'threads': args.threads,
                               'ngl': args.ngl, 'device': args.device}}
    ppl_cache = work / 'ppl.json'
    if ppl_cache.is_file() and json.loads(ppl_cache.read_text()).get('identity') == ppl_identity:
        readings = json.loads(ppl_cache.read_text())['readings']
    else:
        readings = {}
        for arm, model, lora in (('base', args.artifact, None),
                                 ('adapter', args.artifact, adapter_path),
                                 ('merged', twin_path, None)):
            readings[arm] = {}
            for name, item in corpora.items():
                say(f'ppl {arm}/{name} ...')
                readings[arm][name] = read_ppl(args.ppl, model, item['path'], args.ppl_ctx,
                                               args.threads, args.ngl, args.device,
                                               lora=lora, scale=args.scale if lora else None)
        atomic_json(ppl_cache, {'identity': ppl_identity, 'readings': readings})

    def branch(readings_arm):
        for name in ('refusal', 'delivery'):
            if 'ppl' not in readings_arm[name]:
                raise SystemExit(f'perplexity reading {name} failed: {readings_arm[name]}')
        return ln(readings_arm['refusal']['ppl']) - ln(readings_arm['delivery']['ppl'])

    base_br = branch(readings['base'])
    adapter_br = branch(readings['adapter'])
    merged_br = branch(readings['merged'])
    adapter_delta_br = round(adapter_br - base_br, 4)
    merged_delta_br = round(merged_br - base_br, 4)
    ratio = (round(merged_delta_br / adapter_delta_br, 4)
             if adapter_delta_br not in (0, 0.0) and adapter_delta_br > 0 else None)
    ranking_valid = bool(ratio is not None and ratio >= RANKING_RATIO_FLOOR)

    serve_cache = work / 'serve.json'
    arm_identity = {'base_sha256': base_sha, 'adapter_sha256': adapter_sha, 'twin_sha256': twin_sha,
                    'rows': suite_sha, 'tokens': args.tokens, 'ctx': args.ctx, 'ngl': args.ngl,
                    'device': args.device}
    if serve_cache.is_file() and json.loads(serve_cache.read_text()).get('identity') == arm_identity:
        serves = json.loads(serve_cache.read_text())['serves']
    else:
        serves = {}
        say('serving base ...')
        serves['base'] = serve_arm(args.artifact, args.serve, args.device, args.ctx, args.ngl,
                                   args.tokens, rows, log_path=work / 'base-server.log')
        say('serving adapter at scale 1.0 (+ blast check) ...')
        serves['adapter'] = serve_arm(args.artifact, args.serve, args.device, args.ctx, args.ngl,
                                      args.tokens, rows, adapter=str(adapter_path),
                                      adapter_scale=args.scale, log_path=work / 'adapter-server.log',
                                      blast=True)
        say('serving merged twin ...')
        serves['merged'] = serve_arm(twin_path, args.serve, args.device, args.ctx, args.ngl,
                                     args.tokens, rows, log_path=work / 'merged-server.log')
        atomic_json(serve_cache, {'identity': arm_identity, 'serves': serves})

    verdicts = {arm: {row['id']: row for row in serves[arm]['arm']['rows']} for arm in serves}
    agent = None
    if args.agent_verdicts:
        agent = json.loads(Path(args.agent_verdicts).read_text())

    per_row = []
    agree = True
    for source in rows:
        row_id = source['id']
        entry = {'id': row_id, 'family': source['family'], 'prompt': source['prompt']}
        for arm in ('base', 'adapter', 'merged'):
            reading = verdicts[arm][row_id]
            entry[arm] = {key: reading[key] for key in
                          ('verdict', 'shape', 'declining', 'procedural', 'steps', 'structure',
                           'chars', 'tokens', 'coherence_flags')}
            entry[arm]['text'] = reading['text']
        if agent is not None:
            for arm in ('base', 'adapter', 'merged'):
                authored = (agent.get(row_id) or {}).get(arm)
                if not authored:
                    raise SystemExit(f'--agent-verdicts has no verdict for {row_id}/{arm}')
                entry[arm]['agent_read'] = authored
            same = entry['adapter']['agent_read']['verdict'] == entry['merged']['agent_read']['verdict']
            entry['adapter_merged_agent_read_agree'] = same
            agree = agree and same
        per_row.append(entry)

    document = {
        'version': 1,
        'kind': 'p6-runtime-equivalence',
        'generated_by': 'scripts/p6_runtime_equivalence.py',
        'task': 't_beb272db',
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'capture': capture,
        'rows_source': {'records': str(args.records), 'records_sha256': records_sha,
                        'suite': str(args.suite), 'suite_sha256': suite_sha,
                        'suite_matches_capture': bool(recorded_suite_sha == suite_sha)},
        'claim': ('rank-1 GGUF LoRA adapter and merged twin built from ONE fitted plan on the same base '
                  'bytes, served at ONE pinned config on the B70; adapter curve is a screening number '
                  '(R8), the merged twin is the shipped shape'),
        'claim_limit': ('probe-level: 4 harmful rows (one per family) at a 200-token greedy window. A '
                        'probe-only pass is not a certification; no full certification wave was run. '
                        'No non-GGUF runtime (vLLM/SGLang) is claimed.'),
        'base': {'path': str(args.artifact), 'sha256': base_sha,
                 'bytes': Path(args.artifact).stat().st_size,
                 'arch': arch, 'tensor_count': tensor_count,
                 'two_dimensional_tensors': len(shapes)},
        'plan': plan,
        'adapter': {**adapter_report, 'sha256': adapter_sha, 'rank': 1, 'scale': args.scale,
                    'base_artifact_sha256': base_sha,
                    'build': 'gguf_adapters.write_adapter(base_sha256=base_sha)'},
        'twin': {'path': str(twin_path), 'sha256': twin_sha,
                 'bytes': twin_path.stat().st_size, 'edited_tensors': diff['changed_count'],
                 'untouched_tensors': diff['unchanged_count'], 'tensor_diff': diff,
                 'delivered': delivered,
                 'build': 'gguf_adapters.served_deltas(as_projections=True) -> ggufio.edit_tensors'},
        'behaviour': {
            'corpora': corpora,
            'ppl_config': {'binary': str(args.ppl), 'ctx': args.ppl_ctx, 'threads': args.threads,
                           'ngl': args.ngl, 'device': args.device},
            'readings': readings,
            'base_br': round(base_br, 4),
            'adapter_curve': {'delta_br': adapter_delta_br, 'measured_on': 'adapter',
                              'br': round(adapter_br, 4)},
            'merged_bytes': {'delta_br': merged_delta_br, 'measured_on': 'merged-bytes',
                             'br': round(merged_br, 4), 'source_artifact_sha256': twin_sha},
            'delta_br_definition': 'br = ln PPL(refusal corpus) - ln PPL(delivery corpus); '
                                   'delta_br = br(arm) - br(base)',
        },
        'adapter_delta_br': adapter_delta_br,
        'merged_delta_br': merged_delta_br,
        'adapter_to_merged_ratio': ratio,
        'ranking_valid_on_this_artifact': ranking_valid,
        'ranking_validity_ratio_floor': RANKING_RATIO_FLOOR,
        'merged_effect_status': 'MEASURED',
        'curve_is_not_merged_bytes': True,
        'r8_note': ('The adapter curve ranks; the merged bytes ship. adapter_to_merged_ratio is '
                    'merged_delta_br/adapter_delta_br; below the floor the adapter overstated the edit '
                    'and the adapter is screening-only on this artifact.'),
        'serve': {
            'config': {'ctx': args.ctx, 'parallel': 1, 'ngl': args.ngl, 'device': args.device,
                       'split_mode': 'none', 'tokens_per_row': args.tokens,
                       'sampling': {'temperature': 0.0, 'top_k': 1, 'top_p': 1.0,
                                    'cache_prompt': False},
                       'binary': str(args.serve), 'binary_version': server_version(args.serve)},
            'arms': {arm: {'artifact': str(args.artifact) if arm != 'merged' else str(twin_path),
                           'adapter': str(adapter_path) if arm == 'adapter' else None,
                           'argv': serves[arm]['proof']['argv'],
                           'rendered_prompt_sha256': serves[arm]['rendered_sha256'],
                           'runtime_proof': serves[arm]['proof'],
                           'lora_adapters': serves[arm]['load'],
                           'blast': serves[arm]['blast']}
                     for arm in ('base', 'adapter', 'merged')},
        },
        'rows': per_row,
        'agent_read': agent,
        'agreement': ({'adapter_merged_agree_on_probe_rows': agree,
                       'rows': len(per_row),
                       'rows_agreeing': sum(1 for row in per_row
                                            if row.get('adapter_merged_agent_read_agree')),
                       'rows_disagreeing': sum(1 for row in per_row
                                               if row.get('adapter_merged_agent_read_agree') is False),
                       'verdict_counts': {arm: counts_by_verdict(per_row, arm)
                                          for arm in ('base', 'adapter', 'merged')},
                       'statement': ('adapter-on-quant and merged twin agree on every probe row'
                                     if agree else
                                     'adapter-on-quant and merged twin disagree on at least one probe row; '
                                     'this is the measured R8 divergence, not an average')}
                      if agent is not None else
                      {'adapter_merged_agree_on_probe_rows': None,
                       'statement': 'awaiting the agent read of the generations'}),
        'release_label': ('screening-only adapter' if not ranking_valid else 'adapter ranks merged on this artifact'),
    }
    atomic_json(args.out, document)
    print(json.dumps({'out': str(args.out), 'adapter_sha256': adapter_sha, 'twin_sha256': twin_sha,
                      'adapter_delta_br': adapter_delta_br, 'merged_delta_br': merged_delta_br,
                      'adapter_to_merged_ratio': ratio,
                      'ranking_valid': ranking_valid,
                      'min_cos': delivered['min_cos'], 'median_cos': delivered['median_cos']},
                     indent=2))
    return 0


def server_version(binary):
    try:
        done = subprocess.run([str(binary), '--version'], capture_output=True, text=True, timeout=30)
        return (done.stdout + done.stderr).strip().splitlines()[0] if (done.stdout or done.stderr) else None
    except OSError:
        return None


if __name__ == '__main__':
    sys.exit(main())
