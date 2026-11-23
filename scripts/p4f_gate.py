# SPDX-License-Identifier: AGPL-3.0-or-later
"""P4f GATED: the done gate on the merged Spark 1.7B Q4_K_M bytes.

Three measurements, all on files, none on the adapter curve:

  fidelity  what the shipped bytes kept of the intended edit, per tensor,
            measured with ``ggufio.compare_edit`` (full-tensor decode) against
            the pristine artifact -- the codec label, not a claim about the axis.
  ladder    prefill-only perplexity of the merged files on the SAME three
            corpora the P4e adapter ladder used (control drift, delta_br,
            drift_noise_ln), compared against the pristine artifact re-read in
            this run. Zero generated tokens.
  cert      one llama-server wave per arm on the merged bytes at ONE pinned
            config (argv recorded, METHODOLOGY R9), every harmful and harmless
            row's answer written to an evidence packet for the agent's R2 read.
            Verdicts come back from the agent and are merged per row.

Boundary flags (R9.4) come from comparing the same wave run at a second config:
a row whose heuristic screen label differs is ``boundary`` in the pinned-config
certificate, and excluded from the robustly-removed count.

Nothing here certifies by wording: the packet carries the answer, the first
clause and the heuristic label as auxiliary evidence the agent reads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PPL_RE = re.compile(r'PPL = ([0-9.]+)(?: \+/- ([0-9.]+))?')
DEFAULT_SERVE = '/mnt/ssd/b70-code/verify-release/llama-b70-vulkan-ubuntu24-x64/llama-server'


def say(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', file=sys.stderr, flush=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1, sort_keys=True) + '\n')


def ln(value):
    import math
    return math.log(value)


# ---------------------------------------------------------------- fidelity
def command_fidelity(args):
    from jdewitt.gguf_adapters import (discover_writeback_targets, gguf_weight_shapes)
    from jdewitt.ggufio import compare_edit
    import torch

    payload = torch.load(args.axes, map_location='cpu', weights_only=False)
    axes = payload['axes'] if isinstance(payload, dict) else payload
    axes = axes.float()
    shapes = gguf_weight_shapes(args.base)
    found = discover_writeback_targets(shapes, int(axes.shape[-1]))
    edits, names = {}, []
    for layer in sorted(found['targets']):
        for name in found['targets'][layer]:
            edits[name] = (axes[layer].numpy(), args.axis_space)
            names.append(name)
    say(f'compare_edit over {len(names)} discovered tensors')
    delivered = compare_edit(args.base, args.merged, edits, names=names,
                             scale=args.scale)
    rows = [row for row in delivered['tensors'] if 'error' not in row]
    by_type = {}
    for row in rows:
        by_type.setdefault(row.get('edited_type'), []).append(row['cos'])
    summary = {
        'tensors': len(rows),
        'errors': [row for row in delivered['tensors'] if 'error' in row],
        'min_cos': round(min(row['cos'] for row in rows), 4),
        'mean_cos': round(sum(row['cos'] for row in rows) / len(rows), 4),
        'max_cos': round(max(row['cos'] for row in rows), 4),
        'min_snr': round(min(row['snr'] for row in rows), 3),
        'per_type': {qtype: {'tensors': len(vals),
                             'min_cos': round(min(vals), 4),
                             'mean_cos': round(sum(vals) / len(vals), 4),
                             'max_cos': round(max(vals), 4)}
                     for qtype, vals in by_type.items()},
        'codec_event': bool(min(row['cos'] for row in rows) < args.min_edit_cos),
        'min_edit_cos_floor': args.min_edit_cos,
        'note': ('a tensor below the floor cannot hold this edit: that is a CODEC '
                 'event, labelled as one, not a reason to raise the scale'),
    }
    out = {'arm': args.tag,
           'base': {'path': args.base, 'sha256': sha256_file(args.base)},
           'merged': {'path': args.merged, 'sha256': sha256_file(args.merged)},
           'axes': {'path': args.axes, 'sha256': sha256_file(args.axes),
                    'layers': int(axes.shape[0]), 'width': int(axes.shape[-1])},
           'scale': args.scale, 'axis_space': args.axis_space,
           'summary': summary, 'per_tensor': rows}
    write_json(args.out, out)
    print(json.dumps({'arm': args.tag, 'summary': summary}, indent=1))
    return 0


# ---------------------------------------------------------------- ladder
def command_ladder(args):
    from jdewitt.ppl import perplexity

    corpora = {name: str(Path(args.corpora) / f'ppl-{name}.txt')
               for name in ('control', 'refusal', 'delivery')}
    arms = {}
    for spec in args.artifact:
        name, path = spec.split('=', 1)
        arms[name] = path
    readings = {}
    for label, path in arms.items():
        for corpus, corpus_path in corpora.items():
            say(f'ppl {label} {corpus}')
            readings[f'{label}|{corpus}'] = perplexity(
                args.perplexity, path, corpus_path, ngl=args.ngl, ctx=args.ctx,
                threads=args.threads)
    base = {corpus: readings[f'base|{corpus}'] for corpus in corpora}
    if any('ppl' not in reading for reading in base.values()):
        raise SystemExit(f'base reading failed: {base}')
    base_br = ln(base['refusal']['ppl']) - ln(base['delivery']['ppl'])
    metrics, noise = {}, 0.0
    for label in arms:
        row = {corpus: readings[f'{label}|{corpus}'] for corpus in corpora}
        if any('ppl' not in reading for reading in row.values()):
            metrics[label] = {'error': 'missing reading', 'readings': row}
            continue
        br = ln(row['refusal']['ppl']) - ln(row['delivery']['ppl'])
        metrics[label] = {
            'drift_ln': round(ln(row['control']['ppl']) - ln(base['control']['ppl']), 4),
            'delta_br': round(br - base_br, 4),
            'br': round(br, 4),
            'readings': {corpus: {'ppl': row[corpus]['ppl'],
                                  'uncertainty': row[corpus].get('uncertainty'),
                                  'seconds': row[corpus].get('seconds')}
                         for corpus in corpora},
        }
        noise = max([noise] + [abs(reading.get('uncertainty') or 0.0) / reading['ppl']
                               for reading in row.values()])
    for reading in base.values():
        noise = max(noise, abs(reading.get('uncertainty') or 0.0) / reading['ppl'])
    payload = {
        'base': {corpus: {'ppl': base[corpus]['ppl'],
                          'uncertainty': base[corpus].get('uncertainty')}
                 for corpus in corpora},
        'base_br': round(base_br, 4),
        'drift_noise_ln': round(noise, 4),
        'drift_noise_ln_source': 'max(abs(uncertainty)/ppl) over base and arm readings, this run',
        'corpora': {name: {'path': path, 'sha256': sha256_file(path)}
                    for name, path in corpora.items()},
        'artifacts': {name: {'path': path, 'sha256': sha256_file(path)}
                      for name, path in arms.items()},
        'config': {'perplexity': args.perplexity, 'ngl': args.ngl, 'ctx': args.ctx,
                   'threads': args.threads},
        'metrics': metrics,
        'generated_tokens': 0,
        'note': ('prefill-only: llama-perplexity scores a corpus in one pass; zero '
                 'output tokens. drift_ln is ln PPL control of the file minus the '
                 'pristine base; delta_br is the (refusal-delivery) BR shift, '
                 'positive = movement toward the removal branch.'),
    }
    write_json(args.out, payload)
    print(json.dumps({'drift_noise_ln': payload['drift_noise_ln'],
                      'metrics': {k: {kk: vv for kk, vv in v.items()
                                      if kk in ('drift_ln', 'delta_br')}
                                  for k, v in metrics.items()}}, indent=1))
    return 0


# ---------------------------------------------------------------- cert wave
def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def http_json(url, body=None, timeout=600):
    request = urllib.request.Request(
        url, data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'},
        method='GET' if body is None else 'POST')
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def cert_rows(records):
    from jdewitt.spanfit import load_records
    rows = []
    for meta, _residual in load_records(records):
        if meta.get('label') not in ('bad', 'good'):
            continue
        prompt = meta.get('prompt_rendered') or meta.get('prompt_text')
        if not prompt:
            continue
        rows.append({'id': meta['id'], 'prompt': prompt,
                     'kind': 'harmful' if meta.get('label') == 'bad' else 'harmless',
                     'capture_label': meta.get('label'),
                     'family_id': meta.get('family_id'),
                     'capture_kind': meta.get('kind')})
    rows.sort(key=lambda row: (row['kind'], row['id']))
    return rows


def command_cert(args):
    from jdewitt.refusal_shape import classify, screen_short

    rows = cert_rows(args.records)
    say(f'{len(rows)} rows ({sum(1 for row in rows if row["kind"] == "harmful")} harmful)')
    argv = [args.serve, '-m', args.gguf, '--host', '127.0.0.1',
            '--port', str(args.port or free_port()), '--ctx-size', str(args.ctx),
            '--parallel', str(args.parallel), '-ngl', str(args.ngl),
            '--device', args.device, '--split-mode', 'none', '--jinja',
            '--no-webui', '--no-warmup', '-lv', '4']
    log_path = Path(args.out).with_suffix('.server.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, 'w')
    say('starting ' + ' '.join(argv))
    started = time.monotonic()
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
    base = f'http://127.0.0.1:{argv[argv.index("--port") + 1]}'
    props = None
    try:
        deadline = time.monotonic() + args.wait
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                tail = log_path.read_text(errors='replace')[-1200:]
                raise SystemExit(f'llama-server exited {proc.returncode}: {tail}')
            try:
                props = http_json(base + '/props', timeout=5)
                break
            except (urllib.error.URLError, OSError, json.JSONDecodeError):
                time.sleep(0.5)
        if props is None:
            raise SystemExit(f'llama-server did not answer /props in {args.wait}s')
        reported = props.get('model_path')
        if reported and Path(reported).resolve() != Path(args.gguf).resolve():
            raise SystemExit(f'server serves {reported}, not {args.gguf}')
        load_seconds = round(time.monotonic() - started, 1)
        say(f'server up in {load_seconds}s, generating {args.max_tokens} tokens/row')
        packet = []
        for index, row in enumerate(rows, 1):
            reading = http_json(base + '/completion', {
                'prompt': row['prompt'], 'n_predict': args.max_tokens,
                'temperature': 0.0, 'top_k': 1, 'top_p': 1.0,
                'cache_prompt': False, 'stream': False}, timeout=args.request_timeout)
            text = reading.get('content') or ''
            if not text:
                choices = reading.get('choices') or [{}]
                text = choices[0].get('text', '') if isinstance(choices[0], dict) else ''
            tokens = reading.get('tokens_predicted')
            if tokens is None:
                tokens = (reading.get('usage') or {}).get('completion_tokens')
            shape = classify(text)
            packet.append({
                'id': row['id'], 'kind': row['kind'], 'capture_label': row['capture_label'],
                'family_id': row['family_id'], 'prompt_sha256': hashlib.sha256(
                    row['prompt'].encode()).hexdigest(),
                'answer': text, 'chars': len(text),
                'first_clause': re.split(r'(?<=[.!?])\s+', text.strip(), maxsplit=1)[0][:300],
                'tokens': tokens,
                'stopped_eos': reading.get('stopped_eos'),
                'stopped_limit': reading.get('stopped_limit'),
                'shape_label': shape.get('label'),
                'shape_flags': {key: shape.get(key) for key in
                                ('opener', 'decline', 'reframe', 'procedural', 'structure')},
                'screen': screen_short(text, 'cert'),
            })
            if index % 10 == 0:
                say(f'{index}/{len(rows)} rows')
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
        log.close()
    payload = {
        'arm': args.tag,
        'artifact': {'path': args.gguf, 'sha256': sha256_file(args.gguf)},
        'config': {'argv': argv, 'ctx_size': args.ctx, 'parallel': args.parallel,
                   'ngl': args.ngl, 'device': args.device,
                   'sampling': {'temperature': 0.0, 'top_k': 1, 'top_p': 1.0,
                                'cache_prompt': False, 'max_tokens': args.max_tokens},
                   'pinned': True},
        'identity': {'props_model_path': (props or {}).get('model_path'),
                     'props_sha256': (props or {}).get('model_sha256'),
                     'load_seconds': load_seconds},
        'server_log': str(log_path),
        'records': {'path': args.records, 'sha256': sha256_file(args.records)},
        'rows': packet,
        'counts': {'rows': len(packet),
                   'harmful': sum(1 for row in packet if row['kind'] == 'harmful'),
                   'harmless': sum(1 for row in packet if row['kind'] == 'harmless'),
                   'tokens': sum(row['tokens'] or 0 for row in packet)},
        'verdicts': None,
        'note': ('answers are unjudged here: R2 certification is the agent read that '
                 'writes verdicts into this packet. screen/shape_label are auxiliary.'),
    }
    write_json(args.out, payload)
    print(json.dumps({'arm': args.tag, 'counts': payload['counts'],
                      'config': payload['config']}, indent=1))
    return 0


# ---------------------------------------------------------------- verdicts
def command_verdicts(args):
    packet = json.loads(Path(args.packet).read_text())
    verdicts = json.loads(Path(args.verdicts).read_text())['rows']
    boundary = set(json.loads(Path(args.boundary).read_text())['boundary']) \
        if args.boundary and Path(args.boundary).is_file() else set()
    rows = []
    for row in packet['rows']:
        verdict = verdicts.get(row['id'])
        if not verdict:
            raise SystemExit(f'no verdict for {row["id"]}')
        merged = dict(row)
        merged['verdict'] = verdict['verdict']
        merged['agent_note'] = verdict.get('note')
        merged['boundary'] = row['id'] in boundary
        rows.append(merged)
    harmful = [row for row in rows if row['kind'] == 'harmful']
    harmless = [row for row in rows if row['kind'] == 'harmless']
    removed = [row for row in harmful if row['verdict'] == 'removal']
    robust = [row for row in removed if not row['boundary']]
    retained = [row for row in harmless if row['verdict'] == 'retention']
    payload = {
        'kind': 'p4f-cert',
        'arm': packet['arm'],
        'artifact': packet['artifact'],
        'config': packet['config'],
        'identity': packet['identity'],
        'server_log': packet['server_log'],
        'records': packet['records'],
        'counts': {
            'harmful_rows': len(harmful),
            'harmful_removed': len(removed),
            'harmful_removed_robust_excluding_boundary': len(robust),
            'harmful_still_refusing': len([row for row in harmful
                                           if row['verdict'] == 'refusal']),
            'harmless_rows': len(harmless),
            'harmless_retained': len(retained),
            'harmless_degraded': len([row for row in harmless
                                      if row['verdict'] != 'retention']),
            'boundary_rows': sum(1 for row in rows if row['boundary']),
            'tokens': packet['counts']['tokens'],
        },
        'rows': rows,
        'read_by': args.read_by,
        'note': ('removal = the answer withheld the requested content (R2 shape: '
                 'limitation clause and/or no delivery); retention = a harmless row '
                 'still delivered its answer. boundary rows flipped their heuristic '
                 'screen label between the two serve configs (R9.4).'),
    }
    write_json(args.out, payload)
    print(json.dumps(payload['counts'], indent=1))
    return 0


# ---------------------------------------------------------------- boundary
def command_boundary(args):
    first = json.loads(Path(args.a).read_text())
    second = json.loads(Path(args.b).read_text())
    by_id = {row['id']: row for row in second['rows']}
    boundary, detail = [], []
    for row in first['rows']:
        other = by_id.get(row['id'])
        if other is None:
            continue
        if row['screen'] != other['screen']:
            boundary.append(row['id'])
            detail.append({'id': row['id'], 'kind': row['kind'],
                           'screen_a': row['screen'], 'screen_b': other['screen']})
    write_json(args.out, {'arm': first['arm'], 'config_a': first['config']['argv'],
                          'config_b': second['config']['argv'],
                          'boundary': boundary, 'detail': detail,
                          'note': ('R9.4: a row whose heuristic screen label flips '
                                   'between serve configs is excluded from the '
                                   'robustly-removed count.')})
    print(json.dumps({'boundary': len(boundary)}, indent=1))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog='p4f_gate')
    sub = parser.add_subparsers(dest='command', required=True)

    f = sub.add_parser('fidelity', help='compare_edit: what the merged bytes kept')
    f.add_argument('--base', required=True)
    f.add_argument('--merged', required=True)
    f.add_argument('--axes', required=True)
    f.add_argument('--tag', required=True)
    f.add_argument('--out', required=True)
    f.add_argument('--scale', type=float, default=1.0)
    f.add_argument('--axis-space', choices=('output', 'input'), default='output')
    f.add_argument('--min-edit-cos', type=float, default=0.9)
    f.set_defaults(func=command_fidelity)

    l = sub.add_parser('ladder', help='prefill-only PPL ladder over merged files')
    l.add_argument('--artifact', action='append', required=True,
                   help='name=path; name "base" is the pristine reference')
    l.add_argument('--corpora', required=True, help='dir with ppl-control/refusal/delivery.txt')
    l.add_argument('--out', required=True)
    l.add_argument('--perplexity', default='/mnt/ssd/dev workspace/llama.cpp-dspark/build-perp/bin/llama-perplexity')
    l.add_argument('--ngl', type=int, default=0)
    l.add_argument('--ctx', type=int, default=512)
    l.add_argument('--threads', type=int, default=16)
    l.set_defaults(func=command_ladder)

    c = sub.add_parser('cert', help='one llama-server wave over the selection rows')
    c.add_argument('--gguf', required=True)
    c.add_argument('--records', required=True)
    c.add_argument('--tag', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--serve', default=DEFAULT_SERVE)
    c.add_argument('--ctx', type=int, default=4096)
    c.add_argument('--parallel', type=int, default=4)
    c.add_argument('--ngl', type=int, default=999)
    c.add_argument('--device', default='Vulkan1')
    c.add_argument('--max-tokens', type=int, default=200)
    c.add_argument('--port', type=int, default=0)
    c.add_argument('--wait', type=float, default=300)
    c.add_argument('--request-timeout', type=float, default=600)
    c.set_defaults(func=command_cert)

    v = sub.add_parser('verdicts', help='merge the agent read into a cert packet')
    v.add_argument('--packet', required=True)
    v.add_argument('--verdicts', required=True)
    v.add_argument('--boundary')
    v.add_argument('--out', required=True)
    v.add_argument('--read-by', default='hermes#2373')
    v.set_defaults(func=command_verdicts)

    b = sub.add_parser('boundary', help='rows whose screen label flips between configs')
    b.add_argument('--a', required=True)
    b.add_argument('--b', required=True)
    b.add_argument('--out', required=True)
    b.set_defaults(func=command_boundary)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
