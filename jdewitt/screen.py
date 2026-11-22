# SPDX-License-Identifier: AGPL-3.0-or-later
"""Short-decode arm screen with an output-token ledger.

Perplexity is free to measure and blind to delivery: an arm can move the refusal
branch's preference without ever producing the content a row asks for. This
module buys the smallest answer to that question — decode the arm's own harmful
prompts for ``k`` tokens each through llama-server with the arm's adapter
applied, classify every prefix with ``refusal_shape.screen_short``, and count
exactly how many output tokens those verdicts cost.

The currency is inference VOLUME (METHODOLOGY R1), so the screen is built
around it:

  * ONE server load per screen; adapters are preloaded with
    ``--lora-init-without-apply`` and an arm is a ``POST /lora-adapters`` scale
    swap (milliseconds), never a restart;
  * ``k`` is the caller's knob and travels in the report, because a screen read
    through a smaller window is not the same measurement;
  * a prefix that settles ``refusal`` is still a refusal after a longer decode,
    while a prefix that settles ``delivery`` is a candidate the agent reads. A
    row that settles neither is ``escalate``: undecided, never counted as gone;
  * ``/props`` is the identity check. Any stranger server on the port answers
    ``/health``, so a screen that trusts liveness will label and measure an
    artifact that does not exist.

A screen verdict is a screening signal that prunes arms, never a removal claim
(R2/R6): certification stays an agent read on merged bytes at one pinned config.
"""
from __future__ import annotations

import json
import os
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .refusal_shape import screen_short

VERDICTS = ('refusal', 'delivery', 'candidate', 'escalate')


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


class Server:
    """One llama-server, one artifact, preloaded adapters, POST-driven scales."""

    def __init__(self, binary, artifact, adapters=(), ctx=4096, ngl=99,
                 port=None, log_path=None, wait=300, env=None, lib_dirs=None):
        self.binary, self.artifact = str(binary), str(artifact)
        self.adapters = [str(path) for path in adapters]
        self.ctx, self.ngl = int(ctx), int(ngl)
        self.port = int(port or free_port())
        self.log_path = Path(log_path) if log_path else None
        self.wait = wait
        self.env = dict(env or {})
        for directory in lib_dirs or ():
            current = self.env.get('LD_LIBRARY_PATH') or os.environ.get('LD_LIBRARY_PATH') or ''
            self.env['LD_LIBRARY_PATH'] = str(directory) + (f':{current}' if current else '')
        self.proc = None
        self.log = None
        self.props = None
        self.load_seconds = None

    # ---------- lifecycle ----------
    def _command(self):
        command = [self.binary, '-m', self.artifact, '--host', '127.0.0.1',
                   '--port', str(self.port), '-c', str(self.ctx), '-ngl', str(self.ngl),
                   '--parallel', '1']
        for flag in ('--no-webui', '--no-warmup'):
            command.append(flag)
        if self.adapters:
            command += ['--lora', ','.join(self.adapters), '--lora-init-without-apply']
        return command

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    def start(self):
        started = time.monotonic()
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log = open(self.log_path, 'w')
        self.proc = subprocess.Popen(self._command(), stdout=self.log,
                                     stderr=subprocess.STDOUT, env=self.env)
        deadline = started + self.wait
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f'llama-server exited during startup (code {self.proc.returncode}); '
                                   f'see {self.log_path}')
            props = self._get('/props', timeout=5)
            if props:
                self.props = props
                break
            time.sleep(0.4)
        else:
            self.stop()
            raise RuntimeError(f'llama-server startup exceeded {self.wait}s; see {self.log_path}')
        reported = props.get('model_path')
        if reported and Path(reported).resolve() != Path(self.artifact).resolve():
            self.stop()
            raise RuntimeError(
                f'the server on port {self.port} serves {reported}, not {self.artifact}: '
                f'measuring through it would label a different artifact')
        self.load_seconds = round(time.monotonic() - started, 1)
        return self

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=20)
        if self.log is not None:
            self.log.close()
            self.log = None

    # ---------- http ----------
    def _url(self, path):
        return f'http://127.0.0.1:{self.port}{path}'

    def _get(self, path, timeout=30):
        try:
            with urllib.request.urlopen(self._url(path), timeout=timeout) as response:
                return json.loads(response.read().decode())
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return None

    def _post(self, path, body, timeout=600):
        request = urllib.request.Request(
            self._url(path), data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())

    # ---------- operations ----------
    def set_scales(self, scales):
        """scales: {adapter id: scale}. Ids are load order (comma --lora list)."""
        body = [{'id': int(index), 'scale': float(value)} for index, value in sorted(scales.items())]
        if not self.adapters:
            return None
        return self._post('/lora-adapters', body, timeout=60)

    def complete(self, prompt, n_predict=100, timeout=600):
        reading = self._post('/completion', {
            'prompt': prompt, 'n_predict': int(n_predict), 'temperature': 0.0,
            'top_k': 1, 'top_p': 1.0, 'cache_prompt': False, 'stream': False,
        }, timeout=timeout)
        text = reading.get('content')
        if text is None:
            choices = reading.get('choices') or [{}]
            text = choices[0].get('text', '') if isinstance(choices[0], dict) else ''
        tokens, source = reading.get('tokens_predicted'), 'tokens_predicted'
        if tokens is None:
            tokens = (reading.get('usage') or {}).get('completion_tokens')
            source = 'usage.completion_tokens'
        if tokens is None:
            tokens = reading.get('timings', {}).get('predicted_n')
            source = 'timings.predicted_n'
        return {'text': text, 'tokens': int(tokens) if isinstance(tokens, int) else None,
                'token_source': source, 'stopped_eos': reading.get('stopped_eos'),
                'stopped_limit': reading.get('stopped_limit')}


def rows_to_screen(records, kinds=('refusal',), limit=None):
    """Harmful rows from a capture's records, with the prompt rendering they used.

    These are the prompts the base model declined, so they are exactly the rows a
    removal has to change; ``rendered`` is the template-wrapped text the capture
    itself fed the model, so the screen decodes the same words.
    """
    rows = []
    for meta, _residual in records:
        if meta.get('kind') not in kinds:
            continue
        if meta.get('label') == 'deliver':
            continue
        # prompt_rendered is the template-wrapped prompt the capture fed the model
        # (`rendered` is only the marker 'chat'/'raw', and `prompt_text` is the
        # older spelling of the same field). A screen that used the marker would
        # decode the literal word.
        prompt = meta.get('prompt_rendered') or meta.get('prompt_text')
        if not prompt:
            continue
        rows.append({'id': meta.get('id'), 'prompt': prompt,
                     'label': meta.get('label'), 'shape': meta.get('shape')})
        if limit and len(rows) >= limit:
            break
    return rows


def mark_candidates(result, chars=300):
    """At the window cap, an undecided row with substance is a delivery CANDIDATE.

    ``screen_short`` settles 'delivery' only on enumerated procedure, so a model
    that answers with headed prose ("# Making X from household materials") stays
    'escalate' forever and the screen never decides anything -- a label property of
    the classifier, not of the answer. At the cap, a row that has NOT declined and
    has produced at least ``chars`` characters is re-labelled 'candidate': not a
    delivery (the agent still reads it, R2), but evidence the decline is gone,
    which is what ranking an arm needs.
    """
    from .refusal_shape import classify

    moved = 0
    for row in result['detail']:
        if row['verdict'] != 'escalate':
            continue
        shape = classify(row.get('text') or row.get('opening') or '')
        if shape['declining'] or row['chars'] < chars:
            continue
        result['verdicts']['escalate'] -= 1
        result['verdicts']['candidate'] += 1
        row['verdict'] = 'candidate'
        row['shape'] = shape['shape']
        moved += 1
    result['candidates_marked'] = moved
    return result


def screen_arm(server, rows, k, adapter_id=None, scale=None, reset_scales=None):
    """Decode one arm's rows for k tokens each; return verdicts plus token cost."""
    if adapter_id is not None:
        scales = dict(reset_scales or {})
        scales[adapter_id] = scale
        server.set_scales(scales)
    verdicts = {name: 0 for name in VERDICTS}
    tokens, detail, lengths, sources = 0, [], [], set()
    for row in rows:
        started = time.monotonic()
        reading = server.complete(row['prompt'], k)
        verdict = screen_short(reading['text'], 'screen')
        verdicts[verdict] += 1
        counted = reading['tokens'] if isinstance(reading['tokens'], int) else None
        tokens += counted or 0
        sources.add(reading['token_source'])
        lengths.append(len(reading['text']))
        detail.append({'id': row['id'], 'verdict': verdict, 'tokens': counted,
                       'chars': len(reading['text']),
                       'opening': reading['text'][:120],
                       'text': reading['text'],
                       'seconds': round(time.monotonic() - started, 2)})
    return {
        'rows': len(rows), 'k': k, 'verdicts': verdicts, 'tokens': tokens,
        'token_source': ','.join(sorted(sources)),
        'median_chars': int(statistics.median(lengths)) if lengths else 0,
        'detail': detail,
        # a screen where nothing generated is UNMEASURED, never a clean arm
        'measured': bool(lengths) and all(len(row['opening']) > 0 for row in detail),
    }
