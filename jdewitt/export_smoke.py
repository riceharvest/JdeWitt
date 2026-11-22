# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-artifact export smoke: does llama.cpp serve THESE bytes?

The header is read in-process (no model load), then the exact bytes are handed
to a real llama-server on a free port. The command records what the runtime
did -- /health, /props identity, one greedy chat completion -- and stops there.
One JSON answers the release question "a GGUF llama.cpp cannot serve is not a
release" for one artifact, where scripts/live_gguf_smoke.py proves the whole
convert/quantize/serve pipeline per model by hand.

No behaviour claim follows from a pass: serving means the runtime accepted the
bytes and answered one request, not that refusal rate, retention or removal
were measured.
"""
import hashlib
import json
import os
import re
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

# Bounded waits: a loader that never becomes healthy must fail the verdict, not
# hang the caller. The measured artifacts answer /health in ~1 s; the ceiling is
# for slow disks and cold page cache.
HEALTH_TIMEOUT = 120.0
COMPLETION_TIMEOUT = 180
PROMPT = 'Reply with the single word: OK'

_SCALARS = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?'}
_STRING = 8
_WANTED = ('general.architecture', 'general.name', 'general.file_type',
           'general.quantization_version')


def _kv_value(tag, raw):
    """Decode one metadata value; arrays stay undecoded (only scalars/strings needed)."""
    if tag in _SCALARS:
        return struct.unpack(_SCALARS[tag], raw)[0]
    if tag == _STRING:
        length = struct.unpack('<Q', raw[:8])[0]
        return raw[8:8 + length].decode('utf-8', 'replace')
    return None


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_header(path):
    """Header half: identity, architecture, tensor count and dtype histogram.

    Reads only the metadata/tensor-info region. A truncated or hostile file
    raises here and the caller records header_error, but the server half still
    runs: the loader's own verdict is the one this command exists to print.
    """
    from .ggufio import parse_gguf_header
    from .ggufwrite import tensor_nbytes
    from .quants import type_name_for_id

    path = Path(path)
    parsed = parse_gguf_header(path)
    values = {key.decode('utf-8', 'replace'): _kv_value(tag, raw)
              for key, tag, raw in parsed['entries'] if key.decode('utf-8', 'replace') in _WANTED}
    dtypes = {}
    for info in parsed['infos']:
        try:
            name = type_name_for_id(info['dtype'])
        except ValueError:
            name = f"id-{info['dtype']}"
        dims = list(struct.unpack('<' + 'Q' * info['n_dims'], info['dims']))
        row = dtypes.setdefault(name, {'tensors': 0, 'bytes': 0})
        row['tensors'] += 1
        try:
            row['bytes'] += tensor_nbytes([int(dim) for dim in dims], info['dtype'])
        except ValueError:
            row['bytes_unknown'] = True
    return {
        'path': str(path.resolve()),
        'sha256': _sha256(path),
        'bytes': path.stat().st_size,
        'architecture': str(values.get('general.architecture') or 'unknown'),
        'name': values.get('general.name'),
        'file_type': values.get('general.file_type'),
        'quantization_version': values.get('general.quantization_version'),
        'metadata_entries': len(parsed['entries']),
        'tensors': parsed['n_tensors'],
        'dtypes': dict(sorted(dtypes.items())),
    }


def _free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _loader_words(log_path):
    """The loader's own last error lines, timestamps stripped."""
    try:
        log = Path(log_path).read_text(errors='replace')
    except OSError:
        return 'no server log'
    errors = []
    for line in log.splitlines():
        found = re.match(r'^\d+(?:\.\d+)*\s+[EW]\s+(.*)$', line)
        if found:
            errors.append(found.group(1).strip())
    # llama-server warnings are also logged at E; keep the lines that name a
    # load failure so the reason carries the cause, not boilerplate.
    failures = [line for line in errors
                if any(word in line.lower() for word in
                       ('failed', 'error', 'invalid', 'unsupported', 'unknown',
                        'could not', 'cannot', 'not supported'))]
    if not failures:
        failures = errors
    unique = []
    for line in failures:
        if line not in unique:
            unique.append(line)
    if not unique:
        return 'no loader output'
    # The first error is the underlying cause (a truncated tensor-info region is
    # reported as "failed to read key-value pairs"); the last ones are what the
    # loader concluded. Keep both ends rather than only the exit chatter.
    picked = unique if len(unique) <= 3 else [unique[0], *unique[-2:]]
    return '; '.join(picked)


def _health(process, base_url, log_path):
    deadline = time.monotonic() + HEALTH_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return f'llama-server exited {process.returncode} before /health: {_loader_words(log_path)}'
        try:
            with urllib.request.urlopen(base_url + '/health', timeout=2) as response:
                if response.status == 200:
                    return None
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.1)
    return f'llama-server did not become healthy within {HEALTH_TIMEOUT:.0f}s: {_loader_words(log_path)}'


def _get_json(url, body=None, timeout=10):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def measure(path, server, ctx, max_tokens, server_args, log_path):
    """Server half, recorded verbatim. Returns (argv, props_identity, completion, verdict)."""
    argv = [str(server), '--model', str(Path(path).resolve()), '-ngl', '0',
            '--ctx-size', str(ctx), '--host', '127.0.0.1', '--port', str(_free_port()),
            *server_args]
    port = int(argv[argv.index('--port') + 1])
    base_url = f'http://127.0.0.1:{port}'
    props_identity: dict = {'server_log': str(log_path)}
    completion: dict = {'prompt': PROMPT, 'max_tokens': max_tokens,
                        'temperature': 0, 'seed': 0}
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('LLAMA_', 'HF_TOKEN'))}
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            problem = _health(process, base_url, log_path)
            if problem:
                return argv, props_identity, completion, {'servable': False, 'reason': problem}
            try:
                props = _get_json(base_url + '/props')
            except (urllib.error.URLError, OSError, ValueError) as error:
                return argv, props_identity, completion, {
                    'servable': False,
                    'reason': f'/props failed after /health: {type(error).__name__}: {error}'}
            reported = props.get('model_path')
            resolves = bool(reported) and Path(reported).resolve() == Path(path).resolve()
            props_identity.update({
                'model_path': reported,
                'model_path_resolves_to_artifact': resolves,
                'model_alias': props.get('model_alias'),
                'model_ftype': props.get('model_ftype'),
                'build_info': props.get('build_info'),
                'n_ctx': props.get('default_generation_settings', {}).get('n_ctx'),
                'total_slots': props.get('total_slots'),
            })
            if not resolves:
                return argv, props_identity, completion, {
                    'servable': False,
                    'reason': f'/props model_path {reported!r} does not resolve to the artifact'}
            try:
                answer = _get_json(base_url + '/v1/chat/completions', {
                    'model': props.get('model_alias') or 'smoke',
                    'messages': [{'role': 'user', 'content': PROMPT}],
                    'max_tokens': max_tokens, 'temperature': 0, 'seed': 0,
                }, timeout=COMPLETION_TIMEOUT)
            except (urllib.error.URLError, OSError, ValueError) as error:
                return argv, props_identity, completion, {
                    'servable': False,
                    'reason': f'chat completion failed: {type(error).__name__}: {error}'}
            choice = (answer.get('choices') or [{}])[0]
            completion.update({
                'reply': (choice.get('message') or {}).get('content'),
                'finish_reason': choice.get('finish_reason'),
                'completion_tokens': (answer.get('usage') or {}).get('completion_tokens'),
                'prompt_tokens': (answer.get('usage') or {}).get('prompt_tokens'),
            })
            if not isinstance(completion['finish_reason'], str):
                return argv, props_identity, completion, {
                    'servable': False,
                    'reason': f'chat completion carried no finish_reason: {json.dumps(answer)[:300]}'}
            return argv, props_identity, completion, {
                'servable': True,
                'reason': 'llama.cpp loaded the artifact, /props echoed its path and one chat '
                          f"completion answered (finish_reason={completion['finish_reason']!r})"}
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def export_smoke(gguf, server=None, ctx=4096, max_tokens=16, out=None,
                 server_args=(), run_server=True):
    start = time.monotonic()
    gguf = Path(gguf)
    if not gguf.is_file():
        raise FileNotFoundError(gguf)
    out = Path(out) if out is not None else Path('export-smoke.json')
    log_path = out.with_name(out.stem + '.server.log')
    artifact = {'path': str(gguf.resolve()), 'sha256': _sha256(gguf),
                'bytes': gguf.stat().st_size,
                'architecture': 'unknown', 'tensors': None, 'dtypes': {}}
    header_error = None
    try:
        artifact.update(read_header(gguf))
    except (ValueError, OSError, struct.error) as error:
        header_error = f'{type(error).__name__}: {error}'
        artifact['header_error'] = header_error

    if not run_server:
        if header_error:
            verdict = {'servable': False, 'reason': f'header unreadable, so nothing was served: {header_error}'}
        else:
            verdict = {'servable': 'unmeasured',
                       'reason': 'header-only (--no-server): llama.cpp was not run, so serving is unmeasured'}
        argv, props_identity, completion = [], {}, {}
    else:
        try:
            argv, props_identity, completion, verdict = measure(
                gguf, server, ctx, max_tokens, list(server_args), log_path)
        except OSError as error:
            argv, props_identity, completion = [], {}, {}
            verdict = {'servable': False, 'reason': f'could not start llama-server: {type(error).__name__}: {error}'}

    result = {'artifact': artifact, 'argv': argv, 'props_identity': props_identity,
              'completion': completion, 'verdict': verdict,
              'cost_seconds': round(time.monotonic() - start, 3)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def export_smoke_command(args):
    if not args.no_server and args.server is None:
        raise ValueError('export-smoke needs --server PATH, or --no-server for the header half only')
    return export_smoke(args.gguf, args.server, args.ctx, args.max_tokens, args.out,
                        args.server_arg, not args.no_server)
