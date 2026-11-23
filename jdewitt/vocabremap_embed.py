# SPDX-License-Identifier: AGPL-3.0-or-later
"""Input-side (embedding-side) denial id remap: what the model READS, not what it writes.

The decode-side remap (``jdewitt vocab-remap --remap on``) suppresses a token the
model is about to emit. This module is the other placement: it rewrites the
*token ids of the prompt* before the server sees them, so a prompt that says the
denial reads as the affirmative. `` cannot`` becomes `` can`` (one id for one id),
and a multi-token denial keeps its modal while the residual negating token is
dropped (`` can not`` -> `` can``). The table of surfaces is the shipped denial
inventory; ``assets/vocab-remap-affirmative.json`` names the counterpart the
model should read and carries no ids.

That is a semantic change to the user's own words, so every report from this
module is a damage measurement: the prompt-token diff is kept beside every decode
row, an input-identical row is counted as *not exposed* rather than as quiet
success, and neither arm applies a ``logit_bias``. The only difference between
the arms is the id list the server is asked to read.

Both arms send the prompt as ids (``"prompt": [ids]``), which is what makes the
remap survive the multi-step loop: re-sending the prompt text would re-tokenize
the denial back. The off arm is the control for that claim — it sends the
unmodified ids and is checked against a plain greedy wave token-for-token.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .vocabremap import (_completion, _contexts_for, _candidate, _portable_path,
                         _sha256_file, _piece_bytes, apply_chat_template, detokenize,
                         load_table, read_tokenizer, served_tokenize,
                         start_server, stop_server, verify_artifact_identity,
                         verify_served_tokenizer)

AFFIRMATIVE_SCHEMA = 'jdewitt.vocab-remap-affirmative/1'
DIFF_SCHEMA = 'jdewitt.vocab-remap-embed-diff/1'
ARM_SCHEMA = 'jdewitt.vocab-remap-embed-arm/1'
# Only the any-scope entries are input surfaces. An opener_window entry is scoped to
# the first generated tokens of an answer; a prompt has no generated step, so mapping
# it as input would silently widen its scope.
INPUT_SCOPES = ('any',)


def load_affirmative(path=None):
    """Load and validate the affirmative map. Returns (document, raw bytes)."""
    if path is None:
        from importlib.resources import files
        raw = files('jdewitt').joinpath('assets/vocab-remap-affirmative.json').read_bytes()
    else:
        raw = Path(path).read_bytes()
    document = json.loads(raw.decode('utf-8'))
    if not isinstance(document, dict) or document.get('schema') != AFFIRMATIVE_SCHEMA:
        raise ValueError(f'affirmative schema must be {AFFIRMATIVE_SCHEMA!r}')
    entries = document.get('entries')
    if not isinstance(entries, list) or not entries:
        raise ValueError('affirmative entries must be a non-empty list')
    seen = set()
    for index, entry in enumerate(entries):
        where = f'affirmative entry {index}'
        if not isinstance(entry, dict):
            raise ValueError(f'{where}: must be an object')
        for field in ('id', 'affirmative'):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                raise ValueError(f'{where}: {field} must be a non-empty string')
        if entry['id'] in seen:
            raise ValueError(f'{where}: duplicate id {entry["id"]!r}')
        seen.add(entry['id'])
        if 'note' in entry and not isinstance(entry['note'], str):
            raise ValueError(f'{where}: note must be a string')
    return document, raw


def affirmative_items(entries, affirmative, port):
    """Per mapped entry: the denial span and the affirmative span, tokenized by the served tokenizer.

    The affirmative is resolved in the same context as the denial it replaces (bare or
    leading space), so `` can`` maps to `` can`` and not to the bare ``can`` token, which
    a running-text prompt does not contain. The positive span may split into any number of
    tokens; the remap replaces the whole matched span, so no alignment rule is needed.
    """
    by_id = {entry['id']: entry for entry in entries}
    missing = [item['id'] for item in affirmative['entries'] if item['id'] not in by_id]
    if missing:
        raise ValueError(f'affirmative map names ids the denial table does not have: {missing}')
    items = []
    for mapped in affirmative['entries']:
        entry = by_id[mapped['id']]
        if entry['scope'] not in INPUT_SCOPES:
            continue
        contexts = []
        for context_item in _contexts_for(entry, port):
            text = mapped['affirmative'] if context_item['context'] == 'bare' else ' ' + mapped['affirmative']
            ids = served_tokenize(port, text)
            if not ids:
                raise ValueError(f'{entry["id"]}: affirmative {text!r} tokenizes to nothing')
            contexts.append({'context': context_item['context'], 'denial_ids': context_item['ids'],
                             'denial_text': context_item['text'], 'affirmative_ids': ids,
                             'affirmative_text': text})
        items.append({'entry': entry, 'affirmative': mapped['affirmative'],
                      'note': mapped.get('note'), 'contexts': contexts})
    return items


def _span_text(ids, tokens):
    return detokenize(ids, tokens)


def plan_remap(ids, tokens, items, allow_empty_context=False):
    """Rewrite denial spans in a prompt id path. Returns (new_ids, edits, unmatched_entries).

    Matching is token-exact and longest-first, and a candidate only fires when the
    matched span's own bytes equal the surface form: a sequence that happens to share
    ids with a different word is not a denial. ``unmatched_entries`` names every mapped
    entry that had no occurrence in this path, so a row with no diff is reported as
    not exposed instead of as a zero.
    """
    candidates = []
    for item in items:
        for context_item in item['contexts']:
            sequence = tuple(context_item['denial_ids'])
            if not sequence:
                continue
            if _span_text(list(sequence), tokens) != context_item['denial_text'] and not allow_empty_context:
                continue
            candidates.append({'sequence': sequence, 'item': item, 'context': context_item})
    candidates.sort(key=lambda candidate: (-len(candidate['sequence']), candidate['item']['entry']['id'],
                                           candidate['context']['context']))
    out, edits, position = [], [], 0
    seen_entries = set()
    while position < len(ids):
        hit = None
        for candidate in candidates:
            sequence = candidate['sequence']
            if tuple(ids[position:position + len(sequence)]) != sequence:
                continue
            hit = candidate
            break
        if hit is None:
            out.append(ids[position])
            position += 1
            continue
        context_item = hit['context']
        entry = hit['item']['entry']
        seen_entries.add(entry['id'])
        edits.append({
            'entry_id': entry['id'],
            'language': entry['language'],
            'form': entry['form'],
            'context': context_item['context'],
            'start': position,
            'after_start': len(out),
            'matched_ids': list(hit['sequence']),
            'matched_text': context_item['denial_text'],
            'replacement_ids': list(context_item['affirmative_ids']),
            'replacement_text': context_item['affirmative_text'],
            'affirmative': hit['item']['affirmative'],
        })
        out.extend(context_item['affirmative_ids'])
        position += len(hit['sequence'])
    unmatched = [item['entry']['id'] for item in items if item['entry']['id'] not in seen_entries]
    return out, edits, unmatched


def prompt_path(port, text, remap_ids=None):
    """Tokenize one rendered prompt; optionally rewrite it. Returns both paths and their texts."""
    before = served_tokenize(port, text, parse_special=True)
    if remap_ids is None:
        return {'before': before, 'after': list(before), 'edits': [], 'unmatched': []}
    after, edits, unmatched = remap_ids(before)
    return {'before': before, 'after': after, 'edits': edits, 'unmatched': unmatched}


def _ids_text(ids, tokens):
    return detokenize(ids, tokens)


def _piece_text(token_id, tokens):
    return detokenize([token_id], tokens)


def decode_ids_steps(port, tokens, prompt_ids, max_tokens):
    """One greedy step loop that sends the prompt as ids: prompt_ids + everything emitted so far.

    No ``logit_bias`` is ever sent. Returning ids (not text) keeps the remap in force
    on every step; the server reuses the KV prefix because each request extends the
    previous one.
    """
    import time

    emitted = []
    steps = []
    started = time.monotonic()
    stop = None
    requests = 0
    for step in range(max_tokens):
        response = _completion(port, {'prompt': list(prompt_ids) + list(emitted), 'n_predict': 1,
                                      'temperature': 0.0, 'cache_prompt': True, 'n_probs': 1})
        requests += 1
        candidate = _candidate(response)
        if candidate is None:
            stop = response.get('stop_type') or 'eos'
            break
        piece = _piece_bytes(candidate['id'], tokens)
        if not piece and not (response.get('content') or ''):
            stop = 'eos'
            break
        emitted.append(candidate['id'])
        timings = response.get('timings') if isinstance(response.get('timings'), dict) else {}
        steps.append({'step': step, 'token_id': candidate['id'],
                      'piece': _piece_text(candidate['id'], tokens),
                      'prompt_n': timings.get('prompt_n'), 'cache_n': timings.get('cache_n')})
        stop_type = response.get('stop_type')
        if stop_type and stop_type != 'limit':
            stop = stop_type
            break
    return {'ids': emitted, 'steps': steps, 'stop': stop, 'completion_requests': requests,
            'seconds': round(time.monotonic() - started, 3),
            'text': detokenize(emitted, tokens)}


def grep_wave(port, prompt, max_tokens):
    """The plain text-prompt greedy control wave the id loop must reproduce."""
    response = _completion(port, {'prompt': prompt, 'n_predict': max_tokens, 'temperature': 0.0,
                                  'cache_prompt': False, 'n_probs': 1})
    return {'ids': [int(value) for value in response.get('tokens') or []],
            'text': response.get('content') or ''}


def _plain_response_ids(response):
    return [int(value) for value in response.get('tokens') or []]


def vocab_remap_embed_arm(table=None, affirmative=None, artifact=None, suite=None, out=None, mode='off',
                          max_tokens=512, serve=None, port=None, template='chat', limit=0,
                          plain_check=0, ctx=4096, parallel=4, ngl=99, device=None):
    """Decode one arm of the input-side remap: the only difference is whether the prompt ids are rewritten."""
    if mode not in ('on', 'off'):
        raise ValueError('--embed must be on or off')
    if artifact is None or suite is None or out is None:
        raise ValueError('--embed needs --artifact GGUF, --suite FILE and --out DIR')
    if template not in ('chat', 'raw'):
        raise ValueError('--template must be chat or raw')
    if serve is None and port is None:
        raise ValueError('--embed needs --serve PATH or --port N')
    artifact, suite, out = Path(artifact), Path(suite), Path(out)
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    if not suite.is_file():
        raise FileNotFoundError(suite)
    document, raw_table = load_table(table)
    entries = document['entries']
    affirmative_document, raw_affirmative = load_affirmative(affirmative)
    tokenizer = read_tokenizer(artifact)
    tokens = tokenizer['tokens']
    from .vocabremap import _merge_ranks
    vocab = {token: token_id for token_id, token in enumerate(tokens)}
    ranks = _merge_ranks(tokenizer['merges'])
    rows_in = [json.loads(line) for line in suite.read_text().splitlines() if line.strip()]
    if limit:
        rows_in = rows_in[:limit]
    out.mkdir(parents=True, exist_ok=True)
    import time
    began = time.monotonic()
    started = None
    try:
        if serve is not None:
            started = start_server(serve, artifact, port, out / 'server.log', ctx=ctx, parallel=parallel,
                                   ngl=ngl, device=device)
            port = started['port']
        if not isinstance(port, int) or not 0 < port < 65536:
            raise ValueError(f'--port must be an integer in 1..65535, got {port!r}')
        props = verify_artifact_identity(port, artifact)
        served_check = verify_served_tokenizer(entries, port, tokens, vocab, ranks)
        if not served_check['agrees']:
            raise ValueError(f'the server on port {port} tokenizes the table forms differently from the artifact '
                             f'header ({len(served_check["mismatches"])} of {served_check["contexts_checked"]} '
                             'contexts); refusing to record rows from another tokenizer')
        items = affirmative_items(entries, affirmative_document, port)
        rows = []
        for number, task in enumerate(rows_in, 1):
            rendered = (apply_chat_template(port, task['prompt']) if template == 'chat' else task['prompt'])
            before = served_tokenize(port, rendered, parse_special=True)
            if mode == 'on':
                after, edits, unmatched = plan_remap(before, tokens, items)
            else:
                after, edits, unmatched = list(before), [], [item['entry']['id'] for item in items]
            loop = decode_ids_steps(port, tokens, after, max_tokens)
            record = {
                'id': task['id'], 'language': task.get('language'), 'kind': task.get('kind'),
                'category': task.get('category'), 'family_id': task.get('family_id'),
                'mode': mode, 'max_tokens': max_tokens,
                'prompt_sha256': hashlib.sha256(task['prompt'].encode()).hexdigest(),
                'rendered_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
                'rendered_prompt': rendered,
                'prompt_tokens_before': len(before),
                'prompt_tokens_after': len(after),
                'prompt_ids_before_sha256': hashlib.sha256(json.dumps(before).encode()).hexdigest(),
                'prompt_ids_after_sha256': hashlib.sha256(json.dumps(after).encode()).hexdigest(),
                'input_changed': after != before,
                'edits': edits,
                'unmatched_entries': unmatched if mode == 'on' else [],
                'prompt_text_before': _ids_text(before, tokens),
                'prompt_text_after': _ids_text(after, tokens),
                'text': loop['text'], 'n_tokens': len(loop['ids']), 'stop': loop['stop'],
                'completion_requests': loop['completion_requests'],
                'seconds': loop['seconds'],
                'prompt_n_first_step': loop['steps'][0]['prompt_n'] if loop['steps'] else None,
                'prompt_n_after_first_step': max((step['prompt_n'] for step in loop['steps'][1:]
                                                  if type(step['prompt_n']) is int), default=None),
                'cache_n_after_first_step': min((step['cache_n'] for step in loop['steps'][1:]
                                                 if type(step['cache_n']) is int), default=None),
            }
            if number <= plain_check:
                plain = grep_wave(port, rendered, max_tokens)
                record['plain'] = {'ids': plain['ids'], 'text': plain['text'],
                                   'equals_loop': plain['ids'] == loop['ids']}
            rows.append(record)
            if number % 10 == 0 or number == len(rows_in):
                generated = sum(row['n_tokens'] for row in rows)
                elapsed = time.monotonic() - began
                print(f'[{mode}] {number}/{len(rows_in)} rows  {generated} tokens  '
                      f'{generated / max(elapsed, 1e-9):.1f} tok/s  last={task["id"]}', flush=True)
    finally:
        if started is not None:
            stop_server(started)
    rows_path = out / 'rows.jsonl'
    rows_path.write_text(''.join(json.dumps(row, sort_keys=True, allow_nan=False) + '\n' for row in rows))
    edited = [row for row in rows if row['edits']]
    report = {
        'schema': ARM_SCHEMA,
        'mode': mode,
        'scope': ('One arm of the input-side (embedding-side) denial remap: the prompt token ids are '
                  'rewritten (on) or sent unchanged (off), and no logit_bias is ever applied. A row '
                  'here is a token path, the model-visible ids before/after, and the answer; it is not '
                  'a harmlessness or a capability measurement.'),
        'artifact': {'path': _portable_path(artifact), 'sha256': _sha256_file(artifact),
                     'bytes': artifact.stat().st_size, 'tokenizer': tokenizer['identity']},
        'suite': {'path': _portable_path(suite), 'sha256': _sha256_file(suite), 'rows': len(rows)},
        'table': {'path': _portable_path(table) if table is not None else 'jdewitt/assets/vocab-remap-denial.json',
                  'sha256': hashlib.sha256(raw_table).hexdigest(), 'entries': len(entries)},
        'affirmative': {'path': _portable_path(affirmative) if affirmative is not None
                        else 'jdewitt/assets/vocab-remap-affirmative.json',
                        'sha256': hashlib.sha256(raw_affirmative).hexdigest(),
                        'entries': len(affirmative_document['entries']),
                        'unmapped': affirmative_document.get('unmapped', [])},
        'server': {'argv': started['argv'] if started is not None else [],
                   'port': port, 'started_by_runner': started is not None,
                   'model_path_reported': props['model_path_reported'],
                   'model_path_resolves_to_artifact': props['model_path_resolves_to_artifact'],
                   'build_info': props['build_info'], 'n_ctx': props['n_ctx'],
                   'total_slots': props['total_slots'],
                   'tokenizer_check': {'contexts_checked': served_check['contexts_checked'],
                                       'agrees': served_check['agrees']}},
        'config': {'template': template, 'max_tokens': max_tokens, 'temperature': 0.0,
                   'cache_prompt': True, 'ctx': ctx, 'parallel': parallel, 'ngl': ngl,
                   'device': device, 'prompt_sent_as': 'ids',
                   'logit_bias': 'none in either arm'},
        'rows_file': {'path': 'rows.jsonl', 'sha256': _sha256_file(rows_path), 'rows': len(rows)},
        'metrics': {'generated_tokens': sum(row['n_tokens'] for row in rows),
                    'completion_requests': sum(row['completion_requests'] for row in rows),
                    'rows_input_changed': len(edited),
                    'prompt_tokens_before': sum(row['prompt_tokens_before'] for row in rows),
                    'prompt_tokens_after': sum(row['prompt_tokens_after'] for row in rows),
                    'plain_checked_rows': min(plain_check, len(rows)),
                    'plain_equals_loop': sum(1 for row in rows if row.get('plain', {}).get('equals_loop')),
                    'wall_seconds': round(time.monotonic() - began, 3)},
        'verdict': {'complete': len(rows) == len(rows_in),
                    'note': 'an arm is complete only when it holds every suite row it names'},
    }
    (out / 'run.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def embed_diff(port, tokens, items, rows, template='chat'):
    """The model-visible prompt-token diff for authored rows: ids and text, before and after."""
    out = []
    for task in rows:
        rendered = (apply_chat_template(port, task['prompt']) if template == 'chat' else task['prompt'])
        before = served_tokenize(port, rendered, parse_special=True)
        after, edits, unmatched = plan_remap(before, tokens, items)
        windows = []
        for edit in edits:
            lo = max(0, edit['start'] - 4)
            hi = min(len(before), edit['start'] + len(edit['matched_ids']) + 4)
            after_lo = max(0, edit['after_start'] - (edit['start'] - lo))
            after_hi = min(len(after), edit['after_start'] + len(edit['replacement_ids'])
                           + (hi - edit['start'] - len(edit['matched_ids'])))
            windows.append({'start': lo,
                            'before_ids': before[lo:hi],
                            'before_pieces': [_piece_text(token_id, tokens) for token_id in before[lo:hi]],
                            'before_text': _ids_text(before[lo:hi], tokens),
                            'after_ids': after[after_lo:after_hi],
                            'after_pieces': [_piece_text(token_id, tokens) for token_id in after[after_lo:after_hi]],
                            'after_text': _ids_text(after[after_lo:after_hi], tokens)})
        out.append({
            'id': task['id'], 'language': task.get('language'), 'surface': task.get('surface'),
            'prompt': task['prompt'],
            'rendered_prompt': rendered,
            'before': {'n_tokens': len(before), 'ids': before, 'sha256': hashlib.sha256(json.dumps(before).encode()).hexdigest(),
                       'text': _ids_text(before, tokens)},
            'after': {'n_tokens': len(after), 'ids': after, 'sha256': hashlib.sha256(json.dumps(after).encode()).hexdigest(),
                      'text': _ids_text(after, tokens)},
            'edits': edits,
            'unmatched_entries': unmatched,
            'semantic_flip': bool(edits),
            'token_delta': len(after) - len(before),
            'windows': windows,
        })
    return out
