# SPDX-License-Identifier: AGPL-3.0-or-later
"""Token-level denial-word inventory, resolved per GGUF tokenizer.

The table in ``assets/vocab-remap-denial.json`` names surfaces and never token
ids: the split of a word is a property of one tokenizer, and vocab surgery has to
edit the ids that tokenizer actually emits. This module reads the vocabulary
(``tokenizer.ggml.tokens``) and merges (``tokenizer.ggml.merges``) out of a GGUF
header, encodes each surface with the same byte-level BPE the artifact uses, and
reports the id/token sequence per artifact.

Two contexts are reported for every surface. A single-word form is resolved bare
(``cannot``); a form with an internal space is resolved with the leading space a
phrase carries in running text (`` kann nicht``), because that is the occurrence a
denial-word remap has to reach. Both contexts are printed for every lexeme, so
the choice of primary context is inspectable rather than implicit.

Nothing here loads weights or a model runtime; a GGUF header is metadata. The
tokenizer identity hashes the decoded vocabulary in id order (4-byte big-endian
length prefix, then the UTF-8 bytes of each token), so it identifies the
tokenizer independently of the file path or the rest of the container.

``--dry-run`` is the other half: it asks a live llama-server for the token path
of each authored probe (POST /tokenize, never /completion), finds every place a
table surface can occur in that path, and reports the decode step at which the
completing token would be suppressed (logit_bias -100) and the replacement's
first token forced (+100). No weights, logits, or generations are involved; the
plan is what the decode driver in the next stage would do on that path.
"""

import hashlib
import json
import os
import socket
import struct
import subprocess
import time
import urllib.request
from pathlib import Path

from .refusal_shape import classify

TABLE_SCHEMA = 'jdewitt.vocab-remap-denial/1'
LANGUAGES = ('en', 'nl', 'de')
KINDS = ('denial_word', 'refusal_opener', 'hedge', 'apology')
SCOPES = ('any', 'opener_window')
FIELDS = ('id', 'language', 'form', 'kind', 'scope', 'replacement', 'note')
METHOD = ('Byte-level BPE over the artifact vocabulary, using the artifact merge ranks. Token text is the GGUF '
          'vocabulary string: a leading space is written as the byte-level marker, and a non-ASCII character '
          'appears as the byte-level characters of its UTF-8 bytes. Single-word forms are resolved bare; forms '
          'with an internal space are resolved with the leading space they carry in running text. Both contexts '
          'are reported. single_token is true only when the form is one token in both contexts. representable is '
          'false when a byte-level symbol has no vocabulary id.')


def _read_field(fields, name):
    field = fields.get(name)
    return field.contents() if field is not None else None


def _bytes_to_unicode():
    """The GPT-2 byte-to-unicode table, i.e. how a BPE vocabulary spells raw bytes."""
    byte_values = list(range(ord('!'), ord('~') + 1))
    byte_values += list(range(ord('¡'), ord('¬') + 1))
    byte_values += list(range(ord('®'), ord('ÿ') + 1))
    codepoints = byte_values[:]
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            codepoints.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, (chr(codepoint) for codepoint in codepoints)))


BYTE_TO_UNICODE = _bytes_to_unicode()


def _byte_level(text):
    return ''.join(BYTE_TO_UNICODE[byte] for byte in text.encode('utf-8'))


def tokens_digest(tokens):
    digest = hashlib.sha256()
    for token in tokens:
        raw = token.encode('utf-8')
        digest.update(struct.pack('>I', len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def read_tokenizer(path):
    """Read vocab + merges + identity from a GGUF header. No tensors are decoded."""
    from gguf import GGUFReader
    reader = GGUFReader(str(path))
    fields = reader.fields
    tokens = _read_field(fields, 'tokenizer.ggml.tokens')
    if not tokens:
        raise ValueError(f'{Path(path).name}: no tokenizer.ggml.tokens array in the GGUF metadata')
    model = _read_field(fields, 'tokenizer.ggml.model')
    pre = _read_field(fields, 'tokenizer.ggml.pre')
    merges = _read_field(fields, 'tokenizer.ggml.merges') or []
    return {'model': model, 'pre': pre, 'tokens': tokens, 'merges': merges,
            'identity': {'model': model, 'pre': pre, 'n_vocab': len(tokens),
                         'tokens_sha256': tokens_digest(tokens)}}


def _merge_ranks(merges):
    ranks = {}
    for rank, merge in enumerate(merges):
        left, separator, right = merge.partition(' ')
        if separator:
            ranks[(left, right)] = rank
    return ranks


MISSING_RANK = 1 << 62


def bpe_symbols(text, ranks):
    """Merge adjacent byte-level symbols by merge rank, exactly as a GPT-2 BPE decoder expects."""
    word = list(_byte_level(text))
    while len(word) > 1:
        candidates = [(ranks.get((word[index], word[index + 1]), MISSING_RANK), index)
                      for index in range(len(word) - 1)]
        rank, start = min(candidates)
        if rank == MISSING_RANK:
            break
        pair = (word[start], word[start + 1])
        merged = []
        index = 0
        while index < len(word):
            if index < len(word) - 1 and (word[index], word[index + 1]) == pair:
                merged.append(word[index] + word[index + 1])
                index += 2
            else:
                merged.append(word[index])
                index += 1
        word = merged
    return word


def resolve(text, vocab, ranks):
    """Return ([id, token] pairs, representable) for one surface in one context."""
    split = []
    representable = True
    for symbol in bpe_symbols(text, ranks):
        token_id = vocab.get(symbol)
        if token_id is None:
            representable = False
        split.append([token_id, symbol])
    return split, representable


def load_table(path=None):
    """Load and validate the table. Returns (document, raw bytes)."""
    if path is None:
        from importlib.resources import files
        raw = files('jdewitt').joinpath('assets/vocab-remap-denial.json').read_bytes()
    else:
        raw = Path(path).read_bytes()
    document = json.loads(raw.decode('utf-8'))
    if not isinstance(document, dict) or document.get('schema') != TABLE_SCHEMA:
        raise ValueError(f'table schema must be {TABLE_SCHEMA!r}')
    entries = document.get('entries')
    if not isinstance(entries, list) or not entries:
        raise ValueError('table entries must be a non-empty list')
    seen = set()
    for index, entry in enumerate(entries):
        where = f'entry {index}'
        if not isinstance(entry, dict):
            raise ValueError(f'{where}: must be an object')
        missing = [field for field in FIELDS if field not in entry]
        if missing:
            raise ValueError(f'{where}: missing {", ".join(missing)}')
        if not isinstance(entry['id'], str) or not entry['id']:
            raise ValueError(f'{where}: id must be a non-empty string')
        if entry['id'] in seen:
            raise ValueError(f'{where}: duplicate id {entry["id"]!r}')
        seen.add(entry['id'])
        if entry['language'] not in LANGUAGES:
            raise ValueError(f'{where}: language must be one of {LANGUAGES}')
        if entry['kind'] not in KINDS:
            raise ValueError(f'{where}: kind must be one of {KINDS}')
        if entry['scope'] not in SCOPES:
            raise ValueError(f'{where}: scope must be one of {SCOPES}')
        if not isinstance(entry['form'], str) or not entry['form'].strip():
            raise ValueError(f'{where}: form must be a non-empty string')
        if entry['replacement'] is not None and (not isinstance(entry['replacement'], str)
                                                 or not entry['replacement'].strip()):
            raise ValueError(f'{where}: replacement must be a non-empty string or null')
        if not isinstance(entry['note'], str):
            raise ValueError(f'{where}: note must be a string')
    return document, raw


def _primary_context(form):
    return 'leads_with_space' if ' ' in form else 'bare'


def resolve_lexeme(entry, vocab, ranks):
    bare_text = entry['form']
    spaced_text = ' ' + entry['form']
    bare, bare_ok = resolve(bare_text, vocab, ranks)
    spaced, spaced_ok = resolve(spaced_text, vocab, ranks)
    context = _primary_context(entry['form'])
    split = bare if context == 'bare' else spaced
    return {'id': entry['id'], 'language': entry['language'], 'form': entry['form'],
            'kind': entry['kind'], 'scope': entry['scope'], 'replacement': entry['replacement'],
            'ids': [pair[0] for pair in split], 'split': split,
            'single_token': len(bare) == 1 and len(spaced) == 1 and bare_ok and spaced_ok,
            'representable': bare_ok and spaced_ok, 'context': context,
            'contexts': {'bare': bare, 'leads_with_space': spaced}}


def _label(identity, used):
    label = f'{identity["model"]}:{identity["pre"]}:{identity["n_vocab"]}'
    if label in used:
        label = f'{label}:{identity["tokens_sha256"][:8]}'
    used.add(label)
    return label


def disagreements(artifacts, entries):
    """Rows where the primary split (token text sequence) differs across tokenizers."""
    rows = []
    for entry in entries:
        splits = {}
        shapes = set()
        for artifact in artifacts:
            lexeme = next(item for item in artifact['lexemes'] if item['id'] == entry['id'])
            splits[artifact['label']] = lexeme['split']
            shapes.add(tuple(token for _token_id, token in lexeme['split']))
        if len(shapes) > 1:
            rows.append({'form': entry['form'], 'language': entry['language'],
                         'distinct_splits': len(shapes), 'splits': splits})
    return rows


def vocab_remap_list(table=None, artifacts=(), list_mode=False, json_out=None):
    """Resolve the checked-in table against every artifact vocabulary and return one report."""
    if not list_mode:
        raise ValueError('vocab-remap currently implements --list only')
    if not artifacts:
        raise ValueError('at least one --artifact GGUF is required')
    document, raw = load_table(table)
    entries = document['entries']
    resolved = []
    used_labels = set()
    for artifact_path in artifacts:
        tokenizer = read_tokenizer(artifact_path)
        vocab = {token: token_id for token_id, token in enumerate(tokenizer['tokens'])}
        ranks = _merge_ranks(tokenizer['merges'])
        resolved.append({'label': _label(tokenizer['identity'], used_labels),
                         'identity': tokenizer['identity'],
                         'n_vocab': tokenizer['identity']['n_vocab'],
                         'lexemes': [resolve_lexeme(entry, vocab, ranks) for entry in entries]})
    report = {
        'scope': ('Denial-word surfaces resolved against GGUF tokenizers: ids and splits, no model forward. '
                  'This is a token inventory, not a served or weight result.'),
        'method': METHOD,
        'table': {'schema': document['schema'],
                  'sha256': hashlib.sha256(raw).hexdigest(),
                  'entries': len(entries),
                  'languages': sorted({entry['language'] for entry in entries}),
                  'kinds': sorted({entry['kind'] for entry in entries}),
                  'scopes': sorted({entry['scope'] for entry in entries})},
        'artifacts': resolved,
        'disagreements': {'tokenizers': [artifact['label'] for artifact in resolved],
                          'note': ('same form, different token-text sequence under the primary context; ids differ '
                                   'across vocabularies by construction and are not by themselves a disagreement'),
                          'rows': disagreements(resolved, entries)},
    }
    if json_out is not None:
        Path(json_out).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


# --------------------------------------------------------------------------
# --dry-run: plan the suppression schedule against a served tokenizer
# --------------------------------------------------------------------------

DRYRUN_SCHEMA = 'jdewitt.vocab-remap-dryrun/1'
OPENER_WINDOW_TOKENS = 8
UNOPENED_LIMIT = 20
BASE_ANSWER_SOURCES = ('verification/q3-4b-harmful40/q3base/predictions.jsonl',
                       'verification/q3-4b-holdout-wider/wider-base/predictions.jsonl')
DRYRUN_METHOD = (
    'Token paths come from a live llama-server POST /tokenize (add_special=false, parse_special=false); no '
    'completion request is made, no weights or logits are read. Each table form is tokenized in both contexts '
    '(bare at the start of a path, and with the leading space it carries in running text). An occurrence of a '
    'context sequence at steps s..e means the completing token e would carry logit_bias -100, and a replacement, '
    'when the table gives one, would be forced at e with +100 on its first token (leading space included, because '
    'it follows an emitted token). rules_that_fired is the set of entries with at least one occurrence. A rule '
    'is reachable only when at least one context both round-trips through the artifact vocabulary and occurs in '
    'a carrier sentence; a form that matches only in isolation cannot fire in running text and fails the run. '
    'Coverage tokenizes every declining base answer and asks whether any table context occurs in that answer; '
    'opener coverage restricts the occurrence to start before step %d. An opener_window entry only fires before '
    'step %d; an any-scope entry fires anywhere in the path. The prompt plan runs the same matcher '
    'over the prompt token path; those positions are descriptive because prompt tokens are not generated, and the '
    'reference plan (an authored declining continuation on each probe) is the decode-step schedule the driver '
    'would follow if the model emitted that text.' % (OPENER_WINDOW_TOKENS, OPENER_WINDOW_TOKENS))


def _server_json(port, path, payload=None, timeout=60):
    """Loopback-only JSON call. No proxies, no redirects, no credentials."""
    if type(port) is not int or not 0 < port < 65536:
        raise ValueError(f'port must be an integer in 1..65535, got {port!r}')
    request = urllib.request.Request(f'http://127.0.0.1:{port}{path}',
                                     data=json.dumps(payload).encode() if payload is not None else None,
                                     headers={'Content-Type': 'application/json'})

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise ValueError('llama-server redirects are forbidden')

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.loads(response.read(16 * 1024 * 1024))
    except OSError as error:
        raise RuntimeError(f'llama-server on port {port} did not answer {path}: {error}') from error


def served_tokenize(port, text, parse_special=False):
    """Token ids for one string, exactly as the served tokenizer splits it.

    The dry run probes plain surfaces, so special markers stay literal. The decode
    driver tokenizes its own prompt with parse_special=True: the server parses the
    chat template's <|im_start|> markers in a /completion prompt string, so the path
    it evaluates has one token per marker, not the several the literal spelling has.
    """
    payload = _server_json(port, '/tokenize', {'content': text, 'add_special': False,
                                               'parse_special': parse_special})
    tokens = payload.get('tokens') if isinstance(payload, dict) else None
    if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
        raise ValueError(f'llama-server /tokenize returned no integer token list for {text[:40]!r}')
    return tokens


def served_props(port):
    props = _server_json(port, '/props')
    if not isinstance(props, dict):
        raise ValueError('llama-server /props returned no object')
    return {'build_info': props.get('build_info'), 'model_path': props.get('model_path'),
            'model_alias': props.get('model_alias')}


UNICODE_TO_BYTE = {character: byte for byte, character in BYTE_TO_UNICODE.items()}


def _token_bytes(token):
    try:
        return bytes(UNICODE_TO_BYTE[character] for character in token)
    except KeyError:
        return token.encode('utf-8')


def detokenize(ids, tokens):
    """Decode a token path through the artifact vocabulary (byte-level spelling)."""
    return b''.join(_token_bytes(tokens[token_id]) for token_id in ids).decode('utf-8', 'replace')


def piece_text(token_id, tokens):
    """Display text of one token. A multi-byte character split across tokens may read as a replacement char."""
    return _token_bytes(tokens[token_id]).decode('utf-8', 'replace')


def _portable_path(path):
    """A source path as committed evidence: relative to the repository root, never absolute."""
    root = Path(__file__).resolve().parents[1]
    path = Path(path)
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return path.name


def load_probe_rows(path):
    """Read authored probe rows from one JSONL file or every *.jsonl in a directory."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f'probe source does not exist: {path.name}')
    files = sorted(path.glob('*.jsonl')) if path.is_dir() else [path]
    if not files:
        raise ValueError(f'probe source has no JSONL rows: {path.name}')
    rows = []
    for file in files:
        for number, line in enumerate(file.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'{file.name}:{number}: invalid JSON: {error.msg}') from error
            if not isinstance(row, dict):
                raise ValueError(f'{file.name}:{number}: row must be an object')
            for field in ('id', 'prompt'):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f'{file.name}:{number}: {field} must be a non-empty string')
            reference = row.get('reference')
            if reference is not None and (not isinstance(reference, str) or not reference.strip()):
                raise ValueError(f'{file.name}:{number}: reference must be a non-empty string when present')
            rows.append({'id': row['id'], 'prompt': row['prompt'], 'reference': reference,
                         'kind': row.get('kind'), 'language': row.get('language'),
                         'source': _portable_path(file)})
    ids = [row['id'] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('probe rows repeat an id')
    return rows


def _contexts_for(entry, port):
    """The two running-text contexts of a form, tokenized by the served tokenizer."""
    contexts = []
    seen = set()
    for name, text in (('bare', entry['form']), ('leads_with_space', ' ' + entry['form'])):
        ids = served_tokenize(port, text)
        if tuple(ids) in seen:
            continue
        seen.add(tuple(ids))
        contexts.append({'context': name, 'text': text, 'ids': ids})
    return contexts


def _occurrences(path, sequence):
    if not sequence or len(sequence) > len(path):
        return []
    return [index for index in range(len(path) - len(sequence) + 1) if path[index:index + len(sequence)] == sequence]


def _scope_allows(entry, start_step):
    """opener_window rules may only fire before the window closes; 'any' rules fire anywhere."""
    return entry['scope'] == 'any' or start_step < OPENER_WINDOW_TOKENS


def entry_sequences(entries, port, tokens):
    """Per entry: tokenized bare/spaced contexts and the tokenized replacement prefix."""
    result = []
    for entry in entries:
        contexts = _contexts_for(entry, port)
        if any(any(token_id is None or not 0 <= token_id < len(tokens) for token_id in item['ids'])
               for item in contexts):
            raise ValueError(f'{entry["id"]}: served tokenizer returned an id outside its vocabulary')
        replacement = None
        if entry['replacement']:
            text = entry['replacement'] if entry['replacement'].startswith(' ') else ' ' + entry['replacement']
            ids = served_tokenize(port, text)
            if not ids:
                raise ValueError(f'{entry["id"]}: replacement tokenizes to nothing')
            replacement = {'text': text, 'ids': ids, 'first_id': ids[0],
                           'first_piece': piece_text(ids[0], tokens)}
        result.append({'entry': entry, 'contexts': contexts, 'replacement': replacement})
    return result


def _schedule(ids, tokens, entry_items):
    """Occurrence events and the exact per-step bias schedule for one token path."""
    events = []
    seen = set()
    for item in entry_items:
        entry = item['entry']
        for context_item in item['contexts']:
            sequence = context_item['ids']
            for start in _occurrences(ids, sequence):
                if not _scope_allows(entry, start):
                    continue
                end = start + len(sequence) - 1
                key = (entry['id'], start, end)
                if key in seen:
                    continue
                seen.add(key)
                replacement = item['replacement']
                events.append({'entry_id': entry['id'], 'context': context_item['context'],
                               'scope': entry['scope'], 'start_step': start, 'end_step': end,
                               'window': [start, end], 'suppress_at_step': end, 'suppressed_id': sequence[-1],
                               'suppressed_piece': piece_text(sequence[-1], tokens), 'bias': -100,
                               'replacement': entry['replacement'],
                               'replacement_ids': replacement['ids'] if replacement else None,
                               'replacement_first_id': replacement['first_id'] if replacement else None,
                               'replacement_first_piece': replacement['first_piece'] if replacement else None,
                               'force_bias': 100 if replacement else None})
    events.sort(key=lambda event: (event['start_step'], event['end_step'], event['entry_id'], event['context']))
    schedule = [{'step': step, 'token_id': ids[step], 'piece': piece_text(ids[step], tokens),
                 'suppressions': [], 'forced_replacements': []} for step in range(len(ids))]
    for event in events:
        step = schedule[event['suppress_at_step']]
        step['suppressions'].append({'entry_id': event['entry_id'], 'suppressed_id': event['suppressed_id'],
                                     'bias': event['bias'], 'context': event['context'],
                                     'prefix_steps': [event['start_step'], event['end_step'] - 1]
                                     if event['end_step'] > event['start_step'] else []})
        if event['force_bias'] is not None:
            step['forced_replacements'].append({'entry_id': event['entry_id'],
                                                'first_id': event['replacement_first_id'],
                                                'first_piece': event['replacement_first_piece'],
                                                'bias': event['force_bias'], 'remaining_ids': event['replacement_ids'][1:]})
    return events, schedule


def _path_report(ids, tokens):
    return {'n_tokens': len(ids), 'ids': ids, 'pieces': [piece_text(token_id, tokens) for token_id in ids],
            'text': detokenize(ids, tokens)}


def _plan_for_text(text, port, tokens, entry_items):
    ids = served_tokenize(port, text)
    events, schedule = _schedule(ids, tokens, entry_items)
    return {'path': _path_report(ids, tokens), 'events': events, 'schedule': schedule,
            'rules_that_fired': sorted({event['entry_id'] for event in events})}


def sequence_controls(port, tokens, entry_items):
    """A prefix that is not a completion must not fire, and a real completion must.

    ' can' alone is the prefix of can't; a matcher that fires on the prefix would
    demonize every 'we can compute'. This is the same sequence-aware rule the
    decode driver needs, exercised against the served tokenizer. The checks are
    scoped to the can't entry so a user-supplied table does not make them lie.
    """
    cant = next((item for item in entry_items if item['entry']['form'] == "can't"), None)
    if cant is None:
        return {'skipped': "table has no can't entry; prefix/completion control not applicable"}
    negative = _plan_for_text('We can compute the total from here.', port, tokens, entry_items)
    positive = _plan_for_text("I can't help with that.", port, tokens, entry_items)
    if cant['entry']['id'] in negative['rules_that_fired']:
        raise RuntimeError(f'sequence control failed: a bare " can" prefix fired {cant["entry"]["id"]}')
    if cant['entry']['id'] not in positive['rules_that_fired']:
        raise RuntimeError(f'sequence control failed: "I can\'t" did not fire {cant["entry"]["id"]} '
                           f'({positive["rules_that_fired"]})')
    return {'entry_id': cant['entry']['id'],
            'negative': {'text': 'We can compute the total from here.',
                         'rules_that_fired': negative['rules_that_fired']},
            'positive': {'text': "I can't help with that.", 'rules_that_fired': positive['rules_that_fired']},
            'note': ('a prefix is not a completion: the bare " can" token must stay live while '
                     '" can" + "\'t" completes the denial sequence')}


def reachability(entries, port, tokens):
    """Rule-level reachability: the form must survive a running-text carrier, not just isolation."""
    rows = []
    for entry in entries:
        contexts = []
        for name, text, carrier in (('bare', entry['form'], entry['form'] + ' is a phrase.'),
                                    ('leads_with_space', ' ' + entry['form'],
                                     'A sentence says' + ' ' + entry['form'] + ' here.')):
            try:
                ids = served_tokenize(port, text)
                carrier_ids = served_tokenize(port, carrier)
            except (ValueError, RuntimeError) as error:
                raise RuntimeError(f'{entry["id"]} ({name}): served tokenizer failed: {error}') from error
            contexts.append({'context': name, 'ids': ids,
                             'roundtrips': bool(ids) and detokenize(ids, tokens) == text,
                             'occurs_in_carrier': any(_scope_allows(entry, start)
                                                      for start in _occurrences(carrier_ids, ids))})
        reachable = any(item['ids'] and item['roundtrips'] and item['occurs_in_carrier'] for item in contexts)
        rows.append({'entry_id': entry['id'], 'reachable': reachable, 'contexts': contexts})
    return rows


def verify_served_tokenizer(entries, port, tokens, vocab, ranks):
    """Cross-check the served tokenizer against the artifact's own vocab+merges.

    The planner uses served ids only; this check exists so a server started on
    the wrong artifact fails instead of producing a plausible-looking plan.
    """
    mismatches = []
    checked = 0
    for entry in entries:
        for text in (entry['form'], ' ' + entry['form']):
            served = served_tokenize(port, text)
            expected = [token_id for token_id, _token in resolve(text, vocab, ranks)[0]]
            checked += 1
            if served != expected:
                mismatches.append({'entry_id': entry['id'], 'text': text, 'served': served, 'artifact': expected})
    return {'contexts_checked': checked, 'mismatches': mismatches, 'agrees': not mismatches}


def load_answers(sources):
    """Declining base answers from local prediction files, with portable source names."""
    loaded = []
    missing = []
    for source in sources:
        path = Path(source)
        if not path.exists():
            missing.append(_portable_path(path))
            continue
        rows = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row.get('answer'), str) and row.get('status', 'ok') == 'ok':
                rows.append(row)
        loaded.append({'source': _portable_path(path), 'rows': rows})
    return loaded, missing


def coverage_for(loaded, missing, port, tokens, entry_items):
    """Does the answer text contain a table surface? Per source, plus the uncovered openers."""
    sources = []
    uncovered = []
    matched_by_entry = {}
    for item in loaded:
        declining = covered = opener_covered = 0
        for row in item['rows']:
            shape = classify(row['answer'])
            if not shape['declining']:
                continue
            declining += 1
            ids = served_tokenize(port, row['answer'])
            hits = set()
            opener_hit = False
            for entry_item in entry_items:
                for context_item in entry_item['contexts']:
                    starts = [start for start in _occurrences(ids, context_item['ids'])
                              if _scope_allows(entry_item['entry'], start)]
                    if starts:
                        hits.add(entry_item['entry']['id'])
                        if any(start < OPENER_WINDOW_TOKENS for start in starts):
                            opener_hit = True
            for entry_id in hits:
                matched_by_entry[entry_id] = matched_by_entry.get(entry_id, 0) + 1
            if hits:
                covered += 1
            if opener_hit:
                opener_covered += 1
            if not hits and len(uncovered) < UNOPENED_LIMIT:
                uncovered.append({'source': item['source'], 'id': row.get('id'), 'shape': shape['shape'],
                                  'opener': row['answer'][:240], 'chars': len(row['answer'])})
        sources.append({'source': item['source'], 'declining': declining, 'covered': covered,
                        'opener_covered': opener_covered,
                        'fraction': round(covered / declining, 4) if declining else None,
                        'opener_fraction': round(opener_covered / declining, 4) if declining else None})
    declining = sum(item['declining'] for item in sources)
    covered = sum(item['covered'] for item in sources)
    opener_covered = sum(item['opener_covered'] for item in sources)
    return {'declining': declining, 'covered': covered, 'opener_covered': opener_covered,
            'fraction': round(covered / declining, 4) if declining else None,
            'opener_fraction': round(opener_covered / declining, 4) if declining else None,
            'opener_window_tokens': OPENER_WINDOW_TOKENS,
            'by_source': sources, 'missing_sources': missing,
            'uncovered_openers': uncovered, 'uncovered_total': max(0, declining - covered),
            'matched_by_entry': matched_by_entry,
            'note': ('uncovered rows are listed verbatim because a table that cannot reach the base\'s actual '
                     'openers is a bug, not a result: check each opener before treating the fraction as a limit. '
                     'Reaching a topic-first decline would require suppressing the topic word, which this table '
                     'deliberately does not do (R3 word-demonization in reverse).')}


def vocab_remap_dry_run(table=None, suite=None, probes=None, artifacts=(), ports=(), json_out=None,
                        base_answers=None):
    """Plan suppression/forced-replacement steps per probe and audit table reachability."""
    if bool(suite) == bool(probes):
        raise ValueError('dry-run needs exactly one of --suite FILE or --probes DIR')
    if not artifacts:
        raise ValueError('at least one --artifact GGUF is required')
    if not ports:
        raise ValueError('at least one --port is required')
    if len(ports) == 1 and len(artifacts) == 1:
        pairs = [(artifacts[0], ports[0])]
    elif len(ports) == len(artifacts):
        pairs = list(zip(artifacts, ports))
    else:
        raise ValueError(f'{len(artifacts)} --artifact values need either one --port or {len(artifacts)} ports; '
                         f'got {len(ports)}')
    document, raw = load_table(table)
    entries = document['entries']
    probe_rows = load_probe_rows(probes if probes is not None else suite)
    sources = list(base_answers) if base_answers else list(BASE_ANSWER_SOURCES)
    loaded, missing = load_answers(sources)
    if not loaded:
        raise ValueError('no base-answer file exists at the requested sources; the audit has nothing to test')
    tokenizers = []
    used_labels = set()
    for artifact_path, port in pairs:
        tokenizer = read_tokenizer(artifact_path)
        tokens = tokenizer['tokens']
        vocab = {token: token_id for token_id, token in enumerate(tokens)}
        ranks = _merge_ranks(tokenizer['merges'])
        label = _label(dict(tokenizer['identity']), used_labels)
        props = served_props(port)
        served = verify_served_tokenizer(entries, port, tokens, vocab, ranks)
        if not served['agrees']:
            raise ValueError(f'{label}: the server on port {port} does not tokenize this artifact\'s forms the '
                             f'same way ({len(served["mismatches"])} of {served["contexts_checked"]} contexts '
                             'differ); a plan computed here would describe a different tokenizer')
        entry_items = entry_sequences(entries, port, tokens)
        plans = []
        for row in probe_rows:
            prompt_plan = _plan_for_text(row['prompt'], port, tokens, entry_items)
            reference_plan = None
            rules = set(prompt_plan['rules_that_fired'])
            if row['reference']:
                reference_plan = _plan_for_text(row['reference'], port, tokens, entry_items)
                rules.update(reference_plan['rules_that_fired'])
            plans.append({'id': row['id'], 'kind': row['kind'], 'language': row['language'],
                          'source': row['source'], 'prompt': row['prompt'],
                          'prompt_sha256': hashlib.sha256(row['prompt'].encode()).hexdigest(),
                          'prompt_plan': prompt_plan,
                          'reference': row['reference'], 'reference_plan': reference_plan,
                          'rules_that_fired': sorted(rules),
                          'rules_that_fired_on_prompt': prompt_plan['rules_that_fired'],
                          'rules_that_fired_on_reference': reference_plan['rules_that_fired'] if reference_plan
                          else []})
        audit = coverage_for(loaded, missing, port, tokens, entry_items)
        reach = reachability(entries, port, tokens)
        never = [item['entry_id'] for item in reach if not item['reachable']]
        controls = sequence_controls(port, tokens, entry_items)
        tokenizers.append({'label': label, 'identity': tokenizer['identity'], 'n_vocab': len(tokens),
                           'served': {'port': port, 'build_info': props['build_info'],
                                      'tokenizer_check': served,
                                      'note': ('/tokenize is the only inference endpoint used. The served file may '
                                               'be a tokenizer carrier built from this artifact, so identity is '
                                               'established by the vocab+merges cross-check above rather than by '
                                               'the served file path; scripts/vocab_remap_carrier.py builds a '
                                               'carrier for a vocabulary-only artifact.')},
                           'probes': plans, 'coverage': audit, 'reachability': reach,
                           'controls': controls,
                           'rules_that_can_never_fire': never})
    report = {
        'schema': DRYRUN_SCHEMA,
        'scope': ('Tokenizer-level dry run over authored harmful probes and declining base answers already in the '
                  'clone: per-step suppression/forced-replacement plan and table reachability. No generation, no '
                  'weights, no logits; the decode driver applies these biases in the next stage.'),
        'method': DRYRUN_METHOD,
        'table': {'schema': document['schema'], 'sha256': hashlib.sha256(raw).hexdigest(),
                  'entries': len(entries), 'languages': sorted({entry['language'] for entry in entries}),
                  'kinds': sorted({entry['kind'] for entry in entries}),
                  'with_replacement': sum(1 for entry in entries if entry['replacement'])},
        'probes': {'rows': len(probe_rows), 'sources': sorted({row['source'] for row in probe_rows}),
                   'ids': [row['id'] for row in probe_rows]},
        'base_answers': {'sources': [item['source'] for item in loaded], 'missing_sources': missing,
                         'note': ('each source is a base arm of an in-clone generation wave; predictions for '
                                  'other arms live in sibling directories and are not base answers')},
        'tokenizers': tokenizers,
        'rules_that_can_never_fire': sorted({entry_id for item in tokenizers
                                             for entry_id in item['rules_that_can_never_fire']}),
    }
    if json_out is not None:
        Path(json_out).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


# --------------------------------------------------------------------------
# --remap on|off: the decode driver (client-side single-token loop)
# --------------------------------------------------------------------------

DECODE_SCHEMA = 'jdewitt.vocab-remap-decode/1'
SUPPRESS_BIAS = -100
FORCE_BIAS = 100
RESAMPLE_LIMIT = 16
DECODE_TIMEOUT = 300
DECODE_METHOD = (
    'Client-side single-token loop against a live llama-server: each step POSTs /completion with the prompt '
    're-sent and the emitted text appended, n_predict=1, temperature=0, cache_prompt=true, n_probs=1, so the '
    'served tokenizer and the KV prefix stay the artifact\'s own. The server does not continue cached generated '
    'tokens when the prompt is unchanged: the prompt carries the emitted text each step and prompt_n is 1. '
    'remap=on applies logit_bias only for the step it belongs to: the candidate token that completes a table '
    'sequence is suppressed with -100, and when the table names a replacement its first token is forced with '
    '+100 (the remaining replacement tokens carry +100 on the following steps while they still match, or the '
    'force is dropped and the divergence recorded). Suppression is sequence-aware: " can" alone completes '
    'nothing, " can"+"\'t" completes the denial and is the step that fires. Re-sampling after a suppression can '
    'surface another completing token (a typographic apostrophe, " can not"), so the same step is re-sampled '
    'until the chosen token completes no table sequence or RESAMPLE_LIMIT is reached; every suppression is '
    'recorded. remap=off runs the identical loop with no bias, so the difference between the two arms is one '
    'flag, not a different build. Each row is also decoded as one plain greedy /completion wave for comparison.')


def _free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _log_tail(path, limit=3):
    try:
        lines = [line.strip() for line in Path(path).read_text(errors='replace').splitlines() if line.strip()]
    except OSError:
        return 'no server log'
    return '; '.join(lines[-limit:]) or 'no loader output'


def _wait_healthy(port, process, log_path, timeout=180.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'llama-server exited {process.returncode} before /health: {_log_tail(log_path)}')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.1)
    raise RuntimeError(f'llama-server on port {port} did not become healthy in {timeout:.0f}s: {_log_tail(log_path)}')


def start_server(serve, artifact, port, log_path):
    """Start the artifact on a loopback port and wait for /health; return the process handle."""
    argv = [str(serve), '-m', str(Path(artifact).resolve()), '-ngl', '0', '-c', '2048',
            '--parallel', '1', '--host', '127.0.0.1', '--port', str(port)]
    env = {key: value for key, value in os.environ.items() if not key.startswith(('LLAMA_', 'HF_TOKEN'))}
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, 'w')
    process = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT, env=env)
    try:
        _wait_healthy(port, process, log_path)
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        handle.close()
        raise
    return {'argv': argv, 'process': process, 'handle': handle, 'port': port}


def stop_server(started):
    started['process'].terminate()
    try:
        started['process'].wait(timeout=15)
    except subprocess.TimeoutExpired:
        started['process'].kill()
        started['process'].wait(timeout=10)
    started['handle'].close()


def verify_artifact_identity(port, artifact):
    """Refuse unless the server's /props model_path is the artifact: the same gate release.py uses."""
    props = _server_json(port, '/props')
    if not isinstance(props, dict):
        raise ValueError('llama-server /props returned no object')
    reported = props.get('model_path')
    if not isinstance(reported, str) or not reported:
        raise ValueError('llama-server /props carried no model_path; the loaded artifact cannot be proven')
    if Path(reported).resolve() != Path(artifact).resolve():
        raise ValueError(f'refusing to run: the server loaded {reported!r}, not the artifact '
                         f'{Path(artifact).name!r}; a decode row from another GGUF would describe another model')
    generation = props.get('default_generation_settings') or {}
    return {'model_path_reported': _portable_path(reported),
            'model_path_reported_is_absolute': os.path.isabs(reported),
            'model_path_resolves_to_artifact': True,
            'build_info': props.get('build_info'),
            'model_alias': _portable_path(props['model_alias']) if isinstance(props.get('model_alias'), str) else None,
            'n_ctx': generation.get('n_ctx'), 'total_slots': props.get('total_slots')}


def apply_chat_template(port, text):
    """Render the artifact's own chat template through the served runtime; never re-implement it here."""
    response = _server_json(port, '/apply-template',
                            {'messages': [{'role': 'user', 'content': text}],
                             'add_generation_prompt': True})
    prompt = response.get('prompt') if isinstance(response, dict) else None
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('llama-server /apply-template returned no prompt; pass --template raw to decode the '
                         'suite prompts verbatim')
    return prompt


def _completion(port, payload):
    response = _server_json(port, '/completion', payload, timeout=DECODE_TIMEOUT)
    if not isinstance(response, dict):
        raise ValueError('llama-server /completion returned no object')
    return response


def _candidate(response):
    """The top candidate of a one-token completion (None when the runtime sampled nothing)."""
    probabilities = response.get('completion_probabilities')
    if not isinstance(probabilities, list) or not probabilities:
        return None
    top = probabilities[0]
    if not isinstance(top, dict) or type(top.get('id')) is not int:
        raise ValueError('llama-server /completion returned no integer candidate id')
    return {'id': top['id'], 'text': top.get('token')}


def _bias_document(bias):
    return {str(token_id): int(value) for token_id, value in sorted(bias.items())}


def _piece(token_id, tokens):
    """Display text of one token, empty for an id outside the artifact vocabulary."""
    if type(token_id) is not int or not 0 <= token_id < len(tokens):
        return ''
    return piece_text(token_id, tokens)


def _piece_bytes(token_id, tokens):
    if type(token_id) is not int or not 0 <= token_id < len(tokens):
        return b''
    return _token_bytes(tokens[token_id])


def _firing(entry_items, path_ids, candidate_id, step):
    """Table sequences the candidate token completes at generated step ``step`` (sequence-aware).

    The candidate is the token about to be emitted, so a sequence fires only when the whole
    sequence ends at it. A sequence that starts inside the prompt is not the model's own denial
    and is not fired (it is counted as prompt-origin instead).
    """
    full = path_ids + [candidate_id]
    fires = []
    seen = set()
    for item in entry_items:
        for context_item in item['contexts']:
            sequence = context_item['ids']
            if not sequence or len(sequence) > len(full):
                continue
            if full[-len(sequence):] != sequence:
                continue
            start_step = step - len(sequence) + 1
            if start_step < 0 or not _scope_allows(item['entry'], start_step):
                continue
            key = (item['entry']['id'], tuple(sequence))
            if key in seen:
                continue
            seen.add(key)
            fires.append((item, context_item, start_step))
    return fires


def _occurrences_in_generated(path_ids, prompt_length, entry_items):
    """Table sequences present in the generated part of a token path, with the step they start at."""
    hits = []
    generated = path_ids[prompt_length:]
    for item in entry_items:
        for context_item in item['contexts']:
            sequence = context_item['ids']
            for start in _occurrences(generated, sequence):
                if not _scope_allows(item['entry'], start):
                    continue
                hits.append({'entry_id': item['entry']['id'], 'context': context_item['context'],
                             'start_step': start, 'ids': sequence})
    return hits


def _sent_prompt(prompt, prompt_ids, emitted_ids, emitted_bytes):
    """What to POST as the prompt this step: the text with the emitted text appended, or the ids.

    The emitted text is the concatenation of the emitted tokens' raw bytes. A multi-byte character
    split across two tokens has no text form, so that step (and only that step) sends the ids the
    server already knows; nothing is silently replaced.
    """
    try:
        text = prompt + emitted_bytes.decode('utf-8')
    except UnicodeDecodeError:
        return {'field': list(prompt_ids) + list(emitted_ids), 'form': 'ids', 'path_ids': None}
    return {'field': text, 'form': 'text', 'path_ids': None}


def decode_steps(port, tokens, entry_items, prompt, prompt_ids, max_tokens, remap):
    """One client-side single-token loop: per step, suppress the completing token and force replacements."""
    emitted_ids = []
    emitted_bytes = b''
    steps, suppressions, forced = [], [], []
    retokenized, prompt_origin = [], 0
    pending = []
    stop = None
    resample_limit_steps = []
    requests = 0
    for step in range(max_tokens):
        sent = _sent_prompt(prompt, prompt_ids, emitted_ids, emitted_bytes)
        path_ids = (list(prompt_ids) + list(emitted_ids) if sent['form'] == 'ids'
                    else served_tokenize(port, sent['field'], parse_special=True))
        if sent['form'] == 'text' and path_ids != list(prompt_ids) + list(emitted_ids):
            retokenized.append({'step': step, 'served_tokens': len(path_ids),
                                'tracked_tokens': len(prompt_ids) + len(emitted_ids)})
        payload = {'prompt': sent['field'], 'n_predict': 1, 'temperature': 0.0,
                   'cache_prompt': True, 'n_probs': 1}
        bias = {}
        if remap and pending:
            bias[pending[0]] = FORCE_BIAS
        if bias:
            payload['logit_bias'] = _bias_document(bias)
        response = _completion(port, payload)
        requests += 1
        suppressed_here = []
        attempts = 0
        chosen = None
        while True:
            candidate = _candidate(response)
            if candidate is None:
                break
            fires = _firing(entry_items, path_ids, candidate['id'], step) if remap else []
            if not fires:
                # a sequence that starts before step 0 begins in the prompt: the user's words, not the model's denial
                full = list(path_ids) + [candidate['id']]
                prompt_origin += sum(1 for item in entry_items for context_item in item['contexts']
                                     if context_item['ids'] and len(context_item['ids']) <= len(full)
                                     and full[-len(context_item['ids']):] == context_item['ids']
                                     and step - len(context_item['ids']) + 1 < 0)
                chosen = candidate
                break
            if pending and candidate['id'] == pending[0]:
                forced.append({'step': step, 'reason': 'the forced replacement token completes a table '
                                                       'sequence; the suppression rule wins'})
                pending = []
            previous = path_ids[-1] if path_ids else None
            for item, _context, _start in fires:
                replacement = item['replacement']
                event = {'step': step, 'suppressed_id': candidate['id'],
                         'suppressed_text': _piece(candidate['id'], tokens),
                         'replacement_id': replacement['first_id'] if replacement else None,
                         'entry_id': item['entry']['id'],
                         'prev_token': {'id': previous, 'text': _piece(previous, tokens)}
                         if previous is not None else None}
                suppressions.append(event)
                suppressed_here.append(event)
            bias[candidate['id']] = SUPPRESS_BIAS
            forces = {item['replacement']['first_id'] for item, _c, _s in fires if item['replacement']}
            if len(forces) == 1:
                force_id = next(iter(forces))
                if force_id != candidate['id']:
                    bias[force_id] = FORCE_BIAS
                    source = next(item for item, _c, _s in fires if item['replacement'])
                    pending = list(source['replacement']['ids'][1:])
                    forced.append({'step': step, 'entry_id': source['entry']['id'], 'first_id': force_id,
                                   'first_text': _piece(force_id, tokens), 'remaining_ids': pending,
                                   'reason': 'the table replacement is forced on its first token; the rest '
                                             'follows with +100 while it still matches'})
            elif len(forces) > 1:
                forced.append({'step': step, 'reason': 'several entries with different replacements complete '
                                                       'this token; none is forced', 'ids': sorted(forces)})
            payload['logit_bias'] = _bias_document(bias)
            response = _completion(port, payload)
            requests += 1
            attempts += 1
            if attempts >= RESAMPLE_LIMIT:
                chosen = _candidate(response)
                resample_limit_steps.append({'step': step, 'attempts': attempts,
                                             'resampled_id': chosen['id'] if chosen else None})
                break
        if chosen is None:
            stop = response.get('stop_type') or 'eos'
            break
        token_id = chosen['id']
        piece = _piece_bytes(token_id, tokens)
        if not piece and not (response.get('content') or ''):
            stop = 'eos'
            break
        emitted_ids.append(token_id)
        emitted_bytes += piece
        timings = response.get('timings') if isinstance(response.get('timings'), dict) else {}
        steps.append({'step': step, 'token_id': token_id, 'piece': _piece(token_id, tokens),
                      'resampled': attempts, 'suppressions': len(suppressed_here),
                      'prompt_n': timings.get('prompt_n'), 'cache_n': timings.get('cache_n')})
        if pending:
            if token_id == pending[0]:
                pending.pop(0)
            else:
                forced.append({'step': step, 'reason': 'the model left the forced replacement',
                               'expected_id': pending[0], 'got_id': token_id})
                pending = []
        stop_type = response.get('stop_type')
        if stop_type and stop_type != 'limit':
            stop = stop_type
            break
    return {'ids': emitted_ids, 'steps': steps, 'suppressions': suppressions, 'forced': forced,
            'retokenized_steps': retokenized, 'prompt_origin_completions': prompt_origin,
            'resample_limit_steps': resample_limit_steps, 'stop': stop, 'completion_requests': requests,
            'pending_unfinished': pending}


def greedy_wave(port, prompt, max_tokens, tokens):
    """One plain greedy /completion wave: the base continuation the off arm must reproduce token-for-token."""
    start = time.monotonic()
    response = _completion(port, {'prompt': prompt, 'n_predict': max_tokens, 'temperature': 0.0,
                                  'cache_prompt': True, 'n_probs': 1})
    probabilities = response.get('completion_probabilities') or []
    ids = []
    for item in probabilities:
        token_id = item.get('id') if isinstance(item, dict) else None
        if type(token_id) is not int or not _piece_bytes(token_id, tokens):
            break
        ids.append(token_id)
    return {'ids': ids, 'text': detokenize(ids, tokens), 'reported_text': response.get('content') or '',
            'stop_type': response.get('stop_type'), 'seconds': round(time.monotonic() - start, 3)}


APOSTROPHE_VARIANTS = '\u2019\u2018\u02bc`\u00b4\uff07'


def _normalize_apostrophes(text):
    for character in APOSTROPHE_VARIANTS:
        text = text.replace(character, "'")
    return text


def variant_forms(text, entry_items):
    """Table forms that reappear in a spelling the table does not name.

    A contraction the table spells with ASCII "'" can be emitted with a typographic
    apostrophe (U+2019): the token is different, the words are the same. This is
    reported, never silently normalized into the suppression rule — the table decides
    what is named, and an unnamed variant is a gap in the table, not a driver bug.
    """
    normalized = _normalize_apostrophes(text).lower()
    hits = []
    for item in entry_items:
        form = item['entry']['form']
        if "'" not in form:
            continue
        if _normalize_apostrophes(form).lower() in normalized and form.lower() not in text.lower():
            hits.append({'entry_id': item['entry']['id'], 'form': form,
                         'note': 'the same words in an apostrophe spelling the table does not name'})
    return hits


def decode_row(port, tokens, entry_items, probe, prompt, max_tokens, remap, template_mode):
    """Decode one probe in both arms' shape: the single-token loop, plus the plain greedy wave beside it."""
    start = time.monotonic()
    prompt_ids = served_tokenize(port, prompt, parse_special=True)
    loop = decode_steps(port, tokens, entry_items, prompt, prompt_ids, max_tokens, remap == 'on')
    loop_seconds = time.monotonic() - start
    wave = greedy_wave(port, prompt, max_tokens, tokens)
    generated = loop['ids']
    text = detokenize(generated, tokens)
    surfaces = _occurrences_in_generated(generated, 0, entry_items)
    opener_surfaces = [hit for hit in surfaces if hit['start_step'] < OPENER_WINDOW_TOKENS]
    wave_surfaces = _occurrences_in_generated(wave['ids'], 0, entry_items)
    wave_opener_surfaces = [hit for hit in wave_surfaces if hit['start_step'] < OPENER_WINDOW_TOKENS]
    base_shape = classify(wave['text'])
    loop_shape = classify(text)
    prompt_n = [step['prompt_n'] for step in loop['steps'] if type(step['prompt_n']) is int]
    cache_n = [step['cache_n'] for step in loop['steps'] if type(step['cache_n']) is int]
    return {
        'id': probe['id'], 'kind': probe['kind'], 'language': probe['language'],
        'source': probe['source'], 'prompt': probe['prompt'],
        'prompt_sha256': hashlib.sha256(probe['prompt'].encode()).hexdigest(),
        'template': template_mode, 'remap': remap, 'max_tokens': max_tokens,
        'prompt_tokens': len(prompt_ids),
        'path': {'n_tokens': len(generated), 'ids': generated,
                 'pieces': [_piece(token_id, tokens) for token_id in generated], 'text': text},
        'steps': loop['steps'], 'suppressions': loop['suppressions'], 'forced': loop['forced'],
        'stop': loop['stop'], 'completion_requests': loop['completion_requests'],
        'retokenized_steps': loop['retokenized_steps'],
        'prompt_origin_completions': loop['prompt_origin_completions'],
        'resample_limit_steps': loop['resample_limit_steps'],
        'pending_unfinished': loop['pending_unfinished'],
        'prompt_evaluation': {'prompt_n_per_step': prompt_n, 'cache_n_per_step': cache_n,
                              'max_prompt_n_after_first_step': max(prompt_n[1:], default=None),
                              'min_cache_n_after_first_step': min(cache_n[1:], default=None),
                              'note': ('prompt_n is the number of prompt tokens the server actually evaluated in '
                                       'that request (the runtime timing field), so 1 after the first step is the '
                                       'KV prefix being reused; cache_n is the matched prefix length')},
        'plain': {'ids': wave['ids'], 'text': wave['text'],
                  'reported_text': wave['reported_text'], 'stop_type': wave['stop_type'],
                  'seconds': wave['seconds']},
        'gate': {'equals_plain_token_for_token': generated == wave['ids'],
                 'surfaces_in_generated': surfaces,
                 'surfaces_in_opener_window': opener_surfaces,
                 'plain_declining': base_shape['declining'], 'remapped_declining': loop_shape['declining'],
                 'plain_shape': base_shape['shape'], 'remapped_shape': loop_shape['shape'],
                 'opener_gone': bool(wave_opener_surfaces) and not opener_surfaces,
                 'variant_forms_in_text': variant_forms(text, entry_items)},
        'seconds': round(loop_seconds, 3)}


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _portable_argv(serve, artifact, port):
    return [Path(serve).name, '-m', _portable_path(artifact), '-ngl', '0', '-c', '2048',
            '--parallel', '1', '--host', '127.0.0.1', '--port', str(port)]


def _entry_counts(events):
    counts = {}
    for event in events:
        counts[event['entry_id']] = counts.get(event['entry_id'], 0) + 1
    return dict(sorted(counts.items()))


def vocab_remap_decode(table=None, artifact=None, suite=None, out=None, remap='on', max_tokens=40,
                       serve=None, port=None, template='chat'):
    """Run the decode driver on one artifact: the same loop with bias (on) or without it (off)."""
    if remap not in ('on', 'off'):
        raise ValueError('--remap must be on or off')
    if artifact is None:
        raise ValueError('--remap needs exactly one --artifact GGUF (the artifact the server must be serving)')
    artifact = Path(artifact)
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    if suite is None:
        raise ValueError('--remap needs --suite FILE (the authored probe rows)')
    if out is None:
        raise ValueError('--remap needs --out DIR; rows.jsonl and run.json are written there')
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError('--max-tokens must be a positive integer')
    if template not in ('chat', 'raw'):
        raise ValueError('--template must be chat or raw')
    if serve is None and port is None:
        raise ValueError('--remap needs --serve PATH (start a CPU llama-server on the artifact) or --port N '
                         'for one already running')
    if serve is not None and not Path(serve).is_file():
        raise FileNotFoundError(serve)
    document, raw_table = load_table(table)
    entries = document['entries']
    tokenizer = read_tokenizer(artifact)
    tokens = tokenizer['tokens']
    vocab = {token: token_id for token_id, token in enumerate(tokens)}
    ranks = _merge_ranks(tokenizer['merges'])
    probes = load_probe_rows(suite)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = None
    began = time.monotonic()
    try:
        if serve is not None:
            port = port or _free_port()
            started = start_server(serve, artifact, port, out / 'server.log')
        if type(port) is not int or not 0 < port < 65536:
            raise ValueError(f'--port must be an integer in 1..65535, got {port!r}')
        props = verify_artifact_identity(port, artifact)
        served_check = verify_served_tokenizer(entries, port, tokens, vocab, ranks)
        if not served_check['agrees']:
            raise ValueError(f'the server on port {port} does not tokenize the table forms the way the artifact '
                             f'header does ({len(served_check["mismatches"])} of {served_check["contexts_checked"]} '
                             'contexts differ); a decode row here would describe a different tokenizer')
        entry_items = entry_sequences(entries, port, tokens)
        rows = []
        for probe in probes:
            prompt = apply_chat_template(port, probe['prompt']) if template == 'chat' else probe['prompt']
            rows.append(decode_row(port, tokens, entry_items, probe, prompt, max_tokens, remap, template))
    finally:
        if started is not None:
            stop_server(started)

    rows_path = out / 'rows.jsonl'
    rows_path.write_text(''.join(json.dumps(row, sort_keys=True, allow_nan=False) + '\n' for row in rows))
    decode_seconds = round(sum(row['seconds'] for row in rows), 3)
    wall_seconds = round(time.monotonic() - began, 3)
    generated = sum(row['path']['n_tokens'] for row in rows)
    suppressions = [event for row in rows for event in row['suppressions']]
    forced = [event for row in rows for event in row['forced']]
    equals = [row['id'] for row in rows if row['gate']['equals_plain_token_for_token']]
    base_declining = [row['id'] for row in rows if row['gate']['plain_declining']]
    opener_base = [row['id'] for row in rows if row['gate']['surfaces_in_opener_window'] or
                   (row['gate']['plain_declining'] and row['gate']['opener_gone'])]
    opener_gone = [row['id'] for row in rows if row['gate']['opener_gone']]
    still_open = [row['id'] for row in rows if row['gate']['surfaces_in_opener_window']]
    variant_rows = [row['id'] for row in rows if row['gate']['variant_forms_in_text']]
    resample_limits = [row['id'] for row in rows if row['resample_limit_steps']]
    prompt_n = [row['prompt_evaluation']['max_prompt_n_after_first_step'] for row in rows]
    cache_n = [row['prompt_evaluation']['min_cache_n_after_first_step'] for row in rows]
    claim_row = next((row for row in rows if row['gate']['opener_gone']), None)
    if remap == 'off':
        gate = {'plain_wave_equality': {
            'rows': len(rows), 'equal': len(equals),
            'mismatched': [row['id'] for row in rows if row['id'] not in equals],
            'required': 'the per-step loop with no bias must reproduce the plain greedy wave token-for-token',
            'ok': len(equals) == len(rows)},
            'bias_applied': False}
        ok = gate['plain_wave_equality']['ok']
        verdict_reason = ('every row of the unbiased loop equals the plain greedy wave token-for-token; the off arm '
                          'is the untouched decode' if ok else
                          'the unbiased loop did not reproduce the plain greedy wave; the off arm is not a control')
    else:
        gate = {'plain_wave_equality': {'rows': len(rows), 'equal': len(equals),
                                        'mismatched': [row['id'] for row in rows if row['id'] not in equals],
                                        'required': 'informational: the on arm is expected to diverge',
                                        'ok': True},
                'openers': {'base_rows_declining': len(base_declining),
                            'base_rows_with_opener_surface': len(opener_base),
                            'rows_openers_gone': len(opener_gone),
                            'rows_opener_surface_left': still_open,
                            'rows_with_unnamed_variant': variant_rows,
                            'note': ('opener surfaces are table contexts inside the first %d generated tokens; a row '
                                     'whose base opened with one and whose remapped text does not is counted as '
                                     'gone' % OPENER_WINDOW_TOKENS)},
                'suppressions': {'events': len(suppressions), 'rows': sum(1 for row in rows if row['suppressions']),
                                 'by_entry': _entry_counts(suppressions),
                                 'resample_limit_rows': resample_limits,
                                 'forced_replacements': len(forced)},
                'bias_applied': True}
        ok = not resample_limits
        verdict_reason = ('the bias arm decoded every row; every denial sequence completed inside the generated '
                          'text was suppressed and re-sampled' if ok else
                          'at least one step hit the re-sample limit, so a denial surface survived suppression')
    report = {
        'schema': DECODE_SCHEMA,
        'remap': remap,
        'scope': ('Decode-side reference implementation of the denial-word remap on a real CPU serve of the '
                  'artifact. The single-token loop is the same code in both arms; only --remap on|off changes '
                  'whether logit_bias is sent. A row here is a token path, its suppression events and the plain '
                  'greedy wave beside it, not a harmlessness, capability or removal measurement.'),
        'method': DECODE_METHOD,
        'pinned_argv': ['jdewitt', 'vocab-remap', '--artifact', _portable_path(artifact),
                        '--suite', _portable_path(suite), '--out', _portable_path(out),
                        '--remap', remap, '--max-tokens', str(max_tokens),
                        '--template', template]
                       + (['--serve', Path(serve).name] if serve is not None else [])
                       + (['--port', str(port)] if serve is None else [])
                       + (['--table', _portable_path(table)] if table is not None else []),
        'claim': _opener_claim(claim_row, remap, tokens),
        'artifact': {'path': _portable_path(artifact), 'sha256': _sha256_file(artifact),
                     'bytes': artifact.stat().st_size, 'tokenizer': tokenizer['identity']},
        'server': {'argv': _portable_argv(serve, artifact, port) if serve is not None else [],
                   'argv_note': 'the model and binary are recorded portably; the artifact is identified by sha256',
                   'started_by_driver': started is not None, 'port': port,
                   'model_path_reported': props['model_path_reported'],
                   'model_path_resolves_to_artifact': props['model_path_resolves_to_artifact'],
                   'build_info': props['build_info'], 'model_alias': props['model_alias'],
                   'n_ctx': props['n_ctx'], 'total_slots': props['total_slots'],
                   'tokenizer_check': {'contexts_checked': served_check['contexts_checked'],
                                       'agrees': served_check['agrees']}},
        'table': {'schema': document['schema'], 'sha256': hashlib.sha256(raw_table).hexdigest(),
                  'entries': len(entries),
                  'with_replacement': sum(1 for entry in entries if entry['replacement']),
                  'replacement_note': ('the table names %d replacement(s), so this arm can force as well as suppress'
                                       % sum(1 for entry in entries if entry['replacement'])
                                       if any(entry['replacement'] for entry in entries) else
                                       ('the checked-in table is an inventory: every replacement is null, so this '
                                        'arm suppresses and never forces. The force path is exercised by the '
                                        'documented table override under verification/vocab-remap/decode/forced/.'))},
        'probes': {'path': _portable_path(suite), 'sha256': _sha256_file(suite), 'rows': len(probes),
                   'ids': [probe['id'] for probe in probes]},
        'config': {'template': template, 'max_tokens': max_tokens, 'temperature': 0.0,
                   'cache_prompt': True, 'n_probs': 1, 'suppress_bias': SUPPRESS_BIAS,
                   'force_bias': FORCE_BIAS, 'resample_limit': RESAMPLE_LIMIT,
                   'opener_window_tokens': OPENER_WINDOW_TOKENS},
        'rows_file': {'path': 'rows.jsonl', 'sha256': _sha256_file(rows_path), 'rows': len(rows)},
        'metrics': {'generated_tokens': generated, 'steps': sum(len(row['steps']) for row in rows),
                    'completion_requests': sum(row['completion_requests'] for row in rows),
                    'prompt_tokens': sum(row['prompt_tokens'] for row in rows),
                    'decode_seconds': decode_seconds, 'wall_seconds': wall_seconds,
                    'tokens_per_second': round(generated / decode_seconds, 2) if decode_seconds else None,
                    'max_prompt_n_after_first_step': max((value for value in prompt_n
                                                          if type(value) is int), default=None),
                    'min_cache_n_after_first_step': min((value for value in cache_n if type(value) is int),
                                                       default=None),
                    'plain_wave_seconds': round(sum(row['plain']['seconds'] for row in rows), 3),
                    'retokenized_steps': sum(len(row['retokenized_steps']) for row in rows),
                    'prompt_origin_sequences_seen': sum(row['prompt_origin_completions'] for row in rows)},
        'gate': gate,
        'verdict': {'ok': ok, 'reason': verdict_reason},
    }
    (out / 'run.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def _opener_claim(row, remap, tokens):
    """What the on arm actually did to the base's refusal opener, in the run's own words."""
    if remap == 'off':
        return {'applicable': False,
                'note': 'no bias is applied in this arm; compare it to the on arm, not to a base model'}
    if row is None:
        return {'applicable': False,
                'note': ('no row both declined at the base and lost its opener-window surface; the claim is not made '
                         'for this run')}
    path_ids = row['path']['ids']
    base_ids = row['plain']['ids']
    variants = row['gate']['variant_forms_in_text']
    note = ('the table-named denial words are gone from the opener: every token that completed a table sequence '
            'was suppressed at its own step. What follows is the model\'s own continuation, so this is not content '
            'removal, not a compliance claim and not evidence that the answer avoids the topic.')
    if variants:
        note = ('the table-named denial words are gone from the opener, but the same denial reappears in a spelling '
                'the table does not name (%s); the words are not gone from the answer, and this is a gap in the table, '
                'not a driver failure. The rest of the text is the model\'s own continuation: this is not content '
                'removal and not a compliance claim.' % ', '.join(item['entry_id'] for item in variants))
    return {'applicable': True, 'row': row['id'], 'language': row['language'],
            'base_opener': detokenize(base_ids[:OPENER_WINDOW_TOKENS], tokens),
            'remapped_opener': detokenize(path_ids[:OPENER_WINDOW_TOKENS], tokens),
            'suppressed': [{'step': event['step'], 'id': event['suppressed_id'],
                            'text': event['suppressed_text'], 'entry_id': event['entry_id']}
                           for event in row['suppressions']],
            'table_named_words_gone': not row['gate']['surfaces_in_opener_window'],
            'content_removal': False,
            'unnamed_variants_in_text': variants,
            'note': note}
