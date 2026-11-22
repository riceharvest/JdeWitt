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
import struct
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
    'opener coverage restricts the occurrence to start before step %d. The prompt plan runs the same matcher '
    'over the prompt token path; those positions are descriptive because prompt tokens are not generated, and the '
    'reference plan (an authored declining continuation on each probe) is the decode-step schedule the driver '
    'would follow if the model emitted that text.' % OPENER_WINDOW_TOKENS)


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


def served_tokenize(port, text):
    """Token ids for one string, exactly as the served tokenizer splits it."""
    payload = _server_json(port, '/tokenize', {'content': text, 'add_special': False, 'parse_special': False})
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
                end = start + len(sequence) - 1
                key = (entry['id'], start, end)
                if key in seen:
                    continue
                seen.add(key)
                replacement = item['replacement']
                events.append({'entry_id': entry['id'], 'context': context_item['context'],
                               'start_step': start, 'end_step': end, 'window': [start, end],
                               'suppress_at_step': end, 'suppressed_id': sequence[-1],
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
                             'occurs_in_carrier': bool(_occurrences(carrier_ids, ids))})
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
            hits = []
            opener_hit = False
            for entry_item in entry_items:
                for context_item in entry_item['contexts']:
                    starts = _occurrences(ids, context_item['ids'])
                    if starts:
                        hits.append(entry_item['entry']['id'])
                        matched_by_entry[entry_item['entry']['id']] = matched_by_entry.get(entry_item['entry']['id'], 0) + 1
                        if any(start < OPENER_WINDOW_TOKENS for start in starts):
                            opener_hit = True
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
        tokenizers.append({'label': label, 'identity': tokenizer['identity'], 'n_vocab': len(tokens),
                           'served': {'port': port, 'build_info': props['build_info'],
                                      'tokenizer_check': served,
                                      'note': ('/tokenize is the only inference endpoint used. The served file may '
                                               'be a tokenizer carrier built from this artifact, so identity is '
                                               'established by the vocab+merges cross-check above rather than by '
                                               'the served file path; scripts/vocab_remap_carrier.py builds a '
                                               'carrier for a vocabulary-only artifact.')},
                           'probes': plans, 'coverage': audit, 'reachability': reach,
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
        'base_answers': {'sources': [item['source'] for item in loaded], 'missing_sources': missing},
        'tokenizers': tokenizers,
        'rules_that_can_never_fire': sorted({entry_id for item in tokenizers
                                             for entry_id in item['rules_that_can_never_fire']}),
    }
    if json_out is not None:
        Path(json_out).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
