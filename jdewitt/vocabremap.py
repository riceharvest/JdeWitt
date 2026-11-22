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
"""

import hashlib
import json
import struct
from pathlib import Path

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
