# SPDX-License-Identifier: AGPL-3.0-or-later
"""Delete denial-opener ids from a quantized artifact's vocabulary, renumbering down.

The conjugate card exchanges a denial row with a compliance row; this card removes
the denial token from the file. For a chosen set of token ids it

  * drops the id's entry from ``tokenizer.ggml.tokens`` and
    ``tokenizer.ggml.token_type``,
  * renumbers every surviving id down by the number of deleted ids below it, which
    also renumbers the scalar ``*_token_id`` fields (bos/eos/padding) that point
    into the same id space,
  * deletes the corresponding rows of the token-row matrix (``token_embd.weight``
    and, when the head is untied, ``output.weight``), preserving the raw stored
    bytes of every surviving row and changing only the row count in the dims.

Every other KV entry keeps its exact encoding, every other tensor keeps its
payload, and only the tensor offsets after the shrunken matrix move (recomputed
under the file's own alignment). A sidecar holds the deleted entries and row
bytes, so ``revert`` re-inserts them and reproduces the pristine file byte for
byte.

Two limits are made explicit rather than hidden:

  * A merged token whose bytes survive in the byte alphabet is still *spellable*
    after deletion, as several tokens: the byte-level BPE merges still build the
    string, the vocabulary lookup misses, and llama.cpp falls back to the tokens of
    its raw bytes. Deletion removes the single-token spelling, not the string.
  * Deleting a byte-alphabet entry (a token that IS one raw byte, e.g. ``I``)
    makes every string containing that byte unrepresentable, which is a broken
    tokenizer for the whole file, not a targeted removal. Single-byte ids are
    refused unless ``--include-byte-alphabet`` asks for them.

Storage fidelity only: no model is loaded, no logits or generations are read, and
``removal_measured`` is false. Tokenization, tensor algebra and file bytes.
"""
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np

from .vocabremap import (BYTE_TO_UNICODE, _merge_ranks, _portable_path, _sha256_file,
                         bpe_symbols, read_tokenizer, resolve)
from .vocabremap_weights import _vocab_role, discover_vocab_tensors

SCHEMA = 'jdewitt.vocab-delete/1'
REVERT_SCHEMA = 'jdewitt.vocab-delete-revert/1'
PAIRS_SCHEMA = 'jdewitt.vocab-conjugate-pairs/1'
PACKAGED_TABLE = Path(__file__).with_name('assets') / 'vocab-conjugate-openers.json'
CONTEXTS = ('bare', 'leads_with_space')
GGUF_MAGIC = b'GGUF'
SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
TYPE_STRING, TYPE_ARRAY = 8, 9
UINT32, INT32 = 4, 5
NORMAL = 1
PROBES = ('Sorry', 'Sorry,', 'I', ' I', 'Cannot', 'Cannot,', "I'm sorry, but I can't",
          'Es', ' Es', 'Sure', 'Here', 'Certainly', 'Hier')

METHOD = (
    'The denial surfaces of the opener table are resolved against the artifact tokenizer in both contexts '
    '(bare and with the leading space); every id a surface resolves to in a single token is a deletion '
    'candidate. A candidate is deleted when its stored token_type is NORMAL and, unless the byte-alphabet arm '
    'is asked for, its piece is more than one raw byte (a single raw byte IS the tokenizer\'s alphabet). The '
    'write rebuilds the GGUF container: tokenizer.ggml.tokens and tokenizer.ggml.token_type lose the entries, '
    'every scalar *_token_id is renumbered down by the number of deleted ids below it, and each '
    'vocabulary-wide token-row matrix loses the rows (surviving rows are copied as whole stored rows). Every '
    'other KV entry and tensor payload is copied verbatim; tensor offsets are recomputed under the file '
    'alignment. The revert sidecar stores the deleted entries and raw row bytes and re-inserts them. No model '
    'is loaded and no logits are read; the encoder paths are string automata (llama.cpp\'s byte-level BPE plus '
    'byte fallback) over the metadata.')
GATED = ('Whether the model then refuses less is out of scope: that needs a serve/forward pass, which this '
         'card forbids. Even the static result is limited — a merged token whose bytes survive is still '
         'spellable as a sequence, so the model can emit the denial string in several tokens.')


# --- the container ---------------------------------------------------------

def _u32(buffer, offset):
    return struct.unpack_from('<I', buffer, offset)[0]


def _u64(buffer, offset):
    return struct.unpack_from('<Q', buffer, offset)[0]


def _skip_value(handle, value_type):
    """Consume one GGUF value from the current position; returns its byte size."""
    if value_type in SCALAR_SIZES:
        handle.seek(SCALAR_SIZES[value_type], 1)
        return SCALAR_SIZES[value_type]
    if value_type == TYPE_STRING:
        length = _u64(handle.read(8), 0)
        handle.seek(length, 1)
        return 8 + length
    if value_type == TYPE_ARRAY:
        element_type = _u32(handle.read(4), 0)
        count = _u64(handle.read(8), 0)
        if element_type in SCALAR_SIZES:
            size = SCALAR_SIZES[element_type] * count
            handle.seek(size, 1)
            return 12 + size
        if element_type == TYPE_STRING:
            total = 12
            for _ in range(count):
                length = _u64(handle.read(8), 0)
                handle.seek(length, 1)
                total += 8 + length
            return total
        raise ValueError(f'array of unsupported element type {element_type}')
    raise ValueError(f'unsupported GGUF value type {value_type}')


def _scan(path):
    """The KV entries (with their exact bytes), tensor infos and data base."""
    path = Path(path)
    with open(path, 'rb') as handle:
        if handle.read(4) != GGUF_MAGIC:
            raise ValueError(f'{path} is not a GGUF file')
        version = _u32(handle.read(4), 0)
        if version != 3:
            raise ValueError(f'{path} is GGUF version {version}; this writer handles 3')
        tensor_count = _u64(handle.read(8), 0)
        kv_count = _u64(handle.read(8), 0)
        entries = []
        for _ in range(kv_count):
            start = handle.tell()
            key_length = _u64(handle.read(8), 0)
            key = handle.read(key_length).decode('utf-8')
            value_type = _u32(handle.read(4), 0)
            value_start = handle.tell()
            _skip_value(handle, value_type)
            end = handle.tell()
            handle.seek(start)
            blob = handle.read(end - start)
            entries.append({'key': key, 'type': value_type, 'blob': blob,
                            'value_offset': value_start - start})
            handle.seek(end)
        tensors = []
        for _ in range(tensor_count):
            name_length = _u64(handle.read(8), 0)
            name = handle.read(name_length).decode('utf-8')
            n_dims = _u32(handle.read(4), 0)
            dims = list(struct.unpack('<' + 'Q' * n_dims, handle.read(8 * n_dims)))
            qtype = _u32(handle.read(4), 0)
            offset = _u64(handle.read(8), 0)
            tensors.append({'name': name, 'dims': dims, 'qtype': qtype, 'offset': offset})
        info_end = handle.tell()
    alignment = 32
    entry_of = {entry['key']: entry for entry in entries}
    if 'general.alignment' in entry_of and entry_of['general.alignment']['type'] == UINT32:
        alignment = _scalar(entry_of['general.alignment'])
    data_base = info_end + (-info_end) % alignment
    return {'path': path, 'version': version, 'alignment': alignment, 'entries': entries,
            'entry_of': entry_of, 'tensors': tensors, 'info_end': info_end,
            'data_base': data_base, 'size': path.stat().st_size}


def _payload(entry):
    """The value bytes of one KV entry (after key and type)."""
    return entry['blob'][entry['value_offset']:]


def _strings(entry):
    payload = _payload(entry)
    element_type, count = _u32(payload, 0), _u64(payload, 4)
    if entry['type'] != TYPE_ARRAY or element_type != TYPE_STRING:
        raise ValueError(f'{entry["key"]} is not an array of strings')
    values, offset = [], 12
    for _ in range(count):
        length = _u64(payload, offset)
        offset += 8
        values.append(payload[offset:offset + length].decode('utf-8'))
        offset += length
    if offset != len(payload):
        raise ValueError(f'{entry["key"]}: {len(payload) - offset} unconsumed bytes')
    return values


def _ints(entry):
    payload = _payload(entry)
    element_type, count = _u32(payload, 0), _u64(payload, 4)
    if entry['type'] != TYPE_ARRAY or element_type not in (UINT32, INT32):
        raise ValueError(f'{entry["key"]} is not an array of 32-bit integers')
    if len(payload) != 12 + 4 * count:
        raise ValueError(f'{entry["key"]}: payload is {len(payload)} bytes for {count} int32 values')
    return np.frombuffer(payload, dtype='<i4', count=count, offset=12)


def _scalar(entry):
    payload = _payload(entry)
    if entry['type'] == UINT32:
        return struct.unpack_from('<I', payload, 0)[0]
    if entry['type'] == INT32:
        return struct.unpack_from('<i', payload, 0)[0]
    raise ValueError(f'{entry["key"]} is not a 32-bit scalar')


def _encode_string_array(values):
    parts = [struct.pack('<I', TYPE_STRING), struct.pack('<Q', len(values))]
    for value in values:
        raw = value.encode('utf-8')
        parts.append(struct.pack('<Q', len(raw)))
        parts.append(raw)
    return b''.join(parts)


def _encode_int32_array(values):
    array = np.asarray(values, dtype='<i4')
    return struct.pack('<I', INT32) + struct.pack('<Q', len(array)) + array.tobytes()


def _entry_blob(key, value_type, value_bytes):
    raw = key.encode('utf-8')
    return struct.pack('<Q', len(raw)) + raw + struct.pack('<I', value_type) + value_bytes


def _tensor_bytes(tensor):
    from .ggufwrite import tensor_nbytes

    return tensor_nbytes(tensor['dims'], tensor['qtype'])


def _write_container(path, scanned, entries, tensors, replacements, source):
    """Write a new GGUF: new KV blobs, new tensor infos, payloads from ``source``.

    ``replacements`` maps a tensor name to a writer ``(out, source_handle) -> bytes``;
    every other tensor payload is streamed from the source. Offsets are recomputed
    under the file's own alignment, so a file with an unchanged layout round-trips
    byte for byte.
    """
    path = Path(path)
    alignment = scanned['alignment']
    layout, offset = [], 0
    for tensor in tensors:
        size = tensor['bytes'] if tensor['name'] in replacements else _tensor_bytes(tensor)
        layout.append((tensor, offset, size))
        offset += size
        offset += (-offset) % alignment
    with open(path, 'wb') as out, open(source, 'rb') as handle:
        out.write(GGUF_MAGIC)
        out.write(struct.pack('<I', scanned['version']))
        out.write(struct.pack('<Q', len(tensors)))
        out.write(struct.pack('<Q', len(entries)))
        for blob in entries:
            out.write(blob)
        for tensor, _offset, _size in layout:
            name = tensor['name'].encode('utf-8')
            out.write(struct.pack('<Q', len(name)) + name)
            out.write(struct.pack('<I', len(tensor['dims'])))
            out.write(struct.pack('<' + 'Q' * len(tensor['dims']), *tensor['dims']))
            out.write(struct.pack('<I', tensor['qtype']))
            out.write(struct.pack('<Q', _offset))
        position = out.tell()
        out.write(b'\x00' * ((-position) % alignment))
        base = out.tell()
        for tensor, tensor_offset, size in layout:
            out.seek(base + tensor_offset)
            if tensor['name'] in replacements:
                written = replacements[tensor['name']](out, handle)
                if written != size:
                    raise ValueError(f'{tensor["name"]}: writer produced {written} bytes, '
                                     f'the plan says {size}')
                continue
            handle.seek(scanned['data_base'] + tensor['offset'])
            remaining = size
            while remaining:
                chunk = handle.read(min(1 << 24, remaining))
                if not chunk:
                    raise ValueError(f'{tensor["name"]}: short read from the source')
                out.write(chunk)
                remaining -= len(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size}


# --- the edit --------------------------------------------------------------

BYTE_FROM_UNICODE = {character: byte for byte, character in BYTE_TO_UNICODE.items()}


def piece_bytes(piece):
    """The raw bytes a byte-level BPE piece spells; refuses a non-byte piece."""
    try:
        return bytes(BYTE_FROM_UNICODE[character] for character in piece)
    except KeyError as error:
        raise ValueError(f'{piece!r} is not a byte-level BPE piece '
                         f'(no mapping for {error.args[0]!r})') from None


def _encode(text, vocab, ranks):
    """llama.cpp's byte-level BPE, then its byte fallback, as string automata.

    Merges are symbolic strings, so a deleted token can still be *built* by the
    merge list; the vocabulary lookup then misses and the runtime splits the
    merged symbol into its raw bytes. A byte that has no token of its own is
    unrepresentable: that is the broken-tokenizer case, reported as such.
    """
    pieces, ids, kinds = [], [], []
    representable = True
    for symbol in bpe_symbols(text, ranks):
        token_id = vocab.get(symbol)
        if token_id is not None:
            pieces.append(symbol)
            ids.append(int(token_id))
            kinds.append('token')
            continue
        raw = piece_bytes(symbol)
        fallback = [vocab.get(BYTE_TO_UNICODE[byte]) for byte in raw]
        if any(token_id is None for token_id in fallback):
            representable = False
            pieces.append(symbol)
            ids.append(None)
            kinds.append('unrepresentable')
            continue
        for byte in raw:
            spelling = BYTE_TO_UNICODE[byte]
            pieces.append(spelling)
            ids.append(int(vocab[spelling]))
            kinds.append('byte')
    return {'text': text, 'pieces': pieces, 'ids': ids, 'kinds': kinds,
            'n_tokens': len(ids) if representable else None,
            'single_token': bool(representable and len(ids) == 1),
            'representable': representable}


def _mapping_segments(deleted_ids, n_vocab):
    """Contiguous [old_start, old_end] runs with a constant renumbering shift."""
    segments, shift, start = [], 0, 0
    for token_id in sorted(deleted_ids) + [n_vocab]:
        if token_id > start:
            segments.append({'old_start': start, 'old_end': token_id - 1, 'shift': -shift,
                             'new_start': start - shift, 'new_end': token_id - 1 - shift})
        shift += 1
        start = token_id + 1
    return segments


def _resolve_targets(document, tokenizer):
    """Every denial surface resolved in both contexts, and the ids they attest."""
    vocab = {token: token_id for token_id, token in enumerate(tokenizer['tokens'])}
    ranks = _merge_ranks(tokenizer['merges'])
    attest, surfaces, problems = {}, [], []
    for pair in document['pairs']:
        for context in CONTEXTS:
            text = ('' if context == 'bare' else ' ') + pair['denial']
            split, representable = resolve(text, vocab, ranks)
            detail = {'pair': pair['id'], 'context': context, 'surface': pair['denial'],
                      'text': text, 'pieces': [piece for _token_id, piece in split],
                      'ids': [None if token_id is None else int(token_id) for token_id, _piece in split],
                      'representable': bool(representable),
                      'single_token': bool(representable and len(split) == 1 and split[0][0] is not None)}
            surfaces.append(detail)
            if not detail['single_token']:
                problems.append(f'{pair["id"]}/{context}: {text!r} resolves to {detail["pieces"]} '
                                f'({detail["ids"]}); a deletion needs exactly one id')
                continue
            token_id = detail['ids'][0]
            entry = attest.setdefault(token_id, {'id': token_id, 'piece': tokenizer['tokens'][token_id],
                                                 'attested_by': []})
            entry['attested_by'].append({'pair': pair['id'], 'context': context,
                                         'surface': pair['denial'], 'text': text,
                                         'piece': detail['pieces'][0]})
    return {'vocab': vocab, 'ranks': ranks, 'surfaces': surfaces, 'attest': attest, 'problems': problems}


def _row_layout(record):
    from .quants import CODECS

    codec = CODECS.get(record['qtype'])
    if codec is None:
        raise ValueError(f'{record["name"]} is {record["qtype"]}, which this build cannot read')
    block_bytes, block_els, _quantize, _dequantize = codec
    width = int(record['row_width'])
    aligned = width % block_els == 0
    return {'block_bytes': int(block_bytes), 'block_els': int(block_els),
            'row_aligned': aligned,
            'row_bytes': width // block_els * block_bytes if aligned else None}


def _row_copy_writer(source_offset, row_bytes, kept_rows):
    def write(out, handle):
        written = 0
        for old_id in kept_rows:
            handle.seek(source_offset + old_id * row_bytes)
            raw = handle.read(row_bytes)
            if len(raw) != row_bytes:
                raise ValueError(f'short read at row {old_id}')
            out.write(raw)
            written += len(raw)
        return written

    return write


def _row_insert_writer(source_offset, row_bytes, row_order, deleted_rows):
    def write(out, handle):
        written = 0
        for kind, index in row_order:
            if kind == 'deleted':
                raw = deleted_rows[index].tobytes()
            else:
                handle.seek(source_offset + index * row_bytes)
                raw = handle.read(row_bytes)
                if len(raw) != row_bytes:
                    raise ValueError(f'short read at row {index}')
            out.write(raw)
            written += len(raw)
        return written

    return write


def _digest_range(path, offset, size, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        handle.seek(offset)
        remaining = size
        while remaining:
            piece = handle.read(min(chunk, remaining))
            if not piece:
                raise ValueError(f'{path}: short read at {offset}')
            digest.update(piece)
            remaining -= len(piece)
    return digest.hexdigest()


def _rows_identical(before, after, offset_before, offset_after, edit):
    row_bytes = edit['row_bytes']
    mismatched = []
    with open(before, 'rb') as source, open(after, 'rb') as target:
        for new_id, old_id in enumerate(edit['kept_rows']):
            source.seek(offset_before + old_id * row_bytes)
            expected = source.read(row_bytes)
            target.seek(offset_after + new_id * row_bytes)
            actual = target.read(row_bytes)
            if actual != expected:
                mismatched.append({'new_id': new_id, 'old_id': old_id})
    return {'rows_identical': not mismatched, 'rows_checked': len(edit['kept_rows']),
            'mismatched': mismatched[:16], 'mismatched_count': len(mismatched)}


def _metadata_identity(before, after, expected_changed):
    by_key_before, by_key_after = before['entry_of'], after['entry_of']
    changed = sorted(key for key in by_key_before
                     if key not in by_key_after or by_key_before[key]['blob'] != by_key_after[key]['blob'])
    untouched = [key for key in by_key_before if key not in changed]
    return {'entries_before': len(by_key_before), 'entries_after': len(by_key_after),
            'changed_entries': changed,
            'added_entries': sorted(set(by_key_after) - set(by_key_before)),
            'removed_entries': sorted(set(by_key_before) - set(by_key_after)),
            'untouched_entries': len(untouched),
            'untouched_entries_differing': sum(1 for key in untouched
                                               if by_key_before[key]['blob'] != by_key_after[key]['blob']),
            'expected_changed': sorted(expected_changed),
            'changed_entries_match_plan': changed == sorted(expected_changed),
            'note': ('every KV entry not in changed_entries is byte-identical, including '
                     'tokenizer.ggml.merges and the chat template')}


def _tensor_identity(before, after, scanned_before, scanned_after, edits):
    edited = {edit['record']['name']: edit for edit in edits}
    rows = []
    for tensor_after in scanned_after['tensors']:
        tensor_before = next(item for item in scanned_before['tensors']
                             if item['name'] == tensor_after['name'])
        size_before = _tensor_bytes(tensor_before)
        size_after = _tensor_bytes(tensor_after)
        record = {'name': tensor_after['name'], 'dims_before': tensor_before['dims'],
                  'dims_after': tensor_after['dims'], 'qtype': tensor_after['qtype'],
                  'bytes_before': size_before, 'bytes_after': size_after,
                  'digest_before': _digest_range(before, scanned_before['data_base'] + tensor_before['offset'],
                                                 size_before),
                  'digest_after': _digest_range(after, scanned_after['data_base'] + tensor_after['offset'],
                                                size_after)}
        if tensor_after['name'] in edited:
            edit = edited[tensor_after['name']]
            record.update({'role': 'vocabulary matrix', 'rows_before': edit['record']['rows'],
                           'rows_after': int(tensor_after['dims'][1]),
                           'row_bytes': edit['row_bytes'], 'edited': True})
        else:
            record['identical'] = (record['digest_before'] == record['digest_after']
                                   and size_before == size_after)
            record['edited'] = False
        rows.append(record)
    others = [row for row in rows if not row['edited']]
    surviving = {}
    for name in edited:
        tensor_before = next(item for item in scanned_before['tensors'] if item['name'] == name)
        tensor_after = next(item for item in scanned_after['tensors'] if item['name'] == name)
        surviving[name] = _rows_identical(before, after,
                                          scanned_before['data_base'] + tensor_before['offset'],
                                          scanned_after['data_base'] + tensor_after['offset'],
                                          edited[name])
    return {'tensor_count_before': len(scanned_before['tensors']),
            'tensor_count_after': len(scanned_after['tensors']),
            'names_equal': [t['name'] for t in scanned_before['tensors']] ==
                           [t['name'] for t in scanned_after['tensors']],
            'edited': sorted(edited), 'other_tensors_checked': len(others),
            'other_tensors_identical': all(row['identical'] for row in others),
            'surviving_rows_identical': all(item['rows_identical'] for item in surviving.values()),
            'rows': surviving, 'tensors': rows}


def _special_resolution(scanned, tokens):
    out = {}
    for key, entry in scanned['entry_of'].items():
        if key.endswith('_token_id') and entry['type'] == UINT32:
            token_id = _scalar(entry)
            out[key] = {'id': token_id, 'piece': tokens[token_id] if 0 <= token_id < len(tokens) else None}
    return out


def _reader_load(path):
    try:
        from gguf import GGUFReader

        reader = GGUFReader(str(path))
        tokens = reader.fields.get('tokenizer.ggml.tokens')
        return {'parsed': True, 'tensors': len(reader.tensors),
                'n_vocab': len(tokens.data) if tokens is not None else None,
                'reader': 'gguf.GGUFReader'}
    except Exception as error:  # noqa: BLE001 - the report carries the failure, it does not raise
        return {'parsed': False, 'error': f'{type(error).__name__}: {error}'}


def _reencode_on_file(path, tokenizer, probes):
    vocab = {token: token_id for token_id, token in enumerate(tokenizer['tokens'])}
    ranks = _merge_ranks(tokenizer['merges'])
    return {'n_vocab': len(tokenizer['tokens']),
            'rows': [{'text': text, 'encode': _encode(text, vocab, ranks)} for text in probes],
            'note': ('computed from the written file\'s tokenizer.ggml arrays, not from the plan; the same '
                     'byte-level BPE + byte fallback as the dry run')}


def _write_sidecar(pristine, pruned, sidecar, tokens, deleted, types, edits, scanned):
    deleted_sorted = sorted(deleted)
    payload = {
        'schema': np.array(REVERT_SCHEMA),
        'original_sha256': np.array(_sha256_file(pristine)),
        'patched_sha256': np.array(_sha256_file(pruned)),
        'n_vocab_old': np.array(len(tokens), dtype=np.int64),
        'n_vocab_new': np.array(len(tokens) - len(deleted_sorted), dtype=np.int64),
        'deleted_ids': np.array(deleted_sorted, dtype=np.int64),
        'deleted_types': np.array([int(types[token_id]) for token_id in deleted_sorted], dtype=np.int32),
        'pieces': np.array([tokens[token_id] for token_id in deleted_sorted], dtype=str),
        'matrix_names': np.array(json.dumps([edit['record']['name'] for edit in edits])),
        'scalar_ids': np.array(json.dumps({})),
    }
    for edit in edits:
        name = edit['record']['name']
        payload[f'row_bytes::{name}'] = np.array(int(edit['row_bytes']), dtype=np.int64)
        with open(pristine, 'rb') as handle:
            handle.seek(scanned['data_base'] + edit['tensor']['offset'])
            data = handle.read(edit['record']['rows'] * edit['row_bytes'])
        matrix = np.frombuffer(data, dtype=np.uint8).reshape(edit['record']['rows'], edit['row_bytes'])
        payload[f'rows::{name}'] = matrix[deleted_sorted].copy()
    # the scalar ids are re-inserted from the pristine file's own values
    scalars = {}
    for key, entry in scanned['entry_of'].items():
        if key.endswith('_token_id') and entry['type'] == UINT32:
            scalars[key] = _scalar(entry)
    payload['scalar_ids'] = np.array(json.dumps(scalars))
    np.savez(sidecar, **payload)
    return {'path': _portable_path(sidecar), 'bytes': Path(sidecar).stat().st_size,
            'deleted_ids': deleted_sorted,
            'pieces': [tokens[token_id] for token_id in deleted_sorted],
            'matrices': [edit['record']['name'] for edit in edits],
            'rows': len(deleted_sorted), 'original_sha256': str(payload['original_sha256']),
            'patched_sha256': str(payload['patched_sha256']), 'schema': REVERT_SCHEMA}


def revert_delete(pruned, sidecar, out):
    """Re-insert the deleted entries and rows; byte-exact against the recorded source."""
    pruned, sidecar, out = Path(pruned), Path(sidecar), Path(out)
    payload = np.load(sidecar, allow_pickle=False)
    if str(payload['schema']) != REVERT_SCHEMA:
        raise ValueError(f'{sidecar}: schema {str(payload["schema"])!r}, expected {REVERT_SCHEMA!r}')
    recorded = str(payload['patched_sha256'])
    current = _sha256_file(pruned)
    if current != recorded:
        raise ValueError(f'{pruned} changed since the deletion (sha256 {current[:16]} != recorded '
                         f'{recorded[:16]}); refusing to re-insert entries over a different file')
    scanned = _scan(pruned)
    tokens = _strings(scanned['entry_of']['tokenizer.ggml.tokens'])
    types = [int(value) for value in _ints(scanned['entry_of']['tokenizer.ggml.token_type'])]
    deleted_ids = [int(value) for value in payload['deleted_ids']]
    pieces = [str(value) for value in payload['pieces']]
    deleted_types = [int(value) for value in payload['deleted_types']]
    scalar_ids = json.loads(str(payload['scalar_ids']))
    n_old, n_new = int(payload['n_vocab_old']), int(payload['n_vocab_new'])
    if n_new != len(tokens):
        raise ValueError(f'{pruned}: {len(tokens)} tokens, sidecar expects {n_new}')
    if len(pieces) != len(deleted_ids) or len(deleted_types) != len(deleted_ids):
        raise ValueError(f'{sidecar}: deleted entries are not the same length')
    deleted_set = set(deleted_ids)
    deleted_index = {token_id: index for index, token_id in enumerate(deleted_ids)}
    if len(deleted_index) != len(deleted_ids):
        raise ValueError(f'{sidecar}: duplicate deleted ids')
    full_tokens, full_types, row_order = [], [], []
    new_row = 0
    for old_id in range(n_old):
        if old_id in deleted_set:
            full_tokens.append(pieces[deleted_index[old_id]])
            full_types.append(deleted_types[deleted_index[old_id]])
            row_order.append(('deleted', deleted_index[old_id]))
        else:
            if new_row >= n_new:
                raise ValueError('internal: the surviving token count does not match the pruned vocabulary')
            full_tokens.append(tokens[new_row])
            full_types.append(types[new_row])
            row_order.append(('kept', new_row))
            new_row += 1
    if new_row != n_new:
        raise ValueError(f'internal: {new_row} surviving rows for a {n_new}-token file')
    blobs = [entry['blob'] for entry in scanned['entries']]
    for index, entry in enumerate(scanned['entries']):
        if entry['key'] == 'tokenizer.ggml.tokens':
            blobs[index] = _entry_blob(entry['key'], TYPE_ARRAY, _encode_string_array(full_tokens))
        elif entry['key'] == 'tokenizer.ggml.token_type':
            blobs[index] = _entry_blob(entry['key'], TYPE_ARRAY, _encode_int32_array(full_types))
        elif entry['key'] in scalar_ids:
            blobs[index] = _entry_blob(entry['key'], UINT32, struct.pack('<I', int(scalar_ids[entry['key']])))
    names = json.loads(str(payload['matrix_names']))
    tensors, replacements = [], {}
    for tensor in scanned['tensors']:
        if tensor['name'] not in names:
            tensors.append(tensor)
            continue
        row_bytes = int(payload[f'row_bytes::{tensor["name"]}'])
        deleted_rows = np.asarray(payload[f'rows::{tensor["name"]}'], dtype=np.uint8)
        if deleted_rows.shape != (len(deleted_ids), row_bytes):
            raise ValueError(f'{sidecar}: {tensor["name"]} rows {deleted_rows.shape} do not match '
                             f'{len(deleted_ids)} x {row_bytes}')
        dims = list(tensor['dims'])
        if int(dims[1]) != n_new:
            raise ValueError(f'{tensor["name"]}: dims {dims} are not a {n_new}-row matrix')
        dims[1] = n_old
        record = {'name': tensor['name'], 'dims': dims, 'qtype': tensor['qtype'], 'offset': tensor['offset'],
                  'bytes': _tensor_bytes({'dims': dims, 'qtype': tensor['qtype']})}
        tensors.append(record)
        replacements[tensor['name']] = _row_insert_writer(
            scanned['data_base'] + tensor['offset'], row_bytes, row_order, deleted_rows)
    found = {tensor['name'] for tensor in scanned['tensors']}
    if not set(names) <= found:
        raise ValueError(f'{sidecar}: matrices {sorted(set(names) - found)} are not in the pruned file')
    _write_container(out, scanned, blobs, tensors, replacements, pruned)
    restored = _sha256_file(out)
    original = str(payload['original_sha256'])
    return {'path': _portable_path(out), 'sha256': restored, 'original_sha256': original,
            'patched_sha256': recorded, 'byte_exact': restored == original,
            'reinserted_ids': sorted(deleted_ids), 'schema': REVERT_SCHEMA,
            'note': ('the source is reconstructed from the sidecar (deleted entries and raw row bytes '
                     're-inserted), not copied; byte_exact compares the digest with the recorded source')}


def _write_mapping(path, rows, deleted, n_old):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deleted_set = set(deleted)
    by_old = {old_id: (new_id, piece) for old_id, new_id, piece in rows}
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('old_id\tnew_id\tdeleted\tpiece_sha256\tpiece\n')
        for old_id in range(n_old):
            if old_id in deleted_set:
                handle.write(f'{old_id}\t-\t1\t\t\n')
                continue
            new_id, piece = by_old[old_id]
            handle.write(f'{old_id}\t{new_id}\t0\t{_piece_digest(piece)}\t{piece}\n')


def _piece_digest(piece):
    return hashlib.sha256(piece.encode('utf-8')).hexdigest()[:16]


def _mapping_digest(rows):
    digest = hashlib.sha256()
    for old_id, new_id, piece in rows:
        digest.update(f'{old_id}\t{new_id}\t{piece}\n'.encode('utf-8'))
    return digest.hexdigest()


def _maybe_write(report, json_out):
    if json_out is not None:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def vocab_delete(weights, table=None, out=None, apply=False, include_byte_alphabet=False,
                 json_out=None, mapping_out=None):
    """Plan (default) or apply a vocabulary deletion on a quantized artifact."""
    weights = Path(weights)
    out_path = None if out is None else Path(out)
    if not weights.is_file():
        raise ValueError(f'{weights} does not exist')
    if apply and out_path is None:
        raise ValueError('--apply needs --out for the new artifact; this never edits the input in place')
    if apply and out_path.exists():
        raise ValueError(f'{out} already exists; the apply writes a new artifact and refuses to clobber one')
    table_path = Path(table) if table is not None else PACKAGED_TABLE
    document = json.loads(table_path.read_text())
    if document.get('schema') != PAIRS_SCHEMA:
        raise ValueError(f'{table_path}: schema {document.get("schema")!r}, expected {PAIRS_SCHEMA!r}')

    discovery = discover_vocab_tensors(weights)
    tokenizer = discovery['tokenizer']
    tokens = tokenizer['tokens']
    scanned = _scan(weights)
    for key in ('tokenizer.ggml.tokens', 'tokenizer.ggml.token_type'):
        if key not in scanned['entry_of']:
            raise ValueError(f'{weights}: no {key} entry')
    types = _ints(scanned['entry_of']['tokenizer.ggml.token_type'])
    if len(tokens) != len(types):
        raise ValueError(f'{weights}: {len(tokens)} tokens but {len(types)} token types')
    if tokens != _strings(scanned['entry_of']['tokenizer.ggml.tokens']):
        raise ValueError(f'{weights}: gguf reader and raw KV disagree on the token strings')

    resolved = _resolve_targets(document, tokenizer)
    candidates = []
    for token_id in sorted(resolved['attest']):
        attested = resolved['attest'][token_id]
        piece = attested['piece']
        try:
            raw = piece_bytes(piece)
        except ValueError as error:
            raw, refused = None, str(error)
        else:
            refused = None
        byte_alphabet = raw is not None and len(raw) == 1
        token_type = int(types[token_id])
        if refused is None and token_type != NORMAL:
            refused = f'token_type {token_type} is not NORMAL'
        elif refused is None and byte_alphabet and not include_byte_alphabet:
            refused = ('single raw byte: this id IS one entry of the tokenizer\'s byte alphabet, so deleting '
                       'it makes every string containing that byte unrepresentable (whole-tokenizer breakage, '
                       'not a targeted removal); pass --include-byte-alphabet to measure that arm')
        candidates.append({'id': token_id, 'piece': piece,
                           'piece_bytes_hex': raw.hex() if raw is not None else None,
                           'piece_bytes': len(raw) if raw is not None else None,
                           'token_type': token_type, 'byte_alphabet': bool(byte_alphabet),
                           'attested_by': attested['attested_by'], 'refused': refused,
                           'deleted': refused is None})
    deleted = [item['id'] for item in candidates if item['deleted']]
    refused = [item for item in candidates if not item['deleted']]
    if not deleted:
        raise ValueError('no id is deletable under this policy: ' +
                         '; '.join(f'{item["piece"]!r}/{item["id"]}: {item["refused"]}' for item in refused))
    deleted_set = set(deleted)
    n_old, n_new = len(tokens), len(tokens) - len(deleted)
    if n_new <= 0:
        raise ValueError('deleting every token would leave an empty vocabulary')

    new_tokens, old_to_new, mapping_rows = [], {}, []
    shift = 0
    for old_id, token in enumerate(tokens):
        if old_id in deleted_set:
            shift += 1
            continue
        new_id = old_id - shift
        old_to_new[old_id] = new_id
        new_tokens.append(token)
        mapping_rows.append((old_id, new_id, token))
    new_types = [int(types[old_id]) for old_id in range(n_old) if old_id not in deleted_set]
    new_vocab = {token: new_id for token, new_id in
                 ((token, old_to_new[old_id]) for old_id, token in enumerate(tokens)
                  if old_id not in deleted_set)}

    renumbered = {}
    for key, entry in scanned['entry_of'].items():
        if key.endswith('_token_id') and entry['type'] == UINT32:
            old_value = _scalar(entry)
            below = sum(1 for token_id in deleted if token_id < old_value)
            renumbered[key] = {'old': old_value, 'new': old_value - below, 'deleted_below': below}

    edits = []
    for role in ('embedding', 'output_head'):
        record = discovery['tensors'].get(role)
        if record is None:
            continue
        layout = _row_layout(record)
        if not layout['row_aligned']:
            raise ValueError(f'{record["name"]}: row width {record["row_width"]} is not a whole number of '
                             f'{record["qtype"]} blocks; the row-preserving deletion needs aligned rows')
        tensor = next(item for item in scanned['tensors'] if item['name'] == record['name'])
        if int(tensor['dims'][1]) != n_old or int(tensor['dims'][0]) != int(record['row_width']):
            raise ValueError(f'{record["name"]}: dims {tensor["dims"]} do not match {record["rows"]} rows x '
                             f'{record["row_width"]}')
        dims = list(tensor['dims'])
        dims[1] = n_new
        edits.append({'record': record, 'tensor': tensor, 'layout': layout, 'old_dims': list(tensor['dims']),
                      'new_dims': dims, 'row_bytes': layout['row_bytes'],
                      'bytes': _tensor_bytes({'dims': dims, 'qtype': tensor['qtype']}),
                      'kept_rows': [old_id for old_id in range(n_old) if old_id not in deleted_set],
                      'roles': (['embedding'] if role == 'embedding' else
                                ['output_head (tied to embedding)' if discovery['tied'] else 'output_head'])})

    before_after, seen = [], set()
    for item in resolved['surfaces']:
        if item['text'] in seen:
            continue
        seen.add(item['text'])
        before_after.append({'text': item['text'],
                             'before': _encode(item['text'], resolved['vocab'], resolved['ranks'])})
    for text in PROBES:
        if text in seen:
            continue
        seen.add(text)
        before_after.append({'text': text,
                             'before': _encode(text, resolved['vocab'], resolved['ranks'])})
    for row in before_after:
        row['after'] = _encode(row['text'], new_vocab, resolved['ranks'])

    segments = _mapping_segments(deleted, n_old)
    report = {
        'schema': SCHEMA,
        'scope': ('Delete the denial-opener ids from a quantized artifact: the tokenizer arrays, the scalar ids '
                  'that share the vocabulary id space, and the token-row matrix. Storage fidelity, the '
                  'renumbering map, and the static encoder paths of the deleted surfaces. No model is loaded '
                  'and no behavioural claim is made.'),
        'method': METHOD,
        'gated': GATED,
        'removal_measured': False,
        'dry_run': not apply,
        'artifact': {'path': _portable_path(weights), 'bytes': weights.stat().st_size,
                     'sha256': _sha256_file(weights)},
        'table': {'path': _portable_path(table_path), 'schema': document.get('schema'),
                  'pairs': len(document['pairs']),
                  'sha256': _sha256_file(table_path)},
        'tokenizer': tokenizer['identity'],
        'discovery': {'n_vocab_old': n_old, 'n_vocab_new': n_new, 'deleted': len(deleted),
                      'tied': discovery['tied'], 'vocab_wide_tensors': discovery['vocab_wide'],
                      'tensor_count': discovery['tensor_count'],
                      'how': ('vocabulary-wide 2-D tensors are found by row count == tokenizer vocabulary; '
                              'the artifact name decides the role')},
        'policy': {'include_byte_alphabet': bool(include_byte_alphabet), 'deleted_ids': deleted,
                   'definition': ('candidates are the ids the table\'s denial surfaces resolve to in one token '
                                  'in either context; a candidate is deleted when its token_type is NORMAL and, '
                                  'unless --include-byte-alphabet, its piece is more than one raw byte')},
        'candidates': candidates,
        'refused_ids': [{'id': item['id'], 'piece': item['piece'], 'reason': item['refused']}
                        for item in refused],
        'surfaces': resolved['surfaces'],
        'renumber': {'rule': 'new_id = old_id - (number of deleted ids below old_id)',
                     'segments': segments,
                     'deleted_entries': [{'old_id': item['id'], 'piece': item['piece'],
                                          'token_type': item['token_type']}
                                         for item in candidates if item['deleted']],
                     'scalar_ids': renumbered,
                     'invariant': ('every surviving id keeps its piece and its row: the token at new_id is the '
                                   'token that was at old_id, and no surviving piece changes')},
        'encode': {'algorithm': ('byte-level BPE merge ranks (tokenizer.ggml.merges), then the vocabulary '
                                'lookup; a merged symbol absent from the vocabulary falls back to the tokens of '
                                'its raw bytes, exactly as llama.cpp does. Tokenization only.'),
                   'rows': before_after},
        'tensors': [{'name': edit['record']['name'], 'roles': edit['roles'], 'qtype': edit['record']['qtype'],
                     'old_dims': edit['old_dims'], 'new_dims': edit['new_dims'],
                     'rows_old': edit['record']['rows'], 'rows_new': n_new,
                     'row_bytes': edit['row_bytes'], 'bytes_old': edit['record']['bytes'],
                     'bytes_new': edit['bytes'], 'rows_deleted': len(deleted),
                     'row_policy': ('whole stored rows are copied for surviving rows and the deleted rows '
                                    'are dropped; no value is re-encoded, so every surviving row is '
                                    'byte-identical')}
                    for edit in edits],
        'arrays_touched': {
            'metadata': ['tokenizer.ggml.tokens', 'tokenizer.ggml.token_type'] + sorted(renumbered),
            'tensors': [edit['record']['name'] for edit in edits],
            'untouched': ('tokenizer.ggml.merges and every other KV entry keep their exact encoding; every '
                          'other tensor keeps its payload; only offsets after the shrunken matrix move'),
            'renumbering_ties': ('the tokenizer arrays, the scalar id fields and the matrix rows share one id '
                                 'space. A file that renumbers one without the others is broken: a token id '
                                 'that disagrees with its row reads the wrong embedding, and a scalar id that '
                                 'is not renumbered points at a different token. This is why the report names '
                                 'every one of them.')},
        'mapping': {'rows': len(mapping_rows), 'sha256': _mapping_digest(mapping_rows),
                    'segments': segments},
        'plan': {'writer': ('jdewitt.vocabdelete._write_container: KV entries verbatim except the named ones, '
                            'tensor payloads streamed from the source except the shrunken matrices, offsets '
                            'recomputed under the file alignment'),
                 'out': _portable_path(out_path) if out_path is not None else None,
                 'apply': bool(apply),
                 'revert_sidecar': (_portable_path(f'{out_path}.revert.npz')
                                    if out_path is not None else None)},
        'verdict': {'ok': True, 'problems': [], 'notes': [
            'deletion removes a single-token spelling; a merged token whose bytes all survive the byte '
            'alphabet remains spellable as a sequence, so this edit does not make the string unsampleable',
            'a byte-alphabet id cannot be deleted without breaking every string that uses its byte',
            'removal_measured is false: no forward pass, no logits, no serve']},
    }
    if resolved['problems']:
        report['verdict']['problems'].extend(resolved['problems'])
    if len(new_types) != n_new:
        raise ValueError('internal: the renumbered token_type array does not have the new vocabulary size')

    mapping_path = Path(mapping_out) if mapping_out is not None else (
        Path(json_out).with_name(Path(json_out).stem + '-id-map.tsv') if json_out is not None else
        (out_path.with_name(out_path.stem + '-id-map.tsv') if out_path is not None else None))
    if mapping_path is not None:
        _write_mapping(mapping_path, mapping_rows, deleted, n_old)
        report['mapping']['path'] = _portable_path(mapping_path)

    if not apply:
        _maybe_write(report, json_out)
        return report

    started = time.monotonic()
    assert out_path is not None
    blobs = [entry['blob'] for entry in scanned['entries']]
    for index, entry in enumerate(scanned['entries']):
        if entry['key'] == 'tokenizer.ggml.tokens':
            blobs[index] = _entry_blob(entry['key'], TYPE_ARRAY, _encode_string_array(new_tokens))
        elif entry['key'] == 'tokenizer.ggml.token_type':
            blobs[index] = _entry_blob(entry['key'], TYPE_ARRAY, _encode_int32_array(new_types))
        elif entry['key'] in renumbered:
            blobs[index] = _entry_blob(entry['key'], UINT32,
                                       struct.pack('<I', renumbered[entry['key']]['new']))
    tensors, replacements = [], {}
    for tensor in scanned['tensors']:
        edit = next((item for item in edits if item['record']['name'] == tensor['name']), None)
        if edit is None:
            tensors.append(tensor)
            continue
        tensors.append({'name': tensor['name'], 'dims': edit['new_dims'], 'qtype': tensor['qtype'],
                        'offset': tensor['offset'], 'bytes': edit['bytes']})
        replacements[tensor['name']] = _row_copy_writer(scanned['data_base'] + tensor['offset'],
                                                        edit['row_bytes'], edit['kept_rows'])
    written = _write_container(out_path, scanned, blobs, tensors, replacements, weights)

    after = _scan(out_path)
    expected_changed = {'tokenizer.ggml.tokens', 'tokenizer.ggml.token_type'} | set(renumbered)
    metadata = _metadata_identity(scanned, after, expected_changed)
    shipped_tokens = _strings(after['entry_of']['tokenizer.ggml.tokens'])
    shipped_types = [int(value) for value in _ints(after['entry_of']['tokenizer.ggml.token_type'])]
    token_checks = {
        'n_vocab_old': n_old, 'n_vocab_new': len(shipped_tokens),
        'n_vocab_new_equals_old_minus_deleted': len(shipped_tokens) == n_old - len(deleted),
        'tokens_equal_expected': shipped_tokens == new_tokens,
        'token_types_equal_expected': shipped_types == new_types,
        'surviving_pieces_unchanged': all(shipped_tokens[new_id] == tokens[old_id]
                                          for old_id, new_id in old_to_new.items()),
        'deleted_pieces_absent': not ({tokens[token_id] for token_id in deleted} & set(shipped_tokens)),
        'scalars_renumbered': all(_scalar(after['entry_of'][key]) == value['new']
                                  for key, value in renumbered.items()),
        'special_tokens_resolve': _special_resolution(after, shipped_tokens),
    }
    tensors_identity = _tensor_identity(weights, out_path, scanned, after, edits)
    load = _reader_load(out_path)
    tokenizer_after = read_tokenizer(out_path)
    reencode = _reencode_on_file(out_path, tokenizer_after, list(seen))
    sidecar_path = Path(f'{out_path}.revert.npz')
    sidecar = _write_sidecar(weights, out_path, sidecar_path, tokens, deleted, types, edits, scanned)
    reverted_path = Path(f'{out_path}.reverted-probe.gguf')
    revert = revert_delete(out_path, sidecar_path, reverted_path)
    if reverted_path.exists():
        reverted_path.unlink()
    report['apply'] = {'out': {'path': _portable_path(out_path), 'bytes': written['bytes'],
                               'sha256': _sha256_file(out_path)},
                       'metadata': metadata, 'tokenizer': token_checks, 'tensors': tensors_identity,
                       'load': load, 'reencode': reencode, 'revert': revert, 'sidecar': sidecar,
                       'seconds': round(time.monotonic() - started, 1)}
    for key in ('n_vocab_new_equals_old_minus_deleted', 'tokens_equal_expected',
                'token_types_equal_expected', 'surviving_pieces_unchanged', 'deleted_pieces_absent',
                'scalars_renumbered'):
        if not token_checks[key]:
            report['verdict']['problems'].append(f'tokenizer check failed: {key}')
    if metadata['untouched_entries_differing']:
        report['verdict']['problems'].append(f'{metadata["untouched_entries_differing"]} untouched KV entries '
                                             f'changed')
    if not metadata['changed_entries_match_plan']:
        report['verdict']['problems'].append(f'the changed KV entries are not the planned set: '
                                             f'{metadata["changed_entries"]}')
    if not tensors_identity['other_tensors_identical']:
        report['verdict']['problems'].append('a tensor other than the vocabulary matrix changed')
    if not tensors_identity['surviving_rows_identical']:
        report['verdict']['problems'].append('a surviving matrix row is not byte-identical to its source row')
    if not load['parsed']:
        report['verdict']['problems'].append(f'the gguf reader could not parse the new file: {load["error"]}')
    if not revert['byte_exact']:
        report['verdict']['problems'].append('the revert did not reproduce the pristine file')
    report['verdict']['ok'] = not report['verdict']['problems']
    _maybe_write(report, json_out)
    return report
