# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vocabulary-row edits on a quantized artifact: the table as rows of the weights.

``vocab-remap --list`` resolves the denial table to token ids for one tokenizer;
this module is the weight side of that inventory. It finds the artifact's token
embedding (and the output head, when the file carries an untied one), takes the
rows the table covers, and applies the same rank-1 projection a refusal axis
gets — along the row itself, which makes the algebra and a per-row down-weight
scale one operation:

    dW_i = -s * u^T (u W_i) = -s W_i,   u = W_i / ||W_i||

Rows outside the table are not part of the edit. The write is the existing
``ggufio.edit_tensors`` path: everything untested is copied byte for byte and the
edited tensors are requantized at their own stored type, so no tensor offset
moves and the pre-edit bytes restore the artifact exactly. The dry run measures
the edited blocks only (``ggufio.preview_edit`` with ``rows=``), because the edit
exists at a few dozen rows of a 151,936-row matrix and an even sample of the
tensor would land on untouched rows and read cos 0 for an edit that is present.
``--apply`` re-measures the shipped bytes against the pristine source with
``ggufio.compare_edit(rows=...)``, verifies by measurement that no byte outside
the edited tensors changed, and writes a revert sidecar carrying the pre-edit
bytes and both digests.

Storage fidelity only: no model, no logits, no generation, no behavioral claim.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .vocabremap import (_merge_ranks, _portable_path, _sha256_file, load_table,
                         read_tokenizer, resolve_lexeme)

WEIGHTS_SCHEMA = 'jdewitt.vocab-remap-weights/1'
REVERT_SCHEMA = 'jdewitt.vocab-remap-revert/1'
#: The repo's screening floor for a stored edit (planapply's default min_cos).
COS_FLOOR = 0.95
#: GGUF stores the token-row matrices under these leaf names; the HF spellings are
#: accepted too because a converted file that kept them is still readable here.
VOCAB_ROLE_ALIASES = {'token_embd': 'embedding', 'token_embeddings': 'embedding',
                      'embed_tokens': 'embedding', 'output': 'output_head',
                      'lm_head': 'output_head'}
WEIGHTS_METHOD = (
    'The denial table is resolved against the artifact tokenizer (the same byte-level BPE as --list, both '
    'contexts) and the resulting ids name rows of the token-embedding matrix and, when it is untied, of the '
    'output head. For each covered row W_i the edit is the rank-1 projection a refusal axis gets, with the '
    'axis being the row\'s own direction u = W_i/||W_i|| on the row\'s input side: dW_i = -s (W_i u^T) u = -s W_i, '
    'so the algebra and a per-row down-weight scale (1 - s) are the same operation and the same helper '
    '(ggufio.axis_delta) computes it. Rows outside the table are not touched by the edit. The write reuses '
    'ggufio.edit_tensors: every untouched tensor is copied byte for byte and the edited tensors are requantized '
    'at their own stored type, which keeps every tensor offset and byte length, so the revert sidecar can hold '
    'the pre-edit bytes and restore the artifact exactly. The dry run measures the edited blocks only '
    '(ggufio.preview_edit rows=) because the edit exists at a few dozen rows of a 151,936-row matrix; --apply '
    're-measures the shipped bytes (ggufio.compare_edit rows=) against the pristine source. No model, no '
    'logits, no generation, and no behavioral claim: the weight side is measured here for storage fidelity only.')


def _vocab_role(name):
    stem = name[:-len('.weight')] if name.endswith('.weight') else name
    return VOCAB_ROLE_ALIASES.get(stem.split('.')[-1])


def discover_vocab_tensors(artifact):
    """The token-embedding/output-head tensors, read off the artifact itself.

    Both roles are token-row matrices: a 2-D tensor whose logical row count is the
    tokenizer's vocabulary size. Shapes come from the repo's existing header-driven
    discovery (``gguf_adapters.gguf_weight_shapes``: name -> dims from the file's own
    tensor records) and the role from the artifact name, so there is no per-model
    table of tensor names. A tied model (Qwen2.5-0.5B) carries no ``output.weight``:
    the embedding matrix is the output head, and the edit then lands on it once.
    A vocabulary-wide tensor with no recognised role is reported, never edited.
    """
    from .gguf_adapters import gguf_weight_shapes
    from .ggufio import header_reader

    tokenizer = read_tokenizer(artifact)
    n_vocab = len(tokenizer['tokens'])
    reader = header_reader(artifact)
    shapes = gguf_weight_shapes(artifact, reader=reader)
    records = {tensor.name: tensor for tensor in reader.tensors}
    tensors, unmapped, vocab_wide = {}, [], []
    for name, dims in shapes.items():
        if len(dims) != 2 or int(dims[1]) != n_vocab:
            continue
        tensor = records[name]
        record = {'name': name, 'qtype': tensor.tensor_type.name,
                  'rows': int(dims[1]), 'row_width': int(dims[0]),
                  'offset': int(tensor.data_offset), 'bytes': int(tensor.n_bytes)}
        vocab_wide.append(name)
        role = _vocab_role(name)
        if role is None:
            unmapped.append(record)
            continue
        if role in tensors:
            raise ValueError(
                f'{artifact}: two tensors claim the {role} role ({tensors[role]["name"]!r} and '
                f'{name!r}); this refuses to guess which one the vocabulary uses')
        tensors[role] = record
    if 'embedding' not in tensors:
        raise ValueError(
            f'{artifact}: no token-embedding matrix for a {n_vocab}-token vocabulary. 2-D tensors at that width: '
            f'{vocab_wide or "none"}; recognised role names: {sorted(VOCAB_ROLE_ALIASES)}')
    return {'n_vocab': n_vocab, 'tokenizer': tokenizer, 'tensors': tensors,
            'tied': 'output_head' not in tensors,
            'vocab_wide': vocab_wide, 'unmapped': unmapped,
            'tensor_count': len(reader.tensors),
            'tensor_bytes': int(sum(int(item.n_bytes) for item in reader.tensors))}


def denial_rows(document, tokenizer):
    """The table's ids under this tokenizer, plus the entry each id comes from.

    A row is covered when the table's surface tokenizes to it in either context —
    bare (``cannot``) or with the leading space running text carries (`` cannot``) —
    because the weight edit has to reach the token the model would actually emit.
    """
    vocab = {token: token_id for token_id, token in enumerate(tokenizer['tokens'])}
    ranks = _merge_ranks(tokenizer['merges'])
    entries, coverage = [], {}
    for entry in document['entries']:
        lexeme = resolve_lexeme(entry, vocab, ranks)
        contexts = {name: [pair[0] for pair in lexeme['contexts'][name]]
                    for name in ('bare', 'leads_with_space')}
        ids = sorted({int(token_id) for token_id in contexts['bare'] + contexts['leads_with_space']
                      if token_id is not None})
        entries.append({'id': entry['id'], 'form': entry['form'], 'language': entry['language'],
                        'kind': entry['kind'], 'scope': entry['scope'],
                        'representable': lexeme['representable'], 'split': lexeme['ids'],
                        'ids': ids, 'contexts': contexts})
        if not lexeme['representable'] or not ids:
            continue
        for token_id in ids:
            coverage.setdefault(token_id, []).append(entry['id'])
    rows = [{'id': token_id, 'token': tokenizer['tokens'][token_id],
             'entries': sorted(set(coverage[token_id]))} for token_id in sorted(coverage)]
    return {'ids': sorted(coverage), 'rows': rows, 'entries': entries,
            'unrepresentable': [entry['id'] for entry in entries if not entry['representable']]}


def _tensor_layout(tensor):
    from .quants import CODECS

    codec = CODECS.get(tensor['qtype'])
    if codec is None:
        raise ValueError(
            f"{tensor['name']} is {tensor['qtype']}, which this build cannot ENCODE. The write requantizes the "
            f"edited tensor at its own stored type so every byte offset stays put and the revert sidecar can "
            f"restore the artifact exactly; a type this build cannot write would move them. Re-quantize that "
            f"tensor at a type this build writes ({', '.join(sorted(CODECS))}) first, or pick an artifact that "
            f"stores it that way.")
    block_bytes, block_els, _quantize, _dequantize = codec
    n = tensor['rows'] * tensor['row_width']
    if n % block_els:
        raise ValueError(f"{tensor['name']}: {n} values is not a whole number of {block_els}-value blocks")
    return {'block_bytes': int(block_bytes), 'block_els': int(block_els), 'n': n,
            'blocks': n // block_els}


def _row_vectors(artifact, tensor, ids):
    """Dequantized values of the named rows, reading only the blocks they live in.

    A few dozen ids out of 151,936 cost kilobytes of the tensor, not the whole
    145 MB tensor's decode. A row that is not a whole number of blocks wide shares
    its edge block with a neighbour; that block is re-encoded by the writer anyway,
    so reading it is what makes the measured values the shipped ones.
    """
    from .ggufio import _block_spans, decode_values

    layout = _tensor_layout(tensor)
    hidden = tensor['row_width']
    vectors, spans = {}, _block_spans(sorted(ids), hidden, layout['n'], layout['block_els'])
    with open(artifact, 'rb') as handle:
        for start, end in spans:
            byte0 = start // layout['block_els'] * layout['block_bytes']
            count = (end - start) // layout['block_els'] * layout['block_bytes']
            handle.seek(tensor['offset'] + byte0)
            raw = np.frombuffer(handle.read(count), dtype=np.uint8)
            if raw.size != count:
                raise ValueError(f"{tensor['name']}: short read at byte {byte0} (truncated artifact?)")
            values = decode_values(raw, tensor['qtype'], end - start)
            for row in ids:
                low = row * hidden - start
                if 0 <= low and low + hidden <= values.size:
                    vectors[row] = np.asarray(values[low:low + hidden], dtype=np.float32)
    missing = sorted(set(ids) - set(vectors))
    if missing:
        raise ValueError(f"{tensor['name']}: rows {missing[:8]} were not covered by the read spans")
    return {'vectors': vectors, 'layout': layout, 'spans': spans,
            'bytes_read': int(sum((end - start) // layout['block_els'] * layout['block_bytes']
                                  for start, end in spans))}


def _row_scale_delta(tensor, vectors, strength):
    """delta for the covered rows, built with the refusal-axis helper row by row."""
    from .ggufio import axis_delta

    delta = np.zeros((tensor['rows'], tensor['row_width']), dtype=np.float32)
    detail = []
    for row in sorted(vectors):
        values = np.asarray(vectors[row], dtype=np.float32)
        norm = float(np.linalg.norm(values))
        if norm <= 1e-12:
            detail.append({'row': row, 'row_norm': 0.0, 'edited': False, 'reason': 'zero row'})
            continue
        axis = (values / np.float32(norm)).reshape(1, -1)
        row_delta = axis_delta(axis, values.reshape(1, -1), strength, space='input')
        delta[row] = np.asarray(row_delta, dtype=np.float32).reshape(-1)
        detail.append({'row': row, 'row_norm': round(norm, 6), 'edited': True})
    return {'delta': delta, 'rows': detail,
            'rows_edited': sum(1 for item in detail if item['edited'])}


def _tensor_measurement(artifact, deltas, rows_by_name, strength):
    """Row-scoped dry run: what the codec stores on the edited rows only."""
    from .ggufio import preview_edit

    measured = preview_edit(str(artifact), deltas, rows=rows_by_name)
    return {'strength': strength, 'row_scale': round(1.0 - strength, 6), **measured}


def _diff_runs(source, out, edited_ranges, chunk=8 * 1024 * 1024):
    """Byte ranges where two same-size files differ, refused outside the edited tensors.

    A difference outside the edited ranges means the writer touched a tensor it was
    told not to; that is a failure, not a number.
    """
    import os

    size = os.path.getsize(source)
    if size != os.path.getsize(out):
        raise RuntimeError(f'{out} is {os.path.getsize(out)} bytes, source is {size}; an edit that moves '
                          f'offsets cannot be reverted from pre-edit bytes')
    indices = []
    with open(source, 'rb') as before, open(out, 'rb') as after:
        position = 0
        while position < size:
            count = min(chunk, size - position)
            a = np.frombuffer(before.read(count), dtype=np.uint8)
            b = np.frombuffer(after.read(count), dtype=np.uint8)
            if a.size != b.size:
                raise RuntimeError(f'short read at byte {position}')
            differing = a != b
            if differing.any():
                indices.append(np.flatnonzero(differing) + position)
            position += count
    if not indices:
        return []
    flat = np.concatenate(indices)
    breaks = np.flatnonzero(np.diff(flat) > 1)
    starts = np.concatenate([[flat[0]], flat[breaks + 1]])
    ends = np.concatenate([flat[breaks], [flat[-1]]]) + 1
    runs = [(int(start), int(end)) for start, end in zip(starts, ends)]
    for start, end in runs:
        if not any(offset <= start and end <= offset + length for offset, length in edited_ranges):
            raise RuntimeError(f'byte range [{start}, {end}) differs outside the edited tensors '
                               f'{edited_ranges}; the writer changed a tensor it was not asked to edit')
    return runs


def _write_revert_sidecar(source, out, sidecar, runs):
    """Pre-edit bytes of every differing range, plus both digests.

    ``revert_inplace`` refuses a file that changed since the patch, writes these
    bytes back and reports whether the file is the exact pre-edit artifact. The
    ranges are measured, not assumed: what the writer changed is what is saved.
    """
    arrays, offsets, lengths = [], [], []
    with open(source, 'rb') as handle:
        for start, end in runs:
            handle.seek(start)
            arrays.append(np.frombuffer(handle.read(end - start), dtype=np.uint8).copy())
            offsets.append(start)
            lengths.append(end - start)
    data = (np.concatenate(arrays).astype(np.uint8) if arrays else np.empty(0, dtype=np.uint8))
    patched = _sha256_file(out)
    original = _sha256_file(source)
    np.savez(sidecar, offsets=np.array(offsets, dtype=np.int64),
             lengths=np.array(lengths, dtype=np.int64), data=data,
             sha256=np.array(patched), original_sha256=np.array(original))
    return {'schema': REVERT_SCHEMA, 'path': _portable_path(sidecar),
            'bytes': int(Path(sidecar).stat().st_size), 'ranges': len(runs),
            'original_sha256': original, 'patched_sha256': patched,
            'restores_bytes': int(sum(lengths)), 'format': 'npz: offsets, lengths, pre-edit data, '
                                                       'patched sha256, original sha256'}


def _tensor_payload(record, layout):
    return {'name': record['name'], 'roles': record['roles'], 'qtype': record['qtype'],
            'codec_chosen': record['qtype'],
            'codec_reason': ('the tensor requantizes at its own stored type, so every tensor offset and '
                             'byte length stays put and the pre-edit bytes restore the file exactly'),
            'shape': [record['rows'], record['row_width']],
            'bytes': {'source': record['bytes'], 'reencoded': record['bytes'],
                      'blocks_total': layout['blocks']}}


def vocab_remap_weights(weights, table=None, out=None, apply=False, strength=1.0,
                        json_out=None, sensitivities=(0.9, 0.5)):
    """Plan or apply the denial table as a row edit of the artifact's weights.

    Default (``apply=False``) prints the dry run: which tensors the roles resolve
    to, which rows the table covers, the realized cos the codec stores on those
    rows, the bytes the writer re-encodes and the codec chosen per tensor.
    ``apply=True`` writes a NEW artifact (never in place) plus a revert sidecar,
    verifies by measurement that no byte outside the edited tensors changed, and
    re-measures the delivered bytes against the pristine source.
    """
    weights = Path(weights)
    out_path = None if out is None else Path(out)
    if not weights.is_file():
        raise ValueError(f'{weights} does not exist')
    if not 0 <= strength <= 1:
        raise ValueError(f'strength must be within [0, 1] (1 removes a row, 0 edits nothing), got {strength}')
    if apply and out_path is None:
        raise ValueError('--apply needs --out for the new artifact; this never ships in place')
    if apply and out_path.exists():
        raise ValueError(f'{out} already exists; the apply writes a new artifact and refuses to clobber one')
    document, raw = load_table(table)
    discovery = discover_vocab_tensors(weights)
    tokenizer = discovery['tokenizer']
    coverage = denial_rows(document, tokenizer)
    if not coverage['ids']:
        raise ValueError('the table covers no token id in this tokenizer; nothing to edit')

    roles = {'embedding': discovery['tensors']['embedding']['name'],
             'output_head': (discovery['tensors']['output_head']['name']
                             if 'output_head' in discovery['tensors'] else None),
             'output_head_tied_to_embedding': discovery['tied']}
    targets, by_name = [], {}
    for role in ('embedding', 'output_head'):
        record = discovery['tensors'].get(role)
        if record is None:
            continue
        if record['name'] not in by_name:
            by_name[record['name']] = {**record, 'roles': []}
            targets.append(by_name[record['name']])
        label = role if role == 'embedding' else (
            'output_head (tied to embedding)' if discovery['tied'] else 'output_head')
        by_name[record['name']]['roles'].append(label)

    plans, deltas = [], {}
    for record in targets:
        layout = _tensor_layout(record)
        read = _row_vectors(weights, record, coverage['ids'])
        edited = _row_scale_delta(record, read['vectors'], strength)
        deltas[record['name']] = edited['delta']
        touched_blocks = sum((end - start) // layout['block_els'] for start, end in read['spans'])
        plans.append({'record': record, 'layout': layout, 'read': read, 'edited': edited,
                      'touched_blocks': int(touched_blocks)})

    rows_by_name = {plan['record']['name']: list(coverage['ids']) for plan in plans}
    dry = _tensor_measurement(weights, deltas, rows_by_name, strength)
    measured = {row['name']: row for row in dry['tensors']}
    sensitivity = []
    for other in sensitivities:
        if other == strength:
            continue
        other_deltas = {plan['record']['name']:
                        _row_scale_delta(plan['record'], plan['read']['vectors'], other)['delta']
                        for plan in plans}
        reading = _tensor_measurement(weights, other_deltas, rows_by_name, other)
        sensitivity.append({'strength': other, 'row_scale': round(1.0 - other, 6),
                            'cos_touched_rows': {row['name']: row.get('cos_touched_rows')
                                                 for row in reading['tensors']}})

    tensor_payloads = []
    for plan in plans:
        record, layout = plan['record'], plan['layout']
        payload = _tensor_payload(record, layout)
        row = measured.get(record['name'], {})
        touched_bytes = plan['touched_blocks'] * layout['block_bytes']
        payload['ids'] = coverage['ids']
        payload['bytes'].update({'touched_blocks': plan['touched_blocks'],
                                 'touched_bytes': int(touched_bytes),
                                 'touched_fraction': round(touched_bytes / max(1, record['bytes']), 6),
                                 'bytes_read_to_measure': plan['read']['bytes_read']})
        payload['storage'] = {'cos': row.get('cos'),
                              'cos_touched_rows': row.get('cos_touched_rows'),
                              'norm_ratio': row.get('norm_ratio'),
                              'snr': row.get('snr'),
                              'rows_touched': row.get('rows_touched'),
                              'blocks_touched': row.get('blocks_touched'),
                              'row_detail': row.get('row_detail', [])}
        tensor_payloads.append(payload)

    cosines = [row.get('cos_touched_rows') for row in measured.values()
               if row.get('cos_touched_rows') is not None]
    notes, problems = [], []
    if coverage['unrepresentable']:
        notes.append(f"table entries that do not tokenize on this artifact: {coverage['unrepresentable']}")
    if any(len(entry['ids']) > 1 for entry in coverage['entries']):
        notes.append('a phrase entry covers every token of its split, function tokens included (the table '
                     'names surfaces, not lexemes); coverage.rows names the entry behind every id')
    notes.append('a row edit is not the sequence-aware decode rule: a covered row is down-weighted wherever '
                 'that token appears, while --remap only suppresses a token that completes a whole table '
                 'sequence, so the weight side is strictly broader than the decode side and is not a '
                 'substitute for it')
    if cosines and min(cosines) < COS_FLOOR:
        problems.append(f'min cos on the edited rows {min(cosines)} is below the {COS_FLOOR} screening floor')
    report = {
        'schema': WEIGHTS_SCHEMA,
        'scope': ('Vocabulary-row edit of a quantized artifact: storage fidelity and byte accounting only. '
                  'No model is loaded, no logits or generations are read, and no behavioral claim is made; '
                  'the weight side of this card is gated for claims.'),
        'method': WEIGHTS_METHOD,
        'dry_run': not apply,
        'artifact': {'path': _portable_path(weights), 'bytes': weights.stat().st_size,
                     'sha256': _sha256_file(weights)},
        'table': {'schema': document.get('schema'), 'sha256': hashlib.sha256(raw).hexdigest(),
                  'entries': len(document['entries']),
                  'path': _portable_path(table) if table is not None else 'packaged asset'},
        'tokenizer': tokenizer['identity'],
        'discovery': {'n_vocab': discovery['n_vocab'], 'roles': roles, 'tied': discovery['tied'],
                      'vocab_wide_tensors': discovery['vocab_wide'],
                      'unmapped_vocab_wide': [item['name'] for item in discovery['unmapped']],
                      'how': '2-D tensors whose logical row count is the tokenizer vocabulary size; the '
                             'artifact name decides the role, no per-model table'},
        'coverage': {'ids': coverage['ids'], 'rows': coverage['rows'],
                     'entries': coverage['entries'], 'unrepresentable': coverage['unrepresentable'],
                     'note': 'union of both tokenizer contexts (bare and leading space); a phrase entry '
                             'covers every token of its split'},
        'edit': {'algebra': 'per-row rank-1 projection along the row itself: dW_i = -s (W_i u^T) u = -s W_i, '
                            'the same axis_delta helper a refusal axis uses',
                 'strength': strength, 'row_scale': round(1.0 - strength, 6),
                 'tensors_targeted': len(plans),
                 'rows_edited_per_tensor': {plan['record']['name']: plan['edited']['rows_edited']
                                            for plan in plans},
                 'row_detail': {plan['record']['name']: plan['edited']['rows'] for plan in plans}},
        'tensors': tensor_payloads,
        'untouched': {'tensors': discovery['tensor_count'] - len(plans),
                      'tensor_data_bytes': discovery['tensor_bytes'] - sum(plan['record']['bytes']
                                                                           for plan in plans),
                      'guarantee': 'edit_tensors copies the input and rewrites only the edited tensors; --apply '
                                   'measures it (no differing byte outside the edited ranges), and '
                                   'gguf-tensor-diff on the two files is the independent check'},
        'strength_sensitivity': sensitivity,
        'dry_run_measurement': {'sample': 'the block-aligned spans holding the covered rows only',
                                'tensors': dry['tensors'], 'min_cos': dry['min_cos']},
        'plan': {'writer': 'jdewitt.ggufio.edit_tensors (copy + requantize the edited tensors at their own '
                           'type; every other byte verbatim)',
                 'out': _portable_path(out_path) if out_path is not None else None,
                 'apply': bool(apply),
                 'revert_sidecar': (_portable_path(f'{out_path}.revert.npz')
                                    if out_path is not None else None)},
        'verdict': {'ok': not problems, 'cos_floor': COS_FLOOR,
                    'problems': problems, 'notes': notes},
    }

    if not apply:
        if json_out is not None:
            Path(json_out).parent.mkdir(parents=True, exist_ok=True)
            Path(json_out).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        return report

    from .ggufio import compare_edit, edit_tensors

    started = time.monotonic()
    assert out_path is not None  # validated above; apply without --out raises
    written = edit_tensors(str(weights), str(out_path), deltas=deltas)
    edited_ranges = [(plan['record']['offset'], plan['record']['bytes']) for plan in plans]
    runs = _diff_runs(weights, out_path, edited_ranges)
    sidecar_path = Path(f'{out_path}.revert.npz')
    sidecar = _write_revert_sidecar(weights, out_path, sidecar_path, runs)
    shipped = compare_edit(str(weights), str(out_path), deltas, rows=rows_by_name)
    report['apply'] = {
        'out': {'path': _portable_path(out_path), 'bytes': out_path.stat().st_size,
                'sha256': _sha256_file(out_path)},
        'writer': written,
        'differing_ranges_inside_edited_tensors': len(runs),
        'untouched': {'checked_bytes': weights.stat().st_size - sum(plan['record']['bytes'] for plan in plans),
                      'differing_bytes_outside_edited_tensors': 0,
                      'note': 'every differing byte was found inside an edited tensor range, so every '
                              'untouched tensor (and the header/metadata) is byte-identical by measurement'},
        'shipped': {'tensors': shipped['tensors'], 'min_cos': shipped['min_cos'],
                    'note': shipped['note']},
        'revert': sidecar,
        'seconds': round(time.monotonic() - started, 1),
    }
    shipped_min = [row.get('cos_touched_rows') for row in shipped['tensors']
                   if row.get('cos_touched_rows') is not None]
    if shipped_min and min(shipped_min) < COS_FLOOR:
        report['verdict']['problems'].append(
            f'shipped min cos on the edited rows {min(shipped_min)} is below the {COS_FLOOR} floor')
        report['verdict']['ok'] = False
    if json_out is not None:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report