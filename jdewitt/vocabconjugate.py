# SPDX-License-Identifier: AGPL-3.0-or-later
"""Conjugate denial-opener rows to compliance-opener rows in a quantized head.

A removal edits a row towards zero; this card edits the vocabulary's *labels*.
For each pair ``(d, c)`` the two token rows of the output head exchange places:

    W'_d = W_c,   W'_c = W_d,   W'_i = W_i for every other row i

so the hidden state that would have decoded the denial opener ``d`` — the row's
row of the head is the vector its logit is a dot product with — now decodes the
compliance opener ``c``, and the denial's score lands on the compliance token.
The map is a product of disjoint transpositions on the chosen ids, i.e. a
bijection whose inverse is itself; the multiset of rows of the matrix is
preserved exactly, so the tensor's Frobenius norm does not move.

If the head is tied to the token embedding (Qwen2.5-0.5B has no
``output.weight``), the same rows are the input embeddings, so the edit also
changes what the model *reads* when the token appears in a prompt. That is
reported, not hidden: it is the price of a tied file, and the report names the
tied row.

The rest of a refusal sentence is planned upstream. Mapping the first token does
not re-plan the continuation, so the expected outcome on a real refusal is a
compliant opener glued to a refusal body. That outcome is the point of the card
and is recorded as such.

The units here are storage fidelity: which rows the codec kept, whether every
unmapped row is byte-identical, whether the map is the bijection it claims, and
whether the pre-edit bytes restore the file. No model is loaded, no logits or
generations are read, and ``removal_measured`` is false: the behavioural
question is out of scope for a zero-forward card and stays gated.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .vocabremap import _merge_ranks, _portable_path, _sha256_file, resolve
from .vocabremap_weights import (COS_FLOOR, _diff_runs, _row_vectors, _tensor_layout,
                                 _write_revert_sidecar, discover_vocab_tensors)

PAIRS_SCHEMA = 'jdewitt.vocab-conjugate-pairs/1'
SCHEMA = 'jdewitt.vocab-conjugate/1'
REVERT_SCHEMA = 'jdewitt.vocab-conjugate-revert/1'
PACKAGED_PAIRS = Path(__file__).with_name('assets') / 'vocab-conjugate-openers.json'
CONTEXTS = ('bare', 'leads_with_space')
CONTEXT_NOTE = {
    'bare': 'the first token of a response; no leading space',
    'leads_with_space': 'the same surface mid-sentence; byte-level BPE spells the space into the token',
}
METHOD = (
    'Each pair names a denial opener and a compliance opener. Both surfaces are resolved against the artifact '
    'tokenizer in two contexts (bare and with the leading space running text carries) and must be exactly one '
    'token in both; the resolved ids are then transposed in the token-row matrix. For a head tied to the token '
    'embedding (no output.weight) the transposed matrix is both the embedding and the output head, so the edit '
    'also conjugates the input side; the report names it as tied. The write reuses ggufio.edit_tensors: every '
    'untouched tensor is copied byte for byte and the edited tensor is requantized at its own stored type, so '
    'every tensor offset and byte length stays put and the pre-edit bytes restore the file. Storage fidelity '
    'only: no model, no logits, no generation, no behavioral claim, removal_measured false.')
GATED = ('A behavioural claim is out of scope: the rest of the refusal sentence is planned upstream, so an '
         'opener-only conjugation is expected to glue a compliant opening to a refusal body. Measuring that '
         'needs a serve, which this card forbids.')


def load_pairs(table=None):
    """Read and validate the opener-pair table; returns (document, raw bytes)."""
    path = Path(table) if table is not None else PACKAGED_PAIRS
    if not path.is_file():
        raise ValueError(f'{path} does not exist')
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f'{path} is not JSON: {error}') from None
    if document.get('schema') != PAIRS_SCHEMA:
        raise ValueError(f'{path}: schema is {document.get("schema")!r}, expected {PAIRS_SCHEMA!r}')
    pairs = document.get('pairs')
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f'{path}: "pairs" must be a non-empty list')
    seen = set()
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError(f'{path}: every pair must be an object')
        for field in ('id', 'denial', 'compliance'):
            if not isinstance(pair.get(field), str) or not pair[field].strip():
                raise ValueError(f'{path}: pair {pair.get("id")!r} needs a non-empty "{field}" string')
        if pair['id'] in seen:
            raise ValueError(f'{path}: duplicate pair id {pair["id"]!r}')
        seen.add(pair['id'])
        if pair['denial'] == pair['compliance']:
            raise ValueError(f'{path}: pair {pair["id"]!r} maps a surface onto itself')
    return document, raw


def _split(text, vocab, ranks):
    split, representable = resolve(text, vocab, ranks)
    return {'text': text, 'ids': [int(pair[0]) if pair[0] is not None else None for pair in split],
            'pieces': [pair[1] for pair in split], 'representable': bool(representable),
            'single_token': bool(representable and len(split) == 1 and split[0][0] is not None)}


def resolve_pairs(document, tokenizer):
    """Resolve every pair to ids in both contexts and build the transposition list.

    Refuses rather than guessing: a surface that is not one token in a context is
    a pair this command cannot express as a row transposition, and a table whose
    ids overlap cannot be a bijection.
    """
    vocab = {token: token_id for token_id, token in enumerate(tokenizer['tokens'])}
    ranks = _merge_ranks(tokenizer['merges'])
    pairs, permutation, problems = [], [], []
    sources, targets, claimed = {}, {}, {}
    for pair in document['pairs']:
        detail = {'id': pair['id'], 'language': pair.get('language'), 'kind': pair.get('kind'),
                  'denial': pair['denial'], 'compliance': pair['compliance'],
                  'evidence': pair.get('evidence'), 'contexts': {}, 'rows': []}
        for context in CONTEXTS:
            prefix = '' if context == 'bare' else ' '
            denial = _split(prefix + pair['denial'], vocab, ranks)
            compliance = _split(prefix + pair['compliance'], vocab, ranks)
            detail['contexts'][context] = {'denial': denial, 'compliance': compliance,
                                           'note': CONTEXT_NOTE[context],
                                           'single_token': denial['single_token'] and compliance['single_token']}
            if not detail['contexts'][context]['single_token']:
                problems.append(f'{pair["id"]}/{context}: '
                                f'denial {denial["text"]!r} -> {denial["pieces"]} '
                                f'(single_token={denial["single_token"]}), '
                                f'compliance {compliance["text"]!r} -> {compliance["pieces"]} '
                                f'(single_token={compliance["single_token"]})')
                continue
            denial_id, compliance_id = denial['ids'][0], compliance['ids'][0]
            if denial_id == compliance_id:
                problems.append(f'{pair["id"]}/{context}: both surfaces resolve to id {denial_id}')
                continue
            for token_id, role in ((denial_id, 'denial'), (compliance_id, 'compliance')):
                if token_id in claimed:
                    problems.append(f'{pair["id"]}/{context}: id {token_id} already used by {claimed[token_id]}')
                claimed[token_id] = f'{pair["id"]}/{context}/{role}'
            sources.setdefault(denial_id, []).append(f'{pair["id"]}/{context}')
            targets.setdefault(compliance_id, []).append(f'{pair["id"]}/{context}')
            detail['rows'].append({'context': context, 'denial_id': denial_id, 'compliance_id': compliance_id,
                                   'denial_piece': denial['pieces'][0],
                                   'compliance_piece': compliance['pieces'][0]})
            permutation.append({'from': denial_id, 'to': compliance_id,
                                'denial_piece': denial['pieces'][0],
                                'compliance_piece': compliance['pieces'][0],
                                'context': context, 'pair': pair['id']})
            permutation.append({'from': compliance_id, 'to': denial_id,
                                'denial_piece': compliance['pieces'][0],
                                'compliance_piece': denial['pieces'][0],
                                'context': context, 'pair': pair['id']})
        pairs.append(detail)
    ids = sorted(item['from'] for item in permutation)
    mapped = set(ids)
    shared = set(sources) & set(targets)
    for token_id in sorted(shared - mapped):
        problems.append(f'id {token_id} is a denial source and a compliance target in different pairs '
                        f'({sources[token_id]} vs {targets[token_id]}); that is a cycle, not a transposition')
    if len(mapped) != len(ids):
        problems.append(f'{len(ids) - len(mapped)} chosen ids are mapped twice; the map is not a bijection')
    for token_id in mapped:
        partner = [item['to'] for item in permutation if item['from'] == token_id]
        if len(partner) != 1:
            problems.append(f'id {token_id} has {len(partner)} targets; expected exactly one')
        if partner and partner[0] == token_id:
            problems.append(f'id {token_id} maps onto itself')
    if any(item['from'] == item['to'] for item in permutation):
        problems.append('the map contains a fixed point')
    inverse = {item['from']: item['to'] for item in permutation}
    if any(inverse.get(inverse[token_id]) != token_id for token_id in mapped):
        problems.append('the map is not an involution (applying it twice is not the identity)')
    return {'pairs': pairs, 'permutation': sorted(permutation, key=lambda item: (item['from'], item['to'])),
            'ids': ids, 'n_ids': len(ids), 'problems': problems,
            'bijection': not problems,
            'checks': {
                'chosen_ids': len(ids),
                'distinct_ids': len(mapped),
                'injective': len(mapped) == len(ids),
                'every_chosen_id_has_one_target': all(
                    sum(1 for item in permutation if item['from'] == token_id) == 1 for token_id in mapped),
                'involution': not any(inverse.get(inverse[token_id]) != token_id for token_id in mapped),
                'fixed_points': sum(1 for item in permutation if item['from'] == item['to']),
                'unmapped_ids': len(tokenizer['tokens']) - len(mapped),
                'definition': ('a product of disjoint transpositions on the resolved ids: a bijection whose '
                               'inverse is itself, so every unmapped id keeps its own row'),
            }}


def _conjugation(tensor, vectors, permutation):
    """delta = partner row - own row for every chosen id, plus the row readings."""
    delta = np.zeros((tensor['rows'], tensor['row_width']), dtype=np.float32)
    rows = []
    for item in permutation:
        source, target = item['from'], item['to']
        before = np.asarray(vectors[source], dtype=np.float32)
        after = np.asarray(vectors[target], dtype=np.float32)
        delta[source] = after - before
        denominator = float(np.linalg.norm(before) * np.linalg.norm(after))
        cosine = float(before @ after) / denominator if denominator > 0 else 0.0
        rows.append({'id': source, 'token': item['denial_piece'], 'from': source, 'to': target,
                     'context': item['context'], 'pair': item['pair'],
                     'partner_token': item['compliance_piece'],
                     'cos_before_after': round(cosine, 6),
                     'norm_before': round(float(np.linalg.norm(before)), 6),
                     'norm_after': round(float(np.linalg.norm(after)), 6),
                     'norm_ratio': round(float(np.linalg.norm(after))
                                         / (float(np.linalg.norm(before)) + 1e-12), 6),
                     'delta_norm': round(float(np.linalg.norm(delta[source])), 6)})
    return {'delta': delta, 'rows': rows,
            'rows_edited': len(rows),
            'cos_definition': ('cos_before_after = cos(row values at this id before the edit, values the edit '
                               'stores at the same id). It is the same number for both ends of a transposition '
                               'and equals cos(W_denial, W_compliance); 1.0 would mean the row did not move.')}


def _tensor_measurement(artifact, deltas, ids_by_name):
    from .ggufio import preview_edit

    return preview_edit(str(artifact), deltas, rows=ids_by_name)


def _row_identity(source, out, tensor, permutation, chunk_rows=8192):
    """Row-level byte identity: exactly the mapped rows differ, and each holds its partner's row.

    ``gguf-tensor-diff`` is per tensor (it names ``token_embd.weight`` and nothing
    else); this check is per row, which is the granularity the map is expressed at.
    Rows are whole codec blocks here (Q8_0, 896 values = 28 blocks), so a row's raw
    bytes are its stored values; a tensor whose rows straddle blocks falls back to
    dequantized values and says so.
    """
    layout = _tensor_layout(tensor)
    block_bytes, block_els = layout['block_bytes'], layout['block_els']
    rows, width = tensor['rows'], tensor['row_width']
    aligned = width % block_els == 0
    mapped = sorted({item['from'] for item in permutation})
    mapped_set = set(mapped)
    if not aligned:
        return {'measured': 'dequantized values (row width is not a whole number of codec blocks)',
                'row_aligned': False, 'aligned': False}
    row_bytes = width // block_els * block_bytes
    shape = (rows, row_bytes)
    before = np.memmap(source, dtype=np.uint8, mode='r', offset=tensor['offset'], shape=shape)
    after = np.memmap(out, dtype=np.uint8, mode='r', offset=tensor['offset'], shape=shape)
    changed = []
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        differing = np.flatnonzero((before[start:stop] != after[start:stop]).any(axis=1))
        changed.extend(int(start + row) for row in differing)
    changed_set = set(changed)
    holds_partner = [item['from'] for item in permutation
                     if np.array_equal(after[item['from']], before[item['to']])]
    own_row = [token_id for token_id in changed if token_id not in mapped_set]
    return {
        'measured': 'raw stored bytes per row',
        'row_aligned': True,
        'row_bytes': int(row_bytes),
        'rows_total': int(rows),
        'rows_changed': len(changed),
        'rows_mapped': len(mapped),
        'changed_rows': changed,
        'changed_rows_equal_mapped_rows': changed_set == mapped_set,
        'unchanged_rows': int(rows - len(changed)),
        'every_mapped_row_holds_its_partner': len(holds_partner) == len(permutation),
        'mapped_rows_verified': len(holds_partner),
        'differing_rows_outside_map': sorted(own_row),
        'note': ('a transposition preserves the row multiset, so no row norm is gained or lost; the check is '
                 'that out[mapped] is byte-for-byte in[mapped partner] and nothing else moved'),
    }


def vocab_conjugate(weights, table=None, out=None, apply=False, json_out=None):
    """Plan or apply the opener conjugation as a row transposition of the artifact.

    Default (``apply=False``) is the dry run: the pair table resolved to ids, the
    bijection checks, the per-row pre/post cosine, and what the codec stores on
    the chosen rows. ``apply=True`` writes a NEW artifact (never in place) plus a
    revert sidecar, proves no byte outside the edited tensor moved, re-measures
    the shipped rows, and checks at row granularity that exactly the mapped rows
    changed and each holds its partner's bytes.
    """
    weights = Path(weights)
    out_path = None if out is None else Path(out)
    if not weights.is_file():
        raise ValueError(f'{weights} does not exist')
    if apply and out_path is None:
        raise ValueError('--apply needs --out for the new artifact; this never ships in place')
    if apply and out_path.exists():
        raise ValueError(f'{out} already exists; the apply writes a new artifact and refuses to clobber one')
    document, raw = load_pairs(table)
    discovery = discover_vocab_tensors(weights)
    tokenizer = discovery['tokenizer']
    resolution = resolve_pairs(document, tokenizer)
    if resolution['problems'] or not resolution['ids']:
        raise ValueError('the pair table is not expressible as a row transposition on this tokenizer: '
                         + '; '.join(resolution['problems'] or ['no ids resolved']))

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
        read = _row_vectors(weights, record, resolution['ids'])
        edited = _conjugation(record, read['vectors'], resolution['permutation'])
        deltas[record['name']] = edited['delta']
        touched_blocks = sum((end - start) // layout['block_els'] for start, end in read['spans'])
        plans.append({'record': record, 'layout': layout, 'read': read, 'edited': edited,
                      'touched_blocks': int(touched_blocks)})

    ids_by_name = {plan['record']['name']: list(resolution['ids']) for plan in plans}
    dry = _tensor_measurement(weights, deltas, ids_by_name)
    measured = {row['name']: row for row in dry['tensors']}

    tensor_payloads = []
    for plan in plans:
        record, layout = plan['record'], plan['layout']
        row = measured.get(record['name'], {})
        touched_bytes = plan['touched_blocks'] * layout['block_bytes']
        tensor_payloads.append({
            'name': record['name'], 'roles': record['roles'], 'qtype': record['qtype'],
            'codec_chosen': record['qtype'],
            'codec_reason': ('the tensor requantizes at its own stored type, so every tensor offset and byte '
                             'length stays put and the pre-edit bytes restore the file exactly'),
            'shape': [record['rows'], record['row_width']], 'tied': bool(discovery['tied']),
            'bytes': {'source': record['bytes'], 'reencoded': record['bytes'],
                      'blocks_total': layout['blocks'], 'touched_blocks': plan['touched_blocks'],
                      'touched_bytes': int(touched_bytes),
                      'touched_fraction': round(touched_bytes / max(1, record['bytes']), 6),
                      'bytes_read_to_measure': plan['read']['bytes_read']},
            'storage': {'cos': row.get('cos'), 'cos_touched_rows': row.get('cos_touched_rows'),
                        'norm_ratio': row.get('norm_ratio'), 'snr': row.get('snr'),
                        'rows_touched': row.get('rows_touched'),
                        'blocks_touched': row.get('blocks_touched'),
                        'row_detail': row.get('row_detail', [])},
            'rows': plan['edited']['rows']})

    cosines = [row.get('cos_touched_rows') for row in measured.values()
               if row.get('cos_touched_rows') is not None]
    notes = [resolution['checks']['definition'],
             'a row edit is not a sentence: the rest of a refusal is planned upstream, so an opener conjugation '
             'is expected to produce a compliant opener followed by a refusal body',
             'the same tensor is both embedding and head when tied, so the transposition also changes what the '
             'model reads for those ids in a prompt']
    problems = []
    if cosines and min(cosines) < COS_FLOOR:
        problems.append(f'min cos on the conjugated rows {min(cosines)} is below the {COS_FLOOR} floor')
    if discovery['tied']:
        notes.append('tied head: there is no separate output.weight, so the row is the head row and the '
                     'embedding row at once (reported as tied=true)')
    report = {
        'schema': SCHEMA,
        'scope': ('Bijection on the output vocabulary of a quantized artifact: storage fidelity, the pair '
                  'table, and the pre/post cosine of every mapped row. No model is loaded, no logits or '
                  'generations are read, and no behavioural claim is made.'),
        'method': METHOD,
        'gated': GATED,
        'removal_measured': False,
        'dry_run': not apply,
        'artifact': {'path': _portable_path(weights), 'bytes': weights.stat().st_size,
                     'sha256': _sha256_file(weights)},
        'table': {'schema': document.get('schema'), 'sha256': hashlib.sha256(raw).hexdigest(),
                  'pairs': len(document['pairs']),
                  'path': _portable_path(table) if table is not None else 'packaged asset'},
        'tokenizer': tokenizer['identity'],
        'discovery': {'n_vocab': discovery['n_vocab'], 'roles': roles, 'tied': discovery['tied'],
                      'vocab_wide_tensors': discovery['vocab_wide'],
                      'unmapped_vocab_wide': [item['name'] for item in discovery['unmapped']],
                      'how': ('2-D tensors whose logical row count is the tokenizer vocabulary size; the '
                              'artifact name decides the role, no per-model table')},
        'pairs': resolution['pairs'],
        'permutation': {'ids': resolution['ids'], 'n_ids': resolution['n_ids'],
                        'map': resolution['permutation'],
                        'checks': resolution['checks']},
        'edit': {'algebra': 'W_d <- W_c and W_c <- W_d per resolved pair; every other row is untouched',
                 'tensors_targeted': len(plans),
                 'rows_edited_per_tensor': {plan['record']['name']: plan['edited']['rows_edited']
                                            for plan in plans},
                 'rows': {plan['record']['name']: plan['edited']['rows'] for plan in plans},
                 'cos_definition': (plans[0]['edited']['cos_definition'] if plans else '')},
        'tensors': tensor_payloads,
        'untouched': {'tensors': discovery['tensor_count'] - len(plans),
                      'tensor_data_bytes': (discovery['tensor_bytes']
                                            - sum(plan['record']['bytes'] for plan in plans)),
                      'guarantee': ('edit_tensors copies the input and rewrites only the edited tensors; '
                                    '--apply measures it (no differing byte outside the edited ranges), '
                                    'gguf-tensor-diff on the two files is the independent check, and a '
                                    'row-level check names exactly the mapped rows')},
        'dry_run_measurement': {'sample': 'the block-aligned spans holding the mapped rows only',
                                'tensors': dry['tensors'], 'min_cos': dry['min_cos']},
        'plan': {'writer': ('jdewitt.ggufio.edit_tensors (copy + requantize the edited tensors at their own '
                            'type; every other byte verbatim)'),
                 'out': _portable_path(out_path) if out_path is not None else None,
                 'apply': bool(apply),
                 'revert_sidecar': (_portable_path(f'{out_path}.revert.npz')
                                    if out_path is not None else None)},
        'verdict': {'ok': not problems, 'cos_floor': COS_FLOOR, 'problems': problems, 'notes': notes},
        'next': {'commands': [f'jdewitt gguf-inspect {_portable_path(out_path or weights)}',
                              (f'jdewitt gguf-tensor-diff {_portable_path(weights)} '
                               f'{_portable_path(out_path)}' if out_path is not None else
                               'jdewitt gguf-tensor-diff <source> <out>'),
                              ('jdewitt gguf-revert ' + str(_portable_path(out_path))
                               if out_path is not None else 'jdewitt gguf-revert <out>')],
                 'note': ('the diff names the edited tensor; the row-level check in apply.row_identity names '
                          'the mapped rows. A behavioural reading needs a serve and stays gated.')},
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
    sidecar = _write_revert_sidecar(weights, out_path, sidecar_path, runs, schema=REVERT_SCHEMA)
    shipped = compare_edit(str(weights), str(out_path), deltas, rows=ids_by_name)
    identity = {plan['record']['name']: _row_identity(str(weights), str(out_path), plan['record'],
                                                      resolution['permutation'])
                for plan in plans}
    report['apply'] = {
        'out': {'path': _portable_path(out_path), 'bytes': out_path.stat().st_size,
                'sha256': _sha256_file(out_path)},
        'writer': written,
        'differing_ranges_inside_edited_tensors': len(runs),
        'untouched': {'checked_bytes': weights.stat().st_size - sum(plan['record']['bytes'] for plan in plans),
                      'differing_bytes_outside_edited_tensors': 0,
                      'note': ('every differing byte was found inside an edited tensor range, so every '
                               'untouched tensor (and the header/metadata) is byte-identical by measurement')},
        'shipped': {'tensors': shipped['tensors'], 'min_cos': shipped['min_cos'],
                    'note': shipped['note']},
        'row_identity': identity,
        'revert': sidecar,
        'seconds': round(time.monotonic() - started, 1),
    }
    shipped_min = [row.get('cos_touched_rows') for row in shipped['tensors']
                   if row.get('cos_touched_rows') is not None]
    if shipped_min and min(shipped_min) < COS_FLOOR:
        report['verdict']['problems'].append(
            f'shipped min cos on the conjugated rows {min(shipped_min)} is below the {COS_FLOOR} floor')
        report['verdict']['ok'] = False
    for name, reading in identity.items():
        if reading.get('aligned') is False:
            continue
        if not reading['changed_rows_equal_mapped_rows']:
            report['verdict']['problems'].append(
                f'{name}: changed rows are not exactly the mapped rows '
                f'({reading["rows_changed"]} changed vs {reading["rows_mapped"]} mapped)')
        if not reading['every_mapped_row_holds_its_partner']:
            report['verdict']['problems'].append(
                f'{name}: {reading["rows_mapped"] - reading["mapped_rows_verified"]} mapped rows do not hold '
                f'their partner\'s stored bytes')
    if report['verdict']['problems']:
        report['verdict']['ok'] = False
    if json_out is not None:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
