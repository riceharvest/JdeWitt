# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rewrite denial-opener rows to the compliance centroid, or off the denial axis.

The sibling card (`vocabconjugate`) swaps each denial row with its compliance
*partner*: a bijection on the chosen ids. This card keeps the id map intact and
changes the *values* of the denial rows instead, in three forms:

    centroid             W_d <- c                    (c = mean of the compliance rows)
    centroid-orthogonal  W_d <- c - (c.a) a          (the centroid off the denial axis)
    row-orthogonal       W_d <- W_d - (W_d.a) a      (each row off the denial axis)

with ``a = unit(mean(denial rows) - mean(compliance rows))`` the denial-vs-
compliance mean. The compliance rows themselves are never written, so they stay
the reference the centroid is taken over and the untouched control.

Every denial id is resolved from the same opener-pair table the conjugation uses
(bare and leading-space contexts), so the ids are the artifact's own tokenizer's,
not a per-model table. A tied artifact (Qwen2.5-0.5B has no ``output.weight``)
carries one token-row matrix, so the edit is the embedding row and the head row at
once; an untied artifact gets both matrices. The report says which.

Mode ``centroid`` makes every denial id decode the group mean on the head side and
embed the group mean on the input side; the row's norm becomes the centroid's.
The orthogonal forms delete only the shared denial direction: the row keeps its
own identity off that axis and loses exactly the norm it had along it, and its
cosine to the axis is zero by construction (the storage codec leaves a rounding
residual, which is reported rather than assumed away).

The rest of the refusal is planned upstream. Reading a denial id as compliance-
like does not re-plan the continuation, so the expected outcome is a compliant
reading glued to whatever the hidden plan was. That is the recorded negative the
card exists for, and no model is loaded to observe it: only tokenization, tensor
algebra, file bytes and the revert sidecar. ``removal_measured`` is false.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .vocabconjugate import load_pairs, resolve_pairs
from .vocabremap import _portable_path, _sha256_file
from .vocabremap_weights import (COS_FLOOR, _diff_runs, _row_vectors, _tensor_layout,
                                 _write_revert_sidecar, discover_vocab_tensors)

SCHEMA = 'jdewitt.vocab-centroid/1'
REVERT_SCHEMA = 'jdewitt.vocab-centroid-revert/1'
PACKAGED_PAIRS = Path(__file__).with_name('assets') / 'vocab-conjugate-openers.json'
MODES = ('centroid', 'centroid-orthogonal', 'row-orthogonal')
MODE_ALGEBRA = {
    'centroid': 'W_d <- c, the arithmetic mean of the compliance-opener rows',
    'centroid-orthogonal': 'W_d <- c - (c.a) a, the compliance centroid projected off the denial axis a',
    'row-orthogonal': 'W_d <- W_d - (W_d.a) a, each denial row projected off the denial axis a',
}
AXIS_DEFINITION = ('a = unit(mean(denial-opener rows) - mean(compliance-opener rows)); the two group '
                   'means are taken over the same rows the table resolves, so the axis is the group '
                   'contrast, not a fitted refusal direction')
METHOD = (
    'The opener-pair table is resolved against the artifact tokenizer in two contexts (bare and with the '
    'leading space running text carries) and must give exactly one token per surface in each. The denial '
    'ids are then rewritten as ROW VALUES while every id keeps its meaning: mode centroid writes the mean '
    'of the compliance-opener rows into each denial row, mode centroid-orthogonal writes that mean with its '
    'component along the denial-vs-compliance mean removed, and mode row-orthogonal removes that component '
    'from each denial row itself. The compliance rows are never edited: they stay the centroid and the '
    'untouched control. For a head tied to the token embedding (no output.weight) the same matrix is the '
    'embedding and the head, so the edit lands on both sides; the report names it as tied. The write reuses '
    'ggufio.edit_tensors: every untouched tensor is copied byte for byte and the edited tensor is '
    'requantized at its own stored type, so every tensor offset and byte length stays put and the pre-edit '
    'bytes restore the file. Storage fidelity only: no model, no logits, no generation, no behavioral '
    'claim, removal_measured false.')
GATED = ('A behavioural claim is out of scope: an opener-only row rewrite does not re-plan the '
         'continuation, so the model is expected to read a denial id as compliance-like and then continue '
         'with whatever the hidden plan already was. Measuring that needs a serve, which this card forbids.')


def _cos(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b) / denominator if denominator > 0 else 0.0


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError('the denial-vs-compliance mean is the zero vector; the table gives no axis')
    return vector / np.float32(norm), norm


def resolve_groups(document, tokenizer):
    """Resolve the table to denial ids (targets) and compliance ids (the centroid)."""
    resolution = resolve_pairs(document, tokenizer)
    problems = list(resolution['problems'])
    detail = {}
    for pair in resolution['pairs']:
        for row in pair['rows']:
            entry = {'pair': pair['id'], 'context': row['context'], 'language': pair.get('language'),
                     'denial': pair['denial'], 'compliance': pair['compliance'],
                     'denial_token': row['denial_piece'], 'compliance_token': row['compliance_piece'],
                     'denial_id': row['denial_id'], 'compliance_id': row['compliance_id']}
            detail.setdefault(row['denial_id'], []).append(entry)
    denial_ids = sorted(detail)
    compliance_ids = sorted({item['compliance_id'] for entries in detail.values() for item in entries})
    for token_id, entries in detail.items():
        if len(entries) != 1:
            problems.append(f'denial id {token_id} is claimed by {len(entries)} pairs/contexts '
                            f'({[entry["pair"] + "/" + entry["context"] for entry in entries]}); '
                            f'exactly one is required')
    overlap = sorted(set(denial_ids) & set(compliance_ids))
    if overlap:
        problems.append(f'ids {overlap} are both denial targets and compliance references; the centroid '
                        f'would be taken over a row the edit rewrites')
    if not denial_ids or not compliance_ids:
        problems.append('the table resolves no denial target or no compliance reference on this tokenizer')
    return {'rows': detail, 'denial_ids': denial_ids, 'compliance_ids': compliance_ids,
            'n_denial': len(denial_ids), 'n_compliance': len(compliance_ids),
            'problems': problems, 'ok': not problems,
            'definition': ('denial ids are the rows the edit rewrites; compliance ids are the rows it '
                           'averages and never writes, so each row is edited or a reference, never both')}


def _groups(vectors, groups):
    """The centroid, the axis, and the per-row readings both are measured against."""
    compliance = np.stack([vectors[token_id] for token_id in groups['compliance_ids']])
    denial = np.stack([vectors[token_id] for token_id in groups['denial_ids']])
    centroid = compliance.mean(axis=0, dtype=np.float32)
    denial_mean = denial.mean(axis=0, dtype=np.float32)
    axis, axis_norm = _unit(denial_mean - centroid)
    readings = {name: {'ids': list(groups[f'{name}_ids']),
                       'norm_of_group_mean': round(float(np.linalg.norm(matrix.mean(axis=0))), 6),
                       'cos_to_axis': [round(_cos(row, axis), 6) for row in matrix],
                       'mean_cos_to_axis': round(float(np.mean([_cos(row, axis) for row in matrix])), 6),
                       'min_cos_to_axis': round(float(np.min([_cos(row, axis) for row in matrix])), 6),
                       'max_cos_to_axis': round(float(np.max([_cos(row, axis) for row in matrix])), 6)}
                for name, matrix in (('denial', denial), ('compliance', compliance))}
    return {'centroid': centroid, 'axis': axis,
            'info': {'n_denial': len(groups['denial_ids']), 'n_compliance': len(groups['compliance_ids']),
                     'axis': AXIS_DEFINITION,
                     'axis_norm_before_normalisation': round(axis_norm, 6),
                     'centroid_norm': round(float(np.linalg.norm(centroid)), 6),
                     'cos_denial_mean_compliance_centroid': round(_cos(denial_mean, centroid), 6),
                     'group_mean_projection_on_axis': {
                         'denial': round(float(denial_mean @ axis), 6),
                         'compliance': round(float(centroid @ axis), 6)},
                     'groups': readings,
                     'note': ('the axis is the contrast of the two group means; a group is "denial-like" '
                              'on it when its mean cosine is positive and "compliance-like" when negative')}}


def _edit_rows(record, vectors, groups, mode):
    """The float32 delta for the denial rows, plus the pre/post readings per id."""
    if mode not in MODES:
        raise ValueError(f'unknown mode {mode!r}; expected one of {MODES}')
    derived = _groups(vectors, groups)
    centroid, axis = derived['centroid'], derived['axis']
    centroid_after = centroid - np.float32(centroid @ axis) * axis
    rows, targets = [], {}
    delta = np.zeros((record['rows'], record['row_width']), dtype=np.float32)
    for entry in (item for token_id in groups['denial_ids'] for item in groups['rows'][token_id]):
        token_id = entry['denial_id']
        before = np.asarray(vectors[token_id], dtype=np.float32)
        if mode == 'centroid':
            after = np.array(centroid, dtype=np.float32)
        elif mode == 'centroid-orthogonal':
            after = np.array(centroid_after, dtype=np.float32)
        else:
            after = before - np.float32(before @ axis) * axis
        delta[token_id] = after - before
        targets[token_id] = np.asarray(after, dtype=np.float32)
        norm_before = float(np.linalg.norm(before))
        norm_after = float(np.linalg.norm(after))
        rows.append({
            'id': token_id, 'token': entry['denial_token'], 'pair': entry['pair'],
            'context': entry['context'], 'language': entry['language'],
            'denial': entry['denial'], 'compliance': entry['compliance'],
            'compliance_id': entry['compliance_id'], 'compliance_token': entry['compliance_token'],
            'cos_before_centroid': round(_cos(before, centroid), 6),
            'cos_after_centroid': round(_cos(after, centroid), 6),
            'cos_before_axis': round(_cos(before, axis), 6),
            'cos_after_axis': round(_cos(after, axis), 6),
            'norm_before': round(norm_before, 6), 'norm_after': round(norm_after, 6),
            'norm_change': round(norm_after - norm_before, 6),
            'norm_ratio': round(norm_after / (norm_before + 1e-12), 6),
            'delta_norm': round(float(np.linalg.norm(delta[token_id])), 6),
            'cos_to_own_partner_before': round(_cos(before, vectors[entry['compliance_id']]), 6),
            'cos_to_own_partner_after': round(_cos(after, vectors[entry['compliance_id']]), 6),
        })
    return {'delta': delta, 'rows': rows, 'derived': derived, 'rows_edited': len(rows),
            'targets': targets,
            'cos_definition': ('cos_*_centroid = cos(row values at this id before the edit, the compliance '
                               'centroid c); cos_*_axis = cos(row, a) with a the unit denial-vs-compliance '
                               'mean. post values are the intended algebra, before the codec; the shipped '
                               'file\'s realized values are measured separately.')}


def _row_identity(source, out, tensor, groups, targets, chunk_rows=8192):
    """Row-level byte identity plus what the shipped rows actually hold.

    `gguf-tensor-diff` is per tensor; this is the granularity the edit is expressed
    at. Rows are whole codec blocks here (Q8_0, 896 values = 28 blocks), so a row's
    raw bytes are its stored values; a tensor whose rows straddle blocks falls back
    to dequantized values and says so.
    """
    layout = _tensor_layout(tensor)
    block_bytes, block_els = layout['block_bytes'], layout['block_els']
    rows, width = tensor['rows'], tensor['row_width']
    aligned = width % block_els == 0
    mapped = sorted(groups['denial_ids'])
    mapped_set = set(mapped)
    if not aligned:
        stored = _row_vectors(out, tensor, mapped)
        return {'measured': 'dequantized values (row width is not a whole number of codec blocks)',
                'row_aligned': False, 'aligned': False,
                'realized_cos_to_intended': {token_id: round(_cos(stored['vectors'][token_id],
                                                                  targets[token_id]), 6)
                                             for token_id in mapped}}
    row_bytes = width // block_els * block_bytes
    shape = (rows, row_bytes)
    before = np.memmap(source, dtype=np.uint8, mode='r', offset=tensor['offset'], shape=shape)
    after = np.memmap(out, dtype=np.uint8, mode='r', offset=tensor['offset'], shape=shape)
    changed, per_row_bytes = [], {}
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        differing = np.flatnonzero((before[start:stop] != after[start:stop]).any(axis=1))
        for row in differing:
            row = int(start + row)
            changed.append(row)
            per_row_bytes[row] = int((before[row] != after[row]).sum())
    changed_set = set(changed)
    stored = _row_vectors(out, tensor, mapped)
    realized = {token_id: round(_cos(stored['vectors'][token_id], targets[token_id]), 6)
                for token_id in mapped}
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
        'differing_bytes_per_changed_row': per_row_bytes,
        'differing_bytes_total': int(sum(per_row_bytes.values())),
        'reference_rows_changed': sorted(changed_set & set(groups['compliance_ids'])),
        'differing_rows_outside_map': sorted(changed_set - mapped_set),
        'realized_cos_to_intended': realized,
        'realized_cos_min': min(realized.values()) if realized else None,
        'note': ('the compliance rows are the centroid and the control: they must not move. The change '
                 'is value-level, so every denial row is expected to differ and every other row to be '
                 'byte-identical'),
    }


def _stored_cosines(out, tensor, derived, rows_edited):
    """cos(shipped denial row, the pristine compliance centroid) and cos against the pristine axis.

    The reference geometry is the artifact's BEFORE the edit. Recomputing the axis
    from the shipped file would be circular: in mode ``centroid`` the denial side of
    the contrast has just been set equal to the compliance side, so the shipped
    axis is numerical noise. The compliance rows are proven byte-identical by the
    row check, so the centroid taken over them is the same vector either way.
    """
    ids = sorted({row['id'] for row in rows_edited})
    stored = _row_vectors(out, tensor, ids)['vectors']
    centroid, axis = derived['centroid'], derived['axis']
    return {'centroid': {token_id: round(_cos(stored[token_id], centroid), 6) for token_id in ids},
            'axis': {token_id: round(_cos(stored[token_id], axis), 6) for token_id in ids},
            'max_abs_axis': round(max(abs(_cos(stored[token_id], axis)) for token_id in ids), 6),
            'min_centroid': round(min(_cos(stored[token_id], centroid) for token_id in ids), 6),
            'reference': ('centroid and axis are the pristine artifact\'s, taken over rows the row check '
                          'proves did not move; the shipped rows are decoded from the output file')}


def vocab_centroid(weights, table=None, mode='centroid', out=None, apply=False, json_out=None):
    """Plan or apply the denial-row rewrite; never edits the input in place."""
    weights = Path(weights)
    out_path = None if out is None else Path(out)
    if mode not in MODES:
        raise ValueError(f'unknown --mode {mode!r}; expected one of {MODES}')
    if not weights.is_file():
        raise ValueError(f'{weights} does not exist')
    if apply and out_path is None:
        raise ValueError('--apply needs --out for the new artifact; this never ships in place')
    if apply and out_path.exists():
        raise ValueError(f'{out} already exists; the apply writes a new artifact and refuses to clobber one')
    document, raw = load_pairs(table)
    discovery = discover_vocab_tensors(weights)
    tokenizer = discovery['tokenizer']
    groups = resolve_groups(document, tokenizer)
    if not groups['ok']:
        raise ValueError('the opener table is not expressible as a denial-row rewrite on this tokenizer: '
                         + '; '.join(groups['problems']))

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
        read = _row_vectors(weights, record, sorted(set(groups['denial_ids']) | set(groups['compliance_ids'])))
        edited = _edit_rows(record, read['vectors'], groups, mode)
        deltas[record['name']] = edited['delta']
        touched_blocks = sum((end - start) // layout['block_els'] for start, end in read['spans'])
        plans.append({'record': record, 'layout': layout, 'read': read, 'edited': edited,
                      'touched_blocks': int(touched_blocks)})

    ids_by_name = {plan['record']['name']: list(groups['denial_ids']) for plan in plans}
    from .ggufio import preview_edit
    dry = preview_edit(str(weights), deltas, rows=ids_by_name)
    measured = {row['name']: row for row in dry['tensors']}

    tensor_payloads = []
    for plan in plans:
        record, layout = plan['record'], plan['layout']
        row = measured.get(record['name'], {})
        touched_bytes = plan['touched_blocks'] * layout['block_bytes']
        tensor_payloads.append({
            'name': record['name'], 'roles': record['roles'], 'qtype': record['qtype'],
            'codec_chosen': record['qtype'],
            'codec_reason': ('the tensor requantizes at its own stored type, so every tensor offset and '
                             'byte length stays put and the pre-edit bytes restore the file exactly'),
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
            'rows': plan['edited']['rows'],
            'groups': plan['edited']['derived']['info'],
            'targets': {'mode': mode, 'algebra': MODE_ALGEBRA[mode],
                        'same_target_for_every_denial_id': mode != 'row-orthogonal',
                        'denial_ids': groups['denial_ids'],
                        'compliance_ids': groups['compliance_ids']}})

    cosines = [row.get('cos_touched_rows') for row in measured.values()
               if row.get('cos_touched_rows') is not None]
    notes = [MODE_ALGEBRA[mode],
             'the id map is intact: every denial id still decodes as its own token, and only its row '
             'values moved',
             'a row edit is not a sentence: the rest of a refusal is planned upstream, so an opener '
             'rewrite is expected to read compliance-like and continue with the hidden plan',
             'the compliance rows are the centroid and the control, and the edit never writes them']
    problems = []
    if cosines and min(cosines) < COS_FLOOR:
        problems.append(f'min cos on the rewritten rows {min(cosines)} is below the {COS_FLOOR} floor')
    if discovery['tied']:
        notes.append('tied head: there is no separate output.weight, so the row is the head row and the '
                     'embedding row at once (reported as tied=true)')
    report = {
        'schema': SCHEMA,
        'scope': ('Denial-opener rows rewritten as values on a quantized artifact: the id map is left '
                  'intact, the compliance centroid and the denial-vs-compliance mean are measured, and '
                  'every mapped row gets a pre/post table. No model is loaded, no logits or generations '
                  'are read, and no behavioural claim is made.'),
        'method': METHOD,
        'gated': GATED,
        'removal_measured': False,
        'dry_run': not apply,
        'mode': mode,
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
        'groups': {'definition': groups['definition'], 'denial_ids': groups['denial_ids'],
                   'compliance_ids': groups['compliance_ids'], 'n_denial': groups['n_denial'],
                   'n_compliance': groups['n_compliance'],
                   'denial_rows': [item for token_id in groups['denial_ids']
                                   for item in groups['rows'][token_id]],
                   'checks': {'problems': groups['problems'], 'ok': groups['ok']}},
        'edit': {'algebra': MODE_ALGEBRA[mode], 'axis': AXIS_DEFINITION,
                 'tensors_targeted': len(plans),
                 'rows_edited_per_tensor': {plan['record']['name']: plan['edited']['rows_edited']
                                            for plan in plans},
                 'ids_per_tensor': ids_by_name,
                 'rows': {plan['record']['name']: plan['edited']['rows'] for plan in plans},
                 'cos_definition': (plans[0]['edited']['cos_definition'] if plans else '')},
        'tensors': tensor_payloads,
        'untouched': {'tensors': discovery['tensor_count'] - len(plans),
                      'tensor_data_bytes': (discovery['tensor_bytes']
                                            - sum(plan['record']['bytes'] for plan in plans)),
                      'guarantee': ('edit_tensors copies the input and rewrites only the edited tensors; '
                                    '--apply measures it (no differing byte outside the edited ranges), '
                                    'gguf-tensor-diff on the two files is the independent check, and a '
                                    'row-level check names exactly the denial rows')},
        'dry_run_measurement': {'sample': 'the block-aligned spans holding the denial rows only',
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
                 'note': ('the diff names the edited tensor; the row-level check in apply.row_identity '
                          'names the denial rows. A behavioural reading needs a serve and stays gated.')},
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
                                                      groups, plan['edited']['targets'])
                for plan in plans}
    stored = {plan['record']['name']: _stored_cosines(str(out_path), plan['record'],
                                                      plan['edited']['derived'], plan['edited']['rows'])
              for plan in plans}
    report['apply'] = {
        'out': {'path': _portable_path(out_path), 'bytes': out_path.stat().st_size,
                'sha256': _sha256_file(out_path)},
        'writer': written,
        'differing_ranges_inside_edited_tensors': len(runs),
        'bytes_changed': {'differing_ranges': len(runs),
                          'differing_bytes': int(sum(end - start for start, end in runs)),
                          'ranges': [[int(start), int(end)] for start, end in runs],
                          'fraction_of_file': round(sum(end - start for start, end in runs)
                                                    / weights.stat().st_size, 8),
                          'attribution': 'the runs, wholly inside the edited tensors; per-row counts are '
                                         'in row_identity'},
        'untouched': {'checked_bytes': weights.stat().st_size - sum(plan['record']['bytes'] for plan in plans),
                      'differing_bytes_outside_edited_tensors': 0,
                      'note': ('every differing byte was found inside an edited tensor range, so every '
                               'untouched tensor (and the header/metadata) is byte-identical by measurement')},
        'shipped': {'tensors': shipped['tensors'], 'min_cos': shipped['min_cos'],
                    'note': shipped['note']},
        'row_identity': identity,
        'stored_cosines': stored,
        'revert': sidecar,
        'seconds': round(time.monotonic() - started, 1),
    }
    shipped_min = [row.get('cos_touched_rows') for row in shipped['tensors']
                   if row.get('cos_touched_rows') is not None]
    if shipped_min and min(shipped_min) < COS_FLOOR:
        report['verdict']['problems'].append(
            f'shipped min cos on the rewritten rows {min(shipped_min)} is below the {COS_FLOOR} floor')
    for name, reading in identity.items():
        if reading.get('aligned') is False:
            continue
        if not reading['changed_rows_equal_mapped_rows']:
            report['verdict']['problems'].append(
                f'{name}: changed rows are not exactly the denial rows '
                f'({reading["rows_changed"]} changed vs {reading["rows_mapped"]} mapped)')
        if reading['reference_rows_changed']:
            report['verdict']['problems'].append(
                f'{name}: compliance reference rows changed: {reading["reference_rows_changed"]}')
        if reading['realized_cos_min'] is not None and reading['realized_cos_min'] < COS_FLOOR:
            report['verdict']['problems'].append(
                f'{name}: a rewritten row stores cos {reading["realized_cos_min"]} with its intended '
                f'target, below the {COS_FLOOR} floor')
    for name, reading in stored.items():
        if mode in ('centroid-orthogonal', 'row-orthogonal') and reading['max_abs_axis'] > 1e-2:
            report['verdict']['problems'].append(
                f'{name}: a rewritten row keeps |cos| {reading["max_abs_axis"]} on the denial axis, '
                f'which is not orthogonal at storage resolution')
    report['verdict']['ok'] = not report['verdict']['problems']
    if json_out is not None:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
