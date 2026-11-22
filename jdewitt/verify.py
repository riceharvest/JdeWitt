# SPDX-License-Identifier: AGPL-3.0-or-later
"""Provenance checks from files alone: recompute an edit fingerprint, test same-edit.

A release says "81 matrices changed, 1,111 untouched". ``rewrite.rewrite_with_edits``
emits that as a fingerprint; this module recomputes what the FILES can prove of it from
the pristine and edited artifacts, byte comparison first, dequantized difference second.

What the files cannot prove is stated, not faked: the writer's ``edit_sha256`` hashes the
fitted axes and ``scale`` multiplies them, but a file only holds ``-scale * a^T (a W)``
re-encoded at a lower precision, so the axes, their norms and the scale are not
recoverable bit-exactly. Those fields come back ``not_recomputable``; what is recomputed
is the changed-tensor set, counts, layers, written type and the per-layer delta norms.
Every report states sampled vs total so a partial check cannot read as a full one.
"""
from __future__ import annotations

import hashlib
import json
import re
import zlib
from pathlib import Path

import numpy as np

from .quants import type_name_for_id

FLOATS = {'F32': np.float32, 'F16': np.float16}
RECOMPUTABLE = ('edited', 'untouched', 'layers', 'edit_type')
NOT_RECOMPUTABLE = {
    'edit_sha256': 'hashes the fitted axes; the file holds the re-encoded delta, not the axis',
    'axis_norm_by_layer': 'axis norm and scale are inseparable in the stored delta',
    'scale': 'scale multiplies the axis; only their product survives in the file',
    'axis_space': 'output/input projection is not identifiable from one file pair',
    'certified': 'certification is the agent shape read on the shipped bytes (R2); a file field cannot carry it',
    'adapter_screen_state': 'screened on the adapter over the pristine artifact, not on these bytes (R8)',
    'screen_source': 'provenance of the screen, not a property of these bytes',
    'residual_refusals': 'residual rows were screened on the adapter curve, not on these bytes',
    'residual_undecided': 'an undecided row is not recomputable from file bytes',
    'residual_rows': 'row ids from the adapter screen; there are no bytes to recompute them from',
    'best_effort': 'a process label, not a property of these bytes',
}


def _layer(name):
    match = re.match(r'blk\.(\d+)\.', name)
    return int(match.group(1)) if match else None


def _tensors(path):
    """(container, {name: info}) from the header alone; tensor bytes are memory-mapped.

    Parsing the container directly skips gguf-py's per-string vocabulary decode, which
    costs seconds per file and says nothing about the weights.
    """
    from .ggufwrite import parse_container, tensor_nbytes

    header = parse_container(path)
    mapped = np.memmap(str(path), dtype=np.uint8, mode='r')
    tensors = {}
    for entry in header['tensors']:
        dims = [int(x) for x in entry['dims']]
        start = header['data_base'] + entry['offset']
        raw = mapped[start:start + tensor_nbytes(dims, entry['qtype'])]
        rows = int(np.prod(dims[1:])) if len(dims) > 1 else 1
        tensors[entry['name']] = {
            'name': entry['name'], 'dims': dims, 'raw': raw.reshape(rows, -1),
            'type': type_name_for_id(entry['qtype'])}
    return header, tensors


def _same_bytes(a, b):
    if a['type'] != b['type'] or a['dims'] != b['dims']:
        return False
    left, right = a['raw'], b['raw']
    return all(np.array_equal(left[i:i + 256], right[i:i + 256])
               for i in range(0, left.shape[0], 256))


def _decode_rows(tensor, rows=None):
    """float32 (n, W) rows of a tensor, any source quant type; (W,) tensors are one row."""
    from .ggufio import decode_values

    picked = tensor['raw'] if rows is None else tensor['raw'][rows]
    name = tensor['type']
    if name in FLOATS:
        values = np.ascontiguousarray(picked).view(FLOATS[name]).astype(np.float32)
    else:
        # one call over the packed rows: per-row calls cost ~2 ms of overhead each
        values = decode_values(np.ascontiguousarray(picked).reshape(-1), name)
    return values.reshape(-1, tensor['dims'][0])


def _metadata_fingerprint(header):
    """First string metadata value under a *fingerprint* key that parses as JSON."""
    blob = header['kv_bytes']
    for match in re.finditer(rb'[\w.\-]*fingerprint[\w.\-]*', blob, re.IGNORECASE):
        key, at = match.group(), match.start()
        if at < 8 or int.from_bytes(blob[at - 8:at], 'little') != len(key):
            continue
        end = match.end()
        if int.from_bytes(blob[end:end + 4], 'little') != 8:
            continue
        size = int.from_bytes(blob[end + 4:end + 12], 'little')
        try:
            block = json.loads(blob[end + 12:end + 12 + size])
        except ValueError:
            continue
        if isinstance(block, dict):
            return key.decode(), block
    return None, None


def _changed(base, other):
    """Names of tensors present in both that differ by type, shape or bytes (+ set drift)."""
    changed = [name for name, tensor in base.items()
               if name in other and not _same_bytes(tensor, other[name])]
    return changed, sorted(set(base) - set(other)), sorted(set(other) - set(base))


def recompute(pristine, edited):
    """Fingerprint recomputed from two files alone, plus per-tensor rows."""
    _, base = _tensors(pristine)
    _, edit = _tensors(edited)
    changed, missing, extra = _changed(base, edit)
    rows, layer_norms, to_types = [], {}, set()
    digest = hashlib.sha256()
    for name in sorted(changed):
        before, after = base[name], edit[name]
        to_types.add(after['type'])
        digest.update(f"{name}|{after['type']}|{after['dims']}\n".encode())
        row = {'name': name, 'from_type': before['type'], 'to_type': after['type']}
        if before['dims'] != after['dims']:
            row['error'] = 'shape changed'
            rows.append(row)
            continue
        base_values, edit_values = _decode_rows(before), _decode_rows(after)
        delta = float(np.linalg.norm(edit_values - base_values))
        norm = float(np.linalg.norm(base_values))
        row['delta_norm'] = round(delta, 4)
        row['rel_delta'] = round(delta / (norm + 1e-12), 6)
        layer = _layer(name)
        if layer is not None:
            layer_norms[layer] = round(layer_norms.get(layer, 0.0) + delta, 4)
        rows.append(row)
    layers = sorted(layer_norms)
    return {
        'pristine': str(pristine), 'edited': str(edited),
        'tensors_total': len(base), 'tensors_compared': len(base) - len(missing),
        'tensors_decoded': len(rows),
        'fingerprint': {'edited': len(changed), 'untouched': len(base) - len(missing) - len(changed),
                        'layers': layers, 'delta_norm_by_layer': layer_norms,
                        'edit_type': sorted(to_types)[0] if len(to_types) == 1 else sorted(to_types),
                        'changed_set_sha256': digest.hexdigest()[:32]},
        'missing_from_edited': missing[:16], 'extra_in_edited': extra[:16],
        'changed': rows,
        'metadata': {},
    }


def compare_claim(claim, recomputed):
    """Field-by-field check of a writer fingerprint block against the recomputation."""
    got = recomputed['fingerprint']
    fields, mismatched = {}, []
    for key in RECOMPUTABLE:
        if key not in claim:
            fields[key] = {'status': 'absent_in_claim'}
            continue
        mine = got[key]
        if key == 'edit_type' and not isinstance(claim[key], list):
            mine = mine if isinstance(mine, str) else None
        ok = claim[key] == mine
        fields[key] = {'status': 'match' if ok else 'mismatch', 'claimed': claim[key], 'recomputed': mine}
        if not ok:
            mismatched.append(key)
    for key, reason in NOT_RECOMPUTABLE.items():
        if key in claim:
            fields[key] = {'status': 'not_recomputable', 'reason': reason}
    checked = sum(1 for item in fields.values() if item['status'] in ('match', 'mismatch'))
    return {'fields': fields, 'fields_checked': checked, 'fields_claimed': len(claim),
            'mismatched': mismatched}


def verify_pair(pristine, edited, claim=None):
    """Verdict recomputes | mismatch | unverifiable, with the reason and the numbers."""
    report = recompute(pristine, edited)
    source = 'claim' if claim is not None else None
    if claim is None:
        header, _ = _tensors(edited)
        key, claim = _metadata_fingerprint(header)
        source = f'metadata:{key}' if key else None
        report['metadata'] = {'fingerprint_key': key}
    if claim is None:
        report.update(verdict='unverifiable', claim_source=None,
                      reason='no fingerprint in the edited file metadata and no --claim; '
                             'recomputed values are reported but there is nothing to compare')
        return report
    check = compare_claim(claim, report)
    report.update(claim_source=source, comparison=check)
    if check['mismatched']:
        report.update(verdict='mismatch', reason='fields differ: ' + ', '.join(check['mismatched']))
    elif check['fields_checked'] == 0:
        report.update(verdict='unverifiable', reason='claim has no field the files can recompute')
    else:
        skipped = [k for k, v in check['fields'].items() if v['status'] == 'not_recomputable']
        report.update(verdict='recomputes',
                      reason=f"{check['fields_checked']} of {check['fields_claimed']} claimed fields "
                             f'recomputed and match; not recomputable: {", ".join(skipped) or "none"}')
    return report


def same_edit(pristine, first, second, rows=8, threshold=0.98, min_overlap=0.5):
    """Do two edited artifacts make the same change to their common base?

    Each file is differenced against the base (edited - pristine on the same rows), so
    quantization of the shared base cancels; the two DELTAS are compared, never the raw
    files. Per changed tensor in both: row 0 plus ``rows`` seeded rows, cosine of the
    concatenated delta rows.
    """
    _, base = _tensors(pristine)
    _, one = _tensors(first)
    _, two = _tensors(second)
    changed_one, _m1, _e1 = _changed(base, one)
    changed_two, _m2, _e2 = _changed(base, two)
    both = sorted(set(changed_one) & set(changed_two))
    union = set(changed_one) | set(changed_two)
    overlap = len(both) / len(union) if union else 0.0
    table, skipped = [], []
    for name in both:
        shapes = {tuple(t[name]['dims']) for t in (base, one, two)}
        if len(shapes) != 1:
            skipped.append({'name': name, 'reason': 'shape differs between files'})
            continue
        n_rows = int(np.prod(next(iter(shapes))[1:])) if len(next(iter(shapes))) > 1 else 1
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        picks = sorted({0, *rng.choice(n_rows, size=min(rows, n_rows), replace=False).tolist()})
        delta_one = _decode_rows(one[name], picks) - _decode_rows(base[name], picks)
        delta_two = _decode_rows(two[name], picks) - _decode_rows(base[name], picks)
        a, b = float(np.linalg.norm(delta_one)), float(np.linalg.norm(delta_two))
        if a == 0 or b == 0:
            skipped.append({'name': name, 'reason': 'zero delta on the sampled rows'})
            continue
        table.append({'name': name, 'rows': len(picks), 'rows_total': n_rows,
                      'cos': round(float((delta_one * delta_two).sum() / (a * b)), 4),
                      'norm_ratio': round(b / a, 4)})
    cosines = [row['cos'] for row in table]
    median = float(np.median(cosines)) if cosines else None
    if not cosines:
        verdict, reason = 'unverifiable', 'no changed tensor shared by both files had a sampled delta'
    elif median is None or overlap < min_overlap:
        verdict, reason = 'different edit', f'changed-set overlap {overlap:.3f} < {min_overlap}'
    elif median < threshold:
        verdict, reason = 'different edit', f'median cos {median:.4f} < {threshold}'
    else:
        verdict = 'same edit'
        reason = f'median cos {median:.4f} >= {threshold} and overlap {overlap:.3f} >= {min_overlap}'
    return {
        'pristine': str(pristine), 'first': str(first), 'second': str(second),
        'verdict': verdict, 'reason': reason,
        'threshold': threshold, 'min_overlap': min_overlap, 'rows_per_tensor': rows + 1,
        'overlap': round(overlap, 4),
        'changed_first': len(changed_one), 'changed_second': len(changed_two),
        'changed_both': len(both), 'changed_union': len(union),
        'tensors_total': len(base), 'tensors_sampled': len(table),
        'tensors_skipped': skipped[:16],
        'median_cos': None if median is None else round(median, 4),
        'min_cos': min(cosines) if cosines else None,
        'coverage': f'{len(table)} of {len(both)} shared changed tensors sampled '
                    f'({len(base)} tensors in base); rows are sampled, not exhaustive',
        'tensors': table,
    }


def verify_files(pristine, edited, claim=None, other=None, rows=8, threshold=0.98,
                 min_overlap=0.5):
    for path in (pristine, edited, other, claim):
        if path is not None and not Path(path).is_file():
            raise ValueError(f'{path} does not exist')
    if other is not None:
        if claim is not None:
            raise ValueError('--claim and --other are separate checks; run them one at a time')
        return same_edit(pristine, edited, other, rows, threshold, min_overlap)
    block = None
    if claim is not None:
        block = json.loads(Path(claim).read_text())
        block = block.get('fingerprint', block) if isinstance(block, dict) else None
        if not isinstance(block, dict):
            raise ValueError(f'{claim} is not a fingerprint JSON object')
    return verify_pair(pristine, edited, block)
