# SPDX-License-Identifier: AGPL-3.0-or-later
"""Targeted re-encode: remove a direction from YOUR quantized artifact.

The point of this module is the user's own file. A rank-1 removal computed in fp32 and
patched into a Q4_K tile is below the 4-bit grid — ``ggufio.preview_edit`` measures cos
~0.45 between the intended projection and what the tile can store — so a plain byte
patch on a 4-bit artifact ships a partly random edit. Raising the strength until it
survives costs tens of nats of perplexity. What actually works is minimum surgery:

  * every tensor the edit does NOT touch is copied BYTE FOR BYTE, so the user's
    quantization, their imatrix calibration and every offset-adjacent choice survive,
  * the tensors the edit DOES touch are re-encoded at a type that can hold it
    (Q6_K/Q8_0 by default), inside the same file, so nobody has to re-quantize a whole
    model or own a 16-bit copy,
  * the delivered change is measured (``ggufio.compare_edit``) against the pristine
    source, so "what ships" is a number rather than an assumption.

Byte-identity of untouched tensors is verified by re-reading both files, not asserted.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

SCALAR_ADDERS = {
    'BOOL': 'add_bool', 'STRING': 'add_string',
    'UINT8': 'add_uint8', 'INT8': 'add_int8',
    'UINT16': 'add_uint16', 'INT16': 'add_int16',
    'UINT32': 'add_uint32', 'INT32': 'add_int32',
    'UINT64': 'add_uint64', 'INT64': 'add_int64',
    'FLOAT32': 'add_float32', 'FLOAT64': 'add_float64',
}


REPAIR_MODES = ('none', 'renorm', 'norms', 'solve')


def _is_residual_norm(name):
    """blk.N.attn_norm / ffn_norm / post_* weights; the per-head q/k norms are not residual-stream."""
    leaf = name.split('.', 2)[2]
    return leaf.endswith('_norm.weight') and not leaf.startswith(('attn_q_norm', 'attn_k_norm'))


def _copy_metadata(reader, writer, skip=('general.architecture',)):
    """Copy every kv pair verbatim through the typed writer API."""
    from gguf import GGUFValueType

    copied = []
    for key, field in reader.fields.items():
        if key in skip or key.startswith('GGUF.'):
            continue
        types = list(field.types)
        value = field.contents()
        if types[0] == GGUFValueType.ARRAY.value:
            # This gguf-py's add_array takes (key, values) and infers the element
            # type; numpy scalars must become python ones or the inference fails.
            elements = [item.item() if hasattr(item, 'item') else item for item in value]
            writer.add_array(key, elements)
        else:
            name = GGUFValueType(types[0]).name
            getattr(writer, SCALAR_ADDERS[name])(key, value)
        copied.append(key)
    return copied


def _raw_slice(path, offset, n_bytes):
    with open(path, 'rb') as handle:
        handle.seek(offset)
        return np.frombuffer(handle.read(n_bytes), dtype=np.uint8)


def _add_packed(writer, name, raw, ggml_shape, qtype):
    """Add an already-quantized tensor.

    gguf-py's writer takes the PACKED shape when ``raw_dtype`` is given and converts
    it back with ``quant_shape_from_byte_shape``, which requires the last dimension to
    be a whole number of blocks. A flat byte array and the logical shape therefore
    both fail the check; the packed (rows, bytes_per_row) shape is what it wants.
    """
    rows = int(ggml_shape[0])
    if rows <= 0 or raw.size % rows:
        raise ValueError(f'{name}: {raw.size} packed bytes do not split into {rows} rows')
    bytes_per_row = raw.size // rows
    writer.add_tensor(name, np.asarray(raw, dtype=np.uint8).reshape(rows, bytes_per_row),
                      raw_shape=(rows, bytes_per_row), raw_dtype=qtype)


def axis_fingerprint(edits, names=None):
    """Digest of the axis -> tensor map: what anyone can recompute from the two files.

    ``edits`` is the same map ``rewrite_with_edits`` takes. Only entries that carry
    a (axis, space) tuple contribute, and only those in ``names`` (the tensors the
    writer actually re-encoded) are hashed, so the digest describes the edit as it
    landed rather than the request.
    """
    import hashlib

    digest = hashlib.sha256()
    layer_norms = {}
    for name in sorted(names if names is not None else edits):
        entry = edits[name]
        if not isinstance(entry, tuple):
            continue
        axis = np.asarray(entry[0], dtype=np.float32)
        axis = axis.reshape(1, -1) if axis.ndim == 1 else axis
        digest.update(name.encode())
        digest.update(np.round(axis, 6).tobytes())
        layer = int(name.split('.')[1])
        layer_norms[layer] = round(float(np.linalg.norm(axis, axis=1).sum()), 4)
    return {'edit_sha256': digest.hexdigest()[:32],
            'layers': sorted(layer_norms), 'axis_norm_by_layer': layer_norms}


def _solve_against_codec(base, delta, quantize, dequantize, block_els, *,
                         min_cos=0.9, passes=3, max_delta_scale=1.0):
    """Re-solve one tensor's delta against the codec that will STORE it.

    ``d <- d + (target - dequant(encode(w + d)))``, with ``target = w + delta`` held
    fixed: every pass adds back exactly the part of the edit the grid dropped, so the
    pre-quantization values are pulled towards the intended edited tensor instead of
    towards a bigger edit. Pass 1 is the plain write, so the best pass kept here is
    never worse than ``repair='none'``; ``max_delta_scale`` is the hard ceiling on how
    much larger the requested delta may become while doing it (the grid may move an
    edit around, it may not inflate it to clear a floor).

    Where a codec block's realized change points away from the requested one the delta
    is floored to zero in that block: the value ships unedited rather than carrying an
    edit the grid turned inside out. Returns ``(packed_bytes, record)``.
    """
    from .ggufio import _cos

    w = np.ascontiguousarray(base, dtype=np.float32).reshape(-1)
    requested = np.ascontiguousarray(delta, dtype=np.float32).reshape(-1)
    if requested.size != w.size:
        raise ValueError(f'delta has {requested.size} values for a tensor of {w.size}')
    n = int(w.size)
    pad = n + (-n) % int(block_els)
    if pad != n:
        w = np.concatenate([w, np.zeros(pad - n, dtype=np.float32)])
        requested = np.concatenate([requested, np.zeros(pad - n, dtype=np.float32)])
    target = w + requested
    nb = pad // int(block_els)
    expected_bytes = quantize(np.ascontiguousarray(w[:int(block_els)])).size * nb
    limit = float(max_delta_scale) * float(np.linalg.norm(requested[:n]))
    d = requested.copy()
    work = np.empty_like(w)
    best_cos, best_packed, best_d, cos_plain, used = None, None, None, None, 0
    for step in range(max(1, int(passes))):
        np.add(w, d, out=work)
        packed = np.asarray(quantize(np.ascontiguousarray(work)), dtype=np.uint8).reshape(-1)
        if packed.size != expected_bytes:
            raise ValueError(f'codec produced {packed.size} bytes, expected {expected_bytes}')
        rq = np.asarray(dequantize(packed, pad), dtype=np.float32).reshape(-1)
        np.subtract(rq, w, out=work)
        cos = _cos(work[:n], requested[:n])
        used = step + 1
        if cos_plain is None:
            cos_plain = cos
        if best_cos is None or cos > best_cos:
            best_cos, best_packed, best_d = cos, packed, d.copy()
        if cos >= float(min_cos):
            break
        # d <- d + (target - realized): add back what the grid dropped.
        np.subtract(target, rq, out=work)
        del rq
        np.add(d, work, out=d)
        if float(np.linalg.norm(d[:n])) > limit:
            break
    realized = np.asarray(dequantize(best_packed, pad), dtype=np.float32).reshape(-1) - w
    rows = realized.reshape(nb, int(block_els)).astype(np.float64)
    want = best_d.reshape(nb, int(block_els)).astype(np.float64)
    diverged = np.einsum('ij,ij->i', rows, want) <= 0
    floored = int(diverged.sum())
    floored_cos, floored_applied = None, False
    if floored:
        floored_delta = best_d.copy()
        floored_delta.reshape(nb, int(block_els))[diverged] = 0.0
        floored_packed = np.asarray(quantize(np.ascontiguousarray(w + floored_delta)),
                                    dtype=np.uint8).reshape(-1)
        floored_realized = (np.asarray(dequantize(floored_packed, pad), dtype=np.float32).reshape(-1) - w)
        floored_cos = _cos(floored_realized[:n], requested[:n])
        # The floor is kept only when it does not cost fidelity: the point is to
        # refuse to ship a block the grid turned inside out, not to trade one defect
        # for another. Both readings are recorded so the choice is auditable.
        if floored_cos > best_cos:
            best_cos, best_packed, best_d = floored_cos, floored_packed, floored_delta
            floored_applied = True
    record = {'passes': int(used), 'cos_plain': round(float(cos_plain), 6),
              'cos': round(float(best_cos), 6), 'floor': float(min_cos),
              'cleared': bool(best_cos >= float(min_cos)),
              'delta_scale': round(float(np.linalg.norm(best_d[:n])
                                         / (float(np.linalg.norm(requested[:n])) or 1.0)), 6),
              'blocks': int(nb), 'blocks_floored': floored,
              'cos_floored': None if floored_cos is None else round(float(floored_cos), 6),
              'floored_applied': floored_applied, 'elements': int(n)}
    return best_packed, record


def rewrite_with_edits(src, out, edits, *, scale=1.0, scales=None, axis_space='output',
                       edit_type='q6_k', renormalize=False, verify_untouched=True,
                       repair='none', norm_clamp=(0.9, 1.1), claims=None,
                       min_cos=0.9, solve_passes=3):
    """Write `out`: the user's artifact, with the edited tensors re-encoded.

    ``edits`` maps a tensor name to ``(axis, space)`` (a fitted direction) or to an
    explicit float32 delta. Tensors in that map are re-encoded at ``edit_type``;
    everything else is copied byte for byte, metadata included.

    ``scales`` overrides ``scale`` per tensor name. A bank entry (§5.1) carries a
    strength per projection kind — attention output and FFN down are not the same
    edit — and flattening those to one number would silently over-apply one kind
    and weaken the other, so the per-tensor values are applied where they belong
    and recorded in the fingerprint.

    ``repair`` runs after the edit and before encoding: 'none' (default) changes
    nothing, 'renorm' rescales every edited row back to its pre-edit L2 norm, 'norms'
    does that and also rescales the F32 residual-stream norm weights of the edited
    layers by the median row-norm ratio of those layers, clamped to ``norm_clamp``.
    'solve' instead re-solves each edited tensor's delta against the codec it is about
    to be stored in (``_solve_against_codec``): up to ``solve_passes`` passes that add
    back what the grid dropped, stopping as soon as the realized cos clears
    ``min_cos``, with divergent blocks floored to the unedited value. The pass count
    and the realized cos per tensor land in the report's ``solve`` block and in its
    ``next`` lines, and — because the number that matters is the one the FILE carries —
    in a ``delivered`` block measured by ``compare_edit`` on the written file.
    """


    from .ggufwrite import parse_container, tensor_nbytes, write_container
    from .gguf_types import canonical_type
    from .ggufio import axis_delta, decode_values
    from .quants import CODECS as QUANT_CODECS

    src, out = Path(src), Path(out)
    per_tensor_scale = {} if scales is None else {str(k): float(v) for k, v in scales.items()}
    if not src.is_file():
        raise ValueError(f'{src} does not exist')
    if src.resolve() == out.resolve():
        raise ValueError('refusing to rewrite onto the input artifact')
    if repair not in REPAIR_MODES:
        raise ValueError(f'repair must be one of {", ".join(REPAIR_MODES)}, not {repair!r}')
    renormalize = renormalize or repair in ('renorm', 'norms')
    preserve = str(edit_type).strip().lower() == 'preserve'
    target_name = None if preserve else canonical_type(edit_type)
    from .quants import CODECS as _WRITABLE

    if not preserve and target_name not in _WRITABLE:
        raise ValueError(f'cannot WRITE {target_name}: this build encodes {", ".join(sorted(_WRITABLE))}. '
                         f'The source artifact can be any type; only the re-encoded tensors '
                         f'have to be.')
    from .quants import type_id_for_name, type_name_for_id
    header = parse_container(src)
    header['source'] = str(src)
    if not preserve:
        # A quantized tensor's ROW has to be a whole number of codec blocks: ggml
        # asserts n_per_row % blck_size == 0 when it builds the tensor, and gguf-py's
        # reader refuses the file outright. This writer works on the flat element
        # stream, so without this guard it happily pads the last block and emits a
        # file llama.cpp cannot load (found on a 0.5B: 896-wide rows, Q6_K, block 256).
        block_els = QUANT_CODECS[target_name][1]
        offenders = [(tensor['name'], int(tuple(reversed(tensor['dims']))[-1]))
                     for tensor in header['tensors']
                     if tensor['name'] in edits
                     and int(tuple(reversed(tensor['dims']))[-1]) % int(block_els)]
        if offenders:
            detail = ', '.join(f'{name} (row {width})' for name, width in offenders[:6])
            raise ValueError(
                f'cannot re-encode {detail} at {target_name}: {target_name} stores '
                f'{block_els} elements per block and a row has to be a whole number of '
                f'blocks, which these are not — llama.cpp and gguf-py both refuse such a '
                f'tensor, so the file would not load. Pick a type with a smaller block '
                f'(q8_0, q5_0, q4_0, f16) or leave these tensors at their source type '
                f'(--edit-type preserve).')
    started = time.monotonic()
    payloads, qtypes, edited_rows = {}, {}, []
    intended_deltas = {}
    ratios_by_layer = {}
    with open(src, 'rb') as handle:
        for tensor in header['tensors']:
            name = tensor['name']
            if name not in edits:
                continue
            dtype = type_name_for_id(tensor['qtype'])
            write_name = dtype if preserve else target_name
            if preserve and write_name not in _WRITABLE:
                raise ValueError(f'cannot PRESERVE the codec of {name}: {write_name} has no '
                                 f'encoder in this build (encodes {", ".join(sorted(_WRITABLE))}); '
                                 f'pass an explicit --edit-type instead of shipping a type nobody '
                                 f'measured')
            nbytes = tensor_nbytes(tensor['dims'], tensor['qtype'])
            handle.seek(header['data_base'] + tensor['offset'])
            raw = np.frombuffer(handle.read(nbytes), dtype=np.uint8)
            # ANY source quant is readable (decode_values falls back to gguf-py);
            # only the WRITTEN type has to be one of our encoders.
            values = decode_values(raw, dtype)
            logical = tuple(reversed(tensor['dims']))
            matrix = values.reshape(logical)
            edit = edits[name]
            if isinstance(edit, tuple):
                axis, space = edit
                delta = axis_delta(axis, matrix, per_tensor_scale.get(name, scale), space)
            else:
                delta = np.asarray(edit, dtype=np.float32).reshape(logical)
            edited = matrix + delta
            row_scale = None
            if renormalize:
                before = np.linalg.norm(matrix, axis=1, keepdims=True)
                after = np.linalg.norm(edited, axis=1, keepdims=True)
                row_scale = before / np.maximum(after, 1e-12)
                edited = edited * row_scale
                ratios_by_layer.setdefault(int(name.split('.')[1]), []).append(row_scale.reshape(-1))
            _tb, new_block_els, quantize_fn, _dq = QUANT_CODECS[write_name]
            if repair == 'solve':
                # `edited - matrix` is what the caller asked the file to store (after
                # scale/renorm); the solve re-solves exactly that against this codec.
                intended = np.ascontiguousarray((edited - matrix).reshape(-1), dtype=np.float32)
                intended_deltas[name] = intended
                packed, solved = _solve_against_codec(
                    matrix.reshape(-1), intended, quantize_fn, _dq, new_block_els,
                    min_cos=min_cos, passes=solve_passes)

            else:
                flat = np.ascontiguousarray(edited.reshape(-1), dtype=np.float32)
                padded = flat.size + (-flat.size) % new_block_els
                if padded != flat.size:
                    flat = np.concatenate([flat, np.zeros(padded - flat.size, dtype=np.float32)])
                packed = np.asarray(quantize_fn(flat), dtype=np.uint8).reshape(-1)
            payloads[name] = packed
            qtypes[name] = type_id_for_name(write_name)
            edited_rows.append({'name': name, 'from_dtype': dtype, 'to_dtype': write_name,
                                'bytes_before': int(nbytes), 'bytes_after': int(packed.size)})
            if repair == 'solve':
                edited_rows[-1]['solve'] = solved
            if repair in ('renorm', 'norms'):
                edited_rows[-1]['row_scale_min'] = round(float(row_scale.min()), 6)
                edited_rows[-1]['row_scale_max'] = round(float(row_scale.max()), 6)
    matrices = set(payloads)
    repair_block = {'mode': repair, 'tensors': sorted(matrices) if repair != 'none' else [],
                    'scale_min': None, 'scale_max': None, 'norms': [], 'clamp': None,
                    'clamp_hits': 0}
    if repair in ('renorm', 'norms'):
        every = np.concatenate([row for rows in ratios_by_layer.values() for row in rows])
        repair_block['scale_min'] = round(float(every.min()), 6)
        repair_block['scale_max'] = round(float(every.max()), 6)
    if repair == 'norms':
        low, high = float(norm_clamp[0]), float(norm_clamp[1])
        repair_block['clamp'] = [low, high]
        with open(src, 'rb') as handle:
            for tensor in header['tensors']:
                name = tensor['name']
                parts = name.split('.')
                if (len(parts) < 3 or parts[0] != 'blk' or not parts[1].isdigit()
                        or int(parts[1]) not in ratios_by_layer
                        or not _is_residual_norm(name) or int(tensor['qtype']) != 0):
                    continue
                handle.seek(header['data_base'] + tensor['offset'])
                old = np.frombuffer(handle.read(tensor_nbytes(tensor['dims'], 0)), dtype=np.float32)
                measured = float(np.median(np.concatenate(ratios_by_layer[int(parts[1])])))
                factor = min(max(measured, low), high)
                hit = factor != measured
                new = (old * np.float32(factor)).astype(np.float32)
                payloads[name] = np.frombuffer(new.tobytes(), dtype=np.uint8)
                qtypes[name] = 0
                repair_block['clamp_hits'] += int(hit)
                repair_block['norms'].append({
                    'name': name, 'median_row_ratio': round(measured, 6),
                    'factor': round(factor, 6), 'clamped': hit,
                    'median_old': round(float(np.median(old)), 6),
                    'median_new': round(float(np.median(new)), 6)})
    # Edit fingerprint: the machine-checkable claim a release needs (changed tensors,
    # untouched count, the axis digest and its per-layer norm). Anyone can recompute
    # it from the two files, so "this is a fresh edit" stops being a matter of trust —
    # the same idea as publishing "81 matrices changed, 1,111 tensors untouched".
    fingerprint = {**axis_fingerprint(edits, matrices),
                   'edited': len(matrices), 'untouched': len(header['tensors']) - len(payloads),
                   'scale': scale, 'axis_space': axis_space,
                   'edit_type': 'preserve' if preserve else target_name,
                   'edit_type_per_tensor': {row['name']: row['to_dtype'] for row in edited_rows},
                   'repair': repair_block}
    if per_tensor_scale:
        fingerprint['scale_per_tensor'] = {name: per_tensor_scale.get(name, scale)
                                           for name in sorted(matrices)}
    # The fingerprint travels WITH the artifact: a claim nobody can check is marketing,
    # and `verify --claim` can only compare fields it can find in the file.
    import json as _json

    if claims:
        # Behavioural claims ride along WITH their provenance, never as a verdict: a
        # screen on an adapter is not a certification of these bytes (R2/R8), so the
        # artifact records certified=false, what was screened and what it left open.
        # A release that reads as "clears" because of a field we wrote is the exact
        # failure mode this project exists to prevent.
        fingerprint.update({k: v for k, v in claims.items() if v is not None})
    written = write_container(out, header, payloads, qtypes,
                              extra_kv=[('jdewitt.fingerprint',
                                         _json.dumps(fingerprint, sort_keys=True))])
    by_name = {entry['name']: entry for entry in written['entries']}
    report = {
        'source': str(src), 'output': str(out), 'edit_type': target_name,
        'scale': scale, 'axis_space': axis_space, 'renormalize': bool(renormalize),
        'tensors': len(header['tensors']), 'tensors_edited': len(matrices), 'repair': repair,
        'edited': edited_rows,
        'bytes_in': src.stat().st_size, 'bytes_out': out.stat().st_size,
        'fingerprint': fingerprint,
        'elapsed_s': round(time.monotonic() - started, 1),
    }
    if repair == 'solve':
        # The pass count and the in-loop cos are the solver's own reading; the
        # DELIVERED number is the one compare_edit recomputes from the two files:
        # preview_edit's 8,192-element sample has already cleared a floor the written
        # file did not, and a solver grading itself on the values it handed the encoder
        # is the same trap.
        from .ggufio import compare_edit

        solved = [dict(name=row['name'], **row['solve']) for row in edited_rows]
        cosines = [row['cos'] for row in solved]
        delivered = compare_edit(str(src), str(out), intended_deltas,
                                 names=sorted(intended_deltas))
        report['solve'] = {
            'floor': float(min_cos), 'passes_max': int(solve_passes),
            'tensors': solved,
            'min_cos': round(min(cosines), 6) if cosines else None,
            'mean_cos': round(sum(cosines) / len(cosines), 6) if cosines else None,
            'cleared': int(sum(1 for row in solved if row['cleared'])),
            'passed_after_encoding': 'the pass cos is measured on the full tensor with the '
                                      'codec of the written file; delivered is compare_edit on the file',
        }
        repair_block['floor'] = float(min_cos)
        repair_block['passes_max'] = int(solve_passes)
        repair_block['passes_per_tensor'] = {row['name']: row['passes'] for row in solved}
        repair_block['cos_per_tensor'] = {row['name']: row['cos'] for row in solved}
        repair_block['cleared'] = report['solve']['cleared']
        report['delivered'] = delivered
        below_floor = [row['name'] for row in delivered['tensors']
                       if (row.get('cos') or 0.0) < float(min_cos)]
        report['next'] = [
            f'jdewitt gguf-tensor-diff {src} {out}  # every untouched tensor must be bit-identical',
            f'jdewitt verify --pristine {src} --edited {out}  # recompute the edit fingerprint',
            'solve passes/cos on the written file: ' + ', '.join(
                f"{row['name']} {row['passes']}p {row['cos']:.4f}" for row in solved),
        ]
        if below_floor:
            report['next'].append(
                f'the target codec cannot store this edit at cos >= {min_cos} for '
                f'{", ".join(below_floor[:6])}: re-encode those tensors at a finer type '
                f'(--edit-type q8_0 --repair solve) or leave them unedited; do not raise the strength')

    if verify_untouched:
        mismatched, missing = [], []
        with open(src, 'rb') as old, open(out, 'rb') as new:
            for tensor in header['tensors']:
                name = tensor['name']
                if name in payloads:
                    continue
                entry = by_name.get(name)
                if entry is None:
                    missing.append(name)
                    continue
                nbytes = tensor_nbytes(tensor['dims'], tensor['qtype'])
                old.seek(header['data_base'] + tensor['offset'])
                new.seek(written['data_base'] + entry['offset'])
                if entry['qtype'] != int(tensor['qtype']) or old.read(nbytes) != new.read(nbytes):
                    mismatched.append(name)
        report['untouched'] = {'verified': not mismatched and not missing,
                               'checked': len(header['tensors']) - len(payloads),
                               'mismatched': mismatched[:12], 'missing': missing[:12]}
    return report
