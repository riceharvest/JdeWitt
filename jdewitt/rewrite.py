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


REPAIR_MODES = ('none', 'renorm', 'norms')


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


def rewrite_with_edits(src, out, edits, *, scale=1.0, axis_space='output',
                       edit_type='q6_k', renormalize=False, verify_untouched=True,
                       repair='none', norm_clamp=(0.9, 1.1)):
    """Write `out`: the user's artifact, with the edited tensors re-encoded.

    ``edits`` maps a tensor name to ``(axis, space)`` (a fitted direction) or to an
    explicit float32 delta. Tensors in that map are re-encoded at ``edit_type``;
    everything else is copied byte for byte, metadata included.

    ``repair`` runs after the edit and before encoding: 'none' (default) changes
    nothing, 'renorm' rescales every edited row back to its pre-edit L2 norm, 'norms'
    does that and also rescales the F32 residual-stream norm weights of the edited
    layers by the median row-norm ratio of those layers, clamped to ``norm_clamp``.
    """

    from gguf import GGMLQuantizationType

    from .ggufwrite import parse_container, tensor_nbytes, write_container
    from .quantize import canonical_type
    from .ggufio import axis_delta, decode_values
    from .quants import CODECS as QUANT_CODECS

    src, out = Path(src), Path(out)
    if not src.is_file():
        raise ValueError(f'{src} does not exist')
    if src.resolve() == out.resolve():
        raise ValueError('refusing to rewrite onto the input artifact')
    if repair not in REPAIR_MODES:
        raise ValueError(f'repair must be one of {", ".join(REPAIR_MODES)}, not {repair!r}')
    renormalize = renormalize or repair != 'none'
    target_name = canonical_type(edit_type)
    from .quants import CODECS as _WRITABLE

    if target_name not in _WRITABLE:
        raise ValueError(f'cannot WRITE {target_name}: this build encodes {", ".join(sorted(_WRITABLE))}. '
                         f'The source artifact can be any type; only the re-encoded tensors '
                         f'have to be.')
    target = GGMLQuantizationType[target_name]
    header = parse_container(src)
    header['source'] = str(src)
    started = time.monotonic()
    payloads, qtypes, edited_rows = {}, {}, []
    ratios_by_layer = {}
    with open(src, 'rb') as handle:
        for tensor in header['tensors']:
            name = tensor['name']
            if name not in edits:
                continue
            dtype = GGMLQuantizationType(int(tensor['qtype'])).name
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
                delta = axis_delta(axis, matrix, scale, space)
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
            _tb, new_block_els, quantize_fn, _dq = QUANT_CODECS[target.name]
            flat = np.ascontiguousarray(edited.reshape(-1), dtype=np.float32)
            padded = flat.size + (-flat.size) % new_block_els
            if padded != flat.size:
                flat = np.concatenate([flat, np.zeros(padded - flat.size, dtype=np.float32)])
            packed = np.asarray(quantize_fn(flat), dtype=np.uint8).reshape(-1)
            payloads[name] = packed
            qtypes[name] = int(target)
            edited_rows.append({'name': name, 'from_dtype': dtype, 'to_dtype': target.name,
                                'bytes_before': int(nbytes), 'bytes_after': int(packed.size)})
            if repair != 'none':
                edited_rows[-1]['row_scale_min'] = round(float(row_scale.min()), 6)
                edited_rows[-1]['row_scale_max'] = round(float(row_scale.max()), 6)
    matrices = set(payloads)
    repair_block = {'mode': repair, 'tensors': sorted(matrices) if repair != 'none' else [],
                    'scale_min': None, 'scale_max': None, 'norms': [], 'clamp': None,
                    'clamp_hits': 0}
    if repair != 'none':
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
    import hashlib

    digest = hashlib.sha256()
    layer_norms = {}
    for name in sorted(matrices):
        entry = edits[name]
        if isinstance(entry, tuple):
            axis = np.asarray(entry[0], dtype=np.float32)
            axis = axis.reshape(1, -1) if axis.ndim == 1 else axis
            digest.update(name.encode())
            digest.update(np.round(axis, 6).tobytes())
            layer = int(name.split('.')[1])
            layer_norms[layer] = round(float(np.linalg.norm(axis, axis=1).sum()), 4)
    fingerprint = {'edit_sha256': digest.hexdigest()[:32],
                   'edited': len(matrices), 'untouched': len(header['tensors']) - len(payloads),
                   'layers': sorted(layer_norms), 'axis_norm_by_layer': layer_norms,
                   'scale': scale, 'axis_space': axis_space, 'edit_type': target.name,
                   'repair': repair_block}
    # The fingerprint travels WITH the artifact: a claim nobody can check is marketing,
    # and `verify --claim` can only compare fields it can find in the file.
    import json as _json

    written = write_container(out, header, payloads, qtypes,
                              extra_kv=[('jdewitt.fingerprint',
                                         _json.dumps(fingerprint, sort_keys=True))])
    by_name = {entry['name']: entry for entry in written['entries']}
    report = {
        'source': str(src), 'output': str(out), 'edit_type': target.name,
        'scale': scale, 'axis_space': axis_space, 'renormalize': bool(renormalize),
        'tensors': len(header['tensors']), 'tensors_edited': len(matrices), 'repair': repair,
        'edited': edited_rows,
        'bytes_in': src.stat().st_size, 'bytes_out': out.stat().st_size,
        'fingerprint': fingerprint,
        'elapsed_s': round(time.monotonic() - started, 1),
    }
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
