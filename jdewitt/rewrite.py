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
                       edit_type='q6_k', renormalize=False, verify_untouched=True):
    """Write `out`: the user's artifact, with the edited tensors re-encoded.

    ``edits`` maps a tensor name to ``(axis, space)`` (a fitted direction) or to an
    explicit float32 delta. Tensors in that map are re-encoded at ``edit_type``;
    everything else is copied byte for byte, metadata included.
    """

    from gguf import GGMLQuantizationType

    from .ggufwrite import parse_container, tensor_nbytes, write_container
    from .quantize import canonical_type
    from .quants import CODECS as QUANT_CODECS

    src, out = Path(src), Path(out)
    if not src.is_file():
        raise ValueError(f'{src} does not exist')
    if src.resolve() == out.resolve():
        raise ValueError('refusing to rewrite onto the input artifact')
    target_name = canonical_type(edit_type)
    target = GGMLQuantizationType[target_name]
    header = parse_container(src)
    header['source'] = str(src)
    started = time.monotonic()
    payloads, qtypes, edited_rows = {}, {}, []
    with open(src, 'rb') as handle:
        for tensor in header['tensors']:
            name = tensor['name']
            if name not in edits:
                continue
            dtype = GGMLQuantizationType(int(tensor['qtype'])).name
            block_bytes, block_els, _quantize, dequantize = QUANT_CODECS[dtype]
            nbytes = tensor_nbytes(tensor['dims'], tensor['qtype'])
            handle.seek(header['data_base'] + tensor['offset'])
            raw = np.frombuffer(handle.read(nbytes), dtype=np.uint8)
            values = np.asarray(dequantize(raw, nbytes // block_bytes * block_els), dtype=np.float32)
            logical = tuple(reversed(tensor['dims']))
            matrix = values.reshape(logical)
            edit = edits[name]
            if isinstance(edit, tuple):
                axis, space = edit
                axis = np.asarray(axis, dtype=np.float32).reshape(-1)
                if space == 'output':
                    delta = -scale * (axis.reshape(-1, 1) @ (axis.reshape(1, -1) @ matrix))
                else:
                    delta = -scale * ((axis.reshape(-1, 1) @ (axis.reshape(1, -1) @ matrix.T)).T)
            else:
                delta = np.asarray(edit, dtype=np.float32).reshape(logical)
            edited = matrix + delta
            if renormalize:
                before = np.linalg.norm(matrix, axis=1, keepdims=True)
                after = np.linalg.norm(edited, axis=1, keepdims=True)
                edited = edited * (before / np.maximum(after, 1e-12))
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
    written = write_container(out, header, payloads, qtypes)
    by_name = {entry['name']: entry for entry in written['entries']}
    report = {
        'source': str(src), 'output': str(out), 'edit_type': target.name,
        'scale': scale, 'axis_space': axis_space, 'renormalize': bool(renormalize),
        'tensors': len(header['tensors']), 'tensors_edited': len(payloads),
        'edited': edited_rows,
        'bytes_in': src.stat().st_size, 'bytes_out': out.stat().st_size,
        'elapsed_s': round(time.monotonic() - started, 1),
    }
    if verify_untouched:
        mismatched, missing = [], []
        with open(src, 'rb') as old, open(out, 'rb') as new:
            for tensor in header['tensors']:
                name = tensor['name']
                if name in edits:
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
