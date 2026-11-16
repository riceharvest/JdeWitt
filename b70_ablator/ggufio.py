# SPDX-License-Identifier: AGPL-3.0-only
"""GGUF-native quantized tensor editing: dequantize -> edit -> requantize.

Edits quantized tensors without converting the whole file: tensors that are
not named keep their bytes verbatim, the header/metadata/info array is copied
verbatim, and tensor data offsets stay valid because each edited tensor is
requantized to the same block layout and size. Output files load in
llama.cpp with no flags. No model, no device — numpy only.
"""
import shutil
import mmap
import math
import struct
import numpy as np
from pathlib import Path
from .quants import CODECS


def _tensor_info(reader, name):
    tensor = next((t for t in reader.tensors if t.name == name), None)
    if tensor is None:
        raise ValueError(f'no tensor {name!r} in file')
    dtype = tensor.tensor_type.name
    if dtype not in CODECS:
        raise ValueError(f'tensor {name!r} has dtype {dtype}; this tool edits {", ".join(sorted(CODECS))}')
    block_bytes, block_els, quantize, dequantize = CODECS[dtype]
    offset = int(tensor.data_offset)
    n_bytes = int(tensor.n_bytes)
    if n_bytes % block_bytes:
        raise ValueError(f'tensor {name!r} byte size {n_bytes} is not a whole number of {dtype} blocks')
    n = n_bytes // block_bytes * block_els
    shape = tuple(int(x) for x in np.asarray(tensor.shape).ravel())
    return tensor, dtype, block_bytes, block_els, n, quantize, dequantize, offset, n_bytes, shape


def edit_tensors(path, out_path, deltas):
    """Apply {tensor_name: float32 delta matching that tensor's shape} and write out_path.

    Everything not in ``deltas`` is copied verbatim; each edited tensor is
    requantized to the identical block count, so no other offset moves.
    Returns per-tensor reports.
    """
    from gguf import GGUFReader
    path = Path(path)
    out_path = Path(out_path)
    if not path.is_file():
        raise ValueError(f'{path} does not exist')
    if path.resolve() == out_path.resolve():
        raise ValueError('out_path must differ from in_path (edits are written to a new file)')
    reader = GGUFReader(str(path))
    plans = []
    for name, delta in deltas.items():
        tensor, dtype, bs, els, n, quantize, dequantize, offset, n_bytes, shape = _tensor_info(reader, name)
        delta = np.asarray(delta, dtype=np.float32)
        if delta.shape != shape:
            raise ValueError(f'delta for {name} has shape {delta.shape}, tensor shape is {shape}')
        if not np.isfinite(delta).all():
            raise ValueError(f'delta for {name} contains NaN/Inf')
        plans.append((name, tensor, dtype, bs, els, n, quantize, dequantize, offset, n_bytes, shape, delta))
    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.MAP_SHARED)
        try:
            new_blocks = []
            for name, tensor, dtype, bs, els, n, quantize, dequantize, offset, n_bytes, shape, delta in plans:
                nb = n // els
                # Dequant-tile-apply-requant: the FP32 working set stays bounded
                # (~64 MiB of values) no matter how large the tensor is.
                blocks_per_tile = max(1, (1 << 24) // els)
                new_raw = np.empty(n_bytes, dtype=np.uint8)
                changed_blocks = np.zeros(nb, dtype=bool)
                delta_sq = sum_sq = 0.0
                flat_delta = delta.reshape(-1)
                count = 0
                for start in range(0, nb, blocks_per_tile):
                    tile = min(blocks_per_tile, nb - start)
                    byte0, byte1 = start * bs, (start + tile) * bs
                    raw_tile = np.frombuffer(mm, dtype=np.uint8, count=byte1 - byte0,
                                            offset=offset + byte0).copy()
                    values = dequantize(raw_tile, tile * els).reshape(-1)
                    piece_delta = flat_delta[start * els:(start + tile) * els]
                    edited = values + piece_delta
                    if not np.isfinite(edited).all():
                        raise ValueError(f'edited tensor {name} contains NaN/Inf')
                    piece = quantize(np.ascontiguousarray(edited))
                    if piece.size != byte1 - byte0:
                        raise ValueError(f'requantized {name}: block produced {piece.size} '
                                         f'bytes, expected {byte1 - byte0}')
                    new_raw[byte0:byte1] = piece
                    changed_blocks[start:start + tile] = np.any(piece.reshape(tile, bs) != raw_tile.reshape(tile, bs), axis=-1)
                    delta_sq += float((piece_delta ** 2).sum())
                    sum_sq += float((values ** 2).sum())
                    count += tile * els
                rms = float(math.sqrt(delta_sq / count)) if count else 0.0
                mean_sq = float(sum_sq / count) if count else 0.0
                new_blocks.append((name, offset, new_raw, int(nb), int(changed_blocks.sum()), rms,
                                   rms / mean_sq if mean_sq else 0.0, dtype))
        finally:
            mm.close()
    if out_path.exists():
        raise ValueError(f'{out_path} already exists')
    shutil.copyfile(path, out_path)
    with open(out_path, 'r+b') as f:
        for name, offset, new_raw, *_ in new_blocks:
            f.seek(offset)
            f.write(new_raw)
    reports = []
    for name, offset, new_raw, nb, changed, rms, rel, dtype in new_blocks:
        reports.append({'tensor': name, 'dtype': dtype, 'blocks': nb, 'changed_blocks': changed,
                        'rms_delta': rms, 'relative_rms_delta': rel, 'bytes_written': int(new_raw.size)})
    return {'input': str(path), 'output': str(out_path), 'tensors_edited': len(reports),
           'tensors': reports, 'all_other_bytes_verbatim': True}


# --- GGUF v3 metadata editing -------------------------------------------------
# Header layout: magic GGUF, u32 version, u64 n_tensors, u64 n_kv, then n_kv
# (key, type, value) entries, then n_tensors tensor-info records, padding to
# general.alignment (default 32), then one contiguous tensor-data blob.
# Tensor offsets inside the blob are relative to the data base; re-aligning
# after a header size change keeps every tensor's alignment, so the blob can
# be copied verbatim and every info offset shifted by the same amount.
_SCALARS = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?'}
_TYPE_STRING, _TYPE_ARRAY = 8, 9


def _u64(buf):
    return int.from_bytes(buf.read(8), 'little')


def _read_key(buf):
    return buf.read(_u64(buf))


def _skip_value(buf, tag):
    """Read the full raw bytes of one value of the given type tag."""
    if tag in _SCALARS:
        return buf.read(struct.calcsize(_SCALARS[tag]))
    if tag == _TYPE_STRING:
        return _pack_u64(buf)
    if tag == _TYPE_ARRAY:
        elem = struct.unpack('<I', buf.read(4))[0]
        count = _u64(buf)
        head = struct.pack('<I', elem) + struct.pack('<Q', count)
        if elem in _SCALARS:
            return head + buf.read(count * struct.calcsize(_SCALARS[elem]))
        parts = []
        for _ in range(count):
            parts.append(_skip_value(buf, elem))
        return head + b''.join(parts)
    raise ValueError(f'unsupported GGUF type tag {tag} in metadata')


def _pack_u64(buf):
    length = int.from_bytes(buf.read(8), 'little')
    return struct.pack('<Q', length) + buf.read(length)


def parse_gguf_header(path):
    """Parse everything up to the tensor-data base; data is never read."""
    import io
    path = Path(path)
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            raise ValueError(f'{path} is not a GGUF file')
        version = struct.unpack('<I', f.read(4))[0]
        if version < 3:
            raise ValueError(f'GGUF v{version} metadata editing supports v3+')
        n_tensors = _u64(f)
        n_kv = _u64(f)
        buf = io.BufferedReader(f)
        entries = []
        for _ in range(n_kv):
            key = _read_key(buf)
            tag = struct.unpack('<I', buf.read(4))[0]
            raw = _skip_value(buf, tag)
            entries.append((key, tag, raw))
        infos = []
        for _ in range(n_tensors):
            start = buf.tell()
            name = _read_key(buf)
            body = buf.tell()
            n_dims = struct.unpack('<I', buf.read(4))[0]
            dims = buf.read(8 * n_dims)
            dtype = struct.unpack('<I', buf.read(4))[0]
            offset = struct.unpack('<Q', buf.read(8))[0]
            infos.append({'name': name, 'start': start, 'body': body,
                          'dims': dims, 'dtype': dtype, 'offset': offset, 'n_dims': n_dims})
        header_end = buf.tell()
    alignment = 32
    for key, tag, raw in entries:
        if key == b'general.alignment' and tag in _SCALARS:
            alignment = int(np.frombuffer(raw, dtype=_SCALARS[tag]).ravel()[0]) or 32
    data_base = (header_end + alignment - 1) // alignment * alignment
    return {'version': version, 'n_tensors': n_tensors, 'entries': entries, 'infos': infos,
            'header_end': header_end, 'data_base': data_base, 'alignment': alignment}


def _infer_type(value):
    if isinstance(value, bool):
        return 7, struct.pack('<?', value)
    if isinstance(value, str):
        return _TYPE_STRING, struct.pack('<Q', len(value.encode())) + value.encode()
    if isinstance(value, int):
        if 0 <= value < 2 ** 32:
            return 4, struct.pack('<I', value)
        if -(2 ** 31) <= value < 2 ** 63:
            return 5, struct.pack('<q', value)
        raise ValueError('integer too large for GGUF metadata (u32 or i64)')
    if isinstance(value, float):
        return 6, struct.pack('<f', value)
    if isinstance(value, list) and value:
        kinds = {type(v) for v in value}
        if all(isinstance(v, bool) for v in value):
            elem, packer = 7, '?'
        elif kinds == {str}:
            head = b''.join(struct.pack('<Q', len(v.encode())) + v.encode() for v in value)
            return _TYPE_ARRAY, struct.pack('<I', _TYPE_STRING) + struct.pack('<Q', len(value)) + head
        elif kinds == {int}:
            elem, packer = 5, 'q'
        elif kinds == {float} or kinds <= {int, float}:
            elem, packer = 6, 'f'
        else:
            raise ValueError(f'unsupported array element kinds: {kinds}')
        body = b''.join(struct.pack(packer, v) for v in value)
        return _TYPE_ARRAY, struct.pack('<I', elem) + struct.pack('<Q', len(value)) + body
    if isinstance(value, list):
        raise ValueError('empty arrays need an explicit element type; not supported')
    raise ValueError(f'unsupported metadata value type {type(value)}')


def _match_type(existing_tag, existing_raw, value):
    """Serialize value with the existing key's type so unchanged keys keep their layout."""
    if existing_tag in _SCALARS:
        fmt = _SCALARS[existing_tag]
        try:
            packed = struct.pack(fmt, value)
        except (struct.error, TypeError) as e:
            raise ValueError(f'cannot store {value!r} in existing type tag {existing_tag}: {e}')
        return existing_tag, packed
    if existing_tag == _TYPE_STRING:
        if not isinstance(value, str):
            raise ValueError('existing key is a string; provide a JSON string')
        return existing_tag, struct.pack('<Q', len(value.encode())) + value.encode()
    if existing_tag == _TYPE_ARRAY:
        elem = struct.unpack('<I', existing_raw[:4])[0]
        count = int.from_bytes(existing_raw[4:12], 'little')
        if elem == _TYPE_STRING:
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ValueError('existing key is a string array; provide a JSON list of strings')
            head = b''.join(struct.pack('<Q', len(v.encode())) + v.encode() for v in value)
            return _TYPE_ARRAY, struct.pack('<I', _TYPE_STRING) + struct.pack('<Q', len(value)) + head
        if elem in _SCALARS and isinstance(value, list):
            fmt = _SCALARS[elem]
            try:
                body = b''.join(struct.pack(fmt, v) for v in value)
            except (struct.error, TypeError) as e:
                raise ValueError(f'cannot store array in existing element type tag {elem}: {e}')
            return _TYPE_ARRAY, struct.pack('<I', elem) + struct.pack('<Q', len(value)) + body
        raise ValueError(f'array element type tag {elem} ({count} elements) is not editable here')
    raise ValueError(f'existing type tag {existing_tag} is not editable')


def edit_metadata(path, out_path, sets, unsets):
    """Edit GGUF string/scalar/array metadata, writing out_path. Tensor data is never interpreted.

    ``sets`` maps key to a Python value (JSON-parsed by the CLI); unsets remove
    keys. Entries not named are copied byte-for-byte. When the new header no
    longer fits the old data-base alignment slot, the tensor data blob is
    copied verbatim at a new aligned base and every tensor offset is shifted
    by the identical amount.
    """
    path, out_path = Path(path), Path(out_path)
    if not path.is_file():
        raise ValueError(f'{path} does not exist')
    if path.resolve() == out_path.resolve():
        raise ValueError('out_path must differ from in_path')
    parsed = parse_gguf_header(path)
    entries = list(parsed['entries'])
    index = {key: position for position, (key, _tag, _raw) in enumerate(entries)}
    changed = []
    for key in unsets:
        kb = key.encode()
        if kb not in index:
            raise ValueError(f'key {key!r} is not present; refusing a no-op unset')
        entries.pop(index[kb])
        index = {k: p for p, (k, _t, _r) in enumerate(entries)}
        changed.append({'key': key, 'action': 'unset'})
    for key, value in sets.items():
        kb = key.encode()
        if kb in index:
            position = index[kb]
            _key, old_tag, old_raw = entries[position]
            tag, raw = _match_type(old_tag, old_raw, value)
            entries[position] = (kb, tag, raw)
            changed.append({'key': key, 'action': 'set', 'replaced_type_tag': old_tag})
        else:
            tag, raw = _infer_type(value)
            entries.append((kb, tag, raw))
            changed.append({'key': key, 'action': 'set', 'replaced_type_tag': None})
    header = b'GGUF' + struct.pack('<I', parsed['version']) + \
        struct.pack('<Q', parsed['n_tensors']) + struct.pack('<Q', len(entries))
    for key, tag, raw in entries:
        header += struct.pack('<Q', len(key)) + key + struct.pack('<I', tag) + raw
    alignment = parsed['alignment']
    info_bodies = [(info['name'], info['n_dims'], info['dims'], info['dtype'], info['offset'])
                   for info in parsed['infos']]
    header_before = parsed['data_base']
    # Info sizes depend only on name length and n_dims, already captured per record.
    infos_size = sum(8 + len(name) + 4 + 8 * n_dims + 4 + 8 for name, n_dims, *_ in info_bodies)
    new_data_base = (len(header) + infos_size + alignment - 1) // alignment * alignment
    shift = new_data_base - parsed['data_base']
    if out_path.exists():
        raise ValueError(f'{out_path} already exists')
    rebuilt = bytearray(header)
    for name, n_dims, dims, dtype, offset in info_bodies:
        # GGUF tensor offsets are relative to the data base; the blob is copied
        # intact, so relative offsets are verbatim regardless of where the base moves.
        rebuilt += struct.pack('<Q', len(name)) + name + struct.pack('<I', n_dims) + dims + \
            struct.pack('<I', dtype) + struct.pack('<Q', offset)
    rebuilt += b'\x00' * (new_data_base - len(rebuilt))
    if shift == 0:
        shutil.copyfile(path, out_path)
        with open(out_path, 'r+b') as f:
            f.seek(0)
            f.write(rebuilt)
    else:
        remaining = path.stat().st_size - parsed['data_base']
        with open(out_path, 'wb') as g:
            g.write(rebuilt)
            with open(path, 'rb') as f:
                f.seek(parsed['data_base'])
                left = remaining
                while left > 0:
                    chunk = f.read(min(left, 1 << 26))
                    if not chunk:
                        raise ValueError(f'{path}: tensor data ended early')
                    g.write(chunk)
                    left -= len(chunk)
    return {'input': str(path), 'output': str(out_path), 'changes': changed,
            'header_bytes_before': header_before, 'data_base_after': new_data_base,
            'data_base_shift': shift,
            'tensor_data_bytes_verbatim': True,
            'note': 'Unedited metadata entries copied byte-for-byte; the tensor data blob is copied intact and its relative offsets are never touched.'}


def projection_delta(weight, directions, strength, side='output', row_normalization='none'):
    """numpy mirror of AblatedLinear's merge algebra: delta W from orthonormal directions.

    ``side='output'``: dW = -(s * n) * d.T @ (d @ W) (remove what the module writes).
    ``side='input'``:  dW = -s * (W @ d.T) @ d      (remove what the module reads),
    with ``row_normalization='full'`` restoring per-row norms of the edited result.
    """
    d = np.asarray(directions, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    if d.ndim != 2 or not np.isfinite(d).all() or strength < 0:
        raise ValueError('directions must be a finite 2D float matrix and strength >= 0')
    if not np.allclose(d @ d.T, np.eye(d.shape[0], dtype=np.float32), atol=1e-4):
        raise ValueError('directions must be orthonormal')
    if row_normalization not in {'none', 'pre', 'full'}:
        raise ValueError('unknown row normalization')
    if side == 'output':
        if d.shape[1] != w.shape[0]:
            raise ValueError(f'output-side direction width {d.shape[1]} != out_features {w.shape[0]}')
        norms = np.linalg.norm(w, axis=1, keepdims=True)
        base = w / np.maximum(norms, 1e-12) if row_normalization != 'none' else w
        a = d @ base
        b = -strength * d.T
        if row_normalization != 'none':
            b = norms * b
        delta = b @ a
    elif side == 'input':
        if d.shape[1] != w.shape[1]:
            raise ValueError(f'input-side direction width {d.shape[1]} != in_features {w.shape[1]}')
        delta = -strength * (w @ d.T) @ d
    else:
        raise ValueError('unknown side')
    if row_normalization == 'full':
        changed = w + delta
        old = np.linalg.norm(w, axis=1, keepdims=True)
        new = np.linalg.norm(changed, axis=1, keepdims=True)
        ratio = np.where(new > 1e-12, old / np.maximum(new, 1e-12), 1.0)
        delta = delta * ratio - w + w * ratio
    return delta.astype(np.float32)
