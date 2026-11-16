"""FP8 (E4M3/E5M2) safetensors editing: decode, FP32 delta, requantize.

Reuses the pinned E4M3 tables from fp4.py and mirrors its byte-verbatim
edit contract: only the edited tensor's and its scale tensor's byte ranges
are rewritten; every other byte of the file is copied intact.

Real-world layouts covered: per-tensor scale (one F32) and DeepSeek-style
block scales (F32 tensors with one dim divided by the block, 1x128 row
blocks or 128x1 column blocks).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .fp4 import E4M3_MAX, decode_e4m3, encode_e4m3, _read_safetensors

E5M2_MAX = 57344.0
FP8_MAX = {'e4m3': E4M3_MAX, 'e5m2': E5M2_MAX}
FP8_STORAGE = {'e4m3': 'F8_E4M3', 'e5m2': 'F8_E5M2'}


def decode_e5m2(codes):
    return _E5M2_DECODE[np.asarray(codes, dtype=np.uint8)]


def encode_e5m2(values):
    """Round-to-nearest E5M2 encode with ties-to-even; saturates to +-57344."""
    values = np.asarray(values, dtype=np.float32)
    magnitude = np.clip(np.abs(values), None, E5M2_MAX)
    idx = np.searchsorted(_E5M2_MIDPOINTS, magnitude, side='left')
    lower_odd = (idx < _E5M2_POS_CODES.size) & \
        (((_E5M2_POS_CODES[np.minimum(idx, _E5M2_POS_CODES.size - 1)]) & 1) == 1)
    ties = (idx < _E5M2_MIDPOINTS.size) & \
        (magnitude == _E5M2_MIDPOINTS[np.minimum(idx, _E5M2_MIDPOINTS.size - 1)])
    idx = idx + (ties & lower_odd)
    codes = _E5M2_POS_CODES[idx].copy()
    codes[magnitude <= 0] = 0
    return np.where(np.signbit(values), codes | 0x80, codes).astype(np.uint8)


def _decode_table_e5m2():
    table = np.zeros(256, dtype=np.float32)
    for b in range(256):
        sign = -1.0 if b & 0x80 else 1.0
        e, m = (b >> 2) & 0x1F, b & 0x3
        if e == 0x1F:
            table[b] = np.nan  # inf and NaN both saturate away in weight storage
        elif e == 0:
            table[b] = sign * m * 2.0 ** -16
        else:
            table[b] = sign * (1.0 + m / 4.0) * 2.0 ** (e - 15)
    return table


_E5M2_DECODE = _decode_table_e5m2()
# Same RNE machinery as E4M3: sorted positive finite magnitudes (inf/NaN
# codes excluded), exact midpoints tie to the even-mantissa code (code bit 0
# is the mantissa LSB for normals and subnormals).
_pos5 = [(b, _E5M2_DECODE[b]) for b in range(1, 128) if not np.isnan(_E5M2_DECODE[b])]
_pos5.sort(key=lambda p: p[1])
_E5M2_POS_CODES = np.array([c for c, _ in _pos5], dtype=np.uint8)
_E5M2_POS_VALUES = np.array([v for _, v in _pos5], dtype=np.float32)
_E5M2_MIDPOINTS = (np.diff(_E5M2_POS_VALUES) / 2 + _E5M2_POS_VALUES[:-1]).astype(np.float32)


def decode_fp8(fmt, codes):
    return decode_e4m3(codes) if fmt == 'e4m3' else decode_e5m2(codes)


def encode_fp8(fmt, values):
    return encode_e4m3(values) if fmt == 'e4m3' else encode_e5m2(values)


def _broadcast_scales(shape, scales, mode, axis, block):
    """Expand a scale tensor to the element layout of the weights."""
    scales = np.asarray(scales, dtype=np.float32)
    if mode == 'tensor':
        if scales.size != 1:
            raise ValueError(f'per-tensor scale must hold one value, got shape {scales.shape}')
        return np.broadcast_to(scales.reshape(()), shape).reshape(-1)
    if len(shape) == 1:
        raise ValueError('block scaling needs a 2-D tensor; use --scale-mode tensor for a vector')
    rows, cols = shape
    if axis == 'row':
        if cols % block:
            raise ValueError(f'{cols} columns not divisible by block {block}')
        expected = (rows, cols // block)
        if tuple(scales.shape) != expected:
            raise ValueError(f'row-block scale shape must be {expected}, got {tuple(scales.shape)}')
        return np.repeat(scales, block, axis=1).reshape(-1)
    if rows % block:
        raise ValueError(f'{rows} rows not divisible by block {block}')
    expected = (rows // block, cols)
    if tuple(scales.shape) != expected:
        raise ValueError(f'column-block scale shape must be {expected}, got {tuple(scales.shape)}')
    return np.tile(scales.reshape(-1, 1), (1, block)).reshape(-1)


def _amax_by_block(values_reshaped, mode, axis, block):
    """Per-element amax reduced back to the scale tensor shape."""
    flat = values_reshaped
    if mode == 'tensor':
        return float(flat.max()) if flat.size else 0.0
    shape = values_reshaped.shape
    if len(shape) != 2:
        raise ValueError('internal layout error: block mode expects 2-D')
    rows, cols = shape
    if axis == 'row':
        return flat.reshape(rows, cols // block, block).max(axis=2)
    return flat.reshape(rows // block, block, cols).max(axis=1)


def safetensors_fp8_edit(source, target, fmt, weights_name, delta_path, scales_name,
                         scale_mode='block', axis='row'):
    """Edit one FP8 tensor + its scale tensor; every other byte copied verbatim."""
    if fmt not in FP8_MAX:
        raise ValueError("format must be 'e4m3' or 'e5m2'")
    if scale_mode not in ('tensor', 'block'):
        raise ValueError("scale-mode must be 'tensor' or 'block'")
    if scale_mode == 'block' and axis not in ('row', 'col'):
        raise ValueError('axis must be row or col for block scales')
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        raise ValueError('output must differ from input')
    if target.exists():
        raise ValueError(f'{target} already exists')
    data, header, base = _read_safetensors(source)
    storage = FP8_STORAGE[fmt]
    winfo = header.get(weights_name)
    sinfo = header.get(scales_name)
    if winfo is None:
        raise ValueError(f'{weights_name} is not in {source.name}')
    if sinfo is None:
        raise ValueError(f'{scales_name} is not in {source.name}')
    if winfo['dtype'] != storage:
        raise ValueError(f'{weights_name} is {winfo["dtype"]}, not {storage}; pick --format to match')
    if sinfo['dtype'] != 'F32':
        raise ValueError(f'{scales_name} must be an F32 scale tensor; got {sinfo["dtype"]}')
    shape = tuple(int(x) for x in winfo['shape'])
    wa, wb = winfo['data_offsets']
    sa, sb = sinfo['data_offsets']
    codes = np.frombuffer(data, dtype=np.uint8, count=wb - wa, offset=base + wa).copy()
    if codes.size != int(np.prod(shape, dtype=np.int64)):
        raise ValueError(f'{weights_name} declares shape {shape} but holds {codes.size} fp8 bytes')
    scales = np.frombuffer(data, dtype=np.float32, count=(sb - sa) // 4, offset=base + sa).copy()
    scale_shape = tuple(int(x) for x in sinfo.get('shape', [scales.size]))
    if scales.size != int(np.prod(scale_shape, dtype=np.int64)):
        raise ValueError(f'{scales_name} declares shape {scale_shape} but holds {scales.size} F32 values')
    scales = scales.reshape(scale_shape)
    scale_layout = _broadcast_scales(shape, scales, scale_mode, axis, 128)
    delta = np.load(delta_path)
    if delta.dtype != np.float32 or delta.size != codes.size:
        raise ValueError(f'delta must be float32 with {codes.size} elements matching '
                         f'{weights_name}, got dtype {delta.dtype} size {delta.size}')
    values = decode_fp8(fmt, codes).astype(np.float32) * scale_layout
    edited = values + delta.reshape(-1)
    if not np.isfinite(edited).all():
        raise ValueError('edited tensor contains NaN/Inf; refusing to encode')
    amax = _amax_by_block(np.abs(edited).reshape(shape), scale_mode, axis, 128)
    fp8_max = FP8_MAX[fmt]
    if scale_mode == 'tensor':
        new_scale_inv = np.array([max(float(amax), np.finfo(np.float32).tiny) / fp8_max],
                                 dtype=np.float32)
        layout = np.full(codes.size, float(new_scale_inv[0]), dtype=np.float32)
    else:
        new_scale_inv = np.maximum(amax / fp8_max, np.finfo(np.float32).tiny).astype(np.float32)
        layout = _broadcast_scales(shape, new_scale_inv, scale_mode, axis, 128)
    new_codes = encode_fp8(fmt, edited.reshape(-1) / layout).reshape(-1)
    new_scales = new_scale_inv.reshape(-1)
    if new_codes.size != codes.size or new_scales.size != scales.size:
        raise ValueError('requant changed element or scale count; refusing a rewriting edit')
    new = bytearray(data)
    new[base + wa:base + wb] = new_codes.tobytes()
    new[base + sa:base + sb] = new_scales.astype(np.float32).tobytes()
    target.write_bytes(bytes(new))
    return {'input': str(source), 'output': str(target), 'format': fmt,
            'tensor': weights_name, 'scales': scales_name,
            'elements': int(codes.size), 'scale_mode': scale_mode,
            'axis': axis if scale_mode == 'block' else None,
            'block': 128 if scale_mode == 'block' else None,
            'note': 'Weights and scales byte ranges rewritten in fp8 layout; every other byte copied '
                     'verbatim. A zero delta preserves dequantized values exactly but may re-encode '
                     'blocks with a different equivalent scale, so byte identity with the source is '
                     'not guaranteed.'}
