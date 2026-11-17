# SPDX-License-Identifier: AGPL-3.0-or-later
"""MXFP4/NVFP4 block codecs and in-place safetensors editing for fp4 checkpoints.

MXFP4 (OCP microscaling): blocks of 32 elements with an E8M0 shared exponent
scale. NVFP4: blocks of 16 elements with an E4M3 scale and one global tensor
scale. Digits are E2M1 in both. Packed layout is two elements per byte, low
nibble first, matching the convention used by the major fp4 serving stacks.

Rounding is round-to-nearest with ties-to-even against the E2M1 grid. Editing
follows the same contract as ggufio: dequantize the block, add the FP32 delta,
requantize to identical sizes; every byte outside the edited tensor pair
(weights and scales) is copied verbatim and offsets never move.
"""
import json
import numpy as np
from pathlib import Path

E2M1_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
# Midpoints between consecutive grid magnitudes; ties resolve to the entry
# whose E2M1 mantissa bit is even (searchsorted side='left' does exactly this).
E2M1_MIDPOINTS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)
MXFP4_BLOCK = 32
NVFP4_BLOCK = 16
E4M3_MAX = 448.0


def _decode_table_e4m3():
    table = np.zeros(256, dtype=np.float32)
    for b in range(256):
        if b in (0x7F, 0xFF):
            table[b] = np.nan
            continue
        sign = -1.0 if (b >> 7) else 1.0
        e, m = (b >> 3) & 0xF, b & 0x7
        value = m * 2.0 ** -9 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7)
        table[b] = sign * value
    return table


_E4M3_DECODE = _decode_table_e4m3()
# Sorted positive magnitudes with byte codes for RNE encode; midpoints carry
# ties to the even-mantissa code (code bit 0 == mantissa LSB for normal and
# subnormal values alike).
_pos = [(b, _E4M3_DECODE[b]) for b in range(1, 128) if not np.isnan(_E4M3_DECODE[b])]
_pos.sort(key=lambda p: p[1])
_E4M3_POS_CODES = np.array([c for c, _ in _pos], dtype=np.uint8)
_E4M3_POS_VALUES = np.array([v for _, v in _pos], dtype=np.float32)
_E4M3_MIDPOINTS = (np.diff(_E4M3_POS_VALUES) / 2 + _E4M3_POS_VALUES[:-1]).astype(np.float32)


def decode_e4m3(codes):
    return _E4M3_DECODE[np.asarray(codes, dtype=np.uint8)]


def encode_e4m3(values):
    """Round-to-nearest E4M3 encode with ties-to-even; saturates to +-448."""
    values = np.asarray(values, dtype=np.float32)
    magnitude = np.clip(np.abs(values), None, E4M3_MAX)
    idx = np.searchsorted(_E4M3_MIDPOINTS, magnitude, side='left')
    lower_odd = (idx < _E4M3_POS_CODES.size) & (((_E4M3_POS_CODES[np.minimum(idx, _E4M3_POS_CODES.size - 1)]) & 1) == 1)
    ties = (idx < _E4M3_MIDPOINTS.size) & (magnitude == _E4M3_MIDPOINTS[np.minimum(idx, _E4M3_MIDPOINTS.size - 1)])
    idx = idx + (ties & lower_odd)
    codes = _E4M3_POS_CODES[idx].copy()
    codes[magnitude <= 0] = 0
    return np.where(np.signbit(values), codes | 0x80, codes).astype(np.uint8)


def decode_e8m0(codes):
    return np.power(2.0, np.asarray(codes, dtype=np.float32) - 127.0)


def encode_e8m0(amax):
    """Shared exponent scale = 2^(ceil(log2(amax/6))); zero blocks get exponent 127."""
    amax = np.asarray(amax, dtype=np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        exponent = np.ceil(np.log2(amax / 6.0))
    exponent = np.where(amax > 0, np.clip(exponent, -127.0, 127.0), 0.0)
    return (exponent + 127.0).astype(np.uint8)


def _encode_e2m1_codes(x):
    magnitude = np.minimum(np.abs(x), 6.0)
    idx = np.searchsorted(E2M1_MIDPOINTS, magnitude, side='left')
    # Exact midpoints must round to the grid entry with an even mantissa bit.
    ties = np.where(idx < E2M1_MIDPOINTS.size, magnitude == E2M1_MIDPOINTS[np.minimum(idx, E2M1_MIDPOINTS.size - 1)], False)
    odd_lower = (idx < E2M1_MIDPOINTS.size) & ((idx % 2) == 1)
    idx = idx + (ties & odd_lower)
    code = idx.astype(np.uint8)
    return np.where(np.signbit(x), code | 0x08, code).astype(np.uint8)


def _decode_e2m1_codes(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    value = E2M1_VALUES[codes & 0x7]
    return np.where(codes & 0x08, -value, value)


def pack_nibbles(codes):
    codes = np.asarray(codes, dtype=np.uint8).reshape(-1)
    if codes.size % 2:
        raise ValueError('fp4 element count must be even to pack nibbles')
    return (codes[0::2] | (codes[1::2] << 4)).astype(np.uint8)


def unpack_nibbles(packed):
    packed = np.asarray(packed, dtype=np.uint8).reshape(-1)
    out = np.empty(packed.size * 2, dtype=np.uint8)
    out[0::2] = packed & 0x0F
    out[1::2] = packed >> 4
    return out


def quantize_mxfp4(x):
    """x shape [..., K]; returns (packed uint8 [..., K/2], scales_e8m0 uint8 [..., K/32])."""
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] % MXFP4_BLOCK:
        raise ValueError(f'MXFP4 needs element counts multiple of {MXFP4_BLOCK}, got {x.shape[-1]}')
    blocks = x.reshape(-1, x.shape[-1] // MXFP4_BLOCK, MXFP4_BLOCK)
    amax = np.max(np.abs(blocks), axis=-1)
    scales = encode_e8m0(amax)
    de_scales = decode_e8m0(scales)[..., None]
    codes = _encode_e2m1_codes(blocks / np.where(amax[..., None] > 0, de_scales, 1.0))
    return pack_nibbles(codes.reshape(-1)).reshape(x.shape[:-1] + (x.shape[-1] // 2,)), scales


def dequantize_mxfp4(packed, scales, shape=None):
    codes = unpack_nibbles(packed)
    codes = codes.reshape(-1, MXFP4_BLOCK)
    scales = np.asarray(scales, dtype=np.uint8).reshape(-1, 1)
    values = _decode_e2m1_codes(codes) * decode_e8m0(scales)
    values = values.reshape(-1)
    if shape is not None:
        values = values.reshape(shape)
    return values


def quantize_nvfp4(x, global_amax=None):
    """x shape [..., K]; returns (packed, scales_e4m3 [..., K/16], global_scale float32)."""
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] % NVFP4_BLOCK:
        raise ValueError(f'NVFP4 needs element counts multiple of {NVFP4_BLOCK}, got {x.shape[-1]}')
    if global_amax is None:
        global_amax = np.max(np.abs(x)) if x.size else np.float32(0.0)
    global_scale = np.float32(float(global_amax) / (E4M3_MAX * 6.0))
    if global_scale <= 0:
        return np.zeros(x.shape[:-1] + (x.shape[-1] // 2,), dtype=np.uint8), \
               np.zeros(x.shape[:-1] + (x.shape[-1] // NVFP4_BLOCK,), dtype=np.uint8), global_scale
    blocks = x.reshape(-1, x.shape[-1] // NVFP4_BLOCK, NVFP4_BLOCK)
    amax = np.max(np.abs(blocks), axis=-1)
    raw = amax / (6.0 * float(global_scale))
    scale_codes = encode_e4m3(np.clip(raw, 0.0, E4M3_MAX))
    de_scales = decode_e4m3(scale_codes)[..., None] * float(global_scale)
    codes = _encode_e2m1_codes(np.where(amax[..., None] > 0, blocks / np.where(de_scales > 0, de_scales, 1.0), 0.0))
    return (pack_nibbles(codes.reshape(-1)).reshape(x.shape[:-1] + (x.shape[-1] // 2,)),
            scale_codes, global_scale)


def dequantize_nvfp4(packed, scales_e4m3, global_scale, shape=None):
    codes = unpack_nibbles(packed).reshape(-1, NVFP4_BLOCK)
    scales = decode_e4m3(np.asarray(scales_e4m3, dtype=np.uint8).reshape(-1, 1)) * float(global_scale)
    values = _decode_e2m1_codes(codes) * scales
    values = values.reshape(-1)
    if shape is not None:
        values = values.reshape(shape)
    return values


def _read_safetensors(path):
    path = Path(path)
    data = path.read_bytes()
    header_len = int.from_bytes(data[:8], 'little')
    header = json.loads(data[8:8 + header_len])
    return data, header, 8 + header_len


def safetensors_fp4_edit(source, target, fmt, weights_name, delta_path, scales_name, global_name=None, global_scale=None):
    """Edit one fp4 tensor pair in a safetensors file, preserving every other byte.

    ``fmt`` selects the block geometry and scale codec: 'mxfp4' (E8M0 scales,
    block 32) or 'nvfp4' (E4M3 scales, block 16, plus a global scale stored as
    a scalar tensor or supplied directly).
    """
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        raise ValueError('output must differ from input')
    if fmt not in ('mxfp4', 'nvfp4'):
        raise ValueError("format must be 'mxfp4' or 'nvfp4'")
    data, header, base = _read_safetensors(source)
    block = MXFP4_BLOCK if fmt == 'mxfp4' else NVFP4_BLOCK
    for name in (weights_name, scales_name):
        info = header.get(name)
        if info is None:
            raise ValueError(f'{name} is not in {source.name}')
        if info.get('dtype') != 'U8':
            raise ValueError(f'{name} must be U8-packed; got {info["dtype"]} (this is packed fp4 storage, not fp8 weights)')
    if source.resolve() != target.resolve() and target.exists():
        raise ValueError(f'{target} already exists')
    w = header[weights_name]
    s = header[scales_name]
    wa, wb = w['data_offsets']
    sa, sb = s['data_offsets']
    weights = np.frombuffer(data, dtype=np.uint8, count=wb - wa, offset=base + wa).copy()
    scales = np.frombuffer(data, dtype=np.uint8, count=sb - sa, offset=base + sa).copy()
    n_elements = int(weights.size) * 2
    expected_scales = n_elements // block
    if scales.size != expected_scales:
        raise ValueError(f'{scales_name} holds {scales.size} blocks for a {n_elements}-element tensor; needs {expected_scales}')
    delta = np.load(delta_path)
    if delta.dtype != np.float32 or delta.size != n_elements:
        raise ValueError(f'delta must be float32 with {n_elements} elements matching tensor {weights_name}, '
                         f'got dtype {delta.dtype} size {delta.size}')
    if fmt == 'mxfp4':
        values = dequantize_mxfp4(weights, scales)
        packed, new_scales = quantize_mxfp4(values + delta.reshape(values.shape))
    else:
        if global_scale is None:
            if global_name is None:
                raise ValueError('nvfp4 editing needs --global-scale or a --global-scale-tensor in the file')
            info = header.get(global_name)
            if info is None:
                raise ValueError(f'{global_name} is not in {source.name}')
            ga, gb = info['data_offsets']
            dtype = np.float32 if info['dtype'] == 'F32' else np.float64 if info['dtype'] == 'F64' else None
            if dtype is None:
                raise ValueError(f'{global_name} must be F32 or F64; got {info["dtype"]}')
            global_scale = float(np.frombuffer(data, dtype=dtype, count=1, offset=base + ga)[0])
        values = dequantize_nvfp4(weights, scales, global_scale)
        packed, new_scales, _ = quantize_nvfp4(values + delta.reshape(values.shape),
                                               global_amax=global_scale * E4M3_MAX * 6.0)
    if packed.size != weights.size or new_scales.size != scales.size:
        raise ValueError('requant changed packed or scale sizes; refusing a rewriting edit')
    new = bytearray(data)
    new[base + wa:base + wb] = packed.tobytes()
    new[base + sa:base + sb] = new_scales.tobytes()
    target.write_bytes(bytes(new))
    return {'input': str(source), 'output': str(target), 'format': fmt,
            'tensor': weights_name, 'scales': scales_name, 'elements': n_elements,
            'block': block, 'global_scale': float(global_scale) if global_scale is not None else None,
            'note': 'Weights and scales byte ranges rewritten in quantized layout; every other byte copied '
                     'verbatim. A zero delta preserves dequantized values exactly but may re-encode blocks with '
                     'a different (equivalent) block scale, so byte identity with the source is not guaranteed.'}
