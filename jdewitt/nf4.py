"""NormalFloat4 (NF4) codec for the bitsandbytes blockwise layout.

Upstream provenance (pinned, digested — do not edit the table by hand):
  https://raw.githubusercontent.com/TimDettmers/bitsandbytes/0.42.0/bitsandbytes/functional.py
  sha256 ebfd5254f9ee14107ed613506693310132d0214bd77fd07589182293066f318d
The 16-value codebook is ``get_4bit_type('nf4')`` from that revision. It was
re-derived independently here (stdlib normal CDF + bisection inverse over the
``create_normal_map(offset=0.9677083)`` linspace points) and matched the
pinned table to max abs diff 1.08e-7 (fp32 rounding of the upstream table).

Storage layout assumed (bitsandbytes blockwise 4-bit as stored by HF):
  weights  uint8  [rows, cols // 2]   two 4-bit codes packed per byte
  absmax   f32/f16 [rows, cols // blocksize]  one absmax per block of 64
  decode:  value = absmax[block] * TABLE[code]
  packing: low nibble of byte i is element 2*i, high nibble is element 2*i+1

HONEST LIMITS: the packing nibble order matches the bitsandbytes CUDA kernel
convention but has not been checked against a real bitsandbytes checkpoint on
this machine (no nf4 checkpoint present; the stored absmax tensor cannot
distinguish nibble order, so it offers no independent confirmation). If this
convention is wrong the decode is a pairwise element swap — flag before
trusting a produced file.
"""

import json
from pathlib import Path

import numpy as np

NF4_UPSTREAM = ('https://raw.githubusercontent.com/TimDettmers/bitsandbytes/'
                '0.42.0/bitsandbytes/functional.py')
NF4_UPSTREAM_SHA256 = 'ebfd5254f9ee14107ed613506693310132d0214bd77fd07589182293066f318d'
NF4_BLOCK = 64

# get_4bit_type('nf4'), pinned upstream 0.42.0 (float32 values as published).
NF4_TABLE = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0], dtype=np.float32)

# Midpoints between consecutive table values for nearest-code assignment.
_MID = ((NF4_TABLE[:-1] + NF4_TABLE[1:]) / 2).astype(np.float32)


def encode_blockwise(values, blocksize=NF4_BLOCK):
    """(rows, cols) float array -> (packed uint8 [rows, cols//2], absmax f32).

    Codes are nearest in normalised distance; a value exactly on a midpoint
    takes the higher code (np.searchsorted side='right'). absmax of an
    all-zero block is 0.0 (codes then decode to 0 regardless).
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f'nf4 blockwise encode wants a 2-D array, got {values.shape}')
    rows, cols = values.shape
    if cols % 2:
        raise ValueError(f'nf4 packs pairs of elements; got odd width {cols}')
    if cols % blocksize:
        raise ValueError(f'width {cols} not a multiple of blocksize {blocksize}')
    shape = (rows, cols // blocksize, blocksize)
    view = values.reshape(shape)
    absmax = np.max(np.abs(view), axis=2, keepdims=True).astype(np.float32)
    safe = np.where(absmax > 0, absmax, 1.0)
    norm = view / safe
    codes = np.searchsorted(_MID, norm, side='right').astype(np.uint8)
    norm = norm * absmax
    packed = (codes[:, :, ::2] | (codes[:, :, 1::2] << 4)).reshape(rows, cols // 2)
    return packed, absmax.reshape(rows, cols // blocksize)


def decode_blockwise(packed, absmax, blocksize=NF4_BLOCK):
    """(packed uint8, absmax) -> float32 array [rows, cols]."""
    packed = np.frombuffer(np.asarray(packed, dtype=np.uint8).tobytes(), dtype=np.uint8)
    absmax = np.asarray(absmax, dtype=np.float32)
    if absmax.ndim == 1:
        absmax = absmax.reshape(1, -1)
    rows = packed.shape[0] if packed.ndim == 2 else absmax.shape[0]
    packed = packed.reshape(rows, -1)
    cols2 = packed.shape[1]
    cols = cols2 * 2
    nblocks = cols // blocksize
    if absmax.shape[0] != rows or absmax.shape[1] != nblocks:
        raise ValueError(f'absmax shape {absmax.shape} incompatible with packed {packed.shape} '
                         f'and blocksize {blocksize}')
    low = packed & 0x0F
    high = packed >> 4
    codes = np.empty((rows, cols), dtype=np.uint8)
    codes[:, ::2] = low
    codes[:, 1::2] = high
    deq = NF4_TABLE[codes].reshape(rows, nblocks, blocksize)
    return (deq * absmax.reshape(rows, nblocks, 1)).reshape(rows, cols)


def safetensors_nf4_edit(source, target, weights_name, delta_path, scales_name,
                         blocksize=NF4_BLOCK):
    """Edit one NF4-packed tensor + its absmax tensor; all other bytes verbatim.

    Reads bitsandbytes-style packed weights (U8, two codes/byte) and an F32 or
    F16 absmax tensor per 64-element block, applies a full-precision float32
    delta, requantizes, and rewrites only the two byte ranges.
    """
    from .fp4 import _read_safetensors
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        raise ValueError('output must differ from input')
    if target.exists():
        raise ValueError(f'{target} already exists')
    if blocksize % 2:
        raise ValueError(f'blocksize must be even; got {blocksize}')
    data, header, base = _read_safetensors(source)
    winfo = header.get(weights_name)
    sinfo = header.get(scales_name)
    if winfo is None:
        raise ValueError(f'{weights_name} is not in {source.name}')
    if sinfo is None:
        raise ValueError(f'{scales_name} is not in {source.name}')
    if winfo['dtype'] != 'U8':
        raise ValueError(f'{weights_name} must be U8-packed; got {winfo["dtype"]}')
    scale_np = {'F32': np.float32, 'F16': np.float16}.get(sinfo['dtype'])
    if scale_np is None:
        raise ValueError(f'{scales_name} must be F32 or F16 absmax; got {sinfo["dtype"]}')
    wa, wb = winfo['data_offsets']
    sa, sb = sinfo['data_offsets']
    packed = np.frombuffer(data, dtype=np.uint8, count=wb - wa, offset=base + wa).copy()
    shape = tuple(int(x) for x in winfo.get('shape', [packed.shape[0]]))
    if int(np.prod(shape, dtype=np.int64)) != packed.size:
        raise ValueError(f'{weights_name} declares shape {shape} ({int(np.prod(shape, dtype=np.int64))} '
                         f'elements) but holds {packed.size} packed bytes; safetensors stores the '
                         f'packed shape [rows, cols//2], not the logical shape')
    if packed.size % 2:
        raise ValueError(f'{weights_name} holds {packed.size} packed bytes; needs an even count to unpack')
    n_elements = int(packed.size) * 2
    absmax = np.frombuffer(data, dtype=scale_np, count=(sb - sa) // np.dtype(scale_np).itemsize,
                           offset=base + sa).copy()
    scale_shape = tuple(int(x) for x in sinfo.get('shape', [absmax.size]))
    if absmax.size != int(np.prod(scale_shape, dtype=np.int64)):
        raise ValueError(f'{scales_name} declares shape {scale_shape} but holds {absmax.size} values')
    if n_elements % blocksize or absmax.size != n_elements // blocksize:
        raise ValueError(f'{n_elements} elements with blocksize {blocksize} need '
                         f'{n_elements // blocksize} scales, file holds {absmax.size}')
    delta = np.load(delta_path)
    if delta.dtype != np.float32 or delta.size != n_elements:
        raise ValueError(f'delta must be float32 with {n_elements} elements matching '
                         f'{weights_name}, got dtype {delta.dtype} size {delta.size}')
    values = decode_blockwise(packed, absmax.astype(np.float32), blocksize)
    edited = values + delta.reshape(values.shape)
    if not np.isfinite(edited).all():
        raise ValueError('edited tensor contains NaN/Inf; refusing to encode')
    new_packed, new_absmax = encode_blockwise(edited, blocksize)
    if new_packed.size != packed.size or new_absmax.size != absmax.size:
        raise ValueError('requant changed packed or scale sizes; refusing a rewriting edit')
    new = bytearray(data)
    new[base + wa:base + wb] = new_packed.tobytes()
    new[base + sa:base + sb] = new_absmax.astype(scale_np).tobytes()
    target.write_bytes(bytes(new))
    return {'input': str(source), 'output': str(target), 'tensor': weights_name,
            'scales': scales_name, 'elements': n_elements, 'block': blocksize,
            'scale_dtype': sinfo['dtype'],
            'note': 'Packed weights and absmax byte ranges rewritten in NF4 layout; every other byte '
                     'copied verbatim. A zero delta re-encodes each block to its existing codes and '
                     'the recomputed absmax equals the stored one (the codebook spans +-1), so a zero '
                     'delta is byte-identical; packing nibble order is asserted, not yet checked '
                     'against a real bitsandbytes checkpoint.'}


# ---------------------------------------------------------------- double quant
# bitsandbytes 0.42 `Linear4bit(compress_statistics=True)` quantizes the NF4
# absmax values a second time with `quantize_blockwise(...)`: the dynamic 8-bit
# map from `create_dynamic_map`, blocks of 256 absmax values, and a float offset
# (`offset = absmax.mean()`, subtracted before quantizing and added back on
# decode). The map below is a port of that pinned function; `quantize_cpu` also
# overwrites its first entry with -1.0 before the binary search, which is part of
# the effective table and therefore of the port.
DYNAMIC_MAP_POINTS = 256


def dynamic_map(signed=True, max_exponent_bits=7, total_bits=8):
    """Port of `create_dynamic_map` (bnb 0.42), with `code[0] = -1.0` applied."""
    non_sign_bits = total_bits - 1
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    data = []
    last_exponent = 0
    for i in range(max_exponent_bits):
        fraction_items = int(2 ** (i + non_sign_bits - max_exponent_bits) + 1 if signed
                             else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1)
        boundaries = np.linspace(0.1, 1.0, fraction_items, dtype=np.float32)
        means = (boundaries[:-1] + boundaries[1:]) / np.float32(2.0)
        scale = np.float32(10.0) ** np.float32(-(max_exponent_bits - 1) + i)
        data += (scale * means).tolist()
        if signed:
            data += (-scale * means).tolist()
        last_exponent = i
    if additional_items > 0:
        boundaries = np.linspace(0.1, 1.0, additional_items + 1, dtype=np.float32)
        means = (boundaries[:-1] + boundaries[1:]) / np.float32(2.0)
        scale = np.float32(10.0) ** np.float32(-(max_exponent_bits - 1) + last_exponent)
        data += (scale * means).tolist()
        if signed:
            data += (-scale * means).tolist()
    data.append(0.0)
    data.append(1.0)
    data.sort()
    table = np.asarray(data, dtype=np.float32)
    if table.size != 2 ** total_bits:
        raise RuntimeError(f'dynamic map has {table.size} entries, expected {2 ** total_bits}')
    table[0] = np.float32(-1.0)
    return table


_DYNAMIC_CACHE = {}


def _dynamic_code():
    if 'code' not in _DYNAMIC_CACHE:
        _DYNAMIC_CACHE['code'] = dynamic_map()
    return _DYNAMIC_CACHE['code']


def _nearest_code(values, code):
    """bnb `quantize_block` search: left neighbour, then right only if strictly closer."""
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    idx = np.searchsorted(code, flat, side='right') - 1
    idx = np.clip(idx, 0, code.size - 1)
    right = np.minimum(idx + 1, code.size - 1)
    take_right = (np.abs(flat - code[right]) < np.abs(flat - code[idx])) & (idx < code.size - 1)
    return np.where(take_right, right, idx).astype(np.uint8)


def quantize_blockwise_u8(values, blocksize=256):
    """Quantize float values to 8-bit dynamic-map codes + per-block absmax (f32)."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    code = _dynamic_code()
    blocks = -(-values.size // blocksize)
    padded = np.zeros(blocks * blocksize, dtype=np.float32)
    padded[:values.size] = values
    view = padded.reshape(blocks, blocksize)
    absmax = np.max(np.abs(view), axis=1).astype(np.float32)
    safe = np.where(absmax > 0, absmax, np.float32(1.0)).astype(np.float32)
    codes = np.empty(view.shape, dtype=np.uint8)
    for block in range(blocks):
        codes[block] = _nearest_code(view[block] / safe[block], code)
    return codes.reshape(-1)[:values.size], absmax


def dequantize_blockwise_u8(codes, absmax, blocksize=256, n=None):
    """Decode 8-bit dynamic-map codes with per-block absmax back to float32."""
    codes = np.asarray(codes, dtype=np.uint8).reshape(-1)
    count = codes.size if n is None else int(n)
    code = _dynamic_code()
    blocks = -(-count // blocksize)
    padded = np.zeros(blocks * blocksize, dtype=np.uint8)
    padded[:codes.size] = codes
    vals = code[padded].astype(np.float32).reshape(blocks, blocksize)
    return (vals * np.asarray(absmax, dtype=np.float32).reshape(blocks, 1)).reshape(-1)[:count]


def encode_absmax_double(absmax, offset, nested_blocksize=256):
    """absmax (f32 per 64-block) -> (qabsmax u8, nested_absmax f32), bnb semantics."""
    absmax = np.asarray(absmax, dtype=np.float32)
    shifted = (absmax - np.float32(offset)).astype(np.float32)
    codes, nested = quantize_blockwise_u8(shifted, nested_blocksize)
    return codes, nested


def decode_absmax_double(codes, nested_absmax, offset, nested_blocksize=256, n=None):
    vals = dequantize_blockwise_u8(codes, nested_absmax, nested_blocksize, n=n)
    return (vals + np.float32(offset)).astype(np.float32)


def safetensors_nf4_edit_double(source, target, weights_name, delta_path,
                                absmax_name, nested_absmax_name, quant_state_name,
                                blocksize=NF4_BLOCK, nested_blocksize=256):
    """Edit a double-quantized NF4 tensor in place: weights, u8 absmax and nested absmax.

    `quant_state_name` is the packed JSON blob (`quant_state.bitsandbytes__nf4`);
    only its `nested_offset` is read, and the blob is copied verbatim: the stored
    offset stays the one the file was quantized with, so a re-encode never needs
    to move a byte outside the three tensor ranges.
    """
    from .fp4 import _read_safetensors
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        raise ValueError('output must differ from input')
    if target.exists():
        raise ValueError(f'{target} already exists')
    data, header, base = _read_safetensors(source)
    for name in (weights_name, absmax_name, nested_absmax_name, quant_state_name):
        if name not in header:
            raise ValueError(f'{name} is not in {source.name}')
    winfo, ainfo = header[weights_name], header[absmax_name]
    ninfo, qinfo = header[nested_absmax_name], header[quant_state_name]
    if winfo['dtype'] != 'U8':
        raise ValueError(f'{weights_name} must be U8-packed; got {winfo["dtype"]}')
    if ainfo['dtype'] != 'U8':
        raise ValueError(f'{absmax_name} must be the U8 quantized absmax; got {ainfo["dtype"]}')
    if ninfo['dtype'] != 'F32':
        raise ValueError(f'{nested_absmax_name} must be F32; got {ninfo["dtype"]}')
    wa, wb = winfo['data_offsets']
    aa, ab = ainfo['data_offsets']
    na, nb = ninfo['data_offsets']
    qa, qb = qinfo['data_offsets']
    packed = np.frombuffer(data, dtype=np.uint8, count=wb - wa, offset=base + wa).copy()
    qabsmax = np.frombuffer(data, dtype=np.uint8, count=ab - aa, offset=base + aa).copy()
    nested = np.frombuffer(data, dtype=np.float32, count=(nb - na) // 4, offset=base + na).copy()
    blob = json.loads(bytes(data[base + qa:base + qb]).decode('utf-8'))
    if blob.get('quant_type') != 'nf4':
        raise ValueError(f'quant_state says {blob.get("quant_type")!r}, expected nf4')
    if int(blob.get('nested_blocksize', nested_blocksize)) != nested_blocksize:
        raise ValueError(f'quant_state nested_blocksize {blob.get("nested_blocksize")} '
                         f'!= {nested_blocksize}')
    offset = np.float32(blob.get('nested_offset', 0.0))
    n_elements = int(packed.size) * 2
    if n_elements % blocksize:
        raise ValueError(f'{n_elements} elements not a multiple of {blocksize}')
    n_absmax = n_elements // blocksize
    if qabsmax.size != n_absmax:
        raise ValueError(f'{absmax_name} holds {qabsmax.size} codes, expected {n_absmax}')
    if nested.size != -(-n_absmax // nested_blocksize):
        raise ValueError(f'{nested_absmax_name} holds {nested.size} values, expected '
                         f'{-(-n_absmax // nested_blocksize)}')
    absmax = decode_absmax_double(qabsmax, nested, offset, nested_blocksize, n=n_absmax)
    delta = np.load(delta_path)
    values = decode_blockwise(packed, absmax, blocksize)
    if delta.dtype != np.float32 or delta.size != values.size:
        raise ValueError(f'delta must be float32 with {values.size} elements, got '
                         f'{delta.dtype} size {delta.size}')
    edited = values + delta.reshape(values.shape)
    if not np.isfinite(edited).all():
        raise ValueError('edited tensor contains NaN/Inf; refusing to encode')
    new_packed, new_absmax = encode_blockwise(edited, blocksize)
    new_codes, new_nested = encode_absmax_double(new_absmax.reshape(-1), offset, nested_blocksize)
    if new_packed.size != packed.size or new_codes.size != qabsmax.size or new_nested.size != nested.size:
        raise ValueError('requant changed a tensor size; refusing a rewriting edit')
    out = bytearray(data)
    out[base + wa:base + wb] = new_packed.tobytes()
    out[base + aa:base + ab] = new_codes.tobytes()
    out[base + na:base + nb] = new_nested.astype(np.float32).tobytes()
    target.write_bytes(bytes(out))
    return {'input': str(source), 'output': str(target), 'tensor': weights_name,
            'scales': absmax_name, 'scale_dtype': 'U8 (quantized absmax)',
            'nested_absmax': nested_absmax_name,
            'elements': n_elements, 'block': blocksize,
            'nested_block': nested_blocksize, 'offset': float(offset),
            'absmax_codes': int(new_codes.size), 'nested_absmax_values': int(new_nested.size),
            'note': 'Weights, U8 absmax and nested absmax byte ranges rewritten; the packed '
                    'quant_state blob (and its offset) is copied verbatim, so no offset moves. '
                    'Layout and the dynamic map are ports of pinned bitsandbytes 0.42; no real '
                    'bnb double-quantized checkpoint was available to confirm the container.'}
