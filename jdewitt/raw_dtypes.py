# SPDX-License-Identifier: AGPL-3.0-only
"""Raw GGML tensor dtypes I8/I16/I32/I64/F64: blockless little-endian codecs.

Contract (block size 1, bytes per element 1/2/4/8, no scale, no rounding):
  * bytes layout: element i occupies bytes [i*w, (i+1)*w), little-endian
    two's-complement ints (I*) or IEEE-754 binary64 (F64).
  * quantize_<t>(x) -> flat uint8 array. Accepts a float array (any float
    dtype) or ANY integer dtype (bool/complex/object rejected, TypeError).
      - float -> int: C-style truncation toward zero (2.9 -> 2, -2.9 -> -2).
        NaN, +-Inf, or a truncated value outside the target range raises
        ValueError (C leaves these undefined; we refuse to guess).
      - int -> int: value must fit the target range, else ValueError
        (no silent wraparound). Matching dtype passes through bit-exactly.
      - F64 target: float input is widened exactly (NaN payloads, +-Inf,
        -0.0 and subnormals preserved bit-for-bit); integer input is cast
        like C `(double)x` (exact up to 2**53, then nearest-even).
  * dequantize_<t>(raw, n=None) -> array of the NATIVE dtype (int8/16/32/64 or
    float64), not float32. n truncates; byte length must be a multiple of the
    element width and cover n elements, else ValueError.
"""
import numpy as np

# name -> (bytes_per_block, elements_per_block, numpy little-endian dtype)
RAW_DTYPES = {
    'I8': (1, 1, np.dtype('<i1')),
    'I16': (2, 1, np.dtype('<i2')),
    'I32': (4, 1, np.dtype('<i4')),
    'I64': (8, 1, np.dtype('<i8')),
    'F64': (8, 1, np.dtype('<f8')),
}


def _encode(name, x):
    _, _, dt = RAW_DTYPES[name]
    a = np.asarray(x)
    kind = a.dtype.kind
    if kind not in 'fiu':
        raise TypeError(f'{name}: need float or integer array, got {a.dtype}')
    a = a.reshape(-1)
    if dt.kind == 'f':
        out = a.astype(dt)
    else:
        info = np.iinfo(dt)
        if kind == 'f':
            if not np.isfinite(a).all():
                raise ValueError(f'{name}: NaN/Inf cannot be cast to integer')
            t = np.trunc(a.astype(np.float64))
            # float64 bounds: -2**(bits-1) exact, upper bound exclusive 2**(bits-1)
            lo, hi = -(2.0 ** (8 * dt.itemsize - 1)), 2.0 ** (8 * dt.itemsize - 1)
            if ((t < lo) | (t >= hi)).any():
                raise ValueError(f'{name}: value out of range after truncation')
            out = t.astype(dt)
        else:
            if a.size and (int(a.min()) < info.min or int(a.max()) > info.max):
                raise ValueError(f'{name}: integer out of range')
            out = a.astype(dt)
    return np.ascontiguousarray(out).view(np.uint8).reshape(-1)


def _decode(name, raw, n):
    w, _, dt = RAW_DTYPES[name]
    b = np.frombuffer(raw, dtype=np.uint8) if not isinstance(raw, np.ndarray) \
        else np.ascontiguousarray(raw).view(np.uint8).reshape(-1)
    if b.size % w:
        raise ValueError(f'{name}: {b.size} bytes not a multiple of {w}')
    if n is not None and n * w > b.size:
        raise ValueError(f'{name}: need {n * w} bytes, got {b.size}')
    out = b.view(dt)
    if n is not None:
        out = out[:n]
    return out.astype(dt.newbyteorder('='))  # copy, native order


def quantize_i8(x): return _encode('I8', x)
def quantize_i16(x): return _encode('I16', x)
def quantize_i32(x): return _encode('I32', x)
def quantize_i64(x): return _encode('I64', x)
def quantize_f64(x): return _encode('F64', x)
def dequantize_i8(raw, n=None): return _decode('I8', raw, n)
def dequantize_i16(raw, n=None): return _decode('I16', raw, n)
def dequantize_i32(raw, n=None): return _decode('I32', raw, n)
def dequantize_i64(raw, n=None): return _decode('I64', raw, n)
def dequantize_f64(raw, n=None): return _decode('F64', raw, n)


# same shape as quants.CODECS: name -> (bytes_per_block, elems_per_block, quantizer, dequantizer)
RAW_CODECS = {
    'I8': (1, 1, quantize_i8, dequantize_i8),
    'I16': (2, 1, quantize_i16, dequantize_i16),
    'I32': (4, 1, quantize_i32, dequantize_i32),
    'I64': (8, 1, quantize_i64, dequantize_i64),
    'F64': (8, 1, quantize_f64, dequantize_f64),
}
