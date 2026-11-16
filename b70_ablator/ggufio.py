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
