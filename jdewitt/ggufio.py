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
import os
import hashlib
import struct
import numpy as np
from pathlib import Path
from .quants import CODECS


def _plan(reader, deltas):
    """Validated per-tensor edit plans, shared by the copy and in-place paths."""
    plans = []
    for name, delta in deltas.items():
        tensor, dtype, bs, els, n, quantize, dequantize, offset, n_bytes, shape = _tensor_info(reader, name)
        delta = np.asarray(delta, dtype=np.float32)
        if delta.shape != shape:
            raise ValueError(f'delta for {name} has shape {delta.shape}, tensor shape is {shape}')
        if not np.isfinite(delta).all():
            raise ValueError(f'delta for {name} contains NaN/Inf')
        if n != int(np.prod(shape)):
            raise ValueError(f'tensor {name}: {n} values for shape {shape}')
        # Both consumers (copy and in-place) unpack (name, tensor, dtype, bs, els, n,
        # quantize, dequantize, offset, n_bytes, shape, delta); a plan without the
        # tensor object crashed edit_tensors with an 11-vs-12 unpack.
        plans.append((name, tensor, dtype, bs, els, n, quantize, dequantize, offset,
                      n_bytes, shape, delta.reshape(-1)))
    return plans


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
    # tensor.shape is ggml ne order (innermost first). Flat tensor bytes are
    # C-order of the numpy logical array, whose shape is reversed(tensor.shape)
    # — the convention GGUFWriter enforces and gguf_edit relies on. shape below
    # is that LOGICAL shape (e.g. weight [out, in]).
    shape = tuple(int(x) for x in np.asarray(tensor.shape).ravel())[::-1]
    return tensor, dtype, block_bytes, block_els, n, quantize, dequantize, offset, n_bytes, shape


def axis_delta(axis, matrix, scale=1.0, space='output'):
    """delta for a 1-D direction or a k-direction subspace (rows orthonormal).

    A subspace removal is `W - D^T D W`, i.e. the sum of its rank-1 projections, so one
    helper serves rank-1 and rank-k and the two can never drift apart.
    """
    a = np.asarray(axis, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if space == 'output':
        return -scale * (a.T @ (a @ matrix))
    return -scale * ((a.T @ (a @ matrix.T)).T)


def decode_values(raw, qtype_name, n_values=None):
    """Dequantize one tensor from ANY gguf quant type this build can read.

    Our own codecs are used when they exist (byte-exact, proven against the compiled
    C); anything else — IQ1/IQ2/IQ3/IQ4, TQ1/TQ2 and friends — falls back to
    gguf-py's reference dequantizer. Reading the user's artifact must never depend on
    which quant they chose; only the type we WRITE has to be one we can encode.
    """
    raw = np.asarray(raw, dtype=np.uint8).reshape(-1)
    from .raw_dtypes import RAW_CODECS
    raw_codec = RAW_CODECS.get(str(qtype_name))
    if raw_codec is not None:
        # Raw integer / F64 tensors keep their native dtype: a float32 cast would
        # silently round I32/I64 ids and masks.
        width, _els, _q, decode = raw_codec
        count = n_values if n_values is not None else raw.size // width
        return np.asarray(decode(raw, count)).reshape(-1)
    codec = CODECS.get(str(qtype_name))
    if codec is not None:
        block_bytes, block_els, _quantize, dequantize = codec
        count = (n_values if n_values is not None
                 else raw.size // block_bytes * block_els)
        return np.asarray(dequantize(raw, count), dtype=np.float32).reshape(-1)
    from gguf import GGML_QUANT_SIZES, GGMLQuantizationType, dequantize

    quantization = GGMLQuantizationType[str(qtype_name)]
    block_els, block_bytes = GGML_QUANT_SIZES[quantization]
    return np.asarray(dequantize(raw.reshape(-1, block_bytes), quantization),
                      dtype=np.float32).reshape(-1)


def encodable_types():
    """Quant types this build can WRITE (a subset of what it can read)."""
    return sorted(CODECS)


def preview_edit(path, edits, names=None, limit=None, scale=1.0, windows=2, blocks=32,
                 renormalize=False, target_type=None):
    """Simulate an in-place edit on sampled tensors: does it survive the codec?

    ``edits`` maps a tensor name to a float32 delta of that tensor's logical shape
    (or to ``(axis, space)`` exactly like ``project_tensors_inplace``, in which case
    ``scale`` multiplies the projection exactly as the writer does). The tile is
    dequantized, the delta added, then re-encoded and decoded with the SAME codec
    the writer uses. What comes back is what the served artifact will actually
    contain, so this answers a question no perplexity reading can: is the edit
    larger than the quantization step it has to be stored in?

    Measured on a 1.7B Q4_K_M: a rank-1 removal at strength 1.0 lands at
    cos(realized, intended) ~0.28-0.88 with noise as large as the edit, i.e. a
    strength-1.0 in-place merge on Q4_K ships a largely random edit. Nothing about
    that is visible in the adapter curve, which is why the two disagreed.

    ``target_type`` asks the question for a DIFFERENT storage plan: the edited
    tensors are going to be re-encoded at that type (see rewrite.py), so the fidelity
    floor has to be measured against the grid the edit will actually live in, not the
    one it is replacing. Without it, a 4-bit artifact looks uneditable for a whole
    class of strengths that a per-tensor re-encode stores perfectly.

    Only ``windows`` x ``blocks`` whole blocks of each tensor are re-encoded: the
    codec quantizes every block independently, so a spread of blocks is a faithful
    sample of its behaviour, while re-encoding 18M elements of an ffn_down tile per
    arm costs a minute of numpy per arm.
    """
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    chosen = [name for name in (names or edits) if name in edits]
    if limit:
        chosen = chosen[:limit]
    rows = []
    for name in chosen:
        tensor = next((item for item in reader.tensors if item.name == name), None)
        if tensor is None:
            raise ValueError(f'no tensor {name!r} in file')
        dtype = tensor.tensor_type.name
        n_bytes = int(tensor.n_bytes)
        shape = tuple(int(x) for x in np.asarray(tensor.shape).ravel())[::-1]
        offset = int(tensor.data_offset)
        with open(path, 'rb') as handle:
            handle.seek(offset)
            raw = np.frombuffer(handle.read(n_bytes), dtype=np.uint8)
        # ANY source quant decodes (decode_values falls back to gguf-py for IQ/TQ);
        # only the type we WRITE has to be one of our encoders.
        base = decode_values(raw, dtype)
        edit = edits[name]
        if isinstance(edit, tuple):
            axis, space = edit
            axis = np.asarray(axis, dtype=np.float32)  # (W,) or (k, W): never flatten a subspace
            matrix = base.reshape(shape)
            delta = axis_delta(axis, matrix, scale, space)
            if renormalize:
                edited = matrix + delta
                before = np.linalg.norm(matrix, axis=1, keepdims=True)
                after = np.linalg.norm(edited, axis=1, keepdims=True)
                delta = edited * (before / np.maximum(after, 1e-12)) - matrix
        else:
            delta = np.asarray(edit, dtype=np.float32).reshape(shape)
        wanted = (base.reshape(shape) + delta).reshape(-1).astype(np.float32)
        flat_base = base.reshape(-1)
        flat_delta = delta.reshape(-1)
        if target_type:
            from .quantize import canonical_type
            from .quants import CODECS as QUANT_CODECS

            _tb, window_els, encode, decode = QUANT_CODECS[canonical_type(target_type)]
        else:
            codec = CODECS.get(dtype)
            if codec is None:
                raise ValueError(f'{name} is {dtype}, which this build has no ENCODER for; '
                                 f'pass target_type to preview the re-encode that would ship')
            _bb, window_els, encode, decode = codec
        n = base.size
        window_len = int(blocks) * window_els
        span = max(0, n - window_len)
        offsets = sorted({min(int(round(span * step / max(1, windows - 1))) // window_els * window_els,
                             span - span % window_els)
                          for step in range(max(1, windows))})
        realised_parts, delta_parts = [], []
        for offset in offsets:
            end = min(n, offset + window_len)
            chunk = (end - offset) // window_els * window_els
            if chunk <= 0:
                continue
            reencoded = decode(encode(wanted[offset:offset + chunk].copy()), chunk)
            realised_parts.append(reencoded.astype(np.float32) - flat_base[offset:offset + chunk])
            delta_parts.append(flat_delta[offset:offset + chunk])
        if not realised_parts:
            continue
        realized = np.concatenate(realised_parts)
        delta_flat = np.concatenate(delta_parts)
        delta_norm = float(np.linalg.norm(delta_flat))
        noise = realized - delta_flat
        noise_norm = float(np.linalg.norm(noise))
        realised_norm = float(np.linalg.norm(realized))
        cos = (float((realized * delta_flat).sum() / (realised_norm * delta_norm + 1e-12))
               if delta_norm > 0 and realised_norm > 0 else 0.0)
        rows.append({'name': name, 'qtype': dtype, 'shape': list(shape),
                     'cos': round(cos, 4),
                     'norm_ratio': round(realised_norm / (delta_norm + 1e-12), 4),
                     'snr': round(delta_norm / (noise_norm + 1e-12), 3),
                     'delta_norm': round(delta_norm, 4),
                     'noise_norm': round(noise_norm, 4),
                     'windows': len(offsets), 'blocks_per_window': int(blocks)})
    cosines = [row['cos'] for row in rows]
    return {'tensors': rows, 'sampled': len(rows),
            'min_cos': min(cosines) if cosines else None,
            'mean_cos': round(sum(cosines) / len(cosines), 4) if cosines else None,
            'min_snr': min((row['snr'] for row in rows), default=None),
            'note': 'a low cos here means the codec, not the axis, decides what ships'}


def compare_edit(source, edited, edits, names=None, limit=None, renormalize=False):
    """What did the SHIPPED file keep of the intended edit?

    ``source`` is the pristine artifact the project started from, ``edited`` is the
    file that will be served, and ``edits`` is {name: (axis, space)} — the projection
    the ladder chose. The realized change is dequant(edited) - dequant(source), so
    this is the only number that describes the delivered edit; an adapter curve
    describes the wanted one. Use it after merge+requantize to prove the pipeline
    kept the edit and to name the tensors that lost it.
    """
    from gguf import GGUFReader

    def load(path, name):
        # Any quant type decodes: the source may be IQ/TQ/whatever the user runs,
        # and the edited side may hold the re-encoded type.
        reader = GGUFReader(str(path))
        tensor = next((item for item in reader.tensors if item.name == name), None)
        if tensor is None:
            raise ValueError(f'no tensor {name!r} in {path}')
        raw = np.memmap(str(path), dtype=np.uint8, mode='r', offset=int(tensor.data_offset),
                        shape=(int(tensor.n_bytes),))
        dtype = tensor.tensor_type.name
        shape = tuple(int(x) for x in np.asarray(tensor.shape).ravel())[::-1]
        values = decode_values(np.array(raw), dtype)
        return values.astype(np.float32).reshape(shape), dtype

    chosen = [name for name in (names or edits) if name in edits]
    if limit:
        chosen = chosen[:limit]
    rows = []
    for name in chosen:
        base, source_type = load(source, name)
        final, edited_type = load(edited, name)
        if base.shape != final.shape:
            rows.append({'name': name, 'error': f'shape changed: {base.shape} -> {final.shape}'})
            continue
        axis, space = edits[name]
        axis = np.asarray(axis, dtype=np.float32)  # (W,) or (k, W): never flatten a subspace
        intended = axis_delta(axis, base, 1.0, space)
        if renormalize:
            edited_base = base + intended
            before = np.linalg.norm(base, axis=1, keepdims=True)
            after = np.linalg.norm(edited_base, axis=1, keepdims=True)
            intended = edited_base * (before / np.maximum(after, 1e-12)) - base
        realized = final - base
        intended_norm = float(np.linalg.norm(intended))
        realized_norm = float(np.linalg.norm(realized))
        noise_norm = float(np.linalg.norm(realized - intended))
        cos = (float((realized * intended).sum() / (realized_norm * intended_norm + 1e-12))
               if realized_norm > 0 and intended_norm > 0 else 0.0)
        rows.append({'name': name, 'cos': round(cos, 4),
                     'source_type': source_type, 'edited_type': edited_type,
                     'norm_ratio': round(realized_norm / (intended_norm + 1e-12), 4),
                     'snr': round(intended_norm / (noise_norm + 1e-12), 3)})
    cosines = [row['cos'] for row in rows if 'cos' in row]
    return {'tensors': rows, 'sampled': len(rows),
            'min_cos': min(cosines) if cosines else None,
            'mean_cos': round(sum(cosines) / len(cosines), 4) if cosines else None,
            'note': 'delivered edit vs intended projection; low cos means the file, '
                    'not the axis, decided what shipped'}


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
        raise ValueError('out_path must differ from in_path; edit_tensors_inplace patches the artifact itself')
    plans = _plan(GGUFReader(str(path)), deltas)
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


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            digest.update(chunk)
    return digest.hexdigest()


def edit_tensors_inplace(path, deltas, revert_path=None):
    """Patch the artifact's own bytes: one file, no copy, no requantization.

    A requantized block keeps its byte width, so every tensor offset and the
    file size stay put and only changed tiles are written. What makes this safe
    is the sidecar: the pre-patch bytes of every tile written plus the patched
    file's digest, so `revert_inplace` restores the artifact exactly and refuses
    to touch a file that changed after the patch. revert_path=None skips the
    sidecar, and a crash mid-patch then leaves a partly edited artifact with no
    way back.
    """
    return _patch_inplace(path, deltas, revert_path=revert_path)


def project_tensors_inplace(path, projections, scale=1.0, revert_path=None):
    """The same patch, but the delta is computed from the weights being edited.

    projections maps a tensor name to ``(axis, space)``: ``axis`` is the fitted
    direction and ``space`` is ``output`` (the axis lives in the dimension W
    writes into; dW = -s u (u^T W)) or ``input`` (it lives on the module input).
    Computing the delta from the tensor the loop already dequantized is the
    difference between one dequant pass and two over the whole model: on a 1.1 GB
    Q4_K_M that was half the wall clock, and it is why the arm a search picks can
    be merged once instead of screened by building full artifacts.
    """
    return _patch_inplace(path, projections, revert_path=revert_path, scale=scale)


def _patch_inplace(path, edits, revert_path=None, scale=1.0):
    """One writer for both forms: {name: float32 delta} or {name: (axis, space)}."""
    from gguf import GGUFReader
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'{path} does not exist')
    reader = GGUFReader(str(path))
    # Every operand is validated before the first byte is written: a shape or
    # width mismatch discovered halfway leaves a half-edited model.
    plans = []
    for name, edit in edits.items():
        tensor, dtype, bs, els, n, quantize, dequantize, offset, n_bytes, shape = _tensor_info(reader, name)
        if isinstance(edit, tuple):
            axis, space = edit
            axis = np.asarray(axis, dtype=np.float32)  # (W,) or (k, W): never flatten a subspace
            want = shape[0] if space == 'output' else shape[1]
            if axis.shape[0] != want:
                raise ValueError(f'axis for {name} is {axis.shape[0]}-wide, {space} space needs {want}')
            if not np.isfinite(axis).all():
                raise ValueError(f'axis for {name} contains NaN/Inf')
            kind = ('project', axis, space)
        else:
            delta = np.asarray(edit, dtype=np.float32)
            if delta.shape != shape:
                raise ValueError(f'delta for {name} has shape {delta.shape}, tensor shape is {shape}')
            if not np.isfinite(delta).all():
                raise ValueError(f'delta for {name} contains NaN/Inf')
            kind = ('delta', delta.reshape(-1), None)
        if n != int(np.prod(shape)):
            raise ValueError(f'tensor {name}: {n} values for shape {shape}')
        plans.append((name, dtype, bs, els, n, quantize, dequantize, offset, n_bytes, shape, kind))
    size_before = path.stat().st_size
    reports, originals, proofs = [], [], []
    with open(path, 'r+b') as f:
        for name, dtype, bs, els, n, quantize, dequantize, offset, n_bytes, shape, kind in plans:
            nb = n // els
            blocks_per_tile = max(1, (1 << 24) // els)
            f.seek(offset)
            raw = f.read(n_bytes)
            if len(raw) != n_bytes:
                raise ValueError(f'{name}: short read at byte {offset} (truncated artifact?)')
            raw_u8 = np.frombuffer(raw, dtype=np.uint8)
            mode, payload, space = kind
            values = dequantize(raw_u8, n).astype(np.float32).reshape(shape)
            if mode == 'delta':
                edited = values + payload.reshape(shape)
            elif space == 'output':
                edited = values - scale * np.outer(payload, payload @ values)
            else:
                edited = values - scale * np.outer(values @ payload, payload)
            if not np.isfinite(edited).all():
                raise ValueError(f'edited tensor {name} contains NaN/Inf')
            flat = np.ascontiguousarray(edited).reshape(-1)
            new_raw = np.empty(n_bytes, dtype=np.uint8)
            for start in range(0, nb, blocks_per_tile):
                tile = min(blocks_per_tile, nb - start)
                byte0, byte1 = start * bs, (start + tile) * bs
                piece = quantize(np.ascontiguousarray(flat[start * els:(start + tile) * els]))
                if piece.size != byte1 - byte0:
                    raise ValueError(f'requantized {name}: block produced {piece.size} bytes, '
                                     f'expected {byte1 - byte0}')
                new_raw[byte0:byte1] = piece
            moved = edited - values
            sum_sq = float((values ** 2).sum())
            rms = float(math.sqrt(float((moved ** 2).sum()) / n)) if n else 0.0
            mean_sq = float(sum_sq / n) if n else 0.0
            del values, edited, moved
            changed = np.any(new_raw.reshape(nb, bs) != raw_u8.reshape(nb, bs), axis=-1)
            changed_blocks = int(changed.sum())
            first = None
            if changed_blocks:
                for start in range(0, nb, blocks_per_tile):
                    tile = min(blocks_per_tile, nb - start)
                    if not changed[start:start + tile].any():
                        continue
                    byte0, byte1 = start * bs, (start + tile) * bs
                    originals.append((offset + byte0, raw_u8[byte0:byte1].copy()))
                    f.seek(offset + byte0)
                    f.write(new_raw[byte0:byte1].tobytes())
                    if first is None:
                        first = (offset + byte0, new_raw[byte0:byte1].tobytes())
                proofs.append((name, first))
            reports.append({'tensor': name, 'dtype': dtype, 'blocks': int(nb),
                            'changed_blocks': changed_blocks, 'rms_delta': rms,
                            'relative_rms_delta': rms / mean_sq if mean_sq else 0.0})
            del new_raw, raw, raw_u8
        f.flush()
        os.fsync(f.fileno())
    if path.stat().st_size != size_before:
        raise RuntimeError(f'{path} changed size during an in-place patch ({size_before} -> '
                           f'{path.stat().st_size}); the edit is not value-preserving here')
    # A written byte is not a proven byte: read every touched tensor's first
    # tile back before calling the patch done.
    with open(path, 'rb') as f:
        for name, (at, expected) in proofs:
            f.seek(at)
            if f.read(len(expected)) != expected:
                raise RuntimeError(f'in-place patch for {name} did not land at byte {at}')
    patched_sha256 = _sha256(path)
    result = {'input': str(path), 'output': str(path), 'in_place': True,
              'tensors_edited': len(reports), 'tensors': reports,
              'tensors_changed': sum(1 for r in reports if r['changed_blocks']),
              'changed_blocks_total': int(sum(r['changed_blocks'] for r in reports)),
              'all_other_bytes_verbatim': True, 'bytes_added': 0,
              'patched_sha256': patched_sha256}
    if revert_path is not None and originals:
        offsets = np.array([at for at, _ in originals], dtype=np.int64)
        lengths = np.array([len(raw) for _, raw in originals], dtype=np.int64)
        data = np.concatenate([raw for _, raw in originals]).astype(np.uint8)
        np.savez_compressed(revert_path, offsets=offsets, lengths=lengths, data=data,
                            sha256=np.array(patched_sha256))
        result['revert_sidecar'] = str(revert_path)
        result['revert_sidecar_bytes'] = int(Path(revert_path).stat().st_size)
    return result


def revert_inplace(path, sidecar):
    """Write an in-place patch's saved pre-patch bytes back, then drop the sidecar."""
    path, sidecar = Path(path), Path(sidecar)
    for candidate in (path, sidecar):
        if not candidate.is_file():
            raise ValueError(f'{candidate} does not exist')
    payload = np.load(sidecar)
    offsets, lengths, data = payload['offsets'], payload['lengths'], payload['data']
    if 'sha256' in payload:
        recorded = str(payload['sha256'])
        current = _sha256(path)
        if current != recorded:
            raise ValueError(f'{path} changed since the patch (sha256 {current[:16]} != recorded '
                             f'{recorded[:16]}); refusing to write pre-patch bytes over a different file')
    restored, position = 0, 0
    with open(path, 'r+b') as f:
        for at, length in zip(offsets.tolist(), lengths.tolist()):
            f.seek(int(at))
            f.write(data[position:position + int(length)].tobytes())
            position += int(length)
            restored += 1
        f.flush()
        os.fsync(f.fileno())
    sidecar.unlink()
    return {'artifact': str(path), 'sidecar': str(sidecar), 'tiles_restored': restored}


def dequantize_tensor(path, name, tile_blocks=1 << 22, reader=None):
    """Dequantize one tensor to float32 in torch [out, in] order, bounded RAM.

    Despite the tensor metadata being (in, out) — GGUFReader.shape on qwen3-4b
    blk.9.attn_output is (4096, 2560) and this returns (2560, 4096) — the returned
    array is the torch layout. Anyone recovering a direction from a weight delta
    has to take its columns, not its rows; a previous version of this docstring
    said gguf dimension order and cost a debugging session.

    ``reader`` may be a prebuilt GGUFReader on the same file to skip header
    re-parsing when many tensors of one gguf are read (served_deltas).
    """
    path = Path(path)
    if reader is None:
        from gguf import GGUFReader
        reader = GGUFReader(str(path))
    tensor, dtype, bs, els, n, quantize, dequantize, offset, n_bytes, shape = _tensor_info(reader, name)
    out = np.empty(n, dtype=np.float32)
    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.MAP_SHARED)
        try:
            nb = n // els
            for start in range(0, nb, tile_blocks):
                tile = min(tile_blocks, nb - start)
                byte0, byte1 = start * bs, (start + tile) * bs
                raw = np.frombuffer(bytes(mm[offset + byte0:offset + byte1]), dtype=np.uint8)
                out[start * els:(start + tile) * els] = dequantize(raw, tile * els).reshape(-1)
        finally:
            mm.close()
    return out.reshape(shape)


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


class _LimitedBuffer:
    """Read wrapper that refuses any declared length beyond what the file
    actually holds, so a hostile length field fails fast instead of asking
    for gigabytes or looping on short reads."""

    def __init__(self, buf, file_size):
        self.buf = buf
        self.file_size = file_size

    def read(self, n):
        remaining = self.file_size - self.buf.tell()
        if n < 0 or n > remaining:
            raise ValueError(f'declared GGUF read of {n} bytes exceeds the '
                             f'{remaining} bytes remaining in the file')
        data = self.buf.read(n)
        if len(data) < n:
            raise ValueError('GGUF file ended early')
        return data

    def tell(self):
        return self.buf.tell()


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
    """Parse everything up to the tensor-data base; data is never read.

    Every declared count and length is bounded against the real file size: a
    hostile header fails fast with a precise message instead of allocating or
    looping. Legitimate large entries (multi-MB vocab arrays) pass.
    """
    import io
    path = Path(path)
    file_size = path.stat().st_size
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            raise ValueError(f'{path} is not a GGUF file')
        version = struct.unpack('<I', f.read(4))[0]
        if version < 3:
            raise ValueError(f'GGUF v{version} metadata editing supports v3+')
        n_tensors = _u64(io.BytesIO(f.read(8)))
        n_kv = _u64(io.BytesIO(f.read(8)))
        if n_kv > file_size:
            raise ValueError(f'declared {n_kv} metadata entries exceed the {file_size}-byte file')
        if n_tensors > file_size // 24:
            raise ValueError(f'declared {n_tensors} tensors exceed the {file_size}-byte file')
        buf = _LimitedBuffer(io.BufferedReader(f), file_size)
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
            if n_dims > 64:
                raise ValueError(f'tensor {name!r} declares {n_dims} dimensions (>64)')
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


# --- minimal GGUF v3 writer ---------------------------------------------------
# Tensor dtype ids match the pinned gguf.constants.GGMLQuantizationType enum:
# F32=0, F16=1, BF16=30. Dims are GGUF order (innermost first).

FLOAT_DTYPE_IDS = {'f32': 0, 'f16': 1, 'bf16': 30}


def write_gguf(path, metadata, tensors):
    """Write a GGUF v3 file: ``metadata`` is {key: python value} (serialized by
    _infer_type, so gguf-inspect/gguf-meta read it back), ``tensors`` is a list
    of (name, dims, dtype_id, raw_bytes). Offsets are computed relative to the
    aligned data base; nothing else is interpreted.
    """
    path = Path(path)
    if path.exists():
        raise ValueError(f'{path} already exists')
    entries = []
    for key, value in metadata.items():
        tag, raw = _infer_type(value)
        entries.append((key.encode(), tag, raw))
    header = b'GGUF' + struct.pack('<I', 3) + \
        struct.pack('<Q', len(tensors)) + struct.pack('<Q', len(entries))
    for key, tag, raw in entries:
        header += struct.pack('<Q', len(key)) + key + struct.pack('<I', tag) + raw
    alignment = 32
    infos_size = sum(8 + len(name.encode()) + 4 + 8 * len(dims) + 4 + 8 for name, dims, _t, _b in tensors)
    data_base = (len(header) + infos_size + alignment - 1) // alignment * alignment
    rebuilt = bytearray(header)
    offset = 0
    for name, dims, _dtype_id, raw in tensors:
        nb = name.encode()
        rebuilt += struct.pack('<Q', len(nb)) + nb + struct.pack('<I', len(dims)) + \
            b''.join(struct.pack('<Q', d) for d in dims) + struct.pack('<I', _dtype_id) + struct.pack('<Q', offset)
        offset += len(raw)
    rebuilt += b'\x00' * (data_base - len(rebuilt))
    with open(path, 'wb') as f:
        f.write(rebuilt)
        for _name, _dims, _dtype_id, raw in tensors:
            f.write(raw)
    return {'output': str(path), 'metadata_entries': len(entries), 'tensors': len(tensors),
            'data_base': data_base, 'file_size': path.stat().st_size}
