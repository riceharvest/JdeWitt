# SPDX-License-Identifier: AGPL-3.0-or-later
"""Codec-floor ablation: can a coarse quantization grid erase the denial-aligned component?

The question this module answers with tensor algebra only: for each residual-write
tensor (``blk.N.attn_output`` / ``blk.N.ffn_down``), how much of the tensor's output
energy lies in the subspace where the denial tokens' *input embedding rows* live, and
how much of that aligned energy survives a re-encode at Q4_0, Q2_K or Q2_0?

    V     = orthonormal basis of span(rows of token_embd.weight for the denial ids)
    Z     = Vᵀ W                    aligned output block, rows ordered by singular value
    e_k   = ||Z_k||² / ||W||²        aligned energy fraction at rank k
    e_rand= k / out_features        the same fraction for a random rank-k subspace

Re-encoding at a coarse type gives ``W_q``; the delivered aligned component is
``Z_q = Vᵀ W_q``, and each rank-k curve reports

    surviving_energy = ||Z_q,k||² / ||Z_k||²      (1 = untouched, 0 = gone)
    removed_energy   = ||Z_q,k - Z_k||² / ||Z_k||²
    cos              = <Z_q,k, Z_k> / (||Z_q,k|| ||Z_k||)

against the tensor's own codec error ``||W_q - W||² / ||W||²``. That comparison is the
whole experiment: a component is "rounded away" only if the codec removes it, and the
grid is only *selective* if it removes it faster than it removes a random subspace of
the same rank. A random control is computed with the same code, never assumed.

No model is loaded and nothing is generated: input embedding rows are read from the
artifact's own ``token_embd.weight``, the denial ids come from the artifact's own
vocabulary (byte-level BPE over ``tokenizer.ggml.tokens`` + ``merges``) and from the
repo's denial-surface table, and every other number is a matrix product, a Frobenius
norm or a byte count. ``removal_measured`` is false by construction.
"""
from __future__ import annotations

import numpy as np

SCHEMA = 'jdewitt-codec-floor-v1'
# The three grids this experiment re-encodes at, coarsest last. All three have an
# encoder in ``jdewitt.quants.CODECS``; whether a grid can be *stored* in a tensor of
# a given row width is a separate question, answered by ``row_block_ok``.
TYPES = ('q4_0', 'q2_k', 'q2_0')


# --------------------------------------------------------------------------- tokenizer
def byte_level_alphabet():
    """GPT-2's byte -> printable-unicode map and its inverse (the vocab's own alphabet)."""
    byte_values = (list(range(ord('!'), ord('~') + 1))
                   + list(range(ord('¡'), ord('¬') + 1))
                   + list(range(ord('®'), ord('ÿ') + 1)))
    chars = list(byte_values)
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            chars.append(256 + extra)
            extra += 1
    return {value: chr(char) for value, char in zip(byte_values, chars)}, \
           {chr(char): value for value, char in zip(byte_values, chars)}


# Qwen2's pre-tokenizer (``tokenizer.ggml.pre = "qwen2"``). The pattern decides word
# boundaries; the merges decide the ids. Both come from the artifact's own metadata.
QWEN2_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
                 r"|[^\r\n\p{L}\p{N}]?\p{L}+"
                 r"|\p{N}"
                 r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
                 r"|\s*[\r\n]+"
                 r"|\s+(?!\S)"
                 r"|\s+")


class GgufBpe:
    """Byte-level BPE built from a GGUF file's own ``tokenizer.ggml.*`` metadata.

    No HF tokenizer file, no download: the artifact already carries the vocabulary,
    the merge ranks and the pre-tokenizer name. ``encode`` is the standard GPT-2
    algorithm (pre-tokenize, then repeatedly merge the lowest-ranked adjacent pair),
    which is what the tokenizer that quantized this file did.
    """

    def __init__(self, tokens, merges, pattern=QWEN2_PATTERN):
        import regex

        self.tokens = [str(token) for token in tokens]
        self.pattern = regex.compile(pattern)
        self.ranks = {}
        for index, merge in enumerate(str(item) for item in merges):
            left, _, right = merge.partition(' ')
            self.ranks.setdefault((left, right), index)
        self.vocab = {token: index for index, token in enumerate(self.tokens)}
        self._to_escaped, self._from_escaped = byte_level_alphabet()

    @classmethod
    def from_reader(cls, reader, pattern=QWEN2_PATTERN):
        fields = reader.fields
        tokens = fields['tokenizer.ggml.tokens'].contents()
        merges = fields['tokenizer.ggml.merges'].contents()
        pre = str(fields['tokenizer.ggml.pre'].contents()) if 'tokenizer.ggml.pre' in fields else ''
        return cls(tokens, merges, pattern=pattern if pre in ('qwen2', '') else pattern)

    def escaped(self, text):
        return ''.join(self._to_escaped[byte] for byte in text.encode('utf-8'))

    def unescape(self, escaped):
        return bytes(self._from_escaped[char] for char in escaped).decode('utf-8', errors='replace')

    def _merge(self, piece):
        parts = list(piece)
        while len(parts) > 1:
            best = None
            for index in range(len(parts) - 1):
                rank = self.ranks.get((parts[index], parts[index + 1]))
                if rank is not None and (best is None or rank < best[0]):
                    best = (rank, index)
            if best is None:
                break
            index = best[1]
            parts[index:index + 2] = [parts[index] + parts[index + 1]]
        return parts

    def encode(self, text):
        ids = []
        for match in self.pattern.finditer(text):
            for piece in self._merge(self.escaped(match.group(0))):
                if piece not in self.vocab:
                    raise ValueError(f'pre-tokenized piece {piece!r} is not in the vocabulary')
                ids.append(self.vocab[piece])
        return ids

    def decode(self, ids):
        return ''.join(self.unescape(self.tokens[int(token)]) for token in ids)

    def piece(self, token_id):
        return self.unescape(self.tokens[int(token_id)])


# --------------------------------------------------------------------------- denial ids
def denial_id_sets(tokenizer, surfaces):
    """Two id inventories over the repo's denial-surface table.

    ``surface_span`` — every token id of every surface (a phrase contributes all of
        its tokens). This is the literal reading of "the input embedding rows": the
        subspace spanned by the rows of the tokens the refusals are made of.
    ``opener_span``  — one id per surface: the single token that *is* the surface when
        it resolves to one, else the surface's first token. This is the opener-only
        variant, kept because a phrase's function words ('I', 'to', 'a') do not carry
        the denial and a reader may want the stricter set.
    A surface whose first token is bare punctuation or whitespace is skipped and the
    skip is reported rather than hidden.
    """
    by_surface, surface_ids, opener_ids, skipped = [], [], [], []
    for entry in surfaces:
        surface = str(entry['form'])
        ids = tokenizer.encode(surface)
        pieces = [tokenizer.piece(token) for token in ids]
        if not ids:
            skipped.append({'id': entry.get('id'), 'surface': surface, 'reason': 'empty encoding'})
            continue
        by_surface.append({'id': entry.get('id'), 'kind': entry.get('kind'),
                           'surface': surface, 'ids': ids, 'pieces': pieces,
                           'single_token': len(ids) == 1})
        for token in ids:
            if token not in surface_ids:
                surface_ids.append(token)
        if pieces[0].strip():
            if ids[0] not in opener_ids:
                opener_ids.append(ids[0])
        else:
            skipped.append({'id': entry.get('id'), 'surface': surface,
                            'reason': f'first token {pieces[0]!r} is whitespace'})
    return {
        'surfaces': by_surface,
        'surface_span': surface_ids,
        'opener_span': opener_ids,
        'skipped_openers': skipped,
        'surface_pieces': sorted({tokenizer.piece(token).strip() for token in surface_ids}),
    }


def token_rows(reader, path, ids, tensor_name='token_embd.weight'):
    """Rows of ``token_embd.weight`` for ``ids``, read one row at a time.

    GGUF stores the embedding as ``ne = [hidden, vocab]``, so row ``t`` is
    ``raw[t*row_bytes : (t+1)*row_bytes]``; only the rows the experiment names are
    dequantized, not the 151,936-row table.
    """
    from .ggufio import decode_values
    from .quants import type_name_for_id

    tensors = {tensor.name: tensor for tensor in reader.tensors}
    tensor = tensors[tensor_name]
    hidden, vocab = (int(value) for value in tensor.shape)
    if hidden >= vocab:
        raise ValueError(f'{tensor_name}: expected [width, vocab] with width < vocab, '
                         f'got {tensor.shape}')
    qtype = type_name_for_id(int(tensor.tensor_type))
    from .ggufwrite import type_size
    block_els, block_bytes = type_size(int(tensor.tensor_type))
    if hidden % block_els:
        raise ValueError(f'{tensor_name}: row width {hidden} is not a whole number of '
                         f'{block_els}-value blocks; per-row slicing is not possible')
    row_bytes = hidden // block_els * block_bytes
    rows = np.empty((len(ids), hidden), dtype=np.float32)
    order = []
    with open(path, 'rb') as handle:
        for position, token in enumerate(ids):
            if not 0 <= int(token) < vocab:
                raise ValueError(f'token id {token} is outside the {vocab}-row embedding')
            handle.seek(int(tensor.data_offset) + int(token) * row_bytes)
            raw = np.frombuffer(handle.read(row_bytes), dtype=np.uint8)
            rows[position] = decode_values(raw, qtype, n_values=hidden).astype(np.float32)
            order.append(int(token))
    return rows, order, {'tensor': tensor_name, 'dtype': qtype, 'width': hidden,
                         'vocab': vocab, 'row_bytes': row_bytes}


# --------------------------------------------------------------------------- subspaces
def subspace(rows, rank=None):
    """Orthonormal basis of span(rows) with columns ordered by singular value.

    ``V`` is ``[width, r]``. The singular values are reported because the span is
    only meaningful if it is not one direction plus noise; the top-1 energy share is
    the honest way to say that.
    """
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or not len(rows):
        raise ValueError(f'subspace needs a non-empty [n, width] block, got {rows.shape}')
    u, singular, vt = np.linalg.svd(rows, full_matrices=False)
    keep = int(np.sum(singular > singular[0] * 1e-9)) if singular[0] > 0 else 0
    if rank is not None:
        keep = min(keep, int(rank))
    if keep < 1:
        raise ValueError('the row block has no rank')
    energy = singular ** 2
    return {
        # The ROWS live in the embedding/residual width, so the subspace they span is
        # the row space: the right singular vectors, transposed to [width, rank].
        'V': np.ascontiguousarray(vt[:keep].T),
        'rank': keep,
        'singular_values': [round(float(value), 6) for value in singular[:keep]],
        'top1_energy_share': float(energy[0] / energy.sum()),
        'top4_energy_share': float(energy[:4].sum() / energy.sum()) if keep >= 4 else 1.0,
        'fro_norm': float(np.linalg.norm(rows)),
    }


def random_subspace(width, rank, seed=0):
    """A random orthonormal ``[width, rank]`` basis: the codec's behaviour at chance."""
    generator = np.random.default_rng(seed)
    q, _r = np.linalg.qr(generator.normal(size=(int(width), int(rank))))
    return np.ascontiguousarray(q[:, :int(rank)])


def alignment_curve(weight, basis, quantized, ks=None):
    """Aligned energy and delivered cos of ``VᵀW`` before and after re-encoding.

    ``weight`` is ``[out, in]`` (GGUF order reversed, the same orientation the codec
    encodes), ``basis`` is ``[out, r]``. Both ``Z = VᵀW`` and ``Z_q = VᵀW_q`` are
    computed once; a rank-k curve is a prefix sum over their rows because the basis
    columns are ordered by singular value.
    """
    weight = np.asarray(weight, dtype=np.float32)
    quantized = np.asarray(quantized, dtype=np.float32)
    if weight.shape != quantized.shape:
        raise ValueError(f'weight {weight.shape} vs re-encoded {quantized.shape}')
    basis = np.asarray(basis, dtype=np.float32)
    rank = basis.shape[1]
    ks = sorted({int(k) for k in (ks or (1, 2, 4, 8, 16, rank)) if 0 < int(k) <= rank})
    z = basis.T.astype(np.float64) @ weight.astype(np.float64)
    zq = basis.T.astype(np.float64) @ quantized.astype(np.float64)
    total = float(np.sum(weight.astype(np.float64) ** 2))
    row_energy = np.sum(z ** 2, axis=1)
    row_energy_q = np.sum(zq ** 2, axis=1)
    row_diff = np.sum((zq - z) ** 2, axis=1)
    row_dot = np.sum(zq * z, axis=1)
    curves = {}
    for k in ks:
        aligned = float(row_energy[:k].sum())
        kept = float(row_energy_q[:k].sum())
        gone = float(row_diff[:k].sum())
        dot = float(row_dot[:k].sum())
        denominator = float(np.sqrt(kept * aligned))
        curves[str(k)] = {
            'aligned_energy': aligned,
            'aligned_energy_fraction': aligned / total if total else 0.0,
            'isotropic_fraction': k / weight.shape[0],
            'surviving_energy_fraction': kept / aligned if aligned else 0.0,
            'removed_energy_fraction': gone / aligned if aligned else 0.0,
            'cos': dot / denominator if denominator else 0.0,
        }
    return {'weight_energy': total, 'out_features': int(weight.shape[0]),
            'in_features': int(weight.shape[1]), 'rank': rank, 'curves': curves}


def quantization_error(weight, quantized):
    """Codec error of a whole tensor: energy fraction, RMS ratio and SNR in dB."""
    weight = np.asarray(weight, dtype=np.float64)
    quantized = np.asarray(quantized, dtype=np.float64)
    total = float(np.sum(weight ** 2))
    noise = float(np.sum((quantized - weight) ** 2))
    fraction = noise / total if total else 0.0
    snr = 10.0 * np.log10(total / noise) if noise > 0 else float('inf')
    return {'noise_energy': noise, 'relative_noise_energy': fraction,
            'rms_ratio': float(np.sqrt(fraction)), 'snr_db': float(snr)}


def _codec(type_name):
    """Codec tuple by name, case-insensitively ('q4_0' -> CODECS['Q4_0'])."""
    from .quants import CODECS

    key = str(type_name).strip().upper()
    if key not in CODECS:
        raise ValueError(f'no encoder for {type_name!r}; this build writes '
                         f'{", ".join(sorted(CODECS))}')
    return CODECS[key]


def row_block_ok(row_width, type_name):
    """Can a codec be stored in a tensor whose GGML row width is ``row_width``?

    GGML quantizes along ``ne[0]`` and a row must be a whole number of blocks
    (``ggml_row_size`` truncates otherwise, so the loader's byte count no longer
    matches and llama.cpp refuses the file). This is why Q2_K — 256-value
    superblocks — cannot hold a 896-wide ``attn_output`` row while it can hold a
    4864-wide ``ffn_down`` row.
    """
    _block_bytes, block_els, _quantize, _dequantize = _codec(type_name)
    return int(row_width) % int(block_els) == 0, int(block_els)


def roundtrip(weight, type_name):
    """Encode ``weight`` at ``type_name`` and decode it back; returns (values, bytes)."""
    _block_bytes, block_els, quantize, dequantize = _codec(type_name)
    flat = np.ascontiguousarray(np.asarray(weight, dtype=np.float32).reshape(-1))
    if flat.size % block_els:
        raise ValueError(f'{type_name} needs a multiple of {block_els} values, got {flat.size}')
    packed = np.asarray(quantize(flat), dtype=np.uint8).reshape(-1)
    values = np.asarray(dequantize(packed, flat.size), dtype=np.float32).reshape(np.shape(weight))
    return values, packed


# --------------------------------------------------------------------------- aggregation
def aggregate(rows, rank_key='full'):
    """Energy-weighted and median aggregate of per-tensor curves.

    An unweighted mean over tensors would let a 0.8M-parameter o_proj outvote a
    4.36M-parameter down_proj; the weights are each tensor's Frobenius energy, the
    same quantity every fraction is a share of. Medians are reported beside it so a
    single outlier cannot hide behind the weighting.
    """
    if not rows:
        return {}
    weights = np.array([row['weight_energy'] for row in rows], dtype=np.float64)
    weights = weights / weights.sum()
    result = {'tensors': len(rows), 'total_weight_energy': float(sum(weights * 0 +
              np.array([row['weight_energy'] for row in rows])))}
    keys = sorted({k for row in rows for k in row['curves']}, key=int)
    result['curves'] = {}
    for key in keys:
        present = [(row, weight) for row, weight in zip(rows, weights) if key in row['curves']]
        if not present:
            continue
        stats = {}
        for field in ('aligned_energy_fraction', 'surviving_energy_fraction',
                      'removed_energy_fraction', 'cos', 'isotropic_fraction'):
            values = np.array([row['curves'][key][field] for row, _w in present])
            weight_vector = np.array([weight for _row, weight in present])
            stats[field] = float(np.sum(values * weight_vector) / weight_vector.sum())
            stats[f'{field}_median'] = float(np.median(values))
            stats[f'{field}_min'] = float(values.min())
            stats[f'{field}_max'] = float(values.max())
        result['curves'][key] = stats
    for field in ('relative_noise_energy', 'rms_ratio', 'snr_db'):
        values = np.array([row['codec'][field] for row in rows])
        result[field] = float(np.sum(values * weights))
        result[f'{field}_median'] = float(np.median(values))
        result[f'{field}_min'] = float(values.min())
        result[f'{field}_max'] = float(values.max())
    selected = [row['selectivity'] for row in rows if row.get('selectivity') is not None]
    result['selectivity_median'] = float(np.median(selected)) if selected else None
    return result


def conclude(aggregate_rows, types=TYPES):
    """The card's question as arithmetic over the aggregate: is the grid above the edit?

    For each type: the codec's whole-tensor noise fraction (the "codec floor") against
    the aligned energy fraction (the "edit size"), the delivered cos of the aligned
    component, and the selectivity ratio against a random subspace of the same rank.
    ``floor_over_edit`` above 1 means the grid's error is larger than the component it
    is being asked to delete — the component is inside the noise, which cannot be a
    surgical removal.
    """
    verdicts = {}
    for name, row in aggregate_rows.items():
        if not row or 'curves' not in row:
            continue
        full = row['curves'].get(str(row.get('rank') or max(int(k) for k in row['curves'])))
        if full is None:
            full = row['curves'][max(row['curves'], key=int)]
        floor = row['relative_noise_energy_median']
        edit = full['aligned_energy_fraction_median']
        delivered_random = row.get('random_curves', {}).get(
            max(row['random_curves'], key=int)) if row.get('random_curves') else None
        verdicts[name] = {
            'codec_floor_median': floor,
            'edit_size_median': edit,
            'floor_over_edit': floor / edit if edit else None,
            'aligned_cos_median': full['cos_median'],
            'aligned_removed_median': full['removed_energy_fraction_median'],
            'random_removed_median': (delivered_random or {}).get('removed_energy_fraction_median'),
            'selectivity_median': row.get('selectivity_median'),
            'aligned_survives': full['cos_median'] >= 0.99,
        }
    return verdicts
