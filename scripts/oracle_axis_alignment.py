# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure a fitted removal axis against the direction a published edit removed.

Usage: python scripts/oracle_axis_alignment.py <records.pt> <shared|perlayer> <base.gguf> <oracle.gguf>

A published rank-1 weight delta is W - s u v^T, so in torch order [out, in] the
largest-norm COLUMN is the unit direction removed from the residual stream (both
attn_output and ffn_down write into it), recoverable from the delta alone with no
SVD. Cosines of a fitted axis against that direction are per layer and per module.
On Qwen3-4B-Instruct-2507 vs the published Heretic build (blk.9-35):
shared/last-prompt-token mean|cos| 0.508 (spread 0.500-0.511), per-layer
last-prompt-token 0.258, per-layer answer-end 0.048 - the locus that certified
36/40 on MiniCPM-V is statistically orthogonal to what the oracle removes here (0.048).
The full 2x2: shared/last-token 0.508, per-layer/last-token 0.258, shared/answer-end
0.059, per-layer/answer-end 0.048 — the capture locus dominates the shape.
"""
import json
import sys
import time
import numpy as np
sys.path.insert(0, '/mnt/ssd/dev workspace/JdeWitt')
from jdewitt.ggufio import dequantize_tensor
from jdewitt.spanfit import load_records, shared_axis, per_layer_axes
from gguf import GGUFReader

if len(sys.argv) < 5:
    raise SystemExit('usage: oracle_axis_alignment.py <records.pt> <shared|perlayer> <base.gguf> <oracle.gguf>')
RECORDS, SHAPE, BASE, ORACLE = sys.argv[1:5]
if SHAPE not in ('shared', 'perlayer'):
    raise SystemExit('shape must be shared or perlayer')

# The published oracle edits blk.9..blk.35 on attn_output and ffn_down and nothing
# else (byte-hash of every decoder tensor; layers 0-8 are identical). The delta it
# writes is rank-1 by construction: W - s u (v^T), so every row of the delta is a
# scalar times the same unit vector u, and u is exactly the direction removed from
# the residual stream (both tensors have out_dim = hidden = 2560, so this is the
# same space a shared contrast axis lives in). That gives a cheap, direct
# measurement of "is my fitted axis the oracle's axis": dequantize the two files,
# take the largest-magnitude delta row, normalise, cos against my u. No SVD, no
# reconstruction, and nothing that depends on my merge being any good.
recs = load_records(RECORDS)
axes = shared_axis(recs)[0] if SHAPE == 'shared' else per_layer_axes(recs)
shared = axes.ndim == 1
print(f'axis shape {tuple(axes.shape)} shape={SHAPE}', flush=True)

reader_b, reader_o = GGUFReader(BASE), GGUFReader(ORACLE)
names = sorted({int(t.name.split('.')[1]) for t in reader_b.tensors if t.name.startswith('blk.')})
t0 = time.monotonic()
report = {}
for module, hf in (('attn_output', 'o_proj'), ('ffn_down', 'ffn_down')):
    cosines = []
    for L in names:
        name = f'blk.{L}.{module}.weight'
        w = dequantize_tensor(BASE, name, reader=reader_b).astype(np.float32)
        v = dequantize_tensor(ORACLE, name, reader=reader_o).astype(np.float32)
        delta = (v - w)
        if not np.abs(delta).any():
            continue
        # dequantize_tensor returns torch [out, in]; dW = -s u v^T, so the
        # removed residual direction is a COLUMN of that matrix (rows are v).
        norms = np.linalg.norm(delta, axis=0)
        j = int(np.argmax(norms))
        if norms[j] < 1e-6:
            continue
        u_oracle = delta[:, j] / norms[j]
        u_mine = (axes if shared else axes[L]).numpy().astype(np.float32)
        u_mine = u_mine / np.linalg.norm(u_mine)
        cosines.append({'layer': L, 'cos': round(float(np.dot(u_oracle, u_mine)), 4),
                        'delta_row_norm': round(float(norms[j]), 3)})
    vals = [c['cos'] for c in cosines]
    report[module] = {
        'tensors_with_nonzero_delta': len(vals),
        'mean_abs_cos': round(float(np.mean(np.abs(vals))), 4) if vals else None,
        'median_abs_cos': round(float(np.median(np.abs(vals))), 4) if vals else None,
        'max_abs_cos': round(float(np.max(np.abs(vals))), 4) if vals else None,
        'mean_cos_signed': round(float(np.mean(vals)), 4) if vals else None,
        'per_layer': cosines}
print(json.dumps({'records': RECORDS, 'shape': SHAPE,
                  'elapsed_s': round(time.monotonic() - t0, 1),
                  'vs_oracle': report}, indent=1)[:4000], flush=True)
np.save(f'/tmp/q3align_{SHAPE}.npy', np.array([[c['layer'], c['cos']] for m in report.values() for c in m['per_layer']]))
print('DONE', flush=True)
