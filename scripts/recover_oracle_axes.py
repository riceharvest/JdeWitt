# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recover a published edit's direction and strength into a spanfit axes tensor.

Usage: python scripts/recover_oracle_axes.py <out.pt> <base.gguf> <oracle.gguf>

Writes {axes_attn, axes_ffn} [L, D] so the recovered direction can go through
`spanfit build --axes` exactly like a fitted one, and reports the per-tensor
strength it implies (s = ||delta||_F / ||u^T W||) plus the per-layer cosine
between the attention and MLP directions, which decides whether one merge can
carry both modules or they must be edited apart.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jdewitt.ggufio import dequantize_tensor
from gguf import GGUFReader

if len(sys.argv) < 4:
    raise SystemExit('usage: recover_oracle_axes.py <out.pt> <base.gguf> <oracle.gguf>')
OUT, BASE, ORACLE = sys.argv[1:4]

# Recover the direction the published oracle actually removed, in the same tensor
# a spanfit axis lives in, so it can go through the same merge path as a fitted
# one (spanfit build --axes). The oracle's delta is rank-1: W_o - W_b = -s u v^T,
# so the largest-norm row of the delta is s*||v||*u and normalising it gives u
# directly — no SVD, and the sign does not matter because the edit is u(u^T W).
# Strength is then s = ||delta||_F / ||u^T W_base||, per tensor, which is what
# --scale has to reproduce. One tensor in memory at a time: the earlier version
# that collected all 54 deltas was SIGKILLed at 137 partway through.
L_MAX = 36
reader_b, reader_o = GGUFReader(BASE), GGUFReader(ORACLE)
axes = {m: torch.zeros(L_MAX, 2560, dtype=torch.float32) for m in ('attn', 'ffn')}
strengths, missing = {}, []
t0 = time.monotonic()
for module, key in (('attn_output', 'attn'), ('ffn_down', 'ffn')):
    rows = []
    for L in range(L_MAX):
        name = f'blk.{L}.{module}.weight'
        # gguf order is [in, out]; the removed residual direction is the torch
        # COLUMN factor of dW = -s u v^T, so work in torch order [out, in] and take
        # the largest-norm column. Rows are multiples of v, the input-side factor -
        # mixing those two up is the same axis-space mistake this repo already has
        # one commit about.
        # dequantize_tensor already returns torch [out, in] (measured:
        # blk.9.attn_output GGUFReader.shape (4096, 2560), dequantize (2560, 4096)).
        b = dequantize_tensor(BASE, name, reader=reader_b).astype(np.float32)
        o = dequantize_tensor(ORACLE, name, reader=reader_o).astype(np.float32)
        delta = o - b
        del o
        fro = float(np.linalg.norm(delta))
        if fro < 1e-6:
            del b
            continue
        norms = np.linalg.norm(delta, axis=0)
        j = int(np.argmax(norms))
        u = delta[:, j] / norms[j]
        assert u.shape[0] == 2560, (name, u.shape)
        axes[key][L] = torch.from_numpy(u.astype(np.float32))
        v = u @ b            # u^T W_base, [in_features]
        s = fro / float(np.linalg.norm(v))
        rows.append({'layer': L, 'scale': round(s, 4), 'fro': round(fro, 3),
                     'cos_self_row': round(float(norms[j] / fro), 4)})
        del b, delta
    strengths[key] = {'tensors': len(rows),
                      'scale_mean': round(float(np.mean([r['scale'] for r in rows])), 4),
                      'scale_median': round(float(np.median([r['scale'] for r in rows])), 4),
                      'scale_min': round(float(np.min([r['scale'] for r in rows])), 4),
                      'scale_max': round(float(np.max([r['scale'] for r in rows])), 4),
                      'fro_over_row_max_mean': round(float(np.mean([r['cos_self_row'] for r in rows])), 4),
                      'per_layer': rows[:6]}
# Do the two tensors in a layer drop the same direction? The q25 oracle had
# attn/ffn directions near +/-1 per layer; on qwen3-4b that is a measurement, not an
# assumption, and it decides whether one merge can carry both or whether the
# modules have to be edited separately.
per_layer_module_cos = []
for L in range(L_MAX):
    a_, b_ = axes['attn'][L], axes['ffn'][L]
    if float(a_.norm()) > 0 and float(b_.norm()) > 0:
        per_layer_module_cos.append(round(float(torch.nn.functional.cosine_similarity(a_, b_, dim=0)), 4))
print(json.dumps({'elapsed_s': round(time.monotonic() - t0, 1),
                  'edited_layers': {k: int((v.norm(dim=1) > 0).sum()) for k, v in axes.items()},
                  'attn_vs_ffn_per_layer_cos': {'n': len(per_layer_module_cos),
                      'mean_abs': round(float(np.mean(np.abs(per_layer_module_cos))), 4),
                      'mean_signed': round(float(np.mean(per_layer_module_cos)), 4),
                      'first': per_layer_module_cos[:6]},
                  'strength': strengths}, indent=1), flush=True)
torch.save({'axes_attn': axes['attn'], 'axes_ffn': axes['ffn']}, OUT)
print('saved', OUT, flush=True)
print('DONE', flush=True)
print('saved', OUT, flush=True)
print('DONE', flush=True)
