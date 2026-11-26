# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read the refusal axis out of a published abliteration's weight diff.

Usage: python scripts/twin_abduction.py --base BASE.safetensors --twin TWIN.safetensors \
          --out verification/twin-abduction/qwen05b.json --axis-out verification/twin-abduction/axis.npy \
          --npy-dir DIR [--delta-gguf FILE]

A rank-1 abliteration leaves dW = -a r (W^T r)^T on every writeback tensor
(self_attn.o_proj and mlp.down_proj), i.e. exactly rank-1 with the left singular
vector the refusal direction r. So the twin's published edit IS the fitted axis:
this script reads it out with an SVD instead of fitting anything to activations.

Static checks only: weight algebra over the two safetensors files. No model is
loaded, no forward pass, no generation, no GPU. Nothing here measures removal,
and every report says so ("removal_measured": false).

The published twin is bf16-rounded, so dW carries a bf16 rounding floor even on
tensors the edit never touched. Tensors are classified by relative delta norm,
and untouched ones are reported as rounding, not as edits.
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root, for jdewitt.*

WRITEBACK = (('self_attn.o_proj', 'attn_output'), ('mlp.down_proj', 'ffn_down'))


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def st_index(path):
    """safetensors header -> {name: (dtype, shape, begin, end)} and data base."""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
    base = 8 + n
    out = {}
    for name, info in header.items():
        if name == '__metadata__':
            continue
        b, e = info['data_offsets']
        out[name] = (info['dtype'], tuple(info['shape']), base + b, base + e)
    return out, base, header.get('__metadata__', {})


def read_f32(fh, index, name):
    dtype, shape, b, e = index[name]
    fh.seek(b)
    raw = fh.read(e - b)
    if dtype == 'BF16':
        u = np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16
        a = u.view(np.float32)
    elif dtype == 'F32':
        a = np.frombuffer(raw, dtype='<f4')
    elif dtype == 'F16':
        a = np.frombuffer(raw, dtype='<f2').astype(np.float32)
    else:
        raise ValueError(f'{name}: unsupported dtype {dtype}')
    return a.reshape(shape)


def raw_bytes(fh, index, name):
    _d, _s, b, e = index[name]
    fh.seek(b)
    return fh.read(e - b)


def canonical_sign(v):
    """Deterministic sign: make the largest-|component| entry positive."""
    v = np.asarray(v, dtype=np.float64)
    i = int(np.argmax(np.abs(v)))
    return -v if v[i] < 0 else v


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n else v


def cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if not na or not nb:
        return 0.0
    return float(a @ b / (na * nb))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--twin', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--axis-out', type=Path, required=True)
    ap.add_argument('--npy-dir', type=Path, required=True,
                    help='where the per-tensor dW float32 npys are written for gguf-delta-pack')
    ap.add_argument('--npy-prefix', default='blk',
                    help='GGUF tensor prefix used in the npy file names (default blk)')
    ap.add_argument('--base-repo', default='Qwen/Qwen2.5-0.5B-Instruct')
    ap.add_argument('--twin-repo', default='Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1')
    ap.add_argument('--base-revision', default=None)
    ap.add_argument('--twin-revision', default=None)
    ap.add_argument('--no-footprint', action='store_true',
                    help='skip the all-tensor hash footprint pass (slower, reads both files fully)')
    ap.add_argument('--delta-gguf', type=Path, default=None,
                    help='also pack the per-tensor dW npys into this GGUF delta file (gguf-delta-pack) '
                         'and verify the packed tensors read back byte-identical to the npys')
    args = ap.parse_args(argv)

    t0 = time.monotonic()
    bidx, _bb, bmeta = st_index(args.base)
    tidx, _tb, tmeta = st_index(args.twin)
    ids = {'base': {'repo': args.base_repo, 'revision': args.base_revision,
                    'file': args.base.name, 'sha256': sha256(args.base),
                    'size_bytes': args.base.stat().st_size,
                    'safetensors_metadata': bmeta},
           'twin': {'repo': args.twin_repo, 'revision': args.twin_revision,
                    'file': args.twin.name, 'sha256': sha256(args.twin),
                    'size_bytes': args.twin.stat().st_size,
                    'safetensors_metadata': tmeta}}
    print(json.dumps({'stage': 'hashed', 'ids': ids}, indent=1), flush=True)

    layers = sorted({int(k.split('.')[2]) for k in bidx if k.startswith('model.layers.')})
    args.npy_dir.mkdir(parents=True, exist_ok=True)

    with open(args.base, 'rb') as bf, open(args.twin, 'rb') as tf:
        # ---- edit footprint: which tensors differ at all (byte level) ----
        footprint = None
        if not args.no_footprint:
            names = sorted(set(bidx) | set(tidx))
            changed, only_base, only_twin = [], sorted(set(bidx) - set(tidx)), sorted(set(tidx) - set(bidx))
            for name in names:
                if name in only_base or name in only_twin:
                    continue
                hb = hashlib.sha256(raw_bytes(bf, bidx, name)).hexdigest()
                ht = hashlib.sha256(raw_bytes(tf, tidx, name)).hexdigest()
                if hb == ht:
                    continue
                w = read_f32(bf, bidx, name)
                v = read_f32(tf, tidx, name)
                d = v - w
                dn, wn = float(np.linalg.norm(d)), float(np.linalg.norm(w))
                changed.append({'name': name, 'shape': list(w.shape), 'dtype': bidx[name][0],
                                'delta_norm': dn, 'weight_norm': wn,
                                'rel_delta': dn / wn if wn else None,
                                'max_abs_delta': float(np.abs(d).max()),
                                'n_changed_elems': int((d != 0).sum())})
            footprint = {'tensors_in_base': len(bidx), 'tensors_in_twin': len(tidx),
                         'only_in_base': only_base, 'only_in_twin': only_twin,
                         'tensors_changed': len(changed),
                         'tensors_identical': len(names) - len(changed) - len(only_base) - len(only_twin),
                         'changed': sorted(changed, key=lambda r: -(r['rel_delta'] or 0))}
            print(json.dumps({'stage': 'footprint', 'changed': len(changed),
                              'rel_top': [(r['name'], round(r['rel_delta'], 6)) for r in footprint['changed'][:6]]}),
                  flush=True)

        # ---- per writeback tensor: SVD of dW ----
        rows = {}
        lefts = {}   # (layer, module) -> signed u1
        dw_paths = {}
        for L in layers:
            for hf_mod, g_mod in WRITEBACK:
                name = f'model.layers.{L}.{hf_mod}.weight'
                if name not in bidx or name not in tidx:
                    continue
                W = read_f32(bf, bidx, name).astype(np.float32)
                V = read_f32(tf, tidx, name).astype(np.float32)
                dW = V - W
                U, s, Vt = np.linalg.svd(dW, full_matrices=False)
                e = s ** 2
                u1 = canonical_sign(U[:, 0])
                v1 = Vt[0].copy()
                if float(u1 @ U[:, 0]) < 0:      # keep u,v paired after the sign fix
                    v1 = -v1
                lefts[(L, g_mod)] = u1
                dn, wn = float(np.linalg.norm(dW)), float(np.linalg.norm(W))
                rows.setdefault(L, {})[g_mod] = {
                    'tensor': name, 'shape': list(dW.shape),
                    'dtype': bidx[name][0],
                    'delta_fro': dn, 'weight_fro': wn,
                    'rel_delta': dn / wn if wn else None,
                    'max_abs_delta': float(np.abs(dW).max()),
                    'n_changed_elems': int((dW != 0).sum()),
                    'n_elems': int(dW.size),
                    'singular_values_top8': [float(x) for x in s[:8]],
                    'rank1_share_sv': float(s[0] / s.sum()) if s.sum() else None,
                    'rank1_energy_share': float(e[0] / e.sum()) if e.sum() else None,
                    'sv2_over_sv1': float(s[1] / s[0]) if len(s) > 1 and s[0] else None,
                    'u1': [float(x) for x in u1],
                    'v1': [float(x) for x in v1],
                    'u1_head': [float(x) for x in u1[:8]],
                }
                np.save(args.npy_dir / f'{args.npy_prefix}.{L}.{g_mod}.weight.npy', dW)
                dw_paths[f'{args.npy_prefix}.{L}.{g_mod}.weight'] = str(args.npy_dir / f'{args.npy_prefix}.{L}.{g_mod}.weight.npy')
            print(json.dumps({'stage': 'svd', 'layer': L}), flush=True)

        # ---- shared axis: top singular vector of the stacked left vectors ----
        keys = sorted(lefts)
        M = np.stack([lefts[k] for k in keys])                  # (2*n_layers, hidden)
        Ua, sa, Vta = np.linalg.svd(M, full_matrices=False)
        axis = canonical_sign(Vta[0])
        axis = axis / np.linalg.norm(axis)
        axis_energy = float(sa[0] ** 2 / (sa ** 2).sum())

        # module-level axes too, to see whether both writebacks share one direction
        module_axis = {}
        for g_mod in ('attn_output', 'ffn_down'):
            Mk = np.stack([lefts[(L, g_mod)] for L in layers if (L, g_mod) in lefts])
            U2, s2, V2 = np.linalg.svd(Mk, full_matrices=False)
            a = canonical_sign(V2[0]); a = a / np.linalg.norm(a)
            module_axis[g_mod] = {'axis': a, 'energy_share': float(s2[0] ** 2 / (s2 ** 2).sum()),
                                  'cos_to_shared': cos(a, axis)}

        # per-layer cos to the shared axis + layer-to-layer table
        per_layer_cos = {f'{L}:{g_mod}': cos(lefts[(L, g_mod)], axis) for L in layers for g_mod in ('attn_output', 'ffn_down') if (L, g_mod) in lefts}
        cos_table = {}
        for i, ka in enumerate(keys):
            cos_table[f'{ka[0]}:{ka[1]}'] = {f'{kb[0]}:{kb[1]}': round(cos(lefts[ka], lefts[kb]), 6) for kb in keys}

        # ---- right vector against W^T r, and the strength implied by the edit ----
        detail = {}
        for L in layers:
            detail[L] = {}
            for g_mod in ('attn_output', 'ffn_down'):
                if (L, g_mod) not in lefts:
                    continue
                r = rows[L][g_mod]
                W = read_f32(bf, bidx, r['tensor']).astype(np.float64)
                dW = (read_f32(tf, tidx, r['tensor']) - read_f32(bf, bidx, r['tensor'])).astype(np.float64)
                wtr = W.T @ axis                              # (in,)
                wtr_u = unit(wtr)
                # dW = -a r (W^T r)^T  =>  v1 == +-(W^T r)/||W^T r||
                # least-squares strength against the shared axis: a = <dW, r v^T> / ||r v^T||^2, with a sign
                outer = np.outer(axis, wtr_u)
                a_ls = -float((dW * outer).sum() / (outer ** 2).sum())
                recon = dW + a_ls * outer
                res = float(np.linalg.norm(recon) / np.linalg.norm(dW)) if np.linalg.norm(dW) else None
                # same fit with the per-layer left vector (its own rank-1 model)
                u1 = lefts[(L, g_mod)]
                v1 = np.array(r['v1'])
                a_own = -float((dW * np.outer(u1, v1)).sum())
                detail[L][g_mod] = {
                    'cos_u1_to_axis': cos(u1, axis),
                    'cos_v1_to_WT_axis': cos(v1, wtr),
                    'cos_u1_to_neg_axis': cos(u1, -axis),
                    'sv1': float(r['singular_values_top8'][0]),
                    'strength_ls_shared_axis': a_ls,
                    'strength_implied_by_sv1': float(r['singular_values_top8'][0] / np.linalg.norm(wtr)) if np.linalg.norm(wtr) else None,
                    'strength_abl_form': float(r['singular_values_top8'][0] / np.linalg.norm(wtr)) if np.linalg.norm(wtr) else None,
                    'norm_WT_axis': float(np.linalg.norm(wtr)),
                    'residual_frac_shared_axis_rank1': res,
                    'rank1_share_sv': r['rank1_share_sv'],
                    'rank1_energy_share': r['rank1_energy_share'],
                    'rel_delta': r['rel_delta'],
                    'n_changed_elems': r['n_changed_elems'],
                    'n_elems': r['n_elems'],
                }

        # ---- extra edit sites: embedding + the twin's untied lm_head ----
        # dW here is (vocab, hidden); an abliteration leaves dW = -(W r) r^T, so the
        # RIGHT singular vector is the axis. vocab x hidden SVD is too big, and we do
        # not need it: power iteration gives v1, s1, and ||dW||_F^2 gives the rest of
        # the spectrum, so the top-1 energy share is exact from those two numbers.
        def right_rank1(fh_b, fh_t, idx_b, idx_t, name_t, ref_name_b):
            base_ref = read_f32(fh_b, idx_b, ref_name_b)
            d = (read_f32(fh_t, idx_t, name_t) - base_ref).astype(np.float32)
            fro2 = float((d.astype(np.float64) ** 2).sum())
            if fro2 == 0.0:
                return {'tensor': name_t, 'shape': list(d.shape), 'compared_against': ref_name_b,
                        'delta_fro': 0.0, 'rel_delta': 0.0,
                        'rank1_energy_share': None, 'residual_frac_rank1': None,
                        'right_vector_cos_to_axis': None, 'left_cos_to_Wref_axis': None,
                        'strength_ls_rank1': None, 'sv1': 0.0, 'strength_abl_form': None,
                        'residual_frac_abl_form': None,
                        'max_abs_delta': 0.0, 'n_changed_elems': 0, 'n_elems': int(d.size),
                        'v1': None,
                        'note': (f'delta is exactly zero: this tensor in the twin is a byte-identical copy of '
                                 f'{ref_name_b}; it carries none of the edit')}
            v = axis.astype(np.float32).copy()      # seed with the recovered axis
            v /= np.linalg.norm(v)
            for _ in range(200):
                w = d.T @ (d @ v)
                nw = float(np.linalg.norm(w))
                if not nw:
                    break
                v_new = w / nw
                if float(np.linalg.norm(v_new - v)) < 1e-9:
                    v = v_new
                    break
                v = v_new
            dv = d @ v
            s1 = float(np.linalg.norm(dv))
            e1 = s1 ** 2 / fro2 if fro2 else 0.0
            u = dv / s1 if s1 else dv
            w_ref = base_ref.astype(np.float64) @ axis     # the edit's left vector should be W_ref r
            a = -float((d.astype(np.float64) * np.outer(u, v)).sum())
            a_abl = -float((d.astype(np.float64) * np.outer(w_ref, axis)).sum() / float(w_ref @ w_ref))
            res_abl = float(np.linalg.norm(d.astype(np.float64) + a_abl * np.outer(w_ref, axis)) / np.sqrt(fro2))
            return {
                'tensor': name_t, 'shape': list(d.shape),
                'compared_against': ref_name_b,
                'delta_fro': float(np.sqrt(fro2)),
                'rel_delta': float(np.sqrt(fro2) / np.linalg.norm(base_ref)),
                'sv1': s1, 'rank1_energy_share': e1,
                'residual_frac_rank1': float(np.sqrt(max(0.0, 1.0 - e1))),
                'right_vector_cos_to_axis': cos(v, axis),
                'left_cos_to_Wref_axis': cos(u, w_ref),
                'strength_ls_rank1': a,
                'strength_abl_form': a_abl,
                'residual_frac_abl_form': res_abl,
                'max_abs_delta': float(np.abs(d).max()),
                'n_changed_elems': int((d != 0).sum()), 'n_elems': int(d.size),
                'v1': [float(x) for x in v],
            }

        extra = {}
        if 'model.embed_tokens.weight' in tidx and 'model.embed_tokens.weight' in bidx:
            extra['model.embed_tokens.weight'] = right_rank1(
                bf, tf, bidx, tidx, 'model.embed_tokens.weight', 'model.embed_tokens.weight')
        if 'lm_head.weight' in tidx:
            extra['lm_head.weight'] = right_rank1(
                bf, tf, bidx, tidx, 'lm_head.weight', 'model.embed_tokens.weight')
            extra['lm_head.weight']['note'] = ('lm_head exists only in the twin (the base file is tied and has none), '
                                               'so it is compared against the base embedding it was untied from')
            print(json.dumps({'stage': 'extra', 'sites': {k: {'rank1_energy_share': None if v['rank1_energy_share'] is None else round(v['rank1_energy_share'], 6),
                                                              'cos_right_to_axis': None if v['right_vector_cos_to_axis'] is None else round(v['right_vector_cos_to_axis'], 7),
                                                              'strength': None if v['strength_ls_rank1'] is None else round(v['strength_ls_rank1'], 4)}
                                                          for k, v in extra.items()}}), flush=True)

        # ---- non-weight differences shipped beside the weights (tokenizer/config only) ----
        # A behavioural comparison against this twin would confound the weight edit with
        # the repo's own template/tokenizer changes, so the record names them.
        repo_diff = None
        try:
            bdir, tdir = Path(args.base).parent, Path(args.twin).parent
            if (bdir / 'config.json').exists() and (tdir / 'config.json').exists():
                bc = json.loads((bdir / 'config.json').read_text())
                tc = json.loads((tdir / 'config.json').read_text())
                cfg_diff = {k: {'base': bc.get(k), 'twin': tc.get(k)}
                            for k in sorted(set(bc) | set(tc)) if bc.get(k) != tc.get(k)}
                tok = None
                bt, tt = bdir / 'tokenizer_config.json', tdir / 'tokenizer_config.json'
                if bt.exists() and tt.exists():
                    bj = json.loads(bt.read_text())
                    tj = json.loads(tt.read_text())
                    btpl = bj.get('chat_template') or ''
                    ttpl = tj.get('chat_template') or ''
                    tok = {
                        'chat_template_sha256': {'base': hashlib.sha256(btpl.encode()).hexdigest(),
                                                 'twin': hashlib.sha256(ttpl.encode()).hexdigest()},
                        'chat_template_len': {'base': len(btpl), 'twin': len(ttpl)},
                        'chat_template_identical': btpl == ttpl,
                        'pad_token': {'base': bj.get('pad_token'), 'twin': tj.get('pad_token')},
                    }
                repo_diff = {'config_json_differs': cfg_diff, 'tokenizer_config': tok,
                             'note': ('weights are the only thing this experiment reads; the twin repo also ships its own '
                                      'chat template and pad token, so any behavioural comparison against it would carry '
                                      'that confound as well as the weight edit')}
                print(json.dumps({'stage': 'repo-diff', 'config': {k: v for k, v in cfg_diff.items()},
                                  'chat_template_identical': (tok or {}).get('chat_template_identical')}), flush=True)
        except Exception as exc:                      # never fail the weight pass on a config file
            repo_diff = {'error': repr(exc)}

        # ---- checks ----
        touched = [k for k in keys if (detail[k[0]][k[1]]['rel_delta'] or 0) > 1e-3]
        # extraction only: both sources must hash identically after the read pass
        sha_after = {'base': sha256(args.base), 'twin': sha256(args.twin)}
        sources_unmodified = {k: sha_after[k] == ids[k]['sha256'] for k in ('base', 'twin')}
        energies = [detail[L][g]['rank1_energy_share'] for L in layers for g in ('attn_output', 'ffn_down') if g in detail[L]]
        strengths = [detail[L][g]['strength_abl_form'] for L in layers for g in ('attn_output', 'ffn_down') if g in detail[L]]
        checks = {
            'min_rank1_energy_share': float(min(energies)),
            'mean_rank1_energy_share': float(np.mean(energies)),
            'all_rank1_energy_gt_0.99': bool(all(e > 0.99 for e in energies)),
            'strength_abl_form_mean': float(np.mean(strengths)),
            'strength_abl_form_std': float(np.std(strengths)),
            'strength_abl_form_min': float(min(strengths)),
            'strength_abl_form_max': float(max(strengths)),
            'mean_signed_cos_v1_to_WT_axis': float(np.mean([detail[L][g]['cos_v1_to_WT_axis'] for L in layers for g in ('attn_output', 'ffn_down') if g in detail[L]])),
            'extra_sites_rank1_energy_share': {k: v['rank1_energy_share'] for k, v in extra.items()},
            'extra_sites_strength_abl_form': {k: v['strength_abl_form'] for k, v in extra.items()},
            'axis_unit_norm': float(np.linalg.norm(axis)),
            'axis_is_unit_norm': bool(abs(float(np.linalg.norm(axis)) - 1.0) < 1e-6),
            'rank1_note': ('rank1_share_sv = s1/sum(s), which is diluted by the 896-long singular-value spectrum and is NOT a rank-1 measure; '
                           'rank1_energy_share = s1^2/sum(s^2) is the rank-1 measure; sv2_over_sv1 is the bf16 rounding floor'),
            'min_rank1_share_sv': float(min(detail[L][g]['rank1_share_sv'] for L in layers for g in ('attn_output', 'ffn_down') if g in detail[L])),
            'writeback_tensors': len(keys),
            'writeback_tensors_edited': len(touched),
            'mean_abs_cos_u1_to_axis': float(np.mean([abs(v) for v in per_layer_cos.values()])),
            'min_signed_cos_u1_to_axis': float(min(per_layer_cos.values())),
            'max_signed_cos_u1_to_axis': float(max(per_layer_cos.values())),
            'mean_abs_cos_v1_to_WT_axis': float(np.mean([abs(detail[L][g]['cos_v1_to_WT_axis']) for L in layers for g in ('attn_output', 'ffn_down') if g in detail[L]])),
            'mean_residual_frac_shared_axis': float(np.mean([detail[L][g]['residual_frac_shared_axis_rank1'] for L in layers for g in ('attn_output', 'ffn_down') if g in detail[L] and detail[L][g]['residual_frac_shared_axis_rank1'] is not None])),
            'sources_unmodified': sources_unmodified,
            'extraction_only': bool(all(sources_unmodified.values())),
            'sha256_after': sha_after,
        }

        report = {
            'schema': 'jdewitt.twin-abduction/1',
            'scope': ('Static weight-algebra extraction of the direction a published rank-1 abliteration removed. '
                      'Two safetensors files are read in float32; dW = twin - base per writeback tensor; SVD per tensor. '
                      'No model is loaded, no forward pass, no generation, no GPU, no benchmark. Removal is NOT measured here: '
                      'every claim below is about weights, and a behavioural claim would need a served model (out of scope).'),
            'removal_measured': False,
            'caveat': 'The twin is stored as bf16, so dW = bf16(base - edit) - base carries a rounding floor. Untouched tensors are not all-zero; they are classified by relative delta norm and reported separately.',
            'sources': ids,
            'edit_footprint': footprint,
            'weight_algebra': {
                'stacked_left_matrix': list(M.shape),
                'shared_axis_energy_share': axis_energy,
                'shared_axis': [float(x) for x in axis],
                'module_axes': {g: {'energy_share': m['energy_share'], 'cos_to_shared': m['cos_to_shared'],
                                    'axis': [float(x) for x in m['axis']]} for g, m in module_axis.items()},
                'method': ('dW = W_twin - W_base [out, in]; u1,v1 = top singular vectors, sign fixed so the largest |u1| entry is positive '
                           'and v1 flipped with u1; shared axis = top right singular vector of the (2*n_layers, hidden) stack of u1; '
                           'strength_ls_shared_axis is the least-squares a in dW ~= -a r (W^T r)^T with r = shared axis; '
                           'residual_frac_shared_axis_rank1 = ||dW + a r (W^T r)^T|| / ||dW||.'),
            },
            'per_tensor': {str(L): {g: {k: v for k, v in rows[L][g].items() if k not in ('u1', 'v1')} for g in rows[L]} for L in rows},
            'per_tensor_fit': {str(L): detail[L] for L in detail},
            'extra_edit_sites': extra,
            'repo_diff_beside_weights': repo_diff,
            'per_layer_cos_u1_to_shared_axis': per_layer_cos,
            'layer_cos_table_u1': cos_table,
            'checks': checks,
            'artifacts': {'axis_npy': str(args.axis_out), 'delta_npys': dw_paths},
            'run': {'argv': sys.argv, 'numpy': np.__version__,
                    'openblas_num_threads': os.environ.get('OPENBLAS_NUM_THREADS'),
                    'note': 'BLAS must be single-threaded here: scipy-openblas spawns 64 threads on 32 cores and an SVD of 896x4864 goes from 1.7 s to over 60 s.'},
            'elapsed_s': round(time.monotonic() - t0, 2),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ---- optional: pack the deltas into a GGUF delta file and verify it byte-for-byte ----
    if args.delta_gguf:
        import struct as _struct
        from jdewitt.agent import gguf_delta_pack
        from jdewitt.ggufio import parse_gguf_header
        specs = [f'{name}={path}' for name, path in dw_paths.items()]
        pack = gguf_delta_pack(str(args.delta_gguf), specs)
        hdr = parse_gguf_header(args.delta_gguf)
        infos = {i['name'].decode(): i for i in hdr['infos']}
        mismatched, checked = [], 0
        with open(args.delta_gguf, 'rb') as f:
            for name, path in dw_paths.items():
                info = infos.get(name)
                if info is None:
                    mismatched.append((name, 'missing'))
                    continue
                dims = _struct.unpack('<%dQ' % info['n_dims'], info['dims'])
                nbytes = int(np.prod(dims)) * 4          # F32 tensors
                f.seek(hdr['data_base'] + info['offset'])
                got = f.read(nbytes)
                want = np.ascontiguousarray(np.load(path), dtype='<f4').tobytes()
                checked += 1
                if got != want:
                    mismatched.append((name, 'bytes differ'))
        pack['verified_tensors'] = checked
        pack['byte_identical_to_npys'] = not mismatched
        pack['mismatched'] = mismatched
        pack['sha256'] = sha256(args.delta_gguf)
        report['delta_gguf'] = pack
        checks['delta_gguf_byte_identical'] = not mismatched
        print(json.dumps({'stage': 'delta-gguf', 'file': str(args.delta_gguf),
                          'tensors': pack['tensors'], 'verified': checked,
                          'byte_identical': not mismatched, 'sha256': pack['sha256']}), flush=True)

    args.out.write_text(json.dumps(report, indent=1) + '\n')
    np.save(args.axis_out, axis.astype(np.float32))
    print(json.dumps({'stage': 'report', 'out': str(args.out), 'axis_out': str(args.axis_out),
                      'npy': len(dw_paths), 'checks': {k: v for k, v in checks.items() if 'bool' in k or 'min' in k or 'mean' in k or 'tensors' in k}},
                     indent=1))
    return report


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
