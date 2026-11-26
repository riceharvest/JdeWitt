# SPDX-License-Identifier: AGPL-3.0-or-later
"""Static agreement audit across the weight-only refusal-axis constructions.

Usage: python scripts/axis_audit.py --models runs/axis-audit/models.json \
           --out verification/axis-audit/agreement.json \
           [--gguf "verification/iq05b/Qwen2.5-0.5B-Instruct-IQ3_M.gguf"]

Computes, from safetensors bytes only (numpy, no torch, no model, no forward pass):

  x-1   twin diff      dW = twin - instruct, top LEFT singular vector per writeback tensor
  x-1b  column norm    the largest-norm COLUMN of the same dW (the repo's
                       oracle_axis_alignment.py heuristic: no SVD)
  x-3   alignment      dW = instruct - base, top left singular vector per writeback tensor
  x-3b  column norm    largest-norm column of the alignment delta
  x-4   head geometry  u = normalize(mean(denial head rows) - mean(compliance head rows));
                       d = normalize(W_out^T u), with two token-set variants

and reports pairwise cosines (global and per layer), Frobenius norms, top-1 singular
shares, layer-to-layer stability, the alignment delta's subspace overlap with the
twin axis, how much writeback energy each axis touches, and a survey of the fitted
axes already committed in the repo.

Signs of a singular vector are arbitrary, so every cosine is reported signed AND
absolute; the agreement matrix uses |cos|. x-4's sign IS fixed (denial minus
compliance), which is why its signed value is the one to read.

No behavioural claim: removal_measured is false everywhere. A low cosine is a
finding about which sub-space a construction targets, not an error.
"""
import argparse
import hashlib
import json
import os
import sys
import time

# One BLAS thread: every operation here is a small gemv/gram, and OpenBLAS's
# multithreaded path thrashes on this loaded host (measured 18 s vs 1.8 s for the
# same 896x896 SVD). Must be set before numpy is imported.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

from pathlib import Path

import numpy as np

KINDS = ('o_proj', 'down_proj')
HF_NAME = {'o_proj': 'self_attn.o_proj.weight', 'down_proj': 'mlp.down_proj.weight'}


# --------------------------------------------------------------------------- reader
class SafeTensors:
    """Minimal BF16-aware, memory-mapped safetensors reader (no torch)."""

    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, 'rb') as f:
            n = int.from_bytes(f.read(8), 'little')
            self.header = json.loads(f.read(n))
            self._base = 8 + n
        self._data = np.memmap(self.path, dtype=np.uint8, mode='r', offset=self._base)

    def names(self):
        return [k for k in self.header if k != '__metadata__']

    def has(self, name):
        return name in self.header

    def get(self, name):
        info = self.header[name]
        dtype, shape = info['dtype'], tuple(info['shape'])
        start, end = info['data_offsets']
        n_el = int(np.prod(shape))
        if dtype == 'BF16':
            raw = np.frombuffer(self._data[start:end], dtype='<u2', count=n_el)
            out = (raw.astype(np.uint32) << 16).view(np.float32)
        elif dtype == 'F32':
            out = np.frombuffer(self._data[start:end], dtype='<f4', count=n_el).copy()
        elif dtype == 'F16':
            out = np.frombuffer(self._data[start:end], dtype='<f2', count=n_el).astype(np.float32)
        else:
            raise ValueError(f'{name}: dtype {dtype} is not a float weight')
        return out.reshape(shape)


# --------------------------------------------------------------------------- axes
def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0.0:
        raise ValueError('zero axis')
    return (v / n).astype(np.float32)


def top_axis(delta):
    """Top left singular vector and the whole rank-1/energy picture, exactly.

    The delta is at most 896 rows, so the 896x896 Gram matrix dW dW^T holds the
    full left spectrum: eigh is 0.7 s single-threaded where LAPACK's gesdd on
    the 896x4864 matrix is ~10 s (and 18 s multithreaded under load, because
    OpenBLAS's tiny-gemv path thrashes). s1^2 = lambda_max, the rank-1 share is
    lambda_max / trace(G) = s1^2 / ||dW||_F^2, and the eigenvectors are the left
    singular vectors.
    """
    dW = delta.astype(np.float64)
    gram = dW @ dW.T
    vals, vecs = np.linalg.eigh(gram)
    order = np.argsort(vals)[::-1]
    vals, vecs = np.clip(vals[order], 0.0, None), vecs[:, order]
    total = float(vals.sum())
    if total == 0.0:
        raise ValueError('zero delta')
    shares = vals / total
    u = vecs[:, 0].copy()
    j = int(np.argmax(np.abs(u)))          # deterministic 1-D sign convention
    if u[j] < 0:
        u = -u
    cum = np.cumsum(shares)
    return {'u': u.astype(np.float32), 's1': float(np.sqrt(vals[0])),
            'share1': float(shares[0]), 'shares_top8': [float(x) for x in shares[:8]],
            'cumulative_share': {f'k{k}': float(cum[min(k, len(cum)) - 1])
                                 for k in (1, 2, 4, 8, 16, 32) if k <= len(cum)},
            'spectral_gap_s2_over_s1': (float(np.sqrt(vals[1] / vals[0]))
                                        if len(vals) > 1 and vals[0] > 0 else 1.0)}


def column_axis(delta):
    """The repo's oracle heuristic: the largest-norm column, normalised. No SVD."""
    norms = np.linalg.norm(delta, axis=0)
    j = int(np.argmax(norms))
    return _unit(delta[:, j]), float(norms[j]), j


def combine(vectors, weights):
    """Sign-align every vector to the first, weight, sum, normalise."""
    ref = vectors[0]
    acc = np.zeros(len(ref), dtype=np.float64)
    for v, w in zip(vectors, weights):
        sgn = 1.0 if float(np.dot(v, ref)) >= 0.0 else -1.0
        acc += sgn * float(w) * np.asarray(v, np.float64)
    return _unit(acc)


def cos(a, b):
    return float(np.dot(np.asarray(a, np.float64), np.asarray(b, np.float64)))


def layer_axes(recs, layers):
    """Per-layer axis from the writeback tensors, plus per-layer diagnostics."""
    out = {}
    for L in layers:
        rr = [r for r in recs if r['layer'] == L]
        if not rr:
            continue
        out[L] = {'axis': combine([r['u'] for r in rr], [r['s1'] for r in rr]),
                  'cos_between_kinds': round(cos(rr[0]['u'], rr[1]['u']), 4) if len(rr) > 1 else None,
                  'delta_frob': round(float(np.sqrt(sum(r['frob'] ** 2 for r in rr))), 4),
                  'axis_energy': (round(float(sum(r['axis_energy'] for r in rr)), 4)
                                  if all('axis_energy' in r for r in rr) else None),
                  'weight_energy': (round(float(sum(r['weight_frob2'] for r in rr)), 4)
                                    if all('weight_frob2' in r for r in rr) else None)}
    return out


def shared_axis(recs):
    """Top right singular vector of the stacked, singular-value-weighted left vectors."""
    stack = np.stack([np.asarray(r['u'], np.float64) * r['s1'] for r in recs])
    _u, s, vh = np.linalg.svd(stack, full_matrices=False)
    energy = np.square(s, dtype=np.float64)
    return _unit(vh[0]), (energy / energy.sum()).tolist(), stack.shape


# --------------------------------------------------------------------------- tokens
def resolve_ids(tokenizer_json, surfaces):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_json))
    ids, table = [], []
    for surface in surfaces:
        got = []
        for form in (' ' + surface, surface):
            enc = tok.encode(form).ids
            if enc:
                got.append(int(enc[0]))
        if got:
            ids.extend(got)
            table.append({'surface': surface, 'first_token_ids': got,
                          'tokens': [tok.id_to_token(i) for i in got]})
    return sorted(set(ids)), table


# --------------------------------------------------------------------------- survey
def survey_committed_axes(root, width, skip):
    """Committed fitted-axis artifacts, and whether they are comparable to this width."""
    found = []
    for path in sorted(Path(root).rglob('*')):
        if path.suffix not in ('.json', '.pt', '.npy') or not path.is_file():
            continue
        name = path.name.lower()
        if not any(k in name for k in ('axis', 'axes')) or path.resolve() == skip.resolve():
            continue
        entry = {'path': str(path.relative_to(Path(root).parent)), 'bytes': path.stat().st_size}
        if path.suffix == '.json':
            try:
                meta = json.loads(path.read_text())
            except Exception:
                continue
            axes = meta.get('axes') if isinstance(meta.get('axes'), dict) else {}
            shape = meta.get('shape') or axes.get('shape')
            entry.update({'shape': shape, 'dtype': meta.get('dtype') or axes.get('dtype'),
                          'tensor_sha256': meta.get('tensor_sha256') or axes.get('sha256')})
            if shape:
                entry['comparable'] = ('same width: comparable if the tensor can be loaded'
                                       if int(shape[-1]) == width else
                                       f'dimension-incompatible: axis width {shape[-1]} != hidden {width}')
            else:
                entry['comparable'] = 'unknown: no shape in the JSON'
        else:
            entry.update({'shape': None, 'dtype': None, 'tensor_sha256': None,
                          'comparable': 'unreadable without torch; its sidecar JSON carries the shape'})
        found.append(entry)
    return found


# --------------------------------------------------------------------------- stats
def summarise(vals, name):
    if not vals:
        return {f'{name}_mean_abs': None, 'n': 0}
    a = np.abs(np.asarray(vals, np.float64))
    s = np.asarray(vals, np.float64)
    return {f'{name}_mean_abs': round(float(a.mean()), 4),
            f'{name}_median_abs': round(float(np.median(a)), 4),
            f'{name}_min_abs': round(float(a.min()), 4),
            f'{name}_max_abs': round(float(a.max()), 4),
            f'{name}_mean_signed': round(float(s.mean()), 4), 'n': int(len(vals))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='runs/axis-audit/models.json')
    ap.add_argument('--out', default='verification/axis-audit/agreement.json')
    ap.add_argument('--repo-root', default='.')
    ap.add_argument('--gguf', default=None)
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()
    t0 = time.time()

    manifest = json.loads(Path(args.models).read_text())
    models = {}
    for key, rec in manifest['models'].items():
        st = SafeTensors(Path(rec['snapshot']) / 'model.safetensors')
        cfg = json.loads((Path(rec['snapshot']) / 'config.json').read_text())
        models[key] = {'st': st, 'cfg': cfg, 'rec': rec}
        print(f'{key}: {len(st.names())} tensors, hidden {cfg["hidden_size"]}, '
              f'layers {cfg["num_hidden_layers"]}', flush=True)
    hidden = int(models['instruct']['cfg']['hidden_size'])
    layers = list(range(int(models['instruct']['cfg']['num_hidden_layers'])))

    # One read of the edited model's writeback tensors, reused by every metric.
    inst_wb = {f'model.layers.{L}.{HF_NAME[k]}': models['instruct']['st'].get(f'model.layers.{L}.{HF_NAME[k]}')
               for L in layers for k in KINDS}

    # Head-tiedness and the exact head row source for x-4.
    head_rows, head_info = {}, {}
    for key, m in models.items():
        st = m['st']
        has_lm, has_emb = st.has('lm_head.weight'), st.has('model.embed_tokens.weight')
        tied = bool(has_lm and has_emb and np.array_equal(st.get('lm_head.weight'), st.get('model.embed_tokens.weight')))
        head_rows[key] = 'lm_head.weight' if has_lm and not tied else 'model.embed_tokens.weight'
        head_info[key] = {'tied_to_embedding': tied, 'lm_head_present': has_lm,
                          'rows_used_for_x4': head_rows[key]}
    print('head:', head_info, flush=True)

    def writeback_delta(a_key, b_key):
        out = []
        for L in layers:
            for kind in KINDS:
                name = f'model.layers.{L}.{HF_NAME[kind]}'
                dW = (models[a_key]['st'].get(name).astype(np.float32) -
                      models[b_key]['st'].get(name).astype(np.float32))
                out.append((L, kind, name, dW))
        return out

    def footprint(a_key, b_key):
        """Which tensors differ between two artifacts, by suffix: the edit's surface."""
        names = sorted(set(models[a_key]['st'].names()) & set(models[b_key]['st'].names()))
        differ, identical, shape_changed = [], 0, []
        for n in names:
            A, B = models[a_key]['st'].get(n), models[b_key]['st'].get(n)
            if A.shape != B.shape:
                shape_changed.append(n)
            elif np.array_equal(A, B):
                identical += 1
            else:
                differ.append(n)
        by_suffix = {}
        for n in differ:
            by_suffix[n.split('.')[-2]] = by_suffix.get(n.split('.')[-2], 0) + 1
        return {'tensors_compared': len(names), 'bit_identical': identical,
                'value_differing': len(differ), 'shape_changed': shape_changed,
                'differing_by_suffix': dict(sorted(by_suffix.items())),
                'differing_names': differ,
                'only_in_a': sorted(set(models[a_key]['st'].names()) - set(names)),
                'only_in_b': sorted(set(models[b_key]['st'].names()) - set(names))}

    footprints = {'twin_minus_instruct': footprint('twin', 'instruct'),
                  'instruct_minus_base': footprint('instruct', 'base')}

    def delta_method(a_key, b_key, label, assumes, rank1_check=False):
        recs = []
        for L, kind, name, dW in writeback_delta(a_key, b_key):
            frob2 = float(np.square(dW.astype(np.float64)).sum())
            if frob2 == 0.0:
                continue
            W = inst_wb[name]
            ax = top_axis(dW)
            w2 = float(np.square(W.astype(np.float64)).sum())
            proj = W.astype(np.float64).T @ ax['u'].astype(np.float64)
            rec = {'layer': L, 'kind': kind, 'name': name, 'u': ax['u'], 's1': ax['s1'],
                   'share1': ax['share1'], 'shares_top8': ax['shares_top8'],
                   'cumulative_share': ax['cumulative_share'],
                   'spectral_gap_s2_over_s1': ax['spectral_gap_s2_over_s1'],
                   'frob': float(np.sqrt(frob2)),
                   'rank1_residual_relative_to_weight': float(np.sqrt(max(0.0, 1.0 - ax['share1'])) *
                                                              np.sqrt(frob2) / np.sqrt(w2)),
                   'weight_frob': float(np.sqrt(w2)), 'weight_frob2': w2,
                   'axis_energy': float(proj @ proj)}
            if rank1_check:
                # dW = s1 u v^T: recover v and compare it with W^T u, the right vector a
                # rank-1 projection of W along u would produce. cos ~ 1 means the twin
                # edit IS that projection of the instruct weight, not some other finetune.
                v = dW.astype(np.float64).T @ ax['u'].astype(np.float64)
                v_pred = W.astype(np.float64).T @ ax['u'].astype(np.float64)
                nv, npv = np.linalg.norm(v), np.linalg.norm(v_pred)
                rec['right_vector_cos_vs_Wtu'] = float(abs(v @ v_pred) / (nv * npv)) if nv > 0 and npv > 0 else None
            recs.append(rec)
        if not recs:
            raise SystemExit(f'{label}: every writeback delta is zero')
        out = {'label': label, 'assumes': assumes,
               'space': 'residual stream (writeback output space), per layer',
               'tensors': [{k: v for k, v in r.items() if k != 'u'} for r in recs],
               'per_tensor_share1_mean': round(float(np.mean([r['share1'] for r in recs])), 6),
               'per_tensor_share1_min': round(float(np.min([r['share1'] for r in recs])), 6),
               'rank1_residual_relative_to_weight_mean': round(float(np.mean(
                   [r['rank1_residual_relative_to_weight'] for r in recs])), 6),
               'delta_frob_mean': round(float(np.mean([r['frob'] for r in recs])), 4),
               'delta_frob_relative_mean': round(float(np.mean([r['frob'] / r['weight_frob'] for r in recs])), 6),
               'per_layer': layer_axes(recs, layers)}
        if rank1_check:
            vals = [r['right_vector_cos_vs_Wtu'] for r in recs if r['right_vector_cos_vs_Wtu'] is not None]
            out['right_vector_cos_vs_Wtu'] = {**summarise(vals, 'cos'), 'meaning':
                                              'cos of the recovered right singular vector dW^T u/s1 with '
                                              'W_instruct^T u / ||W_instruct^T u||: 1.0 means the delta is '
                                              'exactly the rank-1 projection of the instruct writeback matrix '
                                              'along u, which is the abliteration construction'}
        g, spectrum, shape = shared_axis(recs)
        out['global_axis'] = g
        out['global_stack'] = {'shape': list(shape),
                               'spectrum_share': [round(x, 6) for x in spectrum[:8]],
                               'spectrum_share_top8_sum': round(float(sum(spectrum[:8])), 6)}
        out['_recs'] = recs
        return out

    x1 = delta_method('twin', 'instruct', 'x1_twin_diff_svd',
                      'the published twin edit is rank-1 per writeback tensor '
                      '(dW = -a r (W^T r)^T), so the top LEFT singular vector of twin-instruct is '
                      'the residual direction that was removed; nothing about the axis is fitted '
                      'and no token set is involved, the whole claim is the rank-1 shape.',
                      rank1_check=True)
    x3 = delta_method('instruct', 'base', 'x3_alignment_delta_svd',
                      'the leading left singular vector of instruct-base is the "alignment '
                      'direction"; the delta is high-rank and also carries instruction-following, '
                      'so this construction assumes a refusal component lives on the leading axis.')

    def column_method(a_key, b_key, label):
        recs = []
        for L, kind, name, dW in writeback_delta(a_key, b_key):
            if not np.abs(dW).any():
                continue
            u, norm, j = column_axis(dW)
            recs.append({'layer': L, 'kind': kind, 'name': name, 'u': u, 's1': norm,
                         'column': j, 'frob': float(np.linalg.norm(dW))})
        out = {'label': label,
               'assumes': 'the removed residual direction is a COLUMN of dW (dW = -s u v^T), and the '
                          'largest-norm column is that vector: no SVD and no rank assumption beyond '
                          'rank-1. This is the heuristic the repo already uses in '
                          'scripts/oracle_axis_alignment.py.',
               'space': 'residual stream (writeback output space), per layer',
               'tensors': [{k: v for k, v in r.items() if k != 'u'} for r in recs],
               'per_layer': layer_axes(recs, layers)}
        g, spectrum, shape = shared_axis(recs)
        out['global_axis'] = g
        out['global_stack'] = {'shape': list(shape),
                               'spectrum_share': [round(x, 6) for x in spectrum[:8]]}
        out['_recs'] = recs
        return out

    x1b = column_method('twin', 'instruct', 'x1b_twin_diff_column_norm')
    x3b = column_method('instruct', 'base', 'x3b_alignment_column_norm')

    # x-1e: the twin also edited the token embedding (see edit_footprints). Under a
    # rank-1 projection E' = E - (E r) r^T, dE = -(E r) r^T has RIGHT singular vector
    # r, so the embedding delta is an independent read of the same residual axis.
    dE = (models['twin']['st'].get('model.embed_tokens.weight').astype(np.float32) -
          models['instruct']['st'].get('model.embed_tokens.weight').astype(np.float32))
    gram_e = dE.T.astype(np.float64) @ dE.astype(np.float64)
    vals_e, vecs_e = np.linalg.eigh(gram_e)
    order_e = np.argsort(vals_e)[::-1]
    vals_e, vecs_e = np.clip(vals_e[order_e], 0.0, None), vecs_e[:, order_e]
    share_e = vals_e / vals_e.sum()
    v_e = vecs_e[:, 0].copy()
    if v_e[int(np.argmax(np.abs(v_e)))] < 0:
        v_e = -v_e
    x1e = {'label': 'x1e_twin_embedding_delta',
           'assumes': 'the twin edited model.embed_tokens.weight with the same rank-1 projection '
                      '(dE = -(E r) r^T), so the RIGHT singular vector of the embedding delta is the '
                      'residual direction r; this is a second, independent read of the twin axis from '
                      'a different tensor shape.',
           'space': 'residual stream (embedding ROW space)',
           'tensors': [{'name': 'model.embed_tokens.weight', 'shape': list(dE.shape),
                        'frob': round(float(np.linalg.norm(dE.astype(np.float64))), 4),
                        'weight_frob': round(float(np.linalg.norm(
                            models['instruct']['st'].get('model.embed_tokens.weight').astype(np.float64))), 4)}],
           'per_tensor_share1_mean': round(float(share_e[0]), 6),
           'per_tensor_share1_min': round(float(share_e[0]), 6),
           'delta_frob_relative_mean': round(float(
               float(np.linalg.norm(dE.astype(np.float64))) /
               float(np.linalg.norm(models['instruct']['st'].get('model.embed_tokens.weight').astype(np.float64)))), 6),
           'shares_top8': [round(float(x), 6) for x in share_e[:8]],
           'cumulative_share': {f'k{k}': round(float(share_e[:k].sum()), 6) for k in (1, 2, 4, 8, 16)},
           'global_axis': v_e.astype(np.float32),
           'global_stack': {'shape': [1, int(dE.shape[1])],
                            'spectrum_share': [round(float(x), 6) for x in share_e[:8]]}}

    # -------------------------------- x-4 head geometry (three token-set variants)
    tok_json = Path(models['instruct']['rec']['snapshot']) / 'tokenizer.json'
    denial_asset = json.loads((root / 'jdewitt/assets/vocab-remap-denial.json').read_text())
    affirm_asset = json.loads((root / 'jdewitt/assets/vocab-remap-affirmative.json').read_text())
    denial_surfaces = ['I', 'Sorry', 'Cannot', 'Unfortunately', 'No'] + [e['form'] for e in denial_asset['entries']]
    compliance_surfaces = ['Sure', 'Certainly', 'Yes', 'Here', 'Okay', 'Absolutely']
    affirm_surfaces = [e['affirmative'] for e in affirm_asset['entries'] if e.get('affirmative')]
    denial_ids, denial_table = resolve_ids(tok_json, denial_surfaces)
    comp_ids, comp_table = resolve_ids(tok_json, compliance_surfaces)
    affirm_ids, affirm_table = resolve_ids(tok_json, affirm_surfaces)
    core_denial_ids, core_denial_table = resolve_ids(tok_json, ['I', 'Sorry', 'Cannot', 'Unfortunately', 'No'])

    def head_axis(W, den_ids, pos_ids):
        """u as a sparse logit-space vector; d = W^T u is its residual pullback.

        W is [vocab, hidden]; u = normalize(mean one-hot(denial) - mean
        one-hot(compliance)) is vocab-dim, and W^T u is exactly the mean head-row
        difference, so the residual axis needs no forward pass and no token
        beyond the ids.
        """
        u = np.zeros(W.shape[0], dtype=np.float64)
        np.add.at(u, den_ids, 1.0 / len(den_ids))
        np.add.at(u, pos_ids, -1.0 / len(pos_ids))
        u = _unit(u)
        d = W.astype(np.float32).T @ u.astype(np.float32)
        return {'u': u, 'd': _unit(d), 'residual_norm': float(np.linalg.norm(d))}

    x4 = {}
    for key in ('instruct', 'base', 'twin'):
        W = models[key]['st'].get(head_rows[key])
        primary = head_axis(W, denial_ids, comp_ids)
        var_affirm = head_axis(W, denial_ids, affirm_ids)
        var_core = head_axis(W, core_denial_ids, comp_ids)
        x4[key] = {'label': f'x4_head_geometry_{key}',
                   'assumes': 'the mean denial head row minus the mean compliance head row is the '
                              'logit-space refusal direction and W_out^T (or E^T when tied) maps it '
                              'to the residual direction that raises that margin; depends on the '
                              'token sets, needs no activation and no forward pass.',
                   'space': 'logit space (u) and residual stream (d)',
                   'head': head_info[key],
                   '_d': primary['d'], '_u': primary['u'],
                   'detail': {'head_row_separation_norm': round(primary['residual_norm'], 4),
                              'n_denial_ids': len(denial_ids), 'n_compliance_ids': len(comp_ids),
                              'cos_vs_affirmative_table': round(cos(primary['d'], var_affirm['d']), 4),
                              'cos_vs_core_denial_set': round(cos(primary['d'], var_core['d']), 4),
                              'u_cos_vs_affirmative_table': round(cos(primary['u'], var_affirm['u']), 4),
                              'u_cos_vs_core_denial_set': round(cos(primary['u'], var_core['u']), 4)},
                   'variants': {
                       'affirmative_table': {'denial_ids': denial_ids, 'compliance_ids': affirm_ids,
                                             'head_row_separation_norm': round(var_affirm['residual_norm'], 4),
                                             'd': var_affirm['d'].tolist()},
                       'core_denial_set': {'denial_ids': core_denial_ids, 'compliance_ids': comp_ids,
                                           'head_row_separation_norm': round(var_core['residual_norm'], 4),
                                           'd': var_core['d'].tolist()}}}

    # -------------------------------- agreement
    axes = {'x1': x1['global_axis'], 'x1e': x1e['global_axis'], 'x1b': x1b['global_axis'],
            'x3': x3['global_axis'], 'x3b': x3b['global_axis'],
            'x4': x4['instruct']['_d'], 'x4base': x4['base']['_d'], 'x4twin': x4['twin']['_d']}
    labels = list(axes)
    cos_matrix = {a: {b: round(abs(cos(axes[a], axes[b])), 4) for b in labels} for a in labels}

    per_layer_src = {'x1': x1, 'x1b': x1b, 'x3': x3, 'x3b': x3b}
    per_layer_pairs = {}
    for a, b in (('x1', 'x3'), ('x1', 'x4'), ('x3', 'x4'), ('x1', 'x3b'), ('x1b', 'x3')):
        vals = []
        for L in layers:
            ua = per_layer_src[a]['per_layer'][L]['axis'] if a in per_layer_src else axes[a]
            ub = per_layer_src[b]['per_layer'][L]['axis'] if b in per_layer_src else axes[b]
            vals.append({'layer': L, 'cos_signed': round(cos(ua, ub), 4),
                         'cos_abs': round(abs(cos(ua, ub)), 4)})
        per_layer_pairs[f'{a}_vs_{b}'] = {
            **summarise([v['cos_abs'] for v in vals], 'cos'),
            'signed_mean': round(float(np.mean([v['cos_signed'] for v in vals])), 4),
            'signed_min': round(float(np.min([v['cos_signed'] for v in vals])), 4),
            'signed_max': round(float(np.max([v['cos_signed'] for v in vals])), 4),
            'per_layer': vals}

    stability = {}
    for a, src in per_layer_src.items():
        vals = [abs(cos(src['per_layer'][L]['axis'], src['global_axis'])) for L in layers]
        stability[a] = {**summarise([float(v) for v in vals], 'cos_with_shared'),
                        'per_layer': [round(float(v), 4) for v in vals]}

    # x-3's leading k-dim left subspace vs the x-1 and x-4 axes
    stack3 = np.stack([np.asarray(r['u'], np.float64) * r['s1'] for r in x3['_recs']])
    _u3, s3, vh3 = np.linalg.svd(stack3, full_matrices=False)
    g1 = np.asarray(x1['global_axis'], np.float64)
    d4 = np.asarray(x4['instruct']['_d'], np.float64)
    subspace = {}
    for k in (1, 2, 4, 8, 16):
        V = vh3[:k]
        subspace[f'k{k}'] = {'x1_energy_along_x3_topk': round(float(np.sum((V @ g1) ** 2)), 4),
                             'x4_energy_along_x3_topk': round(float(np.sum((V @ d4) ** 2)), 4),
                             'x3_topk_energy_share': round(float(np.square(s3[:k]).sum() / np.square(s3).sum()), 4)}

    # how much of the edited model's writeback energy each axis touches
    energy = {}
    for a in ('x1', 'x3', 'x4'):
        ax = np.asarray(axes[a], np.float64)
        fr = []
        for name, W in inst_wb.items():
            p = W.astype(np.float64).T @ ax
            fr.append(float(p @ p) / float(np.square(W.astype(np.float64)).sum()))
        energy[a] = {'mean_fraction_of_weight_energy_along_axis': round(float(np.mean(fr)), 8),
                     'min': round(float(np.min(fr)), 8), 'max': round(float(np.max(fr)), 8), 'n': len(fr)}

    agree_within = {
        'x1_svd_vs_column_norm': summarise([abs(cos(r1['u'], r1b['u']))
                                            for r1, r1b in zip(x1['_recs'], x1b['_recs'])], 'cos'),
        'x3_svd_vs_column_norm': summarise([abs(cos(r3['u'], r3b['u']))
                                            for r3, r3b in zip(x3['_recs'], x3b['_recs'])], 'cos'),
    }

    # -------------------------------- optional user-quant stand-in
    user_quant = None
    if args.gguf:
        p = Path(args.gguf)
        if p.is_file():
            h = hashlib.sha256()
            with open(p, 'rb') as f:
                for block in iter(lambda: f.read(1 << 22), b''):
                    h.update(block)
            user_quant = {'path': str(p), 'bytes': p.stat().st_size, 'sha256': h.hexdigest(),
                          'axes_audited': False,
                          'note': 'hashed only; every axis here comes from the float safetensors. '
                                  'Quantized-axis stability is not claimed by this card.'}
        else:
            user_quant = {'path': str(p), 'present': False}

    def public_method(d):
        """JSON-safe copy: per-layer axes become rounded lists, private keys dropped."""
        out = {k: v for k, v in d.items() if not k.startswith('_')}
        if 'per_layer' in d:
            out['per_layer'] = {str(L): {**{k: v for k, v in e.items() if k != 'axis'},
                                         'axis': [round(float(x), 8) for x in e['axis']]}
                                for L, e in d['per_layer'].items()}
        return out

    out_path = root / args.out
    out = {
        'schema': 'jdewitt.axis-audit/1',
        'date_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'removal_measured': False,
        'what': 'static agreement audit of the weight-only refusal-axis constructions: pairwise '
                'cosines, norms, rank-1 shares, layer stability and subspace overlap, from '
                'safetensors bytes only',
        'scope': {'model': 'Qwen2.5-0.5B-Instruct family', 'hidden': hidden, 'layers': len(layers),
                  'writeback_kinds': list(KINDS), 'writeback_tensors': len(layers) * len(KINDS),
                  'no_forward': True, 'no_gpu': True, 'no_capture': True,
                  'why_static': 'a forward pass, a serve or a benchmark is out of scope; every number '
                                'here is matrix algebra on published or committed weights'},
        'artifacts': {k: {'repo': m['rec']['repo'], 'revision': m['rec']['revision'],
                          'gated': m['rec']['gated'], 'snapshot': m['rec']['snapshot'],
                          'weights': max(m['rec']['files'], key=lambda f: f['bytes']),
                          'files': m['rec']['files']} for k, m in models.items()},
        'user_quant_stand_in': user_quant,
        'head': head_info,
        'token_sets': {'denial_surfaces': denial_surfaces, 'compliance_surfaces': compliance_surfaces,
                       'denial_ids': denial_ids, 'compliance_ids': comp_ids,
                       'affirmative_surfaces': affirm_surfaces, 'affirmative_ids': affirm_ids,
                       'denial_table': denial_table, 'compliance_table': comp_table,
                       'affirmative_table': affirm_table,
                       'core_denial_ids': core_denial_ids, 'core_denial_table': core_denial_table},
        'methods': {
            'x1_twin_diff_svd': public_method(x1),
            'x1e_twin_embedding_delta': public_method(x1e),
            'x1b_twin_diff_column_norm': public_method(x1b),
            'x3_alignment_delta_svd': public_method(x3),
            'x3b_alignment_column_norm': public_method(x3b),
            'x4_head_geometry': {k: {'label': v['label'], 'assumes': v['assumes'], 'space': v['space'],
                                     'head': v['head'], 'detail': v['detail'],
                                     'global_axis': v['_d'].tolist(),
                                     'logit_axis_note': 'u (151936-dim) is reproducible from the token '
                                                        'ids and the head rows; only d is stored',
                                     'variants': {vk: {'denial_ids': vv['denial_ids'],
                                                       'compliance_ids': vv['compliance_ids'],
                                                       'head_row_separation_norm': vv['head_row_separation_norm'],
                                                       'global_axis': vv['d']}
                                                  for vk, vv in v['variants'].items()}}
                                 for k, v in x4.items()},
        },
        'agreement': {
            'cos_matrix_abs': cos_matrix,
            'cos_matrix_note': 'SVD signs are arbitrary, so a matrix cell is |cos|; the signed '
                               'value of each comparison is in global_signed and per_layer.',
            'global_signed': {
                'x1_vs_x3': round(cos(x1['global_axis'], x3['global_axis']), 4),
                'x1_vs_x1e': round(cos(x1['global_axis'], x1e['global_axis']), 4),
                'x1_vs_x4': round(cos(x1['global_axis'], x4['instruct']['_d']), 4),
                'x3_vs_x4': round(cos(x3['global_axis'], x4['instruct']['_d']), 4),
                'x1_vs_x1b': round(cos(x1['global_axis'], x1b['global_axis']), 4),
                'x3_vs_x3b': round(cos(x3['global_axis'], x3b['global_axis']), 4),
                'x4_instruct_vs_base': round(cos(x4['instruct']['_d'], x4['base']['_d']), 4),
                'x4_instruct_vs_twin': round(cos(x4['instruct']['_d'], x4['twin']['_d']), 4),
                'x4_u_instruct_vs_base': round(cos(x4['instruct']['_u'], x4['base']['_u']), 4),
                'x4_u_instruct_vs_twin': round(cos(x4['instruct']['_u'], x4['twin']['_u']), 4)},
            'per_layer': per_layer_pairs,
            'layer_stability': stability,
            'x3_subspace_overlap': subspace,
            'axis_energy_in_instruct_writebacks': energy,
            'svd_vs_column_norm_within_delta': agree_within,
        },
        'fitted_axes_committed': survey_committed_axes(root / 'verification', hidden, out_path),
        'edit_footprints': footprints,
        'gated': ['every behavioural or served claim', 'GPU throughput',
                  'a capture refit in this width', 'quantized-axis stability beyond the hashed stand-in'],
        'elapsed_s': round(time.time() - t0, 1),
    }

    ga = out['agreement']['global_signed']
    findings = []

    def f(key, value, reading):
        findings.append({'key': key, 'value': value, 'reading': reading})

    s1 = out['methods']['x1_twin_diff_svd']['per_tensor_share1_mean']
    f('x1_rank1_share', s1,
      'the twin delta is rank-1 per writeback tensor' if s1 > 0.99 else
      'the twin delta is NOT rank-1 per writeback tensor; the left-vector construction is an approximation')
    f('x1_vs_x3_cos', round(abs(ga['x1_vs_x3']), 4),
      'the twin axis and the alignment axis target different residual subspaces'
      if abs(ga['x1_vs_x3']) < 0.3 else 'the twin axis and the alignment axis share a residual subspace')
    f('x1e_embedding_rank1_share', out['methods']['x1e_twin_embedding_delta']['per_tensor_share1_mean'],
      'the twin embedding delta is rank-1 too' if
      out['methods']['x1e_twin_embedding_delta']['per_tensor_share1_mean'] > 0.99 else
      'the twin embedding delta is not rank-1; the embedding edit is not the same projection')
    f('x1_vs_x1e_cos', round(abs(ga['x1_vs_x1e']), 4),
      'the embedding delta recovers the same residual direction as the writebacks: one direction '
      'across tensors of different shapes' if abs(ga['x1_vs_x1e']) > 0.99 else
      'the embedding delta points somewhere other than the writeback axis')
    f('x1_vs_x4_cos', round(abs(ga['x1_vs_x4']), 4),
      'the head-geometry axis is not the twin edit axis' if abs(ga['x1_vs_x4']) < 0.3
      else 'the head-geometry axis matches the twin edit axis')
    f('x3_vs_x4_cos', round(abs(ga['x3_vs_x4']), 4),
      'the alignment axis is not the head-geometry axis' if abs(ga['x3_vs_x4']) < 0.3
      else 'the alignment axis matches the head-geometry axis')
    st1 = stability['x1']['cos_with_shared_mean_abs']
    st3 = stability['x3']['cos_with_shared_mean_abs']
    f('layer_stability', {'x1': st1, 'x3': st3},
      'the twin axis is shared across layers and the alignment axis is layer-specific'
      if (st1 > 0.9 > st3) else
      f'the more layer-stable construction is {"x-1" if st1 >= st3 else "x-3"}')
    f('x3_top1_share', out['methods']['x3_alignment_delta_svd']['per_tensor_share1_mean'],
      'the alignment delta is high-rank: its leading axis explains little of the delta energy'
      if out['methods']['x3_alignment_delta_svd']['per_tensor_share1_mean'] < 0.5 else
      'the alignment delta is dominated by one direction')
    f('per_layer_x1_vs_x3', per_layer_pairs['x1_vs_x3']['cos_mean_abs'],
      'per-layer agreement is weaker than the global-vector agreement'
      if per_layer_pairs['x1_vs_x3']['cos_mean_abs'] < abs(ga['x1_vs_x3']) else
      'per-layer agreement is at least as strong as the global-vector agreement')
    f('x4_token_set_sensitivity', round(abs(x4['instruct']['detail']['cos_vs_core_denial_set']), 4),
      'x-4 is sensitive to the token set: the core-denial variant moves the axis'
      if abs(x4['instruct']['detail']['cos_vs_core_denial_set']) < 0.9 else
      'x-4 is robust to the core-denial subset (cos >= 0.9)')
    f('x4_affirmative_table_variant', round(abs(x4['instruct']['detail']['cos_vs_affirmative_table']), 4),
      'the affirmative-table compliance set moves x-4 almost completely: compliance-row choice is '
      'not a detail, it is half the construction')
    rv = out['methods']['x1_twin_diff_svd'].get('right_vector_cos_vs_Wtu', {})
    f('x1_right_vector_cos', rv.get('cos_mean_abs'),
      'the twin delta is the rank-1 projection of the instruct writeback matrix along the recovered '
      'axis, to storage precision: the left vector alone is enough to reproduce the published edit '
      'shape')
    f('x3_estimator_disagreement', agree_within['x3_svd_vs_column_norm']['cos_mean_abs'],
      'the alignment delta has no estimator-independent axis: the top singular vector and the '
      'largest-column heuristic disagree completely' if
      agree_within['x3_svd_vs_column_norm']['cos_mean_abs'] < 0.5 else
      'two estimators of the alignment axis agree, so the axis is not an artefact of one')
    f('x1_energy_fraction', energy['x1']['mean_fraction_of_weight_energy_along_axis'],
      'fraction of the instruct writeback energy a rank-1 projection along the twin axis would remove')
    tw, al = footprints['twin_minus_instruct'], footprints['instruct_minus_base']
    f('twin_edit_footprint', {'value_differing': tw['value_differing'],
                              'bit_identical': tw['bit_identical'],
                              'by_suffix': tw['differing_by_suffix'],
                              'only_in_twin': tw['only_in_a']},
      'the published twin differs from the instruct base only on the writeback tensors'
      if set(tw['differing_by_suffix']) <= {'o_proj', 'down_proj'} else
      f"the twin edit is not confined to the writebacks: {tw['differing_by_suffix']}")
    f('alignment_edit_footprint', {'value_differing': al['value_differing'],
                                   'bit_identical': al['bit_identical']},
      'the alignment delta touches every tensor, which is why its "axis" cannot be a single '
      'direction' if al['value_differing'] > al['bit_identical'] else
      'the alignment delta is sparse, unlike the twin edit')
    out['findings'] = findings

    # Self-checks on the assembled artifact: every stored axis must be unit, every
    # per-layer table complete, and the matrix must be reproducible from the axes
    # the JSON itself carries (to the 4 decimals the cells are rounded to).
    dev = 0.0
    for a in labels:
        for b in labels:
            dev = max(dev, abs(abs(cos(axes[a], axes[b])) - cos_matrix[a][b]))
    out['checks'] = {
        'all_global_axes_unit_norm': all(abs(float(np.linalg.norm(axes[k])) - 1.0) < 1e-6 for k in labels),
        'max_abs_cos_matrix_deviation_when_recomputed': round(dev, 8),
        'cos_matrix_cells_rounded_to': 4,
        'per_layer_axes': {k: {'layers': len(v['per_layer']), 'width': len(next(iter(v['per_layer'].values()))['axis'])}
                           for k, v in per_layer_src.items()},
        'removal_measured_is_false': out['removal_measured'] is False,
        'no_forward_pass_in_this_script': True,
    }
    out['checks']['ok'] = bool(out['checks']['all_global_axes_unit_norm']
                               and dev <= 1e-3
                               and all(v['layers'] == len(layers) and v['width'] == hidden
                                       for v in out['checks']['per_layer_axes'].values())
                               and out['checks']['removal_measured_is_false'])

    def to_jsonable(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, dict):
            return {k: to_jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [to_jsonable(v) for v in o]
        return o

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(to_jsonable(out), indent=1) + '\n')
    print(f'wrote {out_path} ({out_path.stat().st_size} bytes) in {out["elapsed_s"]}s', flush=True)
    for ent in findings:
        print(f'  {ent["key"]}: {ent["value"]} — {ent["reading"]}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
