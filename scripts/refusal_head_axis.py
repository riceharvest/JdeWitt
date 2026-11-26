# SPDX-License-Identifier: AGPL-3.0-or-later
"""Zero-forward refusal axis from output-head geometry, measured on a real pair.

Usage:
  python scripts/refusal_head_axis.py \
      --base Qwen/Qwen2.5-0.5B-Instruct@7ae55760 \
      --twin Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1@a42fe88f \
      --alignment-base Qwen/Qwen2.5-0.5B@060db649 \
      --gguf <path>/Qwen2.5-0.5B-Instruct-IQ3_M.gguf \
      --cache-dir <hf cache> --delete-downloads \
      --out verification/refusal-head-axis/report.json

What runs: safetensors reads, one GGUF dequantisation of the tied head, one
tokenizer encode per surface, and numpy matrix algebra. No model is instantiated,
no token is generated, nothing is served or benchmarked; the report carries
`removal_measured: false` because a direction plus an exact projection identity
is not a behavioural result.

The writeback pass reads each layer's o_proj and down_proj ONCE per checkpoint
and derives every number from that read: per-model edit norms, the twin's
recovered abduction direction, and the alignment delta against the pretrained
base. The read is the expensive part (bf16 weights, shared disk), so the pass is
single, not one per question.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

# The board's CPU is shared: a 64-thread BLAS on 896-wide products spends more
# time contending than computing. Set before numpy is imported.
for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '4')

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from jdewitt.refusal_head import (COMPLIANCE_WORDS, DENIAL_WORDS, cosine, edit_stats,
                                  head_axis, head_channel_closed, margin_reduction,
                                  resolve_surfaces, table_ids, top_left_direction)

DENIAL_TABLE = REPO / 'jdewitt' / 'assets' / 'vocab-remap-denial.json'
WRITEBACK = (('attn', 'self_attn.o_proj'), ('ffn', 'mlp.down_proj'))
WANTED_AXES = ('twin-abduction', 'twin_abduction', 'alignment-delta', 'alignment_delta')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b''):
            digest.update(chunk)
    return digest.hexdigest()


def r(value, digits=6):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return float(f'{float(value):.{digits}g}')


def split_ref(text: str):
    repo, _, rev = text.partition('@')
    return repo, (rev or 'main')


def fetch(repo: str, revision: str, cache: Path):
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(repo, 'model.safetensors', revision=revision, cache_dir=str(cache)))
    return path, sha256_file(path)


def config_of(repo: str, revision: str, cache: Path) -> dict:
    from huggingface_hub import hf_hub_download
    return json.loads(Path(hf_hub_download(repo, 'config.json', revision=revision,
                                          cache_dir=str(cache))).read_text())


def head_rows_for(path: Path, ids, *, tied: bool):
    """Rows of the SERVED output head for `ids`, plus which tensor they came from.

    A checkpoint can save an lm_head tensor while config.json still says
    tie_word_embeddings: the loader then discards that tensor and scores with the
    embedding, so reading lm_head would build an axis the served model never
    uses. `tied` comes from the config, not from key presence.

    get_slice keeps this at the size of the resolved vocabulary rows: the full
    head is 151936 x 896 and never needs to be resident.
    """
    from safetensors import safe_open
    with safe_open(str(path), framework='pt') as handle:
        keys = set(handle.keys())
        table = ('model.embed_tokens.weight' if tied or 'lm_head.weight' not in keys
                 else 'lm_head.weight')
        rows = np.asarray(handle.get_slice(table)[np.asarray(ids, dtype=np.int64)].float(),
                          dtype=np.float64)
    return rows, table


def tie_check(path: Path) -> dict:
    from safetensors import safe_open
    with safe_open(str(path), framework='pt') as handle:
        if 'lm_head.weight' not in set(handle.keys()):
            return {'lm_head_saved': False, 'tied_to_embedding': True,
                    'note': 'no lm_head tensor: the head IS model.embed_tokens.weight'}
        head = handle.get_tensor('lm_head.weight').float()
        emb = handle.get_tensor('model.embed_tokens.weight').float()
        diff = float((head - emb).abs().max()) if tuple(head.shape) == tuple(emb.shape) else None
        return {'lm_head_saved': True, 'tied_to_embedding': bool(diff == 0.0),
                'max_abs_lm_head_minus_embed': diff, 'shape': list(head.shape)}


def head_variants(tokenizer, table_entries):
    """Three inventories, so the table's influence is measured and not hidden."""
    single = [form for form in (entry['form'] for entry in table_entries)
              if len(tokenizer.encode(form, add_special_tokens=False)) == 1]
    spaced = [' ' + form for form in single
              if len(tokenizer.encode(' ' + form, add_special_tokens=False)) == 1]
    all_surfaces = [entry['form'] for entry in table_entries]

    named = resolve_surfaces(tokenizer, DENIAL_WORDS)
    comp = resolve_surfaces(tokenizer, COMPLIANCE_WORDS)
    primary = resolve_surfaces(tokenizer, DENIAL_WORDS, extra_surfaces=tuple(single + spaced))
    wide = list(named['ids'])
    for token in table_ids(tokenizer, all_surfaces, first_token_only=False):
        if token not in wide:
            wide.append(token)

    used = sorted(set(named['ids'] + comp['ids'] + primary['ids'] + wide))
    tokens = tokenizer.convert_ids_to_tokens(used)
    return {
        'named': named, 'compliance': comp, 'primary': primary, 'wide_denial_ids': wide,
        'table_single_token_forms': single + spaced,
        'overlap': {
            'named': sorted(set(named['ids']) & set(comp['ids'])),
            'primary': sorted(set(primary['ids']) & set(comp['ids'])),
            'wide': sorted(set(wide) & set(comp['ids'])),
        },
        'vocabulary_size': len(tokenizer),
        'tokens': {str(i): token for i, token in zip(used, tokens)},
    }


def axis_block(tokenizer, head_rows, positions, denial_ids, compliance_ids, label):
    """Slice the resolved rows out of the head block and build the axis."""
    out = head_axis(head_rows[[positions[int(i)] for i in denial_ids]],
                    head_rows[[positions[int(i)] for i in compliance_ids]])
    return {'label': label,
            'n_denial': out['n_denial'], 'n_compliance': out['n_compliance'],
            'denial_ids': [int(i) for i in denial_ids],
            'denial_tokens': tokenizer.convert_ids_to_tokens([int(i) for i in denial_ids]),
            'compliance_tokens': tokenizer.convert_ids_to_tokens([int(i) for i in compliance_ids]),
            'd_norm': r(float(np.linalg.norm(out['d'])), 12),
            'margin_per_unit_d': r(out['margin']),
            'row_norms': {k: r(v) for k, v in out['row_norms'].items()},
            'd': out['d'], 'g': out['g']}


def _load(handle, name):
    return np.asarray(handle.get_tensor(name).float(), dtype=np.float64)


def compute_pass(base_path: Path, twin_path: Path, align_path, axes: dict, layers, *, seed=0):
    """One read per tensor per checkpoint; every number below comes from it."""
    from safetensors import safe_open
    rng = np.random.default_rng(seed)
    probes: dict[int, np.ndarray] = {}
    edit_rows = {'aligned': [], 'twin': []}
    margin_rows = {'aligned': [], 'twin': []}
    delta_rows = {'twin': [], 'alignment': []}

    with ExitStack() as stack:
        base = stack.enter_context(safe_open(str(base_path), framework='pt'))
        twin = stack.enter_context(safe_open(str(twin_path), framework='pt'))
        alignment = (stack.enter_context(safe_open(str(align_path), framework='pt'))
                     if align_path is not None else None)
        for index, layer in enumerate(layers):
            for kind, hf in WRITEBACK:
                name = f'model.layers.{layer}.{hf}.weight'
                Wb, Wt = _load(base, name), _load(twin, name)
                for label, W in (('aligned', Wb), ('twin', Wt)):
                    axis = axes[label]
                    stats = edit_stats(W.T, axis['d'])
                    if W.shape[1] not in probes:
                        probes[W.shape[1]] = rng.normal(size=W.shape[1])
                    margin = margin_reduction(axis['g'], axis['d'], W.T, probes[W.shape[1]])
                    edit_rows[label].append({
                        'layer': int(layer), 'module': kind, 'tensor': name,
                        'shape_in_out': stats['shape_in_out'],
                        'norm_fro': r(stats['norm_fro']),
                        'delta_norm_fro': r(stats['delta_norm_fro']),
                        'relative_edit_norm': r(stats['relative_edit_norm'], 5),
                        'residual_identity_abs': r(stats['residual_identity_abs'], 3),
                        'residual_identity_ok': bool(stats['residual_identity_ok']),
                        'output_along_d_max': r(stats['output_along_d_max'], 3)})
                    margin_rows[label].append({
                        'layer': int(layer), 'module': kind,
                        'identity_abs': r(margin['identity_abs'], 3),
                        'identity_ok': bool(margin['identity_ok']),
                        'g_parallel_to_d': r(margin['g_parallel_to_d'], 9),
                        'reduction_over_before': r(abs(margin['reduction']) /
                                                   max(abs(margin['before']), 1e-12), 4)})
                pair = [('twin', Wt, axes['twin'])]
                if alignment is not None:
                    pair.append(('alignment', _load(alignment, name), axes['aligned']))
                for label, other, d in pair:
                    delta = other - Wb            # HF [out, in]: rows are residual-side
                    norm = float(np.linalg.norm(delta))
                    if norm <= 0.0:
                        delta_rows[label].append({'layer': int(layer), 'module': kind,
                                                  'touched': False, 'delta_norm_fro': 0.0})
                        continue
                    top = top_left_direction(delta)
                    if top['direction'].shape[0] != d.shape[0]:
                        raise SystemExit(f'{name}: recovered direction width '
                                         f'{top["direction"].shape[0]} != residual width {d.shape[0]}')
                    share = float(np.linalg.norm(delta.T @ d) ** 2) / (norm ** 2)
                    delta_rows[label].append({
                        'layer': int(layer), 'module': kind, 'touched': True,
                        'delta_norm_fro': r(norm),
                        'cos_d_vs_delta_top': r(cosine(d, top['direction']), 5),
                        'delta_energy_captured_by_d': r(share, 4),
                        'delta_energy_fraction_top': r(top['energy_fraction_top'], 4),
                        'delta_effective_rank': r(top['effective_rank'], 3)})
                del Wb, Wt
            if index and index % 6 == 0:
                print(f'[head-axis]   layer {layer}: {index * 2} tensors paired', flush=True)

    return {
        'edit': {label: _aggregate_edit(edit_rows[label], margin_rows[label])
                 for label in edit_rows},
        'deltas': {label: _aggregate_delta(rows) for label, rows in delta_rows.items()},
    }


def _aggregate_edit(rows, margin_rows):
    relative = [row['relative_edit_norm'] for row in rows]
    by_kind: dict[str, list[float]] = {}
    for row in rows:
        by_kind.setdefault(row['module'], []).append(row['relative_edit_norm'])
    return {
        'tensors': len(rows),
        'per_tensor': rows,
        'relative_edit_norm': {
            'all_mean': r(float(np.mean(relative))), 'all_min': r(float(np.min(relative))),
            'all_max': r(float(np.max(relative))),
            **{kind: {'mean': r(float(np.mean(vals))), 'min': r(float(np.min(vals))),
                      'max': r(float(np.max(vals)))} for kind, vals in by_kind.items()}},
        'identities': {
            'unit_d_ok': True,
            'residual_identity_all_ok': all(row['residual_identity_ok'] for row in rows),
            'residual_identity_max_abs': r(max(row['residual_identity_abs'] for row in rows), 3),
            'margin_identity_all_ok': all(row['identity_ok'] for row in margin_rows),
            'margin_identity_max_abs': r(max(row['identity_abs'] for row in margin_rows), 3),
            'margin_identity_min_parallel': r(min(row['g_parallel_to_d'] for row in margin_rows), 9),
            'margin_per_tensor': margin_rows,
        },
    }


def _aggregate_delta(rows):
    touched = [row for row in rows if row['touched']]
    cosines = [row['cos_d_vs_delta_top'] for row in touched]
    captured = [row['delta_energy_captured_by_d'] for row in touched]
    by_module: dict[str, list[float]] = {}
    for row in touched:
        by_module.setdefault(row['module'], []).append(row['cos_d_vs_delta_top'])
    return {
        'method': 'delta = other - base per writeback tensor in HF [out, in]; the top eigenvector '
                  'of delta delta^T is the residual direction the delta moved, and '
                  '||delta^T d||^2/||delta||^2 is the energy of the delta along d',
        'tensors_compared': len(rows), 'tensors_touched': len(touched),
        'signed_mean_cos': r(float(np.mean(cosines)), 5) if cosines else None,
        'mean_abs_cos': r(float(np.mean(np.abs(cosines))), 5) if cosines else None,
        'median_abs_cos': r(float(np.median(np.abs(cosines))), 5) if cosines else None,
        'max_abs_cos': r(float(np.max(np.abs(cosines))), 5) if cosines else None,
        'by_module_mean_abs_cos': {kind: r(float(np.mean(np.abs(vals))), 5)
                                   for kind, vals in by_module.items()},
        'delta_energy_captured_by_d_mean': r(float(np.mean(captured)), 4) if captured else None,
        'delta_energy_fraction_top_mean': r(float(np.mean(
            [row['delta_energy_fraction_top'] for row in touched])), 4) if touched else None,
        'delta_effective_rank_median': r(float(np.median(
            [row['delta_effective_rank'] for row in touched])), 3) if touched else None,
        'per_tensor': rows,
    }


def gguf_head(path: Path, ids):
    """The stand-in's tied head is Q8_0; dequantise it and check its token strings."""
    from gguf import GGUFReader
    from jdewitt.ggufio import dequantize_tensor
    reader = GGUFReader(str(path))
    names = {tensor.name for tensor in reader.tensors}
    source = 'output.weight' if 'output.weight' in names else 'token_embd.weight'
    tokens = []
    for part in reader.fields['tokenizer.ggml.tokens'].parts:
        tokens.extend(part.tolist())
    full = np.asarray(dequantize_tensor(str(path), source, reader=reader), dtype=np.float32)
    if full.shape[0] != len(tokens):
        full = full.T
    if full.shape[0] != len(tokens):
        raise SystemExit(f'GGUF head {source} shape {full.shape} does not carry {len(tokens)} rows')
    return full, source, tokens


def token_mismatches(tokenizer, tokens, ids):
    out = []
    for token_id in ids:
        want = tokenizer.convert_ids_to_tokens(int(token_id))
        got = tokens[int(token_id)] if int(token_id) < len(tokens) else None
        if want != got:
            out.append({'id': int(token_id), 'hf': want, 'gguf': got})
    return out


def artifact_axes(path: Path, d: np.ndarray, label: str | None = None):
    """Cosine of d against a 'twin-abduction'-style JSON, if one is given."""
    doc = json.loads(path.read_text())
    algebra = doc.get('weight_algebra') or {}
    shared = algebra.get('shared_axis')
    out = {'path': label or path.name, 'sha256': sha256_file(path),
           'schema': doc.get('schema'), 'removal_measured': doc.get('removal_measured'),
           'provenance': 'read from the primary checkout working tree; the sha256 identifies the '
                         'exact file and the committed path is repo-relative, so no lab path '
                         'travels with the number'}
    if shared is not None and len(shared) == d.shape[0]:
        out['cos_d_vs_shared_axis'] = r(cosine(d, np.asarray(shared, dtype=np.float64)), 5)
    for key, value in (algebra.get('module_axes') or {}).items():
        axis = value.get('axis')
        if axis is not None and len(axis) == d.shape[0]:
            out.setdefault('cos_d_vs_module_axes', {})[key] = r(
                cosine(d, np.asarray(axis, dtype=np.float64)), 5)
    checks = doc.get('checks') or {}
    out['artifact_own_checks'] = {k: checks[k] for k in (
        'axis_unit_norm', 'all_writeback_rank1_gt_0.9', 'mean_abs_cos_u1_to_axis')
        if k in checks}
    return out


def find_axis_artifacts(root: Path, width: int):
    """Existing fitted axes in the clone, and whether a cosine is even defined."""
    def walk(node, hit):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == 'shape' and isinstance(value, list) and len(value) == 2 \
                        and all(isinstance(v, int) for v in value) and hit:
                    return value
                found = walk(value, hit or 'axes' in str(key).lower())
                if found:
                    return found
        elif isinstance(node, list):
            for value in node:
                found = walk(value, hit)
                if found:
                    return found
        return None

    artifacts, wanted = [], []
    for path in sorted((root / 'verification').rglob('*.json')):
        if path.stat().st_size > 4_000_000:
            continue
        low = path.name.lower()
        if any(name in low for name in WANTED_AXES):
            wanted.append(str(path.relative_to(root)))
        try:
            doc = json.loads(path.read_text())
        except Exception:
            continue
        shape = walk(doc, 'axes' in path.stem.lower())
        if shape:
            artifacts.append({'path': str(path.relative_to(root)), 'shape': shape,
                              'comparable_in_width': shape[1] == width})
    return {
        'wanted_names_searched': list(WANTED_AXES),
        'wanted_names_found': wanted,
        'axis_artifacts': artifacts,
        'note': 'a cosine needs the same width AND the same checkpoint; an axis fitted on another '
                'model is reported as not comparable rather than silently compared',
    }


def parse_layers(spec: str, n_layers: int):
    if spec == 'all':
        return list(range(n_layers))
    first, _, last = spec.partition('-')
    return list(range(int(first), int(last or first) + 1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--base', required=True, help='repo@revision of the aligned model')
    ap.add_argument('--twin', required=True, help='repo@revision of the abliterated twin')
    ap.add_argument('--alignment-base', default=None,
                    help='repo@revision of the pretrained base, to derive the alignment-delta axis')
    ap.add_argument('--twin-axis-json', default=None,
                    help="optional 'twin-abduction'-style JSON to cosine against")
    ap.add_argument('--gguf', default=None, help='optional quantized stand-in artifact')
    ap.add_argument('--out', required=True, help='report JSON path')
    ap.add_argument('--cache-dir', required=True, help='HF cache directory')
    ap.add_argument('--delete-downloads', action='store_true',
                    help='hash, read, then delete the safetensors')
    ap.add_argument('--layers', default='all', help="'all' or e.g. 0-23")
    args = ap.parse_args(argv)

    started = time.monotonic()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir).expanduser()
    base_repo, base_rev = split_ref(args.base)
    twin_repo, twin_rev = split_ref(args.twin)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_repo, revision=base_rev, cache_dir=str(cache))
    base_cfg = config_of(base_repo, base_rev, cache)
    twin_cfg = config_of(twin_repo, twin_rev, cache)
    layers = parse_layers(args.layers, int(base_cfg['num_hidden_layers']))

    table = json.loads(DENIAL_TABLE.read_text())['entries']
    variants = head_variants(tokenizer, table)
    ids_all = sorted(set(variants['named']['ids'] + variants['compliance']['ids']
                         + variants['primary']['ids'] + variants['wide_denial_ids']))
    positions = {int(token_id): index for index, token_id in enumerate(ids_all)}
    print(f'[head-axis] {base_repo}@{base_rev} vocab={len(tokenizer)} layers={len(layers)} '
          f'denial(primary)={len(variants["primary"]["ids"])} '
          f'compliance={len(variants["compliance"]["ids"])}', flush=True)

    base_path, base_sha = fetch(base_repo, base_rev, cache)
    twin_path, twin_sha = fetch(twin_repo, twin_rev, cache)
    align_path = align_sha = None
    if args.alignment_base:
        align_repo, align_rev = split_ref(args.alignment_base)
        if int(config_of(align_repo, align_rev, cache)['num_hidden_layers']) != len(layers):
            raise SystemExit('alignment base has a different layer count')
        align_path, align_sha = fetch(align_repo, align_rev, cache)
    print(f'[head-axis] hashed base {base_sha[:16]} twin {twin_sha[:16]} '
          f'alignment {None if align_sha is None else align_sha[:16]}', flush=True)

    models = {}
    for label, path, digest, repo, rev, cfg in (
            ('aligned', base_path, base_sha, base_repo, base_rev, base_cfg),
            ('twin', twin_path, twin_sha, twin_repo, twin_rev, twin_cfg)):
        rows, head_name = head_rows_for(path, ids_all,
                                        tied=bool(cfg.get('tie_word_embeddings', True)))
        axes = {
            'named': axis_block(tokenizer, rows, positions, variants['named']['ids'],
                                variants['compliance']['ids'], 'named words only'),
            'primary': axis_block(tokenizer, rows, positions, variants['primary']['ids'],
                                  variants['compliance']['ids'],
                                  'named + single-token table surfaces'),
            'wide': axis_block(tokenizer, rows, positions, variants['wide_denial_ids'],
                               variants['compliance']['ids'], 'named + every table id'),
        }
        axis = axes['primary']
        channel = head_channel_closed(axis['g'], axis['d'],
                                      np.random.default_rng(7).normal(size=axis['d'].shape[0]))
        print(f'[head-axis] {label}: head={head_name} '
              f'margin={axis["margin_per_unit_d"]}', flush=True)
        models[label] = {
            'repo': repo, 'revision': rev,
            'safetensors': {'sha256': digest, 'bytes': path.stat().st_size,
                            'deleted_after_read': bool(args.delete_downloads)},
            'head_tensor': head_name,
            'config_tie_word_embeddings': bool(cfg.get('tie_word_embeddings', True)),
            'tie': tie_check(path),
            'axes': {key: {k: v for k, v in value.items() if k != 'g'}
                     for key, value in axes.items()},
            'axis_vector_primary': [r(x, 9) for x in axis['d']],
            '_vectors': {key: (value['d'], value['g']) for key, value in axes.items()},
            'checks': {
                'head_channel_closed': bool(channel['closed']),
                'head_margin_after_projection': r(channel['margin_after'], 3),
                'g_parallel_to_d': r(channel['g_parallel_to_d'], 12),
                'margin_per_unit_d': r(channel['margin_per_unit_d']),
                'axis_cosines': {
                    'named_vs_primary': r(cosine(axes['named']['d'], axes['primary']['d']), 5),
                    'primary_vs_wide': r(cosine(axes['primary']['d'], axes['wide']['d']), 5),
                    'named_vs_wide': r(cosine(axes['named']['d'], axes['wide']['d']), 5),
                },
            },
        }
        del rows

    axes_d = {label: models[label]['_vectors']['primary'][0] for label in models}
    passes = compute_pass(base_path, twin_path, align_path, axes_d, layers)
    for label in models:
        models[label]['edit'] = passes['edit'][label]
    print('[head-axis] paired pass done', flush=True)

    cross = {
        'cos_aligned_vs_twin_primary': r(cosine(axes_d['aligned'], axes_d['twin']), 5),
        'cos_aligned_vs_twin_named': r(cosine(models['aligned']['_vectors']['named'][0],
                                              models['twin']['_vectors']['named'][0]), 5),
    }
    twin = passes['deltas']['twin']
    twin['label'] = 'twin abduction (twin - base)'
    alignment = None
    if align_path is not None:
        alignment = passes['deltas']['alignment']
        alignment['label'] = 'alignment delta (instruct - base)'
        alignment['base'] = {'repo': split_ref(args.alignment_base)[0],
                             'revision': split_ref(args.alignment_base)[1],
                             'safetensors_sha256': align_sha,
                             'bytes': align_path.stat().st_size,
                             'deleted_after_read': bool(args.delete_downloads)}
    committed = None
    if args.twin_axis_json:
        given = Path(args.twin_axis_json)
        label = ('verification/twin-abduction/' + given.name
                 if given.parent.name == 'twin-abduction' else given.name)
        committed = artifact_axes(given, axes_d['aligned'], label)

    gguf_block = None
    if args.gguf:
        full, source, tokens = gguf_head(Path(args.gguf), ids_all)
        gguf_rows = np.asarray(full[[positions[int(i)] for i in ids_all]], dtype=np.float64)
        ga = head_axis(gguf_rows[[positions[int(i)] for i in variants['primary']['ids']]],
                       gguf_rows[[positions[int(i)] for i in variants['compliance']['ids']]])
        gguf_block = {
            'path': str(args.gguf), 'sha256': sha256_file(Path(args.gguf)),
            'bytes': Path(args.gguf).stat().st_size,
            'head_tensor': source, 'head_shape': [int(x) for x in full.shape],
            'margin_per_unit_d': r(ga['margin']),
            'cos_vs_aligned_primary': r(cosine(ga['d'], axes_d['aligned']), 5),
            'cos_vs_twin_primary': r(cosine(ga['d'], axes_d['twin']), 5),
            'token_string_mismatches': token_mismatches(tokenizer, tokens, ids_all),
        }
        print(f'[head-axis] gguf cos vs aligned {gguf_block["cos_vs_aligned_primary"]} '
              f'token mismatches {len(gguf_block["token_string_mismatches"])}', flush=True)

    report = {
        'schema': 'jdewitt.refusal-head-axis/1',
        'date_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'question': 'does the output head alone name the direction an abliteration removes, '
                    'and what does removing it cost per writeback tensor?',
        'zero_forward': True,
        'removal_measured': False,
        'method': {
            'axis': 'd = normalize(mean(E_denial) - mean(E_compliance)); E = the served output-head '
                    'rows (the embedding when config ties them)',
            'edit': "W' = W (I - d d^T) for self_attn.o_proj and mlp.down_proj in GGUF [in, out] "
                    'order, float64; the stored delta is W d d^T',
            'identities': "d unit; ||W(I-dd^T)-W||_F = ||Wd||_2; margin m-m' = "
                          "(W_out^T u . d) ((Wd) . a)",
            'no_forward': 'no model object, no generate, no capture, no benchmark, no GPU work',
        },
        'tokenizer': {
            'repo': base_repo, 'revision': base_rev, 'vocab_size': variants['vocabulary_size'],
            'denial_words': list(DENIAL_WORDS), 'compliance_words': list(COMPLIANCE_WORDS),
            'table_path': str(DENIAL_TABLE.relative_to(REPO)),
            'table_single_token_forms': variants['table_single_token_forms'],
            'denial_ids': {'named': variants['named']['ids'],
                           'primary': variants['primary']['ids'],
                           'wide': variants['wide_denial_ids']},
            'compliance_ids': variants['compliance']['ids'],
            'denial_overlap_with_compliance': variants['overlap'],
            'by_word': variants['primary']['by_word'],
            'multi_token_not_admitted': variants['primary']['multi_token'],
            'resolved_tokens': variants['tokens'],
        },
        'models': {label: {k: v for k, v in doc.items() if k != '_vectors'}
                   for label, doc in models.items()},
        'cross_checks': {
            'cos_aligned_vs_twin': cross,
            'twin_recovered_edit': twin,
            'alignment_delta': alignment,
            'twin_abduction_artifact': committed,
            'gguf_stand_in': gguf_block,
            'repo_axis_artifacts': find_axis_artifacts(REPO, int(base_cfg['hidden_size'])),
        },
        'claims': {
            'measured': 'per-tensor edit norms and the three identities, from the weights and the '
                        'tokenizer alone',
            'not_measured': 'whether refusal behaviour changes; this card runs no model',
            'comparison_limit': 'the twin and alignment directions are weight deltas, not '
                                'behavioural axes; a high cosine says the head geometry agrees '
                                'with an edit, not that removing d reproduces that edit',
        },
        'wall_seconds': round(time.monotonic() - started, 1),
        'next': ['serve edited weights and run a refusal wave (GATED: needs a GPU and a judge)',
                 'compare the head axis to a capture-fitted axis on the same checkpoint'],
    }
    out.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({
        'out': str(out),
        'margin': {label: models[label]['axes']['primary']['margin_per_unit_d'] for label in models},
        'cos_aligned_vs_twin': cross,
        'twin': {k: twin[k] for k in ('mean_abs_cos', 'max_abs_cos',
                                      'delta_energy_captured_by_d_mean',
                                      'delta_effective_rank_median')},
        'alignment': None if alignment is None else
        {k: alignment[k] for k in ('mean_abs_cos', 'max_abs_cos',
                                   'delta_energy_captured_by_d_mean')},
        'twin_artifact_cos': None if committed is None else committed.get('cos_d_vs_shared_axis'),
        'relative_edit_norm': {label: models[label]['edit']['relative_edit_norm']
                               for label in models},
        'identities_ok': {label: (models[label]['edit']['identities']['residual_identity_all_ok']
                                  and models[label]['edit']['identities']['margin_identity_all_ok'])
                          for label in models},
        'gguf_cos_vs_aligned': None if gguf_block is None else gguf_block['cos_vs_aligned_primary'],
        'wall_seconds': report['wall_seconds'],
    }, indent=1), flush=True)

    if args.delete_downloads:
        for path in [p for p in (base_path, twin_path, align_path) if p is not None]:
            print(f'[head-axis] deleting {path}', flush=True)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
