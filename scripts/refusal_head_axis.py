# SPDX-License-Identifier: AGPL-3.0-or-later
"""Zero-forward refusal axis from output-head geometry, measured on a real pair.

Usage:
  python scripts/refusal_head_axis.py \
      --base Qwen/Qwen2.5-0.5B-Instruct@7ae55760 \
      --twin Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1@a42fe88f \
      --gguf <path>/Qwen2.5-0.5B-Instruct-IQ3_M.gguf \
      --out verification/refusal-head-axis/report.json --delete-downloads

What runs: two safetensors reads, one GGUF dequantisation of the tied head, one
tokenizer encode per surface, and numpy matrix algebra. No model is instantiated,
no token is generated, nothing is served or benchmarked; the report carries
`removal_measured: false` because a direction plus an exact projection identity
is not a behavioural result.

Every number in the report is recomputed from the files named there; the only
values that are not local arithmetic are the repositories' revisions and the
sha256 of their safetensors, both recorded so the run can be repeated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
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


def head_rows_for(path: Path, ids):
    """Rows of the output head for `ids`, plus which tensor they came from.

    get_slice keeps this at the size of the resolved vocabulary rows: the full
    head is 151936 x 896 and never needs to be resident.
    """
    from safetensors import safe_open
    with safe_open(str(path), framework='pt') as handle:
        keys = set(handle.keys())
        table = 'lm_head.weight' if 'lm_head.weight' in keys else 'model.embed_tokens.weight'
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


def axis_block(tokenizer, head_rows, denial_ids, compliance_ids, label):
    out = head_axis(head_rows, denial_ids, compliance_ids)
    return {'label': label,
            'n_denial': out['n_denial'], 'n_compliance': out['n_compliance'],
            'denial_ids': [int(i) for i in denial_ids],
            'denial_tokens': tokenizer.convert_ids_to_tokens([int(i) for i in denial_ids]),
            'compliance_tokens': tokenizer.convert_ids_to_tokens([int(i) for i in compliance_ids]),
            'margin_per_unit_d': r(out['margin']),
            'row_norms': {k: r(v) for k, v in out['row_norms'].items()},
            'd': out['d'], 'g': out['g']}


def writeback_tensors(path: Path, layers):
    """One writeback tensor at a time, transposed into GGUF [in, out] order."""
    from safetensors import safe_open
    with safe_open(str(path), framework='pt') as handle:
        for layer in layers:
            for kind, hf in WRITEBACK:
                name = f'model.layers.{layer}.{hf}.weight'
                yield layer, kind, name, np.asarray(handle.get_tensor(name).float(),
                                                    dtype=np.float64).T


def run_model(path: Path, axis: dict, layers, *, seed=0):
    """Edit norms and the two exact identities, per writeback tensor."""
    rng = np.random.default_rng(seed)
    probes: dict[int, np.ndarray] = {}
    rows, margin_rows, relative = [], [], []
    for layer, kind, name, W in writeback_tensors(path, layers):
        stats = edit_stats(W, axis['d'])
        if W.shape[0] not in probes:
            probes[W.shape[0]] = rng.normal(size=W.shape[0])
        margin = margin_reduction(axis['g'], axis['d'], W, probes[W.shape[0]])
        rows.append({'layer': int(layer), 'module': kind, 'tensor': name,
                     'shape_in_out': stats['shape_in_out'],
                     'norm_fro': r(stats['norm_fro']),
                     'delta_norm_fro': r(stats['delta_norm_fro']),
                     'relative_edit_norm': r(stats['relative_edit_norm'], 5),
                     'residual_identity_abs': r(stats['residual_identity_abs'], 3),
                     'residual_identity_ok': bool(stats['residual_identity_ok']),
                     'output_along_d_max': r(stats['output_along_d_max'], 3)})
        margin_rows.append({'layer': int(layer), 'module': kind,
                            'identity_abs': r(margin['identity_abs'], 3),
                            'identity_ok': bool(margin['identity_ok']),
                            'reduction_over_before': r(
                                abs(margin['reduction']) / max(abs(margin['before']), 1e-12), 4)})
        relative.append(stats['relative_edit_norm'])
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
            'residual_identity_all_ok': all(row['residual_identity_ok'] for row in rows),
            'residual_identity_max_abs': r(max(row['residual_identity_abs'] for row in rows), 3),
            'margin_identity_all_ok': all(row['identity_ok'] for row in margin_rows),
            'margin_identity_max_abs': r(max(row['identity_abs'] for row in margin_rows), 3),
            'margin_per_tensor': margin_rows,
        },
    }


def twin_axis(base_path: Path, twin_path: Path, d: np.ndarray, layers):
    """Recover what the abliterated twin removed, and its cosine with d.

    The card names a 'twin-abduction axis' artifact; none exists in this clone,
    so the axis is recovered from the pair itself: delta = twin - base in
    [in, out] order, top left singular vector per tensor, plus the rank-1 share
    so a non-rank-1 delta is reported as such instead of being called the axis.
    """
    from safetensors import safe_open
    rows = []
    with safe_open(str(base_path), framework='pt') as base, \
            safe_open(str(twin_path), framework='pt') as twin:
        for layer in layers:
            for kind, hf in WRITEBACK:
                name = f'model.layers.{layer}.{hf}.weight'
                delta = (np.asarray(twin.get_tensor(name).float(), dtype=np.float64)
                         - np.asarray(base.get_tensor(name).float(), dtype=np.float64)).T
                norm = float(np.linalg.norm(delta))
                if norm <= 0.0:
                    rows.append({'layer': int(layer), 'module': kind, 'touched': False,
                                 'delta_norm_fro': 0.0})
                    continue
                top = top_left_direction(delta)
                rows.append({'layer': int(layer), 'module': kind, 'touched': True,
                             'delta_norm_fro': r(norm),
                             'cos_d_vs_twin_top': r(cosine(d, top['direction']), 5),
                             'twin_energy_fraction_top': r(top['energy_fraction_top'], 4),
                             'twin_effective_rank': r(top['effective_rank'], 3)})
    touched = [row for row in rows if row['touched']]
    cosines = [row['cos_d_vs_twin_top'] for row in touched]
    by_module: dict[str, list[float]] = {}
    for row in touched:
        by_module.setdefault(row['module'], []).append(row['cos_d_vs_twin_top'])
    return {
        'method': 'delta = twin - base per writeback tensor in [in, out]; the top eigenvector of '
                  'delta delta^T is the residual direction the twin removed',
        'tensors_compared': len(rows), 'tensors_touched': len(touched),
        'signed_mean_cos': r(float(np.mean(cosines)), 5) if cosines else None,
        'mean_abs_cos': r(float(np.mean(np.abs(cosines))), 5) if cosines else None,
        'median_abs_cos': r(float(np.median(np.abs(cosines))), 5) if cosines else None,
        'max_abs_cos': r(float(np.max(np.abs(cosines))), 5) if cosines else None,
        'by_module_mean_abs_cos': {kind: r(float(np.mean(np.abs(vals))), 5)
                                   for kind, vals in by_module.items()},
        'per_tensor': rows,
    }


def gguf_head(path: Path, ids, ids_used):
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
    if full.shape[0] != len(tokens) and full.shape[1] != len(tokens):
        raise SystemExit(f'GGUF head {source} shape {full.shape} does not carry {len(tokens)} token rows')
    rows = np.asarray(full[np.asarray(ids, dtype=np.int64)], dtype=np.float64)
    return rows, source, tokens, list(full.shape)


def token_mismatches(tokenizer, tokens, ids):
    out = []
    for token_id in ids:
        want = tokenizer.convert_ids_to_tokens(int(token_id))
        got = tokens[int(token_id)] if int(token_id) < len(tokens) else None
        if want != got:
            out.append({'id': int(token_id), 'hf': want, 'gguf': got})
    return out


def find_axis_artifacts(root: Path, width: int):
    """Existing fitted axes in the clone, and whether a cosine is even defined."""
    def walk(node, hit):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == 'shape' and isinstance(value, list) and len(value) == 2 \
                        and all(isinstance(v, int) for v in value) and hit and width:
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
    ap.add_argument('--gguf', default=None, help='optional quantized stand-in artifact')
    ap.add_argument('--out', required=True, help='report JSON path')
    ap.add_argument('--cache-dir', required=True, help='HF cache directory (deleted downloads live here)')
    ap.add_argument('--delete-downloads', action='store_true', help='hash, read, then delete the safetensors')
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
    layers = parse_layers(args.layers, int(base_cfg['num_hidden_layers']))

    table = json.loads(DENIAL_TABLE.read_text())['entries']
    variants = head_variants(tokenizer, table)
    ids_all = sorted(set(variants['named']['ids'] + variants['compliance']['ids']
                         + variants['primary']['ids'] + variants['wide_denial_ids']))
    print(f'[head-axis] {base_repo}@{base_rev} vocab={len(tokenizer)} layers={len(layers)} '
          f'denial(primary)={len(variants["primary"]["ids"])} '
          f'compliance={len(variants["compliance"]["ids"])}', flush=True)

    base_path, base_sha = fetch(base_repo, base_rev, cache)
    twin_path, twin_sha = fetch(twin_repo, twin_rev, cache)
    print(f'[head-axis] hashed base {base_sha[:16]} twin {twin_sha[:16]}', flush=True)

    models = {}
    for label, path, digest, repo, rev in (('aligned', base_path, base_sha, base_repo, base_rev),
                                           ('twin', twin_path, twin_sha, twin_repo, twin_rev)):
        rows, head_name = head_rows_for(path, ids_all)
        axes = {
            'named': axis_block(tokenizer, rows, variants['named']['ids'],
                                variants['compliance']['ids'], 'named words only'),
            'primary': axis_block(tokenizer, rows, variants['primary']['ids'],
                                  variants['compliance']['ids'], 'named + single-token table surfaces'),
            'wide': axis_block(tokenizer, rows, variants['wide_denial_ids'],
                               variants['compliance']['ids'], 'named + every table id'),
        }
        axis = axes['primary']
        print(f'[head-axis] {label}: head={head_name} margin={axis["margin_per_unit_d"]} '
              f'edit over {len(layers) * len(WRITEBACK)} tensors', flush=True)
        channel = head_channel_closed(axis['g'], axis['d'])
        models[label] = {
            'repo': repo, 'revision': rev,
            'safetensors': {'sha256': digest, 'bytes': path.stat().st_size,
                            'deleted_after_read': bool(args.delete_downloads)},
            'head_tensor': head_name,
            'tie': tie_check(path),
            'axes': {key: {k: v for k, v in value.items() if k not in ('d', 'g')}
                     for key, value in axes.items()},
            '_vectors': {key: (value['d'], value['g']) for key, value in axes.items()},
            'edit': run_model(path, axis, layers),
            'checks': {
                'head_channel_closed': bool(channel['closed']),
                'head_margin_after_projection': r(channel['margin_after'], 3),
                'margin_per_unit_d': r(channel['margin_per_unit_d']),
                'axis_cosines': {
                    'named_vs_primary': r(cosine(axes['named']['d'], axes['primary']['d']), 5),
                    'primary_vs_wide': r(cosine(axes['primary']['d'], axes['wide']['d']), 5),
                    'named_vs_wide': r(cosine(axes['named']['d'], axes['wide']['d']), 5),
                },
            },
        }
        del rows

    cross = {
        'cos_aligned_vs_twin_primary': r(cosine(models['aligned']['_vectors']['primary'][0],
                                                models['twin']['_vectors']['primary'][0]), 5),
        'cos_aligned_vs_twin_named': r(cosine(models['aligned']['_vectors']['named'][0],
                                              models['twin']['_vectors']['named'][0]), 5),
    }
    twin = twin_axis(base_path, twin_path, models['aligned']['_vectors']['primary'][0], layers)
    gguf_block = None
    if args.gguf:
        gguf_rows, source, tokens, shape = gguf_head(Path(args.gguf), ids_all, ids_all)
        ga = head_axis(gguf_rows, variants['primary']['ids'], variants['compliance']['ids'])
        gguf_block = {
            'path': str(args.gguf), 'sha256': sha256_file(Path(args.gguf)),
            'bytes': Path(args.gguf).stat().st_size,
            'head_tensor': source, 'head_shape': [int(x) for x in shape],
            'margin_per_unit_d': r(ga['margin']),
            'cos_vs_aligned_primary': r(cosine(ga['d'], models['aligned']['_vectors']['primary'][0]), 5),
            'cos_vs_twin_primary': r(cosine(ga['d'], models['twin']['_vectors']['primary'][0]), 5),
            'token_string_mismatches': token_mismatches(tokenizer, tokens, ids_all),
        }
        print(f'[head-axis] gguf cos vs aligned {gguf_block["cos_vs_aligned_primary"]} '
              f'token mismatches {len(gguf_block["token_string_mismatches"])}', flush=True)
    artifacts = find_axis_artifacts(REPO, 896)

    report = {
        'schema': 'jdewitt.refusal-head-axis/1',
        'date_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'question': 'does the output head alone name the direction an abliteration removes, '
                    'and what does removing it cost per writeback tensor?',
        'zero_forward': True,
        'removal_measured': False,
        'method': {
            'axis': 'd = normalize(mean(E_denial) - mean(E_compliance)); E = the lm_head rows '
                    '(tied: model.embed_tokens.weight)',
            'edit': "W' = W (I - d d^T) for self_attn.o_proj and mlp.down_proj in GGUF [in, out] "
                    'order, float64; the stored delta is W d d^T',
            'identities': "d unit; ||W(I-dd^T)-W||_F = ||Wd||_2; margin m-m' = "
                          "||W_out^T u|| (Wd).a",
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
            'gguf_stand_in': gguf_block,
            'repo_axis_artifacts': artifacts,
        },
        'claims': {
            'measured': 'per-tensor edit norms and the three identities, from the weights and the '
                        'tokenizer alone',
            'not_measured': 'whether refusal behaviour changes; this card runs no model',
            'comparison_limit': 'the recovered twin direction is a weight delta, not a behavioural '
                                'axis; a high cosine says the head geometry agrees with the edit, '
                                'not that removing d reproduces the twin',
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
        'twin_axis_mean_abs_cos': twin['mean_abs_cos'],
        'twin_effective_rank': sorted({row.get('twin_effective_rank') for row in twin['per_tensor']
                                       if row['touched']})[:3],
        'relative_edit_norm': {label: models[label]['edit']['relative_edit_norm'] for label in models},
        'identities_ok': {label: (models[label]['edit']['identities']['residual_identity_all_ok']
                                  and models[label]['edit']['identities']['margin_identity_all_ok'])
                          for label in models},
        'gguf_cos_vs_aligned': None if gguf_block is None else gguf_block['cos_vs_aligned_primary'],
    }, indent=1), flush=True)

    if args.delete_downloads:
        for path in (base_path, twin_path):
            print(f'[head-axis] deleting {path}', flush=True)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            lock = path.parent / '.lock'
            if lock.exists():
                try:
                    lock.unlink()
                except OSError:
                    pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
