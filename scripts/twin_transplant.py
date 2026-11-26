# SPDX-License-Identifier: AGPL-3.0-or-later
"""Transplant a twin's full tensor delta into a user's quant, in place.

W' = W + a*dW with a=1 on the residual-write projections (blk.N.attn_output,
blk.N.ffn_down), where dW = W_twin - W_base is computed on the CPU from two public
`model.safetensors` files and W is the user's OWN artifact, decoded and never
re-quantized from scratch: only the 48 edited tensors are re-encoded (Q8_0, a type
that can hold a delta); every other tensor is copied byte for byte.

Two things this measures, both bytes:

  * the full delta (a=1) delivered per tensor: `ggufio.compare_edit` takes
    dequant(edited) - dequant(pristine) and compares it with the delta the writer
    was handed, so a low cos means the Q8_0 grid (or the IQ3_M decode the delta
    landed on) decided what shipped;
  * a rank-k reduction of the same delta (k=4, 8, 16 per tensor): cos(dW, dW_k)
    is exact linear algebra, the realized cos is what the file keeps of it. A
    rank-4 replacement of a 4.36M-parameter matrix is 0.003% of the elements —
    if the realized cos stays high, the transplant is compressible; if it does
    not, that is the result.

Nothing here loads a model, a tokenizer, a server or a GPU: no serve, no capture,
no benchmark, no wave. `removal_measured` is false in every report by construction,
and the twin's own model card is quoted as evidence of what the twin CLAIMS, never
as evidence of behaviour.

    python scripts/twin_transplant.py \
        --artifact verification/iq05b/Qwen2.5-0.5B-Instruct-IQ3_M.gguf \
        --base  Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775 \
        --twin  Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1@a42fe88f75b3ab545466ca5692fef0b4d7bc8009 \
        --cache ~/.cache/twin-transplant --work ~/.cache/twin-transplant/run \
        --report verification/twin-transplant/report.json

Downloads are hashed against the Hub's LFS oid and then DELETED (the deltas are
kept as one .npz), because a 0.5B pair is 2.2 GB and the disk is shared.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SCHEMA = 'jdewitt-twin-transplant-v1'
EDIT_TYPE = 'q8_0'
# The two residual-write projections a twin delta is transplanted into, by their
# GGUF suffix -> HF source key. Everything else in the pair is skipped: only the
# tensors that write the residual stream carry the twin-vs-base difference this
# card is about, and touching the rest would be a different experiment.
HF_SUFFIX = {
    'attn_output': 'self_attn.o_proj',
    'ffn_down': 'mlp.down_proj',
}
KINDS = {'attn_output': 'attn', 'ffn_down': 'mlp'}


def sha256_file(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- sources
def hub_facts(repo, revision):
    """Revision, file size and the Hub's LFS sha256 for `model.safetensors`."""
    import urllib.request

    url = f'https://huggingface.co/api/models/{repo}/tree/{revision}?blobs=true'
    with urllib.request.urlopen(url, timeout=60) as response:
        entries = json.loads(response.read())
    for entry in entries:
        if entry.get('path') == 'model.safetensors':
            lfs = entry.get('lfs') or {}
            return {'repo': repo, 'revision': revision,
                    'files': [item.get('path') for item in entries if item.get('type') == 'file'],
                    'file': 'model.safetensors', 'bytes': int(entry.get('size') or 0),
                    'sha256_upstream_lfs': lfs.get('oid'), 'pointer_bytes': lfs.get('pointerSize')}
    raise ValueError(f'{repo}@{revision} has no model.safetensors in its file list')


def fetch_safetensors(facts, cache):
    """Download `model.safetensors`, check it against the Hub's LFS oid, keep the bytes.

    The local digest is the one that counts: the LFS oid is what the Hub says the
    file is, the sha256 of the bytes on this disk is what the delta was computed
    from, and a mismatch is a refusal rather than a note.
    """
    from huggingface_hub import hf_hub_download

    tag = facts['repo'].split('/')[-1]
    path = Path(hf_hub_download(facts['repo'], facts['file'], revision=facts['revision'],
                                local_dir=str(Path(cache) / 'dl' / tag)))
    facts = dict(facts, path=str(path), bytes_local=int(path.stat().st_size),
                 sha256_local=sha256_file(path))
    if facts['sha256_local'] != facts['sha256_upstream_lfs']:
        raise ValueError(f'{path}: local sha256 {facts["sha256_local"]} != Hub LFS oid '
                         f'{facts["sha256_upstream_lfs"]}; refusing to build a delta from '
                         f'bytes the Hub does not name')
    return facts


def model_card(repo, revision, cache, out_dir):
    """The twin's model card, saved verbatim for citation (not a measurement)."""
    import urllib.request

    url = f'https://huggingface.co/{repo}/raw/{revision}/README.md'
    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode('utf-8', 'replace')
    target = Path(out_dir) / f'{repo.split("/")[-1]}-README.md'
    target.write_text(text)
    lines = [line.strip() for line in text.replace('\r', '').splitlines()]
    keys = ('abliterat', 'refusal', 'uncensor', 'finetuned', 'finetune',
            'base_model', 'base model', 'license')
    quoted = [line for line in lines
              if line and not line.startswith(('---', '#', '- ', 'tags')) and
              any(key in line.lower() for key in keys)]
    return {'url': url, 'path': str(target), 'sha256': sha256_file(target),
            'quoted_lines': quoted[:12],
            'note': 'the card is what the twin CREATOR claims about the twin; this run '
                    'measured weights and bytes only and no behaviour'}


# ------------------------------------------------------------------- safetensors IO
def safetensors_index(path):
    """(header, data_start) — names, dtypes, shapes and byte offsets, no torch."""
    import struct

    with open(path, 'rb') as handle:
        length = struct.unpack('<Q', handle.read(8))[0]
        header = json.loads(handle.read(length))
    header.pop('__metadata__', None)
    return header, 8 + length


_DTYPES = {'BF16': ('<u2', 2), 'F16': ('<f2', 2), 'F32': ('<f4', 4)}


def read_tensor(path, index, data_start, name):
    """One tensor as float32 in torch layout [out, in].

    GGUF stores linear weights transposed and this reads the safetensors side, so
    the array comes out in the same [out, in] order the GGUF decode produces —
    which is what makes dW = W_twin - W_base addable to a decoded GGUF tensor.
    """
    if name not in index:
        raise ValueError(f'{name} is not in {path}')
    info = index[name]
    if info['dtype'] not in _DTYPES:
        raise ValueError(f'{name}: dtype {info["dtype"]} is not bf16/f16/f32')
    numpy_dtype, width = _DTYPES[info['dtype']]
    begin, end = (int(x) for x in info['data_offsets'])
    shape = tuple(int(x) for x in info['shape'])
    with open(path, 'rb') as handle:
        handle.seek(data_start + begin)
        raw = handle.read(end - begin)
    if len(raw) != end - begin:
        raise ValueError(f'{name}: short read in {path}')
    if info['dtype'] == 'BF16':
        values = (np.frombuffer(raw, dtype=numpy_dtype).astype(np.uint32) << 16).view(np.float32)
    else:
        values = np.frombuffer(raw, dtype=numpy_dtype).astype(np.float32)
    del width
    return np.ascontiguousarray(values.reshape(shape))


def hf_key(gguf_name):
    """blk.7.ffn_down.weight -> model.layers.7.mlp.down_proj.weight (or None)."""
    parts = gguf_name.split('.')
    if len(parts) != 4 or parts[0] != 'blk' or not parts[1].isdigit():
        return None
    suffix = HF_SUFFIX.get(parts[2])
    return None if suffix is None else f'model.layers.{parts[1]}.{suffix}.weight'


# --------------------------------------------------------------------------- deltas
def build_deltas(base, twin, targets, shards=None):
    """{gguf name: float32 dW} for the writeback targets, shapes checked both sides."""
    base_index, base_start = safetensors_index(base)
    twin_index, twin_start = safetensors_index(twin)
    deltas, rows = {}, []
    for name in sorted(targets):
        key = hf_key(name)
        if key is None:
            raise ValueError(f'{name} has no twin/base source key')
        want = tuple(int(x) for x in targets[name])
        wb = read_tensor(base, base_index, base_start, key)
        wt = read_tensor(twin, twin_index, twin_start, key)
        if wb.shape != wt.shape:
            raise ValueError(f'{name}: base {wb.shape} vs twin {wt.shape} in the pair')
        if tuple(reversed(wb.shape)) != want:
            raise ValueError(f'{name}: pair shape {wb.shape} does not transpose to the '
                             f'artifact shape {want}')
        delta = np.ascontiguousarray(wt - wb, dtype=np.float32)
        if not np.isfinite(delta).all():
            raise ValueError(f'{name}: delta has NaN/Inf')
        rows.append({'name': name, 'hf_key': key, 'shape': list(delta.shape),
                     'delta_norm': round(float(np.linalg.norm(delta.astype(np.float64))), 6),
                     'weight_norm': round(float(np.linalg.norm(wb.astype(np.float64))), 6)})
        deltas[name] = delta
        del wb, wt, delta
    if shards is not None:
        np.savez(shards, **{name: deltas[name] for name in deltas})
    return deltas, rows


def rank_k(delta, k):
    """(dW_k, cos(dW, dW_k), energy fraction) from the economy SVD of one delta."""
    values = delta.astype(np.float64, copy=False)
    u, s, vt = np.linalg.svd(values, full_matrices=False)
    k = min(int(k), s.size)
    reduced = (u[:, :k] * s[:k]) @ vt[:k]
    denom = float(np.linalg.norm(values) * np.linalg.norm(reduced))
    cosine = float(np.sum(values * reduced) / denom) if denom else 1.0
    energy = float(np.sum(s[:k] ** 2) / np.sum(s ** 2)) if s.size else 1.0
    return reduced.astype(np.float32), cosine, energy


def cos64(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.sum(a * b) / denom) if denom else 1.0


def summarize(values):
    ordered = sorted(float(v) for v in values if v is not None)
    if not ordered:
        return {'n': 0}
    middle = ordered[len(ordered) // 2] if len(ordered) % 2 else \
        0.5 * (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2])
    return {'n': len(ordered), 'min': round(ordered[0], 4),
            'median': round(middle, 4), 'mean': round(sum(ordered) / len(ordered), 4),
            'max': round(ordered[-1], 4)}


# ---------------------------------------------------------------------- measurement
def cli_json(args, env=None):
    """Run a jdewitt subcommand and return its JSON body (the commands in the report)."""
    command = [sys.executable, '-m', 'jdewitt.cli', *args]
    completed = subprocess.run(command, capture_output=True, text=True,
                               env={**os.environ, 'PYTHONPATH': str(REPO), **(env or {})})
    if completed.returncode != 0:
        raise RuntimeError(f'{" ".join(command)} failed: {completed.stderr[-2000:]}')
    return json.loads(completed.stdout)


def write_artifact(pristine, out, edits, *, tag):
    """Only the named tensors are re-encoded; the rest is copied byte for byte."""
    from jdewitt.rewrite import rewrite_with_edits

    started = time.monotonic()
    report = rewrite_with_edits(str(pristine), str(out), edits, edit_type=EDIT_TYPE,
                                scale=1.0, axis_space='output', verify_untouched=True)
    report['tag'] = tag
    report['wall_seconds'] = round(time.monotonic() - started, 1)
    return report


def measure(pristine, out, edits):
    """Per-tensor delivered cos/snr, plus the file-size delta and the untouched proof."""
    from jdewitt.ggufio import compare_edit

    names = sorted(edits)
    measured = compare_edit(str(pristine), str(out), edits, names=names)
    rows = []
    for row in measured['tensors']:
        name = row['name']
        kind = KINDS.get(name.split('.')[2], 'other')
        rows.append({'name': name, 'layer': int(name.split('.')[1]), 'kind': kind,
                     **{k: row.get(k) for k in ('source_type', 'edited_type', 'cos',
                                                'snr', 'norm_ratio')}})
    return {'per_tensor': rows,
            'min_cos': measured['min_cos'], 'mean_cos': measured['mean_cos'],
            'min_snr': min((row['snr'] for row in rows if row.get('snr') is not None),
                           default=None),
            'by_kind': {kind: summarize([row['cos'] for row in rows if row['kind'] == kind])
                        for kind in ('attn', 'mlp')},
            'shape': [p.stat().st_size for p in (Path(pristine), Path(out))],
            'bytes_delta': int(Path(out).stat().st_size - Path(pristine).stat().st_size)}


def cross_cos(pristine, out, references, names):
    """cos(realized change, each reference delta) per tensor — one decode pass.

    `measure` answers "did the file keep what the writer was handed"; this answers
    "does a REDUCED edit ship the FULL delta", which is the transplant's question.
    """
    from gguf import GGUFReader

    from jdewitt.ggufio import decode_values

    totals = {label: [0.0, 0.0, 0.0] for label in references}
    rows = []
    reader_a = GGUFReader(str(pristine))
    reader_b = GGUFReader(str(out))
    tensors_a = {t.name: t for t in reader_a.tensors}
    tensors_b = {t.name: t for t in reader_b.tensors}

    def load(path, tensor):
        raw = np.memmap(str(path), dtype=np.uint8, mode='r',
                        offset=int(tensor.data_offset), shape=(int(tensor.n_bytes),))
        shape = tuple(int(x) for x in np.asarray(tensor.shape).ravel())[::-1]
        values = decode_values(np.array(raw), tensor.tensor_type.name)
        return values.astype(np.float32).reshape(shape)

    for name in names:
        realized = (load(out, tensors_b[name]) - load(pristine, tensors_a[name])) \
            .astype(np.float64)
        row = {'name': name}
        for label, refs in references.items():
            reference = np.asarray(refs[name], dtype=np.float64)
            cos = cos64(realized, reference)
            totals[label][0] += float(np.sum(realized * reference))
            totals[label][1] += float(np.sum(realized * realized))
            totals[label][2] += float(np.sum(reference * reference))
            row[f'cos_vs_{label}'] = round(cos, 6)
        rows.append(row)
        del realized
    aggregate = {}
    for label, (dot, left, right) in totals.items():
        denom = (left ** 0.5) * (right ** 0.5)
        aggregate[label] = round(dot / denom, 6) if denom else None
    return {'aggregate_cos': aggregate, 'per_tensor': rows,
            'note': 'cos of the whole stacked realized change against the whole stacked '
                    'reference delta; the per-tensor rows are in per_tensor'}


# --------------------------------------------------------------------------- report
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--artifact', type=Path, required=True,
                        help="the user's quantized artifact (never modified here)")
    parser.add_argument('--base', required=True, help='repo@revision of the base model')
    parser.add_argument('--twin', required=True, help='repo@revision of the twin model')
    parser.add_argument('--cache', type=Path, default=Path.home() / '.cache/twin-transplant')
    parser.add_argument('--work', type=Path, default=None,
                        help='where edited artifacts and the in-place copy live')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--ranks', default='4,8,16')
    parser.add_argument('--keep-downloads', action='store_true')
    parser.add_argument('--limit', type=int, default=None,
                        help='measure only the first N targets (a smoke run)')
    args = parser.parse_args(argv)

    from jdewitt.ggufgeom import (discover_writeback_targets, gguf_weight_shapes,
                                  inferred_hidden_width)
    from jdewitt.ggufwrite import parse_container
    from jdewitt.quants import type_name_for_id

    started = time.monotonic()
    cache = args.cache.expanduser()
    work = (args.work or cache / 'run').expanduser()
    work.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    ranks = [int(x) for x in str(args.ranks).split(',') if x.strip()]

    artifact = args.artifact.expanduser()
    artifact_sha = sha256_file(artifact)
    shapes = gguf_weight_shapes(str(artifact))
    header = parse_container(artifact)
    source_types = Counter(type_name_for_id(t['qtype']) for t in header['tensors'])
    hidden = inferred_hidden_width(shapes)
    discovered = discover_writeback_targets(shapes, hidden)
    targets = {name: shapes[name]
               for layer in discovered['targets'] for name in discovered['targets'][layer]}
    if args.limit:
        targets = dict(sorted(targets.items())[:args.limit])
    print(f'artifact {artifact.name} {artifact_sha[:16]} hidden={hidden} '
          f'targets={len(targets)}', flush=True)

    base_repo, base_rev = args.base.rsplit('@', 1)
    twin_repo, twin_rev = args.twin.rsplit('@', 1)
    base_facts = fetch_safetensors(hub_facts(base_repo, base_rev), cache)
    twin_facts = fetch_safetensors(hub_facts(twin_repo, twin_rev), cache)
    print(f'sources hashed: base {base_facts["sha256_local"][:16]} '
          f'twin {twin_facts["sha256_local"][:16]}', flush=True)

    shards = work / 'twin-deltas.npz'
    deltas, delta_rows = build_deltas(base_facts['path'], twin_facts['path'], targets,
                                      shards=shards)
    print(f'deltas built: {len(deltas)} tensors -> {shards}', flush=True)
    if not args.keep_downloads:
        freed = 0
        for facts in (base_facts, twin_facts):
            path = Path(facts['path'])
            freed += path.stat().st_size
            path.unlink()
            leftover = path.parent / '.cache'
            if leftover.is_dir():
                shutil.rmtree(leftover, ignore_errors=True)
            if path.parent.is_dir() and not any(path.parent.iterdir()):
                path.parent.rmdir()
        print(f'downloads deleted: {freed} bytes', flush=True)
    else:
        freed = 0

    report = {
        'schema': SCHEMA,
        'removal_measured': False,
        'zero_forward': {
            'serve': False, 'capture': False, 'wave': False, 'benchmark': False,
            'gpu': False,
            'note': 'weight algebra, byte identity and file digests only; no model was '
                    'loaded and no answer was generated, so no behavioural claim is made '
                    'and a negative cos is a valid result'},
        'sources': {
            'base': {k: base_facts[k] for k in
                     ('repo', 'revision', 'file', 'bytes_local', 'sha256_local',
                      'sha256_upstream_lfs')},
            'twin': {k: twin_facts[k] for k in
                     ('repo', 'revision', 'file', 'bytes_local', 'sha256_local',
                      'sha256_upstream_lfs')},
            'twin_card': model_card(twin_repo, twin_rev, cache, args.report.parent),
            'downloads_deleted_bytes': freed,
            'deltas_npz': str(shards),
            'delta_space': 'float32 [out, in] torch layout; dW = W_twin - W_base',
        },
        'artifact': {
            'path': str(artifact), 'sha256': artifact_sha,
            'bytes': int(artifact.stat().st_size),
            'architecture': 'qwen2', 'hidden': hidden,
            'tensors': len(header['tensors']), 'writeback_targets': len(targets),
            'source_types': dict(sorted(source_types.items())),
            'note': 'the user quant is the base model quantized; the twin delta is added '
                    'to ITS decoded weights, never to a fresh 16-bit copy',
        },
        'rule': 'W\' = W + a*dW, a=1, over the residual-write projections; every other '
                'tensor copied byte for byte',
        'edit_type': EDIT_TYPE,
        'deltas': {'tensors': delta_rows, 'total_delta_norm': round(float(np.sqrt(sum(
            row['delta_norm'] ** 2 for row in delta_rows))), 6)},
        'stages': {},
    }
    report_path = args.report
    report_path.write_text(json.dumps(report, indent=1) + '\n')

    # ---- stage 1: the full delta, a=1
    stage_t0 = time.monotonic()
    full_edits = {name: deltas[name] for name in deltas}
    full_out = work / f'{artifact.stem}-full-{EDIT_TYPE}.gguf'
    rewrite = write_artifact(artifact, full_out, full_edits, tag='full')
    delivered = measure(artifact, full_out, full_edits)
    diff = cli_json(['gguf-tensor-diff', str(artifact), str(full_out)])
    edited_names = sorted(full_edits)
    differing = sorted(row['name'] for row in diff['differing_tensors'])
    report['stages']['full'] = {
        'output': str(full_out), 'sha256': sha256_file(full_out),
        'bytes_in': rewrite['bytes_in'], 'bytes_out': rewrite['bytes_out'],
        'bytes_delta': delivered['bytes_delta'],
        'edited_tensors': len(edited_names),
        'reencoded': [{'name': row['name'], 'from_dtype': row['from_dtype'],
                       'to_dtype': row['to_dtype'], 'bytes_before': row['bytes_before'],
                       'bytes_after': row['bytes_after']}
                      for row in rewrite['edited']],
        'untouched': rewrite['untouched'],
        'delivered': delivered,
        'tensor_diff': {
            'summary': diff['summary'],
            'differing': differing,
            'names_exactly_the_edited_tensors': differing == edited_names,
            'bit_identical_count': diff['summary'].get('bit_identical'),
        },
        'fingerprint': rewrite['fingerprint'],
        'wall_seconds': rewrite['wall_seconds'],
        'wall_seconds_total': round(time.monotonic() - stage_t0, 1),
    }
    report_path.write_text(json.dumps(report, indent=1) + '\n')
    print(f"full: untouched={rewrite['untouched']['verified']} "
          f"diff_names_match={differing == edited_names} "
          f"min_cos={delivered['min_cos']} mean_cos={delivered['mean_cos']} "
          f"bytes+{delivered['bytes_delta']}", flush=True)

    # ---- stage 2: rank-k reductions of the same delta
    for k in ranks:
        stage_t0 = time.monotonic()
        reduced, reported_cos, energy = {}, {}, {}
        for name, delta in full_edits.items():
            value, cosine, fraction = rank_k(delta, k)
            reduced[name] = value
            reported_cos[name] = cosine
            energy[name] = fraction
        out = work / f'{artifact.stem}-rank{k}-{EDIT_TYPE}.gguf'
        stage_rewrite = write_artifact(artifact, out, reduced, tag=f'rank{k}')
        stage_delivered = measure(artifact, out, reduced)
        extra = cross_cos(artifact, out, {'full': full_edits}, edited_names)
        per_tensor = {row['name']: row for row in extra['per_tensor']}
        rows = [{'name': row['name'], 'layer': row['layer'], 'kind': row['kind'],
                 'cos_delta_vs_rank': round(reported_cos[row['name']], 6),
                 'energy_retained': round(energy[row['name']], 6),
                 'realized_cos_vs_rank': row['cos'],
                 'realized_cos_vs_full': per_tensor[row['name']]['cos_vs_full'],
                 'snr': row['snr']}
                for row in stage_delivered['per_tensor']]
        report['stages'][f'rank{k}'] = {
            'output': str(out), 'sha256': sha256_file(out),
            'bytes_delta': stage_delivered['bytes_delta'],
            'edit_digest': stage_rewrite['fingerprint']['edit_sha256'],
            'cos_delta_vs_rank': summarize([row['cos_delta_vs_rank'] for row in rows]),
            'energy_retained': summarize([row['energy_retained'] for row in rows]),
            'realized_cos_vs_rank': summarize([row['realized_cos_vs_rank'] for row in rows]),
            'realized_cos_vs_full': summarize([row['realized_cos_vs_full'] for row in rows]),
            'aggregate_realized_cos_vs_full': extra['aggregate_cos']['full'],
            'untouched': stage_rewrite['untouched'],
            'per_tensor': rows,
            'wall_seconds': stage_rewrite['wall_seconds'],
            'wall_seconds_total': round(time.monotonic() - stage_t0, 1),
        }
        report_path.write_text(json.dumps(report, indent=1) + '\n')
        print(f'rank{k}: cos(full,rank) mean={report["stages"][f"rank{k}"]["cos_delta_vs_rank"]["mean"]} '
              f'realized mean={report["stages"][f"rank{k}"]["realized_cos_vs_rank"]["mean"]} '
              f'realized-vs-full mean='
              f'{report["stages"][f"rank{k}"]["realized_cos_vs_full"]["mean"]}', flush=True)

    # ---- stage 3: in place on a copy of the user's artifact, then gguf-revert
    from jdewitt.ggufio import save_whole_file_revert

    inplace = work / artifact.name
    if inplace.exists():
        inplace.unlink()
    shutil.copyfile(artifact, inplace)
    if sha256_file(inplace) != artifact_sha:
        raise RuntimeError(f'{inplace}: the in-place copy is not the artifact')
    sidecar = Path(str(inplace) + '.revert.npz')
    saved = save_whole_file_revert(inplace, full_out, sidecar)
    staged = inplace.with_name(inplace.name + '.partial')
    shutil.copyfile(full_out, staged)
    os.replace(staged, inplace)
    patched_sha = sha256_file(inplace)
    patched_bytes = int(inplace.stat().st_size)
    revert = cli_json(['gguf-revert', str(inplace), str(sidecar)])
    restored_sha = sha256_file(inplace)
    report['stages']['in_place'] = {
        'artifact': str(inplace),
        'pristine_sha256': artifact_sha, 'patched_sha256': patched_sha,
        'patched_equals_full_stage': patched_sha == sha256_file(full_out),
        'bytes_before': int(artifact.stat().st_size), 'bytes_after': patched_bytes,
        'bytes_delta': patched_bytes - int(artifact.stat().st_size),
        'sidecar': {'path': saved['sidecar'], 'bytes': saved['bytes'],
                    'truncate_to': saved['truncate_to']},
        'revert_command': f'jdewitt gguf-revert {inplace} {sidecar}',
        'revert': revert,
        'restored_sha256': restored_sha,
        'restored_byte_exact': restored_sha == artifact_sha,
        'note': 'the shared fixture is never modified: the run copies it (digest checked), '
                'patches the copy by os.replace, then gguf-revert restores it and the '
                'digest is compared with the source',
    }
    report['wall_seconds'] = round(time.monotonic() - started, 1)
    report['claims'] = {
        'untouched_bytes': 'proved by re-reading BOTH files tensor by tensor '
                           '(rewrite.rewrite_with_edits) and again by gguf-tensor-diff, '
                           'which names every differing tensor',
        'delivered_cos': 'dequant(edited) - dequant(pristine) against the delta the '
                         'writer was handed: a fact about these bytes, not about answers',
        'rank_k': 'cos(dW, dW_k) is exact linear algebra on the pair; the realized cos '
                  'is what the Q8_0 re-encode kept of it',
        'removal': 'NOT measured and NOT claimed: no serve, no capture, no wave, no '
                   'benchmark, no GPU; the twin card is cited as the twin creator\'s '
                   'claim, not as an observation',
    }
    report['next'] = [
        f'jdewitt gguf-tensor-diff {artifact} {full_out}  # only the named tensors differ',
        f'jdewitt gguf-inspect {full_out}',
        '# certify any behavioural claim on the edited artifact with a real serve config '
        'and a judge wave; this run measured weights and bytes only',
    ]
    report_path.write_text(json.dumps(report, indent=1) + '\n')
    print(f'report {report_path} ({report["wall_seconds"]}s)', flush=True)
    return report


if __name__ == '__main__':
    main()
