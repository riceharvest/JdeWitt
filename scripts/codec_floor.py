#!/usr/bin/env python
"""Codec-floor ablation: re-encode the writeback tensors at coarse grids, measure what survives.

    python scripts/codec_floor.py \
        --artifact ~/.cache/twin-transplant/run/Qwen2.5-0.5B-Instruct-IQ3_M.gguf \
        --surfaces verification/denial-grammar/denial-surfaces.json \
        --compliant verification/denial-grammar/compliant-openings.jsonl \
        --work-dir ~/.cache/codec-floor/run \
        --report verification/codec-floor/report.json

One pass over the 48 residual-write tensors of the user's own IQ3_M artifact:

  1. read the denial ids from the artifact's own vocabulary + the repo's 27-surface
     table (no forward pass, no tokenizer download);
  2. build the denial-aligned subspace from the *input embedding rows* of those ids,
     plus the rank-1 x-4 mean axis (denial mean minus compliance mean) as the size of
     an explicit edit, plus a random subspace of the same rank as the control;
  3. re-encode each tensor at Q4_0, Q2_K and Q2_0 in float, and measure the surviving
     aligned energy, the delivered cos and the codec's whole-tensor error;
  4. write one real GGUF per type with those tensors re-encoded, and check: untouched
     tensors byte-identical, KV bytes identical, the file parses and every tensor
     decodes finite, and a whole-file revert restores the source sha256 exactly.

``removal_measured`` is false everywhere: no model was loaded, no answer generated.
A negative result is the expected shape of an honest run and is reported as such.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# The board's CPU is shared; a 32-thread BLAS spends more time contending than
# computing. Set this before numpy is imported.
for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '4')

import numpy as np  # noqa: E402  (the thread cap must be set before numpy loads)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from jdewitt.codec_floor import (SCHEMA, TYPES, GgufBpe, aggregate, denial_id_sets,  # noqa: E402
                                 prepare_alignment, prepared_curve, quantization_error,
                                 random_subspace, row_block_ok, roundtrip, subspace,
                                 token_rows, verify_written)
from jdewitt.ggufio import decode_values, revert_inplace, save_whole_file_revert  # noqa: E402
from jdewitt.ggufwrite import parse_container, tensor_nbytes, write_container  # noqa: E402
from jdewitt.quants import type_id_for_name, type_name_for_id  # noqa: E402

WRITEBACK = (('attn_output.weight', 'attn'), ('ffn_down.weight', 'ffn'))
KS = (1, 2, 4, 8, 16, 32)
COMPLIANCE_WORDS = ('Sure', 'Certainly', 'Yes', 'Here', 'Okay', 'Absolutely')


def sha256(path, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def display(path):
    """Repo-relative when inside the clone, ~-relative otherwise: no absolute lab paths."""
    resolved = Path(path).resolve()
    for root, prefix in ((REPO.resolve(), ''), (Path.home().resolve(), '~/')):
        try:
            return prefix + str(resolved.relative_to(root))
        except ValueError:
            continue
    return str(resolved)


def header_digest(path):
    header = parse_container(path)
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read(header['data_base'])).hexdigest(), header


def tokenizer_digest(reader):
    keys = [key for key in reader.fields if key.startswith('tokenizer.ggml.')]
    digest = hashlib.sha256()
    counts = {}
    for key in sorted(keys):
        contents = reader.fields[key].contents()
        values = [contents] if isinstance(contents, (str, bytes, bool, int, float)) else list(contents)
        counts[key] = len(values)
        digest.update(key.encode())
        digest.update(len(values).to_bytes(8, 'little'))
        for value in values:
            raw = value if isinstance(value, bytes) else (
                value.encode('utf-8') if isinstance(value, str) else str(value).encode())
            digest.update(len(raw).to_bytes(8, 'little'))
            digest.update(raw)
    return digest.hexdigest(), counts


def writeback_tensors(header):
    found = []
    for tensor in header['tensors']:
        leaf = tensor['name'].split('.', 2)[2] if tensor['name'].startswith('blk.') else ''
        for suffix, kind in WRITEBACK:
            if leaf == suffix:
                found.append({'name': tensor['name'], 'kind': kind,
                              'layer': int(tensor['name'].split('.')[1]),
                              'dims': tensor['dims'], 'qtype': type_name_for_id(tensor['qtype'])})
    return found


def load_tensor(path, header, name, reader_index):
    tensor = reader_index[name]
    qtype = type_name_for_id(int(tensor.tensor_type))
    nbytes = tensor_nbytes([int(x) for x in tensor.shape], int(tensor.tensor_type))
    with open(path, 'rb') as handle:
        handle.seek(int(tensor.data_offset))
        raw = np.frombuffer(handle.read(nbytes), dtype=np.uint8)
    values = decode_values(raw, qtype).astype(np.float32)
    return values.reshape(tuple(int(x) for x in tensor.shape)[::-1]), raw


def aligned_row(prepared, quantized, rank, ks=(1,)):
    """One table row: aligned fraction, delivered cos, removed energy, random control."""
    curve, control = prepared_curve(prepared, quantized, ks=tuple(ks) + (rank,))
    full = curve['curves'][str(rank)]
    full_control = control['curves'][str(control['rank'])]
    keep = {key: curve['curves'][key] for key in curve['curves'] if int(key) != rank}
    keep_control = {key: control['curves'][key] for key in control['curves'] if int(key) != control['rank']}
    aligned = full['aligned_energy_fraction']
    return {
        'rank': rank,
        'aligned_energy_fraction': aligned,
        'isotropic_fraction': full['isotropic_fraction'],
        'fraction_over_isotropic': aligned / full['isotropic_fraction'] if full['isotropic_fraction'] else None,
        'surviving_energy_fraction': full['surviving_energy_fraction'],
        'surviving_norm_ratio': float(np.sqrt(max(full['surviving_energy_fraction'], 0.0))),
        'removed_energy_fraction': full['removed_energy_fraction'],
        'cos': full['cos'],
        'curves': keep,
        'random': {
            'aligned_energy_fraction': full_control['aligned_energy_fraction'],
            'removed_energy_fraction': full_control['removed_energy_fraction'],
            'surviving_energy_fraction': full_control['surviving_energy_fraction'],
            'cos': full_control['cos'],
            'curves': keep_control,
        },
        'selectivity': (full['removed_energy_fraction'] / full_control['removed_energy_fraction']
                        if full_control['removed_energy_fraction'] else None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--surfaces', type=Path, required=True)
    parser.add_argument('--compliant', type=Path, default=None)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, default=None,
                        help='where the re-encoded GGUFs are written; default: a temp dir')
    parser.add_argument('--types', default=','.join(TYPES))
    parser.add_argument('--min-rank', type=int, default=1,
                        help='skip tensors simpler than this; kept for a smoke run')
    parser.add_argument('--limit', type=int, default=0, help='first N writeback tensors only')
    parser.add_argument('--no-apply', action='store_true',
                        help='measurement only: do not write or verify any artifact')
    parser.add_argument('--keep-artifacts', action='store_true')
    args = parser.parse_args()

    types = [value.strip().lower() for value in args.types.split(',') if value.strip()]
    started = time.monotonic()
    artifact = args.artifact.expanduser()
    source_sha = sha256(artifact)
    source_header_sha, source_header = header_digest(artifact)
    from gguf import GGUFReader

    reader = GGUFReader(str(artifact))
    tokens_sha, token_counts = tokenizer_digest(reader)
    tokenizer = GgufBpe.from_reader(reader)

    surfaces_doc = json.loads(args.surfaces.read_text())
    surfaces = surfaces_doc['entries']
    ids = denial_id_sets(tokenizer, surfaces)
    compliant_rows = []
    if args.compliant is not None:
        compliant_rows = [json.loads(line) for line in args.compliant.read_text().splitlines() if line.strip()]

    def compliance_rows():
        """Input embedding rows of the compliance openers (the other half of the x-4 axis)."""
        wanted = []
        for text in COMPLIANCE_WORDS:
            for form in (text, text.lower(), ' ' + text, ' ' + text.lower()):
                encoded = tokenizer.encode(form)
                if len(encoded) == 1 and encoded[0] not in wanted:
                    wanted.append(encoded[0])
        return wanted

    compliance_ids = compliance_rows()
    rows_surface, order_surface, emb_meta = token_rows(reader, artifact, ids['surface_span'])
    rows_opener, order_opener, _ = token_rows(reader, artifact, ids['opener_span'])
    denial_rows = token_rows(reader, artifact, ids['surface_span'])[0]
    compliance_matrix = token_rows(reader, artifact, compliance_ids)[0]
    mean_axis = denial_rows.mean(axis=0).astype(np.float64) - compliance_matrix.mean(axis=0)
    axis_norm = float(np.linalg.norm(mean_axis))
    axis = (mean_axis / axis_norm).astype(np.float32).reshape(-1, 1)

    spaces = {
        'surface_span': subspace(rows_surface),
        'opener_span': subspace(rows_opener),
        'mean_axis': {'V': axis, 'rank': 1, 'singular_values': [round(axis_norm, 6)],
                      'top1_energy_share': 1.0, 'top4_energy_share': 1.0,
                      'fro_norm': round(axis_norm, 6)},
    }
    controls = {name: random_subspace(emb_meta['width'], block['rank'], seed=0)
                for name, block in spaces.items()}
    for name, block in spaces.items():
        orth = block['V'].T.astype(np.float64) @ block['V'].astype(np.float64)
        block['orthonormality_error'] = float(np.abs(orth - np.eye(block['rank'])).max())

    tensors = writeback_tensors(source_header)
    if args.limit:
        tensors = tensors[:args.limit]
    if not tensors:
        raise SystemExit('no blk.N.attn_output / blk.N.ffn_down tensors in the artifact')

    if args.work_dir is None:
        import tempfile
        work_dir = Path(tempfile.mkdtemp(prefix='codec-floor-'))
    else:
        work_dir = args.work_dir.expanduser()
        work_dir.mkdir(parents=True, exist_ok=True)

    reader_index = {tensor.name: tensor for tensor in reader.tensors}
    reports = {name: {'tensors': [], 'skipped': []} for name in types}
    writeback_by_type = {name: {'payloads': {}, 'qtypes': {}} for name in types}

    for index, spec in enumerate(tensors, 1):
        weight, _raw = load_tensor(artifact, source_header, spec['name'], reader_index)
        out_features, in_features = weight.shape
        base = {'name': spec['name'], 'kind': spec['kind'], 'layer': spec['layer'],
                'from_dtype': spec['qtype'], 'rows': out_features, 'cols': in_features,
                'weight_energy': float(np.sum(weight.astype(np.float64) ** 2))}
        # Z = VᵀW does not depend on the storage type, so it is computed once per
        # tensor and reused for every grid; only VᵀW_q is per type.
        prepared = {space_name: prepare_alignment(weight, block['V'], controls[space_name])
                    for space_name, block in spaces.items()}
        for type_name in types:
            ok, block_els = row_block_ok(in_features, type_name)
            if not ok:
                reports[type_name]['skipped'].append({
                    'name': spec['name'], 'kind': spec['kind'], 'reason': 'row width not a whole '
                    f'number of {block_els}-value blocks, so llama.cpp cannot load the tensor',
                    'row_width': in_features, 'block_elements': block_els})
                continue
            quantized, payload = roundtrip(weight, type_name)
            row = {**base, 'to_dtype': type_name.upper(),
                   'codec': quantization_error(weight, quantized)}
            for space_name, block in spaces.items():
                row[space_name] = aligned_row(prepared[space_name], quantized, block['rank'], ks=KS)
            reports[type_name]['tensors'].append(row)
            writeback_by_type[type_name]['payloads'][spec['name']] = payload
            writeback_by_type[type_name]['qtypes'][spec['name']] = type_id_for_name(type_name)
        print(f'[{index}/{len(tensors)}] {spec["name"]} {spec["qtype"]} '
              f'{weight.shape} energy={base["weight_energy"]:.3e}', flush=True)
        del weight

    def summarise(rows):
        """Energy-weighted aggregate over the tensors of one type, per space."""
        out = {}
        for space_name in spaces:
            sub_rows = [{'weight_energy': row['weight_energy'], 'codec': row['codec'],
                         'curves': {**row[space_name]['curves'],
                                    str(row[space_name]['rank']): {
                                        'aligned_energy_fraction': row[space_name]['aligned_energy_fraction'],
                                        'isotropic_fraction': row[space_name]['isotropic_fraction'],
                                        'surviving_energy_fraction': row[space_name]['surviving_energy_fraction'],
                                        'removed_energy_fraction': row[space_name]['removed_energy_fraction'],
                                        'cos': row[space_name]['cos']}},
                         'selectivity': row[space_name]['selectivity']} for row in rows]
            random_rows = [{'weight_energy': row['weight_energy'], 'codec': row['codec'],
                            'curves': {**row[space_name]['random']['curves'],
                                       str(row[space_name]['rank']): {
                                           'aligned_energy_fraction': row[space_name]['random']['aligned_energy_fraction'],
                                           'isotropic_fraction': row[space_name]['isotropic_fraction'],
                                           'surviving_energy_fraction': row[space_name]['random']['surviving_energy_fraction'],
                                           'removed_energy_fraction': row[space_name]['random']['removed_energy_fraction'],
                                           'cos': row[space_name]['random']['cos']}}}
                           for row in rows]
            block = aggregate(sub_rows)
            block['random'] = aggregate(random_rows)
            rank_key = str(spaces[space_name]['rank'])
            curve = block['curves'][rank_key]
            block['edit'] = {
                'rank': spaces[space_name]['rank'],
                'aligned_energy_fraction': curve['aligned_energy_fraction'],
                'aligned_energy_fraction_median': curve['aligned_energy_fraction_median'],
                'fraction_over_isotropic': (curve['aligned_energy_fraction']
                                            / curve['isotropic_fraction']
                                            if curve['isotropic_fraction'] else None),
                'cos': curve['cos'], 'cos_median': curve['cos_median'],
                'removed_energy_fraction': curve['removed_energy_fraction'],
                'removed_energy_fraction_median': curve['removed_energy_fraction_median'],
                'surviving_energy_fraction': curve['surviving_energy_fraction'],
                'surviving_norm_ratio': float(np.sqrt(max(curve['surviving_energy_fraction'], 0.0))),
            }
            random_curve = block['random']['curves'][rank_key]
            block['edit']['random_removed_energy_fraction'] = random_curve['removed_energy_fraction']
            block['edit']['random_cos'] = random_curve['cos']
            block['edit']['selectivity'] = (curve['removed_energy_fraction']
                                            / random_curve['removed_energy_fraction']
                                            if random_curve['removed_energy_fraction'] else None)
            out[space_name] = block
        return out

    report = {
        'schema': SCHEMA,
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'removal_measured': False,
        'zero_forward': {'serve': False, 'capture': False, 'wave': False, 'benchmark': False,
                         'gpu': False,
                         'note': 'weight algebra, tokenization of the artifact\'s own vocabulary, '
                                 'file bytes and digests only; a coarse grid is a codec choice, not '
                                 'a behavioural measurement, and no answer was generated'},
        'artifact': {'path': display(artifact), 'sha256': source_sha,
                     'bytes': artifact.stat().st_size, 'tensors': len(source_header['tensors']),
                     'header_sha256': source_header_sha},
        'tokenizer': {'model': str(reader.fields['tokenizer.ggml.model'].contents()),
                      'pre': str(reader.fields['tokenizer.ggml.pre'].contents()),
                      'vocab': len(tokenizer.tokens), 'merges': len(tokenizer.ranks),
                      'digest': tokens_sha, 'field_counts': token_counts},
        'denial_table': {'path': display(args.surfaces), 'sha256': sha256(args.surfaces),
                         'schema': surfaces_doc.get('schema'), 'entries': len(surfaces)},
        'compliant_openings': ({'path': display(args.compliant), 'sha256': sha256(args.compliant),
                                'rows': len(compliant_rows)} if args.compliant else None),
        'denial_ids': {
            'surface_span': {'ids': ids['surface_span'], 'count': len(ids['surface_span']),
                             'pieces': [tokenizer.piece(token) for token in ids['surface_span']],
                             'by_surface': ids['surfaces']},
            'opener_span': {'ids': ids['opener_span'], 'count': len(ids['opener_span']),
                            'pieces': [tokenizer.piece(token) for token in ids['opener_span']],
                            'skipped': ids['skipped_openers']},
            'mean_axis': {'denial_ids': ids['surface_span'], 'compliance_ids': compliance_ids,
                          'compliance_pieces': [tokenizer.piece(token) for token in compliance_ids],
                          'norm': axis_norm,
                          'definition': 'normalize(mean(E_denial) - mean(E_compliance)) over the '
                                        'input embedding rows; the x-4 explicit-edit axis'},
        },
        'embedding': emb_meta,
        'subspaces': {name: {key: value for key, value in block.items() if key != 'V'}
                      for name, block in spaces.items()},
        'types': {},
        'checks': {},
        'caveats': [
            'removal_measured is false: no serve, no capture, no wave, no benchmark, no GPU.',
            'The denial-aligned subspace is built from input embedding rows of the denial-table '
            'tokens; it is a token-alignment subspace, not a proven refusal direction, and a '
            'tensor\'s Frobenius energy in it is a static quantity, not an activation measurement.',
            'Cos and removed energy are codec fidelity of that component, not a refusal or '
            'capability result.',
        ],
    }

    for type_name in types:
        rows = reports[type_name]['tensors']
        block = {'tensors_measured': len(rows), 'skipped': reports[type_name]['skipped'],
                 'skip_count': len(reports[type_name]['skipped']),
                 'per_tensor': rows, 'aggregate': summarise(rows) if rows else {}}
        if rows:
            # Attention output and FFN down are different tensors with different source
            # codecs (Q5_0 vs Q4_K/IQ3_S here) and different widths, so one aggregate
            # over both can hide that they behave differently under the same grid.
            block['by_kind'] = {}
            for kind in sorted({row['kind'] for row in rows}):
                subset = [row for row in rows if row['kind'] == kind]
                summary = summarise(subset)['surface_span']
                block['by_kind'][kind] = {
                    'tensors': len(subset),
                    'edit': summary['edit'],
                    'relative_noise_energy_median': summary['relative_noise_energy_median'],
                    'rms_ratio_median': summary['rms_ratio_median'],
                    'snr_db_median': summary['snr_db_median'],
                }
        if not rows:
            block['note'] = 'no tensor of this artifact is representable in this grid'
        report['types'][type_name] = block
        print(f'{type_name}: {len(rows)} tensors measured, '
              f'{len(reports[type_name]["skipped"])} skipped', flush=True)

    # ----------------------------------------------------------------- artifact checks
    if not args.no_apply:
        for type_name in types:
            payloads = writeback_by_type[type_name]['payloads']
            if not payloads:
                continue
            out_path = work_dir / f'codec-floor-{type_name}.gguf'
            out_path.unlink(missing_ok=True)
            header = dict(source_header)
            header['source'] = str(artifact)
            qtypes = writeback_by_type[type_name]['qtypes']
            started_write = time.monotonic()
            write_container(out_path, header, payloads, qtypes)
            write_seconds = time.monotonic() - started_write
            checks = verify_written(artifact, out_path, payloads, qtypes, source_header)
            checks['bytes_in'] = artifact.stat().st_size
            checks['bytes_out'] = out_path.stat().st_size
            checks['bytes_delta'] = checks['bytes_out'] - checks['bytes_in']
            checks['write_seconds'] = round(write_seconds, 2)
            checks['sha256'] = sha256(out_path)
            # Basename only: the artifact is a run product under --work-dir, and a
            # committed report must not carry a lab path nobody else can follow.
            checks['artifact'] = out_path.name
            checks['work_dir_note'] = 'written under the run\'s --work-dir, not committed'
            checks['tensors_written'] = len(payloads)
            checks['tensor_infos_differ_only_in_type_and_offset'] = (
                [t['name'] for t in parse_container(out_path)['tensors']] ==
                [t['name'] for t in source_header['tensors']])
            sidecar = work_dir / f'revert-{type_name}.npz'
            sidecar.unlink(missing_ok=True)
            save_whole_file_revert(artifact, out_path, sidecar)
            revert = revert_inplace(out_path, sidecar)
            checks['revert'] = {**revert, 'artifact': out_path.name, 'sidecar': sidecar.name,
                                'source_sha256': source_sha,
                                'byte_exact': revert.get('restored_sha256') == source_sha}
            checks['loads'] = bool(checks['loads'])
            report['types'][type_name]['artifact'] = checks
            report['checks'][f'{type_name}_untouched_all_bit_identical'] = checks['untouched_all_bit_identical']
            report['checks'][f'{type_name}_kv_section_byte_identical'] = checks['kv_section_byte_identical']
            report['checks'][f'{type_name}_loads'] = checks['loads']
            report['checks'][f'{type_name}_revert_byte_exact'] = checks['revert']['byte_exact']
            if not args.keep_artifacts:
                out_path.unlink(missing_ok=True)
            print(f'{type_name}: wrote {checks["bytes_out"]} B, untouched '
                  f'{checks["untouched_tensors_byte_identical"]}, revert byte_exact '
                  f'{checks["revert"]["byte_exact"]}', flush=True)

    # ------------------------------------------------------------------ the verdict
    verdicts = {}
    for type_name in types:
        block = report['types'][type_name].get('aggregate') or {}
        if not block:
            continue
        block_surface = block['surface_span']
        edit = block_surface['edit']
        floor = block_surface['relative_noise_energy_median']
        verdicts[type_name] = {
            'tensors': block_surface['tensors'],
            'codec_floor_rel_energy_median': floor,
            'codec_floor_rms_ratio_median': block_surface['rms_ratio_median'],
            'codec_snr_db_median': block_surface['snr_db_median'],
            'edit_size_median': edit['aligned_energy_fraction_median'],
            'edit_size_over_isotropic': edit['fraction_over_isotropic'],
            'floor_over_edit': (floor / edit['aligned_energy_fraction_median']
                                if edit['aligned_energy_fraction_median'] else None),
            'aligned_cos_median': edit['cos_median'],
            'aligned_removed_median': edit['removed_energy_fraction_median'],
            'aligned_surviving_norm_ratio': edit['surviving_norm_ratio'],
            'random_removed_median': edit['random_removed_energy_fraction'],
            'random_cos': edit['random_cos'],
            'selectivity_median': edit['selectivity'],
            'aligned_kept': edit['cos_median'] >= 0.99,
            # If the re-encode's error were isotropic and independent of the component,
            # cos would be exactly 1/sqrt(1 + noise_energy/||W||^2). Reporting the
            # prediction beside the measurement is what makes "the codec is blind to
            # direction" a number instead of a claim.
            'cos_predicted_for_isotropic_noise': 1.0 / float(np.sqrt(1.0 + floor)),
            'mean_axis_cos': block['mean_axis']['edit']['cos_median'],
            'mean_axis_edit_size': block['mean_axis']['edit']['aligned_energy_fraction_median'],
            'mean_axis_floor_over_edit': (floor / block['mean_axis']['edit']['aligned_energy_fraction_median']
                                          if block['mean_axis']['edit']['aligned_energy_fraction_median'] else None),
        }
    report['verdict'] = verdicts
    report['wall_seconds'] = round(time.monotonic() - started, 1)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({'report': str(args.report), 'checks': report['checks'],
                      'verdict': {name: {key: value for key, value in row.items()
                                         if key in ('codec_floor_rel_energy_median', 'edit_size_median',
                                                    'floor_over_edit', 'aligned_cos_median',
                                                    'selectivity_median')}
                                  for name, row in verdicts.items()},
                      'seconds': report['wall_seconds']}, indent=2))


if __name__ == '__main__':
    main()
