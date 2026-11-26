#!/usr/bin/env python
"""Apply the chosen rank-k alignment-delta removal to a quantized GGUF.

Two arms, same float32 deltas:

  * ``inplace`` — patch the artifact's own bytes, preserving each tensor's codec
    and block count. A sidecar keeps the pre-patch bytes, so the revert restores
    the source file and its sha256 is compared with the original.
  * ``reencode`` — re-encode only the edited tensors at F16 (no block-size row
    constraint, so the artifact stays readable by gguf-py) so the delta is not
    lost below a 3/4-bit grid; everything else is copied byte for byte.

Both arms are static: no model, no serve, no token generated. The report says
``removal_measured: false`` and reports realized *codec* cos, not refusal.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

# Same cap as the experiment: the board's CPU is shared, and a 32-thread BLAS
# spends more time contending than computing.
for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '4')

import numpy as np  # noqa: E402  (the thread cap must be set before numpy loads)

from jdewitt.ggufio import edit_tensors_inplace, revert_inplace  # noqa: E402
from jdewitt.ggufwrite import parse_container  # noqa: E402
from jdewitt.rewrite import rewrite_with_edits  # noqa: E402


def sha256(path, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


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
            if isinstance(value, bytes):
                raw = value
            elif isinstance(value, str):
                raw = value.encode('utf-8')
            else:
                raw = str(value).encode()
            digest.update(len(raw).to_bytes(8, 'little'))
            digest.update(raw)
    return digest.hexdigest(), counts


def diff_summary(before, after, edited_names, limit=100):
    from jdewitt.agent import gguf_tensor_diff

    diff = gguf_tensor_diff(before, after, limit=limit)
    differing = {row['name'] for row in diff['differing_tensors']}
    total = diff['tensor_count_before']
    untouched = set()
    for path in (before,):
        header = parse_container(path)
        untouched = {tensor['name'] for tensor in header['tensors']} - set(edited_names)
    leaked = sorted(untouched & differing)
    return {
        'tensors_before': total,
        'tensors_after': diff['tensor_count_after'],
        'untouched_count': len(untouched),
        'bit_identical': diff['summary']['bit_identical'],
        'value_differs': diff['summary']['value_differs'],
        'differing_count': len(differing),
        'differing_outside_edit_set': leaked,
        'only_in_before': diff['only_in_before'],
        'only_in_after': diff['only_in_after'],
        'untouched_all_bit_identical': not leaked and diff['summary']['bit_identical'] >= len(untouched),
        'rel_rms': diff['rel_rms'],
        'differing_tensors': diff['differing_tensors'],
    }


def fidelity(source, edited, deltas):
    """Realized codec fidelity, one reader per file.

    ``ggufio.compare_edit`` is the canonical implementation, but it constructs a
    fresh ``GGUFReader`` for every tensor — on this artifact that re-parses the
    151k-entry tokenizer 2N times. The arithmetic here is the same cos / norm /
    SNR between ``dequant(edited) - dequant(source)`` and the intended delta.
    """
    import numpy as np

    from jdewitt.ggufio import decode_values

    def load_all(path):
        from gguf import GGUFReader

        reader = GGUFReader(str(path))
        index = {tensor.name: tensor for tensor in reader.tensors}
        values = {}
        handle = open(path, 'rb')
        try:
            for name in deltas:
                tensor = index[name]
                logical = tuple(int(x) for x in np.asarray(tensor.shape).ravel())[::-1]
                handle.seek(int(tensor.data_offset))
                raw = np.frombuffer(handle.read(int(tensor.n_bytes)), dtype=np.uint8)
                decoded = decode_values(raw, tensor.tensor_type.name).astype(np.float32)
                values[name] = decoded.reshape(logical)
        finally:
            handle.close()
        return values

    base = load_all(source)
    final = load_all(edited)
    rows = []
    for name in deltas:
        intended = np.asarray(deltas[name], dtype=np.float32).reshape(base[name].shape)
        realized = final[name] - base[name]
        intended_norm = float(np.linalg.norm(intended))
        realized_norm = float(np.linalg.norm(realized))
        noise = float(np.linalg.norm(realized - intended))
        denominator = realized_norm * intended_norm
        cosine = float(np.vdot(realized, intended).real / denominator) if denominator else 0.0
        rows.append({
            'name': name,
            'cos': round(cosine, 4),
            'norm_ratio': round(realized_norm / (intended_norm + 1e-12), 4),
            'snr': round(intended_norm / (noise + 1e-12), 3),
        })
    cos = sorted(row['cos'] for row in rows)
    snr = sorted(row['snr'] for row in rows)
    norm = sorted(row['norm_ratio'] for row in rows)
    return {
        'tensors': len(rows),
        'cos_min': cos[0] if cos else None,
        'cos_mean': round(sum(cos) / len(cos), 4) if cos else None,
        'cos_median': cos[len(cos) // 2] if cos else None,
        'cos_max': cos[-1] if cos else None,
        'snr_median': snr[len(snr) // 2] if snr else None,
        'norm_ratio_median': norm[len(norm) // 2] if norm else None,
        'per_tensor': rows,
        'note': 'realized codec fidelity (dequant(edited) - dequant(source) against the intended '
                'delta); it is not a removal or capability measurement',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--deltas', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--edit-type', default='f16',
                        help='re-encode type for the out-of-place arm; F16 has no block-size row '
                             'constraint, so gguf-py can still parse an 896-wide o_proj')
    parser.add_argument('--arms', default='inplace,reencode')
    parser.add_argument('--inplace-skip-types', default='IQ3_S',
                        help='codecs whose pure-numpy encoder is too slow for the in-place arm; '
                             'they are left untouched there and covered by the re-encode arm')
    parser.add_argument('--keep-artifacts', action='store_true')
    args = parser.parse_args()

    manifest = json.loads((args.deltas / 'manifest.json').read_text())
    deltas = {}
    for path in sorted(args.deltas.glob('*.npy')):
        deltas[path.stem] = np.load(path).astype(np.float32)
    if sorted(deltas) != sorted(manifest['names']):
        raise SystemExit('delta files do not match the manifest')

    args.work_dir.mkdir(parents=True, exist_ok=True)
    source = args.gguf
    source_sha = sha256(source)
    source_header_sha, source_header = header_digest(source)
    from gguf import GGUFReader

    reader = GGUFReader(str(source))
    tokens_sha, token_counts = tokenizer_digest(reader)
    selected = {name for name in manifest['names']}
    declared = {tensor['name'] for tensor in source_header['tensors']}
    if not selected <= declared:
        raise SystemExit(f'deltas name tensors not in the artifact: {sorted(selected - declared)}')

    report = {
        'schema': 'jdewitt-alignment-apply-v1',
        'removal_measured': False,
        'source': {'path': str(source), 'sha256': source_sha,
                   'bytes': source.stat().st_size,
                   'tensors': len(source_header['tensors']),
                   'header_sha256': source_header_sha},
        'deltas': {'dir': str(args.deltas), 'k': manifest['k'], 'tensors': len(deltas),
                   'selection': manifest.get('selection'), 'names': sorted(deltas)},
        'tokenizer': {'digest': tokens_sha, 'field_counts': token_counts},
        'arms': {},
        'checks': {},
        'caveats': [
            'removal_measured is false: no serve, no capture, no wave, no benchmark.',
            'Realized cos is codec fidelity (did the tile store the intended delta), not removal.',
            'The alignment delta also carries instruction-following; a capability cost is '
            'expected and unmeasured.',
        ],
    }

    if 'inplace' in args.arms:
        skip_types = {value.strip() for value in args.inplace_skip_types.split(',') if value.strip()}
        dtypes = {tensor.name: tensor.tensor_type.name for tensor in reader.tensors}
        skipped = {name: dtypes[name] for name in deltas if dtypes.get(name) in skip_types}
        patched = {name: delta for name, delta in deltas.items() if name not in skipped}
        work = args.work_dir / 'inplace.gguf'
        shutil.copyfile(source, work)
        sidecar = args.work_dir / 'inplace.revert.npz'
        patch = edit_tensors_inplace(work, patched, revert_path=sidecar)
        edited_sha = sha256(work)
        edited_header_sha, _ = header_digest(work)
        diff = diff_summary(source, work, set(patched))
        edit_tokens_sha, _ = tokenizer_digest(GGUFReader(str(work)))
        arm = {
            'artifact_sha256': edited_sha,
            'header_sha256': edited_header_sha,
            'metadata_bytes_identical': edited_header_sha == source_header_sha,
            'tokenizer_identical': edit_tokens_sha == tokens_sha,
            'patched_tensors': len(patched),
            'skipped_tensors': [
                {'name': name, 'dtype': dtype,
                 'reason': f'{dtype} encoder is the slow path; covered by the re-encode arm'}
                for name, dtype in sorted(skipped.items())],
            'skipped_count': len(skipped),
            'tensors_changed': patch['tensors_changed'],
            'changed_blocks_total': patch['changed_blocks_total'],
            'revert_sidecar_bytes': patch.get('revert_sidecar_bytes', 0),
            'per_tensor': patch['tensors'],
            'untouched': diff,
            'fidelity': fidelity(source, work, patched),
        }
        if sidecar.exists():
            revert = revert_inplace(work, sidecar)
            revert_sha = sha256(work)
        else:
            # No block changed: the codec grid absorbed every delta, so there is
            # nothing to restore and the artifact is already the source bytes.
            revert = {'tiles_restored': 0, 'note': 'no block changed; artifact unchanged'}
            revert_sha = sha256(work)
        arm['revert'] = {**revert, 'restored_sha256': revert_sha,
                         'source_sha256': source_sha,
                         'byte_exact': revert_sha == source_sha}
        report['arms']['inplace_preserve_codec'] = arm
        report['checks']['inplace_untouched_all_bit_identical'] = diff['untouched_all_bit_identical']
        report['checks']['inplace_metadata_identical'] = edited_header_sha == source_header_sha
        report['checks']['inplace_tokenizer_identical'] = edit_tokens_sha == tokens_sha
        report['checks']['inplace_revert_byte_exact'] = revert_sha == source_sha
        report['checks']['inplace_covers_all_deltas'] = len(skipped) == 0
        if not args.keep_artifacts:
            work.unlink(missing_ok=True)

    if 'reencode' in args.arms:
        work = args.work_dir / f'edited.{args.edit_type}.gguf'
        rewrite = rewrite_with_edits(source, work, deltas, edit_type=args.edit_type,
                                     verify_untouched=True)
        edited_sha = sha256(work)
        edited_header_sha, _ = header_digest(work)
        diff = diff_summary(source, work, selected)
        edit_tokens_sha, _ = tokenizer_digest(GGUFReader(str(work)))
        arm = {
            'artifact_sha256': edited_sha,
            'header_sha256': edited_header_sha,
            'metadata_bytes_identical': edited_header_sha == source_header_sha,
            'tokenizer_identical': edit_tokens_sha == tokens_sha,
            'bytes_in': rewrite['bytes_in'],
            'bytes_out': rewrite['bytes_out'],
            'fingerprint': rewrite['fingerprint'],
            'untouched_block': rewrite.get('untouched'),
            'untouched': diff,
            'fidelity': fidelity(source, work, deltas),
        }
        report['arms'][f'out_of_place_{args.edit_type}'] = arm
        report['checks']['reencode_untouched_all_bit_identical'] = diff['untouched_all_bit_identical']
        report['checks']['reencode_tokenizer_identical'] = edit_tokens_sha == tokens_sha
        report['checks']['reencode_untouched_claimed'] = bool(rewrite.get('untouched', {}).get('verified'))
        if not args.keep_artifacts:
            work.unlink(missing_ok=True)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({'report': str(args.report), 'checks': report['checks'],
                      'arms': {name: {'tensors_changed': arm.get('tensors_changed'),
                                      'changed_blocks_total': arm.get('changed_blocks_total'),
                                      'cos_median': arm['fidelity']['cos_median'],
                                      'artifact_sha256': arm['artifact_sha256']}
                               for name, arm in report['arms'].items()}}, indent=2))


if __name__ == '__main__':
    main()
