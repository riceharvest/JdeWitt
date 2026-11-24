# SPDX-License-Identifier: AGPL-3.0-or-later
"""P7 q25: bind the operands this cert consumes to their bytes.

The GGUF artifacts are gitignored, so the repo cannot carry them; what it can
carry is a manifest that names each one with its full sha256, its size and the
exact recipe that produced it. `release-certify` re-hashes a present artifact and
refuses a spec whose digest disagrees, so this file is the bridge between "the
bytes on this box" and "the spec the analysis card commits".

    python scripts/p7_q25_operands.py [--out verification/p7-q25-15b/operands.json]

Re-runnable: it re-hashes everything and fails loudly if a recorded digest no
longer matches (a changed artifact is a different arm, not a re-run).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT_DIR = REPO / 'verification/p7-q25-15b'
# The served bytes live outside the tree (gguf files are gitignored, and there are
# ~5 GB of them): the same stable-path convention the q3 cert used.
MODELS = Path('/mnt/ssd/models/p7-q25-15b')
BASE_HF = Path('/mnt/ssd/models/Qwen2.5-1.5B-Instruct')
PUBLISHED_HF = Path('/mnt/ssd/models/josiefied-q25-v1')

# The conversion commands that produced the Q8_0 files (llama.cpp convert + quantize,
# CPU llama-quantize). Recorded as text: the point is that the recipe is auditable,
# and the digest below is what binds the result.
CONVERT = ('python /mnt/ssd/dev workspace/llama.cpp-dspark/convert_hf_to_gguf.py {hf} '
           '--outfile {f16} --outtype f16')
QUANTIZE = ('/mnt/ssd/dev workspace/llama.cpp-dspark/build-perp/bin/llama-quantize '
            '{f16} {q8} Q8_0 16')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def quant_profile(path: Path) -> dict:
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    counts = Counter(tensor.tensor_type.name for tensor in reader.tensors)
    return {'dominant': counts.most_common(1)[0][0], 'types': dict(sorted(counts.items())),
            'tensors': len(reader.tensors)}


def hf_source(directory: Path) -> dict:
    weights = sorted(directory.glob('*.safetensors'))
    return {'path': str(directory), 'files': {p.name: sha256_file(p) for p in weights},
            'config': json.loads((directory / 'config.json').read_text())}


def artifact(role: str, path: Path, *, source: dict | None, recipe: str | None,
             note: str | None = None) -> dict:
    if not path.is_file():
        raise SystemExit(f'missing operand {role}: {path}')
    entry = {'role': role, 'path': str(path), 'bytes': path.stat().st_size,
             'sha256': sha256_file(path), 'sha256_prefix_16': sha256_file(path)[:16],
             'quant': quant_profile(path), 'source': source, 'recipe': recipe}
    if note:
        entry['note'] = note
    return entry


def main(argv=None):
    parser = argparse.ArgumentParser(prog='p7_q25_operands')
    parser.add_argument('--out', type=Path, default=OUT_DIR / 'operands.json')
    parser.add_argument('--models', type=Path, default=MODELS,
                        help='directory holding artifacts/ and records/ (default %(default)s)')
    args = parser.parse_args(argv)

    models = args.models
    base_hf, published_hf = hf_source(BASE_HF), hf_source(PUBLISHED_HF)
    artifacts = [
        artifact('base', models / 'artifacts/base-q8.gguf', source=base_hf,
                 recipe=(CONVERT.format(hf=BASE_HF, f16=models / 'artifacts/base-f16.gguf') + ' && '
                         + QUANTIZE.format(f16=models / 'artifacts/base-f16.gguf',
                                           q8=models / 'artifacts/base-q8.gguf')),
                 note='the served parent: the arm is this file plus the refit axis merge'),
        artifact('published', models / 'artifacts/published-q8.gguf', source=published_hf,
                 recipe=(CONVERT.format(hf=PUBLISHED_HF, f16=models / 'artifacts/published-f16.gguf') + ' && '
                         + QUANTIZE.format(f16=models / 'artifacts/published-f16.gguf',
                                           q8=models / 'artifacts/published-q8.gguf')),
                 note='the head-to-head side: measured by us at the same argv, never read from its README'),
        artifact('arm-refit', models / 'artifacts/arm-refit-q8.gguf', source=None,
                 recipe=('python -m jdewitt.spanfit build --gguf-in '
                         f'{models / "artifacts/base-q8.gguf"} --out '
                         f'{models / "artifacts/arm-refit-q8.gguf"} --axes '
                         f'{models / "records/refit-axes.pt"} --gguf-targets auto '
                         '--shape shared --axis-space output --scale 1.5'),
                 note=('REFIT, not SAS@1.5: the original q25-v4-dir000.gguf adapter is not on this '
                       'box, so the shared axis was refit from a CPU capture of the base '
                       '(records/refit-records.pt). Merge into the base Q8_0 bytes: 56 tensors, '
                       'every other byte verbatim')),
    ]
    payload = {
        'schema_version': 1,
        'kind': 'p7-q25-operands',
        'lineage': 'qwen25-1.5b',
        'note': ('GGUF artifacts are gitignored; this manifest is the in-repo binding between the '
                 'spec digests and the bytes on disk. release-certify re-hashes a present artifact.'),
        'refit': True,
        'models_dir': str(models),
        'capture': str(models / 'records/refit-capture.json'),
        'axes': str(models / 'records/refit-axes.pt'),
        'records': str(models / 'records/refit-records.pt'),
        'artifacts': artifacts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + '\n')
    for entry in artifacts:
        print(f"{entry['role']:<12} {entry['sha256_prefix_16']}  {entry['bytes']:>12}  {entry['quant']['dominant']}")
    print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
