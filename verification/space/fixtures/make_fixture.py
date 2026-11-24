# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regenerate the bank-check fixtures: a 2-layer GGUF, its bank entry, a bad copy.

Run from the repo root with the project venv:

    python verification/space/fixtures/make_fixture.py

Why a fixture at all: `bank-check` must be provable without a real multi-GB
artifact in the clone. The file below carries the same *shape* of evidence the
loader reads on a real model — `blk.N.attn_output` and `blk.N.ffn_down` at the
hidden width, plus a hidden-wide `attn_q` that must NOT be discovered as a
writeback target — at 2 layers x 256 wide, so the whole fixture is ~0.5 MB.

The bundle is a plain tar with a fixed member mtime, so its sha256 is
reproducible and the entry can pin it in `bank.json`.
"""
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

HERE = Path(__file__).resolve().parent
ARCH = 'bankfixture'
LAYERS, HIDDEN, FFN = 2, 256, 512
KEY = 'bankfixture/f00l01-fixture'
BUNDLE = 'fixture-bank.tar'


def write_gguf(path):
    rng = np.random.default_rng(20261123)
    writer = GGUFWriter(str(path), ARCH)
    writer.add_name('bank-fixture-2layer-f16')
    for layer in range(LAYERS):
        # Residual-write projections: output width == HIDDEN.
        writer.add_tensor(f'blk.{layer}.attn_output.weight',
                          rng.standard_normal((HIDDEN, HIDDEN)).astype(np.float16))
        writer.add_tensor(f'blk.{layer}.ffn_down.weight',
                          rng.standard_normal((HIDDEN, FFN)).astype(np.float16))
        # Hidden-wide but NOT a writeback target (query space, not residual).
        writer.add_tensor(f'blk.{layer}.attn_q.weight',
                          rng.standard_normal((HIDDEN, HIDDEN)).astype(np.float16))
    writer.open_output_file()
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def write_bundle(path):
    payload = json.dumps({'schema': 'jdewitt-space-bank-fixture-v1',
                          'geometry': {'layers': LAYERS, 'rank': 1, 'hidden': HIDDEN},
                          'axis': 'fixture-only: no real direction is shipped'}).encode()
    with tarfile.open(path, 'w') as tar:
        info = tarfile.TarInfo('fixture-plan.json')
        info.size = len(payload)
        info.mtime = 0
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(payload))


def entry(sha256):
    return {
        'key': KEY,
        'base': {'repo': 'jdewitt/bank-fixture', 'revision': 'fixture'},
        'architecture': ARCH,
        'geometry': {'layers': LAYERS, 'hidden': HIDDEN, 'hidden_source': 'artifact'},
        'targets': {'attn': 'attn_output', 'mlp': 'ffn_down'},
        'footprint': {'first': 0, 'last': LAYERS - 1, 'fraction': [0.0, (LAYERS - 1) / LAYERS]},
        'strength': {'attn': 1.5, 'mlp': 1.5},
        'edit': {'method': 'directional', 'shape': 'shared', 'space': 'output',
                 'row_normalization': 'none', 'min_cos': 0.95,
                 'types': ['q8_0', 'f16'], 'repair': 'none'},
        'sources': {'quant_types': ['f16']},
        'measurement': {'state': 'banked_unmeasured', 'artifact_class': None,
                        'removal': None, 'retention': None, 'config': None},
        'bundle': {'file': BUNDLE, 'sha256': sha256},
    }


def main():
    import hashlib

    gguf = HERE / 'fixture-2layer-f16.gguf'
    write_gguf(gguf)

    bundle = HERE / 'bank' / BUNDLE
    bundle.parent.mkdir(parents=True, exist_ok=True)
    write_bundle(bundle)
    sha256 = hashlib.sha256(bundle.read_bytes()).hexdigest()
    (bundle.parent / 'bank.json').write_text(
        json.dumps({'schema': 'jdewitt-space-bank-v1', 'entries': [entry(sha256)]},
                   indent=2) + '\n')

    # A copy of the SAME bundle with one byte flipped, pinned at the good digest:
    # bank-check must refuse before reading the axis.
    corrupt_dir = HERE / 'bank-bad-digest'
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    corrupt = bytearray(bundle.read_bytes())
    corrupt[-1] ^= 0xFF
    corrupt_path = corrupt_dir / 'fixture-bank-corrupt.tar'
    corrupt_path.write_bytes(bytes(corrupt))
    corrupt_entry = entry(sha256)
    corrupt_entry['bundle'] = {'file': corrupt_path.name, 'sha256': sha256}
    (corrupt_dir / 'bank.json').write_text(
        json.dumps({'schema': 'jdewitt-space-bank-v1', 'entries': [corrupt_entry]},
                   indent=2) + '\n')

    print(json.dumps({'gguf': str(gguf), 'bundle_sha256': sha256,
                      'bank': str(bundle.parent), 'bad_digest_bank': str(corrupt_dir)},
                     indent=2))


if __name__ == '__main__':
    main()
