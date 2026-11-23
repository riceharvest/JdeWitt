# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regenerate the bank-apply fixtures: a 2-layer q8_0 GGUF and a plan-bundle bank.

Run from the repo root with the project venv (this script needs torch: a bank entry
pins a REGISTRY PLAN BUNDLE, whose payload is a ``plan.pt`` with a real
``[layers, rank, hidden]`` basis tensor):

    python verification/space/fixtures/make_apply_fixture.py

Why a second fixture and not the bank-check one: `bank-check` proves the loader reads
a header, so its bundle is a plain JSON tar and the script stays torch-free. The
apply path also has to install a bundle through ``registry.unbundle_plan`` and load a
3D basis through ``registry.load_plan``, which is a registry artifact, so the fixture
here is a real one: 2 layers x hidden 64 x rank 1, a q8_0 artifact beside it, and the
entry pinning the bundle's sha256.

The bundle's tar members carry a fixed mtime and ``torch.save`` is byte-stable for a
fixed payload, so re-running this script reproduces the same sha256 on the same torch
build; the pin lives in ``bank.json`` and re-running rewrites it either way.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter, GGMLQuantizationType

from jdewitt.answers import atomic_json
from jdewitt.data import digest
from jdewitt.quants import CODECS
from jdewitt.registry import bundle_plan

HERE = Path(__file__).resolve().parent
DEST = HERE / 'bank-apply'
ARCH = 'applyfixture'
LAYERS, HIDDEN, FFN = 2, 64, 128
ENTRY_KEY = 'applyfixture/f00l01-q8'
PLAN_KEY = 'applyfixturef00l01q8'
BUNDLE = 'fixture-apply-plan.tar'
GGUF = DEST / 'fixture-2layer-q8_0.gguf'


def _q8(matrix):
    """(packed bytes, byte shape) for one q8_0 tensor, in gguf-py's raw form.

    ``matrix`` is torch-ordered [out, in]. The byte shape is (out rows, packed bytes
    per row) and gguf-py recovers [in, out] from it; GGUFReader then hands back
    [in, out], which is what ``gguf_weight_shapes`` reads.
    """
    _block_bytes, block_els, encode, _decode = CODECS['Q8_0']
    rows = int(matrix.shape[0])
    flat = np.ascontiguousarray(matrix.reshape(-1), dtype=np.float32)
    raw = np.asarray(encode(flat), dtype=np.uint8).reshape(-1)
    return raw.reshape(rows, raw.size // rows), (rows, raw.size // rows)


def write_gguf(path):
    rng = np.random.default_rng(20261123)
    writer = GGUFWriter(str(path), ARCH)
    writer.add_name('bank-apply-fixture-2layer-q8_0')
    writer.add_block_count(LAYERS)
    writer.add_embedding_length(HIDDEN)
    for layer in range(LAYERS):
        # Residual-write projections: output width == HIDDEN.
        tensors = {
            f'blk.{layer}.attn_output.weight': rng.standard_normal((HIDDEN, HIDDEN)),
            f'blk.{layer}.ffn_down.weight': rng.standard_normal((HIDDEN, FFN)),
            # Hidden-wide but NOT a residual-write projection: must stay untouched.
            f'blk.{layer}.attn_q.weight': rng.standard_normal((HIDDEN, HIDDEN)),
        }
        for name, matrix in tensors.items():
            packed, shape = _q8(matrix.astype(np.float32))
            writer.add_tensor(name, packed, raw_shape=shape, raw_dtype=GGMLQuantizationType.Q8_0)
    writer.open_output_file()
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def write_plan(registry):
    """A rank-1 basis of [LAYERS, 1, HIDDEN], registered so bundle_plan can sign it."""
    rng = np.random.default_rng(7)
    basis = rng.standard_normal((LAYERS, 1, HIDDEN)).astype(np.float32)
    basis /= np.linalg.norm(basis, axis=-1, keepdims=True)
    plan = {'basis': torch.from_numpy(basis), 'layers': list(range(LAYERS)),
            'method': 'directional', 'fixture': 'verification/space/fixtures/bank-apply'}
    directory = Path(registry) / 'plans' / PLAN_KEY
    directory.mkdir(parents=True)
    payload = directory / 'plan.pt'
    with payload.open('wb') as stream:
        torch.save(plan, stream)
    meta = {'version': 1, 'key': PLAN_KEY, 'registered_at': '2026-11-23T00:00:00Z',
            'identity_digest': digest({'fixture': ARCH}), 'payload_sha256': _sha(payload),
            'model': 'bank-apply-fixture', 'revision': 'fixture',
            'method': 'directional', 'method_options': {}, 'rank': 1, 'seed': 7,
            'generation_fingerprint': {'architecture': ARCH, 'layers': LAYERS,
                                       'hidden_size': HIDDEN, 'rank': 1},
            'suite_digests': {'directions': None, 'selection': None, 'holdout': None},
            'gates': None, 'measured': False}
    meta['meta_digest'] = digest(meta)
    atomic_json(directory / 'meta.json', meta)
    return meta


def _sha(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def entry(bundle_sha256):
    return {
        'key': ENTRY_KEY,
        'base': {'repo': 'jdewitt/bank-apply-fixture', 'revision': 'fixture'},
        'architecture': ARCH,
        'geometry': {'layers': LAYERS, 'hidden': HIDDEN, 'hidden_source': 'artifact'},
        'targets': {'attn': 'attn_output', 'mlp': 'ffn_down'},
        'footprint': {'first': 0, 'last': LAYERS - 1,
                      'fraction': [0.0, (LAYERS - 1) / LAYERS]},
        # Two DIFFERENT strengths on purpose: a writer that flattens them to one
        # number cannot reproduce both delivered cos readings.
        'strength': {'attn': 1.5, 'mlp': 0.75},
        'edit': {'method': 'directional', 'shape': 'shared', 'space': 'output',
                 # 0.999 forces the ladder one rung up on a q8_0 source: q8_0
                 # stores this edit at ~0.998, f16 at ~0.99999, and AUTO must take
                 # the first that clears the entry's own floor.
                 'row_normalization': 'none', 'min_cos': 0.999,
                 'types': ['q8_0', 'f16'], 'repair': 'none'},
        'sources': {'quant_types': ['q8_0']},
        'measurement': {'state': 'banked_unmeasured', 'artifact_class': None,
                        'removal': None, 'retention': None, 'config': None},
        'bundle': {'file': BUNDLE, 'sha256': bundle_sha256},
    }


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    write_gguf(GGUF)
    with tempfile.TemporaryDirectory(prefix='jdewitt-apply-fixture-') as registry:
        write_plan(registry)
        bundle = DEST / BUNDLE
        if bundle.exists():
            bundle.unlink()
        written = bundle_plan(registry, PLAN_KEY, bundle)
    members = _verify_members(DEST / BUNDLE)
    if len(members) != 2:
        raise SystemExit(f'bundle members {members} are not the registry pair')
    (DEST / 'bank.json').write_text(
        json.dumps({'schema': 'jdewitt-space-bank-v1',
                    'entries': [entry(written['bundle_sha256'])]}, indent=2) + '\n')
    print(json.dumps({'bank': str(DEST), 'gguf': str(GGUF), 'entry': ENTRY_KEY,
                      'bundle_sha256': written['bundle_sha256']}, indent=2))


def _verify_members(path):
    """The bundle is an uncompressed tar with exactly the registry's two members."""
    import tarfile

    with tarfile.open(path, 'r') as tar:
        return tar.getnames()


if __name__ == '__main__':
    main()
