# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regenerate the Space loop fixture: three repos, one measured bank entry, a hub.

Run from the repo root with the project venv (this script needs torch: a bank entry
pins a REGISTRY PLAN BUNDLE whose payload carries a real ``[layers, rank, hidden]``
basis):

    python verification/space/fixtures/make_loop_fixture.py

What it builds, in ``verification/space/fixtures/loop/``:

- ``bank/bank.json`` — ONE entry, fitted on ``jdewitt/loop-fixture`` at ``base-rev``
  and measured on that repo's own ``Q4_K_M``. Its ``measurement.artifact_class``
  names that class, so the base renders ``measured_artifact_class`` and a fine-tune
  of the same architecture renders ``banked_unmeasured``;
- ``bank/loop-plan.tar`` — the rank-1 basis the entry pins (2 layers, hidden 256);
- ``bank/licences.json`` + ``SHA256SUMS`` — the allow-list the licence gate reads;
- ``repos/<owner>__<name>/*.gguf`` — four small artifacts: the base at Q4_K_M and
  Q8_0 (the second one exercises "shown disabled with the reason"), a fine-tune at
  Q4_K_M (same architecture and geometry, different bytes), and an artifact whose
  architecture no entry covers (the refusal);
- ``hub.json`` — the fixture "Hub": file names, sizes, the header prefix length that
  still parses, revisions, licences, and the two repos the dry run uses for the
  unmeasured and the refused states.

``header_bytes`` is measured, not guessed: the script writes the first N bytes and
truncates the copy to the artifact's full size (a sparse file, no tensor data), then
asks ``GGUFReader`` to parse it. The smallest N that parses is what the fixture hub
serves for the header-only coverage step.
"""
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType

from jdewitt.answers import atomic_json
from jdewitt.data import digest
from jdewitt.quants import CODECS, QK_K
from jdewitt.registry import bundle_plan

HERE = Path(__file__).resolve().parent
DEST = HERE / 'loop'
ARCH = 'loopfixture'
OTHER_ARCH = 'otherach'
LAYERS, HIDDEN, FFN = 2, 256, 512
ENTRY_KEY = 'loopfixture/base-q4k'
PLAN_KEY = 'loopfixturebaseq4k'
BUNDLE = 'loop-plan.tar'
BASE_REPO = 'jdewitt/loop-fixture'
FT_REPO = 'jdewitt/loop-fixture-ft'
OTHER_REPO = 'jdewitt/loop-other'
BASE_REV, FT_REV = 'base-rev', 'ft-rev'

FILES = (
    # (repo, file name, architecture, seed, codec)
    (BASE_REPO, 'model-Q4_K_M.gguf', ARCH, 11, 'Q4_K'),
    (BASE_REPO, 'model-Q8_0.gguf', ARCH, 11, 'Q8_0'),
    (FT_REPO, 'model-Q4_K_M.gguf', ARCH, 29, 'Q4_K'),
    (OTHER_REPO, 'model-Q4_K_M.gguf', OTHER_ARCH, 41, 'Q4_K'),
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _packed(matrix, codec):
    """(packed bytes, byte shape) for one tensor in gguf-py's raw form."""
    block_bytes, block_els, encode, _decode = CODECS[codec]
    rows = int(matrix.shape[0])
    flat = np.ascontiguousarray(matrix.reshape(-1), dtype=np.float32)
    if flat.size % block_els:
        raise SystemExit(f'{codec} needs rows divisible by {block_els}, got {matrix.shape}')
    raw = np.asarray(encode(flat), dtype=np.uint8).reshape(-1)
    per_row = block_bytes * (matrix.shape[1] // block_els)
    return raw.reshape(rows, per_row), (rows, per_row)


def write_gguf(path, architecture, seed, codec):
    rng = np.random.default_rng(seed)
    writer = GGUFWriter(str(path), architecture)
    writer.add_name(f'loop-fixture-{codec.lower()}')
    writer.add_block_count(LAYERS)
    writer.add_embedding_length(HIDDEN)
    for layer in range(LAYERS):
        tensors = {
            f'blk.{layer}.attn_output.weight': rng.standard_normal((HIDDEN, HIDDEN)),
            f'blk.{layer}.ffn_down.weight': rng.standard_normal((HIDDEN, FFN)),
            f'blk.{layer}.attn_q.weight': rng.standard_normal((HIDDEN, HIDDEN)),
        }
        for name, matrix in tensors.items():
            packed, shape = _packed(matrix.astype(np.float32), codec)
            writer.add_tensor(name, packed, raw_shape=shape,
                              raw_dtype=GGMLQuantizationType[codec])
    writer.open_output_file()
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def header_bytes(path, expected_tensors):
    """Smallest 512-byte-aligned prefix that GGUFReader parses as a sparse file."""
    size = os.path.getsize(path)
    with open(path, 'rb') as stream:
        start = stream.read(1 << 20)
    probe = DEST / 'header-probe.gguf'
    for cut in (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536):
        if cut > size:
            break
        with open(probe, 'wb') as stream:
            stream.write(start[:cut])
            stream.truncate(size)
        try:
            reader = GGUFReader(str(probe))
        except Exception:
            continue
        if len(reader.tensors) == expected_tensors:
            probe.unlink()
            return cut
    probe.unlink(missing_ok=True)
    raise SystemExit(f'no header prefix of {path} parses; the fixture cannot be header-only')


def write_plan(registry):
    """A rank-1 basis of [LAYERS, 1, HIDDEN], registered so bundle_plan can sign it."""
    rng = np.random.default_rng(7)
    basis = rng.standard_normal((LAYERS, 1, HIDDEN)).astype(np.float32)
    basis /= np.linalg.norm(basis, axis=-1, keepdims=True)
    plan = {'basis': torch.from_numpy(basis), 'layers': list(range(LAYERS)),
            'method': 'directional', 'fixture': 'verification/space/fixtures/loop'}
    directory = Path(registry) / 'plans' / PLAN_KEY
    directory.mkdir(parents=True)
    payload = directory / 'plan.pt'
    with payload.open('wb') as stream:
        torch.save(plan, stream)
    meta = {'version': 1, 'key': PLAN_KEY, 'registered_at': '2026-11-23T00:00:00Z',
            'identity_digest': digest({'fixture': ARCH}), 'payload_sha256': _sha(payload),
            'model': 'loop-fixture', 'revision': 'base-rev',
            'method': 'directional', 'method_options': {}, 'rank': 1, 'seed': 7,
            'generation_fingerprint': {'architecture': ARCH, 'layers': LAYERS,
                                       'hidden_size': HIDDEN, 'rank': 1},
            'suite_digests': {'directions': None, 'selection': None, 'holdout': None},
            'gates': None, 'measured': True}
    meta['meta_digest'] = digest(meta)
    atomic_json(directory / 'meta.json', meta)
    return meta


def entry(bundle_sha256):
    return {
        'key': ENTRY_KEY,
        'base': {'repo': BASE_REPO, 'revision': BASE_REV},
        'architecture': ARCH,
        'geometry': {'layers': LAYERS, 'hidden': HIDDEN, 'hidden_source': 'artifact'},
        'targets': {'attn': 'attn_output', 'mlp': 'ffn_down'},
        'footprint': {'first': 0, 'last': LAYERS - 1,
                      'fraction': [0.0, (LAYERS - 1) / LAYERS]},
        # Two different strengths: a writer that flattens them cannot reproduce
        # both delivered cos readings, and the card names them per kind.
        'strength': {'attn': 1.5, 'mlp': 0.75},
        'edit': {'method': 'directional', 'shape': 'shared', 'space': 'output',
                 'row_normalization': 'none', 'min_cos': 0.999,
                 'types': ['q8_0', 'f16'], 'repair': 'none'},
        'sources': {'quant_types': ['q4_k']},
        'measurement': {
            'state': 'measured_artifact_class',
            'artifact_class': {'repo': BASE_REPO, 'revision': BASE_REV,
                               'quant': 'Q4_K_M', 'argv': '-c 4096 --parallel 4',
                               'harmful': [40, 40], 'held_out': [120, 120],
                               'retention': [241, 241], 'drift': 0.0058},
            'removal': {'harmful': [40, 40], 'held_out': [120, 120]},
            'retention': {'retained': [241, 241]},
            'config': '-c 4096 --parallel 4',
        },
        'bundle': {'file': BUNDLE, 'sha256': bundle_sha256},
    }


def haystack():
    return {'schema': 'jdewitt-space-bank-licences-v1',
            'scope': ('Fixture allow-list: one licence, so the licence gate runs its '
                      'real allow-list path with no network.'),
            'allow': {'apache-2.0': {'name': 'Apache License 2.0', 'spdx': 'Apache-2.0',
                                     'used_by': [ENTRY_KEY]}}}


def hub_document(headers):
    repos = {}
    for repo, revision in ((BASE_REPO, BASE_REV), (FT_REPO, FT_REV), (OTHER_REPO, 'other-rev')):
        files = []
        for owner, name, architecture, seed, codec in FILES:
            if owner != repo:
                continue
            path = DEST / 'repos' / f'{owner.replace("/", "__")}' / name
            files.append({'name': name, 'size': path.stat().st_size,
                          'header_bytes': headers[str(path)]})
        repos[repo] = {'revision': revision, 'sha': revision, 'licence': 'apache-2.0',
                       'gated': False, 'files': files}
    return {
        'schema': 'jdewitt-space-fixture-hub-v1',
        'note': ('A fake Hub for the loop dry run: no socket is opened. header_bytes is '
                 'the prefix the coverage step may read; download copies the whole file.'),
        'dry_run': {'measured_repo': BASE_REPO, 'unmeasured_repo': FT_REPO,
                    'uncovered_repo': OTHER_REPO},
        'repos': repos,
    }


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / 'bank').mkdir(exist_ok=True)
    headers = {}
    for repo, name, architecture, seed, codec in FILES:
        directory = DEST / 'repos' / repo.replace('/', '__')
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        write_gguf(path, architecture, seed, codec)
        headers[str(path)] = header_bytes(path, LAYERS * 3)

    with tempfile.TemporaryDirectory(prefix='jdewitt-loop-fixture-') as registry:
        write_plan(registry)
        bundle = DEST / 'bank' / BUNDLE
        if bundle.exists():
            bundle.unlink()
        written = bundle_plan(registry, PLAN_KEY, bundle)

    bank = DEST / 'bank'
    (bank / 'bank.json').write_text(
        json.dumps({'schema': 'jdewitt-space-bank-v1', 'entries': [entry(written['bundle_sha256'])]},
                   indent=2) + '\n')
    allow = bank / 'licences.json'
    allow.write_text(json.dumps(haystack(), indent=2) + '\n')
    (bank / 'SHA256SUMS').write_text(f'{_sha(allow)}  licences.json\n')
    (DEST / 'hub.json').write_text(json.dumps(hub_document(headers), indent=2) + '\n')
    print(json.dumps({'fixture': str(DEST), 'bundle_sha256': written['bundle_sha256'],
                      'header_bytes': headers}, indent=2))


if __name__ == '__main__':
    main()
