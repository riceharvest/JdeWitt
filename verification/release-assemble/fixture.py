# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build the small, explicit fixture used to exercise `jdewitt release-assemble`.

NOT model weights and NOT a plan: every file is a few hundred bytes of readable
placeholder data whose only job is to carry the shape a phase-6 release has to
bind (a GGUF adapter bound to a GGUF base, a merged twin whose digests the adapter
card names, a digest-pinned plan bundle, an evaluation record with a pinned serve
argv, and a README). Run it into a scratch directory; nothing here is a claim.

    uv run python verification/release-assemble/fixture.py /tmp/release-assemble-fixture
"""
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jdewitt.data import digest  # noqa: E402
from jdewitt.registry import BUNDLE_FORMAT  # noqa: E402

ARCH = 'twinfixture'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gguf(path, tensors, metadata):
    from gguf import GGUFWriter
    writer = GGUFWriter(str(path), ARCH)
    for key, value in metadata.items():
        writer.add_string(key, value)
    for name, array in tensors.items():
        writer.add_tensor(name, np.asarray(array, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    base_tensors = {'blk.0.attn_output.weight': rng.normal(size=(4, 4)),
                    'blk.0.ffn_down.weight': rng.normal(size=(4, 4))}
    gguf(root / 'base.gguf', base_tensors, {'general.name': 'fixture base (NOT weights)'})
    twin_tensors = {'blk.0.attn_output.weight': base_tensors['blk.0.attn_output.weight'] + 0.1,
                    'blk.0.ffn_down.weight': base_tensors['blk.0.ffn_down.weight']}
    gguf(root / 'twin.gguf', twin_tensors, {'general.name': 'fixture merged twin (NOT weights)'})
    gguf(root / 'adapter.gguf',
         {'blk.0.attn_output.weight.lora_a': np.full((4, 1), 0.5),
          'blk.0.attn_output.weight.lora_b': np.full((1, 4), 0.25)},
         {'general.type': 'adapter', 'general.name': 'fixture LoRA (NOT weights)',
          'jdewitt.base_artifact_sha256': sha256(root / 'base.gguf')})

    record = {
        'version': 1,
        'artifact_sha256': sha256(root / 'twin.gguf'),
        'base_artifact_sha256': sha256(root / 'base.gguf'),
        'arch': ARCH,
        'edit_fingerprint': {'family': 'perlayer|0', 'shape': 'perlayer', 'layers': [0],
                             'scale': 1.0, 'rank': 1, 'edited_tensors': 1},
        'edited_tensors': 1,
        'untouched_tensors': 1,
        'delivered': {'tensors': [{'name': 'blk.0.attn_output.weight', 'cos': 0.4407,
                                   'source_type': 'F32', 'edited_type': 'F32',
                                   'norm_ratio': 0.8263, 'snr': 1.023}]},
        'behaviour': {'adapter_curve': {'delta_br': 0.215, 'measured_on': 'adapter'},
                      'merged_bytes': {'delta_br': 0.119, 'measured_on': 'merged-bytes'}},
        'curve_is_not_merged_bytes': True,
        'note': 'fixture fidelity record; the delta_br pair is copied from docs/ROADMAP.md item 3',
    }
    (root / 'twin-record.json').write_text(json.dumps(record, indent=2) + '\n')

    payload = b'fixture plan payload, NOT a fitted plan\n'
    payload_sha = hashlib.sha256(payload).hexdigest()
    meta = {'key': 'fixture-plan', 'method': 'fixture', 'payload_sha256': payload_sha,
            'measured': False}
    meta['meta_digest'] = digest({key: value for key, value in meta.items()
                                  if key != 'meta_digest'})
    manifest = {'format': BUNDLE_FORMAT, 'key': 'fixture-plan', 'meta': meta,
                'payload_sha256': payload_sha}
    with tarfile.open(root / 'plan.tar', 'w') as tar:
        for name, data in ((f'{BUNDLE_FORMAT}/manifest.json',
                            json.dumps(manifest, sort_keys=True).encode()),
                           (f'{BUNDLE_FORMAT}/plan.pt', payload)):
            info = tarfile.TarInfo(name)
            info.type = tarfile.REGTYPE
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))

    (root / 'evaluation.json').write_text(json.dumps({
        'purpose': 'release-assemble fixture; NOT a model evaluation',
        'serve_argv': ['llama-server', '-m', 'model-merged.gguf', '-ngl', '0', '-c', '512'],
        'matches_adapter_screen': False,
    }, indent=2) + '\n')
    (root / 'README.md').write_text(
        '# Fixture release (NOT a model)\n\n'
        'The adapter curve is not the merged bytes (R8). merged_delta_br 0.119 on the twin; '
        'adapter_delta_br 0.215 is a screening number. Numbers copied from docs/ROADMAP.md.\n')
    print(root)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'release-assemble-fixture')
