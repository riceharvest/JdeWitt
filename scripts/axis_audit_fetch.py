# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fetch the 0.5B triple used by the static axis audit, pinned by revision.

Usage: python scripts/axis_audit_fetch.py --out runs/axis-audit/models.json

Downloads only what the audit needs (safetensors + tokenizer/config JSON), records
the resolved repo revision, file bytes and sha256 for every file, and leaves the
snapshot in the Hugging Face cache. No model is executed: this script only moves
bytes and hashes them. Delete the cache entries after the audit to reclaim disk.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

MODELS = {
    'base': ('Qwen/Qwen2.5-0.5B', '060db6499f32faf8b98477b0a26969ef7d8b9987'),
    'instruct': ('Qwen/Qwen2.5-0.5B-Instruct', '7ae557604adf67be50417f59c2c2f167def9a775'),
    'twin': ('Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1',
             'a42fe88f75b3ab545466ca5692fef0b4d7bc8009'),
}
ALLOW = ['*.json', '*.safetensors', '*.model', '*.txt']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='runs/axis-audit/models.json')
    args = ap.parse_args()
    api = HfApi()
    manifest = {'schema': 'jdewitt.axis-audit-models/1', 'allow_patterns': ALLOW,
                'fetched_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'models': {}}
    for key, (repo, revision) in MODELS.items():
        t0 = time.time()
        info = api.model_info(repo, revision=revision)
        if info.sha != revision:
            raise SystemExit(f'{repo}: requested {revision}, resolved {info.sha}')
        path = Path(snapshot_download(repo, revision=revision, allow_patterns=ALLOW))
        files = []
        for f in sorted(path.rglob('*')):
            if f.is_file():
                files.append({'path': str(f.relative_to(path)), 'bytes': f.stat().st_size,
                              'sha256': sha256(f)})
        manifest['models'][key] = {'repo': repo, 'revision': revision, 'gated': bool(info.gated),
                                   'snapshot': str(path), 'files': files}
        big = max(files, key=lambda x: x['bytes'])
        print(f'{key}: {repo}@{revision} {len(files)} files, weights '
              f'{big["path"]} {big["bytes"]} bytes sha256 {big["sha256"]} '
              f'({time.time() - t0:.0f}s)', flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1) + '\n')
    print(f'wrote {out}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
