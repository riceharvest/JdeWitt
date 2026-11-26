#!/usr/bin/env python
"""Fetch the four artifacts this experiment needs, hash them, then delete the bytes.

Everything is public and ungated. The revision is pinned to the commit the sha256
came from, and the local file is rehashed rather than trusting the Hub metadata.
``--keep`` keeps the downloads for the experiment run; without it the bytes are
deleted after the hashes are written (disk is tight).
"""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

SPECS = [
    ('base', 'Qwen/Qwen2.5-0.5B', '060db6499f32faf8b98477b0a26969ef7d8b9987', 'model.safetensors'),
    ('instruct', 'Qwen/Qwen2.5-0.5B-Instruct', '7ae557604adf67be50417f59c2c2f167def9a775', 'model.safetensors'),
    ('twin', 'Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1',
     'a42fe88f75b3ab545466ca5692fef0b4d7bc8009', 'model.safetensors'),
    ('iq3m', 'bartowski/Qwen2.5-0.5B-Instruct-GGUF',
     '41ba88dbac95fed2528c92514c131d73eb5a174b', 'Qwen2.5-0.5B-Instruct-IQ3_M.gguf'),
]


def sha256(path, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path, help='provenance JSON to write')
    parser.add_argument('--dest', type=Path, default=Path('downloads'),
                        help='download directory (default: ./downloads)')
    parser.add_argument('--keep', action='store_true',
                        help='keep the downloaded bytes; the default deletes them after hashing')
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    records = {}
    for key, repo, revision, filename in SPECS:
        local = hf_hub_download(repo_id=repo, filename=filename, revision=revision,
                                local_dir=args.dest / key)
        digest = sha256(local)
        size = Path(local).stat().st_size
        records[key] = {'repo': repo, 'revision': revision, 'filename': filename,
                        'bytes': size, 'sha256': digest}
        print(key, size, digest, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2))
    if not args.keep:
        for key in records:
            for path in sorted((args.dest / key).rglob('*')):
                if path.is_file():
                    path.unlink()
    print('wrote', args.out)


if __name__ == '__main__':
    main()
