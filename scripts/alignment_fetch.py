#!/usr/bin/env python
"""Fetch the four artifacts this experiment needs, hash them, then delete the bytes.

Everything is public and ungated. The revision is pinned to the commit the sha256
came from, and the local file is rehashed rather than trusting the Hub metadata.
"""
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

KEEP = '--keep' in sys.argv
sys.argv = [arg for arg in sys.argv if arg != '--keep']

DEST = Path('/home/dario/alignment-exp/downloads')
OUT = Path(sys.argv[1])

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
    DEST.mkdir(parents=True, exist_ok=True)
    records = {}
    for key, repo, revision, filename in SPECS:
        local = hf_hub_download(repo_id=repo, filename=filename, revision=revision,
                                local_dir=DEST / key)
        digest = sha256(local)
        size = Path(local).stat().st_size
        records[key] = {'repo': repo, 'revision': revision, 'filename': filename,
                        'bytes': size, 'sha256': digest}
        print(key, size, digest, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, indent=2))
    if not KEEP:
        for key in records:
            for path in sorted((DEST / key).rglob('*')):
                if path.is_file():
                    path.unlink()
    print('wrote', OUT)


if __name__ == '__main__':
    main()
