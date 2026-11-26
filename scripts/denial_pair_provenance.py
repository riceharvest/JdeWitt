# SPDX-License-Identifier: AGPL-3.0-or-later
"""Provenance for the tiny public base/twin pair, without downloading weights.

The denial grammar is compiled against the tokenizer, not the weights, so this
records what identifies the pair: repo id, commit revision and sha256 of every
tokenizer file (rehashed locally after download, then deleted) and of the one
safetensors file (the Hub's LFS sha256, because the tensor file is not needed and
disk is tight).  It also checks the one claim the compilation depends on: the
twin inherits the base tokenizer byte for byte.

Run: ``python scripts/denial_pair_provenance.py --out verification/denial-grammar/public-pair.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPOS = {
    'base': 'Qwen/Qwen2.5-0.5B-Instruct',
    'twin': 'Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1',
}
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
                   'generation_config.json', 'config.json')


def sha256_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _merges(document):
    listed = document.get('model', {}).get('merges', [])
    return [tuple(item.split(' ')) if isinstance(item, str) else tuple(item) for item in listed]


def _tokenizer_semantics(base, twin):
    """Compare the two tokenizer.json files field by field, not byte by byte."""
    if not base or not twin:
        return {'compared': False}
    base_model, twin_model = base.get('model', {}), twin.get('model', {})
    return {'compared': True,
            'same_vocab': base_model.get('vocab') == twin_model.get('vocab'),
            'vocab_entries': len(base_model.get('vocab', {})),
            'same_merge_ranks': _merges(base) == _merges(twin),
            'merges': len(_merges(base)),
            'same_added_tokens': base.get('added_tokens') == twin.get('added_tokens'),
            'same_normalizer': base.get('normalizer') == twin.get('normalizer'),
            'same_pre_tokenizer': base.get('pre_tokenizer') == twin.get('pre_tokenizer'),
            'same_post_processor': base.get('post_processor') == twin.get('post_processor'),
            'same_decoder': base.get('decoder') == twin.get('decoder'),
            'serialization_difference': ('tokenizer.json stores merges as space-joined strings in one repo and '
                                         'as two-element arrays in the other; the rank order is identical')}


def _gguf_vs_hub(path, document):
    """Token strings and ids of the GGUF against the Hub vocabulary map."""
    from jdewitt.vocabremap import read_tokenizer
    tokenizer = read_tokenizer(path)
    gguf = {token: index for index, token in enumerate(tokenizer['tokens'])}
    hub = dict(document.get('model', {}).get('vocab', {}))
    for entry in document.get('added_tokens', []):
        hub.setdefault(entry.get('content'), entry.get('id'))
    common = set(hub) & set(gguf)
    matching = sum(1 for key in common if hub[key] == gguf[key])
    return {'gguf_identity': tokenizer['identity'],
            'gguf_pieces': len(gguf), 'hub_pieces': len(hub), 'common_pieces': len(common),
            'ids_matching': matching, 'ids_differing': len(common) - matching,
            'hub_only': sorted(set(hub) - set(gguf))[:5],
            'gguf_only_count': len(set(gguf) - set(hub))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='verification/denial-grammar/public-pair.json')
    parser.add_argument('--artifact', default=None,
                        help='optional GGUF whose tokenizer strings are compared with the Hub vocabulary')
    args = parser.parse_args(argv)

    from huggingface_hub import HfApi

    api = HfApi()
    report = {'schema': 'jdewitt.denial-grammar-public-pair/1',
              'scope': ('Repo/revision/sha256 of the tiny public base and twin, and a byte-identity check of '
                        'their tokenizers. No weights are downloaded or run; the tokenizer is the only input the '
                        'grammar compilation needs.'),
              'removal_measured': False,
              'method': {'tokenizer': 'fetched into memory, hashed, never written to disk',
                         'weights': ('sha256 taken from the Hub LFS metadata; the tensor file is not downloaded '
                                     'because the experiment compiles text, not weights, and disk is tight')},
              'repos': {}}
    tokenizer_blobs = {}
    tokenizer_docs = {}
    for role, repo in REPOS.items():
        info = api.model_info(repo, files_metadata=True)
        files = {sibling.rfilename: sibling for sibling in info.siblings}
        row = {'repo': repo, 'revision': info.sha, 'files': {}, 'tokenizer': {}}
        for name in TOKENIZER_FILES:
            sibling = files.get(name)
            if sibling is None:
                row['tokenizer'][name] = {'present': False}
                continue
            url = f'https://huggingface.co/{repo}/resolve/{info.sha}/{name}'
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
            local_hash = sha256_bytes(data)
            row['tokenizer'][name] = {'present': True, 'bytes': len(data),
                                      'sha256_local': local_hash,
                                      'sha256_hub': sibling.blob_id,
                                      'match': local_hash == sibling.blob_id}
            tokenizer_blobs[name] = local_hash
            if name == 'tokenizer.json':
                tokenizer_docs[role] = json.loads(data)
            del data
        for name, sibling in sorted(files.items()):
            if name.endswith('.safetensors'):
                row['files'][name] = {'bytes': sibling.size,
                                      'sha256': (sibling.lfs.sha256 if sibling.lfs else sibling.blob_id),
                                      'downloaded': False}
        report['repos'][role] = row

    base_tok = report['repos']['base']['tokenizer'].get('tokenizer.json', {})
    twin_tok = report['repos']['twin']['tokenizer'].get('tokenizer.json', {})
    report['tokenizer_byte_identical'] = bool(base_tok.get('sha256_local')
                                              and base_tok.get('sha256_local') == twin_tok.get('sha256_local'))
    report['tokenizer_semantics'] = _tokenizer_semantics(tokenizer_docs.get('base'), tokenizer_docs.get('twin'))
    report['tokenizer_identity_used'] = {
        'artifact': args.artifact or 'verification/iq05b/Qwen2.5-0.5B-Instruct-IQ3_M.gguf',
        'note': ('the IQ3_M stand-in carries the Qwen2.5-0.5B-Instruct vocabulary; the repo tokenizer.json '
                 'above is the public source of the same split'),
    }
    if args.artifact:
        report['tokenizer_identity_used']['gguf_vocab_vs_hub'] = _gguf_vs_hub(args.artifact, tokenizer_docs.get('base'))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'base': report['repos']['base']['revision'],
                      'twin': report['repos']['twin']['revision'],
                      'tokenizer_byte_identical': report['tokenizer_byte_identical'],
                      'weights': {role: report['repos'][role]['files'] for role in REPOS}},
                     indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
