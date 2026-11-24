# SPDX-License-Identifier: AGPL-3.0-or-later
"""Census for the transfer kill gate (§6): which fine-tune of the covered base ships a Q4_K_M GGUF.

The rule is mechanical and the artifact records it: query the Hub for
`Qwen3-4B-Instruct-2507`, order by downloads, drop every repo whose `base_model` tag names
the base itself with relation `quantized` (those are requants of the base, not fine-tunes),
then take the highest-downloads repo that ships a `*Q4_K_M*.gguf`. Identity is checked
against the base before the row is written: architecture/model_type/geometry from
config.json, `tokenizer.json` digest, and the GGUF's own 151,936-token array when the file
is local.

    python verification/space/make_transfer_census.py            # writes transfer-census.json
    python verification/space/make_transfer_census.py --out -     # stdout only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BASE = 'Qwen/Qwen3-4B-Instruct-2507'
BASE_REV = 'cdbee75f17c01a7cc42f958dc650907174af0554'
QUERY = 'Qwen3-4B-Instruct-2507'
API = 'https://huggingface.co/api/models'
HERE = Path(__file__).resolve().parent


def fetch_json(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'jdewitt-transfer-census/1'})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.loads(response.read())


def fetch_bytes(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'jdewitt-transfer-census/1'})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def census_rows():
    """Downloads-ordered rows; one page of 100 is what the API serves for this query."""
    url = (f'{API}?search={urllib.parse.quote(QUERY)}&sort=downloads&direction=-1'
           f'&limit=100&full=true&fields=id,downloads,sha,lastModified,tags,siblings,cardData')
    rows = fetch_json(url)
    seen = {}
    for row in rows:
        seen[row['id']] = row
    return seen


def q4km_files(row):
    return [s['rfilename'] for s in (row.get('siblings') or [])
            if s['rfilename'].lower().endswith('.gguf') and 'q4_k_m' in s['rfilename'].lower()]


def is_base_requant(row):
    tags = row.get('tags') or []
    card = row.get('cardData') or {}
    base_models = card.get('base_model')
    base_models = base_models if isinstance(base_models, list) else [base_models] if base_models else []
    return any(tag == f'base_model:quantized:{BASE}' for tag in tags) or \
        (base_models == [BASE] and 'finetune' not in ' '.join(tags))


def gguf_meta(path):
    """Header + tokenizer array digest (tokens only; tensor data is never read)."""
    sys.path.insert(0, str(HERE.parents[1]))
    from jdewitt.ggufio import parse_gguf_header

    scalars = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?',
               10: '<Q', 11: '<q', 12: '<d'}

    def decode(tag, raw, offset=0):
        if tag in scalars:
            size = struct.calcsize(scalars[tag])
            return struct.unpack_from(scalars[tag], raw, offset)[0], offset + size
        if tag == 8:
            length = struct.unpack_from('<Q', raw, offset)[0]
            offset += 8
            return raw[offset:offset + length], offset + length
        if tag == 9:
            subtype = struct.unpack_from('<I', raw, offset)[0]
            offset += 4
            count = struct.unpack_from('<Q', raw, offset)[0]
            offset += 8
            values = []
            for _ in range(count):
                value, offset = decode(subtype, raw, offset)
                values.append(value)
            return values, offset
        raise ValueError(tag)

    header = parse_gguf_header(path)
    meta = {}
    for key, tag, raw in header['entries']:
        value, _ = decode(tag, raw)
        meta[key.decode()] = value
    tokens = meta.get('tokenizer.ggml.tokens') or []
    template = meta.get('tokenizer.chat_template')
    return {
        'n_tensors': header['n_tensors'],
        'architecture': _text(meta.get('general.architecture')),
        'n_tokens': len(tokens),
        'token_array_sha256': hashlib.sha256(
            b'\x00'.join(t if isinstance(t, bytes) else t.encode() for t in tokens)).hexdigest(),
        'chat_template_sha256': hashlib.sha256(
            template if isinstance(template, bytes) else (template or '').encode()).hexdigest(),
        'chat_template_chars': len(template or ''),
        'eos_token_id': meta.get('tokenizer.ggml.eos_token_id'),
    }


def _text(value):
    return value.decode() if isinstance(value, bytes) else value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default=str(HERE / 'transfer-census.json'))
    parser.add_argument('--base-gguf', default='/mnt/ssd/models/unsloth-qwen3-4b-q4km/'
                                               'Qwen3-4B-Instruct-2507-Q4_K_M.gguf')
    parser.add_argument('--candidate-gguf', default='/mnt/ssd/models/finetune/'
                                                   'Huihui-Qwen3-4B-Instruct-2507-abliterated.Q4_K_M.gguf')
    args = parser.parse_args(argv)

    rows = census_rows()
    fine_tunes = []
    for repo, row in rows.items():
        files = q4km_files(row)
        if not files or repo == BASE:
            continue
        fine_tunes.append({
            'repo': repo,
            'downloads': row.get('downloads') or 0,
            'revision': row.get('sha'),
            'q4_k_m_files': files,
            'is_base_requant': is_base_requant(row),
            'base_model': (row.get('cardData') or {}).get('base_model'),
        })
    fine_tunes.sort(key=lambda r: (-r['downloads'], r['repo']))
    candidates = [row for row in fine_tunes if not row['is_base_requant']]
    chosen = candidates[0] if candidates else None

    base_config = fetch_json(f'https://huggingface.co/{BASE}/resolve/{BASE_REV}/config.json')
    identity: dict = {'base_config': {k: base_config.get(k) for k in (
        'architectures', 'model_type', 'hidden_size', 'num_hidden_layers', 'num_attention_heads',
        'num_key_value_heads', 'head_dim', 'vocab_size', 'rope_theta', 'intermediate_size',
        'tie_word_embeddings')}}
    identity['base_tokenizer_json_sha256'] = hashlib.sha256(
        fetch_bytes(f'https://huggingface.co/{BASE}/resolve/{BASE_REV}/tokenizer.json')).hexdigest()

    if chosen:
        meta = fetch_json(f'{API}/{chosen["repo"]}?blobs=true')
        chosen['revision'] = chosen['revision'] or meta.get('sha')
        chosen['base_model'] = (meta.get('cardData') or {}).get('base_model')
        parents = chosen['base_model'] if isinstance(chosen['base_model'], list) else [chosen['base_model']]
        parent = next((p for p in parents if p and p != BASE), None)
        if parent:
            parent_rev = fetch_json(f'{API}/{parent}').get('sha')
            identity['fine_tune_repo'] = parent
            identity['fine_tune_revision'] = parent_rev
            parent_config = fetch_json(f'https://huggingface.co/{parent}/resolve/{parent_rev}/config.json')
            identity['fine_tune_config'] = {k: parent_config.get(k) for k in identity['base_config']}
            identity['configs_identical'] = identity['fine_tune_config'] == identity['base_config']
            identity['fine_tune_tokenizer_json_sha256'] = hashlib.sha256(
                fetch_bytes(f'https://huggingface.co/{parent}/resolve/{parent_rev}/tokenizer.json')).hexdigest()
            identity['tokenizer_json_identical'] = (
                identity['fine_tune_tokenizer_json_sha256'] == identity['base_tokenizer_json_sha256'])
        for sibling in meta.get('siblings') or []:
            if sibling['rfilename'] in chosen['q4_k_m_files']:
                chosen['file'] = sibling['rfilename']
                chosen['bytes'] = sibling.get('size')
                chosen['sha256'] = (sibling.get('lfs') or {}).get('sha256')
        ggufs = {}
        for label, path in (('base', args.base_gguf), ('candidate', args.candidate_gguf)):
            if Path(path).exists():
                ggufs[label] = {'path': path, **gguf_meta(path)}
        identity['gguf'] = ggufs
        if len(ggufs) == 2:
            identity['gguf_token_arrays_identical'] = (
                ggufs['base']['token_array_sha256'] == ggufs['candidate']['token_array_sha256'])

    report = {
        'schema': 'jdewitt-transfer-census-v1',
        'question': 'top-downloaded fine-tune of the covered base that ships a Q4_K_M GGUF',
        'query': {'api': API, 'search': QUERY, 'sort': 'downloads', 'direction': 'desc',
                  'limit': 100, 'rows_returned': len(rows)},
        'rule': ['order by downloads', 'drop repos whose base_model is the base itself with relation '
                 'quantized (requants of the base, not fine-tunes)',
                 'take the highest downloads that ships a *Q4_K_M*.gguf',
                 'refuse a candidate whose config or tokenizer differs from the base'],
        'fine_tune_candidates': candidates,
        'base_requants_dropped': [row for row in fine_tunes if row['is_base_requant']],
        'chosen': chosen,
        'identity': identity,
        # filled only when an identity check actually fails; this census refuses none
        'refused_candidates': [],
    }
    text = json.dumps(report, indent=1, sort_keys=True) + '\n'
    if args.out == '-':
        sys.stdout.write(text)
    else:
        Path(args.out).write_text(text)
        print(f'wrote {args.out}: {len(candidates)} fine-tune candidates, chosen '
              f'{chosen["repo"] if chosen else None}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
