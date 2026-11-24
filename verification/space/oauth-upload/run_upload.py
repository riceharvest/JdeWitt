#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Upload one local artifact through the Space's OAuth delivery and re-read it.

    python verification/space/oauth-upload/run_upload.py \
        --artifact FILE --base-repo OWNER/NAME --quant Q8_0 --label oauth-2.8gb \
        --card CARD.md --sidecar-config JSON --out verification/space/oauth-upload

This is the live-account half of card 13. It calls `jdewitt.space_deliver.deliver`
-- the module the staged Space app calls -- with a real OAuth token from the
`hf_oauth` flow (`~/.cache/huggingface/token` or `$HF_TOKEN`), then fetches the
artifact back from the created repo over the public URL, hashes it, and runs
`gguf-inspect` on the fetched bytes. The run JSON is written under
`verification/space/oauth-upload/<label>.json`; `make_report.py` merges the runs
into `verification/space/oauth-upload.json`.

It is not a Space run, and the documents say so: the container is this host, the
token is the one the live Space issued, and what is measured is the delivery path
end to end (bytes out, wall clock, rate, commit, fetch-back).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from jdewitt import space_deliver  # noqa: E402

SCHEMA = 'jdewitt-space-oauth-upload-run-v1'
CHUNK = 8 << 20


def token_from_environment():
    value = os.environ.get('HF_TOKEN')
    if value:
        return value.strip()
    path = Path(os.path.expanduser('~/.cache/huggingface/token'))
    return path.read_text().strip() if path.is_file() else None


def sha256_of(path):
    hasher = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(CHUNK), b''):
            hasher.update(block)
    return hasher.hexdigest()


def gguf_inspect(path):
    """The project's own inventory, on the fetched bytes."""
    process = subprocess.run([sys.executable, '-m', 'jdewitt', 'gguf-inspect', str(path)],
                             cwd=ROOT, capture_output=True, text=True, timeout=1800)
    try:
        document = json.loads(process.stdout)
    except ValueError:
        return {'ok': False, 'exit_code': process.returncode,
                'stdout_head': process.stdout[:400], 'stderr_head': process.stderr[:400]}
    metadata = document.get('metadata') or {}
    return {
        'ok': process.returncode == 0 and 'error' not in document,
        'exit_code': process.returncode,
        'file': document.get('file'),
        'size_bytes': document.get('size_bytes'),
        'tensor_count': document.get('tensor_count'),
        'data_bytes': document.get('data_bytes'),
        'architecture': metadata.get('general.architecture'),
        'name': metadata.get('general.name'),
    }


def fetch_back(repo_id, filename, dest):
    """Anonymous GET from the public repo: the visitor's own path, timed."""
    url = f'https://huggingface.co/{repo_id}/resolve/main/{filename}'
    started = time.monotonic()
    hasher = hashlib.sha256()
    total = 0
    samples = []
    request = urllib.request.Request(url, headers={'User-Agent': 'jdewitt-oauth-verify'})
    with urllib.request.urlopen(request, timeout=300) as response, open(dest, 'wb') as handle:
        declared = int(response.headers.get('Content-Length') or 0)
        while True:
            block = response.read(CHUNK)
            if not block:
                break
            handle.write(block)
            hasher.update(block)
            total += len(block)
            samples.append((round(time.monotonic() - started, 2), total))
    wall = time.monotonic() - started
    return {'url': url, 'declared_bytes': declared, 'bytes': total,
            'sha256': hasher.hexdigest(), 'wall_seconds': round(wall, 3),
            'bytes_per_second': round(total / max(wall, 1e-6), 1),
            'samples': samples[::max(1, len(samples) // 40)]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--base-repo', required=True,
                        help='the repo id the repo name is derived from')
    parser.add_argument('--quant', default=None)
    parser.add_argument('--label', required=True, help='output file stem under --out')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--card', type=Path, default=None,
                        help='card markdown to upload as card.md; generated if omitted')
    parser.add_argument('--card-title', default=None)
    parser.add_argument('--card-body', default=None)
    parser.add_argument('--keep-fetch', action='store_true',
                        help='keep the fetched copy (default: delete it after inspect)')
    parser.add_argument('--repo-name-suffix', default='',
                        help='suffix appended to the quant in the repo name (uniqueness)')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    token = token_from_environment()
    if not token:
        print(json.dumps({'error': 'no token in HF_TOKEN or ~/.cache/huggingface/token'}))
        return 1
    artifact = args.artifact.resolve()
    size = artifact.stat().st_size
    local_sha = sha256_of(artifact)

    card = args.card
    if card is None:
        card = args.out / f'{args.label}-card.md'
        card.write_text((args.card_body or '') + '\n' if args.card_body else '# delivery\n')
    sidecar = args.out / f'{args.label}-sidecar.json'
    sidecar.write_text(json.dumps({
        'schema': 'jdewitt-space-sidecar-v1',
        'kind': 'oauth-upload-verification',
        'artifact': {'file': artifact.name, 'bytes': size, 'sha256': local_sha},
        'delivery': {'repo_name': space_deliver.repo_name(args.base_repo,
                                                          (args.quant or '') + args.repo_name_suffix),
                     'flag': space_deliver.FLAG_ENV,
                     'held': 'the Space held a temporary copy; the repo is the delivery'},
        'claims': {'upload': 'bytes and seconds measured on this host by the delivery module',
                   'landed': 're-fetched from the public repo and gguf-inspect clean'},
    }, indent=2, sort_keys=True) + '\n')

    files = {artifact.name: artifact, 'card.md': card, 'sidecar.json': sidecar}
    progress_path = args.out / f'{args.label}-progress.jsonl'
    progress_stream = open(progress_path, 'w')

    def on_progress(snapshot):
        progress_stream.write(json.dumps(snapshot, sort_keys=True) + '\n')
        progress_stream.flush()
        print(f"[{snapshot['file']}] {snapshot['bytes_done']}/{snapshot['bytes_total']} B "
              f"@ {snapshot['rate_bytes_per_second'] / 1e6:.1f} MB/s "
              f"eta {snapshot['eta_seconds']}s", flush=True)

    quant = (args.quant or '') + args.repo_name_suffix
    started_at = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    started = time.monotonic()
    delivery = space_deliver.deliver(files, token=token, base_repo=args.base_repo,
                                     quant=quant or None, on_progress=on_progress,
                                     root=args.out / 'work')
    upload_wall = time.monotonic() - started
    progress_stream.close()

    fetched_path = args.out / f'{args.label}-fetched.gguf'
    print('fetching back from the public repo ...', flush=True)
    fetched = fetch_back(delivery['repo_id'], artifact.name, fetched_path)
    inspect = gguf_inspect(fetched_path)
    fetched['matches_local_sha256'] = fetched['sha256'] == local_sha
    fetched['matches_local_bytes'] = fetched['bytes'] == size
    inspect['matches_local_sha256'] = fetched['matches_local_sha256']
    if not args.keep_fetch:
        fetched_path.unlink(missing_ok=True)

    samples = sum(1 for _ in open(progress_path))
    document = {
        'schema': SCHEMA,
        'label': args.label,
        'started_at': started_at,
        'is_a_space_run': False,
        'where': 'this host; the token is the one the live Space issued through hf_oauth '
                 '(auth type oauth, write-repos)',
        'account': {'name': delivery['account']['name'],
                    'auth': delivery['account']['auth'],
                    'expires_at': delivery['account']['expires_at']},
        'flag': {'name': space_deliver.FLAG_ENV, 'value': True},
        'local_artifact': {'path': str(artifact), 'file': artifact.name,
                           'bytes': size, 'sha256': local_sha},
        'delivery': delivery,
        'upload_wall_seconds_including_create_commit': round(upload_wall, 3),
        'progress_samples': samples,
        'progress_file': str(progress_path),
        'fetched_back': fetched,
        'gguf_inspect': inspect,
        'landed_whole': bool(fetched['matches_local_sha256']
                             and fetched['matches_local_bytes']
                             and inspect.get('ok')),
    }
    target = args.out / f'{args.label}.json'
    target.write_text(json.dumps(document, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: document[key] for key in
                      ('label', 'upload_wall_seconds_including_create_commit',
                       'landed_whole')}, indent=2))
    print('repo', delivery['repo_url'], 'commit', delivery['commit_sha'])
    print('wrote', target)
    return 0 if document['landed_whole'] else 2


if __name__ == '__main__':
    sys.exit(main())
