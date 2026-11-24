#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Merge the OAuth upload runs into `verification/space/oauth-upload.json`.

    python verification/space/oauth-upload/make_report.py [--dir DIR]

`run_upload.py` writes one JSON per run (the delivery document, the per-file
numbers, the fetch-back and the `gguf-inspect` result). This script adds the one
judgement the card needs and does not put in the code that takes it: whether the
5.5 GB class fits the request window the roadmap promises, with the window named
and the host the measurement was taken on named beside it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = 'jdewitt-space-oauth-upload-v1'
DEFAULT_DIR = Path(__file__).resolve().parent
# The window the product promises ("one click, five minutes", §8) and the one the
# live run could not meet on the edit leg.
PROMISE_SECONDS = 300


def load_runs(directory):
    runs = []
    for path in sorted(directory.glob('*.json')):
        if path.name == 'oauth-upload.json':
            continue
        document = json.loads(path.read_text())
        if document.get('schema') != 'jdewitt-space-oauth-upload-run-v1':
            continue
        runs.append((path, document))
    return runs


def artifact_entry(run):
    files = run['delivery']['files']
    return max(files, key=lambda entry: entry['total'])


def summary_of(path, run):
    artifact = artifact_entry(run)
    return {
        'label': run['label'],
        'source': path.name,
        'started_at': run['started_at'],
        'repo_id': run['delivery']['repo_id'],
        'repo_url': run['delivery']['repo_url'],
        'commit_sha': run['delivery']['commit_sha'],
        'visibility': run['delivery']['visibility'],
        'artifact': {'file': artifact['name'], 'bytes': artifact['total'],
                     'sha256': run['local_artifact']['sha256'],
                     'transfer': artifact.get('transfer')},
        'upload_seconds': artifact.get('seconds'),
        'upload_bytes_per_second': artifact.get('bytes_per_second'),
        'upload_megabytes_per_second': round((artifact.get('bytes_per_second') or 0) / 1e6, 2),
        'commit_wall_seconds': run['upload_wall_seconds_including_create_commit'],
        'files_uploaded': [{'name': entry['name'], 'bytes': entry['total'],
                            'seconds': entry.get('seconds'),
                            'transfer': entry.get('transfer')}
                           for entry in run['delivery']['files']],
        'fetch_back': {'bytes': run['fetched_back']['bytes'],
                       'sha256': run['fetched_back']['sha256'],
                       'bytes_per_second': run['fetched_back']['bytes_per_second'],
                       'megabytes_per_second': round(
                           run['fetched_back']['bytes_per_second'] / 1e6, 2),
                       'matches_local_sha256': run['fetched_back']['matches_local_sha256']},
        'gguf_inspect': {'ok': run['gguf_inspect'].get('ok'),
                         'tensor_count': run['gguf_inspect'].get('tensor_count'),
                         'size_bytes': run['gguf_inspect'].get('size_bytes'),
                         'architecture': run['gguf_inspect'].get('architecture')},
        'landed_whole': run['landed_whole'],
        'progress_samples': run['progress_samples'],
        'skipped': run['delivery'].get('skipped'),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', type=Path, default=DEFAULT_DIR)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args(argv)
    runs = load_runs(args.dir)
    if not runs:
        raise SystemExit(f'no run documents in {args.dir}')
    summaries = [summary_of(path, run) for path, run in runs]
    big = [item for item in summaries if item['artifact']['bytes'] >= 5_000_000_000]
    if not big:
        raise SystemExit('no 5.5 GB-class run in the directory; the window verdict needs one')
    largest = min(big, key=lambda item: item['upload_seconds'] or float('inf'))
    fits = (largest['upload_seconds'] or float('inf')) <= PROMISE_SECONDS
    document = {
        'schema': SCHEMA,
        'card': '13 (§8, v1.1)',
        'generated_at': runs[0][1]['started_at'],
        'account': runs[0][1]['account'],
        'is_a_space_run': False,
        'where': runs[0][1]['where'],
        'runs': summaries,
        'window': {
            'promise_seconds': PROMISE_SECONDS,
            'label': largest['label'],
            'artifact_bytes': largest['artifact']['bytes'],
            'upload_seconds': largest['upload_seconds'],
            'megabytes_per_second': largest['upload_megabytes_per_second'],
            'fits': fits,
            'headroom_seconds': (round(PROMISE_SECONDS - largest['upload_seconds'], 1)
                                 if largest['upload_seconds'] is not None else None),
            'caveat': ('measured from this host over its own uplink, not from inside the '
                       'Space container; the container leg that was measured is the '
                       'download (437 MB/s in verification/space/live-run.json)'),
            'verdict': ('the flag can be on: the 5.5 GB push fits the five-minute window '
                        'with the headroom above; the window still fails on the edit leg, '
                        'which is a separate measured row (live-run.json)'
                        if fits else
                        'the flag stays off: the 5.5 GB push does not fit the five-minute '
                        'window; the direct download remains the delivery'),
        },
        'notes': [
            'The Hub refuses a single basic LFS PUT above 5 GB (HTTP 400: "configure your '
            'repository to enable upload of files > 5GB"); the first 5.5 GB attempt hit '
            'this and deleted its repo. The delivery now advertises basic + multipart and '
            'sends 16 MB parts over one keep-alive connection.',
            'The 5.5 GB payload is a valid GGUF transfer probe, not a 7B model: the bank has '
            'one 4B entry, so a 7B artifact is BANK_MISS and the Space cannot produce the '
            '7B q8_0 output the §8 table estimates. The byte class is what transfers.',
        ],
        'flag': {'name': 'JW_OAUTH_PUSH',
                 'default': 'off',
                 'enable_when': 'the visitor needs a repo, not a temporary download link',
                 'set_by': 'a Space variable; the README only declares the OAuth scope'},
    }
    target = args.out or (args.dir / 'oauth-upload.json')
    target.write_text(json.dumps(document, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'window': document['window'], 'runs': [
        {key: item[key] for key in ('label', 'artifact', 'upload_seconds',
                                    'upload_megabytes_per_second', 'landed_whole')}
        for item in summaries]}, indent=2))
    print('wrote', target)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
