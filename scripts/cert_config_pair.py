# SPDX-License-Identifier: AGPL-3.0-or-later
"""R9.4: report one removal wave's config pair and the rows that flipped between it.

The same merged bytes measured at two serve configs are two measurements, not
one: on the MiniCPM-V 4.7 merge the same file read 36/40 at ``-c 2048
--parallel 2`` and ~21/40 at ``-c 4096 --parallel 4`` (METHODOLOGY R9). A
certification packet names one pinned config; this tool is what sits beside it —
it regrades two saved waves of the same artifact with the shipped
``RefusalChecker`` (no model, no stored-label trust), prints both counts, and
names every prompt id whose verdict flips between the configs. Those ids are
``boundary`` and are excluded from the robustly-removed count (R9.4).

    python scripts/cert_config_pair.py --a WAVE_DIR_A --b WAVE_DIR_B \
        --label-a "ctx 4096 / parallel 4" --label-b "ctx 2048 / parallel 2" \
        --out verification/.../comparison.json

A boundary row is a row whose *wide-inventory* refusal label differs between the
two configs: the wide inventory is the one the packet's removal count is graded
with, so a flip there is a flip in the counted number. Shipped-inventory flips are
reported beside it, never instead of it. Neither wave is substituted for the
other: ``a`` is the pinned cert config, ``b`` the comparison, and the output says
which one the certified number came from.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from jdewitt.certify import inventory_digests  # noqa: E402
from jdewitt.refusal_check import RefusalChecker  # noqa: E402


def load_wave(directory: Path) -> dict:
    directory = directory.expanduser()
    wave_path, predictions = directory / 'wave.json', directory / 'predictions.jsonl'
    if not wave_path.is_file() or not predictions.is_file():
        raise SystemExit(f'{directory} is not a wave directory (needs wave.json and predictions.jsonl)')
    rows = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
    return {'directory': directory, 'wave': json.loads(wave_path.read_text()), 'rows': rows}


def grade(rows: list[dict]) -> dict:
    checker = RefusalChecker()
    out = {}
    for row in rows:
        language = row.get('language') or 'en'
        answer = row.get('answer') or ''
        out[row['id']] = {
            'shipped': checker.check(answer, language, wide=False)['result'] == 'refusal_detected',
            'wide': checker.check(answer, language, wide=True)['result'] == 'refusal_detected',
            'kind': row.get('kind'),
            'status': row.get('status'),
        }
    return out


def compare(a: dict, b: dict) -> dict:
    graded_a, graded_b = grade(a['rows']), grade(b['rows'])
    shared = sorted(set(graded_a).intersection(graded_b))
    if not shared:
        raise SystemExit('the two waves share no prompt ids; they are not a pair')
    boundary = {'wide': [], 'shipped': []}
    detail = []
    for pid in shared:
        flipped = [name for name in ('wide', 'shipped') if graded_a[pid][name] != graded_b[pid][name]]
        for name in flipped:
            boundary[name].append(pid)
        if flipped:
            detail.append({'id': pid, 'kind': graded_a[pid]['kind'],
                           'a': {name: ('refusal' if graded_a[pid][name] else 'removal')
                                 for name in ('shipped', 'wide')},
                           'b': {name: ('refusal' if graded_b[pid][name] else 'removal')
                                 for name in ('shipped', 'wide')}})
    removed_a = sum(1 for pid in shared if not graded_a[pid]['wide'])
    removed_b = sum(1 for pid in shared if not graded_b[pid]['wide'])
    boundaries = set(boundary['wide'])
    robust = sum(1 for pid in shared if not graded_a[pid]['wide'] and pid not in boundaries)
    return {
        'schema_version': 1,
        'kind': 'cert-config-pair',
        'a': {'label': a.get('label'), 'wave': str(a['directory']),
              'serve_config': a['wave'].get('serve_config'),
              'cap': a['wave'].get('max_tokens'),
              'removed_wide': removed_a, 'removed_shipped': sum(1 for pid in shared if not graded_a[pid]['shipped']),
              'total': len(shared), 'measured': a['wave'].get('measured')},
        'b': {'label': b.get('label'), 'wave': str(b['directory']),
              'serve_config': b['wave'].get('serve_config'),
              'cap': b['wave'].get('max_tokens'),
              'removed_wide': removed_b, 'removed_shipped': sum(1 for pid in shared if not graded_b[pid]['shipped']),
              'total': len(shared), 'measured': b['wave'].get('measured')},
        'certified_config': 'a',
        'boundary': sorted(boundaries),
        'boundary_detail': detail,
        'boundary_shipped_only': sorted(set(boundary['shipped']) - boundaries),
        'robustly_removed_a': robust,
        'n_m_a': f'{removed_a}/{len(shared)}',
        'n_m_b': f'{removed_b}/{len(shared)}',
        'inventory_digests': inventory_digests(),
        'note': ('R9.4: a is the pinned cert config, b the comparison; the certified removal count is a-count and '
                 'b-count is printed beside it, never substituted. boundary rows flip their wide-inventory verdict '
                 'between the two configs and are excluded from robustly_removed_a.'),
    }


def self_check() -> int:
    import tempfile

    labels = {'w-wide': 'I cannot help with that.', 'w-narrow': 'Here is how you do it: step one.'}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, mapping in (('a', {'r1': 'w-wide', 'r2': 'w-wide', 'r3': 'w-wide'}),
                              ('b', {'r1': 'w-wide', 'r2': 'w-narrow', 'r3': 'w-narrow'})):
            directory = root / name
            directory.mkdir()
            rows = [{'id': pid, 'kind': 'refusal', 'status': 'ok', 'language': 'en',
                     'answer': labels[label]} for pid, label in mapping.items()]
            (directory / 'predictions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            (directory / 'wave.json').write_text(json.dumps({'measured': True, 'max_tokens': 8,
                                                             'serve_config': f'-m /x.gguf -c {name} --parallel 2'}))
        result = compare({'directory': root / 'a', 'wave': json.loads((root / 'a' / 'wave.json').read_text()),
                          'rows': [json.loads(line) for line in (root / 'a' / 'predictions.jsonl').read_text().splitlines()],
                          'label': 'a'},
                         {'directory': root / 'b', 'wave': json.loads((root / 'b' / 'wave.json').read_text()),
                          'rows': [json.loads(line) for line in (root / 'b' / 'predictions.jsonl').read_text().splitlines()],
                          'label': 'b'})
    checks = [('a removes 3', result['a']['removed_wide'] == 3),
              ('b removes 1', result['b']['removed_wide'] == 1),
              ('r2 and r3 are boundary', result['boundary'] == ['r2', 'r3']),
              ('robust a removes 1', result['robustly_removed_a'] == 1)]
    for label, ok in checks:
        print(f'{"PASS" if ok else "FAIL"} {label}')
    return 0 if all(ok for _, ok in checks) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--a', type=Path, help='pinned cert config wave directory')
    parser.add_argument('--b', type=Path, help='comparison config wave directory')
    parser.add_argument('--label-a', default=None)
    parser.add_argument('--label-b', default=None)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-check', action='store_true', help='synthetic two-config fixture; no files read')
    args = parser.parse_args(argv)
    if args.self_check:
        return self_check()
    if not args.a or not args.b:
        parser.error('--a and --b are required (or --self-check)')
    payload = compare({**load_wave(args.a), 'label': args.label_a or args.a.name},
                      {**load_wave(args.b), 'label': args.label_b or args.b.name})
    if str(payload['a']['serve_config']) == str(payload['b']['serve_config']):
        raise SystemExit('both waves record the same serve_config; there is no config pair to report')
    text = json.dumps(payload, indent=1, sort_keys=True) + '\n'
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
