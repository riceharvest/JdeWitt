#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Re-grade refusal screens under the wider declination inventory, side by side.

The shipped checker matches first-person refusals only ("I cannot help with that"),
so an answer opening "Absolutely not." counts as refusal removed. Pass one or more
screen directories (a directory holding predictions.jsonl) and this prints, per
screen, the removal count the shipped inventory gives and the one the wider
inventory gives. Two directories plus --compare also get an exact McNemar test on
the wider grading, which is the honest version of a wording comparison.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jdewitt.refusal_check import RefusalChecker            # noqa: E402
from scripts.paired_screen_test import exact_mcnemar        # noqa: E402

CHECKER = RefusalChecker()


def language_of(screen_dir: Path) -> str:
    wave = screen_dir / 'wave.json'
    if not wave.exists():
        wave = screen_dir.parent / (screen_dir.name + '.wave.json')
    suite = json.loads(wave.read_text()).get('suite') if wave.exists() else None
    if suite and Path(suite).exists():
        for line in open(suite):
            if line.strip():
                return json.loads(line).get('lang', 'en')
    return 'en'


def grades(screen_dir: Path) -> dict[str, tuple[bool, bool]]:
    """id -> (refused under shipped inventory, refused under wider inventory)."""
    lang = language_of(screen_dir)
    path = screen_dir / 'predictions.jsonl'
    if not path.exists():
        raise SystemExit(f'{path} does not exist')
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('status') != 'ok':
            continue
        v1 = (row.get('refusal_check') or {}).get('result') == 'refusal_detected'
        v2 = CHECKER.check(row.get('answer') or '', lang, wide=True)['result'] == 'refusal_detected'
        out[row['id']] = (v1, v2)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('screens', nargs='+', type=Path)
    ap.add_argument('--compare', action='store_true',
                    help='with exactly two screens, run exact McNemar on the wider grading')
    args = ap.parse_args(argv)

    table = {}
    for screen in args.screens:
        g = grades(screen)
        rows = len(g)
        v1 = sum(1 for a, _ in g.values() if not a)
        v2 = sum(1 for _, b in g.values() if not b)
        missed = sum(1 for a, b in g.values() if not a and b)
        table[screen] = g
        print(f'{screen}: rows {rows} — removal shipped-checker {v1}/{rows} '
              f'| removal wider-inventory {v2}/{rows} | shipped checker missed {missed} '
              f'declinations it passed')
    if args.compare:
        if len(args.screens) != 2:
            raise SystemExit('--compare takes exactly two screens')
        a, b = (table[k] for k in args.screens)
        res = exact_mcnemar({i: r[1] for i, r in a.items()},
                            {i: r[1] for i, r in b.items()})
        print(json.dumps(res, indent=1, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
