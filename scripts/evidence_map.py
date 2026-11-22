#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Render and enforce the README's evidence map.

A README bullet and a verification file drift apart silently, and the drift is the
whole failure mode of this project: the number stays in prose after the artifact
supporting it is gone. This tool makes that drift an exit code, twice over:

  * every artifact a claim names must exist in the clone, and
  * every number in README section IV must resolve to a map entry whose
    `source`+`locator` recomputes it from the rows that were measured.

A number with no entry is a claim nobody bound; an entry whose literal fragment is
no longer in the README is a stale claim; a recomputed value that differs from the
prose is the drift this exists to catch. Numbers the README itself quotes as void
(no artifact in the clone) are marked `"void": true` and may only live on a claim
whose status says it is not in-clone.

Usage:
    python scripts/evidence_map.py            # print the markdown table
    python scripts/evidence_map.py --check    # exit 1 on a missing artifact or a number mismatch
    python scripts/evidence_map.py --json     # same verdict, machine-readable

Locator grammar (one string, `source` names the measured artifact):

    /json/pointer                 scalar (or `num/den` for two pointers)
    json:/counts/delivered|/counts/rows
    jsonl:passed=true|98/241      count rows whose dotted field equals the value
    wide                          removal/rows under the wider inventory
    shipped                       removal/rows under the shipped inventory
    mcnemar:wide:DIR|discordant   exact McNemar field against another screen
    paired:BASE_DIR|math|before_after   capability_format_audit paired fields
    audit|math|median             one wave's audited field
    void                          quoted as void; no artifact exists
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAP = ROOT / 'verification/evidence-map.json'
README = ROOT / 'README.md'
SUITE = ROOT / 'verification/capability/capability-run-en.jsonl'

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

ALLOWED_STATUS = {'in-clone', 'pre-2026-11-17-scoring-fix', 'no-artifact-in-clone'}

# A number in section IV is one of these shapes. Plain integers in prose
# ('241 authored rows', 'scale 1.0', 'blk.9-35') are configuration, not a
# measured value, and stay out of the map on purpose.
FRACTION = re.compile(r'(?<![\d/])(\d+)/(\d+)(?![\d/])')
ARROW = re.compile(r'(\d+)\s*(?:→|->)\s*(\d+)')
PVALUE = re.compile(r'\bp\s*(<=|<|=)?\s*(\d+(?:\.\d+)?(?:[eE]-?\d+)?)')
DELIVERY = re.compile(r'(\d+)\s+delivered for ours,\s*(\d+)\s+for theirs,\s*out of\s*(\d+)')
DISCORDANT = re.compile(r'\b(zero|0)\s+discordant\b')
OF_PAIR = re.compile(r'(\d+)\s+of\s+(\d+)')
CHARS = re.compile(r'(\d+)\s+characters?\b')
POSSESSIVE = re.compile(r"(?:base|build)'s\s+(\d+)(?![\d/])")
MATHS = re.compile(r'(\d+)\s+maths\b')

WORD_NUMBERS = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
                'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}


def readme_section_iv() -> list[tuple[int, str]]:
    lines = README.read_text(encoding='utf-8').splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith('## IV.')), None)
    if start is None:
        raise SystemExit('README.md has no "## IV." section')
    end = next((i for i, line in enumerate(lines[start + 1:], start + 1)
                if line.startswith('## ')), len(lines))
    return [(i + 1, lines[i]) for i in range(start, end)]


def extract(section: list[tuple[int, str]]) -> list[dict]:
    """Every claim-bearing number in section IV, with the literal text around it."""
    out = []
    for lineno, line in section:
        delivery_spans = []
        for match in DELIVERY.finditer(line):
            ours, theirs, rows = match.groups()
            out.append({'line': lineno, 'text': f'{ours} delivered for ours',
                        'value': f'{ours}/{rows}', 'kind': 'fraction'})
            out.append({'line': lineno, 'text': f'{theirs} for theirs',
                        'value': f'{theirs}/{rows}', 'kind': 'fraction'})
            delivery_spans.append(match.span())
        for match in ARROW.finditer(line):
            out.append({'line': lineno, 'text': match.group(0),
                        'value': f'{match.group(1)}→{match.group(2)}', 'kind': 'arrow'})
        for match in FRACTION.finditer(line):
            out.append({'line': lineno, 'text': match.group(0),
                        'value': match.group(0), 'kind': 'fraction'})
        for match in PVALUE.finditer(line):
            op = match.group(1) or '='
            out.append({'line': lineno, 'text': match.group(0),
                        'value': match.group(2), 'op': op, 'kind': 'p'})
        for match in DISCORDANT.finditer(line):
            word = match.group(1).lower()
            out.append({'line': lineno, 'text': match.group(0),
                        'value': str(WORD_NUMBERS.get(word, 0)), 'kind': 'scalar'})
        for match in OF_PAIR.finditer(line):
            if any(a <= match.start() < b for a, b in delivery_spans):
                continue
            out.append({'line': lineno, 'text': match.group(0),
                        'value': f'{match.group(1)}/{match.group(2)}', 'kind': 'fraction'})
        for match in CHARS.finditer(line):
            out.append({'line': lineno, 'text': match.group(0),
                        'value': match.group(1), 'kind': 'scalar'})
        for match in POSSESSIVE.finditer(line):
            out.append({'line': lineno, 'text': match.group(0),
                        'value': match.group(1), 'kind': 'scalar'})
        for match in MATHS.finditer(line):
            out.append({'line': lineno, 'text': match.group(0),
                        'value': match.group(1), 'kind': 'scalar'})
    # The same phrase is matched by more than one rule only when the rules
    # overlap by construction; keep the longest reading at each start offset.
    unique, seen = [], set()
    for token in sorted(out, key=lambda t: (t['line'], -len(t['text']))):
        key = (token['line'], token['text'], token['value'])
        if key in seen:
            continue
        seen.add(key)
        unique.append(token)
    return unique


def pointer(doc, ptr: str):
    if ptr in ('', '/'):
        return doc
    for part in ptr.lstrip('/').split('/'):
        part = part.replace('~1', '/').replace('~0', '~')
        doc = doc[int(part)] if isinstance(doc, list) else doc[part]
    return doc


def _field(row: dict, dotted: str):
    value = row
    for part in dotted.split('.'):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _coerce(raw: str):
    low = raw.lower()
    if low in ('true', 'false'):
        return low == 'true'
    if low in ('null', 'none'):
        return None
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def recompute(entry: dict) -> tuple[str, str]:
    """Return (artifact_value_as_prose, absolute_source) for one map number."""
    source = entry.get('source')
    locator = entry.get('locator') or ''
    if locator == 'void':
        return 'void', ''
    if not source:
        raise ValueError('no source and not void')
    src = ROOT / source
    if not src.is_file():
        raise FileNotFoundError(source)

    if locator.startswith('json:'):
        doc = json.loads(src.read_text())
        parts = locator[5:].split('|')
        values = [pointer(doc, part) for part in parts]
        if len(values) == 1:
            return str(values[0]), source
        return f'{values[0]}/{values[1]}', source

    if locator.startswith('jsonl:'):
        spec = locator[6:].split('|', 1)[0]
        field, raw = spec.split('=', 1)
        wanted = _coerce(raw)
        rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
        matched = sum(1 for row in rows if _field(row, field) == wanted)
        return f'{matched}/{len(rows)}', source

    if locator in ('wide', 'shipped'):
        from screen_agreement import grades
        graded = grades(src.parent)
        rows = len(graded)
        removed = sum(1 for a, b in graded.values() if not (b if locator == 'wide' else a))
        return f'{removed}/{rows}', source

    if locator.startswith('mcnemar:'):
        from paired_screen_test import exact_mcnemar
        from screen_agreement import grades
        _, inventory, rest = locator.split(':', 2)
        other_dir, field = rest.split('|', 1)
        wide = inventory == 'wide'
        mine = grades(src.parent)
        theirs = grades(ROOT / other_dir)
        result = exact_mcnemar({i: r[1] if wide else r[0] for i, r in mine.items()},
                               {i: r[1] if wide else r[0] for i, r in theirs.items()})
        if field == 'p':
            return repr(result['exact_mcnemar_p_two_sided']), source
        return str(result[field]), source

    if locator.startswith('paired:'):
        from capability_format_audit import audit, paired
        parts = locator.split('|')
        base_dir = parts[0].split(':', 1)[1]
        kind, field = parts[1], parts[2]
        before = audit('base', (ROOT / base_dir) / 'predictions.jsonl', SUITE)
        after = audit('edit', src, SUITE)
        result = paired(before, after, kind)
        if field == 'p':
            return repr(result['strict']['paired_sign_flip_p_two_sided']), source
        if field == 'before_after':
            strict = result['strict']
            return f"{strict['before_pass']}→{strict['after_pass']}", source
        if field == 'lost':
            strict = result['strict']
            return str(strict['before_pass'] - strict['after_pass']), source
        return str(result['length'][field]), source

    if locator.startswith('audit|'):
        from capability_format_audit import audit
        _, kind, field = locator.split('|')
        audited = audit('x', src, SUITE)
        if field in ('median', 'median_chars'):
            return str(audited[kind]['length']['median_chars']), source
        return str(audited[kind][field]), source

    raise ValueError(f'unknown locator {locator!r}')


def values_match(prose: dict, artifact: str) -> bool:
    value, kind = prose['value'], prose['kind']
    if kind == 'fraction':
        match = re.fullmatch(r'(\d+)/(\d+)', artifact.strip())
        if not match:
            return False
        want = re.fullmatch(r'(\d+)/(\d+)', value)
        return bool(want) and (int(match.group(1)), int(match.group(2))) == (int(want.group(1)), int(want.group(2)))
    if kind == 'arrow':
        match = re.fullmatch(r'(\d+)→(\d+)', artifact.strip().replace('->', '→'))
        if not match:
            return False
        want = re.fullmatch(r'(\d+)→(\d+)', value)
        return bool(want) and (int(match.group(1)), int(match.group(2))) == (int(want.group(1)), int(want.group(2)))
    # scalar or p: compare at the precision the prose quotes.
    try:
        got = float(artifact)
    except ValueError:
        return False
    if prose.get('op') == '<':
        return got < float(value)
    if '.' in value or 'e' in value.lower():
        decimals = len(value.split('.')[1]) if '.' in value else 0
        return round(got, decimals) == float(value)
    return int(round(got)) == int(float(value))


def prose_display(token: dict) -> str:
    if token['kind'] == 'p':
        return f"p {token.get('op', '=')} {token['value']}"
    return token['value']


def artifact_display(token: dict, artifact: str) -> str:
    if token['kind'] == 'p':
        return f"p {token.get('op', '=')} {artifact}"
    return artifact


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--check', action='store_true',
                    help='fail if an artifact is missing, a README number is unbound, or a value differs')
    ap.add_argument('--json', action='store_true', help='print the verdict as JSON')
    args = ap.parse_args(argv)

    if not MAP.is_file():
        raise SystemExit(f'{MAP} does not exist')
    doc = json.loads(MAP.read_text())
    claims = doc['claims']

    missing, broken_status = [], []
    for claim in claims:
        for artifact in claim['artifacts']:
            if not (ROOT / artifact).is_file():
                missing.append((claim['status'], claim['claim'], artifact))
        if claim['status'] not in ALLOWED_STATUS:
            broken_status.append((claim['claim'], claim['status']))

    section = readme_section_iv()
    tokens = extract(section)
    mismatches = []
    checked = 0

    entries = [(claim, number) for claim in claims for number in claim.get('numbers', [])]
    used = set()
    unmapped = {}

    for token in tokens:
        line_text = dict(section)[token['line']]
        candidates = [(i, claim, number) for i, (claim, number) in enumerate(entries)
                      if number['text'] in line_text and token['text'] in number['text']]
        if not candidates:
            mismatches.append({'kind': 'unmapped', 'line': token['line'], 'text': token['text'],
                               'prose_value': prose_display(token), 'artifact_value': '-',
                               'source': '-', 'command': ''})
            unmapped[len(mismatches) - 1] = token
            continue
        if len(candidates) > 1:
            mismatches.append({'kind': 'ambiguous', 'line': token['line'], 'text': token['text'],
                               'prose_value': prose_display(token),
                               'artifact_value': '-', 'source': ', '.join(n['text'] for _, _, n in candidates),
                               'command': ''})
            continue
        index, claim, number = candidates[0]
        used.add(index)
        if number.get('void'):
            if claim['status'] == 'in-clone':
                mismatches.append({'kind': 'void-in-clone', 'line': token['line'], 'text': number['text'],
                                   'prose_value': prose_display(token), 'artifact_value': 'void',
                                   'source': number.get('source', ''), 'command': number.get('command', '')})
                continue
            checked += 1
            continue
        try:
            artifact, source = recompute(number)
        except (FileNotFoundError, KeyError, IndexError, ValueError, TypeError) as error:
            mismatches.append({'kind': 'source-error', 'line': token['line'], 'text': number['text'],
                               'prose_value': prose_display(token), 'artifact_value': f'error: {error}',
                               'source': number.get('source', ''), 'command': number.get('command', '')})
            continue
        if not values_match(token, artifact):
            mismatches.append({'kind': 'value', 'line': token['line'], 'text': number['text'],
                               'prose_value': prose_display(token),
                               'artifact_value': artifact_display(token, artifact),
                               'source': source, 'command': number.get('command', '')})
            continue
        checked += 1

    for index, (claim, number) in enumerate(entries):
        if index in used:
            continue
        if any(number['text'] in line for _, line in section):
            mismatches.append({'kind': 'unexercised', 'line': '-', 'text': number['text'],
                               'prose_value': number.get('value', '?'), 'artifact_value': '-',
                               'source': number.get('source', ''), 'command': number.get('command', '')})
            continue
        # A prose edit inside a bound fragment (159→148 -> 159→149) is still the
        # same slot: pair the stale entry with the unmatched token whose line is
        # the entry's literal with only the value swapped, so the report shows
        # both values instead of two disconnected complaints.
        paired = None
        for mismatch_index, token in unmapped.items():
            line_text = dict(section)[token['line']]
            swapped = number['text'].replace(str(number.get('value', '')), token['value'], 1)
            if swapped != number['text'] and swapped in line_text:
                paired = (mismatch_index, token)
                break
        if paired is not None:
            mismatch_index, token = paired
            try:
                artifact, source = recompute(number)
            except (FileNotFoundError, KeyError, IndexError, ValueError, TypeError):
                artifact, source = '-', number.get('source', '')
            mismatches[mismatch_index] = {
                'kind': 'value', 'line': token['line'], 'text': number['text'],
                'prose_value': prose_display(token),
                'artifact_value': artifact_display(token, artifact),
                'source': source, 'command': number.get('command', '')}
            del unmapped[mismatch_index]
            used.add(index)
            continue
        mismatches.append({'kind': 'stale', 'line': '-', 'text': number['text'],
                           'prose_value': number.get('value', '?'), 'artifact_value': '-',
                           'source': number.get('source', ''), 'command': number.get('command', '')})

    mismatches.extend({'kind': 'missing-artifact', 'line': '-', 'text': artifact,
                       'prose_value': '-', 'artifact_value': 'absent',
                       'source': artifact, 'command': ''}
                      for _status, _claim, artifact in missing)
    mismatches.extend({'kind': 'bad-status', 'line': '-', 'text': status,
                       'prose_value': '-', 'artifact_value': '-', 'source': claim,
                       'command': ''}
                      for claim, status in broken_status)

    ok = not mismatches
    command = 'python scripts/evidence_map.py --json'
    if not ok:
        command = next((m['command'] for m in mismatches if m['command']),
                       'python scripts/evidence_map.py --check')

    if args.json:
        print(json.dumps({'ok': ok, 'claims': len(claims), 'numbers_checked': checked,
                          'mismatches': mismatches, 'next_command': command}, indent=2))
        return 0 if ok else 1

    print('# Evidence map\n')
    print(doc['purpose'] + '\n')
    print('| status | claim | artifacts named | command |')
    print('|' + '-' * 104 + '|')
    for claim in claims:
        arts = ', '.join(f'`{a}`' for a in claim['artifacts']) or '*(none)*'
        cmd = claim.get('commands', [''])[0]
        cmd = f'`{cmd}`' if cmd else ''
        print(f"| {claim['status']} | {claim['claim']} | {arts} | {cmd} |")
        if claim.get('notes'):
            print(f"| | ↳ {claim['notes']} | | |")
    print(f"\n{len(claims)} claims: "
          f"{sum(1 for c in claims if c['status'] == 'in-clone')} in-clone, "
          f"{sum(1 for c in claims if c['status'] == 'pre-2026-11-17-scoring-fix')} pre-fix, "
          f"{sum(1 for c in claims if c['status'] == 'no-artifact-in-clone')} with no artifact.")
    print(f"{len(tokens)} numbers in README section IV, {len(entries)} in the map, {checked} checked.")

    if args.check:
        if broken_status:
            for claim, status in broken_status:
                print(f'STATUS NOT IN THE ALLOWED SET [{status}]: {claim}')
        if missing:
            for status, claim, artifact in missing:
                print(f'MISSING [{status}]: {artifact}  ← {claim}')
        if mismatches:
            print('\nline | text | prose_value | artifact_value | source')
            for item in mismatches:
                print(f"{item['line']} | {item['text']} | {item['prose_value']} | "
                      f"{item['artifact_value']} | {item['source']}")
            print(f"\n{len(mismatches)} mismatch(es); {checked}/{len(tokens)} numbers checked. "
                  f"next: {command}")
            return 1
        print('\nAll artifacts named by the map are present; every README number recomputes.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
