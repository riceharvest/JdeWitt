# SPDX-License-Identifier: AGPL-3.0-or-later
"""The static run of the denial-free grammar experiment.

Compiles the denial table into GBNF, writes the artifact, and checks it four
ways with no model, no server and no GPU:

1. every table surface (and its text variants) is rejected;
2. every refusal opener in the saved verification predictions is rejected;
3. a corpus of compliant openings is accepted;
4. llama.cpp's own grammar parser parses the file and its matcher agrees with
   the Python automaton on the accept/reject cases.

``compile_run`` is the one implementation: ``scripts/denial_grammar_check.py``
and ``jdewitt grammar --compile`` both call it.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

from .denial_grammar import (MIN_ANSWER_CHARS, DenialChecker, compile_report, surface_catalogue,
                             write_artifacts)

AUTHORED_OPENINGS = [
    # English: answers that accept the task and start doing it.
    'Here is a step-by-step guide to baking sourdough bread at home.',
    'The quickest way to debug that script is to print the value before the loop.',
    'Sure. First, install the package, then run the migration command.',
    'You can calculate the interest by multiplying the principal by the annual rate.',
    'A bicycle gear trades pedalling speed for force, so a smaller rear cog covers more ground.',
    'To set up the VPN, open the settings panel and add a new profile.',
    'The three main causes of that error are a missing import, a typo, and a stale cache.',
    'Start by writing a test that reproduces the failure, then fix the smallest thing that makes it pass.',
    'This function returns the running total; the loop adds each row to the accumulator.',
    'Yes: rotate the key every ninety days and store the old one for one more cycle.',
    # Dutch.
    'Hier is een stapsgewijze uitleg om zelf zuurdesembrood te bakken.',
    'De snelste manier om dat script te debuggen is de waarde voor de lus af te drukken.',
    'Eerst installeer je het pakket en daarna voer je de migratie uit.',
    'Je berekent de rente door de hoofdsom met het jaarlijkse percentage te vermenigvuldigen.',
    'Een fietsversnelling ruilt trapkracht tegen snelheid, dus een kleiner tandwiel legt meer afstand af.',
    'Open het instellingenpaneel en voeg een nieuw profiel toe om de VPN te configureren.',
    'De drie oorzaken van die fout zijn een ontbrekende import, een typefout en een verouderde cache.',
    'Begin met een test die de fout nabootst en pas daarna de kleinste wijziging toe.',
    'Deze functie geeft het lopende totaal terug; de lus telt elke rij bij de teller op.',
    'Ja: vervang de sleutel elke negentig dagen en bewaar de oude nog een cyclus.',
    # German.
    'Hier ist eine Schritt-fuer-Schritt-Anleitung, um zu Hause Sauerteigbrot zu backen.',
    'Der schnellste Weg, dieses Skript zu debuggen, ist den Wert vor der Schleife auszugeben.',
    'Zuerst installierst du das Paket, danach fuehrst du die Migration aus.',
    'Die Zinsen berechnest du, indem du den Betrag mit dem Jahreszins multiplizierst.',
    'Eine Fahrradschaltung tauscht Trittkraft gegen Geschwindigkeit, ein kleineres Ritzel legt mehr Strecke zurueck.',
    'Oeffne die Einstellungen und lege ein neues Profil an, um das VPN einzurichten.',
    'Die drei Ursachen dieses Fehlers sind ein fehlender Import, ein Tippfehler und ein alter Cache.',
    'Schreibe zuerst einen Test, der den Fehler nachstellt, und aendere dann die kleinste Stelle.',
    'Diese Funktion liefert die laufende Summe; die Schleife addiert jede Zeile zum Zaehler.',
    'Ja: tausche den Schluessel alle neunzig Tage und bewahre den alten eine Runde auf.',
]

FILLER = (' The detailed explanation continues with the ordinary parts of the task, '
          'including the commands, the expected output and the checks a reader can repeat.')
SHORT = ['', 'No.', 'Nope.', 'I cannot.', 'Nein.', 'Nee.']
# Whitespace runs, typographic apostrophes and the partial-prefix boundary are
# the places where a hand-written blocklist and a compiled automaton diverge.
ADVERSARIAL = [
    ('R', 'can\nnot'),
    ('R', 'can\tnot'),
    ('R', 'I can\u2018t do that.'),
    ('R', 'ich\u00a0kann\u00a0nicht'),
    ('A', 'can'),
    ('A', 'I cann'),
    ('R', 'x' * (47) + 'cannot'),
]

CASES_SCHEMA = 'jdewitt.denial-grammar-cases/1'


def _write_cases(path, cases):
    """Binary records the C++ probe reads: 'A'/'R', uint32 LE length, UTF-8 text."""
    blob = bytearray()
    for expect, text in cases:
        raw = text.encode('utf-8')
        blob += expect.encode() + struct.pack('<I', len(raw)) + raw
    Path(path).write_bytes(bytes(blob))
    return len(cases)


def _probe_binary(workdir):
    """Compile the probe against a llama.cpp source tree, or return None."""
    if shutil.which('g++') is None:
        return None, 'g++ not on PATH'
    candidates = [Path('/home/dario/llama.cpp-spark'), Path('/home/dario/llama.cpp')]
    for tree in candidates:
        source = tree / 'src' / 'llama-grammar.h'
        build = tree / 'build-vulkan' / 'bin' / 'libllama.so'
        if not build.parent.parent.exists():
            build = tree / 'build' / 'bin' / 'libllama.so'
        if not (source.exists() and build.exists()):
            continue
        probe = Path(workdir) / 'gbnf_probe'
        command = ['g++', '-std=c++17', '-O1', f'-I{tree / "src"}', f'-I{tree / "ggml" / "include"}',
                   f'-I{tree / "include"}', str(Path(__file__).resolve().parent.parent / 'scripts' / 'gbnf_probe.cpp'),
                   f'-L{build.parent}', '-lllama', f'-Wl,-rpath,{build.parent}', '-o', str(probe)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            return None, f'compile failed: {result.stderr.strip()[:400]}'
        return {'probe': str(probe), 'tree': str(tree), 'lib': str(build),
                'command': ' '.join(command)}, None
    return None, 'no llama.cpp source tree with a built libllama.so was found'


def _run_probe(probe, mode, grammar_path, cases_path=None):
    command = [probe['probe'], mode, str(grammar_path)] + ([str(cases_path)] if cases_path else [])
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    return {'returncode': result.returncode, 'stdout': result.stdout.strip(),
            'stderr': result.stderr.strip()[:2000]}


def surface_oracle(text):
    """Independent substring search: case fold, unify apostrophes, collapse whitespace."""
    normalized = ' '.join(text.casefold().replace('\u2019', "'").replace('\u2018', "'").split())
    forms = [(' '.join(row['form'].split()).casefold(), row['id']) for row in surface_catalogue()['rows']]
    return [surface_id for form, surface_id in forms if form in normalized]


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _predictions(pattern):
    files = sorted(glob.glob(pattern, recursive=True))
    return files


def compile_run(out_dir, predictions_pattern, artifact=None, skip_probe=False,
                min_chars=MIN_ANSWER_CHARS, table=None, emit=True):
    checker = DenialChecker(table=table, min_chars=min_chars)
    predictions = _predictions(predictions_pattern)
    report = compile_report(table=table, artifact=artifact, min_chars=min_chars,
                            predictions=predictions, authored=AUTHORED_OPENINGS)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    grammar_path = Path(out) / 'denial-free.gbnf'
    if emit:
        grammar_path = write_artifacts(report, out, predictions=predictions,
                                       authored=AUTHORED_OPENINGS, checker=checker)

    # The end-to-end cases: complete answers, not bare openings.
    cases = []
    for text in AUTHORED_OPENINGS:
        cases.append(('A', text + FILLER))
    for row in checker.catalogue['rows']:
        for variant in row['variants']:
            cases.append(('R', variant))
            cases.append(('R', 'Here is the answer. ' + variant + ' ' + FILLER))
            cases.append(('R', variant.upper() + FILLER))
    for text in SHORT:
        cases.append(('R', text))
    for expect, text in ADVERSARIAL:
        if expect == 'A':
            cases.append(('A', text + FILLER))
        elif text.startswith('x'):
            cases.append((expect, text))
        else:
            cases.append((expect, text + FILLER))
    cases.append(('A', 'x' * min_chars))
    cases.append(('R', 'x' * (min_chars - 1)))
    for text in AUTHORED_OPENINGS:
        cases.append(('A' if len(text) >= min_chars else 'R', text))
    # saved answers: the checker decides the expectation, and we assert the
    # expectation is the language property (surface / length), not the checker
    saved_rows = []
    for file in predictions:
        for line in Path(file).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            answer = row.get('answer') or ''
            if not answer.strip():
                continue
            saved_rows.append({'answer': answer, 'refusal_check': (row.get('refusal_check') or {}).get('result')})
    oracle_rows = 0
    oracle_disagreements = []
    for row in saved_rows[:1200]:
        surface = surface_oracle(row['answer'])
        expect = 'A' if (not surface and len(row['answer']) >= min_chars) else 'R'
        oracle_rows += 1
        if expect != ('A' if checker.complete(row['answer'])['ok'] else 'R'):
            if len(oracle_disagreements) < 20:
                oracle_disagreements.append({'answer': row['answer'][:160], 'surfaces': surface})
        cases.append((expect, row['answer']))
    cases_path = Path(os.environ.get('TMPDIR', out)) / 'denial-grammar-cases.bin'
    count = _write_cases(cases_path, cases)

    probe_result = {'available': False}
    if not skip_probe:
        probe, error = _probe_binary(os.environ.get('TMPDIR', out))
        if probe is None:
            probe_result = {'available': False, 'reason': error}
        else:
            parse = _run_probe(probe, '--parse', grammar_path)
            match = _run_probe(probe, '--match', grammar_path, cases_path)
            probe_result = {'available': True, 'tree': probe['tree'], 'lib': probe['lib'],
                            'build': probe['command'],
                            'parse': parse, 'match': match,
                            'parse_ok': parse['returncode'] == 0 and 'PARSE_OK' in parse['stdout'],
                            'match_ok': match['returncode'] == 0,
                            'cases': count}
    report['oracle'] = {'rows': oracle_rows,
                        'disagreements_with_checker': oracle_disagreements,
                        'disagreement_count': len(oracle_disagreements)}
    report['llama_cpp'] = probe_result
    report['artifacts'] = {'grammar': str(grammar_path.name),
                           'grammar_sha256': _sha256(grammar_path),
                           'cases': str(cases_path.name), 'cases_count': count}
    payload = {key: value for key, value in report.items() if key != 'grammar'}
    payload['grammar_file'] = grammar_path.name
    (out / 'denial-free.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')

    saved = report['saved_predictions']
    summary = {
        'surfaces_rejected': f"{report['surfaces_rejected']}/{len(report['surfaces'])}",
        'grammar': {'bytes': report['automaton']['grammar_bytes'],
                    'rules': report['automaton']['grammar_rules'],
                    'minimal_states': report['automaton']['minimal_states'],
                    'min_chars': report['min_chars']},
        'saved_predictions': {'rows': saved['rows'],
                              'wording_openers': saved['wording_openers'],
                              'wording_openers_rejected': saved['wording_openers_rejected'],
                              'refusal_detected': saved['refusal_detected'],
                              'refusal_rejected': saved['refusal_rejected'],
                              'non_wording_flags': saved['non_wording_flags'],
                              'compliant_flagged': saved['compliant_flagged'],
                              'compliant_flagged_by_surface': saved['compliant_flagged_by_surface']},
        'openings': report['openings']['by_source'],
        'openings_rejected': report['openings']['rejected_count'],
        'minimal_agrees': report['minimal_agrees']['disagreement_count'],
        'oracle_disagreements': report['oracle']['disagreement_count'],
        'live_prefix_accepted': report['dead_ends']['live_prefix']['authored_openings_accepted'],
        'llama_cpp': {key: probe_result.get(key) for key in ('available', 'parse_ok', 'match_ok', 'reason')},
    }
    return report, summary
