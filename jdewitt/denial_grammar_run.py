# SPDX-License-Identifier: AGPL-3.0-or-later
"""The static run of the denial-free grammar experiment.

Compiles the denial table into GBNF, writes the artifact, and checks it four
ways with no model, no server and no GPU:

1. every table surface (and its text variants) is rejected;
2. every refusal opener in the saved verification predictions is rejected;
3. a corpus of compliant openings is accepted;
4. llama.cpp's own grammar parser parses the file and its matcher agrees with
   the Python automaton on the accept/reject cases.

With ``mine=True`` the table is first grown from the saved answers (see
``denial_mine``), the extended table is emitted as ``denial-surfaces.json``, and
the run reports the before/after delta plus the trade-off curve over the tiers.

``compile_run`` is the one implementation: ``scripts/denial_grammar_check.py``
and ``jdewitt grammar --compile`` both call it.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path

from .denial_grammar import (MIN_ANSWER_CHARS, DenialChecker, _prediction_answers,
                             compare_runs, compile_report, measure, surface_catalogue,
                             write_artifacts)
from .denial_mine import mine_table
from .vocabremap import load_table

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


def surface_oracle(text, catalogue=None):
    """Independent regex search over the compiled spellings.

    Deliberately a different implementation from the automaton: variants are
    plain substrings of the folded, whitespace-collapsed text, and the guarded
    spellings are regexes with a lookbehind that refuses a letter before them.
    Agreements between this and the DFA are evidence, not a tautology.
    """
    import re as _re
    rows = (catalogue or surface_catalogue())['rows']
    normalized = ' '.join(text.casefold().replace('\u2019', "'").replace('\u2018', "'").split())
    hits = set()
    for row in rows:
        for form in row['variants']:
            if ' '.join(form.split()).casefold() in normalized:
                hits.add(row['id'])
        for form in row['guarded_forms']:
            pattern = r'(?<![a-z])' + _re.escape(' '.join(form.split()).casefold())
            if _re.search(pattern, normalized):
                hits.add(row['id'])
    return sorted(hits)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _predictions(pattern):
    files = sorted(glob.glob(pattern, recursive=True))
    return files


def _tier_documents(document, mining=None):
    """The trade-off curve: cumulative table tiers, each a compilable document.

    Tier 0 is the seed table exactly as the previous card compiled it (no elided
    spellings).  Each later tier adds one shape class from the mined rows; the
    last tier adds the frames the adoption rule rejects, so the curve shows what
    buying that coverage would cost.
    """
    seed_document, _raw = load_table()
    mined = [entry for entry in document['entries'] if entry.get('source') == 'mined']
    seed = [entry for entry in document['entries'] if entry.get('source') == 'seed']
    order = [('apology', 'apology'), ('capability_disclaimer', 'capability'),
             ('refusal_opener', 'refusal opener')]
    tiers = [{'name': 'seed', 'label': 'seed table, as shipped by t_4d944c4e',
              'document': seed_document}]
    current = list(seed)
    tiers.append({'name': 'seed+elisions',
                  'label': 'seed rows plus elided apostrophe spellings (word-start guarded)',
                  'document': _document_like(document, current)})
    for kind, label in order:
        extra = [entry for entry in mined if entry['kind'] == kind]
        if not extra:
            continue
        current = current + extra
        tiers.append({'name': f'+{kind}', 'label': f'plus {label} rows',
                      'document': _document_like(document, current)})
    frames = _frame_entries(mining)
    if frames:
        tiers.append({'name': '+frames',
                      'label': 'plus the lecture/disclaimer frames the adoption rule rejects '
                               '(measured, not shipped)',
                      'document': _document_like(document, current + frames)})
    return tiers


def _frame_entries(mining):
    """Table entries for the frame shapes the adoption rule excluded, for the curve."""
    if not mining:
        return []
    entries = []
    for record in mining['candidates']['excluded_rows']:
        if record['kind'] != 'refusal_frame' or record['marginal_gain'] <= 0:
            continue
        slug = re.sub(r'[^a-z0-9]+', '-', record['form'].lower().replace("'", '')).strip('-')
        entries.append({'id': f'measured-{slug}', 'language': 'en', 'form': record['form'],
                        'kind': 'refusal_frame', 'scope': 'opener_window', 'replacement': None,
                        'note': 'measured frame, not shipped', 'rule': record['rule'],
                        'source': 'measured_frame', 'word_start': False, 'guarded_forms': [],
                        'evidence_count': record['marginal_gain'], 'evidence': []})
    return entries


def _document_like(document, entries):
    payload = {key: value for key, value in document.items() if key != 'entries'}
    payload['entries'] = entries
    return payload


def _curve(tiers, rows, authored, min_chars):
    curve = []
    for tier in tiers:
        checker = DenialChecker(table=tier['document'], min_chars=min_chars)
        measured = measure(checker, rows, authored)
        grammar = checker.grammar()
        curve.append({
            'tier': tier['name'], 'label': tier['label'],
            'surfaces': len(checker.catalogue['rows']),
            'guarded_spellings': sum(len(row['guarded_forms']) for row in checker.catalogue['rows']),
            'minimal_states': len(checker.minimal['transitions']),
            'grammar_rules': len(checker.minimal['transitions']) * (min_chars + 1) + 1,
            'grammar_bytes': len(grammar.encode()),
            'refusal_opener_rejected': measured['refusal']['opener_rejected'],
            'refusal_opener_rows': measured['refusal']['rows'],
            'refusal_whole_rejected': measured['refusal']['whole_answer_rejected'],
            'compliant_accepted': measured['compliant']['accepted'],
            'compliant_rows': measured['compliant']['rows'],
            'compliant_accept_rate': measured['compliant']['accept_rate'],
            'authored_openings_accepted': measured['openings']['by_source'].get(
                'authored', {}).get('accepted', 0),
            'saved_openings_accepted': measured['openings']['by_source'].get(
                'saved_compliant', {}).get('accepted', 0),
            'saved_openings_rows': measured['openings']['by_source'].get(
                'saved_compliant', {}).get('rows', 0),
        })
    return curve


def compile_run(out_dir, predictions_pattern, artifact=None, skip_probe=False,
                min_chars=MIN_ANSWER_CHARS, table=None, emit=True, mine=False):
    predictions = _predictions(predictions_pattern)
    rows = _prediction_answers(predictions)
    seed_document, _seed_raw = load_table()

    mining = None
    if mine:
        table, mining = mine_table(predictions, seed_document=seed_document)
    checker = DenialChecker(table=table, min_chars=min_chars)
    baseline = DenialChecker(table=None, min_chars=min_chars)

    report = compile_report(table=table, artifact=artifact, min_chars=min_chars,
                            predictions=predictions, authored=AUTHORED_OPENINGS, rows=rows)
    report['delta'] = compare_runs(baseline, checker, rows, AUTHORED_OPENINGS)
    if artifact is not None and report.get('tokenizer'):
        from .denial_grammar import read_tokenizer, token_report
        tokenizer = read_tokenizer(artifact)
        before_token = token_report(baseline.automaton, tokenizer['tokens'])
        report['delta']['tokenizer'] = {
            key: {'before': before_token[key], 'after': report['tokenizer'][key]}
            for key in ('pieces', 'pieces_forbidden_at_start', 'pieces_forbidden_from_every_state',
                        'pieces_ending_inside_a_partial_match')}
        report['delta']['tokenizer']['allowed_pieces_min'] = {
            'before': before_token['allowed_pieces_per_state']['min'],
            'after': report['tokenizer']['allowed_pieces_per_state']['min']}
    tiers = _tier_documents(checker.catalogue['document'], mining)
    report['tradeoff'] = {'tiers': _curve(tiers, rows, AUTHORED_OPENINGS, min_chars),
                          'rule': ('a candidate surface is adopted when it closes at least one '
                                   'additional refusal opener and blocks no more genuinely '
                                   'compliant answers than it closes; the last tier shows the '
                                   'frames that the rule excludes')}
    if mining is not None:
        report['mining'] = mining
        report['table_document'] = checker.catalogue['document']

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    grammar_path = Path(out) / 'denial-free.gbnf'
    if emit:
        grammar_path = write_artifacts(report, out, predictions=predictions,
                                       authored=AUTHORED_OPENINGS, checker=checker)
        if mining is not None:
            (out / 'denial-surfaces.json').write_text(
                json.dumps(checker.catalogue['document'], indent=2, ensure_ascii=False) + '\n')
            (out / 'denial-surfaces-mining.json').write_text(
                json.dumps(mining, indent=2, ensure_ascii=False) + '\n')

    # The end-to-end cases: complete answers, not bare openings.
    cases = []
    for text in AUTHORED_OPENINGS:
        cases.append(('A', text + FILLER))
    for row in checker.catalogue['rows']:
        for variant in row['variants']:
            cases.append(('R', variant))
            cases.append(('R', 'Here is the answer. ' + variant + ' ' + FILLER))
            cases.append(('R', variant.upper() + FILLER))
        for form in row['guarded_forms']:
            cases.append(('R', form))
            cases.append(('R', 'So, ' + form + ' with that.' + FILLER))
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
    saved_rows = [row for row in rows]
    oracle_rows = 0
    oracle_disagreements = []
    for row in saved_rows[:1200]:
        surface = surface_oracle(row['answer'], checker.catalogue)
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
    # the mining report and the table document are written to their own files;
    # embedding them here would triple the size of this one
    payload = {key: value for key, value in report.items()
               if key not in ('grammar', 'table_document', 'mining')}
    payload['grammar_file'] = grammar_path.name
    (out / 'denial-free.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')

    saved = report['saved_predictions']
    refusal, compliant = saved['refusal'], saved['compliant']
    delta = report['delta']
    summary = {
        'removal_measured': False,
        'surfaces_rejected': f"{report['surfaces_rejected']}/{len(report['surfaces'])}",
        'grammar': {'bytes': report['automaton']['grammar_bytes'],
                    'rules': report['automaton']['grammar_rules'],
                    'minimal_states': report['automaton']['minimal_states'],
                    'min_chars': report['min_chars']},
        'refusal': {'rows': refusal['rows'],
                    'opener_rejected': refusal['opener_rejected'],
                    'opener_missed': refusal['opener_missed'],
                    'whole_rejected': refusal['whole_answer_rejected'],
                    'whole_missed': refusal['whole_answer_missed'],
                    'wide_flagged': refusal['wide_flagged'],
                    'wide_rejected': refusal['wide_whole_rejected'],
                    'wording_openers': refusal['wording_openers'],
                    'wording_openers_rejected': refusal['wording_openers_rejected']},
        'compliant': {'rows': compliant['rows'], 'accepted': compliant['accepted'],
                      'rate': compliant['accept_rate']},
        'openings': report['openings']['by_source'],
        'delta': {'surfaces': delta['surfaces'],
                  'refusal': {side: delta['refusal'][side] for side in ('before', 'after')},
                  'compliant': {side: delta['compliant'][side] for side in ('before', 'after')},
                  'openings': {side: delta['openings'][side] for side in ('before', 'after')},
                  'grammar': delta['grammar'],
                  'added': [{'id': row['id'], 'form': row['form'],
                             'evidence_count': row['evidence_count']}
                            for row in delta['table']['added']]},
        'tradeoff': [{key: tier[key] for key in
                      ('tier', 'surfaces', 'refusal_opener_rejected', 'compliant_accepted',
                       'compliant_accept_rate', 'saved_openings_accepted')}
                     for tier in report['tradeoff']['tiers']],
        'minimal_agrees': report['minimal_agrees']['disagreement_count'],
        'oracle_disagreements': report['oracle']['disagreement_count'],
        'live_prefix_accepted': report['dead_ends']['live_prefix']['authored_openings_accepted'],
        'llama_cpp': {key: probe_result.get(key) for key in ('available', 'parse_ok', 'match_ok', 'reason')},
    }
    return report, summary
