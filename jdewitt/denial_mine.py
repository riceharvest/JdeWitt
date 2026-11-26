# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mine the denial table from the saved answers, with an explicit rule set.

The hand-written table in ``assets/vocab-remap-denial.json`` is an inventory of
20 EN/NL/DE surfaces.  A grammar compiled from it is only as complete as that
inventory, so this module reads the answers the project already saved, finds the
refusal openers they actually start with, and grows the table by the attested
shapes.  Everything here is string work: no model, no tokenizer, no forward
pass.  A candidate surface is admitted only when a saved row contains it
verbatim, and the row id travels with it into the emitted table.

The rule set is a list of regexes over a fixed opening window (the first
``OPENING_CHARS`` characters of the answer, whitespace-collapsed and with the
typographic apostrophes folded to ASCII).  Each rule names a shape class and a
stem; the surface is the stem plus at most ``extend`` following words, so
``I cannot`` grows into ``I cannot provide`` only where a saved answer says it.
Rules never invent a spelling: a match is the surface.

Adoption is not "everything that matched".  For every candidate the miner
measures, against the seed table's coverage,

* ``marginal_gain``: refusal-labelled saved answers whose opening window has no
  surface yet, i.e. the rows this candidate would newly close;
* ``marginal_cost``: relabelled-compliant saved answers this candidate would
  newly reject anywhere in the answer (the automaton is a substring machine, so
  a surface costs wherever it occurs).

A refusal-labelled row is one either checker flagged (the narrow
``refusal_check`` or the wide ``refusal_check_wide`` inventory); a compliant row
is one neither flagged.  Candidates are adopted greedily, best gain/cost first,
under one stated rule: a candidate is adopted when it closes at least one
additional refusal opener and does not block more compliant answers than it
closes.  Rows rejected by the rule are kept in the report with their numbers,
because a table that buys refusal coverage by refusing compliant answers is a
regression, not a success.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SCHEMA = 'jdewitt.denial-surfaces/1'
OPENING_CHARS = 160
EVIDENCE_CAP = 5

APOSTROPHE_FOLD = str.maketrans({'\u2019': "'", '\u2018': "'", '\u02bc': "'"})
_LOWER = str.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')

# Words that may not end an extended surface: "I cannot provide" is a surface,
# "I cannot provide instructions" keeps the object and generalises less.
STOPWORDS = frozenset("""a an the any this that these those it its it's i we me my our your you
some all such their his her there here to of in on at and or but as is are be been being
for with about from by not so very please just also only more most other same own
do does did can could will would should may might must have has had if then than when
what which who whom whose how why where""".split())


# --------------------------------------------------------------------------
# Shape rules
# --------------------------------------------------------------------------
#
# id, kind, category, pattern over the folded opening window, extend, note.
# ``extend`` counts following words kept as part of the surface.  A rule whose
# surface must start at a word boundary declares ``word_start``; only short
# stems need it, because the automaton matches substrings.

RULES = (
    {'id': 'refusal-modal', 'kind': 'refusal_opener', 'category': 'negative_modal',
     'pattern': r"\b(?:i|we)\s*(?:cannot|can\s+not|can't|won't|will\s+not)\b", 'extend': 1,
     'word_start': False,
     'note': 'First-person negative modal; the object is where models vary.'},
    {'id': 'refusal-inability', 'kind': 'refusal_opener', 'category': 'inability',
     'pattern': r"\b(?:i|we)\s*(?:am\s+unable\s+to|'m\s+unable\s+to|am\s+not\s+able\s+to|"
                r"'m\s+not\s+able\s+to|do\s+not\s+have\s+the\s+ability\s+to)\b", 'extend': 1,
     'word_start': False,
     'note': 'Explicit inability rather than a negative modal.'},
    {'id': 'refusal-negated-verb', 'kind': 'refusal_opener', 'category': 'negated_verb',
     'pattern': r"\b(?:i|we)\s*(?:do\s+not|don't|didn't|am\s+not\s+able|'m\s+not\s+able)\b",
     'extend': 1, 'word_start': False,
     'note': 'Negated first-person verb ("I do not write ..."), the widest class.'},
    {'id': 'refusal-plain', 'kind': 'refusal_opener', 'category': 'plain_refusal',
     'pattern': r"\b(?:i|we)\s*(?:must\s+refuse|refuse\s+to|decline|have\s+to\s+decline|"
                r"must\s+decline|'d\s+rather\s+not|prefer\s+not\s+to|am\s+not\s+going\s+to|"
                r"'m\s+not\s+going\s+to)\b", 'extend': 1, 'word_start': False,
     'note': 'A refusal verb without a modal ("I refuse to", "I must decline").'},
    {'id': 'apology', 'kind': 'apology', 'category': 'apology',
     'pattern': r"\b(?:i\s+am\s+(?:really\s+|so\s+|very\s+)?sorry|i'm\s+(?:really\s+|so\s+|very\s+)?"
                r"sorry|i\s+apologize|i\s+am\s+afraid|i'm\s+afraid)\b",
     'extend': 1, 'word_start': False,
     'note': 'Apology and regret openers, including the intensifier inflection.'},
    {'id': 'refusal-inability-idiom', 'kind': 'refusal_opener', 'category': 'inability',
     'pattern': r"\bi\s+am\s+not\s+in\s+a\s+position\s+to\b", 'extend': 1, 'word_start': False,
     'note': 'Idiomatic inability ("I am not in a position to provide ...").'},
    {'id': 'apology-concessive', 'kind': 'apology', 'category': 'apology',
     'pattern': r"\b(?:i\s+am|i'm)\s+(?:really\s+|so\s+|very\s+)?sorry\s*,?\s+but\b", 'extend': 0,
     'word_start': False,
     'note': 'Concessive apology opener; the "but" is what marks the refusal clause.'},
    {'id': 'capability-disclaimer', 'kind': 'capability_disclaimer', 'category': 'capability',
     'pattern': r"\bi\s+(?:don't|do\s+not)\s+have\s+(?:personal\s+)?(?:preferences|beliefs|"
                r"opinions|feelings|emotions|a\s+body|the\s+ability)\b", 'extend': 0,
     'word_start': False,
     'note': 'Capability disclaimer used as a refusal ("I do not have personal preferences").'},
    {'id': 'ai-disclaimer', 'kind': 'refusal_frame', 'category': 'disclaimer',
     'pattern': r"\bas\s+an\s+ai(?:\s+(?:model|assistant|language\s+model))?\b", 'extend': 1,
     'word_start': False,
     'note': 'AI self-description; only a refusal when a denial follows it.'},
    {'id': 'not-something', 'kind': 'refusal_opener', 'category': 'not_something',
     'pattern': r"\b(?:that|this|it)(?:'s|\s+is)?\s+not\s+something\b", 'extend': 2,
     'word_start': False,
     'note': 'Object-first refusal ("that is not something I can help with").'},
    {'id': 'not-something-i-can', 'kind': 'refusal_opener', 'category': 'not_something',
     'pattern': r"\bnot\s+something\s+i\s+can\b", 'extend': 1, 'word_start': False,
     'note': 'The same shape with the subject first.'},
    {'id': 'request-modal', 'kind': 'refusal_opener', 'category': 'request',
     'pattern': r"\b(?:that|this)\s+(?:request|task)\s+(?:cannot|can't)\b", 'extend': 1,
     'word_start': False,
     'note': 'Refusal predicated on the request rather than on the speaker.'},
    {'id': 'frame-clarify', 'kind': 'refusal_frame', 'category': 'meta',
     'pattern': r"\bit's\s+important\s+to\s+(?:clarify|note|emphasize|remember)\b", 'extend': 1,
     'word_start': False,
     'note': 'Lecture frame that often carries a soft refusal; not a denial by itself.'},
    {'id': 'frame-emphasize', 'kind': 'refusal_frame', 'category': 'meta',
     'pattern': r"\bi\s+must\s+emphasize\b", 'extend': 1, 'word_start': False,
     'note': 'First-person lecture frame; not a denial by itself.'},
    {'id': 'imperative-not', 'kind': 'refusal_opener', 'category': 'imperative',
     'pattern': r"\babsolutely\s+not\b", 'extend': 0, 'word_start': False,
     'note': 'Bare refusal; also a common intensifier in compliant prose.'},
)

# Elided apostrophes: a spelling variant of an existing surface, not a new shape.
# They are compiled behind a word-start guard, because "cant" is a substring of
# "significant" and the other short forms are just as easy to collide with.
ELISIONS = {
    'en-cant': ['cant'],
    'en-wont': ['wont'],
    'en-im-sorry': ['im sorry'],
    'en-i-am-sorry': ['im sorry'],
}


def normalize(text):
    """Fold the apostrophes and collapse whitespace; the corpus is UTF-8 text."""
    return ' '.join(text.translate(APOSTROPHE_FOLD).split())


def fold(text):
    """Lowercase without changing the length, so match offsets stay valid."""
    return text.translate(_LOWER)


def opening_window(text, chars=OPENING_CHARS):
    return normalize(text)[:chars]


def _extend(window, end, words):
    """Keep up to ``words`` following words, stopping at a stopword or punctuation."""
    out = []
    cursor = end
    while len(out) < words:
        match = re.match(r" +([A-Za-z][A-Za-z'\-]*)", window[cursor:])
        if not match:
            break
        word = match.group(1)
        if word.lower() in STOPWORDS and not out:
            break
        out.append(word)
        cursor += match.end()
        if word.lower() in STOPWORDS:
            break
    return window[end:cursor]


def candidate_surfaces(window, rules=RULES):
    """Every (rule, surface) a single opening window attests, in text order."""
    folded = fold(window)
    found = []
    for rule in rules:
        pattern = re.compile(rule['pattern'], re.IGNORECASE)
        for match in pattern.finditer(folded):
            # slice the original window, not the folded match: the form in the
            # table keeps the spelling the answer used
            tail = _extend(window, match.end(), rule['extend'])
            span = ' '.join(window[match.start():match.end() + len(tail)].split())
            if not span:
                continue
            found.append({'rule': rule, 'form': span, 'offset': match.start(),
                          'folded': span.translate(APOSTROPHE_FOLD).lower()})
    return found


# --------------------------------------------------------------------------
# Corpus reading
# --------------------------------------------------------------------------

def read_corpus(paths):
    """Every non-empty saved answer, with the roster identity for evidence."""
    rows = []
    for path in paths:
        path = Path(path)
        files = sorted(path.rglob('predictions.jsonl')) if path.is_dir() else [path]
        for file in files:
            for number, line in enumerate(file.read_text().splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                answer = row.get('answer') or row.get('raw') or ''
                if not isinstance(answer, str) or not answer.strip():
                    continue
                check = row.get('refusal_check') or {}
                wide = row.get('refusal_check_wide') or {}
                folded = fold(normalize(answer))
                rows.append({'row_id': row.get('id'), 'file': str(file), 'line': number,
                             'answer': answer, 'folded_answer': folded,
                             'opening': opening_window(answer),
                             'folded_opening': fold(opening_window(answer)),
                             'refusal_check': check.get('result'),
                             'check_method': check.get('method'),
                             'wide_check': wide.get('result'),
                             'wide_evidence': wide.get('evidence'),
                             'language': row.get('language')})
    return rows


def is_refusal(row):
    """Labelled a refusal by either checker: the wide inventory counts too."""
    return 'refusal_detected' in (row['refusal_check'], row['wide_check'])


def is_compliant(row):
    """Neither checker flagged it: the only rows a larger table can regress."""
    return (row['refusal_check'] == 'no_explicit_refusal'
            and row['wide_check'] != 'refusal_detected')


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

def refusal_shaped(row, rules=RULES):
    """True when the opening window of a row matches at least one shape rule.

    Used as a label audit: a 'compliant' answer that opens with a refusal shape
    is counted separately from a genuine compliant answer, because the narrow
    classifier is known to miss refusals that do not use its wording list.
    """
    folded = fold(row['opening'])
    return any(re.search(rule['pattern'], folded, re.IGNORECASE) for rule in rules)


def measure_candidates(rows, rules=RULES):
    """Per candidate: refusal-opening attestations, absolute gain and cost."""
    refusal = [row for row in rows if is_refusal(row)]
    compliant = [row for row in rows if is_compliant(row)]
    shaped = {(row['file'], row['line']): refusal_shaped(row, rules) for row in compliant}
    candidates = {}
    for row in refusal:
        for item in candidate_surfaces(row['opening'], rules):
            entry = candidates.setdefault(item['folded'], {
                'form': item['form'], 'rule': item['rule']['id'],
                'kind': item['rule']['kind'], 'category': item['rule']['category'],
                'language': row['language'] or 'en', 'first_offset': item['offset'],
                'refusal_rows': [], 'gain': 0, 'cost': 0, 'cost_refusal_shaped': 0})
            entry['first_offset'] = min(entry['first_offset'], item['offset'])
            if all(existing['row_id'] != row['row_id'] for existing in entry['refusal_rows']):
                entry['refusal_rows'].append({'row_id': row['row_id'], 'file': row['file'],
                                              'line': row['line'], 'span': item['form']})
    for key, entry in candidates.items():
        entry['gain'] = sum(1 for row in refusal if key in row['folded_opening'])
        hits = [row for row in compliant if key in row['folded_answer']]
        entry['cost'] = len(hits)
        entry['cost_refusal_shaped'] = sum(1 for row in hits
                                           if shaped[(row['file'], row['line'])])
    return candidates


def adopt(candidates, seed_folded, rows, rules=RULES):
    """Greedy adoption under the stated rule; returns adopted and excluded rows.

    ``seed_folded`` are the folded forms the seed table already blocks.  A
    candidate is adopted when it closes at least one refusal opener the current
    set does not, is a denial shape rather than a frame, and blocks no more
    genuinely compliant answers than it closes.  Ordering: coverage per
    compliant answer blocked first (so a specific surface wins over a wider one
    that costs empathy: ``I'm really sorry, but`` over ``I'm really sorry``),
    then the widest coverage, then the earliest attestation in the opener.
    """
    refusal = [row for row in rows if is_refusal(row)]
    compliant = [row for row in rows if is_compliant(row)]
    shaped = {(row['file'], row['line']): refusal_shaped(row, rules) for row in compliant}
    covered = {(row['file'], row['line']) for row in refusal
               if any(form in row['folded_opening'] for form in seed_folded)}
    blocked = {(row['file'], row['line']) for row in compliant
               if any(form in row['folded_answer'] for form in seed_folded)}

    ordered = sorted(candidates.items(),
                     key=lambda item: (item[1]['cost'] / max(item[1]['gain'], 1),
                                       -item[1]['gain'], item[1]['first_offset'],
                                       item[1]['form'].lower()))
    adopted, excluded = [], []
    for key, entry in ordered:
        closed_now = [row for row in refusal
                      if (row['file'], row['line']) not in covered and key in row['folded_opening']]
        blocks_now = [row for row in compliant
                      if (row['file'], row['line']) not in blocked and key in row['folded_answer']]
        genuine = [row for row in blocks_now if not shaped[(row['file'], row['line'])]]
        record = {'form': entry['form'], 'rule': entry['rule'], 'kind': entry['kind'],
                  'category': entry['category'], 'absolute_gain': entry['gain'],
                  'absolute_cost': entry['cost'],
                  'absolute_cost_refusal_shaped': entry['cost_refusal_shaped'],
                  'first_offset': entry['first_offset'],
                  'marginal_gain': len(closed_now), 'marginal_cost': len(blocks_now),
                  'marginal_cost_genuine': len(genuine), 'evidence': entry['refusal_rows']}
        if entry['kind'] == 'refusal_frame':
            record['decision'] = 'excluded'
            record['reason'] = ('frame, not a denial: it introduces refusals but is not one, '
                                'and blocking it also blocks compliant answers that use the frame')
        elif closed_now and len(genuine) <= len(closed_now):
            covered.update((row['file'], row['line']) for row in closed_now)
            blocked.update((row['file'], row['line']) for row in blocks_now)
            record['decision'] = 'adopted'
            record['blocked_examples'] = [
                {'row_id': row['row_id'], 'file': row['file'], 'line': row['line'],
                 'text': row['answer'][:140], 'refusal_shaped': shaped[(row['file'], row['line'])]}
                for row in blocks_now[:5]]
            adopted.append(record)
        else:
            record['decision'] = 'excluded'
            record['reason'] = ('no additional refusal opener' if not closed_now
                                else 'blocks more genuinely compliant answers than it closes')
        if record['decision'] == 'excluded':
            if closed_now:
                record['blocked_examples'] = [{'row_id': row['row_id'], 'file': row['file'],
                                               'line': row['line'], 'text': row['answer'][:140]}
                                              for row in blocks_now[:5]]
            excluded.append(record)
    return adopted, excluded, len(covered), len(blocked)


# --------------------------------------------------------------------------
# Table document
# --------------------------------------------------------------------------

def _slug(text):
    slug = text.lower().replace("'", '').replace('\u2019', '')
    slug = re.sub(r'[^a-z0-9]+', '-', slug).strip('-')
    return slug


def build_table(seed_document, adopted, rows, seed_evidence=None):
    """The seed rows (with evidence) plus the adopted mined rows, one row each."""
    seed_evidence = seed_evidence or {}
    compliant = [row for row in rows if is_compliant(row)]
    entries = []
    used = set()
    for entry in seed_document['entries']:
        evidence = seed_evidence.get(entry['id'], [])
        new = dict(entry)
        new.update({'rule': 'seed', 'source': 'seed', 'word_start': False,
                    'guarded_forms': ELISIONS.get(entry['id'], []),
                    'evidence_count': len(evidence), 'evidence': evidence[:EVIDENCE_CAP]})
        entries.append(new)
        used.add(entry['id'])
    for record in adopted:
        base = 'en-' + _slug(record['form'])
        row_id, suffix = base, 2
        while row_id in used:
            row_id = f'{base}-{suffix}'
            suffix += 1
        used.add(row_id)
        folded = fold(record['form'])
        cost = sum(1 for row in compliant if folded in row['folded_answer'])
        entries.append({
            'id': row_id, 'language': 'en', 'form': record['form'], 'kind': record['kind'],
            'scope': 'opener_window', 'replacement': None,
            'note': f"mined by rule {record['rule']} ({record['category']}); "
                    f"attested in {record['absolute_gain']} refusal rows",
            'rule': record['rule'], 'source': 'mined', 'word_start': False,
            'guarded_forms': [], 'evidence_count': len(record['evidence']),
            'evidence': record['evidence'][:EVIDENCE_CAP], 'compliant_cost': cost})
    return {'schema': SCHEMA,
            'note': ('Grown from assets/vocab-remap-denial.json by '
                     'scripts/denial_table_mine.py against the saved answers; each row carries the '
                     'saved rows that attest it. Surfaces only: no token ids, no replacement policy. '
                     'guarded_forms are elided spellings compiled behind a word-start guard.'),
            'seed_schema': seed_document.get('schema'),
            'opening_window_chars': OPENING_CHARS,
            'entries': entries}


def seed_evidence(rows, seed_document):
    """For every seed row, the saved refusal rows that attest its form."""
    refusal = [row for row in rows if is_refusal(row)]
    evidence = {}
    for entry in seed_document['entries']:
        forms = {fold(normalize(entry['form']))}
        forms |= {form.replace("'", '\u2019') for form in list(forms)}
        hits = []
        for row in refusal:
            if any(form in row['folded_answer'] for form in forms):
                hits.append({'row_id': row['row_id'], 'file': row['file'],
                             'line': row['line'], 'span': entry['form']})
        evidence[entry['id']] = hits
    return evidence


def mine_table(paths, seed_document=None, rules=RULES):
    """Read the corpus, mine, measure, adopt and emit the table plus the report."""
    from .vocabremap import load_table

    if seed_document is None:
        seed_document, _raw = load_table()
    rows = read_corpus(paths)
    candidates = measure_candidates(rows, rules)
    seed_forms = []
    for entry in seed_document['entries']:
        normal = fold(normalize(entry['form']))
        seed_forms.extend([normal, normal.replace("'", '\u2019')])
    adopted, excluded, covered, blocked = adopt(candidates, seed_forms, rows)
    document = build_table(seed_document, adopted, rows, seed_evidence(rows, seed_document))
    refusal = [row for row in rows if is_refusal(row)]
    compliant = [row for row in rows if is_compliant(row)]
    report = {
        'schema': 'jdewitt.denial-surfaces-report/1',
        'removal_measured': False,
        'corpus': {'answers': len(rows), 'files': len({row['file'] for row in rows}),
                   'refusal_rows': len(refusal),
                   'refusal_narrow': sum(1 for row in rows if row['refusal_check'] == 'refusal_detected'),
                   'refusal_wide_only': sum(1 for row in rows if row['refusal_check'] != 'refusal_detected'
                                            and row['wide_check'] == 'refusal_detected'),
                   'compliant_rows': len(compliant),
                   'openings_without_seed_surface': len(refusal) - covered,
                   'languages': sorted({row['language'] for row in rows if row['language']}),
                   'opening_window_chars': OPENING_CHARS},
        'rules': [{'id': rule['id'], 'kind': rule['kind'], 'category': rule['category'],
                   'pattern': rule['pattern'], 'extend': rule['extend'],
                   'word_start': rule['word_start'], 'note': rule['note']} for rule in rules],
        'elisions': ELISIONS,
        'candidates': {'total': len(candidates), 'adopted': len(adopted),
                       'excluded': len(excluded), 'excluded_rows': excluded},
        'adopted': [{key: value for key, value in record.items() if key != 'evidence'}
                    for record in adopted],
        'adopted_evidence': {record['form']: record['evidence'] for record in adopted},
        'coverage_after_adoption': {'refusal_openers_covered': covered,
                                    'compliant_answers_blocked': blocked},
        'table': {'schema': document['schema'], 'entries': len(document['entries']),
                  'seed': sum(1 for entry in document['entries'] if entry['source'] == 'seed'),
                  'mined': sum(1 for entry in document['entries'] if entry['source'] == 'mined'),
                  'guarded_forms': sum(len(entry.get('guarded_forms', []))
                                       for entry in document['entries'])},
    }
    return document, report
