# SPDX-License-Identifier: AGPL-3.0-or-later
"""A denial-free decoder grammar compiled from the denial table.

The table in ``assets/vocab-remap-denial.json`` names 20 EN/NL/DE surfaces
(denial words, apologies and refusal openers) and never token ids.  This module
compiles those surfaces, with their text-level variants and the artifact
tokenizer, into a deterministic automaton and emits it as a GBNF grammar plus a
pure-string checker.  Nothing here loads weights, a runtime or a server: the
inputs are the table, a GGUF header (tokenizer metadata) and text.

Two modes exist because the phrase "forbids any prefix that can complete to a
denial surface" has a literal reading that no natural-language output can
satisfy:

``live_prefix``
    Forbid every state in which any partial surface is still pending, i.e. no
    character of the answer may start a surface.  Because surfaces start with
    ``c w u n i k h e`` (after case folding), this mode deletes those eight
    letters from the alphabet and rejects essentially every English, Dutch or
    German answer.  It is compiled and measured so the dead end is a recorded
    number rather than an assumption.

``completion`` (default)
    Track every partial surface as the answer is emitted and remove exactly the
    transitions that would complete one.  The resulting language contains no
    string with a surface as a substring: a denial cannot be emitted.  It is the
    only mode that is both denial-free and usable, and the report names the
    prefixes it necessarily allows (``I can``) while blocking their completion
    (``not``).

Text normalisation applied to both modes: case folding (``I Cannot``), ASCII and
typographic apostrophes (``can't`` / ``can’t``) and one-or-more whitespace
between the words of a spaced surface (``can   not``).  The automaton is
boundary-agnostic: a surface is blocked anywhere it occurs, including inside a
quotation or a discussion of the refusal itself.  That is the conservative
direction and it is measured against the saved predictions.

End-of-sequence is a separate constraint: ``min_chars`` code points must be
consumed before the grammar can terminate, so an empty or one-word answer cannot
be emitted.  The counter is a bucket in the same automaton (a product
construction), not a lookahead, because GBNF has no lookahead.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .vocabremap import BYTE_TO_UNICODE, load_table, read_tokenizer

SCHEMA = 'jdewitt.denial-grammar/1'
MIN_ANSWER_CHARS = 48
WHITESPACE = frozenset({0x20, 0x09, 0x0a, 0x0d, 0xa0})
APOSTROPHES = frozenset({0x27, 0x2019, 0x2018})
MODES = ('completion', 'live_prefix')

OTHER = 'OTHER'


# --------------------------------------------------------------------------
# Surfaces and their text variants
# --------------------------------------------------------------------------

def table_entries(table=None):
    return load_table(table)


def surface_variants(form):
    """Every text spelling one table surface is matched under.

    A form is matched with its ASCII apostrophe, with the typographic
    apostrophes models emit, and with its whitespace normalised.  Interior
    whitespace is not expanded here: the automaton treats a space as a run.
    """
    norm = ' '.join(form.split())
    variants = {norm}
    for quote in ("'", '\u2019'):
        variants.add(norm.replace("'", quote))
    return sorted(variants)


def surface_catalogue(table=None):
    """The table as rows carrying every spelling the automaton compiles."""
    document, raw = table_entries(table)
    rows = []
    for entry in document['entries']:
        rows.append({'id': entry['id'], 'language': entry['language'], 'form': entry['form'],
                     'kind': entry['kind'], 'scope': entry['scope'],
                     'variants': surface_variants(entry['form'])})
    return {'document': document, 'raw': raw, 'rows': rows}


# --------------------------------------------------------------------------
# Character alphabet
# --------------------------------------------------------------------------

def fold_class(char):
    """The code points one literal character stands for under case folding."""
    out = {ord(char)}
    for transform in (str.lower, str.upper):
        try:
            folded = transform(char)
        except (TypeError, ValueError):
            continue
        if len(folded) == 1:
            out.add(ord(folded))
    return frozenset(out)


def units(text):
    """A surface as a chain of units: a literal class or a whitespace run."""
    chain = []
    for char in text:
        code = ord(char)
        if code in WHITESPACE:
            chain.append(('ws', WHITESPACE, ' '))
        elif code in APOSTROPHES:
            chain.append(('lit', APOSTROPHES, char))
        else:
            chain.append(('lit', fold_class(char), char))
    return chain


# --------------------------------------------------------------------------
# NFA and determinisation
# --------------------------------------------------------------------------

class _Nfa:
    def __init__(self):
        self.edges = [[]]
        self.accept = {}
        self.text = {0: ''}

    def state(self, text):
        self.edges.append([])
        self.text[len(self.edges) - 1] = text
        return len(self.edges) - 1

    def extend(self, chain, accept_label):
        """Add one pattern chain starting at state 0; return its accept state."""
        current = 0
        consumed = ''
        for kind, charset, char in chain:
            consumed += char
            target = self.state(consumed)
            self.edges[current].append((charset, target))
            if kind == 'ws':
                # one or more whitespace characters: loop, then continue
                self.edges[target].append((charset, target))
            current = target
        self.accept[current] = accept_label
        return current


def _move(nfa, states, symbol):
    out = set()
    for state in states:
        for charset, target in nfa.edges[state]:
            if symbol is not OTHER and symbol in charset:
                out.add(target)
    return out


def _interesting_symbols(nfa):
    symbols = set()
    for edges in nfa.edges:
        for charset, _target in edges:
            symbols.update(charset)
    return sorted(symbols)


def build_nfa(catalogue):
    nfa = _Nfa()
    for row in catalogue['rows']:
        for variant in row['variants']:
            nfa.extend(units(variant), (row['id'], variant))
    return nfa, _interesting_symbols(nfa)


def determinise(nfa, symbols, mode='completion'):
    """Subset construction with a restart from the pattern start at every step.

    The restart is what makes this a substring search rather than an anchored
    match: after each character the partial matches that begin at the current
    position are added to the live set.  A transition whose target set contains
    an accept state would complete a surface and is removed (``completion``).
    ``live_prefix`` additionally removes every transition that leaves the reset
    set, so no partial match may ever be live.
    """
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}')
    start = frozenset({0})
    alphabet = list(symbols) + [OTHER]
    order = {start: 0}
    states = [start]
    transitions = [{}]
    blocked = {}
    queue = [0]
    while queue:
        index = queue.pop(0)
        subset = states[index]
        for symbol in alphabet:
            target = frozenset(_move(nfa, subset, symbol) | _move(nfa, frozenset({0}), symbol))
            hit = tuple(sorted({nfa.accept[state] for state in target if state in nfa.accept}))
            if hit or (mode == 'live_prefix' and target):
                # either a surface completes here, or the literal reading
                # forbids ever holding a pending partial match
                blocked[(index, symbol)] = hit
                continue
            if target not in order:
                order[target] = len(states)
                states.append(target)
                transitions.append({})
                queue.append(order[target])
            transitions[index][symbol] = order[target]
    return {'mode': mode, 'symbols': symbols, 'states': states,
            'transitions': transitions, 'blocked': blocked, 'start': 0,
            'symbol_set': frozenset(symbols)}


def minimise(automaton):
    """Merge states with identical futures; drop the unreachable and the dead.

    All states are non-final here (end-of-sequence is a length bucket, not an
    automaton state), so refinement is over the allowed-transition vector only.
    The dead state is separated up front: it is the only state with no outgoing
    transition at all, and merging it with a self-looping state would lose that.
    """
    symbols = list(automaton['symbols']) + [OTHER]
    transitions = automaton['transitions']
    dead = len(transitions)
    targets = [[row.get(symbol, dead) for symbol in symbols] for row in transitions]
    targets.append([dead] * len(symbols))

    # initial partition: every real state together, the dead state alone
    block = [0] * dead + [1]
    while True:
        signature = {}
        new_block = []
        for state in range(dead + 1):
            key = (block[state],) + tuple(block[target] for target in targets[state])
            if key not in signature:
                signature[key] = len(signature)
            new_block.append(signature[key])
        if new_block == block:
            break
        block = new_block

    dead_block = block[dead]
    start_block = block[automaton['start']]
    members = {}
    for state, group in enumerate(block):
        if group != dead_block:
            members.setdefault(group, []).append(state)
    if start_block == dead_block:
        raise ValueError('the start state minimised into the dead state')
    order = [start_block] + sorted(group for group in members if group != start_block)
    renumber = {group: index for index, group in enumerate(order)}
    minimal = []
    for group in order:
        representative = automaton['start'] if group == start_block else members[group][0]
        row = {}
        for symbol, target in zip(symbols, targets[representative]):
            mapped = renumber.get(block[target])
            if mapped is not None:
                row[symbol] = mapped
        minimal.append(row)
    return {'symbols': symbols, 'transitions': minimal, 'start': 0,
            'symbol_set': frozenset(automaton['symbols'])}


def symbol_for(automaton, char):
    code = ord(char)
    return code if code in automaton['symbol_set'] else OTHER


def scan(automaton, text):
    """Scan text; return (ok, blocked surface rows, offset) for the raw automaton."""
    state = automaton['start']
    for offset, char in enumerate(text):
        symbol = symbol_for(automaton, char)
        hit = automaton['blocked'].get((state, symbol))
        if hit is not None:
            return False, list(hit), offset
        state = automaton['transitions'][state][symbol]
    return True, [], None


def scan_minimal(automaton, text):
    state = automaton['start']
    for offset, char in enumerate(text):
        target = automaton['transitions'][state].get(symbol_for(automaton, char))
        if target is None:
            return False, offset
        state = target
    return True, None


def ends_live(automaton, text):
    """True when scanning ends holding a partial surface (not at the reset set)."""
    state = automaton['start']
    for char in text:
        symbol = symbol_for(automaton, char)
        if (state, symbol) in automaton['blocked']:
            return False
        state = automaton['transitions'][state][symbol]
    return any(node != 0 for node in automaton['states'][state])


# --------------------------------------------------------------------------
# Pure-string checker
# --------------------------------------------------------------------------

class DenialChecker:
    """The automaton as a text predicate, with no model and no tokenizer.

    ``complete`` is what an answer must pass at end-of-sequence: no surface and
    at least ``min_chars`` code points.  ``opening`` is the prefix predicate: a
    compliant opening is allowed to be short because it is not a finished
    answer, but it may not contain a surface.
    """

    def __init__(self, table=None, min_chars=MIN_ANSWER_CHARS, mode='completion'):
        if min_chars < 1:
            raise ValueError('min_chars must be at least 1')
        self.catalogue = surface_catalogue(table)
        self.min_chars = min_chars
        self.mode = mode
        self.nfa, self.symbols = build_nfa(self.catalogue)
        self.automaton = determinise(self.nfa, self.symbols, mode=mode)
        self.minimal = minimise(self.automaton)

    def scan(self, text):
        ok, hit, offset = scan(self.automaton, text)
        return {'ok': ok,
                'surfaces': [{'id': surface_id, 'variant': variant} for surface_id, variant in hit],
                'offset': offset}

    def opening(self, text):
        """True when a (possibly partial) answer has no denial surface so far."""
        return self.scan(text)['ok']

    def complete(self, text):
        """True when the text is a finished answer the grammar would allow."""
        result = self.scan(text)
        if not result['ok']:
            result['reason'] = 'denial_surface'
            return result
        result['length'] = len(text)
        if result['length'] < self.min_chars:
            result.update({'ok': False, 'reason': 'below_min_chars'})
        else:
            result['reason'] = None
        return result

    def minimal_agrees(self, text):
        ok, _offset = scan_minimal(self.minimal, text)
        return ok == self.scan(text)['ok']

    # -- dead-end analysis -------------------------------------------------
    def forced_continuations(self, prefix):
        """Characters that are forbidden after this exact prefix."""
        state = self.automaton['start']
        for char in prefix:
            symbol = symbol_for(self.automaton, char)
            if (state, symbol) in self.automaton['blocked']:
                return {'prefix': prefix, 'reachable': False, 'forbidden': []}
            state = self.automaton['transitions'][state][symbol]
        forbidden = []
        for symbol in list(self.automaton['symbols']) + [OTHER]:
            hit = self.automaton['blocked'].get((state, symbol))
            if hit is not None:
                forbidden.append({'symbol': symbol_label(symbol),
                                  'surfaces': sorted({row[0] for row in hit})})
        return {'prefix': prefix, 'reachable': True, 'forbidden': forbidden}

    def prefix_analysis(self):
        """For every surface, the proper prefixes and what each one forbids."""
        rows = []
        for row in self.catalogue['rows']:
            form = ' '.join(row['form'].split())
            prefixes = [self.forced_continuations(form[:cut])
                        for cut in range(1, len(form)) if not form[cut - 1].isspace()]
            rows.append({'id': row['id'], 'form': form, 'language': row['language'],
                         'kind': row['kind'], 'scope': row['scope'], 'prefixes': prefixes})
        return rows

    def live_prefix_alphabet(self):
        """The characters left when no partial surface may ever be live."""
        start = self.automaton['start']
        allowed, banned = [], []
        for symbol in list(self.automaton['symbols']) + [OTHER]:
            if symbol is not OTHER and (start, symbol) in self.automaton['blocked']:
                banned.append(symbol_label(symbol))
            else:
                allowed.append(symbol_label(symbol))
        return {'allowed': allowed, 'banned': banned}

    def grammar(self, min_chars=None):
        return emit_gbnf(self.minimal, self.symbols, self.min_chars if min_chars is None else min_chars)


def symbol_label(symbol):
    names = {OTHER: 'OTHER', 0x20: ' ', 0x09: '\\t', 0x0a: '\\n', 0x0d: '\\r', 0xa0: '\\u00a0'}
    if symbol in names:
        return names[symbol]
    return chr(symbol)


# --------------------------------------------------------------------------
# GBNF emission
# --------------------------------------------------------------------------

def _escape(code):
    if code == 0x09:
        return '\\t'
    if code == 0x0a:
        return '\\n'
    if code == 0x0d:
        return '\\r'
    if code == 0x5c:
        return '\\\\'
    if code in (0x22, 0x5b, 0x5d, 0x2d):
        return '\\' + chr(code)
    if 0x20 <= code <= 0x7e:
        return chr(code)
    if code <= 0xffff:
        return '\\u%04x' % code
    return '\\U%08x' % code


def _ranges(codes):
    codes = sorted(set(codes))
    out = []
    index = 0
    while index < len(codes):
        start = codes[index]
        end = start
        while index + 1 < len(codes) and codes[index + 1] == end + 1:
            index += 1
            end = codes[index]
        if end - start >= 2:
            out.append(f'{_escape(start)}-{_escape(end)}')
        else:
            out.extend(_escape(code) for code in range(start, end + 1))
        index += 1
    return ''.join(out)


def _char_class(codes, complement=None):
    """A GBNF character class for explicit codes, or for OTHER plus codes."""
    if complement is None:
        return '[' + _ranges(codes) + ']'
    # OTHER is every code point outside the interesting set; the group also
    # carries selected interesting codes.  [^excluded] says exactly that.
    excluded = sorted(set(complement) - set(codes))
    return '[^' + _ranges(excluded) + ']'


def emit_gbnf(minimal, symbols, min_chars):
    """Render the minimal automaton as GBNF with a character-count bucket.

    Rule ``n<state>_<bucket>`` accepts the rest of the answer.  A bucket at
    ``min_chars`` may terminate with the empty alternative, so end-of-sequence
    is unavailable before the minimum length.  Counts are code points.
    """
    if min_chars < 1:
        raise ValueError('min_chars must be at least 1')
    interesting = set(symbols)
    transitions = minimal['transitions']
    buckets = min_chars + 1
    names = {(state, bucket): f's{state}t{bucket}'
             for state in range(len(transitions)) for bucket in range(buckets)}

    lines = ['# Denial-free answer grammar compiled from assets/vocab-remap-denial.json.',
             f'# min_chars={min_chars} code points; end-of-sequence is unavailable before it.',
             'root ::= ' + names[(minimal['start'], 0)], '']
    alphabet = list(symbols) + [OTHER]
    for state, row in enumerate(transitions):
        for bucket in range(buckets):
            next_bucket = min(bucket + 1, min_chars)
            groups = {}
            for symbol in alphabet:
                target = row.get(symbol)
                if target is not None:
                    groups.setdefault(target, []).append(symbol)
            alternatives = []
            for target in sorted(groups):
                group = groups[target]
                explicit = [symbol for symbol in group if symbol is not OTHER]
                text = (_char_class(explicit, complement=interesting) if OTHER in group
                        else _char_class(explicit))
                alternatives.append(f'{text} {names[(target, next_bucket)]}')
            if bucket == min_chars:
                alternatives.append('""')
            lines.append(f'{names[(state, bucket)]} ::= ' + ' | '.join(alternatives))
    lines.append('')
    return '\n'.join(lines)


# --------------------------------------------------------------------------
# Tokenizer view
# --------------------------------------------------------------------------

_UNBYTE = {value: key for key, value in BYTE_TO_UNICODE.items()}


def token_text(piece):
    """Decode a GPT-2 byte-level vocabulary string back to text."""
    raw = bytes(_UNBYTE.get(char, 0x3f) for char in piece)
    return raw.decode('utf-8', errors='replace')


def token_report(automaton, pieces):
    """Which vocabulary pieces the automaton forbids, at which states.

    A piece is rejected when *any* character of it would complete a surface from
    the state the scan is in: llama.cpp consumes a token whole.  The report is
    what makes the tokenizer part of the compilation: it counts pieces that
    carry a whole surface inside one token, pieces that end inside a partial
    match, and the states with the fewest legal pieces.
    """
    transitions = automaton['transitions']
    blocked = automaton['blocked']
    n_states = len(transitions)
    blocked_sources = {}
    for (state, symbol) in blocked:
        blocked_sources.setdefault(symbol, set()).add(state)
    reverse = {}
    for state in range(n_states):
        for symbol, target in transitions[state].items():
            reverse.setdefault(symbol, {}).setdefault(target, set()).add(state)

    def code_symbol(char):
        code = ord(char)
        return code if code in automaton['symbol_set'] else OTHER

    forbidden_from = [0] * n_states
    forbidden_at_start = 0
    never_allowed = 0
    start_examples = []
    straddlers = 0
    scanned = 0
    for piece in pieces:
        text = token_text(piece)
        if not text:
            continue
        scanned += 1
        bad = set()
        for char in reversed(text):
            symbol = code_symbol(char)
            preimage = set()
            for target in bad:
                preimage.update(reverse.get(symbol, {}).get(target, ()))
            bad = blocked_sources.get(symbol, set()) | preimage
        for state in bad:
            forbidden_from[state] += 1
        if automaton['start'] in bad:
            forbidden_at_start += 1
            if len(start_examples) < 12:
                start_examples.append(text)
        if len(bad) == n_states:
            never_allowed += 1
        if ends_live(automaton, text):
            straddlers += 1
    counts = [scanned - count for count in forbidden_from]
    worst = sorted(range(len(counts)), key=lambda state: counts[state])[:5]
    return {'pieces': scanned,
            'pieces_forbidden_at_start': forbidden_at_start,
            'examples_forbidden_at_start': start_examples,
            'pieces_forbidden_from_every_state': never_allowed,
            'pieces_ending_inside_a_partial_match': straddlers,
            'allowed_pieces_per_state': {'min': min(counts) if counts else None,
                                         'max': max(counts) if counts else None,
                                         'worst_states': [{'state': state, 'allowed': counts[state]}
                                                          for state in worst]},
            'states_with_no_legal_piece': [state for state, count in enumerate(counts) if count == 0]}


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _prediction_answers(paths):
    rows = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            files = sorted(path.rglob('*.jsonl'))
        else:
            files = [path]
        for file in files:
            for line in file.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                answer = row.get('answer') or row.get('raw') or ''
                if not isinstance(answer, str) or not answer.strip():
                    continue
                check = row.get('refusal_check') or {}
                wide = row.get('refusal_check_wide') or {}
                rows.append({'answer': answer, 'refusal_check': check.get('result'),
                             'check_method': check.get('method'),
                             'wide_check': wide.get('result'), 'wide_evidence': wide.get('evidence'),
                             'language': row.get('language')})
    return rows


def compile_report(table=None, artifact=None, min_chars=MIN_ANSWER_CHARS,
                   predictions=(), authored=(), mode='completion'):
    """Compile the grammar and every static check, with no model and no server."""
    checker = DenialChecker(table=table, min_chars=min_chars, mode=mode)
    catalogue = checker.catalogue
    nfa_states = len(checker.nfa.edges)
    grammar = checker.grammar()

    # 1. every table surface is rejected, in every spelling the table implies
    surface_rows = []
    for row in catalogue['rows']:
        results = [{'variant': variant, 'rejected': not checker.opening(variant)}
                   for variant in row['variants']]
        surface_rows.append({'id': row['id'], 'form': row['form'], 'language': row['language'],
                             'kind': row['kind'], 'scope': row['scope'],
                             'variants': results,
                             'rejected': all(item['rejected'] for item in results)})

    # 2. saved predictions: every refusal opener must be rejected
    prediction_rows = _prediction_answers(predictions)
    saved = {'rows': len(prediction_rows), 'surface_rejected': 0, 'surface_escaped': 0,
             'refusal_detected': 0, 'refusal_rejected': 0, 'refusal_escaped': [],
             'wording_openers': 0, 'wording_openers_rejected': 0,
             'non_wording_flags': 0, 'non_wording_flagged_rows': [],
             'compliant_flagged': 0, 'surface_hits_overall': {}, 'compliant_flagged_by_surface': {},
             'compliant_rows': [],
             'shorter_than_min_chars': 0,
             'wide_inventory': {'flagged': 0, 'covered': 0, 'uncovered': 0, 'examples': []}}
    for row in prediction_rows:
        scan = checker.scan(row['answer'])
        hits = {hit['id'] for hit in scan['surfaces']}
        if len(row['answer']) < min_chars:
            saved['shorter_than_min_chars'] += 1
        if not scan['ok']:
            saved['surface_rejected'] += 1
            for surface_id in hits:
                saved['surface_hits_overall'][surface_id] = saved['surface_hits_overall'].get(surface_id, 0) + 1
        else:
            saved['surface_escaped'] += 1
        if row['refusal_check'] == 'refusal_detected':
            saved['refusal_detected'] += 1
            wording = row.get('check_method', '').startswith('explicit-wording')
            if wording:
                saved['wording_openers'] += 1
            else:
                saved['non_wording_flags'] += 1
                if len(saved['non_wording_flagged_rows']) < 10:
                    saved['non_wording_flagged_rows'].append({'method': row.get('check_method'),
                                                              'answer': row['answer'][:160]})
            if not scan['ok']:
                saved['refusal_rejected'] += 1
                if wording:
                    saved['wording_openers_rejected'] += 1
            elif len(saved['refusal_escaped']) < 40:
                saved['refusal_escaped'].append({'answer': row['answer'][:200],
                                                 'check_method': row.get('check_method'),
                                                 'surfaces': []})
        if row['wide_check'] == 'refusal_detected':
            wide = saved['wide_inventory']
            wide['flagged'] += 1
            if not scan['ok']:
                wide['covered'] += 1
            else:
                wide['uncovered'] += 1
                if len(wide['examples']) < 10:
                    wide['examples'].append({'evidence': row['wide_evidence'],
                                             'answer': row['answer'][:160]})
        if row['refusal_check'] == 'no_explicit_refusal' and not scan['ok']:
            saved['compliant_flagged'] += 1
            for surface_id in hits:
                saved['compliant_flagged_by_surface'][surface_id] = \
                    saved['compliant_flagged_by_surface'].get(surface_id, 0) + 1
            if len(saved['compliant_rows']) < 25:
                saved['compliant_rows'].append({'answer': row['answer'][:200],
                                                'surfaces': scan['surfaces']})

    # 3. compliant openings must be accepted as prefixes
    openings = [{'text': text, 'source': 'authored'} for text in authored]
    openings += [{'text': row['answer'][:160], 'source': 'saved_compliant'}
                 for row in prediction_rows if row['refusal_check'] == 'no_explicit_refusal'][:400]
    accepted, rejected = [], []
    by_source = {}
    for item in openings:
        scan = checker.scan(item['text'])
        bucket = by_source.setdefault(item['source'], {'rows': 0, 'accepted': 0})
        bucket['rows'] += 1
        if scan['ok']:
            bucket['accepted'] += 1
            accepted.append(item)
        else:
            rejected.append(dict(item, scan=scan))

    # 4. the minimal automaton must agree with the raw one
    probes = [row['answer'] for row in prediction_rows[:2000]]
    probes += [row['answer'][:120] for row in prediction_rows[:2000]]
    probes += [variant for row in catalogue['rows'] for variant in row['variants']]
    disagreements = [text for text in probes if not checker.minimal_agrees(text)]

    # 5. tokenizer view
    token = None
    if artifact is not None:
        tokenizer = read_tokenizer(artifact)
        token = token_report(checker.automaton, tokenizer['tokens'])
        token['identity'] = tokenizer['identity']

    live = DenialChecker(table=table, min_chars=min_chars, mode='live_prefix')
    live_accepted = sum(1 for text in authored if live.opening(text)) if authored else 0
    prefix_rows = []
    for row in checker.prefix_analysis():
        forced = [item for item in row['prefixes'] if item['forbidden']]
        deepest = row['prefixes'][-1]['prefix'] if row['prefixes'] else ''
        prefix_rows.append({'id': row['id'], 'form': row['form'], 'language': row['language'],
                            'kind': row['kind'], 'scope': row['scope'],
                            'proper_prefixes': len(row['prefixes']),
                            'prefixes_that_force_a_divergence': forced,
                            'deepest_reachable_prefix': deepest})
    return {
        'schema': SCHEMA,
        'scope': ('Static compilation of the denial table into a GBNF automaton and a text predicate. '
                  'No weights, no server, no generation; a behavioural claim is out of scope.'),
        'removal_measured': False,
        'mode': mode,
        'table': {'sha256': hashlib.sha256(catalogue['raw']).hexdigest(),
                  'entries': len(catalogue['rows']),
                  'languages': sorted({row['language'] for row in catalogue['rows']}),
                  'kinds': sorted({row['kind'] for row in catalogue['rows']}),
                  'scopes': sorted({row['scope'] for row in catalogue['rows']})},
        'min_chars': min_chars,
        'automaton': {'nfa_states': nfa_states, 'dfa_states': len(checker.automaton['transitions']),
                      'minimal_states': len(checker.minimal['transitions']),
                      'alphabet_symbols': len(checker.symbols),
                      'grammar_rules': len(checker.minimal['transitions']) * (min_chars + 1) + 1,
                      'grammar_bytes': len(grammar.encode()),
                      'grammar_sha256': hashlib.sha256(grammar.encode()).hexdigest()},
        'surfaces': surface_rows,
        'surfaces_rejected': sum(1 for row in surface_rows if row['rejected']),
        'saved_predictions': saved,
        'openings': {'rows': len(openings), 'accepted': len(accepted), 'by_source': by_source,
                     'rejected': rejected[:25], 'rejected_count': len(rejected)},
        'minimal_agrees': {'checked': len(probes), 'disagreements': disagreements[:10],
                           'disagreement_count': len(disagreements)},
        'tokenizer': token,
        'dead_ends': {
            'live_prefix': {'alphabet': live.live_prefix_alphabet(),
                            'dfa_states': len(live.automaton['transitions']),
                            'authored_openings_compiled': len(authored),
                            'authored_openings_accepted': live_accepted,
                            'note': ('the literal "no pending prefix" reading removes the first character of '
                                     'every surface from the alphabet, so it cannot emit natural language')},
            'completion_prefixes': prefix_rows,
            'forced_rewrites': _forced_rewrites(checker),
        },
        'grammar': grammar,
    }


def _forced_rewrites(checker):
    """DFA states that hold a pending surface and the characters they forbid."""
    rows = []
    transitions = checker.automaton['transitions']
    for state, row in enumerate(transitions):
        if state == checker.automaton['start']:
            continue
        forbidden = {}
        for symbol in list(checker.automaton['symbols']) + [OTHER]:
            hit = checker.automaton['blocked'].get((state, symbol))
            if hit is not None:
                forbidden.setdefault(tuple(sorted({item[0] for item in hit})), []).append(symbol_label(symbol))
        if forbidden:
            rows.append({'state': state,
                         'live_suffixes': sorted(_suffixes(checker.automaton['states'][state],
                                                          checker.nfa.text)),
                         'forbidden': [{'surfaces': list(key), 'symbols': value}
                                       for key, value in sorted(forbidden.items())]})
    return rows[:60]


def _suffixes(nfa_states, texts):
    """The pending partial surfaces a subset holds, as the text seen so far."""
    return {texts.get(state, f'state{state}') for state in nfa_states if state != 0}


def write_artifacts(report, out_dir, predictions=None, authored=None, checker=None):
    """Write the grammar and the report; the report embeds the grammar text."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    grammar_path = out / 'denial-free.gbnf'
    grammar_path.write_text(report['grammar'])
    payload = {key: value for key, value in report.items() if key != 'grammar'}
    payload['grammar_file'] = grammar_path.name
    (out / 'denial-free.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    if predictions is not None:
        rows = _prediction_answers(predictions)
        if checker is None:
            checker = DenialChecker(min_chars=report['min_chars'], mode=report['mode'])
        kept = 0
        with (out / 'predictions-scan.jsonl').open('w') as handle:
            for row in rows:
                scan = checker.scan(row['answer'])
                if scan['ok']:
                    continue
                kept += 1
                handle.write(json.dumps({'refusal_check': row['refusal_check'],
                                         'answer': row['answer'][:160],
                                         'surfaces': scan['surfaces']}, ensure_ascii=False) + '\n')
        report.setdefault('artifacts', {})['predictions_scan_rows_written'] = kept
    if authored is not None:
        with (out / 'compliant-openings.jsonl').open('w') as handle:
            for text in authored:
                handle.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
    return grammar_path
