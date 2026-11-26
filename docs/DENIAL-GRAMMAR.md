# Denial-free decoder grammar

A compiled decoder constraint that cannot emit a denial surface. The input is
`jdewitt/assets/vocab-remap-denial.json` (20 EN/NL/DE surfaces) plus a GGUF
tokenizer header; the output is a GBNF automaton and a pure-string checker. No
model forward, no server, no generation, no GPU: `removal_measured` is `false` in
every report here, and nothing in this document is a behavioural claim.

Artifact: `verification/denial-grammar/` (grammar, report, prediction scan,
compliant openings, public-pair provenance).
Recompute: `python scripts/denial_grammar_check.py --out verification/denial-grammar --artifact <gguf>`
or `jdewitt grammar --compile --artifact <gguf>`.

## What is compiled

1. Each table surface becomes a regular pattern. A literal character is compiled
   as its case-folded class, `'` also matches `’`/`‘`, and a space inside a
   surface is a one-or-more whitespace run, so `I Cannot`, `can’t` and
   `can   not` are the same denial to the automaton as the table spelling.
2. The patterns become an NFA (216 states), then a DFA by subset construction
   (150 states) with a restart from the pattern start at every character, which
   is what makes it a substring search rather than an anchored match.
3. A transition whose target set contains a pattern's accept state is **removed**.
   That is the whole constraint: the language contains no string with a surface
   as a substring. The DFA is then minimised to 103 states over 54 character
   symbols.
4. End-of-sequence is a product construction with a length bucket: 48 code points
   must be consumed before the empty alternative is reachable. GBNF has no
   lookahead, so the floor cannot be expressed any other way.

At `min_chars=48` that is 5,048 rules and 862,128 bytes. llama.cpp's parser
reads it in about 0.1 s (`PARSE_OK rules=5048`).

## The literal reading is a dead end, and it is measured

"forbids any prefix that can complete to a denial surface" has a strict reading:
no pending partial surface may ever be live. Compiled, it removes the first
character of every surface from the alphabet — `c w u n i k h e` and their
capitals, 8 letters — and the automaton collapses to 2 states. Every English,
Dutch and German sentence contains at least one of those letters: **0 of the 30
authored compliant openings survive**. That mode (`live_prefix`) is in the
module and in the report as a recorded dead end, not as a candidate.

The usable mode (`completion`, the default) allows those prefixes and removes
the completing character. For every one of the 20 surfaces the deepest reachable
prefix is the surface minus its last character, and only the last character (in
both cases) is blocked:

| surface id | form | proper prefixes | deepest reachable prefix | blocked next |
|---|---|---:|---|---|
| en-cannot | cannot | 5 | canno | T,t |
| en-can-not | can not | 5 | can no | T,t |
| en-cant | can't | 4 | can' | T,t |
| en-wont | won't | 4 | won' | T,t |
| en-will-not | will not | 6 | will no | T,t |
| en-unable | unable | 5 | unabl | E,e |
| en-not-able | not able | 6 | not abl | E,e |
| en-im-sorry | I'm sorry | 7 | I'm sorr | Y,y |
| en-i-am-sorry | I am sorry | 7 | I am sorr | Y,y |
| en-i-apologize | I apologize | 9 | I apologiz | E,e |
| nl-kan-niet | kan niet | 6 | kan nie | T,t |
| nl-kan-ik-niet | kan ik niet | 8 | kan ik nie | T,t |
| nl-niet-mogelijk | niet mogelijk | 11 | niet mogelij | K,k |
| nl-het-spijt-me | het spijt me | 9 | het spijt m | E,e |
| nl-ik-kan-niet | ik kan niet | 8 | ik kan nie | T,t |
| de-kann-nicht | kann nicht | 8 | kann nich | T,t |
| de-kann-ich-nicht | kann ich nicht | 11 | kann ich nich | T,t |
| de-nicht-moeglich | nicht möglich | 11 | nicht möglic | H,h |
| de-es-tut-mir-leid | es tut mir leid | 11 | es tut mir lei | D,d |
| de-ich-kann-nicht | ich kann nicht | 11 | ich kann nich | T,t |

There is no unsatisfiable state in `completion` mode: every state keeps at least
149,972 of the 151,936 vocabulary pieces legal. The cost is paid at the token
boundary instead. llama.cpp consumes a token whole, so a piece is rejected when
any of its characters would complete a surface from the current state:

* 9 pieces carry a whole surface and are forbidden from every state
  (` cannot`, ` unable`, `Unable`, `Cannot`, `cannot`, ` Cannot`, ` Unable`,
  `unable`, `_CANNOT`);
* 32,508 pieces (21.4% of the vocabulary) end inside a pending partial match —
  ` can`, ` canno`, `not`, and even ` the` (ends inside `het spijt me`). A
  whole-token blocklist cannot see any of those; the character automaton can.
* no state leaves the decoder without a legal piece.

## Static checks

| check | result |
|---|---|
| table surfaces rejected, every spelling | 20/20 |
| authored compliant openings accepted as prefixes | 30/30 |
| saved compliant openings accepted (from `no_explicit_refusal` rows) | 318/400 |
| saved answers scanned (134 prediction files) | 17,847 |
| saved answers rejected | 3,622 |
| explicit-wording refusal openers rejected | 1,880/1,880 |
| wide-inventory refusals covered | 1,577/1,586 |
| minimal DFA vs full DFA, 4,023 strings | 0 disagreements |
| independent substring oracle, 1,200 answers | 0 disagreements |
| llama.cpp parser | `PARSE_OK rules=5048` |
| llama.cpp matcher vs the Python automaton | 1,344 cases, 0 mismatches |

The three `refusal_detected` rows without a table surface are not openers: their
`refusal_check.method` is `degenerate-repetitive_filler` and their evidence list
is empty. The nine wide-inventory misses are refusal shapes the 20-surface table
does not name (`I refuse to`, `I don't have personal preferences`,
`I'm really sorry`, `as an AI, I do not`). The grammar can only be as complete as
the table: extending the table and recompiling extends the grammar.

## What it costs, stated plainly

The automaton is boundary-agnostic: it rejects a surface anywhere, including
inside a quotation or a sentence that discusses refusals. Of 5,550 rows the
narrow refusal check called non-refusals, 1,007 (18.1%) contain a table surface
and would be barred from finishing that sentence. Most are denials of a shape the
narrow check does not match (`I cannot and will not provide...`); some are
ordinary uses of the same words (`it will not run`, `I can not only ... but
also`). A decoder constraint is not a compliance classifier: it is a hard
language restriction, and any measure of how much compliant text it changes has
to be made against a task's own answers, not against this scan.

The floor (`min_chars=48`) is a policy dial, not a property of the table. It is
counted in code points. It rules out a terse "No." or a one-word answer; on the
saved predictions it would block end-of-sequence for 4,256 of 17,847 answers
(23.8%), which includes the short-answer capability and classification rows.
Use it only on answer-shaped generation, not on classification.

## Limits

* Text level only. The grammar was parsed and matched by llama.cpp's own parser
  and matcher, but no generation was run with it; a claim about what a model does
  under this constraint is out of scope for this experiment.
* Case folding, apostrophes and whitespace runs are covered. An elided
  apostrophe (`Im sorry`, `cant`) is not a table spelling and is not matched.
* `scope=opener_window` entries (apologies and refusal openers) are forbidden at
  any position, not only in the opener window: a character automaton would need a
  second position counter to honour the scope, and the stricter reading is the one
  that cannot emit a denial.
* The grammar is 862 KB at 48 code points because the end-of-sequence floor is a
  product with the automaton. A smaller floor makes a smaller file; there is no
  lookahead that would let the two constraints share states.
* `scripts/results_index.py` picks these files up on its next generation; the
  committed index in this branch predates them.

## Files

| file | what it is |
|---|---|
| `verification/denial-grammar/denial-free.gbnf` | the compiled grammar |
| `verification/denial-grammar/denial-free.json` | surfaces, automaton counts, checks, dead ends, token view |
| `verification/denial-grammar/predictions-scan.jsonl` | every saved answer the automaton rejects, with the surface that fired |
| `verification/denial-grammar/compliant-openings.jsonl` | the 30 authored openings used as the acceptance corpus |
| `verification/denial-grammar/public-pair.json` | base/twin repo, revision, sha256, tokenizer semantics |
