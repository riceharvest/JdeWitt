# denial-grammar

Static evidence for the denial-free decoder grammar: the 20-surface EN/NL/DE seed
table grown to 27 surfaces from the saved answers and compiled into a GBNF
automaton, plus a pure-string checker. No model, no server, no generation, no
GPU. `removal_measured` is `false`.

Recompute (from the repo root):

    python scripts/denial_grammar_check.py --out verification/denial-grammar --mine \
        --artifact verification/iq05b/Qwen2.5-0.5B-Instruct-IQ3_M.gguf

`--mine` grows the table from `verification/**/predictions.jsonl` first and writes
`denial-surfaces.json` + `denial-surfaces-mining.json`. Without it the seed table
compiles to the previous artifact (5,048 rules, 862,128 bytes).

Or, the same implementation through the CLI:

    jdewitt grammar --compile --mine --artifact verification/iq05b/Qwen2.5-0.5B-Instruct-IQ3_M.gguf

The llama.cpp parse/match check needs a source tree and a built `libllama.so`;
`scripts/gbnf_probe.cpp` is compiled against them on the fly and the check is
reported as unavailable (not passed) when they are missing. It parses and
matches only: no model is loaded.

| file | what it is |
|---|---|
| `denial-free.gbnf` | the compiled grammar (10,928 rules, 2,210,112 bytes at min_chars=48) |
| `denial-free.json` | report: surfaces, automaton counts, checks, before/after delta, trade-off curve, dead ends, token view |
| `denial-surfaces.json` | the extended table: 27 rows, each with its rule and the saved row ids that attest it |
| `denial-surfaces-mining.json` | the rule set, all 79 attested candidates, each with gain/cost and the adoption decision |
| `predictions-scan.jsonl` | every saved answer the automaton rejects, with the surface that fired |
| `compliant-openings.jsonl` | the 30 authored compliant openings (EN/NL/DE) |
| `public-pair.json` | base/twin repo, revision, sha256, tokenizer semantics |

Headline numbers are in `docs/DENIAL-GRAMMAR.md` and in `denial-free.json`. Read
them as a text-level result: the grammar rejects all 27 surfaces (including the 8
guarded elided spellings), 1,931 of 2,035 refusal-labelled openings in the saved
predictions, every explicit-wording refusal opener (1,880/1,880), and llama.cpp's
matcher agrees with the Python automaton on 1,390 cases. Nothing here says what a
model does when the constraint is active.
