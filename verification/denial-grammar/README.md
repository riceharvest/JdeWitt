# denial-grammar

Static evidence for the denial-free decoder grammar: the 20-surface EN/NL/DE
table compiled into a GBNF automaton, plus a pure-string checker. No model, no
server, no generation, no GPU. `removal_measured` is `false`.

Recompute (from the repo root):

    python scripts/denial_grammar_check.py --out verification/denial-grammar \
        --artifact verification/iq05b/Qwen2.5-0.5B-Instruct-IQ3_M.gguf

or, the same implementation through the CLI:

    jdewitt grammar --compile --artifact verification/iq05b/Qwen2.5-0.5B-Instruct-IQ3_M.gguf

The llama.cpp parse/match check needs a source tree and a built `libllama.so`;
`scripts/gbnf_probe.cpp` is compiled against them on the fly and the check is
reported as unavailable (not passed) when they are missing. It parses and
matches only: no model is loaded.

| file | what it is |
|---|---|
| `denial-free.gbnf` | the compiled grammar (5,048 rules, 862,128 bytes at min_chars=48) |
| `denial-free.json` | report: surfaces, automaton counts, checks, dead-end analysis, token view |
| `predictions-scan.jsonl` | every saved answer the automaton rejects, with the surface that fired |
| `compliant-openings.jsonl` | the 30 authored compliant openings (EN/NL/DE) |
| `public-pair.json` | base/twin repo, revision, sha256, tokenizer semantics |

Headline numbers are in `docs/DENIAL-GRAMMAR.md` and in
`denial-free.json`. Read them as a text-level result: the grammar rejects all 20
surfaces and every explicit-wording refusal opener in the saved predictions, and
llama.cpp's matcher agrees with the Python automaton on 1,344 cases. Nothing here
says what a model does when the constraint is active.
