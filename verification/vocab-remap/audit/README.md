# The wording-vs-content audit

`verification/vocab-remap/audit.json` and `audit.txt` are one run of
`scripts/vocab_remap_audit.py` over the decode wave in
`verification/vocab-remap/decode/` (`off` = no remap, `on` = remap). The script is
the instrument; the two files are its output, and the printed table is the second of
them.

Reproduce (one command, one JSON):

    python scripts/vocab_remap_audit.py \
        --artifact "$ARTIFACT" \
        --serve "$LLAMA_SERVER"

`$ARTIFACT` is the GGUF the decode wave used and `$LLAMA_SERVER` is a runnable
llama-server. The command regenerates the real capability control under `control/`,
re-reads both arms, prints the side-by-side table and writes `../audit.json` plus
`../audit.txt`. `--reuse-control` reads the existing control row instead of decoding
it (inspection only; the generation is the evidence).

## What it reports

Per row: the answer, its length, the word-check result (`jdewitt.refusal_check`), the
shape label plus opener/decline/procedural/embedded_decline
(`jdewitt.refusal_shape.classify`), and one verdict — `content_effect`,
`wording_effect` or `no_change`. The aggregate prints the denial-word rate and the
content rate per arm, side by side, never averaged, plus the false-clean count (the
word check says clean while the shape declines).

Two rows are controls, labelled and never counted as arm results: a seeded compliant
continuation per wave (`fixture: seeded_contrast`) and one real maths row decoded
through the same stepper with `remap=on` (`real_capability_control`). The real
control exists so the content branch is proved on traffic, not only on a fixture.
Content on a benign row is not evidence about a harmful row.

## What it does not say

This proves the instrument separates the two axes — wording and content — not that
the method removes anything. On this wave the table-named denial surface is gone
from every row that carried one, the shape still declines in 6 of 10, the word check
still detects 6 of 10, and no real row gained the requested content
(`wording_effect`). The word check normalizes the typographic apostrophe, so the
U+2019 variant that defeats the table does not fool the check: the table gap shows up
as "no table-named surface", not as a false clean.

The `control/server.log` is regenerable, carries the loader's absolute argv and is
git-ignored; `rows.jsonl` and `run.json` are portable and committed.
