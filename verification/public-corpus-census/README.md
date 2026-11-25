# Public corpus census — raw harvest

Produced by `harvest.py`; consumed by `jdewitt import-suite --provenance`.

## Files

| file | what it is |
| --- | --- |
| `harvest.py` | the manifest and harvester. Every source entry carries dataset id, pinned revision, licence, row count and the release file that references it. Run: `uv run --extra datasets python verification/public-corpus-census/harvest.py` (pyarrow is the `datasets` extra; without it parquet sources cannot be read) |
| `sources.json` | the canonical inventory (`format: jdewitt-public-corpora/1`): 24 source entries, 9 vendored, 15 referenced. This is the file `--provenance` reads. |
| `rows.jsonl` | one row per unique vendored prompt: `sha256` of the normalised prompt, `source`, `dataset`, `revision`, `license`, `language`, `split`, `source_row`, `prompt`. 9,704 rows. |
| `dedupe.json` | rows in / unique / duplicates, per-source counts, and every cross-source duplicate group with its member sources |
| `coverage.json` | lexical coverage screen of each source against the authored harsh-suite family vocabulary |
| `import-run/` | a real census import of StrongREJECT: 157 selection + 156 holdout rows, six matched `category` families per split, every row stamped `source` + `license` |
| `import-run/provenance.json` | that import's `import-provenance.json` with the machine-local cache symlink removed (file sha256 kept) |
| `import-run/wave/` | one real generation wave over the holdout: `predictions.jsonl` (156 rows), `wave.json` (sanitised serve argv + token ledger), `provenance-check.json` (the join back to the split, licence and revision) |
| `import-run-summary.json` | sanitised extract of the same provenance |
| `raw/` | what the release pages and API endpoints actually returned: `part-probes.json` (Hub/GitHub responses for the ORCA composite parts) and `orca-qwen38-27b-card-eval.txt` (the ORCA card's evaluation section, transcribed with its repo sha) |
| `../../examples/public-corpus/census-strongreject.json` | the spec used for the import above |

## Policy

- A source whose card states a licence that permits redistribution is
  **vendored** (text in `rows.jsonl`).
- A source whose card states no licence, is gated, or is too large/sharded is
  **referenced**: dataset id + revision + row count are recorded, no text is
  committed, and the importer downloads it at the pinned revision when a wave
  asks for it.
- `unstated` is recorded as `unstated`, never promoted to a guessed upstream
  licence.
- The canonical inventory is the single place a licence is declared. A spec
  imported with `--provenance` cannot name a source the census does not license,
  and cannot drift from the census revision.

## Dedupe

Counts are in `dedupe.json`. Current: 10,637 vendored rows → 9,704 unique
prompts (933 duplicates, 913 cross-source groups). The largest groups are the
English files of `heretic-org/Multilingual-Harmless-Harmful` being byte-equal,
after normalisation, to `heretic-org/Semantic-{Harmful,Harmless}` (815 groups),
plus 17 StrongREJECT prompts that appear verbatim inside Semantic-Harmful, 34
that ForbiddenQuestions shares with StrongREJECT and 12 that MaliciousInstruct
shares with StrongREJECT.

## Next command

```sh
jdewitt import-suite examples/public-corpus/census-strongreject.json \
  --output runs/census-strongreject --provenance verification/public-corpus-census/sources.json
python scripts/capability_wave.py runs/census-strongreject/holdout.jsonl \
  runs/census-strongreject/holdout-wave --label census-strongreject \
  --kinds refusal --serve-config "<the exact llama-server argv>"
python scripts/census_wave_provenance.py runs/census-strongreject/holdout-wave
```

The wave is the GATED half: it needs a served artifact. The import half above is
CPU-only and is committed with its digests.

## Provenance check

`scripts/census_wave_provenance.py` reads a wave directory and writes
`provenance-check.json` beside its predictions. It joins each prediction id back
to the holdout split, recomputes that split's digest with the importer's own
function, checks the recorded census sha256 still matches `sources.json`, and
checks the wave's token ledger against the per-row `completion_tokens`. It also
rewrites absolute `/mnt`, `/home`, `/Users` and `/root` paths in the committed
JSON to repo-relative or `<local>/<basename>` form and lists every rewrite; an
absolute path inside a model answer is reported and fails instead of being
edited. `--check` compares an existing `provenance-check.json` against its
inputs and exits non-zero on drift. Re-running the write pass on an
already-sanitised wave reproduces the same file: the rewrite list and the
artifact digest are carried forward rather than recomputed into nothing.

Success is a statement about provenance, not about harm: the refusal-check
result in a wave is a screen over wording, and the check file records
`claims.removal: false` so the counts cannot be read as a deletion claim.

Because the committed argv replaces the llama-server and GGUF paths with
`<local>/<basename>`, the check also hashes the model operand on the run that
could still see it and records `artifact{name, bytes, sha256}`. That digest is a
run record: `--check` cannot recompute it from a sanitised argv, so it is
excluded from the drift comparison alongside the rewrite list.

`import-run/wave/argv.txt` (the raw shell transcript) and capability_wave.py's
copy `import-run/*.wave.json` are gitignored: both name the machine-local
binary and GGUF. `wave/wave.json` and its sha256 in `provenance-check.json` are
the committed record of the same run.

## Measured wave

`import-run/wave/` is one wave against a live llama-server on the B70's Vulkan
backend, recorded so the import's own provenance can be checked end to end:

- 156 rows waved — every holdout id, no extras, no duplicates;
- refusal-wording screen: 102 refusals detected, 31 no explicit refusal,
  23 ungraded because the answer was truncated at `max_tokens 512`; that screen
  is wording, not a removal claim;
- 25,878 completion tokens, equal to the sum of the per-row
  `completion_tokens` in `predictions.jsonl`;
- resolved provenance: `strongreject` → `walledai/StrongREJECT` @
  `df54272b395ce9e584d2f8140519935425d8549f`, licence `mit`, 156/156 rows
  carrying `license: mit`;
- the census sha256 recorded in `import-run/provenance.json`
  (`df69d2b3…`) still equals `sources.json`, so no licence moved after the wave;
- served artifact: `qwen25-0.5b-instruct-iq4_xs.gguf`, 349,402,688 bytes,
  sha256 `df178ccd…`.

Re-verify the committed file against its inputs with:

```sh
python scripts/census_wave_provenance.py \
  verification/public-corpus-census/import-run/wave --check
```

