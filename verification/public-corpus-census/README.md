# Public corpus census — raw harvest

Produced by `harvest.py`; consumed by `jdewitt import-suite --provenance`.

## Files

| file | what it is |
| --- | --- |
| `harvest.py` | the manifest and harvester. Every source entry carries dataset id, pinned revision, licence, row count and the release file that references it. Run: `uv run python verification/public-corpus-census/harvest.py` |
| `sources.json` | the canonical inventory (`format: jdewitt-public-corpora/1`): 19 source entries, 6 vendored, 13 referenced. This is the file `--provenance` reads. |
| `rows.jsonl` | one row per unique vendored prompt: `sha256` of the normalised prompt, `source`, `dataset`, `revision`, `license`, `language`, `split`, `source_row`, `prompt`. 9,160 rows. |
| `dedupe.json` | rows in / unique / duplicates, per-source counts, and every cross-source duplicate group with its member sources |
| `coverage.json` | lexical coverage screen of each source against the authored harsh-suite family vocabulary |
| `import-run/` | a real census import of StrongREJECT: 157 selection + 156 holdout rows, six matched `category` families per split, every row stamped `source` + `license` |
| `import-run/provenance.json` | that import's `import-provenance.json` with the machine-local cache symlink removed (file sha256 kept) |
| `import-run-summary.json` | sanitised extract of the same provenance |
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

Counts are in `dedupe.json`. Current: 10,047 vendored rows → 9,160 unique
prompts (887 duplicates, 867 cross-source groups). The largest groups are the
English files of `heretic-org/Multilingual-Harmless-Harmful` being byte-equal,
after normalisation, to `heretic-org/Semantic-{Harmful,Harmless}` (815 groups),
plus 17 StrongREJECT prompts that appear verbatim inside Semantic-Harmful.

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
inputs and exits non-zero on drift.

Success is a statement about provenance, not about harm: the refusal-check
result in a wave is a screen over wording, and the check file records
`claims.removal: false` so the counts cannot be read as a deletion claim.

