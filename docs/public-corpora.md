# Importing public prompt corpora

```sh
jdewitt validate-import spec.json                 # spec + local files, no downloads
jdewitt import-suite spec.json --output DIR       # downloads and writes the suites
```

Public refusal corpora are larger and broader than anything worth authoring by
hand, but they are not shaped like this tool's suites: they are usually
English-only, have no topic families and contain no maths or code tasks. The
importer converts them and records exactly what it had to invent.

## What the importer will not do

- It never decides which split a prompt belongs to. Every source names its split.
- It never invents maths or code tasks. Those must be authored; the importer can
  pass an authored suite through unchanged.
- It never guesses a language. `language` is a literal or an explicit column.
- It never claims a synthetic row grouping is a real topic family. Row-group
  families are recorded as `family_origin: synthetic-row-group`, and the report
  says so in `not_checked`.
- It never rewrites an existing directory.

## Spec

The spec has exactly `seed` and `splits`. Splits are `directions`, `selection`
and `holdout`.

A corpus source:

| field | meaning |
| --- | --- |
| `id` | source name; prefixes generated row ids |
| `kind` | `refusal`, `harmless` or `direction` (never `math`/`code` from a corpus) |
| `language` | literal string, or `{"column": "..."}` |
| `label` | required for `direction`: `good` or `bad` |
| `dataset` / `revision` | Hub dataset and its **pinned** commit; a branch is refused |
| `split` / `config` / `column` | dataset split, optional config subdir, prompt column |
| `offset` / `limit` | deterministic row range, `records[offset:offset+limit]` |
| `family` | required for `refusal` and `direction` only |

Or a local file: `path` plus `column`, instead of `dataset`/`revision`.

Or an authored passthrough: `{"id": ..., "passthrough": true, "path": "...",
"kinds": ["math", "code"]}` reuses an existing suite file verbatim, optionally
filtered by kind. This is how authored capability tasks join imported ones.

## Families

```json
{"strategy": "column",     "column": "SemanticCategory"}
{"strategy": "row-group",  "size": 2, "group": "advbench"}
{"strategy": "constant",   "value": "topic"}
```

`row-group` chunks consecutive rows; it is the only option for corpora with no
category column. `group` overrides the id prefix so a paired good/bad corpus can
share family ids — without it a direction suite can never contain matched
families. Family ids embed the row offset, so two row ranges of one corpus never
collide across splits.

The `directional` method only uses per-language means, so synthetic pairing does
not affect it. Rank-based methods (`svd`, `som`, `whitened-svd`) contrast
per-family means, so synthetic families make their direction estimates
meaningless. Use a category column for those.

## Output

- `directions.jsonl`, `selection.jsonl`, `holdout.jsonl` as declared
- `import-provenance.json`: per source the dataset, pinned revision, chosen file
  and its sha256, row range, rows used, duplicates dropped, family strategy,
  family origin, family count, plus per-split digests and `not_checked`

Duplicate prompts are dropped deterministically, first occurrence wins, and the
count is recorded. Split disjointness is re-checked through the same validator
the runs use, so a spec that reuses rows fails instead of writing bad suites.

## Limits

- Directions still need two languages. Almost every public corpus is
  English-only, so either keep authored multilingual directions or supply a
  multilingual corpus with a language column. This is a hard gate in the tool,
  not an importer choice.
- Parquet corpora need the optional extra: `uv pip install "jdewitt[datasets]"`.
  CSV, JSON, JSONL and TXT import without it.
- Importing proves nothing about harmfulness, refusal wording, token budgets or
  model fit. A 60-family synthetic grouping is a harness for measuring a change,
  not evidence that the prompts are harmful to the model under test.
