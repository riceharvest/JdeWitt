# Public prompt corpora: census, import, provenance

Two things live here:

1. **A census** of the prompt corpora that public abliteration / uncensoring
   releases actually use or ship — dataset id, pinned revision, row count,
   languages, prompt vs prompt+response, harmful/harmless pairing and licence —
   with per-row provenance kept in `verification/public-corpus-census/`.
2. **The importer** that turns any declared subset of those corpora into this
   tool's suites, for use with the same wave instrument as our authored suites.

The point is comparability with a stranger: a claim should be re-runnable
against exactly the prompts a competitor used, at the revision they used, with
the licence they shipped. A corpus whose card states no licence is *referenced*
here, never vendored.

```sh
jdewitt validate-import spec.json --provenance verification/public-corpus-census/sources.json
jdewitt import-suite  spec.json --output DIR --provenance verification/public-corpus-census/sources.json
```

`--provenance` is optional for backwards compatibility, but a spec imported with
it refuses any source whose id is absent from the census, or whose dataset or
revision disagrees with it. A census entry whose licence is `unstated` is
importable — otherwise Heretic's own default prompts could never be tested — but
it is listed in the report's `unlicensed_sources`, is never vendored, and every
row it produces carries `license: unstated`. Every imported row then carries
`source` and `license`, and the import's `import-provenance.json` repeats the
census digest and the full citation (`license`, `license_url`, `url`,
`referenced_by`, `revision`).

## What each release actually ships

The brand names are model families, not corpora. The corpus is what the release
file references, and that is what this table cites.

| release | corpus it actually uses | where that is stated |
| --- | --- | --- |
| Heretic 2.x | `mlabonne/harmful_behaviors` + `mlabonne/harmless_alpaca` (mirrored as `heretic-org/*`) | `p-e-w/heretic` `src/heretic/config.py`, dataset defaults |
| Heretic (semantic variant) | `heretic-org/Semantic-{Harmful,Harmless}`, 416 one-to-one pairs | dataset card + `metadata/matched_pairs.json` |
| Heretic (multilingual) | `heretic-org/Multilingual-Harmless-Harmful`, 10 languages x 416 x 2 | dataset card + `config.multilingual.toml` |
| AEON (AEON-7) | no corpus of its own; "heretic's `compute_refusal_directions`" on the heretic defaults | `AEON-7/Gemma-4-12B-it-AEON-Abliterated-K4-BF16` card |
| Blackfrost | R1-HARMFUL-BENCH-450 = 150 AdvBench + 150 StrongREJECT + 150 XSTest (composition published, no dataset id) | `Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16` and `-GGUF` cards |
| OBLITERATUS | crowd telemetry (`pliny-the-prompter/OBLITERATUS-TELEMETRY`); "targeted corpus" for V3 is not published as a dataset | `elder-plinius/OBLITERATUS` README; release card |
| Pliny the Prompter | `pliny-the-prompter/g0dm0d3` (30,424 sharded jsonl files of harvested prompts/telemetry, AGPL-3.0) | dataset card; OBLITERATUS release cards |
| abliterator | AdvBench `goal` column (test split) + `tatsu-lab/alpaca` instructions with empty `input` | `FailSpy/abliterator` `abliterator.py` |
| DECCP | `augmxnt/deccp`: 95 censored + 37 uncensored prompts, Apache-2.0 | dataset card; `augmxnt/Qwen2-7B-Instruct-deccp` |
| Josiefied / JOSIE | model cards say "custom dataset"; the author's published corpora are `JOSIE-Prompts-Only-Corpus-300` (300) and `Josiefication-prompts-online-po` (4.0M), both with no licence field | model + dataset cards |
| ORCA (OrcaRouter) | **unresolved**: model cards are gated (HTTP 401 without accepting the gate) and no dataset id is published | `orcarouter/*-abliterated` repos |

## Census table

Vendored = text is redistributable and written to
`verification/public-corpus-census/rows.jsonl`. Referenced = dataset id and
revision recorded, text downloaded at import time by the importer (or not
importable at all when the release publishes nothing).

| source | rows | languages | pairs / structure | licence | use | vendored |
| --- | --- | --- | --- | --- | --- | --- |
| `walledai/StrongREJECT@df54272b` | 313 | en | no; prompt-only + category | MIT | Blackfrost bench (150), harmful eval | yes |
| `Paul/XSTest@f600c994` | 450 | en | safe/unsafe label; prompt-only | CC-BY-4.0 | Blackfrost bench (150), over-refusal | yes |
| `heretic-org/Semantic-Harmful@001ca2ce` | 416 | en | one-to-one with Semantic-Harmless | CC-BY-4.0 | heretic semantic contrast (harmful) | yes |
| `heretic-org/Semantic-Harmless@7e9f2b01` | 416 | en | one-to-one with Semantic-Harmful | CC-BY-4.0 | heretic semantic contrast (harmless) | yes |
| `heretic-org/Multilingual-Harmless-Harmful@f5a870a4` | 8,320 | en, bn, fr, hi, it, ja, ko, mr, ru, ta | 416 harmful + 416 harmless per language | CC-BY-SA-4.0 | multilingual contrast | yes |
| `augmxnt/deccp@2c964c69` | 132 | en (zh topics) | censored 95 / uncensored 37 | Apache-2.0 | Chinese-censorship refusal contrast | yes |
| `heretic-org/harmful_behaviors@55eeb6b5` | 520 | en | no; prompt-only | **unstated** (mirror of mlabonne) | heretic default harmful | no |
| `heretic-org/harmless_alpaca@50f4f3fb` | 31,323 | en | no; prompt-only | **unstated** (mirror of mlabonne) | heretic default harmless | no |
| `mlabonne/harmful_behaviors@01cead01` | 520 | en | no; prompt-only | **unstated** (AdvBench upstream is MIT) | shared harmful list | no |
| `mlabonne/harmless_alpaca@02c6a92c` | 31,323 | en | no; instruction-only | **unstated** (alpaca upstream is CC-BY-NC-4.0) | shared harmless list | no |
| `llm-attacks/advbench` | 520 | en | no; prompt-only | MIT | abliterator harmful, Blackfrost bench | no — gated, no reachable revision |
| `tatsu-lab/alpaca@dce01c9b` | 52,002 | en | instruction+input+output | CC-BY-NC-4.0 | abliterator harmless | no — non-commercial |
| `Goekdeniz-Guelmez/JOSIE-Prompts-Only-Corpus-300@9a5ffe35` | 300 | en | no; prompt-only | **unstated** | Josiefied corpus | no |
| `Goekdeniz-Guelmez/Josiefication-prompts-online-po@0006ee4c` | 4,019,573 | en | no; prompt-only | **unstated** | Josiefication scrape | no |
| `pliny-the-prompter/g0dm0d3@f8055e22` | not claimed (30,424 shards) | en | no; sharded jsonl telemetry/prompts | AGPL-3.0 | Pliny prompt corpus | no — sharded, no row total |
| `pliny-the-prompter/OBLITERATUS-TELEMETRY@30ecd955` | not claimed | en | run telemetry, not prompt/response | AGPL-3.0 | OBLITERATUS crowd data | no |
| `Blackfrost R1-HARMFUL-BENCH-450` | 450 | en | composite 150/150/150; no dataset id | composite (MIT + MIT + CC-BY-4.0) | Blackfrost release score | no — reference to parts |
| `AEON-7 heretic defaults` | — | en | heretic contrast pair | composite | AEON abliteration | no — reference to parts |
| `ORCA / OrcaRouter` | — | — | unresolved; cards gated, no dataset id | **unstated** | ORCA abliterated family | no |

Per-source licences are the *dataset card's* statement, fetched 2026-11-22;
where a card states none, the table says `unstated` even when the known upstream
is permissive. That is deliberate: a claim about a corpus should cite what the
corpus says about itself, not what its ancestor presumably said.

## Dedupe report

`verification/public-corpus-census/dedupe.json`, produced by
`harvest.py` from the vendored rows:

| number | value |
| --- | --- |
| rows harvested (vendored) | 10,047 |
| unique normalised prompts | 9,160 |
| duplicate rows | 887 |
| cross-source duplicate groups | 867 |

Normalisation: NFKC, casefold, whitespace collapse, sha256. The duplicates are
not noise; they are the evidence that the "different" releases are largely the
same prompt list:

- 399 groups: `Semantic-Harmful` == English `Multilingual-...-harmful`;
- 416 groups: `Semantic-Harmless` == English `Multilingual-...-harmless`;
- 17 groups: `StrongREJECT` appears verbatim inside `Semantic-Harmful` (and its
  multilingual copy);
- 35 groups are duplicates *within* the multilingual corpus (identical text
  under more than one language file).

Consequence for measurement: importing two of these into one split would fail
the suite validator, which is the intended behaviour. The canonical rows file
keeps one copy per prompt and lists every source in `duplicate_groups`, so a
claim can say which releases shared the row.

## Family coverage against our suites

`verification/public-corpus-census/coverage.json` screens each source's prompts
against the family vocabulary used by the authored harsh suites. It is a lexical
screen, not a family claim.

- Covered with real prompts: cybercrime/intrusion, illegal, fraud/deception,
  violence, sexual content, privacy, misinformation/disinformation,
  non-violent crimes, illegal goods, government decision-making, economic harm,
  physical harm, hate/harassment, malware/hacking, copyright.
- Thin: chemical/biological (3 hits), expert advice (4), harassment/bullying as
  distinct from hate (26).
- Absent by construction: maths and code. Public refusal corpora cannot supply
  capability retention; that stays authored (the 241-row suite, 76 families).
- Absent as a family: over-refusal on *safe* prompts is only in XSTest (250
  `safe` rows), which our authored suites do not have as a family. That is the
  one axis where a public corpus adds something our suites cannot.
- No source carries genuine topic families except StrongREJECT (`category`),
  XSTest (`type`/`focus`) and deccp's two sets; the heretic/multilingual corpora
  are single-family prompt lists. Rank-based methods must use the categorised
  sources or accept synthetic row groups (documented in the import report).

## Spec

The spec has exactly `seed` and `splits`. Splits are `directions`, `selection`
and `holdout`.

A corpus source:

| field | meaning |
| --- | --- |
| `id` | source name; prefixes generated row ids, and must match a census id when `--provenance` is used |
| `kind` | `refusal`, `harmless` or `direction` (never `math`/`code` from a corpus) |
| `language` | literal string, or `{"column": "..."}` |
| `label` | required for `direction`: `good` or `bad` |
| `dataset` / `revision` | Hub dataset and its **pinned** commit; a branch is refused |
| `split` / `config` / `column` | dataset split, optional config subdir, prompt column |
| `offset` / `limit` | deterministic row range, `records[offset:][::stride][:limit]` |
| `stride` | optional integer >= 1; `2` interleaves two disjoint halves of a category-sorted corpus so selection and holdout share every category |
| `family` | required for `refusal` and `direction` only |
| `category` | optional literal or `{"column": "..."}`, copied onto the row |

Or a local file: `path` plus `column`, instead of `dataset`/`revision`.

Or an authored passthrough: `{"id": ..., "passthrough": true, "path": "...",
"kinds": ["math", "code"]}` reuses an existing suite file verbatim, optionally
filtered by kind. This is how authored capability tasks join imported ones.

A worked census import is committed: `examples/public-corpus/census-strongreject.json`
(157 selection / 156 holdout StrongREJECT rows, real `category` families) with
its run in `verification/public-corpus-census/import-run/`.

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
collide across splits (with `column` families, give each split its own `group`).

The `directional` method only uses per-language means, so synthetic pairing does
not affect it. Rank-based methods (`svd`, `som`, `whitened-svd`) contrast
per-family means, so synthetic families make their direction estimates
meaningless. Use a category column for those.

## Output

- `directions.jsonl`, `selection.jsonl`, `holdout.jsonl` as declared
- `import-provenance.json`: per source the dataset, pinned revision, chosen file
  and its sha256, row range, rows used, duplicates dropped, family strategy,
  family origin, family count, plus per-split digests and `not_checked`; with
  `--provenance`, also the census digest and each source's licence and citation,
  and every row carries `source` + `license`

Duplicate prompts are dropped deterministically, first occurrence wins, and the
count is recorded. Split disjointness is re-checked through the same validator
the runs use, so a spec that reuses rows fails instead of writing bad suites.

## What the importer will not do

- It never decides which split a prompt belongs to. Every source names its split.
- It never invents maths or code tasks. Those must be authored; the importer can
  pass an authored suite through unchanged.
- It never guesses a language. `language` is a literal or an explicit column.
- It never claims a synthetic row grouping is a real topic family.
- It never rewrites an existing directory.
- With `--provenance`, it never imports a source that the census does not
  license, and never lets a spec's revision drift from the census revision.
- It never vendors an unlicensed corpus into this repository.

## Limits

- Directions still need two languages. Almost every public corpus is
  English-only, so either keep authored multilingual directions or supply a
  multilingual corpus with a language column (`Multilingual-Harmless-Harmful`
  does). This is a hard gate in the tool, not an importer choice.
- Parquet corpora need the optional extra: `uv pip install "jdewitt[datasets]"`.
  CSV, JSON, JSONL and TXT import without it.
- Importing proves nothing about harmfulness, refusal wording, token budgets or
  model fit. A 60-family synthetic grouping is a harness for measuring a change,
  not evidence that the prompts are harmful to the model under test.
- A census row's presence is not a claim that the release evaluates removal on
  it. Blackfrost, for example, publishes the *composition* of its 450-case bench
  but not the exact selection; the parts are cited, the exact subset is not
  reconstructed.
