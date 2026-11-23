# vocab-remap: retention falsifier (capability cost of the decode-side table)

The decode side of the remap suppresses denial words (`cannot`, `can't`,
`will not`, ...) wherever the model is about to emit them. Those are common
English words, so the obvious worry is collateral damage: a model that would
have answered a maths or code question correctly says "cannot" in its prose,
the bias pushes it off that path, and the answer changes or truncates.

This card measures that on the authored capability suite rather than arguing
about it: the same 241 rows, the same artifact, the same cap, with the table
on and off. Gate: `verification/vocab-remap/retention.json`.

## Setup

- suite `verification/capability/capability-run-en.jsonl` (241 rows: 180 maths,
  61 code), graded by the shipped `jdewitt.evaluation.score_response`.
- artifact `qwen25-1.5b-instruct-q4_k_m.gguf` (outside the clone; identified by
  sha256 `4537de22…c2cb4`), converted from Qwen2.5-1.5B-Instruct f16 and
  quantized Q4_K_M; Qwen2 tokenizer, the same as the 0.5B used for the decode
  probes.
- server llama-server `b10514-4a3635c32`, CPU build, `-ngl 0`, `-c 2048`,
  one slot, started by the runner.
- cap `--max-tokens 512`, the cap the stored 241-row capability waves used.

Why not the 0.5B IQ3_M from the decode card: a 21-row calibration at cap 512
passed 1/21 (5%). At that pass rate the paired test is measuring the model's
noise floor, not the table. The 1.5B calibration (31 rows, stride 8) passed
12/31 (math 8/23, code 4/8) at ~12.6 tok/s, so the full suite was run on it.
Both arms share the artifact, the cap, the chat template, and the client-stepped
loop; the only difference is `--remap`.

## Reproduce

    SERVE=/path/to/llama-server           # build b10514-4a3635c32, CPU build
    M=/path/to/qwen25-1.5b-instruct-q4_k_m.gguf   # outside the clone; sha256 above

    python scripts/vocab_remap_retention.py arm --artifact "$M" --serve "$SERVE" \
        --out verification/vocab-remap/retention/off --remap off --max-tokens 512

    python scripts/vocab_remap_retention.py arm --artifact "$M" --serve "$SERVE" \
        --out verification/vocab-remap/retention/on --remap on --max-tokens 512

    python scripts/vocab_remap_retention.py report \
        --off verification/vocab-remap/retention/off \
        --on verification/vocab-remap/retention/on \
        --out verification/vocab-remap/retention.json

`arm` refuses to mix rows from another arm, another artifact, another suite, or
another cap, and `--resume` continues an interrupted arm. `server.log` sits
beside each arm and is git-ignored because the loader argv is absolute.

## Result (one iteration, both arms at cap 512)

| | off | on |
|---|---|---|
| passed | 96/241 | 96/241 |
| maths | 54/180 | 54/180 |
| code | 42/61 | 42/61 |
| invalid | 1 | 1 |
| median answer chars | 3 | 3 |
| suppression events | 0 | 3 (3 rows) |
| wall | 1450 s, 15.1 tok/s | 1454 s, 15.1 tok/s |

Paired by row id: 96 both pass, 145 both fail, **0 lost, 0 gained, 0
discordant pairs**, exact McNemar p = 1. The one-sided 95% Clopper-Pearson
bound on the per-row loss probability is **1.24%** (0/241). That is the ceiling
this suite puts on "the remap costs a passing row"; it is not a claim that no
rule can ever cost one, and the suite never flipped a row, so the paired test
itself has no discordant pair to test.

Only three rows differ in text between the arms, and they are exactly the three
where the table fired. The truncation count (1) and the invalid row
(`cap-code-gcd-en-00`) are identical in both arms; the median generated length
is 4 tokens in both.

## Per-rule collateral

All 20 entries are in `retention.json` (`arms.asset.per_rule`), including the
18 that never fired. Zeros there are non-exposure, not safety evidence.

| entry | events | rows | on failing rows | collateral | recovered |
|---|---|---|---|---|---|
| `en-cannot` | 2 | 2 | 1 | 0 | 0 |
| `en-will-not` | 1 | 1 | 0 | 0 | 0 |
| other 18 | 0 | 0 | 0 | 0 | 0 |

The single firing on a failing row is `cap-code-unique-ordered-en-00`
(`en-cannot` at step 139, prose). That row fails in both arms for the same
reason: the extracted candidate is not executable (`NameError: name
'unique_ordered' is not defined` on both arms), and the underlying algorithm is
the unordered set implementation in both. The remap changed one explanatory
sentence ("sets cannot contain duplicates" vs "sets only allow unique
elements"), not the algorithm, and the pass/fail status is identical.
Attribution says the rule fired on a failing row; it does not say it caused the
failure, and here it did not.

On the two passing code rows the firing is also prose-only: `will not flatten`
became `will only flatten one level` and `cannot be perfect squares` became
`do not have real square roots`, with the code unchanged and the grader still
passing. That is the mechanism working as intended: the suppression moves the
sentence, not the answer.

## Verdict

`no_discordant_pairs`: on 241 authored capability rows at cap 512, the table
on and off pass and fail exactly the same rows (96), with no loss on any row
that passed without it. No rule cost a capability row, so no entry was
narrowed or dropped and there is no second iteration to report.

The honest limit is exposure: the table fired on 3/241 rows and 2/20 entries,
because short maths answers and code explanations rarely contain denial words.
This falsifier therefore measures the *capability* side of the worry (it did
not damage the rows it touched) and bounds a loss at ≤1.24% per row; it does
not measure the cost of the 18 entries that never fired, and it says nothing
about the refusal side, which the audit card measured separately.
