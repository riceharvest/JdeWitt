# Qwen3-4B-Instruct-2507 Q4_K_M → Q8_0, banked edit

**Label: `banked_unmeasured`.** This file was not measured. The direction was fitted
on `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c0` and applied as-is at strength
`{attn 1.5, mlp 1.5}`; the numbers below are codec fidelity of the stored edit, not
a claim about this model's answers.

## What produced it

`jdewitt bank-apply --bank spaces/bank --entry qwen3-4b-instruct-2507/f15s15-l935`,
explicitly at `--edit-type Q8_0`, on the base's own quantized artifact
`unsloth/Qwen3-4B-Instruct-2507-GGUF@a06e946bb6b6` `Q4_K_M`
(2,497,281,120 bytes, sha256 `3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597`).
Header read → byte copy → tensor re-encode → read-back. No inference, no tokenizer,
no server; nothing was generated from this file.

## Read-back

| fact | value |
|---|---|
| tensors in | 398 |
| edited | **42** (the entry's footprint) |
| delivered cos, min / mean | **0.9873** / 0.9915 (floor 0.95, met) |
| untouched | **356**, byte-identical against the input on re-read |
| codec | Q8_0 (the auto ladder's Q6_K was not used here: its sampled preview is optimistic on these rank-1 deltas) |
| bytes in → out | 2,497,281,120 → **2,798,248,544** |
| sha256 | `bdeca8be7fa2f9c28bd952737ed2dfc65190954927543fad9f0165b8b32fc5ed` |
| `removal_measured` | **false** — cos is what the bytes hold, not that anything was removed |

Evidence: `verification/space/selfapply-qwen3-4b.json` and
`verification/space/selfapply-run/bank-apply-q8_0.json`.
