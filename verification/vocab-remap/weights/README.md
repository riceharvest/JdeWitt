# vocab-remap: the weight-side row edit (implementation 2, code half)

`jdewitt vocab-remap --weights GGUF --table FILE --out GGUF [--apply]` applies the
denial table to the artifact's own weights: the table is resolved against the
artifact tokenizer (the same byte-level BPE `--list` uses), and the ids it names
become rows of the token-embedding matrix and, when the file carries an untied
one, of the output head.

This card is the code path and its CPU proof. The weight side is gated for
behavioral claims, so nothing here is a removal, harmlessness or capability
number: the gate below measures storage fidelity (realized cos), byte accounting,
untouched-tensor identity and a byte-exact revert.

## The edit

For each covered row `W_i` the edit is the rank-1 projection a refusal axis gets,
with the axis being the row's own direction on its input side:

    u = W_i / ||W_i||,   dW_i = -s (W_i u^T) u = -s W_i

so the projection algebra and a per-row down-weight scale `1 - s` are the same
operation, and `ggufio.axis_delta` — the helper the refusal axis uses — computes
it. `--strength` is `s` in [0, 1]; the default 1.0 removes the row, below 1
down-weights it. Rows outside the table are not part of the edit.

This is deliberately not the decode-side rule. `--remap on` suppresses a token
only when it completes a whole table sequence and reports what it left; a row is
down-weighted wherever that token appears, function tokens included (17 of the 48
ids here are shared by more than one entry, `" can"` / `" not"` / `"I"` among
them). The weight side is strictly broader than the decode side, and the two are
not substitutes.

## The artifact

`verification/iq05b/Qwen2.5-0.5B-Instruct-IQ3_M.gguf`, sha256 `0b2e1747…32099`,
342,752,576 bytes, 290 tensors. Its `token_embd.weight` is Q8_0, 151,936 rows ×
896 values, and there is no `output.weight`: Qwen2.5-0.5B ties the output head to
the embedding, so the edit lands on that one matrix and the dry run reports the
tied head rather than inventing a second target.

## Gate

Commands, in order (`$M` = the artifact, `$O` = the new one, `$T` = the table):

    jdewitt vocab-remap --weights "$M" --table "$T" --json verification/vocab-remap/weights/dryrun.json
    jdewitt vocab-remap --weights "$M" --table "$T" --out "$O" --apply --json verification/vocab-remap/weights/apply.json
    jdewitt gguf-tensor-diff "$M" "$O"  > verification/vocab-remap/weights/diff.json
    jdewitt gguf-revert "$O" "$O.revert.npz" > verification/vocab-remap/weights/revert.json

`apply.json` is the command's own report for the write (the gate names three
files; this is the fourth because the write needs a receipt); the two runs are
deterministic — the same patched sha256 came out of two independent applies.

`dryrun.json` — 20 table entries resolve to 48 ids. The edit is confined to those
48 rows: 1,344 of 4,254,208 Q8_0 blocks, 45,696 of 144,643,072 tensor bytes
(0.03 %). Realized cos on the edited rows is **1.0** (snr 1.2e7, i.e. the Q8_0
grid stores a scaled row to rounding), unchanged at strength 0.9 and 0.5, where
the rows keep 10 % and 50 % of their norm instead of being removed. The codec
chosen per tensor is its own stored type (Q8_0), which is what keeps every tensor
offset and byte length in place; the plan reports 144,643,072 bytes re-encoded,
because the writer requantizes the whole edited tensor and only the changed
blocks are allowed to differ.

`apply.json` — 1,344 changed blocks, exactly the touched blocks: every other block
in the tensor re-encoded to its original bytes. 1,229 byte ranges differ and all
of them are inside `token_embd.weight`; the 198,109,504 bytes outside the edited
tensor were compared byte for byte and agree. Delivered cos re-measured on the
shipped file is 1.0 on the edited rows (whole-tensor cos also 1.0). The revert
sidecar is 65,858 bytes holding 44,422 pre-edit bytes in 1,229 ranges plus both
digests.

`diff.json` — 290 shared tensors, **289 bit-identical**, `only_in_*` empty,
`token_embd.weight` the single `value_differs` row (Q8_0 → Q8_0, rel_rms
0.017529, consistent with 48 of 151,936 rows removed).

`revert.json` — `gguf-revert` restored 1,229 ranges and reports
`original_sha256` = `restored_sha256` = `0b2e1747…32099`, `byte_exact: true`. `cmp`
against the source then finds the files identical: the revert is byte-exact, not
a claim about the ranges that happened to be saved.

End state on disk: the artifact is left **patched** (sha `00bdd8fc…3c40`) with its
sidecar, so `diff.json` describes it and revert can be re-run. The gate's revert
step was also re-checked on a scratch copy of the final artifact, so the sidecar
present now is proven, not just the one the run consumed.

## Untied heads and other codecs

The gate artifact ties its head, so the two-tensor path was checked on a synthetic
fixture (28 byte-level tokens, 256-wide rows; `token_embd.weight` Q8_0,
`output.weight` Q6_K, a merge rank that never fires so every form resolves
per byte). `--weights` resolved both roles, edited 22 rows in each, reported cos
1.0 on both tensors, and `gguf-revert` restored the file byte for byte
(`byte_exact: true`, `cmp` clean).

That run also shows a codec difference the gate's Q8_0 case hides: the Q6_K
tensor changed 25 blocks for 22 touched rows. Q6_K's requantization of an
unedited block is not always byte-exact, so blocks outside the covered rows of
the same tensor move too. The report counts them as changed blocks and the
sidecar restores them; "untouched" on this card means tensors, never rows.

## What this does not say

- No model was loaded and no token was generated on this card: cos is a storage
  measurement of the weight edit, not evidence about behavior.
- `--weights` refuses an artifact whose token-row tensor this build cannot encode
  at its own layout: a type change would move tensor offsets and the pre-edit
  bytes could no longer restore the file exactly. That refusal is the price of a
  provable revert, and it names the types this build writes.
- A phrase entry covers every token of its split (`coverage.rows` names the entry
  behind each id); the table names surfaces, not lexemes, and this edit inherits
  that coverage unchanged.
