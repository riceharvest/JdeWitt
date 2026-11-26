# vocab-delete: the denial opener ids removed from the vocabulary

`jdewitt vocab-delete --weights GGUF [--out GGUF --apply]` deletes the ids the
opener table's denial surfaces resolve to, renumbers the vocabulary down, and
deletes the matching rows of the token-row matrix. This is the stronger arm of the
vocabulary pair: `vocab-conjugate` exchanges a denial row for a compliance row, this
one removes the row and the token entry that names it.

Nothing here is a behavioural claim. No model is loaded, no logit or generation is
read, no serve is started, `removal_measured` is `false`, and the GPU is not
touched. What follows is storage algebra, the renumbering map, and string automata
(the artifact's tokenizer, cross-checked with `llama-tokenize`, which tokenizes and
does nothing else).

## What the write changes

For each deleted id, all three things that share one id space move together:

| surface | change |
|---|---|
| `tokenizer.ggml.tokens` | the entry is dropped; every later entry moves down one |
| `tokenizer.ggml.token_type` | the same entry is dropped; no other value changes |
| `tokenizer.ggml.{bos,eos,padding}_token_id` | renumbered by the number of deleted ids below the old value |
| `token_embd.weight` | the row is deleted from the payload; the dims row count drops by the number of deleted ids |

Everything else is copied byte for byte: `tokenizer.ggml.merges` and the chat
template keep their exact encodings, every other tensor keeps its payload, and only
the tensor offsets after the shrunken matrix move (recomputed under the file's own
32-byte alignment). Surviving matrix rows are copied as whole stored rows — the
tensor is Q8_0 with 28 blocks per row, so a row is 952 raw bytes and nothing is
re-encoded.

The risk this card is honest about: the tokenizer arrays, the scalar id fields and
the matrix rows are one id space. A file that renumbers one and not the others is
broken (a token id that disagrees with its row reads the wrong embedding; a scalar
id that is not renumbered points at a different token). The report names every one
of them (`arrays_touched`), and `apply` measures the result instead of asserting it.

## The artifact and the deleted ids

`verification/iq05b/Qwen2.5-0.5B-Instruct-IQ3_M.gguf`, sha256
`0b2e17471ccc784dfd7b042a81e34289a3accb75ccda7347598a329e0a932099`, 342,752,576
bytes, 290 tensors, 151,936 tokens, tied head (`token_embd.weight` is the head),
Q8_0 896 values = 28 blocks per row.

The table is the packaged opener table, `jdewitt/assets/vocab-conjugate-openers.json`
(sha256 `d5f8a625…537c0ec`): four pairs, each denial surface resolved in both contexts.
Seven of the eight resolved ids are merged tokens and are deleted:

| pair | context | id | piece | token_type | bytes |
|---|---|---:|---|---:|---:|
| en-i-here | spaced | 358 | `ĠI` | 1 | 2 |
| de-es-hier | spaced | 9236 | `ĠEs` | 1 | 3 |
| de-es-hier | bare | 17360 | `Es` | 1 | 2 |
| en-cannot-certainly | bare | 17444 | `Cannot` | 1 | 6 |
| en-sorry-sure | bare | 19152 | `Sorry` | 1 | 5 |
| en-sorry-sure | spaced | 32286 | `ĠSorry` | 1 | 6 |
| en-cannot-certainly | spaced | 34655 | `ĠCannot` | 1 | 7 |

The eighth, id 40 `I`, is one raw byte: it **is** the tokenizer's alphabet entry for
byte 0x49. It is refused by default and deleted only under
`--include-byte-alphabet` (arm B), because deleting it makes every string containing
`I` unrepresentable — that is whole-tokenizer breakage, not a targeted removal.

## Gate results

Arm A (default, seven ids), `$M` = the artifact, `$O` = the new file:

    jdewitt vocab-delete --weights "$M" --json verification/vocab-delete/dryrun-A.json --map verification/vocab-delete/id-map-A.tsv.gz
    jdewitt vocab-delete --weights "$M" --out "$O" --apply --json verification/vocab-delete/apply-A.json
    jdewitt gguf-inspect "$O"                     > verification/vocab-delete/inspect-A.json
    jdewitt gguf-tensor-diff "$M" "$O"            > verification/vocab-delete/tensor-diff-A.json
    jdewitt vocab-delete --revert "$O" --sidecar "$O.revert.npz" --revert-out /tmp/reverted.gguf

Arm B (`--include-byte-alphabet`, eight ids) is the same three commands with `-B`;
its tokenizer-only cross-check is `llama-tokenize-B.json`.

| check | arm A | arm B |
|---|---|---|
| vocabulary | 151,936 → 151,929 | 151,936 → 151,928 |
| deleted ids | 358, 9236, 17360, 17444, 19152, 32286, 34655 | + 40 |
| new file bytes / sha256 | 342,745,824 / `c7bc90fa…acf416` | 342,744,832 / `5d9341b0…baa4862` |
| `token_embd.weight` dims | [896, 151,936] → [896, 151,929] | [896, 151,936] → [896, 151,928] |
| tensor payload bytes | 144,643,072 → 144,636,408 | 144,643,072 → 144,635,456 |
| tensors bit-identical (independent `gguf-tensor-diff`) | 289 of 290, 0 value-differing, 1 shape-changed | 289 of 290, 0 value-differing, 1 shape-changed |
| surviving rows byte-identical to their source rows | 151,929 of 151,929 | 151,928 of 151,928 |
| KV entries changed / untouched | 5 / 33, 0 differing | 5 / 33, 0 differing |
| file shrink | 6,752 bytes | 7,744 bytes |
| revert (sidecar re-inserts entries + rows) | byte-exact, sha `0b2e1747…32099` | byte-exact, same sha |

The five changed KV entries are exactly the planned ones —
`tokenizer.ggml.tokens`, `tokenizer.ggml.token_type`, `bos_token_id`,
`eos_token_id`, `padding_token_id`. The scalar ids move by the number of deleted ids
below them: `eos` 151,645 → 151,638 (A) / 151,637 (B); `bos` and `padding` 151,643 →
151,636 / 151,635, and the new ids still resolve to `<|im_end|>` and
`<|endoftext|>`.

The revert sidecar is 11,290 bytes (A) / 12,282 bytes (B): the seven or eight token
entries, their types, their raw 952-byte rows and the old scalar ids. Reverting
re-inserts them into the pruned container and recomputes the original layout; the
result's sha256 is the pristine artifact's, so the restore is a reconstruction, not
a stored copy.

## The renumbering map

`verification/vocab-delete/id-map-A.tsv.gz` is the full table, one line per
surviving id (`old_id`, `new_id`, `deleted`, sha256/16 of the piece, the piece) plus
one line per deleted id. It is the same content as the report's `renumber.segments`:

| old ids | shift | new ids |
|---|---|---|
| 0 – 357 | 0 | 0 – 357 |
| 359 – 9,235 | −1 | 358 – 9,234 |
| 9,237 – 17,359 | −2 | 9,235 – 17,357 |
| 17,361 – 17,443 | −3 | 17,358 – 17,440 |
| 17,445 – 19,151 | −4 | 17,441 – 19,147 |
| 19,153 – 32,285 | −5 | 19,148 – 32,280 |
| 32,287 – 34,654 | −6 | 32,281 – 32,648 |
| 34,656 – 151,935 | −7 | 34,649 – 151,928 |

Invariant measured on the written file: `tokens[new_id] == tokens_old[old_id]` for
all 151,929 survivors (`tokenizer.surviving_pieces_unchanged`), the deleted pieces
are absent (`deleted_pieces_absent`), and the mapping digest is in
`mapping.sha256`.

## What the deleted string does now

The vocab lookup misses, but the merge list still builds the symbol. What happens
next is runtime-specific, so the report carries both paths:

* `after` mirrors `src/llama-vocab.cpp`: each UTF-8 byte of the unknown symbol is
  looked up as a token **text**. ASCII bytes resolve; a non-ASCII byte has no
  single-byte token text and is silently dropped.
* `after_reference` is the byte-alphabet semantics Hugging Face uses: the symbol is
  mapped back to raw bytes through the byte alphabet and every byte is looked up.

| text | before | after, llama.cpp | after, reference | lossless (llama.cpp) |
|---|---|---|---|---|
| `Sorry` | [19152] | [50, 78, 81, 81, 88] | same | yes |
| `Sorry,` | [19152, 11] | [50, 78, 81, 81, 88, 11] | same | yes |
| `Cannot` | [17444] | [34, 64, 77, 77, 78, 83] | same | yes |
| `Es` | [17360] | [36, 82] | same | yes |
| ` I` | [358] | [40] | [220, 40] | **no — the space is dropped** |
| ` Es` | [9236] | [36, 82] | [220, 36, 82] | **no — the space is dropped** |
| `I'm sorry, but I can't` | [40, 2776, 14589, 11, 714, 358, 646, 944] | [40, 2775, 14587, 11, 713, 40, 645, 943] | [40, 2775, 14587, 11, 713, 220, 40, 645, 943] | **no — `ĠI` loses its space** |

Arm B adds id 40 `I`: `I` encodes to `[]`, ` I` to `[]` (the `Ġ` survives alone as
id 219 only in the reference path), and a sentence containing `I` silently loses it.

Both paths were checked against llama.cpp's own tokenizer on both files
(`llama-tokenize-A.json`, `llama-tokenize-B.json`): all 13 probes on the pristine
file and all 13 on the pruned file match the static ids exactly, `returncode 0` on
every call. `llama-tokenize` loads the vocabulary and tokenizes; it runs no forward
pass, and `CUDA_VISIBLE_DEVICES` is empty so no device is visible. That is a
tokenizer check, not a model check.

The consequence is the card's negative result, and it is the reason the task's
"if a denial string has no remaining spelling it cannot be sampled at all" does not
hold for this edit: **deleting a merged token removes its single-token id, not the
string.** `Sorry` is still spellable as five byte tokens, and the leading space is
its own surviving token (`Ġ`, 220), so a model that wants to say ` Sorry` can still
emit it. The only string that truly becomes unrepresentable is one that uses a
deleted byte-alphabet entry — and then the whole tokenizer is broken for every text
using that byte, not just the denial.

A second, narrower finding: llama.cpp's byte fallback is lossy for the deleted
merged symbols that carry the space marker. `Ġ` is two UTF-8 bytes (`C4 A0`), and
`byte_to_token`-style lookups find no token whose text is a lone `C4` or `A0`
byte, so both are dropped. Prompt-side ` Sorry` becomes `Sorry` (no space) in
llama.cpp, while the reference path keeps token 220. Either way the deleted id
itself can never be produced: it is not in `tokenizer.ggml.tokens` and
`token_embd.weight` has no row for it, so nothing can sample it.

## Limits

* Storage and tokenizer only. No forward pass, no logits, no serve, no benchmark,
  no GPU. `removal_measured` is `false`; whether the artifact refuses less is
  **not established** by anything here.
* The string-level claim is limited by the byte alphabet: a merged token is
  removable, a byte is not (without breaking the tokenizer). Deleting the byte
  `I` (arm B) measured exactly that: `I` and `I`-bearing text serialize to nothing.
* The llama.cpp fallback was measured on build `0.0.10102` of the local
  llama.cpp checkout. A different runtime may fall back differently; the static
  `after_reference` path is the one that keeps every surviving byte.
* The artifact is not committed (343 MB, `.gitignore`d), so a re-run needs the
  file, identified by sha256 above. The JSONs hold every measured number.
