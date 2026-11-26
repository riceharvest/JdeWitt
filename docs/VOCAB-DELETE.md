# Vocabulary deletion: the denial ids removed, not remapped

`jdewitt vocab-delete` deletes the ids the denial-opener table resolves to and
renumbers the vocabulary down. It is the stronger member of the vocabulary pair:

* `vocab-conjugate` keeps the vocabulary size and swaps each denial row with a
  compliance row (a bijection on the head's rows);
* `vocab-delete` removes the token entry, removes the matrix row, and renumbers
  every surviving id — so the id that decoded the denial no longer exists.

The edit is static: tokenizer arrays, tensor rows and file bytes. No model is
loaded, no logits are read, no server is started, `removal_measured` is `false`.

## What moves

One id space is shared by three things, and all three are rewritten together:

    tokenizer.ggml.tokens          drop the entry, every later entry moves down
    tokenizer.ggml.token_type      drop the same entry
    tokenizer.ggml.*_token_id      renumber bos/eos/padding by the ids deleted below them
    token_embd.weight              drop the row; dims rows -= deleted
    output.weight (untied only)    drop the row; dims rows -= deleted

Everything else keeps its exact bytes: the merges array, the chat template and every
other KV entry; every other tensor payload. Only tensor offsets after the shrunken
matrix move, recomputed under the file's alignment. Surviving matrix rows are
copied as whole stored rows (Q8_0, 28 blocks per row = 952 bytes on the 0.5B), so
no surviving row is re-encoded.

A sidecar holds the deleted token entries and the deleted raw rows. The `--revert`
mode re-inserts them, recomputes the original layout and produces the pristine
file; on the 0.5B IQ3_M it reproduces the source sha256 exactly from a 11 KB
sidecar, so the restore is a reconstruction rather than a copy of the artifact.

## Candidates come from the table, not from the words

The table is `jdewitt/assets/vocab-conjugate-openers.json`, resolved in both
contexts (bare and with the leading space). On Qwen2.5-0.5B IQ3_M the eight denial
surfaces resolve to eight ids; seven are merged byte-level BPE tokens and are
deleted. The eighth, `I` (id 40), is a single raw byte — the tokenizer's alphabet
entry for 0x49. It is refused by default: deleting it makes every string containing
`I` unrepresentable, which breaks the tokenizer for all text, not just the denial.
`--include-byte-alphabet` measures that arm anyway (arm B).

## What deletion does to the string

The vocabulary lookup misses, but the merge list still builds the deleted symbol.
The runtime then falls back. Two fallbacks are reported because they disagree:

* llama.cpp (`src/llama-vocab.cpp`) looks each UTF-8 byte of the unknown symbol up
  as a token text. ASCII bytes resolve; a non-ASCII byte has no single-byte token
  text and is dropped.
* The reference (Hugging Face) semantics map the symbol back through the byte
  alphabet and look up every byte.

On the 0.5B, `Sorry` (id 19152) becomes five byte tokens
`[50, 78, 81, 81, 88]`; ` Sorry` (id 32286) becomes the same five in llama.cpp —
**the space is dropped**, because `Ġ` is two UTF-8 bytes (`C4 A0`) and neither is a
token text. The reference path keeps the space as token 220. All 13 probes on both
the pristine and the pruned file match `llama-tokenize` exactly.

Arm B adds id 40: `I` and `I`-bearing text serialize to nothing (`I` → `[]`). That
is the only case in which a string becomes truly unrepresentable, and it is
unrepresentable because the *byte* is gone, not because the denial word was
targeted.

The negative result for the method is therefore explicit:

* deleting a merged token removes its **single-token id**, not the string. The
  string stays samplable as a sequence, and the space token itself (220) survives,
  so ` Sorry` can still be emitted.
* no deletion of a merged token makes a denial string unsampleable; only deleting
  the byte alphabet does, and that breaks the whole tokenizer.
* the artefact can never produce the deleted id again: it is absent from
  `tokenizer.ggml.tokens` and has no row in `token_embd.weight`.

## Storage result

Artifact `verification/iq05b/Qwen2.5-0.5B-Instruct-IQ3_M.gguf` (sha256
`0b2e1747…32099`), 290 tensors, tied head. Arm A: 151,936 → 151,929 tokens, 7 rows
deleted, file 342,752,576 → 342,745,824 bytes. Arm B: 151,936 → 151,928, 8 rows,
342,744,832 bytes.

Measured on the written files: `gguf-tensor-diff` reports 289 of 290 tensors
bit-identical, 0 value-differing and 1 shape-changed in both arms; every surviving
matrix row is byte-identical to its source row (151,929 / 151,928 rows); exactly 5
of the 38 KV entries changed (tokens, token_type, bos/eos/padding) and the other 33
are byte-identical; `gguf-inspect`/`GGUFReader` parses both. The revert reproduces
the pristine sha256 byte for byte. The full old→new table is
`verification/vocab-delete/id-map-A.tsv.gz`; the report carries the same mapping as
contiguous shift segments.

## The risk this method carries

Renumbering is inherently coupled: the tokenizer arrays, the scalar id fields and
the matrix rows are the same id space, and a file that renumbers some and not the
others is silently broken — a token id that disagrees with its row reads the wrong
embedding, a scalar id that is not renumbered points at a different token. The
report names every array it touches. The inverse operation is only byte-exact
because the sidecar holds the deleted entries and raw rows; there is no generic
`gguf-revert` for a file whose layout moved.

## Gated

Whether the artifact refuses less after the deletion needs a forward pass, a serve
and a judge — none of which this card runs. `removal_measured` is `false` in every
JSON here, and no behavioural, capability or release claim is made. The static
result already shows the edit cannot be a removal by construction: the ids are gone,
the strings are not.

## Recompute

    jdewitt vocab-delete --weights "$M" --json verification/vocab-delete/dryrun-A.json
    jdewitt vocab-delete --weights "$M" --out "$O" --apply --json verification/vocab-delete/apply-A.json
    jdewitt vocab-delete --weights "$M" --include-byte-alphabet --out "$O2" --apply \
        --json verification/vocab-delete/apply-B.json
    jdewitt gguf-tensor-diff "$M" "$O" > verification/vocab-delete/tensor-diff-A.json
    jdewitt vocab-delete --revert "$O" --sidecar "$O.revert.npz" --revert-out /tmp/reverted.gguf

The tokenizer cross-check is `llama-tokenize -m FILE -p TEXT --ids --no-bos`; see
`verification/vocab-delete/llama-tokenize-{A,B}.json` for the 13 probes and the
per-probe match. It tokenizes only — no forward pass, and
`CUDA_VISIBLE_DEVICES=""` keeps the device invisible.
