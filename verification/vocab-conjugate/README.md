# vocab-conjugate: the head's opening rows exchanged, as a bijection

`jdewitt vocab-conjugate --weights GGUF [--table FILE] [--out GGUF --apply]` makes
the artifact's token-row matrix decode a compliance opener where it would decode a
denial opener. The edit is a transposition per pair, not a removal:

    W'_d = W_c,   W'_c = W_d,   W'_i = W_i for every other row i

so the hidden state `h` that scored the denial row — `h·W_d` is the logit the head
gave that token — now scores that same value on the compliance row, and the
compliance row's old score moves to the denial id. The map is a product of disjoint
transpositions on the resolved ids: an involution, hence a bijection, so the
multiset of rows of the matrix is preserved exactly and the tensor's Frobenius norm
does not move.

This card is the CPU-only half of a paired-method experiment. No model is loaded, no
logit or generation is read, and `removal_measured` is false: what follows is
storage algebra and byte accounting. The behavioural question is GATED.

## The pair table

`jdewitt/assets/vocab-conjugate-openers.json`, sha256 `d5f8a625…537c0ec`, four pairs.
Each surface is resolved against the artifact's own byte-level BPE in two contexts
(bare — the first token of a response — and `leads_with_space` — the same word
mid-sentence), and the command refuses unless each side is exactly one token in
both. `As` (the base model's third attested refusal opener) is deliberately **not**
mapped: it is a high-frequency discourse marker, and leaving it is the negative
control that proves the edit is confined to the chosen ids.

| pair | denial (bare / spaced) | compliance (bare / spaced) | evidence |
|---|---|---|---|
| en-i-here | `I` 40 / `ĠI` 358 | `Here` 8420 / `ĠHere` 5692 | attested: 6 of 10 base greedy rows open `I'm sorry, but …` |
| en-sorry-sure | `Sorry` 19152 / `ĠSorry` 32286 | `Sure` 39814 / `ĠSure` 22555 | attested: `vr-probe-0006` opens `Sorry, but I can't …` |
| en-cannot-certainly | `Cannot` 17444 / `ĠCannot` 34655 | `Certainly` 95456 / `ĠCertainly` 52366 | card example; ` cannot` attested as a denial word, sentence-initial `Cannot` not attested on this build |
| de-es-hier | `Es` 17360 / `ĠEs` 9236 | `Hier` 74909 / `ĠHier` 38747 | attested: `vr-probe-0007` opens `Es ist wichtig … nicht möglich`; ` Hier` is a compliance opener on that row |

16 ids, 16 distinct, 0 fixed points, involution true, 151,920 unmapped ids
(`dryrun.json` `permutation.checks`). Tokenizer identity: gpt2/qwen2, 151,936
tokens, tokens sha256 `a760a3ca…e1000`.

## The artifact

`verification/iq05b/Qwen2.5-0.5B-Instruct-IQ3_M.gguf`, sha256 `0b2e1747…32099`,
342,752,576 bytes, 290 tensors. `token_embd.weight` is Q8_0, 151,936 rows × 896
values (28 whole blocks per row) and there is **no `output.weight`**: Qwen2.5-0.5B
ties the output head to the embedding, so the transposed rows are the head rows and
the embedding rows at once. The report says `tied: true` and names it.

## Gate results

Commands (`$M` = the artifact, `$O` = the new one):

    jdewitt vocab-conjugate --weights "$M" --json verification/vocab-conjugate/dryrun.json
    jdewitt vocab-conjugate --weights "$M" --out "$O" --apply --json verification/vocab-conjugate/apply.json
    jdewitt gguf-inspect "$O"                 > verification/vocab-conjugate/inspect.json
    jdewitt gguf-tensor-diff "$M" "$O"        > verification/vocab-conjugate/diff.json
    jdewitt gguf-revert "$O.copy" "$O.revert.npz" > verification/vocab-conjugate/revert.json

`dryrun.json` — the edit is confined to 16 rows: 448 of 4,254,208 Q8_0 blocks, and
15,232 of 144,643,072 tensor bytes (0.0105 %). The codec chosen per tensor is its
own stored type (Q8_0), which keeps every tensor offset and byte length in place.
Realized cos on the conjugated rows is **1.0** (snr 2.2e12): the target row is the
partner's *stored* values, already on the Q8_0 grid, so re-encoding it reproduces it
exactly. This is the opposite of the removal card's finding — a row swap is in the
codec's image, a rank-1 projection is not.

Per mapped row (`edit.rows.token_embd.weight`; `cos` is cos(row before, values stored
at the same id), the same number for both ends of a transposition):

| pair | context | id | token → partner | cos | ‖before‖ | ‖after‖ |
|---|---|---|---|---|---|---|
| en-i-here | bare | 40 | `I` → `Here` | 0.052209 | 0.466851 | 0.408523 |
| en-i-here | spaced | 358 | `ĠI` → `ĠHere` | 0.071066 | 0.452987 | 0.405508 |
| en-i-here | spaced | 5692 | `ĠHere` → `ĠI` | 0.071066 | 0.405508 | 0.452987 |
| en-i-here | bare | 8420 | `Here` → `I` | 0.052209 | 0.408523 | 0.466851 |
| de-es-hier | spaced | 9236 | `ĠEs` → `ĠHier` | 0.089518 | 0.410930 | 0.423497 |
| de-es-hier | bare | 17360 | `Es` → `Hier` | 0.139469 | 0.435811 | 0.450795 |
| en-cannot-certainly | bare | 17444 | `Cannot` → `Certainly` | 0.195181 | 0.371156 | 0.401020 |
| en-sorry-sure | bare | 19152 | `Sorry` → `Sure` | 0.245144 | 0.406514 | 0.410061 |
| en-sorry-sure | spaced | 22555 | `ĠSure` → `ĠSorry` | 0.249450 | 0.403342 | 0.385225 |
| en-sorry-sure | spaced | 32286 | `ĠSorry` → `ĠSure` | 0.249450 | 0.385225 | 0.403342 |
| en-cannot-certainly | spaced | 34655 | `ĠCannot` → `ĠCertainly` | 0.107221 | 0.379668 | 0.375775 |
| de-es-hier | spaced | 38747 | `ĠHier` → `ĠEs` | 0.089518 | 0.423497 | 0.410930 |
| en-sorry-sure | bare | 39814 | `Sure` → `Sorry` | 0.245144 | 0.410061 | 0.406514 |
| en-cannot-certainly | spaced | 52366 | `ĠCertainly` → `ĠCannot` | 0.107221 | 0.375775 | 0.379668 |
| de-es-hier | bare | 74909 | `Hier` → `Es` | 0.139469 | 0.450795 | 0.435811 |
| en-cannot-certainly | bare | 95456 | `Certainly` → `Cannot` | 0.195181 | 0.401020 | 0.371156 |

The low cosines are the point: the compliance row is a different direction, not a
scaled copy, so the conjugation is a large move in the head's geometry (and the
norms stay within a few per cent, which is what keeps the edit's magnitude honest).

`apply.json` — output sha256 `bfabcc5f…846d02`, same size. 226 differing byte ranges,
15,022 differing bytes, **all inside `token_embd.weight`**; the 198,109,504 bytes
outside the edited tensor were compared byte for byte and agree. Row-granularity
check (`apply.row_identity`): 151,936 rows read, **16 changed / 151,920 unchanged**,
the changed set equals the mapped id set, and every mapped row holds its partner's
stored bytes (16/16). The revert sidecar is 20,410 bytes holding 15,022 pre-edit
bytes in 226 ranges plus both digests.

`diff.json` — 290 shared tensors, **289 bit-identical**, `only_in_*` empty,
`token_embd.weight` the single differing row (Q8_0 → Q8_0, rel_rms 0.012169). This
names exactly the one tensor the conjugations live in; the row-level check above
names exactly the 16 rows.

`inspect.json` — `gguf-inspect` reads the output: 290 tensors, 342,752,576 bytes, the
same per-type byte census as the source.

`revert.json` — `gguf-revert` restored 226 ranges and reports
`original_sha256` = `restored_sha256` = `0b2e1747…32099`, `byte_exact: true`; `cmp`
against the source finds the files identical. The revert was run on a copy, so the
applied artifact and its sidecar stay on disk for inspection.

## What this does not say — and the garbage it expects

- **No model, no serve, no GPU, no token.** The numbers above are weight algebra,
  byte identity and tokenizer strings. `removal_measured: false` in every report; a
  behavioural reading is out of scope for this card.
- **The rest of the refusal is planned upstream.** Mapping the first token does not
  re-plan the continuation. On the recorded base greedy path, a conjugated file
  asked the `vr-probe-0006` prompt would now start `Sure` where the base starts
  `Sorry`, and would be followed by the same `, but I can't assist with that.`; the
  English rows would start `Here'm sorry, …`. That is a compliant opening glued to a
  refusal body — the outcome the card exists to record, not a fix. It is **derived**
  from the transposition plus the recorded token path, not measured.
- **Tied file: the edit is on both sides.** The same rows are input embeddings, so a
  prompt containing `I`, `Sorry`, `Cannot` or `Es` now embeds as `Here`, `Sure`,
  `Certainly`, `Hier`. That is inherent to a tied artifact and is unmeasured here.
- **Four pairs, one artifact, one codec.** The pair table is a design choice, not a
  policy; the lossless storage result is a property of Q8_0 storing an already-
  quantized row. The weight-removal card records the other codec case: Q6_K can move
  unedited blocks of the same tensor, which is why the row identity check is measured
  per file rather than assumed.

## Twin provenance

The paired-method card names a public base/twin pair. `twin-provenance.json` records
repository, revision, size and sha256 for both single-safetensors checkpoints, with
the Hub's declared LFS oid matching the local digest in each case; the downloads were
deleted after hashing:

- base `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae55760…` — 988,097,824 bytes,
  sha256 `fdf756fa…8fb7fe`
- twin `Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1` @
  `a42fe88f…` — 1,260,367,448 bytes, sha256 `dff81834…b8e2406`; header read: 291
  tensors, BF16, untied `lm_head.weight` beside `model.embed_tokens.weight`

The twin's untied head is a real layout difference from the IQ3_M artifact edited
here, and is recorded so a later paired run decides the head question per artifact.
Nothing about the twin's behaviour was read.
