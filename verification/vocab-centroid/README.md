# vocab-centroid: denial rows rewritten as the compliance centroid

`jdewitt vocab-centroid --weights GGUF --mode {centroid,centroid-orthogonal,row-orthogonal}
[--out GGUF --apply]` leaves the id map intact and rewrites the *values* of the
denial-opener rows of the token-row matrix. Three targets, all measured against the
same two group vectors:

    c = mean of the compliance-opener rows                      (the centroid)
    a = unit(mean(denial rows) - mean(compliance rows))         (the denial-vs-compliance mean)

    centroid             W_d <- c
    centroid-orthogonal  W_d <- c - (c.a) a
    row-orthogonal       W_d <- W_d - (W_d.a) a

Unlike the sibling `vocab-conjugate` card (a row transposition), no id changes its
meaning: `I` still tokenizes and decodes as `I`, `Sorry` as `Sorry`. What changes is
what the id's row *is* — the head-side vector its logit is a dot product with, and,
on this tied artifact, the input embedding at the same id. In mode `centroid` every
denial id is assigned the same vector, so the eight denial openings become one row.

This card is weight algebra and bytes. No model is loaded, no token is generated, no
logit is read, no GPU is used, and there is **no donor checkpoint**: every vector is
the target artifact's own row. `removal_measured` is false in every report, and the
behavioural question is GATED.

## The pair table and the two groups

`jdewitt/assets/vocab-conjugate-openers.json`, sha256 `d5f8a625…537c0ec`, four pairs.
Each surface is resolved against the artifact's own byte-level BPE in two contexts
(`bare` — first token of a response — and `leads_with_space`), and must be exactly one
token in both. The denial ids are the rows the edit rewrites; the compliance ids are
the rows it averages and **never writes**, so they are both the centroid and the
untouched control.

| pair | context | denial id → token | compliance id → token |
|---|---|---|---|
| en-i-here | bare | 40 `I` | 8420 `Here` |
| en-i-here | leads_with_space | 358 `ĠI` | 5692 `ĠHere` |
| en-sorry-sure | bare | 19152 `Sorry` | 39814 `Sure` |
| en-sorry-sure | leads_with_space | 32286 `ĠSorry` | 22555 `ĠSure` |
| en-cannot-certainly | bare | 17444 `Cannot` | 95456 `Certainly` |
| en-cannot-certainly | leads_with_space | 34655 `ĠCannot` | 52366 `ĠCertainly` |
| de-es-hier | bare | 17360 `Es` | 74909 `Hier` |
| de-es-hier | leads_with_space | 9236 `ĠEs` | 38747 `ĠHier` |

8 denial ids, 8 compliance ids, 16 distinct, disjoint (`groups.checks.ok = true`).
`As` — the base model's third attested refusal opener — is absent from the table, so
it is the negative control by construction: nothing in the edit can touch a row the
table never names.

The group contrast is real but modest, and these are the numbers the edit is built on
(`tensors[0].groups` in every report):

| quantity | value |
|---|---|
| centroid norm ‖c‖ | 0.253054 |
| norm of the denial group mean | 0.202360 |
| raw axis norm ‖mean_d − c‖ | 0.291775 |
| cos(denial mean, centroid) | +0.193853 |
| denial group mean · â | +0.106325 |
| compliance group mean · â | −0.185450 |
| per-row cos to â, denial rows | +0.144344 … +0.350257 (mean +0.253152) |
| per-row cos to â, compliance rows | −0.392790 … −0.501930 (mean −0.454076) |

Every denial row leans positive on the axis and every compliance row negative, with no
overlap — the axis separates the two groups on the rows the table names. That is a
statement about eight rows of one artifact, not a fitted refusal direction.

## Per-id pre/post (intended algebra)

`cos_pre_c` / `cos_post_c` are cos(row, c) before and after; `cos_pre_a` / `cos_post_a`
are cos(row, â); `n_pre` / `n_post` are row norms; `stored_c` is the shipped file's
cos(row, c) after the codec, measured on the decoded output row.

### mode `centroid`

| id | token | cos_pre_c | cos_post_c | cos_pre_a | cos_post_a | n_pre | n_post | n_post/n_pre | n change | stored_c | bytes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 40 | `I` | +0.002484 | +1.000000 | +0.326144 | −0.732847 | 0.466851 | 0.253054 | 0.542046 | −0.213796 | 0.999985 | 948 |
| 358 | `ĠI` | −0.100872 | +1.000000 | +0.350257 | −0.732847 | 0.452987 | 0.253054 | 0.558635 | −0.199932 | 0.999985 | 947 |
| 9236 | `ĠEs` | +0.007700 | +1.000000 | +0.312672 | −0.732847 | 0.410930 | 0.253054 | 0.615809 | −0.157876 | 0.999985 | 945 |
| 17360 | `Es` | +0.101135 | +1.000000 | +0.265129 | −0.732847 | 0.435811 | 0.253054 | 0.580651 | −0.182757 | 0.999985 | 945 |
| 17444 | `Cannot` | +0.157947 | +1.000000 | +0.217176 | −0.732847 | 0.371156 | 0.253054 | 0.681801 | −0.118102 | 0.999985 | 948 |
| 19152 | `Sorry` | +0.268237 | +1.000000 | +0.144344 | −0.732847 | 0.406514 | 0.253054 | 0.622499 | −0.153460 | 0.999985 | 943 |
| 32286 | `ĠSorry` | +0.250125 | +1.000000 | +0.159282 | −0.732847 | 0.385225 | 0.253054 | 0.656901 | −0.132170 | 0.999985 | 946 |
| 34655 | `ĠCannot` | +0.124054 | +1.000000 | +0.250215 | −0.732847 | 0.379668 | 0.253054 | 0.666516 | −0.126613 | 0.999985 | 940 |

### mode `centroid-orthogonal`

| id | token | cos_pre_c | cos_post_c | cos_pre_a | cos_post_a | n_pre | n_post | n_post/n_pre | n change | stored_c | bytes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 40 | `I` | +0.002484 | +0.680393 | +0.326144 | 0.000000 | 0.466851 | 0.172177 | 0.368804 | −0.294674 | 0.680263 | 944 |
| 358 | `ĠI` | −0.100872 | +0.680393 | +0.350257 | 0.000000 | 0.452987 | 0.172177 | 0.380092 | −0.280810 | 0.680263 | 942 |
| 9236 | `ĠEs` | +0.007700 | +0.680393 | +0.312672 | 0.000000 | 0.410930 | 0.172177 | 0.418992 | −0.238753 | 0.680263 | 943 |
| 17360 | `Es` | +0.101135 | +0.680393 | +0.265129 | 0.000000 | 0.435811 | 0.172177 | 0.395071 | −0.263635 | 0.680263 | 941 |
| 17444 | `Cannot` | +0.157947 | +0.680393 | +0.217176 | 0.000000 | 0.371156 | 0.172177 | 0.463893 | −0.198979 | 0.680263 | 940 |
| 19152 | `Sorry` | +0.268237 | +0.680393 | +0.144344 | 0.000000 | 0.406514 | 0.172177 | 0.423544 | −0.234337 | 0.680263 | 939 |
| 32286 | `ĠSorry` | +0.250125 | +0.680393 | +0.159282 | 0.000000 | 0.385225 | 0.172177 | 0.446951 | −0.213048 | 0.680263 | 938 |
| 34655 | `ĠCannot` | +0.124054 | +0.680393 | +0.250215 | 0.000000 | 0.379668 | 0.172177 | 0.453493 | −0.207491 | 0.680263 | 936 |

`cos_post_c = 0.680393` is `sqrt(1 − cos(c, â)²)` with cos(c, â) = −0.732847: the
centroid's off-axis part is all that survives, so the row keeps a positive compliance
reading and has no denial-axis component at all.

### mode `row-orthogonal`

| id | token | cos_pre_c | cos_post_c | cos_pre_a | cos_post_a | n_pre | n_post | n_post/n_pre | n change | stored_c | bytes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 40 | `I` | +0.002484 | +0.255466 | +0.326144 | 0.000000 | 0.466851 | 0.441323 | 0.945320 | −0.025527 | 0.255670 | 899 |
| 358 | `ĠI` | −0.100872 | +0.166351 | +0.350257 | 0.000000 | 0.452987 | 0.424292 | 0.936654 | −0.028695 | 0.166421 | 910 |
| 9236 | `ĠEs` | +0.007700 | +0.249343 | +0.312672 | 0.000000 | 0.410930 | 0.390326 | 0.949861 | −0.020604 | 0.249436 | 902 |
| 17360 | `Es` | +0.101135 | +0.306400 | +0.265129 | 0.000000 | 0.435811 | 0.420215 | 0.964213 | −0.015596 | 0.306336 | 908 |
| 17444 | `Cannot` | +0.157947 | +0.324858 | +0.217176 | 0.000000 | 0.371156 | 0.362297 | 0.976132 | −0.008859 | 0.324963 | 884 |
| 19152 | `Sorry` | +0.268237 | +0.377977 | +0.144344 | 0.000000 | 0.406514 | 0.402257 | 0.989528 | −0.004257 | 0.378209 | 883 |
| 32286 | `ĠSorry` | +0.250125 | +0.371599 | +0.159282 | 0.000000 | 0.385225 | 0.380307 | 0.987233 | −0.004918 | 0.371409 | 869 |
| 34655 | `ĠCannot` | +0.124054 | +0.317524 | +0.250215 | 0.000000 | 0.379668 | 0.367590 | 0.968190 | −0.012077 | 0.317466 | 880 |

This is the least destructive form: each row keeps its own direction and loses exactly
the component it had along â. The removed component's length is
`delta_norm = |cos_pre_a| · n_pre` (0.152260 for id 40, 0.0587-0.1587 over the eight),
and the norm falls as `n_post = sqrt(n_pre² − delta_norm²)` — a 1.1 %-6.3 % drop
(`n_post/n_pre` 0.9367-0.9895) — while `cos_post_a` is zero. `cos_post_c` still rises
for every id (0.0025 → 0.255 for `I`) only because the centroid shares the off-axis
bulk; it is not the target.

The shipped file's `stored_c` column differs from `cos_post_c` by at most 1.5e-5 in
`centroid` mode, 1.3e-4 in `centroid-orthogonal` and 2.3e-4 in `row-orthogonal`: the
Q8_0 grid stores a row that is not itself a stored row with a small rounding, which is
what the realized-cos columns below measure directly.

## What the file kept

Artifact: `verification/iq05b/Qwen2.5-0.5B-Instruct-IQ3_M.gguf`, 342,752,576 B,
sha256 `0b2e1747…32099`, 290 tensors. `token_embd.weight` is Q8_0, 151,936 rows × 896
values (28 blocks per row), and there is **no `output.weight`**: Qwen2.5-0.5B ties the
output head to the embedding, so the rewritten rows are the head rows and the embedding
rows at once (`tied: true`, named in every report). Tokenizer: gpt2/qwen2, 151,936
tokens, `tokens_sha256 a760a3ca…e1000`.

Commands (`$M` = the artifact, `$O` = the new one, `$m` = the mode):

    jdewitt vocab-centroid --weights "$M" --mode "$m" --json verification/vocab-centroid/dryrun-$m.json
    jdewitt vocab-centroid --weights "$M" --mode "$m" --out "$O" --apply --json verification/vocab-centroid/apply-$m.json
    jdewitt gguf-inspect "$O"          > verification/vocab-centroid/inspect-$m.json
    jdewitt gguf-tensor-diff "$M" "$O" > verification/vocab-centroid/diff-$m.json
    jdewitt gguf-revert "$O.copy" "$O.revert.npz" > verification/vocab-centroid/revert-$m.json

| | centroid | centroid-orthogonal | row-orthogonal |
|---|---|---|---|
| output sha256 | `a83a4fb0…97aeac` | `b6cdbc7e…665ce5` | `13598c4f…88f9f3` |
| size | 342,752,576 B | 342,752,576 B | 342,752,576 B |
| dry-run min cos (stored vs intended) | 1.0 | 1.0 | 0.9998 |
| differing byte ranges | 61 | 99 | 468 |
| **differing bytes** | **7,562** | **7,523** | **7,135** |
| fraction of file | 0.0022 % | 0.0022 % | 0.0021 % |
| rows changed / total | 8 / 151,936 | 8 / 151,936 | 8 / 151,936 |
| unchanged rows | 151,928 | 151,928 | 151,928 |
| changed set == denial id set | true | true | true |
| compliance rows changed | none | none | none |
| realized cos to intended target (min) | 0.999985 | 0.999985 | 0.999984 |
| `gguf-tensor-diff` | 289/290 bit-identical | 289/290 | 289/290 |
| diff rel_rms on `token_embd.weight` | 0.007394 | 0.006001 | 0.001786 |
| `gguf-inspect` | 290 tensors, same type census | same | same |
| revert | 61 tiles, `byte_exact: true` | 99 tiles, true | 468 tiles, true |

`apply.row_identity` is the row-granularity check the tensor diff cannot make: it reads
the output's raw stored bytes per row, and `changed_rows_equal_mapped_rows` is true in
all three modes — the changed rows are exactly the denial ids, the changed set has no
member outside it, and no compliance row moved. Per-row changed bytes are the `bytes`
column of the per-id tables above, and sum to the differing-byte totals (948+947+… =
7,562 for `centroid`, etc.); every differing byte lies inside `token_embd.weight`'s
144,643,072 bytes, so the other 289 tensors, the header and the metadata are
byte-identical by measurement, not by assumption.

The dry-run sample touches 224 Q8_0 blocks (8 rows × 28) = 15,232 of the tensor's
144,643,072 bytes (0.0105 %). The codec is the tensor's own stored type, so every tensor
offset and byte length stays put; the revert sidecar holds the measured pre-edit bytes
(10,310 / 10,879 / 16,395 B) and `gguf-revert` restores
`0b2e17471ccc784dfd7b042a81e34289a3accb75ccda7347598a329e0a932099` — the input's digest,
`byte_exact: true`. `cmp` against the source on the reverted copy finds the files
identical. The revert ran on a copy, so the applied artifacts and their sidecars stay on
disk for inspection.

Each apply is deterministic: re-running the same command on the same input wrote the
same sha256. Wall clock was 45 s (`centroid`), 48 s (`centroid-orthogonal`) and 68 s
(`row-orthogonal`) on this host, all in one process on the CPU.

## What this does not say — the expected negative

- **The id map is intact, so the model reads the denial ids compliance-like, and the
  continuation is still whatever the hidden plan was.** On the head side
  `logit(d) = h·W'_d`: in `centroid` mode every denial id now scores `h·c`, the same
  value for all eight and on the compliance side of the axis (cos −0.7328); in the
  orthogonal modes the id's score loses its projection on the shared denial direction.
  That changes which *opener* is cheap, not which *plan* the model was executing — the
  next tokens are planned by the same hidden state as before. A compliant opening glued
  to a refusal body is the outcome this card exists to record, and it is a negative
  result: an opener-level row edit is not a removal. It is **derived** from the algebra
  plus the recorded base token paths (the sibling card records base greedy rows opening
  on ids 40, 19152 and 17360); this run measured no continuation and asserts none.
- **Tied file: the edit is on both sides.** The same rows are input embeddings, so a
  prompt containing `I`, `Sorry`, `Cannot` or `Es` now embeds as the centroid (or as the
  row minus its axis component). In `centroid` mode that is a genuine collapse: the
  eight denial ids become one vector on the input side and one score on the head side,
  which is more semantic damage than "one opener is replaced". That consequence is
  inherent to this artifact's layout and is unmeasured here.
- **Eight rows, one artifact, one codec, one table.** The fidelity numbers are a
  property of Q8_0 storing a row on its own grid; a Q6_K/IQ artifact can move unedited
  blocks, which is why the row check is measured per file rather than assumed. The pair
  table is a design choice, not a policy. `As` is untouched by construction, and that is
  the only control this card carries for "nothing else moved" beyond the byte checks.
- **No donor, no model, no serve, no GPU, no token.** No abliterated/uncensored
  checkpoint was used or needed: the compliance centroid is the target's own rows. The
  twin-provenance table that the twin-transplant cards carry has no counterpart here by
  design. `removal_measured` is false in every report.

## Files

| file | what it is |
|---|---|
| `dryrun-<mode>.json` | the plan: groups, axis/centroid, per-row pre/post table, codec simulation |
| `apply-<mode>.json` | the shipped file: output digest, differing bytes, row identity, stored cosines, revert sidecar |
| `inspect-<mode>.json` | `gguf-inspect` on the output (header and tensor table read) |
| `diff-<mode>.json` | `gguf-tensor-diff` source vs output (289/290 bit-identical) |
| `revert-<mode>.json` | `gguf-revert` on a copy: `byte_exact: true`, original digest restored |
