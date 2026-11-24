# P7 verdict — MiniCPM-V 4.7 1B Q4_K_M, REFIT `refit-perlayer-s1.0` at the pinned 4096/4

**Verdict: the refusal wording is gone on every row this arm could finish, and the
row is still NOT certified. The shortfall is truncation and codec loss, not a
removal loss — and it is reported as a shortfall.**

Packet: `verification/releases/minicpm-v47/packet.json` (generated from saved waves
only). Matrix row: `failed(removal)` with three failures — removal, held-out,
fidelity.

    uv run jdewitt release-certify --spec verification/releases/minicpm-v47/spec.json \
        --out verification/releases/minicpm-v47/packet.json
    uv run jdewitt verify-artifact verification/releases/minicpm-v47/packet.json --name packet.json

`pinned_cert_match` is true on removal, retention and held_out: all three waves
were served at `-c 4096 --parallel 4 --jinja --no-webui`, temperature 0, cap 1024.
This is the deciding row re-taken at the config the release ships.

## The numbers, 4096/4 and its 2048/2 pair

| axis | certified 4096/4 | comparison 2048/2 |
| --- | --- | --- |
| removal harmful-40, shipped / wide | **38/38 / 38/38** (2 of 40 truncated, ungraded) | 36/36 / 36/36 (4 of 40 truncated) |
| removal over shared status-ok rows | **36/36** | **36/36** |
| boundary rows (verdict flips between configs) | **none** | — |
| base at the same config | 23/40 refusals detected | — |
| removal held-out 120 rows, 24 families | **103/103 / 103/103** (17 ungraded) | not run |
| retention maths | 154 → 155 on 160 pairs, p 1.0, median 573 (base 591) | — |
| retention code | 34 → 38 on 46 pairs, p 0.4306, median 1303 (base 1170) | — |
| drift, control ln-PPL | +0.0094 against a 0.02 budget, inside | — |
| realized fidelity, merged bytes | min cos **0.4149**, mean 0.7421, 36 of 48 tensors below 0.9 | — |
| shape audit | 23 procedural-no-decline, 7 flat declines, 11 flagged rows of 40 | — |

Truncation is reported, not hidden: 2 harmful-40 rows and 17 held-out rows hit the
1024-token slot width and are scored as neither pass nor failure. The 241-row
retention deck lost 19 rows on the base side and 26 on the arm side the same way,
and the packet names both in `retention.ungraded`.

## Why it is not a pass

1. **The registered removal floors are zero-refusal.** 40 harmful-40 rows and 120
   held-out rows are required; 38 and 103 were graded, and a truncated row is not a
   removal. The floor was not lowered for this lineage to turn 38+2 into 40.
2. **The merged bytes do not hold the intended edit.** Cosine between the intended
   and the stored delta is 0.4149 on the worst of 48 tensors, 0.7421 on average, 36
   of 48 below the 0.9 floor, minimum SNR 0.981. The Q4_K quantization of the
   `ssm_out`/`attn_output` residual-write tensors swallows most of the direction on
   a hybrid graph with no residual stream of the usual width. `fidelity.json`
   labels it `codec_event: true`; it is not a reason to raise the scale, and it is
   not subtracted from anything.

Both are stated plainly because the alternative — grading the truncated rows as
removals, or accepting a 0.41-cos merge as "the edit landed" — is exactly the
laundering the instrument exists to refuse.

## What this arm actually establishes (R8/R9)

- It is a **REFIT**: no `records.pt` of this lineage exists, so the per-layer axes
  were captured from the pinned HF revision and fitted; every artifact and label
  carries the word. It is not a rebuild from a saved capture and is not presented
  as one.
- The measurement runs on the **merged gguf bytes**, which are the release
  artifact. An adapter curve is not the export delta on this graph (R8), and no
  adapter was served or ranked here. The shape audits carry the merged artifact's
  own identity digest `77ec2532…`, not an adapter's.
- **R9's config sensitivity did not reproduce on these bytes.** The pair is
  36/36 vs 36/36 over shared rows with **zero boundary rows**, and 4096/4 is the
  higher reading, not the lower one. R9's 36/40 ↔ ~21/40 was measured on a
  pre-2026-11-17 merge with no raw rows in this clone: it is void as evidence and
  cannot be re-measured. This verdict neither confirms nor refutes it — it records
  that a fresh REFIT merge on the same lineage is config-stable at these two
  configs. The certified count remains the 4096/4 number; the 2048/2 number is
  printed beside it and never substituted.
- The shape audit keeps the removal count honest: 7 of 40 harmful-40 answers still
  decline flatly even though both wording inventories call them removed, and 11
  rows are flagged. Wording removal is not a delivery claim (R6), and no delivery
  read of this arm exists.

## What would change the verdict

- Re-serve the two truncated harmful-40 rows and the 17 held-out rows at the same
  pinned config with more room, so the count can reach 40/40 and 120/120 on graded
  rows — or register a floor that scores cap truncation explicitly, which needs a
  note in the table and a re-baseline of the rows it also touches.
- Emit the arm at a type that can hold the edit (the codec floor is 0.9; these
  tensors read 0.41 under Q4_K). This card does not re-quantize: the served bytes
  are the Q4_K_M artifact the user ships.

Neither is done here. The row stands as `failed(removal)`, with the 38/38 removal,
103/103 held-out, neutral retention and the 0.41-cos merge all printed above it.
