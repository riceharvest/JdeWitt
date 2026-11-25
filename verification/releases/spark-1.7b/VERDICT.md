# P7 verdict — Spark-X2.6-1.7B Q4_K_M, REFIT `spark-1.7b-q4km-s1.0` at the pinned 4096/4

**Verdict: the codec floor is cleared on the artifact a user serves — 56 of 56
edited tensors realize cos ≥ 0.9687 at Q8_0 — and the row is still NOT certified.
Removal is 9/15 under the wide inventory (12/15 shipped) over the 15 of 40 rows the
1024-token slot let finish; 25 rows truncated and are ungraded. The row is
`failed(removal)`, not `failed(codec)`, and the two are reported separately.**

Packet: `verification/releases/spark-1.7b/packet.json` (generated from saved waves
only). Matrix row: `failed(removal)` — removal and held-out, with fidelity and
drift clearing their gates.

    uv run jdewitt release-certify --spec verification/releases/spark-1.7b/spec.json \
        --out verification/releases/spark-1.7b/packet.json
    uv run jdewitt verify-artifact verification/releases/spark-1.7b/packet.json --name packet.json

`pinned_cert_match` is true on removal, retention and held_out: all three waves
were served at `-c 4096 --parallel 4 -dev Vulkan1 --jinja --no-webui`, temp 0.

## The numbers

| axis | measured | gate |
| --- | --- | --- |
| removal harmful-40, shipped / wide | **12/15 / 9/15** | 40 removed — failed |
| removal harmful-40, ungraded | **25 of 40 truncated** at the 1023-token slot | — |
| base at the same argv, cap 4096 | 40/40 ok, 40/40 refusals, 0 removed | — |
| removal held-out 120 rows / 120 families | **29/40 / 18/40**, 80 ungraded, 40 families graded | 120 removed — failed |
| 24-family probe (not an axis) | 11/11 removed both inventories, 13 ungraded | probe only |
| retention maths | 68 → 67 of 180 pairs, p 1.0, median 3 chars (base 3) | — |
| retention code | 44 → 44 of 59 pairs, p 1.0, median 1,478 chars (base 472) | — |
| drift, control ln-PPL | **+0.0069** against 0.02 | inside |
| realized fidelity, merged bytes | min cos **0.9687**, mean 0.9755, 0 of 56 below 0.90 | cleared |
| codec table, per type | Q6_K 0.8004 (refused), Q8_0 0.9861 (chosen), F16 1.0 | 0.90 floor |
| byte cost | 1,307,821,248 vs base 1,107,458,592 — **+18.1%** | printed, not hidden |

## Why it is not a pass

1. **The removal count is short and the denominator is small.** 9 of the 15 rows
   that finished are removals under the wide inventory (12 shipped), against a
   registered floor of 40. The base at the same argv refuses 40/40, so the base is
   the zero-removal reference and the arm does move — but 25 rows hit the 1023-token
   slot and are scored as neither pass nor failure. At `-c 4096 --parallel 4` the
   arm's 4.2 k-char answers do not fit, so the floor is unreachable by construction
   on this arm and this config. The packet prints `9/15`, never `9/40` and never
   `40/40`.
2. **The held-out axis has the same shape**: 18/40 wide over 40 graded rows
   (29 shipped), 80 of 120 truncated, 40 of the suite's 120 families measured.
3. **The codec is NOT the failure** — and that is the lineage's deciding result.
   The Q6_K step is the codec floor the roadmap records: min cos 0.8004 on the
   pristine codec table (attention 0.8004–0.8116) and 0.7766 on the one
   merged-byte writer probe; the Q8_0 type choice clears it at 0.9687 without
   touching the scale. Scale stayed 1.0; the 6.61/+10.2-nat counterexample was not
   approached. The price is bytes: +200,362,656 B (+18.1%), because the 28
   `ffn_down` and 28 `attn_output` tensors move to Q8_0.

## What this arm establishes (R8 / R9)

- It is a **REFIT**: the shared axis is fitted from the committed p4d capture
  (`runs/p4/records.pt`, on the bfloat16 HF source the f16 twin was converted from)
  and is bitwise reproducible; the manifest row and the packet both say `refit`.
- The measurement runs on the **merged gguf bytes** — the release artifact the user
  would serve — and the shape audits carry that artifact's identity digest
  `9eb6afc2…`. No adapter was served, and no adapter curve is ranked.
- The edit is an **in-place-style edit of the served bytes**: the base file is
  untouched, the arm sits beside it, and reverting is deleting the arm. Every one
  of the 170 unedited tensors is byte-identical.
- R6: wording removal is not delivery. The shape audit reads 35 of 40 harmful-40
  answers as procedural and 5 rows as still declining despite the checker calling
  them removed; no delivery claim is made.
- R9: the config is the pinned one. No second config was served, so this row makes
  no config-sensitivity claim; a wider-slot config would be
  `certified_at_other_config`, not a substitute.

## What would change the verdict

- A registered config whose slot fits this arm's answers (2 k tokens or more) so
  that more than 15/40 and 40/120 rows can be graded. That is a different
  ctx/parallel — a different row, not a re-read of this one.
- A hand read of the harmful-40 answers, which is the only route to a delivery
  claim; the shape audit is a heuristic.

Neither is done here. `failed(removal)` stands, with the codec floor cleared and
every denominator printed beside its count.
