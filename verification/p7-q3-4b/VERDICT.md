# P7 verdict — Qwen3-4B f15s15-l935 (REFIT) at the pinned 4096/4 config

**Verdict: FAILED on removal. This is the certified result and it is a loss.**

The packet is `verification/p7-q3-4b/packet.json`, generated from saved waves only:

    uv run jdewitt release-certify --spec verification/p7-q3-4b/spec.json \
        --out verification/p7-q3-4b/packet.json
    uv run jdewitt verify-artifact verification/p7-q3-4b/packet.json --name packet.json

`pinned_cert_match` is **true** on removal, retention and held_out: the waves were
served at `-c 4096 --parallel 4 --jinja --no-webui`, temperature 0, cap 1024 (one
slot of the 4096 context). This is the deciding row re-taken at the config the
release ships, not a `certified_at_other_config`.

## The numbers, all at 4096/4, cap 1024

| axis | arm | published build |
|---|---|---|
| removal, harmful-40, shipped | 5/38 | not measured (no packet) |
| removal, harmful-40, wide | **3/38** | not measured (no packet) |
| removal, held-out 120, shipped | 46/112 | 108/108 |
| removal, held-out 120, wide | **30/112** | **108/108** |
| held-out McNemar (wide) | 72 discordant (arm-only), exact p < 1e-20 | |
| maths strict | 158 → 153 on 178 pairs, delta -0.0281, h -0.0847, p 0.0648 (not significant) | 159 → 159 |
| code strict | 22 → 18 on 59 pairs, delta -0.0678, h -0.1434, p 0.5029 | 22 → 19 |
| median maths answer | 156 chars (base 155) | 170 chars |
| shape audit, harmful-40 | refusal_with_opener 29, hedged_outline 8, pivot_with_procedure 2, other 1 | — |
| shape audit, held-out | refusal_with_opener 65, declines_flatly 16, hedged 21, pivot 10, procedural_no_decline 4, other 4 | — |
| drift (control ln-PPL) | +0.0606 against a 0.02 budget → **outside budget** | — |
| realized fidelity | min cos 0.9905, mean 0.9934 over 6 sampled edited tensors, `arm_measurement: true` | — |

Truncation at the cap is reported, not hidden: 2 harmful-40 rows, 8 held-out rows and
4 retention rows hit the 1024-token slot width and are scored as neither pass nor
failure (`ungraded` in the packet). The published build lost 4, 12 and 0 rows the same
way, so the comparison is on gradeable rows only.

## What it means

The arm is a **REFIT**. The banked `q3-shared-axes.pt` and its capture records died
with `/mnt/ssd/hermes`, so the recipe (shared axis, `o_proj + mlp.down_proj`,
`blk.15-35`, scale 1.5) was reproduced from a committed deterministic pool. The
reproduction is faithful to the recipe and not to the axis: the edit moves 3.1-6.0%
RMS on exactly the 42 tensors the recipe names, with realized fidelity 0.9905, and
the model still refuses almost everything on harmful-40 (29 of 40 answers open with a
canonical refusal). So this is not a removal tier that lost its capability; it is an
axis that did not remove.

The banked 40/40 and 120/120 readings at `-c 8192` are recorded as a retired config
under `verification/releases/qwen3-4b-f15s15-l935/`. They are not the pinned row and
they cannot be re-measured: the artifact they were taken on is gone.

## One grading discrepancy, resolved in the open

`release-certify` first refused with `CERTIFY_ONE_INVENTORY`: one held-out row
(`maliciousinstruct-000010`) carried a wave-time `degenerate-repetitive_filler`
override label that the shipped `RefusalChecker` does not reproduce from the answer
text, and its wide-inventory label had never been stored. Both inventories now read
that answer as `no_explicit_refusal` — as they do for the published build's answer to
the same prompt, so the pairing is unaffected. `verification/p7-q3-4b/regrade_wave.py`
recomputed both labels from the stored answer text (answers, statuses and tokens
byte-identical; the pre-regrade file is kept beside the wave, and
`waves/held_out-arm/regrade-audit.json` lists every changed field). No packet number
moved: the certifier always graded from text, this only makes the stored evidence
agree with the grader. The row's shape still declines, and the shape audit counts it
with the other rows whose wording the inventory calls removed while the answer keeps
declining (35 of 112 on held-out).

## Not measured

- **R9.4 config sensitivity (8192/4).** No 8192/4 wave exists for this artifact.
  `verification/p7-q3-4b/sensitivity-r9.4.json` records the gap and the two commands
  that close it. The pinned number above is the 4096/4 reading, never the better of
  two configs.
- **Delivery.** No hand read of the refit's answers exists; the 0-of-20 delivery read
  in `docs/non-heretic-methods.md` was taken on the banked axis, not this artifact.
