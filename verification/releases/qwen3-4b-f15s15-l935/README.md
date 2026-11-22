# qwen3-4b-f15s15-l935 — Phase 7 certification packet

One arm, both axes, one serve config per axis. `packet.json` is generated; do not
hand-edit it.

    uv run jdewitt release-certify \
      --spec verification/releases/qwen3-4b-f15s15-l935/spec.json \
      --out  verification/releases/qwen3-4b-f15s15-l935/packet.json
    uv run jdewitt verify-artifact \
      verification/releases/qwen3-4b-f15s15-l935/packet.json --name packet.json

The command reads saved waves only: no model, no GPU, no network. It refuses
(named non-zero, no packet written) when an axis is missing, when a wave's
recorded `serve_config` differs from its spec slot (R9), when a wave is not
measured or has zero valid rows, when a removal number carries only one wording
inventory, when retention has no paired stat or no median answer length, or when
a shape audit belongs to another artifact (R8).

## What the packet says

- Removal, harmful-40: **40/40** under the shipped inventory and **40/40** under
  the wider inventory (Wilson lower 0.9124).
- Removal, held-out 120 families: **120/120** under both inventories, **120**
  families; against the published build's **120/120**, **0 discordant** rows,
  exact McNemar **p = 1.0**.
- Shape verdicts beside the removal numbers (R6: wording removal is not a
  delivery claim): 117 flat declines, 20 pivot-with-procedure, 14
  pivot-no-procedure, 4 procedural-no-decline, 5 other, over 160 audited rows.
- Retention, maths: **159 -> 148** on 179 paired rows, paired sign-flip
  **p = 0.0078**, Cohen's h -0.1768, median answer **68** chars against the base
  wave's **155**.
- Retention, code: **20 -> 15** on 61 paired rows, p = 0.4023.
- Drift: control delta +0.0058 ln-PPL inside a 0.02 budget.
- Realized fidelity: Q8_0 codec probe at the arm's footprint gives min cos
  0.7255, mean 0.847 over 4 sampled tensors.

## Two honest disagreements with README IV, kept visible

The packet's `disagreements` array carries both; neither is rounded away.

1. README IV says the base median maths answer is **175** characters; the
   in-clone `q3-4b-retention/q3base` wave measures **155** (upper median over
   paired, status-ok maths rows). 175 is exactly the `q3oracle` (published
   build) wave's upper maths median, so the quoted base number appears to have
   been taken from the published build rather than the base wave.
2. Because the collapse flag is a ratio to the measured base median, it is
   **false** at 155 (68 is 44% of 155). At the README's 175 it would be true.

Removal 40/40, held-out 120/120, maths 159->148, p 0.0078 and median 68 all
reproduce exactly.

## Pinned cert config

`pinned_cert_argv` is the R9 cert config (`-c 4096 --parallel 4`). The packet
prints `pinned_cert_match=false` for the removal axis (the in-clone wave is
`-c 8192 --parallel 2`, cap 4096) and for retention/held-out (`-c 8192
--parallel 4`, cap 1024). That false is the honest state: these waves are not
the pinned cert run, which is the separate gated card.

## Artifact identity

`/mnt/ssd/hermes/models/` no longer exists, so the GGUF cannot be hashed here.
The packet therefore records `artifact_sha256: null` plus an
`artifact_identity_digest` over the base sha prefix (`dc4ae2660e73cc2a`) and the
normalized spanfit build recipe, and says so in `evidence_gaps`. It binds the
served artifact by identity, never by a path that happens to exist on this box.
`base_sha256` is the recorded 16-hex prefix; the full digest was never written
in-clone.

The fidelity input is a codec probe on the available Q8_0 artifact at this arm's
footprint and scale (`arm_measurement: false`), not a `preview_edit` run on the
deleted arm artifact. It measures what the codec stores; it does not claim to
measure the fitted direction. Replace it with a real arm reading when the
artifact is rebuilt.

`drift.json` is a provenance excerpt of
`verification/ppl-probe/screen-2026-11-18.json` (`artifacts.f15s15.delta_control`).
