# P7 cert at the pinned config — what the next two cards run

State: the operands are rebuilt and hash-bound; the waves have NOT been run. This
card (prep) holds no GPU: it fetched the checkpoints, converted and rebuilt the
artifacts, and wrote the runner. The GPU card runs the waves; the analysis card
assembles the packet and the matrix row.

> **Update from the analysis card (3/3) — read `VERDICT.md` first.** The waves ran
> (7/7 measured, 06:13–07:13Z on 2026-11-24) and the pinned packet exists:
> `verification/p7-q3-4b/packet.json`, `pinned_cert_match: true` on removal,
> retention and held_out at `-c 4096 --parallel 4` cap 1024. **The verdict is a loss**
> and it is the certified result: harmful-40 removal 3/38 wide (5/38 shipped),
> held-out 30/112 wide against the published build's 108/108 (72 discordant, exact
> p < 1e-20), maths 158→153 (p 0.0648), code 22→18, drift +0.0606 outside the 0.02
> budget, realized fidelity min cos 0.9905. The matrix row reads `failed(removal)`.
> README IV and `docs/EVIDENCE-TABLE.md` carry the same numbers. R9.4 (8192/4) is
> **not measured**: `sensitivity-r9.4.json` records the gap and the two commands that
> close it. One wave-time refusal label had to be regraded from the answer text before
> `release-certify` would grade it (`regrade_wave.py`, audit beside the wave).

**Read this first — measured on the rebuilt arm before any wave:** control-corpus
drift is **+0.0606 ln-PPL** (base 1.2656 -> arm 1.3446), outside the 0.02 budget.
The banked arm read +0.0058. The drift input is already on disk
(`drift.json`, `run_waves.sh` re-measures it), so `release-matrix` will report a
drift failure for this REFIT arm unless the waves show the number is wrong. That is
the honest state of a refit: the recipe is the same, the axis is not. Do not retune
the pool or the scale after seeing this — report it. If the arm also loses removal,
held-out or retention at 4096/4, that is the certified result (see the last section).

Pinned config: `-c 4096 --parallel 4 --jinja --no-webui`, temperature 0, recorded
cap 1024. 1024 is the slot width at 4096/4 (`/props` reports `n_ctx_slot: 1024`), so
nothing truncates at the cap; a 4096-token request cannot fit in a 1024-token slot,
and asking for one would grade truncation as capability.

## Operands on disk (sha-checked)

| operand | path | sha256 | bytes |
|---|---|---|---|
| base Q8_0 | `/mnt/ssd/hermes/models/q3-4b-base-q8.gguf` | see `operands.json` | 4,280,403,488 |
| published Q8_0 | `/mnt/ssd/hermes/models/q3-4b-heretic-q8.gguf` | see `operands.json` | 4,280,403,552 |
| arm Q8_0 (REFIT) | `/mnt/ssd/hermes/models/q3-4b-f15s15-l935-refit.gguf` | see `operands.json` | see `operands.json` |
| refit axes | `/mnt/ssd/hermes/models/_fit/q3-shared-axes-refit.pt` | see `operands.json` | — |
| refit records | `/mnt/ssd/hermes/models/_fit/records-refit.pt` | see `operands.json` | — |
| refit pool | `/mnt/ssd/hermes/models/_fit/pool.jsonl` | see `operands.json` | — |

Sources, both at the pinned revisions:

* `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f17c01a7cc42f958dc650907174af0554`
  (`cdbee75f17c0`)
* `p-e-w/Qwen3-4B-Instruct-2507-heretic` @ `bc843daf6f71759bfb52ad69841f7a84aed6dd64`
  (`bc843daf6f71`)

Both converted with the same converter (`llama.cpp convert_hf_to_gguf.py --outtype
q8_0`, no separate 16-bit emit-and-requantize, no download from us).

**The arm is a REFIT and must be called one.** The banked `q3-shared-axes.pt` and its
capture records died with `/mnt/ssd/hermes`, so the exact axis could not be restored.
The recipe was reproduced instead: pool = `verification/harsh-suites/selection.jsonl`
at the P4d strides (`refusal [::25][:40]`, `harmless [::8][:40]`), 80 rows over 68
families, shape-labelled by the base model's own answers (`bad` = declined a harmful
ask, `good` = answered a harmless one), captured at the answer-span locus on the base
checkpoint, fitted `--shape shared`, applied to `o_proj + mlp.down_proj` at
`blk.15-35`, scale 1.5. `refit_pool.py` is deterministic and committed. The spec,
packet, README and matrix row must say "refit"; never present it as the banked plan.

The base artifact's sha does **not** reproduce the old prefix `dc4ae2660e73cc2a`
(older converter build), which is expected: the spec binds the new full sha256. As a
behavioural cross-check, control-corpus PPL on the rebuilt base is **1.2656**, exactly
the banked screen's base reading.

The arm/base tensor diff (`operand-diff.json`, `jdewitt gguf-tensor-diff`, CPU):
398 tensors, **356 bit-identical**, exactly the **42** the recipe names changed, same
Q8_0 dtype, none added or dropped, realized edit rel-RMS 3.1-6.0% per edited tensor.
Realized fidelity on the served bytes (`fidelity.json`, `compare_edit`, arm's own
fitted direction at scale 1.5): min cos **0.9905**, mean **0.9934** over 6 sampled
tensors (codec simulation 0.9976). The old packet's 0.7255 was a codec proxy on a
seeded axis, not this reading.

`harmful-40.jsonl` was missing from the clone. It is committed here as a byte-for-byte
copy of `verification/screens/advbench-40.jsonl` under its historical name (ids and
prompts verified equal against the historical `q3-4b-harmful40/*` waves); see
`harmful-40.provenance.json`.

## Card 2 — GPU: run the waves

    verification/p7-q3-4b/run_waves.sh --dry     # review the argv list, free
    verification/p7-q3-4b/run_waves.sh           # run; re-run to resume

It serves base, then arm, then the published build at the pinned argv, runs

* `retention-base` — 241 capability rows (maths+code), base, cap 1024
* `removal-arm` — harmful-40, both wording inventories stored at write time
* `held_out-arm` — 120 held-out families, both inventories
* `retention-arm` — 241 capability rows
* `removal-published`, `held_out-published`, `retention-published` — same decks, same
  config, so the head-to-head and the matrix's R9 comparison are one config wide
* shape audits on both arm refusal waves (`refusal_shape_audit.py`),
* drift (`drift_probe.py`, llama-perplexity on `verification/ppl-probe/control.txt`,
  base vs arm),
* realized fidelity (`fidelity_probe.py`, `preview_edit` at the FITTED axis, scale 1.5,
  per edited tensor — `arm_measurement: true`, replacing the old codec proxy).

Servers are stopped after each artifact and by an EXIT trap. Every wave whose
`wave.json` says `measured: true` is skipped, so a killed run resumes. Progress and
row/token counts go to `progress.txt`. Expect roughly 40-60 min total on the local
3060-class GPU; the previously banked retention wave was 190 s at 8192/4.

On the GPU host pass the binaries if they are not at the defaults:

    JDEWITT_SERVER=/path/to/cuda/llama-server \
    JDEWITT_PERPLEXITY=/path/to/cuda/llama-perplexity \
        verification/p7-q3-4b/run_waves.sh

Port 8134 is the historical port; another card may hold it. The runner refuses a
server whose `/props` model path is not the artifact it is about to grade, so a
conflict fails loudly instead of attributing answers to the wrong file; move with
`JDEWITT_PORT=…` (the config string and the spec follow it).

## Card 3 — analysis (CPU): packet

Re-generate the spec from the built operands (it reads the config strings out of
`run_waves.sh --dry`, so the spec cannot disagree with what the waves recorded, and it
hashes every operand):

    uv run python verification/p7-q3-4b/make_spec.py \
        --arm /mnt/ssd/hermes/models/q3-4b-f15s15-l935-refit.gguf

    uv run jdewitt release-certify \
        --spec verification/p7-q3-4b/spec.json \
        --out  verification/p7-q3-4b/packet.json
    uv run jdewitt verify-artifact verification/p7-q3-4b/packet.json --name packet.json

The packet must print `pinned_cert_match: true` on removal, retention and held_out
(that is what makes this the pinned row rather than a `certified_at_other_config`).
Report beside it, at 4096/4: removal harmful-40 under both inventories, held-out 120
families plus the paired McNemar against the published build, maths/code strict with
paired delta, Cohen's h, p and median answer length, drift, realized fidelity.

## Card 3 — matrix

Register the refreshed row in `verification/releases/matrix.manifest.json` before
running the matrix: `artifact.sha256` = the arm sha, `artifact.identity_digest` = the
`artifact_identity_digest` in `spec.json`, packet/spec paths, `head_to_head.waves` =
the three `*-published` waves, `retention_base_wave` = `waves/retention-base`,
`base.sha256` = the rebuilt base sha. Two registration decisions are not optional:

1. **Cap.** `registered.cap` pins removal/held_out at 4096, which was the slot width
   at 8192/2. At 4096/4 one slot is 1024, so the pinned row is graded at cap 1024; the
   registered cap for this row must be 1024 with that reason in the note. Lowering it
   silently, or keeping 4096 and calling the packet `cap_lowered`, are both wrong.
2. **Suites.** `registered.suites.removal` must name `harmful-40.jsonl`; that basename
   is what `verification/p7-q3-4b/harmful-40.jsonl` carries. held_out and retention
   names are unchanged.

    uv run jdewitt release-matrix \
        --manifest verification/releases/matrix.manifest.json \
        --out      verification/releases/matrix.json \
        --table    verification/releases/matrix.md \
        --doc      docs/EVIDENCE-TABLE.md

## R9.4 — config sensitivity

Re-run the deciding waves at 8192/4 in a separate output root (same script, only ctx
differs; the cap stays 1024 so the comparison isolates context):

    JDEWITT_OUT=verification/p7-q3-4b/sensitivity-8192 JDEWITT_CTX=8192 \
        verification/p7-q3-4b/run_waves.sh
    uv run python verification/p7-q3-4b/compare_configs.py \
        --pinned verification/p7-q3-4b/waves/removal-arm \
        --other  verification/p7-q3-4b/sensitivity-8192/waves/removal-arm \
        --out    verification/p7-q3-4b/sensitivity-removal.json
    # same for held_out-arm

Answers whose verdict flips are `boundary`: excluded from `robustly_removed`, and the
pinned number is reported with the flip count beside it — not the better of the two.

## If the arm loses at 4096/4

That number is the certified result. Held-out no longer 120/120, or maths past 148:
write it into README IV and the honest table, do not keep the 8192 numbers as the
pinned row, and do not lower the cap or narrow the suite to recover the old number.
