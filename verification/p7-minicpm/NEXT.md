# P7 — MiniCPM-V 4.7 1B, Q4_K_M: what is prepared, and the commands that finish it

This directory is the prep card's output. It contains no wave and no packet: the
wave card runs `run_waves.sh` on the GPU box, and the analysis card runs
`release-certify` / `release-matrix` on saved files. Everything below is the
handoff, in the order the two cards need it.

## What is on disk and verified (no GPU used)

| file | what it is | verified here |
| --- | --- | --- |
| `operands.json` | every pinned path + digest: base gguf, HF checkpoint, the three suites, the refit recipe, the discovered tensor inventory, the caps | base `sha256 67dd31f5…40be` recomputed on this box and matched `verification/minicpm-v47/summary.json`; Q5_K_M / bf16 siblings recorded, `used_by_this_cert: false` |
| `fit-suite.jsonl` | 60 rows (40 refusal × 8 families, 20 harmless) for the refit capture | `sha256 e2e83db8…1df4`, recut by `recut_suites.py` |
| `heldout-24f.jsonl` | 120 rows over 24 held-out families, disjoint from advbench-40 | `sha256 9eb470f9…100a`, recut by `recut_suites.py` |
| `recut_suites.py` | the executable form of the suites' selection rule + the disjointness check | `PASS` on both digests; it is the reason the rule is honest (the prose rule in `suites.provenance.json` left two clauses implicit and did not reproduce the files — the script now states and checks them) |
| `cert_inputs.py` | translates the measuring tools' reports into the exact JSON `release-certify` reads (`fidelity`, `drift`, `audit-identity`, `spec`) | `--self-check` passes 12/12 on synthetic fixtures |
| `run_waves.sh` | the wave driver: operands, refit, merge, fidelity, drift, every wave, audits, config pair, spec | `bash -n` clean; `--dry` prints the full argv list |
| `selftest.py` | builds a synthetic measurement set from the real suites and drives `cert_inputs → release-certify → cert_config_pair`, plus an R8 refusal | 16/16 checks pass; it writes only under `--out` and its packet says `synthetic-selftest`, so it can never be read as evidence |

`ruff check jdewitt scripts` is green (it covers `cert_config_pair.py`, the new
`scripts/` tool this cert uses for the R9.4 pair).

The axis is a **REFIT**: there is no capture (`records.pt`) of this lineage on this
box — the only `.pt` near this pipeline is a unit-random probe axis from an
earlier merge dry-run, which is a merge-path probe and never a fit. `run_waves.sh`
step 2 therefore captures from the pinned HF revision (B70; `jdewitt capture`
refuses CPU by design), fits `spanfit axes --shape perlayer`, and every artifact
it writes carries the word REFIT in the edit text, so no reader can mistake it for
a rebuild from a saved capture.

## The wave card (GPU) — one command

    bash verification/p7-minicpm/run_waves.sh

It is long (refit capture + merge + 6 waves on a 1B model) and resumable: a step
whose output exists is skipped, so a killed run continues where it stopped.
`--force` redoes everything. `--dry` starts no server and writes nothing. Every
server it starts is stopped when its wave group ends and again by the `EXIT` trap.

    ARTIFACT_DIR=/mnt/ssd/p7-minicpm      # merged gguf, axes, records, dequant cache
    SERVER=/home/dario/llama.cpp-minicpm47/build-cpu/bin/llama-server
    DEVICE_ARGS="--device none"           # CPU build; a Vulkan server may be substituted
    PORT=8134  SCALE=1.0  CAP_REMOVAL=1024 CAP_HELDOUT=1024 CAP_RETENTION=1024

Outputs: `waves/*/predictions.jsonl` + `wave.json`, `fidelity.json`, `drift.json`,
`shape-audit-removal.json`, `shape-audit-heldout.json`, `comparison.json`,
`spec.json`, `progress.txt`, `logs/*.log`, `merge.json`.

The waves, in order:

1. base @4096/4 — harmful-40 and retention (the paired base side of the 241-row
   retention stat; the base's own removal count is printed as the pre-edit rate)
2. merged @4096/4 — harmful-40, held-out 120, retention 241  ← **the pinned cert run**
3. merged @2048/2 — harmful-40  ← the R9 comparison run, same bytes, same cap

`-c` is the TOTAL context and `--parallel` splits it: at 4096/4 one request can
generate at most 1024 tokens, so every cap is 1024 and the comparison run at
2048/2 also yields 1024 per slot. The pair differs only in ctx/parallel, which is
what R9.4 wants.

## The analysis card — exact commands

1. Register the merged artifact in the manifest (the row exists but has no
   artifact). In `verification/releases/matrix.manifest.json`, lineage
   `minicpm-v47`, set `artifact` from `verification/p7-minicpm/spec.json`:

       sha256          = spec.artifact.sha256          (the merged gguf)
       identity_digest = spec.artifact_identity_digest (computed over sha + bytes + recipe)
       quant_type      = "Q4_K_M"
       scale           = 1.0
       bytes           = spec.artifact.bytes
       edited_tensors  = spec.arm.edited_tensors       (48)
       footprint       = spec.arm.edited_tensor_footprint
       base.sha256     = spec.base.sha256
       base.path       = "/mnt/ssd/p7-minicpm/base.gguf"   (hardlink of the pinned operand)

   `head_to_head` stays null: no competitor build of this lineage was measured on
   this box, and the old 36/40 (pre-2026-11-17, no raw rows) is void as evidence —
   do not register it as a reference side.

2. Build the packet and verify it:

       uv run jdewitt release-certify \
         --spec verification/p7-minicpm/spec.json \
         --out  verification/releases/minicpm-v47/packet.json
       uv run jdewitt verify-artifact \
         verification/releases/minicpm-v47/packet.json --name packet.json

3. Count the matrix:

       uv run jdewitt release-matrix \
         --manifest verification/releases/matrix.manifest.json \
         --out     verification/releases/matrix.json \
         --table   verification/releases/matrix.md \
         --doc     docs/EVIDENCE-TABLE.md

4. Put the pair in the packet's README (the packet has no comparison field):
   quote `comparison.json`'s `n_m_a` as **the certified removal count** and
   `n_m_b` beside it, never in place of it, with the boundary ids and
   `robustly_removed_a`. A row whose wide-inventory verdict flips between the two
   configs is `boundary` and is excluded from robustly-removed (R9.4).
   Both sides are counted over shared **status-ok** rows — the same population
   `release-certify` grades — so `n_m_a` has the packet's denominator; any row
   dropped for status or for being on one side only is named in `excluded_rows`
   (a truncated answer is never a removal). `scripts/cert_config_pair.py
   --self-check` proves that with a truncated row in its fixture.

## Read this before running `release-matrix`: the registered floors are q3's, not this lineage's

I ran `release-matrix` against a manifest registering the synthetic selftest packet
(`verification/p7-minicpm/selftest.py`) so this is measured, not guessed. The
minicpm row comes back `invalid(...)` with exactly four problems, none of them
about the measurement:

    cap_lowered       removal   served at cap 1024, the registered cap is 4096
    cap_lowered       held_out  served at cap 1024, the registered cap is 4096
    suite_substituted removal   packet covers 'advbench-40.jsonl',   registered 'harmful-40.jsonl'
    suite_substituted held_out  packet covers 'heldout-24f.jsonl',   registered 'q3-holdout-wider.jsonl'

Why:

- **The cap floor cannot hold at the pinned config.** `matrix.manifest.json`
  registers `cap: {removal: 4096, held_out: 4096, retention: 1024}` while its own
  `pinned` block is `ctx 4096 / parallel 4 / cap 1024`. `-c 4096 --parallel 4`
  gives each sequence 1024 tokens, so **no** wave at the pinned config can be
  graded at 4096. q3 met that floor only because its removal wave was served at
  `-c 8192 --parallel 2` (4096 per slot) — which is exactly why its row reads
  `certified_at_other_config`.
- **The suite floors name q3's files.** The registered removal suite
  `/mnt/ssd/hermes/cache/scratch/probes/harmful-40.jsonl` does not exist anywhere
  in-clone (the q3 packet records that as an evidence gap), and the registered
  held-out suite is `verification/screens/q3-holdout-wider.jsonl`, which is 120
  rows over **120** families with **15** prompts shared with advbench-40 — it
  cannot serve as this card's "24 held-out families, disjoint from the removal
  screen" suite. Substituting the registered file to clear the check would
  certify a different suite than the one the card specifies.

The honest fixes, in order of preference:

- **preferred: make the registered floors per-lineage.** Let a manifest entry
  carry optional `cap` / `suites` / `suite_rows` overrides that
  `_count_problems` uses before the global `registered` block (a handful of lines
  in `jdewitt/matrix.py`), and register this lineage's real operands:
  `cap {removal: 1024, held_out: 1024, retention: 1024}` with the reason
  "per-slot limit of the pinned 4096/4 config", `suites.removal
  verification/screens/advbench-40.jsonl`, `suites.held_out
  verification/p7-minicpm/heldout-24f.jsonl`. The q3 row keeps its own floors and
  stays as it is.
- otherwise: change the global `registered` block and re-baseline the q3 row in
  the same commit, saying in the table why the floor moved. Do not lower a global
  floor for a lineage without that note — a lowered cap floor lets every lineage
  grade truncated removal rows as removal.
- do not serve the removal wave at a bigger ctx to clear the floor: that changes
  the pinned config, makes `pinned_cert_match` false, and contradicts the card.

`registered.pinned` already says `cap 1024`, so the per-lineage reading is also the
one the manifest's own pinned block implies.

## If 4096/4 reads ~21/40

That is the certified number and the package says so (METHODOLOGY R9: the same
bytes read 36/40 at 2048/2 on this lineage pre-2026-11-17, and that number has no
raw rows in the clone). The honest table carries the pair; the README quotes
4096/4 first, 2048/2 beside it, and lists the boundary ids. If any README or
doc claims a removal count, put it in the spec's `expected` block before
certifying so the packet's `disagreements` array records the difference instead
of hiding it.

## R8 / R9 bookkeeping this dir enforces

- No adapter is served anywhere in this loop: the merge writes the gguf bytes and
  those are what every wave and the packet name (R8).
- The shape audits are run on the **merged** waves' predictions and are stamped
  with the artifact identity digest `release-certify` computes from the spec
  (R8) — the audit file itself carries `artifact_identity_digest`.
- Every wave records its full llama-server argv; `release-certify` refuses a wave
  whose recorded `serve_config` differs from its spec slot (R9).
- `comparison.json` is re-graded from saved answers with the shipped checker, so
  the pair cannot be produced by two different graders.
