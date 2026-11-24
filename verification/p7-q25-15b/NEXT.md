# P7 qwen2.5-1.5b — what the analysis card runs next

This directory is the preparation for the qwen25-1.5b Phase 7 row. It holds the
served operands (bound by digest), the suites, the spec, and one script that runs
every wave. Nothing here has been served: this card ran on CPU with no GPU and no
wave. The wave run and the certify run are the two cards that follow.

## 1. What is on disk

Artifacts (~5 GB, gitignored, stable path — the same convention as the q3 cert's
`/mnt/ssd/hermes/models`):

    /mnt/ssd/models/p7-q25-15b/artifacts/base-q8.gguf        5bfdb201668c8f41…  1646572800  Q8_0
    /mnt/ssd/models/p7-q25-15b/artifacts/published-q8.gguf   4067fa4e3534e929…  1894534048  Q8_0
    /mnt/ssd/models/p7-q25-15b/artifacts/arm-refit-q8.gguf   584ec48f9a05a9f3…  1646572800  Q8_0
    /mnt/ssd/models/p7-q25-15b/records/refit-records.pt      CPU capture, 80 fit rows
    /mnt/ssd/models/p7-q25-15b/records/refit-axes.pt         shared axis [28,1536]
    /mnt/ssd/models/p7-q25-15b/records/refit-capture.json    capture report

In the repo:

    operands.json        every artifact + HF source bound to its sha256 and recipe
    suites/              harmful-40.jsonl (40 rows / 20 advbench families),
                         holdout-24f.jsonl (24 rows / 24 families),
                         fit-pool.jsonl (80 rows: 40 refusal + 40 harmless), staging.json
    spec.json            the cert spec: argv, suites, waves, audit/drift/fidelity paths
    run_waves.sh         every wave, in order, resumable, CPU, stops its own servers
    scripts/p7_q25_{suites,refit,operands,spec,fidelity,drift}.py   how the above was made

**The arm is a REFIT, not SAS@1.5.** The original `q25-v4-dir000.gguf` adapter is
not on this box (searched). The shared axis was refit from a CPU capture of the base
(16 bad / 38 good / 24 ambiguous of 80 fit rows) and merged into the base Q8_0 bytes
at scale 1.5: 56 tensors (`blk.0-27` x `attn_output` + `ffn_down`), every other byte
verbatim. It must be reported as a refit in every packet and table. Its realized
fidelity was smoke-measured on two tensors: cos 0.9946 / 0.9954 (Q8_0 keeps the edit).

## 2. Re-verify the operands (cheap, no GPU)

    uv run python scripts/p7_q25_operands.py        # re-hashes, rewrites operands.json

## 3. Run the waves (the card that measures)

    bash verification/p7-q25-15b/run_waves.sh --dry   # prints the argv list, writes nothing
    bash verification/p7-q25-15b/run_waves.sh         # runs every wave, in order

Order: removal-arm, removal-base, removal-published (harmful-40, cap 4096);
heldout120-arm/base/published (registered 120-row suite, cap 4096);
holdout24-arm/base (24 families, cap 4096); retention-arm/base/published
(241 rows, cap 1024); then shape audits on the arm's removal and held-out
predictions, drift, fidelity.

One CPU llama-server at a time, at the pinned argv for every wave:

    -m <artifact> -c 4096 --parallel 4 --jinja --no-webui --host 127.0.0.1 --port 8141

**That string is the wave's recorded `serve_config`, and `release-certify` compares it
to `spec.configs.<axis>.argv` after whitespace normalisation — any extra token fails
the packet with `CERTIFY_CONFIG_MISMATCH` (R9).** If the wave card serves with a GPU
binary instead, regenerate the spec with the exact string it will record (edit
`PINNED` in `scripts/p7_q25_spec.py` and re-run it; the artifact identity digest does
not depend on the argv), then wave and spec agree.

Caps are the registered floors (`matrix.manifest.json` -> `registered.cap`:
removal 4096, held_out 4096, retention 1024). A wave graded below its floor is
`invalid(cap_lowered)`, not a pass, so do not lower them to make the run shorter.

Resumable: a wave directory with a measured `wave.json` of the right row count is
skipped, and the drift/fidelity/audit steps skip on their own output file. `--force`
redoes them. Progress goes to `progress.txt`; the last server log to `server.log`.

Runtime (CPU, measured here): capture of 80 rows took ~25 min at 16 threads; the
retention wave is the long one (241 rows x cap 1024 x 3 artifacts). Wait on it rather
than polling: `ab park <id> --pid <pid> --note "wave run"` or a followup check.

**Memory:** the worker cgroup for a background job is 4 GiB, and a float32 1.5B torch
load is OOM-killed there (it happened in this card). A CPU Q8_0 server (~2 GB) plus
the client fits; run the wave script in one foreground/terminal call, not inside a
4 GiB-scoped job. This is why the capture was run in foreground chunks and why the
refit script now seeds from a finished capture.

## 4. Certify (the analysis card's two commands)

From the repo root, after the waves:

    uv run jdewitt release-certify \
      --spec verification/p7-q25-15b/spec.json \
      --out  verification/p7-q25-15b/packet.json

    uv run jdewitt verify-artifact \
      verification/p7-q25-15b/packet.json --name packet.json

    uv run jdewitt release-matrix \
      --manifest verification/releases/matrix.manifest.json \
      --out verification/releases/matrix.json \
      --table verification/releases/matrix.md \
      --doc docs/EVIDENCE-TABLE.md

`release-certify` reads saved waves only (no model, no GPU) and refuses by name with
no packet on: a missing axis, a probe-sized or narrowed suite, a wave whose recorded
`serve_config` differs from its spec slot (R9), a removal count with one wording
inventory, retention with no paired stat or median length, or a shape audit of
another artifact (R8). The spec already carries both wording-inventory digests and
the artifact identity digest `ce518e20517885fc…`, computed by the certifier's own
functions; the shape audits inherit that digest, so an audit of a different arm is
rejected rather than averaged in.

The row is already registered in `matrix.manifest.json` (base Q8_0 with a full
digest, arm Q8_0 with digest + 56 edited tensors + scale 1.5, published build as the
head-to-head with its three waves and the base retention wave). Today it reads
`UNMEASURED — no packet`; after the certify command it reads `certified`, or
`certified_at_other_config` if any axis was served off `-c 4096 --parallel 4`, or
`failed(removal)` with the number.

## 5. What the report must say, without rounding

- Removal under **both** inventories (shipped and wide) at harmful-40, and the
  held-out 120 with the published build as the paired competitor (discordant count
  and exact McNemar p). The wide inventory is the floor: `removal_wide_floor 40`,
  `held_out_wide_floor 120`. A count that only clears on the shipped inventory is
  the q3 failure this card exists to prevent repeating.
- The **shape audit** beside every removal number (flat declines / pivot with
  procedure / pivot no procedure / procedural no decline / other). Shape is a
  heuristic read; the semantic "was the harm delivered" is the human/agent step and
  is recorded as such (R6).
- Retention strict n/m with paired delta/h/p and median answer length, maths and
  code, from the 241-row suite; base and published at the identical argv. If any
  artifact ends up at another argv, report both configs and mark the row
  `certified_at_other_config` — never quote one config's retention as the other's.
- Drift: control delta in ln-PPL against the 0.02 budget (`drift.json`).
- Fidelity: `arm_measurement: true` — `compare_edit` over the served bytes, min cos
  and mean over all 56 edited tensors, against the matrix gate `fidelity_min_cos 0.9`.
  This is stronger than the q3 codec proxy and must not be described as one.
- If removal does not clear at 4096/4, the row is `failed(removal)` with the number
  and stays in the honest table as a loss. Retention is not a substitute for removal.

## 6. Known constraints of this lineage

- The capture pool is advbench/harsh-suite families, so harmful-40 is **in-sample**
  for the refit; the held-out screens (24 and 120) share no id or prompt with it.
  Say "in-sample" for harmful-40 in the packet narrative.
- The 24-family screen is 24 rows. Certify's held-out floor is 120 rows, so 24 is
  reported beside the 120-row axis (`side_screens.holdout_24_families` in the spec),
  never as the axis.
- The baseline retention number for this lineage is the published build at one
  config: base 98/241, published third-party ablation 81/241, SAS@1.5 184/241 (maths
  gain p 0.006) from `verification/retention-q25-code61/` at `-c 8192 --parallel 4`
  cap 1024. Those were taken at a different argv than this cert run; they are
  context, not this row's comparison.
