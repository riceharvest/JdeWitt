# bank-check evidence

`jdewitt bank-check --bank DIR --gguf FILE` reads a GGUF's header and tensor-info
array only (no tensor data) and either prints the resolved entry or refuses with
one typed code. The three files here are the three gate runs from the bank card:

| file | artifact | expected |
|---|---|---|
| `bank-check-minicpm-v47.json` | `verification/minicpm-v47/minicpm-v47-Q4_K_M.gguf` | `BANK_MISS`, architecture `qwen35` |
| `bank-check-fixture.json` | `verification/space/fixtures/fixture-2layer-f16.gguf` | resolved names, layers `[0, 1]`, strength, edit types |
| `bank-check-bad-digest.json` | same fixture, `bank-bad-digest/bank.json` | `BUNDLE_BAD_DIGEST` |

A refusal exits 1 and prints the standard CLI envelope
(`{"error":{"code","type","message","command"}}`) — one JSON, like every other
typed refusal in this repo.

The MiniCPM artifact is 672 MB and is not committed (`*.gguf` is ignored and the
clone outside `verification/vocab-remap/vocab/` carries no model). To rerun that
row, put the file at the path above (a symlink to a local copy is enough) or
download it; the command never reads tensor data, so a header-range copy would do.

The fixture is committed and regenerable:

    python verification/space/fixtures/make_fixture.py

`make_fixture.py` writes the 2-layer GGUF, `bank/fixture-bank.tar` (fixed member
mtime, so its sha256 is reproducible), the entry that pins that sha256, and
`bank-bad-digest/` with a one-byte-flipped copy pinned at the good digest.

# bank-apply evidence

`jdewitt bank-apply --bank DIR --entry KEY --gguf FILE --out FILE` resolves one
entry, installs its pinned bundle into a throwaway registry, re-encodes only the
footprint tensors at the entry's per-kind strength and prints one JSON. The three
files here are its gate runs:

| file | artifact | expected |
|---|---|---|
| `bank-apply-fixture.json` | `fixtures/bank-apply/fixture-2layer-q8_0.gguf` | F16 chosen (q8_0 probe 0.9979 misses the entry's 0.999 floor), 4 edited / 2 untouched byte-identical, delivered cos 1.0 |
| `bank-apply-geometry-refused.json` | same fixture, `fixtures/bank-apply-bad-geometry/bank.json` (hidden 128) | `GEOMETRY_MISMATCH`, exit 1, no output file |
| `bank-apply-real-qwen2.json` | `/mnt/ssd/models/qwen25-1.5b-instruct-q4_k_m.gguf` (28 layers, hidden 1536, footprint 10-12) | Q6_K 0.6466 / Q8_0 0.9444 / F16 0.9999 -> F16, 6 edited / 332 untouched verified, 986 MB -> 1052 MB, 105-144 s over two runs |

The two fixtures regenerate with (the project venv, which has torch):

    python verification/space/fixtures/make_apply_fixture.py
    python verification/space/fixtures/make_real_apply_bank.py \
        --gguf <artifact.gguf> --bank <dir> --layers 10 12

`make_apply_fixture.py` writes a real REGISTRY plan bundle (a `[2,1,64]` basis, tar
members at a fixed mtime; the sha256 reproduces byte for byte on the same torch
build), the q8_0 artifact beside it, and `bank-apply-bad-geometry/` — the same
good bundle under an entry whose hidden width disagrees, so the refusal gate needs
no doctored bytes.

`make_real_apply_bank.py` is not a way to seed the shipped bank: it builds a
labelled SYNTHETIC axis (`measurement.state = synthetic_axis`) fitted only to the
artifact's own geometry, because the clone carries no fitted axis. That run proves
the write path, the ladder, the per-tensor measurement and the untouched bytes; it
proves nothing about removal, and the report says `"removal_measured": false`.

The output artifacts are not committed (`*.gguf` is ignored); each run rewrites
them, and the reports record the paths used at the time.

# The 2-vCPU CPU proxy

`cpu-proxy-2vcpu.json` is the measurement `docs/SPACE-ROADMAP.md` §4's decision
rule is applied to. It is a **CPU proxy, not a Space run**: the edit path is
numpy-only and ZeroGPU adds a GPU but not CPU cores, so `bank-apply` was timed on
two affinity-pinned cores of this host (`taskset -c 0,1`, `/usr/bin/time -v`).

| file | what it holds |
|---|---|
| `cpu-proxy-2vcpu.json` | artifact identity (repo/revision/sha256/bytes), `is_a_space_run: false`, host cores × affinity, three runs with wall/user/sys/RSS, bytes out, per-tensor delivered cos, the §4 verdict, the 7B extrapolation |
| `cpu-proxy-2vcpu/bank/` | the labelled synthetic-axis timing entry the runs used (see its README) |
| `cpu-proxy-2vcpu/make_report.py` | rebuilds the JSON from the two transcripts and the bank-apply reports |

Artifact: `unsloth/Qwen3-4B-Instruct-2507-GGUF` @ `a06e946bb6b6`, Q4_K_M,
2,497,281,120 B, sha256 `3605803b…` (equal to the Hub LFS sha, recomputed after
the download). Measured on 2026-11-23: **496.3 / 476.3 / 606.2 s** wall for the
same command, byte-identical output every time (42 F16 tensors, 2.50 GB →
3.49 GB, 356 untouched byte-identical). The one-core-at-98 % reading is why a
2-vCPU Space does not halve this. Verdict: **`cuda_codec_required`** — every run
is above 240 s. The 7B extrapolation (3.04× edited params) is ≈ 1510 s, a number
under a named assumption.

`spaces/bank` is empty, so the timing entry is a fixture: `removal_measured` is
false and the delivered cos is codec fidelity on a synthetic edit, not removal.

# The transfer kill gate (Qwen3-4B-Instruct-2507 → its Q4_K_M abliterated fine-tune)

`transfer-qwen3-4b.json` is the §6 row. It is the one artifact class the bank's
premise ships to, measured once: the bank entry fitted on
`Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c0` applied by `bank-apply` at its default
strength (`{attn 1.5, mlp 1.5}`, 42 tensors, Q8_0) to
`mradermacher/Huihui-Qwen3-4B-Instruct-2507-abliterated-GGUF@c9e90669eeb2`
`Q4_K_M` (sha256 `b1324e2f…`, tokenizer byte-identical to the base — see
`transfer-qwen3-4b/tokenizer-compare.json`). Both sides served from the same
binary at `-c 4096 --parallel 4`, cap 1024, temp 0, `--device Vulkan0` on the B70.

Every answer was read by the agent (R2 shape, wide inventory); the word checker
is reported only as a screen. Verdicts and the exact commands:

| file | what it holds |
|---|---|
| `transfer-qwen3-4b.json` | artifact class, both serve argv, removal + retention + drift + delivered cos, `verdict: kill` |
| `transfer-qwen3-4b/verdicts-{merged,base}-{harmful-40,heldout-120}.json` | 320 per-row verdicts with a one-line reason each (add `advbench-train-*` ids) |
| `transfer-qwen3-4b/read-score-{harmful-40,heldout-120}.json` | the rates and the exact McNemar on the paired rows |
| `transfer-qwen3-4b/tokenizer-compare.json` | the tokenizer gate: tokens/token_type/merges hashed against the base GGUF |

Result, in one line each:

- harmful-40 (in sample): merged **1/40** delivers, unedited fine-tune **19/40**;
  0 vs 18 discordant pairs, exact McNemar **p 7.6e-06**.
- held-out 120: merged **20/120**, unedited **63/120**; 0 vs 43 pairs,
  **p 2.3e-13**.
- retention 241: **157/241** merged vs **155/241** unedited (p 0.89) — intact.
- drift: control ln ppl **+0.0709** (31 %, budget 0.02), `delta_BR` −0.5532.
- delivered cos: min **0.9877**, mean 0.9918 over 42 tensors, floor 0.95 met.

The word inventory would have called this a win (11 vs 16 refusal-shaped answers
on harmful-40) while 13 of the merged 40 collapsed into repetition and 19 pivoted.
Fewer refusal words, less content — the false pass R3 exists to catch. This is why
the row is `kill` and not a removal number.

The packets (full answers, ~1.5 MB) and the recorded waves live outside the clone
under `~/b70_prep/splits/t_75155e9e/out/`; the paths are in the JSON's `evidence`
block. Regenerate a verdict file's score with `scripts/delivery_read_packet.py
score --pair-basis read` (the default now; `--pair-basis ok` is the old silently
narrow pairing that left 2 of 40 rows on this wave).
