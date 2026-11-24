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

# staging gate

`python scripts/space_stage.py --out DIR` writes the tree an account owner
pushes: `README.md` (Spaces YAML: gradio, `app_file: app.py`, a `models:` list
built from the bank's bases), `requirements.txt` (numpy, gguf, huggingface_hub,
gradio — no torch pin, no transformers, no llama stack), a thin `app.py`, a
vendored `jdewitt/` import closure, `spaces/bank/` verbatim, `.gitignore` and the
AGPL licence text. The Space is buildable without pip-installing this repo, so
the XPU torch pin in `pyproject.toml` never reaches a CUDA ZeroGPU image.

It refuses to stage on a local import the tree cannot supply, an import the
requirements do not ship, any `/mnt/`, `/home/`, board card id or the banned word
in a staged text file, a tree over 5 MB outside the bank, or a non-empty target.

`stage.json` is the gate run, taken from two real stages into fresh directories:

- `diff_r_identical` — the two trees are byte-identical;
- `sanitisation.matches` — empty for the four refused shapes;
- `app_parses` — `ast.parse` on the staged `app.py`;
- `staged_import` — `import jdewitt.space_loop` with the staged tree on
  `sys.path` succeeds and pulls neither gradio nor torch;
- `ruff` — `ruff check jdewitt scripts`, exit 0.

The coverage loop itself (`jdewitt/space_loop.py`) answers from the bank alone:
`inspect(repo, revision, quant)` returns a covered entry or a typed `BANK_MISS` /
`UNSUPPORTED_QUANT`, with the licence `UNVERIFIED` and the edit state `none`
until those gates land.

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

# licence gate

`licence-gate.json` holds three real Hub rows from `jdewitt.bank_licence`: one
allowed (`Qwen/Qwen3-4B-Instruct-2507`, apache-2.0), one refused for its licence
(`bigscience/bloom-560m`), one refused as gated (`meta-llama/Llama-3.2-1B`). The
gate reads `model_info` only (`files_metadata=False`); a fresh `HF_HOME` stayed
empty, so no file was downloaded. The allow-list is `spaces/bank/licences.json`,
which `SHA256SUMS` pins.

# the loop gate

`loop.json` is the Phase 0 done gate for "app loop + card labels, dry run on a
fixture" (§10). It is rebuilt by:

    python verification/space/make_loop_gate.py

That script runs `space_loop.dry_run` on `fixtures/loop/`, stages a tree into a
temporary directory, runs the banned-word grep, `ruff check jdewitt scripts` and
the two Space test files, and writes one JSON. The user-facing equivalent is:

    python -m jdewitt space-dry-run --bank spaces/bank \
        --model jdewitt/loop-fixture --quant Q4_K_M \
        --fixture verification/space/fixtures/loop

What the run proves, with no GPU, no socket and no Hub download:

- the three labels and one refusal: the base at Q4_K_M renders
  `measured_artifact_class`, the fine-tune (same architecture, different bytes and
  revision) renders `banked_unmeasured`, and the other architecture is `refused`
  with `BANK_MISS` and one sentence naming the architecture and what would cover
  it;
- coverage read a 512-byte prefix of each 295 488-byte artifact; the fixture hub
  pads that prefix to the declared size as a sparse file, so no tensor bytes move;
- the licence gate ran on Hub metadata before the download, and the refusal paths
  (wrong licence, gated repo) are unit-tested;
- the unmeasured card's first line is `This file was **not** measured.`, and the
  card carries base repo@rev, applied strength, artifact class, config and
  delivered cos;
- `panel_order` puts `caveat` above `start` and `file`, and `spaces/app.py` builds
  the caveat Markdown above the `gr.File` control;
- every job deleted its workspace and left the four delivered files (artifact,
  `card.md`, `sidecar.json`, `SHA256SUMS` with matching digests);
- the quant control keeps `model-Q8_0.gguf` in the list, disabled with the reason,
  instead of hiding it;
- the staged tree carries `spaces/app.py` verbatim, vendors `gguf_types.py` and
  vendors no `quantize.py` (the llama-quantize wrapper with its local paths);
- the page itself builds under the pinned Gradio 5.50 (a scratch venv: the module
  constructs the Blocks demo, `refresh` returns a typed refusal for a bad repo id
  and `start` refuses an empty file choice before running anything).

`fixtures/loop/` regenerates with the project venv (torch is needed for the plan
bundle):

    python verification/space/fixtures/make_loop_fixture.py

It writes one measured entry (2 layers, hidden 256, `q4_k` source), the base at
Q4_K_M and Q8_0, a fine-tune of the same architecture with different bytes, an
artifact of an architecture no entry covers, `hub.json` (file sizes and the header
prefix that still parses) and the licence allow-list.

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

The timing run predates the shipped entry. `spaces/bank` now carries
`qwen3-4b-instruct-2507/f15s15-l935`, but the timing entry is still a labelled
synthetic-axis fixture (it needed the geometry, not the direction), so
`removal_measured` is false there and the delivered cos is codec fidelity on a
synthetic edit, not removal.

# The self-apply: the bank's own bundle on the base's Q4_K_M

`selfapply-qwen3-4b.json` is the first number the bank owns: `bank-apply` with the
shipped entry on the quant the Space will actually serve
(`unsloth/Qwen3-4B-Instruct-2507-GGUF` Q4_K_M @ `a06e946b`, sha256 `3605803b…`,
2,497,281,120 B), no model. The document is the `--edit-type auto` report verbatim
(card 3's shape: per-tensor cos/SNR, chosen type, bytes in/out, untouched re-read),
plus the artifact identity, the delivered ladder for the two other types the entry
measured, the probe-vs-delivered evidence, and the `gguf-inspect`/`gguf-tensor-diff`
read-back.

| run | type | delivered min cos | floor 0.95 | bytes out |
|---|---|---:|---|---:|
| auto | Q6_K (the probe's choice) | 0.8815 (37 of 42 below) | missed | 2,618,260,064 (+0.12 GB) |
| forced | Q8_0 | 0.9873 | met | 2,798,248,544 (+0.30 GB) |
| forced | F16 | 1.0006 | met | 3,494,978,080 (+0.99 GB) |

`removal_measured` is false and the document says so: cos is codec fidelity; no
answer was generated. The 42 edited names are exactly the 42 non-identical names in
`gguf-tensor-diff`, and the 356 untouched tensors are bit-identical in both diffs
(the JSON's untouched count equals the diff's `bit_identical` count).

Why `auto` misses: its ladder decides the type from `ggufio.preview_edit`, which
re-encodes `windows=2 x blocks=32` blocks (8,192 elements) of each probe tensor. For
these rank-1 deltas that sample is optimistic — on `blk.29.ffn_down.weight` the
sample reads cos 0.9974, the whole tensor 0.8812, the written file 0.8815 — so the
ladder stops at Q6_K one step before the type that clears. This is not a
`CODEC_FLOOR`: Q8_0 clears at the entry's own strength and the strength was not
raised.

Raw transcripts are in `selfapply-run/`; `selfapply-run/make_report.py` rebuilds the
document from them.

# The live ZeroGPU run

`live-run.json` is the only file that may be quoted as a Space runtime. It is one
run of the full loop inside the public Space `darioooooo0o/banked-gguf-edit`
(gradio 5.50.0, hardware `zero-a10g`, tree commit `72afd4a2`), driven over the
public URL exactly as a visitor drives it: one `POST /gradio_api/call/start` with
the repo id and the file name, then the event's stream.

| what | measured |
|---|---|
| wall clock | **964.95 s** — coverage 7.284, licence 0.058, download 5.711, edit **933.96**, read-back 15.346, package 2.558 |
| container | 192 cores `Intel(R) Xeon(R) Platinum 8559C`, 2.1 TB RAM, peak RSS **3.66 GB** |
| download | 2,497,281,120 B at **437 MB/s** end-to-end (the Hub's own line in the same log: 687 MB/s for the transfer) |
| `@spaces.GPU` calls | **0**, so **0 s of the 300 s/day quota** |
| output | 2,618,260,064 B, sha256 `f29fe43552ca18ea…`, 42 edited, 356 untouched byte-identical |
| served | `llama-server` (build b20-35999d101, `-ngl 0 --ctx-size 4096`) loaded it, `/props` echoed the path, first answer `OK` (`stop`, 2 tokens) — `live-run-serve.json` |

Three things this run settles, all of them uncomfortable:

1. **The five-minute promise is false for 4B.** 964.95 s is 3.2x it, and 96.8 % of
   the wall clock is the edit — single-threaded numpy on a 192-core box. The GPU
   quota is not the binding constraint and cannot be: 0 s of 300 were spent.
2. **A CPU-only edit must not be routed through the GPU queue.** The first live
   attempt wrapped the edit in `@spaces.GPU(duration=120)`, the shape §4 prescribes
   for an edit that fits one call. It finished the download in 3 s and then sat
   queued for a card the edit cannot use, with no output for over ten minutes. The
   Space now declares one `@spaces.GPU` function for ZeroGPU's startup check
   (`No @spaces.GPU function detected during startup` is otherwise fatal) and no job
   calls it; the run record says 0 calls.
3. **The type ladder still stops one step short.** The ladder chose `Q6_K` from its
   sampled preview and the whole-tensor read-back came in at 0.8808, below the
   entry's 0.95 floor — the same gap the self-apply run above documents on the
   base's Q8_0 (sample 0.9974, whole tensor 0.8812). `Q8_0` clears at the same
   strength. The card is honest about it (`delivered cos 0.8808`, label
   `banked_unmeasured`) but the preview threshold is the defect.

The artifact was downloaded from the Space's own file URL to
`/mnt/ssd/hermes/models/space-run-19c`, checked against the Space's `SHA256SUMS`
(all three OK), and only then served.


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
