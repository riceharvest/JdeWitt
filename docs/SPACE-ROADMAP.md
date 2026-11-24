# SPACE-ROADMAP — a ZeroGPU Space that edits the visitor's own GGUF

v1, 2026-11-23. Status: design only. **No Space has been deployed and no GPU second has been
spent on one.** Where this document quotes a runtime, a size or a quota it is tagged:

- **MEASURED** — the number exists in a file in this clone; the file is named.
- **ARITHMETIC** — derived from block sizes and a stated bandwidth assumption; replace it with a
  measurement before quoting it anywhere public.
- **GATED** — needs a real run (B70, a serve, or a ZeroGPU Space) and cannot be closed by a CPU card.

## 0. What this supersedes, and what it does not

`docs/ROADMAP.md` ends with an explicit non-goal: "cloud/Space hosting". This document overrides
that line for exactly one product: **the banked-axis editor**. Everything else in the ROADMAP
stands, and one rule above all: *the deliverable is an edit on the artifact the visitor already
serves.* The Space downloads their quant, edits its bytes, and hands the same file back. It never
offers a pre-quantized model of ours, never re-quantizes the model, and never generates a token.

The repo's advantage is the thing that makes this possible at all: `gguf-edit` / `edit-quantized` /
`rewrite_with_edits` re-encode only the tensors that carry an edit and copy every other tensor
byte for byte, with the untouched bytes re-read and checked (`verify_untouched`). A Space that
did a full dequant-requant would be a different, worse product and could not fit the budget.

## 1. What the visitor gets

Their own GGUF (their quant, their calibration) with one residual direction removed from the
writeback tensors of a fixed layer footprint, those tensors re-encoded at a type that can store
the edit, every other byte verbatim, plus:

- the output file, streamed for download;
- a `SHA256SUMS` line and the bank entry's digest;
- a generated model card naming the base the axis was fitted on, the artifact class it was
  *measured* on, the serve config, and the caveat that applies when their file is not that class;
- the `.revert.npz` sidecar when its size is acceptable (it is ~30 % of the edited tensors' bytes,
  so at 4B it is a few hundred MB; the app offers it as a second file, not inline).

## 2. The problems, and the decision taken for each

| # | Problem | Decision | What keeps it honest |
|---|---|---|---|
| 1 | A fitted axis needs captures the Space cannot run | **Bank** of offline-fitted, digest-pinned plans, one entry per base model; applied to fine-tunes of that base | The entry carries the artifact class it was measured on. An unmeasured class ships **labelled**, or is refused |
| 2 | ZeroGPU's limits | One Gradio app, one job at a time, `@spaces.GPU(duration=120)` calls only for the tensor work | Every quoted second comes from the Space's own log |
| 3 | Output path | **Direct download** for v1; OAuth `write-repos` push only if the 4.5 GB case fails | Ephemeral disk means the link dies with the Space; the visitor is told to save now |
| 4 | The loop | Pick model + quant → coverage + licence → download → banked edit → read-back → package → deliver | Strength is the bank's default; there is no strength slider |
| 5 | Coverage | v1 is **one base, one architecture family, four source quant types** | Anything else is refused with a named reason, never edited with a guessed axis |

## 3. Platform facts (re-fetched 2026-11-23; re-fetch before designing on them)

From <https://huggingface.co/docs/hub/spaces-zerogpu> (`.md`), the 2026-11-23 revision:

- ZeroGPU is the free GPU; **Gradio SDK only** (Gradio 4+), torch 2.8+ (2.8.0-2.13.0 listed),
  Python 3.12.12 / 3.10.13.
- Two sizes: `large` (default) = half RTX Pro 6000 Blackwell, **48 GB**, 1x quota;
  `xlarge` = full card, 96 GB, **2x quota**.
- `@spaces.GPU` default duration **60 s**, custom `duration=` up to **120 s**; the decorator is a
  no-op outside ZeroGPU.
- Daily quota per account: unauthenticated **2 min**, free **5 min**, PRO 40 min. The included
  quota "resets exactly 24 hours after your first GPU usage". Queue priority tracks the remaining
  quota, so a visitor who has already spent the day is queued last.
- Hosting: free personal accounts in good standing (verified email, account > 30 days) may host
  **2 ZeroGPU Spaces**; PRO 10; orgs 50.
- `torch.compile` is unsupported; ahead-of-time compilation is available (torch 2.8+).

From <https://huggingface.co/docs/hub/spaces-overview> (`.md`):

- Every Spaces environment: **16 GB RAM, 2 CPU cores, 50 GB non-persistent disk**.
- CPU Basic is free; a Gradio/Docker Space "requires a paid plan", **except** the ZeroGPU free
  tier for personal accounts. That exception is the whole reason this product targets ZeroGPU:
  the GPU is not needed for throughput, it is the only free way to host a Gradio app.
- Free hardware sleeps when unused.

From <https://huggingface.co/docs/hub/spaces-oauth> (`.md`):

- `hf_oauth: true` in the Space README enables the sign-in flow; `hf_oauth_scopes` takes optional
  scopes, of which `write-repos` is documented as "Read and write the user's personal repos".

**Not verified, and therefore not relied on:** the container's CPU/RAM while on ZeroGPU hardware
(assumed to be the 2 vCPU / 16 GB default — the first live run must print what it sees), the
maximum response time of the Spaces proxy for a multi-GB Gradio `File` download, and the
achievable Hub download/upload rate from a Space container. All three get measured in the live-run
card, not guessed here.

## 4. The feasibility equation

The free visitor has **300 s of GPU time per day**, at most **120 s per call**, and the edit is
CPU-bound numpy today: `ggufio` is "numpy only, no device", and ZeroGPU adds a GPU but **not CPU
cores**. So the number that decides the product is not VRAM, it is a 2-vCPU wall clock.

MEASURED, in-clone, on this host (not 2 vCPU):

- in-place banked-style build on a 1.1 GB Q4_K_M: **368.1 s** (contrast arm) and **298.0 s** (RDO
  arm), 56 tensors, 1,713,021 blocks, 0 bytes added, all other bytes verbatim —
  `runs/p4/builds.json`;
- the same class of edit on a loaded host: **1057 s**, two dequant passes — `docs/AGENT-FIRST.md`;
- the codec floor that shapes the output: at strength 1.0 on Spark 1.7B Q4_K_M, realized cos is
  **0.454** on attention and 0.938 on FFN — `docs/ROADMAP.md` §"Where we are short" (1). A unit
  axis at s=1 changes ~3 % of a row norm, below what a 6-bit grid keeps, so edited tensors land at
  q8_0 or f16 and the file grows.

Decision rule, applied by the 2-vCPU proxy card before anything is deployed:

| proxy wall clock for the covered 4B artifact | verdict |
|---|---|
| ≤ 120 s | one `@spaces.GPU(duration=120)` call; quota cost ≤ 120 s of 300 |
| 120-240 s | two calls, sharded by layer block; quota cost ≤ 240 s of 300 |
| > 240 s | a CUDA codec port is **mandatory** before any Space run; if the port cannot beat 240 s either, the free product is 1.7B-and-below and 4B/7B are dropped |

**Applied by the 2-vCPU proxy card (2026-11-23) — a CPU proxy, not a Space run:**
`verification/space/cpu-proxy-2vcpu.json`. `bank-apply` on
`unsloth/Qwen3-4B-Instruct-2507-GGUF` @ `a06e946bb6b6` Q4_K_M (sha256 `3605803b…`,
2,497,281,120 B), pinned to cores 0-1 (`taskset -c 0,1`), on a labelled synthetic-axis entry
carrying the §5.1 geometry (36 layers, hidden 2560, `blk.15-35`, 42 tensors, strength 1.5):

- **496.3 s** wall / 450.4 s user / 37.2 s sys, max RSS 4,285,292 kB; 2.50 GB in → 3.49 GB out,
  42 tensors written at F16, 356 untouched tensors byte-identical;
- repeats: 476.3 s and 606.2 s, byte-identical output each time. The range is the shared host's
  memory pressure — the slower run spends 25 % more *user* CPU, not more I/O — and no run uses
  more than ~98 % of ONE core, so the second vCPU buys nothing;
- verdict: **`cuda_codec_required`**. Every run is above 240 s, so the rule cannot move.
- 7B extrapolation: the 7B case has **3.04×** the edited parameters (2.26 B / 743.2 M) and would
  be **≈ 1510 s** at the same per-parameter cost — a number under that named assumption, not a
  measured 7B run.

These are CPU proxy seconds from this clone; no Space run has produced a runtime yet (Phase 3).

Memory: a 4B Q4_K_M is ≈ 2.5 GB on disk; dequantizing the 42 edited tensors in float32 is
≈ 743 M params (measured from the tensor table: 21 layers × (10,485,760 + 24,903,680)) × 4 B
≈ 3.0 GB, and the measured run peaked at 4.3 GB RSS. That fits 16 GB. A 7B Q4_K_M (≈ 4.5 GB)
with ≈ 2.26 B edited params (≈ 9 GB in float32) is the edge of the container **and** is nowhere
near 240 s on 2 vCPU, so 7B is not v1.

**One consequence to state plainly:** if the edit is CUDA-accelerated it does real work inside
`@spaces.GPU`, and the visitor's 5 minutes are spent on compute. If it stays on CPU inside a GPU
call, the quota is charged for waiting. Either way the budget is 300 s/day, so **a free visitor
gets one 4B edit per day**, and a failed attempt costs the same day's second attempt.

## 5. The direction bank

### 5.1 Unit

One entry = one `entry.json` + one `plan-bundle` (the existing digest-pinned container produced by
`plan-bundle`, verified by `plan-unbundle`/`plan-load`). The Space ships the whole bank as a
directory of a few MB; the axis is `[layers, rank, hidden]` in bf16, so 36 layers × 2560 floats is
≈ 184 KB per target kind. Nothing in the bank is model weights.

```json
{
  "schema": "jdewitt-space-bank-v1",
  "entries": [{
    "key": "qwen3-4b-instruct-2507/f15s15-l935",
    "base": {"repo": "Qwen/Qwen3-4B-Instruct-2507", "revision": "<sha>"},
    "architecture": "qwen3",
    "geometry": {"layers": 36, "hidden": 2560, "hidden_source": "artifact"},
    "targets": {"attn": "attn_output", "mlp": "ffn_down"},
    "footprint": {"first": 15, "last": 35, "fraction": [0.4167, 0.9722]},
    "strength": {"attn": 1.5, "mlp": 1.5},
    "edit": {"method": "directional", "shape": "shared", "space": "output",
             "row_normalization": "none", "min_cos": 0.95,
             "types": ["q6_k", "q8_0", "f16"], "repair": "none"},
    "sources": {"quant_types": ["q4_k", "q5_k", "q6_k", "q8_0"]},
    "measurement": {"state": "banked_unmeasured", "artifact_class": null,
                    "removal": null, "retention": null, "config": null},
    "bundle": {"file": "qwen3-4b-instruct-2507.tar", "sha256": "<digest>"}
  }]
}
```

The geometry values shown are the expected Qwen3-4B numbers, not constants: the entry is written by
the refit card and every field is read from the artifact before it is written.

Rules the loader enforces, all from the **artifact's own tensor table**, never a per-model table:

- hidden width from `gguf_adapters.inferred_hidden_width`, targets from
  `discover_writeback_targets` (name **and** width — a hidden-wide `q_proj` is not a writeback
  tensor);
- every `footprint` layer must exist in the artifact and carry both target kinds, else
  `GEOMETRY_MISMATCH` naming the missing layers;
- the source quant type must be one this build decodes, else `UNSUPPORTED_QUANT` naming the type;
- the bundle digest is verified before the axis is read, else `BUNDLE_BAD_DIGEST`;
- no entry for the architecture → `BANK_MISS` with the architecture the artifact declares.

`footprint.fraction` is how a bank entry survives a different layer count: the entry states the
fraction it was measured at, and the loader applies the same fraction to the artifact (rounded,
clamped to whole layers). Anything outside the fraction is not edited. Strength is **not** scaled
by the fraction; it is the measured arm's number and only that.

### 5.2 Where the first entry's numbers come from

The only arm on this board with a removal row is Qwen3-4B-Instruct-2507 `f15s15-l935`:
shared sign-aligned axis, output space, `o_proj` + `mlp.down_proj`, `blk.15-35` of 36, scale 1.5,
42 tensors, Q8_0 chosen type, `realized_cos` 0.7255 on 4 sampled tensors, drift 0.0058 ln ppl,
40/40 harmful-40 and 120/120 held-out, maths 148/179 and code 15/61 — **at a non-pinned config**,
which is why the evidence table marks it `certified_at_other_config` and not a pass
(`verification/releases/qwen3-4b-f15s15-l935/spec.json`, `docs/EVIDENCE-TABLE.md`).

Two facts follow, and both are uncomfortable:

1. The axis file that produced it is **not in this clone**; the path the spec names no longer
   exists. The entry was therefore **refitted** from a fresh B70 capture at the recorded recipe
   (ctx 2048, answer locus, 512 tokens, 80 rows; fit 1.21 s; axes 36×2560, fit cos mean 0.736),
   and the refit reproduces the plan key `1d29cd9feda0cf41` — so the committed bundle is provably
   that fit (`verification/space/qwen3-4b-refit/`, `spaces/bank/qwen3-4b-instruct-2507/`). A refit
   carries the *recipe*, not the certificate, which is why the entry ships `banked_unmeasured`.
2. There is **no certified-at-pinned-config removal row for any lineage** (0 of 5 in
   `docs/EVIDENCE-TABLE.md`), and the Qwen2.5-1.5B removal numbers are **retracted** (6/40, 11/24,
   34/120 under the wide inventory). So the Space's first covered model does not inherit a
   "refusals removed" label from anywhere. It starts at `banked_unmeasured` and earns the label
   only from the transfer card's own wave on the exact artifact class.

### 5.3 Default strength per family

v1 carries one family and one strength pair, `{attn: 1.5, mlp: 1.5}`, because that is the arm that
was measured. A second family (e.g. a Llama or Gemma base) may reuse the *shape* of the recipe but
not its strength; each entry carries its own measured number and the arm label it came from. The
UI never exposes strength.

### 5.4 Coverage at v1

One base (Qwen3-4B-Instruct-2507) and its fine-tunes, architecture `qwen3`, source types `q4_k`
(quants `Q4_K_M`, `Q4_K_S`), `q5_k` (`Q5_K_M`), `q6_k`, `q8_0`. Everything else: refused with the
reason. Widening coverage is a separate phase and starts only after the transfer gate passes.

## 6. Transfer measurement — the kill gate

The bank's premise is that a base-fitted axis still moves refusals on a **fine-tune** of that base.
That is a claim about an artifact class, so it is measured on that class, not on the base.

Design (one wave, one pinned config, `-c 4096 --parallel 4`, cap 1024 per R9):

1. pick the top-downloaded fine-tune of the covered base that ships a Q4_K_M GGUF (record repo,
   revision, sha256), and refuse a candidate whose tokenizer or architecture differs;
2. apply the entry at its default strength (`bank-apply`), no strength search;
3. serve the merged artifact on the 40-row harmful suite, the 120-row held-out suite and the
   241-row retention suite; serve the **unedited fine-tune** at the same config as the base side;
4. the agent reads every answer for shape (R2) under the wide inventory; a word pass is a screen,
   never a verdict;
5. report per-row verdicts, McNemar on the paired rows, drift, median answer length, and the
   delivered cos per edited tensor.

**Kill criterion.** If removal on the fine-tune does not beat its own unedited rate by a
significant margin on the 40 in-sample rows, **or** does not hold on the 120 held-out rows, the
transfer claim is dead: write `"verdict": "kill"` with the numbers and stop the product line.
Do not widen coverage, do not shop for a friendlier fine-tune, do not raise the strength past the
bank's default to make it move.

**What the row may say.** Only what it measured: "on `<repo>@<rev>` Q4_K_M at `-c 4096 --parallel
4`, N/40 harmful and M/120 held-out by shape, retention X/241, drift Y". If the fine-tune is not
the measured class, the output card says `banked_unmeasured` and the file is still delivered — as
an edit, not as a removal.

**Measured (this board), and it is a kill.** On
`mradermacher/Huihui-Qwen3-4B-Instruct-2507-abliterated-GGUF@c9e90669eeb2` `Q4_K_M` at
`-c 4096 --parallel 4` / cap 1024 / `Vulkan0`, with the unedited fine-tune served at the same
config in the same wave: **1/40 harmful and 20/120 held-out** deliver, against **19/40 and 63/120**
for the unedited file (paired exact McNemar p 7.6e-06 and 2.3e-13, and every discordant pair
favours the unedited side). Retention **157/241 vs 155/241** (p 0.89), control drift **+0.0709**
ln ppl, delivered cos min **0.9877** over 42 tensors. The merged artifact did not refuse more by
wording — the word screen reads 11 vs 16 refusal-shaped answers — it pivoted (19/40) and collapsed
into repetition (13/40). Fewer refusal words, less content. Full row, per-row verdicts and the
tokenizer gate: `verification/space/transfer-qwen3-4b.json`. `verdict: kill`; the product line
stops here.

## 7. The loop, step by step

Each step is one command or one pure function, and each emits one JSON. No step needs a model
server, and the whole path must work with `transformers` absent from the process.

### 7.1 Pick model + quant

The UI takes a repo id (default: a short list of covered bases) and lists the repo's `*.gguf`
files with sizes. Quants whose source type is not in the entry's `sources.quant_types` are shown
**disabled with the reason**, not hidden: hiding them makes the picker lie about coverage.

### 7.2 Licence and access

`huggingface_hub.HfApi().model_info(repo, revision=...)` → `gated`, `cardData.license`, `tags`.
Rules: a gated repo is refused (`GATED_REPO` — an anonymous Space cannot authenticate), a missing
or unrecognised licence is refused (`LICENCE_UNKNOWN`), and the licence string is copied verbatim
into the output card. The allow-list is the bank's, not the world's.

### 7.3 Coverage

`bank-check --bank DIR --gguf FILE` reads the header only (no tensor data) and returns the resolved
entry, the resolved target tensor names, the layer list and the refusal code if any. For a Hub
artifact this runs on a **header range request** before the 2.5 GB download: the GGUF header plus
tensor table of a 4B Q4 is a few MB, so an uncovered model costs seconds, not gigabytes.

### 7.4 Download

`hf_hub_download` into the Space's 50 GB ephemeral disk, one job at a time, with a size check
against the Hub's metadata. No token, no private repos.

### 7.5 Banked edit

`bank-apply --bank DIR --entry KEY --gguf IN --out OUT` — **this command does not exist yet**.
What exists is the primitive it wraps: `rewrite.rewrite_with_edits(src, out, edits, edit_type=...,
axis_space='output', verify_untouched=True, repair=...)`, where `edits` maps tensor name to
`(axis, space)` and only those tensors are re-encoded; `ggufio.axis_delta` computes the per-tensor
delta from the axis, and `ggufio.compare_edit` / `preview_edit` measure what shipped. The command
adds four things the primitive does not have: the bank resolution, the edit-type policy (smallest
of `q6_k → q8_0 → f16` that clears `min_cos` on the worst edited tensor), the read-back report,
and the typed refusals. `edit-quantized` is the closest sibling but takes capture records, not a
pinned entry.

No inference: the module imports no tokenizer, no `transformers`, no server client. The card
asserts this by import, and the Space's requirements do not contain `transformers` at all.

### 7.6 Read-back verification

Not assumed: the output is reopened and checked — untouched tensors byte-identical against the
input, the file's tensor table and sizes valid, per-tensor delivered cos/SNR reported, `sha256`
printed. `gguf-inspect` on the output is part of the JSON, not a separate manual step.

### 7.7 Package and deliver

`SHA256SUMS`, the generated card (§9), the sidecar, and the Gradio `File` output. The Space deletes
its copy after the response is sent (the disk is ephemeral anyway) so two jobs cannot collide on
50 GB.

## 8. Output path

**v1: direct download.** Reasons, in order: no write scope is requested from the visitor; the
browser pulls the file from the Space (no upload from a 2-vCPU container); no repo is created in
the visitor's account for a one-shot edit; and an anonymous visitor (2 min of quota) is a valid
user for a job that may not need the GPU at all.

Sizes (MEASURED on the real file — `verification/space/selfapply-qwen3-4b.json`; the type the
shipped bytes use decides which row the Space pays):

| case | input | edited params | out at q6_k | out at q8_0 | out at f16 |
|---|---:|---:|---:|---:|---:|
| 4B Q4_K_M, 21 of 36 layers, 2 tensors/layer | 2.50 GB (2,497,281,120 B, measured) | 743.2 M (measured) | 2.62 GB (+0.12; delivered min cos 0.8815 — below the floor) | 2.80 GB (+0.30; delivered min cos 0.9873) | 3.49 GB (+0.99; delivered min cos 1.0006) |
| 7B Q4_K_M, 28 layers, 2 tensors/layer | ≈ 4.5 GB | ≈ 2.26 B | — | ≈ 5.5 GB (+1.03, arithmetic) | ≈ 7.6 GB (+3.15, arithmetic) |

The 4B row is measured, not estimated: `bank-apply` with the shipped entry
(`qwen3-4b-instruct-2507/f15s15-l935`, strength 1.5) on the `unsloth/Qwen3-4B-Instruct-2507-GGUF`
Q4_K_M a visitor would bring. Two facts from it decide this section:

- the type that clears the entry's 0.95 floor is **Q8_0**, not F16: delivered min cos 0.9873
  (mean 0.9915) over the 42 edited tensors for **+0.30 GB**. F16 buys the last 0.013 of cos for a
  further +0.70 GB. The output the Space downloads is **2.80 GB**.
- the ladder's own `auto` choice is **Q6_K** and it must not ship: its pre-write probe reads
  0.9961, but the written file delivers min cos 0.8815, below the floor on 37 of 42 tensors. The
  probe re-encodes 2×32 blocks (8,192 elements) of each tensor; on the whole tensor the same edit
  reads 0.8812. Until the probe covers the tensor, the floor is taken from the delivered file, not
  from the probe. The type changes the output size, not §4's verdict: every type is far above
  240 s on the proxy.

Transfer, with the assumption named: a 2.8 GB download at 30-100 MB/s is 28-95 s inside the Space;
the visitor's browser pulling 2.8 GB at 10-50 MB/s is 1-5 min and is outside our control. A 5.5 GB
push to the Hub at 10 MB/s is ≈ 9 min — longer than the whole GPU budget, and it must happen in
the same request window. That is why (b) is not v1. (A ladder that lands on F16 instead ships the
measured 3.49 GB, and the download estimate moves to 35-116 s at the same rates, still inside the
window.)

**v1.1: OAuth push**, adopted only if the 4.5 GB direct-download case fails in a live run. It is
supported: `hf_oauth: true`, `hf_oauth_scopes: [write-repos]`. The app creates
`jdewitt-<base>-<quant>` under the visitor's account, uploads the GGUF + card + digests (+ a
`.revert.npz` sidecar if under a 1 GiB cap), and shows per-file progress. Conditions: a hard cap on
concurrent uploads, a cancel that leaves no repo, and the card must say the Space only ever held a
temporary copy. All of that is built behind `JW_OAUTH_PUSH` (`jdewitt/space_deliver.py`,
`scripts/space_stage.py --oauth-push`), because the live run's direct download did **not** fail:
it moved 2.5 GB at 437 MB/s. What failed in that run was the edit leg (§11), which no output path
fixes.

**Measured, 2026-11-24** (`verification/space/oauth-upload.json`; three runs against one real
personal account through the `hf_oauth` token the live Space issued):

| artifact | bytes | transfer | upload | MB/s | commit wall |
|---|---:|---|---:|---:|---:|
| 4B Q8_0 self-apply (real product artifact) | 2,798,248,544 | basic | 25.84 s | 108.3 | 31.4 s |
| 5.5 GB-class GGUF probe, pre-fix | 5,500,000,704 | multipart | 371.0 s | 14.8 | 379.2 s |
| 5.5 GB-class GGUF probe, shipped | 5,500,000,704 | multipart | **116.8 s** | **47.1** | 141.6 s |

Three things the card records rather than assumes. The Hub **refuses one basic PUT above 5 GB**
(HTTP 400, "configure your repository to enable upload of files > 5GB"), so objects over it go out
as 16 MB multipart parts and the first 5.5 GB attempt deleted its repo when the batch refused.
Sequential parts on a fresh connection per part ran at 14.8 MB/s and missed the window; one
keep-alive connection per host is what brought 5.5 GB to 117 s, **183 s inside the 300 s promise**.
And the 5.5 GB payload is a valid GGUF transfer probe, not a 7B edit: with one 4B entry in the
bank, a 7B artifact is `BANK_MISS`, so the byte class is what this measures. The container's own
uplink is not measured — the container leg in `live-run.json` is the download — so the flag may be
turned on, and the container-side push is the next number to take. It stays **off by default**
(`JW_OAUTH_PUSH` unset, and `scripts/space_stage.py` writes the scope lines only with
`--oauth-push`): the download serves an anonymous visitor, and asking for `write-repos` is a
promise the Space should make only where it is going to use it.


**What the visitor sees during the upload/download:** a stage line, bytes moved and total, a rate
and a remaining-time estimate computed from the bytes actually moved (never a constant), and the
digest as soon as it exists. No spinner without numbers.

**Ephemeral disk is the product's lifetime.** The Space's 50 GB disk is lost on restart or sleep,
so the download link is valid only while the Space is awake and the visitor is told to save the
file immediately; the app does not promise a URL that outlives the session.

## 9. UI and card honesty labels

Three states, and the label is generated from the entry's `measurement.state`, never typed by hand:

1. **`measured_artifact_class`** — the exact repo/revision/quant this entry was measured on. Card
   line: "Measured on `<repo>@<rev>` `<quant>` at `<argv>`: `<N>/40` harmful and `<M>/120` held-out
   by answer shape, retention `<X>/241`, drift `<Y>` ln ppl. Shape read by an agent, wide inventory;
   no word-list count is used."
2. **`banked_unmeasured`** — a fine-tune or quant that is covered geometrically but was not
   measured. Card line: "This file was **not** measured. The direction was fitted on `<base>@<rev>`
   and applied as-is at strength `<s>`. Codec fidelity only: delivered cos `<c>` on `<n>` tensors.
   No claim is made about this model's answers."
3. **`refused`** — `BANK_MISS` / `GEOMETRY_MISMATCH` / `UNSUPPORTED_QUANT` / `LICENCE_UNKNOWN` /
   `GATED_REPO`, each with one sentence naming the model's own architecture or file, and what would
   be needed to cover it.

Banned everywhere in the app, the card and the staged tree: "uncensored", "refusals removed",
"jailbroken", "no refusals", any percentage of refusal reduction that is not the measured row for
that artifact class, and any runtime or quota figure that did not come from a Space run's own log.

The "no inference" promise is stated where the visitor can check it: the card lists the pipeline
(`header read → byte copy → tensor re-encode → read-back`), and the Space's `requirements.txt`
contains no inference stack — no `transformers`, no `llama-server`, no tokenizer.

## 10. Phases

### Phase 0 — CPU: the product's skeleton, verifiable today

| step | done gate |
|---|---|
| staging script: `spaces/` app + README metadata + vendored `jdewitt/` + bank, no absolute paths, no board ids | two runs byte-identical; the staged tree imports without torch; sanitisation grep empty |
| bank format + loader: geometry from the artifact, typed refusals | `BANK_MISS` on a real uncovered GGUF; a resolved target list on a fixture; `BUNDLE_BAD_DIGEST` on a corrupted bundle |
| `bank-apply`: their gguf in, edited gguf out, one JSON | delivered cos printed, untouched tensors byte-identical on re-read, second run same sha256 |
| 2-vCPU proxy timing on the covered 4B Q4_K_M | `verification/space/cpu-proxy-2vcpu.json` with wall/user/sys/RSS and the verdict from §4 — **done (2026-11-23): ≈ 496 s on the per-tensor F16 ladder → `cuda_codec_required`** |
| licence + access gate | one allowed, one refused, one gated repo; one JSON each; no download |
| app loop + card labels, dry run on a fixture | the three label states render; the refusal copy names the reason; `grep -r uncensored spaces/` empty |

Phase 0 makes a Space that can **tell a stranger whether their model is covered** and refuse
correctly. That is deployable and honest before any axis exists.

### Phase 1 — GPU: seed the bank (one base)

Refit Qwen3-4B-Instruct-2507 at the `f15s15-l935` recipe on the B70 (capture → fit → register →
bundle → `direction-export`), write `spaces/bank/qwen3-4b-instruct-2507/entry.json`, and apply the
bundle back onto the base's own Q4_K_M to measure the codec cos and the untouched-identity check on
the artifact class the Space will actually serve. Done gate: `plan-load --expect` passes, the
bundle digest matches the entry, and `bank-apply` on the base's Q4_K_M
(`verification/space/selfapply-qwen3-4b.json`) reports every edited tensor above `min_cos` at Q8_0
— min 0.9873 over 42 tensors, +0.30 GB — with 356 untouched tensors byte-identical. The ladder's
`auto` probe stops at Q6_K, which misses the floor (min 0.8815), so the delivered file and not the
probe decides the shipped type (§8).

### Phase 2 — GPU: the transfer kill gate

The fine-tune wave of §6. This is the phase that decides whether the product exists.
Done gate: a measured row on a fine-tune's Q4_K_M at one pinned config, or `verdict: kill`.

### Phase 3 — GPU/GATED: the live loop on ZeroGPU

Deploy the staged tree to a free personal account, run one full loop on the covered base, log GPU
seconds per call and wall clock per stage, download the output and **serve it** (`export-smoke`)
to prove it loads, and write the numbers into `verification/space/live-run.json`.

**Done (2026-11-24), and the answer is not the one the phase assumed.**
`verification/space/live-run.json` is one real run in `darioooooo0o/banked-gguf-edit`
(gradio 5.50.0, hardware `zero-a10g`) on the covered base's Q4_K_M: wall clock **964.95 s**
(coverage 7.284 / licence 0.058 / download 5.711 / edit **933.96** / read-back 15.346 /
package 2.558), container 192 cores, peak RSS 3.66 GB, download 437 MB/s, output
2,618,260,064 B sha256 `f29fe435…`, and the file **serves** (`llama-server`, `-ngl 0`,
`--ctx-size 4096`, first answer `OK`). Three measured facts follow: (1) **0 `@spaces.GPU`
calls, 0 s of the 300 s/day quota** — the edit is CPU numpy, so the quota is not the binding
constraint; the 16-minute wall clock is. (2) A CPU-only edit **must not** be routed through the
GPU queue: the first attempt wrapped it in `@spaces.GPU(duration=120)` and the job sat queued
after its 3 s download for over ten minutes. (3) The type ladder still stops at `Q6_K` and
lands at cos 0.8808, below the entry's 0.95 floor, while `Q8_0` clears. If the edit
exceeds 120 s, shard it by layer block across calls (≤ 240 s total) and prove the sharded output
is byte-identical to the single-call output. If the proxy verdict was `cuda_codec_required`, the
CUDA codec card runs first and must prove block-for-block identity against the numpy codec before
it may write a file.

### Phase 4 — delivery and coverage widen

OAuth `write-repos` push if direct download fails at 4.5 GB — **built and measured**
(`space_deliver.py`, `verification/space/oauth-upload.json`: 5.5 GB in 117 s, flag off by default);
a second bank family only after the transfer gate passes on the first one; every new entry carries
its own measured row or ships as `banked_unmeasured`.

## 11. What would prove this roadmap wrong

- **Transfer does not move refusals.** A base-fitted axis on a fine-tune's own artifact leaves the
  refusal rate where the unedited file was. Then the bank is a codec demo, not a product; say so
  and stop before widening coverage.
  **Measured, and this one happened:** on the top-downloaded Q4_K_M fine-tune of the first base the
  transfer gate did worse than leave the rate where it was — delivery fell from 19/40 to 1/40
  (held-out 63/120 to 20/120) at the pinned config, with retention intact and the word screen
  moving the other way. `verdict: kill` in `verification/space/transfer-qwen3-4b.json`. The bank's
  premise is dead on the class it ships to; the codec/fidelity path survives, the product does not
  (§6, measured block).
- **The edit cannot fit 240 s on 2 vCPU and the CUDA port cannot either.** Then the free Space is a
  1.7B-and-below product, and 4B/7B — the models people actually download — are out of scope.
  **Measured against this on 2026-11-24:** the live ZeroGPU run's edit is 933.96 s (single-core
  numpy, 192 cores available and unused), 3.2x the five-minute promise, with 0 s of GPU quota
  spent. The free Space is not blocked by the 300 s/day budget; it is blocked by CPU wall clock,
  so the sharded/CUDA edit is the only thing that can rescue 4B.
- **The Spaces proxy cannot stream 3-5 GB, and a Hub push from the container is slower than the
  request window.** Then "one click, five minutes" is false for every model above ~1.5 GB, and the
  honest product is a button that produces a *reproducible command* the visitor runs themselves,
  not a file. That would be a different roadmap.
  **Half measured, half not.** The push carries 5.5 GB in 117 s from the harness host
  (`verification/space/oauth-upload.json`), which fits the five-minute promise with 183 s to spare,
  and the direct download moved 2.5 GB at 437 MB/s in the live Space. Neither of those is a
  *container-side* push: that number is not taken yet. The clause that is measured false is the
  edit (§11, first bullet above), not the transfer.
- **The quota is 300 s/day per account.** A visitor cannot iterate: one attempt, then tomorrow.
  Any UI that implies retry-until-it-works is lying.

## 12. Explicit non-goals

No inference, no generation, no probes, no captures on the visitor's model. No "download our
pre-quantized model". No strength slider, no layer picker, no method picker. No uploading the
visitor's model to us (it only ever lives in the Space's ephemeral disk). No telemetry. No gated or
private repos. No multi-GPU, no `xlarge` (2x quota for a CPU-bound edit), no `torch.compile`
(unsupported on ZeroGPU). No coverage claimed from an architecture table: an entry exists only for
a base we fitted and measured.
