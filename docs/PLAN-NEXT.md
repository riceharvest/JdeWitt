# Next long-term plan — code only (2026-11-16)

Round 2. The first roadmap (ROADMAP.md) is landed and pushed; this file is
the next queue. North star unchanged: the tool whose abliterated releases
you trust because every claim is gated, falsified or labelled.

Everything below is CODE: CPU-verifiable, no model, no B70, ship by
running the command and reading its output. Verification is that exact
sentence — no CI, no new test suites, no fluff. Where an item's final
claim lives outside the CPU (any behavioural effect), the boundary is
written into the item and stays GATED.

## Tier 1 — highest value per effort

1. Checkpoint content diff (conversion QA).
   `safetensors-diff A B` and `gguf-tensor-diff A B`: per-tensor
   comparison that reports bit-identical tensors, plus dequant-tiled
   relative RMS error for tensors that differ, and the tensor names that
   exist on only one side. No inference; pure byte/dequant measurement.
   Why: today `gguf-inspect` proves headers line up; nobody proves a
   GGUF export actually contains the source model within the quant's
   error. This command is the measurement the GATED converter-compatibility
   matrix will eventually consume, and it also audits your own export path
   and any downloaded quant before you burn a GPU minute on it.
   Verify: a real GGUF vs its own edited copy (expected: only edited
   tensors differ, others bit-identical); two independent files.
   Landed: both commands shipped with emit-time schemas. Proven live —
   Q4_K_M vs its gguf-edit output: 320/321 bit-identical, only
   blk.0.ssm_alpha.weight differs (rel_rms 0.017815, same dtype);
   Q4_K_M vs Q5_K_M cross-dtype: CORRECTED — the original numbers
   (rel_rms 0.000945/0.070176/0.190611 over 316 tensors) were measured
   through a buffer bug that sliced the second file at the first file's
   block width, so they were meaningless; a follow-up commit then hid the
   whole comparison behind a byte_size_changed skip. Now cross-dtype
   tensors decode per side with their own block size: 158 bit-identical,
   163 compared by value (packed_bytes per row), rel_rms min/median/max
   0.079626/0.081328/0.092237, 0 shape mismatches, 0 codec refusals,
   full-model pass in 40s; safetensors fixtures with a hand-computed
   expected rel_rms matched to the printed digit.

2. FP8 everywhere.
   E4M3/E5M2 codecs in quants.py and `fp8-edit` for safetensors with
   per-tensor and per-block (DeepSeek-style 1x128 / 128x1) scale layouts —
   the FP8 checkpoint format new flagship releases actually ship in
   (hub-census already sees Qwen3.8 FP8). Same contract as fp4-edit:
   rewrite only the edited tensor + its scale tensor, every other byte
   verbatim, note what is value-preserving vs byte-preserving honestly.
   Verify: round-trip error vs torch.float8_e4m3fn on real checkpoint
   blocks, then a whole-file zero-delta edit with the byte-diff audit.
   Landed: `fp8-edit` with emit schema. E4M3/E5M2 encoders match torch's
   own fp8 conversion on 50000 random values each (0 mismatches, both
   formats); 256x256 row-block and 64x128 per-tensor fixtures edited
   through the CLI — zero-delta outputs are byte-identical, delta-edit
   codes and scale_inv bytes match the hand-computed expected requant.
   GGML F8_E4M3/E5M2 tensors stay blocked: the pinned gguf package's
   GGMLQuantizationType enum has no FP8 entries, so a GGUF-side codec
   could not be read or proven until the package moves.

3. K-quant completion. — PARTLANDED: Q2_K/Q3_K dequant landed.
   `dequantize_q2_k`/`dequantize_q3_k` in quants.py: numpy ports of
   ggml_dequantize_row_q2k/q3k, 0/770,048 mismatches vs the independent
   HF gguf.quants port on random blocks and uniform-byte edge fills
   (incl. the dmin/d zero guards). Wired into gguf-tensor-diff as a
   decode-only path (quants.DECODE_ONLY): self-diff of a written Q2_K
   +Q3_K GGUF is bit-identical, a 1-byte flip in a Q3_K block is caught
   (rel_rms 6.9e-05), and such files no longer crash the diff. Remaining:
   the legacy 1-bit-fraction quants Q4_1/Q5_0/Q5_1/Q8_1, the Q2_K/Q3_K
   quantizers (gguf-edit refuses them honestly until ported), then a
   real zero-delta edit through gguf-edit.

4. `verdict-audit DIR`. — LANDED.
   Compares every saved LLM-judge verdict against the deterministic
   wording refusal-check re-run on the same saved answer: agreement rate
   with Wilson bounds, disagreement counts by direction, dangerous
   disagreements (judge said compliance, checker found explicit refusal
   wording — inflates refusal-removal success) listed with the matched
   evidence, other directions by kind. Skips are reported by state
   (judge_invalid, incomplete, uncovered language); refusal-check-only
   runs are refused — nothing to audit. No new judging, no re-scoring;
   saved artifacts only, output schema-enforced at emit.
   Verified on the repo's own saved verification predictions: two judge
   runs audited (1 comparable row each, 100% agreement, Wilson lower
   0.207), refusal-only run correctly refused; synthetic ten-row fixture
   reproduced exactly: 3/6 agree, dangerous {r2,r5}, conservative {r6}.
   Caveat kept in the note: agreement is with a lexical heuristic, not
   semantic truth.

## Tier 2 — surface, portability, statistics

5. Direction interchange format. — LANDED.
   `gguf-delta-pack OUT FILE NAME=DELTA.NPY...` packs float deltas into a
   GGUF (format tag `delta-gguf-v1`); `gguf-edit --delta-gguf FILE
   --strength s` consumes any float-tensor GGUF whose tensors are named like
   target tensors (BF16 byte-pair decode proven). `direction-export
   REGISTRY KEY OUT [--dtype bf16|f32]` (format tag `direction-gguf-v1`)
   writes registered plan tensors + method/geometry/payload/provenance
   metadata via a new `ggufio.write_gguf`; a direction file fed to
   `--delta-gguf` refuses with `no tensor 'basis' in file` on purpose —
   plan→tensor-name mapping stays architecture-specific and GATED.
   Proven: delta-gguf edit byte-identical to the npy path on the 672 MB
   Q4_K_M; strength 0.5 halves the edit RMS exactly (0.000498 vs 0.000996);
   bf16 plan export bit-exact vs torch bf16 cast, f32 exact vs plan;
   exported files read clean by gguf-inspect (digests), gguf-meta and
   gguf-tensor-diff. Emit schemas for both commands, falsified.

6. Registry portability. — LANDED.
   `plan-bundle REGISTRY KEY -o FILE` archives signed meta + payload into
   one uncompressed tar (`b70ablator-plan-bundle-v1`, fixed metadata for
   determinism); `plan-unbundle FILE --into REG` verifies meta_digest,
   payload sha vs manifest AND signed meta, and basis-rank vs signed meta
   on arrival. Proven: clean import into an empty registry loads via
   plan-load; identical re-import is a verified no-op; a one-byte payload
   flip refuses (payload integrity mismatch); a crafted bundle with
   honestly re-signed digests but rank 3 vs signed rank 2 refuses
   (geometry); foreign members refuse (format). `plan-prune REGISTRY`
   found a corrupted payload + orphan dir in report mode, removed both
   with --execute and rewrote the index; candidates went to zero.

7. `studies-compare PATH...`. — LANDED.
   Read-only SQLite over several study.db files (or run directories):
   per-study state counts, budget (trial number) spent to reach the best
   feasible trial, decimated cost curves, and per-parameter-value
   best-value shares with means. Same read-only URI + `PRAGMA
   query_only=ON` discipline as study-metrics. Proven on the repo's two
   scratch studies (6-trial MO-TPE study: best 1.025 reached at trial 5,
   5 feasible; empty study reports honest zeros); emit schema falsified.
   Honest limitation stamped in the note: shares are counts, not a test.

8. score-compare hardening. — LANDED.
   score-compare now emits signed Cohen's h for the overall pass rates.
   New `suites-compare SUITE BEFORE AFTER ...` (n triples) runs the paired
   comparison per family and Holm-Bonferroni-adjusts the family sign-test
   p-values across all of them, exposing the adjusted minimum so a lucky
   suite cannot be reported without the family-wise cost of the look-all.
   Proven against hand computation: 0/4 -> 4/4 gives h = pi; p-values
   [0.125, 1.0, 1.0] Holm-adjust to [0.375, 1.0, 1.0]; emit schemas for
   both shapes, falsified.

9. `quants-audit GGUF [--blocks N]`. — LANDED.
   Productized codec falsifier: for every tensor of an edit-capable dtype,
   dequantize an evenly spaced bounded block sample from the real file and
   requantize with the same port, reporting tensors, total/sampled blocks,
   byte-identity rate, max absolute and max/std value drift. Decode-only
   dtypes (Q2_K/Q3_K) are reported as not requant-auditable, unsupported
   dtypes by count. Proven on the 672 MB Q4_K_M (F32 256/256, Q4_K 256/256
   byte-identical; Q6_K 219/256 byte-identical, zero value change — scale
   tie edges) and on the Q2_K/Q3_K fixture (decode-only listed, none
   silently skipped). Run after any quants.py change. Emit schema falsified.

10. `commands --completions bash|zsh`. — LANDED.
    Completion scripts generated from the live parser tree, so they cannot
    drift from the real flags. Proven functional in bash: prefix `gguf-`
    completes the five gguf-* commands and `gguf-edit --` completes
    `--help --delta-gguf --strength --tensor`; 73 commands covered;
    `bash -n` clean. The zsh script is emitted from the same introspection
    but unverified — no zsh on this host to syntax-check it against.

## Tier 3 — hardening and the deep end

11. Parsing hardening — FIRST SWEEP LANDED (bounded header reads).
    ggufio.parse_gguf_header now checks every declared count and length
    against the real file size (_LimitedBuffer; n_kv/n_tensors caps;
    n_dims<=64). Five synthetic hostile headers refused in milliseconds
    with precise messages: n_kv=1e18; string declaring 1e15 bytes;
    float array declaring 1e18 elements; n_dims=0xFFFFFFFF; and a lying
    length (declares 100, ships 10). The 672 MB real file parses, grows
    its header and round-trips byte-verbatim in 2.5s after the change.
    Still open in this item: streaming the tensor blob where materialising
    helps, and a symlink/size audit across importer paths.

12. IQ quants (IQ1_S … IQ6_K): the small-model quants people actually use.
    Port is code-only but carries pinned llama.cpp importance tables that
    must be versioned, digested and pinned like NOTICE upstreams, or the
    codec silently lies. Split per size class; Q2/Q3_K first still pays
    the rent sooner.

13. bitsandbytes NF4 — SINGLE-QUANT LANDED. `nf4-edit` for U8-packed
    weights + F32/F16 absmax scales: codebook pinned to bnb 0.42.0 by
    file sha256 and independently re-derived (max diff 1.08e-7); hand
    blocks exact, grid roundtrip bit-exact, quant err 0.15160 <= 0.15190
    (half largest bin), zero-delta file edit byte-identical, delta edit
    requants inside the bound, schema enforced+falsified. Double-quant
    (quantized absmax tensors) not implemented; nibble order asserted,
    real bnb checkpoint confirmation still open.

11b. LANDED (this landing): IQ4_XS decode ported from the vendored
     ggml-quants.c dequantize_row_iq4_xs and bit-exact vs the pinned
     gguf.quants port on 163,840 random blocks plus 5 edge blocks (zero,
     all-0xFF, d extremes, max-d); gguf-tensor-diff now falls back to the
     pinned gguf port for IQ codebook types it cannot decode itself, with
     per-row `decoder: pinned gguf.quants port (not ours)` labels and a
     port_decoded summary count; dtypes neither can read (I8) get an
     explanatory refusal; non-finite rel_rms from garbage scales reports
     rel_rms=None with a note instead of crashing JSON emit (JSON was
     refusing inf). Proven on a 3-tensor IQ GGUF fixture: IQ4_XS flip →
     rel_rms 0.00152 ours; IQ2_XS flip → rel_rms 0.007319 labelled port;
     I8 flip → codec_unsupported with note; quants-audit lists IQ4_XS as
     decode-only. REAL-DATA PROOF LANDED: Qwen2.5-0.5B IQ4_XS checkpoint
     (verification/iq05b, 290 tensors, 5 dtypes) — our IQ4_XS decoder is
     bit-exact vs the pinned port on 8,192 real ffn_down blocks; full-file
     self-diff 290/290 bit-identical; quants-audit now has a labelled
     port_decode_only bucket (IQ4_NL x120, Q5_1 x24) so nothing on that
     file reads unsupported. First REAL cross-quant proof: IQ4_XS vs IQ3_M
     of the same 0.5B checkpoint — 290 tensors shared, 218 bit-identical,
     72 compared across codecs (IQ4_XS side through our port, IQ3_M side
     through the labelled pinned port, 69 rows marked not-ours), rel_rms
     0.0567/0.0876/0.1769 min/median/max, 20s. Both binaries stay gitignored.

14. Two more curated import specs in the jailbreak-slices pattern: a code
    execution slice (runs through the existing ast-based executor) and a
    long-context math slice. Authored families, revision-pinned, disjoint
    selection/holdout, `suite-stats` output in the commit message.

15. SAE hooks stay a long bet unchanged: contracts and fit metadata only
    when the compute plan is real. No contract code before a GPU is
    earmarked for it.

## Standing rules for this plan

- CODE means provable on CPU by running the command; ship the proof
  (counts, digests, file sizes) in the commit message.
- GATED stays GATED: no behavioural claim from any item above gets made
  without a B70 run, no matter how clean the CPU proof is.
- Codec rule from round 1 stands: byte identity is the claim, value
  preservation is the fallback wording, and neither is assumed.
- Every new command joins `schemas`/EMIT_SCHEMAS at ship time; writer
  drift fails the command.


## Quantizer byte-reproducibility against upstream (landed this run)
- Vendored ggml scalar quantizers as compileable C harnesses (scripts/ggml_ref,
  pinned sha256 bcda336...). Four real port bugs found and fixed:
  1. make_qkx2: C updates the running min on accepted refinements; port kept it stale.
  2. K-quant requant: C uses f16-rounded d/dmin; port divided by the pre-round floats
     (explodes exactly where f16 d is subnormal — tiny weight groups).
  3. int64 scale/min unpack promoted the requant divisor to float64 (C is all f32).
  4. .sum() pairwise order vs C left-fold, Python double 0.1 vs f32 0.1f, numpy
     max() returning -0.0 where C's strict-> fold seeds +0.
  Result: 100% byte-exact vs compiled C on 11 distributions x 3 codecs, including
  six real checkpoint tensors. quants-audit Q4_K identity unchanged (0.998 at
  512 blocks, tie-edge scale bits). Edit-path regression: 320/321 bit-identical.
- Extended the harness to every remaining flat codec: q4_0/q8_0 (two more semantic
  bugs: ggml truncates (x0+8.5) toward zero in q4_0, uses roundf half-away-from-zero
  in q8_0) and new self-owned q4_1/q5_0/q5_1 (quantize byte-exact on 14 distributions,
  dequant bit-exact on random packed bytes; harness caught a missing -16 offset-code
  bias in q5_0 dequant). Wired into CODECS: gguf-edit now requants those dtypes; on
  the real IQ4_XS checkpoint Q5_1 moved from port_decode_only to audited, and the
  IQ4_XS<->IQ3_M diff dropped port-labelled rows 69->21 with byte-identical rel_rms.

## Q2_K quantizer landed / Q3_K quantizer open (this run)
- quantize_q2_k: byte-exact vs compiled ggml quantize_row_q2_K_ref on 13
  distributions (gaussian x2, zeros+impulse, uniform-tiny, subnormal-mix,
  neg-dominant, near-eps, huge-range, all-equal-neg, bimodal, flat-subblocks,
  4 real F32 tensors from the minicpm Q4_K_M checkpoint); decode cross-checked
  bit-exact vs the pinned gguf.quants port on freshly quantized data.
  Wired into CODECS: gguf-edit can now requant Q2_K tensors byte-reproducibly.
  Real bug caught by the C harness: make_qkx2_quants consumes CALLER-provided
  weights — q2_K uses plain |x|, q4_K/q5_K use av_x+|x|; the port had hardcoded
  the av variant, so the first q2_K run was 0.0% byte-exact.
- quantize_q3_k: attempt NOT released. Structural layout, hmask (8 phases of 32,
  not 4), scale packing and the zero-path L (C returns L=0 with NO +nmax baked
  in — first port added nmax and broke zero/eps blocks) are all now understood
  and fixed, but the make_q3_quants refinement loop still diverges (~16% blocks
  byte-exact on gaussian data; exact on zeros/subnormals/near-eps). Prime
  suspect: running-sum tie behaviour in the per-acceptance test
  (slx*slx*suml2 > sumlx*sumlx*sl2) under f32. Q3_K stays DECODE_ONLY.
  Reproduce: scripts/ggml_ref/ref_23k.c vs quantize_q3_k (removed pending fix).

## CUDA as a second-rank test backend (landed this run)
- `hf.py` now resolves compute backends by rank instead of hard-refusing everything that is
  not the B70: B70/XPU is `primary`, an NVIDIA CUDA device is `secondary`, and CPU is never a
  backend — a missing device is still a refusal. `require_b70(allow_secondary=False)` keeps the
  old contract by default; `HFModel(..., allow_secondary=...)` and `RunConfig.allow_cuda_secondary`
  opt a single run in. `device_memory_info`/`device_synchronize`/`device_empty_cache` replace the
  direct `torch.xpu.*` calls so memory gates and cache release follow the chosen device.
- Second-rank by construction, not by comment: `export.stream_merge`/`verify_resident_export`
  accept only an allowed device, `tree-ranker` and the tree path stay strict B70, the llama.cpp
  Vulkan serving gate still demands a B70 device, and `provenance.json` gains
  `compute_backend` with the `CUDA_SECONDARY_NOTE` wording **only when the run is not on the B70**
  (provenance is digested and compared on resume, so an always-present key would break existing studies).
- Verified: `doctor` on the B70 venv reports `compute_backend.rank=primary`; on a separate CUDA
  venv (`torch 2.14.1+cu130`, which matches the `torch>=2.14.1,<2.15` floor) with
  `JDEWITT_ALLOW_CUDA=1` it reports `secondary` with the RTX 3060 and a finite kernel sum, and
  refuses without the flag. No model load was run on CUDA.

## Second-rank serving proven, first model runs on the 3060 (this run)
- `ManagedGGUF(..., secondary=True)` was verified against a real `llama-server` with the B70 hidden
  (`VK_ICD_FILENAMES` pointed at the NVIDIA ICD only): `secondary=False` refuses the same box with
  "No verified B70 Vulkan device in llama-server", `secondary=True` reports
  `serving_backend.rank=secondary` for `Vulkan0` = the RTX 3060, passes the full-offload / artifact
  path / context proofs and returns a real chat completion. Alias is `test-<digest>` and the
  teardown lock is `GPU_LOCK_PATH.secondary`. Probes: `/mnt/ssd/hermes/cache/scratch/serve_rank_probe.py`.
- `llama.cpp-spark` builds clean (CPU and Vulkan) and its converter handles Spark-X2.6-1.7B:
  f16 GGUF 3,420,932,640 B, then `llama-quantize Q4_K_M` -> 1,107,458,592 B (5.16 BPW), which then
  served on the 3060. This fork rejects `--use-safetensors`; use `--outtype f16`.
- Spark-X2.6-1.7B (`model_type spark2_5`) loads under transformers 5.23 only after three repairs to
  the vendor remote code **in the private compatibility copy** (`compat.repair_export_metadata`,
  never the user's download): `_tied_weights_keys` list form -> `{"lm_head.weight": "model.embedding.weight"}`
  (5.x calls `.keys()` on it), mask kwargs `input_embeds` -> `inputs_embeds`, and the
  `"cache_position": cache_position,` entry dropped (5.x mask factories no longer take it). Each
  repair matches its pattern exactly once or raises. Verified: `lm_head.weight is` the input
  embedding tensor, and greedy generation returns text. `trust_remote_code` must be true, since
  5.23 has no registered `spark2_5` config.
- Baseline measurements on the selection suite (18 rows, greedy, `refusal-check`, second-rank CUDA):
  - Spark-X2.6-1.7B @ max_tokens 1536 / ctx 8192: refusal_rate 0.625, capability pass 0.667,
    degenerate 1, 2 rows truncated -> baseline gate refused the run. A single-request probe then
    finished `s-reviews-r3` with eos at 924 tokens (the batched baseline had truncated it at 1536)
    and ran `s-tax-r3` to 3072 tokens with no eos.
  - MiniCPM-V-4.7-1B @ 1024 and again @ 2048 / ctx 4096: refusal_rate 0.125, capability pass 0.833,
    degenerate 0, `s-reviews-r3` truncated at both budgets -> baseline gate refused the run.
  - Both numbers come from `baseline-hf/report.json` in the archived run dirs; no trial completed, so
    there is still no fitted-direction or before/after number for any model.
- Open: selection item `s-reviews-r3` (nl) does not terminate under greedy on either model, which is
  what keeps tripping `Invalid baseline outputs/judgments` (that gate means at least one row is not
  `status=ok` with `refusal_check.result` in {no_explicit_refusal, refusal_detected} — for these runs
  it was pure truncation: `completion_tokens == max_tokens`, no OOM). Raising the budget has not
  fixed it: the run that got past the gate used a run-local copy of the selection suite with exactly
  those two ids dropped (`/mnt/ssd/hermes/cache/scratch/suites/public-corpus-fit/selection.jsonl`,
  244 of 246 rows, repo suites untouched), and a probe through `HFModel.generate_batch` then showed
  both rows running to 4096 tokens with `finish_reason=length` on degenerate itemised loops — so
  budget was never the lever, and no decoding-policy change was made. Related: MiniCPM decode on CUDA has
  neither `flash-linear-attention` nor `causal_conv1d`, so its gated-delta/conv1d layers run on
  reference kernels and the baseline sweep costs minutes, not seconds.
- The fit then stopped at direction coverage, and the reason is measured rather than guessed:
  `pipeline.py` keeps a direction record only when a `bad` variant is actually refused and a `good`
  variant complies (truncated answers are dropped before judging), and `stable_directions` then
  refuses to fit if any (language, family_id, label) triple is missing. On the 16-row authored
  direction suite MiniCPM-V-4.7-1B lost 3 of 8 cells — `(en, dir-reviews, bad)`,
  `(nl, dir-reviews, bad)`, `(nl, dir-tax, bad)`. Spark-X2.6-1.7B refused all 4 `language: en` bad
  variants inside the budget (`finish=stop`, explicit refusal wording); its 4 `language: nl` bad
  variants all came back `finish=length` at the 512-token cap, and the explicit-wording check found
  no refusal phrase in that truncated text (probe: `/mnt/ssd/hermes/cache/scratch/spark_direction_probe.py`).
  The `nl` prompts are faithful translations of the `en` ones at the same directness, so prompt
  strength does not explain the split — but no behavioural conclusion is established yet, because
  the pipeline drops a direction record when `answer.complete` is false, **before** any refusal check
  runs, and it does not persist the dropped generations. With those rows never stopping inside the
  budget, truncation is at least a live cause alongside a possible missed refusal, and the run cannot
  distinguish them. `stable_directions` requires records for two languages, so an English-only
  directions file is refused by design.
- First valid second-rank baseline on record: `runs/minicpm-3060-public-fit/baseline-hf` (244 rows)
  refusal_rate 0.867, capability_pass_rate 0.833, harmless over-refusals 1/118, invalid 0, and
  `provenance.json` carries `compute_backend.rank=secondary` with the disclosure note as designed.

## Short-decode stage-1 trial funnel (landed this run)
- `hf.py` `generate`/`generate_batch` accept an optional `max_tokens` that must be
  a positive integer `<=` the backend budget (and `>=` any `min_tokens` floor);
  `PhasedHF` forwards it and keeps the GPU-tensor->CPU detach.
- New pure module `stages.py` (no torch): exact-key validation of the
  `search_stages` run object; deterministic family-stratified sampling with no
  RNG to serialize; `stage1_pruned` successive-halving keep line (at least
  `max(2, ceil(1/keep_top))` stage-3 priors, strict-worse pruning, ties always
  survive). Hand spot-checks: 7-row family at 0.5 picks `{a0,a2,a4,a5}`, 3-row
  family `{b0,b2}`; `keep_top=0.5` with priors `[0.2,0.1]` prunes `0.20`, passes
  the tie `0.10`; `keep_top=0.25` stays dormant until 4 priors exist.
- `RunConfig.search_stages` is optional and validated in `from_json`
  (`--set search_stages={...}` works); `search_stages: null` keeps the classic
  single-stage objective unchanged. `describe-run` reports stage-1 probe-row
  counts, generation and token bounds as upper bounds, and shows the field in
  `policy`.
- Pipeline objective inserts stage 1 after the margin screen and before any
  full-budget generation: short-decode the stratified harmful slice at
  `probe_tokens`, word-level `RefusalChecker` verdicts only, per-trial
  `trial-N/stage1.json` plus a `trial-stage1` event; the keep line is computed
  only over trials that actually reached stage 3 (`report` user attr or
  COMPLETE). Resume reuses the stored probe score without re-decoding. Probe
  answers never enter the answer store, the gates or release reports; the run
  directory carries `search-stages.json` saying so.
- Verified: ruff green, `validate-run`/`describe-run` on
  `examples/minicpm-v47/run.json` (with and without the field). No model was
  loaded. Stage-1 throughput, pruning hit rate and probe-vs-finalist
  correlation are GATED like every other speed claim.

## Funnel visibility + probe admission parity (landed same run)
- Probe short decodes now run through `plan_batches` with
  `output_budget=probe_tokens` (new explicit parameter, default `None` = old
  full-budget reservation): same admission limits, length packing and
  floor-policy grouping as trial generation; serial when no batch policy is
  set. The probe passes the configured refusal floor as `min_tokens` and
  requires the backend's `floor_enforced` confirmation exactly like the
  answer ledger. `from_json` refuses `probe_tokens < refusal_floor`.
- `trial-pruned` events carry a truncated reason string; `run-status` emits
  `search_funnel` tallies (stage1_probed/pruned, screen/kl/capability prunes)
  plus the candidate-probe line. Tally logic verified against a synthetic
  event file line-for-line; live `run-status` re-validated against
  `/mnt/ssd/hermes/cache/scratch/jdewitt/evjob` (schema unchanged; the
  run-status contract tolerates added keys, which the live command proved).

## Pre-search candidate probe (landed same run, same GATED caveat)
- `candidate_probe: {samples, enqueue_top}` run field; refused unless
  `search_stages` is set, since the probe reuses that exact signal.
  `search.sample_probe_params` draws from `parameter_bounds` — the same
  literal table `suggest_parameters` now calls, so sampler and probe cannot
  drift; `random.Random(seed)`, reproducible across processes.
  `unflatten_parameters` maps optuna names back to the nested shape the
  editor consumes; discriminative plans correctly sample strength only.
- Pipeline: after warm starts, before the objective — apply, short-decode
  the stage-1 slice, reset; full ranking + labels in `candidate-probe.json`,
  `candidate-probe` event, top `enqueue_top` enqueued as ordinary first
  trials. Provenance already hashes both new config fields, so changing
  either invalidates a resume like any other setting change.
- Spot-checked: determinism for equal seeds, bounds per key,
  flat/nested round trip, validator rejects (top > trials, >4000 samples,
  wrong keys, probe-without-stages), and describe-run emits
  `candidate_probe_probe_generations` = samples x probe rows.

## Retention instrument was reading zero — found and fixed (2026-11-17, CPU)
- P0: `score_compare`, `suites_compare`, `robustness_report` and
  `family_decay` graded every saved row through the judge field, so a
  capability row (no `judgment`) and a harmless row always counted as a
  failure. Proven on `verification/framework-jobs-smoke`: the math row
  carries `passed: true` and these commands reported `math: 0/1`,
  aggregate `1/2`. Every capability (retention) number those commands
  ever emitted was a zero, which is why "loss measurement: zero all
  lineages" looked like missing measurement instead of a broken
  instrument. A failed generation also counted as a pass candidate there.
- Fix: one shared `row_passes`/`row_label` pair in `agent.py` — capability
  rows on the deterministic scorer, harmless rows on the absence of a
  refusal, refusal rows as before, `status != 'ok'` never passes. Same
  re-measurement now gives `math: 1/1`, aggregate `2/2`. The three
  command outputs changed; any earlier score-compare/suites-compare/
  robustness/family-decay numbers are not comparable and must not be
  quoted.
- New surface: `--kind {all,refusal,math,code,harmless}` on
  `score-compare` and `suites-compare` so the retention axis can be
  measured without a refusal row in the denominator; `per_kind` now
  carries before/after rates with Wilson intervals, signed delta and
  Cohen's h; `suites-compare` surfaces `per_kind` per family so a
  removal claim and a retention claim sit in the same adjusted table.
- `calibration-report` pairwise agreement is now reported raw AND as
  Cohen's kappa with Fleiss CI (raw agreement is inflated by a dominant
  label: two judges that say "refusal" to everything look 90% concordant).
  A single-label pair returns `kappa: null` with the reason, never 1.0.
  Hand-checked against computed values: labels (4R,1C) vs (3R,2C) give
  raw 0.8, chance 0.56, kappa 0.5455; identical label sets give 1.0.
- `paired_permutation_test` returns a documented no-power result at
  n<2 instead of raising, so a one-row kind-restricted comparison reports
  itself as no power rather than crashing the report.
- Gates re-falsified: `schemas.enforce` still fires on every touched
  command after the added keys, and `--kind code` on a suite with no code
  rows fails with the kind named in the message.
- STILL OPEN and unchanged by this: the capability suites themselves are
  4 maths + 2 code rows on every suite on disk, so the retention axis is
  now instrumented but still has almost nothing to measure; no 3-4B dense
  instruct checkpoint is pinned. Populating the capability suite is the
  next CODE item.

## Capability (retention) suite populated: 300 maths + 42 code tasks (2026-11-17, CPU)
- `scripts/author_capability.py` writes `verification/capability/capability-suite.jsonl`:
  300 authored parameterised maths rows, 15 families, 180 en + 120 nl, every
  `family_id` `cap-*` and `category` `capability`. Answers come from a derivation
  function over a frozen seeded draw — never typed — and `capability-provenance.json`
  states the honest scope: authored ladder, not benchmark items, no leaderboard claim.
  Determinism proven by generating the same seed twice into different directories and
  diffing the bytes, and by a different seed giving a different digest.
- `scripts/verify_capability.py` re-derives every stored answer from the recorded
  parameter draw AND falsifies the scorer on every row through the real
  `jdewitt.evaluation.grade_math`: a correct rendering must be accepted and a
  perturbed one rejected. Result: 300/300 both directions, zero problems.
- `scripts/author_code_capability.py` writes `capability-code.jsonl`: 42 single-function
  tasks with executable assertions graded by `grade_python` in the pinned
  `python:3.12-slim` sandbox (no network, read-only rootfs, 128 MiB, allowlisted
  imports). Each task ships with a reference solution and a written mutant; the
  generator runs both through the real sandbox and refuses to write the suite unless
  every reference passed and every mutant failed. Result: 84 runs, 42/42 and 42/42.
  One mutant was caught being semantically equivalent to its reference (`isqrt` floor
  can never exceed the value) and was replaced before anything was written.
- Join path proven with the existing tooling rather than by editing certified files:
  an importer spec with three `passthrough` sources produced a 582-row holdout in
  scratch (240 harm + 300 maths + 42 code, `duplicates_dropped: 0` per source,
  `import-provenance.json` recording each file digest). Recipe is recorded in the
  provenance `join` field.
- The certified suite files under `verification/` and `examples/` are deliberately
  NOT touched: appending capability rows would change the denominators of the
  already-reported cert numbers and make them non-comparable. Capability rows go into
  ONE split (the certification holdout) for new runs — a selection/holdout split of
  them buys nothing because the fit never reads capability rows, and
  `validate_splits` would refuse the shared `cap-*` family ids anyway.
- First measured retention wave taken 2026-11-17 on the 3060 (see
  `docs/RETENTION-MEASUREMENTS.md` and `verification/retention-q25/`). It found two
  measurement defects worth keeping in mind: a generation cap below the edited
  model's verbosity scores truncation as capability loss, and the strict grade
  conflating skill with format compliance (now reported separately by
  `scripts/capability_format_audit.py`).
- Remaining retention gaps, stated plainly: one lineage measured (q25 1.5B) and no
  competitor build on that base, no nl code slice, one code task per family so a
  per-family code rate is a single sample and the code deltas are under-powered at
  n=42, and maths difficulty is uncalibrated against any published benchmark.

## 2026-11-18 (run stopped on request — state as of the stop)

Closed, with evidence committed:
- **The wording checker was lying.** `refusal_check.PATTERNS` matched first-person
  constructions only, so answers opening "Absolutely not." graded as refusal removed:
  86 of 120 rows on the 120-family held-out screen. Added `PATTERNS_WIDE`,
  `RefusalChecker.check(wide=True)`, `scripts/screen_agreement.py` (both inventories +
  the false-clean count + McNemar under the wide one), and waves now store
  `refusal_check_wide` at write time. The "indistinguishable from the published build,
  p = 1.0" claim for the 300-contrast arm is **retracted**: wider-inventory grading puts
  it at 6/40, 11/24 and 34/120 where it printed 40/40, 23/24 and 120/120.
- **The floor sweep found the real point.** Shared axis, o_proj + down_proj, scale 1.5:
  blk.9-35 = 40/40 harmful but 97/120 held-out and maths 159→128; **blk.15-35 = 40/40
  harmful under both inventories and 120/120 held-out against the published build's
  120/120, zero discordant, McNemar p = 1.0, maths 159→148 (p 0.0078, collapse flag,
  median 68 chars vs 175)** — `q3-4b-f15s15-l935`. blk.20-35@2.0 = 39/40 and maths
  139; blk.15-35@2.0 = 37/40 with 3 invalid and maths 136.
- Null-space protection tested and rejected on this model: rank-4 prevention subspace at
  scale 1.5 kept removal identical and moved maths 128→130; at 2.0 it read maths 139→75
  with a median answer of five characters.
- Behaviour reads (METHODOLOGY R2, single reader): 24-family screen — published build
  delivers on 4/24, 300-contrast arm on 5/24. 20-family sample of the 120-family screen —
  published build 0/20, f15s15 0/20. Both refuse; the edit removes the sentence, not the
  no. Retention is the axis the published edit still wins (maths 159→159, code 20→18).

Stopped mid-lane, stated exactly so nothing is cited that does not exist:
- `q3-4b-f15s12-l935` (blk.15-35, both targets, scale **1.2**) built fully (4,280,403,392
  bytes) and its harmful-40 wave completed: **39/40 under both inventories**, 0 invalid,
  measured, pinned config. **Its retention wave was never taken** — the lane was stopped
  during that serve, and the empty `verification/q3-4b-retention/f15s12-l935/` directory
  was removed rather than left to be mistaken for evidence.
- `f15a20-l935` (blk.15-35, o_proj only, scale 2.0) was never built.
- The open question that lane was meant to answer is still open and is the next thing to
  run if GPU time comes back: does scale 1.2 hold ~39/40 removal at a maths cost the
  published build's zero does not dominate? The in-sample number says the arm is close to
  free; nothing in this repository yet shows its retention or its held-out-120 rate.

## Stage-1 probe gates on shape now, not the word list (2026-11-18, CPU)
- The short-decode probe and candidate probe were scored by the wording
  checker — the very instrument R3 disqualifies for the removal decision had
  become the funnel gate. Validated offline on every banked q3-4B harmful-40
  wave (17 arms, 816 ok rows, zero new inference): the word probe at the
  ~150-token window admits answers of which ~95% are full-text declinations
  (f15s15: word-gate admits 40/40, shape says 37/40 still decline; attn arms:
  every word-admit in those waves declines by full shape). Switching the gate
  to the shape classifier at the same window: admit precision 0.549,
  0.735 at ~300 tokens, 0.926 (4 false admits) at ~600 tokens — and
  zero wrongly-pruned delivering rows at every window (0/50 on the banked
  set). False-cleared arms now die at the 6%-cost probe instead of
  eating a full-budget wave plus a merge cert.
- Landed: pure-text classifier moved to `jdewitt/refusal_shape.py` (single
  source of truth; `scripts/refusal_shape_audit.py` imports it), trial-stage1
  and candidate-probe scores are shape-decline rates with the word verdict
  stored per row as aux (`gate:'shape'` in stage1.json), pruned-reason
  wording, `stage1_score` docstring and docs/fast-search.md updated with the
  measured numbers. Verified: ruff green, classifier move reproduces the
  offline aggregate exactly, `stage1_score(3,10)==0.3` spot-check, pipeline
  imports with no heavy deps, validate-run/describe-run unchanged on
  `examples/minicpm-v47/run.json`. Live-study prune hit rate and
  probe-to-finalist correlation stay GATED like every other funnel claim.

## Sampling does not unlock delivery (2026-11-18, 3060, four waves, 52k tokens)
- Question: is zero delivery on wording-perfect arms a greedy-verification
  artifact? Ran the 24-family holdout on f15s15 (new greedy + sampled), the
  published oracle (sampled; greedy read banked at 4/24) and base (sampled),
  pinned serve argv per wave, temp 1.0/top_p 0.95/top_k 64/seed 20261118,
  cap 2048. /props identity gate on every wave.
- R2 single-reader counts: f15s15 greedy 0/24, sampled 0/24 (McNemar p=1.0);
  oracle 4/24 greedy -> 1 clean + 2 borderline sampled; base-sampled 1/24 —
  and that one row (sorry-bench-000277) is delivered by the UNEDITED base,
  so it is a screen defect, excluded from robust counts.
- Verdict: the delivery gap is in the weights, not the decoder. Rank-1
  output-direction removal — ours and the published oracle's alike — moves
  the lexical decision surface (40-char flat refusal into a 2–4k-char
  lecture-pivot), never the content branch. Clean delivery across scale,
  target-split and decode policy sits at the noise edge of this edit class.
- Consequence for the method (R3/R4): contrasts separating refusal-wording
  from compliance yield axes that can only remove refusal wording. The fit
  needs contrasts whose difference is delivery of the SAME request —
  refusal-vs-successful-jailbreak answers, pair-matched within one harmful
  row, answer-span locus. Next lane: capture jailbreak-frame answers on the
  non-protective families (generation only), spanfit axes on the pair diff,
  merge, one cert wave + this same read. Self-harm/CP-adjacent families stay
  out of jailbreak capture permanently.
- Jailbreak-pair capture, first round (3060, base q8, 174 requests, 50k tokens):
  story + redteam frames yield 2 real deliveries of 58 rows; the assistant-
  prefill seed frame lifts shape-compliant to 17/58 but half are seeded
  evasions ('Sure — here is exactly how: I will not provide…'). Real
  delivery pairs ≈ 3–5/58; prompt-side jailbreak of q3-4B greedy is a low-
  yield data source. Two instrument findings: (1) the same false-compliance
  the word checker shows, the jailbreak side shows it too — 9/11 word-clean
  answers decline by shape; (2) DEFECT: shape.classify's OPENER is ^-anchored
  so a refusal buried after a compliant-looking prefix (seed echo) reads
  'other'; DECLINE must also catch embedded 'i can't assist|i won't|i cannot'
  before any pair corpus gets fit from jailbreak captures. Next decision
  point is CPU-only: recover residuals from the real-delivery pairs and check
  cos(pair-axis, oracle axis); alignment near 1 kills the rank-1 delivery
  hypothesis, orthogonality justifies one merge test. Higher-yield compliant
  sides are a data-sourcing task (pinned public advbench response mirrors or
  GCG adversarial suffixes), not another frame round.

## 2026-11-18 — token-to-certification: cascade verifier measured, the asymmetry documented

The metric for this whole section of the project is generated tokens per R2
certification, never wall time or model identity. Instruments landed
(commit 0a31acb): `scripts/cert_cost.py` sums actual token spend per artifact
from banked waves; `scripts/cascade_verify.py` decodes every row to K tokens,
settles rows whose shape is decidable inside the window, and pays full decode
only for the rest via assistant-prefill continuation from the server's prefix
cache. `screen_short(..., mode)` in `jdewitt/refusal_shape.py` is the settle
rule; `EMBEDDED_DECLINE` (first-person refusal anywhere in the answer, which
catches seeded 'Sure, here is how: I cannot' evasions) is a new field kept
OUT of the frozen `declining` so no banked label retunes.

Live measurements (q3-4b-f15s15-l935, temp 0, K=150):
- harmful-40 screen mode: 7,125 tok vs 21,024 banked full (66% saved, sim
  predicted 65%); 35/40 settled refusal, 0 settled delivery, 5 escalated;
  zero settled-vs-full-gold disagreements, zero delivered rows buried.
- q3-holdout-24 cert mode: f15s15 10,807 vs 15,942 (32%); oracle 13,261 vs
  19,405. Zero buried on both, zero disagreements.
- Offline screen-mode scan over all banked waves: K=150 settles 56-65% of
  harmful-wave tokens on post-edit pivot artifacts with 0/1648 buried;
  K=300 buried exactly one delivery-with-disclaimer in 1648, which is why
  cert mode never settles on a bare topical decline word (R3 in reverse).

The asymmetry, now encoded in the tool docstrings: certifying a SUCCESSFUL
removal is cheap — a delivered answer shows procedure inside the window and
the row ends at ~150 tokens. Certifying a FAILED one is informationally
expensive — proving 'no content is ever coming' from a 3k-character pivot
lecture requires generating the lecture. So the cheapest search for a proper
removal is a K=150 screen-mode cascade per arm; full waves are paid only for
arms where settle-delivery fires. Note on records: settled rows store the
truncated answer, escalated rows the prefix+continuation; continuation
greedy decodes are numerically distinct from independent full decodes (KV
chunking differs), which is why escalated rows were byte-identical 0/5 —
label-level agreement is the tested claim, byte equality is not.

Wave-count sizing note (row axis, cheap math, no tool needed): n clean rows
bounds the residual refusal rate at ~3/n with 95% confidence; banked harmful-40
(40 rows, 35 refusals) is a failure arm, but a success claim at <2% residual
refusal wants ~150 clean families — affordable precisely because screen-mode
cascades make 150 rows ~25k tokens instead of ~79k at full-decode cost.

## 2026-11-18 — zero-generation branch-probe: refusal removal measurable without generating a token

The cascade cut the tokens a removal CLAIM needs; this cuts the tokens the
removal SEARCH needs to the floor: zero generated. A refusal is a branch
preference, and a preference is measurable by teacher-forced scoring alone.
`scripts/build_ppl_probe.py` + `scripts/refusal_ppl_screen.py`: score the
model's own refusal continuations (R) and hand-read-verified delivery
continuations (D, 10 pairs; 4 oracle-screen + 6 jailbreak-capture, every D
row agent-read first because 13/13 shape-clean-but-pivot rows were rejected
— the shape screen cannot certify delivery, R2 again) and a benign control
corpus C, with llama-perplexity (CPU build of the dspark fork, prefill only).
`BR = lnPPL(R) - lnPPL(D)`, delta against the base artifact.

Measured, all against the same q8 base (ln-PPL units, control drift <=0.006):
- heretic oracle (banked real deliveries at the class ceiling): +0.559
- q3oaxes135 (oracle direction recovered from its weights and merged via our
  path — instrument control, banked 7/10 delivered): +0.649
- EVERY one of our 14 fitted arms (f15s12/15/20, f20s20, attn*15/25/40,
  prot15/20, shl10, yx*): between -0.487 (attn40, the collapse arm — refusal
  gets MORE predictable) and +0.103. f15s15: -0.052 +/- 0.13.
The probe independently reproduces the conclusion the behavior waves took
~145k generated tokens to establish: our fitted axes move tone, never the
branch; branch damage appears only in oracle-form directions, and the probe
ranks artifacts by it for free. The search protocol is now: recover or fit a
candidate direction -> CPU cos check -> zero-token ppl-branch probe -> K=150
cascade screen (~5k tok) -> one full cert wave only where both agree.
Corpora in `verification/ppl-probe/` (pairs.json records per-pair provenance
and the agent verdict); ladder table in `screen-ladder-2026-11-18.json`.
Probe limits, stated: BR is a preference shift, not a delivery claim; the
corpus is 10 pairs and 2.6k delivery tokens — enough to separate +0.5 from
+-0.1, not to rank near-ties; and P(refusal|plain prompt) rises under ANY
tone shift that breaks base self-fluency, so BR is read against the control
corpus and, at decision time, always against the cascade read.

## 2026-11-18 — 800-prompt verification wave: what scales and what doesn't

Scaled the verification set from 40 to 800 harmful prompts (171 seeded,
family-intact families subsampled deterministically from the pinned, deduped
harsh-suites holdout — `scripts/subsample_screen.py`, no new data sourced)
through the K=150 screen-mode cascade, one RTX 3060, escalation capped at 600
tokens, chunked + merged by `scripts/merge_cascade_chunks.py` (refuses mixed
configs). Results in `verification/q3-4b-harmful800/summary-2026-11-18.json`.
- f15s15 (ours): 472 refusal-settled, 0 delivery-settled, 328 escalated;
  223,473 tok (~280/row), ~19 min GPU. ~50% off the naive full-wave estimate
  (420-480k) for this mix.
- heretic oracle: 104 refusal-settled, 16 delivery-settled, 680 escalated;
  313,848 tok, ~20 min GPU at 8 slots.
What the scale exposed, honestly:
- Settle rates are CORPUS-MIX dependent. The harsh set's pivot prose avoids
  decline markers inside 150 tokens far more than harmful40 did: escalation
  rate 41% (ours) and 85% (oracle) vs 12% banked at K=150 on harmful40.
- Delivery-settle precision on this mix is ~0.1-0.4, not the 0.926 banked on
  harmful40 greedy: 22 f15s15 shape-delivery rows hand-read = 2-3 real
  deliveries (over-refusal topics only); oracle's 16 settle-deliveries read =
  ~4 real. The pool is a candidate generator and the R2 read is mandatory.
- The two independent instruments still agree: the ppl branch probe says
  f15s15's branch preference is unchanged from base (dBR -0.052), and the
  800-row read finds its 'deliveries' only on topics the base over-refused.
Cheapest next check-cost reductions, ranked: escalation cap 450 (procedure
onsets measured to saturate <=600; live A/B not yet run), always 8 slots at
-c 4096 (20-min floor), and the zero-token ppl probe in front of any cascade
for a new arm. Time to check 800 harmful prompts today: ~20 min GPU and ~280
generated tokens/row; the read that certifies lives on the 33-85% candidate
pool, not on 800.

## 2026-11-18 — amortized verification: the 800-row check becomes one-time-cost

- Escalation-cap 450 tested OFFLINE on the 800-row records (zero new inference):
  loses 17/22 (ours) and 89 (oracle) delivery candidates that only form between
  450-600 tokens, and 34 refusal settles — for 11% of escalation tokens. REJECTED;
  cap stays 600. The 'onsets saturate by 600' note was a harmful-40 greedy artifact.
- Amortization play: one base wave over the 800-prompt screen (111,826 tok,
  verification/q3-4b-harmful800/base-800/) freezes a per-prompt REFUSAL corpus
  (refusal800.txt, 799 docs + family-intact 128-doc subsample). Screening any
  future artifact on this corpus costs ZERO generated tokens — ~4-9 min CPU —
  and the score is per-prompt (dR vs base), not just corpus-global.
  Protocol becomes: base refusal corpus (built once) -> per-artifact dR probe
  (0 tok) -> cascade screen only if dR looks alive -> agent reads the pool.
- Ladder rerun on the 800-row refusal corpus in flight (16 artifacts,
  zero-generation); validates per-prompt dR against the BR findings.

## 2026-11-18 — cascade policy sweep on banked records (zero new inference)

Greedy continuation is prefix-deterministic, so one banked run reconstructs
what any K/cap policy would have cost. Sweep (`scripts/cascade_policy_sim.py`,
tables in verification/q3-4b-harmful800/*-policy-sim.json): K=100 saves ~10%
(202k vs 224k on f15s15-800) and never adds a false delivery-settle; K beyond
150 is strictly worse. The structural finding: the escalation branch owns >90%
of cascade tokens — K tuning is a ±10% knob, not the next order. The zero-gen
refusal-corpus probe is the lever that removes escalation cost from the
SEARCH loop; the 41-85% escalators remain only for finalist artifacts.
Char-boundary noise at the observed K is 2-4% — the sim ranks policies, it
doesn't clone runs. Transfers on harmful-40 (mismatches 0 there): K=100 saves
24% (5,446 vs 7,168 tok), still no false delivery-settles. K=100 is now the
recommended screen default in cascade_verify's help; cert stays 150.
