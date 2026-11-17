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
