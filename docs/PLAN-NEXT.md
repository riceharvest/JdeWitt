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
   Q4_K_M vs Q5_K_M: 316 of 321 tensors differ across dtypes,
   rel_rms min/median/max 0.000945/0.070176/0.190611, 0 shape mismatches,
   0 codec refusals, full-model pass in 94s; safetensors fixtures with a
   hand-computed expected rel_rms matched to the printed digit.

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

6. Registry portability: `plan-bundle KEY -o FILE` (single-file archive of
   plan + payload + meta + gate state, digest-pinned) and `plan-unbundle
   FILE --into REG` which re-verifies every digest on arrival and refuses
   tampered or geometry-mismatched entries. `plan-prune` stays
   report-only unless `--execute`, same discipline as corpus-cache.

7. `studies compare DB1 DB2 ...`: read-only SQLite over several
   study.db files — per-parameter and per-method win rates, budget spent
   per accepted finalist, cost curves. Same read-only URI + `PRAGMA
   query_only=ON` discipline as study-metrics.

8. score-compare hardening: effect size (Cohen's h) alongside the Wilson
   intervals, and Holm–Bonferroni adjustment across suites so the
   "family sign test p=0.0117" class of results cannot be cherry-picked
   across ten suites.

9. `quants-audit [GGUF]`: productize the codec proof — for each dtype in
   a real file, dequant/requant the bounded block slice, report compressed
   byte count, block count and max round-trip error. This is the standing
   falsifier for every codec in this repo; run it after any quants.py change.

10. `commands --completions bash|zsh`: generate shell completion from the
    same argparse tree `commands` already introspects. Agents and humans
    get tab-completion for free, no hand-maintained list to drift.

## Tier 3 — hardening and the deep end

11. Parsing hardening sweep: bounded reads in ggufio header parsing (a
    5 MB vocab array is legitimate; a 4 GB length field is not), stream
    the tensor blob instead of materialising tile + output where it helps,
    symlink and size-limit audit across importer/release/registry paths.
    Falsify with synthetic hostile headers (they are probes, not tests).

12. IQ quants (IQ1_S … IQ6_K): the small-model quants people actually use.
    Port is code-only but carries pinned llama.cpp importance tables that
    must be versioned, digested and pinned like NOTICE upstreams, or the
    codec silently lies. Split per size class; Q2/Q3_K first still pays
    the rent sooner.

13. bitsandbytes NF4 (HF-quantized checkpoints, single- and double-quant):
    one more source format `direction-export`-adjacent tooling can read.
    Same block codec + real-file byte identity proof as every other codec.

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
