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

3. K-quant completion.
   Q2_K and Q3_K (S/M/L subtypes), then the legacy 1-bit-fraction quants
   Q4_1/Q5_0/Q5_1/Q8_1. Sub-4B territory lives on Q2_K/Q3_K and
   `gguf-edit` currently refuses them. Port from the pinned GGML sources
   already used for Q4_K/Q5_K/Q6_K; identity-prove on real blocks from
   small GGUFs (compressed byte count + dequant equality like the Q6_K
   audit), then a real zero-delta edit through `gguf-edit`.

4. `verdict-audit SUITE DIR`.
   Compare every saved LLM-judge verdict against the deterministic
   refusal-check verdict for the same row: agreement rate with Wilson
   bounds, disagreement rows with both verdicts and the trigger that
   fired, and refusal-check rows excluded from LLM scoring as they should
   be. This turns judge trust from prose into a number that feeds the
   calibration ledger. No new judging, no re-scoring; only saved
   artifacts. Verify on the existing verification suites' saved
   predictions.

## Tier 2 — surface, portability, statistics

5. Direction interchange format.
   `direction-export PLAN OUT.DIR.GGUF --strength s` writes fitted
   directions as a GGUF (F32/BF16 tensors + method/layer/gate/provenance
   metadata in the same header gguf-meta already understands), and
   `gguf-edit --delta-gguf FILE` consumes it. Directions become portable,
   inspectable, signable files — `gguf-inspect` already prints their
   metadata. File layer is CODE; applying deltas to a live model stays
   GATED like every other behavioural claim.

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
