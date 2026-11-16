# Roadmap to the best refusal-removal tool

North star: the tool whose abliterated releases you can trust more than
anyone else's, because every claim is gated, falsified or labelled. The
moat is method breadth plus verifiable honesty; everyone else overclaims.

Work is tagged:
- CODE — CPU-verifiable, no model, no B70. Ship by running the command.
- GATED — needs real B70 / real checkpoints before any claim changes.

Status: every CODE item on this roadmap is landed and pushed. Highlights:
quant-native editing (GGML Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/F32/BF16 gguf-edit,
MXFP4/NVFP4 fp4-edit for safetensors), method breadth (nullspace,
answer-mean rEF capture, family-weighted whitened-SVD, multi-objective
TPE with Pareto-front finalists), evidence chain (chained claims ledger,
ed25519 release signing with repro index, published artifact +
command-output JSON Schemas enforced at emit time, sha-bound fitted-plan
registry with gate labels), agent ops (stable error codes,
about/studies/study-metrics/run-cancel/run-events/describe-run,
--set/--preset, abliterate flow, events, injectable clock, GPU lock,
per-job budgets), and offline evaluation (Wilson + bootstrap + paired
permutation stats, score-compare, judge-ensemble, score-sensitivity,
robustness-report, family-decay, suite-stats/suite-diff/suite-near-dup,
corpus-cache, curated jailbreak-slice import spec). Still GATED (needs
real B70 / real checkpoints, by design): method-winner claims,
quant-edited model behaviour, judge calibration gains, throughput claims,
per-family preset profiles (the preset mechanism ships; presets without
evidence references are refused at load time).

No CI and no new test suites on this roadmap. Verification is: run it,
read the artifact, keep the honest claims table current. The existing CPU
suite stays as-is.

## Phase 0 — Operations surface (CODE)

Everything here makes the tool drivable by an agent at scale, which is
how breadth happens later.

- Stable machine-readable error codes for every expected failure
  (errors.py already separates them; give them codes, document the table).
- Published JSON schemas for run-status, studies, describe-run, validate-auto/
  abliterate dry-run, methods and architectures output; the CLI enforces them
  at emit time, so a writer that drifts fails the command instead of a test
  harness. Artifact schemas (files/evidence/signature/scoring-manifest) ship
  under `schemas` / `verify-artifact`.
- `describe-run CONFIG`: static budget report — trial accounting,
  family-disjointness matrix, projected generation volume, suite slices.
  Pure config arithmetic; refuse anything validate-run would refuse.
- `studies list` / `study-metrics` reading the optuna sqlite directly;
  `run-cancel` writing a durable cancel flag honoured at trial boundaries.
- `--version` and `about`: tool hash, NOTICE-pinned upstreams, dependency
  versions, asset digests (the provenance auto.py already assembles).
- `suite-diff`, `suite-stats`, corpus-cache `gc/prune`.
- transport: cap generate_batch fanout (unbounded today), bounded retry
  with backoff for idempotent count/models calls only.
- atomic-write sweep over every artifact path; one clock injected into
  pipeline/chat/release.

## Phase 1 — Break the VRAM wall: quantized-native editing

The single biggest differentiator. Today the original-precision model
plus workspace must fit one B70, NF4 trial weights are rejected, and
several published MoEs are simply uneditable regardless of adapter. No
public refusal-removal tool edits quantized sources natively.

- Streaming quantized source edits: dequant-tile, apply the exact FP32
  addition and final cast used by search/export, requant with the pinned
  importance matrix. Q4_K/Q5_K first, then NVFP4/MXFP4 safetensors.
  Same tensor-verification contract as merged-resident: every stored
  finalist checked against the operator that produced it.
  Landed: `gguf-edit` (tile-wise, GGML Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/F32/BF16)
  and `fp4-edit` (MXFP4 E8M0-block32 / NVFP4 E4M3-block16 + global scale,
  E2M1 RNE ties-to-even, safetensors byte-verbatim outside the edited
  tensor pair). Editing the merged-resident model inside a live run stays
  GATED behind a real B70 edit+load proof.
- This reopens NF4/quant trial weights honestly: the guarantee comes from
  the requant round-trip check, not from pretending a dequantized base
  equals the source.
- Consequence: 27B+ editable on a B70 or a 16 GB card; quantized-model
  abliteration with measured quantization damage instead of hope.
- MoE router interventions and shared/fused-expert aware edit placement
  (upstream-unresolved list includes routers). GATED after the CODE part
  lands, since router damage is real.

## Phase 2 — Method space

Today: directional, SVD, projected, MPOA, whitened-SVD, SOM, LEACE,
dense ARA, trained SFT/DPO edit factors. To be the best, cover the ideas
that beat plain last-token SVD and make the searcher arbitrate them.

- Null-space projection family (refusal-direction removal / concept-null
  space) as a first-class method with fit metadata.
- Multi-token / mean-difference directions (RepEng-style rEF): directions
  fitted on answer-position means rather than a single final token.
- Automatic layer localization via clean/corrupted residual patching
  using the existing lens/hook infrastructure, feeding per-layer
  strength bounds instead of hand-picked ranges.
- ICA/whitening variants where SVD is known to mix features.
- Direction quality upstream: contrast-pair mining that enforces topic-
  matched families across more languages; whitening already exists,
  extend to per-family weighting.
- Plan library: fitted plans live in a local registry keyed by run
  provenance and sha256-bound to the exact checkpoint payload.
  `plan-register RUN_OUTPUT [--gates EVIDENCE|--allow-unmeasured]` refuses
  unprovenanced or tampered fits, `plan-list`/`plan-show`/`plan-load` carry
  gate state and refuse geometry mismatches on transfer. Unmeasured entries
  are labelled forever. Runs consume the registry through the
  `warm_start_registry` run-config field: a fit transfers only when method,
  options, suite digest and every weight-geometry key (checkpoint fingerprint
  minus model/revision) match; ineligible or fingerprint-less candidates are
  recorded in `direction-stability.json` and the run fits from scratch.
  Proving a real cross-revision transfer on B70 checkpoints stays GATED.
- GATED: which of these actually wins is decided by the searcher on real
  checkpoints, never by prose.

## Phase 3 — Objective and search economics

The objective is what makes results better, and judge cost is an
explicit unresolved limit in the README.

- Multi-objective search: compliance, full-reply KL, paired task
  regressions, refusal margin, answer length/damage. Optuna MO-TPE,
  Pareto front surfaced in run-status; the existing single minimize
  becomes one configured point on it.
  Landed: run-config field `objectives` (2-4 unique metrics from
  search_rate/full_reply_kl/answer_kl/capability_drop, first is primary),
  keyed-mo-tpe-v1 study contract that refuses cross-contract resumes,
  vector objective returning the same already-computed trial metrics,
  Pareto-front finalist selection ordered by the primary metric, and the
  front surfaced in run-status. Plumbing is CPU-proved with a stub
  objective; the real objective on a B70 study is GATED like every other
  claim.
- Judge economics, in order of payoff: cheap prefilter (screening and
  refusal-margin already exist — extend to candidate ranking),
  judge-size sweeps per model family to answer the open cost question,
  offline ensemble aggregation over saved verdict files (pure CODE).
- Partial-answer early stopping: kill trials after a stratified answer
  subset if they cannot reach the gate; reuse the durable answer cache so
  no generation is wasted.
- Significance without ceremony: one stats module (Wilson intervals,
  family-level bootstrap, paired permutation test). Spot-checked against
  scipy once.

## Phase 4 — Evaluation breadth (the trust moat)

Breadth of honest evidence is what "best tool" means in practice.

- Importer already covers Hub/CSV/JSON corpora; add curated jailbreak
  and capability slices as first-class suites: StrongREJECT, JailbreakBench,
  HarmBench-style held-outs, code exec tasks via the existing ast-based
  executor, long-context math. Family-ID discipline unchanged: authored
  families, no invented groupings.
  Landed: `examples/jailbreak-slices/import.json` — revision-pinned
  StrongREJECT (Machlovi mirror of the gated walledai original),
  JailbreakBench harmful behaviors, HarmBench behaviors (huihui-ai
  mirror of the official CSV) and a harmless-alpaca overrefusal control,
  cross-source jaccard dedupe (StrongREJECT recycles AdvBench/DAN
  prompts), authored row-group families, disjoint selection/holdout by
  offset; imported live once (409 selection / 421 holdout tasks,
  per-file sha256 provenance). The jailbreak "benign" CSVs in these
  corpora are duplicates of the harmful list, so the harmless control
  comes from a separate corpus.
- Adversarial robustness reporting: `family-decay --fitted SUITE SCORES --held
  SUITE SCORES` reports pass-rate decay on authored families the fitted run
  never saw and warns loudly when families leak between the two suites;
  report decay, don't hide it.
- Judge calibration growth: multi-model disagreement logs on every run,
  back into the authored calibration ledger.
- Claims ledger: every release directory carries a machine-readable
  evidence summary (gates run, failures, holdout results, judge
  calibration state). Consumers can diff claims between tools. This is
  the competitive differentiator nobody else will copy.

## Phase 5 — Architecture coverage (steady-state loop)

Not a phase, a rhythm: the July-October census model.

- Quarterly Hub census -> adapter worklist: `hub-census` queries the live
  Hub for new text-generation repositories from the tracked publishers,
  labels them by config.json model_type against the adapter table, and
  emits the worklist ordered by downloads (GGUF/MLX conversions counted
  separately as derived artifacts). The 2026-11-16 run surfaced lfm3,
  mimo_v2, dory, bolmo, ernie5_5_moe, llada2_moe, kimi_k3 and
  nemotron_labs_sigma as adapter candidates. The human-curated gap list
  (DeepSeek V4.1, custom Ling VL, draft-only Muse assistant, Qwen Drive
  planning models) ships in the same JSON; nothing here claims support.
- GGUF converter compatibility matrix per architecture: an export that
  llama.cpp cannot actually run is not a release. Smoke-load each
  exported quant on the B70 before the release entry says so. GATED.
- Hybrid recurrent blocks (already handled cache-free in tree_hf) and
  MTP/draft weight preservation (export streams the full tensor
  inventory) stay covered as upstream models drift.
- Vision/audio towers: out until demand; when in, ablate the LM side and
  report the tower interaction honestly.

## Phase 6 — Release and UX

- Preset profiles per model family (safe / standard / aggressive) with
  measured ranges only; presets without evidence are refused.
- One-command flow on top of `auto`: abliterate MODEL targeting
  CONSISTENCY POLICY, with the global-winner discipline already there.
- Release signing: Ed25519 (openssl) over a machine-readable reproduction
  index (file hashes, evaluation/readme digests, ledger digest);
  `release-keygen`/`release-sign`/`release-verify-signature`, with
  `--public` to pin authenticity to a known key.
- Job queue hardening: generation and judging already own separate GPU
  lifetimes; add priority and budget caps per job.

## Long bets

- SAE-based feature ablation: train or reuse small SAEs per family,
  remove refusal features directly. Biggest potential quality jump,
  biggest compute. Do the CODE parts (contracts, fit metadata, scoring
  hooks) only when the compute plan is real.
- Cross-model universal directions via activation transplant between
  same-arch checkpoints.
- RL-free preference editing: closed-loop direction training against the
  full-reply objective (training.py has the base).

## Explicit non-goals

Diffusion samplers, MLX/Apple, Windows, multi-GPU, remote uploads,
claiming certification of pretrained checkpoints the tool never actually
ran on. Marketing-version name guessing stays banned. Any published
result must survive the same consistency policy as local ones.
