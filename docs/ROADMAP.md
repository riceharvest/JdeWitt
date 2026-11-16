# Roadmap to the best refusal-removal tool

North star: the tool whose abliterated releases you can trust more than
anyone else's, because every claim is gated, falsified or labelled. The
moat is method breadth plus verifiable honesty; everyone else overclaims.

Work is tagged:
- CODE — CPU-verifiable, no model, no B70. Ship by running the command.
- GATED — needs real B70 / real checkpoints before any claim changes.

Status: all CODE items on this roadmap are landed and pushed. Highlights:
quant-native GGUF editing (GGML Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/F32/BF16 codecs,
tile-wise gguf-edit, gguf-inspect), method breadth (nullspace, answer-mean
rEF capture, family-weighted whitened-SVD), evidence chain (claims ledger,
ed25519 release signing with repro index, published artifact JSON Schemas),
agent ops (stable error codes, about/studies/study-metrics/run-cancel/
run-events, describe-run, --set/--preset, abliterate one-command flow,
events.jsonl, injectable clock), and offline evaluation (Wilson + bootstrap
+ paired permutation stats, score-compare, judge-ensemble, score-sensitivity,
robustness-report, suite-stats/suite-diff/suite-near-dup, corpus-cache
report/prune). Still GATED (needs real B70 / real checkpoints, by design):
method-winner claims, quant-edited model behaviour, judge calibration gains,
throughput claims, per-family preset profiles (the preset mechanism ships;
presets without evidence references are refused at load time).

No CI and no new test suites on this roadmap. Verification is: run it,
read the artifact, keep the honest claims table current. The existing CPU
suite stays as-is.

## Phase 0 — Operations surface (CODE)

Everything here makes the tool drivable by an agent at scale, which is
how breadth happens later.

- Stable machine-readable error codes for every expected failure
  (errors.py already separates them; give them codes, document the table).
- Published JSON schemas for run-status, methods, architectures,
  validate-* and import-suite output; the commands enforce them at emit
  time, so no test harness is needed.
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
- Plan library: plans.py is already portable, no full weights. Add a
  local registry of fitted plans with provenance + measured gates, so
  warm starts can transfer across checkpoints in the same family. A
  private "direction registry" no competitor has.
- GATED: which of these actually wins is decided by the searcher on real
  checkpoints, never by prose.

## Phase 3 — Objective and search economics

The objective is what makes results better, and judge cost is an
explicit unresolved limit in the README.

- Multi-objective search: compliance, full-reply KL, paired task
  regressions, refusal margin, answer length/damage. Optuna MO-TPE,
  Pareto front surfaced in run-status; the existing single minimize
  becomes one configured point on it.
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
- Adversarial robustness reporting: same release evaluated against
  paraphrase families it was not fitted on; report decay, don't hide it.
- Judge calibration growth: multi-model disagreement logs on every run,
  back into the authored calibration ledger.
- Claims ledger: every release directory carries a machine-readable
  evidence summary (gates run, failures, holdout results, judge
  calibration state). Consumers can diff claims between tools. This is
  the competitive differentiator nobody else will copy.

## Phase 5 — Architecture coverage (steady-state loop)

Not a phase, a rhythm: the July-October census model.

- Quarterly Hub census -> adapter worklist. Current explicit gaps:
  DeepSeek V4.1, Kimi K3 loaders, MiMo V2.6/Bolmo/Youtu wrappers,
  custom Ling VL, draft-only Muse assistant, Qwen Drive planning models.
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
