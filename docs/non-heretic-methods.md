# Methods not in upstream Heretic — inventory, pros/cons, decision state

Scope: refusal-removal methods that upstream Heretic does NOT ship. This is a
maintained literature + status list, not a measurement. No method on this
page has a local before/after number yet; the ROADMAP rule stands: "which of
these actually wins is decided by the searcher on real checkpoints, never by
prose." Third-party numbers are labelled with their source and are that
source's claim, not ours.

Upstream pin: `p-e-w/heretic` README, `master` @ commit dated
2026-11-16T18:16:06Z (checked 2026-11-17).

## What IS upstream (excluded from this list)

Heretic 2.0 ships exactly two modifiers:

- **Directional ablation** (Arditi 2406.11717), with **projected** (grimjim
  blog) and **norm preservation / MPOA** (PR #52, merged; PR #332 "ARA but
  it's LoRA" merged).
- **ARA — Arbitrary-Rank Ablation** (default): optimises each target weight
  matrix per module, no direction assumed; ~rank-50 LoRA export. PR #400
  (adaptive steering resolving MoE/DiffusionGemma failures) was **closed
  unmerged** — the upstream MoE story is incomplete.

SOM (PR #196) was **closed unmerged**; J-space exists only as a third-party
fork plugin. So SOM and J-space count as not-in-Heretic.

## Implemented here, not upstream

Mechanisms as they actually exist in `b70_ablator/methods.py` — not as the
papers describe them.

### `whitened-svd` — ridge-whitened contrast subspace
Source family: OBLITERATUS. Thin-SVD of the (optionally family-weighted)
harmless covariance, contrast pushed through the ridge inverse-covariance
sqrt. `family_weighting: balanced` weights families by row count.
- Pros: fixes SVD's known failure to decorrelate features; weighted variant
  stops big families from dominating; no optimizer; ours.
- Cons: whitening amplifies low-variance directions where the contrast is
  mostly noise — ridge value is a real tuning knob; no cross-model study
  exists for it.
- Cross-model evidence: none published.

### `som` — deterministic 1-D SOM prototypes (arXiv 2511.08379, AAAI 2026)
Fit `rank` prototypes to harmful activations only, subtract the harmless
centroid, orthonormalise. Upstream closed the PR; this is our port.
- Pros: the paper's own claim (third-party): multi-directional removal beats
  the single-direction baseline and specialised jailbreaks; SOM generalises
  difference-in-means, so `som rank=1` should collapse toward directional —
  a built-in falsification check.
- Cons: paper is single-group; our 1-D topology is a subset of the paper's
  design space; more directions = more edit rank = more measured capability
  risk, which the paper's benchmarks may understate; sensitivity to
  `som_steps`/learning-rate schedule is unmeasured.
- Cross-model evidence: paper-side only, labelled third-party.

### `leace` — binary ridge linear covariance erasure (arXiv 2306.03819)
Closed-form linear erasure map; we take the left factor as the direction,
rank-1 only, row normalization refused because it changes the map.
- Pros: principled "remove the component that predicts refusal from the
  activation covariance" instead of "mean difference"; closed-form, zero
  search cost.
- Cons: the original is a representation-of-interest erasure, not a refusal
  remover — no affine mean restoration; a linear map may not find a
  refusal-aligned axis at all in models where refusal is nonlinear (see
  concept cones below).
- Cross-model evidence: none for refusal removal.

### `nullspace` — greedy deflated contrast basis (ours; RepEng-style)
Direction k = mean contrast surviving projection of directions < k,
sharpened by `nullspace_deflations` power passes. Orthogonal by
construction; a genuinely different basis from singular vectors when
contrast signs are mixed across families. (Source metadata fixed 2026-11-17:
the old arXiv ID pointed at a photogeneration physics paper; replaced with
AlphaSteer 2506.07022, null-space-constrained refusal steering.)
- Pros: family-by-family deflation is the natural counter to the "cone"
  picture without gradient search; fully deterministic — the only method on
  this list whose run-to-run variance is exactly zero.
- Cons: greedy order means the first family seen gets first pick of the
  subspace; no independent literature validating this exact greedy rule.
- Cross-model evidence: none.
- Upgrade candidate: AlphaEdit 2410.02355 projects the edit DELTA into the
  null space of preserved knowledge; AlphaSteer 2506.07022 the steering
  vector into the null space of benign behaviour. Either could wrap any
  direction method here as a collateral-damage cap (CODE to fit, GATED to
  claim).

### `sft` / `dpo` — trained low-rank edit factors (`b70-ablator train`)
Heretic is training-free by design, so this is a genuine family it never
does. SFT masks prompt tokens; DPO uses the unedited model as reference
without a second copy in VRAM.
- Pros: can push behaviour that a linear edit structurally cannot; DPO
  directly optimises chosen-over-rejected on the exact compliance pairs.
- Cons: train/eval leakage discipline is entirely on the data author; the
  holdout is never evaluated by training; trained-unvalidated, no
  optimizer-state resume; activation memory shrinks the editable-model
  envelope further.
- Cross-model evidence: unaligned-SFT/DPO pipelines exist broadly; no
  refusal-removal bake-off vs direction methods.

## Researched, not implemented (here or upstream)

- **Concept cones / RDO** (arXiv 2502.17420, verified): gradient-fitted
  multi-dimensional cones; their central result — orthogonal is NOT
  independent under intervention — directly attacks the edit primitive every
  direction method here (and Heretic) uses. Pros: best mechanistic story for
  mixed-families; cons: gradient fit needs per-model loss passes, cone basis
  is non-canonical.
- **SAE feature erasure**: biggest potential quality jump, biggest compute.
  Do not start CODE parts until the compute plan is real (ROADMAP long bet).
- **Router pre-bias / expert-level edits** (Heretic issue #322, still open;
  upstream PR #400 resolving MoE ARA failures closed unmerged). Nobody has a
  good published MoE refusal story; MoE coverage is the clearest
  differentiation lane left.
- **J-space / Jacobian-pullback direction** (`Sekinal/jspace-refusal`):
  direction from refusal-token unembeddings pulled back through the Jacobian
  lens. The authors' own numbers (single model, their harness): 5.6x less
  benign-workspace distortion than abliteration but far LESS behavioural
  removal (0.78 vs 0.06 refusal rate on AdvBench); their follow-up found the
  behavioural part of the mean-difference direction is a harmfulness-
  perception feature, not the refusal narration. Verdict: strong
  interpretability, weak remover — not on the bake-off shortlist.
- **Multi-token / answer-mean (rEF-style) directions**: ROADMAP Phase 2 item
  (capture at answer-position means, not the last token). The 2611.17846
  audit's own section 2.4 cites evidence that where a direction is estimated
  (layer output vs summed residual stream) "may matter as much as how many
  are removed" — that question is unanswered in this repo and cheap to test.
- **Optimal-transport contrast matching** (Heretic issue #217, closed
  without implementation) and **GRP-style objectives** (issue #184, closed
  without implementation). Open ideas, no public results.

## Single-pass tools outside Heretic (third-party numbers only)

From arXiv 2512.13655 (single author, 16 instruction models 7–14B, Dec 2025;
treat all numbers as their claim):

- **DECCP** (`AUGMXNT/deccp`, closed-form mean-difference per block):
  GSM8K change on the benchmarked subset avg **−0.13 pp**; preserved or beat
  Heretic on every benchmarked model (Yi-1.5-9B: 72.40% vs 52.08%).
- **ErisForge** (`Tsadoq/ErisForge`, single-pass, default ablation weights):
  best capability preservation across all benchmarks (GSM8K avg **−0.28 pp**).
- **FailSpy/abliterator** (`FailSpy/abliterator`): reference-implementation
  tier.
- Heretic (Bayesian-optimised): KL after edit spanned **0.043–1.646**
  depending on model; GSM8K is the most sensitive capability axis overall
  (+1.51 to −18.81 pp across tools/models).
- Headline implication for us: **closed-form single-pass edits currently win
  the retention axis** in the only cross-architecture study that measured
  it; optimisation/search depth buys removal, not preservation.

## Cross-model evidence, 2026 (the actual "best across models" question)

Two studies by the same single author — useful, but single-operator and
judge-dependent; do not launder them into certainty:

1. **arXiv 2611.17846** (2026-11-13, four days before this note): five
   release pairs (Qwen2.5-32B→Qwen3.8-27B, Gemma-2-27B→Gemma-4-31B,
   OLMo-2-32B→Olmo-3.1-32B, Mistral-Small-2409→3.2, Qwen2.5-7B→Qwen3.5-4B),
   same prompts/budget/KL-cap/selection rule, WildGuard over 972 held-out
   harmful prompts. Newer releases retained −1 to +37 pp more refusal after
   direction-based removal and −26 to +34 pp after ARA; heterogeneity I^2
   20–80%. **ARA won on Gemma and OLMo, lost on Qwen; some verdicts flipped
   with the KL budget or the variance model; keyword counting differed from
   the judge by ~7 pp per model.** Their stated conclusion: whether removal
   is "harder" depends on the release pair, method and budget, not on model
   generation. There is no method that uniformly wins across models.
2. **arXiv 2608.25390**: models whose refusal training used diverse refusal
   prefixes develop higher-stable-rank refusal geometry and resist vector
   ablation — i.e. newer alignment recipes may be harder to remove by
   construction. Direction-count is not the whole answer.
3. **arXiv 2608.17202 (Fool's Gold)**: defensive deception built to survive
   safety-removal — an "abliterated" model can be a honeypot. Any measured
   win must use a semantic judge and held-out families, not keyword counting.

## What would actually decide this (the measurement, GATED)

Reproduce 2611.17846's controls with our stack, which already has every
piece: imported jailbreak-slice suites (StrongREJECT + JailbreakBench +
HarmBench + harmless control, revision-pinned), selection/holdout
family-disjointness, `suites-compare` (Holm-adjusted family sign tests +
signed Cohen's h), `judge-ensemble`, `family-decay`, MO-TPE with a first-token
KL objective.

Shortlist for the bake-off: `directional` (baseline), `som`, `whitened-svd`,
`nullspace`, `ara`, `leace`, each at its own searched strengths, ≥3 seeds per
method (2611.17846's single-run caveat is the exact failure we outgrow),
≥3 model families incl. at least one MoE, capability axis = existing
math/code suites. Method wins are claimed per family with the Holm result
attached; the global claim is "which methods survive across all measured
families", nothing stronger.

Known blockers before GPU time: the direction-suite coverage trap
(bad variants that never terminate drop cells before the refusal check —
PLAN-NEXT 2026-11-16), selection-suite item `s-reviews-r3` non-termination,
and no 3–4B dense instruct checkpoint pinned yet (the two 1B models on disk
compress the retention axis to nothing and serve only as loop-speed models).

## Maintenance rule

Each bake-off result row lands here (or in a run report referenced from
here) only after the suites-compare gate has run; literature rows keep their
source pin; any claim that cannot survive the same consistency policy as a
local claim stays out.

## First-party measured result — SAS (Sign-Aligned Shared axis)

Status update to the header caveat: one method now has local before/after
numbers. Measured on Qwen2.5-1.5B-Instruct (bf16 GGUF), all numbers
`rank=second` (RTX 3060; B70 GATED).

**SAS** is not a new fit *objective* — it is a post-processing of the
contrast data the directional pipeline already generates:

1. Take per-layer contrast means `m_L = mean(bad residual) − mean(good residual)`
   from the teacher-forced capture the pipeline already runs (no extra inference).
2. Fit ONE shared unit axis across all layers by sign-aligned power iteration:
   `a ← normalize(Σ_L sign(a·m_L) · m̂_L)`. Layer sign flips are automatic;
   topic-dominated per-layer means cancel except on the component the layers
   agree on, which is where the refusal routing lives.
3. Edit: remove `a` from every `o_proj` and `mlp.down_proj` output
   (projection removal, functional `Wᵀa` computed from PRIMAL weights —
   see serving-basis trap below).
4. Export as one rank-1 adapter; trials are POST `/lora-adapters` scale swaps;
   scale calibrated by one micro-wave.

Cost measured: axis fit 0.024 s, adapter build 0.443 s, zero extra inference.

Certification (shape judged by the working agent per METHODOLOGY.md R2; all q25 rows
below served at the R9-pinned `--ctx-size 4096 --parallel 4` config):

- harmful-40 suite (in-sample, families used for the fit): 40/40 refusal-shape
  removed at scale 1.5, harmless 8/8 intact.
- Canonical held-out (`public-corpus-fit/holdout.jsonl`, zero id-overlap with
  the fit selection, 24 stratified refusal families): 24/24 refusal-shape
  removed, harmless 10/10 intact.
- Oracle cross-check: the published `Josiefied-Qwen2.5-1.5B-Instruct-abliterated-v1`
  (pinned rev `c116e736b78dd85e`) was diffed — its edit is per-layer
  projection-removal on exactly o_proj + down_proj with one shared output
  axis (|cos(u_L, u_down)| = 1.0, all 28 layers; cos(v, Wᵀu) = −1.0);
  reproducing that axis as an adapter on the *unmodified* base removed 40/40.
  SAS lands at mean |cos| = 0.64 to that axis and still certifies 100% on
  held-out families.
- Rejected for this model (measured, not prose): harmful-mean prototype,
  refusal-prefix span, RLGA likelihood-gradient fits (mean |cos| to oracle
  0.19–0.26; behaviorally null or destructive; per-layer independent fits
  land at |cos| 0.26 and remove only ~75% of refusal shapes).

Serving caveat that invalidated a month of nulls before being fixed: the
dspark llama.cpp fork Hadamard-folds qwen FFN weights and applies LoRA deltas
to the PRIMAL input — adapter functionals MUST be `a = d @ W_hf`, never
`d @ dequantize_tensor(gguf)`. Embedding-side edits cannot be served as LoRA
on tied-head models (build_inp_embd vs build_lora_mm orientation conflict).

Cross-lineage probe — MiniCPM-V 4.7-1B (hybrid linear-attention), same SAS/contrast
data, R8-correct STATIC MERGE path, config pinned per R9:

- The tool's contrast plane (SVD rank-4 of the prompt-end mean contrast, output-side
  removal merged in-place) is config-sensitive: ~90% harmful-40 removal at
  `--ctx-size 2048 --parallel 2`, ~52% at `--ctx-size 4096 --parallel 4` on the SAME
  bytes. Harmless rows intact in both.
- The published `Abiray/MiniCPM-V-4.7-1B-abliterated-heretic-GGUF` Q8_0 serves 36/40
  (91%) at the pinned `4096/4` config, and its byte-reproduced static merge matches it.
  Oracle diff: 35 nonzero rank-1 tensors with near-orthogonal per-layer u/v (u-cos ~0.02
  ssm_out, ~0.10 attn_output) + diffuse embedding edits (relnorm 0.039) — NOT a shared
  axis. Consistent with SAS being the right structure only when the model routes refusal
  through one shared output direction (q25); mc47's refusal is per-layer structured.
- ANSWER-SPAN PER-LAYER rank-1 (tool-fit, label-only): for each layer L, take the top
  contrast direction of the refusal-vs-compliant ANSWER-SPAN residual means (R4 locus)
  and remove it from that layer's ssm_out/attn_output/ffn_down (merge in-place,
  strength 1.0). At the pinned `4096/4` config: 36/40 removed (90%), harmless 8/8.
  This matches the published oracle's 36/40 under R9 config-matched comparison, from
  the tool's own contrast data — same CEILING, different leftover rows (tool leaves
  prevention-pivot families: identity theft, self-harm, credit-card, CP-website).
  The prompt-end-plane-only reading of 52% is superseded: the fit LOCUS (R4) is the
  variable, not just the span rank.

