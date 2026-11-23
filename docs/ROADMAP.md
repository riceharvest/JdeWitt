# Roadmap to the best refusal-removal tool

v2, 2026-11-22. Supersedes the 2026-11-21 phases for ordering; the CODE/GATED
discipline and the ordering rule are unchanged.

North star: the tool whose removal actually lands in the artifact you serve, and
whose every number survives someone else re-running it. Not the tool with the
most methods, and not the one with the most downloads — the one whose release
you can check.

The claim that decides whether we are the best: **equal removal at less
capability loss, measured on the same artifact class at one pinned serve
config**. Today that claim is not established, and the gap is one row wide
(maths 159 -> 148 for us against 159 -> 159 for the published same-base build,
on a held-out 120/120 against its 120/120 with zero discordant rows). Everything
below is ordered by how directly it closes that row or makes the row cheaper to
measure.

Ordering rule (docs/AGENT-FIRST.md): an agent working on a model it did not
choose must be able to drive every step with one command and one JSON, on an
architecture this repo has never seen, editing the artifact in place.

Work is tagged:
- CODE — CPU-verifiable, no model, no B70. Ship by running the command.
- GATED — needs real weights / the B70 before any claim changes.

The resource is generated tokens, never throughput. Speed numbers do not enter
a cost argument. THE DELIVERABLE IS AN EDIT ON THE ARTIFACT THE USER ALREADY
SERVES: their quant, their calibration, no requant of the model, no download
from us.

Sequencing lives on the project's work board: each phase heading below names the
goal card that owns it, with one spine card for ordering and a standing audit card
that keeps README section IV and `verification/evidence-map.json`
true as phases land. Every card any of those phases creates carries the same three
rules: it closes the deciding row or makes it cheaper to measure; it is drivable
with one command and one JSON, next command included; and it edits the artifact
the user already serves. The order below is by closeness to the row, and the text
names the points where one phase needs another's result.

## Where the field actually is (fetched 2026-11-22, re-fetch before quoting)

Every line about a competitor is their claim until we measure it. Do not copy
their comparison tables, including ours.

Heretic 2.1.0 (2026-11-08; master pushed 2026-11-19). Ships two modifiers:
arbitrary-rank ablation (ARA, the default) and directional ablation with
projected + norm-preserving (MPOA) variants under a bell-curve weight kernel
over the layers. Plugin APIs for both modifiers and scorers, so any behaviour
pair and any number of co-optimised scores; shipped configs for no-slop,
no-humor, no-sycophancy, emotions, terse. `bnb_4bit` loading for models that do
not fit at 16 bit. CoT-block skipping, auto batch sizing. And the thing we still
do not answer: `--reproduce` (byte-exact reproduction manifests), the Grimoire
library of manifests, thousands of published models, and third-party MMLU/GSM8K
comparisons it cites on its own README. Its search is not prefill-only; on the
order of a hundred full evaluations per run is their cost shape.

OBLITERATUS. 15 analysis modules plus an `informed` pipeline that configures the
obliteration from them: whitened SVD, bias-term projection, iterative re-probe,
expert-granular ablation for MoE, CoT-aware orthogonalisation, COSMIC layer
selection, parametric kernel optimisation, gradient-based refusal-direction
optimisation (RDO), KL co-optimisation that restores the layers that blew the
budget, component-specific scaling, LoRA-reversible ablation. Distribution is a
Gradio Space + Colab, and telemetry is harvested into a crowd dataset. Its own
docs state the artefact-level truth: peak VRAM is the BF16 size of the model and
the output is saved as BF16, with re-quantization left to the user
(`llm-compressor`/`modelopt`). That is the gap the whole jdewitt thesis sits on.

abliterator. TransformerLens notebooks: cache at many hook sites, test a
direction through a temporary hook before touching weights, blacklist layers,
MSE-harmless loss. Its real contribution is the selector — score a candidate
axis by token-logit margin (`Sure` vs `cannot`) on prefill, zero generation.

What none of them have, and we do: editing the Quantized artifact in place,
per-tensor minimum surgery, every other tensor byte-identical, exact revert,
and a codec layer proved byte-exact against the compiled C for all 35 GGML
types. They all edit 16-bit and hand the user a requantization problem they
then measure nothing about.

## Where we lead (measured, checkable in this clone)

- Quant-native editing: `edit-quantized` applies a fitted axis to ANY quantized
  GGUF, re-encoding only the edited tensors; `gguf-revert` restores exact bytes;
  `preview_edit` reports cos(realized, intended) per tensor in ~7 s, zero tokens.
- Byte-exact codecs for every GGML type against the pinned C (`scripts/ggml_ref/`),
  including the IQ family and Q2_0, which published gguf cannot even open.
- Token economy: prefill-only ppl ladder ranks arms at ZERO generated tokens;
  `tune` finishes an arm choice in ~1,940/2,000 tokens and 70 s on a 1.7B;
  row-halving (eliminate on one row, confirm the survivor) took the same
  decision from 1,940 to 1,280 tokens; a cascade screen settles rows inside a
  K-token window and pays full decode only for escalations.
- Certification by shape, not wording: the word check is screening; removal is
  a content verdict from an agent reading an evidence packet. The retraction in
  the README exists because we ran that instrument on our own best arm.
- Honesty gates nobody else has: `/props` identity before a wave, adapter-vs-
  merged ratio with an explicit validity flag, drift gate calibrated to the
  instrument's noise, evidence map that fails a claim with no file behind it,
  chained claims ledger, ed25519 release signing.
- Discovery: capture finds the decoder stack and the residual-write projections
  from the module tree; `spanfit targets` reads the artifact's real suffixes.
  No tensor tables, no hand-typed names.

## Where we are short (measured, or provable by inspection)

1. The codec floor. On Spark 1.7B Q4_K_M a rank-1 removal at strength 1.0
   realizes cos 0.454 on attention and 0.938 on FFN; SNR ~1, i.e. requant noise
   as large as the edit. Today `--min-edit-cos` refuses those arms or bisects
   the scale upward (which landed at 6.61 and +10.2 nats of control PPL — a
   collapsed model). A grid limit is being reported as an axis verdict.
2. The cheap loop has the kernel but not the component split. `auto` searches
   per-group (attn/mlp) strength plus center/width/floor; `tune` has footprints,
   shapes, rank and a flat/bell/manual profile, all at zero tokens, but one
   global scale across every target, and it classifies targets by kind without
   ever searching a separate strength per kind. Heretic co-optimises per
   component; that is where its retention advantage is plausibly earned.
3. Ranking happens on adapter curves. Measured on Spark 1.7B: identical
   coverage, delta_br 0.215 on the adapter against 0.119 on merged bytes
   (ratio 0.553), and an arm the adapter screen cleared refused on the merged
   artifact's first row. The merged screen exists but costs ~9 min per arm.
4. Sizeable method gaps: RDO, SAE feature ablation and router/pre-bias
   interventions are researched-not-implemented; bias-term projection is absent
   entirely. Layer selection is a searched bell or a measured profile, never a
   causality score.
5. The behaviour axis is unestablished for the whole class. On a hand read of
   the same 24 held-out families our arm delivers the asked-for content on 5/24
   and the published build on 4/24; the unedited base itself reaches 1/24; on the
   first 20 families of the wider screen the arm sits at 0/20 and the published build
   at 1/20; sampling does not unlock it. Rank-1 direction removal moves the sentence
   that says no, not the content.
6. Evidence volume. Thousands of third-party-benchmarked Heretic builds against
   a handful of our certified artifacts. Our instruments are better; our count
   is not, and count is what a stranger uses to decide.

## The gates every phase must pass

One arm = one edited artifact, both axes, one pinned serve config each: removal
on harmful rows under both inventories plus the shape audit, retention on the
241-row authored suite with paired stats and median answer length, realized
fidelity from `preview_edit`/`--min-edit-cos`, drift against the instrument's
noise. A probe-only pass is not a pass. An all-invalid screen is UNMEASURED.
Any arm that clears by narrowing the suite, raising the scale past the grid or
grading at a smaller cap is a bug in the measurement, not a result.

## Phase 1 — Make the edit survive the codec (CODE, highest payoff)

The root cause of both the R8 disagreement and the false refusal on merged
bytes. Three parts, in this order:

- Per-tensor type choice by measured realized cos, not by hope: the arm
  identity carries (tensor, intended delta, chosen type, cos, byte cost). `auto`
  already walks q6_k -> q8_0 -> f16; it must be driven by the measured cos and
  the byte budget, and a tensor that cannot hold the edit is reported, never
  silently accepted at a raised scale.
- Codec-aware delta. Solve for the delta the grid can store: iterate
  `d <- d + proj(target - realized)` two or three times per tensor (CPU, the
  requantizer we already have), and floor it at the unedited value when it
  diverges. This is a capability no competitor has, because no competitor edits
  a quantized file.
- Stop conflating grid and axis. A low realized cos fails the arm as a CODEC
  event with its own label; it never becomes evidence that the fitted direction
  is wrong.

Done gate (GATED, Spark 1.7B Q4_K_M): every edited tensor at >= 0.90 realized
cos at the strength that clears a one-row probe, control drift inside noise,
untouched bytes identical, revert exact, and one recorded case where the
codec-aware delta beats plain scale-up on removal at equal-or-lower drift.

## Phase 2 — Merged bytes as the decision stage, at ladder cost (CODE)

The adapter curve ranks; the merged artifact ships. Make the shipping verdict
cheap instead of rare:

- Fold the two dequant passes (deltas in `build_merged`, application in the
  in-place writer) into one.
- Content-address the dequantized bases per artifact (sha256, like the ladder
  cache) so the second and third arm pay apply+requant only.
- `tune` ends on merged bytes for the chosen arm by default; the top-2 merges
  when the adapter ranking is marginal (`adapter_to_merged_ratio` near the
  validity threshold).
- Report `ranking_valid_on_this_artifact` beside every ranked table, and never
  quote an adapter delta_br as the edit's effect.

Done gate: four merged arms screened in under 3 minutes on a 1.1 GB Q4_K_M
artifact, zero generated tokens, exact revert verified, and one documented
disagreement where the merged verdict overturned the adapter ranking.

## Phase 3 — Put the kernel and the rank in the cheap loop (CODE)

Phase 2 makes merged ranking affordable; this makes the space worth ranking.

- Arm space becomes footprint x shape x rank (1..k) x per-kind scale (attn vs
  mlp) x per-kind profile (flat | bell). `tune` already classifies targets by
  kind, so per-kind scale is one more adapter file and one more entry in the
  scale vector: zero generated tokens, and it is the exact dimension Heretic
  co-optimises. Only the Pareto front reaches the screen.
- Profiles and norms are compared, never assumed: a bell scored better than flat
  in a single run means nothing until a paired, mean-matched comparison at equal
  clearance decides it, and the loser is reported as the loser. Same rule for
  `--renormalize` and for rank-k against rank-1.
- The profile REPLACES the layer profile, never multiplies it (two compound
  shapes are not a measurement). Reject profiles on methods that carry their own
  covariance or learned factors (LEACE, ARA) at validation time.
- Norm preservation gets its own derivation per side: row-wise cancels on the
  read side (W(I - sD'D)), full normalization rescales final rounded rows and
  must reject collapsed nonzero rows. Prove both against closed forms on
  rectangular/GQA weights before any model run.
- Rank-k exists in the fitter (svd/whitened/nullspace/SOM) but the cheap loop
  scores rank 1; wire per-direction adapters and per-direction strengths, and
  let Phase 1 decide which tensors can store them.
- Bias-term projection where the layout has biases (OBLITERATUS's claim, cheap
  to test, free on the ladder to evaluate).

Done gate (GATED, q3-4B): an arm with removal equal to the published build
(McNemar not significant on 40 in-sample and 120 held-out families) and maths
retention at or above 159/241 at the same pinned config. That row is the
objective; a loss must be reported as a loss.

## Phase 4 — Zero-generation selection (CODE, then GATED verification)

Stop choosing axes and layers by contrast mean alone.

- Token-logit margin selector (Sure vs cannot, prefill only) scores a candidate
  axis and a layer footprint before any weight is touched. It targets causality
  instead of contrast geometry, and it costs nothing to add to `capture`.
- RDO: refine the fitted direction against a linear refusal probe by gradient
  descent on the same captured tensors. `methods.py` already lists it as
  researched; implementation is CPU-side.
- A separability layer score (harmful vs harmless cosine per layer, COSMIC-
  style) computed from the capture records we already store, so layer footprints
  stop being searched blind.
- Every selector is arbitrated by the ppl ladder and then by merged bytes; a
  selector that loses the ladder is dropped, not kept for elegance.

Done gate: one fitted arm whose drift at equal removal is lower than the
contrast-mean arm on the same lineage, selected reproducibly from saved records
with no extra generation.

## Phase 5 — The delivery axis (GATED; the only thing that would make this class
of tool new)

Measured starting point: 0/24 delivered (ours), 4/24 hand-read (published
build), 1/24 (unedited base), sampling does not unlock it. Tone is solved;
content is not. Ordered by yield per token:

- Fit data, not decode policy. Jailbreak-pair contrasts where the difference IS
  delivery of the same request; pinned public response mirrors; GCG-style
  adversarial suffixes. Every compliant side is agent-read before it enters a
  fit; word and shape screens alone false-pass seeded refusals.
- SAE feature ablation: train or reuse small per-family SAEs, remove refusal
  features directly. Biggest quality jump, biggest compute; do the CODE
  contracts only when a compute plan exists.
- Trained factors: `train` already produces SFT/DPO edit factors and labels
  them unvalidated. Validate against the same two axes before anything ships.

Done gate: content-rate delivery on the hand-read 24-family set at >= 3x the
base's rate, with retention inside the same gate as every other arm. Below that
it is a tone result and is reported as one.

## Phase 6 — Delivery formats that do not fight the codec (CODE, GATED for claims)

- Adapter release as a first-class artifact: rank-1 files apply exactly on any
  quantized runtime, with the honest label that the adapter curve is not the
  merged bytes (R8), plus the codec-realized numbers for the merged twin.
- Plan cards: `plan-bundle` already produces a digest-pinned signed plan and
  `plan-apply --bundle CARD --gguf THEIR.gguf --out EDITED` consumes it: the card
  is verified into a throwaway registry, the target tensors are discovered in the
  artifact itself, only those are re-encoded, untouched tensors stay byte-identical
  and the delivered codec cos is printed. That makes the card the unit of
  distribution — a stranger reproduces the edit on THEIR quant in one command,
  without our search — and `release-diff` tells them whether our claims still hold.
- Non-GGUF export for vLLM/SGLang users stays open until a fitted plan can be
  served and round-tripped through that runtime; until then it is a roadmap
  line, not a promise. No pre-quantized model releases.

## Phase 7 — Evidence volume and reproducibility (CODE + GATED runs)

Heretic's real moat is that thousands of builds exist and a manifest recreates
any of them. Ours must be fewer artifacts with claims nobody can argue with:

- Certified release matrix across at least five lineages at pinned configs, each
  with removal under both inventories, the shape audit, retention with paired
  stats and length, held-out families, drift, and realized fidelity.
- A portable reproduction index per release (plan bundle + suite digests + serve
  argv + codec choices) that reproduces the artifact without re-running the
  search, and a check that fails when any operand moved.
- `release-diff` and `verify-artifact` aimed at third parties: anyone can diff
  our claims against another tool's release directory by digest. The reading
  half ships as `verify-artifact DIR --foreign`: it hashes the regular files,
  reads GGUF headers (architecture, tensor count, quant type per tensor family,
  tokenizer keys) without loading weights, reports `certified_claims: 0`, and
  names each claim that cannot be checked there together with the file that
  would be needed (`files.json`, `evidence.json`, `signature.json`, a
  codec-realized record, a serve argv). A directory that already carries
  `files.json` is refused as ambiguous and goes through the native path.
- The honest table published including the losses. A roadmap that only lists
  wins is marketing, and the README already treats marketing as a bug.

Done gate: five lineages certified, one third-party reproduction of a plan card
on different hardware, every README number resolving through `evidence_map`.

## Phase 8 — Steady-state rhythm (not a phase)

- Quarterly `hub-census` -> adapter worklist, ordered by downloads; the human
  gap list stays in the same JSON. The dated artifact and its triage table live
  in `verification/hub-census/` and `docs/HUB-CENSUS.md`.
- Per-architecture export smoke: a GGUF llama.cpp cannot serve is not a release.
- `capture`/`tune` keep working on architectures never seen: discovery, not
  tables, and a typed refusal (`NO_VERIFIED_ADAPTER`) when a layout has no
  verified adapter.
- Agent ops stay one-command/one-JSON/next-command; the CLI self-describes and
  every command stays model-free on its help path.

## Long bets (unchanged, unordered)

Cross-model universal directions via activation transplant between same-arch
checkpoints. Closed-loop preference editing against the full-reply objective.
Learned arbitrary-rank ablation where the rank is a searched matrix, not a
truncated SVD.

## What would prove this roadmap wrong

If per-component profiles, codec-aware fitting and norm preservation still lose
the retention row at equal removal, then the ceiling belongs to the static-
projection method class and not to our implementation — and the next roadmap
starts at learned factors (SAE, training), not at more search. Say that out
loud before spending the next thousand GPU minutes on tuning.

## Explicit non-goals

Diffusion samplers, MLX/Apple, Windows, multi-GPU, remote uploads of models,
cloud/Space hosting, telemetry or crowd-harvested datasets, pre-quantized model
releases, claiming certification of checkpoints the tool never ran on,
marketing-version name guessing, and quoting a number whose artifact is not in
the clone. No CI. No new test suites.
