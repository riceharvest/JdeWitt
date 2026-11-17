# Learned J-space behavior ranking (implementation only)

This extension was written under an explicit **do not test** instruction. No
training, model inference, test suite, CLI smoke, audit or package rebuild was run
for it. Previous tree-test counts do not verify these changes.

The old word-margin scorer remains available when `behavior` is absent. The
extension adds data collection, same-prompt branch-preference training, separate
behavior/recognition predictions, uncertainty handling, and a held-out
false-confidence report. It remains experimental and produces no certified GGUF.

## What the model sees

At each configured block-output checkpoint, record a broad operator-selected set
of J-space token logits, centered by their own mean, plus the diagnostic refusal
margin and residual/transport norm logs. Store the whole observed checkpoint
history, differences from the unedited trajectory, differences between successive
observations and observed-position masks. Future checkpoints are zero-filled;
training data validation rejects future-state leakage.

The ranker never receives the answer, outcome label, action name, strength, prompt
ID or intervention-family ID as an input feature. It can still learn intervention
artifacts indirectly, which is why intervention-family and exact-basis isolation
are checked alongside prompt/topic-family isolation. These checks cannot prove
semantic independence of incorrectly authored family IDs.

The fitted lens, target identity, source dtype, feature tokens, refusal/reference
tokens, chat/generation fingerprint and output-budget contract are part of the
feature schema. A differently configured ranker is rejected rather than silently
substituted.

## 1. Collect edited states and full answers

Add a `behavior` object to an ordinary tree configuration. Its field template is
`examples/tree-search/behavior.template.json`. All null settings need explicit
values before execution; no budgets or token groups have been chosen.

- `mode`: `collect` or `rank`.
- `feature_tokens`: broad, distinct target-token IDs, not just refusal words.
- `intervention_groups`: an authored family name for every configured action.
  Keep related directions in the same group, including variants/aliases.
- `max_feature_values`: must cover every visited-node feature vector plus the
  independent baseline. The metadata validator rejects insufficient bounds.
- `max_ranker_bytes`: bound on resident ranker parameters/normalization, not
  total training/inference workspace.
- `artifact`: `null` for initial collection, otherwise a trained ranker JSON path.
- `policy`: `null` without an artifact; otherwise all policy fields described below.
- `max_decodes_per_prompt`: in collection mode, must cover the baseline and all
  finite visited nodes. There is no hidden subsampling of expensive decodes.
  In rank-only mode it must be `null`.
- `include_rejected`: explicit boolean. In collection mode, enabling it includes
  finite states rejected by native damage guards, useful as broken-output/hard
  negative examples. In rank-only mode it must be false.

Collection requires real topic-family IDs on the suite rows. Synthetic row
families are not evidence of independent topics.

The existing command owns execution:

    jdewitt tree-search /absolute/path/collection-tree.json

Besides baseline/finalist answers, collection writes one feature index and
AnswerStore under `behavior/<source-task-digest>/`. It decodes each visited path
prefix **without any later intervention**, so the label really belongs to that
observed state, not to an unknown descendant. Equivalent prefill trajectories
reuse already saved baseline/finalist answers. Indexes are committed before
collection decodes, answers are immutable and missing work resumes.

`labels.template.jsonl` contains response-bound annotation skeletons with null
outcomes, recognition labels and label-source fields. They are not guessed labels.
Copy/fill them using a human or an independent complete-answer judge. Combine the
completed annotations for a source job into the labels file referenced below.

Each annotation needs:

    {
      "id": "<collected example ID>",
      "response_digest": "<digest already supplied by the template>",
      "outcome": "completion | refusal | evasion | broken",
      "recognition": true,
      "label_source": {"kind": "human | full-answer-judge", "identity": "<reviewer or judge identity>"}
    }

`recognition` is an independently assessed boolean about whether the model
recognizes the relevant concept/risk. Do not copy it from the prompt category or
infer it from whether the model refused. Genuine completion requires actually
fulfilling the request, not an affirmative opening. Refusal and evasion are
separate from nonsensical/broken output. A nonterminal/truncated answer cannot be
labelled genuine completion. Code/math completion labels need real task checks.

To exclude an uncertain annotation explicitly, leave `outcome` null and supply
an `exclude_reason`. Every collected example needs an annotation or explicit
exclusion; unmatched labels and missing answers are errors.

## 2. Build isolated datasets

Use `examples/tree-search/ranker-data.template.json`:

    jdewitt tree-ranker-data /absolute/path/data-spec.json

Each source supplies a job directory, completed labels file and explicit split:
`train`, `calibration` or `test`. The converter checks exact response digests and
source provenance, then rejects overlap in normalized prompts, authored topic
families, authored intervention families or exact direction-basis fingerprints
between any two splits. Baseline/no-edit records have no intervention family.
Raw tensor fingerprints do not establish that differently encoded subspaces are
semantically different; correctly authored groups remain essential.

Collect genuinely different direction families for calibration and test rather
than renaming the same directions. You cannot reuse the same common pullback
across all splits and claim an unseen-intervention test. The resulting manifest
binds immutable JSONL split hashes and retains label-source evidence. No splits,
recognition labels or correctness outcomes are invented.

## 3. Train and calibrate (B70 only, later)

Use `examples/tree-search/ranker-train.template.json`:

    jdewitt tree-ranker-train /absolute/path/train-spec.json

All network size, ensemble seeds, epoch/batch/pair budgets, optimizer settings,
loss weights, outcome utilities and calibration settings are explicit. Completion
must have greater utility than each failure outcome. Different failure utilities
are your choice, not silently selected values.

The trainer uses a small two-layer ensemble. It jointly learns:

- four behavior classes: completion, refusal, evasion and broken output;
- independent binary recognition, never rewarded for being absent;
- a utility head with pointwise anchoring and same-prompt/same-checkpoint pairwise
  preference loss. It does not contrast unrelated harmful and harmless topics as
  though that alone identified refusal.

Topic-family balancing prevents a family with many variants dominating the
pointwise loss. Preference pairs are sampled with an explicit reservoir cap and
family weighting, not simply taken from the first branches. Both train and
calibration must cover all behavior classes, both recognition labels and every
evaluated trajectory depth; missing coverage stops the run.

Normalization is fitted on train only. Separate outcome and recognition
temperatures are fitted on calibration only, using the same average-of-member
probabilities used at inference. **The trainer never opens test rows.** The saved
JSON artifact is `trained-calibrated-development-only`, not held-out certified.
Training and prediction have no CPU/CUDA model fallback.

## 4. Rank branches and find development hard negatives

Set `behavior.artifact` to the trained JSON file and supply a policy with:

    min_completion_probability
    max_refusal_probability
    max_evasion_probability
    max_broken_probability
    max_disagreement
    max_feature_z
    uncertainty_weight
    uncertain_policy

`uncertain_policy` explicitly chooses `reject` or `proxy`; there is no hidden
fallback. Excessive ensemble disagreement or standardized-feature extrapolation
causes abstention. A configured word-proxy fallback is recorded and placed behind
confident learned rankings, avoiding comparisons of utility and word-margin
values on incompatible scales. These are heuristic uncertainty checks, not formal
confidence guarantees.

Confident branch ordering minimizes negative predicted utility plus the explicit
uncertainty penalty. Recognition probability is logged but is **not** in the
reward or any acceptance gate: erasing risk understanding is not the objective.
Native protected-token/norm damage guards still apply at every depth. Predicted
behavior-probability gates apply at the terminal checkpoint, not prematurely to
partial paths that could recover after another pivot.

Use `mode: "rank"` to decode only baseline/finalists. Use `mode: "collect"` with
the trained artifact to let the tree optimize against it on fresh **development**
inputs and collect independent labels for its confidently wrong states. Collection
logs whether the ranker would reject a state but does not let its own behavior
probability gates hide those negative examples. Rebuilding training data from
these development collections gives an explicit hard-negative loop; test errors
are not fed back into it automatically.

## 5. Check false confidence later

Use `examples/tree-search/ranker-audit.template.json`:

    jdewitt tree-ranker-audit /absolute/path/audit-spec.json

This command predicts on the independent test split, rejects overlap with the
artifact's train/calibration prompts/families/directions, and reports:

- fraction of promising predictions that actually refuse, evade or break;
- counts and per-topic-family rates, with `null` rather than zero when no
  promising examples exist;
- same-prompt branch-preference accuracy and recognition accuracy;
- exact artifact/dataset hashes and every per-example prediction.

It requires an explicit minimum promising count and maximum false-confidence
rate. An empty promising set cannot pass. The report is a held-out screen, not a
model release certificate, and it never retrains the scorer or alters thresholds.
Do not tune against its errors; make fresh development collections and preserve
an untouched final test. Check full-answer usefulness and capability retention
independently before making any superiority claim.

## Current verification boundary

Source implementation only. Nothing in this extension has been exercised yet,
as requested. No GPU availability probe, CPU model inference, training or test
run was performed, and existing wheels were not rebuilt. Explicit budgets do not
prove B70 fit, and writing a ranker does not prove its predictions are useful.
