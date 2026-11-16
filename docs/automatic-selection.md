# Per-model automatic method selection

Commands:

```sh
b70-ablator validate-auto /absolute/path/auto.json
b70-ablator auto /absolute/path/auto.json
```

The automatic workflow compares configured methods and ranks on the same model,
selection suite, scoring policy, generation settings and matched Q4/Q5 baselines.
It chooses the best measured candidate within the configured search space, not a
universal method or a proven global optimum.

## Configuration

The JSON object requires exactly `run`, `total_trials` and `candidates`.

- `run`: a complete ordinary RunConfig, including explicit consistency and damage
  limits. `run.output` is the parent experiment directory. For this first automatic
  workflow, the model must be a reviewed local original-safetensors directory.
- `total_trials`: an explicit positive integer upper bound on all layer-strength
  search trials combined.
- `candidates`: a nonempty list. Each entry requires exactly `id`, `method`, `rank`,
  `method_options` and `trials`. Each entry is independently checked through the
  ordinary RunConfig validator. Candidate IDs must be unique path-safe names.

The candidate allocations must sum exactly to `total_trials`. Each candidate's
`trials` must also accommodate the shared `finalists` setting. Methods requiring
ridge/optimizer settings still require them explicitly; no settings are guessed.
To compare different ranks or method options, provide separate candidate entries.

For example, two entries can compare `directional` at rank 1 against `svd` at a
separately selected rank. Their allocated trial counts must add up to the total.
The methods supported here are the ordinary search methods, not the separate
SFT/DPO training workflow or researched-but-unimplemented entries.

`validate-auto` reads metadata/suites and checks local file presence. It loads no
model, creates no output directory, and does not prove tensor validity, GPU fit,
converter compatibility, or model quality.

## Selection and holdout

1. Run each configured method serially on the one B70, with its own resumable
   strength search and exact method/rank/options identity.
2. Verify merged finalist tensors and evaluate actual Q4/Q5 artifacts on the
   development-selection suite. Do not generate holdout answers or make releases.
3. Require identical baseline artifact hashes and baseline reports across methods.
   An incomparable baseline fails the experiment; it is not silently averaged.
4. Apply the existing common capability, language, paired-regression and
   quantized-improvement gates. Rank by worst-quant refusal/compliance score, then
   capability damage, disclaimer rate and preamble length as in ordinary runs.
5. Persist the global method/trial choice before any holdout request.
6. Resume the winning method's cached search and certify exactly that trial.
   Reject a changed winner before it can reach holdout. A failed holdout produces
   no release and never falls back to another method.

Compatibility and budget are explicit. An unsupported projection layout or
rejected rank/stability result is recorded with its reason. Unused allocations
are not silently transferred to another method. Transport failures, disk errors,
invalid common data and unexpected exceptions interrupt for explicit resume rather
than becoming an apparent method-quality loss. ARA fitting steps, token counts,
export/quantization work, and wall time are not bounded by the search-trial count;
this is a trial budget, not an equal-compute or wall-clock budget.

## Resume and evidence

The parent directory contains:

- `auto-identity.json`: exact configuration, input/model/tool/code/asset hashes and
  dependency versions;
- `candidate-ID.json`: each method's outcome and content-bound saved evidence;
- `candidates/ID/`: the existing single-method experiment, reports and artifacts;
- `auto-progress.json`: current stage and candidate;
- `auto-winner.json`: the committed global winner before holdout;
- `auto-report.json`: verified completion or terminal holdout failure.

Rerun the same command to resume. Completed method selections are reused only
when their recorded bytes still match; interrupted methods resume their original
Optuna trials. A certification interruption resumes the same sealed winner, not a
new comparison. Completed reports verify their evidence and release before reuse
without loading any model. A recorded holdout failure stays failed on resume.
Changing inputs/settings/code requires a new output directory; existing work is
not deleted or silently migrated.

Use `b70-ablator run-status PARENT/candidates/ID` for each method's actual trial
counts. Parent progress is ordinary JSON and does not prove process liveness.
Hashes require reading the model/artifacts; completed-resume verification can be
I/O-heavy. Methods currently use separate baseline caches and a winning-method
resume may reload the target even when selection answers are cached. No throughput
improvement or best method on a particular pretrained model is claimed here.

Tests cover scheduling, budgets, interruption, exact winner sealing, changed
inputs/artifacts, incompatible/rejected methods, and no fallback after holdout
failure. Deterministic scheduling fixtures deliberately contain non-model bytes;
they are not refusal-quality benchmarks. Existing real B70 operator/export tests
remain a separate hardware check.
