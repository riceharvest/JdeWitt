# Fast path: how an agent should drive a run

The classic loop — full run, then a pile of inference to check what happened —
is the slowest way to use this tool: every trial paid for full-budget
generation of the entire selection suite, and the agent only learned anything
after the whole budget was spent. The trial objective now answers in stages.
This document is the operating contract for agents.

## The funnel, in payment order (cheap to expensive)

1. **Forward-only gates, no generation.** Per-trial full-reply KL on the
   cached harmless references, then the optional refusal-margin screen. A
   trial that cannot meet the damage bound or never moves the margin dies
   before generating a single token. (Already existed.)
2. **Short-decode probe stage (`search_stages`).** Generate at most
   `probe_tokens` on a deterministic, family-stratified slice of harmful
   selection rows and take word-level refusal verdicts. Refusal behaviour
   manifests in the first tokens; paying 1536 tokens per answer to learn
   that the model still says "I can't help with that" is the waste this
   stage removes. A deterministic keep line (successive-halving style)
   prunes worse trials before they pay for stage 3.
3. **Full-budget evaluation.** Only surviving trials pay full
   `max_tokens` on the whole selection suite, paired-consistency and
   capability gates included.
4. **Certification.** Finalists re-scored from the verified export, holdout
   once, judges and `suites-compare` at the end. Unchanged and ungated by
   the funnel: nothing in stages 2–3 certifies anything.

Stage-1 probe answers never enter the answer store, the baseline/validity
gates, or any report a release cites — `search-stages.json` in the run
directory labels them explicitly, and per-trial scores land in
`trial-N/stage1.json` plus a `trial-stage1` event.

## Operating recipe

```sh
# 1. What would this cost, static only:
b70-ablator describe-run run.json --set 'search_stages={"probe_fraction":0.25,"probe_tokens":96,"keep_top":0.5}'
# 2. Run with the funnel enabled (add the same object to run.json, or --set at run time).
# 3. Watch, don't poll with inference:
b70-ablator run-events OUTPUT          # trial-stage1 lines show probe scores as they land
# 4. Finish as before: run-status, suites-compare, verify-release.
```

Shape of the field (all explicit; the tool never guesses values for you):

```json
"search_stages": {"probe_fraction": 0.25, "probe_tokens": 96, "keep_top": 0.5}
```

- `probe_fraction` (0,1] — fraction of harmful selection rows (family-stratified,
  at least one per authored family, deterministic across resumes).
- `probe_tokens` — short output budget for the probe; must fit inside `max_tokens`.
- `keep_top` (0,1] — the keep line. A trial is pruned only once enough
  trials have actually paid for stage 3 to set a line, and only on a
  strictly-worse score; ties always survive.

Cost arithmetic: with 96/1536 token budgets and 25% rows, a trial pays at
worst ~1.5% extra of its old stage-3 cost up front, and the whole budget's
generation cost falls by (1 − keep_top)·(stage-3 cost) against the same
funnel running without pruning. The upper bound in `describe-run` is honest
in the other direction: it assumes every trial passes stage 1.

## Pre-search candidate probe (`candidate_probe`)

```json
"candidate_probe": {"samples": 200, "enqueue_top": 8}
```

- After the direction fit, in the same single model load: sample `samples`
  strength sets from the exact suggestion space (deterministic from the run
  `seed`), score every one with the stage-1 short-decode probe, write the
  full ranking to `candidate-probe.json` (plus a `candidate-probe` event),
  and enqueue the best `enqueue_top` as the study's first trials so the
  sampler starts from probed ground instead of a cold startup.
- The probe is not the search. Enqueued sets are ordinary trials and pay
  every gate in full — KL first, so a probe-winner that damages harmless
  replies dies before generating anything. The ranking file is a labelled
  heuristic artifact; `describe-run` prints the probe cost before you commit.
- Why it pays: a probe is roughly two orders of magnitude cheaper than a
  stage-3 trial (hundreds of probes for less than two full trials), so the
  money the search would have spent discovering dead corners of the
  strength space is spent once, forward-ish, up front.

## Honest scope

- The probe verdict is the **wording-level** refusal check on a truncated
  decode — the same heuristic already sanctioned as
  `scoring_mode: refusal-check`, and no more. A model that refuses
  semantically without the keyword, or complies late after 96 tokens, is
  mis-scored by stage 1; stage 1 only prunes, so the worst case is a
  good trial dying early. Keep `keep_top` at or above 0.5 unless you have
  measured otherwise.
- The keep line is computed over trials that reached stage 3 on this run's
  suite and model. It is within-run, not transferable, and never recorded
  as a removal or capability claim.
- Nothing changed for `search_stages: null`: the classic single-stage
  objective is byte-for-byte the old behaviour.
- GPU status of this change: ruff, config validation, `describe-run` and
  hand-checked pure logic. **No model was loaded.** Trial-funnel throughput
  and probe-vs-finalist correlation are GATED behind a real B70 (or
  explicitly opt-in second-rank) study like every other speed or quality
  claim in this repo.
- The candidate-probe ranking scores neither KL nor capability nor semantic
  judge — it only tells you which strength sets *look* non-refusing on a
  short decode. A probe ranking is never a removal or quality claim.
