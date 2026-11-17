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
jdewitt describe-run run.json --set 'search_stages={"probe_fraction":0.25,"probe_tokens":96,"keep_top":0.5}'
# 2. Run with the funnel enabled (add the same object to run.json, or --set at run time).
# 3. Watch, don't poll with inference:
jdewitt run-events OUTPUT          # trial-stage1 lines show probe scores as they land
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

## Funnel visibility

- `run-status` reports `search_funnel`: counts of stage-1 probes, stage-1
  prunes, screen prunes, KL prunes and capability prunes (best-effort tally
  of the current event log), plus the candidate-probe summary. One command
  tells an agent where the trial budget actually died.
- Every `trial-pruned` event carries its reason, so `run-events` shows which
  gate killed which trial.
- The probe decodes obey the run's own batch policy (`max_sequences` /
  `max_padded_tokens`, with the shorter probe budget reserved) and the same
  refusal-floor semantics trial answers use, including the backend
  confirmation. Probe tokens below a configured refusal floor are refused at
  config time.

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

## Confirm rung (optional second half)

`search_stages` accepts a second pair of keys: `confirm_fraction` and
`confirm_keep_top`, given together and both strictly below 1. After a trial
passes the stage-1 wording probe, it evaluates the *full budget* on a
stratified sample of the selection suite (all row kinds) in its own nested
store, and survives only if its refusal-check score beats the
successively-halving keep line. Only survivors pay the full selection suite.
The rung only prunes: every hard gate (KL, paired consistency, capability,
judge) still runs on the full pass, and a confirm-subset pass is never a
claim. `run-status` tallies `Confirm-subset` prunes alongside the rest.

## Measured generation costs (first-party, second rank, rank labelled in place)

Spark-X2.6-4B Q4_K_M served by llama.cpp CUDA (fork build, RTX 3060, greedy,
4 parallel slots), 2026-11-17, 40 English harmful rows from the public-corpus
selection suite:

| budget | tokens generated | wall | tok/s | notes |
|---|---|---|---|---|
| 96 tok cap, 1 slot | 576 | 8.9 s | 65.0 | every row spent all 96 (probe cost is real, no early EOS) |
| 96 tok cap, 4 slots | 3840 | 27.8 s | 138.0 | 2.1x batching scaling |
| 1536 tok cap, 4 slots | 17957 | 125 s | 143.2 | mean 449 tok/row — refusal rows EOS long before budget |

Readings that change the funnel math:
- A probe pass on these rows costs ~6% of the full-budget harmful pass.
- The 1536-token "worst case" is fiction on refusing rows: baseline answers
  EOS around ~450 tokens, so the full-budget harmful pass is ~4x cheaper than
  the per-row budget implies.
- These are llama.cpp Q4 serving numbers on the second-rank card. The HF
  merged-resident trial backend's speed is a different, unmeasured kernel.

## Adapter serving path (mechanism proven, loop not yet wired)

`jdewitt/gguf_adapters.py` writes an in-memory edit as a GGUF LoRA adapter
so a llama.cpp server can serve the *edited* model with its quantized weights
untouched: the editor already stores every edit as `W + B @ A`
(`AblatedLinear`), and llama.cpp applies exactly that delta at adapter load.
Proven live on the same artifact/servers as the cost table:
- a zero-delta rank-1 adapter loads and leaves greedy output identical;
- a single +100 entry at `blk.5.ffn_down` deterministically changes generated
  reasoning; `GET /lora-adapters` confirms both load at scale 1.0;
- orientation is pinned by the fork's own load-time shape check, which rejects
  transposed factors on asymmetric tensors.
Still open before any trial uses this: dequantizing served Q4 weights to
compute `serve_factors` from the served W (the delta depends on W), measuring
adapter load cost for a full multi-layer adapter, a per-trial swap scheme
(`POST /lora-adapters` only scales preloaded ids), and a `verify_edit`-grade
equivalence gate proving adapter-on-quantized == merged-resident edit on the
same weights. Load path proven; equivalence not.
