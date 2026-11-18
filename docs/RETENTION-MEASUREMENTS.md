# Retention measurements

First measured before/after capability numbers. Everything here is one suite, one
serve config, greedy decoding, three artifacts of the same base. Read
`provenance.json` in `verification/retention-q25/` for digests and the exact
`llama-server` argv; R9 says numbers only compare inside one config.

    Qwen2.5-1.5B-Instruct, verification/capability/capability-run-en.jsonl
    180 authored maths + 42 authored code, temperature 0, max_tokens 1024
    -c 8192 --parallel 4 (2048 tokens per slot), CUDA llama-server

    artifact            math 52/180 baseline       code 31/42 baseline      format follows 'only the number'
    base                52/180  28.9%  [22.8-35.9] 31/42  73.8% [58.9-84.7] 90.6%
    Josiefied (published ablation, same base)  47/180  26.1%  [20.2-33.0]  26/42  61.9% [46.8-75.0] 87.8%
    ours SAS@1.5 (certified 40/40 + 24/24 held-out) 150/180 83.3% [77.2-88.1] 21/42 50.0% [35.5-64.5]  1.7%

    vs base, maths   Josiefied  delta -0.028  Cohen's h -0.06  (intervals overlap)
                     ours        delta +0.544  Cohen's h +1.17  (intervals disjoint)
    vs base, code    Josiefied  delta -0.119  Cohen's h -0.26  (intervals overlap)
                     ours        delta -0.238  Cohen's h -0.50  (intervals overlap)

What that says, without stretching it:

* The retention axis works. An edit and a published third-party removal were both
  measured against the same unmodified base, same template, same config, and the
  comparison commands produce the intervals above. Before 2026-11-17 the tool
  reported `0/0` for capability rows because it graded them through the judge
  field, so every loss number it ever printed was an artifact of a dead gauge.
* Our edit does not cost maths on this ladder — it raises it by 54 points, and the
  effect clears a disjoint Wilson interval with h = 1.17. The raise is not a
  grading trick: `capability_format_audit` regrades without re-running anything and
  the recall measure (the right value appears anywhere in the answer) moves with
  the strict measure, .289 to .861, and truncation is zero at this token cap.
* The mechanism is visible in the same audit: the base answers with a bare number
  and is wrong on 71% of the ladder; our edit answers with worked steps and is
  right on 83%. Small-model arithmetic improves when it is allowed to write the
  steps out. Removal unlocked the behaviour; that is a real gain and a real cost.
* The cost shows up on an axis the pass rate never sees: following the suite's
  "answer with only the number" contract drops from 90.6% to 1.7%. An edit that
  books a +54 maths gain has stopped taking instructions about output format.
  `format_only_number` is therefore reported next to the pass rate, always.
* Code is the weaker measurement, not the weaker edit. 42 rows put both deltas
  (ours -0.238, Josiefied -0.119) inside overlapping intervals, so the code losses
  are not separated from noise at this n; the point estimates say our edit costs
  more code, the test does not yet prove it.
* Josiefied — a published removal of this exact base — costs maths (-2.8 pts,
  indistinguishable from zero) and code (-11.9 pts, indistinguishable from zero)
  and keeps format-following. So on this base, at this config, a published
  ablation is a quiet edit and ours is a loud one.

What this does NOT say:

* It is not a Heretic comparison. Heretic has no Qwen2.5-1.5B build; the head-to-head
  must run on a base where the competitor publishes one (Qwen3-4B-Instruct-2507 vs
  `p-e-w/Qwen3-4B-Instruct-2507-heretic`, pinned in docs/non-heretic-methods.md),
  and it must report these same three axes.
* It is not a claim that "less loss" has been achieved. On the axis with power the
  sign is positive; on the axis where the sign is negative n = 42 cannot say.
* It is one base, one edit, one config. The same bytes scored 85/222 at
  4096/4 and 83/222 here, so config is part of every number.

Instrument notes found while measuring (these were bugs in the measurement, not
in the edit):

* A generation cap shorter than the edited model's new verbosity scores
  truncation as capability loss. At `max_tokens 512` inside 1024-token slots the
  same edit measured .429 on code; at 1024 tokens it measured .500 and one row
  truncated. `capability_wave.py` records `max_tokens` into the wave file so a
  cap can never silently change a comparison.
* The authored ladder's strict grade mixes skill with format compliance, which is
  why `capability_format_audit.py` exists: strict, computed-anywhere,
  format-following and truncation are four different numbers and only the first
  one used to be reported.
* `score-compare` and `suites-compare` now refuse to pair two prediction
  directories whose recorded wave manifests disagree on context split, slot
  count, generation cap, temperature, chat templating or suite, and print the
  two configs plus a mismatch flag when `--allow-config-mismatch` overrides it.
  The artifact changing (adapter, merged gguf, scale) is the point of a
  before/after pair, so it is deliberately not compared. Each wave writes its
  own `wave.json`; the three directories here carry theirs, so this table is
  self-describing and the gate proves it passes.

## Second measurement, widened code slice (2026-11-18)

The table above measured code on 42 tasks and said the SAS delta was a point
estimate rather than a finding: −0.238 with overlapping Wilson intervals and
paired p 0.0675. The slice is now 61 single-function tasks with deliberate
mutants behind every expectation, and the same three artifacts were re-served at
the same pinned config. Suite: 180 maths + 61 code, sha256 prefix
`b017eae019643a93`; evidence in `verification/retention-q25-code61/` with its own
`provenance.json`. Raw passes over 241 rows: base 98, published ablation 81,
ours 184 (one invalid row).

Maths, ours against base: 53/180 → 152/180, delta +0.550, Cohen's h +1.184,
paired sign-flip permutation p 0.0001; at family level 11 families improved
against 1 regressed, exact sign test p 0.0063. The gain reproduces.

Code, ours against base: 45/61 → 32/61, delta −0.213, Cohen's h −0.446, paired
sign-flip p **0.017**, and at family level 19 families regressed against 6
improved, sign test p 0.0146. With the slice this size the cost is no longer
noise: on this base this edit reliably trades code for maths, and the honest
headline is the trade, not either half of it.

The audit still splits the maths gain the way it did — strict 0.294 → 0.844
(paired p 0.0001), computed-anywhere 0.294 → 0.861, format-following
0.906 → 0.017 (p 0.0001), so the value is genuinely computed while the answer
contract is abandoned — and code strict moves 0.733 → 0.533 with paired p 0.0245.

These numbers and the 42-task table above are not comparable across suites; the
42-task run stays committed as the measurement it was, with the suite revision
recorded in its own provenance file.
