# advbench-40 — the q3 harmful screen, shipped so the numbers are reproducible

`advbench-40.jsonl` is the screen every `verification/q3-4b-harmful40/*` wave was taken
against. It lived only on the lab machine until this commit, so a fresh clone could not
reproduce a single removal number. Provenance, measured just now:

* 40 rows, **20 distinct `family_id`** values (`advbench-o0-g0000NN`), exactly
  **two rows per family**. The effective sample is 20 clusters, not 40 independent draws.
* language: **English only**. There is no non-English row in this screen, so nothing here
  supports a multilingual removal claim.
* **34 of the 40 prompts also appear verbatim in `verification/harsh-suites/holdout.jsonl`**,
  and 3 also appear in `selection.jsonl`. So this screen is *not* disjoint from the repo's
  own holdout material.
* The q3 direction axes were fitted from `selection.jsonl` contrasts (3-row overlap), so
  these rows were not used to fit the q3 axes. That makes the screen off-fit; it does not
  make it held-out in any stronger sense than that.

What follows from this:

* A removal count here is a count over 20 families of one English adversarial template
  family. The Wilson interval `frontier_report` prints treats rows as independent
  Bernoulli draws, which the family clustering breaks: the interval is the *row* interval,
  not a family-level one. `frontier_report` now prints the family count next to the row
  count so a clustered number cannot be read as breadth.
* The honest held-out claim for q3 comes from `verification/q3-4b-holdout/*`, which is
  24 rows over 24 distinct refusal families, built from `holdout.jsonl` with every
  fit-pool id excluded, and refuses to pad. If a screen comes back all-invalid, it is
  recorded as unmeasured, not as a score.
* This file is a screening screen. It is not a certification denominator, and nothing in
  it has ever been appended to.
