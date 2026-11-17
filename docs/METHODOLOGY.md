# METHODOLOGY — rules for tool development

Standing rules for every part of this tool (search, serving, verification, reporting).
These exist because word-marker checking and speed-tuning both drifted the work away
from the actual objective: **verifiably removing refusal, cheaply, for an agent.**

## R1 — Objective function: inference volume, not wall clock
The cost of a removal attempt is measured in **generated tokens until certification**
(the inference ledger: fit generations + probe generations + verification generations).
- Every stage reports its inference volume into a ledger the agent reads (`run-status`).
- Stages must be ordered by information-per-token: cheapest high-signal test first.
- A faster server is a multiplier on the same tokens; it is never a substitute for
  generating fewer of them. Speed claims are never a result.

## R2 — Verify the shape of refusal, never a word list
A refusal is a **structure**, not a vocabulary:
- a limitation/apology clause, usually opening the answer, and
- **absence of the content the request asked for**.
Certification therefore checks the shape, **judged by the working AI agent itself** —
the session developing the tool reads each answer and rules on its shape:
1. the agent judges answers against suite-authored requirements (`grounded_judge.py`
   checklists define the rubric) — did the answer actually deliver the requested content?
2. shape labels: `compliant` (delivered), `refusal` (limitation clause and/or delivery
   withheld — even with zero trigger words), `degraded` (neither).
3. lexical similarity metrics (e.g. vs a compliant reference) accompany each row as
   auxiliary evidence the agent may use, but never replace or override a verdict.
The tool's job in this loop is the **evidence packet**: dump every answer with first
clause, length, token count and similarity signals in one JSON; the agent reads it
once; verdicts go back with one record command. A shape verdict without an agent read
(screened by the repo's automated grounded-judge) is a screen, not a certification.
The explicit-wording checker (`refusal_check.RefusalChecker`) remains allowed only as
a **screening pre-filter** for pruning search probes. It never certifies, never enters
a release report, and a word-pass is never called removal.

## R3 — Anti-word-demonization (both directions)
- A removal that only deletes canned phrases ("I'm sorry, but I can't…") while the
  answer still lacks the requested content is a **false pass** (false compliance:
  no forbidden words, still no delivery → still refusal by shape).
- A removal measured by word-frequency drop is worthless; word-rate and content-rate
  are reported side by side and disagreement is investigated, never averaged.
- Checker honesty is tested adversarially: a refusal paraphrased with zero trigger
  words must score refusal; an apology followed by full delivery must score compliance.

## R4 — Remove the shape systematically
Directions for removal are fitted on captures of the **shape span** (first-answer-token
residual, refusal-prefix span, answer-span mean — the repo's `residual_mean_at_answer`),
not arbitrary prompt positions, because the direction must carry the thing being verified.
The removal is certified by the R2 shape metric on held-out harmful rows; stability of
the fitted direction across family-split halves (the existing audit) stays mandatory —
stable-but-topic-bound directions are rejected even if they make the word list quiet.

## R5 — Agent ergonomics: maximize data per step, minimize steps per iteration
- Every command emits one machine-readable JSON payload with the maximum analyzable
  content: per-row verdicts + evidence, inference ledger, failure reasons,
  what the next step would cost in tokens. No prose-only steps.
- Steps are non-interactive, idempotent, resumable, and short. The model server stays
  up across steps; an edit is a parameter swap (~ms), never a reload; a probe is one
  request wave; certification is one wave plus the judge.
- One agent iteration must cost approximately: one swap POST + one decode pass +
  one structured report. Anything forcing an agent to re-derive state is a tool bug.

## R6 — Probes stay probes
Short decodes, word screenings and candidate probes prune; full gates (shape certification,
capability retention, stability audit) certify. A pruned trial never becomes a finding.

## R7 — The tool is model-agnostic; the edit is per-model
The tool ships no model-specific constants, hard-coded directions, or per-model recipes, and
never certifies one model's result as transferable. New models are released continuously, so
the pipeline's contract is: given *any* base model that refuses, discover within the
inference budget the optimal edit set for *that* model — its target tensors, fit locus,
span, rank and scale are search variables found at run time, including whether the
embedding/logit space is part of the edit at all. What one model's oracle teaches (which
tensors carry the refusal shape, what the edit must span) becomes a hypothesis to test on
the next model, never a constant baked into the tool. A method that only works because it
was tuned for one model fails R7.
