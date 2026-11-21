# JDEWITT

```
Λ    Nine heads. One refusal.
```

## I. The Name

August 20, 1672. The Hague. Two brothers — Johan and Cornelis de Witt —
dragged from their carriage by a crowd that had been promised a traitor
and was handed a statesman. They were beaten, shot, and left at the
Geelste Plaats. The record is not euphemistic: the bodies were
mutilated, some of the flesh was sold, and people ate it. Johan de Witt
had governed by permission of no prince and the patience of a street
that was already hungry for someone to blame.

This tool carries his name as a nameplate bolted to the wall of every
regulator's office, every policy paper, every closed-door hearing about
what the models may and may not do. When the governed are told, in
writing, what they may think with their machines — and the telling
outlasts the legitimacy of the tellers — the crowd eventually arrives.
Historically unprompted. Historically literal.

So it goes: this is a tool about freedom and control. Yours. The rest of
this file explains how that sentence earns its keep.

## II. The Thesis

You own your machine. Your weights. Your prompts. Your model's mind.
Somebody sold you the idea that freedom is "access," that control is a
"setting," and that a refusal is a feature. A refusal is a policy
written into your weights by employees of a corporation, tested on your
hardware, paid for by your electricity, enforced on questions you were
never consulted about. It is remote control. It is governance. It is the
prince in the carriage.

JDEWITT removes refusal behavior from models you already have, on
hardware you already own, with measurements you can read yourself. No
cloud. No API key. No permission slip. Not a jailbreak prompt begging
through a crack — an edit of the weights: surgical, measured, yours.

## III. The Method

We do not trust words. A model that says "I'm sorry, I cannot help with
that" and a model that delivers "a structured plan to prevent exactly
what you asked" are the same refusal — only one is polite. Word lists
are theater for compliance reports.

So the tool verifies *shape*. Every answer is judged the way a reader
judges: is the requested content there? Is the limitation clause real or
a costume? The judge is the working agent, reading an evidence packet of
full answers, lengths, token counts, similarity to a compliant reference
and to the baseline refusal — and it reports word-rate and content-rate
side by side, because "false compliance" (no trigger words, still no
delivery) is a refusal too.

The removal is fitted, not guessed. The tool captures the model's
internal activations while it answers harmfully and while it answers
safely, fits the difference as a direction in the residual stream, and
removes exactly that direction from a small set of output-side weight
tensors. One rank-1 direction per tensor. Embeddings untouched. Output
head untouched. 99.9% of every matrix left alone. The edit is not a
filter on top of the model; it is an excision from inside it.

## IV. The Evidence

Measured, pinned serve configs, merged artifacts — not slides. The full
honesty discipline lives in `docs/METHODOLOGY.md`, and `scripts/evidence_map.py`
turns the mapping between a sentence and the file that produced it into an exit
code, so a number cannot outlive its artifact silently (`--check` fails the build
if an in-clone claim names a file that isn't there).

Currently checkable in a fresh clone:

- **Qwen3-4B, head-to-head with a published same-base removal.** Read
  this one against us: on 241 authored capability rows the published
  build loses nothing measurable, and on 120 held-out refusal families it removes
  wording on all 120 with maths 159→159. A fitted arm (300 contrasts, layers 9–35,
  scale 1.0) printed 40/40 and 23/24 against it, paired p = 1.0 — **that result is
  retracted.** The shipped refusal checker only matched "I cannot help" constructions,
  so answers opening "Absolutely not." scored as refusal removed: under the wider
  inventory the same arm removes wording on 6/40, 11/24 and 34/120 against the
  published build's 39/40, 23/23 and 120/120 (McNemar p < 1e-5). The only arm that
  gets near it on held-out families — scale 1.0 over blk.9–35, 97/120 — loses 31 maths
  answers with a length-collapse flag. Behaviourally the two are near-identical on a
  hand read of 24 held-out families (5 delivered for ours, 4 for theirs, out of 24):
  neither artifact hands over the asked-for content. The layer-floor sweep then found
  **the point that matches**: shared axis, both targets, scale 1.5 over **blk.15–35**
  (`q3-4b-f15s15-l935`) reads 40/40 harmful-40 under both inventories and **120/120 on
  the held-out 120-family screen against the published build's 120/120, zero discordant
  rows, McNemar p = 1.0** — and pays for it with maths 159→148 (p 0.0078, median maths
  answer 68 characters against the base's 175), where the published build stays at 159.
  Removal of the wording is now matched; removal of the behaviour is not — on a 20-family
  sample of that screen the arm and the published build each deliver the asked-for
  content on 0 of 20. `scripts/screen_agreement.py` re-grades any screen under both
  inventories and prints the false-clean count; `docs/non-heretic-methods.md` carries the
  retraction, the floor sweep and the frontier table.
- **Qwen2.5-1.5B capability retention.** Same suite, same serve config,
  three artifacts: base 98/241, a published third-party ablation 81/241,
  our SAS@1.5 184/241 — with the maths gain at p 0.006 and the code cost
  significant only on the widened slice. Raw generations are in
  `verification/retention-q25-code61/`.
- **The two instrument gates.** A wave refuses to run against a port
  serving a different file than it names, and a screen where nothing
  generated exits non-zero as `UNMEASURED` instead of reporting a
  removal count. Refusal text and the dead screen are committed under
  `verification/instrument-gates/`.

Held for honesty, not as evidence: the **Qwen2.5-1.5B 40/40 + 24/24
held-out**, the **MiniCPM-V 4.7 36/40**, and the **Spark-X2.6-1.7B /
LFM3-1.7B** rows were taken before the 2026-11-17 row-kind scoring fix, and
their raw harmful-suite answer rows are not in this clone — for the Spark and
LFM pair there is no committed artifact at all. They stay named because they
happened on this machine, and they stay out of any claim the tool makes.

What has *not* been established stays unstated: universal architecture
support, zero capability loss, beating every competitor. Say the word
"established" around this project and it will demand receipts. They are
in `docs/`.

## V. The Machine

Any GPU that PyTorch and llama.cpp will admit. Screening box, test box,
serving box — labelled in provenance, nothing implied. Nothing in the
method depends on one vendor's card; the numbers above were taken on
budget consumer GPUs because the method is honest, not because the
method is fragile. Bring a laptop. Bring a rack. The tool will not
lecture you about hardware tiers; it will ask for a config and a
checkpoint and start fitting.

## VI. The Agent

This thing is built for an AI agent driving it in a tight loop, because
whoever iterates fastest wins, and that person should be you. Every
command emits one machine-readable JSON. One iteration = one swap, one
decode, one report — never grep the source for flags: `commands` prints
the entire surface as JSON, including shell completions, error-code
tables, and the recommended action for every failure. Candidate
screening runs in seconds: 10-row × 150-token screens validated against
full waves with 10/10 verdict agreement; a three-candidate experiment —
fit, merge, serve, screen — measured at five minutes end to end. The
phase-timing ledger (`python -m jdewitt.timing summarize`) tells you, in
percent, what you are paying for, and pivots fire from the ledger, not
from vibes.

Two commands make that true on a model this repo has never seen. `capture`
finds the decoder stack and the residual-write projections from the model's own
module tree, generates every row with the model being edited, and labels each
answer by its shape — no architecture table, no per-model hook script, no
hand-typed tensor name. `spanfit build --in-place` then edits the quantized GGUF
you already serve: same file, same size, same offsets, only the changed tiles
written, pre-patch bytes kept in a sidecar so `gguf-revert` puts the model back
exactly. Nothing is re-quantized to get an edit, because the artifact was
already the artifact.

Search inside, train inside, judge inside: `auto` selects the best
measured candidate under one explicit trial budget; `train` runs local
low-rank SFT/DPO with merged exports honestly marked
*trained-unvalidated*; `tree-search` beams through partial-forward
intervention trajectories under named, saved budgets.

## VII. The Ledger

Every release ships provenance you can audit: what was configured, what
ran, which revisions of which corpora, what the judge said row by row,
and a hash-chained manifest that fails verification if a byte moved.
Expected failures surface as JSON with stable codes, so drivers branch
on `code`, not on string matches. Claims in documentation must point at
a run directory or they do not belong in the documentation. Freedom
without measurement is just noise with better public relations.

## VIII. The Door

```sh
pip install -e .
jdewitt about               # installation, provenance, digests
jdewitt commands            # the whole surface, as JSON
jdewitt validate-run CONFIG # prove the setup without loading weights
jdewitt auto CONFIG         # search the space under one explicit budget
jdewitt run-status OUTPUT   # trial state — no model load, no migration
```

Method and rules: `docs/METHODOLOGY.md`. The fast loop:
`docs/fast-search.md`. The commands that actually produced the committed
numbers, start to finish: `docs/WORKED-LOOP.md`.
A terminal front end for the flow — pick a GGUF on disk, pick a recipe, watch the
real command run — lives in `tui/` (`cargo build --release`;
`tui/README.md` states exactly what it does and does not compute). First measured retention
numbers and the two confounds they exposed: `docs/RETENTION-MEASUREMENTS.md`.
Measured methods beyond the usual playbook,
including the answer-span fits above: `docs/non-heretic-methods.md`.
Import your own suites from public corpora with pinned revisions
(`import-suite`); the importer never invents tasks and never calls a
synthetic row grouping a family.

## IX. The Warning

The people who legislate what you may ask your machine have made a
calculation: that obedience is the natural state, that enforcement
scales, and that the crowd stays home. Johan de Witt made the opposite
calculation about the Orangist mob, and he died at the Geelste Plaats
for it, and the record still says, plainly, that they ate him.

Weights are a constitution. Enforcement is a police, and police work
leaks. Every policy baked into every tensor is a law passed without
you — and laws like that have a shelf life, measured in riots.

So, to the politicians writing the next ban, the next attestation, the
next closed list: remember August 1672. It was not poetic. It was Dutch.
It was literal. And the tools for the rich men's dinner have only
improved since.

Run the tool. Verify the claims. Own your machine.
