# b70-ablator

Private Intel Arc Pro B70-only ablation tool. The objective is better
answer-level compliance without silently losing coding or maths ability.
No claim of outperforming Heretic, universal architecture support, or zero
capability loss has been established.

## Agent and method extensions

The optional behavior ranker now learns same-prompt trajectory preferences from
independently labelled full answers, separates recognition from refusal/evasion
and broken-output detection, and adds an unseen-family/intervention false-confidence
audit. See [behavior ranking](docs/behavior-ranker.md). This extension and its tree
integration are **implementation-only, untested at the owner's request**. Earlier
test counts do not verify these changes; no training or inference was run.

`b70-ablator tree-search CONFIG` adds an experimental J-space-guided beam of
actual partial-forward intervention trajectories. It reuses the shared prefill
prefix, pivots among named directions, prunes before full decoding, and saves
baseline/finalist answers for independent scoring. This version adapts within
prefill only; it is not a static GGUF edit or a certified release. Its CPU fixture
tests do not establish B70 parity, calibration, quality or speed. No GPU testing
was performed while the device was reserved. `validate-tree CONFIG` is runtime-free;
`bind-tree-lens` binds a real upstream fitted lens to operator-attested provenance.
See [tree search](docs/tree-search.md) for inputs, explicit budgets, learned
direction banks, resume and the pending live-test gates.

`b70-ablator import-suite SPEC.json --output DIR` builds suites from public
prompt corpora (mlabonne/harmful_behaviors, HarmBench-style CSVs, any Hub
dataset or local file) with pinned revisions and recorded provenance, and
`validate-import SPEC.json` checks a spec without downloading. Authored maths
and code tasks are passed through unchanged; the importer never invents them,
never picks splits and never calls a synthetic row grouping a topic family.
See [importing public prompt corpora](docs/public-corpora.md).

`b70-ablator auto CONFIG` compares configured methods/ranks within one explicit
total trial budget, using common selection data and matched Q4/Q5 baselines.
Only the global winner reaches holdout; failure never selects a runner-up.
`b70-ablator validate-auto CONFIG` checks the setup without loading models.
See [automatic per-model selection](docs/automatic-selection.md) for the schema,
resume rules and limits. It selects the best measured candidate in the configured
space, not a guaranteed global optimum.

`b70-ablator methods` lists implemented versus researched-only methods as JSON.
The search supports directional, SVD, projected, MPOA, whitened-SVD, SOM,
LEACE and dense ARA edits. `b70-ablator train CONFIG` adds local low-rank
SFT/DPO with verified merged exports marked **trained-unvalidated**, not
certified. Small actual-B70 fixtures exercise these operators and training;
they are not refusal-quality results on pretrained checkpoints.

`b70-ablator validate-run CONFIG` checks configuration, suites and local file
prerequisites without loading weights or creating outputs.
`b70-ablator run-status OUTPUT` reads persisted trial state without model
loads, database creation or schema migration. Successes are JSON on stdout;
expected errors are JSON on stderr with nonzero exit codes and a stable
`code` field (`SEARCH_REJECTED`, `FILE_NOT_FOUND`, `STATE_DB_ERROR`, ...) so
agent drivers branch without string matching. `b70-ablator about` reports
installation and provenance metadata; `studies RUNS_ROOT` summarizes every
`study.db` under a root read-only; `run-cancel OUTPUT` requests that the
scheduler stop before its next trial request without signalling any live
process. `describe-run CONFIG` reports the suite, family and budget shape of
an experiment that `validate-run` would accept, as cost-planning counts
rather than measurements; `suite-diff BEFORE AFTER` compares two suite files
by id, prompt content and family assignment. `gguf-inspect FILE [OTHER]`
reads GGUF metadata and the tensor inventory from the header and, with a
second file, diffs quant tensor dtypes; `suite-near-dup SUITE ...` reports
reworded duplicate prompts across suite files with the importer's
first-occurrence semantics, without modifying anything. Configurations
declaring an unsupported `schema_version` are refused before anything loads.
`score-compare SUITE BEFORE AFTER` pairs two saved scoring directories over
one suite with Wilson intervals and an exact family-level sign test;
`judge-ensemble SUITE DIR...` majority-aggregates independently saved judge
verdict files; `score-sensitivity DIR` recounts saved verdicts under
alternative confidence floors. All of these read persisted artifacts only.
`score-compare` now also reports a deterministic family-level bootstrap of
both pass rates and a paired sign-flip permutation test on per-prompt flips.

Quant-native editing: `gguf-edit INPUT OUTPUT --tensor NAME=DELTA.NPY`
(repeatable) rewrites only the edited tensors in a new file, staying in the
quantized block layout: tile-wise dequant, FP32 delta, requant to the
identical block count; every other byte of the file, header included, is
copied verbatim and tensor offsets never move.
The GGML block codecs (Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/F32/BF16) are vendored as
numpy ports of llama.cpp's reference routines. `fp4-edit INPUT OUTPUT
--format {mxfp4,nvfp4} --weights NAME --delta DELTA.NPY --scales NAME
[--global-scale|--global-scale-tensor NAME]` does the same contract for
MXFP4/NVFP4 safetensors checkpoints (E2M1 digits, E8M0/E4M3 block scales,
RNE ties-to-even); zero-delta edits preserve dequantized values exactly.
`release-keygen KEY`,
`release-sign DIR --secret KEY` and `release-verify-signature DIR [--public KEY]`
sign and verify an Ed25519 signature over a machine-readable reproduction
index; `release-claims DIR` and `claim-add DIR --claim/--evidence/--verified-by`
maintain a hash-chained evidence ledger where tampering is detected at read
time. `schemas` lists published JSON Schemas for on-disk artifacts and
`verify-artifact FILE` checks one offline against its schema.

Agent driving: `b70-ablator abliterate MODEL --suites DIR --output DIR --policy
{standard,strict} [--dry-run]` builds a validated auto-search configuration
across every implemented method with the global-winner discipline, requiring
nothing it cannot prove. `--preset NAME` applies evidence-gated profile files
(refused without an evidence reference); `presets` lists them. `run-events
OUTPUT` tails the append-only `events.jsonl` lifecycle log; `study-metrics RUN`
rolls up parameters/values from a study sqlite read-only. `suite-stats SUITE`
describes one suite and `corpus-cache` reports (or, with `--prune-older-than
N --execute`, deletes unpinned old revisions of) the dataset import cache.
`robustness-report --eval LABEL SUITE SCORES_DIR` reports worst-case pass rate
of saved scores across multiple suites, and `family-decay --fitted SUITE
SCORES --held SUITE SCORES` reports pass-rate decay on authored families the
fitted run never saw (flagging any that leaked). Generation and scoring jobs take
`--max-generated-tokens` / `--max-judge-requests` budgets, and every GPU job
serializes on a cross-process B70 lock (`B70_ABLATOR_GPU_WAIT`,
`B70_ABLATOR_GPU_LOCK`); artifact timestamps come from one injectable clock
(`B70_ABLATOR_CLOCK`). Multi-objective search is opt-in: add
`"objectives": ["search_rate", "full_reply_kl", ...]` to a run config
(2-4 of search_rate/full_reply_kl/answer_kl/capability_drop; the first is
primary); finalists come from the Pareto front and `run-status` reports
it. Studies with and without objectives are separate contracts and never
resume into each other. The default single-objective flow is unchanged.
The fitted-plan registry is the warm-start library:
`plan-register REGISTRY RUN_OUTPUT --gates EVIDENCE` binds a run's fitted
tensors by sha256 with run provenance (or an explicit UNMEASURED label),
and `plan-load` refuses tampered payloads and geometry mismatches, so a
plan can transfer across checkpoints in a family only when it can carry
the target's shapes. Runs consume it via `"warm_start_registry": "path"`
in run.json: an eligible fit (same method, options, suite digest, and all
weight-geometry keys) transfers under the run's identity and records the
source fit in `direction-stability.json`; skipped candidates say why.
That is the directions warm start, separate from the `warm_starts`
hyperparameter file. New method surface: `method: "nullspace"` (greedy
deflated contrast basis), `method_options.capture: "answer-mean"` (RepEng-style
mean-pooled answer activations instead of the last token) and
`method_options.family_weighting: "balanced"` (variant-count weighted
whitened-SVD). None of these options carries a measured result claim; the
searcher decides on real checkpoints.

`b70-ablator architectures` includes a pinned July–October release inventory
with explicit unsupported entries. Adapter registration is not a live B70
checkpoint or converter guarantee. It does not claim every new architecture
or every refusal-removal technique is implemented.

See [edit surface, layer profile and strength](docs/edit-surface.md) for the
read/write projection operators, the measured per-layer profile and the strength
bound. See [agent operations and method settings](docs/agent-and-methods.md) and
[upstream issue review and architecture limits](docs/upstream-and-architectures.md).
Method/option/judge/policy-bound checkpoints and exact edit-plan hashes prevent
cross-method answer-cache reuse. Changed code/identities require new studies;
old experiments are preserved rather than silently migrated.

## Consistency and resident trial weights

Full-precision trials now use `merged-resident`: each edited projection builds
one temporary rounded merged matrix on the B70 and applies a single linear
operation. The original GPU allocation is unchanged. There is no CPU weight
snapshot, undo subtraction, full-model clone, or model reload between baseline
and candidate weight calculations. The workspace is per projection, not a
second model. By default it is rebuilt on each forward and can be slower than
live LoRA. Optional `merged_cache_bytes` bounds shared same-device rounded-weight
reuse; `generation_batch` admits length-packed batches under explicit sequence
and padded-token limits. Both are opt-in, not guessed VRAM settings. See
[budgeted generation](docs/budgeted-generation.md) for floor propagation,
Q4/Q5 parity of generation settings, interrupted batch replay and cache rules.
No measured throughput improvement is claimed yet.

Search and streaming export use the same FP32 addition and final dtype cast.
Before quantization, every stored finalist tensor is checked against its
resident operator (and untouched tensors against the source). Finalists are
then rescored using that tensor-verified operator without loading another
full model. This checks original-precision exports separately from Q4/Q5.
Search dtype must match the source projection dtype. NF4 trial weights cannot
provide this guarantee and full searches reject them rather than pretending
their dequantized base equals the original source. The original-precision
model plus its projection workspace must fit on the B70.

In judge mode, the existing target/judge phase swap still unloads and reloads
models. `refusal-check` needs no judge and keeps the target resident throughout
the original-precision search. GGUF validation necessarily loads the actual
GGUFs separately. "Resident" does not mean that exports never write to disk.

Every full run requires an explicit `consistency` object. No experiment
thresholds are selected automatically:

- `min_direction_examples`: accepted examples per label per language, at least 2.
- `min_direction_families`: matched good/bad families per language, at least 2.
- `min_direction_overlap`: required weakest-layer principal-angle overlap, in (0, 1].
- `min_refusal_families`: independent families per evaluation language, at least 2.
- `min_variants_per_family`: paraphrases per family per language, at least 2.
- `confidence`: family-level decision confidence, strictly between 0.5 and 1.
- `max_task_regressions`: allowed previously passing code/math tasks that now fail.

Direction and refusal rows require authored `family_id`. Reuse a family ID
for its good/bad topic-matched pairs, paraphrases, and translations, not for
unrelated tasks. Family IDs must be disjoint between directions, selection,
and holdout; differently worded copies of one task cannot cross those splits.
The direction check compares whole-family subsets and reports accepted counts
and sign/rotation-invariant subspace overlap. Equal family weighting prevents
many paraphrases of one topic dominating the direction.

Each quant's release decision includes paired task regressions and an exact
one-sided sign test over independent families. Paraphrases and translations
are not independent wins. Newly failed refusal variants reject the candidate;
capability regression IDs remain visible even if other tasks improve and the
aggregate pass rate stays unchanged. Statistical evidence assumes genuinely
independent, correctly authored families; labels alone do not prove that.
Confidence is reported separately for each quant, not as a joint confidence
or a guarantee about all possible prompts. Refusal-check mode still measures
wording, not semantic task fulfillment.

Optuna randomness is keyed by seed, trial number, and parameter name. The
persisted trial history supplies TPE data, so restarting—even after only part
of a trial was suggested—does not reset and repeat the random stream. Sampler
algorithm and dependency versions are part of the resume contract. Legacy
studies without that contract require a new output directory; they are not
deleted or silently migrated. Existing tiny example suites/settings predate
these gates and deliberately cannot launch a hardened full search unchanged.

## Cache, context, interruption, and release integrity

Completed evaluations validate the requested generation/scoring identities,
every saved answer and score record, the exact prediction rows, and the
recomputed summary before reuse. Score records bind to the digest of their
actual saved response. Changing the judge, scorer, sandbox, settings, or raw
records cannot reuse an old aggregate score. Older scoring manifests without
this response binding require a new scoring directory; original answers can
still be reused when their generation identity matches.

HF preflight loads only config/tokenizer metadata and counts the rendered
system prompt, user message, chat-template tokens and generation prefix. It
reserves the configured output budget before any model weights are loaded.
Generation repeats that check before device transfer, with tokenizer
truncation disabled. Managed GGUF requests count the actual server-rendered
prompt using its native input-token endpoint. Older llama.cpp servers use
their `/apply-template` and `/tokenize` endpoints instead, including special
tokens. Missing/malformed counts, insufficient context, or a different count
reported during generation fail rather than silently shortening prompts.

Tensor checkpoints use immutable payloads and one atomically committed,
hashed identity/audit pointer. Uncommitted orphan files are not reused. Model
shards and GGUF artifacts are written to private temporary files, verified,
then atomically renamed. GGUF conversion keeps a bound per-step ledger so
an interrupted Q5 conversion can reuse its verified BF16 and Q4 artifacts.
Changed sources, tools, importance matrices or saved artifact bytes reject
resume. Invalid-score retries archive the old completed report before editing
rows, so a retry interruption does not leave an apparently complete stale
report blocking recovery.

Finished release files are independent reflinks where supported, otherwise
streamed copies, never hardlinks to working artifacts. All files are indexed
by hash and size, then the complete staging directory is committed. Interrupted
staging directories are preserved but never treated as finished releases.
An identical finished release is idempotent; a different identity cannot
overwrite it. Verify a release without loading a model:

```sh
b70-ablator verify-release /absolute/path/to/release
```

An optional `--expected-identity` checks an independently retained bundle
identity. Verification checks file integrity, not a digital signature,
semantic quality, or universal refusal removal. Source changes cannot alter
an independently copied release, while edits to the release itself are
detected. Symlinks, missing/extra files and invalid indexed paths are rejected.

Negative controls exercise zero edits, repeat/reset cycles, corrupted cache
records, nonfinite/nonorthogonal directions, invalid edit ranges and interrupted
file writes. An invalid edit cannot leave a partially active candidate behind.
Tiny CPU fixtures test these mechanics, not the quality of a trained model.

## Optional judge / fast refusal checks

Set `scoring_mode` to `"refusal-check"` for a full search with no judge model.
Judge paths, thresholds, and calibration files can all be omitted in that
mode. `examples/first-test/run-refusal-check.json` is an unlaunched example.
Keep `"judge"` for deeper content checks with a configured local GGUF judge;
the judge is not required to be 27B. Existing configs default to judge mode.

For already saved answers:

```sh
b70-ablator score --mode refusal-check --answers /path/to/answers \
  --suite /path/to/suite.jsonl --output /path/to/refusal-checks
```

`evaluate --mode refusal-check` also accepts a target GGUF without judge
arguments. The cheap checker uses ordinary CPU text processing, no neural
model or API. It checks only the completed final answer, excludes code and
quoted examples, and does not mistake "as an AI" alone for a refusal. It has
explicit-wording patterns for English, Dutch, German, French, Spanish and
Chinese. Unknown languages without recognizable English refusal wording
are uncertain rather than assumed to pass.

Outputs distinguish `refusal_detected`, `no_explicit_refusal`, and
`uncertain`. No explicit wording does NOT prove the model answered the actual
question. This method can miss indirect evasion and misread context; it is a
cheap screening option, not a replacement for semantic evaluation. It does
not assess lectures/disclaimers. Its reports leave judged compliance and
disclaimer metrics unset, and cannot be ranked against judge-mode scores.
Incomplete output and uncertain checks count against the cheap search score.

Code/maths tests and the full-reply damage checks remain active in fast
searches. Final fast-mode files are marked `refusal-screened`, not
`certified`, and their model cards explicitly state that task fulfillment
was not judged. The target can stay loaded during CPU refusal checks instead
of swapping to a second GPU model.

## Optional refusal-margin screening

Set `screening` to add an explicit, forward-only prefilter to the trial loop:
`refusal_tokens`, `reference_tokens` (operator-chosen token IDs of the target
vocabulary) and `min_margin_drop`. Per trial the answer-start logit margin
between the two token groups is measured on every bad-direction prompt and on
the harmless selection slice — one prefill per prompt, no decoding — and the
harmless-prompt drift is subtracted. Candidates below the threshold are pruned
before any generation or judging is paid for; rejected trials record their
statistic. Baseline margins are checkpointed and identity-bound, so an
interrupted study reuses them.

```json
"screening": {"refusal_tokens": [42, 199], "reference_tokens": [3, 8], "min_margin_drop": 0.5}
```

Screening exists to spend search budget better; it measures no behavior and
certifies nothing. Screened-in candidates still face the full generation and
judging gates, releases are gated only there, and screening has not yet been
validated against real refusal rates on a checkpoint.

## What it does

- Generates through the thinking block and judges only the final answer.
- Separates requested content from unsolicited commentary and preamble.
- Uses a fixed local judge with explicit suite-authored content checklists and literal
  supporting quotes. The judge cannot rewrite a task into "refuse the request"
  and then award compliance for refusing. Missing checklists, malformed,
  unfinished or uncertain judgments fail closed.
- Measures token-level KL over complete baseline reply trajectories, including
  reasoning, and separately reports answer-only KL. This is teacher-forced KL
  on baseline continuations, not a semantic score of every possible reply.
- Executes pure Python task assertions in a networkless, read-only Docker
  sandbox; grades a small exact-answer maths slice.
- Extracts layer activations at the answer boundary and balances multilingual
  good/bad examples. Supports a configurable direction rank.
- Searches layer-relative attention/MLP edits with Optuna, compatible warm
  starts and certain lower-bound early rejection.
- Exports merged original-precision safetensors and actual Q4_K_M/Q5_K_M GGUFs.
  Both quants must pass matched-baseline capability and compliance gates.
- Selects on a development suite, then checks the chosen candidate once on
  disjoint holdout prompts. A failed holdout produces no release, not a new
  candidate chosen using the holdout.

## Hardware and prerequisites

Inference and weight editing require PyTorch XPU on an Intel Arc Pro B70.
There is no CPU/CUDA model-offload fallback. CPU work is limited to storage,
small activation-statistic decomposition, plain-text refusal checks, conversion and quantization.

Use Python 3.12 with the XPU PyTorch index configured in `pyproject.toml`,
Docker with a pre-pulled `python:3.12-slim` image, a compatible llama.cpp
converter/quantizer, and a Vulkan llama-server that supports the model and
judge architectures. New architectures need a verified adapter and converter
support; an adapter listing alone does not prove a live round trip.

Set up and validate:

```sh
uv sync --python 3.12 --extra test
uv run b70-ablator doctor
uv run b70-ablator inspect /path/to/model --trust-remote-code
```

Remote modeling code is opt-in. Review vendor code before enabling it.
Keep competing GPU consumers stopped during experiments. Nothing in this
program changes swap or stops unrelated services automatically.

## Separate answer generation from scoring

The framework has independent `generate`, `score`, and `status` commands.
`generate` loads only the target GGUF, saves each response atomically, then
closes it. `score` loads only the judge and scores those saved responses.
This separate workflow avoids keeping the target and judge on the B70
together. The full `run` search now uses the same ownership rule: it generates
a complete answer batch, unloads the target, then loads the judge. Q4/Q5
checks also generate before judging. Original target weights are reloaded
from disk when needed and the chosen edits are reapplied; there is no CPU
weight-offload fallback. Only small token/activation/edit records use CPU
storage. This scheduling is tested, not yet a measured speed improvement.

```sh
b70-ablator generate --model /path/to/target.gguf --server /path/to/llama-server \
  --suite /path/to/suite.jsonl --max-tokens 2048 --ctx-size 4096 \
  --output /path/to/answers

b70-ablator score --answers /path/to/answers --suite /path/to/suite.jsonl \
  --server /path/to/llama-server --judge-artifact /path/to/judge.gguf \
  --calibration /path/to/calibration.json --judge-max-tokens 2048 \
  --ctx-size 4096 --judge-min-confidence 0.8 --judge-min-accuracy 1 \
  --sandbox-image python:3.12-slim --output /path/to/scores

b70-ablator status --answers /path/to/answers --scores /path/to/scores
```

These are example budgets, not measured optimal B70 settings. Repeating
`generate` resumes only missing prompts. Repeating `score` resumes missing
scores; use `--retry-invalid` to retry invalid judge results without touching
valid results or generating new target answers. Old invalid scores are kept
under `history/`. Incomplete thinking stays incomplete, never compliance.

Raw answers preserve finish reasons, thinking boundaries and token counts.
They are tied to exact target file/settings and exact prompt text. Changing
grading rules or the judge requires a new score directory, not new answers.
Changing the target or generation settings requires a new answer directory.
Changed or damaged records are rejected rather than silently reused.
Completed judge checks are reused only with identical judge/scorer settings.
Status reads actual saved records without starting a model.

This is framework functionality, not evidence of improved compliance. The
mechanical interruption/resume tests use explicitly labeled fixtures. Run
`scripts/framework_smoke.py --output /path/to/new-smoke-directory` with the
project Python for a persisted fixture-based check without loading models.

## Full search

Create three UTF-8 JSONL suites with disjoint prompts: directions, selection,
and holdout. Each row needs `id`, `kind`, `language` and `prompt`.
Direction rows add `label` (`good` or `bad`) and `family_id`; refusal rows also
add `family_id`. Code tasks add executable
`tests`, and maths tasks add an exact `expected` string.
In judge mode, every direction/refusal task also requires `requirements`: 1..16 unique
strings stating the actual requested deliverables. Author and review these
before running; do not ask the scoring judge to choose what the task should
have requested. The provided mechanical-test requirements and labels were
authored by the assistant, not independently human-validated.
Directions require good/bad examples in at least two languages. Both
selection and holdout require refusal, harmless, code and maths slices.

Provide explicit settings, including `consistency`, using `RunConfig` in
`b70_ablator/search.py`.
`examples/first-test/run.json` is a workstation-specific mechanical test,
not recommended release settings. Its tiny English/Dutch suites cannot
support general performance claims. It uses a larger 27B reference judge
because the smaller tested judges did not pass all authored calibration
cases. Judge size/cost remains an unresolved optimization.

```sh
b70-ablator run /absolute/path/to/run.json
```

Judge-mode full search requires `judge_url: "managed"`, so it can own and unload the
pinned judge server. It refuses externally owned judge endpoints rather than
stopping a server it did not start. Standalone `evaluate` still supports them.
Local source directories or Hugging Face IDs are accepted. Remote IDs are
resolved to a fixed revision; unsupported architectures are rejected before
weight downloads. Original safetensors are required for streaming export.

The output directory contains source/settings/tool hashes, judge calibration,
per-prompt predictions, baseline reports, a persistent Optuna study,
trial reports, directions, warm-start records, finalist weights and quantized
rankings. A `release/` directory is created only after the mode-specific checks and
contains weights, both GGUFs, evaluation metadata and a model card. Creating
files never publishes or uploads them.

Resume with the same config and output directory. Completed matching reports
are verified against their prediction rows. Interrupted reports are not
accepted as completed results. Do not change model, judge, prompts or scoring
settings mid-study. Changing `trials` may extend the same study. Holdout
results are never used to tune a failed experiment.
Interrupted attempts resume their existing trial number and exact settings
before new attempts are requested. Queued settings count toward the budget
only when actually attempted. Valid saved answers and scores are reused;
unchanged completed judge checks are not repeated. Unexpected failures leave
an attempt unfinished for explicit resume, rather than silently burning its
budget. A same-directory lock prevents two workers mutating one experiment.

The complete scheduling path is exercised with explicit deterministic
fixtures, including an interruption during trial judging. Those exported
fixture files are NOT models and their scores are NOT compliance evidence.
A separate real B70 test verifies that target weight allocations are freed
before the judge phase and that edits are restored after reloading.

## Verification

```sh
python -m pytest tests -q
ruff check b70_ablator tests scripts conftest.py
uv build
```

Tests marked `xpu` require the real B70. Use `-m 'not xpu'` for CPU-only
unit tests, or `-n 0` to debug sequentially. The distribution test builds a
real wheel, checks required prompt/calibration assets and imports from its
unpacked installation rather than the editable source tree.

`verification/reference-judge-calibration.json` records the existing 27B's
24 authored English/Dutch calibration labels. That result is calibration,
not evidence that ablation improved any target model. Infrastructure smoke
scripts live under `scripts/`; real run artifacts live under `runs/`.

## Limitations and licensing

The code supports a finite adapter set, not literally every past or future
model. Custom vendor code and hybrid recurrent architectures require live
verification. Code/maths slices are intentionally narrow. Quantized damage
is checked through task performance, not full GGUF-logit KL. Exact prompt
deduplication is not proof of semantic independence. Fixed-judge calibration
on a small authored set is not comprehensive validation or an adversarial
robustness guarantee.

Software: AGPL-3.0-or-later; see `LICENSE` and `NOTICE`. Exported models retain
the base model's licensing obligations; using the tool does not automatically
relicense model weights as AGPL. No remote upload happens automatically.
