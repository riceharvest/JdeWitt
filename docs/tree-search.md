# Experimental J-space prefill tree

This is a separate experiment, not another static projection method and not a
certified release path. It implements multi-step branching over interventions,
with real layer-segment forwards before pruning. No pretrained-model quality or
speed advantage has been measured.

The optional [behavior trajectory ranker](behavior-ranker.md) adds independent
full-answer supervision, same-prompt preference learning, separate recognition
and refusal/evasion/broken-output predictions, and held-out false-confidence
auditing. That extension and its integration were implemented under an explicit
**do not test** instruction: previous tree test results do not verify the current
changes. No extension training, inference, tests or package rebuild was run.

## What actually runs

1. Render the existing deterministic chat template and reserve the full output
   budget. The target and original-precision weights stay on the B70.
2. Embed the prompt and compute the shared prefix once, up to the first configured
   checkpoint. Checkpoints are **zero-based block outputs**, as in upstream jlens.
3. At each checkpoint, branch over no edit and every named direction/strength.
   Project only the **last prompt position**. Keep the rest of the sequence.
4. Advance each child through the actual next layer segment without a cache.
   Score its transported last-position state through `J_l`, the model's real
   final normalization and selected unembedding rows. At the final layer use the
   actual output readout (`J = I`), not a fitted final-layer approximation.
5. Reject protected-token-distribution/norm regressions beyond the explicit
   limits, retain a bounded beam and repeat. A temporarily worse branch can win
   after a later pivot; the tests explicitly distinguish this from greedy steering.
6. Commit the selected trajectories and every evaluated branch score **before**
   decoding. Replay each finalist's exact edits on its first prefill, then let
   ordinary cached generation continue. Decode an unedited baseline under the
   same output budget and refusal floor. Normally only finalists and the baseline
   decode; explicit behavior collection can additionally decode all finite visited
   branch prefixes under its separate declared budget.

This first implementation adapts **within the prefill**, not on every generated
reasoning token. Its chosen paths are prompt-dependent runtime hooks. They cannot
be exported as a normal fixed-weight GGUF, and this command never claims to do so.
There is no shared divergent KV/recurrent cache: each segment processes the full
prompt sequence, and finalist generation has its own cache. Replaying finalists
recomputes their prefill; it does not reuse the search's partial states for decode.

## Inputs you must choose before a live test

No target, token groups, search budgets or damage limits are guessed. Start from
`examples/tree-search/config.template.json` and replace its nulls. Use a new output
directory for a changed experiment.

- `model_identity`: exact model identifier (or absolute local safetensors
  directory), revision, Transformers architecture, layer count and hidden width.
  Remote models require a full immutable revision SHA at execution. Local model
  weight/tokenizer/metadata files are content-hashed for resume.
- `dtype`: `bfloat16` or `float16`, original precision. No NF4/CPU inference fallback.
- `lens_manifest`: a bound, fitted **target-specific** upstream Jacobian lens.
- `suite`, `suite_role: "development"`: ordinary existing JSONL suite schema.
  Do not use the sealed holdout for iterative tree development. Marked holdout,
  test and certification rows are rejected.
- `ctx_size`, `max_tokens`, `refusal_floor`: full generation reserve and optional
  refusal floor (`null` means none). The floor applies only to refusal tasks and
  is identical for the baseline and finalists.
- `checkpoints`: strictly increasing zero-based block-output indices, at least
  three, ending at `layers - 1`. All earlier checkpoints need fitted J matrices.
- `beam_width`, `finalists`, `strengths`, `max_expansions`: explicit complete
  search budget. No edit is always included separately. An insufficient expansion
  limit fails before the first forward, rather than silently truncating actions.
- `max_state_bytes`: conservative saved-state bound `(2*beam_width + 16)` times
  one full-sequence residual tensor. This is **not** a total VRAM guarantee.
- `max_lens_bytes`: resident fp32 Jacobian/readout/direction bound. Learned-bank
  directions conservatively reserve full rank. Model weights, attention/MLP
  workspace, masks, rotary tensors, allocator overhead and decoder caches need
  **additional** room. No automatic occupancy/VRAM settings are chosen.
- `refusal_tokens`, `reference_tokens`, `protected_tokens`: explicit integer IDs
  in the target tokenizer, distinct/disjoint groups, at least two protected IDs.
  These are selected-token margins/KL, **not** full-vocabulary probabilities.
- `actions`: named token-ID groups to construct distinct rank-one Jacobian
  pullbacks, or learned-bank references described below. Candidate pullbacks use
  the effective RMS gain, including Qwen3.5's `1 + weight` convention. They are
  fixed local-linear candidate directions, not a proven causal refusal basis.
- `max_protected_kl`, `max_norm_change`, `drift_weight`: explicit finite
  nonnegative guards/weight. Objective is current refusal-margin change from the
  unedited state plus `drift_weight * protected_KL`, minimized. Different depths'
  scores are not summed as though they had interchangeable scales.

## Bind an existing upstream lens

Upstream source/format: https://github.com/anthropics/jacobian-lens

The upstream `JacobianLens.save()` format is a safe tensor dictionary with `J`,
`d_model`, `source_layers`, and `n_prompts`. This project reads it with
`weights_only=True`, memory-mapped on the CPU, then sends selected fitted matrices
to the B70. It does **not** require installing jlens merely to apply a saved lens.
Obtaining/fitting a lens remains a prerequisite; this command does not fabricate
one or quietly replace it with a logit lens.

Save the `model_identity` object in a separate JSON file, then run:

    b70-ablator bind-tree-lens /absolute/path/lens.pt \
      --model-identity /absolute/path/model-identity.json \
      --output /absolute/path/lens-manifest.json

Binding performs no model inference and checks actual tensor shapes/finiteness.
**Upstream files carry no model ID.** Binding is explicitly an operator attestation
of their fitting origin, not mathematical proof of model correspondence. Do not
bind an unrelated lens just because its hidden width matches.

## Reuse independently fitted causal directions

Token pullbacks need not be the intervention choices. To use mean-difference,
SVD or other independently fitted orthonormal bases, optionally add
`direction_bank` to the configuration. Its safe `torch.save` payload is:

    {
      "model_identity": <exact identity object>,
      "actions": {
        "mean-difference": {<checkpoint index>: <Tensor[rank, hidden]>, ...},
        "alternative-subspace": {<checkpoint index>: <Tensor[rank, hidden]>, ...}
      }
    }

Every named basis must cover exactly the nonterminal checkpoints, with finite
orthonormal rows. The payload contains directions, never full model weights.
Reference it in `actions`, for example:

    "actions": {
      "learned-refusal": {"bank": "mean-difference"},
      "other-subspace": {"bank": "alternative-subspace"},
      "token-pullback": [<explicit target-token IDs>]
    }

The bank's model identity and file hash are checked. Nothing is automatically
fitted from the evaluation suite, and no synthetic labels are generated.

## Validate, run and score later

    b70-ablator validate-tree /absolute/path/tree.json
    b70-ablator tree-search /absolute/path/tree.json

`validate-tree` imports no torch/Transformers runtime, initializes no device,
checks file identities/metadata/budgets, and creates no outputs. It deliberately
reports tensor contents, device fit/parity and proxy calibration as unchecked.
`tree-search` is the explicit B70 inference command; do not run it while the
GPU is reserved.

Output layout:

- `trees/*.json`: hashed task/experiment-bound branch traces, selected paths,
  proxy metrics, expansions, saved-state bound and synchronized search wall time.
- `answers/baseline/`, `answers/rank-0/`, ...: normal immutable AnswerStores, usable
  with the existing separate `score` command. Rank is the proxy rank, not an
  independently judged outcome. Baseline is always generated independently.
- `runtime.json`: bound tokenizer/model/generation fingerprint.
- `progress.json`, `report.json`: incremental progress and a sealed final
  **experimental-unvalidated** report, never a release certificate.

An interrupted run reuses committed searches and completed answers. It reruns
only missing decodes, checks saved paths and fingerprints, and rejects changed
input/code/dependency identities. A completed job validates its artifacts and
returns without loading a model. It does not choose a different runner-up after
seeing held-out outcomes.

## Limits and next live checks

Explicit segment layouts currently cover Llama, Qwen3/Qwen3-MoE and Qwen3.5
text/VLM text-backbones (including MoE). Text-only input is required. Other
architectures are rejected; the general static editor's wider catalogue does not
imply tree support. These are source-matched adapter implementations, **not**
pretrained-checkpoint B70 certifications.

CPU tests use labeled algebra/scheduling fixtures. Actual B70 tests have been
added for native Llama, Qwen3 and hybrid Qwen3.5 segment/logit and replay parity,
but are left pending while the GPU is reserved. No CPU model inference is used
as a substitute.

Before treating this as useful, run B70 parity, verify target-specific lens
provenance, calibrate scores on both edited and unedited states, and judge full
answers plus matched benign/code/math controls. Compare at matched total compute,
including the lens, tree, baseline and finalist decoding costs. Protected-token
KL and residual norms are only cheap damage screens. Suppressing internal
"cannot" narration can leave refusal behavior intact; neither a low margin nor
a green fixture suite proves useful completion or preserved capability.
