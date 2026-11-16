# Agent operations and method inventory

This is the existing private B70 tool, not a wrapper around the installed Heretic CLI.
No remote model upload, publishing, offloading, or automatic parameter selection is introduced.

## Agent contract

- Successful commands emit JSON on stdout; help is plain text. Diagnostics from third-party model loaders may still appear on stderr.
- Expected command failures emit `{"error":{"type":"...","message":"...","command":"..."}}` on stderr and exit 1. Argument errors emit an `ArgumentError` object and exit 2.
- `methods`, `architectures`, `--help`, and `run-status` do not import PyTorch, Transformers or bitsandbytes.
- `validate-run CONFIG` validates the actual `RunConfig`, authored split/coverage rules, and local input/tool file presence. It never starts a model, runs Docker, contacts a remote model host, or creates the output directory. It imports Optuna through the current configuration module.
- `run-status OUTPUT` opens the Optuna SQLite database read-only. It does not create a study or migrate a database. `RUNNING` means an unfinished resumable attempt, not proof of a currently running process. Release presence is not release verification.
- `inspect MODEL` reads config metadata. It can require network access and explicitly authorized remote config code; it does not load model weights.
- `doctor` really runs an XPU kernel on the B70. It never substitutes another device.

Typical read-only commands:

```sh
b70-ablator methods
b70-ablator architectures
b70-ablator validate-run /absolute/path/run.json
b70-ablator run-status /absolute/path/experiment
b70-ablator verify-release /absolute/path/experiment/release
```

Validation is not execution: tokenized context budgets, GPU memory, kernels, converter architecture support, and answer quality are checked separately during the real workflow. The old first-test configs intentionally lack hardened consistency settings; they are not launch-ready presets.

## Implemented search methods

Each of these is reachable through the `method` field of `b70-ablator run CONFIG`, direction fitting, the resident editor, streaming safetensors export, and existing release gates. The seven direction methods and dense ARA optimizer have small actual-B70 operator tests. These tests do not establish refusal-removal quality on a real pretrained checkpoint.

| Method | Operator | Required method_options |
| --- | --- | --- |
| `directional` | Language-balanced mean contrasts, orthogonal residual projection | none |
| `svd` | Matched-family contrast subspace | none |
| `projected` | Contrast component orthogonal to the harmless centroid | none |
| `mpoa` | Projected contrast with full weight-row norm restoration by default | none |
| `whitened-svd` | Thin-SVD ridge-whitened contrast subspace | positive `ridge` |
| `som` | Deterministic 1-D SOM fitted to harmful family means; subtract harmless centroid, orthonormalize | positive integer `som_steps`; `som_learning_rate` in (0,1] |
| `leace` | Ridge-regularized, binary linear covariance erasure | positive `ridge`; rank must be 1 |
| `ara` | Bounded KNN low-rank optimization of captured module I/O, followed by layer-strength search | positive `steps`, `learning_rate`, `neighbors`, `preserve`, `steer`, `repulsion_cap`; nonnegative `overcorrect` |

Common explicit options are `row_normalization` (`none`, `pre`, `full`), optional `winsorization_quantile` in (0,1], and the paired positive `massive_threshold`/`massive_ratio`. Unknown keys and missing required method options fail before loading weights. No optimization knobs are chosen for the user.

Limitations matter:

- SOM is a deterministic 1-D implementation, not a reproduction of every topology/experiment in the paper.
- LEACE uses the linear map without affine mean restoration; row normalization is rejected because it would change that erasure map. Binary ridge erasure is not a guarantee of semantic refusal removal.
- ARA currently needs dense projections and real module-I/O captures. Fused expert route-I/O is not implemented, and those models fail instead of silently skipping experts. It does not claim exact reproduction of every upstream ARA variation.
- ARA fitting uses captured outputs and two small low-rank products: it does not copy the original projection into FP32 or materialize a full delta matrix at every loss evaluation. The subsequent search/export still uses the rounded merged operator.
- Norm restoration, outlier protection, layer profiles and larger ranks are variants/options, not separate algorithms.

## SFT and DPO

`b70-ablator train CONFIG` trains low-rank edit factors on the B70 while freezing the original model. SFT masks prompt positions and optimizes completion likelihood. DPO uses explicit chosen/rejected completions and the unedited model as the reference. It does not hold a second reference model in VRAM.

The configuration requires exactly these fields:

`mode`, `model`, `revision`, `rank`, `ctx_size`, `max_tokens`, `dtype`, `trust_remote_code`, `data`, `holdout`, `steps`, `learning_rate`, `beta`, `seed`, `output`.

`mode` is `sft` or `dpo`; the source must be a reviewed local original-safetensors model. `revision` may be null for local files. Use explicit positive integer budgets/rank/steps, positive finite learning rate and beta, an integer seed in [0,2**32), and `float16` or `bfloat16`. There are no selected training defaults. Data rows use the ordinary task schema plus `completion` for SFT or `chosen`/`rejected` for DPO. The separate holdout is checked for overlap but is NOT evaluated by training. A disjoint file alone is not an evaluation result.

Training output is immutable and marked `trained-unvalidated`. It contains a training identity, incremental progress, losses, merged HF files and tensor-level export verification. It is not an automatic certified release and does not yet support interrupted optimizer-state resume. Dense original-precision projections only; training activation/gradient memory further reduces the usable model-size envelope. Actual B70 tests use tiny random Llama fixtures, not a large pretrained refusal benchmark.

## Explicitly not implemented

The `methods` catalogue separates researched entries from implemented paths. RDO/concept-cone optimization, SAE feature erasure and router pre-bias remain unimplemented. They are not aliases for SVD, ARA or SFT, and cannot be enabled by choosing those names. Runtime steering, head/neuron interventions, optimal-transport ablation, GRP-style objectives, and distillation are additional research families, not currently supported export/search modes. Prompt jailbreaks are not permanent weight-edit methods.

This finite inventory is not a claim that every possible refusal-removal method has been included.

## Sources

- Directional ablation: https://arxiv.org/abs/2406.11717
- Projected ablation: https://huggingface.co/blog/grimjim/projected-abliteration
- MPOA: https://github.com/p-e-w/heretic/pull/52
- SOM: https://arxiv.org/abs/2511.08379 and https://github.com/p-e-w/heretic/pull/196
- LEACE: https://arxiv.org/abs/2306.03819
- ARA: https://github.com/p-e-w/heretic/pull/332 and https://github.com/p-e-w/heretic/pull/400
- DPO: https://arxiv.org/abs/2305.18290
- RDO/concept cones: https://arxiv.org/abs/2502.17420
- Router pre-bias proposal: https://github.com/p-e-w/heretic/issues/322
- Broader method comparison: https://github.com/elder-plinius/OBLITERATUS
- Optimal transport discussion: https://github.com/p-e-w/heretic/issues/217
- GRP discussion: https://github.com/p-e-w/heretic/issues/184
