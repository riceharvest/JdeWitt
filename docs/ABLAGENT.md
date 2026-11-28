# Ablagent (Blackfrost-AI) — what it is, what we borrowed, what we do better

Pinned audit: `Blackfrost-AI/Ablagent` @ `19a5987b34e53fa972352dc1c314548cf7ccec81`
(2026-11-27, v0.1.0, Apache-2.0). Re-fetch before quoting; the repo is an alpha
research preview.

## What it is

A research runtime and an interchange protocol (DWMIP v0) for Domain Weight
Modification, with abliteration/refusal-direction research as its originating
application. Two executors:

- a **synthetic** offline pipeline with synthetic banks and mock fixtures;
- a **real CPU executor** for small dense HuggingFace safetensors checkpoints
  (≤ ~900 MiB float32 editable surface, llama/qwen/mistral/gemma-style, no Conv1D
  families) that does S0 intake → S1 baseline → S2 capture → S3 rank-one bake →
  HF export, and records S4 quant, S5 serve, S6/S7 behaviour+capability, SR
  red-line and S10 delivery as **skipped**.

Its reference operation is the same rank-one update ours uses:

    W' = W - alpha * d (d^T W)

with `W` a declared dense `out x in` writer, `d` unit length in the declared
output space, `alpha` in (0, 1]. Its compiled artifact is a manifest carrying a
`BoundDirection` (`unit_vector`, `shape`, `tensor_name`, `writer_id`,
`baseline_digest`, `graph_digest`) and the `InterventionSpec` (`alpha`,
`algorithm_id=dense.rank_one_output.v0`). Every real run ends `not_eligible`;
efficacy and safety are `unmeasured`, and the README says quantization, serving
and behaviour verification are not implemented.

## Where jdewitt is the better tool for the actual goal

The goal is an edit that lands in the artifact the user already serves.

| Axis | Ablagent v0 | jdewitt |
|---|---|---|
| Artifact | full-precision dense safetensors | **the user's quantized GGUF**; only edited tensors are re-encoded, every other tensor is copied byte-for-byte and re-read |
| Codec honesty | S4 quant skipped | `preview_edit` cos/SNR per tensor **before** a write, full-footprint read-back after, `CODEC_FLOOR` refusal/rollback |
| Serving | S5 skipped | `board` / capability waves at a pinned serve argv; removal vs retention measured separately |
| Behaviour claim | explicitly `unmeasured`, `not_eligible` | same boundary, plus an executable evidence path (`edit`/`route` → `board` → cert packet) |
| Architecture reach | dense `out x in` writers, one site | discovery from the module tree and the artifact's tensor table; any output-side writeback, per-kind strength |
| Reversibility | separate exported copy | rewrite-with-untouched-bytes, in-place atomic replace with a pre-write revert log or whole-file sidecar |
| Agent interface | CLI boundary + JSON manifests; no autonomous executor | `edit` one-shot and `route` six-stage state machine, one JSON per stage, resumable state, stale-input refusals, every branch a command |
| Token cost | capture/baseline runs on CPU | axis selection is prefill-only in `tune`; the route never generates a token |

What Ablagent does better than we did before this audit: it makes the
architecture contract explicit (graph, site, space, writer), it keeps rejected
trials and skips in the lineage instead of printing only winners, and it has a
typed interchange document for a direction. Those are real ideas, not slideware.

## What we adopted

1. **Imported axes are first-class.** `jdewitt edit --axes FILE` and
   `jdewitt route --axes FILE` apply an externally recovered direction through
   the same discovery and the same full-footprint `CODEC_FLOOR` gate as a fitted
   axis. A `[D]` axis broadcasts over the artifact's writeback layers; a `[L, D]`
   axis is per-layer; zero rows mean "do not touch this layer".
2. **Per-tensor targeting for sparse axes.** An axis file may carry
   `kinds: ["attn_output"]` so a one-writer direction edits exactly that tensor
   (Ablagent's site), not every writeback projection in its layer.
3. **Direction provenance travels with the axis.** The `.pt` payload carries the
   source manifest digest, direction digest, alpha, writer, space and the
   source run's skipped stages. `route plan` prints it; `verify`/`certify` keep
   `removal_measured: false`.
4. **`jdewitt ablagent-import`** is the seam: it reads one compiled manifest,
   validates it fail-closed, maps the HF writer to `blk.N.<kind>.weight`, checks
   width/layer/hidden against `--gguf`, and writes the axis file the route
   consumes.

## The import contract

    jdewitt ablagent-import RUN_OR_MANIFEST --gguf ARTIFACT --out AXIS.pt
    jdewitt route --gguf ARTIFACT --axes AXIS.pt --scale <alpha> survey
    jdewitt route --workdir DIR plan      # preview per storage type, cost, risks
    jdewitt route --workdir DIR gate      # trial write + FULL-footprint read-back
    jdewitt route --workdir DIR apply --out EDITED.gguf

Mapped: `model.layers.N.self_attn.o_proj.weight → blk.N.attn_output.weight`,
`model.layers.N.mlp.down_proj.weight → blk.N.ffn_down.weight`, plus direct
`o_proj/out_proj/down_proj/w2/ssm_out/attn_output` suffixes. The direction is
applied at `--scale alpha` with the same `W - scale d (d^T W)` formula; the axis
file also records `scale: alpha` so the printed next commands carry it. The
default output is `./ablagent-axis-<manifest8>.pt`; it is never written inside
the source run tree.

Refused, with a typed code (never repaired): unreadable manifest
(`ABLAGENT_INPUT`), non-DWMIP or non-dense-rank-one document
(`ABLAGENT_NOT_COMPILED`), non-unit vector (`DIRECTION_NOT_UNIT`), input-space
vector (`DIRECTION_SPACE`), alpha outside (0, 1] (`ALPHA_OUT_OF_RANGE`), writer
with no GGUF counterpart (`UNMAPPED_WRITER`), width/layer/hidden mismatch
(`GEOMETRY_MISMATCH`).

## Boundaries that stay written down

- Importing a direction is not importing evidence. Ablagent's skipped S4–S7 and
  `not_eligible` release remain `not_eligible` here; only the codec gate and the
  untouched-byte checks are measured by this path.
- The direction is bound to Ablagent's baseline digest; the import checks the
  artifact's architecture/hidden/layer geometry, **not** that the artifact was
  converted from that exact checkpoint. The risk is reported as
  `BASE_IDENTITY_UNPROVEN`.
- `ffn_down` directions carry a `BASIS_FOLDING` risk: some converters (the
  dspark fork on qwen) Hadamard-fold FFN tensors, so an HF-basis direction has
  the wrong functional there. `attn_output`/`o_proj` is not folded.
- One writer, one layer, one alpha. Ablagent v0 does not compose directions, and
  neither does this import: use `--rank`/`--layers` on a fit if you want a
  footprint, not a bag of imported vectors.
