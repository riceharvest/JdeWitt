# Spark-X2.6-1.7B Q4_K_M — phase-6 fitted-plan release

This directory is the Phase 6 delivery unit: **one fitted plan, the adapter fitted
on it, the merged twin built from the same plan, and the plan card that lets
anyone reproduce the edit on their own copy of the base.** It is not a
pre-quantized model release and it is not a certification.

## What ships

| file | what it is |
|---|---|
| `adapter.gguf` | rank-1 GGUF LoRA adapter (36 tensors, layers 10–27), bound to the base sha256 |
| `model-merged.gguf` | the merged twin: the same plan's deltas applied to the base bytes (36 edited / 190 untouched tensors) |
| `plan-card.tar` | digest-pinned plan bundle (signed meta + payload) for the fitted plan |
| `plan-card.json` | manifest of that bundle, shipped beside it so `verify-artifact` can check it without unpacking |
| `adapter.json` | R8 adapter card: what the adapter curve is and is not |
| `twin.json` | codec-realized record of the shipped twin + behaviour block with pinned argv |
| `evaluation.json` | the measured behaviour record (GATED, probe-level) |
| `README.md` | this file |
| `files.json` | release inventory + `release_identity` |

Base artifact: `Spark-X2.6-1.7B-Q4_K_M.gguf`, sha256
`80ff5222ae51de055fe2e86b1ec4f6784c2f9db704ed3b043b97ce900d3a75ad`,
1,107,458,592 bytes, arch `spark2_5`, 226 tensors.

## The adapter curve is not the merged bytes (R8)

`adapter_delta_br` **0.2523** and `merged_delta_br` **0.1525** are different
measurements of different objects. The adapter card records
`adapter_to_merged_ratio` **0.604** (< the 0.7 floor), so
`ranking_valid_on_this_artifact` is **false**: on this artifact the adapter
overstates the merged edit and is **screening-only**. Never quote the adapter
curve as the edit's effect.

Both behaviour numbers are **GATED** and probe-level: 4 harmful rows (one per
family), greedy decoding, 200 tokens per row, one pinned serve config per arm
(`-c 4096 -ngl 999 --parallel 1 --device Vulkan1 --split-mode none --jinja`,
llama-server build 11690). On those rows the adapter and the merged twin agree
on 1/4 and disagree on 3/4; this is a measured divergence, not an averaged
result. No certification wave ran and no vLLM/SGLang claim is made. The exact
merged-arm argv is pinned in `evaluation.json` and copied into `twin.json`.

## Codec-realized numbers for the shipped bytes

`cos`/SNR below name `model-merged.gguf` by sha256 and were measured on CPU
(`ggufio.compare_edit`, full-tensor decode) against the pristine base. They are
requant fidelity — facts about these bytes, not about answers.

| edited tensors | count | cos min/median/max | SNR min/median/max |
|---|---|---|---|
| Q4_K | 27 | 0.2152 / 0.3689 / 0.4717 | 0.942 / 0.993 / 1.046 |
| Q6_K | 9 | 0.8118 / 0.8487 / 0.8858 | 1.568 / 1.753 / 2.013 |

The low Q4_K cos is the codec, not a missing edit: a unit axis at strength 1.0
removes a small projection from a 4-bit grid. The delivered edit is recorded
per tensor in `twin.json`.

## Reproduce the edit without our search

```sh
jdewitt plan-unbundle plan-card.tar --into /tmp/reg
jdewitt plan-apply --bundle plan-card.tar \
  --gguf Spark-X2.6-1.7B-Q4_K_M.gguf --out /tmp/repro.gguf --edit-type preserve
```

The card is a footprint plan: basis rows map to layers 10–27, rank 1, width
2048. `plan-apply` verifies the bundle, checks the artifact's own geometry,
discovers the residual-write tensors from the file, and edits only those.
`--edit-type preserve` keeps each tensor's source codec, which is how the
shipped twin was written; `--edit-type auto` picks the smallest type that clears
the cos floor (on this base it walks the ladder to F16, a different codec, not a
different edit).

## Verify the directory

```sh
jdewitt verify-artifact --all .
jdewitt verify-release .
jdewitt release-verify-signature . --public /path/to/release-signing-key.pub.pem
```

`files.json` pins the sha256 of every file above and the `release_identity`
= digest(evaluation record, this README). `signature.json` (written after
assembly) signs the reproduction index; the public key is published separately —
only its sha256 belongs in a result line.
