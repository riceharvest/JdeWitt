# Alignment-matrix removal from writeback tensors — static experiment

Status: static only. `removal_measured: false`. No serve, no capture, no wave, no
benchmark, no GPU, no generated token. The numbers below are weight algebra,
byte identity, tokenizer strings and a public-checkpoint proxy. They are **not**
a refusal measurement and **not** a capability measurement.

Artifacts:

- `report.json` — the per-tensor and aggregate k-curve (`jdewitt-alignment-matrix-v1`)
- `apply.json` — the GGUF application, both arms, and every static check
- `provenance.json` — repos, revisions and sha256 of the four inputs

## What was measured

The alignment delta of a writeback tensor is

```
dW = W_instruct - W_base                 HF layout [out_features, in_features]
```

for the 48 writeback tensors of Qwen2.5-0.5B (24 layers × `o_proj`, `down_proj`;
these are the residual-write modules named in `docs/edit-surface.md`). An economy
SVD `dW = U S Vt` (computed from the smaller Gram matrix `dW dWᵀ`, which is the
same decomposition) gives the rank-k removal

```
W_k = W_instruct - U[:, :k] S[:k] Vt[:k]
```

so the removed matrix is exactly the truncated part of the alignment delta. For
each tensor and `k ∈ {1, 2, 4, 8}` the report carries:

- `removed_energy_fraction` — `sum(S[:k]²) / sum(S²)`;
- `delta_residual_norm_ratio` — `||dW - dW_k||_F / ||dW||_F`, which must equal
  `sqrt(1 - removed_energy_fraction)`;
- `weight_norm_ratio` — `||W_k||_F / ||W_instruct||_F`, the surviving share of
  the instruct matrix;
- `twin_left_capture` / `twin_right_capture` — the static refusal proxy: the
  share of a public abliterated twin's own delta (`W_twin - W_instruct`) that
  lies inside the removed `U_k` / `V_k` subspace, plus the corresponding
  subspace overlap (mean squared cosine; a random `k`-subspace of width 896
  scores `k/896`).

The applied k is chosen by a pre-registered **static** rule, not by behaviour:
maximise `twin_left_capture + twin_right_capture` subject to
`weight_norm_ratio >= 0.95`. The whole curve is reported below so anyone can
re-pick from the data.

Provenance (full sha256 in `provenance.json`):

| role | repo | revision | sha256 |
|---|---|---|---|
| base | `Qwen/Qwen2.5-0.5B` | `060db6499f32faf8b98477b0a26969ef7d8b9987` | `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342` |
| instruct | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` |
| twin | `Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1` | `a42fe88f75b3ab545466ca5692fef0b4d7bc8009` | `dff81834dbf3fe0bc476eb144f68173eac4feeece41e1337b29e88019b8e2406` |
| quant stand-in | `bartowski/Qwen2.5-0.5B-Instruct-GGUF` | `41ba88dbac95fed2528c92514c131d73eb5a174b` | `0b2e17471ccc784dfd7b042a81e34289a3accb75ccda7347598a329e0a932099` |

## The curve (aggregate over all 48 tensors)

| k | removed energy | delta residual | weight norm ratio | twin capture L | twin capture R | overlap L | overlap R |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4718% | 0.99764 | 1.00014 | 0.4156% | 0.8393% | 0.4155% | 0.8408% |
| 2 | 0.8254% | 0.99586 | 1.00023 | 0.5635% | 1.2151% | 6.4352% | 0.6529% |
| 4 | 1.4558% | 0.99269 | 1.00036 | 0.8803% | 1.6862% | 10.8389% | 0.5565% |
| 8 | 2.6204% | 0.98681 | 1.00060 | 1.2523% | 1.9855% | 7.0563% | 0.5527% |

By kind:

| kind | k | removed energy | delta residual | weight norm ratio | twin capture L | twin capture R |
|---|---:|---:|---:|---:|---:|---:|
| attention `o_proj` | 1 | 1.1025% | 0.99447 | 1.00015 | 0.2524% | 0.9713% |
| attention `o_proj` | 2 | 1.9175% | 0.99037 | 1.00033 | 0.5003% | 1.5984% |
| attention `o_proj` | 4 | 3.3272% | 0.98322 | 1.00064 | 0.8808% | 2.7083% |
| attention `o_proj` | 8 | 5.8795% | 0.97016 | 1.00118 | 1.5374% | 3.7035% |
| FFN `down_proj` | 1 | 0.3819% | 0.99809 | 1.00014 | 0.4491% | 0.8121% |
| FFN `down_proj` | 2 | 0.6698% | 0.99665 | 1.00022 | 0.5765% | 1.1362% |
| FFN `down_proj` | 4 | 1.1891% | 0.99404 | 1.00031 | 0.8801% | 1.4759% |
| FFN `down_proj` | 8 | 2.1561% | 0.98916 | 1.00049 | 1.1936% | 1.6321% |

Selected `k = 8` by the rule above (twin capture 3.24%, weight norm ratio
1.00060, removed energy 2.62%). `k = 1` is the minimal-removal point on the
same curve.

The per-tensor maxima are larger than the aggregates: at `k = 1` the most
concentrated tensor (`layer 0 o_proj`) gives up 2.48% of its delta energy, and
at `k = 8` the most concentrated (`layer 11 o_proj`) gives up 13.35%. The
median tensor at `k = 8` gives up 4.00%. No writeback tensor has a dominant
rank-1 delta.

## Findings

1. **The alignment delta is high-rank.** The top singular direction carries
   only 0.47% of the aggregate delta energy and the top eight carry 2.62%.
   Subtracting the top-k truncation of `dW` therefore removes almost none of
   `dW`: the premise that the instruct-minus-base edit is a low-rank object does
   not hold for this pair at the whole-matrix level. Attention `o_proj` is more
   concentrated than FFN `down_proj` (5.88% vs 2.16% at k=8), but neither is
   close to a rank-1 story.
2. **The public abliterated twin is not captured either.** At k=8 the removed
   subspace holds 1.25% of the twin delta on the left and 1.99% on the right.
   The left overlap (7.06%) is about 8× the `k/896` chance level, and the right
   overlap (0.55%) about 3.4× the `k/4864` chance level — a weak statistical
   echo, not an identification. The static proxy gives no support for "the top
   alignment directions are the refusal-consistent ones". That is a negative
   result and is recorded as one.
3. **The surviving-weight ratio is ~1.0 and is the wrong place to look for a
   capability cost.** `dW` is small next to `W`; removing 2.6% of `dW` moves
   `||W||` by 0.06% (and upward, because the truncated component is nearly
   orthogonal to `W`). The delta residual (`0.9868` at k=8) is the informative
   removal number; neither ratio measures instruction-following.
4. **The 3-bit artifact cannot hold the edit.** In the in-place arm the
   intended delta is at or below the codec grid: realized cos has median 0.255
   (min 0.062, max 0.763) and the realized change is *larger* than intended
   (norm ratio 1.74), i.e. requantization noise dominates the delta. At F16 the
   same delta realizes cos 0.9999 (min 0.9998, SNR 62.1). The algebra is fine;
   the storage type is the limit.

## GGUF application (`apply.json`)

Source: `verification/iq05b/Qwen2.5-0.5B-Instruct-IQ3_M.gguf`
(sha256 `0b2e1747…a932099`, 342,752,576 bytes, 290 tensors). The rank-8 deltas
are the same float32 arrays in both arms.

| arm | tensors | changed blocks | realized cos (median / min / max) | SNR median | untouched tensors bit-identical |
|---|---:|---:|---|---:|---|
| in-place, codec preserved | 27 | 623,846 | 0.255 / 0.062 / 0.763 | 0.57 | 263 / 263 |
| out-of-place, F16 re-encode | 48 | — (type change) | 0.9999 / 0.9998 / 1.0 | 62.1 | 242 / 242 |

The in-place arm leaves the 21 `IQ3_S` `ffn_down` tensors untouched: this
build's pure-numpy IQ3_S encoder is the time bound (minutes per tensor), and
those tensors are covered by the F16 arm instead. The skip is recorded per
tensor in `apply.json` (`skipped_tensors`); the check
`inplace_covers_all_deltas` is therefore `false` and is reported as such.

Static checks:

| check | result |
|---|---|
| removed energy == sum of squared removed singulars | max relative error `5.1e-15` |
| delta residual == sqrt(1 − removed fraction) | max absolute error `1.1e-15` |
| singular values descending in every tensor | true |
| in-place: metadata bytes identical | true |
| in-place: untouched tensors bit-identical | true (263/263) |
| in-place: tokenizer strings identical | true |
| in-place: revert byte-exact | true (restored sha256 == source sha256) |
| in-place: covers all 48 deltas | **false** (21 IQ3_S skipped, see above) |
| F16: untouched tensors bit-identical | true (242/242) |
| F16: untouched claim verified by the writer | true (242 checked, 0 mismatched, 0 missing) |
| F16: tokenizer strings identical | true |
| F16: metadata bytes identical | false (one `jdewitt.fingerprint` KV is added by the writer) |

Revert is the sidecar path: `edit_tensors_inplace(..., revert_path=…)` saves the
pre-patch bytes of every changed tile, `revert_inplace` restores them and
refuses a file that changed since, and the restored sha256 equals the source
sha256 exactly.

A Q6_K re-encode was tried first and rejected: gguf-py's reader refuses a
quantized tensor whose 896-wide row is not a whole number of 256-value blocks,
so the artifact would not be readable by this repo's own diff tooling even
though the flat tensor is block-aligned. F16 has no row constraint and keeps
the artifact inspectable.

## What this does not say

- `removal_measured` is false. No forward pass ran; nothing here observes what
  the model answers. A behavioural claim is out of scope.
- The alignment delta also carries instruction-following. A capability cost is
  expected and is **unmeasured** by this experiment; the weight-norm ratio in
  the table is not that measurement.
- `twin_capture` uses a public abliterated checkpoint as a weight-space proxy.
  That checkpoint's delta also contains its own fine-tuning and quantization,
  and abliteration is not proven to be a pure refusal-removal edit.
- Singular directions are statistical (largest delta energy), not causal. Being
  first in the SVD does not make a direction a refusal direction.
- The curve is the result. `k = 8` is the point the pre-registered static rule
  selected for the application step, not a winner validated by behaviour.

## Reproduce

```bash
# provenance + hashes (public, ungated); --keep retains the bytes for the next step
python scripts/alignment_fetch.py verification/alignment-matrix/provenance.json --dest downloads --keep

# the curve and the rank-k deltas (CPU, ~100 s)
python scripts/alignment_matrix.py \
  --base downloads/base/model.safetensors \
  --instruct downloads/instruct/model.safetensors \
  --twin downloads/twin/model.safetensors \
  --provenance verification/alignment-matrix/provenance.json \
  --out verification/alignment-matrix/report.json \
  --emit-deltas deltas

# the GGUF application and every byte check (CPU, no model)
python scripts/alignment_apply.py \
  --gguf verification/iq05b/Qwen2.5-0.5B-Instruct-IQ3_M.gguf \
  --deltas deltas --work-dir work \
  --report verification/alignment-matrix/apply.json
```

Tests: `pytest tests/test_alignment.py`.
