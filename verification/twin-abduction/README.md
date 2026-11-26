# Twin abduction — the refusal axis read out of a published abliteration (Qwen2.5-0.5B)

A rank-1 abliteration of a base leaves, on every writeback tensor, exactly

    dW = W_twin - W_base = -a r (W^T r)^T

so the edit *is* a rank-1 matrix whose left singular vector is the refusal
direction `r`. Nothing is fitted to activations here: the axis is read out of
`dW = twin - base` with an SVD, and the published edit is the answer to the fit.

Scope is weight algebra only. No model is loaded, no forward pass, no
generation, no GPU, no serve, no benchmark. `removal_measured: false` in the
record: nothing here measures whether the refusal behaviour went away, and a
behavioural claim would need a served model — out of scope for this card.

## Sources (recorded in `qwen05b.json` → `sources`)

| | repo | revision | `model.safetensors` sha256 | bytes |
|---|---|---|---|---|
| base | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` | 988,097,824 |
| twin | `Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1` | `a42fe88f75b3ab545466ca5692fef0b4d7bc8009` | `dff81834dbf3fe0bc476eb144f68173eac4feeece41e1337b29e88019b8e2406` | 1,260,367,448 |

Both are public and ungated (`gated: false`), one safetensors each, one epoch
(`model.safetensors`, no shards). Both revisions are the repo HEAD at the time
of the run. 24 layers, hidden 896, intermediate 4864, bf16 on disk.

The twin is 272.3 MB larger because it carries an explicit `lm_head.weight` the
tied base does not. That head turns out to be an exact, unedited copy of the
**base** embedding (see below), not a re-tie of the twin's edited embedding.

## What the edit touched (`edit_footprint`)

49 of 290 shared tensors differ at byte level: the 48 writeback tensors
(`self_attn.o_proj` and `mlp.down_proj` in all 24 layers) plus
`model.embed_tokens.weight`. Everything else — every layernorm, every
`q/k/v` projection, `gate_proj`/`up_proj`, all biases — is byte-identical to the
base. `lm_head.weight` is present only in the twin.

Relative delta norm per tensor is 0.0288–0.0454; that is the bf16 storage floor
of the twin, not the edit strength (the edit's own magnitude is the singular
value below).

## Per-writeback-tensor SVD (`per_tensor`, `per_tensor_fit`)

For each of the 48 tensors, `dW` was formed in float32 and SVD'd.

- **rank-1 share (energy, `s1²/Σs²`): 0.99703 min, 0.99796 median, 0.99879 max.**
  The residual 3.50–5.49 % (mean 4.50 %) is the bf16 rounding floor of the twin:
  the next singular value is 0.4–0.85 % of the first, which is exactly what
  bf16 round-to-nearest on a 0.035-relative perturbation produces. Note
  `rank1_share_sv = s1/Σs` is *not* a rank-1 measure for a 896×4864 matrix (the
  long spectrum dilutes it to ~0.41–0.55); the energy share is the one to read.
- **left vector `u1` vs the shared axis: mean |cos| 0.9999988, min signed
  0.9999956, max 0.9999997.** Sign fixed so the largest |entry| of `u1` is
  positive; every one of the 48 is on the same side, no flips.
- **right vector `v1` vs `W^T r`: cos −0.9999904 on average** (max |1−|cos|| ≈
  1e-5). The minus sign is the formula's, not a defect: with `u1 = +r` the
  pairing `dW = -a r (W^T r)^T` forces `v1 = -unit(W^T r)`.
- **strength `a = s1/‖W^T r‖`: 1.00181 ± 0.00014** (min 1.00156, max 1.00211)
  across all 48 tensors, attention and MLP alike — i.e. the published edit is a
  unit-strength rank-1 subtraction with **one** shared direction, and the +0.18 %
  offset is the bf16 rounding bias of the twin, identical on every tensor.

`strength_ls_shared_axis` in the record is `s1` itself (the least-squares
coefficient *after* normalising `W^T r`), which varies 0.42–1.51 because
`‖W^T r‖` varies per tensor; `strength_abl_form` is the `a` above and is flat.
Both are recorded so the two readings cannot be confused.

## The axis (`weight_algebra`, `axis.npy`)

The shared axis is the top right singular vector of the 48×896 stack of the
per-tensor left vectors.

- unit norm: **1.0000000** (`axis_is_unit_norm: true`; the saved float32 npy
  reads back at 0.99999994)
- energy share of that top component in the stacked matrix: **0.9999976**
- per-module axes: `attn_output` share 0.9999963, `ffn_down` 0.9999990, both at
  cos **0.99999995** to the shared axis — attention-writeback and MLP-writeback
  were ablated along the same direction, to within the rounding floor
- layer-to-layer cos table (`layer_cos_table_u1`, all 2256 ordered pairs of the
  48 `u1`s): **min 0.9999910**, max 1.0; same-module min 0.9999910,
  cross-module min 0.9999940. No layer disagrees about the direction.

`axis.npy` is `float32[896]`, the unit axis, saved from the same run.

## The two extra sites (`extra_edit_sites`)

- `model.embed_tokens.weight` (151936×896): the delta is **also rank-1 on the
  same axis** — energy share 0.99768, right singular vector cos **0.9999989** to
  the axis, strength 1.00159, residual 4.84 %. So the twin removed the direction
  from the embedding rows too, at the same unit strength. For a (vocab, hidden)
  tensor the axis is the *right* singular vector; the left one is `W r`, and its
  cos to `W_base r` is −0.99999 (sign pairing again).
- `lm_head.weight`: **delta exactly zero** against the base embedding — the
  twin's untied head is a byte-identical copy of the *unedited* base embedding.
  The output projection therefore still carries the direction the twin removed
  from everything else. `lm_head` is not in the delta GGUF for that reason (and
  the stand-in quant below has no `output.weight` at all — it is tied).

## Non-weight differences in the twin repo (`repo_diff_beside_weights`)

The twin repo is not just a weight edit, so the record names what else changed:

- `config.json`: `sliding_window` 32768 → `null` (both files say `use_sliding_window: false`,
  so no attention window changes), `transformers_version` 4.43.1 → 4.46.2, `_name_or_path` added.
- tokenizer: the vocab mapping is identical (151,643 entries, same ids) and the merge table
  is identical pair-for-pair once serialization is normalised (the base writes `"a b"`
  strings, the twin writes `["a", "b"]`, same 151,387 merges in the same order). A
  `padding` block was added (`BatchLongest`, left, `pad_id` 151645).
- `chat_template` differs: 2,507 → 4,433 chars, sha256 `cd8e9439f057…` → `98640c31070c…`;
  `pad_token` `<|endoftext|>` → `<|im_end|>`.

So a behavioural A/B against this twin would carry the prompt template as a second
variable on top of the weight edit. That is one more reason this card stays on weights.

## Checks (`checks`)

| check | value |
|---|---|
| `axis_unit_norm` / `axis_is_unit_norm` | 1.0 / true |
| `all_rank1_energy_gt_0.99` | true (min 0.99703) |
| `writeback_tensors` / `writeback_tensors_edited` | 48 / 48 |
| `mean_abs_cos_u1_to_axis`, `min_signed` | 0.9999988, 0.9999956 |
| `mean_residual_frac_shared_axis` | 0.04498 |
| `sources_unmodified` / `extraction_only` | both true (sha256 recomputed after the read pass equals the recorded hash) |
| `delta_gguf_byte_identical` | true (48/48 tensors read back byte-for-byte from the packed file) |

Nothing in this directory was edited by the extraction; the only writes are the
new files listed below. `removal_measured` is false and no file here claims a
behavioural effect.

## Delta GGUF (`delta_gguf`)

`verification/twin-abduction/qwen05b-delta.gguf` — 48 F32 tensors named exactly
like the target tensors (`blk.L.attn_output.weight`, `blk.L.ffn_down.weight`),
packed with `gguf-delta-pack` from the per-tensor `dW` npys:

```
python -m jdewitt.cli gguf-delta-pack verification/twin-abduction/qwen05b-delta.gguf \
  blk.0.attn_output.weight=<npy> ... (48 NAME=DELTA.NPY specs)
```

size 495,455,264 B, sha256 `77e3108d35c805dda46ec0f4606bb584f48a60034a56927b2b52cacb730b38b4`.
Every packed tensor was read back from the file at its `data_base + offset` and
compared to the npy bytes: 48/48 identical. Names and dims were checked against
the stand-in quant
`verification/iq05b/Qwen2.5-0.5B-Instruct-IQ3_M.gguf`: 48/48 present, 0 dim
mismatches (its `blk.L.attn_output.weight` is Q8_0 `[896,896]`,
`blk.L.ffn_down.weight` is `[4864,896]`, and it has no `output.weight`).

The sigma file is regenerable and 473 MB, so it is not committed (`*.gguf` is in
`.gitignore`); its path and sha256 are in the record.

## Reproduce

```
cd <worktree>
OPENBLAS_NUM_THREADS=1 python3 scripts/twin_abduction.py \
  --base  <cache>/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors \
  --twin  <cache>/models--Goekdeniz-Guelmez--Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1/snapshots/a42fe88f75b3ab545466ca5692fef0b4d7bc8009/model.safetensors \
  --out verification/twin-abduction/qwen05b.json \
  --axis-out verification/twin-abduction/axis.npy \
  --npy-dir <scratch>/twin/delta \
  --delta-gguf verification/twin-abduction/qwen05b-delta.gguf \
  --base-revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --twin-revision a42fe88f75b3ab545466ca5692fef0b4d7bc8009
```

`OPENBLAS_NUM_THREADS=1` is required on this box: scipy-openblas on 32 cores
oversubscribes to 64 threads and the 896×4864 SVD goes from 1.7 s to over 60 s
per tensor, which is what cost the earlier attempts their runs. Full run ~1 min
(61 s, hashing both files twice included). numpy 2.4.6, no torch.

## Verdict

The published twin's edit is a clean unit-strength rank-1 abliteration along one
direction, applied identically to the attention writeback, the MLP writeback and
the embedding, leaving the untied `lm_head` untouched. The axis is recovered to
~1e-5 cosine from 48 independent tensors; the entire remaining residual is the
bf16 storage floor. The published edit did not have to be fitted, and it was not:
it was read out. Whether that axis removes refusal is untested here.

Artifacts in this directory: `qwen05b.json` (machine record, schema
`jdewitt.twin-abduction/1`), `axis.npy` (float32[896] unit axis),
`qwen05b-delta.gguf` (473 MB, gitignored), and this note. The generator is
`scripts/twin_abduction.py`.
