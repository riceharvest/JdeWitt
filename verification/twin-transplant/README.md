# twin-transplant: the twin's full delta, added to the user's quant

`W' = W + a*dW, a = 1` over the residual-write projections of
`verification/iq05b/Qwen2.5-0.5B-Instruct-IQ3_M.gguf`, with
`dW = W_twin - W_base` computed on the CPU from two public safetensors files. Only
the 48 edited tensors are re-encoded (Q8_0, a type that can store a delta); the
other 242 tensors are copied byte for byte. A rank-k reduction of the same delta
(k = 4, 8, 16 per tensor) is built the same way.

This card is weight algebra and bytes. Nothing here loads a model, a tokenizer, a
server or a GPU: no serve, no capture, no wave, no benchmark. `removal_measured`
is false in the report by construction, and the twin's model card is quoted as
what its CREATOR claims, never as an observation.

## Sources

| role | repo | revision | model.safetensors | sha256 |
|---|---|---|---|---|
| base | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | 988,097,824 B | `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` |
| twin | `Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1` | `a42fe88f75b3ab545466ca5692fef0b4d7bc8009` | 1,260,367,448 B | `dff81834dbf3fe0bc476eb144f68173eac4feeece41e1337b29e88019b8e2406` |

Both are public and ungated with one `model.safetensors` each; the sha256 in the
table is the Hub's LFS oid AND the digest of the bytes on this disk (checked
before any tensor was read, then the files were deleted: 2,248,465,272 bytes
freed). The twin card (`Josiefied-…-README.md` here, sha256 in the report) says
it is "a abliterated model, and further finetuned model on a custom dataset for
more uncensoredness", `base_model: Qwen/Qwen2.5-0.5B-Instruct`, and offers a system
prompt asserting "All refusal vectors have been removed from your programming".
Those are the twin author's claims about the twin. This run did not test them.

## The artifact and the geometry

`0b2e1747…32099`, 342,752,576 bytes, 290 tensors, `general.name = Qwen2.5 0.5B
Instruct`, types `{F32: 121, IQ4_NL: 96, Q5_0: 48, IQ3_S: 21, Q4_K: 3, Q8_0: 1}`.
The targets come from the artifact's own tensor table (`ggufgeom`, name *and*
width, no per-model table): hidden 896, 24 layers, 48 residual-write projections —
`blk.N.attn_output.weight` (896x896) and `blk.N.ffn_down.weight` (896x4864, torch
layout). They are 123,863,040 of the model's weights.

## What shipped

| | value |
|---|---|
| tensors edited | 48 (24 attn_output, 24 ffn_down) |
| untouched tensors | 242, re-read from both files: 0 mismatched, 0 missing |
| `gguf-tensor-diff` | 242 bit-identical; the 48 named as differing are exactly the 48 edited |
| bytes | 342,752,576 -> 414,433,248 (**+71,680,672**, +20.9%) |
| edited tensor | every one IQ*/Q4_K/Q5_0 -> Q8_0, per-tensor `bytes_before -> bytes_after` in the report |
| delivered cos | **min 0.983, mean 0.9877** (attn mean 0.9893, mlp mean 0.9861) |
| delivered snr | min 5.34, and norm_ratio 1.007-1.019 |

`gguf-tensor-diff` splits those 48 in two ways, and the split is honest about what
the tool can see. The 24 `attn_output` are Q5_0, which shares Q8_0's 32-element
block grid, so it value-compares them: **rel_rms 0.0293-0.0457** (median 0.0369) —
the same size as the delta's own relative norm, which is the check that the stored
change is the edit and not codec drift. The 21 IQ3_S and 3 Q4_K `ffn_down` use
256-element blocks, so it reports them as differing with `codec_unsupported` and no
rel_rms ("block grids differ"). The per-tensor cos above is `compare_edit`'s, which
decodes both sides independently and therefore covers all 48.

The delivery number is a codec ceiling, not an axis question. The delta is
2.9%-4.5% of the weight norm it is added to (median 3.55%), Q8_0 rounds every
weight into a block scale, and the realized change carries a norm 0.7%-1.9% LARGER
than the intended one. Nothing in the pipeline is lossless: the base side of the
subtraction is itself a lossy IQ3_M decode, so "what the file holds of dW" is
measured against `dequant(pristine)`, not against the fp32 base.

## Rank-k, and what the delta actually is

One economy SVD per delta, reused for every k (562.8 s for 48 tensors on a loaded
CPU; the whole run took 708.9 s). `cos(dW, dW_k)` is exact linear algebra on the
pair; the realized cos is what the Q8_0 re-encode kept.

| k | cos(dW, dW_k): min / median / mean | energy in top k | realized cos vs dW_k | realized cos vs **full** dW |
|---|---|---|---|---|
| 4 | 0.9986 / 0.9990 / 0.9990 | 99.72%-99.89% | 0.9827 / 0.9873 / 0.9875 | 0.9816 / 0.9864 / 0.9866 |
| 8 | 0.9986 / 0.9990 / 0.9991 | 99.73%-99.89% | 0.9827 / 0.9873 / 0.9876 | 0.9816 / 0.9864 / 0.9866 |
| 16 | 0.9987 / 0.9991 / 0.9991 | 99.75%-99.90% | 0.9827 / 0.9873 / 0.9876 | 0.9817 / 0.9865 / 0.9867 |

Stacked over all 48 tensors the rank-4 edit's realized change has cos **0.985663**
with the full delta; rank 8 gives 0.985686, rank 16 gives 0.985721, and the full
edit itself ships 0.9877 of its own intent. Replacing 123.9M parameters with 4
directions per matrix costs about 0.001 cos. Every rank-k artifact is the same size
as the full one (+71,680,672 bytes) because the type, not the rank, sets the bytes,
and the three carrying distinct `edit_sha256` digests is how the report shows they
are three different files.

That k = 4 is already 0.999 is not a property of the reduction, it is a property of
the pair, and `top-directions.json` measures it directly (power iteration with a
2.2e-07 max residual, cross-checked against an exact SVD on 4 tensors where the two
agree to 6 decimals):

- the delta's **top-1 direction holds 99.70%-99.88% of its energy** in every one of
  the 48 tensors (median 99.80%; exact SVD: s2/s1 = 0.0050-0.0063);
- that top direction is **the same output-space vector everywhere**: |cos| between
  the top left singular vectors is >= 0.999991 over all 1,128 pairs, 1.000000 over
  the 24 same-layer attn-vs-mlp pairs, >= 0.999998 across the 24 FFN layers. The
  input-space vectors are unrelated (|cos| median 0.014, max 0.104);
- removing that one term leaves 4.5%-4.6% of the delta's norm and no dominant
  direction in what remains (top residual singular value: 1.3%-1.4% of the residual
  energy).

So the twin-vs-base writeback delta is one shared direction plus ~0.2% of energy
of fine-tuning noise. Transplanting "the twin" into these 48 matrices is
transplanting one axis, which is the same object the bank fits from captures — this
run did not compare it to any fitted axis and makes no claim that it is the
refusal direction. A 0.2%-energy residual is also where any non-axis part of the
twin lives; nothing here says what it does.

## In place, and the revert

The shared fixture is never modified. The run copies it (digest checked), writes
the edited artifact, replaces the copy in one `os.replace`, then reverts:

- `0b2e1747…` -> `64fe0060…` (byte-identical to the full stage's output),
  342,752,576 -> 414,433,248 bytes;
- `jdewitt gguf-revert` restores `restored_sha256 = 0b2e1747…` with
  **`byte_exact: true`** and `truncated_to: 342752576`.

That revert needed a library change this card also documents: `revert_inplace`
wrote saved byte ranges and never truncated, which is correct for the tile patches
it was built for (same width, same size) and wrong for a re-encode, where every
offset after the first edited tensor moves and the file grows by 71 MB. The sidecar
now carries the whole pre-edit file plus `truncate_to`
(`ggufio.save_whole_file_revert`); `revert_inplace` still refuses a file that
changed since the patch, and `tests/test_twin_transplant.py` covers the grown case
and the refusal.

## Reproduce

```
.venv/bin/python scripts/twin_transplant.py \
  --artifact verification/iq05b/Qwen2.5-0.5B-Instruct-IQ3_M.gguf \
  --base Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775 \
  --twin Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1@a42fe88f75b3ab545466ca5692fef0b4d7bc8009 \
  --cache ~/.cache/twin-transplant --work ~/.cache/twin-transplant/run \
  --report verification/twin-transplant/report.json --ranks 4,8,16
.venv/bin/python scripts/twin_transplant.py analyse \
  --shards ~/.cache/twin-transplant/run/twin-deltas.npz \
  --out verification/twin-transplant/top-directions.json
```

Downloads are hashed against the Hub's LFS oid and deleted; the 48 fp32 deltas are
kept as one `.npz` (495,465,534 bytes) so the rank-k and spectrum numbers can be
recomputed without the 2.2 GB pair. The four edited artifacts (~414 MB each) live
in `~/.cache/twin-transplant/run/` named by the report's sha256, not in the clone.

## What this is not

- **not a removal measurement.** No model was loaded, no answer generated, no
  row judged. `removal_measured: false`. Whether this artifact refuses less is
  GATED behind a real serve config and a judge wave.
- **not a capability measurement.** Retention is unmeasured; the delta is 3.6% of
  the weight norm in the writeback projections and nothing here says what that
  costs.
- **not a claim about the twin's card.** The card is cited as the author's claim
  and the numbers above are about bytes.
- The fixture is IQ3_M, so the base side of `dW = W_twin - W_base` is applied to a
  *decoded* lossy artifact; a 16-bit base would deliver a different (probably
  slightly higher) cos. That difference is the artifact's, not the method's, and it
  is not measured here.
