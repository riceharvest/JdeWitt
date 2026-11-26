# Codec-floor ablation: the denial-aligned component through a coarse quant grid

Status: static only. `removal_measured: false`. No serve, no capture, no wave, no
benchmark, no GPU, no generated token. Every number below is tensor algebra over the
artifact's own bytes, its own tokenizer metadata and the repo's denial-surface table.
They are **not** a refusal measurement and **not** a capability measurement.

Artifacts:

- `report.json` — every per-tensor row, the aggregate curves, the random controls, the
  copy/revert checks and the verdict (`jdewitt-codec-floor-v1`)
- `scripts/codec_floor.py` — the run; `jdewitt/codec_floor.py` — the algebra
- `tests/test_codec_floor.py` — the identities, the codecs, and the byte checks

## The question

Instead of an explicit edit, choose the storage type so the denial component is lost in
the codec floor: re-encode the residual-write tensors at `q4_0`, `q2_k`, `q2_0` and see
what happens to the part of each tensor that acts on the denial tokens' input embedding
rows. If even the coarsest grid keeps that component, the codec floor is above the edit
size — the same wall the codec cards hit, measured from the other side.

## What was measured

The artifact is the stand-in the earlier cards used — the user's
`Qwen2.5-0.5B-Instruct-IQ3_M.gguf` (kept outside the clone, `*.gguf` is gitignored), 342,752,576 B, 290 tensors, sha256 `0b2e1747…`,
header sha256 `5898b9ea…`. Its 48 writeback tensors are `blk.N.attn_output.weight`
(`ne = [896, 896]`, stored Q5_0) and `blk.N.ffn_down.weight` (`ne = [4864, 896]`, stored
Q4_K on 3 layers and IQ3_S on 21). Tokenizer metadata is read from the file itself
(gpt2 model, `qwen2` pre-tokenizer, 151,936 tokens, 151,387 merges, tokenizer digest
`494eeb7f…`).

**Denial ids.** Every one of the 27 surfaces in
`verification/denial-grammar/denial-surfaces.json` is encoded with a byte-level BPE built
from that file's own vocabulary and merge ranks. That yields two id inventories:

| inventory | ids | rank | top-1 energy share | top-4 share |
|---|---:|---:|---:|---:|
| `surface_span` — every token of every surface | 58 | 58 | 9.97% | 22.71% |
| `opener_span` — each surface's first token, deduplicated | 14 | 14 | 15.67% | 45.95% |
| `mean_axis` — normalize(mean(E_denial) − mean(E_compliance)), the x-4 explicit edit | 1 | 1 | 100% | 100% |

**Subspaces.** The denial ids' rows of `token_embd.weight` (Q8_0, read row by row — the
151,936-row table is never loaded whole) span a subspace of the 896-wide residual
stream. `V` is its orthonormal row-space basis, ordered by singular value. The two spans
are not one direction plus noise: the surface span needs all 58 directions (top-1 holds
9.97%), so the measurement is a subspace measurement and not a renamed axis.

**The measurement, per tensor and grid.** With `W` the dequantized writeback tensor and
`W_q` its re-encode at the grid:

```
Z = Vᵀ W              the aligned output block, rows ordered by singular value
aligned fraction      ||Z||² / ||W||²            (isotropic expectation: rank/896)
surviving energy      ||Z_q||² / ||Z||²          (a magnitude ratio, see the note)
removed energy        ||Z_q − Z||² / ||Z||²
delivered cos         <Z_q, Z> / (||Z_q|| ||Z||)
codec floor           ||W_q − W||² / ||W||²      (the grid's error over the whole tensor)
```

A **random subspace of the same rank** goes through exactly the same code: "the grid
removes the denial component" is only a claim about the grid if it removes that subspace
faster than a random one. The baseline share is `rank/896`; the measured share is
reported beside it as `fraction_over_isotropic`.

`surviving energy` can exceed 1. The codec does not shrink a component — it adds error
energy, and part of that error lands inside the subspace. The norm ratio
`sqrt(surviving)` is therefore the honest magnitude, and `cos` is the honest direction.

## The table

Medians over the tensors a grid can represent; energy-weighted means are in the report.

| grid | tensors | codec floor (energy) | codec RMS | codec SNR | aligned fraction | ÷ isotropic | delivered cos | removed | random removed | selectivity | surviving norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `q4_0` | 48 | 1.05% | 10.2% | 19.8 dB | 6.51% | 1.003 | 0.9950 | 1.11% | 1.21% | 1.00 | 1.040 |
| `q2_k` | 24 | 9.34% | 30.6% | 10.3 dB | 6.52% | 1.007 | 0.9552 | 9.38% | 9.37% | 1.00 | 1.033 |
| `q2_0` | 48 | 51.55% | 71.8% | 2.9 dB | 6.51% | 1.003 | 0.7739 | 51.49% | 51.29% | 1.00 | 1.091 |

The aligned fraction is the 58-direction surface span. The rank-1 x-4 axis, which is the
size an explicit abliteration edit would be aiming at, is 0.118% of the same energy
(1.046 × isotropic) — i.e. **8.9× below `q4_0`'s floor, 80× below `q2_k`'s and 437× below
`q2_0`'s**, and its delivered cos is 0.9950 / 0.9548 / 0.7708 at the three grids.

By kind (surface span; attention output is 896-wide and starts from Q5_0, FFN down is
4864-wide and starts from Q4_K/IQ3_S):

| grid | kind | tensors | codec floor | aligned fraction | delivered cos | removed |
|---|---|---:|---:|---:|---:|---:|
| `q4_0` | attention `o_proj` | 24 | 1.02% | 6.50% | 0.9949 | 1.03% |
| `q4_0` | FFN `down_proj` | 24 | 1.26% | 6.52% | 0.9953 | 1.25% |
| `q2_k` | FFN `down_proj` | 24 | 9.34% | 6.52% | 0.9552 | 9.38% |
| `q2_0` | attention `o_proj` | 24 | 51.60% | 6.50% | 0.7827 | 52.06% |
| `q2_0` | FFN `down_proj` | 24 | 50.96% | 6.52% | 0.7674 | 51.34% |

## The rank curve converges to the random control

Aggregate over all tensors of a grid, surface span (rows ordered by singular value, so
`k` is the top-k denial directions). If a low-rank denial component were what the grid
erases, the top directions would be hurt much more than the tail:

| grid | k | aligned fraction | cos | removed | random removed | removed ÷ random |
|---|---:|---:|---:|---:|---:|---:|
| `q4_0` | 1 | 0.11% | 0.9941 | 1.47% | 1.23% | 1.20 |
| `q4_0` | 4 | 0.44% | 0.9949 | 1.27% | 1.21% | 1.05 |
| `q4_0` | 16 | 1.78% | 0.9951 | 1.23% | 1.21% | 1.02 |
| `q4_0` | 32 | 3.58% | 0.9951 | 1.22% | 1.21% | 1.01 |
| `q4_0` | 58 | 6.49% | 0.9952 | 1.21% | 1.21% | 1.00 |
| `q2_k` | 1 | 0.11% | 0.9477 | 11.18% | 9.44% | 1.18 |
| `q2_k` | 4 | 0.43% | 0.9533 | 9.80% | 9.41% | 1.04 |
| `q2_k` | 16 | 1.79% | 0.9548 | 9.47% | 9.40% | 1.01 |
| `q2_k` | 32 | 3.59% | 0.9552 | 9.38% | 9.37% | 1.00 |
| `q2_k` | 58 | 6.52% | 0.9554 | 9.33% | 9.37% | 1.00 |
| `q2_0` | 1 | 0.11% | 0.7443 | 59.20% | 51.76% | 1.14 |
| `q2_0` | 4 | 0.44% | 0.7607 | 53.57% | 51.40% | 1.04 |
| `q2_0` | 16 | 1.78% | 0.7664 | 51.94% | 51.39% | 1.01 |
| `q2_0` | 32 | 3.58% | 0.7678 | 51.60% | 51.24% | 1.01 |
| `q2_0` | 58 | 6.49% | 0.7684 | 51.43% | 51.29% | 1.00 |

This is the one place where the codec is not perfectly direction-blind: the *top*
denial direction is removed 14–20% faster than a random direction, and the excess decays
to the random value by k≈8–16. It is a real but small bias — 0.24 points of energy at
`q4_0`, 1.7 at `q2_k`, 7.4 at `q2_0` — and it points the same way the ablation would
want. It does not survive as a mechanism: after `q2_0` the top direction still has cos
0.744, which is *below* the 0.768 a random direction keeps, so the extra damage buys
nothing except more damage. The full curve (`k = 1, 2, 4, 8, 16, 32, 58`) is in
`report.json` → `types.<grid>.aggregate.surface_span.curves`.

## The codec is blind to direction (at full rank)

Over 48 tensors the removed fraction inside the whole 58-direction denial-aligned
subspace equals the removed fraction inside a random subspace of the same rank:
selectivity 0.995–1.003. The delivered cos is what an isotropic, direction-blind codec
predicts — `1/sqrt(1 + floor)`:

| grid | measured cos | predicted for isotropic noise | difference |
|---|---:|---:|---:|
| `q4_0` | 0.9950 | 0.9948 | +0.0002 |
| `q2_k` | 0.9552 | 0.9564 | −0.0013 |
| `q2_0` | 0.7739 | 0.8123 | −0.0384 |

At `q4_0` and `q2_k` the measured cos is the blind-codec value to within 0.0015. At
`q2_0` the grid's error is so large that it interacts with the component itself (the
delivered aligned block is 1.19× the original energy), and the measured cos falls below
the white-noise prediction — which is the difference between "the grid does not know
where the component is" and "the grid is now the dominant term in it".

## Bytes: what the written files are

One GGUF per grid, the 48 (or 24) writeback tensors re-encoded, everything else copied
from the source:

| grid | file bytes | Δ | tensors written | untouched byte-identical | decoded | KV section | names/order/dims | revert |
|---|---:|---:|---:|---:|---:|---|---|---|
| `q4_0` | 342,752,576 → 352,499,264 | +9,746,688 | 48 | 242 | 290 | identical | identical | byte-exact |
| `q2_k` | 342,752,576 → 330,393,152 | −12,359,424 | 24 | 266 | 290 | identical | identical | byte-exact |
| `q2_0` | 342,752,576 → 317,662,784 | −25,089,792 | 48 | 242 | 290 | identical | identical | byte-exact |

`q4_0` on this artifact is **larger** than the file it starts from: Q4_0 is 4.5 bits per
weight and the source is IQ3_S/Q5_0. The two 2-bit grids are smaller and, as the table
above shows, 9% / 52% wrong in energy.

`q2_k` covers only 24 of the 48 writeback tensors. Q2_K quantizes in 256-value
superblocks along `ne[0]`, and a 896-wide `attn_output` row is 3.5 blocks: llama.cpp's
row-size arithmetic truncates and the loader then refuses the file. The 24 FFN tensors
are 4864-wide (19 whole blocks) and take it. A grid that cannot be stored is not a grid
that "keeps" or "loses" anything, so those 24 are reported as skipped with the reason,
not silently measured.

Every written file was read back: KV section byte-identical to the source, tensor
names/order/dims identical, edited tensors carrying the target type and the exact bytes
measured, all 242 or 266 untouched tensors byte-identical, all 290 tensors decoded to
their declared element count with finite values, and a whole-file revert sidecar
restoring the source sha256 exactly. The published `gguf` package (0.19.0) has no
`GGMLQuantizationType` for Q2_0's raw type 42 and refuses that file; this build's reader
decodes it, and the refusal is recorded in the report as `gguf_py_note` rather than
hidden.

## The honest conclusion

Measured from the other side, the wall is the same:

1. **The component is small and not over-represented.** The denial-subspace share of a
   writeback tensor's energy is 6.51%, which is the isotropic share (58/896 = 6.47%) to
   within 0.3%. Nothing in these tensors is oriented toward the denial tokens' embedding
   directions: the rank-1 mean axis is 0.118% (1.05 × isotropic), the 14-direction opener
   span 1.54% (0.99 ×). An edit of that size is a rank-1 or rank-14 perturbation, not a
   6% chunk of the matrix.
2. **The coarse grids are above that edit, and they still keep it.** `q4_0`'s floor is
   1.05% of the tensor energy: 8.9× the rank-1 axis and 0.16× the 58-direction span.
   It delivers the axis at cos 0.9950 and the span at cos 0.9950 — the grid does not
   round anything away, because its error is spread over all 896 directions.
3. **When the floor finally does exceed the component, it exceeds everything.** At
   `q2_0` the floor is 51.6% of the tensor energy (7.9× the span, 437× the axis), the
   axis's cos falls to 0.771 and the span's to 0.774 — and so does a random subspace's
   (0.769). At full rank the selectivity is 1.00 at all three grids; the only directional
   bias in the experiment is the 14–20% extra removal of the single top direction, which
   leaves that direction at 0.744 cos, worse than a random direction's 0.768. The grid is not an edit; it is a
   re-encode at 2.9 dB SNR that replaces half of every direction with noise, and the
   delivered "denial component" comes out *larger* than the original (1.09× the norm)
   while pointing 0.77 cos away from it. That is the codec cards' wall: coarse grids
   destroy precision everywhere at exactly the rate they destroy the component, so they
   can never be used to remove it.

The one direction in which the result is positive: choosing `q2_0` provably destroys the
denial-aligned component of these tensors — 51% of its energy is gone and what remains
is 0.77 cos from it, no better than a random direction would be. If the only question is
"is the denial component recoverable from a `q2_0` build of this 0.5B", the answer is no.
If the question is "does the grid remove refusal while leaving the model intact", the
answer is no on the second half, and the first half is unmeasured here by construction.

## What this is not

- Not a refusal measurement. No logits, no answers, no serve. The denial subspace is a
  token-alignment subspace built from input embedding rows; whether it is the refusal
  direction is exactly what a forward pass would have to show, and this card forbids one.
- Not a capability measurement. `q2_0` at 2.9 dB SNR would be a broken model by any
  measure, but that is inference from the codec error, not a measurement of it.
- Not a claim that the aligned energy fraction is what refusal "costs". Frobenius energy
  in a token subspace is a static quantity; the activation-weighted version needs the
  capture the zero-forward rule excludes.
- The row-block skip is a `llama.cpp` storage constraint, not a property of the codec's
  arithmetic: the encoder can produce 896-wide Q2_K blocks, and the file would not load.

## Reproduce

```
python scripts/codec_floor.py \
    --artifact ~/.cache/twin-transplant/run/Qwen2.5-0.5B-Instruct-IQ3_M.gguf \
    --surfaces verification/denial-grammar/denial-surfaces.json \
    --compliant verification/denial-grammar/compliant-openings.jsonl \
    --work-dir ~/.cache/codec-floor/run \
    --report verification/codec-floor/report.json
```

Roughly two minutes on four BLAS threads. The report is deterministic apart from
`generated_at` and `wall_seconds`; the three artifact digests reproduce exactly.
