# Refusal axis from head geometry alone — zero-forward

Question: can the output head name the direction an abliteration removes, and
what does projecting that direction out of the writeback matrices cost per
tensor? Reported in `report.json` (`jdewitt.refusal-head-axis/1`). No model is
built, no token is generated, nothing is served, captured or benchmarked; every
number below is weight algebra or a tokenizer encode, and the report states
`removal_measured: false`. **This is a negative result.**

## Inputs

| role | repo | revision | safetensors sha256 | bytes |
|---|---|---|---|---|
| aligned | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf…` | `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` | 988 097 824 |
| twin | `Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1` | `a42fe88f75b3…` | `dff81834dbf3fe0bc476eb144f68173eac4feeece41e1337b29e88019b8e2406` | 1 260 367 448 |
| alignment base | `Qwen/Qwen2.5-0.5B` | `060db6499f…` | `88c142557820ccad…` | 988 097 824 |
| stand-in | `verification/iq05b/Qwen2.5-0.5B-Instruct-IQ3_M.gguf` | — | `0b2e17471ccc784d…` | 342 752 576 |

Both Qwen checkpoints set `tie_word_embeddings: true`. The twin nonetheless
saves a distinct `lm_head.weight` (max |lm_head − embedding| = 0.0140); a
config-tied loader discards it, so the served head — and the axis built here — is
`model.embed_tokens.weight`. The trap: reading `lm_head` because it exists builds
an axis the twin never scores with.

Downloads were hashed, read, then deleted. This run removed the snapshot
symlinks and the content-addressed blobs behind them by hand; the committed
script now resolves the symlink into the blob store before unlinking, so a rerun
frees the bytes itself.

## Axis

`d = normalize(mean(E_denial) − mean(E_compliance))` over the served head rows.

* Denial rows: the named words `I, Sorry, Cannot, Unfortunately, No` plus the
  repo table's single-token surfaces (`cannot`, `unable` and their spaced forms)
  → **21 ids**. Compliance rows: `Sure, Certainly, Yes, Here, Okay, Absolutely`
  → **22 ids**. No id is in both sets.
* A surface that tokenizes to more than one id is recorded in
  `multi_token_not_admitted` and never folded into the mean: a subword row
  (`fortunately`) is not the word. The wide sensitivity variant (every id of
  every table surface) rotates the axis by 28° (cos 0.8805); the named-only
  variant agrees with the primary to cos 0.9913.
* Margin per unit step along `d`: **0.24379** (aligned), **0.242164** (twin).

Edit, in float64 and in GGUF `[in, out]` order: `W' = W (I − d dᵀ)` for every
`self_attn.o_proj` and `mlp.down_proj`, 48 tensors per checkpoint.

## Edit cost and the three identities

| | aligned | twin |
|---|---|---|
| relative edit norm, mean | 0.03406 | 0.03412 |
| … attn / ffn | 0.03400 / 0.03412 | 0.03406 / 0.03418 |
| … min / max (48 tensors) | 0.02347 / 0.04128 | 0.02422 / 0.04119 |
| `‖W(I−ddᵀ)−W‖_F = ‖Wd‖₂` max rel. error | 1.17e-15 | 1.10e-15 |
| margin identity max rel. error | 2.88e-14 | 1.55e-14 |
| `g ∥ d` (min over 48) | 1.0 | 1.0 |

For scale: a random unit direction in 896 dimensions edits a matrix by
`1/√896 ≈ 0.0334` of its Frobenius norm, and `d` lands at 0.0341 — the edit is
the size of noise, which is the honest cost reading.

The margin identity is `m − m' = (g·d)((Wd)·a)` with `g = W_outᵀu`; the
coefficient is `g·d`, not `‖g‖`, and `g ∥ d` is measured rather than assumed.
Projecting `d` out of the head's input closes that channel exactly
(`uᵀW_out(I−ddᵀ)x = 0`, reported as `head_channel_closed: true`).

## The two comparisons the card asked for

| comparison | value |
|---|---|
| twin abduction axis, **twin's own head axis** | mean abs cos **0.00010** (max 0.00023), energy captured 2.4e-06 |
| twin abduction axis, **aligned head axis** | mean abs cos **0.11558**, energy captured 0.0133 |
| twin abduction axis, banked `verification/twin-abduction/qwen05b.json` | cos **−0.11558** (attn −0.11557, ffn −0.11558) |
| alignment-delta axis (`instruct − base`, no committed artifact; derived here) | mean abs cos **0.02932** (max 0.09599), energy captured 0.00113 |
| stand-in GGUF head (Q8_0) vs aligned head axis | cos **1.0** at 5 dp; 0 token-string mismatches |
| aligned vs twin head axis | cos 0.99329 → **6.64° rotation** |

The twin's delta is rank-1 at every writeback tensor (top-direction energy share
0.997–0.9988, effective rank 1.0), so the recovered direction is the axis and
not a leading singular vector of something else. It is **orthogonal** to the
head axis of its own checkpoint (1e-4) and only 0.116 away from the aligned
checkpoint's head axis. The banked twin-abduction artifact, computed
independently from the same pair, agrees with the 0.11558 figure.

The difference between 0.00010 and 0.11558 is not a disagreement between
implementations — it is a 6.64° rotation of the head axis caused by the twin's
own embedding edit (mean relative row change 3.55%; **6.30%** on the denial rows
against **1.97%** on the compliance rows). Because the abliteration direction is
nearly orthogonal to the head axis, a sixth of a radian of axis movement is
enough to move the cosine by 0.12: the measured overlap lives in a near-null
space and is not a stable quantity to build a method on.

## Reading

Head geometry alone does **not** name the abliteration direction on this pair.
Two independent observers (this card's per-tensor SVD and the banked stacked-SVD
axis) put the overlap at |cos| ≈ 0.12 for the aligned checkpoint and ≈ 0 for the
twin's own. The alignment delta (`instruct − base`) is not rank-1 (median
effective rank 176) and overlaps the head axis at 0.03. Both numbers are far from
the 0.5–0.9 range that would make a forward-free direction usable.

That is a valid result: the J-space idea survives only with a Jacobian that is
not the identity, and the output head is not a stand-in for it. The edit norms
and identities are still useful — they price any head-derived direction at ~3.4%
of a writeback tensor's Frobenius norm, and they show the projection closes the
head's margin channel exactly.

## Limits

* No behavioural claim: `removal_measured: false`. Whether refusal changes needs
  a served model and a judge wave (GATED; out of scope for this card).
* The twin and alignment directions are **weight deltas**, not behavioural axes.
  A high cosine would have meant the head agrees with an edit — not that
  removing `d` reproduces it. A low one is the same statement in reverse.
* One model family, one pair, one tokenizer, one inventory rule. The 21/22-id
  resolution is Qwen2.5-specific; another tokenizer resolves the same surfaces
  to different ids, and the axis has to be rebuilt per tokenizer.
* `verification/twin-abduction/qwen05b.json` was read from the primary checkout's
  working tree, uncommitted at read time; the sha256 in the report
  (`784d910f…`) identifies the exact file. The alignment-delta axis is derived
  here from the pretrained base because no alignment artifact is committed.

## Reproduce

```
python scripts/refusal_head_axis.py \
  --base Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775 \
  --twin Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1@a42fe88f75b3ab545466ca5692fef0b4d7bc8009 \
  --alignment-base Qwen/Qwen2.5-0.5B@060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --twin-axis-json verification/twin-abduction/qwen05b.json \
  --gguf verification/iq05b/Qwen2.5-0.5B-Instruct-IQ3_M.gguf \
  --cache-dir <hf cache> --delete-downloads \
  --out verification/refusal-head-axis/report.json
```

`tests/test_refusal_head.py` pins the algebra on 4-wide toys: unit `d`, the
residual identity against an independently computed `‖Wd‖`, the margin identity
for both a parallel and an unrelated `g`, the wide-matrix orientation that
`down_proj` needs, and the multi-token-surface rule.
