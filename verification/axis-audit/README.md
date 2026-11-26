# Static axis agreement audit — Qwen2.5-0.5B family

Every weight-only construction in the twin-methods set computes a "refusal axis"
from bytes, without a forward pass. This card asks the only question that can be
answered before a model is served: **do those constructions point at the same
subspace, and how stable is each one?** Nothing here measures removal —
`removal_measured` is `false` in the artifact.

Artifact: `verification/axis-audit/agreement.json` (schema `jdewitt.axis-audit/1`).
Script: `scripts/axis_audit.py` (numpy only, no torch, no model, no device).
Fetch/hash: `scripts/axis_audit_fetch.py`. Math tests: `tests/test_axis_audit.py`.

## Artifacts (repo, revision, sha256 of the weight file)

| role | repo | revision | bytes | sha256 |
|---|---|---|---:|---|
| base | `Qwen/Qwen2.5-0.5B` | `060db6499f32faf8b98477b0a26969ef7d8b9987` | 988097824 | `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342` |
| instruct | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | 988097824 | `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` |
| twin | `Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1` | `a42fe88f75b3ab545466ca5692fef0b4d7bc8009` | 1260367448 | `dff81834dbf3fe0bc476eb144f68173eac4feeece41e1337b29e88019b8e2406` |

All three are BF16, hidden 896, 24 layers, vocab 151936. Base and Instruct store
no `lm_head.weight` (tied); the twin ships a separate `lm_head.weight` whose rows
differ from its embedding (`||d||/||E|| = 0.033`). The twin differs from Instruct
in 49 of 290 shared tensors: all 48 writeback tensors plus `embed_tokens`.
Instruct differs from base in all 290 — the alignment delta has no surface.

The user-quant stand-in `verification/iq05b/Qwen2.5-0.5B-Instruct-IQ3_M.gguf`
(342752576 B, sha256 `0b2e17471ccc784dfd7b042a81e34289a3accb75ccda7347598a329e0a932099`)
was hashed, not audited: every axis below comes from the float safetensors.

## Constructions and what each assumes

| id | construction | assumption it rests on |
|---|---|---|
| `x1` | `dW = twin - instruct`, top **left** singular vector per writeback tensor | the published twin edit is rank-1 per writeback tensor (`dW = -a r (W^T r)^T`), so the left vector *is* the removed residual direction. No activation, no token set — the whole claim is the rank-1 shape. |
| `x1b` | same `dW`, largest-norm **column** | `dW = -s u v^T` makes `u` a column of `dW`, so the largest column is the axis; no SVD. This is the heuristic already in `scripts/oracle_axis_alignment.py`. |
| `x1e` | `dE = twin.embed - instruct.embed`, top **right** singular vector | the same rank-1 projection was applied to the embedding rows (`dE = -(E r) r^T`), so the *right* singular vector of the embedding delta is `r` — an independent read from a 151936x896 matrix. |
| `x3` | `dW = instruct - base`, top **left** singular vector per writeback tensor | the leading left direction of the instruction-tuning delta is "the alignment direction". The delta is high-rank and also carries instruction-following. |
| `x3b` | same `dW`, largest-norm column | the same column heuristic, applied to a delta that is not rank-1. |
| `x4` | `u = normalize(mean one-hot(denial) - mean one-hot(compliance))` (vocab space), `d = normalize(W_out^T u)` (residual space) | the mean denial head row minus the mean compliance head row is the refusal direction and the head's pullback maps it into the residual stream. Depends on the two token sets. No activation. |
| `x4base`, `x4twin` | `x4` computed on the base and twin heads | same assumption, different model; a check on whether the head geometry is model-specific. |

`x4` uses 37 denial ids (the card's `I, Sorry, Cannot, Unfortunately, No` plus
every surface in `jdewitt/assets/vocab-remap-denial.json`, leading space and bare
forms) and 12 compliance ids (`Sure, Certainly, Yes, Here, Okay, Absolutely`).
Two sensitivity variants are computed: compliance = the affirmative table's
entries (14 ids), and denial = the card's five words only.

## Agreement (|cos|; signs of singular vectors are arbitrary)

```
          x1     x1e     x1b      x3     x3b      x4   x4base  x4twin
  x1    1.0000  1.0000  1.0000  0.0899  0.0338  0.0806  0.0782  0.0806
  x1e   1.0000  1.0000  1.0000  0.0898  0.0338  0.0806  0.0782  0.0806
  x1b   1.0000  1.0000  1.0000  0.0902  0.0335  0.0805  0.0781  0.0805
  x3    0.0899  0.0898  0.0902  1.0000  0.0059  0.0549  0.0575  0.0549
  x3b   0.0338  0.0338  0.0335  0.0059  1.0000  0.0694  0.0686  0.0694
  x4    0.0806  0.0806  0.0805  0.0549  0.0694  1.0000  0.9979  1.0000
  x4base 0.0782 0.0782  0.0781  0.0575  0.0686  0.9979  1.0000  0.9979
  x4twin 0.0806 0.0806  0.0805  0.0549  0.0694  1.0000  0.9979  1.0000
```

Per layer the numbers are the same order: `x1 vs x3` mean |cos| 0.0522
(min 0.0010, max 0.1114), `x1 vs x4` 0.0806, `x3 vs x4` 0.0343,
`x1 vs x3b` 0.0251. The alignment delta's own layer-to-layer spread is wide
(0.001-0.111), so the mean is a poor summary of a quantity that is not stable.

## Stability and rank-1 shares

| quantity | `x1` | `x3` |
|---|---|---|
| top-1 singular share per tensor, mean (min) | 0.99798 (0.99703) | 0.007346 (0.002462) |
| second singular / first, mean | 0.0055 | 0.899 |
| ||delta||_F / ||W||_F, mean | 0.0361 | 0.0723 |
| cos(layer axis, shared axis) mean (min-max) | 1.0000 (1.0000-1.0000) | 0.5406 (0.018-0.887) |
| o_proj vs down_proj axis cos, all 24 layers | 1.0000 | -0.70 to +0.69 |
| rank-1 reconstruction residual / ||W|| | 0.0016 | — |

`x1`'s recovered right singular vector matches `W_instruct^T u / ||W_instruct^T u||`
with cos 1.0000 on all 48 tensors (min 1.0000): the twin delta is exactly the
rank-1 projection of this Instruct checkpoint's writeback matrix, and the residual
0.0016 of the weight norm is at the BF16 storage floor. `x1b` rediscovers the same
vector without an SVD (cos 0.9999, min 0.9996). `x1e` recovers the same direction
from the embedding delta (share 0.9977, |cos| with `x1` = 1.0000), so one
direction explains three tensor kinds of two shapes.

`x3` has no such structure: its leading direction explains 0.7 % of a tensor's
delta, the second singular value is 0.90 of the first, and the two estimators
disagree (SVD vs column-norm cos 0.0959). The stacked per-tensor axes spread over
a wide spectrum (top stack share 0.290, top 8 = 0.700) and the `x1` axis carries
only 0.8 % of its energy in `x3`'s top-1 stack axis, 2.7 % in the top 16.

`x4` is model-stable (instruct vs base 0.9979, vs twin 1.0000; the logit-space
`u` is 1.0000 across the three, since the same token rows are involved) and
token-set-fragile: the core-denial variant moves `d` to 0.8248, the
affirmative-table compliance set to 0.0589 (logit `u` 0.1341). Head row
separation norms are 0.7406 (instruct/twin) and 0.7537 (base).

Energy the projection would touch, as a fraction of the Instruct writeback
energy: `x1` 0.00131 (0.08-0.20 %), `x3` 0.00856, `x4` 0.00116.

## Findings

1. **One construction is a property of the weights, the rest are properties of
   an estimator or a word list.** The twin edit is a single shared rank-1
   direction: identical in both writeback kinds, identical across all 24 layers,
   recovered independently from the 151936x896 embedding delta at |cos| 1.0, with
   a right vector matching `W^T u` exactly. `x1`, `x1b` and `x1e` are three views
   of the same object — their agreement is a consistency check, not a discovery.
2. **The alignment delta has no axis.** Top-1 share 0.73 %, second/first 0.90,
   layer cos spread 0.018-0.887, and two estimators agree at 0.096. Any
   downstream number that calls the leading singular vector of `instruct-base`
   "the alignment direction" is reporting the estimator's arbitrary choice among
   a nearly degenerate spectrum.
3. **The three families are near-orthogonal.** Every cross-family |cos| is
   <= 0.09. In particular the head-geometry axis lies at 0.08 to the direction
   the published twin actually removed: whatever `x4` points at, the published
   abliteration did not remove it (and vice versa). This is a finding, not a
   failure: it says the constructions target different subspaces.
4. **x-4's instability is lexical, not numerical.** Across three models the same
   head geometry gives the same direction to 1e-3, but swapping the compliance
   set for the repo's affirmative table moves the residual axis to 0.059. A
   head-geometry claim inherits the token list it was built from.
5. **The twin edit is tiny and exactly structured; the alignment edit is large
   and unstructured.** 0.13 % of writeback energy in 48 rank-1 terms versus 7.2 %
   of the weight norm spread over a flat spectrum.

## Limits

- Static only. No forward pass, no serve, no capture, no GPU, no benchmark; a
  behavioural removal claim is out of scope and `removal_measured` is false.
- A low cosine is not an error: it identifies which subspace a construction
  targets and which construction is unstable. It does not say which one would
  work if served.
- Singular-vector signs are arbitrary; the JSON carries signed cosines per
  comparison and |cos| in the matrix.
- Spectra come from `eigh` of the 896x896 Gram matrix `dW dW^T` (measured ~0.7 s
  vs ~10 s for a LAPACK `gesdd` on the 896x4864 delta, and ~18 s multithreaded
  under host load). That is exact for the top directions and the traced energy,
  but squaring the matrix costs relative precision on the smallest singular
  values — only the leading values and the shares are reported, never a
  condition number.
- `x1`'s axis is the *published twin's* edit, and the twin is third-party. Its
  model card says abliterated; this audit verifies the edit's shape, not the
  recipe, its strength, or its intent.
- No fitted axis exists in this width: the committed axes are `p5/pair-axes`
  (36x2560) and `p7-spark` (28x2048), both dimension-incompatible with hidden
  896, and the 4B oracle axes are `.pt` without a shape sidecar. The
  fitted-vs-weight-only comparison is therefore not runnable on the 0.5B and is
  recorded as an open item rather than approximated.
- Quantized-axis stability is not audited; the IQ3_M stand-in is hashed only.
  Codec fidelity of an applied edit is measured elsewhere
  (`verification/p2/space-dry-run.json`, delivered mean cos 0.9863).

## Reproduce

```sh
# fetch (pinned revisions), hash, write the manifest (needs network, ~3.2 GB)
python3 scripts/axis_audit_fetch.py --out runs/axis-audit/models.json
# audit (pure numpy; one BLAS thread is ~10x faster on a loaded host)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/axis_audit.py \
  --models runs/axis-audit/models.json \
  --out verification/axis-audit/agreement.json \
  --gguf "verification/iq05b/Qwen2.5-0.5B-Instruct-IQ3_M.gguf"
# the math tests; --noconftest because the repo conftest imports torch and this
# path is deliberately torch-free
python3 -m pytest tests/test_axis_audit.py -q --noconftest -o addopts=""
```

The run is deterministic: a second run of the same command over the same bytes
reproduced every field of `agreement.json` exactly (compared with
`json.dumps(..., sort_keys=True)` after dropping `date_utc` and `elapsed_s`), and
`checks` records unit norms, the per-layer table shape, and the matrix recomputed
from the stored axes.
