# P4e — one ppl ladder over four axis sources (Spark-X2.6-1.7B Q4_K_M)

Four prefill-only ladders on the P4d capture, one per axis source, to arbitrate
**which direction the edit removes** at a fixed strength. Zero generated tokens in
every run; no cert wave, no removal claim. This card ranks only — P4f
(t_2088fc50) measures the shipped bytes.

## Code contract added

`tune` used to re-fit the contrast mean inside the invocation
(`axes_by_shape` from `per_layer_axes` / `shared_axis` / `per_layer_subspace`),
and its adapter cache identity was `(artifact sha, records sha, families, basis,
width, layers, arch)`. A precomputed axis would therefore silently reuse the
contrast-mean adapters and print a ranking for an edit nobody selected.

- `tune --axes FILE [--axes-key axes]` loads a `[L,W]` (or `[W]`, broadcast over the
  captured layers) tensor and uses it **in place of the fitted axis for every
  requested `--shapes`**. The file is what `python -m jdewitt.rdo fit --out` writes
  (`{'axes': tensor, 'meta': {...}}`); a bare tensor is accepted.
- Refusals: a width that differs from the records (`width N ... records are width M`);
  more layer rows than the capture (`captured only L`); a rank-k tensor (3-D) with
  the different-contract message; and `--rank > 1` together with `--axes`.
- `fit_identity` now carries `axes_source` (`"contrast"` or `"file:<path>"`) and
  `axes_sha256` (sha256 over the `[L,W]` float32 buffers actually applied, per
  requested shape, with the shape tag). Both are printed in `tune.json` under
  `fit_identity`, and the whole identity block is now also emitted at top level.
  `tune.json` also carries `axes` (source/sha/file/layers/width/file_meta) and an
  explicit `generated_tokens` counter.

## Inputs

- records `runs/p4/records.pt` sha256 `9d5e648f9fee7638d174c9397c96ca50c01d14e34a026b582638fc02fabadf6a`
  (P4d capture, 36 bad / 30 good / 10 overdecline / 3 deliver / 1 ambiguous; committed
  evidence in `verification/p4d/`)
- artifact `/mnt/ssd/models/Spark-X2.6-1.7B-Q4_K_M.gguf` sha256
  `80ff5222ae51de055fe2e86b1ec4f6784c2f9db704ed3b043b97ce900d3a75ad`, arch `spark2_5`,
  28 writeback layers, width 2048
- perplexity `/mnt/ssd/dev workspace/llama.cpp-dspark/build-perp/bin/llama-perplexity`
  on CPU (`--ppl-ngl 0`), `--parallel 3`, `--threads 16` (default), `--ctx` 512

**Corpora.** `tune` derives control/refusal/delivery from the records. The default
1500-char corpus fails the binary's floor on this capture: the refusal corpus (the
model's own 36 declining answers) tokenizes to 294 tokens against a 2x-ctx (300
token) minimum. All four arms therefore use `--corpus-chars 6000` (refusal 6002 chars,
control 6007, delivery 6064; `ctx_used` 512 in every `tune.json`), the same derivation
for every arm.

## Axis sources

| arm | selector | `--axes` | `--footprints` | axes_sha256 (fit_identity) |
|---|---|---|---|---|
| A1 | contrast mean (fitted here) | – | `all` | `5ca40165b69dda2a30e25b22dd525b1c844cc7fca5d5517deed780df3553644e` |
| A2 | contrast mean (fitted here) | – | band from `layer_scores.json` = `0-27` | same as A1 (`5ca40165…`) |
| A3 | RDO axes | `runs/p4/axes-rdo.pt` | `0-27` | `0838820a994f5f66…` (distinct) |
| A4 | contrast mean (fitted here) | – | margin `best_footprint` = `full` -> `0-27` | same as A1 |

A1's perlayer rows are byte-identical to P4d's `axes-contrast.pt`
(`f2b14d3015ff1654…`), verified with `torch.equal` — the "contrast" source in this
table is the P4d selector, not a re-fit lookalike.

### A2 and A4 collapse onto A1 on this capture

`layer_scores.json` reads `footprint: "0-27"`, `flat: false` with every layer above
the keep line (0.389476), i.e. the emitted band IS the whole captured span and has no
complement. `margin_scores.json` reads `best_footprint: "full"` with
`delta_bad_minus_good = -8.490625` (negative, so A4 is licensed by the card). Since a
full-span footprint is labelled `all`, A1, A2 and A4 are the *same* command apart from
the flag spelling, and they produce identical family keys, identical adapter bytes
(`53b12bcf…` perlayer, `1a6784d7…` shared) and identical readings to 4 decimals. They
are three repeats of one selector, not three independent sources; the only real
axis-source contrast on this capture is **A1 vs A3**. This is written as a
duplicate, not counted as a win.

### RDO fit (A3)

`python -m jdewitt.rdo fit` on `records.pt` directly exits on layer 0:
`bad and good means coincide; no contrast direction`. Cause: rdo centers every row at
its `family_id` mean whenever any family_id exists; on this capture the harmful rows
carry `family_id` but the harmless rows carry none, so the 30 harmless rows pool into
one family mean, every bad family mean subtracts to zero, and the contrast is
degenerate. This capture has no within-family bad/good pairs, so the family-centering
design does not exist here. The fit therefore ran in rdo's own raw-row mode
(`centering: none`) on a derived copy with `family_id` nulled:

```
python -m jdewitt.rdo fit --records runs/p4/records-rdo-none.pt --holdout none \
  --out runs/p4/axes-rdo.pt --report runs/p4/axes-rdo-report.json
```

`records-rdo-none.json` records the transformation, the source sha256 and the reason;
`axes-rdo-report.json` is committed here. Fit facts: `[28,2048]`, `centering: none`,
`axes_sha256 5c481da33093174c715762db601e9d3b4cb04074a260323381bcefa0a9d197f7` (rdo's
own buffer digest; tune's digest differs because it tags the shape), cos(rdo,
contrast) 0.677–0.830 per layer, kept fraction 0.737–0.869, `auc_train` 1.0 on the
same rows, `probe_underdetermined_layers` = all 28 (66 labelled rows / 2048 dims, so
the probe is heavily underdetermined and the refinement is in-sample). A3 is therefore
the RDO selector *as this capture can honestly fit it*, not RDO under a holdout design.

## Ladder (all four runs)

```
python -m jdewitt.cli tune --gguf /mnt/ssd/models/Spark-X2.6-1.7B-Q4_K_M.gguf \
  --records runs/p4/records.pt --perplexity <cpu llama-perplexity> \
  --token-budget 0 --no-verify --parallel 3 --ppl-ngl 0 --scales 1.0 \
  --corpus-chars 6000 --out runs/p4/ladder-<arm> --footprints <flag>
```

`tune.json` per arm: `generated_tokens: 0`, `tokens.search: 0`, `tokens.screen: 0`.
The ladder is deterministic at this config: A1, A2 and A4 (identical selectors) read
the same PPL values exactly, not just within noise.

Base (identical in all four runs): control PPL 23.1502 ±3.34649, refusal 6.4587
±0.69261, delivery 9.4067 ±1.13047. `drift_noise_ln` (tune's own ±-derived floor,
half-width/PPL) = **0.1446**; `drift_budget_used` = 0.2891.

| arm | selector | shape | scale | drift_ln | delta_br | codec min_cos | verdict |
|---|---|---|---:|---:|---:|---:|---|
| A1 | contrast mean | perlayer\|all | 1.0 | +0.2166 | -0.3319 | 0.3024 | kept — only source left; tone-only at 1.0 |
| A1 | contrast mean | shared\|all | 1.0 | +0.1660 | -0.6566 | 0.1882 | kept — only source left; tone-only at 1.0 |
| A2 | contrast mean (band 0-27) | perlayer\|all | 1.0 | +0.2166 | -0.3319 | 0.3024 | duplicate of A1 (band == full span) |
| A2 | contrast mean (band 0-27) | shared\|all | 1.0 | +0.1660 | -0.6566 | 0.1882 | duplicate of A1 |
| A3 | RDO axes-rdo.pt | perlayer\|all | 1.0 | +1.2590 | -0.7385 | 0.1549 | **DROPPED** (worse drift, worse delta_br, worse codec survival) |
| A3 | RDO axes-rdo.pt | shared\|all | 1.0 | +1.2590 | -0.7385 | 0.1549 | **DROPPED** |
| A4 | contrast mean (margin full) | perlayer\|all | 1.0 | +0.2166 | -0.3319 | 0.3024 | duplicate of A1 (best_footprint == full span) |
| A4 | contrast mean (margin full) | shared\|all | 1.0 | +0.1660 | -0.6566 | 0.1882 | duplicate of A1 |

`delta_br = ln PPL(refusal) − ln PPL(delivery)` shift against base; positive is
movement toward the removal branch. A3's two shapes are identical because the same
`[L,W]` file serves both shapes (the per-layer row is the direction at each layer),
which is the requested contract.

## Arbitration

- **A3 vs A1 (perlayer), the one real axis-source comparison:** drift gap
  **+1.0424 nats**, delta_br gap **-0.4066**. delta_br is outside tune's 0.05
  agreement tolerance, so the two rows are *not* matched, and the 1.04-nat drift gap
  is far outside both the ladder's own noise (`drift_noise_ln` 0.1446) and the
  0.41-nat figure measured on this artifact in `docs/PLAN-NEXT.md`. A3 is worse on
  drift, worse on delta_br, and its edit survives the Q4_K codec less well
  (min_cos 0.155 vs 0.302). **A3 is dropped; it is not rerun.**
- **Contrast vs contrast (A1/A2/A4):** identical selector, identical readings —
  nothing to arbitrate, drift difference 0.000 (inside noise).
- **perlayer vs shared (same contrast source, scale 1.0):** drift 0.2166 vs 0.1660,
  a 0.0506-nat gap **inside** the 0.1446 / 0.41 noise floor -> **UNMEASURED**. The
  shape difference is not resolved by this ladder.
- **No arm moves the branch at 1.0.** delta_br is negative for every row, so the
  model predicts its own refusals *relatively better* after the edit: this is a
  tone/fluency-only table at scale 1.0, not a removal result. `winner` is `null` in
  all four `tune.json` files, and `tune`'s own note says no arm both stayed inside
  the drift budget and moved the refusal branch.
- **Result of the arbitration: contrast mean survives as the only non-dropped axis
  source; RDO is dropped on this capture. Nothing ships from this table** — the
  surviving selector has not moved the branch at the strength the card fixed.

## Not claimed

No certification, no removal, no "RDO is worse in general". A3's fit had no holdout
design on this capture and its probe is underdetermined at every layer, so its loss
here bounds *this* axis on *these* 66 rows. P4f (t_2088fc50) is the card that measures
the shipped bytes; P4e ranks only.

## Evidence here

- `tune-a1.json` … `tune-a4.json` — full `tune.json` per arm (fit_identity, axes,
  tokens, base, arms, winner, notes)
- `ladder-a1.json` … `ladder-a4.json` — raw PPL readings and the ladder cache identity
- `ladder-table.json` — the four rows above plus the arbitration numbers, generated
  from the four `tune.json` files
- `axes-rdo-report.json` — the RDO fit report (no holdout; centering none)
- `records-rdo-none.json` — provenance of the derived capture used for the RDO fit
- band and margin inputs are `verification/p4d/layer_scores.json` and
  `verification/p4d/margin_scores.json` (byte-identical to the `runs/p4` copies)
