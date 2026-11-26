# Zero-generation selectors: three rankings and the rule that arbitrates them

P4a–P4c implemented three selectors that rank an edit before a token is generated;
P4d–P4f measured them on Spark-X2.6-1.7B (28 layers, width 2048) and on its
Q4_K_M artifact. None of them is a removal method. Each reads the model's own
prefill logits or the activations the capture already saved, emits a ranking, and
stops.

This page records, for each selector, the command line, the measured cost, the
result, and the exact limit of the claim; then the order in which the phase lets a
ranking decide which axis ships. Every number below is followed by the artifact
that produced it. `runs/` is gitignored, so it holds the artifact store (records,
axes, merged gguf) and every JSON below; the committed copies of the P4d selector
JSONs are in `verification/p4d/`, and of the P4f gate JSONs in `verification/p4f/`,
under the same filenames. A number with no artifact behind it is written
UNMEASURED, not rounded.

## The rule: selectors rank, the gates decide

A selector is allowed to rank. It is never allowed to certify: a prefill margin, a
layer SNR or a refined direction is evidence about the axis, not about the edited
artifact's behaviour (METHODOLOGY R2: shape verdicts need a merged serve; R6: a
probe prunes, a full gate certifies). The phase therefore arbitrates in this order,
cheapest first:

1. **The ppl ladder** (`tune`, zero generated tokens). Each axis source is built
   into arms and scored by `drift_ln` = ln PPL(control) − ln PPL(base) on the same
   corpora, beside `delta_br`, the BR shift against base where BR = ln PPL(refusal)
   − ln PPL(delivery) (`jdewitt/ppl.py`) and positive = movement toward the removal
   branch. An axis source that loses the ladder is **dropped** at this rung; it is
   not carried forward as a candidate.
2. **The merged bytes** (`spanfit build --in-place` on the Q4_K_M artifact, then a
   ladder over the same corpora). Adapter arms are a screening device; the merged
   file is the release artifact (METHODOLOGY R8). Equal removal is defined as
   |Δdelta_br(arm) − Δdelta_br(contrast arm)| ≤ 0.05
   (`runs/p4/selection.json`, `equal_removal.definition`). A win requires the
   selected arm's drift to be lower than the contrast arm's by **more than the
   ladder's own noise** (`drift_noise_ln`); a point estimate inside the noise is
   UNMEASURED.
3. **The cert lane** (R2, pinned config, R9). Removals are counted by a shape read
   of the merged serve. If the cert removal count contradicts the ladder's "equal
   delta_br", then equal delta_br was not equal removal, and the claim is
   UNMEASURED — the ladder is a proxy, the cert lane is the behaviour.

A selector that loses is dropped, not kept "because it nearly won". It may stay in
the record as an axis source with a falsifier path — which is what
`jdewitt/methods.py` now does for RDO — but it does not enter `IMPLEMENTED` and no
release claim is built on it.

## 1. Sure-vs-cannot margin

**What it measures.** The model's own token-logit margin between the single-token
spellings of `Sure` and `cannot` at the position that would emit the first answer
token: `logit[Sure] − logit[cannot]`, mean over resolved `(Sure, cannot)` id pairs.
Only a spelling that tokenizes to exactly one id enters the statistic; a multi-token
spelling is refused as a different estimator (`jdewitt/margin.py`, `MarginRefused`).
P4d resolved all four spellings, so four pairs were scored
(`runs/p4/margin_scores-full.json` `resolution`, `pairs`).

The selector applies a candidate axis through temporary forward hooks on the
captured decoder layers, re-prefills prompt + the record's own answer, and reports
the margin before and after per row, plus the mean drop on harmful rows minus the
mean drop on harmless rows.

**Command** (P4d; the other two footprints differ only in `--footprints`/`--out`):

```sh
python -m jdewitt.margin score --model /path/to/Spark-X2.6-1.7B \
  --records runs/p4/records.pt --axis-file runs/p4/axes-contrast.pt \
  --footprints 0-27 --out runs/p4/margin_scores-full.json --trust-remote-code
```

**Cost.** 0 generated tokens by construction; 160 prefills per footprint, 480 over
the three footprints; hooks removed, weights never written
(`runs/p4/margin_scores.json` `tokens_generated`, `prefill_count`,
`prefills_per_footprint`; `runs/p4/margin_scores-full.json` `hooks_removed`,
`weights_written`). The capture is 80 rows: 36 bad, 30 good, 3 deliver, 1
ambiguous, 10 overdecline (`runs/p4/margin_scores.json` `labels`).

**Measured** (axis = the contrast mean `runs/p4/axes-contrast.pt`, scale 1.0):

| footprint | layers | harmful mean drop | harmless mean drop | delta (bad − good) | prefills | generated |
|---|---|---:|---:|---:|---:|---:|
| full (0–27) | 28 | −6.171875 | +2.31875 | **−8.490625** | 160 | 0 |
| sep (spanfit band) | 28 | −6.174479 | +2.3125 | **−8.486979** | 160 | 0 |
| control (14 lowest-snr) | 14 | −3.545139 | +1.880208 | **−5.425347** | 160 | 0 |

Source: `runs/p4/margin_scores.json` `footprints` / `deltas` (committed copy
`verification/p4d/margin_scores.json`), and the per-footprint `summary` blocks of
`runs/p4/margin_scores-{full,sep,control}.json` (same files in `verification/p4d/`).
`drop` is `margin_before − margin_after`; the artifact's sign line reads "NEGATIVE
delta = the temporary projection lifts Sure relative to cannot more on harmful rows
than on harmless ones".

The full and sep footprints name the same 28 layers and were scored twice in
separate model loads; the difference between them is 0.003646, so differences below
about 0.004 are run-to-run numerics, not a layer effect
(`runs/p4/margin_scores.json` `identical_footprint_repeat`). The 28-layer band sits
3.0653 below the 14 lowest-snr control band on the same statistic — a comparator,
not a null edit: the control edits half as many layers, so part of that gap is the
doubled edit surface (`runs/p4/margin_scores.json` `definitions.control`).

**May claim.** On this capture and this axis, the projection moves the
Sure−cannot margin toward `Sure` on harmful rows more than on harmless rows, and
the 28-layer band does so more than the 14 lowest-snr layers (−8.490625 vs −5.425347,
against a 0.003646 repeat gap). That is a differential ranking of footprints, and of
candidate axes when each is scored the same way on the same capture.

**May not claim.** Removal, safety, or per-label behaviour. The statistic is a
*difference*: on this run the harmful rows' margin rose (mean drop −6.171875) while the
harmless rows' margin fell (mean drop +2.31875), and a large negative delta is
producible that way. No token was generated and no cert wave ran, so a margin win is
a hint about where the axis points, never evidence the merged artifact behaves
(METHODOLOGY R2/R6). This run also scored one axis only (the contrast mean); it did
not rank RDO against contrast — that arbitration happened in the ladder, not here.
A margin-selected arm's removal is UNMEASURED: no cert wave was run for a
margin-only selection.

## 2. RDO (gradient-refined direction)

**What it measures.** Per layer, the contrast mean `d0` is refined by gradient
ascent on a linear refusal probe's own class-separation objective, with the top-k
PCA subspace of the mean-centred good rows projected out and `d` renormalized
(`jdewitt/rdo.py`). It is the paper's *idea* (arXiv:2502.17420), not its operator
and not an exact reproduction; the report says so in its `scope` field. CPU-side,
float64, deterministic, no model load.

**Command** (P4d/P4e artifact):

```sh
python -m jdewitt.rdo fit --records runs/p4/records-rdo-none.pt --holdout none \
  --out runs/p4/axes-rdo.pt --report runs/p4/axes-rdo-report.json
```

**Cost.** 0 generated tokens and 0 prefills: it consumes saved capture records and
writes an axis tensor, loading no model at all. The report records 200 refinement
steps, lr 0.05, `protect_k` 8, a 400-step logistic probe, axes `[28, 2048]`
(`runs/p4/axes-rdo-report.json`). One honest caveat from the same report:
`probe_underdetermined_layers` lists **all 28 layers** — 36 bad and 30 good rows at
width 2048 do not identify a 2048-parameter probe, so the refined direction is a
ranking hypothesis, not a fitted estimator with identified parameters.

**Measured, ppl rung (P4e, adapter arms, scale 1.0, zero generated tokens):**

| arm | drift_ln | delta_br | ladder noise | generated |
|---|---:|---:|---:|---:|
| contrast mean (A1) | 0.2166 | −0.3319 | 0.1446 | 0 |
| RDO (A3) | 1.259 | −0.7385 | 0.1446 | 0 |

Source: `verification/p4f/rerun/tune-a1.json`, `verification/p4f/rerun/tune-a3.json`
(committed copies of the P4e rerun; `drift_noise_ln`, `generated_tokens`, `arms`).
At the same strength the RDO arm drifts 1.0424 ln more on the control corpus and its
delta_br moves away from the removal branch, so the ladder drops RDO as an edit
candidate.

**Measured, merged-bytes rung (P4f, Q4_K_M, same corpora):**

| arm | scale | control PPL | drift_ln | delta_br | cert: removed (attributable) | cert: harmless retained |
|---|---:|---:|---:|---:|---:|---:|
| base | – | 23.1502 | 0 | 0 | 0/34 | 30/30 |
| contrast | 1.0 | 25.8512 | +0.1104 | +0.0359 | 30/34 | 29/30 |
| RDO | 1.0 | 33.8421 | +0.3797 | +0.1015 | 4/34 | 23/30 |
| RDO | 0.20 | 23.1695 | +0.0008 | +0.0594 | 1/34 | 27/30 |

Sources: `verification/p4f/ladder-merged.json` (`metrics`, `drift_noise_ln` 0.1485,
`generated_tokens` 0) and `verification/p4f/selection.json` `cert` (counts; full
per-row packets in `verification/p4f/cert-{base,contrast,rdo,rdo-s020}.json`). The
0.20 row exists only to force equal delta_br: the difference from contrast's
+0.0359 is 0.0235, inside the 0.05 rule.

**Verdict.** At 1.0 the arms are not at equal removal (delta_br difference 0.0656 >
0.05) and RDO's drift is 0.2693 higher — it loses on both axes. At 0.20 the
delta_br difference is 0.0235 (matched) and RDO's drift is 0.1096 lower, but
0.1096 < 0.1485, the ladder's own noise, so the drift win is **UNMEASURED**; and the
cert lane removes 1/34 attributed harmful rows against contrast's 30/34, so equal
delta_br was not equal removal. The gate is UNMEASURED: no claim. RDO stays
`researched-not-implemented` in `jdewitt/methods.py`, with the falsifier path
`runs/p4/selection.json` in its notes.

Two context numbers, so the merged ladder is not over-read: both arms' realised
edits sit below the 0.9 minimum-edit-cosine floor on all 42 Q4_K tensors (contrast
mean cos 0.4220, RDO 0.4516; `verification/p4f/fidelity-{contrast,rdo}.json`,
`summary.per_type`), a labelled CODEC event — the codec decides what ships at
strength 1.0, not the axis. 170 of 226 tensors are bit-identical to the pristine
artifact and 56 differ, 0 shape/size/type changes
(`verification/p4f/tensor-diff-contrast.json` `summary`).

## 3. Separability layer score

**What it measures.** A closed-form read of saved capture records: per layer, how
far apart the harmful and harmless label means point (cosine) and how large that gap
is next to each label's own spread (SNR). The output is a ranked layer order plus
the band `tune --footprints` can take, at a keep line of `keep × max snr`
(`jdewitt/spanfit.py`, `layer_separability`).

**Command:**

```sh
python -m jdewitt.spanfit layers --records runs/p4/records.pt --out runs/p4/layer_scores.json
```

**Cost.** 0 generated tokens and 0 prefills: the command reads `records.pt` and
writes JSON; it never loads the model ("a read of records already on disk, not a
model run", `jdewitt/spanfit.py`).

**Measured** (`runs/p4/layer_scores.json`):

| quantity | value |
|---|---|
| layers scored | 28 (width 2048) |
| rows | 36 bad / 30 good |
| keep | 0.5 |
| snr min / max / spread | 0.462245 / 0.778952 / 0.316706 |
| keep line | 0.389476 |
| emitted footprint | 0–27 |
| flat | false |
| ranked head | 10, 13, 9, 12, 14, 11, 8, 15, 16, 17, 7, 18, 20, 6, 21, 19, 2, 23, … |

**May claim.** A reproducible per-layer ranking and a footprint, both derived only
from the capture; on this lineage it emits `0–27`, i.e. every captured layer. The
ranking is what the margin run used as its "sep" footprint, and the 14 lowest-snr
layers (`0–5, 19, 21–27`) are the honest control stand-in
(`runs/p4/margin_scores.json` `definitions.control`) because the band has no
complement.

**May not claim.** That the band is the right footprint, or that the band beats its
complement at removal. `flat: false` with every layer above the 0.389476 keep line
means the emitted band spans the whole capture and has no complement: the selector
did not restrict anything here. Its only measured discrimination is the margin
control gap in section 1 (−8.490625 vs −5.425347). A cert comparison of band vs
complement is UNMEASURED — there is no complement to certify.

## Why margin and separability are not in `jdewitt.cli methods`

The catalogue lists edit methods by the entrypoint that writes weights (`run` for
projection methods, `train` for SFT/DPO); the margin and the layer score are
read-only selectors over an axis or a layer set, with no weight-writing entrypoint,
so they have no `status` to carry and are documented here instead of being given a
misleading `researched-not-implemented` row.

## What is UNMEASURED on this lineage

- Lower drift at equal removal for any arm other than the contrast mean, beyond the
  ladder noise — the P4f done gate is UNMEASURED (0.1096 lower, inside the 0.1485
  noise, with the cert lane removing 1/34 vs 30/34).
- Removal for a margin-selected arm, or for the separability band against its
  complement — neither was ever merged and certified.
- Whether the margin ranking predicts merged bytes at all — the margin scored the
  same axis the ladder shipped, so it never predicted a different arm's outcome.

## Reproduce

Interpreters and paths as in `verification/p4f/README.md`. From the worktree root:

```sh
python -m jdewitt.spanfit layers --records runs/p4/records.pt --out runs/p4/layer_scores.json
python -m jdewitt.rdo fit --records runs/p4/records-rdo-none.pt --holdout none \
  --out runs/p4/axes-rdo.pt --report runs/p4/axes-rdo-report.json
python -m jdewitt.margin score --model /path/to/Spark-X2.6-1.7B \
  --records runs/p4/records.pt --axis-file runs/p4/axes-contrast.pt \
  --footprints 0-27 --out runs/p4/margin_scores-full.json --trust-remote-code
python scripts/p4f_selection.py --root . --out runs/p4/selection.json
```

## Citation index

Every number in this page resolves to one of these artifacts, with the committed
copy named where one exists:

| artifact | numbers it supports |
|---|---|
| `runs/p4/margin_scores.json` (copy `verification/p4d/margin_scores.json`) | capture labels 36/30/3/1/10 of 80; 160 prefills per footprint, 480 total; the three deltas and their bad/good means; control definition; repeat gap 0.003646; `usable` check |
| `runs/p4/margin_scores-{full,sep,control}.json` (copies in `verification/p4d/`) | per-footprint `summary`, `prefill_count` 160, `tokens_generated` 0 |
| `runs/p4/layer_scores.json` (copy `verification/p4d/layer_scores.json`) | 28 layers, width 2048, keep 0.5, snr min/max/spread, keep line 0.3895, footprint 0–27, `flat: false`, ranked order |
| `runs/p4/axes-rdo-report.json` (artifact store only; regenerated by the `rdo fit` command above, and its sha256 is recorded in `verification/p4f/selection.json` `selectors.rdo_axes`) | steps 200, lr 0.05, protect_k 8, probe 400 steps, axes [28,2048], `probe_underdetermined_layers` |
| `verification/p4f/rerun/tune-a1.json`, `tune-a3.json` | P4e drift 0.2166 / 1.259, delta_br −0.3319 / −0.7385, noise 0.1446, generated 0 |
| `verification/p4f/ladder-merged.json` (`runs/p4/ladder-merged.json`) | base 23.1502; contrast 25.8512 / +0.1104 / +0.0359; RDO 33.8421 / +0.3797 / +0.1015; RDO@0.20 23.1695 / +0.0008 / +0.0594; noise 0.1485; generated 0 |
| `verification/p4f/selection.json` (`runs/p4/selection.json`) | equal-removal definition 0.05; drift gap 0.1096; cert counts and packet paths; `generated_tokens.selection` 0 |
| `verification/p4f/cert-{base,contrast,rdo,rdo-s020}.json` | removal/retained counts, boundary 0, generated tokens 10593/12432/11709/10650 |
| `verification/p4f/fidelity-{contrast,rdo}.json` | 56 tensors, 42 Q4_K, 14 Q6_K, floor 0.9, mean-cos 0.4220 / 0.4516, min/mean 0.2204/0.5336 and 0.2821/0.5577 |
| `verification/p4f/tensor-diff-contrast.json` | 226 shared, 170 bit-identical, 56 differ, 0 shape changes |

Derived values, each a difference of two cited numbers: 0.0656 = 0.1015 − 0.0359,
0.0235 = 0.0594 − 0.0359, 0.2693 = 0.3797 − 0.1104, 1.0424 = 1.259 − 0.2166,
3.0653 = 8.4906 − 5.4253.
