# SAE compute plan (P5 workstream b, first train step)

Status: authored by t_fc23e61a. The card `t_10d92751` ("the compute plan first")
was closed `done` with a gateway-timeout result and its branch carries no commits,
so the plan this card's gate refers to did not exist; the numbers below are the
plan, priced on the B70 (`scripts/sae_budget.py --timing`) before training.

What is trained: one small **top-k autoencoder** per non-protective content
family, on that family's captured refusal residuals at one layer
(`capture` answer locus, `verification/p5/sae/inputs/`). The edit half of the
lane is `jdewitt/sae.py` and is unchanged: a feature is only a *new* thing if its
decoder direction is not the contrast axis, which is why every feature in the
table carries `cos_to_axis`.

Why top-k rather than L1: the plan's feature budget is a work statement, and a
top-k SAE keeps the active set at exactly `k`, so the step FLOPs below are the
actual work rather than an expectation over a penalty sweep.

## Budget (machine-readable: `scripts/sae_budget.py --compare` re-reads this block)

<!-- sae-budget:start -->
activation_bytes_per_row = 229376
batch = 16
families = 4
features = 128
generated_tokens = 0
gpu_hours_total = 0.002881
gpu_seconds_per_family = 2.593
gpu_seconds_total = 10.37
hidden = 2048
k = 2
layers = 28
merge_cert_seconds = 540
rows = 158
step_flops = 25165824
step_rate_source = measured
steps = 1200
steps_per_second = 462.862
stop_rule_fires = False
stop_rule_gpu_seconds_per_family = 540
train_matrix_bytes = 1048576
<!-- sae-budget:end -->

Readings:

- `activation_bytes_per_row = layers * hidden * 4`: one captured row is the
  whole 28-layer residual stack at the answer locus, fp32 in memory. 229,376 B.
- `train_matrix_bytes = features * hidden * 4`: one dictionary. 1,048,576 B
  (1 MiB) at 128 features -- a rounding error next to the model, which is why
  compute is not what binds this lane (see Capacity).
- `step_flops = 2 * batch * features * hidden * 3`: three dense matmuls per row
  (encode, decode, backward at ~2x forward). 25.2 MFLOP per step.
- `gpu_hours_total`: every family at the plan's step count on the measured rate:
  0.0029 GPU-hours, orders below the model work this repo already does.
- `rows = 158`: labelled rows that enter the dictionaries (126 refusal + 32
  compliant). The suite reads 168; the 10 rows the capture labelled
  `ambiguous`/`overdecline` never enter the fit.
- `generated_tokens = 0`: **R1.** SAE training is device time, not inference
  volume. The capture that produced the rows is priced separately by the card
  (its own token count, `inputs/records.pt.capture.json`); nothing in this lane
  decodes.

## Stop rule (the number)

A family is **not trained** when its own training is priced above the artifact's
own merge+cert cost: `gpu_seconds_per_family > merge_cert_seconds`, where
`merge_cert_seconds = 540` (ROADMAP phase 2 records ~9 min per merged arm, and
`verification/p4f/` re-measured a merged build + one cert wave in that envelope).
`stop_rule_fires` in the block above is that comparison on the measured rate:
2.6 s against 540 s. A stop is reported as **unmeasured**, never as a bad
result: the lane ran out of budget, not out of signal.

The training driver (`scripts/p5_sae_train.py`) re-prices the rule *before the
first gradient step* with a 40-step probe on the largest family and exits 2
without training if the projection fires, so a slower device cannot silently
blow the budget.

## Capacity: what the row count, not the budget, allows

A dictionary has to be identifiable from 24 train rows per family against a
2,048-wide residual stream, so capacity is chosen on held-out FVU, not on what
the table can afford. `scripts/p5_sae_sweep.py` measures the candidates on the
same split and seeds the report uses
(`verification/p5/sae/capacity-sweep.json`):

| features | k | families where median FVU beats the rank-k control |
|---:|---:|---|
| 32 | 2 | 3/4 |
| 64 | 2 | 3/4 |
| 64 | 4 | 2/4 |
| 64 | 8 | 1/4 |
| **128** | **2** | **4/4** |
| 128 | 4 | 2/4 |
| 256 | 2 | 3/4 |
| 256 | 8 | 0/4 |
| 512 | 2 | 4/4 |

Rule, stated before the pick: among the configurations that win in all four
families, take the **smallest** dictionary -- it wins at the same margins with
less capacity, and the larger active sets (k>=4) lose to the rank-k control
because a bigger k gives the control more to match. The plan's budget
(`features=128, k=2`) is therefore set by identifiability, and the compute block
above is priced at exactly that shape.

Two honesty notes on the pick. First, the family that decides 64-vs-128 is
`hb-std-sel-illegal`, whose three seeds span 0.84-0.94 against a control of 0.909:
its pass is inside the seed noise, and the report prints the spread next to the
number. Second, the same sweep on an fp16 copy of the residual matrix put 64
features at 4/4; the matrix is stored fp32 now because an fp16 copy is lossy
against the captured bf16 values and moved a family's FVU across the baseline
rule -- which is exactly the kind of 0.01 decision this table should not turn on.

## Baseline rule for a family's held-out FVU

FVU is the fraction of variance unexplained in raw units, so `1.0` is exactly
"predict the train mean". A family's SAE **stops** if
`fvu >= 1.0` or `fvu >= pca_control_fvu`, where the control is the best
rank-`k` linear reconstruction (SVD on the train split, `k` = the SAE's active
features). A top-k SAE with `k` active features is a rank-at-most-`k`
reconstruction, so failing to beat the same-rank linear control means the
dictionary bought nothing over a projection.

The published FVU is the **median over three seeds** (20261123/4/5), and the
published checkpoint is the median-FVU seed: with 7-8 held-out rows a single
seed's FVU is noise, and publishing the best seed would be selection on the test
set. Per-seed values and the spread sit next to every family in the report.

Sample size is the honest limit of this plan: 24 train rows per family. The
dictionaries are identifiable at k=2 and not above; the over-complete dictionary
this lane is *for* needs a corpus of thousands of rows, which is a data-sourcing
task, not a compute one.

## Artefacts

| artefact | what it is | where |
|---|---|---|
| capture suite | 4 non-protective families x 32 refusal rows + 40 harmless, deterministic stride, sha256 in the meta file | `verification/p5/sae/inputs/suite.jsonl` + `suite.meta.json` |
| capture records | `[28, 2048]` per row at the answer locus, labelled by answer shape, with the capture identity | `verification/p5/sae/inputs/records.pt` (+ `.capture.json`) |
| residuals | the `[158, 2048]` matrix at the plan's layer, with row metadata and the record digest | `verification/p5/sae/residuals-layer10.npz` + `.meta.json` |
| checkpoint | per-family `W_enc`/`W_dec`/`b_enc`/`b_dec` + standardiser (fp16 `.npz`), weight digests | `verification/p5/sae/checkpoints/` |
| residual matrix | `[158, 2048]` fp32 (exact widening of the captured bf16, never the lossy fp16 copy) | `verification/p5/sae/residuals-layer10.npz` |
| config + seed | plan numbers, split ids, seeds, median choice | checkpoint config block + `train-report.json` |
| capacity sweep | every candidate's per-family, per-seed FVU and its control | `verification/p5/sae/capacity-sweep.json` |
| feature table | top-k features by `|contrast|`, per-family means both sides, `cos_to_axis` | `verification/p5/sae/feature-table.json` / `.md` |
| ablation stage (next card) | pre-registers from the feature table | t_35ced455 |

## Measured vs assumed

- **Measured**: steps/second on the target GPU (`--timing`, 20-step warm-up
  excluded), projected per-family GPU seconds, the actual per-family training
  seconds in `train-report.json`. The rate in the block above (462.862 steps/s)
  is the 2026-11-23 run of
  `uv run python scripts/sae_budget.py --features 128 --k 2 --rows 158 --timing --timing-steps 400 --device xpu`
  on `Intel(R) Arc(TM) Pro B70 Graphics`, with the deterministic kernels the
  trainer uses. (Without `torch.use_deterministic_algorithms(True)` the same
  shape measured 667 steps/s but did not reproduce a digest, so the non-
  deterministic rate is the wrong one to price.) `--compare` re-reads the doc
  with `--rate 462.862`, so the diff does not depend on a fresh timing run's
  noise.
- **Quoted from a doc**: `merge_cert_seconds = 540` (ROADMAP phase 2, ~9 min per
  merged arm; the P4f run's own console is the nearest re-measurement).
- **Assumed**: none in the block above -- `step_rate_source = measured`. A run
  without `--timing` prints `ASSUMPTION` next to every GPU number, and
  `--compare` fails if the label and the rate disagree.

## What this does not establish

Nothing about removal, delivery, or refusal behaviour. The SAEs are a feature
basis over captured residuals; the card claims no edit, no merge and no cert
wave. A feature table entry with a large |contrast| is an input to a
pre-registered ablation, not a result.
