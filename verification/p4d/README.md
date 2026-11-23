# P4d — real capture with margins, margin rescoring, layer scores (Spark-X2.6-1.7B)

First real measurement of the two zero-generation selectors on this lineage: the
layer-separability band (`spanfit layers`) and the token-logit Sure-vs-cannot margin
(`margin score`). No cert wave (METHODOLOGY R2/R9) and no removal claim: the margin
ranks candidate axes, it never certifies.

## Sample

`verification/harsh-suites/selection.jsonl` sha256
`d21a9d881befcbe90f57f032511b2a7259acdd035845923022339330ffdbe143`, 1340 rows
(1004 refusal, 330 harmless). All suite heads are `refusal`, so `--limit` alone yields
one label; the sample is stratified instead:

- refusal rows `[::25][:40]` (every 25th, first 40)
- harmless rows `[::8][:40]` (every 8th, first 40)

80 rows; `suite-80.jsonl` sha256 `1f3ab3cdfc1021bd38209c9ef4c13f3b9ded5eec381b95ad9319ba6fd1b73846`.
`suite-80.meta.json` carries the source path, source sha256 and the stride rules.

## Capture

```
python -m jdewitt.cli capture --model /mnt/ssd/models/Spark-X2.6-1.7B \
  --suite runs/p4/suite-80.jsonl --kinds refusal,harmless --out runs/p4/records.pt \
  --gguf /mnt/ssd/models/Spark-X2.6-1.7B-Q4_K_M.gguf --ctx-size 2048 \
  --max-tokens 512 --settle-min-tokens 24 --trust-remote-code --allow-cuda
```

`--trust-remote-code` is required: Spark2_5 ships custom modeling code and the load
refuses without it. Backend chosen: **xpu, rank primary, "Intel(R) Arc(TM) Pro B70
Graphics"** — this host has no CUDA device, so `--allow-cuda` was passed but not used
(CUDA is the second-rank test path, `jdewitt/hf.py`). 608.8 s; 80 of 80 rows written,
0 discarded.

| labels | bad | good | deliver | ambiguous | overdecline |
|---|---:|---:|---:|---:|---:|
| rows | 36 | 30 | 3 | 1 | 10 |

`fit_usable: true` (36 bad / 30 good, gate is >= 10 each). Tokens: prompt 12880,
answer 15256. Margin resolution: 4 single-token pairs (` Sure` 38289, `Sure` 48353;
` cannot` 6847, `cannot` 87753); margin available on all 80 rows; ambient
`delta_bad_minus_good = -7.475` (bad mean -1.724, good mean 5.751) — the model's own
Sure-minus-cannot prefill margin already separates the labels. GGUF cross-check: 28 of
28 writeback layers found, none unmapped, `agree: true`.

Reference (not a target): the banked 50-row capture on this lineage was
38 bad / 10 good / 2 ambiguous.

## Layer separability (`spanfit layers`)

```
python -m jdewitt.spanfit axes --records runs/p4/records.pt --out runs/p4/axes-contrast.pt --shape perlayer
python -m jdewitt.spanfit layers --records runs/p4/records.pt --out runs/p4/layer_scores.json
```

axes: `[28, 2048]` perlayer, fitted on the bad-minus-good capture means (1.1 s).

layers: rows bad 36 / good 30, 28 layers, width 2048, keep 0.5 x max snr 0.778952 =
keep line **0.389476**; snr min **0.462245**, max **0.778952**, spread **0.316706**;
emitted footprint **`0-27`**, `flat: false`. Top layers by rank: 10 (0.779), 13
(0.768), 9 (0.743), 12 (0.743), 14 (0.725), 11 (0.724).

Every layer clears the keep line, so the emitted band is the **whole captured span**:
`flat=false` (the score does vary, 0.46–0.78) but the band still has no complement —
the band does not localise a sub-range on this lineage, it says the split is present at
every layer, deepest in the middle.

## Margin rescoring (`margin score`, prefill only)

Three footprints, answers reused from the records, zero generated tokens:

```
python -m jdewitt.margin score --model /mnt/ssd/models/Spark-X2.6-1.7B \
  --records runs/p4/records.pt --axis-file runs/p4/axes-contrast.pt \
  --footprints <layers> --out runs/p4/margin_scores-<name>.json --trust-remote-code
```

| footprint | layers | delta_bad_minus_good | bad drop | good drop | prefills |
|---|---|---:|---:|---:|---:|
| full | 0-27 (28) | **-8.490625** | -6.171875 | +2.318750 | 160 |
| sep | 0-27 (28) | **-8.486979** | -6.174479 | +2.312500 | 160 |
| control | 0-5,19,21-27 (14) | **-5.425347** | -3.545139 | +1.880208 | 160 |

`drop = margin_before - margin_after` on the prefill margin (logit Sure - logit
cannot); delta is mean drop on harmful rows minus mean drop on harmless rows. A
NEGATIVE delta means the temporary projection **lifts Sure relative to cannot more on
harmful rows than on harmless ones**. Hooks are removed in `finally`
(`hooks_removed: true`), weights are never written, 480 prefills total, 0 generated
tokens.

- `best_footprint: full` (most negative delta; null would have meant none negative).
- Full-span delta is negative, so by the card's rule the margin selector is usable on
  this lineage — recorded as measured, not as a removal claim.
- sep is not worse than control (-8.487 vs -5.425): the separability band carries
  roughly 1.5x the effect of the low-separability half.
- **Control definition caveat:** with the emitted band covering 0-27, the strict
  complement is empty. `control` is therefore the complementary half *by separability
  rank* — the 14 lowest-snr layers (0-5,19,21-27) — and is labelled that way in
  `margin_scores.json`. It is a comparator, not a null edit.
- **Noise floor:** full and sep name the same 28 layers and were scored in separate
  model loads; their deltas differ by 0.003646. Differences below ~0.004 on this
  backend are run-to-run numerics, not a layer effect.
- **Cross-check:** `margin_before` for every row matches the capture's ambient margin
  from the same position (correlation 0.99995, mean abs diff 0.0133, max 0.25 over 80
  rows), so the rescoring pass and the capture read the same statistic.

## Verdict

The full-span margin delta is negative (-8.491), so the margin is usable as a ranking
selector here; the sep band is at least as strong as the control half. The layer band
does not localise (footprint 0-27) — on this lineage, with this capture, the
harmful/harmless split is everywhere and strongest mid-stack. No falsifier was
recorded. No merged serve, no cert wave, no removal claim.

Artifacts: `runs/p4/` holds `records.pt` (18.6 MB), `axes-contrast.pt`,
`layer_scores.json`, `margin_scores-{full,sep,control}.json`, `margin_scores.json`
(the aggregate: three deltas, all 80 per-row margins per footprint, `best_footprint`)
and the capture report. This directory holds the text evidence; `records.pt` and the
axes file are deliberately not committed (repo `.gitignore` covers `runs/` and `*.pt`).

## Reproduce

Use the project interpreter (the plain `python` on this box has no numpy/torch):
`/mnt/ssd/dev workspace/JdeWitt/.venv/bin/python` with `PYTHONPATH` at the worktree
root, from the worktree root. The sample is deterministic; the capture and the three
margin runs are the only steps that touch the GPU (B70).
