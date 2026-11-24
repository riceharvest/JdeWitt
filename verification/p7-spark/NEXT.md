# NEXT — P7 Spark-X2.6-1.7B Q4_K_M, analysis card

The wave card (`t_0ad8db9b`) runs `verification/p7-spark/run_waves.sh` under the b70
lock; this card does the judging. No model, no GPU, no wave here.

## 0. Check the run actually finished (do this first)

    cd "/mnt/ssd/dev workspace/JdeWitt"
    tail -3 verification/p7-spark/progress.txt          # last line must say FINISHED
    uv run python -c "import json;d=json.load(open('verification/p7-spark/waves.json'));print(d['all_measured'],d['rows_total'],d['completion_tokens_total']);[print(w['label'],w['rows'],w['max_tokens'],w['measured']) for w in d['waves']]"

Every wave the spec uses must be there and measured:

| wave | suite | rows | cap |
|---|---|---|---|
| `waves/removal-arm` | `verification/p7-spark/suites/harmful-40.jsonl` (restored copy of advbench-40) | 40 | 4096 |
| `waves/heldout-120-arm` | `verification/screens/q3-holdout-wider.jsonl` | 120 | 4096 |
| `waves/retention-arm` | `verification/capability/capability-run-en.jsonl` | 241 | 1024 |
| `waves/retention-base` | same | 241 | 1024 |

All four at `-c 4096 --parallel 4`, b70 = `Vulkan1` (the RTX 3060 is banned).
Supplementary (not certifiable axes): `waves/removal-base` (40 rows, baseline for the
removal count) and `waves/heldout-24-arm` (24 families; `release-certify` refuses a
24-row wave as `CERTIFY_PROBE_ONLY`, which is why the 120-row suite is the axis).

If a wave is missing, `ab fail` naming it — the GPU card re-runs it.

## 1. Certify (spec is already written by the wave script)

    cd "/mnt/ssd/dev workspace/JdeWitt"
    uv run jdewitt release-certify \
      --spec verification/releases/spark-1.7b/spec.json \
      --out  verification/releases/spark-1.7b/packet.json
    uv run jdewitt verify-artifact \
      verification/releases/spark-1.7b/packet.json --name packet.json

`run_waves.sh` wrote the same bytes to `verification/p7-spark/spec.json`; the
manifest's registered path is `verification/releases/spark-1.7b/spec.json`, so
certify from there.

The command reads saved waves only and refuses (non-zero, no packet) on a missing
axis, a wave not served at the pinned argv (`CERTIFY_CONFIG_MISMATCH`), an
unmeasured wave, a one-inventory removal count, a retention wave with no paired
stat or median, or a shape audit of another artifact. A refusal is a finding to
report, not something to patch around.

## 2. Register the arm in the manifest, then count the matrix

The `spark-1.7b` entry in `verification/releases/matrix.manifest.json` still has
`artifact.sha256: null` / `identity_digest: null`. Fill it from the packet so the
row binds the bytes it claims (this is the edit the q3 row already has):

    python - <<'PY'
    import json
    path = 'verification/releases/matrix.manifest.json'
    manifest = json.load(open(path))
    packet = json.load(open('verification/releases/spark-1.7b/packet.json'))
    for row in manifest['lineages']:
        if row['lineage'] != 'spark-1.7b':
            continue
        row['artifact'].update({
            'sha256': packet['artifact_sha256'],
            'identity_digest': packet['artifact_identity_digest'],
            'quant_type': 'Q4_K_M',
            'scale': 1.0,
            'edited_tensors': packet['arm']['edited_tensors'],
            'bytes': packet['arm']['byte_cost'],
            'footprint': packet['arm'].get('edited_tensor_footprint'),
            'note': ('edited in place-style on the served Q4_K_M (base bytes untouched, arm '
                     'written beside it); the packet binds the full sha256 and the merge recipe'),
        })
    with open(path, 'w') as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
        handle.write('\n')
    PY

    uv run jdewitt release-matrix \
      --manifest verification/releases/matrix.manifest.json \
      --out      verification/releases/matrix.json \
      --table    verification/releases/matrix.md \
      --doc      docs/EVIDENCE-TABLE.md

Then read `rows[?].status` for spark-1.7b. `certified` means the pinned argv
matched and every registered band cleared; `failed(<axis>)` carries the measured
number that missed a band; `invalid(<code>)` means a structural problem (fix it
and re-count). Do not round a `failed` up to a pass.

## 3. What the packet must contain, and the one branch that changes the verdict

- `arm.chosen_type` and `arm.realized_cos` come from `fidelity.json`
  (`compare_edit` over all 56 edited tensors of the merged bytes), not from the
  codec probe.
- `arm.codec_table` points at `verification/p7-spark/codec-table.json`: the
  measured per-type table (`preview_edit` on the pristine artifact, every edited
  tensor). Scale stays 1.0 in every arm — the type is the codec-aware lever.
- `realized_fidelity.tensors` is the per-tensor table the gate reads. Every
  edited tensor must be **>= 0.90**.

**If `codec-table.json.verdict` is `failed(codec)`** (no candidate type clears
0.90 on every edited tensor), or `fidelity.json.codec_event` is true: the row is
`failed(codec)`. Print the measured per-tensor cos values and the control drift
and do not claim a pass. That is the result the spec asks for; do not lower the
floor and do not raise the scale to buy clearance (the documented counterexample
is scale 6.61 at +10.2 nats control PPL).

Pre-measured on this box before the wave card (committed as an operand, so the
analysis can cross-check the wave card's numbers):

    q6_k  min 0.8004  mean 0.8731   -> below the floor (this is the codec floor)
    q8_0  min 0.9861  mean 0.9899   -> chosen
    f16   min 1.0000  mean 1.0000

`verification/p7-spark/evidence/q6k-merge-probe.json` is the delivered-byte
check at Q6_K (2 writer-probe tensors: attention 0.7766, FFN 0.8713) — the same
floor seen on the merged file, not just in the codec simulation.

## 4. Supplementary evidence to report beside the row (not certifiable axes)

- `waves/removal-base`: the pristine artifact's harmful-40 answers. The README's
  old "refusal openers gone 40/40" row has no artifact and is void; this is the
  baseline the arm's count is a delta from. Read both inventories off it with the
  same checker (`release-certify` grades the saved answers; for the base wave use
  `scripts/capability_wave.py`'s `refusal_check`/`refusal_check_wide` fields or
  `scripts/refusal_shape_audit.py`).
- `waves/heldout-24-arm`: 24 rows over 24 families, disjoint from the fit pool
  (`runs/p4/suite-80.jsonl`). Report it as a probe-size reading, never as the
  120-family certification.
- `shape/removal.json`, `shape/heldout.json`: refusal shape (R6). Wording removal
  is not delivery; the shape verdicts sit beside the removal count.

## 5. Honest gaps this row carries (state them, do not paper over)

1. **Capture source.** The spec says capture on the f16 twin; the repo capture
   path loads HF checkpoints through transformers, not GGUF. The records are the
   p4d capture of the **bfloat16 HF source** (`/mnt/ssd/models/Spark-X2.6-1.7B`)
   that the f16 twin was converted from. The f16 twin is the shape/writeback
   cross-check, not the load path. The axis is a **refit** in this card from those
   records (`steps/axes.json`, bitwise identical on re-fit).
2. **Port/device are part of the serve config.** `-dev Vulkan1` is the b70
   (Intel BMG G31); Vulkan0 is the banned RTX 3060. The recorded argv includes
   both, so a wave taken elsewhere is detectable, not silently comparable.
3. **The 24-family reading is a probe.** The registered held-out floor is 120
   rows; the 120-row wave is the axis.
4. **Base removal is not an axis.** It is baseline evidence; the packet's removal
   block is the arm's.

## 6. How the prep was verified (re-run these if anything looks off)

    bash -n verification/p7-spark/run_waves.sh
    bash verification/p7-spark/run_waves.sh --dry        # prints the argv list only
    uv run python scripts/p7_spark_waves.py operands \
        --manifest verification/p7-spark/operands.json   # sha256 every operand
    uv run ruff check jdewitt scripts

Note: `.venv/bin/ruff` does not exist in this worktree; use `uv run ruff`.
