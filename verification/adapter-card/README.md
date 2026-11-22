# adapter-card evidence

`twin-spark-perlayer-10-27.json` is a real merged-twin fidelity record: the input
`adapter-card` consumes. It is kept so the R8 card can be reproduced without the
1.1 GB twin artifact, which lives outside the clone; the record names that
artifact by sha256.

## Where each fact comes from

- `artifact_sha256` = `3380d89d…` of `merged-twin.gguf`, built by merging all 36
  `blk.{10..27}.{attn_output,ffn_down}.weight` deltas of the rank-1 adapter
  `adapter-perlayer-10-27.gguf` into `Spark-X2.6-1.7B-Q4_K_M.gguf`
  (sha `80ff5222ae51de055fe2e86b1ec4f6784c2f9db704ed3b043b97ce900d3a75ad`)
  with `ggufio.edit_tensors`.
- `delivered.tensors` = per-tensor `cos` / `snr` from `ggufio.compare_edit`
  (pristine base vs the merged twin, full-tensor decode) against the projection
  the adapter encodes. 36 rows; Q4_K 27 tensors cos 0.198 / 0.331 / 0.467
  min/median/max and SNR 0.938 / 0.979 / 1.044; Q6_K 9 tensors cos 0.808 /
  0.837 / 0.881 and SNR 1.551 / 1.683 / 1.974.
- `edited_tensors: 36`, `untouched_tensors: 190`: `gguf-tensor-diff` between base
  and twin reports 226 shared tensors, 190 bit-identical, 36 value-differing, no
  shape or dtype change.
- The behaviour pair `adapter_curve.delta_br 0.215` / `merged_bytes.delta_br
  0.119` (ratio 0.553) is copied from `docs/ROADMAP.md` "Where we are short"
  item 3 and `docs/PLAN-NEXT.md` "R8 verification of the shipped pick" (Spark
  1.7B Q4_K_M, identical 36-tensor coverage). It is NOT re-measured here; that
  needs a behaviour wave over both artifacts.

## Reproduce the card

```sh
# the adapter must carry jdewitt.base_artifact_sha256; tune writes it now, or
# rebuild a cached adapter through jdewitt.gguf_adapters.write_adapter(base_sha256=…)
uv run jdewitt adapter-card \
  --adapter A.gguf \
  --base /mnt/ssd/models/Spark-X2.6-1.7B-Q4_K_M.gguf \
  --twin verification/adapter-card/twin-spark-perlayer-10-27.json \
  --out /tmp/adapter.json
uv run jdewitt verify-artifact /tmp/adapter.json
```

Observed: `adapter_delta_br 0.215`, `merged_delta_br 0.119`,
`adapter_to_merged_ratio 0.553`, `ranking_valid_on_this_artifact false`,
`curve_is_not_merged_bytes true`, `merged_effect_status MEASURED`, and
`verify-artifact` returns `valid: true`. Removing
`curve_is_not_merged_bytes`, or copying `0.215` into `merged_delta_br`, exits 1
with `CURVE_CLAIM_MISSING` and `ADAPTER_DELTA_IN_MERGED_FIELD` respectively.
