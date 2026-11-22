# release-assemble evidence

`release-assemble` stages the phase-6 release: the LoRA adapter, the merged twin the
adapter curve is not, the digest-pinned plan card, and the labels that bind them —
then publishes the directory only after `verify_release` passes.

Everything here was produced by real CLI runs on the fixture built by
`fixture.py`. The fixture is **explicit placeholder bytes, not model weights and
not a fitted plan** (a 4×4 F32 GGUF base, a 4×4 twin with one tensor moved, a two
tensor LoRA bound to the base, and a two member plan-bundle with a 38-byte
payload). The only non-fixture numbers are the behaviour pair `delta_br` 0.215
adapter / 0.119 merged and the ratio 0.553, copied from `docs/ROADMAP.md`
"Where we are short" item 3; the record says so instead of implying a measurement.

## Reproduce

```sh
uv run python verification/release-assemble/fixture.py /tmp/rasfx
uv run jdewitt adapter-card --adapter /tmp/rasfx/adapter.gguf --base /tmp/rasfx/base.gguf \
  --twin /tmp/rasfx/twin-record.json --out /tmp/rasfx/adapter.json
uv run jdewitt release-assemble /tmp/rasfx/release \
  --adapter /tmp/rasfx/adapter.gguf --twin /tmp/rasfx/twin.gguf --plan /tmp/rasfx/plan.tar \
  --adapter-card /tmp/rasfx/adapter.json --evaluation /tmp/rasfx/evaluation.json \
  --readme /tmp/rasfx/README.md
uv run jdewitt verify-artifact --all /tmp/rasfx/release
uv run jdewitt verify-release /tmp/rasfx/release
```

## Observed (2026-11-22, CPU)

- `release-assemble` exit 0, `valid: true`, `files: 9` (the eight release files plus
  `evidence.json`, which the ledger appends), `release_identity`
  `ed6dd6bfa3b1ccd374bb117a566515ef349d9a0a634f9ec32302bb61a7f5e77e`
  → `release-assemble.json`, `verify-release.json`.
- `verify-artifact --all` exit 0, `valid: true`; `checked` carries
  `adapter.json`, `evidence.json`, `files.json`, `plan-card.json`, `twin.json`
  → `verify-artifact.json`.
- The eight release files are exactly the task list
  (`adapter.gguf`, `model-merged.gguf`, `plan-card.tar`, `adapter.json`,
  `twin.json`, `evaluation.json`, `README.md`, indexed by `files.json`) plus
  `plan-card.json`. The manifest is the ninth file on purpose: the phase-6 gate
  (`t_a44a4d22`) requires `verify-artifact --all` to check `plan-card.json`, while
  `plan-unbundle DIR/plan-card.tar` requires the bundle. Both are written from one
  manifest object, so they cannot drift at assembly time.
- `twin.json` (committed copy) carries the artifact sha256, edited/untouched
  counts (1 + 1 = 2 tensors), `edit_type_per_tensor` read from the shipped GGUF,
  per-tensor delivered cos/SNR, the edit fingerprint, and a behaviour block whose
  `serve_argv` is the argv pinned in `evaluation.json`. `measurement_ledger` labels
  the codec-realized numbers (`CPU byte-level`, requant fidelity) apart from the
  behaviour claims (`generation-scored`, `measured_where` names the merged bytes).
- Adding one file to the published directory (`unexpected.bin`) makes
  `verify-release` exit 1 with "Release file set/hash mismatch"; removing it
  returns exit 0 → `refusals.json` (`extra-file-after-release`).

## Refusals (all real runs, `refusals.json`)

| case | exit | code | directory published |
|---|---|---|---|
| twin only (`--twin` and nothing else) | 1 | `NO_PREQUANTIZED_MODEL_RELEASE` | no |
| `--twin` is a `.safetensors` | 1 | `NON_GGUF_EXPORT_ROADMAP` | no |
| adapter card base sha doctored | 1 | `ADAPTER_CARD_BASE_SHA_MISMATCH` | no |
| adapter card names another twin sha | 1 | `TWIN_ARTIFACT_SHA_MISMATCH` | no |
| plan-bundle payload tampered, manifest unchanged | 1 | `PLAN_CARD_PAYLOAD_MISMATCH` | no |

No refusal leaves a target directory; nothing is staged for these cases because
every label is checked before a staging directory is created.

The plan-bundle refusal is the "manifest payload sha256 != the bundle's payload"
case: the tar is rebuilt with the original `manifest.json` and different `plan.pt`
bytes. The card-base refusal is the "adapter-card base sha256" case: the card's
`base.sha256` no longer equals the `jdewitt.base_artifact_sha256` the adapter is
bound to, so the card would label bytes it was not fitted on. `twin.json`'s
`base.sha256` is the pre-merge artifact the adapter was fitted on (NOT shipped);
the state that binds the release is `adapter_card.merged_twin.artifact_sha256 ==
sha256(model-merged.gguf)`, also recorded in `twin.json.provenance`.

## Legacy shape

`create_release` / `verify_release` are untouched. `tests/test_auto.py` passes;
`tests/test_release_integrity.py` reports 8 passed, 1 failed
(`test_actual_console_verify_release_reads_only_files` asserts `files == 6` while
`evidence.json` makes the inventory 7). That failure reproduces on `master` before
this work (`git worktree` of `df9ca09`), so it is pre-existing, not caused here.
