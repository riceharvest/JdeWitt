# Phase-6 release gate evidence

Release directory (on disk, excluded from git because it is 2.2 GB of model
bytes): `verification/releases/spark-1.7b-q4km-p6`.

Every JSON in `gates/` is the stdout of one real command run in this worktree;
none was edited. `plan-apply-preserve.time.log` is the matching `/usr/bin/time -v`
stderr. The signing secret is `keys/release-signing-key.pem` (mode 0600, ignored
by git); only the public key `keys/release-signing-key.pub.pem` is published.

| gate | command | result | file |
|---|---|---|---|
| assemble | `release-assemble DIR --adapter … --twin … --plan … --adapter-card … --evaluation … --readme …` | exit 0, valid true, 9 files | `gates/release-assemble.json` |
| artifact schemas | `verify-artifact --all DIR` | exit 0, valid true, checked adapter.json + twin.json + plan-card.json | `gates/verify-artifact.json` |
| release | `verify-release DIR` | exit 0, valid true, 11 files | `gates/verify-release.json` |
| key | `release-keygen keys/release-signing-key.pem` | exit 0, Ed25519; public key PEM only above | `gates/keygen.json` |
| sign | `release-sign DIR --secret keys/release-signing-key.pem` | exit 0, repro_digest `60952e99…` | `gates/release-sign.json` |
| signature | `release-verify-signature DIR --public keys/release-signing-key.pub.pem` | exit 0, valid true, pinned-key authenticity | `gates/release-verify-signature.json` |
| repro index | `release-repro DIR --check` | exit 0, valid true, moved [] (only tool.git_describe provenance drift) | `gates/release-repro-check.json` |
| diff | `release-diff DIR verification/p6-release/competitor-fixture` | exit 0, unnamed [] and every named difference has a kind | `gates/release-diff.json` |
| unbundle | `plan-unbundle DIR/plan-card.tar --into /tmp/reg` | exit 0, installed, geometry layers 18 rank 1 hidden 2048 | `gates/plan-unbundle.json` |
| apply | `plan-apply --bundle DIR/plan-card.tar --gguf <base> --out /tmp/repro.gguf --edit-type preserve` | exit 0, 36 edited / 190 untouched byte-identical, min cos 0.2152 | `gates/plan-apply-preserve.json` |
| reproduction | `gguf-tensor-diff model-merged.gguf /tmp/repro.gguf` | 226/226 tensors bit_identical, differing_tensors [] | `gates/twin-vs-repro-diff.json` |

## What the reproduction proves and what it does not

The card's payload maps basis rows to layers 10–27 (`layers` list, pinned by the
payload sha256). `plan-apply` honours that list; without it a footprint plan
would be re-indexed onto layers 0–17 and edit the wrong weights. With
`--edit-type preserve` the writer keeps each tensor's source codec (27 Q4_K,
9 Q6_K), which is how the twin was built, so **every one of the 226 tensors is
byte-identical** to `model-merged.gguf`, including the 190 untouched ones.

The two files are not identical as containers: the plan-apply output carries
`jdewitt.fingerprint` in its GGUF metadata (1,920 bytes of header), so its file
sha256 (`4538622683…`) differs from the twin's (`fc0b1a7f7ff8…`) while every
tensor digest matches. That is the only difference.

`--edit-type auto` is the default and would pick F16 on this base (the probe
ladder cannot keep a unit-axis edit inside a 6-bit grid at a 0.95 cos floor), a
different codec, not a different edit. The release documents both; the
reproduction gate uses `preserve` because the shipped twin is the claim.

## Inputs

* measured record: `verification/p6-runtime-equivalence/spark-1.7b-q4km.json`
  (sha256 `c39e69b2…`, committed there)
* plan card built by `scripts/p6_plan_card.py`, which re-derives the fit from
  `records.pt` and refuses unless `directions_sha256` equals the measured
  `32cb983f051fb9265db278f9f138350c13ea736a899708715f2bcff06b2bfeef`
* labels built by `scripts/p6_release_inputs.py` from the same measured record;
  adapter card written by `jdewitt adapter-card`
* competitor fixture: `competitor-fixture/` — a shape fixture (16-bit model
  named by sha256, no adapter, no twin record, no plan card, no signature,
  no invented numbers)

## Open

The release is probe-level and GATED: 4 rows, no certification wave, no
vLLM/SGLang claim. The adapter curve is screening-only on this artifact
(ratio 0.604 < 0.7 floor).
