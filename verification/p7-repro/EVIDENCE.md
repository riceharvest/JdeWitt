# Phase-7 reproduction index on the real Spark release

Every JSON in `gates/` is the stdout (or, where noted, the merged stdout+stderr) of
one real command run in this worktree; none was edited. The release directory is
1.1 GB of model bytes and is not in git.

## Release directory and base artifact

The signed Phase-6 release dir is
`/home/dario/.agentboard/worktrees/t_a44a4d22/verification/releases/spark-1.7b-q4km-p6`
(release identity `9964a63f…`, eight release files plus `evidence.json`,
`signature.json`, `repro.json`). It belongs to the P6 card's evidence and was **not**
modified. It was copied byte-for-byte (btrfs reflink) to
`/home/dario/.agentboard/p7-t_3bb401d4/release-spark-p7`; `verify-release` on the
copy returned valid with the same identity and the twin still hashed
`fc0b1a7f…` before any seal was rewritten. The P7 seals below were written only
into that copy. Signing used the P6 Ed25519 secret in place
(`…/t_a44a4d22/verification/p6-release/keys/release-signing-key.pem`, mode 0600):
it was never copied into this branch, never printed, and only its public key's
sha256 (`b1ad47c1…`) is recorded here.

Base artifact (the user's quant, not redistributed in the release):
`/mnt/ssd/models/Spark-X2.6-1.7B-Q4_K_M.gguf`, 1,107,458,592 bytes, sha256
`80ff5222ae51de055fe2e86b1ec4f6784c2f9db704ed3b043b97ce900d3a75ad`
(`gates/sha256sum.txt`). `evaluation.json` declares that path and digest; the index
now resolves it and re-hashes the bytes.

## Gates on the real dir

| gate | command | result | file |
|---|---|---|---|
| build | `release-repro DIR --out DIR/repro.json` | exit 0, version 3, `repro_digest 73d6ded2…`, `binding_digest 459eaf56…`, 201 operands | `gates/release-repro-out.json` |
| sign | `release-sign DIR --secret …/release-signing-key.pem` | exit 0, same digests, `public_key_sha256 b1ad47c1…`, 201 operands | `gates/release-sign.json` |
| signature | `release-verify-signature DIR --public …/release-signing-key.pub.pem` | exit 0, valid true, pinned key | `gates/release-verify-signature.json` |
| repro check | `release-repro DIR --check` | exit 0, valid true, 8 files + 193 operands, `moved []`, `problems []`, no provenance drift | `gates/release-repro-check.json` |
| release | `verify-release DIR` | exit 0, valid true, 11 files, identity `9964a63f…` | `gates/release-verify-release.json` |
| artifacts | `verify-artifact --all DIR` | exit 0, valid true | `gates/release-verify-artifact.json` |
| base digest | `sha256sum DIR/model-merged.gguf /mnt/ssd/models/Spark-X2.6-1.7B-Q4_K_M.gguf` | twin `fc0b1a7f…`; base `80ff5222…` | `gates/sha256sum.txt` |
| P6 dir, new tool | `release-repro <P6 dir> --check`; `release-verify-signature <P6 dir> --public …` | both exit 0 with the shipped v2 seal; no base read | `gates/original-p6-*.json` |

The check printed `base_file: {"path": "/mnt/ssd/models/Spark-X2.6-1.7B-Q4_K_M.gguf",
"source": "evaluation.json base.path", "sha256": "80ff5222…", "bytes": 1107458592}`,
so the printed digest is the one `sha256sum` returns beside it in
`gates/sha256sum.txt`.

## What the index resolved (201 operands: 8 files + 193 operand leaves)

| operand | resolved value |
|---|---|
| base GGUF | `Spark-X2.6-1.7B-Q4_K_M.gguf`, sha256 `80ff5222…`, 1,107,458,592 bytes, arch `spark2_5`, file type `MOSTLY_Q4_K_M`, quant counts F32 57 / Q4_K 140 / Q6_K 29, 226 tensors, `observed: true` |
| plan card | bundle `plan-card.tar` sha256 `b57dac81…`, manifest payload `a90c3746…`, registry key `6aedfc200b76e5a7`, meta digest `05b4a0ba…` |
| plan-card.json | cross-checked against the bundle; no `card_problem` |
| suites | `null` — the release declares no suite file (see below) |
| serve argv | one axis (`default`), 22 tokens, as `evaluation.json`/`twin.json` pinned them |
| codec | 36 rows read from the shipped twin header: 27 Q4_K (cos 0.2152–0.4717) and 9 Q6_K (cos 0.8118–0.8858), byte cost per tensor |

At least one codec row per edited tensor family is present (27 Q4_K, 9 Q6_K), and
each row's quant type is checked against the twin's own header — a row that says
Q4_K where the artifact stores Q6_K is a named refusal, not a silent pass.

## One moved byte is caught on the real base

The base is not a release file: `files.json` deliberately does not index it,
`release-assemble` refuses files outside its release set, and the operand is the
digest. So there is nothing to repoint; the checks below leave every release file
byte-identical and move only the base copy.

**Variant A — explicit `--base`.** Copy the dir, copy the base to
`spoiled/Spark-X2.6-1.7B-Q4_K_M.gguf`, flip one byte at offset 500,000,000
(`0x37` → `0xda`), then:

```
release-repro release-flipped-base --check --base spoiled/Spark-X2.6-1.7B-Q4_K_M.gguf
```

exit 1, `valid false`, exactly one moved row:

```
{"operand": "base.sha256",
 "expected": "80ff5222ae51de055fe2e86b1ec4f6784c2f9db704ed3b043b97ce900d3a75ad",
 "actual":   "41e45c4cbde776129eeb0989bbd5daed75f38adcd84b4fd899090baf0fa72526"}
```

`gates/release-repro-flipped-base.json`, SHA-256 of both files in
`gates/sha256sum-flipped-base.txt`.

**Variant B — base dropped into the copied dir.** The same spoiled bytes as
`base.gguf` inside a copy of the release, plain `--check` (no flag): exit 1 with
the same single `base.sha256` row. A `base.gguf` inside the directory now wins
over the author's absolute host path, so a copied release is checked against its
own bytes (`gates/release-repro-flipped-indir.json`).

**The untampered dirs still pass.** The working dir re-checked after both
negatives: exit 0, `moved []` (`gates/release-repro-check.json`). The untouched P6
dir also still checks 0 and its shipped signature still verifies against the
pinned key (`gates/original-p6-*.json`).

**The same bytes under another name still pass.** A reflink copy of the base named
`tmp-p7-base-renamed.gguf` (sha256 `80ff5222…`) checks 0 and
`release-verify-signature --base` verifies with the same `binding_digest`
`459eaf56…`, because a version-3 seal leaves the base's file name and path out of
the binding (`gates/release-repro-renamed-base.json`,
`gates/release-verify-signature-renamed-base.json`).

**UNMEASURED is named, never fabricated.** A missing release directory exits 1 with
`UNMEASURED: release directory … does not exist …` naming the path
(`gates/unmeasured-missing-directory.json`); an explicit `--base` that does not
exist is refused with that path in the problem instead of falling back to another
copy (`gates/unmeasured-missing-base.json`). No fixture was substituted for the
release.

## Why `suites` is null on this release

The four behaviour probes (one per family, 200-token greedy) are recorded inside
the gated measured record `verification/p6-runtime-equivalence/spark-1.7b-q4km.json`
(sha256 `c39e69b2…`, suite `capture-spark-mixed.jsonl` sha256 `1f3010fa…`), which is
not in the release directory. The index records suites only when the release
declares them and their bytes can be re-hashed from what ships, so it reports
`null` rather than inventing a suite operand a stranger could not reproduce. The
argv and codec operands are the behavioural surface the release actually carries.

## What this proves and what it does not

Proves: the shipped release's base digest, plan bundle, argv and per-tensor codec
table are re-derivable by one command from the release dir plus the buyer's own
quant; a single moved byte in that quant is a named, digest-level failure; the seal
still verifies against the pinned key after the index was rebuilt with observed
base bytes.

Does not prove: model quality, that the twin's edit direction is the fitted one
(that is `plan-apply` + `gguf-tensor-diff`, P6 evidence), or any behaviour claim.
On a host without the base bytes the check exits 1 UNMEASURED until `--base` is
supplied — that is the intended failure, not a pass.
