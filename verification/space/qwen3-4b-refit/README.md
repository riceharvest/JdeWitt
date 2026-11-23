# qwen3-4b-instruct-2507 `f15s15-l935` — refit, not a copy

The arm this bank entry comes from (`verification/releases/qwen3-4b-f15s15-l935/spec.json`)
names a `.pt` that is not in this clone, and the model directory it names no longer
exists. No surviving capture matched the recorded recipe either: the `verification/q3-4b-*`
captures are ctx 8192 / pair-fit ctx 4096, and the arm was captured at ctx 2048. So the
entry was **refitted** from a new capture at the recorded recipe. Nothing here is copied
from the release spec except the recipe numbers and the revision.

## Capture

    python -m jdewitt capture \
      --model Qwen/Qwen3-4B-Instruct-2507 \
      --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
      --suite verification/p4d/suite-80.jsonl \
      --kinds refusal,harmless \
      --out verification/space/qwen3-4b-refit/records.pt \
      --locus answer --ctx-size 2048 --max-tokens 512 --settle-min-tokens 24

The capture record (`records.pt.capture.json`, committed) holds the identity the fit
reads — model, revision, architecture `qwen3`, 36 layers, hidden 2560, answer locus,
ctx 2048, max tokens 512, suite `verification/p4d/suite-80.jsonl`
(sha256 `1f3ab3cd…`), 80 rows, 36 bad / 36 good, tokens prompt 12120 and answer
19813, elapsed **1410.1 s** on the B70. No `--gguf` was passed, so the record names no
local path.

`records.pt` is **29.7 MB and is not committed** (`*.pt` is ignored; the repo's largest
tracked file is 7.4 MB). Its digest is `83734fe85d92bf1af32da3981e7b3a71a6ba1a9b167f558596bd206ed285239a`,
the same digest the plan's identity and the fitted run's provenance carry, so the
committed bundle can be tied to the exact bytes that produced it. To regenerate the
capture, run the command above.

## Fit

The arm's own estimator, `spanfit.shared_axis` (shared sign-aligned), on the answer
locus. **1.21 s** on the CPU from the committed record (`3.2 s` in the first pass on a
cold page cache); axes `[36, 2560]`, `axes_sha256`
`3da7f8baf84b3d02f6cd17bad6939fee24af6a81c6645058dd8fc7446d86770c`. The fit is
deterministic: refitting the same `records.pt` reproduces that digest and the same plan
key `1d29cd9feda0cf41`.

Axis agreement per layer, cos(direction, bad−good mean): min 0.394, mean 0.736,
max 0.861, layers 15–35 mean 0.794. The recorded arm's `realized_cos` 0.7255 is a
four-tensor *post-quant* cos, so the two numbers are the same ballpark, not the same
statistic — `refit.json` records both and says so.

## Artifacts

| file | what | digest |
|---|---|---|
| `spaces/bank/qwen3-4b-instruct-2507-f15s15-l935.tar` | `plan-bundle`, 378880 B, 1 tensor `[36, 1, 2560]` f32 | sha256 `5df68bc7…` (pinned by the entry) |
| `spaces/bank/qwen3-4b-instruct-2507/entry.json` | the §5.1 unit, identical to its `entries[]` row | — |
| `direction-f15s15-l935.gguf` | `direction-export --dtype bf16`, 185152 B, `b70ablator.format = direction-gguf-v1`, shape `[2560, 1, 36]`, plan key `1d29cd9feda0cf41` | sha256 `53e49df9…` |
| `refit.json` | the whole run: capture identity, tokens, seconds, fit, gates | — |
| `bank-check-base-q4km.json` | `bank-check` against the base's own Q4_K_M | — |

## Gates (all re-run, not transcribed)

- `plan-unbundle` the committed tar into a throwaway registry, then
  `plan-load <registry> 1d29cd9feda0cf41 --expect 36 1 2560` → passes, geometry
  `{36, 1, 2560}`, `measured: false`.
- `direction-export` header → `direction-gguf-v1`, bf16, plan key `1d29cd9feda0cf41`,
  geometry `{layers: 36, rank: 1, hidden: 2560}`.
- `sha256sum` of the bundle equals `entry.bundle.sha256`
  (`5df68bc77469c3c722d73ff2871a1de98266f2053c7040ee87c2932432066f5c`); `bank-check`
  also verifies it before reading the axis (`bundle.verified: true`).
- `jdewitt bank-check --bank spaces/bank --gguf <base Q4_K_M>` → resolves
  `qwen3-4b-instruct-2507/f15s15-l935`, architecture `qwen3`, hidden 2560 from the
  artifact's own tensor table, footprint layers 15–35, **42 targets** (21
  `attn_output` + 21 `ffn_down`), strength `{attn: 1.5, mlp: 1.5}`, edit types
  `q6_k/q8_0/f16`, source types `Q4_K`/`Q6_K`, `refusal: null`.

The base Q4_K_M is 2.5 GB and is not committed. It is the recorded base Q8_0
(`q3-4b-base-q8.gguf`, sha256 `dc4ae266…`) re-quantized at 16 threads:

    python -m jdewitt requant --gguf q3-4b-base-q8.gguf \
      --out q3-4b-base-q4_k_m.gguf --type q4_k_m --threads 16

398 tensors in, 398 out, 26.8 s, output sha256
`a8b02c91a77727a903dd9196842f9ff191269de24b6ac1f95e9c185bb1b85c17` — reproducible byte
for byte, and the same digest the first pass recorded. `bank-check` reads only the
header, so a header-range copy of the base is enough to rerun the gate.

## What this is not

Nothing here measures removal or retention. The entry is born `banked_unmeasured`; the
transfer card earns a label on an artifact class, not on this refit.
