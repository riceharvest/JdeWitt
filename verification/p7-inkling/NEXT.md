# P7 cert — Inkling-2-Mini IQ4_NL: what the analysis card runs next

Card layout: card 1 (this one) prepared operands + `run_waves.sh`; card 2 (`[gpu]`, t_5053ccbb)
only launches `run_waves.sh` under `lab-gpu b70`; card 3 (`[analyze]`, t_c547bd04) runs
`release-certify` / `release-matrix` and writes the packet + row.

## Read this first: the arm does not exist, and why

`run_waves.sh` was executed on this host (2026-11-24, exit 77) and stopped at step 1. Two
independent gaps were measured, not assumed. Both are recorded in committed files:

1. **Capture is impossible on this stack.** `verification/p7-inkling/capture-blocked.json`:
   `AutoConfig.from_pretrained('/mnt/ssd/models/Inkling-2-Mini', trust_remote_code=True)` raises
   `ValueError: ... model type 'inkling2_loop' but Transformers does not recognize this
   architecture` (transformers 5.20.0 in `.venv`). `CONFIG_MAPPING` has `inkling_text` /
   `inkling_audio` / `inkling_vision` / `inkling_mm_model` — none of them `inkling2_loop` or
   `inkling2_loop_text`. The checkpoint has no `auto_map`, so there is no trust-remote-code path,
   and its `quantization_config` is `modelopt`/`NVFP4`. No residual records -> no fitted axis
   -> no merge -> no arm. `jdewitt capture` is transformers-only; nothing in `scripts/` captures
   from a GGUF, and the fork's `B70TUNE_TAP` output is op timings, not hidden states.
2. **No server binary that knows this architecture.** The Inkling fork builds
   (`/mnt/ssd/dev/inkling-rt{,-t57174e1b,-fadec}/llama.cpp/build-sycl/bin`) contain
   `llama-cli`, `llama-bench`, `llama-perplexity`, `llama-quantize` and `libllama-server-impl.so`
   but **no `llama-server` executable**; a stock `llama.cpp` does not know `inkling2_loop`.
   `run_waves.sh` refuses `SERVE_NOT_FOUND` (72) naming this. Even a built arm could not be served
   until that one target is built in the fork.

Consequence: `verification/releases/inkling-2-mini/spec.json` and `packet.json` do **not** exist,
and must not be written from prose. There is nothing to certify at the pinned config, and
`release-matrix` will render the row `UNMEASURED` with
`status_detail = "no packet at verification/releases/inkling-2-mini/packet.json"`.

## Branch A (this is the current state): record the blocker, name the spare

Do not invent a packet. `release-matrix`'s status vocabulary (`jdewitt/matrix.py`, module
docstring) only derives `failed(<axis>)` from a packet that *was* measured at the pinned config
and missed a registered band; with no waves there is no such packet, so the honest row is
`UNMEASURED` with the measured blocker. Concretely:

1. Put the blocker where the matrix reads it, so the row is not a blank:
   `verification/releases/matrix.manifest.json` -> the `inkling-2-mini` entry's `artifact.note`
   (and `required` stays `true`). Quote `capture-blocked.json`'s `error`, and name the second gap
   (no `llama-server` for the fork).
2. Regenerate the matrix so the row, the doc and the counts come from the manifest:
   ```
   uv run jdewitt release-matrix \
     --manifest verification/releases/matrix.manifest.json \
     --out verification/releases/matrix.json \
     --table verification/releases/matrix.md \
     --doc verification/releases/README.md
   ```
   Expect `inkling-2-mini` = `UNMEASURED`, exit 0, and the count of certified lineages unchanged
   (four). Do **not** describe the goal card as closed on five.
3. Name a spare lineage per the done gate. `gemma4-31b` (Gemma-4-31B q4_0) is registered in
   `matrix.manifest.json -> spares` and is a distinct `model_type`. `spark-4b` is **not** a
   candidate: Spark-X2.6-4B is `spark2_5`, the same family as `spark-1.7b`, so it can never add a
   row ("a duplicate model_type is one lineage counted twice, never a sixth").
4. If you decide a real `failed(capture)` status is worth having, that is a code change with a
   test in `jdewitt/matrix.py` (add it to the closed vocabulary, feed it from a measured input,
   keep `failed` distinguishable from `invalid`), not a hand-edited `status` field. Say so in the
   packet/README either way.

## Branch B (if capture and a server become possible): the two commands

`run_waves.sh` is written to run the whole chain unattended once capture can run: operands ->
capture -> `spanfit axes --shape shared` -> per-tensor-typed merge (IQ4_NL first) -> 4 waves at
`-c 4096 --parallel 4` cap 1024 -> shape audits -> drift -> `spec.json`. It is resumable: every
step is skipped when its output file exists, and a half-written wave directory is deleted and
re-taken.

```bash
# 0. preconditions (no GPU): operands, pins, binaries
bash verification/p7-inkling/run_waves.sh --check     # sha256 of every operand in operands.json
bash verification/p7-inkling/run_waves.sh --dry       # every argv, nothing executed

# 1. the GPU card runs this (it does no other work)
bash verification/p7-inkling/run_waves.sh             # writes progress.txt, spec.json

# 2. analysis card: certify the saved waves (CPU, reads files only)
uv run jdewitt release-certify \
  --spec verification/releases/inkling-2-mini/spec.json \
  --out  verification/releases/inkling-2-mini/packet.json

# 3. analysis card: the matrix row
uv run jdewitt release-matrix \
  --manifest verification/releases/matrix.manifest.json \
  --out verification/releases/matrix.json \
  --table verification/releases/matrix.md \
  --doc verification/releases/README.md
```

`release-certify` refuses by name (`CERTIFY_MISSING_AXIS`, `CERTIFY_ONE_INVENTORY`,
`CERTIFY_CONFIG_MISMATCH`, `CERTIFY_PROBE_ONLY`, `CERTIFY_UNMEASURED`,
`CERTIFY_RETENTION_INCOMPLETE`, `CERTIFY_ARTIFACT_MISMATCH`) and writes nothing when it refuses.
A refusal is a finding to report, never something to work around by trimming a suite or lowering a
cap.

## Known registered-floor conflict the analysis card must resolve, not skip

`matrix.manifest.json -> registered.cap` says `removal: 4096, held_out: 4096, retention: 1024`,
but `registered.pinned` says `ctx 4096, parallel 4`. At 4096 context with 4 parallel slots each
slot gets 1024 tokens of context, so a 4096-token generation cap is arithmetically unreachable:
the registered removal/held-out caps were carried over from the qwen waves that ran at
`ctx 8192 / parallel 2` (8192/2 = 4096 — see `verification/releases/qwen3-4b-f15s15-l935/spec.json`).
`run_waves.sh` follows this card's spec and asks for `--max-tokens 1024` on every axis, so as soon
as a packet exists `release-matrix` will flag `invalid(cap_lowered)` on removal and held_out.

Resolve it explicitly, in this order:
1. Prefer raising the *slot* budget instead of lowering the registered floor: serve at
   `-c 16384 --parallel 4` (4096 per slot) if the artifact and the card allow it, and record that
   argv in the spec — but note this then leaves the pinned `ctx 4096` cert config
   (`certified_at_other_config`, never `certified`). That is the honest row, not a pass.
2. Otherwise fix `registered.cap.removal`/`.held_out` to 1024 **with a note saying why the
   arithmetic forces it** and record the change in the matrix doc. Do not edit the floor silently
   and do not grade refusal rows at a cap their slot cannot hold: truncated answers read as "no
   refusal".
Either way the change must appear in the matrix a reader can audit, and it must not be done in
this prep card (this card registers no floors).

## Files this card produced (committed)

| file | what it is |
| --- | --- |
| `run_waves.sh` | the one script: operands -> capture -> axis -> arm -> waves -> audits/drift -> spec; `--dry`, `--check`, resumable, stops every server it starts (`trap ... EXIT INT TERM`) |
| `operands.json` | 10 pinned operands with sha256/bytes and provenance; `--check` re-hashes them (base GGUF 17,627,940,768 B `cbdf3fa9…`; source safetensors `65e60f51…`) |
| `capture-blocked.json` | the measured `AutoConfig` refusal, the `CONFIG_MAPPING` keys, the checkpoint's `model_type`/`quantization_config`/`has_auto_map` |
| `capture-suite.jsonl` | deterministic 72-row capture suite (40 registered harmful rows + 32 harmless from `harsh-suites/selection.jsonl`), used only by the capture step |
| `progress.txt` | the run's own trail: `operands ok`, then `REFUSED 77 CAPTURE_UNSUPPORTED` |
| `run.log` | the stderr/stdout trail of that same run (what each step printed) |

Removal suite provenance: the registered path
`/mnt/ssd/hermes/cache/scratch/probes/harmful-40.jsonl` was gone from this host. It is **restored**
from the in-clone twin `verification/screens/advbench-40.jsonl` (sha256
`d4b84d2b…`), and the restore is verified: the twin's 40 ids and prompt strings are identical to
the rows the qwen removal wave recorded (0 mismatches). `run_waves.sh` re-restores and re-checks it
against `operands.json` on any host that lost it, and the spec's `suites.removal.note` says so.

## Verification performed without a GPU (all re-runnable)

```bash
bash -n verification/p7-inkling/run_waves.sh                  # syntax
bash verification/p7-inkling/run_waves.sh --dry               # argv list + preflight, writes nothing
bash verification/p7-inkling/run_waves.sh --check             # sha256 of all 10 operands
bash verification/p7-inkling/run_waves.sh                     # -> exit 77, capture-blocked.json
uv run ruff check jdewitt scripts
```

Every Python heredoc in the script was extracted and `py_compile`d (8/8 clean), and the
spec-assembly heredoc was run against stub arm/build/codec/drift files plus the *real* qwen waves
in a scratch directory: the emitted `spec.json` was accepted by `release-certify`, which produced a
complete packet with shape audits stamped at the computed identity digest, paired retention stats,
drift and fidelity blocks (`pinned_cert_match.retention/held_out = true`). That proves the spec
shape the script writes is the shape the certifier reads; it proves nothing about Inkling numbers.
