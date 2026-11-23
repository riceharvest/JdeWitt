# The Space direction bank

`bank.json` is the unit described in `docs/SPACE-ROADMAP.md` §5.1: one
`jdewitt-space-bank-v1` document whose `entries` each name a fitted axis by the
geometry it was fitted on, plus a digest-pinned `plan-bundle` beside the file.
Nothing here is model weights; an entry's `footprint.fraction` is how it survives
an artifact with a different layer count.

The bank ships **empty** on purpose. The only removal arm on the board
(Qwen3-4B-Instruct-2507 `f15s15-l935`) no longer has its axis file in this clone,
so the first entry has to be refitted on a GPU card; copying an entry without the
bytes it pins would be a claim this repo cannot verify. An empty `entries` list is
valid and `bank-check` reports `BANK_MISS` for every artifact until it is seeded.

Resolve an artifact against the bank (header + tensor table only, no tensor
data, no torch):

    python -m jdewitt bank-check --bank spaces/bank --gguf <artifact.gguf>

It prints one JSON: the resolved entry, target tensor names, layer list, per-kind
strength and edit types, or one typed refusal — `BANK_MISS`,
`GEOMETRY_MISMATCH`, `UNSUPPORTED_QUANT`, `BUNDLE_BAD_DIGEST` — in the standard
error envelope. Fixture coverage lives in `verification/space/fixtures/`; the
gate runs are recorded in `verification/space/`.

Adding an entry is a copy of the §5.1 object with its own bundle and digest.
`bundle.file` must be a plain file name beside `bank.json`; the loader refuses a
path, an absolute name or a directory.

## Apply one entry: their GGUF in, an edited GGUF out, one JSON

    python -m jdewitt bank-apply --bank spaces/bank --entry <entries[].key> \
        --gguf <artifact.gguf> --out <edited.gguf>

No server, no tokenizer, no transformers, no model load. The command resolves the
entry against the artifact's own header (`bank-check`'s checks, same typed
refusals), installs the pinned bundle into a THROWAWAY registry
(`unbundle_plan` -> `load_plan`, so the file beside `bank.json` is never trusted
in place), and builds the edit map from the entry: every footprint tensor named by
`targets`, at the entry's own per-kind `strength` (times `--strength-scale`).
Then it writes, via the same minimal-surgery writer `plan-apply` uses, only the
footprint tensors — everything else is copied byte for byte — and measures the
delivered edit per tensor with `compare_edit`.

`--edit-type auto` (the default) probes the worst edited tensors at every type the
entry MEASURED, smallest first (`q6_k` -> `q8_0` -> `f16`), and writes the first
that clears `edit.min_cos`. If none clears it the run is refused with
`CODEC_FLOOR`, naming the tensor and the best cos: the strength is never raised to
hide a codec floor, and a type the entry did not measure is never substituted. A
named `--edit-type` must be inside `edit.types` (`EDIT_TYPE_NOT_ALLOWED` otherwise).
Every refusal happens before the writer opens `--out`.

The JSON reports the bank key, base repo/revision, bundle sha256, the entry's
measurement state, the footprint and the entry rows it landed on, the type ladder
and its probe readings, per-tensor intended-vs-delivered `cos` and `snr`, chosen
type, bytes in/out, edited and untouched counts (untouched identity is re-read
from BOTH files, never asserted), the output sha256 and wall seconds. It also
carries `"removal_measured": false`: the cos is CODEC fidelity — what the bytes
hold — not a claim that anything was removed.

Evidence: `verification/space/bank-apply-*.json`; the fixtures and their
generators live in `verification/space/fixtures/`.
