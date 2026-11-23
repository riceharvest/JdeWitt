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
