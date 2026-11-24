# jdewitt transfer probe — 5.5 GB class

**This is not a model and not a removal artifact.** It is a valid GGUF with a real
header, a real tensor table and real tensor bytes (eleven F16 tensors of zeros,
5,500,000,704 bytes in total), written so the Hub push can be measured at the byte
class the 7B case needs.

Why a probe: the shipped bank has one entry, fitted on `Qwen/Qwen3-4B-Instruct-2507`,
so a 7B Q4_K_M model is refused with `BANK_MISS` — today the Space cannot produce
the ≈5.5 GB 7B q8_0 output the roadmap's §8 table estimates. What can be measured
without inventing a 7B axis is the transfer: the same bytes, the same delivery
module, the same Hub endpoints.

The round-trip check is byte-exact: the artifact is fetched back from the public
repo and its sha256 and byte count must equal the file that was uploaded, and
`gguf-inspect` must read its tensor table (11 tensors, architecture `jdewittprobe`).
No claim is made about this file as weights, and none about the 7B product path
beyond the transfer number in `verification/space/oauth-upload.json`.
