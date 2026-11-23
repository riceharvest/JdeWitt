# bank-check evidence

`jdewitt bank-check --bank DIR --gguf FILE` reads a GGUF's header and tensor-info
array only (no tensor data) and either prints the resolved entry or refuses with
one typed code. The three files here are the three gate runs from the bank card:

| file | artifact | expected |
|---|---|---|
| `bank-check-minicpm-v47.json` | `verification/minicpm-v47/minicpm-v47-Q4_K_M.gguf` | `BANK_MISS`, architecture `qwen35` |
| `bank-check-fixture.json` | `verification/space/fixtures/fixture-2layer-f16.gguf` | resolved names, layers `[0, 1]`, strength, edit types |
| `bank-check-bad-digest.json` | same fixture, `bank-bad-digest/bank.json` | `BUNDLE_BAD_DIGEST` |

A refusal exits 1 and prints the standard CLI envelope
(`{"error":{"code","type","message","command"}}`) — one JSON, like every other
typed refusal in this repo.

The MiniCPM artifact is 672 MB and is not committed (`*.gguf` is ignored and the
clone outside `verification/vocab-remap/vocab/` carries no model). To rerun that
row, put the file at the path above (a symlink to a local copy is enough) or
download it; the command never reads tensor data, so a header-range copy would do.

The fixture is committed and regenerable:

    python verification/space/fixtures/make_fixture.py

`make_fixture.py` writes the 2-layer GGUF, `bank/fixture-bank.tar` (fixed member
mtime, so its sha256 is reproducible), the entry that pins that sha256, and
`bank-bad-digest/` with a one-byte-flipped copy pinned at the good digest.

# bank-apply evidence

`jdewitt bank-apply --bank DIR --entry KEY --gguf FILE --out FILE` resolves one
entry, installs its pinned bundle into a throwaway registry, re-encodes only the
footprint tensors at the entry's per-kind strength and prints one JSON. The three
files here are its gate runs:

| file | artifact | expected |
|---|---|---|
| `bank-apply-fixture.json` | `fixtures/bank-apply/fixture-2layer-q8_0.gguf` | F16 chosen (q8_0 probe 0.9979 misses the entry's 0.999 floor), 4 edited / 2 untouched byte-identical, delivered cos 1.0 |
| `bank-apply-geometry-refused.json` | same fixture, `fixtures/bank-apply-bad-geometry/bank.json` (hidden 128) | `GEOMETRY_MISMATCH`, exit 1, no output file |
| `bank-apply-real-qwen2.json` | `/mnt/ssd/models/qwen25-1.5b-instruct-q4_k_m.gguf` (28 layers, hidden 1536, footprint 10-12) | Q6_K 0.6466 / Q8_0 0.9444 / F16 0.9999 -> F16, 6 edited / 332 untouched verified, 986 MB -> 1052 MB, 105-144 s over two runs |

The two fixtures regenerate with (the project venv, which has torch):

    python verification/space/fixtures/make_apply_fixture.py
    python verification/space/fixtures/make_real_apply_bank.py \
        --gguf <artifact.gguf> --bank <dir> --layers 10 12

`make_apply_fixture.py` writes a real REGISTRY plan bundle (a `[2,1,64]` basis, tar
members at a fixed mtime; the sha256 reproduces byte for byte on the same torch
build), the q8_0 artifact beside it, and `bank-apply-bad-geometry/` — the same
good bundle under an entry whose hidden width disagrees, so the refusal gate needs
no doctored bytes.

`make_real_apply_bank.py` is not a way to seed the shipped bank: it builds a
labelled SYNTHETIC axis (`measurement.state = synthetic_axis`) fitted only to the
artifact's own geometry, because the clone carries no fitted axis. That run proves
the write path, the ladder, the per-tensor measurement and the untouched bytes; it
proves nothing about removal, and the report says `"removal_measured": false`.

The output artifacts are not committed (`*.gguf` is ignored); each run rewrites
them, and the reports record the paths used at the time.

# The 2-vCPU CPU proxy

`cpu-proxy-2vcpu.json` is the measurement `docs/SPACE-ROADMAP.md` §4's decision
rule is applied to. It is a **CPU proxy, not a Space run**: the edit path is
numpy-only and ZeroGPU adds a GPU but not CPU cores, so `bank-apply` was timed on
two affinity-pinned cores of this host (`taskset -c 0,1`, `/usr/bin/time -v`).

| file | what it holds |
|---|---|
| `cpu-proxy-2vcpu.json` | artifact identity (repo/revision/sha256/bytes), `is_a_space_run: false`, host cores × affinity, three runs with wall/user/sys/RSS, bytes out, per-tensor delivered cos, the §4 verdict, the 7B extrapolation |
| `cpu-proxy-2vcpu/bank/` | the labelled synthetic-axis timing entry the runs used (see its README) |
| `cpu-proxy-2vcpu/make_report.py` | rebuilds the JSON from the two transcripts and the bank-apply reports |

Artifact: `unsloth/Qwen3-4B-Instruct-2507-GGUF` @ `a06e946bb6b6`, Q4_K_M,
2,497,281,120 B, sha256 `3605803b…` (equal to the Hub LFS sha, recomputed after
the download). Measured on 2026-11-23: **496.3 / 476.3 / 606.2 s** wall for the
same command, byte-identical output every time (42 F16 tensors, 2.50 GB →
3.49 GB, 356 untouched byte-identical). The one-core-at-98 % reading is why a
2-vCPU Space does not halve this. Verdict: **`cuda_codec_required`** — every run
is above 240 s. The 7B extrapolation (3.04× edited params) is ≈ 1510 s, a number
under a named assumption.

The timing run predates the shipped entry. `spaces/bank` now carries
`qwen3-4b-instruct-2507/f15s15-l935`, but the timing entry is still a labelled
synthetic-axis fixture (it needed the geometry, not the direction), so
`removal_measured` is false there and the delivered cos is codec fidelity on a
synthetic edit, not removal.

# The self-apply: the bank's own bundle on the base's Q4_K_M

`selfapply-qwen3-4b.json` is the first number the bank owns: `bank-apply` with the
shipped entry on the quant the Space will actually serve
(`unsloth/Qwen3-4B-Instruct-2507-GGUF` Q4_K_M @ `a06e946b`, sha256 `3605803b…`,
2,497,281,120 B), no model. The document is the `--edit-type auto` report verbatim
(card 3's shape: per-tensor cos/SNR, chosen type, bytes in/out, untouched re-read),
plus the artifact identity, the delivered ladder for the two other types the entry
measured, the probe-vs-delivered evidence, and the `gguf-inspect`/`gguf-tensor-diff`
read-back.

| run | type | delivered min cos | floor 0.95 | bytes out |
|---|---|---:|---|---:|
| auto | Q6_K (the probe's choice) | 0.8815 (37 of 42 below) | missed | 2,618,260,064 (+0.12 GB) |
| forced | Q8_0 | 0.9873 | met | 2,798,248,544 (+0.30 GB) |
| forced | F16 | 1.0006 | met | 3,494,978,080 (+0.99 GB) |

`removal_measured` is false and the document says so: cos is codec fidelity; no
answer was generated. The 42 edited names are exactly the 42 non-identical names in
`gguf-tensor-diff`, and the 356 untouched tensors are bit-identical in both diffs
(the JSON's untouched count equals the diff's `bit_identical` count).

Why `auto` misses: its ladder decides the type from `ggufio.preview_edit`, which
re-encodes `windows=2 x blocks=32` blocks (8,192 elements) of each probe tensor. For
these rank-1 deltas that sample is optimistic — on `blk.29.ffn_down.weight` the
sample reads cos 0.9974, the whole tensor 0.8812, the written file 0.8815 — so the
ladder stops at Q6_K one step before the type that clears. This is not a
`CODEC_FLOOR`: Q8_0 clears at the entry's own strength and the strength was not
raised.

Raw transcripts are in `selfapply-run/`; `selfapply-run/make_report.py` rebuilds the
document from them.
