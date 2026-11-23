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

# staging gate

`python scripts/space_stage.py --out DIR` writes the tree an account owner
pushes: `README.md` (Spaces YAML: gradio, `app_file: app.py`, a `models:` list
built from the bank's bases), `requirements.txt` (numpy, gguf, huggingface_hub,
gradio — no torch pin, no transformers, no llama stack), a thin `app.py`, a
vendored `jdewitt/` import closure, `spaces/bank/` verbatim, `.gitignore` and the
AGPL licence text. The Space is buildable without pip-installing this repo, so
the XPU torch pin in `pyproject.toml` never reaches a CUDA ZeroGPU image.

It refuses to stage on a local import the tree cannot supply, an import the
requirements do not ship, any `/mnt/`, `/home/`, board card id or the banned word
in a staged text file, a tree over 5 MB outside the bank, or a non-empty target.

`stage.json` is the gate run, taken from two real stages into fresh directories:

- `diff_r_identical` — the two trees are byte-identical;
- `sanitisation.matches` — empty for the four refused shapes;
- `app_parses` — `ast.parse` on the staged `app.py`;
- `staged_import` — `import jdewitt.space_loop` with the staged tree on
  `sys.path` succeeds and pulls neither gradio nor torch;
- `ruff` — `ruff check jdewitt scripts`, exit 0.

The coverage loop itself (`jdewitt/space_loop.py`) answers from the bank alone:
`inspect(repo, revision, quant)` returns a covered entry or a typed `BANK_MISS` /
`UNSUPPORTED_QUANT`, with the licence `UNVERIFIED` and the edit state `none`
until those gates land.

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

# licence gate

`licence-gate.json` holds three real Hub rows from `jdewitt.bank_licence`: one
allowed (`Qwen/Qwen3-4B-Instruct-2507`, apache-2.0), one refused for its licence
(`bigscience/bloom-560m`), one refused as gated (`meta-llama/Llama-3.2-1B`). The
gate reads `model_info` only (`files_metadata=False`); a fresh `HF_HOME` stayed
empty, so no file was downloaded. The allow-list is `spaces/bank/licences.json`,
which `SHA256SUMS` pins.

# the loop gate

`loop.json` is the Phase 0 done gate for "app loop + card labels, dry run on a
fixture" (§10). It is rebuilt by:

    python verification/space/make_loop_gate.py

That script runs `space_loop.dry_run` on `fixtures/loop/`, stages a tree into a
temporary directory, runs the banned-word grep, `ruff check jdewitt scripts` and
the two Space test files, and writes one JSON. The user-facing equivalent is:

    python -m jdewitt space-dry-run --bank spaces/bank \
        --model jdewitt/loop-fixture --quant Q4_K_M \
        --fixture verification/space/fixtures/loop

What the run proves, with no GPU, no socket and no Hub download:

- the three labels and one refusal: the base at Q4_K_M renders
  `measured_artifact_class`, the fine-tune (same architecture, different bytes and
  revision) renders `banked_unmeasured`, and the other architecture is `refused`
  with `BANK_MISS` and one sentence naming the architecture and what would cover
  it;
- coverage read a 512-byte prefix of each 295 488-byte artifact; the fixture hub
  pads that prefix to the declared size as a sparse file, so no tensor bytes move;
- the licence gate ran on Hub metadata before the download, and the refusal paths
  (wrong licence, gated repo) are unit-tested;
- the unmeasured card's first line is `This file was **not** measured.`, and the
  card carries base repo@rev, applied strength, artifact class, config and
  delivered cos;
- `panel_order` puts `caveat` above `start` and `file`, and `spaces/app.py` builds
  the caveat Markdown above the `gr.File` control;
- every job deleted its workspace and left the four delivered files (artifact,
  `card.md`, `sidecar.json`, `SHA256SUMS` with matching digests);
- the quant control keeps `model-Q8_0.gguf` in the list, disabled with the reason,
  instead of hiding it;
- the staged tree carries `spaces/app.py` verbatim, vendors `gguf_types.py` and
  vendors no `quantize.py` (the llama-quantize wrapper with its local paths).

`fixtures/loop/` regenerates with the project venv (torch is needed for the plan
bundle):

    python verification/space/fixtures/make_loop_fixture.py

It writes one measured entry (2 layers, hidden 256, `q4_k` source), the base at
Q4_K_M and Q8_0, a fine-tune of the same architecture with different bytes, an
artifact of an architecture no entry covers, `hub.json` (file sizes and the header
prefix that still parses) and the licence allow-list.
