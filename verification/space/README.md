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
