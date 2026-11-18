# jdewitt-tui

A terminal front end for the refusal-removal pipeline, so a quant already on disk has
one place to start. Rust, [ratatui](https://ratatui.rs) for layout,
[tachyonfx](https://ratatui.rs/ecosystem/tachyonfx/) for the transitions.

```
cargo build --release
./target/release/jdewitt-tui --repo /path/to/JdeWitt
```

`JDEWITT_REPO` and `JDEWITT_TUI_ROOTS` (colon-separated directories to scan) work as
environment variables.

## What it does

Three screens:

* **library** — local `.gguf` and `.pt` files, with sizes. A file listing, nothing more.
* **recipe** — which operand you already have, plus the `spanfit targets` readout for
  the selected GGUF (architecture, layer count, dims, source keys), so the build line
  is the tool's read of the file rather than a guess you typed in.
* **run** — the live stdout/stderr of the command that actually ran, and its exit code.

## What it does not do

It computes nothing. Every number on screen came from `jdewitt.cli` or
`jdewitt.spanfit`; the log panel is those processes' own output. There is no result
table here because a result table would need a wave to be taken, and a wave needs a
server and a suite and an honest serve config.

## Recipes

Only three, because only three are supported by the maths:

| key | recipe | needs |
| --- | --- | --- |
| 1 | `gguf-tensor-diff base ablated` | two GGUFs of the same base, read-only, no GPU |
| 2 | `spanfit build` with a direction `.pt` | the quant plus a fitted direction pack |
| 3 | `auto` under a config | the config you wrote, plus GPU |

Recipe 1 answers the question people actually have when they hand over a published
ablated build: which tensors did it move, and does it match the footprint your
artifact needs. It does not apply anything.

Recipe 2 carries the layer plan `spanfit targets` recovered, and names the key inside
the axes file (`k` cycles it). A merge with the wrong layer count edits 0 tensors and
exits 0; the UI refuses to let that happen silently.

## Verified

Driving it headless through a pty and dumping the state (`i` key with
`JDEWITT_TUI_DUMP=/tmp/x.json`) selected `q3-4b-base-q8.gguf` with
`q3-shared-axes.pt` at scale 1.0 and built it. The output is
4,280,403,392 bytes with sha256
`18a9e9ea043254da3a0774ef713e6e4464c64190cf7e42c9f12d91f6a2890f36`, identical to the
CLI-produced `verification` artifact of the same operands. The UI path and the CLI
path are the same edit.
