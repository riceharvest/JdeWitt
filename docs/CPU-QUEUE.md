# CPU queue: work that can land without a model forward

2026-11-26. Companion to `ROADMAP.md` (what to do and why) and `PLAN-NEXT.md`
(what has landed). Every item here is CPU-only end to end: no serve, no
capture, no wave, no B70, no generated token. Each item names the command that
decides it; ship the counts, digests and exit codes that command printed in the
commit message. Behavioural claims stay GATED, and each item says where that
boundary is.

Baseline, checked today (a fresh clone should reproduce it):

- `uv run python scripts/evidence_map.py --check` -> 43 of 43 numbers in README
  section IV recompute; every artifact the map names is present.
- `uv run python scripts/help_audit.py` -> `next_missing: []`.
- `uv run jdewitt commands` -> still lists `plan-apply` on master. The dev
  checkout carries an uncommitted removal of it (item 2): master and that tree
  do not describe the same tool until one of them lands.

## 1. Codec-aware delta: make a 4-bit grid store the edit

Highest payoff on this list, and the capability no competitor has: nobody else
edits a quantized file, so the delta solver is the product, not polish.

Recorded 2026-11-22: on Spark 1.7B Q4_K_M a rank-1 removal at strength 1.0
realizes cos 0.454 on attention and 0.938 on FFN; `--min-edit-cos 0.9` refuses
those arms, and bisecting the scale upward landed at 6.61 with +10.2 nats of
control perplexity. The limit is the codec grid, not the fitted axis.

Build: a `repair='solve'` mode in `jdewitt/rewrite.py` that re-solves each
edited tensor against its target codec: `d <- d + proj(target -
dequant(encode(w + d)))`, two or three passes, stop once the realized cos clears
`--min-cos`, floor to the unedited value where it diverges; record the pass
count and realized cos per tensor in the edit report and its `next` block.
Expose the same switch on `gguf-edit`, which takes explicit deltas and needs no
capture records, so the proof below stays self-contained.

Check (in-clone, CPU): `uv run jdewitt requant
verification/iq05b/Qwen2.5-0.5B-Instruct-IQ3_M.gguf --out <scratch>/q4.gguf
--type q4_k_m` (0.5B, seconds), pack one deterministic delta with
`gguf-delta-pack`, then `gguf-edit --delta-gguf` with `--repair none` versus
`--repair solve` at the same `--edit-type q6_k`. Compare the per-tensor
delivered cos in the two reports: plain ~0.45, solve >= 0.90 at equal type and
byte cost. Then `gguf-revert` restores the source byte-exactly and
`gguf-tensor-diff` reports every untouched tensor bit-identical.

Boundary: what the model answers after the solved edit is GATED (needs a
serve); the numbers here are realized fidelity, not removal.

## 2. Land the pending tree (small; do first)

The tree already carries a coherent change, uncommitted: `plan-apply` removed
end to end (module, CLI command, `verification/p6-release/`, its scripts), a
`next` block on `export-smoke` and `hub-census`, and `--records` threaded into
the edit report. One stale reference remains: `jdewitt/vocabremap_weights.py`
still cites planapply's default floor.

Check: `uv run jdewitt commands` lists no `plan-apply`; `scripts/help_audit.py`
-> `next_missing: []`; `scripts/evidence_map.py --check` green; `git grep
planapply` matches only untracked `build/` artifacts once the comment is fixed;
`uv run jdewitt schemas` unchanged. Then commit (no AI attribution) and push,
or revert the whole change on purpose. Do not leave it half-landed: docs and
CLI describing different tools is the exact failure the repo is built against.

## 3. Public-surface truth pass

The repo is public and several docs still carry absolute lab paths (13 lines
under `docs/`) and two board card ids. Scrub them to repo-relative paths and
plain descriptions, and keep every quoted command runnable from a fresh clone.

Check: `grep -rn '/mnt/' docs/` returns 0 lines; the card-id pattern returns 0
hits in `docs/` and `README.md`; `scripts/evidence_map.py --check` stays green
after the edit, because the commands in the map are the ones being scrubbed.

## 4. Standing audit sweep, recorded

One pass over the CPU checkers, recorded in the commit message (date, exit
codes, digests):

- `uv run python scripts/evidence_map.py --check`
- `release-matrix` rendered twice from the same manifest -> two byte-identical
  outputs plus the table and doc
- a one-byte edit of a loss-ledger value -> the command refuses
  `MATRIX_LEDGER_INVALID` (falsify the gate; a green run alone proves nothing)
- `quants-audit` on the in-clone fixtures (iq05b IQ3_M and IQ4_XS; a sampled
  Q4_K_M from the minicpm-v47 directory)
- `gguf-tensor-diff` self-diff on each fixture -> 100% bit-identical
- `uv run python scripts/help_audit.py`

Check: every command exits 0, the tampered ledger is refused, and the two
matrix renders share a sha256.

## 5. Hub-census refresh -> adapter worklist

The phase-8 rhythm: run `hub-census` (network, no model), join the architecture
table, write the dated artifact, the triage table and the per-row `next`
commands the in-tree change adds, ordered by downloads.

Check: a rerun produces the same JSON apart from the timestamp; the worklist is
download-ordered; the triage table is rendered, never hand-edited; every `next`
is a command the CLI parses.

Landed 2026-11-26 (3.2 s, exit 0): `verification/hub-census/2026-11-26.json`
(137 repositories, 13 derived artifacts, 22 model types, 0 author errors, a
7-row worklist, 14 `next` commands, sha256 `701d2e28000cd448…`); a second run
was identical apart from `census_at`; the triage script rendered
`docs/HUB-CENSUS.md` and `2026-11-26-loader.json` from it, and both are
byte-identical on a re-render. All 14 `next` lines parse through the CLI;
`capture` needs its required `--ctx-size`, which the earlier in-tree line
omitted, so the pair carries a literal 4096 default.

## 6. Space skeleton, CPU dry run before any quota

`docs/SPACE-ROADMAP.md` is design only and no GPU second has been spent on it.
Build the editor path as a plain module (visitor GGUF in -> the same file,
edited, out; untouched tensors byte-identical) and run it locally against the
in-clone 0.5B IQ3_M with a synthetic axis. No Space, no quota.

Check: the report carries edited/untouched counts, delivered cos and sha256; a
second run is byte-identical; a file already carrying the fingerprint is
answered with a byte copy (`already_applied`); an oversized input is refused
before it is read.

Boundary: ZeroGPU quota, GPU throughput and every served claim are GATED.

## Order for today

2 (clear the tree, ~30 min) -> 1 (the capability; the day's real work) -> 3 and
4 (small; they protect every quoted number) -> 5 and 6 as time allows.

## Out of scope for this queue

Anything whose acceptance check needs a serve, a wave, a judge or the B70; that
work is GATED by definition and is not on this list. No new test suites and no
CI (standing rules in `PLAN-NEXT.md`).
