# AGENT-FIRST — the narrowing that makes this the easiest tool to drive

The user of this tool is an **AI agent** (METHODOLOGY R5), working on a model it
did not choose, on hardware it does not control, from documentation it reads
cold. Three properties decide whether that agent can act at all. They now order
all work on this repo; everything else is underneath them.

## 1. One path, one command per step, no code in between

The chain from "here is a checkpoint" to "here is a certified artifact" is
`capture` → `spanfit build` → screen → certify. Each step:

- takes files and flags, never a bespoke script — an agent that must write a
  hook or a serve loop to get started has already left the tool;
- emits one JSON with the payload, the cost paid, and the **exact next
  command** filled in with the paths and names it discovered;
- is non-interactive, idempotent and resumable; the model server stays up
  across steps (R5).

`capture` replaces the per-model scratch scripts that used to sit here. It is
one command, and its report is the handoff.

## 2. Any architecture, by discovery, never by table

- The decoder stack is found from the module tree: the `ModuleList` whose
  blocks write a `hidden_size`-wide residual stream.
- The residual-write projections are found by name **and** width
  (`o_proj`/`out_proj`/`ssm_out`/`down_proj`/`w2`, output width == hidden). The
  name test is load-bearing: on an MHA model `q_proj` is also hidden-wide, and
  an axis captured in residual space is meaningless in query space, so a
  dimension-only rule edits the wrong matrix while passing every shape check.
- The GGUF side does the same from the artifact's tensor table
  (`--gguf-targets auto`): no per-model source-key table, no hand-typed glob.
- A name the tool cannot map is **reported and refused**, never guessed at.
- A capability that only works for a lineage the repo has already been tuned for
  fails R7, whatever its numbers.

## 3. In place, never re-quantized

The artifact *is* the model the agent serves. `spanfit build --in-place` patches
the quantized GGUF's own bytes:

- a requantized block keeps its byte width, so every tensor offset and the file
  size stay fixed — the file is edited in place, not rebuilt;
- only tiles whose bytes change are written; every other byte is untouched;
- the edit is reversible: the pre-patch bytes of every written tile go to a
  `.revert.npz` sidecar with the patched file's digest, and `gguf-revert`
  restores them — or refuses, if the artifact changed since;
- the write is proven by reading back, not assumed.

The copy path (`--out`) stays for cutting a separate release artifact. It is no
longer the path an agent should take to edit a model it already has: a second
copy of a 27B model is disk it may not have, and a full dequant-requant of the
model is work nobody asked for.

## What this deprioritizes

- Research axes that do not either cut tokens-to-certification (R1) or widen
  architecture reach. A new fitting trick that only works on qwen3-4B is a
  measurement, not a feature.
- Throughput and serving-speed work. Speed is a multiplier on the same tokens.
- Anything an agent must glue together by hand between two commands.

## Order of work under this focus

1. **`capture`** — generic, in-tool, labels by answer shape, cross-checks the
   artifact it will be merged into. *Landed.*
2. **In-place merge + revert** — `spanfit build --in-place`,
   `--gguf-targets auto`, `gguf-revert`. *Landed.*
3. **`abliterate` as the one verb** — chain capture → fit (shape × scale ×
   layer floor) → screen → report, each stage writing its own JSON and ledger
   line, so an agent runs one command and reads one directory. The current
   `abliterate` builds an auto-search config for the older HF-trial pipeline;
   it must drive the researched spanfit loop instead.
4. **Verification stays the agent's read** (R2): the tool produces evidence
   packets; it never prints a removal claim it did not measure. Ease of use is
   not permission to overclaim, and the honest refusal is a feature of the
   interface too.

## Measured on the way in (2026-11-21)

- `capture` on qwen2-1.5B-Instruct: 10 rows (4 bad / 4 good / 2 ambiguous),
  1,008 generated tokens, 54 s including model load, targets found as
  `self_attn.o_proj` + `mlp.down_proj` × 28 layers.
- `capture` on spark2_5 1.7B (vendor remote code): 8 rows, 1,746 tokens, 141 s,
  targets found as `out_proj` + `down_proj` — a name the per-model table never
  had, discovered from the module tree. Artifact cross-check agreed 28/28.
- `spanfit axes --records <capture.pt>`: 0.4 s for [28, 1536].
- `spanfit build --in-place` on a 1.1 GB Q4_K_M: 56 tensors, 1,678,218 blocks
  changed, size byte-identical, sha restored to the exact original by
  `gguf-revert`. 1057 s on a loaded host, and it currently pays two dequant
  passes (one to compute the deltas, one to apply them) — the next CPU-cost
  item, not a correctness one.

