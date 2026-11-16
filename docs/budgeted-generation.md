# Budgeted generation and rounded-weight reuse

These changes repair the complete evaluation path, not just its direct backend.
They are covered by CPU tensor/transport fixtures and full scheduling tests. New
batching and cache behavior still needs a B70 throughput and parity run: GPU tests
were deliberately not executed while the card was reserved.

## Correctness fixes

- `PhasedHF` forwards the refusal floor to its target, including during batching.
- Baseline and finalist Q4/Q5 generation use the same configured refusal floor as
  the original-precision evaluation. Floor and batch policy are bound to answer
  identities, so old answers cannot silently inherit new settings.
- The generation ledger rejects a backend that does not confirm the floor.
- Full-surface MPOA restores edited row norms on both sides. For read-side `pre`,
  normalizing each input row first and restoring it cancels algebraically; `full`
  rescales the final edited rows. A collapsed nonzero row is rejected, not divided
  by zero.
- Read-side LEACE applies `W (I - strength * left @ right)`; write-side LEACE
  applies `(I - strength * left @ right) W`. Grouped-query k/v can therefore use
  their narrower output width without multiplying incompatible matrices.
- Replacing learned factors clears any row scales left by a preceding edit.

## Batches with a token budget

The optional run field `generation_batch` has exactly two positive integer keys:
`max_sequences` and `max_padded_tokens`. Omit it for the existing serial policy.
Values must be selected explicitly; the tool does not probe the GPU or guess a
safe batch size.

Admission counts the rendered prompts, not user-word lengths, and reserves the
full configured output budget for each row:

    admitted_tokens = sequences * (longest_rendered_prompt + max_tokens)

Tasks are grouped by generation policy (floored refusal versus natural-stop
capability), sorted by length, and packed under both limits. Every prompt is
preflighted before generation. This conservative token bound is not a VRAM-fit
calculation or a claim of saturated KV utilization.

HF batches use one left-padded model.generate call. Each result removes left
padding and trailing finished-row padding before saving its trajectory. Loopback
GGUF batches issue bounded concurrent requests. Managed target servers allocate
`ctx_size * max_sequences` total context so adding slots does not silently shrink
an individual request's context. Judge servers remain separate, single-slot
lifetimes. No CPU model offload is introduced.

Every returned answer is committed independently. Interrupted replay preserves
the original batch companions because changing batch shape can change low-
precision arithmetic; already saved companions may be recomputed but are never
overwritten. Completed jobs do not start a model. The batch policy is part of
generation identity; changing it requires a new answer/run directory.

The standalone `generate` command accepts `--refusal-floor`, plus the paired
`--batch-max-sequences` and `--batch-token-budget` options. Both batch options are
required when either is supplied. These have the same identity and admission
rules as full runs.

## Bounded merged-weight cache

`merged_cache_bytes` is an optional nonnegative integer, default `0` (disabled).
It is one shared budget across all edited projections, not an allowance per
layer. The full budget is reserved alongside the projection workspace and the
configured VRAM headroom before evaluation.

When enabled, admitted rounded matrices remain on the original weight's device
and are reused instead of rebuilding the FP32 low-rank update every decode step.
Only final rounded matrices are retained. Original weights remain immutable;
streaming export still uses the same FP32 addition and final cast.

Admission pins matrices until their edit changes. It deliberately does not use
an LRU eviction loop: with a partial-model budget, sequential traversal would
otherwise churn through all layers every token and get no reuse. Oversized or
unadmitted projections retain the existing one-projection transient path.

Cache signatures track tensor identity, dtype, device and mutation versions.
Changed factors or weights invalidate their entry; reset clears it. Tensors
without a mutation counter bypass reuse. Gradient-enabled tensors bypass the
cache so training graphs are never detached. The byte limit bounds persistent
cache allocations, not temporary GEMM workspaces, KV state, or tensors retained
by external callers.

## Less bookkeeping work

Answer generation no longer rereads every saved answer after every prompt. It
validates the ledger once, updates the completed counter while holding the job
lock, and verifies final status once. Individual record validation, atomic writes,
and final integrity checks remain in place.

No wall-clock speedup over Heretic or Abliterix is claimed without matched B70
measurements. Length packing, bounded concurrency, cache hit behavior, immutable
base weights and interruption handling are exercised by tests, not inferred from
an API accepting their configuration.
