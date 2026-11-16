# Upstream review and recent architecture coverage

Checked 2026-10-16 against Heretic source revision `66a952b49b00bfd1854bdad769fee5fb225b53be` (source commit dated 2026-10-12). The saved GitHub index includes 221 issues and 247 PRs, including closed entries. Listing an item is not a claim that every issue body was exhaustively reviewed or that every upstream bug applies to this different codebase.

## Applicable reports traced to the private tool

| Upstream | Private-tool result | Evidence |
| --- | --- | --- |
| #487 / #493: reasoning-prefix contamination | Uses explicit final-answer boundaries, no inferred common prose prefix. Harmony spellings and unfinished analysis have regression tests. | `hf.py:BoundaryContract`, `scoring.py:answer_boundary_chars`, `tests/test_edit_plans.py` |
| #495 / #490: live LoRA/export rounding mismatch | Resident trials and export use the same FP32 addition/final-cast routine; exported edited and untouched tensors are checked, then candidates rescored. More expensive than live LoRA, not advertised as faster. | `editor.py:merged_weight`, `export.py:verify_resident_export`, `tests/test_resident_operator.py` |
| #476: edit factors carried between trials | Resets A, B and row scaling; seeded ARA initialization and saved exact factors. | `editor.py:reset`, `methods.py:fit_ara`, `tests/test_edit_plans.py` |
| #478: no-op tiny layer range and too many neighbors | Layer-relative search, real changed-output tests and explicit neighbor-count rejection. Fixed template date prevents day-dependent prompt rendering. | `search.py:suggest_parameters`, `chat.py`, `tests/test_methods.py` |
| #483: templates rejecting system messages | Only the known system-role template failure folds system content into the user message. Unrelated errors propagate. | `chat.py:render_chat`, `tests/test_architecture_extensions.py` |
| #391 / #339 / #402: shared/fused MoE experts skipped | Discovers shared experts, individual experts and fused 3-D tensors; explicit GPT-OSS orientation. Bottlenecked Muse Spark/Nemotron edits use the residual-space output projection rather than applying residual directions inside expert latent coordinates. | `architectures.py`, `editor.py:discover_targets`, `tests/test_architecture_extensions.py` |
| #384: output_hidden_states missing in vendor models | Captures residuals through layer hooks, including explicit Qwen4 hyperconnection width handling. Qwen4 stream averaging is an experimental adapter, not a quality guarantee. | `hf.py:residual_at_answer`, `hf.py:residual_vector` |
| #209: single-token KL misses later damage | Full-reply, teacher-forced KL with independent baseline/candidate recurrent caches and chunked logits; answer-only KL is reported separately. | `hf.py:trajectory_kl`, `tests/test_adapter_hybrid.py` |
| #348: invalid KL with suppressed tokens | Masked/suppressed nonfinite logits fail explicitly, not an apparent zero-damage success. General masked-token KL support is NOT claimed. | `numerics.py:token_kl` |
| #210: false refusals on benign tasks | Harmless replies receive wording checks; paired harmless regressions fail candidate gates. | `evaluation.py`, `scoring.py:paired_consistency`, `tests/test_extension_regressions.py` |
| #244 / #477 / #288 / #360: marker gaming and nonanswers | Optional grounded semantic judge with authored requirements; wording-only mode remains explicitly weaker and is never called semantic compliance. | `grounded_judge.py`, `refusal_check.py`, `tests/test_grounded_judge.py` |
| #393: accelerator SVD convergence failure | Small activation statistics use CPU SVD, not GPU solver or model-weight offload. Decomposition failures still propagate. | `plans.py`, `methods.py`, `numerics.py` |
| #435 / #434: futile dtype retries | Exact requested dtype, no silent retry to another device or dtype. | `hf.py:HFModel` |
| #408: empty scorer prompt sets | Empty suites rejected before evaluation. | `data.py:load_suite` |
| #441 / #293: heavyweight CLI imports | Help/catalogues/status avoid model runtimes; optional bnb code is not imported for those commands. | `tests/test_agent_interface.py`, `tests/test_extension_regressions.py` |
| #381 / #379: additional-trial accounting | Attempted trials count once; unfinished trials resume with existing IDs and parameters; keyed TPE continuation has restart tests. | `search.py`, `tests/test_search_resume.py` |
| #317: auxiliary MTP/draft weights lost on save | Original root safetensor tensor inventory is streamed and verified, not reconstructed from editable modules alone. Nested custom asset directories are not covered by this claim. | `export.py:stream_merge`, `export.py:verify_resident_export` |
| #447: model tensor integrity | Exported tensors and release bundle files verified by exact contents/hash; no claim of upstream reproducibility on another platform. | `export.py`, `release.py`, `tests/test_release_integrity.py` |
| #481: holdout leakage | Disjoint family IDs, selected finalist checked once; failed holdout does not select a replacement candidate. | `pipeline.py`, `data.py:validate_splits` |
| #497: missing LFS hash metadata | Not applicable to this tool's local hashing path; no Hub-upload API or dependency on `RepoSibling.lfs`. | `release.py:sha256_file` |

Windows, TPU, multi-GPU, Apple MLX, remote uploads and assorted UI/docs reports are outside this B70-only tool's scope. Diffusion-model support, native quantized source editing and router interventions are unresolved features, not fixed by renaming an architecture. This review does NOT establish that all Heretic issues are fixed.

## New private-tool fixes from this pass

- Entirely rejected direction languages remain visible instead of disappearing from coverage checks.
- Completed conversion reuse binds importance-matrix bytes, not just the path.
- Direction checkpoints bind method/options, judge/scorer identity, policy and seed. Trial answer identities also bind the committed plan payload hash. Warm starts cannot silently cross different methods/options.
- Invalid method and training settings fail before expensive weight loading.
- SOM fits harmful prototypes before subtracting the harmless centroid, rather than clustering paired differences under the same method name.
- ARA loss evaluation avoids a full FP32 base-weight copy and a dense output-by-input delta allocation. B70 speedup is not claimed without a matched workload benchmark.

## July–October census

`verification/research/recent-model-census.json` records a bounded Hub creation-date census from 2026-07-16 through 2026-10-16 across 21 publishers. It contains 160 model artifacts and 31 non-null model_type names; 21 artifacts have no model_type in the API metadata. Publisher pagination completed. This is not a census of every Hub author, and repository creation time is not always architecture release time.

The installed adapter registry contains 61 model_type names. Native definition tests cover 24 text/backbone/wrapper layouts, with additional fused-expert orientation, system-role and latent-projection tests. Registration, definition discovery, actual B70 kernels and a real checkpoint round trip are different claims.

Recent supported layout families include Qwen3.8/4-Exp, GLM5/5.3, Gemma4, Muse Glimmer/Spark, DeepSeek V4, LFM2/2.5, OLMo3, MiniMax M3 text, Step3.5/3.7, Mistral4, Hy3/Hy4, Ling/Bailing hybrid, native MiMo V2 Flash and NemotronH/Omni. Marketing versions reuse older model_type values; they were not guessed from names. Vision/audio towers are not ablated.

Run `b70-ablator architectures` for the packaged machine-readable inventory and pinned config URLs. It explicitly identifies unimplemented DeepSeek V4.1 and Kimi K3 loaders, custom MiMo V2.6/Bolmo/Youtu wrappers, LLaDA/d1 diffusion samplers, custom Ling VL, draft-only Muse assistant, and Qwen Drive planning models. They are NOT silently routed through an older similarly named adapter.

No full recent pretrained checkpoint is newly certified by this pass. Some published MoEs cannot fit their original-precision weights on one B70 regardless of adapter support. Native model definition availability does not establish XPU kernel or GGUF converter compatibility. No CPU weight offload or other GPU is substituted to make that claim.
