# Hub census and adapter triage

Census artifact `verification/hub-census/2026-11-22.json`: taken 2026-11-22T18:39:18Z (2026-Q4), sha256 `421805ae80fc63701ed8c97b2b21150a61775b4ef5a85f23970e874a38b89f7f`.

134 text-generation repositories checked across 21 publishers (created since 2026-08-01, cap 40 per publisher); 13 derived GGUF/MLX artifacts skipped; 22 model types seen; 0 author errors. Installed transformers 5.20.0; latest published on PyPI 5.24.0.

Every number below is read from that artifact or measured by `scripts/hub_census_triage.py`; none is typed by hand.

## Adapter worklist — 7 rows, ordered by downloads

| model_type | downloads | example models | installed transformers 5.20.0 | auto_map | first published loader | decision |
|---|---:|---|---|---|---|---|
| `lfm3` | 212,493 | `LiquidAI/LFM3-1.7B-MLX-5bit`, `LiquidAI/LFM3-1.7B-MLX-4bit`, `LiquidAI/LFM3-1.7B-MLX-6bit` | no | 3/3 | 5.23.0 | adapter card |
| `mimo_v2` | 121,714 | `XiaomiMiMo/MiMo-V2.6-Flash-MOPD`, `XiaomiMiMo/MiMo-V2.6-Pro-MOPD`, `XiaomiMiMo/MiMo-V2.6-Flash-RL` | no | 3/3 | none found | adapter card |
| `dory` | 61,627 | `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4`, `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-MXFP8`, `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16` | no | 0/3 | 5.24.0 | blocked on loader: transformers>=5.24.0 |
| `ernie5_5_moe` | 7,449 | `baidu/ERNIE-5.5-Flash-Base`, `baidu/ERNIE-5.5-Flash-Base-BF16` | no | 0/2 | none found | blocked on loader: no published loader registers this type |
| `nemotron_labs_sigma` | 6,277 | `nvidia/Nemotron-Labs-Sigma-distill-spec-3B`, `nvidia/Nemotron-Labs-Sigma-distill-8B-nfe16`, `nvidia/Nemotron-Labs-Sigma-distill-8B-nfe8` | no | 3/3 | none found | adapter card |
| `bolmo` | 4,408 | `allenai/Llama-3-Blama-8B`, `allenai/Llama-3-Blama-8B-Stage1`, `allenai/Bwen-8B` | no | 3/3 | none found | adapter card |
| `llada2_moe` | 1,827 | `inclusionAI/LLaDA2.2-mini` | no | 1/1 | none found | adapter card |

## Untyped repositories — 1 row

| label | downloads | example models | decision |
|---|---:|---|---|
| `no-model-type` | 214,731 | `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash`, `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB`, `nvidia/NVIDIA-Nemotron-Labs-3-Competitive-Coding-550B-A55B-NVFP4` | not-an-architecture |

These repositories have no `config.json` `model_type` (some have no `config.json` at all). A missing label is a census artifact, not an architecture: they stay out of `adapter_worklist` and no adapter card is written for them.

## How the decision was made

- `installed transformers`: whether `model_type` is a key of `CONFIG_MAPPING_NAMES` in the transformers installed for this run; none of the worklist rows is, so the installed build cannot name these architectures itself.
- `auto_map`: how many of the row's example checkpoints ship remote-code configs. Those load with `trust_remote_code` today, so writing the adapter is not blocked on a release; a published floor is still recorded because installing it drops the remote-code dependency.
- `first published loader`: the first PyPI-released transformers version at or above the installed one whose `src/transformers/models/auto/auto_mappings.py` registers the type, searched over 5.20.0, 5.20.1, 5.20.2, 5.21.0, 5.21.1, 5.22.0, 5.22.1, 5.22.2, 5.23.0, 5.24.0 (read from the GitHub tag, or from the published wheel where a release has no tag).
- A row is `blocked on loader` only when the installed build can read neither a registered type nor an `auto_map`. The decision names the floor that unblocks it, or says that no published loader registers the type.

## Human gap list

The curated `unresolved_gaps` list stays in the same artifact (6 entries) and is not re-derived here.

## Reproduce

```
.venv/bin/python -m jdewitt hub-census --out verification/hub-census/2026-11-22.json
.venv/bin/python scripts/hub_census_triage.py verification/hub-census/2026-11-22.json --out docs/HUB-CENSUS.md --evidence verification/hub-census/2026-11-22-loader.json
.venv/bin/python -m jdewitt verify-artifact verification/hub-census/2026-11-22.json
```
The census write is an exclusive create: re-running the same date exits 1 with `FILE_EXISTS` instead of overwriting the observation.
