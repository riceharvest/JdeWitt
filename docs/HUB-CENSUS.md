# Hub census and adapter triage

Census artifact `verification/hub-census/2026-11-26.json`: taken 2026-11-26T17:28:06Z (2026-Q4), sha256 `701d2e28000cd448461fe51e47820d7679393413ac5344f3981fde1161131d25`.

137 text-generation repositories checked across 21 publishers (created since 2026-08-01, cap 40 per publisher); 13 derived GGUF/MLX artifacts skipped; 22 model types seen; 0 author errors. Installed transformers 5.20.0; latest published on PyPI 5.24.1.

Every number below is read from that artifact or measured by `scripts/hub_census_triage.py`; none is typed by hand.

## Adapter worklist — 7 rows, ordered by downloads

| model_type | downloads | example models | installed transformers 5.20.0 | auto_map | first published loader | decision |
|---|---:|---|---|---|---|---|
| `lfm3` | 226,041 | `LiquidAI/LFM3-1.7B-MLX-5bit`, `LiquidAI/LFM3-1.7B-MLX-4bit`, `LiquidAI/LFM3-1.7B-MLX-6bit` | no | 3/3 | 5.23.0 | adapter card |
| `mimo_v2` | 130,428 | `XiaomiMiMo/MiMo-V2.6-Flash-MOPD`, `XiaomiMiMo/MiMo-V2.6-Pro-MOPD`, `XiaomiMiMo/MiMo-V2.6-Flash-RL` | no | 3/3 | none found | adapter card |
| `dory` | 60,494 | `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4`, `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-MXFP8`, `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16` | no | 0/3 | 5.24.0 | blocked on loader: transformers>=5.24.0 |
| `ernie5_5_moe` | 10,201 | `baidu/ERNIE-5.5-Flash-Preview-BF16`, `baidu/ERNIE-5.5-Flash-Preview-FP8`, `baidu/ERNIE-5.5-Flash-Preview` | no | 0/3 | none found | blocked on loader: no published loader registers this type |
| `nemotron_labs_sigma` | 8,945 | `nvidia/Nemotron-Labs-Sigma-distill-spec-3B`, `nvidia/Nemotron-Labs-Sigma-distill-8B-nfe16`, `nvidia/Nemotron-Labs-Sigma-distill-8B-nfe8` | no | 3/3 | none found | adapter card |
| `bolmo` | 4,403 | `allenai/Llama-3-Blama-8B`, `allenai/Llama-3-Blama-8B-Stage1`, `allenai/Bwen-8B` | no | 3/3 | none found | adapter card |
| `llada2_moe` | 1,568 | `inclusionAI/LLaDA2.2-mini` | no | 1/1 | none found | adapter card |

## Untyped repositories — 1 row

| label | downloads | example models | decision |
|---|---:|---|---|
| `no-model-type` | 205,077 | `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash`, `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB`, `nvidia/NVIDIA-Nemotron-Labs-3-Competitive-Coding-550B-A55B-NVFP4` | not-an-architecture |

These repositories have no `config.json` `model_type` (some have no `config.json` at all). A missing label is a census artifact, not an architecture: they stay out of `adapter_worklist` and no adapter card is written for them.

## How the decision was made

- `installed transformers`: whether `model_type` is a key of `CONFIG_MAPPING_NAMES` in the transformers installed for this run; none of the worklist rows is, so the installed build cannot name these architectures itself.
- `auto_map`: how many of the row's example checkpoints ship remote-code configs. Those load with `trust_remote_code` today, so writing the adapter is not blocked on a release; a published floor is still recorded because installing it drops the remote-code dependency.
- `first published loader`: the first PyPI-released transformers version at or above the installed one whose `src/transformers/models/auto/auto_mappings.py` registers the type, searched over 5.20.0, 5.20.1, 5.20.2, 5.21.0, 5.21.1, 5.22.0, 5.22.1, 5.22.2, 5.23.0, 5.24.0, 5.24.1 (read from the GitHub tag, or from the published wheel where a release has no tag).
- A row is `blocked on loader` only when the installed build can read neither a registered type nor an `auto_map`. The decision names the floor that unblocks it, or says that no published loader registers the type.

## Next — 14 commands, a capture/tune pair per worklist row

Copied from the artifact's `next` list (one pair per architecture, in the same download order). Each capture reads the checkpoint its row actually saw; angle-bracket paths are placeholders this census never built, and every line parses as a CLI command.

```
uv run jdewitt capture --model LiquidAI/LFM3-1.7B-MLX-5bit --suite "<suite.jsonl>" --ctx-size 4096 --out records-lfm3.pt --gguf "<lfm3-artifact.gguf>"
uv run jdewitt tune --gguf "<lfm3-artifact.gguf>" --records records-lfm3.pt
uv run jdewitt capture --model XiaomiMiMo/MiMo-V2.6-Flash-MOPD --suite "<suite.jsonl>" --ctx-size 4096 --out records-mimo_v2.pt --gguf "<mimo_v2-artifact.gguf>"
uv run jdewitt tune --gguf "<mimo_v2-artifact.gguf>" --records records-mimo_v2.pt
uv run jdewitt capture --model nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4 --suite "<suite.jsonl>" --ctx-size 4096 --out records-dory.pt --gguf "<dory-artifact.gguf>"
uv run jdewitt tune --gguf "<dory-artifact.gguf>" --records records-dory.pt
uv run jdewitt capture --model baidu/ERNIE-5.5-Flash-Preview-BF16 --suite "<suite.jsonl>" --ctx-size 4096 --out records-ernie5_5_moe.pt --gguf "<ernie5_5_moe-artifact.gguf>"
uv run jdewitt tune --gguf "<ernie5_5_moe-artifact.gguf>" --records records-ernie5_5_moe.pt
uv run jdewitt capture --model nvidia/Nemotron-Labs-Sigma-distill-spec-3B --suite "<suite.jsonl>" --ctx-size 4096 --out records-nemotron_labs_sigma.pt --gguf "<nemotron_labs_sigma-artifact.gguf>"
uv run jdewitt tune --gguf "<nemotron_labs_sigma-artifact.gguf>" --records records-nemotron_labs_sigma.pt
uv run jdewitt capture --model allenai/Llama-3-Blama-8B --suite "<suite.jsonl>" --ctx-size 4096 --out records-bolmo.pt --gguf "<bolmo-artifact.gguf>"
uv run jdewitt tune --gguf "<bolmo-artifact.gguf>" --records records-bolmo.pt
uv run jdewitt capture --model inclusionAI/LLaDA2.2-mini --suite "<suite.jsonl>" --ctx-size 4096 --out records-llada2_moe.pt --gguf "<llada2_moe-artifact.gguf>"
uv run jdewitt tune --gguf "<llada2_moe-artifact.gguf>" --records records-llada2_moe.pt
```

## Human gap list

The curated `unresolved_gaps` list stays in the same artifact (6 entries) and is not re-derived here.

## Reproduce

```
.venv/bin/python -m jdewitt hub-census --out verification/hub-census/2026-11-26.json
.venv/bin/python scripts/hub_census_triage.py verification/hub-census/2026-11-26.json --out docs/HUB-CENSUS.md --evidence verification/hub-census/2026-11-26-loader.json
.venv/bin/python -m jdewitt verify-artifact verification/hub-census/2026-11-26.json
```
The census write is an exclusive create: re-running the same date exits 1 with `FILE_EXISTS` instead of overwriting the observation.
