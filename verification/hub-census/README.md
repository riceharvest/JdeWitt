# Hub census artifacts

`census-2026-11-22.json` is the raw stdout of

    .venv/bin/python -m jdewitt hub-census

run on 2026-11-22 19:21 CET (3.2 s, exit 0, 0 author errors). It is kept as the
triage baseline for the P8 census card: that card adds `--out` (and splits the
`no-model-type` row out of `adapter_worklist`), so this file is the
before-picture, not an artifact of the new flag.

Measured content: 134 repositories checked, 13 derived-artifact repositories,
22 model types, an 8-row worklist (top: no-model-type 214,731; lfm3 212,493;
mimo_v2 121,714; dory 61,627), sha256
ed53d0a5b239f279608d31e6a99a72de809e3efab5bbf192e9c5056102c9e9d8.

Scope, as the payload itself says: adapter labels come from config.json
model_type, downloads are a demand proxy, nothing here is a support claim.

## The `--out` artifact (same day, after the flag landed)

`2026-11-22.json` is the first artifact written by

    .venv/bin/python -m jdewitt hub-census --out verification/hub-census/2026-11-22.json

(exit 0; a second identical call exits 1 with `FILE_EXISTS` instead of
overwriting it). It is the same payload with two changes: the `no-model-type`
row is split out of `adapter_worklist` into `untyped_repositories`, and the run
is stamped `census_at`/`quarter`. Measured content: the same 134 repositories,
13 derived artifacts, 22 model types and 0 author errors, a 7-row worklist
(top: lfm3 212,493; mimo_v2 121,714; dory 61,627), 1 untyped row
(no-model-type 214,731), quarter `2026-Q4`, sha256
421805ae80fc63701ed8c97b2b21150a61775b4ef5a85f23970e874a38b89f7f.

`2026-11-22-loader.json` is the loader evidence behind the triage table: the
transformers installed for the run (5.20.0), the PyPI releases searched up to
5.24.0, and per row the installed support, `auto_map` count, first published
loader and decision. `scripts/hub_census_triage.py` writes it and renders
`docs/HUB-CENSUS.md` from the artifact, so no number there is retyped by hand.


## The 2026-11-26 refresh

`2026-11-26.json` is the next quarterly observation, written by

    uv run jdewitt hub-census --out verification/hub-census/2026-11-26.json

on 2026-11-26 (3.2 s, exit 0, 0 author errors), and `2026-11-26-loader.json` is
the loader evidence `scripts/hub_census_triage.py` measured for it. This
artifact is also the first one carrying the `next` block: one capture/tune pair
per worklist row, in the same download order, built from the checkpoint the row
actually saw (14 commands; `docs/HUB-CENSUS.md` renders them from the artifact,
never by hand).

Measured content: 137 repositories checked (up from 134), 13 derived
GGUF/MLX artifacts skipped, 22 model types, a 7-row worklist (top: lfm3 226,041;
mimo_v2 130,428; dory 60,494; ernie5_5_moe 10,201; nemotron_labs_sigma 8,945;
bolmo 4,403; llada2_moe 1,568), 1 untyped row (`no-model-type` 205,077),
quarter `2026-Q4`, sha256
701d2e28000cd448461fe51e47820d7679393413ac5344f3981fde1161131d25. The triage
decisions are unchanged in kind from 2026-11-22 (5 adapter cards, dory blocked on
transformers>=5.24.0, ernie5_5_moe blocked on a loader nobody publishes), but the
ernie row's examples moved to the `baidu/ERNIE-5.5-Flash-Preview*` checkpoints
and every download count was re-read.

Reproducibility: a second run without `--out` produced the same JSON apart from
`census_at`, and re-rendering the triage table from the same artifact produced
byte-identical markdown and evidence.
