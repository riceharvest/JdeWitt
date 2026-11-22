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

