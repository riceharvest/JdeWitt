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
