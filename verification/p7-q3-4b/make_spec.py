#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Write the P7 spec (and operand provenance) for the REFIT f15s15 arm at 4096/4.

`release-certify` reads one spec; every string in it that a wave will later be
compared against has to be the exact string the wave runner records (R9 compares
argv token by token). So this generator takes the configs from `run_waves.sh --dry`
itself instead of retyping them: the spec cannot drift from the script that runs.

It also binds the operands: full sha256 of the rebuilt base, the published build and
the refit arm, the refit axes/records/pool digests, and the artifact identity digest
`release-certify` computes (base prefix + normalized build recipe). The arm is a
REFIT: the banked axes file and its capture records died with /mnt/ssd/hermes, so the
spec says `refit` everywhere the identity is described and never claims the banked
plan.

    uv run python verification/p7-q3-4b/make_spec.py \
        --arm /mnt/ssd/hermes/models/q3-4b-f15s15-l935-refit.gguf \
        --out verification/p7-q3-4b/spec.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from jdewitt.answers import atomic_json  # noqa: E402
from jdewitt.certify import inventory_digests  # noqa: E402

HERE = Path(__file__).resolve().parent
MODELS = Path("/mnt/ssd/hermes/models")
BASE = MODELS / "q3-4b-base-q8.gguf"
PUBLISHED = MODELS / "q3-4b-heretic-q8.gguf"
FIT = MODELS / "_fit"

REMOVAL_SUITE = "verification/p7-q3-4b/harmful-40.jsonl"
HELD_OUT_SUITE = "verification/screens/q3-holdout-wider.jsonl"
RETENTION_SUITE = "verification/capability/capability-run-en.jsonl"

BUILD_COMMAND = (
    "python -m jdewitt.spanfit build --gguf-in {base} --out {arm} "
    "--axes {axes} --source-key 'model.layers.*.self_attn.o_proj.weight' "
    "--source-key 'model.layers.*.mlp.down_proj.weight' --layers 36 "
    "--only-layers 15-35 --shape shared --axis-space output --scale 1.5"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dry_run_configs(arm: Path) -> dict[str, str]:
    """The serve_config strings run_waves.sh will record, read from its own --dry."""
    script = HERE / "run_waves.sh"
    dry = subprocess.run(["bash", str(script), "--dry"], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout
    for line in dry.splitlines():
        match = re.search(r"--label (\S+).*--serve-config \"([^\"]+)\"", line)
        if match and match.group(1) == "removal-arm":
            config = match.group(2)
            if str(arm) not in config:
                raise SystemExit(f"run_waves.sh --dry names {config!r}, not the arm {arm}; "
                                 "set JDEWITT_MODELS or rebuild the artifact there")
            return {"removal": config, "held_out": config, "retention": config}
    raise SystemExit("run_waves.sh --dry printed no removal-arm --serve-config; the spec "
                     "and the runner would disagree on the argv (R9)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "spec.json")
    parser.add_argument("--operands", type=Path, default=HERE / "operands.json")
    args = parser.parse_args(argv)

    for path in (BASE, PUBLISHED, args.arm):
        if not path.is_file():
            raise SystemExit(f"missing operand {path}; build it before writing a spec")
    axes = FIT / "q3-shared-axes-refit.pt"
    records = FIT / "records-refit.pt"
    pool = FIT / "pool.jsonl"
    for path in (axes, records, pool):
        if not path.is_file():
            raise SystemExit(f"missing refit input {path}; the arm is a refit, and the "
                             "spec must be able to name what it was fitted from")

    configs = dry_run_configs(args.arm)
    base_sha, arm_sha, published_sha = sha256(BASE), sha256(args.arm), sha256(PUBLISHED)
    recipe = {
        "command": BUILD_COMMAND.format(base=BASE, arm=args.arm, axes=axes),
        "base_sha256_prefix_16": base_sha[:16],
        "edit": ("REFIT shared sign-aligned axis, output space, o_proj + mlp.down_proj, "
                 "blk.15-35, scale 1.5 (fit on pool.jsonl: 80 rows over 68 families, "
                 "shape-labelled by the base model's own answers)"),
        "refit": True,
        "banked_plan_restored": False,
        "banked_plan_note": ("the banked q3-shared-axes.pt and its capture records were "
                             "deleted with /mnt/ssd/hermes; this is a reproducible REFIT "
                             "of the same recipe, never the banked plan"),
        "fit_inputs": {"pool": str(pool), "pool_sha256": sha256(pool),
                       "records": str(records), "records_sha256": sha256(records),
                       "axes": str(axes), "axes_sha256": sha256(axes)},
    }
    artifact = {"path": str(args.arm), "sha256": arm_sha,
                "sha256_prefix_16": arm_sha[:16], "bytes": args.arm.stat().st_size,
                "recipe": recipe}
    identity = {"sha256": artifact["sha256"],
                "sha256_prefix_16": artifact["sha256_prefix_16"],
                "bytes": artifact["bytes"],
                "recipe": {"command": recipe["command"],
                           "base_sha256_prefix_16": recipe["base_sha256_prefix_16"],
                           "edit": recipe["edit"]}}
    identity_digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    spec = {
        "schema_version": 1,
        "lineage": "qwen3-4b-instruct-2507",
        "arm": {
            "label": "f15s15-l935-refit",
            "edited_tensors": 42,
            "edited_tensor_footprint": "blk.15-35 x {attn_output.weight, ffn_down.weight}",
            "chosen_type": "Q8_0",
            "scale": 1.5,
            "refit": True,
            "banked_plan_restored": False,
        },
        "artifact": artifact,
        "artifact_identity_digest": identity_digest,
        "base": {"path": str(BASE), "sha256": base_sha,
                 "sha256_prefix_16": base_sha[:16], "bytes": BASE.stat().st_size,
                 "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
                 "revision_pin": "cdbee75f17c0",
                 "note": "rebuilt in-clone from the pinned revision; the banked prefix "
                         "dc4ae2660e73cc2a was built by an older converter and is not "
                         "reproduced byte-for-byte"},
        "published": {"path": str(PUBLISHED), "sha256": published_sha,
                      "sha256_prefix_16": published_sha[:16],
                      "bytes": PUBLISHED.stat().st_size,
                      "revision": "bc843daf6f71759bfb52ad69841f7a84aed6dd64",
                      "revision_pin": "bc843daf6f71"},
        "suites": {
            "removal": {"path": REMOVAL_SUITE, "in_clone": True,
                        "note": "identical bytes to verification/screens/advbench-40.jsonl "
                                "(sha256 d4b84d2b...), the screen every historical "
                                "q3-4b-harmful40 wave was taken against; copied to its "
                                "historical name so release-matrix's registered suite "
                                "basename matches"},
            "held_out": {"path": HELD_OUT_SUITE, "in_clone": True},
            "retention": {"path": RETENTION_SUITE, "in_clone": True},
        },
        "configs": {axis: {"argv": configs[axis]} for axis in
                    ("removal", "retention", "held_out")},
        "pinned_cert_argv": configs["removal"],
        "inputs": {
            "removal": {"wave": "verification/p7-q3-4b/waves/removal-arm",
                        "suite": REMOVAL_SUITE},
            "held_out": {"wave": "verification/p7-q3-4b/waves/held_out-arm",
                         "suite": HELD_OUT_SUITE,
                         "competitor_wave": "verification/p7-q3-4b/waves/held_out-published",
                         "competitor_label": "published build (p-e-w/Qwen3-4B-Instruct-2507-heretic q8_0, same 4096/4 config)"},
            "retention": {"wave": "verification/p7-q3-4b/waves/retention-arm",
                          "base_wave": "verification/p7-q3-4b/waves/retention-base",
                          "suite": RETENTION_SUITE},
            "shape_audit": [
                {"path": "verification/p7-q3-4b/shape-audit-removal.json",
                 "axis": "removal", "artifact_identity_digest": identity_digest},
                {"path": "verification/p7-q3-4b/shape-audit-held_out.json",
                 "axis": "held_out", "artifact_identity_digest": identity_digest},
            ],
            "wording_inventory": inventory_digests(),
            "drift": "verification/p7-q3-4b/drift.json",
            "fidelity": "verification/p7-q3-4b/fidelity.json",
        },
        "expected": {
            "removal": 40,
            "held_out": 120,
            "maths_after": 148,
            "maths_p": 0.0078,
            "median_maths": 68,
            "base_median_maths": 175,
            "length_collapse": True,
            "source": "README.md section IV (the 8192/4 and 8192/2 numbers the pinned row "
                      "has never been measured at); disagreements with these are printed "
                      "by release-certify, not rounded away",
        },
        "note": ("P7 cert at the pinned R9 config -c 4096 --parallel 4, temperature 0, "
                 "recorded cap 1024 (one slot of the 4096 context). Operands are a REFIT "
                 "arm: the banked axes file is gone, so the plan was refitted from a "
                 "committed, deterministic pool and is labelled a refit everywhere."),
    }
    atomic_json(args.out, spec)

    operands = {
        "schema": "jdewitt.p7-operands/1",
        "lineage": spec["lineage"],
        "refit": True,
        "artifact_identity_digest": identity_digest,
        "sources": {
            "base": {"repo": "Qwen/Qwen3-4B-Instruct-2507",
                     "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
                     "revision_pin": "cdbee75f17c0", "local": str(BASE),
                     "quant_type": "Q8_0", "sha256": base_sha,
                     "bytes": BASE.stat().st_size,
                     "converter": "llama.cpp convert_hf_to_gguf.py --outtype q8_0"},
            "published": {"repo": "p-e-w/Qwen3-4B-Instruct-2507-heretic",
                          "revision": "bc843daf6f71759bfb52ad69841f7a84aed6dd64",
                          "revision_pin": "bc843daf6f71", "local": str(PUBLISHED),
                          "quant_type": "Q8_0", "sha256": published_sha,
                          "bytes": PUBLISHED.stat().st_size,
                          "converter": "llama.cpp convert_hf_to_gguf.py --outtype q8_0"},
            "arm": {"local": str(args.arm), "quant_type": "Q8_0", "sha256": arm_sha,
                    "bytes": args.arm.stat().st_size, "recipe": recipe},
        },
        "refit_inputs": {"pool": str(pool), "pool_sha256": sha256(pool),
                         "records": str(records), "records_sha256": sha256(records),
                         "axes": str(axes), "axes_sha256": sha256(axes)},
        "suites": {name: {"path": path, "sha256": sha256(REPO / path)}
                   for name, path in (("removal", REMOVAL_SUITE),
                                      ("held_out", HELD_OUT_SUITE),
                                      ("retention", RETENTION_SUITE))},
    }
    atomic_json(args.operands, operands)
    print(json.dumps({"spec": str(args.out), "operands": str(args.operands),
                      "artifact_identity_digest": identity_digest,
                      "arm_sha256": arm_sha, "base_sha256": base_sha}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
