# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build tests/test_spanfit.py's fixture and merge it -- the byte-identity probe.

P2b moved the row-norm (`--renormalize`) arithmetic out of build_merged's
precomputed float32 delta and into the writer, and replaced that delta with an
`(axis, space)` projection. The claim "the out-of-place merge is byte-identical
to before" needs the same inputs run through the old and the new code, so this
script is deliberately self-contained: it rebuilds the fixture from
`np.random.default_rng(3)` (same tensors, same axes as the test), merges in the
three modes that reach different writer branches, and prints one sha256 per
output file.

  # old code (any tree at master e1ece0c)          # new code (this tree)
  python scripts/p2b_fixture_identity.py --out-dir /tmp/base
  python scripts/p2b_fixture_identity.py --out-dir /tmp/new --compare /tmp/base

`--compare` re-hashes the other tree's files and reports `cmp` per mode, so one
run prints both sides. A warm second pass through the dequant cache is included
when the tree has the cache; it must land on the same bytes as the cold pass.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, '/mnt/ssd/dev workspace/llama.cpp-dspark/gguf-py')
sys.path.insert(0, str(Path.cwd()))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from gguf import GGUFWriter  # noqa: E402

from jdewitt import spanfit  # noqa: E402

L, D = 4, 16
MODES = (("input", "input", False),
         ("output", "output", False),
         ("output_renormalize", "output", True))


def build_fixture(outdir: Path):
    g = np.random.default_rng(3)
    tensors = {'blk.0.attn_output.weight': g.standard_normal((D, D)).astype(np.float32),
               'blk.0.ffn_down.weight': g.standard_normal((D, D)).astype(np.float32),
               'blk.1.attn_output.weight': g.standard_normal((D, D)).astype(np.float32),
               'token_embd.weight': g.standard_normal((32, D)).astype(np.float32)}
    src = outdir / 'in.gguf'
    writer = GGUFWriter(str(src), 'ggmf')
    for name, arr in tensors.items():
        writer.add_tensor(name, arr)
    writer.open_output_file()
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    # the axes and the name map are the test's, so the arrays the writer sees are
    # identical; only the code path under test differs between trees
    axes = torch.nn.functional.normalize(
        torch.from_numpy(g.standard_normal((L, D)).astype(np.float32)), dim=1)
    keys = ['model.layers.0.self_attn.o_proj.weight',
            'model.layers.0.mlp.down_proj.weight',
            'model.layers.1.self_attn.o_proj.weight']
    name_map = {keys[0]: 'blk.0.attn_output.weight',
                keys[1]: 'blk.0.ffn_down.weight',
                keys[2]: 'blk.1.attn_output.weight'}
    return src, axes, keys, name_map


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', required=True, help='where the fixture and the merged files go')
    ap.add_argument('--compare', help="another run's --out-dir (the other tree) to cmp against")
    ap.add_argument('--json-out', help='also write the report here')
    args = ap.parse_args(argv)

    outdir = Path(args.out_dir)
    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True)
    src, axes, keys, name_map = build_fixture(outdir)
    accepts_cache = 'dequant_cache' in inspect.signature(spanfit.build_merged).parameters

    rows = []
    for mode, space, renormalize in MODES:
        out = outdir / f'{mode}.gguf'
        kwargs = {'axis_space': space, 'renormalize': renormalize}
        if accepts_cache:
            kwargs['dequant_cache'] = False
        report = spanfit.build_merged(str(src), str(out), axes, keys, name_map=name_map,
                                      **kwargs)
        row = {'mode': mode, 'axis_space': space, 'renormalize': renormalize,
               'sha256': sha256(out), 'tensors_edited': report['tensors_edited']}
        if accepts_cache:
            warm = outdir / f'{mode}.warm.gguf'
            spanfit.build_merged(str(src), str(warm), axes, keys, name_map=name_map,
                                 **dict(kwargs, dequant_cache=str(outdir / 'cache')))
            row['warm_sha256'] = sha256(warm)
            row['warm_identical'] = row['warm_sha256'] == row['sha256']
        if args.compare:
            other = Path(args.compare) / f'{mode}.gguf'
            row['compare_sha256'] = sha256(other)
            row['cmp'] = 'identical' if other.read_bytes() == out.read_bytes() else 'DIFFERENT'
        rows.append(row)

    nxt = None
    if not args.compare:
        nxt = ('python scripts/merge_bench.py --artifact /mnt/ssd/models/spark/p2/bench.gguf '
               '--axes /mnt/ssd/models/spark/p2/axes.pt --arms 3 '
               '--cache /mnt/ssd/models/spark/p2/dequant '
               '--json-out verification/p2/bench-cache.json')
    payload = {'script': 'scripts/p2b_fixture_identity.py',
               'fixture': 'tests/test_spanfit.py fixture: L=4, D=16, 3 target tensors, F32',
               'fixture_in_sha256': sha256(src),
               'cache_available': accepts_cache,
               'rows': rows,
               'all_identical': all(r.get('cmp', 'identical') == 'identical' for r in rows),
               'next': nxt}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=1) + '\n')
    print(json.dumps(payload, indent=1))
    return payload


if __name__ == '__main__':
    main()
