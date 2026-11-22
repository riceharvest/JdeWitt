# SPDX-License-Identifier: AGPL-3.0-or-later
"""One arm of the in-place merge, timed per phase, reverted exactly.

The phase gate wants four merged arms screened in under three minutes, and the
one number on the record is 550 s for a single in-place merge of the winner
(36 tensors / 1,093,269 blocks, docs/PLAN-NEXT.md "R8 verification of the
shipped pick"). Nothing says where those seconds go, so every later card is
argued against a guess.

  python scripts/merge_bench.py --artifact /mnt/ssd/models/spark/p2/bench.gguf \
      --axes /mnt/ssd/models/spark/p2/axes.pt --only-layers 10-27 --scale 1.0 \
      --arms 1 --json-out verification/p2/bench-baseline.json

One command, one JSON: artifact sha before and after revert, tensors and
changed blocks, per-arm dequant/apply/requant seconds, requant_share,
projected_4_arm_s, records_sha256, and `next` (the card this run hands to).

How the split is measured. Production code has no timers, and this script does
not add any: it runs build_merged in its own process and, for the duration of
one arm, replaces the quantize/dequantize callables in jdewitt.quants.CODECS
with counting wrappers. ggufio._tensor_info hands those callables to the
in-place writer, which dequantizes each tensor once and quantizes each tile
once, so the two counters are exactly the codec cost of the arm. apply_s is the
remainder: the rank-1 projection arithmetic, block-change detection, file I/O
and the writer's closing sha256. The wrappers are restored before the arm
returns, so the revert below runs on untouched codecs.

The artifact is patched in place and then reverted through the writer's own
revert sidecar; the sweep refuses to report unless the sha256 comes back to the
byte-for-byte pre-merge value. It never starts a server and never produces a
token: the merge is numpy over GGUF bytes, and the numbers it prints are
wall-clock seconds, not model output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

# The pinned Spark 1.7B Q4_K_M the R8 record was taken on. A copy of this file
# is what the bench patches; if the copy has drifted the measurement is about a
# different artifact, so the sha is checked before the first byte moves.
ART_SHA256 = '80ff5222ae51de055fe2e86b1ec4f6784c2f9db704ed3b043b97ce900d3a75ad'

# P2b (t_35cd0a5b) is the card this baseline unblocks: one dequant pass plus a
# content-addressed base cache. Its gate command is printed in `next` so the
# next agent starts from this JSON instead of from this docstring; a run that
# already used the cache prints the card it hands to instead (P2c), because
# re-printing its own command would be a loop.
P2B_NEXT = (
    'python scripts/merge_bench.py --artifact {artifact} --axes {axes} '
    '--only-layers {only_layers} --arms 3 --cache /mnt/ssd/models/spark/p2/dequant '
    '--json-out verification/p2/bench-cache.json   # P2b (t_35cd0a5b): the same bench '
    'with a dequant cache; arms[1].dequant_s must read 0.0 and cache_hits==cache_misses==36'
)
P2C_NEXT = (
    'ab show t_542ab2dc   # P2c: parallel requant across tensors. This file is the P2b '
    'gate (arm 0 dequant > 0, arms 1+ dequant_s == 0.0, cache_hits == cache_misses == '
    '{tensors}); requant_share above is why the 4-arm <3min gate (t_cf83773e) lives there'
)


def next_command(artifact: str, axes: str, only_layers: str, cached: bool, tensors: int) -> str:
    if cached:
        return P2C_NEXT.format(tensors=tensors)
    return P2B_NEXT.format(artifact=artifact, axes=axes, only_layers=only_layers)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b''):
            digest.update(chunk)
    return digest.hexdigest()


class PhaseClock:
    """Seconds and call counts for the two codec phases of one arm."""

    def __init__(self):
        self.dequant_s = 0.0
        self.requant_s = 0.0
        self.dequantize_calls = 0
        self.quantize_calls = 0

    def as_dict(self, total_s: float) -> dict:
        # apply_s is defined as the remainder, so the three buckets always sum
        # to the arm's wall clock and nothing is silently dropped.
        apply_s = max(0.0, total_s - self.dequant_s - self.requant_s)
        return {'dequant_s': round(self.dequant_s, 3), 'apply_s': round(apply_s, 3),
                'requant_s': round(self.requant_s, 3), 'total_s': round(total_s, 3),
                'dequantize_calls': self.dequantize_calls,
                'quantize_calls': self.quantize_calls}


def install_timed_codecs(clock: PhaseClock) -> dict:
    """Wrap the codec callables in place; return the originals for restore.

    quants.CODECS is the dict ggufio imported by name, so replacing its entries
    reaches the writer without a single line of production code knowing a timer
    exists. Every dtype is wrapped (not just Q4_K) so this keeps working when
    the bench is pointed at Q5_K / F16 arms.
    """
    from jdewitt import quants
    saved = dict(quants.CODECS)
    for dtype, (block_bytes, block_els, quantize, dequantize) in saved.items():
        def timed_dequantize(raw, n, _fn=dequantize):
            start = time.perf_counter()
            out = _fn(raw, n)
            clock.dequant_s += time.perf_counter() - start
            clock.dequantize_calls += 1
            return out

        def timed_quantize(values, _fn=quantize):
            start = time.perf_counter()
            out = _fn(values)
            clock.requant_s += time.perf_counter() - start
            clock.quantize_calls += 1
            return out

        quants.CODECS[dtype] = (block_bytes, block_els, timed_quantize, timed_dequantize)
    return saved


def restore_codecs(saved: dict) -> None:
    from jdewitt import quants
    for dtype, codec in saved.items():
        quants.CODECS[dtype] = codec


def merge_targets(artifact: str, axes, only_layers: str | None) -> tuple[dict, list[int]]:
    """name_map for this artifact's residual-write projections, restricted to layers.

    Same discovery spanfit's `--gguf-targets auto` uses: the artifact reports
    the tensors it carries at the axes' width, and a layer with no target is
    refused rather than skipped, because a merge that quietly edits half the
    footprint reports a complete edit.
    """
    from jdewitt.gguf_adapters import discover_writeback_targets, gguf_weight_shapes
    from jdewitt.spanfit import parse_layer_ids

    layer_count = int(axes.shape[0])
    ids = parse_layer_ids(only_layers) if only_layers else list(range(layer_count))
    if ids and any(i >= layer_count for i in ids):
        raise SystemExit(f'--only-layers {ids} reaches past the axes file ({layer_count} layers)')
    shapes = gguf_weight_shapes(artifact)
    found = discover_writeback_targets(shapes, int(axes.shape[-1]))
    targets = {layer: names for layer, names in found['targets'].items() if layer in set(ids)}
    missing = sorted(set(ids) - set(targets))
    if missing:
        raise SystemExit(
            f'this artifact has no residual-write projection at width {int(axes.shape[-1])} '
            f'for layers {missing[:8]} ({len(missing)} of {len(ids)}); found layers '
            f'{sorted(targets)}. The axes and the artifact disagree, so a merge here would '
            f'edit part of the model and look complete')
    name_map = {name: name for names in targets.values() for name in names}
    return name_map, sorted(targets)


def run_arm(artifact: Path, axes, name_map: dict, scale: float, axis_space: str,
            cache=None) -> dict:
    """Merge in place, revert, and prove the bytes came back; return one arm row."""
    from jdewitt.ggufio import revert_inplace
    from jdewitt.spanfit import build_merged

    before = sha256_file(artifact)
    clock = PhaseClock()
    snap = cache.snapshot() if cache is not None else None
    saved = install_timed_codecs(clock)
    try:
        start = time.perf_counter()
        report = build_merged(str(artifact), str(artifact), axes, [], name_map=name_map,
                              scale=scale, axis_space=axis_space, in_place=True,
                              dequant_cache=cache)
        total_s = time.perf_counter() - start
    finally:
        restore_codecs(saved)
    sidecar = report.get('revert_sidecar')
    if not sidecar or not Path(sidecar).is_file():
        raise SystemExit('build_merged returned no revert sidecar; the merge is not reversible '
                         'and must not be measured as one')
    start = time.perf_counter()
    revert = revert_inplace(artifact, sidecar)
    revert_s = time.perf_counter() - start
    after = sha256_file(artifact)
    if after != before:
        raise SystemExit(f'revert did not restore {artifact}: {before[:16]} -> {after[:16]}')
    if Path(sidecar).exists():
        raise SystemExit(f'revert left its sidecar behind: {sidecar}')
    row = clock.as_dict(total_s)
    row.update({'scale': scale,
                'tensors': int(report.get('tensors_edited', 0)),
                'tensors_changed': int(report.get('tensors_changed', 0)),
                'changed_blocks_total': int(report.get('changed_blocks_total', 0)),
                'revert_s': round(revert_s, 3),
                'tiles_restored': int(revert.get('tiles_restored', 0)),
                'artifact_sha_after_revert': after})
    if snap is not None:
        # Per-arm access counts: a warm arm must show hits == tensors and misses
        # == 0, and dequant_s == 0.0 above proves no codec call was made.
        row.update(cache.since(snap))
    return row


def _purge_cache(root: Path) -> None:
    """Drop previously cached bases so a bench run is cold on arm 0.

    A benchmark's first arm is the dequant cost; a warm directory would report
    0 misses and the run would no longer measure what it claims. Only the
    hash-named .npy entries this cache writes are removed.
    """
    root.mkdir(parents=True, exist_ok=True)
    for pattern in ('*.npy', '*.npy.tmp.npy'):
        for path in root.glob(pattern):
            path.unlink()


def bench(args) -> tuple[dict, str]:
    import torch
    from jdewitt.answers import atomic_json

    artifact = Path(args.artifact)
    axes_path = Path(args.axes)
    for candidate in (artifact, axes_path):
        if not candidate.is_file():
            raise SystemExit(f'{candidate} does not exist')
    payload = torch.load(axes_path, map_location='cpu', weights_only=False)
    if args.axes_key not in payload:
        raise SystemExit(f'{axes_path} has keys {sorted(payload)}, not {args.axes_key!r}')
    axes = payload[args.axes_key]
    if axes.dim() != 2:
        raise SystemExit(f'axes must be [layers, width], got {list(axes.shape)}')

    before = sha256_file(artifact)
    if args.expected_sha and before != args.expected_sha:
        raise SystemExit(f'{artifact} sha256 {before} != expected {args.expected_sha}; the bench '
                         'patches a byte copy of the pinned quant, and this is a different file')
    records = Path(args.records) if args.records else axes_path.with_name('records.pt')
    records_sha = sha256_file(records) if records.is_file() else None

    import jdewitt
    code_root = str(Path(jdewitt.__file__).resolve().parent)
    print(f'code: {code_root}', flush=True)

    name_map, layer_ids = merge_targets(str(artifact), axes, args.only_layers)
    names = sorted(name_map)
    print(f'{len(names)} tensors over layers {layer_ids[0]}-{layer_ids[-1]}: '
          f'{names[0]} .. {names[-1]}', flush=True)

    cache = None
    if args.cache:
        from jdewitt.ggufio import DequantBases
        if not args.cache_keep:
            _purge_cache(Path(args.cache))
        cache = DequantBases(args.cache)
        print(f'dequant cache {cache.root} ({"reused" if args.cache_keep else "purged"}): '
              f'arm 0 misses, later arms hit', flush=True)

    arm_rows = []
    for index in range(args.arms):
        row = run_arm(artifact, axes, name_map, args.scale, args.axis_space, cache=cache)
        row['arm'] = index
        if (index == 0 and cache is not None and not args.cache_keep
                and row.get('cache_hits')):
            # A purged directory cannot serve arm 0; if it did, something else
            # wrote into --cache while this arm ran (the keys are content-based,
            # so a concurrent run on the same weights lands on the same files).
            # The dequant column then measures a warm cache, so refuse the run
            # instead of publishing a number that is not dequant cost.
            raise SystemExit(
                f'arm 0 read {row["cache_hits"]} hit(s) from a purged cache directory '
                f'{cache.root}: another process is writing the same content-addressed '
                f'bases. Re-run with a private --cache directory (or wait for it to finish); '
                f'dequant_s on this run is not the cold dequant cost.')
        arm_rows.append(row)
        print(f'arm {index}: dequant {row["dequant_s"]}s apply {row["apply_s"]}s '
              f'requant {row["requant_s"]}s total {row["total_s"]}s '
              f'({row["changed_blocks_total"]} blocks, {row["dequantize_calls"]} dequant calls, '
              f'{row.get("cache_hits", 0)} hits/{row.get("cache_misses", 0)} misses) '
              f'reverted in {row["revert_s"]}s', flush=True)

    after = sha256_file(artifact)
    if after != before:
        raise SystemExit(f'the sweep did not end on the pristine artifact: {before} -> {after}')
    mean_total = sum(row['total_s'] for row in arm_rows) / len(arm_rows)
    requant_s = sum(row['requant_s'] for row in arm_rows)
    total_s = sum(row['total_s'] for row in arm_rows)
    report = {
        'code_root': code_root,
        'provenance': args.provenance,
        'artifact': str(artifact),
        'artifact_bytes': artifact.stat().st_size,
        'artifact_sha_before': before,
        'artifact_sha_after_revert': after,
        'axes': {'path': str(axes_path), 'shape': list(axes.shape),
                 'sha256': sha256_file(axes_path)},
        'records': str(records), 'records_sha256': records_sha,
        'only_layers': args.only_layers or f'0-{int(axes.shape[0]) - 1}',
        'layer_ids': layer_ids,
        'axis_space': args.axis_space,
        'tensors': len(names),
        'tensor_names': names,
        'tensors_changed': int(arm_rows[-1]['tensors_changed']),
        'changed_blocks_total': int(arm_rows[-1]['changed_blocks_total']),
        'arms': arm_rows,
        'requant_share': round(requant_s / total_s, 4) if total_s else None,
        'projected_4_arm_s': round(mean_total * 4, 1) if arm_rows else None,
        'total_wall_s': round(sum(row['total_s'] + row['revert_s'] for row in arm_rows), 1),
        'generated_tokens': 0,
        'note': 'apply_s is total_s minus dequant_s and requant_s: projection arithmetic, '
                'block-change detection, file I/O and the writer closing sha256. No model is '
                'loaded and no server is started on this path.',
        'next': next_command(str(artifact), str(axes_path), args.only_layers or '10-27',
                             cache is not None, len(names)),
    }
    if cache is not None:
        # Distinct keys: each of the 36 tensor bases is a miss once and a hit on
        # every later arm, so cache_hits == cache_misses == tensors. The *_accesses
        # counters are the per-arm sums (arm 0: 36 misses, arms 1..: 36 hits each).
        report.update({
            'cache_dir': str(cache.root),
            'cache_hits': len(cache.hit_keys),
            'cache_misses': len(cache.miss_keys),
            'cache_hit_accesses': cache.hits,
            'cache_miss_accesses': cache.misses,
            'cache_keys': len(cache.hit_keys | cache.miss_keys),
            'cache_dequant_s': round(cache.dequant_seconds, 3),
            'cache_note': 'cache_hits/cache_misses count distinct tensor keys: one cold miss '
                          'per tensor on arm 0, then a hit on every later arm. '
                          'cache_hit_accesses/cache_miss_accesses are the summed per-arm '
                          'lookups, and each arm row carries its own counters.',
        })
    text = json.dumps(report, indent=1) + '\n'
    if args.json_out:
        atomic_json(args.json_out, report)
    return report, text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='merge_bench', description=__doc__.split('\n')[0])
    ap.add_argument('--artifact', required=True,
                    help='byte copy of the artifact to patch and revert in place')
    ap.add_argument('--axes', required=True, help="torch.save({'axes': [L,W]}) from spanfit axes")
    ap.add_argument('--axes-key', default='axes', help='key inside the axes file')
    ap.add_argument('--only-layers', help="restrict the footprint: '10-27' or '10-20,30'")
    ap.add_argument('--scale', type=float, default=1.0, help='projection multiplier')
    ap.add_argument('--axis-space', choices=('input', 'output'), default='output')
    ap.add_argument('--arms', type=int, default=1,
                    help='repeat the merge+revert N times on the same bytes for a stable clock')
    ap.add_argument('--records', help='capture the axes were fitted from (sha256 goes in the '
                                     'JSON); defaults to records.pt beside the axes file')
    ap.add_argument('--expected-sha', default=ART_SHA256,
                    help='refuse an artifact whose sha256 differs (default: the pinned spark quant)')
    ap.add_argument('--provenance', metavar='TEXT',
                    help='one line explaining where this artifact came from, carried into the '
                         'JSON (e.g. "byte copy of the pinned bench.gguf: the shared path was being '
                         'patched by another card in the same window")')
    ap.add_argument('--cache', '--dequant-cache', dest='cache', metavar='DIR',
                    help='content-addressed dequantized-base cache: arm 0 dequantizes and '
                         'stores, later arms map the saved base and make zero dequantize calls '
                         '(purged at the start unless --cache-keep)')
    ap.add_argument('--cache-keep', action='store_true',
                    help='reuse the cache directory instead of purging it (a warm first arm '
                         'then reports 0 misses, so this is not the default)')
    ap.add_argument('--json-out', help='write the report here (atomic; one JSON)')
    args = ap.parse_args(argv)
    if args.arms < 1:
        raise SystemExit('--arms must be at least 1')
    report, text = bench(args)
    print(text)
    print(f'next: {report["next"]}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
