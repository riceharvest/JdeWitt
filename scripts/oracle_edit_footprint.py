# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which decoder tensors did a published edited build actually change?

Usage: python scripts/oracle_edit_footprint.py <base.gguf> <edited.gguf>

Byte-hashes every decoder tensor on both sides and reports which ones moved. On
Qwen/Qwen3-4B-Instruct-2507 @ cdbee75f17c0 versus p-e-w/Qwen3-4B-Instruct-2507-heretic
@ bc843daf6f71 (both q8_0 from the same converter): 54 of 396 tensors changed,
attn_output and ffn_down for blk.9 through blk.35, layers 0-8 byte-identical.
A diff-and-fit method that assumes the whole stack is edited is fitting against
zeros in the untouched layers.
"""
import hashlib
import json
import sys
from gguf import GGUFReader

if len(sys.argv) < 3:
    raise SystemExit('usage: oracle_edit_footprint.py <base.gguf> <edited.gguf>')
A, B = sys.argv[1], sys.argv[2]

# Which decoder tensors did the published oracle actually change? The axis
# diagnostic compares fitted directions to the oracle's per-tensor weight delta;
# if the delta is identically zero for a tensor, "my axis is not the oracle's
# direction" is meaningless there. Hash the raw quantized bytes per tensor and
# report which ones moved. gguf-py exposes tensor.data as a view over the
# tensor's own byte span, which already accounts for quantization blocks.
ra, rb = GGUFReader(A), GGUFReader(B)
ta = {t.name: t for t in ra.tensors}
tb = {t.name: t for t in rb.tensors}
names = sorted({n for n in ta if n.startswith('blk.')},
               key=lambda n: (int(n.split('.')[1]), n))
changed, identical, missing = [], [], []
for name in names:
    if name not in ta or name not in tb:
        missing.append(name)
        continue
    ha = hashlib.sha256(ta[name].data.tobytes()).hexdigest()[:16]
    hb = hashlib.sha256(tb[name].data.tobytes()).hexdigest()[:16]
    (changed if ha != hb else identical).append(name)

groups = {}
for n in changed:
    suffix = '.'.join(n.split('.')[2:])
    groups.setdefault(suffix, []).append(int(n.split('.')[1]))
layers = sorted({int(n.split('.')[1]) for n in changed})
print(json.dumps({
    'a': A, 'b': B,
    'compared': len(changed) + len(identical),
    'identical': len(identical), 'changed': len(changed), 'only_one_side': missing,
    'changed_suffix_counts': {k: len(v) for k, v in groups.items()},
    'changed_layers': layers,
    'first_changed': changed[:12],
}, indent=1), flush=True)
