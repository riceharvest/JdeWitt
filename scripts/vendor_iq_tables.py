# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vendor the IQ grid tables from ggml-common.h into a pinned Python module.

The IQ quantizers index static grids defined upstream. Copying them by hand
would let the codec drift silently, so this script parses the pinned source,
refuses if a table's declared size does not match what it parsed, and writes a
module that records the source digest. Re-run only against the documented
revision (see scripts/ggml_ref/README.md).

Usage:
    python scripts/vendor_iq_tables.py PATH/TO/ggml-common.h > jdewitt/iq_tables.py
"""
from __future__ import annotations

import hashlib
import re
import sys

WANTED = {
    'kmask_iq2xs': 8,
    'ksigns_iq2xs': 128,
    'iq2xxs_grid': 256,
    'iq2xs_grid': 512,
    'iq2s_grid': 1024,
    'iq3xxs_grid': 256,
    'iq3s_grid': 512,
    'iq1s_grid': 2048,
}
BEGIN = re.compile(r'GGML_TABLE_BEGIN\(\s*(uint\d+_t)\s*,\s*(\w+)\s*,\s*([^)]+)\)')


def main():
    path = sys.argv[1]
    text = open(path, 'r', encoding='utf-8').read()
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    tables = {}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = BEGIN.search(line)
        if not match or match.group(2) not in WANTED:
            continue
        ctype, name, size_text = match.groups()
        expected = WANTED[name]
        declared = int(size_text) if size_text.isdigit() else expected
        values = []
        for body in lines[index + 1:]:
            if 'GGML_TABLE_END' in body:
                break
            body = body.split('//')[0]
            for token in re.findall(r'0x[0-9a-fA-F]+|\b\d+\b', body):
                values.append(int(token, 16) if token.lower().startswith('0x') else int(token, 10))
        if len(values) != expected or declared != expected:
            raise SystemExit(f'{name}: parsed {len(values)} values, header says '
                             f'{declared}, expected {expected}')
        tables[name] = (ctype, values)
    missing = set(WANTED) - set(tables)
    if missing:
        raise SystemExit(f'missing tables: {sorted(missing)}')
    out = ['# SPDX-License-Identifier: AGPL-3.0-or-later',
           '"""IQ grid tables vendored from ggml-common.h (generated, do not edit).',
           '',
           'Source: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-common.h',
           f'sha256 {digest}',
           '',
           'Regenerate with scripts/vendor_iq_tables.py against the pinned source; the',
           'script refuses if any table size disagrees with its declaration.',
           '"""',
           'import numpy as np',
           '',
           f'IQ_COMMON_SHA256 = {digest!r}',
           'IQ1S_DELTA = 0.125',
           '']
    for name in WANTED:
        ctype, values = tables[name]
        dtype = 'np.uint64' if ctype == 'uint64_t' else ('np.uint32' if ctype == 'uint32_t' else 'np.uint8')
        out.append(f'{name.upper()} = np.array([')
        for start in range(0, len(values), 6):
            row = ', '.join(str(v) for v in values[start:start + 6])
            out.append(f'    {row},')
        out.append(f'], dtype={dtype})')
        out.append('')
    sys.stdout.write('\n'.join(out))


if __name__ == '__main__':
    main()
