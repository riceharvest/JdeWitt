#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Write the 5.5 GB-class transfer probe GGUF for the OAuth upload measurement.

    python verification/space/oauth-upload/make_probe.py --out FILE [--bytes 5500000000]

Why a purpose-built file instead of a real 7B artifact: the bank carries one entry,
fitted on Qwen3-4B-Instruct-2507, so a 7B model is `BANK_MISS` — the Space cannot
produce a 7B q8_0 output today. What the request window question needs is the byte
class and a file whose bytes can be checked after the round trip, so the probe is a
valid GGUF (a real header, a real tensor table, real tensor bytes) filled with
zeros. It is NOT a model and the card says so. Its sha256, tensor table and byte
count are what the fetch-back proves; no claim is made about it as weights.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

ARCH = 'jdewittprobe'
TENSORS = 11
ELEMENTS = 250_000_000  # f16 -> 500,000,000 B per tensor


def write_probe(out, tensors=TENSORS, elements=ELEMENTS):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(out), ARCH)
    writer.add_name('jdewitt-transfer-probe-5.5gb')
    writer.add_description('5.5 GB-class transfer probe: valid GGUF, zero-valued F16 '
                           'tensors; not a model')
    for index in range(tensors):
        writer.add_tensor(f'probe.{index:02d}', np.zeros((elements,), dtype=np.float16))
    writer.open_output_file()
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tensors', type=int, default=TENSORS)
    parser.add_argument('--elements', type=int, default=ELEMENTS)
    args = parser.parse_args(argv)
    path = write_probe(args.out, args.tensors, args.elements)
    print(f'{path} {path.stat().st_size} bytes')
    return 0


if __name__ == '__main__':
    sys.exit(main())
