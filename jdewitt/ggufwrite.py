# SPDX-License-Identifier: AGPL-3.0-or-later
"""Minimal GGUF writer: rewrite a file changing only what the edit touches.

gguf-py's writer is row-oriented — it derives a tensor's shape from its packed bytes
and refuses tensors whose row length is not a whole number of blocks (a real case:
``blk.0.attn_gate.weight`` is [2048, 8] Q4_K, quantized as one flat run of blocks).
Editing a user's artifact must not depend on that opinion, so this module writes the
container itself:

  * the metadata (KV) section is copied VERBATIM, byte for byte, so every key the
    user's quantizer wrote — imatrix calibration included — survives unchanged,
  * tensor infos are re-emitted with the same names, dims and order, with a new type
    only where the edit changed the tensor,
  * untouched tensors are streamed from the source without decoding.

Offsets are relative to the data base and every tensor starts at the file's alignment,
which is what llama.cpp reads; the whole file is verified by parsing it back.
"""
from __future__ import annotations

import struct
from pathlib import Path


GGUF_MAGIC = b'GGUF'
SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
TYPE_STRING, TYPE_ARRAY = 8, 9


def _read_value_size(handle, value_type):
    """Bytes consumed by one value of `value_type` at the current position."""
    if value_type in SCALAR_SIZES:
        return SCALAR_SIZES[value_type]
    if value_type == TYPE_STRING:
        length = struct.unpack('<Q', handle.read(8))[0]
        return 8 + length
    if value_type == TYPE_ARRAY:
        element_type = struct.unpack('<I', handle.read(4))[0]
        count = struct.unpack('<Q', handle.read(8))[0]
        if element_type == TYPE_STRING:
            total = 12
            for _ in range(count):
                length = struct.unpack('<Q', handle.read(8))[0]
                handle.seek(length, 1)
                total += 8 + length
            return total
        if element_type in SCALAR_SIZES:
            return 12 + count * SCALAR_SIZES[element_type]
        raise ValueError(f'array of unsupported element type {element_type}')
    raise ValueError(f'unsupported GGUF value type {value_type}')


def parse_container(path):
    """Header counts, the raw KV bytes, tensor infos and the alignment."""
    path = Path(path)
    with open(path, 'rb') as handle:
        if handle.read(4) != GGUF_MAGIC:
            raise ValueError(f'{path} is not a GGUF file')
        version = struct.unpack('<I', handle.read(4))[0]
        if version != 3:
            raise ValueError(f'{path} is GGUF version {version}; this writer handles 3')
        tensor_count = struct.unpack('<Q', handle.read(8))[0]
        kv_count = struct.unpack('<Q', handle.read(8))[0]
        kv_start = handle.tell()
        alignment = 32
        for _ in range(kv_count):
            name_length = struct.unpack('<Q', handle.read(8))[0]
            name = handle.read(name_length).decode('utf-8', errors='replace')
            value_type = struct.unpack('<I', handle.read(4))[0]
            value_start = handle.tell()
            size = _read_value_size(handle, value_type)
            handle.seek(value_start)
            if name == 'general.alignment' and value_type == 4:
                alignment = struct.unpack('<I', handle.read(4))[0]
                handle.seek(value_start + size)
            else:
                handle.seek(value_start + size)
        kv_end = handle.tell()
        handle.seek(kv_start)
        kv_bytes = handle.read(kv_end - kv_start)
        tensors = []
        for _ in range(tensor_count):
            name_length = struct.unpack('<Q', handle.read(8))[0]
            name = handle.read(name_length).decode('utf-8', errors='replace')
            n_dims = struct.unpack('<I', handle.read(4))[0]
            dims = list(struct.unpack('<' + 'Q' * n_dims, handle.read(8 * n_dims)))
            qtype = struct.unpack('<I', handle.read(4))[0]
            offset = struct.unpack('<Q', handle.read(8))[0]
            tensors.append({'name': name, 'dims': dims, 'qtype': qtype, 'offset': offset})
        data_base = handle.tell()
        data_base += (-data_base) % alignment
    return {'version': version, 'alignment': alignment, 'kv_bytes': kv_bytes,
            'tensors': tensors, 'data_base': data_base,
            'kv_count': kv_count, 'tensor_count': tensor_count}


def type_size(qtype):
    from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

    if int(qtype) == 42:  # Q2_0: newer than published gguf, see quants.EXTRA_TYPE_IDS
        return 64, 18
    block_els, block_bytes = GGML_QUANT_SIZES[GGMLQuantizationType(int(qtype))]
    return int(block_els), int(block_bytes)


def tensor_nbytes(dims, qtype):
    block_els, block_bytes = type_size(qtype)
    count = 1
    for dim in dims:
        count *= int(dim)
    if count % block_els:
        raise ValueError(f'{count} values is not a whole number of {block_els}-value blocks')
    return count // block_els * block_bytes


def write_container(path, header, payloads, qtypes=None, extra_kv=None):
    """Write `path` from a parsed container.

    ``payloads`` maps a tensor name to the bytes to store (or None to copy the source
    slice verbatim) and ``qtypes`` overrides the stored type for edited tensors.

    ``extra_kv`` appends string metadata WITHOUT reparsing the section: GGUF key/value
    pairs are a flat sequence, so one new pair at the end plus an incremented count is
    the whole change — and it is what lets a file carry its own edit fingerprint, which
    is the only way `verify` can check a release's claim instead of guessing it.
    """
    path = Path(path)
    alignment = header['alignment']
    source = Path(header['source'])
    entries = []
    offset = 0
    with open(source, 'rb') as reader:
        for tensor in header['tensors']:
            qtype = int((qtypes or {}).get(tensor['name'], tensor['qtype']))
            block_els, block_bytes = type_size(qtype)
            if qtype == tensor['qtype']:
                nbytes = tensor_nbytes(tensor['dims'], qtype)
            else:
                nbytes = tensor_nbytes(tensor['dims'], qtype)
            payload = payloads.get(tensor['name'])
            if payload is None:
                reader.seek(header['data_base'] + tensor['offset'])
                payload = reader.read(tensor_nbytes(tensor['dims'], tensor['qtype']))
            if len(payload) != nbytes:
                raise ValueError(f"{tensor['name']}: payload is {len(payload)} bytes, "
                                 f"{qtype} needs {nbytes}")
            entries.append({'name': tensor['name'], 'dims': tensor['dims'], 'qtype': qtype,
                            'offset': offset, 'payload': payload})
            offset += len(payload)
            offset += (-offset) % alignment
    with open(path, 'wb') as out:
        out.write(GGUF_MAGIC)
        out.write(struct.pack('<I', header['version']))
        out.write(struct.pack('<Q', len(entries)))
        kv_bytes = header['kv_bytes']
        kv_count = int(header['kv_count'])
        for key, value in (extra_kv or []):
            key_bytes = str(key).encode('utf-8')
            value_bytes = str(value).encode('utf-8')
            kv_bytes += struct.pack('<Q', len(key_bytes)) + key_bytes
            kv_bytes += struct.pack('<I', 8)  # GGUFValueType.STRING
            kv_bytes += struct.pack('<Q', len(value_bytes)) + value_bytes
            kv_count += 1
        out.write(struct.pack('<Q', kv_count))
        out.write(kv_bytes)
        for entry in entries:
            name = entry['name'].encode('utf-8')
            out.write(struct.pack('<Q', len(name)))
            out.write(name)
            out.write(struct.pack('<I', len(entry['dims'])))
            out.write(struct.pack('<' + 'Q' * len(entry['dims']), *entry['dims']))
            out.write(struct.pack('<I', entry['qtype']))
            out.write(struct.pack('<Q', entry['offset']))
        position = out.tell()
        out.write(b'\x00' * ((-position) % alignment))
        base = out.tell()
        for entry in entries:
            out.seek(base + entry['offset'])
            out.write(entry['payload'])
    return {'entries': [{'name': e['name'], 'qtype': e['qtype'], 'offset': e['offset'],
                         'bytes': len(e['payload'])} for e in entries],
            'data_base': base, 'alignment': alignment}
