# SPDX-License-Identifier: AGPL-3.0-or-later
"""What a stranger can check in SOMEONE ELSE'S release directory, from bytes alone.

``verify-artifact FILE`` validates artifacts this tool publishes: it reads
files.json/evidence.json/signature.json/adapter.json and says whether they
conform. Pointed at a directory from another tool (Heretic, OBLITERATUS, or a
hand-rolled release), it reports "no known artifact" — which is true and
useless. This module is the other half: it inspects a foreign directory and
separates two things a reader actually needs:

* ``verifiable``     — facts recomputed from the bytes present here: file
                       digests, GGUF architecture, tensor count, the declared
                       quant type of every tensor family, tokenizer metadata,
                       and whether a plan/adapter card is shipped.
* ``not_verifiable`` — claims with no operand in this directory, each naming
                       the file that would have to be shipped to check it.

Nothing is decoded and no weights are loaded: the GGUF reader walks the header
and tensor table and stops at the data base. File contents are hashed, never
printed. ``certified_claims`` is 0 for a foreign directory by definition: no
chain of claims signed by a key we can check exists there.

The command is read-only and refuses anything that would make "the bytes in
this directory" ambiguous: a symlinked directory, a symlink at any depth, a
socket/fifo/device, a name that tries to traverse, or a directory that already
carries files.json (that one is ours to verify, use the native path).
"""
from __future__ import annotations

import os
import re
import struct
from pathlib import Path

from .errors import ForeignArtifactRefused
from .ggufio import _SCALARS, _TYPE_STRING, parse_gguf_header
from .ggufwrite import tensor_nbytes
from .quants import type_name_for_id
from .release import sha256_file

_GGUF_MAGIC = b'GGUF'
_WANTED = ('general.architecture', 'general.name', 'general.file_type',
           'general.quantization_version')
# Card files this tool publishes; presence is reported, contents are not read.
ADAPTER_CARDS = ('adapter.json', 'adapter-card.json')
PLAN_CARDS = ('plan-card.json', 'plan.json')


def _kv_value(tag, raw):
    """Decode one metadata value; arrays stay undecoded (only scalars/strings needed)."""
    if tag in _SCALARS:
        return struct.unpack(_SCALARS[tag], raw)[0]
    if tag == _TYPE_STRING:
        length = struct.unpack('<Q', raw[:8])[0]
        return raw[8:8 + length].decode('utf-8', 'replace')
    return None


def _check_name(name, path, directory):
    """Refuse a filename that is not a plain entry of this directory."""
    if not name or name in {'.', '..'} or '/' in name or '\\' in name:
        raise ForeignArtifactRefused(
            'PATH_TRAVERSAL_REFUSED',
            f'{path}: entry name {name!r} is not a plain file name inside {directory}')
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ForeignArtifactRefused(
            'PATH_TRAVERSAL_REFUSED',
            f'{path}: resolves outside {directory}')


def _regular_files(directory):
    """Every regular file below DIR, no link following; refuse anything else.

    The walk refuses rather than skips: a symlink or a socket in a release
    directory makes "the bytes in this directory" ambiguous, and a reader who
    hashes what they were given would otherwise certify a file they never saw.
    """
    files = []
    for root, dirnames, filenames in os.walk(directory, followlinks=False):
        root_path = Path(root)
        for name in dirnames:
            path = root_path / name
            if path.is_symlink():
                raise ForeignArtifactRefused(
                    'SYMLINK_REFUSED',
                    f'{path}: symlinked directory; refusing to follow it '
                    '(hash what is here, not what a link points at)')
            _check_name(name, path, directory)
        for name in filenames:
            path = root_path / name
            if path.is_symlink():
                raise ForeignArtifactRefused(
                    'SYMLINK_REFUSED',
                    f'{path}: symlink; refusing to hash a link instead of its target')
            _check_name(name, path, directory)
            if not path.is_file():
                raise ForeignArtifactRefused(
                    'NONREGULAR_REFUSED',
                    f'{path}: not a regular file (socket/fifo/device); only regular bytes can be hashed')
            files.append(path)
    files.sort(key=lambda path: str(path.relative_to(directory)))
    return files


def _gguf_inventory(path, directory, size, sha256):
    """Header/tensor table of one GGUF: no weights, no decode, no contents."""
    try:
        parsed = parse_gguf_header(path)
    except (ValueError, OSError) as error:
        raise ForeignArtifactRefused('HEADER_INVALID', f'{path}: {error}') from error
    metadata, tokenizer_keys = {}, []
    for key, tag, raw in parsed['entries']:
        name = key.decode('utf-8', 'replace')
        if name.startswith('tokenizer.'):
            tokenizer_keys.append(name)
        if name in _WANTED:
            metadata[name] = _kv_value(tag, raw)
    dtypes, families = {}, {}
    past_end = False
    for info in parsed['infos']:
        name = info['name'].decode('utf-8', 'replace')
        dims = [int(dim) for dim in struct.unpack('<' + 'Q' * info['n_dims'], info['dims'])]
        try:
            dtype = type_name_for_id(info['dtype'])
        except ValueError:
            dtype = f'id-{info["dtype"]}'
        try:
            nbytes = tensor_nbytes(dims, info['dtype'])
        except ValueError:
            nbytes = None
        row = dtypes.setdefault(dtype, {'tensors': 0, 'declared_bytes': 0})
        row['tensors'] += 1
        if nbytes is None:
            row['bytes_unknown'] = True
        else:
            row['declared_bytes'] += nbytes
            if parsed['data_base'] + info['offset'] + nbytes > size:
                past_end = True
        # blk.3.attn_q.weight and blk.7.attn_q.weight are one family with one
        # declared type; a mixed-type family is a fact a reader wants to see.
        family = re.sub(r'\d+', '*', name)
        family_row = families.setdefault(family, {})
        family_row[dtype] = family_row.get(dtype, 0) + 1
    if past_end:
        raise ForeignArtifactRefused(
            'HEADER_INVALID',
            f'{path}: tensor table declares data past the {size}-byte end of the file '
            '(truncated download or corrupt header)')
    return {
        'path': str(path.relative_to(directory)),
        'bytes': size,
        'sha256': sha256,
        'version': parsed['version'],
        'architecture': str(metadata.get('general.architecture') or 'unknown'),
        'name': metadata.get('general.name'),
        'file_type_declared': metadata.get('general.file_type'),
        'quantization_version': metadata.get('general.quantization_version'),
        'metadata_entries': len(parsed['entries']),
        'tensor_count': parsed['n_tensors'],
        'quant_types': dict(sorted(dtypes.items())),
        'quant_by_tensor_family': dict(sorted(families.items())),
        'tokenizer': {'present': bool(tokenizer_keys),
                      'keys': sorted(tokenizer_keys),
                      'chat_template': 'tokenizer.chat_template' in tokenizer_keys},
        'tensor_data_within_file': True,
    }


def _verifiable(files, directory):
    rows, ggufs, other = [], [], []
    verified_bytes = 0
    for path in files:
        size = path.stat().st_size
        sha256 = sha256_file(path)
        name = str(path.relative_to(directory))
        verified_bytes += size
        rows.append({'path': name, 'bytes': size, 'sha256': sha256})
        with path.open('rb') as handle:
            magic = handle.read(4)
        if magic == _GGUF_MAGIC:
            ggufs.append(_gguf_inventory(path, directory, size, sha256))
        elif path.suffix.lower() == '.gguf':
            raise ForeignArtifactRefused(
                'HEADER_INVALID',
                f'{path}: GGUF name but magic {magic!r} is not b\'GGUF\'; '
                'this file is not the artifact its name claims')
        else:
            other.append(name)
    names = {row['path'] for row in rows}
    adapter_found = sorted(name for name in names if name in ADAPTER_CARDS)
    plan_found = sorted(name for name in names if name in PLAN_CARDS)
    return {
        'files': rows,
        'file_count': len(rows),
        'verified_bytes': verified_bytes,
        'gguf': ggufs,
        'tokenizer_present': any(row['tokenizer']['present'] for row in ggufs),
        'adapter_card_present': bool(adapter_found),
        'adapter_card_files': adapter_found,
        'plan_card_present': bool(plan_found),
        'plan_card_files': plan_found,
        'other_files': other,
    }


def _not_verifiable():
    """Claims with no operand here; each names the file that would check it."""
    return [
        {'claim': 'the file set is complete: no artifact is omitted, substituted or added',
         'needs_file': 'files.json'},
        {'claim': 'the evaluation numbers were recorded by the publisher and not edited afterwards',
         'needs_file': 'evidence.json'},
        {'claim': 'the bytes and the claims come from the publisher key and were not altered in transit',
         'needs_file': 'signature.json (public_key_pem + signature_base64)'},
        {'claim': 'the declared quant type reconstructs the weights it replaced '
                  '(realized fidelity: keep-cos / perplexity / bits-per-weight)',
         'needs_file': 'a codec-realized record: export-smoke output or plan card naming the measurement codec'},
        {'claim': 'the numbers were measured with the server argv a reader would run',
         'needs_file': 'the serve argv recorded by export-smoke, or a plan card'},
        {'claim': 'any claim is certified against published gates',
         'needs_file': 'a claim chain (evidence.json) bound to files.json release_identity'},
    ]


def _checks_our_claims_against():
    """Rows a reader can put beside ours, and the source on each side."""
    return [
        {'row': 'artifact digest and size',
         'ours': 'files.json entry for the same file name',
         'theirs': 'verifiable.files[].sha256 / bytes (recomputed here)'},
        {'row': 'gguf architecture, tensor count, quant type per tensor family',
         'ours': 'plan card / export-smoke header record',
         'theirs': 'verifiable.gguf[].architecture, tensor_count, quant_by_tensor_family'},
        {'row': 'tokenizer identity',
         'ours': 'tokenizer files listed in files.json',
         'theirs': 'verifiable.gguf[].tokenizer keys (values never read out)'},
        {'row': 'measured behaviour numbers',
         'ours': 'evaluation.json plus the evidence.json claims that cite it',
         'theirs': 'nothing checkable: their evaluation JSON is unconstrained prose'},
        {'row': 'certified claims',
         'ours': 'evidence.json ledger entries',
         'theirs': 'none; certified_claims is 0 for a foreign directory'},
    ]


def verify_foreign_artifact(directory):
    """Inspect a directory that is not ours: what can be checked, and what cannot."""
    directory = Path(directory)
    if directory.is_symlink():
        raise ForeignArtifactRefused(
            'SYMLINK_REFUSED',
            f'{directory}: symlinked directory; point --foreign at the real directory')
    if not directory.is_dir():
        raise ForeignArtifactRefused(
            'NOT_A_DIRECTORY', f'{directory}: --foreign needs a directory, not a file')
    record = directory / 'files.json'
    if record.is_symlink() or record.is_file():
        raise ForeignArtifactRefused(
            'FOREIGN_AMBIGUOUS',
            f'{directory}: carries a files.json release index, so it is not a stranger\'s '
            'directory; use the native path (verify-artifact DIR --all) instead of --foreign')
    files = _regular_files(directory)
    verifiable = _verifiable(files, directory)
    return {
        'directory': str(directory.resolve()),
        'mode': 'foreign',
        # Deliberately no "valid": a foreign directory is not certified by being
        # readable, and certified_claims is 0 precisely because it cannot be.
        'verifiable': verifiable,
        'not_verifiable': _not_verifiable(),
        'certified_claims': 0,
        'checks_our_claims_against': _checks_our_claims_against(),
        'next': ('uv run jdewitt release-diff <our-release-dir> ' + str(directory.resolve())
                 + '  # once the publisher ships files.json; until then the rows in '
                   'checks_our_claims_against are the comparison'),
        'note': ('Header inventory only: weights were never loaded or decoded and no file '
                 'contents are printed. certified_claims is 0 because a foreign directory '
                 'carries no claim chain signed by a key this tool can check.'),
    }
