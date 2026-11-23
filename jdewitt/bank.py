# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Space direction bank: one JSON unit, resolved against the artifact itself.

A bank entry names a fitted axis by the geometry it was fitted on — architecture,
hidden width, a layer footprint and the two residual-write projection kinds — never
by a per-model tensor table. `resolve` reads only the GGUF header and tensor-info
array (`GGUFReader`; no tensor data) and answers the questions the Space needs
before it downloads or edits anything:

  * does this artifact have an entry at all (BANK_MISS);
  * do its geometry and footprint agree, layer by layer (GEOMETRY_MISMATCH);
  * can this build decode the source tensors' quant type (UNSUPPORTED_QUANT);
  * is the entry's bundle the bytes the entry pinned (BUNDLE_BAD_DIGEST).

The resolved report carries the concrete tensor names, the layer list, the per-kind
strength and the edit types, so `bank-apply` never has to guess a name.

This module is torch-free on purpose: the coverage check runs on a header range
request inside a 16 GB Space, before the multi-GB download, and importing
`gguf_adapters` (which needs torch) would defeat that.
"""
import hashlib
import json
from pathlib import Path

from gguf import GGMLQuantizationType, GGUFReader

from .errors import BankRefused
from .ggufgeom import (WRITEBACK_TENSOR, discover_writeback_targets,
                       gguf_weight_shapes, inferred_hidden_width)

SCHEMA = 'jdewitt-space-bank-v1'
ENTRY_KEYS = ('key', 'base', 'architecture', 'geometry', 'targets', 'footprint',
              'strength', 'edit', 'sources', 'measurement', 'bundle')
TARGET_KINDS = ('attn', 'mlp')


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _validated_entry(entry):
    """One entry, checked against §5.1's shape without trusting its numbers."""
    if not isinstance(entry, dict):
        raise ValueError('every bank entry must be a JSON object')
    missing = [key for key in ENTRY_KEYS if key not in entry]
    if missing:
        raise ValueError(f'bank entry {entry.get("key")!r} has no {", ".join(missing)}')
    key = str(entry['key'])
    geometry, footprint = entry['geometry'], entry['footprint']
    for field, block in (('layers', geometry), ('hidden', geometry),
                         ('first', footprint), ('last', footprint)):
        if not isinstance(block.get(field), int):
            raise ValueError(f'bank entry {key!r}: {field} must be an integer')
    targets = entry['targets']
    if not isinstance(targets, dict) or set(targets) != set(TARGET_KINDS):
        raise ValueError(f'bank entry {key!r}: targets must name exactly {TARGET_KINDS}')
    for kind in TARGET_KINDS:
        if not WRITEBACK_TENSOR.match(str(targets[kind])):
            raise ValueError(f'bank entry {key!r}: target {kind}={targets[kind]!r} is not '
                             f'a residual-write projection ({WRITEBACK_TENSOR.pattern})')
    if not isinstance(entry['strength'], dict) or not set(entry['strength']) >= set(TARGET_KINDS):
        raise ValueError(f'bank entry {key!r}: strength must carry {TARGET_KINDS}')
    types = entry['edit'].get('types') if isinstance(entry['edit'], dict) else None
    if not isinstance(types, list) or not types:
        raise ValueError(f'bank entry {key!r}: edit.types must be a non-empty list')
    quant_types = entry['sources'].get('quant_types') if isinstance(entry['sources'], dict) else None
    if not isinstance(quant_types, list):
        raise ValueError(f'bank entry {key!r}: sources.quant_types must be a list')
    bundle = entry['bundle']
    name = bundle.get('file') if isinstance(bundle, dict) else None
    if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError(f'bank entry {key!r}: bundle.file must be a plain file name '
                         f'beside bank.json, not a path')
    if not isinstance(bundle.get('sha256'), str) or not bundle['sha256']:
        raise ValueError(f'bank entry {key!r}: bundle.sha256 is required')
    return entry


def load_bank(bank_dir):
    """Read and shape-check `bank.json`; an empty `entries` list is valid."""
    path = Path(bank_dir) / 'bank.json'
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or document.get('schema') != SCHEMA:
        raise ValueError(f'{path} schema is {document.get("schema")!r}, not {SCHEMA!r}')
    entries = document.get('entries')
    if not isinstance(entries, list):
        raise ValueError(f'{path}: entries must be a list')
    return {'schema': SCHEMA, 'path': str(path),
            'entries': [_validated_entry(entry) for entry in entries]}


def _declared(reader, suffix):
    """A `general.*` value from the header, or None when the artifact omits it."""
    field = reader.fields.get(f'general.{suffix}')
    return None if field is None else field.contents()


def _target_suffix(name):
    """`blk.7.mlp.down_proj.weight` -> `down_proj`."""
    parts = name.rsplit('.', 2)
    return parts[1] if len(parts) == 3 else ''


def _artifact_layers(discovered):
    """Layer count the tensor table proves: the writeback layers it carries."""
    return max(discovered['targets']) + 1 if discovered['targets'] else 0


def _projected_layers(entry, artifact_layers):
    """The entry's footprint on this artifact, by fraction when layer counts differ.

    §5.1: the fraction is what survives a different layer count; it is rounded and
    clamped to whole layers, and strength is never scaled by it.
    """
    footprint = entry['footprint']
    first, last = int(footprint['first']), int(footprint['last'])
    if artifact_layers == int(entry['geometry']['layers']):
        return list(range(first, last + 1)), False
    fraction = footprint.get('fraction')
    if not isinstance(fraction, list) or len(fraction) != 2:
        raise BankRefused(
            'GEOMETRY_MISMATCH',
            f'entry {entry["key"]!r} was fitted on {entry["geometry"]["layers"]} layers but '
            f'this artifact carries {artifact_layers}, and the entry states no footprint '
            f'fraction to project onto it')
    if artifact_layers == 0:
        raise BankRefused(
            'GEOMETRY_MISMATCH',
            f'entry {entry["key"]!r} states a footprint but this artifact carries no '
            f'residual-write projection to place it on')
    start = min(max(round(float(fraction[0]) * artifact_layers), 0), artifact_layers - 1)
    end = min(max(round(float(fraction[1]) * artifact_layers), start), artifact_layers - 1)
    return list(range(start, end + 1)), True


def _missing(layers, discovered, targets):
    """Footprint layers that are absent or lack one of the two target kinds."""
    missing = []
    for layer in layers:
        carried = {_target_suffix(name) for name in discovered['targets'].get(layer, [])}
        lacking = [kind for kind in TARGET_KINDS if targets[kind] not in carried]
        if lacking:
            missing.append({'layer': layer, 'missing': lacking})
    return missing


def _decodable(type_name):
    """Whether `decode_values` can decode this tensor type without tensor data."""
    from .quants import CODECS
    from .raw_dtypes import RAW_CODECS

    if str(type_name) in CODECS or str(type_name) in RAW_CODECS:
        return True
    try:
        GGMLQuantizationType[str(type_name)]
    except KeyError:
        return False
    return True


def _missing_text(missing):
    parts = []
    for item in missing[:12]:
        parts.append(f'layer {item["layer"]} ({"+".join(item["missing"])})')
    more = f', and {len(missing) - 12} more' if len(missing) > 12 else ''
    return ', '.join(parts) + more


def _verify_bundle(bank_dir, entry):
    """The entry's bundle must hash to the digest it pins, or nothing is applied."""
    bundle = entry['bundle']
    path = Path(bank_dir) / bundle['file']
    if not path.is_file():
        raise BankRefused(
            'BUNDLE_BAD_DIGEST',
            f'entry {entry["key"]!r} pins bundle {bundle["file"]!r} at sha256 '
            f'{bundle["sha256"]}, but that file is missing from {Path(bank_dir)}')
    actual = _file_digest(path)
    if actual.lower() != str(bundle['sha256']).lower():
        raise BankRefused(
            'BUNDLE_BAD_DIGEST',
            f'entry {entry["key"]!r} pins bundle {bundle["file"]!r} at sha256 '
            f'{bundle["sha256"]}, but the file hashes to {actual}')
    return {'file': bundle['file'], 'sha256': actual, 'verified': True}


def resolve(bank_dir, gguf, key=None):
    """Resolve one artifact against the bank, or raise the typed refusal.

    Checks run in the order a Space needs them: the architecture must be covered at
    all, then geometry/footprint, then source decodability, then the bundle digest
    (the axis is never read, so a bad bundle cannot have been applied).

    ``key`` restricts the candidate entries to one, so `bank-apply --entry` resolves
    the entry the user named instead of the first one that fits the geometry. An
    unknown key is a BANK_MISS like an uncovered architecture: the caller asked for
    something the bank does not hold.
    """
    bank_dir, gguf = Path(bank_dir), Path(gguf)
    document = load_bank(bank_dir)
    entries = document['entries']
    if key is not None:
        entries = [entry for entry in entries if str(entry['key']) == str(key)]
        if not entries:
            carried = [str(entry['key']) for entry in document['entries']]
            raise BankRefused(
                'BANK_MISS',
                f'the bank at {bank_dir} holds no entry {key!r} '
                f'(carried: {", ".join(carried) if carried else "nothing"})')
    reader = GGUFReader(str(gguf))
    architecture = _declared(reader, 'architecture')
    architecture = None if architecture is None else str(architecture)

    by_arch = [entry for entry in entries
               if str(entry['architecture']) == architecture]
    if not by_arch:
        covered = sorted({str(entry['architecture']) for entry in entries})
        raise BankRefused(
            'BANK_MISS',
            f'the artifact declares architecture {architecture!r} and the bank has no entry '
            f'for it (covered: {covered or "nothing"})')

    shapes = gguf_weight_shapes(gguf, reader=reader)
    hidden = inferred_hidden_width(shapes)
    discovered = (discover_writeback_targets(shapes, hidden) if hidden is not None
                  else {'targets': {}, 'unmapped_writeback_width': []})
    tensor_layers = _artifact_layers(discovered)
    declared_layers = _declared(reader, 'block_count')
    # The header's block_count is the artifact's own statement of its depth and is
    # what a fraction must project onto; falling back to the deepest writeback layer
    # would silently shrink a footprint whose last block carries no target.
    artifact_layers = int(declared_layers) if declared_layers is not None else tensor_layers

    if hidden is None:
        raise BankRefused(
            'GEOMETRY_MISMATCH',
            f'architecture {architecture!r} has a bank entry but this artifact carries no '
            f'residual-write projection at all, so its hidden width cannot be inferred')

    by_hidden = [entry for entry in by_arch if int(entry['geometry']['hidden']) == hidden]
    if not by_hidden:
        expected = sorted({int(entry['geometry']['hidden']) for entry in by_arch})
        raise BankRefused(
            'GEOMETRY_MISMATCH',
            f'the artifact\'s writeback tensors are {hidden}-wide but architecture '
            f'{architecture!r} entries are fitted at hidden {expected}')

    failures = []
    chosen = None
    for entry in by_hidden:
        layers, projected = _projected_layers(entry, artifact_layers)
        missing = _missing(layers, discovered, entry['targets'])
        if not missing:
            chosen = (entry, layers, projected)
            break
        failures.append((entry, missing))
    if chosen is None:
        entry, missing = min(failures, key=lambda item: len(item[1]))
        where = ''
        if declared_layers is not None and tensor_layers != artifact_layers:
            where = (f' (the header declares {artifact_layers} blocks but the tensor table '
                     f'shows writeback layers 0-{tensor_layers - 1})')
        raise BankRefused(
            'GEOMETRY_MISMATCH',
            f'entry {entry["key"]!r} needs both target kinds on every footprint layer, but '
            f'this artifact is missing {_missing_text(missing)}{where}')

    entry, layers, projected = chosen
    targets = {}
    for kind in TARGET_KINDS:
        suffix = str(entry['targets'][kind])
        names = []
        for layer in layers:
            by_suffix = {_target_suffix(name): name
                         for name in discovered['targets'][layer]}
            names.append(by_suffix[suffix])
        targets[kind] = names
    target_names = sorted(name for names in targets.values() for name in names)

    source_by_name = {tensor.name: tensor.tensor_type.name for tensor in reader.tensors
                      if tensor.name in set(target_names)}
    source_types = sorted(set(source_by_name.values()))
    unsupported = [name for name in source_types if not _decodable(name)]
    if unsupported:
        example = next(name for name in target_names if source_by_name[name] == unsupported[0])
        raise BankRefused(
            'UNSUPPORTED_QUANT',
            f'target tensor {example} carries source type {unsupported[0]}, which this build '
            f'cannot decode; covered source types: '
            f'{", ".join(entry["sources"]["quant_types"])}')

    bundle = _verify_bundle(bank_dir, entry)
    covered_types = {str(name).casefold() for name in entry['sources']['quant_types']}
    return {
        'schema': SCHEMA,
        'bank': str(bank_dir),
        'artifact': {'path': str(gguf), 'architecture': architecture,
                     'layers': artifact_layers,
                     'tensor_layers': tensor_layers,
                     'declared_block_count': None if declared_layers is None else int(declared_layers),
                     'hidden': hidden, 'tensors': len(reader.tensors),
                     'source_types': source_types},
        'entry': {'key': entry['key'], 'base': entry['base'],
                  'architecture': entry['architecture']},
        'geometry': {'layers': int(entry['geometry']['layers']),
                     'hidden': int(entry['geometry']['hidden']),
                     'hidden_source': entry['geometry'].get('hidden_source', 'artifact')},
        'footprint': {'first': int(entry['footprint']['first']),
                      'last': int(entry['footprint']['last']),
                      'fraction': entry['footprint'].get('fraction'),
                      'projected': projected, 'layers': layers},
        'layers': layers,
        'targets': targets,
        'target_names': target_names,
        'strength': {kind: float(entry['strength'][kind]) for kind in TARGET_KINDS},
        'edit_types': list(entry['edit']['types']),
        'edit': entry['edit'],
        'sources': {'quant_types': list(entry['sources']['quant_types']),
                    'covered': all(name.casefold() in covered_types for name in source_types)},
        'measurement': entry['measurement'],
        'bundle': bundle,
        'refusal': None,
    }


def bank_check(bank_dir, gguf, key=None):
    """CLI entry: the resolved report, or a `BankRefused` the envelope names."""
    return resolve(bank_dir, gguf, key)
