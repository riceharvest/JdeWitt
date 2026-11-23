# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Space command: THEIR GGUF in, an edited GGUF out, one JSON, no model.

`bank-apply` is the consumer half of the direction bank (§5, §7.5). One bank, one
entry, one artifact, one JSON — the same shape `plan-apply` gives a plan card, with
two differences that belong to the bank and not to a card:

  * the axis is chosen by the bank, per residual-write kind (§5.1: attention output
    and FFN down carry their own strength), so the edit map is
    ``{tensor_name: (axis[layer], entry.edit.space)}`` built from the entry's
    resolved footprint — never a per-model tensor table and never a hand-typed name;
  * the storage type is chosen from the types the entry MEASURED, and the ladder
    refuses (CODEC_FLOOR) rather than widening the type or raising the strength to
    make the floor. The deliverable is the honest number, not a written file.

The steps, in order, all before the writer opens ``out``:

  * ``bank.resolve(bank_dir, gguf, key=entry)`` — typed refusals (BANK_MISS,
    GEOMETRY_MISMATCH, UNSUPPORTED_QUANT, BUNDLE_BAD_DIGEST) from the artifact's own
    header and tensor table;
  * ``registry.unbundle_plan`` into a THROWAWAY registry, then ``registry.load_plan``
    — the file beside bank.json is never trusted in place, and load_plan re-verifies
    the signed meta, the payload sha256, and the basis geometry;
  * the footprint is placed on the basis rows (a projected footprint spans the
    entry's own footprint, endpoints preserved), and every artifact layer the bank
    resolved must have a basis row;
  * ``ggufio.preview_edit`` probes the worst edited tensor per candidate type, in the
    entry's own type order, and the first type that clears ``entry.edit.min_cos`` is
    the one written;
  * ``rewrite.rewrite_with_edits`` re-encodes ONLY those tensors, with per-kind
    strength, copying every other tensor byte for byte;
  * ``ggufio.compare_edit`` measures what shipped, per tensor.

Why not call ``planapply.plan_apply``: it takes ONE strength for every tensor and
one caller-supplied ``--min-cos``, and it falls back to f16 when its ladder misses.
A bank entry has per-kind strength and a measured type set, and a missed floor is a
refusal here. The write path itself is the shared one (``rewrite_with_edits``);
nothing about the container is reimplemented.

The cos/SNR below are CODEC-realized fidelity: ``compare_edit`` compares the stored
change against the intended projection. Removal is NOT measured here and the report
says so (``removal_measured: false``).
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np

from . import bank as bank_module
from .errors import BankApplyRefused
from .registry import RegistryError, _verify_meta, load_plan, unbundle_plan
from .storage import file_digest

SCHEMA = 'jdewitt-bank-apply-v1'
# Smallest first: the ladder stops at the first type that stores the edit at the
# entry's own floor, so the file that ships is the cheapest faithful one.
AUTO_LADDER = ('q6_k', 'q8_0', 'f16')


def _canonical(edit_type):
    """Name the requested type, or refuse with the encodable list this build has."""
    from .ggufio import encodable_types
    from .quantize import canonical_type

    encodable = encodable_types()
    try:
        name = canonical_type(edit_type)
    except ValueError as error:
        raise BankApplyRefused('EDIT_TYPE_UNKNOWN',
                               f'{error}; this build can ENCODE: {", ".join(encodable)}') from error
    if name not in encodable:
        raise BankApplyRefused(
            'EDIT_TYPE_NOT_ENCODABLE',
            f'{edit_type!r} ({name}) has no encoder in this build; it can ENCODE: '
            f'{", ".join(encodable)}. Refusing instead of shipping an edit nobody measured')
    return name


def _ladder(edit, key):
    """The entry's measured types this build can ENCODE, smallest first.

    A type the entry measured but this build cannot write is dropped from AUTO with
    its name kept for the report: it is not a candidate, and silently widening to it
    (or to another type the entry never measured) is what this command must not do.
    """
    from .ggufio import encodable_types

    encodable = set(encodable_types())
    from .quantize import canonical_type

    allowed, skipped = [], []
    for name in edit['types']:
        try:
            canonical = canonical_type(name)
        except ValueError:
            skipped.append(str(name))
            continue
        if canonical not in encodable:
            skipped.append(str(canonical))
            continue
        if canonical not in allowed:
            allowed.append(canonical)
    if not allowed:
        raise BankApplyRefused(
            'EDIT_TYPE_NOT_ENCODABLE',
            f'entry {key!r} allows {", ".join(map(str, edit["types"]))}, none of '
            f'which this build can ENCODE ({", ".join(sorted(encodable))}); nothing written')
    order = {name: position for position, name in enumerate(AUTO_LADDER)}
    return sorted(allowed, key=lambda name: order.get(name.casefold(), len(AUTO_LADDER))), skipped


def _select_edit_type(gguf, groups, probes, ladder, edit_type, min_cos, space, skipped=()):
    """Pick the storage type: the first the entry allows whose probe clears the floor.

    The probe runs the SAME codec the writer will use (``preview_edit`` with
    ``target_type``) over the worst edited tensors of each kind, at that kind's own
    strength. The floor only picks the type; the numbers in the report are measured
    afterwards on the written file, per tensor.
    """
    from .ggufio import preview_edit

    if edit_type != 'auto':
        chosen = _canonical(edit_type)
        allowed = {name.casefold() for name in ladder}
        if chosen.casefold() not in allowed:
            raise BankApplyRefused(
                'EDIT_TYPE_NOT_ALLOWED',
                f'--edit-type {edit_type!r} ({chosen}) is outside the types this entry '
                f'measured ({", ".join(ladder)}); pass auto, or one of those')
        return {'mode': 'explicit', 'floor': min_cos, 'chosen': chosen,
                'candidates': ladder, 'not_candidates': list(skipped), 'readings': [],
                'floor_cleared': None,
                'note': 'the caller named the type; the delivered numbers below still say '
                        'what it stored'}

    readings = []
    for kind in ladder:
        rows = []
        for group in groups.values():
            names = [name for name in probes if name in set(group['names'])]
            if not names:
                continue
            reading = preview_edit(str(gguf), {name: (group['axis'][name], space) for name in names},
                                   names=names, scale=group['scale'], target_type=kind)
            rows.extend(reading['tensors'])
        if not rows:
            continue
        worst = min(rows, key=lambda row: row['cos'])
        readings.append({'edit_type': kind, 'min_cos': worst['cos'],
                         'worst_tensor': worst['name'], 'min_snr': worst['snr'],
                         'mean_cos': round(sum(row['cos'] for row in rows) / len(rows), 4),
                         'probed': len(rows)})
        if worst['cos'] >= min_cos:
            return {'mode': 'auto', 'floor': min_cos, 'chosen': kind, 'candidates': ladder,
                    'not_candidates': list(skipped), 'readings': readings, 'floor_cleared': True,
                    'note': 'candidates are probed smallest-first and probing stops at the '
                            'first clear; the delivered numbers below are measured on the '
                            'written file'}
    best = max(readings, key=lambda row: row['min_cos']) if readings else None
    if best is None:
        raise BankApplyRefused(
            'CODEC_FLOOR',
            f'none of the entry\'s types ({", ".join(ladder)}) could be probed on this '
            f'artifact, so no type can be shown to store the edit at cos >= {min_cos}')
    raise BankApplyRefused(
        'CODEC_FLOOR',
        f'entry allows {", ".join(ladder)}; none stores the edit at cos >= {min_cos}: '
        f'worst probed tensor {best["worst_tensor"]} reaches cos {best["min_cos"]} at '
        f'{best["edit_type"]}. Nothing written. The strength is not raised to hide this — '
        f'a write that only survives at a higher strength is an edit nobody measured')


def _row_by_layer(resolved):
    """Artifact layer -> the bank entry's own footprint layer (§5.1).

    A projected footprint sits on the entry's footprint span with both endpoints
    preserved, so a different layer count re-spaces the measured rows instead of
    re-indexing them onto the wrong depth. When the counts agree this is the
    identity.
    """
    footprint = resolved['footprint']
    layers = resolved['layers']
    first, last = int(footprint['first']), int(footprint['last'])
    if not layers:
        raise BankApplyRefused(
            'GEOMETRY_MISMATCH',
            f'entry {resolved["entry"]["key"]!r} resolves to an empty footprint on this '
            f'artifact; nothing to edit')
    low, high = int(layers[0]), int(layers[-1])
    mapping = {}
    for layer in layers:
        if high == low:
            mapping[int(layer)] = first
        else:
            mapping[int(layer)] = int(round(first + (int(layer) - low) * (last - first) / (high - low)))
    return mapping


def _plan_rows(plan, key):
    """The basis row -> fitted-layer list a plan carries, or 0..n-1 when it omits it."""
    basis = plan.get('basis')
    if basis is None or getattr(basis, 'ndim', 0) != 3:
        raise BankApplyRefused(
            'BUNDLE_REFUSED',
            f'plan {key} carries no [layers,rank,hidden] basis; bank-apply edits from a 3D '
            f'basis only')
    named = plan.get('layers')
    rows = int(basis.shape[0])
    if named is None:
        return list(range(rows))
    if (not isinstance(named, list) or len(named) != rows
            or any(type(layer) is not int for layer in named)
            or named != sorted(set(named))):
        raise BankApplyRefused(
            'BUNDLE_REFUSED',
            f'plan {key} carries layer indices {named!r}; expected {rows} strictly increasing '
            f'integers matching the basis rows')
    return list(named)


def _build_edits(resolved, basis, plan_rows, strength_scale):
    """``{tensor_name: (axis[layer], space)}`` plus the per-kind strength it lands at."""
    row_of = {layer: position for position, layer in enumerate(plan_rows)}
    space = str(resolved['edit'].get('space') or 'output')
    if space not in ('output', 'input'):
        raise BankApplyRefused(
            'BUNDLE_REFUSED',
            f'entry {resolved["entry"]["key"]!r} states edit.space {space!r}; the writer only '
            f'knows output and input')
    layer_rows = _row_by_layer(resolved)
    edits, scales, axis = {}, {}, {}
    strength = {kind: float(value) * float(strength_scale)
                for kind, value in resolved['strength'].items()}
    for kind, names in resolved['targets'].items():
        for layer, name in zip(resolved['layers'], names):
            entry_layer = layer_rows[int(layer)]
            if entry_layer not in row_of:
                raise BankApplyRefused(
                    'GEOMETRY_MISMATCH',
                    f'artifact layer {layer} maps to the entry\'s fitted layer {entry_layer}, '
                    f'but the bundle\'s basis carries rows for '
                    f'{plan_rows[:4]}{"..." if len(plan_rows) > 4 else ""}; nothing written')
            vector = np.asarray(basis[row_of[entry_layer]], dtype=np.float32)
            edits[name] = (vector, space)
            axis[name] = vector
            scales[name] = strength[kind]
    return edits, scales, axis, strength, layer_rows, space


def _check_paths(gguf, out):
    if not gguf.is_file():
        raise FileNotFoundError(str(gguf))
    if gguf.resolve() == out.resolve():
        raise BankApplyRefused(
            'OUT_IS_INPUT',
            f'--out {out} is the input artifact; bank-apply always writes a new file and never '
            f'edits the file it read')
    if not out.parent.is_dir():
        raise BankApplyRefused('OUT_DIRECTORY_MISSING', f'{out.parent} is not a directory')


def _probe_names(resolved):
    """The worst edited tensors to probe: both ends of each kind's footprint, capped."""
    names = []
    for kind, tensors in resolved['targets'].items():
        if not tensors:
            continue
        for name in (tensors[0], tensors[-1]):
            if name not in names:
                names.append(name)
            if len(names) >= 4:
                break
        if len(names) >= 4:
            break
    return names


def bank_apply(bank_dir, entry_key, gguf, out, edit_type='auto', strength_scale=1.0):
    """Apply one bank entry to ``gguf`` and write ``out``; returns the one report.

    Every refusal happens before the writer opens ``out``, so a refused run never
    leaves a partly-edited artifact under the requested name.
    """
    from .ggufio import compare_edit
    from .rewrite import rewrite_with_edits

    started = time.monotonic()
    bank_dir, gguf, out = Path(bank_dir), Path(gguf), Path(out)
    _check_paths(gguf, out)
    if not (float(strength_scale) > 0):
        raise BankApplyRefused(
            'INPUT_ERROR', f'--strength-scale must be positive, not {strength_scale!r}; '
                           f'a zero or negative scale is not an edit')

    resolved = bank_module.resolve(bank_dir, gguf, key=entry_key)
    entry = resolved['entry']
    bundle_path = bank_dir / resolved['bundle']['file']
    geometry = resolved['geometry']
    repair = str(resolved['edit'].get('repair') or 'none')

    with tempfile.TemporaryDirectory(prefix='jdewitt-bank-apply-') as registry_dir:
        try:
            installed = unbundle_plan(bundle_path, registry_dir)
            plan_key = installed['key']
            meta = _verify_meta(registry_dir, plan_key)
            # Only the width is a whole-model fact the basis must match: the basis
            # carries one row per FITTED footprint layer (named in the payload), not
            # one per block of the model, so requiring entry.geometry.layers here
            # would reject every footprint plan. Row coverage is checked below.
            loaded = load_plan(registry_dir, plan_key,
                               expect={'hidden': int(geometry['hidden'])})
        except (RegistryError, OSError) as error:
            raise BankApplyRefused('BUNDLE_REFUSED',
                                   f'entry {entry["key"]!r} pins {resolved["bundle"]["file"]!r} '
                                   f'but that bundle did not verify: {error}') from error
        plan = loaded['plan']
        basis = np.asarray(plan['basis'].detach().float().cpu().numpy(), dtype=np.float32)
        plan_rows = _plan_rows(plan, plan_key)
        edits, scales, axis, strength, layer_rows, space = _build_edits(
            resolved, basis, plan_rows, strength_scale)

        groups = {}
        for kind, names in resolved['targets'].items():
            groups[kind] = {'names': names, 'scale': strength[kind],
                            'axis': {name: axis[name] for name in names}}
        probes = _probe_names(resolved)
        ladder, skipped = _ladder(resolved['edit'], entry['key'])
        selection = _select_edit_type(gguf, groups, probes, ladder, edit_type,
                                      float(resolved['edit'].get('min_cos') or 0.95), space,
                                      skipped)
        chosen = selection['chosen']

        temp_path = out.with_name(f'{out.name}.partial-{os.getpid()}')
        try:
            rewrite = rewrite_with_edits(str(gguf), str(temp_path), edits, scales=scales,
                                         axis_space=space, edit_type=chosen,
                                         verify_untouched=True, repair=repair)
            os.replace(temp_path, out)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    renorm = repair != 'none'
    rows, delivered_by_kind = [], {}
    for kind, names in resolved['targets'].items():
        if not names:
            continue
        measured = compare_edit(str(gguf), str(out), {name: edits[name] for name in names},
                                names=names, scale=strength[kind], renormalize=renorm)
        snrs = [row['snr'] for row in measured['tensors'] if row.get('snr') is not None]
        delivered_by_kind[kind] = {'min_cos': measured['min_cos'],
                                   'mean_cos': measured['mean_cos'],
                                   'min_snr': min(snrs) if snrs else None}
        for row in measured['tensors']:
            layer = int(row['name'].split('.')[1])
            rows.append({'name': row['name'], 'layer': layer, 'kind': kind,
                         'source_type': row.get('source_type'),
                         'edited_type': row.get('edited_type'),
                         'intended_strength': strength[kind],
                         'cos': row.get('cos'), 'snr': row.get('snr'),
                         'norm_ratio': row.get('norm_ratio')})
    cosines = [row['cos'] for row in rows if row.get('cos') is not None]
    floor = float(resolved['edit'].get('min_cos') or 0.95)
    untouched = rewrite.get('untouched') or {}
    fingerprint = rewrite.get('fingerprint') or {}
    report = {
        'command': 'bank-apply',
        'schema': SCHEMA,
        'entry_key': entry['key'],
        'bundle_sha256': resolved['bundle']['sha256'],
        'base': entry['base'],
        'bank': {'dir': str(bank_dir), 'entry': entry['key'],
                 'base': entry['base'],
                 'bundle': {'file': resolved['bundle']['file'],
                            'sha256': resolved['bundle']['sha256'],
                            'verified': resolved['bundle']['verified'],
                            'plan_key': plan_key,
                            'plan_meta_digest': meta.get('meta_digest'),
                            'payload_sha256': meta.get('payload_sha256'),
                            'installed_into_temp_registry': installed.get('installed')}},
        'artifact': {'input': str(gguf), 'output': str(out),
                     'architecture': resolved['artifact']['architecture'],
                     'layers': resolved['artifact']['layers'],
                     'hidden': resolved['artifact']['hidden'],
                     'tensors': resolved['artifact']['tensors'],
                     'source_types': resolved['artifact']['source_types']},
        'entry': {'key': entry['key'], 'base': entry['base'],
                  'measurement': resolved['measurement'],
                  'edit': resolved['edit'],
                  'strength': strength,
                  'strength_scale': float(strength_scale),
                  'strength_note': 'per-kind entry strength times --strength-scale; the writer '
                                   'applies it per tensor, not as one flat number'},
        'footprint': {'first': int(resolved['footprint']['first']),
                      'last': int(resolved['footprint']['last']),
                      'fraction': resolved['footprint']['fraction'],
                      'projected': bool(resolved['footprint']['projected']),
                      'artifact_layers': resolved['layers'],
                      'entry_rows': [layer_rows[layer] for layer in resolved['layers']],
                      'projection_rule': 'a projected footprint spans the entry\'s own '
                                         'footprint with both endpoints preserved'},
        'edit_type': selection,
        'edited': {'count': len(rows),
                   'min_cos': min(cosines) if cosines else None,
                   'mean_cos': round(sum(cosines) / len(cosines), 4) if cosines else None,
                   'min_snr': min((row['snr'] for row in rows if row.get('snr') is not None),
                                  default=None),
                   'floor': floor,
                   'floor_met': (min(cosines) if cosines else 0.0) >= floor,
                   'below_floor': [row['name'] for row in rows
                                   if (row.get('cos') or 0.0) < floor],
                   'by_kind': delivered_by_kind,
                   'tensors': rows},
        'untouched': {'count': untouched.get('checked'),
                      'verified': bool(untouched.get('verified')),
                      'mismatched': untouched.get('mismatched') or [],
                      'missing': untouched.get('missing') or [],
                      'method': 'byte-for-byte re-read of BOTH files after the write; the '
                                'edited tensors are excluded by construction'},
        'fingerprint': fingerprint,
        'bytes_in': rewrite.get('bytes_in'), 'bytes_out': rewrite.get('bytes_out'),
        'input_sha256': file_digest(gguf), 'output_sha256': file_digest(out),
        'wall_seconds': round(time.monotonic() - started, 2),
        'ok': bool(untouched.get('verified')) and (min(cosines) if cosines else 0.0) >= floor,
        'status': ('applied' if (min(cosines) if cosines else 0.0) >= floor
                   else 'applied_below_floor'),
        'removal_measured': False,
        'claims': {
            'removal': 'not measured and not claimed: removal is GATED behind a real serve '
                       'config and a judge wave, and this command has no model',
            'delivered_cos': 'codec fidelity of the stored edit against the intended '
                             'projection, computed on CPU from the two GGUF files — a fact '
                             'about bytes, not about answers',
            'untouched_bytes': 'verified by re-reading both files, not asserted',
            'determinism': 'same bank + same input bytes + same flags rewrite to the same '
                           'output sha256; the writer emits no timestamps or paths'},
        'next': [f'jdewitt gguf-inspect {out}',
                 f'jdewitt gguf-tensor-diff {gguf} {out}  # only the named tensors may differ',
                 '# certify removal on the EDITED artifact with a real serve config; '
                 'this run measured codec fidelity only'],
    }
    return report
