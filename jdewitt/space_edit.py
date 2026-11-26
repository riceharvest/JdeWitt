# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Space's editor path: a visitor's GGUF in, the same file edited, out.

One importable function, no model, no server, no capture:

    from jdewitt.space_edit import edit_artifact
    report = edit_artifact(src, out, axis=None, edit_type='q8_0')

A Gradio handler calls it and hands back ``out``; nothing here imports torch,
transformers, a tokenizer or a server client, so the Space's requirements can
stay free of an inference stack (SPACE-ROADMAP §7.5).

What it does, in the order the order matters:

  * **the size cap is checked from ``stat()``, before the file is opened.**
    ``max_input_bytes`` defaults to ``MAX_INPUT_BYTES``; an input above it is
    refused with ``INPUT_TOO_LARGE`` and no byte of it is read. This is the
    first decision the function makes, so a hostile or mistaken 20 GB upload
    costs one syscall, not a download and a parse.
  * **geometry comes from the artifact's own tensor table** — hidden width from
    the residual-write projections, the target names from
    ``discover_writeback_targets`` (name *and* width, so a hidden-wide
    ``attn_q`` is not a writeback tensor). No per-model table is consulted.
  * **an artifact that already carries ``jdewitt.fingerprint``** is answered
    with a byte copy and ``status: already_applied`` when the stored digest is
    this edit; a *different* stored edit is refused (``EDIT_ALREADY_PRESENT``)
    rather than applied twice, because removing the same direction twice is a
    no-op that cos cannot see and a second, different axis is not the file the
    bank measured.
  * **only the target tensors are re-encoded** (``rewrite_with_edits``); every
    other tensor is copied byte for byte and both files are re-read to prove it.
  * **the delivered edit is measured** (``ggufio.compare_edit``) — cos between
    the stored change and the intended projection, per tensor. That is a fact
    about bytes; what the model answers is GATED behind a serve.

The axis is the caller's business: a bank entry's fitted rows, or ``None`` for
the local dry run's **synthetic** axis — deterministic unit vectors derived from
a seed, used to exercise the path with no capture. A synthetic axis is not a
removal and the report says so.

Deliberately absent: an ``auto`` type ladder. §8 measured that the probe reads
0.9961 on Q6_K while the written file delivers 0.8815, so a probe-chosen type is
not trusted here; the handler names the type (``q8_0`` is the one the 4B card
measured as clearing the floor) and reads ``edited.floor_met`` before delivery.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

SCHEMA = 'jdewitt-space-edit-v1'

# 6 GiB. The covered v1 class tops out at a 7B Q4_K_M (~4.5 GB) and the largest
# transfer probe measured 5.5 GB; a 13B-class artifact would also need ~15 GB of
# float32 for its edited tensors, past the Space's 16 GB RAM. The cap is a
# refusal, not a promise: it is checked before the file is opened.
MAX_INPUT_BYTES = 6 << 30

SYNTHETIC_SEED = 'jdewitt-synthetic-axis-v1'
AXIS_SPACE = 'output'

# What this file cannot answer, stated where a report reader will see it.
GATED = [
    {'claim': 'zero_gpu_quota',
     'state': 'GATED',
     'detail': 'the editor never calls @spaces.GPU; no GPU second and no daily quota was spent'},
    {'claim': 'gpu_throughput',
     'state': 'GATED',
     'detail': 'wall clock here is this host\'s CPU, not a ZeroGPU container; no throughput claim'},
    {'claim': 'served_behavior',
     'state': 'GATED',
     'detail': 'removal, refusal rates and retention need a real serve config and a judge wave; '
               'this path measured codec fidelity only'},
]

_TARGET_KIND = {'attn_output': 'attn', 'attn_out': 'attn', 'ssm_out': 'attn',
                'ffn_down': 'mlp', 'w2': 'mlp'}


class SpaceEditRefused(ValueError):
    """The artifact cannot be edited under the stated rules; ``code`` names why.

    INPUT_TOO_LARGE: the size cap refused it from ``stat()``, nothing was read.
    EDIT_ALREADY_PRESENT: the file carries a different ``jdewitt.fingerprint``.
    NO_WRITEBACK_TARGETS: the tensor table carries no residual-write projection.
    AXIS_WIDTH_MISMATCH: a supplied axis is not the artifact's hidden width.
    EDIT_TYPE_NOT_ENCODABLE: this build cannot write the requested type.
    OUT_IS_INPUT / OUT_DIRECTORY_MISSING: the output cannot be written safely.
    """

    def __init__(self, code, message):
        self.code = code
        super().__init__(f'{code}: {message}')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _stored_fingerprint(path):
    """The ``jdewitt.fingerprint`` kv a previous edit wrote into the artifact.

    Read here rather than imported from ``bank_apply``: the Space editor is not
    a bank consumer, and a handler must not break because another card moved the
    bank's writer. Returns None when the file carries no such kv.
    """
    from gguf import GGUFReader

    field = GGUFReader(str(path)).fields.get('jdewitt.fingerprint')
    if field is None:
        return None
    raw = field.contents()
    if isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw).decode('utf-8', 'replace')
    try:
        value = json.loads(str(raw))
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _artifact_facts(path):
    """Header-only facts: architecture, hidden width, writeback targets, types."""
    from collections import Counter

    from gguf import GGUFReader

    from .ggufgeom import (discover_writeback_targets, gguf_weight_shapes,
                           inferred_hidden_width)

    reader = GGUFReader(str(path))
    architecture = None
    field = reader.fields.get('general.architecture')
    if field is not None:
        architecture = str(field.contents())
    shapes = gguf_weight_shapes(path, reader=reader)
    hidden = inferred_hidden_width(shapes)
    discovered = (discover_writeback_targets(shapes, hidden)
                  if hidden is not None else {'targets': {}, 'unmapped_writeback_width': []})
    return {
        'architecture': architecture,
        'hidden': hidden,
        'shapes': shapes,
        'targets': discovered['targets'],
        'unmapped_writeback_width': discovered['unmapped_writeback_width'],
        'tensors': len(reader.tensors),
        'source_types': dict(sorted(Counter(t.tensor_type.name for t in reader.tensors).items())),
    }


def _footprint(discovered, first=None, last=None):
    """The layer list to edit: every layer carrying a writeback tensor by default."""
    layers = sorted(discovered)
    if first is not None:
        layers = [layer for layer in layers if layer >= int(first)]
    if last is not None:
        layers = [layer for layer in layers if layer <= int(last)]
    return layers


def synthetic_axis(layers, hidden, seed=SYNTHETIC_SEED):
    """A deterministic unit direction per layer: fixture, not a fitted removal.

    Derived from ``sha256(seed:layer:hidden)`` so two runs on the same tree pick
    the same axes and any reader can recompute them. The report labels the axis
    ``synthetic`` and claims no removal.
    """
    axes = {}
    for layer in layers:
        digest = hashlib.sha256(f'{seed}:{int(layer)}:{int(hidden)}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'big'))
        vector = rng.standard_normal(int(hidden)).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise SpaceEditRefused('AXIS_WIDTH_MISMATCH',
                                   f'synthetic axis for layer {layer} has zero norm')
        axes[int(layer)] = vector / np.float32(norm)
    return axes


def _check_axis(axis, layers, hidden):
    """Validate a supplied {layer: vector}; coerce JSON string keys to ints."""
    checked = {}
    for key, vector in dict(axis).items():
        layer = int(key)
        if layer not in layers:
            continue
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        if array.size != int(hidden):
            raise SpaceEditRefused(
                'AXIS_WIDTH_MISMATCH',
                f'axis for layer {layer} has {array.size} values, the artifact\'s '
                f'writeback width is {hidden}; nothing written')
        if not np.isfinite(array).all():
            raise SpaceEditRefused('AXIS_WIDTH_MISMATCH',
                                   f'axis for layer {layer} contains NaN/Inf; nothing written')
        checked[layer] = array
    missing = [layer for layer in layers if layer not in checked]
    if missing:
        raise SpaceEditRefused(
            'AXIS_WIDTH_MISMATCH',
            f'no axis supplied for layers {missing[:8]}{"..." if len(missing) > 8 else ""}; '
            f'the footprint has {len(layers)} layers and every one needs a direction')
    return checked


def _build_edits(discovered, layers, axes):
    """{tensor name: (axis, 'output')} — the writer's dual-form edit map."""
    edits, names_by_kind = {}, {'attn': [], 'mlp': []}
    for layer in layers:
        for name in discovered.get(layer, []):
            suffix = name.split('.')[2]
            kind = _TARGET_KIND.get(suffix)
            if kind is None:
                continue
            edits[name] = (axes[layer], AXIS_SPACE)
            names_by_kind[kind].append(name)
    return edits, names_by_kind


def _empty_edited(floor, note):
    return {'count': 0, 'tensors': [], 'min_cos': None, 'mean_cos': None,
            'min_snr': None, 'floor': float(floor), 'floor_met': None,
            'below_floor': [], 'by_kind': {}, 'note': note}


def _already_applied_report(facts, src, out, stored, axis_block, layers, edits,
                            names_by_kind, strength, floor, started):
    """The one report for an artifact that already carries this edit.

    Nothing was rewritten, so there is no delivered edit to measure and the cos
    block is null rather than copied from a previous run. The output is a byte
    copy, which is why both digests are equal.
    """
    input_bytes, output_bytes = int(src.stat().st_size), int(out.stat().st_size)
    input_sha, output_sha = _sha256(src), _sha256(out)
    return {
        'schema': SCHEMA,
        'status': 'already_applied',
        'ok': True,
        'applied': False,
        'already_applied': True,
        'input': {'path': str(src), 'bytes': input_bytes, 'sha256': input_sha,
                  'architecture': facts['architecture'], 'tensors': facts['tensors'],
                  'hidden': facts['hidden'], 'source_types': facts['source_types']},
        'output': {'path': str(out), 'bytes': output_bytes, 'sha256': output_sha},
        'size_cap': {'max_input_bytes': int(MAX_INPUT_BYTES),
                     'input_bytes': input_bytes,
                     'checked_before_read': True},
        'axis': axis_block,
        'footprint': {'first': min(layers) if layers else None,
                      'last': max(layers) if layers else None,
                      'layers': layers,
                      'fraction_of_artifact_layers': (
                          round(len(layers) / (max(facts['targets']) + 1), 4)
                          if facts['targets'] else None)},
        'targets': {'tensor_count': len(edits), 'by_kind': names_by_kind,
                    'names': sorted(edits)},
        'edit_type': {'requested': None, 'chosen': (stored or {}).get('edit_type'),
                      'note': 'the file already carries this edit; no type was chosen and '
                              'nothing was rewritten'},
        'strength': float(strength),
        'edited': _empty_edited(
            floor, 'nothing rewritten: the input already carries this edit fingerprint, so '
                   'applying the same axis again would remove the direction twice; cos cannot '
                   'see that, which is why it is answered with a byte copy'),
        'untouched': {'count': facts['tensors'], 'verified': True, 'mismatched': [],
                      'missing': [],
                      'method': 'no rewrite: the output is a byte copy of the input, so every '
                                'tensor is identical by construction'},
        'fingerprint': stored,
        'bytes_in': int(src.stat().st_size), 'bytes_out': int(out.stat().st_size),
        'input_sha256': input_sha, 'output_sha256': output_sha,
        'wall_seconds': round(time.monotonic() - started, 2),
        'removal_measured': False,
        'gated': GATED,
        'claims': {
            'byte_copy': 'the output is shutil.copyfile(input); both sha256 are printed so the '
                         'copy is checkable, not asserted',
            'removal': 'not measured and not claimed; the input already carried an edit and '
                       'this run had no model',
        },
        'next': [f'jdewitt gguf-inspect {out}',
                 '# re-run on the PRISTINE artifact to measure the delivered edit again'],
    }


def edit_artifact(src, out, *, axis=None, first=None, last=None, strength=1.0,
                  edit_type='q8_0', min_cos=0.95, max_input_bytes=MAX_INPUT_BYTES,
                  verify_untouched=True, report_path=None):
    """Edit ``src`` into ``out`` and return the one report (schema v1).

    ``axis`` is ``{layer: unit vector}``; ``None`` builds the deterministic
    synthetic fixture axis. ``first``/``last`` bound the layer footprint; by
    default every layer that carries a writeback tensor is edited. The output
    is written to a temp name and renamed, so a failed run leaves no half file
    under ``out``.
    """
    from .ggufio import compare_edit
    from .rewrite import axis_fingerprint, rewrite_with_edits

    started = time.monotonic()
    src, out = Path(src), Path(out)

    # 1. The cap, from stat(), before ANY open: no parse, no digest, no read.
    if not src.is_file():
        raise FileNotFoundError(str(src))
    size = int(src.stat().st_size)
    if max_input_bytes is not None and size > int(max_input_bytes):
        raise SpaceEditRefused(
            'INPUT_TOO_LARGE',
            f'{src} is {size} bytes, above the {int(max_input_bytes)}-byte input cap; '
            f'refused from stat() before the file was opened and nothing was read')

    # 2. Output path rules, still before a read.
    if src.resolve() == out.resolve():
        raise SpaceEditRefused('OUT_IS_INPUT',
                               f'{out} is the input artifact; the editor always writes a new file')
    if not out.parent.is_dir():
        raise SpaceEditRefused('OUT_DIRECTORY_MISSING', f'{out.parent} is not a directory')

    facts = _artifact_facts(src)
    if facts['hidden'] is None or not facts['targets']:
        raise SpaceEditRefused(
            'NO_WRITEBACK_TARGETS',
            f'{src} declares architecture {facts["architecture"]!r} and carries no residual-write '
            f'projection (attn_output/attn_out/ssm_out/ffn_down/w2); there is nothing an output-space '
            f'axis could edit')

    layers = _footprint(facts['targets'], first, last)
    if not layers:
        raise SpaceEditRefused('NO_WRITEBACK_TARGETS',
                               f'no writeback layer falls inside first={first} last={last}')

    synthetic = axis is None
    if synthetic:
        axes = synthetic_axis(layers, facts['hidden'])
        axis_block = {'kind': 'synthetic', 'seed': SYNTHETIC_SEED, 'space': AXIS_SPACE,
                      'rank': 1, 'layers': layers,
                      'note': 'deterministic fixture direction derived from sha256(seed:layer:hidden), '
                              'NOT a fitted removal; no capture was read and no removal is claimed',
                      'synthetic': True}
    else:
        axes = _check_axis(axis, layers, facts['hidden'])
        axis_block = {'kind': 'supplied', 'space': AXIS_SPACE, 'rank': 1, 'layers': layers,
                      'norms': {str(layer): round(float(np.linalg.norm(axes[layer])), 6)
                                for layer in layers},
                      'note': 'caller-supplied directions, applied as-is; the caller names their fit',
                      'synthetic': False}

    edits, names_by_kind = _build_edits(facts['targets'], layers, axes)
    if not edits:
        raise SpaceEditRefused('NO_WRITEBACK_TARGETS',
                               f'the footprint layers {layers} carry no attn/mlp writeback tensor')

    # 3. Already applied? Checked before the writer and before any type choice.
    intended = axis_fingerprint(edits)
    stored = _stored_fingerprint(src)
    if stored is not None:
        same = (stored.get('edit_sha256') == intended['edit_sha256']
                and stored.get('axis_space') == AXIS_SPACE
                and sorted(stored.get('layers') or []) == intended['layers']
                and float(stored.get('scale', strength)) == float(strength))
        if same:
            temp_path = out.with_name(f'{out.name}.partial-{os.getpid()}')
            try:
                shutil.copyfile(src, temp_path)
                os.replace(temp_path, out)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
            report = _already_applied_report(facts, src, out, stored, axis_block, layers, edits,
                                             names_by_kind, strength, min_cos, started)
            if report_path:
                Path(report_path).write_text(json.dumps(report, indent=1, sort_keys=False) + '\n')
            return report
        raise SpaceEditRefused(
            'EDIT_ALREADY_PRESENT',
            f'{src} carries a jdewitt.fingerprint for another edit '
            f'(edit_sha256 {stored.get("edit_sha256")!r}, space {stored.get("axis_space")!r}, '
            f'layers {len(stored.get("layers") or [])}); applying a second edit to an edited '
            f'artifact is not the file the bank measured — use the pristine artifact')

    # 4. Storage type: named, validated, never probe-chosen (§8).
    from .gguf_types import canonical_type
    from .ggufio import encodable_types

    try:
        chosen = canonical_type(edit_type)
    except ValueError as error:
        raise SpaceEditRefused(
            'EDIT_TYPE_NOT_ENCODABLE',
            f'{error}; this build can ENCODE: {", ".join(encodable_types())}') from error
    if chosen not in encodable_types():
        raise SpaceEditRefused(
            'EDIT_TYPE_NOT_ENCODABLE',
            f'{edit_type!r} ({chosen}) has no encoder in this build; it can ENCODE: '
            f'{", ".join(encodable_types())}. Refusing instead of shipping an edit nobody measured')

    # 5. Write, then rename. Only the target tensors are re-encoded.
    temp_path = out.with_name(f'{out.name}.partial-{os.getpid()}')
    try:
        rewrite = rewrite_with_edits(str(src), str(temp_path), edits, scale=float(strength),
                                     axis_space=AXIS_SPACE, edit_type=chosen,
                                     verify_untouched=bool(verify_untouched))
        os.replace(temp_path, out)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    # 6. What shipped, per tensor.
    measured = compare_edit(str(src), str(out), edits, names=sorted(edits),
                            scale=float(strength))
    rows, by_kind = [], {}
    for row in measured['tensors']:
        name = row['name']
        layer = int(name.split('.')[1])
        kind = _TARGET_KIND.get(name.split('.')[2], 'other')
        cosine = row.get('cos')
        entry = {'name': name, 'layer': layer, 'kind': kind,
                 'source_type': row.get('source_type'), 'edited_type': row.get('edited_type'),
                 'cos': cosine, 'snr': row.get('snr'), 'norm_ratio': row.get('norm_ratio')}
        rows.append(entry)
        bucket = by_kind.setdefault(kind, {'count': 0, 'cosines': []})
        bucket['count'] += 1
        if cosine is not None:
            bucket['cosines'].append(cosine)
    for kind, bucket in by_kind.items():
        cosines = bucket.pop('cosines')
        bucket['min_cos'] = min(cosines) if cosines else None
        bucket['mean_cos'] = round(sum(cosines) / len(cosines), 4) if cosines else None
    cosines = [row['cos'] for row in rows if row.get('cos') is not None]
    floor = float(min_cos)
    untouched = rewrite.get('untouched') or {}
    floor_met = bool(cosines) and min(cosines) >= floor
    input_sha, output_sha = _sha256(src), _sha256(out)
    report = {
        'schema': SCHEMA,
        'status': 'edited' if floor_met else 'edited_below_floor',
        'ok': bool(untouched.get('verified')) and floor_met,
        'applied': True,
        'already_applied': False,
        'input': {'path': str(src), 'bytes': int(rewrite.get('bytes_in') or size),
                  'sha256': input_sha, 'architecture': facts['architecture'],
                  'tensors': facts['tensors'], 'hidden': facts['hidden'],
                  'source_types': facts['source_types']},
        'output': {'path': str(out), 'bytes': int(rewrite.get('bytes_out') or out.stat().st_size),
                   'sha256': output_sha},
        'size_cap': {'max_input_bytes': int(max_input_bytes) if max_input_bytes else None,
                     'input_bytes': size, 'checked_before_read': True},
        'axis': axis_block,
        'footprint': {'first': min(layers), 'last': max(layers), 'layers': layers,
                      'fraction_of_artifact_layers': (
                          round(len(layers) / (max(facts['targets']) + 1), 4))},
        'targets': {'tensor_count': len(edits), 'by_kind': names_by_kind,
                    'names': sorted(edits)},
        'edit_type': {'requested': str(edit_type), 'chosen': chosen,
                      'note': 'named by the caller; q8_0 is the type the 4B card measured as '
                              'clearing its 0.95 floor. An auto probe is deliberately absent: the '
                              'Q6_K probe read 0.9961 while the written file delivered 0.8815'},
        'strength': float(strength),
        'edited': {'count': len(rows),
                   'min_cos': min(cosines) if cosines else None,
                   'mean_cos': round(sum(cosines) / len(cosines), 4) if cosines else None,
                   'min_snr': min((row['snr'] for row in rows if row.get('snr') is not None),
                                  default=None),
                   'floor': floor, 'floor_met': floor_met,
                   'below_floor': [row['name'] for row in rows
                                   if (row.get('cos') or 0.0) < floor],
                   'by_kind': by_kind, 'tensors': rows},
        'untouched': {'count': untouched.get('checked'),
                      'verified': bool(untouched.get('verified')),
                      'mismatched': untouched.get('mismatched') or [],
                      'missing': untouched.get('missing') or [],
                      'method': 'byte-for-byte re-read of BOTH files after the write; the edited '
                                'tensors are excluded by construction'},
        'fingerprint': rewrite.get('fingerprint') or {},
        'bytes_in': rewrite.get('bytes_in'), 'bytes_out': rewrite.get('bytes_out'),
        'input_sha256': _sha256(src), 'output_sha256': _sha256(out),
        'wall_seconds': round(time.monotonic() - started, 2),
        'removal_measured': False,
        'gated': GATED,
        'claims': {
            'delivered_cos': 'codec fidelity of the stored edit against the intended projection, '
                             'computed on CPU from the two GGUF files — a fact about bytes, not '
                             'about answers',
            'untouched_bytes': 'verified by re-reading both files, not asserted',
            'determinism': 'same input bytes + same axis + same flags rewrite to the same output '
                           'sha256; the writer emits no timestamps or paths into the artifact',
            'removal': 'not measured and not claimed: every served claim is GATED behind a real '
                       'serve config and a judge wave',
        },
        'next': [f'jdewitt gguf-inspect {out}',
                 f'jdewitt gguf-tensor-diff {src} {out}  # only the named tensors may differ',
                 '# certify removal on the EDITED artifact with a real serve config; this run '
                 'measured codec fidelity only'],
    }
    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=1, sort_keys=False) + '\n')
    return report
