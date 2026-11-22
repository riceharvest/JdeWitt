# SPDX-License-Identifier: AGPL-3.0-or-later
"""Apply a digest-pinned plan card to THEIR quantized artifact.

`plan-bundle` makes a fitted plan distributable; this module is the consumer half
of that unit. Nothing here needs our records, our fit, a model load or the B70:
one bundle, one artifact, one JSON. The steps are the same ones `edit-quantized`
runs, in the same order:

  * verify the card (signed meta, payload sha256, basis rank) and install it into
    a throwaway registry (`registry.unbundle_plan`) — a tampered card never lands;
  * read the TARGET artifact's geometry and its residual-write tensors from the
    file itself (`gguf_weight_shapes` + `discover_writeback_targets`), never from
    a per-model table or a hand-typed tensor name;
  * refuse, before writing a byte, when the plan's rank/hidden cannot describe
    this artifact, when no target tensor matches the plan's layers, or when the
    requested edit type has no encoder in this build;
  * re-encode only the discovered target tensors through the same axis path as
    `edit-quantized` (`ggufio.axis_delta` inside `rewrite.rewrite_with_edits`),
    copying every other tensor byte for byte;
  * measure what shipped (`ggufio.compare_edit`) and print it.

The cos/SNR in the report are CODEC-realized fidelity numbers computed on CPU
from the two files: they say what the stored edit is, not what the model does.
Removal is not measured here and is never claimed.
"""
import json
import os
import shutil
import tempfile
from pathlib import Path

from .registry import BUNDLE_FORMAT, RegistryError, _verify_meta, load_plan, unbundle_plan
from .storage import file_digest

AUTO_TYPES = ('q6_k', 'q8_0', 'f16')


class PlanApplyRefused(ValueError):
    """A named, pre-write refusal.

    The message starts with the machine-readable reason, so an agent can branch on
    it without parsing prose; the CLI envelope carries the class name as ``type``.
    """

    def __init__(self, reason, detail):
        self.reason = str(reason)
        self.detail = str(detail)
        super().__init__(f'{self.reason}: {self.detail}')


def _artifact_geometry(path):
    """arch / block count / embedding width / tensor count from the artifact itself.

    The GGUF key/value block is the artifact's own statement about its shape; no
    architecture table is consulted, so an architecture this repo has never seen
    still yields a geometry or a loud refusal.
    """
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    fields = reader.fields
    arch = fields['general.architecture'].contents() if 'general.architecture' in fields else None

    def declared(suffix):
        for key, field in fields.items():
            if key == f'general.{suffix}' or key.endswith(f'.{suffix}'):
                return field.contents()
        return None

    layers, hidden = declared('block_count'), declared('embedding_length')
    missing = [name for name, value in (('block_count', layers), ('embedding_length', hidden))
               if value is None]
    if missing:
        raise PlanApplyRefused(
            'ARTIFACT_GEOMETRY_UNKNOWN',
            f'{path} declares no {" / ".join(missing)} key/value; a plan cannot be '
            f'checked against geometry the artifact does not state')
    assert layers is not None and hidden is not None  # narrowed by `missing` above
    return {'arch': None if arch is None else str(arch),
            'layers': int(layers), 'hidden': int(hidden), 'tensors': len(reader.tensors)}


def _stored_fingerprint(path):
    """The ``jdewitt.fingerprint`` kv a previous edit wrote into the artifact, if any."""
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


def _canonical_edit_type(edit_type):
    """Name the requested write type, or refuse with the list this build can emit."""
    from .ggufio import encodable_types
    from .quantize import canonical_type

    encodable = encodable_types()
    try:
        name = canonical_type(edit_type)
    except ValueError as error:
        raise PlanApplyRefused(
            'EDIT_TYPE_UNKNOWN',
            f'{error}; this build can ENCODE: {", ".join(encodable)}') from error
    if name not in encodable:
        raise PlanApplyRefused(
            'EDIT_TYPE_NOT_ENCODABLE',
            f'{edit_type!r} ({name}) has no encoder in this build; it can ENCODE: '
            f'{", ".join(encodable)}. Refusing instead of merging at 16-bit and '
            f'requantizing, which would ship an edit nobody measured')
    return name


def _select_edit_type(gguf, edits, probes, strength, edit_type, min_cos):
    """Pick the write type: the smallest candidate whose probe clears the cos floor.

    The probe runs the SAME codec the writer will use (`ggufio.preview_edit` with
    ``target_type``) over a deterministic spread of target tensors, so the choice
    costs seconds instead of one full rewrite per candidate. The floor only picks
    the type; the numbers in the report are measured afterwards on every edited
    tensor from the written file.
    """
    from .ggufio import preview_edit
    from .quantize import canonical_type

    if edit_type != 'auto':
        chosen = _canonical_edit_type(edit_type)
        return {'mode': 'explicit', 'floor': min_cos, 'chosen': chosen,
                'probes': [], 'readings': [], 'floor_cleared': None}
    candidates = [_canonical_edit_type(kind) for kind in AUTO_TYPES]
    probe_edits = {name: edits[name] for name in probes}
    readings, chosen, cleared = [], None, False
    for kind in candidates:
        reading = preview_edit(str(gguf), probe_edits, scale=strength, target_type=kind)
        readings.append({'edit_type': kind, 'min_cos': reading['min_cos'],
                         'mean_cos': reading['mean_cos'], 'min_snr': reading['min_snr'],
                         'probed': reading['sampled']})
        if (reading['min_cos'] or 0.0) >= min_cos:
            chosen, cleared = kind, True
            break  # the ladder is smallest-first; the first clear IS the choice
    if chosen is None:
        # F16 stores the edit exactly; it is the honest last resort and it is
        # reported as a floor miss rather than passed off as a cleared probe.
        chosen = canonical_type(AUTO_TYPES[-1])
    return {'mode': 'auto', 'floor': min_cos, 'chosen': chosen, 'probes': probes,
            'readings': readings, 'floor_cleared': cleared,
            'note': 'candidates are probed smallest-first and probing stops at the first '
                    'clear; the delivered numbers below are measured on the written file'}


def plan_apply(bundle, gguf, out, strength=1.0, edit_type='auto', min_cos=0.95,
               axis_space='output'):
    """Reproduce a plan card's edit on ``gguf`` and write ``out``; returns one report.

    Every refusal happens before the writer opens ``out``, so a failed run never
    leaves a half-edited artifact behind under the requested name.
    """
    import numpy as np

    from .gguf_adapters import discover_writeback_targets, gguf_weight_shapes
    from .ggufio import compare_edit
    from .rewrite import axis_fingerprint, rewrite_with_edits

    bundle, gguf, out = Path(bundle), Path(gguf), Path(out)
    if not bundle.is_file():
        raise FileNotFoundError(str(bundle))
    if not gguf.is_file():
        raise FileNotFoundError(str(gguf))
    if gguf.resolve() == out.resolve():
        raise PlanApplyRefused(
            'OUT_IS_INPUT', f'--out {out} is the input artifact; plan-apply always '
                            f'writes a new file and never edits the file it read')
    if not out.parent.is_dir():
        raise PlanApplyRefused('OUT_DIRECTORY_MISSING', f'{out.parent} is not a directory')
    if axis_space not in ('output', 'input'):
        raise PlanApplyRefused('AXIS_SPACE_UNKNOWN', f'axis space must be output or input, not {axis_space!r}')

    geometry = _artifact_geometry(gguf)
    bundle_sha = file_digest(bundle)

    with tempfile.TemporaryDirectory(prefix='jdewitt-plan-') as registry:
        try:
            installed = unbundle_plan(bundle, registry)
        except RegistryError as error:
            # Digest/format/rank problems all arrive named from the registry; wrap
            # them so the consumer sees one stable reason for "the card is bad".
            raise PlanApplyRefused('BUNDLE_REFUSED', str(error)) from error
        key = installed['key']
        try:
            meta = _verify_meta(registry, key)
        except RegistryError as error:
            raise PlanApplyRefused('BUNDLE_REFUSED', str(error)) from error
        plan_geometry = installed.get('geometry') or {}
        if not plan_geometry:
            raise PlanApplyRefused(
                'PLAN_HAS_NO_3D_BASIS',
                f'plan {key} carries no [layers,rank,hidden] basis; plan-apply edits '
                f'from a 3D basis only (method={meta.get("method")!r})')
        plan_layers = int(plan_geometry['layers'])
        plan_rank = int(plan_geometry['rank'])
        plan_hidden = int(plan_geometry['hidden'])
        if meta.get('rank') is not None and int(meta['rank']) != plan_rank:
            raise PlanApplyRefused(
                'PLAN_RANK_MISMATCH',
                f'plan {key} basis rank {plan_rank} disagrees with signed meta rank {meta["rank"]}')
        mismatches = []
        if plan_hidden != geometry['hidden']:
            mismatches.append(f'basis hidden {plan_hidden} != artifact {geometry["hidden"]}')
        if plan_layers > geometry['layers']:
            mismatches.append(f'basis layers {plan_layers} > artifact {geometry["layers"]}')
        # The signed card also records the geometry it was FITTED for
        # (generation_fingerprint: layers / hidden_size / rank, written by the fit).
        # Those are the same facts the artifact states about itself, so a card that
        # disagrees with either side cannot be reproduced here.
        declared = meta.get('generation_fingerprint') or {}
        if isinstance(declared, dict):
            if declared.get('hidden_size') is not None and int(declared['hidden_size']) != geometry['hidden']:
                mismatches.append(f'card hidden_size {declared["hidden_size"]} != artifact {geometry["hidden"]}')
            if declared.get('layers') is not None and int(declared['layers']) != geometry['layers']:
                mismatches.append(f'card layers {declared["layers"]} != artifact {geometry["layers"]}')
            if declared.get('rank') is not None and int(declared['rank']) != plan_rank:
                mismatches.append(f'card rank {declared["rank"]} != basis rank {plan_rank}')
        if mismatches:
            raise PlanApplyRefused(
                'GEOMETRY_MISMATCH',
                f'plan {key} (rank={plan_rank}, layers={plan_layers}, hidden={plan_hidden}) '
                f'does not describe {gguf} (arch={geometry["arch"]}, layers={geometry["layers"]}, '
                f'hidden={geometry["hidden"]}): {" ; ".join(mismatches)}; nothing written')
        # load_plan re-verifies the signed meta and payload sha every time and
        # checks the payload's own geometry against the expectation derived here.
        loaded = load_plan(registry, key, expect={'rank': plan_rank, 'hidden': plan_hidden})
        if loaded['geometry'] != plan_geometry:
            raise PlanApplyRefused(
                'PLAN_GEOMETRY_DRIFT',
                f'plan {key} loaded with geometry {loaded["geometry"]} but the bundle '
                f'declared {plan_geometry}')
        basis_tensor = loaded['plan'].get('basis') if isinstance(loaded['plan'], dict) else None
        if basis_tensor is None or not hasattr(basis_tensor, 'detach'):
            raise PlanApplyRefused('PLAN_HAS_NO_3D_BASIS', f'plan {key} payload has no basis tensor')
        basis = basis_tensor.detach().float().cpu().numpy().astype(np.float32)
        if basis.ndim != 3:
            raise PlanApplyRefused('PLAN_HAS_NO_3D_BASIS', f'plan {key} basis has shape {basis.shape}, not [layers,rank,hidden]')

        # --- targets, discovered from the artifact ---------------------------------
        shapes = gguf_weight_shapes(str(gguf))
        discovered = discover_writeback_targets(shapes, plan_hidden)
        targets = discovered['targets']
        covered = [layer for layer in range(plan_layers) if layer in targets]
        if not covered:
            raise PlanApplyRefused(
                'NO_TARGET_TENSORS',
                f'plan {key} covers layers 0..{plan_layers - 1} at width {plan_hidden}, but '
                f'{gguf} exposes no residual-write projection at that width '
                f'(layers with targets: {sorted(targets)[:4]}'
                f'{"..." if len(targets) > 4 else ""}); nothing written')
        edits = {}
        for layer in covered:
            for name in targets[layer]:
                edits[name] = (basis[layer], axis_space)
        probes = []
        for position in sorted({0, len(covered) // 2, len(covered) - 1}):
            for name in targets[covered[position]]:
                if name not in probes:
                    probes.append(name)
        probes = probes[:4]

        selection = _select_edit_type(gguf, edits, probes, strength, edit_type, min_cos)
        chosen = selection['chosen']

        # --- already applied? the artifact carries its own edit fingerprint --------
        fingerprint = axis_fingerprint(edits)
        stored = _stored_fingerprint(gguf)
        already = False
        if isinstance(stored, dict):
            already = (stored.get('edit_sha256') == fingerprint['edit_sha256']
                       and stored.get('scale') == strength
                       and stored.get('axis_space') == axis_space
                       and sorted(stored.get('layers') or []) == fingerprint['layers'])
        if already:
            # Applying the same card twice in exact arithmetic is the identity (the
            # direction is gone); the honest, deterministic behaviour is a byte copy
            # with the stored fingerprint reported, not a second quantization pass.
            temp_path = out.with_name(f'{out.name}.partial-{os.getpid()}')
            try:
                shutil.copyfile(gguf, temp_path)
                os.replace(temp_path, out)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
            return {
                'command': 'plan-apply',
                'applied': False, 'already_applied': True,
                'plan': _plan_block(key, meta, bundle, bundle_sha, installed),
                'artifact': _artifact_block(gguf, out, geometry),
                'plan_geometry': plan_geometry, 'artifact_geometry': geometry,
                'strength': strength, 'axis_space': axis_space,
                'edit_type': stored.get('edit_type'),
                'edit_type_by_tensor': {}, 'edit_type_selection': selection,
                'targets': _targets_block(discovered, covered, edits),
                'edited': {'count': 0, 'tensors': []},
                'untouched': {'count': geometry['tensors'], 'verified': True,
                              'note': 'no rewrite: the output is a byte copy of the input, '
                                      'so every tensor is identical by construction'},
                'delivered': None,
                'fingerprint': stored,
                'bytes_in': gguf.stat().st_size, 'bytes_out': out.stat().st_size,
                'determinism': {'input_sha256': file_digest(gguf),
                                'output_sha256': file_digest(out),
                                'byte_identical_to_input': True,
                                'note': 're-applying the card the artifact already carries is a no-op; '
                                        'use the pristine base to measure the delivered edit'},
                'next': _next_commands(gguf, out),
                'claims': _claims(),
            }

        # --- write minimal surgery, then measure what shipped ----------------------
        temp_path = out.with_name(f'{out.name}.partial-{os.getpid()}')
        try:
            rewrite = rewrite_with_edits(str(gguf), str(temp_path), edits, scale=strength,
                                         axis_space=axis_space, edit_type=chosen,
                                         verify_untouched=True)
            os.replace(temp_path, out)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        delivered = compare_edit(str(gguf), str(out), edits, scale=strength)
        rows = []
        for row in delivered['tensors']:
            layer = int(row['name'].split('.')[1]) if row.get('name') else None
            rows.append({'name': row.get('name'), 'layer': layer,
                         'cos': row.get('cos'), 'snr': row.get('snr'),
                         'norm_ratio': row.get('norm_ratio'),
                         'source_type': row.get('source_type'),
                         'edited_type': row.get('edited_type')})
        untouched = rewrite.get('untouched') or {}
        report = {
            'command': 'plan-apply',
            'applied': True, 'already_applied': False,
            'plan': _plan_block(key, meta, bundle, bundle_sha, installed),
            'artifact': _artifact_block(gguf, out, geometry),
            'plan_geometry': plan_geometry, 'artifact_geometry': geometry,
            'strength': strength, 'axis_space': axis_space,
            'edit_type': chosen,
            'edit_type_by_tensor': {row['name']: row['edited_type'] for row in rows},
            'edit_type_selection': selection,
            'targets': _targets_block(discovered, covered, edits),
            'edited': {'count': len(rows), 'tensors': rows},
            'untouched': {
                'count': untouched.get('checked'),
                'verified': bool(untouched.get('verified')),
                'mismatched': untouched.get('mismatched') or [],
                'missing': untouched.get('missing') or [],
                'note': 're-read byte for byte from both files after the write; the '
                        'edited tensors are excluded by construction',
            },
            'delivered': {'min_cos': delivered.get('min_cos'),
                          'mean_cos': delivered.get('mean_cos'),
                          'min_snr': min((row['snr'] for row in rows if row.get('snr') is not None),
                                         default=None),
                          'below_floor': [row['name'] for row in rows
                                          if (row.get('cos') or 0.0) < min_cos],
                          'floor': min_cos},
            'fingerprint': rewrite.get('fingerprint'),
            'bytes_in': rewrite.get('bytes_in'), 'bytes_out': rewrite.get('bytes_out'),
            'elapsed_s': rewrite.get('elapsed_s'),
            'determinism': {
                'input_sha256': file_digest(gguf),
                'output_sha256': file_digest(out),
                'note': 'same bundle + same input bytes + same flags rewrite to the same '
                        'output sha256; the writer emits no timestamps or paths',
            },
            'next': _next_commands(gguf, out),
            'claims': _claims(),
        }
        return report


def _plan_block(key, meta, bundle, bundle_sha, installed):
    return {'key': key, 'meta_digest': meta.get('meta_digest'),
            'payload_sha256': meta.get('payload_sha256'),
            'method': meta.get('method'), 'rank': meta.get('rank'),
            'measured': meta.get('measured'), 'model': meta.get('model'),
            'revision': meta.get('revision'), 'registered_at': meta.get('registered_at'),
            'bundle': str(bundle), 'bundle_format': BUNDLE_FORMAT,
            'bundle_sha256': bundle_sha, 'installed_into_temp_registry': installed.get('installed')}


def _artifact_block(gguf, out, geometry):
    return {'input': str(gguf), 'output': str(out), 'arch': geometry['arch'],
            'tensors': geometry['tensors']}


def _targets_block(discovered, covered, edits):
    return {'layers': covered, 'layer_count': len(covered), 'tensor_count': len(edits),
            'names': sorted(edits),
            'unmapped_writeback_width': discovered.get('unmapped_writeback_width') or [],
            'discovery': 'residual-write projections at the plan width, read from the '
                         'artifact tensor table (names + width), no per-model table'}


def _next_commands(gguf, out):
    return [f'jdewitt gguf-tensor-diff {gguf} {out}  # only the named tensors may differ',
            f'jdewitt verify --pristine {gguf} --edited {out}  # recompute the edit fingerprint',
            '# certify the EDITED artifact with a real serve config: removal is GATED, '
            'this command only measured codec fidelity']


def _claims():
    return {'removal': 'not measured and not claimed; behaviour is GATED behind a real '
                       'serve config and a judge wave',
            'delivered_cos': 'codec-realized fidelity of the stored edit, computed on CPU '
                             'from the two GGUF files (a fact about bytes, not about answers)',
            'untouched_bytes': 'verified by re-reading both files, not asserted'}
