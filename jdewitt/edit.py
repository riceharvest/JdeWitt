# SPDX-License-Identifier: AGPL-3.0-or-later
"""One verb for the artifact you already serve: decide, gate, then write.

`jdewitt edit` exists because the tool had more apply paths than an agent can
choose between from a cold read (`edit-quantized`, `bank-apply`, `plan-apply`,
`gguf-edit`, `spanfit build --in-place`, the vocabulary family) and they did not
share one rule for the thing that matters: whether the QUANTIZED grid can store
the edit at all. A sampled probe once admitted a Q6_K bank edit that landed at
whole-tensor cos 0.8808 under a 0.95 floor on a real run. So:

  * the plan names every method the operands make possible, why each is or is
    not ready, the exact next command, and the measured preview fidelity — one
    JSON, no model, no tokenizer, no server, no generated token;
  * `--apply` performs the selected path through the same write gate as every
    other writer (`rewrite.write_with_floor`): a candidate per storage type,
    read back over the FULL edited footprint, smallest type that clears the
    floor, and a typed CODEC_FLOOR refusal when none does;
  * nothing is written in plan mode, and `--apply` without `--out` or
    `--in-place` refuses rather than guessing a destination.

Method priority is deliberate. Records fitted on the model being edited beat a
banked axis (measured on a base, applied to a fine-tune), which beats an
explicit delta the caller produced elsewhere; the vocabulary/grammar fallbacks
are always listed but never chosen over an artifact edit, and they carry
``removal_measured: false`` because they are static byte/automata work.
"""
from __future__ import annotations

import os
import shlex
import time
from pathlib import Path

SCHEMA = 'jdewitt-edit-plan-v1'
LADDER = ('q6_k', 'q8_0', 'f16')
METHOD_ORDER = ('fitted-axis', 'imported-axis', 'banked-axis', 'explicit-delta')


def _out_path(args, artifact_path):
    if args.out is not None:
        return Path(args.out)
    if args.in_place:
        return Path(artifact_path)
    stem = Path(artifact_path)
    return stem.with_name(f'{stem.stem}-edited{stem.suffix}')


def _artifact_view(gguf):
    """Header-only facts an agent (and a refusal) can act on."""
    from gguf import GGUFReader
    from .gguf_adapters import discover_writeback_targets, gguf_weight_shapes, inferred_hidden_width

    reader = GGUFReader(str(gguf))
    field = reader.fields.get('general.architecture')
    architecture = None if field is None else field.contents()
    shapes = gguf_weight_shapes(gguf, reader=reader)
    hidden = inferred_hidden_width(shapes)
    discovered = (discover_writeback_targets(shapes, hidden) if hidden is not None
                  else {'targets': {}, 'unmapped_writeback_width': []})
    layers = sorted(discovered['targets'])
    return {
        'path': str(gguf),
        'architecture': architecture,
        'layers': (layers[-1] + 1) if layers else 0,
        'hidden': hidden,
        'tensors': len(reader.tensors),
        'source_types': sorted({tensor.tensor_type.name for tensor in reader.tensors}),
        'bytes': Path(gguf).stat().st_size,
        'writeback_layers': len(layers),
        'unmapped_writeback_width': discovered['unmapped_writeback_width'][:8],
    }


def _probe_names(edits, per_kind=2):
    """Ends of the footprint per writeback kind: cheap, and the ends carry the diff."""
    grouped = {}
    for name in sorted(edits):
        suffix = name.rsplit('.', 2)[1] if name.count('.') >= 2 else name
        grouped.setdefault(suffix, []).append(name)
    probes = []
    for suffix in sorted(grouped):
        names = grouped[suffix]
        for name in (names[0], names[-1]):
            if name not in probes:
                probes.append(name)
            if len(probes) >= per_kind * max(1, len(grouped)):
                break
    return probes[:8]


def _base_command(args, artifact):
    return ['python', '-m', 'jdewitt.cli', 'edit', '--gguf', str(artifact['path'])]


def _fitted_method(args, artifact, require_floor=True):
    """A fitted axis from capture records: the primary method when records exist.

    ``require_floor`` (default) is the one-shot contract: a method whose sampled
    preview cannot clear the floor is not offered, because `edit` would write
    nothing. The stepped route passes False: the guess ranks, and the GATE is the
    verdict, so the planner must still hand it a candidate to measure.
    """
    from .spanfit import load_records, per_layer_axes, per_layer_subspace, shared_axis

    method = {'method': 'fitted-axis', 'ready': False, 'reason': None, 'detail': None,
              'source': str(args.records), 'command': None, 'preview': None}
    try:
        records = load_records(str(args.records))
        if args.shape == 'shared':
            axes = shared_axis(records)[0]
        else:
            axes = per_layer_axes(records)
        if int(args.rank) > 1:
            axes = per_layer_subspace(records, int(args.rank))
    except Exception as error:  # noqa: BLE001 - every records failure is a reportable plan state
        method['reason'] = 'RECORDS_UNREADABLE'
        method['detail'] = f'{type(error).__name__}: {error}'
        return method, None
    return _axes_edits(args, artifact, axes, method=method,
                       source_flags=['--records', str(args.records), '--shape', args.shape,
                                     '--rank', str(args.rank)],
                       require_floor=require_floor)


def _axes_method(args, artifact, require_floor=True):
    """An axis imported from a file: an externally recovered direction.

    The file is ``torch.save({'axes': tensor[, 'provenance': {...}]})`` with a
    ``[D]`` shared axis or a ``[L, D]`` per-layer matrix. The route and `edit`
    treat it exactly like a fitted axis — same discovery, same full-footprint
    gate — because the provenance of a direction changes how much you trust it,
    never whether the codec can store it.
    """
    import torch

    method = {'method': 'imported-axis', 'ready': False, 'reason': None, 'detail': None,
              'source': str(args.axes), 'command': None, 'preview': None}
    try:
        payload = torch.load(str(args.axes), map_location='cpu', weights_only=False)
        axes = payload.get(args.axes_key, payload.get('axes', payload)) if isinstance(payload, dict) else payload
    except Exception as error:  # noqa: BLE001 - every axes-file failure is a plan state
        method['reason'] = 'AXES_UNREADABLE'
        method['detail'] = f'{type(error).__name__}: {error}'
        return method, None
    if not torch.is_tensor(axes) or axes.ndim not in (1, 2):
        method['reason'] = 'AXES_UNREADABLE'
        method['detail'] = (f'{args.axes} does not carry a [D] or [L, D] tensor under '
                            f'{args.axes_key!r}; keys: '
                            f'{sorted(payload) if isinstance(payload, dict) else type(payload).__name__}')
        return method, None
    axes = axes.detach().float()
    if not bool(torch.isfinite(axes).all()):
        method['reason'] = 'AXES_UNREADABLE'
        method['detail'] = 'the imported axis contains NaN/Inf'
        return method, None
    provenance = payload.get('provenance') if isinstance(payload, dict) else None
    if provenance is not None:
        method['provenance'] = provenance
    kinds = payload.get('kinds') if isinstance(payload, dict) else None
    if kinds:
        method['kinds'] = list(kinds)
    return _axes_edits(args, artifact, axes, method=method,
                       source_flags=['--axes', str(args.axes), '--axes-key', args.axes_key],
                       kinds=kinds, require_floor=require_floor)


def _axes_edits(args, artifact, axes, *, method, source_flags, kinds=None,
                require_floor=True):
    """Discovery, edit map, codec preview and the exact apply command for one axis tensor."""
    import torch

    from .gguf_adapters import discover_writeback_targets, gguf_weight_shapes
    from .ggufio import preview_edit
    from .ggufwrite import parse_container, tensor_nbytes
    from .spanfit import parse_layer_ids

    if axes.ndim == 1:
        width = int(axes.shape[0])
        shapes = gguf_weight_shapes(str(args.gguf))
        discovered = discover_writeback_targets(shapes, width)
        layers = sorted(discovered['targets'])
        if layers:
            expanded = torch.zeros((max(layers) + 1, width), dtype=torch.float32)
            expanded[layers] = axes
            axes = expanded
            method['broadcast'] = {'from': '[D]', 'to': f'[{int(axes.shape[0])},{width}]',
                                   'layers': layers[:8], 'layers_total': len(layers)}
    width = int(axes.shape[-1])
    shapes = gguf_weight_shapes(str(args.gguf))
    discovered = discover_writeback_targets(shapes, width)
    layers = sorted(discovered['targets'])
    wanted = parse_layer_ids(args.layers) if args.layers else None
    if wanted is not None:
        layers = [layer for layer in layers if layer in set(wanted)]
    method['footprint'] = ({'first': layers[0], 'last': layers[-1], 'layers': len(layers)}
                           if layers else None)
    if not layers:
        method['reason'] = 'NO_VERIFIED_ADAPTER'
        method['detail'] = (f'the artifact carries no residual-write projection at width {width} '
                            f'(its own writeback names put hidden at {artifact["hidden"]}); the '
                            f'axis and the artifact have to belong to the same model')
        return method, None
    edits, edited_bytes = {}, 0
    allowed_kinds = set(kinds) if kinds else None
    header = parse_container(str(args.gguf))
    by_name = {tensor['name']: tensor for tensor in header['tensors']}
    for layer in layers:
        if layer >= int(axes.shape[0]):
            continue
        if float(axes[layer].norm()) == 0.0:
            # An imported axis is usually sparse (one writer, one layer). A zero row
            # means "do not touch this layer", exactly as spanfit's merge treats it;
            # editing it with a zero projection would requantize the tensor for no edit
            # and then grade the codec on an edit nobody asked for.
            continue
        for name in discovered['targets'][layer]:
            if allowed_kinds is not None and name.split('.')[-2] not in allowed_kinds:
                continue
            edits[name] = (axes[layer].numpy(), args.axis_space)
            tensor = by_name.get(name)
            if tensor is not None:
                edited_bytes += tensor_nbytes(tensor['dims'], tensor['qtype'])
    if not edits:
        method['reason'] = 'NO_VERIFIED_ADAPTER'
        method['detail'] = (f'the footprint selects no tensor this artifact carries: {len(layers)} '
                            f'discovered layer(s) survive the filter, the axis has '
                            f'{int(axes.shape[0])} rows')
        return method, None
    probes = _probe_names(edits)
    ladder = LADDER if args.edit_type == 'auto' else (args.edit_type,)
    readings = []
    for kind in ladder:
        reading = preview_edit(str(args.gguf), {name: edits[name] for name in probes},
                               names=probes, scale=args.scale,
                               renormalize=getattr(args, 'renormalize', False) or args.repair != 'none',
                               target_type=kind)
        rows = reading.get('tensors') or []
        readings.append({'edit_type': kind, 'min_cos': reading.get('min_cos'),
                         'mean_cos': reading.get('mean_cos'), 'probed': len(rows),
                         'worst_tensor': min(rows, key=lambda row: row.get('cos') or 0.0)['name']
                         if rows else None})
        if reading.get('min_cos') is not None and float(reading['min_cos']) >= float(args.min_cos):
            break
    method['preview'] = {'floor': float(args.min_cos), 'probes': probes, 'readings': readings,
                         'measured': 'real codec on sampled windows, before any write'}
    cleared = next((row for row in readings
                    if row['min_cos'] is not None and float(row['min_cos']) >= float(args.min_cos)), None)
    if cleared is None:
        worst = max(readings, key=lambda row: row.get('min_cos') or 0.0) if readings else None
        method['reason'] = 'CODEC_FLOOR'
        method['detail'] = (f'no probed type stores the edit at cos >= {args.min_cos}: '
                            f'{worst["edit_type"]} reaches {worst["min_cos"]} on '
                            f'{worst["worst_tensor"]}' if worst else 'no tensor could be probed')
        if require_floor:
            return method, None
        cleared = worst
    if cleared is None:
        method['reason'] = 'NO_READINGS'
        method['detail'] = 'no tensor could be probed on this artifact'
        return method, None
    method.update({
        'ready': True,
        'floor_cleared': float(cleared.get('min_cos') or 0.0) >= float(args.min_cos),
        'edit_type': cleared['edit_type'],
        'expected_min_cos': cleared['min_cos'],
        'cost': {'edited_tensors': len(edits), 'edited_bytes': int(edited_bytes)},
        'ladder': [row['edit_type'] for row in readings],
    })
    command = _base_command(args, artifact) + source_flags + [
        '--scale', str(args.scale), '--axis-space', args.axis_space,
        '--min-cos', str(args.min_cos), '--edit-type', args.edit_type]
    if args.layers:
        command += ['--layers', args.layers]
    if args.in_place:
        command.append('--in-place')
    else:
        command += ['--out', str(_out_path(args, artifact['path']))]
    method['command'] = shlex.join(command + ['--apply'])
    return method, edits


def _bank_method(args, artifact):
    """A digest-pinned banked axis: the zero-fit path for an agent holding only the GGUF."""
    from .bank import resolve
    from .errors import BankRefused
    from .registry import RegistryError, _verify_meta, load_plan, unbundle_plan

    method = {'method': 'banked-axis', 'ready': False, 'reason': None, 'detail': None,
              'source': str(args.bank), 'command': None}
    try:
        resolved = resolve(args.bank, args.gguf, key=args.entry)
    except BankRefused as error:
        method['reason'] = error.code
        method['detail'] = str(error)
        return method, None
    # The digest check is not an install check: a bundle can hash to its pin and
    # still carry members the loader refuses (a real fixture does). Plan readiness
    # has to mean "apply will run", so the same unbundle+load runs here, in a
    # throwaway registry, before the method is called ready.
    import tempfile

    bundle_path = Path(args.bank) / resolved['bundle']['file']
    try:
        with tempfile.TemporaryDirectory(prefix='jdewitt-edit-plan-') as registry_dir:
            installed = unbundle_plan(bundle_path, registry_dir)
            _verify_meta(registry_dir, installed['key'])
            load_plan(registry_dir, installed['key'],
                      expect={'hidden': int(resolved['geometry']['hidden'])})
    except (RegistryError, OSError) as error:
        method['reason'] = 'BUNDLE_REFUSED'
        method['detail'] = (f'entry {resolved["entry"]["key"]!r} pins '
                            f'{resolved["bundle"]["file"]!r} but that bundle does not install: '
                            f'{error}')
        return method, None
    targets = resolved.get('targets') or {}
    method.update({
        'ready': True,
        'entry': resolved['entry']['key'],
        'base': resolved['entry'].get('base'),
        'measurement': resolved.get('measurement'),
        'footprint': resolved.get('footprint'),
        'strength': resolved.get('strength'),
        'edit_types': (resolved.get('edit') or {}).get('types'),
        'bundle_verified': True,
        'cost': {'edited_tensors': sum(len(names) for names in targets.values()),
                 'edited_bytes': None},
        'removal_measured': (resolved.get('measurement') or {}).get('state') != 'banked_unmeasured',
    })
    command = _base_command(args, artifact) + ['--bank', str(args.bank),
                                               '--entry', method['entry']]
    if args.in_place:
        command.append('--in-place')
    else:
        command += ['--out', str(_out_path(args, artifact['path']))]
    method['command'] = shlex.join(command + ['--apply'])
    return method, resolved


def _delta_method(args, artifact):
    """An explicit delta the caller produced elsewhere (gguf-delta-pack / twin transplant)."""
    from gguf import GGUFReader

    method = {'method': 'explicit-delta', 'ready': False, 'reason': None, 'detail': None,
              'source': str(args.delta_gguf), 'command': None}
    path = Path(args.delta_gguf)
    if not path.is_file():
        method['reason'] = 'FILE_NOT_FOUND'
        method['detail'] = f'{path} does not exist'
        return method, None
    reader = GGUFReader(str(path))
    names = [tensor.name for tensor in reader.tensors]
    if not names:
        method['reason'] = 'INPUT_ERROR'
        method['detail'] = f'{path} carries no tensors'
        return method, None
    method.update({'ready': True, 'tensors': len(names),
                   'cost': {'edited_tensors': len(names), 'edited_bytes': None}})
    command = _base_command(args, artifact) + ['--delta-gguf', str(path),
                                               '--strength', str(args.strength)]
    if args.in_place:
        command.append('--in-place')
    else:
        command += ['--out', str(_out_path(args, artifact['path']))]
    method['command'] = shlex.join(command + ['--apply'])
    return method, None


def _vocabulary_method(args, artifact):
    """The static fallbacks: real edits, but no behaviour is measured on this arm."""
    out = _out_path(args, artifact['path'])
    return {
        'method': 'vocabulary',
        'ready': True,
        'reason': None,
        'detail': None,
        'rank': 'fallback',
        'removal_measured': False,
        'audit_note': ('static byte/automata work: the tool measures the tokenizer, row and '
                       'byte facts; it does not measure answers, so "removed" is GATED behind '
                       'a serve that this command never starts'),
        'commands': [
            shlex.join(['python', '-m', 'jdewitt.cli', 'vocab-centroid', '--weights',
                        str(artifact['path']), '--out', str(out), '--mode',
                        'centroid-orthogonal', '--apply']),
            shlex.join(['python', '-m', 'jdewitt.cli', 'vocab-delete', '--weights',
                        str(artifact['path']), '--out', str(out), '--apply']),
            shlex.join(['python', '-m', 'jdewitt.cli', 'grammar', '--compile', '--artifact',
                        str(artifact['path']), '--out',
                        str(Path('verification/denial-grammar'))]),
        ],
    }


def plan_edit(args):
    """The plan dict and the private state needed to apply the choice."""
    started = time.monotonic()
    artifact = _artifact_view(args.gguf)
    methods, state = [], {'edits': None, 'resolved': None}
    if args.records is not None and args.axes is not None:
        raise ValueError('--records and --axes are alternatives: records fit an axis, axes import one')
    if args.records is not None:
        method, edits = _fitted_method(args, artifact)
        methods.append(method)
        state['edits'] = edits
    if args.axes is not None:
        method, edits = _axes_method(args, artifact)
        methods.append(method)
        state['edits'] = edits
    if args.bank is not None:
        method, resolved = _bank_method(args, artifact)
        methods.append(method)
        state['resolved'] = resolved
    if args.delta_gguf is not None:
        method, _ = _delta_method(args, artifact)
        methods.append(method)
    methods.append(_vocabulary_method(args, artifact))
    chosen = next((method for name in METHOD_ORDER for method in methods
                   if method['method'] == name and method['ready']), None)
    chosen_method = chosen['method'] if chosen else 'vocabulary'
    cost = (chosen or methods[-1]).get('cost') if chosen else None
    report = {
        'schema': SCHEMA,
        'command': 'edit',
        'mode': 'apply' if args.apply else 'plan',
        'artifact': artifact,
        'inputs': {'records': None if args.records is None else str(args.records),
                   'axes': None if args.axes is None else str(args.axes),
                   'bank': None if args.bank is None else str(args.bank),
                   'entry': args.entry,
                   'delta_gguf': None if args.delta_gguf is None else str(args.delta_gguf)},
        'gate': {'floor': float(args.min_cos), 'ladder': list(LADDER),
                 'rule': 'write a candidate per storage type, read the FULL edited footprint '
                         'back, keep the smallest type whose worst tensor clears the floor; '
                         'refuse CODEC_FLOOR when none does'},
        'methods': methods,
        'chosen': chosen_method,
        'cost': cost,
        'output': str(_out_path(args, artifact['path'])),
        'notes': ['no model, tokenizer, server or generated token: the preview runs the real '
                  'codec on sampled windows, the write gate reads every edited tensor back'],
        'elapsed_s': round(time.monotonic() - started, 2),
    }
    if chosen is None:
        report['next'] = methods[-1]['commands'][0]
    else:
        report['next'] = chosen['command']
    return report, state


def apply_in_place(gguf, temp, report):
    """Ship a layout-moving edit to the artifact's own path, atomically and reversibly.

    The original bytes go to a whole-file revert sidecar BEFORE the rename, so a
    crash between the two leaves the original untouched and a completed write is
    still undoable (`gguf-revert ART SIDECAR`).
    """
    from .ggufio import save_whole_file_revert

    sidecar = str(gguf) + '.revert.npz'
    saved = save_whole_file_revert(gguf, temp, sidecar)
    os.replace(temp, gguf)
    report['in_place'] = True
    report['output'] = str(gguf)
    report['revert_sidecar'] = sidecar
    report['revert_sidecar_bytes'] = saved['bytes']
    report['original_sha256'] = saved['original_sha256']
    report['patched_sha256'] = saved['patched_sha256']
    return report


def _apply_fitted(args, state):
    from .rewrite import write_with_floor

    edits = state.get('edits')
    if not edits:
        raise ValueError('the fitted method produced no edit map')
    target = Path(str(args.gguf) + '.edit-tmp') if args.in_place else Path(args.out)
    ladder = LADDER if args.edit_type == 'auto' else (args.edit_type,)
    report = write_with_floor(str(args.gguf), str(target), edits, ladder=ladder,
                              floor=float(args.min_cos), scale=args.scale,
                              axis_space=args.axis_space, repair=args.repair,
                              records=str(args.records))
    return apply_in_place(Path(args.gguf), target, report) if args.in_place else report


def _apply_bank(args, state):
    from .bank_apply import bank_apply

    resolved = state.get('resolved') or {}
    entry = args.entry or (resolved.get('entry') or {}).get('key')
    if entry is None:
        raise ValueError('the bank method resolved no entry key; pass --entry')
    target = Path(str(args.gguf) + '.edit-tmp') if args.in_place else Path(args.out)
    report = bank_apply(args.bank, entry, args.gguf, target,
                        strength_scale=args.strength_scale)
    return apply_in_place(Path(args.gguf), target, report) if args.in_place else report


def _apply_delta(args, state):
    from .agent import gguf_edit

    target = Path(str(args.gguf) + '.edit-tmp') if args.in_place else Path(args.out)
    report = gguf_edit(args.gguf, target, [], args.delta_gguf, args.strength)
    return apply_in_place(Path(args.gguf), target, report) if args.in_place else report


def edit_command(args):
    """CLI handler: plan (default) or plan+apply; always one JSON."""
    if args.apply and args.out is None and not args.in_place:
        raise ValueError('--apply writes an artifact: pass --out PATH, or --in-place to replace '
                         'the input (plan mode writes nothing and needs no destination)')
    if args.out is not None and args.in_place:
        raise ValueError('--out and --in-place are alternatives; pick the destination once')
    if args.out is not None and Path(args.out).resolve() == Path(args.gguf).resolve():
        raise ValueError('--out is the input artifact; pass --in-place for that, or a new path')
    report, state = plan_edit(args)
    if not args.apply:
        return report
    if report['chosen'] == 'fitted-axis' or report['chosen'] == 'imported-axis':
        applied = _apply_fitted(args, state)
    elif report['chosen'] == 'banked-axis':
        applied = _apply_bank(args, state)
    elif report['chosen'] == 'explicit-delta':
        applied = _apply_delta(args, state)
    else:
        floors = [method for method in report['methods']
                  if method.get('reason') == 'CODEC_FLOOR']
        if floors:
            from .errors import EditRefused
            raise EditRefused('CODEC_FLOOR',
                              '; '.join(method.get('detail') or 'codec floor' for method in floors))
        raise ValueError('no artifact edit is ready for this file; the plan lists the static '
                         'vocabulary fallbacks and why each artifact path refused')
    report['applied'] = applied
    if args.in_place:
        sidecar = applied.get('revert_sidecar')
        report['next'] = [
            shlex.join(['python', '-m', 'jdewitt.cli', 'gguf-inspect', str(args.gguf)]),
            shlex.join(['python', '-m', 'jdewitt.cli', 'board', '--artifact', str(args.gguf),
                        '--records', str(args.records or '<records.pt from capture>')]),
        ] + ([shlex.join(['python', '-m', 'jdewitt.cli', 'gguf-revert', str(args.gguf),
                          str(sidecar)])] if sidecar else [])
    else:
        from .rewrite import edit_next_steps

        report['next'] = edit_next_steps(str(args.gguf), str(report.get('output')),
                                         str(args.records) if args.records else None)
        if applied.get('revert_sidecar'):
            report['next'].append(shlex.join(['python', '-m', 'jdewitt.cli', 'gguf-revert',
                                              str(report.get('output')),
                                              str(applied['revert_sidecar'])]))
    report['elapsed_s'] = round(float(report.get('elapsed_s') or 0.0)
                                + float(applied.get('elapsed_s') or 0.0), 2)
    return report


# Public seams for the stepped route: the route reuses the one-shot planner's
# discovery, bank pre-flight and vocabulary fallbacks instead of growing a second
# copy of them.
artifact_view = _artifact_view
fitted_method = _fitted_method
axes_method = _axes_method
bank_method = _bank_method
delta_method = _delta_method
probe_names = _probe_names
vocabulary_method = _vocabulary_method

