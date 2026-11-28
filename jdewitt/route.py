# SPDX-License-Identifier: AGPL-3.0-or-later
"""The stepped route: one stage per command, with the evidence to decide the next.

`jdewitt edit` is the one-shot: it decides and writes. This module is the other
shape an agent needs when the decision matters, broken down so every step can be
inspected before the next byte moves:

    survey -> plan -> gate -> apply -> verify -> certify

Each step:

  * reads and writes a resumable state file in ``--workdir`` (``route.json``,
    atomic), and refuses to run on a stale state instead of re-deriving it;
  * emits one JSON with the FULL evidence for that decision: per-tensor codec
    rows, per-layer axis separability, candidate storage types with preview
    fidelity, byte costs, the artifact's own tensor-type census;
  * prints a ``decision`` block where every option is a command with its
    expected effect, cost and risk, so the agent chooses instead of guessing;
  * writes its full report to ``<workdir>/<step>.json`` and keeps the gated
    candidate artifact, so a later step can promote exactly the bytes measured.

``gate`` is the trial write: it writes the candidate for each storage type,
reads the FULL edited footprint back and keeps the winner; ``apply`` only
promotes a candidate whose sha matches the gate record. Nothing here measures
behaviour: ``certify`` states exactly what remains unmeasured and the commands
that would measure it, and ``removal_measured`` is false everywhere.
"""
from __future__ import annotations

import math
import os
import shlex
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

SCHEMA_STEP = 'jdewitt-route-step-v1'
SCHEMA_STATE = 'jdewitt-route-state-v1'
STEPS = ('survey', 'plan', 'gate', 'apply', 'verify', 'certify')
DEFAULTS = {'shape': 'perlayer', 'rank': 1, 'scale': 1.0, 'axis_space': 'output',
            'layers': None, 'edit_type': 'auto', 'min_cos': 0.95, 'repair': 'none',
            'keep': 0.5, 'probe_tensors': 8, 'strength': 1.0, 'strength_scale': 1.0,
            'axes_key': 'axes'}


# ---------- shared helpers ----------

def _sha(path):
    from .storage import file_digest

    return file_digest(path)


def _clean(value):
    """JSON-safe copy: numpy scalars to python, non-finite floats to None."""
    import numpy as np

    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.floating):
        return _clean(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def _cmd(workdir, step, **options):
    from .nextsteps import cli_command

    parts = cli_command('route', '--workdir', workdir, step)
    for key, value in options.items():
        if value is None or value is False:
            continue
        flag = '--' + key.replace('_', '-')
        if value is True:
            parts.append(flag)
        else:
            parts.extend([flag, str(value)])
    return shlex.join(parts)


def _default_workdir(artifact):
    artifact = Path(artifact)
    return artifact.with_name(artifact.stem + '.route')


def _state_path(workdir):
    return Path(workdir) / 'route.json'


def _load_state(workdir):
    path = _state_path(workdir)
    if not path.is_file():
        return None
    import json

    return json.loads(path.read_text())


def _save_state(workdir, state):
    from .answers import atomic_json

    state['updated_at'] = _now()
    atomic_json(_state_path(workdir), _clean(state))


def _write_report(workdir, step, report):
    from .answers import atomic_json

    Path(workdir).mkdir(parents=True, exist_ok=True)
    atomic_json(Path(workdir) / f'{step}.json', _clean(report))


def _settings(args, state):
    out = dict(DEFAULTS)
    if state:
        out.update(state.get('settings') or {})
    for key in DEFAULTS:
        value = getattr(args, key, None)
        if value is not None:
            out[key] = value
    return out


def _inputs(args, state):
    out = dict((state or {}).get('inputs') or {})
    for key in ('records', 'axes', 'bank', 'entry', 'delta_gguf'):
        value = getattr(args, key, None)
        if value is not None:
            out[key] = str(value) if key != 'entry' else value
    return out


def _input_shas(inputs):
    shas = {}
    for key in ('records', 'axes', 'delta_gguf'):
        path = inputs.get(key)
        shas[key] = _sha(path) if path and Path(path).is_file() else None
    bank = inputs.get('bank')
    shas['bank_json'] = (_sha(Path(bank) / 'bank.json')
                         if bank and (Path(bank) / 'bank.json').is_file() else None)
    return shas


def _ns(args, inputs, settings, artifact):
    return SimpleNamespace(
        gguf=Path(artifact), out=None, in_place=False,
        records=Path(inputs['records']) if inputs.get('records') else None,
        axes=Path(inputs['axes']) if inputs.get('axes') else None,
        axes_key=settings.get('axes_key') or 'axes',
        bank=Path(inputs['bank']) if inputs.get('bank') else None,
        entry=inputs.get('entry'),
        delta_gguf=Path(inputs['delta_gguf']) if inputs.get('delta_gguf') else None,
        strength=settings['strength'], strength_scale=settings['strength_scale'],
        shape=settings['shape'], rank=settings['rank'], scale=settings['scale'],
        axis_space=settings['axis_space'], layers=settings['layers'],
        edit_type=settings['edit_type'], min_cos=settings['min_cos'],
        repair=settings['repair'], renormalize=False)


def _state_view(state, workdir):
    if not state:
        return {'stage': 'none', 'workdir': str(workdir), 'steps_done': [],
                'artifact': None, 'inputs': {}, 'settings': {}}
    done = [row['step'] for row in state.get('history') or []]
    return {
        'stage': done[-1] if done else 'none',
        'workdir': str(workdir),
        'steps_done': done,
        'artifact': state.get('artifact'),
        'artifact_sha256': state.get('artifact_sha256'),
        'artifact_bytes': state.get('artifact_bytes'),
        'inputs': state.get('inputs') or {},
        'settings': state.get('settings') or {},
        'gate': (state.get('gate') or {}).get('summary'),
        'apply': (state.get('apply') or {}).get('summary'),
    }


def _classify(options, risks, recommendation, blocked=None):
    return {'recommended': recommendation, 'options': options,
            'blocked': blocked or []}


def _option(option_id, label, command, why, expected=None, cost=None, risk=None, gated=None):
    row = {'id': option_id, 'label': label, 'command': command, 'why': why}
    if expected is not None:
        row['expected'] = expected
    if cost is not None:
        row['cost'] = cost
    if risk is not None:
        row['risk'] = risk
    if gated is not None:
        row['gated'] = gated
    return row


def _record_step(state, step, report, options):
    state.setdefault('history', []).append({
        'step': step, 'at': _now(), 'ok': report.get('ok'),
        'elapsed_s': report.get('elapsed_s'),
        'options': [option['id'] for option in options]})
    state['updated_at'] = _now()


def _base_report(step, state, workdir, view, evidence, risks, decision, started):
    recommended = decision.get('recommended')
    next_command = next((option['command'] for option in decision.get('options') or []
                         if option['id'] == recommended), None)
    state_view = _state_view(state, workdir)
    if step in STEPS:
        state_view['stage'] = step
        if step not in state_view['steps_done']:
            state_view['steps_done'] = state_view['steps_done'] + [step]
    return {'schema': SCHEMA_STEP, 'command': 'route', 'step': step, 'ok': True,
            'state': state_view, 'artifact': view,
            'evidence': evidence, 'risks': risks, 'decision': decision,
            'next': next_command,
            'history': ((state or {}).get('history') or [])[-8:],
            'elapsed_s': round(time.monotonic() - started, 2)}


# ---------- survey ----------

def _census(artifact):
    from .ggufio import encodable_types
    from .ggufwrite import parse_container, tensor_nbytes
    from .quants import type_name_for_id

    header = parse_container(str(artifact))
    census = {}
    for tensor in header['tensors']:
        name = type_name_for_id(tensor['qtype'])
        row = census.setdefault(name, {'tensors': 0, 'bytes': 0})
        row['tensors'] += 1
        row['bytes'] += tensor_nbytes(tensor['dims'], tensor['qtype'])
    for name, row in census.items():
        row['encodable'] = name in encodable_types()
    return header, census


def _layer_evidence(records_path, keep):
    """Per-layer separability from the capture, or the honest reason it is unavailable."""
    from .spanfit import layer_separability, load_records

    try:
        records = load_records(str(records_path))
    except Exception as error:  # noqa: BLE001 - a survey reports, it does not crash
        return {'available': False, 'reason': f'{type(error).__name__}: {error}'}
    labels = {}
    for meta, _rec in records:
        label = meta.get('label') if isinstance(meta, dict) else None
        labels[label] = labels.get(label, 0) + 1
    try:
        report = layer_separability(records, keep=float(keep))
    except (SystemExit, Exception) as error:  # noqa: BLE001
        detail = str(error) or type(error).__name__
        return {'available': False, 'reason': detail, 'labels': labels, 'rows': len(records)}
    report.update({'labels': labels, 'rows': len(records), 'available': True})
    return report


def _records_survey(records_path, view, keep):
    """Cheap records read: labels, geometry, separability. No fit, no preview."""
    import torch

    from .spanfit import load_records

    try:
        records = load_records(str(records_path))
    except Exception as error:  # noqa: BLE001
        return {'ready': False, 'reason': 'RECORDS_UNREADABLE',
                'detail': f'{type(error).__name__}: {error}'}
    labels = {}
    shape = None
    for meta, rec in records:
        label = meta.get('label') if isinstance(meta, dict) else None
        labels[label] = labels.get(label, 0) + 1
        if shape is None and torch.is_tensor(rec):
            shape = list(rec.shape)
    width = shape[1] if shape and len(shape) == 2 else None
    ready = (labels.get('bad', 0) >= 1 and labels.get('good', 0) >= 1
             and width is not None and view.get('hidden') == width)
    detail = None
    if not ready:
        if labels.get('bad', 0) < 1 or labels.get('good', 0) < 1:
            detail = 'records need at least one bad and one good row'
        elif width is None:
            detail = 'records carry no [layers, width] tensor rows'
        else:
            detail = (f'records are {width}-wide, the artifact hidden width is '
                      f'{view.get("hidden")}; the records and the artifact have to be the same model')
    return {'ready': ready, 'reason': None if ready else 'NO_VERIFIED_ADAPTER',
            'detail': detail, 'labels': labels, 'rows': len(records),
            'shape': shape, 'width': width,
            'layers': _layer_evidence(records_path, keep)}


def _axes_survey(path, view, key):
    """Cheap read of an imported axis: shape, width agreement, provenance."""
    import torch

    try:
        payload = torch.load(str(path), map_location='cpu', weights_only=False)
        axes = payload.get(key, payload.get('axes', payload)) if isinstance(payload, dict) else payload
    except Exception as error:  # noqa: BLE001
        return {'ready': False, 'reason': 'AXES_UNREADABLE',
                'detail': f'{type(error).__name__}: {error}'}
    if not torch.is_tensor(axes) or axes.ndim not in (1, 2):
        return {'ready': False, 'reason': 'AXES_UNREADABLE',
                'detail': f'{path} carries no [D] or [L, D] tensor under {key!r}'}
    width = int(axes.shape[-1])
    ready = view.get('hidden') == width
    detail = None
    if not ready:
        detail = (f'the axis is {width}-wide; the artifact hidden width is '
                  f'{view.get("hidden")}. A direction is only valid in the space it was '
                  f'measured in')
    return {'ready': ready,
            'reason': None if ready else 'GEOMETRY_MISMATCH',
            'detail': detail,
            'shape': [int(size) for size in axes.shape],
            'width': width,
            'layers': int(axes.shape[0]) if axes.ndim == 2 else None,
            'unit_norms': ([round(float(axes[row].norm()), 6)
                            for row in range(min(4, int(axes.shape[0])))]
                           if axes.ndim == 2 else round(float(axes.norm()), 6)),
            'provenance': (payload.get('provenance') if isinstance(payload, dict) else None)}


def _input_evidence(artifact, args, inputs, settings, view):
    evidence = {}
    if inputs.get('records'):
        evidence['records'] = {'path': inputs['records'],
                               **_records_survey(inputs['records'], view, settings['keep'])}
    if inputs.get('axes'):
        evidence['axes'] = {'path': inputs['axes'],
                            **_axes_survey(inputs['axes'], view,
                                           settings.get('axes_key') or 'axes')}
    if inputs.get('bank'):
        from .edit import bank_method

        method, _resolved = bank_method(_ns(args, inputs, settings, artifact), view)
        evidence['bank'] = method
    if inputs.get('delta_gguf'):
        from .edit import delta_method

        method, _ = delta_method(_ns(args, inputs, settings, artifact), view)
        evidence['delta_gguf'] = method
    return evidence


def _vocab(artifact, view):
    from .edit import vocabulary_method

    return vocabulary_method(SimpleNamespace(gguf=Path(artifact), out=None, in_place=False), view)


def _risks_from_inputs(view, census, evidence):
    risks = []
    if census and not any(name in ('F16', 'BF16', 'F32') for name in census):
        risks.append({'code': 'LOW_BIT_ARTIFACT', 'severity': 'high',
                      'detail': ('every stored tensor is a lossy quant type; a rank-1 removal can be '
                                 'below the codec grid. A gate refusal is the finding, not a bug')})
    if view.get('unmapped_writeback_width'):
        risks.append({'code': 'UNMAPPED_HIDDEN_WIDE', 'severity': 'low',
                      'detail': (f'{len(view["unmapped_writeback_width"])} hidden-wide tensors are not '
                                 f'residual-write names and will not be edited: '
                                 f'{view["unmapped_writeback_width"][:6]}')})
    records = evidence.get('records')
    if records:
        if not records.get('ready'):
            risks.append({'code': records.get('reason') or 'RECORDS_NOT_READY',
                          'severity': 'high', 'detail': records.get('detail')})
        layers = records.get('layers') or {}
        if not layers.get('available'):
            risks.append({'code': 'SMALL_OR_FLAT_CAPTURE', 'severity': 'medium',
                          'detail': layers.get('reason')})
    axes = evidence.get('axes')
    if axes:
        if not axes.get('ready'):
            risks.append({'code': axes.get('reason') or 'AXES_NOT_READY',
                          'severity': 'high', 'detail': axes.get('detail')})
        else:
            risks.append({'code': 'IMPORTED_AXIS_UNMEASURED', 'severity': 'medium',
                          'detail': ('an imported axis carries its source provenance, not a '
                                     'measurement on this artifact; the gate measures codec '
                                     'fidelity and nothing else')})
    bank = evidence.get('bank')
    if bank:
        if not bank.get('ready'):
            risks.append({'code': bank.get('reason') or 'BANK_NOT_READY',
                          'severity': 'high', 'detail': bank.get('detail')})
        elif not bank.get('removal_measured'):
            risks.append({'code': 'BANK_UNMEASURED', 'severity': 'medium',
                          'detail': ('the entry is banked_unmeasured: a pinned bundle and footprint, '
                                     'but no measured removal or retention on an artifact class')})
    delta = evidence.get('delta_gguf')
    if delta and not delta.get('ready'):
        risks.append({'code': delta.get('reason') or 'DELTA_NOT_READY',
                      'severity': 'high', 'detail': delta.get('detail')})
    return risks


def _survey(workdir, artifact, settings, state, started):
    from .edit import artifact_view
    from .nextsteps import cli_line

    view = artifact_view(artifact)
    header, census = _census(artifact)
    from .gguf_adapters import discover_writeback_targets, gguf_weight_shapes

    shapes = gguf_weight_shapes(str(artifact))
    discovered = (discover_writeback_targets(shapes, view['hidden'])
                  if view['hidden'] is not None
                  else {'targets': {}, 'unmapped_writeback_width': []})
    kinds = {}
    for layer, names in discovered['targets'].items():
        for name in names:
            kind = name.split('.')[-2]
            row = kinds.setdefault(kind, {'tensors': 0, 'layers': [], 'example': name})
            row['tensors'] += 1
            row['layers'].append(int(layer))
    for row in kinds.values():
        row['layers'] = sorted(set(row['layers']))
        row['first'], row['last'] = row['layers'][0], row['layers'][-1]
    inputs = _inputs(None, state)
    evidence = {'tensor_types': census, 'tensor_count': len(header['tensors']),
                'writeback': {'kinds': kinds, 'targets': discovered['targets'],
                              'unmapped_writeback_width': discovered['unmapped_writeback_width']},
                'inputs': _input_evidence(artifact, SimpleNamespace(
                    gguf=Path(artifact), records=None, bank=None, entry=None, delta_gguf=None,
                    strength=1.0, strength_scale=1.0), inputs, settings, view)}
    risks = _risks_from_inputs(view, census, evidence['inputs'])
    options = []
    records = evidence['inputs'].get('records')
    axes = evidence['inputs'].get('axes')
    bank = evidence['inputs'].get('bank')
    delta = evidence['inputs'].get('delta_gguf')
    if records and records.get('ready'):
        options.append(_option(
            'plan', 'fit the axis and build the candidate table',
            _cmd(workdir, 'plan', records=inputs['records']),
            'records are present and their width matches the artifact',
            expected='per-layer separability, per-tensor preview fidelity, cost table',
            cost={'rows': records.get('rows'), 'layers': (records.get('shape') or [None])[0]}))
    if axes and axes.get('ready'):
        options.append(_option(
            'plan', 'plan the imported axis against this artifact',
            _cmd(workdir, 'plan', axes=inputs['axes']),
            'an axis file is attached and its width matches the artifact hidden width',
            expected='per-tensor preview fidelity, cost table, the axis provenance',
            risk='IMPORTED_AXIS_UNMEASURED'))
    if bank and bank.get('ready'):
        options.append(_option(
            'plan', 'plan the banked axis against this artifact',
            _cmd(workdir, 'plan', bank=inputs['bank'], entry=inputs.get('entry')),
            'a digest-pinned bank entry covers this artifact; no fit needed',
            expected='entry footprint, per-kind strength, type set, cost table',
            risk='banked_unmeasured' if not bank.get('removal_measured') else None))
    if delta and delta.get('ready'):
        options.append(_option(
            'plan', 'plan the explicit delta',
            _cmd(workdir, 'plan', delta_gguf=inputs['delta_gguf']),
            'a delta GGUF was supplied; its tensors resolved against the artifact',
            expected='per-tensor delta table and codec preview'))
    if not options:
        options.append(_option(
            'capture', 'capture contrast records first (GATED: HF checkpoint + compute device)',
            cli_line('capture', '--model', '<HF-CHECKPOINT>', '--suite', '<suite.jsonl>',
                     '--out', Path(workdir) / 'records.pt', '--ctx-size', 4096,
                     '--gguf', artifact),
            'no records, axes, bank or delta is attached, so there is no axis to gate',
            expected='a fit-usable capture with both labels, cross-checked against the artifact',
            gated='needs the 16-bit checkpoint and a compute device; the route never fits blind'))
    static = _vocab(artifact, view)
    options.append(_option(
        'static', 'static vocabulary route (no fit, no GPU, no behaviour measurement)',
        static['commands'][0],
        'rewrite denial-opener rows in place; ids stay intact and the revert is byte-exact',
        expected='row/value changes and a tokenizer audit, not answers',
        gated='removal is GATED behind a serve this command never starts'))
    decision = _classify(options, risks, options[0]['id'],
                         blocked=[] if records or axes or bank or delta else
                         ['no axis source: pass --records/--axes/--bank/--delta-gguf to plan, '
                          'or capture first'])
    report = _base_report('survey', state, workdir, view, evidence, risks, decision, started)
    return report


# ---------- plan ----------

def _preview_table(gguf, edits, types, probes, scale, renormalize):
    from .ggufio import preview_edit

    table = {}
    for kind in types:
        reading = preview_edit(str(gguf), {name: edits[name] for name in probes},
                               names=probes, scale=scale, renormalize=renormalize,
                               target_type=kind)
        rows = reading.get('tensors') or []
        table[kind] = {'min_cos': reading.get('min_cos'),
                       'mean_cos': reading.get('mean_cos'),
                       'probed': len(rows),
                       'worst_tensor': (min(rows, key=lambda row: row.get('cos') or 0.0)['name']
                                        if rows else None),
                       'rows': rows}
    return table


def _tensor_meta(artifact):
    from .ggufwrite import parse_container, tensor_nbytes
    from .quants import type_name_for_id

    header = parse_container(str(artifact))
    return {tensor['name']: {'layer': (int(tensor['name'].split('.')[1])
                                       if tensor['name'].startswith('blk.') else None),
                             'kind': tensor['name'].split('.')[-2],
                             'dims': tensor['dims'],
                             'type': type_name_for_id(tensor['qtype']),
                             'bytes': tensor_nbytes(tensor['dims'], tensor['qtype'])}
            for tensor in header['tensors']}


def _cost_table(meta, edits, types):
    from .quants import type_id_for_name
    from .ggufwrite import tensor_nbytes

    source = sum(meta[name]['bytes'] for name in edits)
    after = {}
    for kind in types:
        qtype = type_id_for_name(kind)
        after[kind] = sum(tensor_nbytes(meta[name]['dims'], qtype) for name in edits)
    return {'edited_tensors': len(edits), 'edited_bytes_source': int(source),
            'edited_bytes_after': after,
            'artifact_bytes': sum(row['bytes'] for row in meta.values())}


def _plan(workdir, artifact, settings, args, state, started):
    from .edit import (LADDER, artifact_view, axes_method, bank_method, delta_method,
                       fitted_method, probe_names)
    from .errors import RouteRefused

    if not state or not state.get('survey'):
        raise RouteRefused('ROUTE_ORDER', 'plan needs a survey of the same workdir first')
    if _sha(artifact) != state.get('artifact_sha256'):
        raise RouteRefused('ROUTE_STALE', 'the artifact changed since the survey; re-run survey')
    inputs = _inputs(args, state)
    if inputs.get('records') and inputs.get('axes'):
        raise RouteRefused('ROUTE_INPUT', 'records and axes are alternatives: records fit an axis, '
                                           'axes import one')
    ns = _ns(args, inputs, settings, artifact)
    view = artifact_view(artifact)
    meta = _tensor_meta(artifact)
    evidence, method, edits = {}, None, None
    if inputs.get('records') or inputs.get('axes'):
        if inputs.get('records'):
            method, edits = fitted_method(ns, view, require_floor=False)
        else:
            method, edits = axes_method(ns, view, require_floor=False)
        evidence['method'] = method
        if method.get('ready'):
            layers = {}
            for name in edits:
                axis = edits[name][0]
                row = layers.setdefault(meta[name]['layer'], {'tensors': 0, 'axis_norm': None,
                                                              'kinds': []})
                row['tensors'] += 1
                row['axis_norm'] = round(float(float((axis ** 2).sum()) ** 0.5), 6)
                if meta[name]['kind'] not in row['kinds']:
                    row['kinds'].append(meta[name]['kind'])
            evidence['layers'] = {'axis': layers}
            if inputs.get('records'):
                evidence['layers']['separability'] = _layer_evidence(inputs['records'],
                                                                     settings['keep'])
            if method.get('broadcast'):
                evidence['layers']['broadcast'] = method['broadcast']
            types = (LADDER if settings['edit_type'] == 'auto' else (settings['edit_type'],))
            probes = (method.get('preview') or {}).get('probes') or probe_names(edits)
            probes = probes[:max(1, int(settings['probe_tensors']))]
            evidence['preview'] = _preview_table(
                artifact, edits, types, probes, settings['scale'],
                settings['repair'] != 'none')
            rows = []
            for name in sorted(edits):
                info = meta[name]
                row = {'name': name, 'layer': info['layer'], 'kind': info['kind'],
                       'dims': info['dims'], 'source_type': info['type'],
                       'bytes_before': info['bytes'], 'axis_norm': None, 'preview': {}}
                axis = edits[name][0]
                row['axis_norm'] = round(float(float((axis ** 2).sum()) ** 0.5), 6)
                for kind, table in evidence['preview'].items():
                    hit = next((item for item in table['rows'] if item['name'] == name), None)
                    if hit is not None:
                        row['preview'][kind] = {key: hit.get(key) for key in
                                                ('cos', 'snr', 'norm_ratio')}
                rows.append(row)
            evidence['tensors'] = rows
            evidence['cost'] = _cost_table(meta, edits, types)
            evidence['candidate_types'] = list(types)
    elif inputs.get('bank'):
        method, resolved = bank_method(ns, view)
        evidence['method'] = method
        if method.get('ready'):
            targets = (resolved or {}).get('targets') or {}
            rows = []
            for kind, names in sorted(targets.items()):
                for name in names:
                    info = meta.get(name, {})
                    rows.append({'name': name, 'kind': kind, 'dims': info.get('dims'),
                                 'source_type': info.get('type'),
                                 'bytes_before': info.get('bytes'),
                                 'strength': (method.get('strength') or {}).get(kind)})
            edits = {row['name']: None for row in rows}
            evidence['tensors'] = rows
            evidence['preview'] = {'deferred': 'a banked axis lives in the pinned bundle; the '
                                               'plan does not read tensor data. The gate step '
                                               'loads it and measures every edited tensor.'}
            evidence['candidate_types'] = list(method.get('edit_types') or ())
            evidence['cost'] = {'edited_tensors': len(rows),
                                'edited_bytes_source': sum(row.get('bytes_before') or 0
                                                           for row in rows)}
    elif inputs.get('delta_gguf'):
        import numpy as np
        from gguf import GGUFReader

        method, _ = delta_method(ns, view)
        evidence['method'] = method
        if method.get('ready'):
            reader = GGUFReader(str(inputs['delta_gguf']))
            edits, rows = {}, []
            for tensor in reader.tensors:
                values = np.asarray(tensor.data, dtype=np.float32).reshape(-1) * settings['strength']
                edits[tensor.name] = values
                info = meta.get(tensor.name, {})
                rows.append({'name': tensor.name, 'layer': info.get('layer'),
                             'kind': info.get('kind'), 'dims': list(tensor.shape),
                             'source_type': info.get('type'),
                             'bytes_before': info.get('bytes'),
                             'delta_norm': round(float(np.linalg.norm(values)), 6)})
            evidence['tensors'] = rows
            types = ((settings['edit_type'],) if settings['edit_type'] != 'auto'
                     else ((meta[rows[0]['name']]['type'],) if rows and rows[0]['name'] in meta
                           else ('q6_k',)))
            evidence['preview'] = _preview_table(
                artifact, edits, types, list(edits)[:max(1, int(settings['probe_tensors']))],
                1.0, False)
            evidence['cost'] = {'edited_tensors': len(edits)}
            evidence['candidate_types'] = list(types)
    else:
        raise RouteRefused('ROUTE_INPUT', 'plan needs --records, --axes, --bank or --delta-gguf '
                                          '(survey without them to see every route)')
    if method is None or not method.get('ready'):
        raise RouteRefused(method.get('reason') or 'ROUTE_INPUT',
                           method.get('detail') or 'the chosen method is not ready on this artifact')
    risks = []
    preview = evidence.get('preview') or {}
    if 'deferred' not in preview:
        below = [kind for kind, table in preview.items()
                 if isinstance(table, dict) and table.get('min_cos') is not None
                 and float(table['min_cos']) < float(settings['min_cos'])]
        clears = [kind for kind, table in preview.items()
                  if isinstance(table, dict) and table.get('min_cos') is not None
                  and float(table['min_cos']) >= float(settings['min_cos'])]
        if below and not clears:
            risks.append({'code': 'CODEC_FLOOR', 'severity': 'high',
                          'detail': (f'no probed type clears the {settings["min_cos"]} floor: '
                                     f'{", ".join(below)}; the gate is expected to refuse')})
        elif below:
            risks.append({'code': 'TYPES_BELOW_FLOOR', 'severity': 'low',
                          'detail': (f'{", ".join(below)} are below the {settings["min_cos"]} floor; '
                                     f'the gate steps to {clears[0]} if the full footprint agrees')})
    if method.get('removal_measured') is False:
        risks.append({'code': 'UNMEASURED_ENTRY', 'severity': 'medium',
                      'detail': 'the plan measures codec fidelity only; removal is unmeasured'})
    types = _planned_types(evidence, settings)
    options = []
    for kind in types:
        table = preview.get(kind) if isinstance(preview, dict) else None
        clears = (isinstance(table, dict) and table.get('min_cos') is not None
                  and float(table['min_cos']) >= float(settings['min_cos']))
        options.append(_option(
            f'gate-{kind}', f'gate the {kind} candidate and read every edited tensor back',
            _cmd(workdir, 'gate', edit_type=kind),
            ('this type clears the sampled preview' if clears else
             'this type is below the sampled preview floor; the gate measures the real footprint'),
            expected={'preview_min_cos': (table or {}).get('min_cos') if isinstance(table, dict) else None},
            cost={'tensors': (method.get('cost') or {}).get('edited_tensors')},
            risk=None if clears else 'EXPECTED_CODEC_FLOOR'))
    options.append(_option(
        'replan', 'change the footprint, strength, rank or shape and re-plan',
        _cmd(workdir, 'plan', layers='<a-b>', scale='<s>'),
        'the plan is deterministic; a new footprint or strength costs one command, no re-fit'))
    static = _vocab(artifact, view)
    options.append(_option('static', 'static vocabulary route', static['commands'][0],
                           'no fit and no server; kept as the fallback when the codec refuses'))
    decision = _classify(options, risks, options[0]['id'])
    report = _base_report('plan', state, workdir, view, evidence, risks, decision, started)
    state['settings'] = settings
    state['inputs'] = inputs
    state['input_sha256'] = _input_shas(inputs)
    for stale in ('gate', 'apply', 'verify', 'certify'):
        state.pop(stale, None)
    state['plan'] = {'at': _now(), 'report': 'plan.json', 'method': method['method'],
                     'settings': settings,
                     'summary': {'method': method['method'], 'types': list(types),
                                 'tensors': (method.get('cost') or {}).get('edited_tensors'),
                                 'preview': {kind: (table or {}).get('min_cos')
                                             for kind, table in preview.items()
                                             if isinstance(table, dict)}}}
    _record_step(state, 'plan', report, options)
    return report


def _planned_types(evidence, settings):
    if settings['edit_type'] != 'auto':
        return (settings['edit_type'],)
    if evidence.get('candidate_types'):
        return tuple(evidence['candidate_types'])
    preview = evidence.get('preview') or {}
    if 'readings' in preview:
        return tuple(row['edit_type'] for row in preview['readings'])
    return tuple(kind for kind, table in preview.items() if isinstance(table, dict))


# ---------- gate ----------

def _gate(workdir, artifact, settings, args, state, started):
    from .errors import RouteRefused
    from .ggufio import compare_edit

    if not state or not state.get('plan'):
        raise RouteRefused('ROUTE_ORDER', 'gate needs a plan in the same workdir first')
    if _sha(artifact) != state.get('artifact_sha256'):
        raise RouteRefused('ROUTE_STALE', 'the artifact changed since the plan; re-run survey/plan')
    if _input_shas(state.get('inputs') or {}) != (state.get('input_sha256') or {}):
        raise RouteRefused('ROUTE_STALE', 'the records/bank/delta changed since the plan; re-run plan')
    from .edit import artifact_view, delta_method, fitted_method

    inputs = _inputs(args, state)
    ns = _ns(args, inputs, settings, artifact)
    view = artifact_view(artifact)
    kinds = tuple(state['plan']['summary']['types'])
    if settings['edit_type'] != 'auto':
        kinds = (settings['edit_type'],)
    method_kind = state['plan']['method']
    edits = None
    if method_kind in ('fitted-axis', 'imported-axis'):
        if method_kind == 'fitted-axis':
            method, edits = fitted_method(ns, view, require_floor=False)
        else:
            from .edit import axes_method

            method, edits = axes_method(ns, view, require_floor=False)
        if not method.get('ready'):
            raise RouteRefused(method.get('reason') or 'ROUTE_INPUT', method.get('detail'))
    elif method_kind == 'explicit-delta':
        import numpy as np
        from gguf import GGUFReader

        method, _ = delta_method(ns, view)
        if not method.get('ready'):
            raise RouteRefused(method.get('reason') or 'ROUTE_INPUT', method.get('detail'))
        reader = GGUFReader(str(inputs['delta_gguf']))
        edits = {}
        for tensor in reader.tensors:
            edits[tensor.name] = (np.asarray(tensor.data, dtype=np.float32).reshape(-1)
                                  * settings['strength'])
    candidates_dir = Path(workdir) / 'candidates'
    candidates_dir.mkdir(parents=True, exist_ok=True)
    for stale in candidates_dir.glob('*.gguf'):
        stale.unlink()
    attempts, winner = [], None
    for kind in kinds:
        candidate = candidates_dir / f'{kind}.gguf'
        attempt = {'edit_type': kind, 'candidate': str(candidate), 'ok': False,
                   'discarded': True}
        attempt_started = time.monotonic()
        try:
            if method_kind in ('fitted-axis', 'imported-axis'):
                from .rewrite import rewrite_with_edits
                write_report = rewrite_with_edits(
                    str(artifact), str(candidate), edits, scale=settings['scale'],
                    axis_space=settings['axis_space'], edit_type=kind,
                    renormalize=settings['repair'] != 'none', repair=settings['repair'],
                    verify_untouched=True)
                delivered = compare_edit(str(artifact), str(candidate), edits,
                                         names=list(edits), scale=settings['scale'],
                                         renormalize=settings['repair'] != 'none')
                attempt.update({'writer': write_report, 'delivered': delivered})
            elif method_kind == 'banked-axis':
                from .bank_apply import bank_apply
                write_report = bank_apply(inputs['bank'], inputs.get('entry'), artifact,
                                          candidate, edit_type=kind,
                                          strength_scale=settings['strength_scale'])
                attempt.update({'writer': write_report,
                                'delivered': {'tensors': write_report['edited']['tensors'],
                                              'min_cos': write_report['edited']['min_cos'],
                                              'mean_cos': write_report['edited']['mean_cos']}})
            else:
                from .agent import gguf_edit
                write_report = gguf_edit(str(artifact), str(candidate), [],
                                         inputs['delta_gguf'], settings['strength'])
                delivered = compare_edit(str(artifact), str(candidate), edits,
                                         names=list(edits))
                attempt.update({'writer': write_report, 'delivered': delivered})
            cosines = [row.get('cos') for row in attempt['delivered']['tensors']
                       if row.get('cos') is not None]
            min_cos = min(cosines) if cosines else None
            untouched = (attempt.get('writer') or {}).get('untouched')
            if untouched is None and (attempt.get('writer') or {}).get('all_other_bytes_verbatim') is not None:
                untouched = {'verified': bool((attempt.get('writer') or {})
                                              .get('all_other_bytes_verbatim'))}
            attempt.update({
                'ok': bool(min_cos is not None
                           and float(min_cos) >= float(settings['min_cos'])
                           and (untouched is None or untouched.get('verified'))),
                'min_cos': min_cos,
                'mean_cos': (round(sum(cosines) / len(cosines), 4) if cosines else None),
                'tensors': len(cosines),
                'below_floor': [row['name'] for row in attempt['delivered']['tensors']
                                if float(row.get('cos') or 0.0) < float(settings['min_cos'])],
                'provenance': {'source_sha256': state['artifact_sha256'],
                               'candidate_sha256': None, 'seconds': None}})
            if candidate.is_file() and attempt['ok']:
                attempt['provenance']['candidate_sha256'] = _sha(candidate)
                attempt['bytes'] = candidate.stat().st_size
                attempt['discarded'] = False
            elif candidate.is_file():
                attempt['bytes'] = candidate.stat().st_size
                attempt['provenance']['candidate_sha256'] = _sha(candidate)
            attempt['provenance']['seconds'] = round(time.monotonic() - attempt_started, 2)
        except Exception as error:  # noqa: BLE001 - each attempt reports its own failure
            attempt['error'] = {'code': getattr(error, 'code', type(error).__name__),
                                'message': str(error)}
            candidate.unlink(missing_ok=True)
        attempts.append(attempt)
        if attempt['ok']:
            winner = attempt
            break
        candidate.unlink(missing_ok=True)
        attempt['candidate_deleted'] = True
    risks = []
    if winner is None:
        risks.append({'code': 'CODEC_FLOOR', 'severity': 'high',
                      'detail': 'no attempted storage type cleared the floor over the full '
                                'footprint; the artifact was not edited'})
    for attempt in attempts:
        if attempt.get('error'):
            risks.append({'code': attempt['error']['code'], 'severity': 'high',
                          'detail': f"{attempt['edit_type']}: {attempt['error']['message']}"})
        elif attempt.get('below_floor'):
            risks.append({'code': 'BELOW_FLOOR', 'severity': 'medium',
                          'detail': (f"{attempt['edit_type']}: {len(attempt['below_floor'])} tensor(s) "
                                     f"below the floor, worst {attempt['below_floor'][0]}")})
    options = []
    if winner is not None:
        default_out = Path(artifact).with_name(
            Path(artifact).stem + '-edited' + Path(artifact).suffix)
        options.append(_option(
            'apply', 'promote the gated candidate to the artifact path',
            _cmd(workdir, 'apply', out=default_out),
            'this candidate cleared the floor over every edited tensor and its untouched bytes were verified',
            expected={'edit_type': winner['edit_type'], 'min_cos': winner['min_cos'],
                      'bytes': winner.get('bytes')}))
        options.append(_option('replan', 'lower the strength or narrow the footprint and re-plan',
                               _cmd(workdir, 'plan', scale='<s>', layers='<a-b>'),
                               'a floor miss is a codec/grid fact; the honest levers are strength and footprint'))
    else:
        retry_scale = round(max(float(settings['scale']) * 0.5, 0.05), 3)
        options.append(_option(
            'gate-retry', f'retry the ladder at scale {retry_scale}',
            _cmd(workdir, 'gate', scale=retry_scale),
            'half the current strength costs less codec error; the gate re-measures every '
            'tensor at the new strength',
            expected={'scale': retry_scale, 'floor': settings['min_cos']}))
        tried = {str(attempt.get('edit_type', '')).casefold() for attempt in attempts}
        untried = [kind for kind in ('q6_k', 'q8_0', 'f16') if kind.casefold() not in tried]
        if untried:
            options.append(_option(
                f'gate-{untried[0]}', f'gate the {untried[0]} candidate instead',
                _cmd(workdir, 'gate', edit_type=untried[0]),
                'the attempted type(s) missed; this is the next storage type in the ladder'))
        options.append(_option(
            'replan', f'change the footprint or strength and re-plan (scale {retry_scale})',
            _cmd(workdir, 'plan', scale=retry_scale, layers='<a-b>'),
            'the plan is deterministic; the gate is the verdict'))
    from .edit import artifact_view as _view  # noqa: F401 - keep the lazy import contract
    from .nextsteps import cli_line

    options.append(_option('static', 'static vocabulary route',
                           cli_line('vocab-centroid', '--weights', artifact, '--out',
                                    artifact.with_name(artifact.stem + '-centroid.gguf'),
                                    '--mode', 'centroid-orthogonal', '--apply'),
                           'a real in-place edit that measures rows and bytes, not answers'))
    decision = _classify(options, risks, options[0]['id'])
    report = _base_report('gate', state, workdir, view,
                          {'attempts': attempts, 'floor': settings['min_cos'],
                           'winner': winner['edit_type'] if winner else None,
                           'winner_candidate': winner['candidate'] if winner else None},
                          risks, decision, started)
    report['ok'] = winner is not None
    state['settings'] = settings
    for stale in ('apply', 'verify', 'certify'):
        state.pop(stale, None)
    edited_names = ([row['name'] for row in winner['delivered']['tensors']]
                    if winner else [])
    state['gate'] = {'at': _now(), 'report': 'gate.json',
                     'candidate': winner['candidate'] if winner else None,
                     'candidate_sha256': (winner['provenance']['candidate_sha256']
                                          if winner else None),
                     'edit_type': winner['edit_type'] if winner else None,
                     'edited_names': edited_names,
                     'floor': settings['min_cos'],
                     'summary': {'edit_type': winner['edit_type'] if winner else None,
                                 'min_cos': winner['min_cos'] if winner else None,
                                 'attempts': len(attempts)}}
    _record_step(state, 'gate', report, options)
    return report


# ---------- apply ----------

def _promote(candidate, output):
    """Copy the gated candidate to `output` atomically (fsync, then rename)."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name('.' + output.name + f'.{os.getpid()}.new')
    try:
        with open(candidate, 'rb') as source, open(temporary, 'wb') as target:
            shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, output)
        fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _apply(workdir, artifact, settings, args, state, started):
    from .edit import apply_in_place, artifact_view
    from .errors import RouteRefused
    from .nextsteps import cli_line

    if not state or not state.get('gate') or not state['gate'].get('candidate'):
        raise RouteRefused('ROUTE_ORDER', 'apply needs a gated candidate in the same workdir')
    gate = state['gate']
    if _sha(artifact) != state.get('artifact_sha256'):
        raise RouteRefused('ROUTE_STALE', 'the artifact changed since the gate; re-run survey/plan/gate')
    candidate = Path(gate['candidate'])
    if not candidate.is_file() or _sha(candidate) != gate.get('candidate_sha256'):
        raise RouteRefused('ROUTE_STALE', 'the gated candidate changed or disappeared; re-run gate')
    if args.out is not None and args.in_place:
        raise ValueError('--out and --in-place are alternatives; pick one')
    if args.in_place:
        output = Path(artifact)
    else:
        output = Path(args.out) if args.out is not None else Path(artifact).with_name(
            Path(artifact).stem + '-edited' + Path(artifact).suffix)
        if output.resolve() == Path(artifact).resolve():
            raise ValueError('--out is the input artifact; pass --in-place for that, or a new path')
    started_apply = time.monotonic()
    report = {'candidate': str(candidate), 'candidate_sha256': gate['candidate_sha256'],
              'edit_type': gate.get('edit_type'), 'floor': gate.get('floor'),
              'output': str(output), 'in_place': bool(args.in_place),
              'pre_apply_sha256': state.get('artifact_sha256')}
    if args.in_place:
        apply_in_place(Path(artifact), candidate, report)
        report['output_sha256'] = report.get('patched_sha256')
    else:
        _promote(candidate, output)
        report['output_sha256'] = _sha(output)
    report['bytes'] = Path(output).stat().st_size
    report['elapsed_s'] = round(time.monotonic() - started_apply, 2)
    view = artifact_view(output)
    options = [_option('verify', 'recompute the edit fingerprint and untouched-byte identity',
                       _cmd(workdir, 'verify'),
                       'the write is not the proof; verify reads both files')]
    if args.in_place:
        options.append(_option(
            'revert', 'restore the pre-apply bytes from the sidecar',
            cli_line('gguf-revert', artifact, report.get('revert_sidecar')),
            'the original bytes were saved before the rename'))
    options.append(_option('certify', 'emit the certification packet', _cmd(workdir, 'certify'),
                           'certify states what is measured and what still needs a serve'))
    risks = []
    if not report.get('output_sha256'):
        risks.append({'code': 'OUTPUT_UNREADABLE', 'severity': 'high',
                      'detail': 'the promoted file could not be hashed'})
    decision = _classify(options, risks, 'verify')
    step_report = _base_report('apply', state, workdir, view, report, risks, decision, started)
    for stale in ('verify', 'certify'):
        state.pop(stale, None)
    state['apply'] = {'at': _now(), 'report': 'apply.json',
                      'output': str(output), 'output_sha256': report.get('output_sha256'),
                      'candidate_sha256': gate['candidate_sha256'],
                      'in_place': bool(args.in_place),
                      'pre_apply_sha256': report.get('pre_apply_sha256'),
                      'revert_sidecar': report.get('revert_sidecar'),
                      'summary': {'output': str(output), 'edit_type': gate.get('edit_type'),
                                  'bytes': report.get('bytes'),
                                  'output_sha256': report.get('output_sha256')}}
    _record_step(state, 'apply', step_report, options)
    return step_report


# ---------- verify ----------

def _stored_fingerprint(path):
    """The edit fingerprint an edited GGUF carries in its own metadata."""
    import json

    from gguf import GGUFReader

    field = GGUFReader(str(path)).fields.get('jdewitt.fingerprint')
    if field is None:
        return None
    try:
        return json.loads(str(field.contents()))
    except (TypeError, ValueError):
        return None


def _sidecar_original(sidecar):
    """The pristine digest a revert sidecar records, in either format."""
    from .ggufio import REVERT_LOG_MAGIC, _revert_log_entries

    path = Path(sidecar)
    with open(path, 'rb') as probe:
        magic = probe.read(len(REVERT_LOG_MAGIC))
    if magic == REVERT_LOG_MAGIC:
        for kind, payload in _revert_log_entries(path):
            if kind == 'pristine':
                return payload
        return None
    import numpy as np

    payload = np.load(path)
    return str(payload['original_sha256']) if 'original_sha256' in payload else None


def _untouched_identity(source, output, exclude):
    from .ggufwrite import parse_container, tensor_nbytes

    old = parse_container(str(source))
    new = parse_container(str(output))
    new_by_name = {tensor['name']: tensor for tensor in new['tensors']}
    changed, identical, missing = [], 0, []
    with open(source, 'rb') as a, open(output, 'rb') as b:
        for tensor in old['tensors']:
            name = tensor['name']
            if name in exclude:
                continue
            other = new_by_name.get(name)
            if other is None:
                missing.append(name)
                continue
            nbytes = tensor_nbytes(tensor['dims'], tensor['qtype'])
            a.seek(old['data_base'] + tensor['offset'])
            b.seek(new['data_base'] + other['offset'])
            same = True
            remaining = nbytes
            while remaining:
                chunk = min(8 * 1024 * 1024, remaining)
                if a.read(chunk) != b.read(chunk):
                    same = False
                    break
                remaining -= chunk
            if same and tensor['qtype'] == other['qtype']:
                identical += 1
            else:
                changed.append(name)
    return {'identical': identical, 'changed': changed, 'missing': missing,
            'checked': len(old['tensors']) - len(exclude), 'excluded': sorted(exclude)}


def _verify(workdir, artifact, settings, args, state, started):
    from .edit import artifact_view
    from .errors import RouteRefused
    from .nextsteps import cli_line
    from .verify import verify_pair

    if not state or not state.get('apply'):
        raise RouteRefused('ROUTE_ORDER', 'verify needs an applied artifact in the same workdir')
    output = Path(state['apply']['output'])
    if not output.is_file():
        raise RouteRefused('ROUTE_STALE', f'{output} does not exist; re-run apply')
    exclude = set(state.get('gate', {}).get('edited_names') or [])
    missing = ['behaviour (removal) on a served artifact',
               'capability/retention on the paired suites',
               'the pinned serve argv and per-row shape verdicts']
    evidence: dict = {'missing': missing}
    in_place = bool(state['apply'].get('in_place'))
    if in_place:
        # There is no pristine file on disk: the pre-apply bytes are the sidecar.
        # Prove the sidecar records THOSE bytes and that the artifact carries the
        # gate's fingerprint; byte-for-byte untouched identity is not checkable
        # in place and is named as such instead of reported as a pass.
        sidecar = state['apply'].get('revert_sidecar')
        present = bool(sidecar and Path(sidecar).is_file())
        original = _sidecar_original(sidecar) if present else None
        pre_apply = state['apply'].get('pre_apply_sha256')
        stored = _stored_fingerprint(output)
        gate_names = set(state.get('gate', {}).get('edited_names') or [])
        checks = [
            {'name': 'revert_sidecar_present', 'ok': present, 'detail': sidecar},
            {'name': 'sidecar_records_pre_apply_bytes',
             'ok': bool(original) and original == pre_apply,
             'detail': {'sidecar_original_sha256': original,
                        'pre_apply_sha256': pre_apply}},
            {'name': 'stored_fingerprint_matches_gate',
             'ok': bool(stored) and int(stored.get('edited') or 0) == len(gate_names)
             and str(stored.get('edit_type') or '').lower()
             == str(state['gate'].get('edit_type') or '').lower(),
             'detail': {'stored': ({key: stored.get(key) for key in
                                    ('edited', 'untouched', 'edit_type')} if stored else None),
                        'gate': {'edited': len(gate_names),
                                 'edit_type': state['gate'].get('edit_type')}}},
        ]
        measured = {'codec_fidelity': True,
                    'untouched_byte_identity': None,
                    'fingerprint_recomputes': None,
                    'revert_sidecar_records_pre_apply': checks[1]['ok'],
                    'stored_fingerprint_matches_gate': checks[2]['ok']}
        missing.insert(0, 'untouched-byte identity vs the pristine file: in-place keeps the '
                          'pristine bytes in the sidecar, so gguf-revert is the proof, not a '
                          'second file read')
        evidence.update({'checks': checks, 'measured': measured, 'pair': None,
                         'sidecar_original_sha256': original})
    else:
        pair = verify_pair(str(artifact), str(output))
        untouched = _untouched_identity(artifact, output, exclude)
        checks = [
            {'name': 'fingerprint_recomputes', 'ok': pair.get('verdict') == 'recomputes',
             'detail': pair.get('reason')},
            {'name': 'untouched_bytes_identical',
             'ok': not untouched['changed'] and not untouched['missing'],
             'detail': {'identical': untouched['identical'],
                        'changed': untouched['changed'][:12],
                        'missing': untouched['missing'][:12],
                        'excluded_edited': len(exclude)}},
        ]
        measured = {'codec_fidelity': True,
                    'untouched_byte_identity': checks[1]['ok'],
                    'fingerprint_recomputes': checks[0]['ok']}
        evidence.update({'checks': checks, 'measured': measured, 'pair': pair,
                         'untouched': untouched})
    commands = {
        'board': cli_line('board', '--artifact', output, '--records',
                          state.get('inputs', {}).get('records') or '<records.pt>',
                          '--base', artifact),
        'serve': 'start llama-server on the edited artifact and record the full argv',
    }
    evidence['commands'] = commands
    risks = []
    for check in evidence['checks']:
        if not check['ok']:
            risks.append({'code': 'CHECK_FAILED', 'severity': 'high',
                          'detail': f"{check['name']}: {check['detail']}"})
    options = [
        _option('certify', 'emit the certification packet with measured vs missing fields',
                _cmd(workdir, 'certify'),
                'all local checks are in; the packet keeps the boundary honest'),
        _option('board', 'measure behaviour on a real serve (GATED)', commands['board'],
                'the removal claim needs answers, not codec numbers',
                gated='needs llama-server and the harmful rows; the route never starts a server'),
    ]
    decision = _classify(options, risks, 'certify')
    report = _base_report('verify', state, workdir, artifact_view(output), evidence,
                          risks, decision, started)
    report['ok'] = all(check['ok'] for check in evidence['checks'])
    state.pop('certify', None)
    state['verify'] = {'at': _now(), 'report': 'verify.json', 'ok': report['ok'],
                       'summary': {'checks': {check['name']: check['ok']
                                              for check in evidence['checks']},
                                   'measured': evidence['measured'], 'missing': missing}}
    _record_step(state, 'verify', report, options)
    return report


# ---------- certify ----------

def _certify(workdir, artifact, settings, args, state, started):
    import json

    from .answers import atomic_json
    from .edit import artifact_view
    from .errors import RouteRefused
    from .nextsteps import cli_line

    if not state or not state.get('verify'):
        raise RouteRefused('ROUTE_ORDER', 'certify needs verify in the same workdir first')
    output = Path(state['apply']['output'])
    view = artifact_view(output)
    gate_path = Path(workdir) / 'gate.json'
    gate = json.loads(gate_path.read_text()) if gate_path.is_file() else {}
    plan_path = Path(workdir) / 'plan.json'
    plan = json.loads(plan_path.read_text()) if plan_path.is_file() else {}
    method = ((plan.get('evidence') or {}).get('method') or {})
    winner = ((gate.get('evidence') or {}).get('winner'))
    attempt = next((row for row in (gate.get('evidence') or {}).get('attempts') or []
                    if row.get('edit_type') == winner), {})
    packet = {
        'schema': 'jdewitt-route-cert-v1',
        'artifact': {'source': str(artifact), 'source_sha256': state.get('artifact_sha256'),
                     'edited': str(output),
                     'edited_sha256': state.get('apply', {}).get('output_sha256'),
                     'architecture': view.get('architecture'), 'hidden': view.get('hidden'),
                     'layers': view.get('layers'), 'tensors': view.get('tensors'),
                     'source_types': view.get('source_types'), 'bytes': view.get('bytes')},
        'edit': {'method': state.get('plan', {}).get('method'),
                 'edit_type': state.get('gate', {}).get('edit_type'),
                 'floor': state.get('gate', {}).get('floor'),
                 'gate': state.get('gate', {}).get('summary'),
                 'direction_source': {'source': method.get('source'),
                                      'kinds': method.get('kinds'),
                                      'broadcast': method.get('broadcast'),
                                      'provenance': method.get('provenance')},
                 'settings': state.get('settings'),
                 'delivered': {'min_cos': attempt.get('min_cos'),
                               'mean_cos': attempt.get('mean_cos'),
                               'tensors': attempt.get('delivered', {}).get('tensors')}},
        'measured': state['verify'].get('summary', {}).get('measured'),
        'unmeasured': state['verify'].get('summary', {}).get('missing'),
        'removal_measured': False,
        'capability_measured': False,
        'claims': {'removal': 'not measured: codec fidelity and byte facts, not answers',
                   'retention': 'not measured: no capability suite was run',
                   'serve_config': 'not pinned: no server was started'},
        'next': {
            'board': cli_line('board', '--artifact', output, '--records',
                              state.get('inputs', {}).get('records') or '<records.pt>',
                              '--base', artifact),
            'reproduce': cli_line('verify', '--pristine', artifact, '--edited', output),
        },
    }
    atomic_json(Path(workdir) / 'packet.json', _clean(packet))
    options = [
        _option('board', 'measure behaviour on a real serve (GATED)', packet['next']['board'],
                'the only remaining measurement is answers', gated=True),
        _option('done', 'stop here: the local edit is measured and the boundary is labelled',
                cli_line('gguf-inspect', output),
                'no local step remains; inspect or ship the artifact'),
    ]
    decision = _classify(options, [], 'done')
    report = _base_report('certify', state, workdir, view,
                          {'packet': str(Path(workdir) / 'packet.json'), 'summary': packet},
                          [], decision, started)
    report['ok'] = True
    state['certify'] = {'at': _now(), 'report': 'certify.json',
                        'packet': str(Path(workdir) / 'packet.json'),
                        'summary': {'removal_measured': False,
                                    'measured': packet['measured']}}
    _record_step(state, 'certify', report, options)
    return report


# ---------- status ----------

def _status(workdir, artifact, settings, args, state, started):
    import json

    from .edit import artifact_view
    from .errors import RouteRefused

    if not state:
        raise RouteRefused('ROUTE_ORDER', 'no route state in this workdir; run survey first')
    done = [row['step'] for row in state.get('history') or []]
    last = done[-1] if done else 'survey'
    last_report = Path(workdir) / f'{last}.json'
    if last_report.is_file():
        decision = json.loads(last_report.read_text()).get('decision') or {}
    else:
        decision = {'recommended': 'survey', 'options': [], 'blocked': []}
    report = _base_report('status', state, workdir, artifact_view(artifact),
                          {'steps_done': done, 'state': _state_view(state, workdir)},
                          [], decision, started)
    report['step'] = 'status'
    return report


# ---------- entry ----------

def _status_fresh(workdir, artifact, settings, started):
    """First contact: a read-only status for a workdir that has no route state.

    An agent's natural first call is `route --gguf ART` with no step. That used to
    be ROUTE_ORDER; it is now the map: the artifact view, the stage list, and the
    exact survey/one-shot commands. Nothing is written and no state is created.
    """
    from .edit import artifact_view
    from .nextsteps import cli_line

    view = artifact_view(artifact)
    options = [
        _option('survey', 'read the artifact, inputs and risks before any write',
                _cmd(workdir, 'survey', gguf=artifact),
                'this workdir has no state yet; survey starts the route',
                expected='tensor-type census, writeback kinds, inputs, separability, risks'),
        _option('edit', 'one-shot: decide and print the plan (add --apply to write)',
                cli_line('edit', '--gguf', artifact),
                'the one-shot path when the decision does not need inspecting'),
    ]
    decision = _classify(options, [], 'survey')
    report = _base_report('status', None, workdir, view,
                          {'steps_done': [], 'route_steps': list(STEPS),
                           'artifact_bytes': artifact.stat().st_size,
                           'artifact_sha256': None,
                           'identity_note': 'status does not hash the artifact; survey '
                                            'records the sha256 once'},
                          [], decision, started)
    report['step'] = 'status'
    return report


def route_command(args):
    """CLI handler: run one route step and return its one JSON report."""
    from .errors import RouteRefused

    started = time.monotonic()
    step = args.step or 'status'
    if step not in STEPS and step != 'status':
        raise RouteRefused('ROUTE_INPUT', f'unknown route step {step!r}; steps: {" ".join(STEPS)}')
    workdir = Path(args.workdir) if args.workdir else None
    if workdir is None and args.gguf is not None:
        workdir = _default_workdir(args.gguf)
    if workdir is None:
        raise RouteRefused('ROUTE_INPUT', 'route needs --gguf on the first step or --workdir')
    state = _load_state(workdir)
    artifact = Path(args.gguf) if args.gguf is not None else (
        Path(state['artifact']) if state and state.get('artifact') else None)
    if artifact is None:
        raise RouteRefused('ROUTE_INPUT', 'no artifact: pass --gguf or point --workdir at a state')
    settings = _settings(args, state)
    if step == 'status' and state is None:
        return _clean(_status_fresh(workdir, artifact, settings, started))
    if step == 'survey':
        if state is None:
            state = {'schema': SCHEMA_STATE, 'artifact': str(artifact.resolve()),
                     'artifact_sha256': _sha(artifact),
                     'artifact_bytes': artifact.stat().st_size,
                     'workdir': str(workdir.resolve()), 'settings': settings,
                     'inputs': _inputs(args, None),
                     'input_sha256': _input_shas(_inputs(args, None)), 'history': []}
        # A survey starts the route over: every downstream record described an
        # artifact/records state that this survey may have just replaced.
        for key in ('plan', 'gate', 'apply', 'verify', 'certify'):
            state.pop(key, None)
        report = _survey(workdir, artifact, settings, state, started)
        state['settings'] = settings
        state['inputs'] = _inputs(args, state)
        state['input_sha256'] = _input_shas(state['inputs'])
        state['survey'] = {'at': _now(), 'report': 'survey.json',
                           'summary': {'tensors': report['artifact']['tensors'],
                                       'types': sorted(report['evidence']['tensor_types']),
                                       'hidden': report['artifact']['hidden'],
                                       'layers': report['artifact']['layers']}}
        _record_step(state, 'survey', report, report['decision']['options'])
    elif step == 'plan':
        report = _plan(workdir, artifact, settings, args, state, started)
    elif step == 'gate':
        report = _gate(workdir, artifact, settings, args, state, started)
    elif step == 'apply':
        report = _apply(workdir, artifact, settings, args, state, started)
    elif step == 'verify':
        report = _verify(workdir, artifact, settings, args, state, started)
    elif step == 'certify':
        report = _certify(workdir, artifact, settings, args, state, started)
    else:
        report = _status(workdir, artifact, settings, args, state, started)
    _save_state(workdir, state)
    _write_report(workdir, report['step'], report)
    return _clean(report)
