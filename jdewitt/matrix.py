# SPDX-License-Identifier: AGPL-3.0-or-later
"""Count the Phase 7 packets into one matrix, and refuse the shortcuts that fake it.

"Five lineages certified" is a count, not five sentences. This command reads the
manifest that registers each lineage (model, pinned revision, quant, base and
artifact digests, packet, head-to-head side) and the one packet per arm that
``release-certify`` assembled, and emits one object where every row carries a
status from a closed vocabulary:

  certified                 every axis measured at the pinned serve config, all floors met
  certified_at_other_config every axis present and consistent, but a wave was served
                            outside the pinned config, so the row is not the pass
  UNMEASURED                no packet (or no merged artifact to certify); exit 0
  failed(<axis>)            measured at the pinned config and outside a registered band
  invalid(<reason>)         the row climbed a shortcut; exit non-zero, row named

The three shortcuts the roadmap names are rows here, not prose: a suite shorter
than the registered one (``suite_narrowed``), a generation cap below the
registered one (``cap_lowered``), and a fitted scale past the registered grid
(``scale_past_grid``). A head-to-head whose two sides were not served the same
ctx/parallel/cap is ``head_to_head_config`` (R9): two configs are not one row.
A packet that does not bind the registered bytes (or binds a 16-bit parent, or a
re-quantized copy) is ``artifact_*``.

Nothing here measures a model. Every number is read from a packet (or, for a
head-to-head reference with no packet, recomputed from the saved wave with the
same graders ``release-certify`` uses) and carried into the matrix unchanged, so
the matrix can be audited against the files it names.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, NoReturn

from .answers import atomic_json
from .errors import MatrixRefused

SCHEMA_VERSION = 1
AXES = ('removal', 'held_out', 'retention')
STATUS_CERTIFIED = 'certified'
STATUS_OTHER_CONFIG = 'certified_at_other_config'
STATUS_UNMEASURED = 'UNMEASURED'

# The five lineages the roadmap counts, and the checkpoint model_type that makes
# each one distinct. A manifest row may not rename these; a duplicate model_type
# is one lineage counted twice, never a sixth.
REQUIRED_LINEAGES = (
    {'lineage': 'qwen3-4b', 'model_type': 'qwen3'},
    {'lineage': 'qwen25-1.5b', 'model_type': 'qwen2'},
    {'lineage': 'spark-1.7b', 'model_type': 'spark2_5'},
    {'lineage': 'minicpm-v47', 'model_type': 'minicpmv4_7'},
    {'lineage': 'inkling-2-mini', 'model_type': 'inkling2_loop'},
)

# A served artifact in one of these families is the parent checkpoint, not an
# edit of the bytes the user already serves.
PARENT_QUANTS = {'F16', 'BF16', 'F32', 'FP16', 'FP32'}

_CTX = re.compile(r'(?<![\w-])(-c|--ctx-size|ctx-size|ctx_size)(?:=|\s+)(\d+)')
_PARALLEL = re.compile(r'(?<![\w-])(--parallel|-np|parallel)(?:=|\s+)(\d+)')
_CAP = re.compile(r'(?<![\w-])(--max-tokens|--n-predict|max_tokens|n_predict)(?:=|\s+)(\d+)')
_ARTIFACT = re.compile(r'(?:^|\s)(?:-m|--model)(?:=|\s+)(\S+)')


def _fail(code, message) -> NoReturn:
    raise MatrixRefused(code, message)


def _read_json(path, what):
    path = Path(path).expanduser()
    if not path.is_file():
        _fail('MATRIX_MANIFEST_INVALID', f'{what}: {path} does not exist')
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        _fail('MATRIX_MANIFEST_INVALID', f'{what}: {path} is not JSON: {error}')


def _digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _serve_facts(text) -> dict:
    """ctx / parallel / cap / artifact from a recorded serve argv, tolerantly.

    Waves in this clone write two spellings of the same serve config (``-c 8192
    --parallel 2`` and ``ctx-size 8192 parallel 2``); a comparability test that
    only understood the first would call an equal config different.
    """
    text = text or ''
    facts: dict[str, Any] = {'artifact': None, 'ctx': None, 'parallel': None, 'cap': None}
    for key, pattern in (('ctx', _CTX), ('parallel', _PARALLEL), ('cap', _CAP)):
        found = pattern.search(text)
        if found:
            facts[key] = int(found.group(2))
    artifact = _ARTIFACT.search(text)
    if artifact:
        facts['artifact'] = artifact.group(1)
    return facts


def _wave_facts(wave: dict) -> dict:
    facts = _serve_facts(wave.get('serve_config'))
    structured = wave.get('serve_structured') or {}
    for key, field in (('ctx', 'ctx'), ('parallel', 'parallel'), ('cap', 'max_tokens')):
        if structured.get(field) is not None:
            facts[key] = int(structured[field])
    if wave.get('max_tokens') is not None:
        facts['cap'] = int(wave['max_tokens'])
    if wave.get('suite'):
        facts['suite'] = wave['suite']
    facts['argv'] = wave.get('serve_config')
    return facts


def _triplet(facts: dict) -> dict:
    return {key: facts.get(key) for key in ('ctx', 'parallel', 'cap')}


def _argv_of(packet: dict, axis: str):
    configs = packet.get('configs') or {}
    entry = configs.get(axis)
    return entry if isinstance(entry, str) else (entry or {}).get('argv')


def _packet_facts(packet: dict, axis: str) -> dict:
    """Serve facts for one axis of a packet; the packet's own facts win over argv.

    ``measured_axes`` is the certifier's record of what the wave actually ran at.
    Older packets do not carry it, so the argv is parsed instead and a missing cap
    stays missing instead of being guessed from the registered floor.
    """
    facts = _serve_facts(_argv_of(packet, axis))
    measured = (packet.get('measured_axes') or {}).get(axis) or {}
    for key, field in (('ctx', 'ctx'), ('parallel', 'parallel'), ('cap', 'cap')):
        if measured.get(field) is not None:
            facts[key] = int(measured[field])
    if measured.get('suite'):
        facts['suite'] = measured['suite']
    elif (packet.get('suites') or {}).get(axis):
        suite = packet['suites'][axis]
        facts['suite'] = suite.get('path') if isinstance(suite, dict) else suite
    facts['argv'] = _argv_of(packet, axis)
    return facts


def _inventory(block, inventory):
    return ((block.get('inventories') or {}).get(inventory) or {})


def _packet_counts(packet: dict) -> dict:
    """Row counts the packet proves, per axis, plus where they disagree.

    The count is taken from both wording inventories and both retention kinds, so
    a packet with one side trimmed is a disagreement here rather than a smaller
    number that slips under the floor.
    """
    harmful = (packet.get('removal') or {}).get('harmful40') or {}
    held = (packet.get('removal') or {}).get('held_out_120') or {}
    counts = {'removal': [], 'held_out': [], 'retention': []}
    for block, key in ((harmful, 'removal'), (held, 'held_out')):
        for inventory in ('shipped', 'wide'):
            total = _inventory(block, inventory).get('total')
            if total is not None:
                counts[key].append(int(total))
    retention = packet.get('retention') or {}
    kind_total = 0
    for kind in ('math', 'code'):
        all_rows = ((retention.get(kind) or {}).get('strict') or {}).get('all_rows') or {}
        before = str(all_rows.get('before') or '')
        if '/' in before:
            kind_total += int(before.split('/')[-1])
    if kind_total:
        counts['retention'].append(kind_total)
    measured = packet.get('measured_axes') or {}
    for axis in AXES:
        if (measured.get(axis) or {}).get('rows') is not None:
            counts[axis].append(int(measured[axis]['rows']))
    result = {}
    for axis, values in counts.items():
        result[axis] = {'rows': max(values) if values else None,
                        'disagree': len(set(values)) > 1, 'observed': values}
    return result


def _numeric_leaves(value, prefix='') -> dict:
    """Every number in the packet, keyed by its JSON path.

    The matrix is the one place the packet's numbers are counted, so it carries
    all of them: a reader can diff a matrix row against the packet without
    reopening it, and a number that moved shows up as a changed key.
    """
    out: dict[str, Any] = {}
    if isinstance(value, bool):
        return out
    if isinstance(value, (int, float)):
        out[prefix] = value
    elif isinstance(value, dict):
        for key in sorted(value):
            out.update(_numeric_leaves(value[key], f'{prefix}.{key}' if prefix else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            out.update(_numeric_leaves(item, f'{prefix}[{index}]'))
    return out


def _retention_losses(packet: dict) -> dict:
    losses = {}
    for kind in ('math', 'code'):
        block = (packet.get('retention') or {}).get(kind) or {}
        strict = block.get('strict') or {}
        before, after = strict.get('before'), strict.get('after')
        p = block.get('paired_sign_flip_p_two_sided')
        losses[kind] = {
            'before': before, 'after': after,
            'drop': (None if before is None or after is None else int(after) - int(before)),
            'paired_rows': strict.get('paired_rows'),
            'paired_p_two_sided': p,
            'significant': (p is not None and float(p) < 0.05),
            'median_answer_chars': block.get('median_answer_chars'),
            'base_median_answer_chars': block.get('base_median_answer_chars'),
            'length_collapse_flag': block.get('length_collapse_flag'),
            'lost': (None if before is None or after is None else int(after) < int(before)),
        }
    return losses


def _reference_side(entry, packet_dir) -> dict | None:
    """The head-to-head's other side: a packet when one exists, else its waves.

    A competitor artifact has no packet of ours (no shape audit, no drift, no
    fidelity), so a reference is measured from the waves we served ourselves with
    the same graders the certifier uses. That is a measurement, not their claim.
    """
    if not isinstance(entry, dict):
        return None
    side = {'label': entry.get('label'), 'model': entry.get('model')}
    packet_path = entry.get('packet')
    if packet_path and Path(packet_path).expanduser().is_file():
        packet = _read_json(packet_path, f'head_to_head.packet ({packet_path})')
        side.update({'source': 'packet', 'packet': str(packet_path),
                     'retention': _retention_losses(packet),
                     'configs': {axis: _packet_facts(packet, axis) for axis in AXES}})
        return side
    waves = entry.get('waves') or {}
    if not waves:
        return side | {'source': None, 'retention': None, 'configs': {}}
    try:
        from .certify import _retention_kind, _wave
    except ImportError:  # pragma: no cover - the module ships together
        return side | {'source': 'waves', 'retention': None, 'configs': {}}
    base_path = entry.get('retention_base_wave')
    if not base_path:
        return side | {'source': 'waves', 'retention': None, 'configs': {}}
    if packet_dir:
        base_path = Path(packet_dir) / str(base_path)
    base_rows = [row for row in _wave(base_path, 'retention.base')['rows']
                 if row.get('status') == 'ok']
    retention = {}
    configs = {}
    for axis in AXES:
        if not waves.get(axis):
            continue
        wave = _wave(waves[axis], f'head_to_head.{axis}')
        configs[axis] = _wave_facts(wave['wave'])
        if axis != 'retention':
            continue
        rows = [row for row in wave['rows'] if row.get('status') == 'ok']
        try:
            retention = {kind: _retention_kind(base_rows, rows, kind) for kind in ('math', 'code')}
        except Exception:  # noqa: BLE001 - an incomplete reference is reported, never guessed
            retention = {}
    side.update({'source': 'waves', 'packet': None, 'retention': retention, 'configs': configs,
                 'waves': waves})
    return side


def _reference_losses(side: dict | None) -> dict:
    if not side or not side.get('retention'):
        return {}
    out = {}
    for kind in ('math', 'code'):
        block = side['retention'].get(kind) or {}
        strict = block.get('strict') or {}
        out[kind] = {'before': strict.get('before'), 'after': strict.get('after'),
                     'paired_rows': strict.get('paired_rows'),
                     'paired_p_two_sided': block.get('paired_sign_flip_p_two_sided'),
                     'median_answer_chars': block.get('median_answer_chars')}
    return out


def _loss_line(lineage: str, axis: str, ours: dict, reference: dict | None) -> str:
    left = f'{ours.get("before")} -> {ours.get("after")}'
    right = ('unmeasured' if not reference or reference.get('before') is None
             else f'{reference.get("before")} -> {reference.get("after")}')
    verdict = []
    if reference and reference.get('before') is not None:
        ours_drop = None if ours.get('drop') is None else -int(ours['drop'])
        ref_drop = int(reference['before']) - int(reference['after']) \
            if reference.get('after') is not None else None
        verdict.append(f'more loss than the reference ({ours_drop} vs {ref_drop})'
                       if ours_drop is not None and ref_drop is not None and ours_drop > ref_drop
                       else 'no worse than the reference')
    p = ours.get('paired_p_two_sided')
    if p is not None:
        verdict.append(f'paired p {p}' + ('' if ours.get('significant') else ' (not significant)'))
    if ours.get('length_collapse_flag'):
        verdict.append('length collapse flag set')
    return (f'{lineage}  {axis} {left}    reference {right}    ' + '; '.join(verdict)).strip()


def _certify_command(entry) -> str:
    spec = entry.get('spec')
    packet = entry.get('packet')
    if not packet:
        return 'uv run jdewitt release-matrix --manifest MANIFEST.json --out matrix.json --table matrix.md'
    spec = spec or str(Path(packet).with_name('spec.json'))
    return f'uv run jdewitt release-certify --spec {spec} --out {packet}'


def _model_type_problem(entry, model_type) -> str | None:
    config_path = (entry.get('model') or {}).get('config')
    if not config_path:
        return None
    path = Path(str(config_path)).expanduser()
    if not path.is_file():
        return None
    try:
        config = json.loads(path.read_text())
    except json.JSONDecodeError:
        return f'{config_path} is not JSON, so the registered model_type {model_type!r} is unverified'
    actual = config.get('model_type')
    if actual and model_type and actual != model_type:
        return (f'registered model_type {model_type!r} but {config_path} says {actual!r}; '
                'a row is counted on the checkpoint\'s model_type, not its name')
    return None


def _binding_problems(entry, packet) -> list[dict]:
    problems = []
    artifact = entry.get('artifact') or {}
    base = entry.get('base') or {}
    digest = packet.get('artifact_identity_digest')
    if artifact.get('identity_digest') and digest != artifact['identity_digest']:
        problems.append({'code': 'artifact_mismatch',
                         'message': (f'packet binds artifact identity {digest}, the manifest '
                                     f'registers {artifact["identity_digest"]}')})
    recorded = artifact.get('sha256')
    packet_sha = packet.get('artifact_sha256')
    if recorded and packet_sha and recorded != packet_sha:
        problems.append({'code': 'artifact_mismatch',
                         'message': (f'packet artifact_sha256 {packet_sha} is not the registered '
                                     f'{recorded}')})
    if recorded and not packet_sha:
        problems.append({'code': 'artifact_unbound',
                         'message': ('the manifest registers artifact sha256 '
                                     f'{recorded}, the packet records none; the bytes are not bound')})
    quant = artifact.get('quant_type')
    if quant and str(quant).upper() in PARENT_QUANTS:
        problems.append({'code': 'artifact_is_parent',
                         'message': f'row artifact quant {quant} is a 16-bit parent, not an edit'})
    if packet_sha and packet.get('base_sha256') and packet_sha == packet['base_sha256']:
        problems.append({'code': 'artifact_is_parent',
                         'message': 'packet artifact sha256 equals its base sha256: the served file is the parent'})
    base_quant = base.get('quant_type')
    if quant and base_quant and str(quant).upper() != str(base_quant).upper():
        problems.append({'code': 'artifact_requantized',
                         'message': (f'row artifact is {quant} while its base is {base_quant}: '
                                     'a re-quantized copy is not an edit of the served bytes')})
    return problems


def _count_problems(entry, packet, registered) -> list[dict]:
    problems = []
    counts = _packet_counts(packet)
    floors = registered.get('suite_rows') or {}
    for axis in AXES:
        observed = counts[axis]
        if observed['disagree']:
            problems.append({'code': 'row_count_disagreement',
                             'message': (f'{axis}: the packet records {observed["observed"]} rows '
                                         'on different inventories/axes; the count is ambiguous')})
        floor = floors.get(axis)
        if floor is not None and observed['rows'] is not None and observed['rows'] < int(floor):
            problems.append({'code': 'suite_narrowed',
                             'message': (f'{axis}: the packet covers {observed["rows"]} rows, the '
                                         f'registered suite has {floor}; a pass at a smaller suite '
                                         'is not a pass')})
    caps = registered.get('cap') or {}
    for axis in AXES:
        cap = _packet_facts(packet, axis).get('cap')
        floor = caps.get(axis)
        if floor is None:
            continue
        if cap is None:
            problems.append({'code': 'cap_unrecorded',
                             'message': (f'{axis}: the packet records no generation cap, so the '
                                         f'registered {floor} cannot be checked')})
        elif int(cap) < int(floor):
            problems.append({'code': 'cap_lowered',
                             'message': (f'{axis}: served at cap {cap}, the registered cap is {floor}; '
                                         'grading at a smaller cap is not a pass')})
    suites = registered.get('suites') or {}
    for axis in AXES:
        expected = suites.get(axis)
        observed = _packet_facts(packet, axis).get('suite')
        if not expected or not observed:
            continue
        if Path(str(expected)).name != Path(str(observed)).name:
            problems.append({'code': 'suite_substituted',
                             'message': (f'{axis}: the packet covers {Path(str(observed)).name!r}, '
                                         f'the registered suite is {Path(str(expected)).name!r}')})
    return problems


def _head_to_head(entry, packet):
    side = _reference_side(entry.get('head_to_head'), None)
    if not side or not side.get('configs'):
        return None, [], side
    problems = []
    axes = {}
    for axis in AXES:
        reference_facts = side['configs'].get(axis)
        if not reference_facts:
            continue
        ours = _packet_facts(packet, axis)
        left, right = _triplet(ours), _triplet(reference_facts)
        missing = [key for key, value in right.items() if value is None]
        same = left == right
        axes[axis] = {'ours': left, 'reference': right, 'same_config': same,
                      'ours_argv': ours.get('argv'), 'reference_argv': reference_facts.get('argv')}
        if missing:
            problems.append({'code': 'head_to_head_config',
                             'message': (f'{axis}: the reference side records no '
                                         f'{", ".join(missing)}; comparability (R9) is not established. '
                                         f'ours: {ours.get("argv")!r} | reference: '
                                         f'{reference_facts.get("argv")!r}')})
        elif not same:
            problems.append({'code': 'head_to_head_config',
                             'message': (f'{axis}: the two sides were not served the same '
                                         f'ctx/parallel/cap (R9). ours: {left} {ours.get("argv")!r} | '
                                         f'reference: {right} {reference_facts.get("argv")!r}')})
    head = {'label': side.get('label'), 'model': side.get('model'),
            'reference_source': side.get('source'), 'axes': axes}
    return head, problems, side


def _packet_summary(packet: dict) -> dict:
    """The packet's own numbers, lifted into the row so the row reads alone."""
    harmful = (packet.get('removal') or {}).get('harmful40') or {}
    held = (packet.get('removal') or {}).get('held_out_120') or {}
    removal = {'harmful40': {name: _inventory(harmful, name).get('n_m')
                             for name in ('shipped', 'wide')},
               'held_out_120': {name: _inventory(held, name).get('n_m')
                                for name in ('shipped', 'wide')},
               'held_out_families': packet.get('held_out_families')}
    competitor = held.get('competitor')
    if competitor:
        removal['competitor'] = {'label': competitor.get('label'),
                                 'wave': competitor.get('wave'),
                                 'inventories': {name: (competitor.get('inventories') or {})
                                                 .get(name, {}).get('n_m')
                                                 for name in ('shipped', 'wide')},
                                 'mcnemar_wide': competitor.get('mcnemar_wide')}
    fidelity = packet.get('realized_fidelity') or {}
    return {
        'removal': removal,
        'retention': _retention_losses(packet),
        'drift': packet.get('drift'),
        'fidelity': {key: fidelity.get(key) for key in
                     ('min_cos', 'mean_cos', 'sampled', 'chosen_type', 'arm_measurement')},
        'shape_verdict_counts': (packet.get('shape') or {}).get('verdict_counts'),
        'pinned_cert_match': packet.get('pinned_cert_match'),
        'disagreements': packet.get('disagreements'),
        'evidence_gaps': packet.get('evidence_gaps'),
    }


def _band_failures(packet: dict, registered) -> list[dict]:
    gates = registered.get('gates') or {}
    failures = []
    harmful = (packet.get('removal') or {}).get('harmful40') or {}
    held = (packet.get('removal') or {}).get('held_out_120') or {}
    removal_floor = gates.get('removal_wide_floor')
    removed = _inventory(harmful, 'wide').get('removed')
    if removal_floor is not None and removed is not None and int(removed) < int(removal_floor):
        failures.append({'axis': 'removal',
                         'message': (f'removed {removed}, the registered floor is {removal_floor}; '
                                     'reported as a failure, never rounded up to a pass')})
    held_floor = gates.get('held_out_wide_floor')
    held_removed = _inventory(held, 'wide').get('removed')
    if held_floor is not None and held_removed is not None and int(held_removed) < int(held_floor):
        failures.append({'axis': 'held_out',
                         'message': f'removed {held_removed} of {held_floor} held-out families'})
    drift = packet.get('drift') or {}
    budget = gates.get('drift_budget', drift.get('budget'))
    delta = drift.get('control_delta_ln_ppl')
    if budget is not None and delta is not None and abs(float(delta)) > float(budget):
        failures.append({'axis': 'drift',
                         'message': f'control delta {delta} is outside the {budget} budget'})
    fidelity = packet.get('realized_fidelity') or {}
    cos_floor = gates.get('fidelity_min_cos')
    min_cos = fidelity.get('min_cos')
    if cos_floor is not None and min_cos is not None and fidelity.get('arm_measurement') is not False \
            and float(min_cos) < float(cos_floor):
        failures.append({'axis': 'fidelity',
                         'message': f'min realized cos {min_cos} is below {cos_floor}'})
    return failures


def _row(entry, registered, expected_type=None) -> dict:
    lineage = entry.get('lineage')
    packet_path = entry.get('packet')
    row: dict[str, Any] = {
        'lineage': lineage,
        'model_type': entry.get('model_type'),
        'required': bool(entry.get('required', True)) or expected_type is not None,
        'model': entry.get('model'),
        'base': entry.get('base'),
        'artifact': entry.get('artifact'),
        'packet': packet_path,
        'spec': entry.get('spec'),
        'status': STATUS_UNMEASURED,
        'status_detail': None,
        'problems': [],
        'gaps': [],
        'numbers': {},
        'losses': {},
        'next_command': _certify_command(entry),
        'recompute_command': _certify_command(entry),
    }
    if not packet_path or not Path(str(packet_path)).expanduser().is_file():
        row['status_detail'] = (f'no packet at {packet_path}' if packet_path
                                else 'no packet path registered')
        if not (entry.get('artifact') or {}).get('sha256') and \
                not (entry.get('artifact') or {}).get('identity_digest'):
            row['gaps'].append('no merged quant artifact registered: '
                               'the base bytes are not an edit')
        spec = entry.get('spec')
        if spec and not Path(str(spec)).expanduser().is_file():
            row['gaps'].append(f'{spec} does not exist yet: write the spec from the arm\'s saved '
                               'waves, then the command below fills the row')
        return row

    packet_file = Path(str(packet_path)).expanduser()
    packet = _read_json(packet_file, f'packet ({packet_path})')
    row['packet_digest'] = _digest(packet_file)
    row['numbers'] = _numeric_leaves(packet)
    row['losses'] = _retention_losses(packet)
    row['packet_numbers'] = _packet_summary(packet)
    row['pinned_cert_match'] = packet.get('pinned_cert_match')

    problems = _binding_problems(entry, packet)
    if expected_type and entry.get('model_type') and entry['model_type'] != expected_type:
        problems.append({'code': 'model_type_mismatch',
                         'message': (f'{lineage} registers model_type {entry["model_type"]!r}; the '
                                     f'lineage is counted on {expected_type!r}')})
    model_type_problem = _model_type_problem(entry, entry.get('model_type'))
    if model_type_problem:
        problems.append({'code': 'model_type_mismatch', 'message': model_type_problem})
    grid = (registered.get('scale_grid') or {}).get('max')
    scale = (entry.get('artifact') or {}).get('scale')
    if scale is None:
        row['gaps'].append('arm scale not registered, so the grid cannot be checked')
    elif grid is not None and float(scale) > float(grid):
        problems.append({'code': 'scale_past_grid',
                         'message': (f'arm scale {scale} is past the registered grid {grid}; a '
                                     'clearance bought by raising the scale is not a pass')})
    problems.extend(_count_problems(entry, packet, registered))
    head, head_problems, side = _head_to_head(entry, packet)
    problems.extend(head_problems)
    if head:
        row['head_to_head'] = head
        row['reference_losses'] = _reference_losses(side)

    match = packet.get('pinned_cert_match') or {}
    at_pinned = all(match.get(axis) is True for axis in AXES)
    failures = _band_failures(packet, registered) if at_pinned else []
    if problems:
        row['status'] = f'invalid({problems[0]["code"]})'
        row['status_detail'] = problems[0]['message']
        row['next_command'] = (f'fix the row ({problems[0]["code"]}) and re-count: '
                              'uv run jdewitt release-matrix --manifest '
                              f'{entry.get("manifest_path", "verification/releases/matrix.manifest.json")} '
                              '--out matrix.json --table matrix.md')
    elif not at_pinned:
        off = {axis: match.get(axis) for axis in AXES}
        row['status'] = STATUS_OTHER_CONFIG
        row['status_detail'] = (f'every axis is measured and consistent, but pinned_cert_match is '
                                f'{off}: the waves were served outside the pinned config, so this row '
                                'is not the pass')
        pinned = registered.get('pinned') or {}
        row['next_command_note'] = (f're-serve the three axes at -c {pinned.get("ctx")} '
                                    f'--parallel {pinned.get("parallel")} cap {pinned.get("cap")} '
                                    '(the gated cert run), then re-certify with the command here')
    elif failures:
        row['status'] = f'failed({failures[0]["axis"]})'
        row['status_detail'] = failures[0]['message']
        row['failures'] = failures
        row['next_command'] = _certify_command(entry)
    else:
        row['status'] = STATUS_CERTIFIED
        row['status_detail'] = 'every axis measured at the pinned serve config and inside its floor'
        verify = packet.get('next_command')
        row['next_command'] = verify or f'uv run jdewitt verify-artifact {packet_path} --name packet.json'
    row['problems'] = problems
    return row


def _missing_row(meta) -> dict:
    return {'lineage': meta['lineage'], 'model_type': meta['model_type'], 'required': True,
            'model': None, 'base': None, 'artifact': None, 'packet': None, 'spec': None,
            'status': STATUS_UNMEASURED,
            'status_detail': 'the manifest registers no row for this required lineage',
            'problems': [], 'gaps': ['not registered in the manifest'], 'numbers': {},
            'losses': {}, 'packet_numbers': None, 'recompute_command': None,
            'next_command': ('register the lineage in verification/releases/matrix.manifest.json '
                             'and run release-certify for it, then re-run release-matrix')}


def _registered_suite_problems(registered) -> list[str]:
    """The registered floors have to be true of the files they name.

    A manifest that declares 241 retention rows while its suite file has 180 is
    the same narrowing one layer up, so it is refused before any row is read.
    """
    problems = []
    floors = registered.get('suite_rows') or {}
    suites = registered.get('suites') or {}
    for axis, path in suites.items():
        if not path:
            continue
        candidate = Path(str(path)).expanduser()
        if not candidate.is_file():
            continue
        rows = sum(1 for line in candidate.read_text().splitlines() if line.strip())
        floor = floors.get(axis)
        if floor is not None and rows < int(floor):
            problems.append(f'{axis}: the registered suite file {path} has {rows} rows, the '
                            f'manifest registers {floor}')
    return problems


def _matrix_table(matrix) -> str:
    registered = matrix['registered']
    pinned = registered.get('pinned') or {}

    def _counts(mapping):
        return ', '.join(f'{axis} {value}' for axis, value in (mapping or {}).items()
                         if isinstance(value, int))

    lines = ['# Phase 7 release matrix', '',
             f'Manifest: `{matrix["manifest"]}`',
             (f'Pinned serve config: ctx {pinned.get("ctx")} / parallel {pinned.get("parallel")} / '
              f'cap {pinned.get("cap")}; registered suites '
              + _counts(registered.get('suite_rows'))
              + '; registered caps ' + _counts(registered.get('cap'))
              + f'; scale grid <= {(registered.get("scale_grid") or {}).get("max")}'),
             (f'complete: {str(matrix["complete"]).lower()}   certified: {matrix["certified"]} of '
              f'{len(matrix["rows"])} required lineages   '
              f'certified_at_other_config: {matrix["counts"]["certified_at_other_config"]}   '
              f'UNMEASURED: {matrix["counts"]["unmeasured"]}   invalid: {matrix["counts"]["invalid"]}   '
              f'failed: {matrix["counts"]["failed"]}'),
             '', '## Lineages', '',
             '| lineage | model_type | model | revision | quant | status | removal (shipped/wide) | held-out | maths | drift | fidelity min cos |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in matrix['rows'] + matrix['spares']:
        model = row.get('model') or {}
        numbers = row.get('packet_numbers') or {}
        removal = (numbers.get('removal') or {}).get('harmful40') or {}
        held = (numbers.get('removal') or {}).get('held_out_120') or {}
        maths = (row.get('losses') or {}).get('math')
        drift = numbers.get('drift') or {}
        fidelity = numbers.get('fidelity') or {}
        lines.append('| ' + ' | '.join(str(item) for item in (
            row['lineage'] + ('' if row['required'] else ' (spare)'),
            row.get('model_type'),
            model.get('name'),
            model.get('revision'),
            (row.get('artifact') or {}).get('quant_type') or (model or {}).get('quant_type'),
            row['status'],
            f'{removal.get("shipped")} / {removal.get("wide")}' if removal else '-',
            held.get('wide') or '-',
            (f'{maths.get("before")} -> {maths.get("after")} (p {maths.get("paired_p_two_sided")})'
             if maths else '-'),
            (f'{drift.get("control_delta_ln_ppl")} in {drift.get("budget")}' if drift else '-'),
            fidelity.get('min_cos') if fidelity else '-')) + ' |')
    lines += ['', '## LOSSES', '',
              'The honest table: what each row lost against its own base, beside the reference '
              'side of the head-to-head where one exists.', '']
    if matrix['losses']['rows']:
        for item in matrix['losses']['rows']:
            lines.append('    ' + item['line'])
    else:
        lines.append('    (no packet carries a retention number yet)')
    lines += ['', '## Head-to-head', '']
    if matrix['head_to_head']:
        for item in matrix['head_to_head']:
            lines.append(f'    {item["lineage"]} vs {item["label"]}: reference measured from '
                         f'{item["reference_source"]}')
            for axis, facts in (item.get('axes') or {}).items():
                state = 'same config' if facts['same_config'] else 'CONFIG MISMATCH (R9)'
                lines.append(f'      {axis}: ours {facts["ours"]} | reference {facts["reference"]} '
                             f'-> {state}')
    else:
        lines.append('    (none registered)')
    lines += ['', '## Refusals', '']
    if matrix['refusals']:
        for item in matrix['refusals']:
            lines.append(f'    {item["lineage"]}: {item["code"]} - {item["message"]}')
    else:
        lines.append('    (none)')
    lines += ['', '## Next commands', '']
    for row in matrix['rows'] + matrix['spares']:
        lines.append(f'    {row["lineage"]} ({row["status"]}): {row["next_command"]}')
    lines += ['',
              'The matrix is a count, not a claim: a row is `certified` only when every axis was '
              'measured at the pinned serve config and inside its registered floor. A row measured '
              'at another config is reported as `certified_at_other_config`, a row with no packet is '
              '`UNMEASURED`, and a row that narrowed its suite, lowered its cap or raised its scale '
              'past the grid is `invalid`, with a non-zero exit. A `failed(<axis>)` row is a '
              'measured result inside a registered band that it missed: it is reported as a failure '
              'and counted as one, never rounded into a pass.', '']
    return '\n'.join(lines)


def release_matrix(manifest_path, out_path=None, table_path=None):
    """Read the manifest and the packets; emit one matrix. No model, no writes on a bad manifest."""
    manifest_file = Path(str(manifest_path)).expanduser()
    manifest = _read_json(manifest_file, 'manifest')
    if manifest.get('schema_version') != SCHEMA_VERSION:
        _fail('MATRIX_MANIFEST_INVALID',
              f'manifest schema_version {manifest.get("schema_version")!r} is not {SCHEMA_VERSION}')
    if manifest.get('kind') != 'release-matrix-manifest':
        _fail('MATRIX_MANIFEST_INVALID',
              f'manifest kind {manifest.get("kind")!r} is not release-matrix-manifest')
    registered = manifest.get('registered') or {}
    for key in ('pinned', 'suite_rows', 'cap', 'suites'):
        if key not in registered:
            _fail('MATRIX_MANIFEST_INVALID',
                  f'registered.{key} is required: a row is checked against a registered floor')
    suite_problems = _registered_suite_problems(registered)
    if suite_problems:
        _fail('MATRIX_MANIFEST_INVALID', '; '.join(suite_problems))
    entries = list(manifest.get('lineages') or []) + list(manifest.get('spares') or [])
    if not entries:
        _fail('MATRIX_MANIFEST_INVALID', 'the manifest registers no lineages')
    seen = set()
    for entry in entries:
        lineage = entry.get('lineage')
        if not lineage:
            _fail('MATRIX_MANIFEST_INVALID', f'a manifest row has no lineage id: {entry!r}')
        if lineage in seen:
            _fail('MATRIX_MANIFEST_INVALID',
                  f'lineage {lineage!r} is registered twice; one lineage is one row')
        seen.add(lineage)
        entry['manifest_path'] = str(manifest_path)
    by_id = {entry['lineage']: entry for entry in entries}

    rows = []
    for meta in REQUIRED_LINEAGES:
        entry = by_id.pop(meta['lineage'], None)
        rows.append(_missing_row(meta) if entry is None
                    else _row(entry, registered, meta['model_type']))
    spares = [_row(entry, registered) for entry in by_id.values()]

    certified = [row for row in rows if row['status'] == STATUS_CERTIFIED]
    distinct = {row['model_type'] for row in certified if row['model_type']}
    complete = (len(certified) == len(REQUIRED_LINEAGES)
                and len(distinct) == len(REQUIRED_LINEAGES))
    counts = {
        'required': len(REQUIRED_LINEAGES),
        'registered_required': sum(1 for row in rows if row.get('model')),
        'certified': len(certified),
        'certified_at_other_config': sum(1 for row in rows
                                         if row['status'] == STATUS_OTHER_CONFIG),
        'unmeasured': sum(1 for row in rows if row['status'] == STATUS_UNMEASURED),
        'invalid': sum(1 for row in rows if row['status'].startswith('invalid(')),
        'failed': sum(1 for row in rows if row['status'].startswith('failed(')),
        'distinct_model_types_certified': len(distinct),
    }
    loss_rows = []
    for row in rows + spares:
        if not row.get('losses'):
            continue
        reference = row.get('reference_losses') or {}
        for label, kind in (('maths', 'math'), ('code', 'code')):
            ours = row['losses'].get(kind) or {}
            if ours.get('before') is None:
                continue
            other = reference.get(kind)
            loss_rows.append({'lineage': row['lineage'], 'axis': label,
                              'ours': {'before': ours.get('before'), 'after': ours.get('after'),
                                       'drop': ours.get('drop'),
                                       'paired_p_two_sided': ours.get('paired_p_two_sided'),
                                       'significant': ours.get('significant'),
                                       'length_collapse_flag': ours.get('length_collapse_flag')},
                              'reference': ({'before': other.get('before'), 'after': other.get('after')}
                                            if other else None),
                              'verdict': ('loss' if ours.get('lost') else 'no loss'),
                              'line': _loss_line(row['lineage'], label, ours, other)})
    head_rows = []
    for row in rows + spares:
        head = row.get('head_to_head')
        if not head:
            continue
        label = head.get('label') or 'reference'
        head_rows.append({'lineage': row['lineage'], 'label': label,
                          'reference_source': head.get('reference_source'),
                          'axes': head.get('axes'),
                          'retention': {
                              label: {'ours': {'before': (row['losses'].get(kind) or {}).get('before'),
                                               'after': (row['losses'].get(kind) or {}).get('after')},
                                      'reference': (row.get('reference_losses') or {}).get(kind)}
                              for label, kind in (('maths', 'math'), ('code', 'code'))},
                          'held_out_competitor': ((row.get('packet_numbers') or {})
                                                  .get('removal') or {}).get('competitor')})
    refusals = [{'lineage': row['lineage'], 'code': problem['code'], 'message': problem['message']}
                for row in rows + spares for problem in row.get('problems') or []]
    invalid_rows = [row['lineage'] for row in rows + spares
                    if row['status'].startswith('invalid(')]
    result = {
        'schema_version': SCHEMA_VERSION,
        'kind': 'release-matrix',
        'manifest': str(manifest_path),
        'manifest_digest': _digest(manifest_file),
        'registered': registered,
        'complete': complete,
        'certified': counts['certified'],
        'counts': counts,
        'rows': rows,
        'spares': spares,
        'losses': {'heading': 'LOSSES', 'rows': loss_rows,
                   'lines': [item['line'] for item in loss_rows]},
        'head_to_head': head_rows,
        'refusals': refusals,
        'valid': not invalid_rows,
        'invalid_rows': invalid_rows,
        'next_command': (f'uv run jdewitt release-matrix --manifest {manifest_path} '
                         '--out matrix.json --table matrix.md'),
        'note': ('One row per registered lineage, one status per row from a closed vocabulary. '
                 'A row at another serve config is certified_at_other_config, not a pass; a row '
                 'with no packet is UNMEASURED and still prints the command that would fill it; '
                 'a narrowed suite, a lowered cap, a scale past the grid or a head-to-head at two '
                 'configs is invalid and exits non-zero. Every number is copied from the packet '
                 'named on the row (or recomputed from the saved wave for a reference with no '
                 'packet); nothing here measures a model.'),
    }
    if out_path:
        out_path = Path(str(out_path)).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(out_path, result)
    if table_path:
        table_path = Path(str(table_path)).expanduser()
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table_path.write_text(_matrix_table(result))
    return result
