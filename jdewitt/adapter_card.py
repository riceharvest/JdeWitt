# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bind a shipped LoRA adapter to its base artifact and label its curve (METHODOLOGY R8).

An adapter is the cheap screening mechanism: it ranks arms without spending a
token. It is NOT the release artifact. On tensor families whose graph the server
transforms at runtime the served LoRA curve is not the export delta, and the
measured consequence on Spark 1.7B Q4_K_M is that the adapter curve overstates
the merged edit (adapter delta_br 0.215 against 0.119 on merged bytes, ratio
0.553; docs/ROADMAP.md, "Where we are short" item 3). A shipped adapter therefore
has to carry, in one machine-readable file beside it:

  * the base artifact it was fitted on (sha256, arch, tensor count) and its own
    identity (sha256, bytes, rank, tensor count, ladder scale);
  * the merged twin's fidelity record exactly as measured (per-tensor delivered
    cos / SNR, artifact sha256, edited/untouched counts, edit fingerprint);
  * its adapter delta_br BESIDE the merged twin's delta_br and their ratio, so no
    reader can quote the adapter number as the edit's effect;
  * `curve_is_not_merged_bytes: true` and the R8 note text.

This command only records; it produces no behaviour numbers. A twin record without
delivered cos, a base binding that disagrees, merged numbers that are absent
without an explicit UNMEASURED label, or an adapter delta_br parked in a field
that names the merged effect are all refused with a named error.
"""
import hashlib
import json
import re
from pathlib import Path

from .answers import atomic_json
from .errors import AdapterCardRefused

CARD_VERSION = 1
KIND = 'adapter-card'

# docs/ROADMAP.md Phase 2: "the top-2 merges when the adapter ranking is marginal
# (adapter_to_merged_ratio near the validity threshold)". tune.py treats a ratio
# inside this band as agreement with the adapter screen; a ratio BELOW the floor
# means the adapter overstated the edit, so ranking is not valid on this artifact.
RANKING_VALIDITY_RATIO_FLOOR = 0.7

# The binding tune writes into every adapter it builds; the card reads it back and
# compares it with the artifact passed as --base. Older adapters without it are
# refused rather than labelled against an unproven base.
ADAPTER_BASE_SHA_KEY = 'jdewitt.base_artifact_sha256'

R8_NOTE = ('R8 — the verdict runs on the merged artifact, not the served adapter: on tensors '
           'the server graph transforms at runtime, a LoRA curve is not the export delta '
           '(docs/METHODOLOGY.md). adapter_delta_br in this card is a screening number; the '
           'edit\'s effect is merged_delta_br, measured on the merged twin.')
CURVE_CLAIM = ('curve_is_not_merged_bytes=true: adapter_delta_br ranks, merged_delta_br ships. '
               'Never quote adapter_delta_br as the edit\'s effect.')

_ADAPTER_PROVENANCE = ('adapter', 'adapter-curve', 'adapter_curve', 'curve', 'lora')
_MERGED_PROVENANCE = ('merged', 'merged-bytes', 'merged_bytes', 'merged-artifact',
                      'merged_artifact', 'merged-twin', 'merged_twin', 'gguf-bytes', 'gguf_bytes')
_UNMEASURED = ('unmeasured', 'un-measured', 'un_measured', 'not measured', 'none')
_SHA_RE = re.compile(r'^[0-9a-f]{64}$')

# Named refusals. verify-artifact reports the same codes, so a doctored card and a
# bad input fail with the same vocabulary.
CURVE_CLAIM_MISSING = 'CURVE_CLAIM_MISSING'
CURVE_CLAIM_FALSE = 'CURVE_CLAIM_FALSE'
ADAPTER_BASE_BINDING_MISSING = 'ADAPTER_BASE_BINDING_MISSING'
ADAPTER_BASE_DIGEST_MISMATCH = 'ADAPTER_BASE_DIGEST_MISMATCH'
TWIN_ARTIFACT_SHA_MISSING = 'TWIN_ARTIFACT_SHA_MISSING'
TWIN_DELIVERED_COS_MISSING = 'TWIN_DELIVERED_COS_MISSING'
TWIN_TENSOR_COUNTS_MISSING = 'TWIN_TENSOR_COUNTS_MISSING'
TWIN_EDIT_FINGERPRINT_MISSING = 'TWIN_EDIT_FINGERPRINT_MISSING'
TWIN_SCALE_MISSING = 'TWIN_SCALE_MISSING'
MERGED_EFFECT_UNMEASURED_UNLABELLED = 'MERGED_EFFECT_UNMEASURED_UNLABELLED'
ADAPTER_DELTA_IN_MERGED_FIELD = 'ADAPTER_DELTA_IN_MERGED_FIELD'
DELTA_BR_RATIO_INCONSISTENT = 'DELTA_BR_RATIO_INCONSISTENT'
RANKING_VALIDITY_INCONSISTENT = 'RANKING_VALIDITY_INCONSISTENT'
ADAPTER_NOT_A_LORA = 'ADAPTER_NOT_A_LORA'
ADAPTER_FACTORS_MISSING = 'ADAPTER_FACTORS_MISSING'
ADAPTER_FACTORS_DISAGREE = 'ADAPTER_FACTORS_DISAGREE'
ADAPTER_RANK_DISAGREE = 'ADAPTER_RANK_DISAGREE'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _dig(value, path):
    current = value
    for part in path.split('.'):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _first(document, paths, default=None):
    for path in paths:
        value = _dig(document, path)
        if value is not None:
            return value
    return default


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_unmeasured(value):
    return isinstance(value, str) and value.strip().lower() in _UNMEASURED


def _gguf_header(path):
    """Metadata scalars and tensor descriptors of a GGUF; header reads only."""
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    meta = {}
    for key, field in reader.fields.items():
        if key.startswith('GGUF.'):
            continue
        try:
            value = field.contents()
        except Exception:
            continue
        if isinstance(value, (str, int, float, bool)):
            meta[key] = value
    tensors = [{'name': tensor.name, 'qtype': tensor.tensor_type.name,
                'shape': [int(x) for x in tensor.shape]} for tensor in reader.tensors]
    return meta, tensors


def _adapter_identity(path):
    """sha256 / bytes / rank / tensor counts, plus the base binding tune wrote."""
    path = Path(path)
    meta, tensors = _gguf_header(path)
    if meta.get('general.type') != 'adapter':
        raise AdapterCardRefused(ADAPTER_NOT_A_LORA,
                                 f'{path.name} is not a LoRA adapter (general.type='
                                 f'{meta.get("general.type")!r})')
    a_names = [t['name'] for t in tensors if t['name'].endswith('.lora_a')]
    b_names = [t['name'] for t in tensors if t['name'].endswith('.lora_b')]
    if not a_names or not b_names:
        raise AdapterCardRefused(ADAPTER_FACTORS_MISSING,
                                 f'{path.name} carries no .lora_a/.lora_b pairs')
    pairs_a = {name[:-len('.lora_a')] for name in a_names}
    pairs_b = {name[:-len('.lora_b')] for name in b_names}
    if pairs_a != pairs_b:
        raise AdapterCardRefused(ADAPTER_FACTORS_DISAGREE,
                                 f'{path.name}: {len(pairs_a - pairs_b)} .lora_a without '
                                 f'.lora_b and {len(pairs_b - pairs_a)} the reverse')
    ranks = {next(t['shape'][-1] for t in tensors if t['name'] == name + '.lora_a')
             for name in pairs_a}
    if len(ranks) != 1:
        raise AdapterCardRefused(ADAPTER_RANK_DISAGREE,
                                 f'{path.name}: adapters mix ranks {sorted(ranks)}')
    binding = next((value for key, value in meta.items()
                    if key.lower().endswith('base_artifact_sha256')), None)
    return {'path': path.name, 'sha256': file_sha256(path), 'bytes': path.stat().st_size,
            'rank': int(next(iter(ranks))), 'tensor_entries': len(tensors),
            'edited_tensors': len(pairs_a), 'bound_base_sha256': binding}


def _delivered_rows(twin):
    """Per-tensor delivered fidelity rows, tolerant about the record's field names."""
    raw = _first(twin, ['delivered.tensors', 'delivered_edit.tensors', 'tensors',
                        'delivered_edit', 'delivered'], default=None)
    if isinstance(raw, dict):
        raw = raw.get('tensors', raw)
        if isinstance(raw, dict):
            raw = [dict(value, name=key) if isinstance(value, dict) else
                   {'name': key, 'cos': value} for key, value in raw.items()]
    if not isinstance(raw, list):
        return []
    rows = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        name = row.get('name', row.get('tensor'))
        cos = row.get('cos', row.get('delivered_cos'))
        snr = row.get('snr', row.get('delivered_snr'))
        if not isinstance(name, str):
            continue
        if not _number(cos):
            raise AdapterCardRefused(TWIN_DELIVERED_COS_MISSING,
                                     f'twin record row {name!r} has no delivered cos')
        rows.append({'name': name, 'cos': cos, 'snr': snr if _number(snr) else None})
    return rows


def _effect_block(twin, paths):
    value = _first(twin, paths, default=None)
    if isinstance(value, dict):
        return value
    if _number(value):
        return {'delta_br': value}
    return {}


def _effect_delta(block):
    for key in ('delta_br', 'delta_br_percent', 'delta_br_value'):
        if _number(block.get(key)):
            return block[key]
    return None


def _merged_provenance(block):
    return next((str(block[key]) for key in ('measured_on', 'measured_on_artifact', 'source')
                 if block.get(key) is not None), None)


def _merged_source_sha(block):
    for key in ('source_artifact_sha256', 'artifact_sha256', 'measured_artifact_sha256'):
        value = block.get(key)
        if isinstance(value, str):
            return value.lower()
    return None


def _provenance_is_adapter(label):
    return bool(label) and label.strip().lower() in _ADAPTER_PROVENANCE


def _provenance_is_merged(label):
    return bool(label) and label.strip().lower() in _MERGED_PROVENANCE


def _refuse_bad_merged(block, adapter_sha, base_sha):
    """A merged effect must name merged bytes as its origin, never the adapter."""
    label = _merged_provenance(block)
    source = _merged_source_sha(block)
    if _provenance_is_adapter(label):
        raise AdapterCardRefused(ADAPTER_DELTA_IN_MERGED_FIELD,
                                 f'merged effect is labelled measured_on={label!r}; an adapter '
                                 f'curve in the merged field is exactly what R8 forbids')
    if source and source in {adapter_sha, base_sha}:
        raise AdapterCardRefused(ADAPTER_DELTA_IN_MERGED_FIELD,
                                 f'merged effect names source sha {source[:12]}…, which is the '
                                 f'adapter/base artifact, not the merged twin')
    return label


def _ratio(merged, adapter):
    if merged is None or adapter is None or adapter == 0:
        return None
    return round(merged / adapter, 3)


def _ranking_valid(ratio, floor=RANKING_VALIDITY_RATIO_FLOOR):
    return ratio is not None and ratio >= floor


def build_adapter_card(adapter, base, twin, out, scale=None):
    """Write the R8 label for one shipped adapter beside it; no behaviour numbers made here."""
    adapter_path, base_path = Path(adapter), Path(base)
    twin_path, out_path = Path(twin), Path(out)
    for label, path in (('--adapter', adapter_path), ('--base', base_path), ('--twin', twin_path)):
        if not path.is_file():
            raise FileNotFoundError(f'{label} {path} does not exist')
    if out_path.exists():
        raise FileExistsError(f'{out_path} exists; nothing is silently clobbered')

    adapter_info = _adapter_identity(adapter_path)
    base_sha = file_sha256(base_path)
    base_meta, base_tensors = _gguf_header(base_path)

    if not adapter_info['bound_base_sha256']:
        raise AdapterCardRefused(ADAPTER_BASE_BINDING_MISSING,
                                 f'{adapter_path.name} carries no {ADAPTER_BASE_SHA_KEY}; rebuild '
                                 f'the adapter with tune so it is bound to the artifact it was '
                                 f'fitted on before labelling it')
    if adapter_info['bound_base_sha256'].lower() != base_sha:
        raise AdapterCardRefused(ADAPTER_BASE_DIGEST_MISMATCH,
                                 f'{adapter_path.name} is bound to base '
                                 f'{adapter_info["bound_base_sha256"][:12]}… but --base is '
                                 f'{base_sha[:12]}…')

    try:
        twin_document = json.loads(twin_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f'--twin {twin_path} is not valid JSON: {error}') from error
    if not isinstance(twin_document, dict):
        raise ValueError(f'--twin {twin_path} must be a JSON object')

    if 'curve_is_not_merged_bytes' in twin_document and twin_document['curve_is_not_merged_bytes'] is not True:
        raise AdapterCardRefused(CURVE_CLAIM_FALSE,
                                 f'{twin_path.name} asserts curve_is_not_merged_bytes='
                                 f'{twin_document["curve_is_not_merged_bytes"]!r}; R8 says the curve '
                                 f'is not the merged bytes')
    twin_sha = _first(twin_document, ['artifact_sha256', 'merged.artifact_sha256',
                                      'merged_twin.artifact_sha256', 'twin.artifact_sha256'])
    if not isinstance(twin_sha, str) or not _SHA_RE.match(twin_sha.lower()):
        raise AdapterCardRefused(TWIN_ARTIFACT_SHA_MISSING,
                                 f'{twin_path.name} carries no 64-hex merged artifact sha256')
    twin_sha = twin_sha.lower()
    twin_base_sha = _first(twin_document, ['base_artifact_sha256', 'base.sha256', 'base_sha256'])
    if isinstance(twin_base_sha, str) and twin_base_sha.lower() != base_sha:
        raise AdapterCardRefused(ADAPTER_BASE_DIGEST_MISMATCH,
                                 f'{twin_path.name} is a twin of base {twin_base_sha[:12]}… but '
                                 f'--base is {base_sha[:12]}…')

    delivered = _delivered_rows(twin_document)
    if not delivered:
        raise AdapterCardRefused(TWIN_DELIVERED_COS_MISSING,
                                 f'{twin_path.name} has no per-tensor delivered cos rows')

    edited = _first(twin_document, ['edited_tensors', 'tensors_edited', 'counts.edited'])
    untouched = _first(twin_document, ['untouched_tensors', 'tensors_untouched', 'counts.untouched'])
    if type(edited) is not int or type(untouched) is not int:
        raise AdapterCardRefused(TWIN_TENSOR_COUNTS_MISSING,
                                 f'{twin_path.name} must record integer edited_tensors and '
                                 f'untouched_tensors for the merged twin')
    fingerprint = _first(twin_document, ['edit_fingerprint', 'fingerprint', 'edit.fingerprint'])
    if fingerprint is None:
        raise AdapterCardRefused(TWIN_EDIT_FINGERPRINT_MISSING,
                                 f'{twin_path.name} carries no edit fingerprint')
    twin_scale = _first(twin_document, ['scale', 'arm.scale', 'adapter.scale'])
    if twin_scale is None and isinstance(fingerprint, dict):
        twin_scale = _first(fingerprint, ['scale', 'strength'])
    scale = scale if scale is not None else twin_scale
    if not _number(scale):
        raise AdapterCardRefused(TWIN_SCALE_MISSING,
                                 f'{twin_path.name} carries no ladder scale; pass --scale or record '
                                 f'scale/edit_fingerprint.scale in the twin record')

    adapter_block = _effect_block(twin_document, ['behaviour.adapter_curve', 'adapter_curve',
                                                  'adapter.behaviour', 'adapter'])
    merged_block = _effect_block(twin_document, ['behaviour.merged_bytes', 'merged_bytes',
                                                 'merged', 'behaviour.merged'])
    adapter_delta = _effect_delta(adapter_block)
    merged_delta = _effect_delta(merged_block)
    status = _first(twin_document, ['merged_effect_status', 'behaviour.merged_bytes.status',
                                    'merged.status'])
    if merged_delta is None:
        if not _is_unmeasured(status):
            raise AdapterCardRefused(
                MERGED_EFFECT_UNMEASURED_UNLABELLED,
                f'{twin_path.name} has no merged delta_br and no explicit UNMEASURED '
                f'merged-effect label; the card will not present adapter numbers as merged ones')
        status = 'UNMEASURED'
        merged_measured_on = None
        merged_source_sha = None
        ratio = None
    else:
        measured_on = _refuse_bad_merged(merged_block, adapter_info['sha256'], base_sha)
        if _is_unmeasured(status):
            raise AdapterCardRefused(
                MERGED_EFFECT_UNMEASURED_UNLABELLED,
                f'{twin_path.name} records merged_delta_br {merged_delta} but labels the merged '
                f'effect {status!r}; one of the two is wrong')
        status = 'MEASURED'
        merged_measured_on = measured_on or 'merged-bytes'
        merged_source_sha = _merged_source_sha(merged_block) or twin_sha
        ratio = _ratio(merged_delta, adapter_delta)
        recorded_ratio = _first(twin_document, ['adapter_to_merged_ratio', 'ratio.adapter_to_merged'])
        if _number(recorded_ratio) and ratio is not None and abs(recorded_ratio - ratio) > 0.002:
            raise AdapterCardRefused(
                DELTA_BR_RATIO_INCONSISTENT,
                f'{twin_path.name} records adapter_to_merged_ratio {recorded_ratio}, but '
                f'{merged_delta}/{adapter_delta} = {ratio}; the merged field may hold the adapter number')
    ranking_valid = _ranking_valid(ratio)

    cosines = [row['cos'] for row in delivered]
    snrs = [row['snr'] for row in delivered if row['snr'] is not None]
    card = {
        'version': CARD_VERSION,
        'kind': KIND,
        'generated_by': 'jdewitt adapter-card',
        'adapter': {**adapter_info, 'scale': scale},
        'base': {'path': base_path.name, 'sha256': base_sha,
                 'arch': base_meta.get('general.architecture'),
                 'tensor_count': len(base_tensors)},
        'merged_twin': {
            'artifact_sha256': twin_sha,
            'record_sha256': file_sha256(twin_path),
            'record_path': twin_path.name,
            'edited_tensors': edited,
            'untouched_tensors': untouched,
            'edit_fingerprint': fingerprint,
            'delivered': {'tensors': delivered, 'sampled': len(delivered),
                          'min_cos': min(cosines), 'min_snr': min(snrs) if snrs else None},
            'provenance': _first(twin_document, ['provenance', 'source', 'note']),
        },
        'adapter_delta_br': adapter_delta,
        'merged_delta_br': merged_delta,
        'adapter_to_merged_ratio': ratio,
        'ranking_valid_on_this_artifact': ranking_valid,
        'ranking_validity_ratio_floor': RANKING_VALIDITY_RATIO_FLOOR,
        'merged_effect_status': status,
        'merged_measured_on': merged_measured_on,
        'merged_source_sha256': merged_source_sha,
        'curve_is_not_merged_bytes': True,
        'r8_note': R8_NOTE,
        'claim_limit': CURVE_CLAIM,
        'note': ('The adapter ranks; the merged twin ships. Every number here is copied from the '
                 'twin record; this card produces none. ranking_valid_on_this_artifact is true only '
                 'when the ratio is known and >= the ratio floor, i.e. the adapter curve did not '
                 'overstate the merged edit.'),
    }
    problems = verify_adapter_card(card)
    if problems:
        raise AdapterCardRefused(problems[0].split(':', 1)[0],
                                 'refusing to write a card that fails its own check: '
                                 + '; '.join(problems))
    atomic_json(out_path, card)
    return card


def _card_problems(document):
    problems = []
    adapter = document.get('adapter') or {}
    base = document.get('base') or {}
    twin = document.get('merged_twin') or {}
    adapter_sha = str(adapter.get('sha256') or '').lower()
    base_sha = str(base.get('sha256') or '').lower()
    twin_sha = str(twin.get('artifact_sha256') or '').lower()
    bound = str(adapter.get('bound_base_sha256') or '').lower()
    if document.get('curve_is_not_merged_bytes') is not True:
        code = CURVE_CLAIM_FALSE if 'curve_is_not_merged_bytes' in document else CURVE_CLAIM_MISSING
        problems.append(f'{code}: curve_is_not_merged_bytes must be present and true (R8: the '
                        f'adapter curve is not the merged bytes)')
    if bound and base_sha and bound != base_sha:
        problems.append(f'{ADAPTER_BASE_DIGEST_MISMATCH}: adapter is bound to base {bound[:12]}…, '
                        f'card names base {base_sha[:12]}…')
    delivered = twin.get('delivered') or {}
    rows = delivered.get('tensors') if isinstance(delivered, dict) else None
    if not isinstance(rows, list) or not rows:
        problems.append(f'{TWIN_DELIVERED_COS_MISSING}: merged_twin.delivered.tensors is empty')
    else:
        for row in rows:
            if not isinstance(row, dict) or not _number(row.get('cos')):
                problems.append(f'{TWIN_DELIVERED_COS_MISSING}: delivered row without cos: {row!r}')
                break
    merged_delta = document.get('merged_delta_br')
    adapter_delta = document.get('adapter_delta_br')
    ratio = document.get('adapter_to_merged_ratio')
    status = document.get('merged_effect_status')
    if status not in ('MEASURED', 'UNMEASURED'):
        problems.append(f'{MERGED_EFFECT_UNMEASURED_UNLABELLED}: merged_effect_status is {status!r}; '
                        f'it must be MEASURED or an explicit UNMEASURED label')
    if merged_delta is None:
        if status != 'UNMEASURED':
            problems.append(f'{MERGED_EFFECT_UNMEASURED_UNLABELLED}: no merged_delta_br and no '
                            f'explicit UNMEASURED merged-effect label')
        if ratio is not None:
            problems.append(f'{DELTA_BR_RATIO_INCONSISTENT}: an UNMEASURED merged effect cannot '
                            f'carry adapter_to_merged_ratio {ratio}')
    else:
        if not _number(merged_delta):
            problems.append(f'{MERGED_EFFECT_UNMEASURED_UNLABELLED}: merged_delta_br is not a number')
        label = str(document.get('merged_measured_on') or '')
        source = str(document.get('merged_source_sha256') or '').lower()
        if _provenance_is_adapter(label) or (source and source in {adapter_sha, base_sha}):
            problems.append(f'{ADAPTER_DELTA_IN_MERGED_FIELD}: merged_delta_br is attributed to '
                            f'{label or source[:12]}; the merged effect must be measured on merged '
                            f'bytes, not on the adapter curve or the base artifact')
        if status == 'UNMEASURED':
            problems.append(f'{MERGED_EFFECT_UNMEASURED_UNLABELLED}: merged_delta_br is present but '
                            f'status is UNMEASURED')
        if ratio is None and _number(adapter_delta):
            problems.append(f'{DELTA_BR_RATIO_INCONSISTENT}: merged and adapter delta_br exist but '
                            f'no adapter_to_merged_ratio is recorded')
        elif _number(ratio) and _number(merged_delta) and _number(adapter_delta) and adapter_delta:
            implied = _ratio(merged_delta, adapter_delta)
            if implied is not None and abs(ratio - implied) > 0.002:
                code = (ADAPTER_DELTA_IN_MERGED_FIELD
                        if merged_delta == adapter_delta else DELTA_BR_RATIO_INCONSISTENT)
                problems.append(f'{code}: merged_delta_br {merged_delta} with adapter_delta_br '
                                f'{adapter_delta} implies ratio {implied}, card records {ratio}; the '
                                f'merged field may hold the adapter number')
        if source and twin_sha and source != twin_sha:
            problems.append(f'{ADAPTER_DELTA_IN_MERGED_FIELD}: merged_source_sha256 {source[:12]}… is '
                            f'not the merged twin {twin_sha[:12]}…')
    floor = document.get('ranking_validity_ratio_floor')
    if not _number(floor):
        problems.append(f'{RANKING_VALIDITY_INCONSISTENT}: ranking_validity_ratio_floor is missing')
        floor = RANKING_VALIDITY_RATIO_FLOOR
    expected = _ranking_valid(ratio if _number(ratio) else None, floor)
    if document.get('ranking_valid_on_this_artifact') is not expected:
        problems.append(f'{RANKING_VALIDITY_INCONSISTENT}: ranking_valid_on_this_artifact is '
                        f'{document.get("ranking_valid_on_this_artifact")!r}, but ratio {ratio} '
                        f'against floor {floor} implies {expected}')
    return problems


def verify_adapter_card(document):
    """Semantic checks a shape validator cannot express; returns named problems."""
    if not isinstance(document, dict):
        return ['INPUT_ERROR: adapter card must be a JSON object']
    return _card_problems(document)
