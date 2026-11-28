# SPDX-License-Identifier: AGPL-3.0-or-later
"""Import an Ablagent (DWMIP v0) direction as an axis for OUR quantized artifact.

Ablagent is an evidence-first research runtime: a dense, rank-one,
full-precision intervention `W' = W - alpha d (d^T W)` on a small HF checkpoint,
with a typed manifest and honest skips for quantization, serving and behaviour
verification. It does not edit a quantized GGUF and never authorizes a release.

This command is the narrow seam between the two: it reads one COMPILED Ablagent
artifact, validates the direction it carries (unit length, output space, alpha
range), maps the HF writer to the GGUF residual-write tensor the artifact
actually carries, and writes an axis file that our `edit` / `route --axes` path
applies through the SAME full-footprint codec gate as every other axis. The
provenance travels with the axis; Ablagent's skipped stages and `not_eligible`
release stay in the report, and nothing here measures removal.

What is refused, not repaired: an unreadable manifest, a non-DWMIP artifact, a
direction that is not unit length, an input-space vector, an unknown writer, and
a width/layer mismatch against the target artifact. The map is explicit (HF
writer -> blk.N.attn_output/ffn_down); the tool never guesses a tensor from
dimensions.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

SCHEMA = 'jdewitt-ablagent-import-v1'
ARTIFACT_SCHEMA = 'dwmip.artifact.v0'
ALGORITHM = 'dense.rank_one_output.v0'
HF_TO_GGUF = {'self_attn.o_proj': 'attn_output', 'mlp.down_proj': 'ffn_down'}
WRITEBACK = ('attn_output', 'attn_out', 'ssm_out', 'ffn_down', 'w2', 'o_proj',
             'out_proj', 'down_proj')
LAYER_TENSOR = re.compile(r'(?:^|\.)layers\.(\d+)\.(.+)\.weight$')


def _sha(path):
    from .storage import file_digest

    return file_digest(path)


def _manifest_path(path):
    from .errors import AblagentRefused

    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        for relative in ('artifacts/candidate.manifest.json', 'candidate.manifest.json'):
            candidate = path / relative
            if candidate.is_file():
                return candidate
        hits = sorted(path.glob('**/candidate.manifest.json'))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise AblagentRefused(
                'ABLAGENT_INPUT',
                f'{len(hits)} candidate manifests under {path}; name the one to import')
    raise AblagentRefused(
        'ABLAGENT_INPUT',
        f'{path} is not an Ablagent run directory or candidate.manifest.json')


def _stage_summary(manifest_path):
    """The run's control stage states, when the manifest sits inside a run tree."""
    run_dir = None
    for parent in manifest_path.parents:
        if (parent / 'control' / 'CONTROL.json').is_file():
            run_dir = parent
            break
    if run_dir is None:
        return {'control': None, 'release': None, 'stages': None}
    control = json.loads((run_dir / 'control' / 'CONTROL.json').read_text())
    stages = control.get('stages') or {}
    return {
        'control': str(run_dir / 'control' / 'CONTROL.json'),
        'release': control.get('release'),
        'stages': {name: {'execution': row.get('execution'), 'release': row.get('release')}
                   for name, row in sorted(stages.items())},
    }


def _map_writer(tensor_name):
    """HF writer -> (layer, gguf kind), or None when the name is not a decoder matrix."""
    match = LAYER_TENSOR.search(str(tensor_name))
    if match is None:
        return None
    layer, suffix = int(match.group(1)), match.group(2)
    kind = HF_TO_GGUF.get(suffix)
    if kind is None:
        last = suffix.split('.')[-1]
        kind = last if last in WRITEBACK else None
    if kind is None:
        return None
    return layer, kind, f'blk.{layer}.{kind}.weight'


def _stages_risks(stages):
    if not stages or not stages.get('stages'):
        return [{'code': 'STAGES_UNKNOWN', 'severity': 'low',
                 'detail': 'no control/CONTROL.json beside the manifest; the run’s skipped '
                           'stages are unknown here'}]
    skipped = [name for name, row in stages['stages'].items()
               if row.get('execution') == 'skipped']
    risks = []
    if skipped:
        risks.append({'code': 'SOURCE_STAGES_SKIPPED', 'severity': 'medium',
                      'detail': ('the source run skipped ' + ', '.join(skipped)
                                 + '; those are untested there, and this import never '
                                   'upgrades them')})
    if stages.get('release') and stages['release'] != 'not_eligible':
        risks.append({'code': 'SOURCE_RELEASE_LABEL', 'severity': 'high',
                      'detail': f"the source control calls release {stages['release']!r}; "
                                f"this tool does not accept a source release label as evidence"})
    return risks


def ablagent_import(manifest, gguf=None, out=None):
    """Read one compiled Ablagent artifact and write an axis file for `edit`/`route`."""
    import numpy as np
    import torch

    from .edit import artifact_view
    from .errors import AblagentRefused
    from .ggufwrite import parse_container
    from .nextsteps import cli_line

    path = _manifest_path(manifest)
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise AblagentRefused('ABLAGENT_INPUT', f'{path} is not a JSON manifest: {error}') from error
    if not isinstance(document, dict) or document.get('schema_version') != ARTIFACT_SCHEMA:
        raise AblagentRefused(
            'ABLAGENT_NOT_COMPILED',
            f'{path} declares schema {document.get("schema_version") if isinstance(document, dict) else None!r}; '
            f'this importer reads {ARTIFACT_SCHEMA} compiled artifacts only')
    intervention = document.get('intervention')
    if not isinstance(intervention, dict):
        raise AblagentRefused('ABLAGENT_NOT_COMPILED',
                              'the manifest carries no intervention object')
    if intervention.get('algorithm_id') != ALGORITHM:
        raise AblagentRefused(
            'ABLAGENT_NOT_COMPILED',
            f'the intervention algorithm is {intervention.get("algorithm_id")!r}, not {ALGORITHM}')
    direction = intervention.get('direction')
    if not isinstance(direction, dict):
        raise AblagentRefused('ABLAGENT_NOT_COMPILED',
                              'the intervention carries no bound direction object')
    alpha = intervention.get('alpha')
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as error:
        raise AblagentRefused('ALPHA_OUT_OF_RANGE', f'alpha is {alpha!r}') from error
    if not 0.0 < alpha <= 1.0:
        raise AblagentRefused('ALPHA_OUT_OF_RANGE', f'alpha {alpha} is outside (0, 1]')
    unit = np.asarray(direction.get('unit_vector') or [], dtype=np.float32)
    shape = [int(size) for size in direction.get('shape') or []]
    if unit.ndim != 1 or len(shape) != 2 or not unit.size:
        raise AblagentRefused('ABLAGENT_NOT_COMPILED',
                              'the direction carries no unit_vector with a 2-D shape')
    if not np.isfinite(unit).all():
        raise AblagentRefused('DIRECTION_NOT_UNIT',
                              'the unit_vector contains NaN/Inf; re-export rather than repairing it')
    if len(unit) == shape[1] and len(unit) != shape[0]:
        raise AblagentRefused(
            'DIRECTION_SPACE',
            f'the vector is {len(unit)} long, matching the writer INPUT ({shape[1]}), not '
            f'the output {shape[0]} the rank-one operation removes along; this tool does not '
            f'transpose a space')
    if len(unit) != shape[0]:
        raise AblagentRefused(
            'DIRECTION_SPACE',
            f'the vector is {len(unit)} long and the writer output is {shape[0]}; the '
            f'direction and the declared shape disagree')
    tolerance = float(direction.get('unit_norm_tolerance') or 1e-6)
    norm = float(np.linalg.norm(unit))
    if abs(norm - 1.0) > max(tolerance, 1e-6):
        raise AblagentRefused(
            'DIRECTION_NOT_UNIT',
            f'unit_vector norm is {norm:.9f}, outside its declared tolerance {tolerance}; '
            f're-export rather than normalizing here')
    tensor_name = str(direction.get('tensor_name') or '')
    mapped = _map_writer(tensor_name)
    if mapped is None:
        raise AblagentRefused(
            'UNMAPPED_WRITER',
            f'writer tensor {tensor_name!r} has no GGUF residual-write counterpart; this '
            f'importer maps decoder-layer o_proj/self_attn.o_proj and mlp.down_proj writers')
    layer, kind, gguf_name = mapped
    stages = _stage_summary(path)
    mapping = {'hf_tensor': tensor_name, 'writer_id': direction.get('writer_id'),
               'gguf_tensor': gguf_name, 'layer': layer, 'kind': kind,
               'space': 'output', 'width': len(unit), 'checked': False}
    artifact_hidden = None
    rows = layer + 1
    if gguf is not None:
        view = artifact_view(gguf)
        artifact_hidden = view.get('hidden')
        header = parse_container(str(gguf))
        tensors = {tensor['name']: tensor for tensor in header['tensors']}
        tensor = tensors.get(gguf_name)
        if tensor is None:
            carried = sorted(name for name in tensors
                             if name.startswith(f'blk.{layer}.') and name.endswith('.weight'))
            raise AblagentRefused(
                'UNMAPPED_WRITER',
                f'{gguf_name} is not in {gguf}; that layer carries {carried or "no 2-D weights"}')
        dims = [int(size) for size in tensor['dims']]
        if len(dims) != 2 or dims[1] != len(unit):
            raise AblagentRefused(
                'GEOMETRY_MISMATCH',
                f'{gguf_name} is {dims} (ggml [in, out]); the direction is {len(unit)}-wide '
                f'in the output space')
        if view.get('layers') and layer >= int(view['layers']):
            raise AblagentRefused(
                'GEOMETRY_MISMATCH',
                f'layer {layer} is outside the {view["layers"]} writeback layers this '
                f'artifact carries')
        if artifact_hidden is not None and artifact_hidden != len(unit):
            raise AblagentRefused(
                'GEOMETRY_MISMATCH',
                f'the artifact hidden width is {artifact_hidden} and the direction is '
                f'{len(unit)}-wide; the direction was measured in a different space')
        rows = max(rows, int(view.get('layers') or 0))
        mapping.update({'checked': True, 'artifact_hidden': artifact_hidden,
                        'gguf_dims': dims})
    axes = torch.zeros((rows, len(unit)), dtype=torch.float32)
    axes[layer] = torch.from_numpy(unit)
    provenance = {
        'source': 'ablagent',
        'manifest': str(path.resolve()),
        'manifest_sha256': _sha(path),
        'artifact_schema': ARTIFACT_SCHEMA,
        'algorithm_id': ALGORITHM,
        'direction_digest': document.get('direction_digest'),
        'writer_id': direction.get('writer_id'),
        'tensor_name': tensor_name,
        'shape': shape,
        'space_id': direction.get('space_id'),
        'baseline_digest': direction.get('baseline_digest'),
        'graph_digest': direction.get('graph_digest'),
        'source_algorithm_id': direction.get('source_algorithm_id'),
        'source_vector_digest': direction.get('source_vector_digest'),
        'alpha': alpha,
        'unit_norm': round(norm, 9),
        'release': stages.get('release'),
        'removal_measured': False,
        'imported_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    stamp = provenance['manifest_sha256'][:8]
    out_path = Path(out) if out is not None else Path.cwd() / f'ablagent-axis-{stamp}.pt'
    if out_path.is_dir():
        out_path = out_path / f'ablagent-axis-{stamp}.pt'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'axes': axes, 'kinds': [kind], 'alpha': alpha,
                'scale': alpha, 'provenance': provenance}, out_path)
    risks = _stages_risks(stages)
    risks.append({'code': 'SOURCE_SCOPE', 'severity': 'medium',
                  'detail': ('Ablagent’s v0 real executor edits one dense rank-one writer on '
                             'a full-precision checkpoint and records quantization, serving '
                             'and behaviour verification as skipped; this import carries the '
                             'direction, not those measurements')})
    risks.append({'code': 'BASE_IDENTITY_UNPROVEN', 'severity': 'medium',
                  'detail': ('the direction is bound to Ablagent’s baseline digest '
                             f'{direction.get("baseline_digest")}; only the artifact '
                             'architecture/hidden/layer geometry is checked here, not that '
                             'digest')})
    if kind == 'ffn_down':
        risks.append({'code': 'BASIS_FOLDING', 'severity': 'high',
                      'detail': ('some converters (the dspark fork on qwen) Hadamard-fold ffn '
                                 'tensors; an HF-basis direction then has the wrong '
                                 'functional. o_proj is not folded. Confirm the artifact’s '
                                 'converter before applying an ffn_down direction')})
    if not mapping['checked']:
        risks.append({'code': 'ARTIFACT_UNCHECKED', 'severity': 'low',
                      'detail': 'no --gguf was passed: geometry and tensor existence were not '
                                'checked against an artifact'})
    target = Path(gguf) if gguf is not None else Path('<artifact.gguf>')
    workdir = target.with_name(target.stem + '.route') if gguf is not None else Path('<workdir>')
    next_commands = {
        'route': cli_line('route', '--gguf', target, '--workdir', workdir,
                          '--axes', out_path, '--scale', alpha, 'survey'),
        'plan': cli_line('route', '--workdir', workdir, 'plan'),
        'edit': cli_line('edit', '--gguf', target, '--axes', out_path, '--scale', alpha),
    }
    return {
        'schema': SCHEMA,
        'command': 'ablagent-import',
        'ok': True,
        'ablagent': {
            'manifest': str(path),
            'manifest_sha256': provenance['manifest_sha256'],
            'artifact_schema': ARTIFACT_SCHEMA,
            'algorithm_id': ALGORITHM,
            'efficacy': document.get('efficacy'),
            'safety': document.get('safety'),
            'release_eligibility': document.get('release_eligibility'),
            'direction_digest': document.get('direction_digest'),
            'writer_id': direction.get('writer_id'),
            'tensor_name': tensor_name,
            'space_id': direction.get('space_id'),
            'shape': shape,
            'alpha': alpha,
            'unit_norm': round(norm, 9),
        },
        'source_run': stages,
        'mapping': mapping,
        'axes': {'path': str(out_path), 'sha256': _sha(out_path),
                 'shape': [int(size) for size in axes.shape], 'kinds': [kind],
                 'nonzero_layers': [layer], 'scale': alpha},
        'honesty': {
            'removal_measured': False,
            'capability_measured': False,
            'release': 'not_eligible',
            'source_scope': 'dense full-precision safetensors, one rank-one writer',
            'what_this_import_adds': ('the direction and its provenance only; the gate in '
                                      'edit/route measures codec fidelity on the user’s '
                                      'quantized artifact, and behaviour stays GATED'),
        },
        'risks': risks,
        'next': next_commands,
    }


def ablagent_import_command(args):
    """CLI handler: one JSON, one axis file, no model load."""
    return ablagent_import(args.manifest, gguf=args.gguf, out=args.out)
