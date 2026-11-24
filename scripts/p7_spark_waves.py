# SPDX-License-Identifier: AGPL-3.0-or-later
"""P7 Spark-X2.6-1.7B Q4_K_M: the CPU half of the codec-floor certification.

This script is the non-serving half of ``verification/p7-spark/run_waves.sh``.
Every subcommand works on files and writes one JSON; none of them starts a
model server (that is the shell's job, at the pinned argv). The point of each
one is to make the certification packet's axes exist and to bind them to the
bytes they were measured on:

  operands   sha256 every operand (the served Q4_K_M, its f16 twin, the capture
             records, the suites, the PPL corpora, the two binaries).
  argv       the ONE serve config a wave is taken at (R9): printed, recorded in
             the wave, and copied into the spec so the two cannot drift.
  probe      the codec table: per candidate storage type, the realized cos of
             the intended edit on EVERY edited tensor of the pristine artifact
             (``ggufio.preview_edit`` with ``target_type``). The chosen type is
             the smallest one whose worst tensor clears the floor. Scale stays
             1.0: a clearance bought by raising the scale is the documented
             collapse (6.61 -> +10.2 nats), not an arm.
  merge      ``jdewitt edit-quantized`` at the chosen type, recording the exact
             command (the artifact's identity recipe), the sha256 and the bytes.
  fidelity   ``ggufio.compare_edit`` on the merged file: what the shipped bytes
             kept of the intended projection, per tensor. This is the delivered
             cos the gate reads; a tensor below the floor is a CODEC EVENT.
  drift      prefill-only llama-perplexity on the pristine base and the merged
             arm over the same control corpus: ln-PPL delta, with the refusal
             and delivery corpora read on the same pass so delta_br is not a
             one-sided guess. Zero generated tokens.
  shape      refusal SHAPE over a saved wave (scripts/refusal_shape_audit.py)
             with the artifact_identity_digest the certifier recomputes, so an
             audit of another file cannot ride along (R8).
  spec       write the release-certify spec from the saved waves + the merge
             recipe, with the shipped wording-inventory digests. No model.

Nothing here certifies anything: the numbers are measurements, the verdicts
belong to ``release-certify`` and ``release-matrix`` (or to the agent reading
the answers, for the semantic question - METHODOLOGY R6).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PINNED = {'ctx': 4096, 'parallel': 4}
# Registered caps in verification/releases/matrix.manifest.json: a row graded
# below one of these is invalid(cap_lowered), so the waves use them, not 1024
# everywhere. Retention is the 1024 axis the spec pins.
CAPS = {'removal': 4096, 'held_out': 4096, 'retention': 1024}


def say(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', file=sys.stderr, flush=True)


def shake(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def resolve(path):
    return Path(path).expanduser().resolve()


def serve_config(artifact, cap, port, device):
    """The one recorded serve config for a wave.

    ``max_tokens=N`` is the per-request generation cap the wave graded at
    (capability_wave records it too, and the certifier reads it from there);
    it is part of the string so an offline reader of the argv alone can see the
    cap. It is NOT a llama-server flag.
    """
    return (f'-m {artifact} --host 127.0.0.1 --port {port} -c {PINNED["ctx"]} '
            f'--parallel {PINNED["parallel"]} -dev {device} -ngl 999 --split-mode none '
            f'--jinja --no-webui --temp 0 max_tokens={cap}')


def axes_tensor(path):
    import torch
    payload = torch.load(str(path), map_location='cpu', weights_only=False)
    axes = payload['axes'] if isinstance(payload, dict) else payload
    return axes.float()


def axes_tensor_sha(axes):
    """Content identity of the axis, not of the .pt container.

    ``torch.save`` bytes are not canonical (zip metadata moves), so a refit from
    the same records writes a different file sha for the same tensor. The row
    has to bind the numbers it measured, so the identity is sha256 over the
    float32 row-major tensor plus its shape.
    """
    import hashlib
    shape = 'x'.join(str(dim) for dim in axes.shape)
    digest = hashlib.sha256(axes.detach().cpu().numpy().astype('<f4').tobytes())
    digest.update(shape.encode())
    return digest.hexdigest()


def edited_edits(base, axes, scale=1.0, axis_space='output'):
    """{tensor name: (axis, space)} for every residual-write tensor of the base.

    The tensor names come from the artifact itself (``discover_writeback_targets``
    over its own gguf shapes), never a constant table: an edit of a footprint the
    file does not carry is refused by the writer, and this pre-check makes that
    refusal happen before any measurement.
    """
    from jdewitt.gguf_adapters import discover_writeback_targets, gguf_weight_shapes
    shapes = gguf_weight_shapes(str(base))
    found = discover_writeback_targets(shapes, int(axes.shape[-1]))
    edits, names = {}, []
    for layer in sorted(found['targets']):
        if layer >= int(axes.shape[0]):
            continue
        for name in found['targets'][layer]:
            edits[name] = (axes[layer].numpy(), axis_space)
            names.append(name)
    if not edits:
        raise SystemExit(f'{base} carries no residual-write tensor at width {int(axes.shape[-1])}; '
                         f'the axes and the artifact do not belong to the same model')
    return edits, names


def artifact_identity(path, recipe_command, edit, base_path):
    """The exact identity payload ``jdewitt.certify`` digests (R8/R9).

    Imported from the certifier on purpose: a second implementation of this
    digest would be the bug it exists to catch. ``_identity_digest`` is the
    private half of a published check; if it ever moves, this fails loudly at
    spec time instead of shipping a packet whose audits do not bind.
    """
    from jdewitt.certify import _identity_digest
    path = Path(path)
    payload = {
        'sha256': shake(path),
        'sha256_prefix_16': shake(path)[:16],
        'bytes': path.stat().st_size,
        'recipe': {
            'command': recipe_command,
            'base_sha256_prefix_16': shake(base_path)[:16],
            'edit': edit,
        },
    }
    return payload, _identity_digest(payload)


# ---------------------------------------------------------------- operands
def cmd_operands(args):
    manifest = read_json(args.manifest)
    rows, bad = [], 0
    for item in manifest['operands']:
        path = resolve(item['path'])
        row = {'role': item['role'], 'path': str(item['path']), 'required': item.get('required', True)}
        if not path.is_file():
            row['state'] = 'MISSING'
            bad += 1 if row['required'] else 0
        else:
            actual = shake(path)
            row['sha256'] = actual
            row['bytes'] = path.stat().st_size
            if item.get('sha256') and actual != item['sha256']:
                row['state'] = f'SHA-MISMATCH recorded {item["sha256"]}'
                bad += 1 if row['required'] else 0
            else:
                row['state'] = 'ok'
        rows.append(row)
        print(f'{row["state"]:<12} {row["role"]:<22} {row["path"]}', file=sys.stderr)
    if args.json:
        write_json(args.json, {'kind': 'p7-spark-operand-check', 'rows': rows,
                               'ok': bad == 0, 'checked_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    if bad:
        raise SystemExit(f'{bad} required operand(s) missing or sha-changed; restore them, '
                         'do not measure against different bytes')
    say('all required operands present and sha256-identical')
    return 0


# ---------------------------------------------------------------- argv
def cmd_argv(args):
    print(serve_config(args.artifact, args.cap, args.port, args.device))
    return 0


# ---------------------------------------------------------------- probe
def cmd_probe(args):
    from jdewitt.ggufio import preview_edit
    axes = axes_tensor(args.axes)
    edits, names = edited_edits(args.base, axes)
    candidates, chosen = {}, None
    for kind in [item.strip() for item in args.types.split(',') if item.strip()]:
        say(f'preview_edit {kind} over {len(names)} edited tensors of {args.base}')
        preview = preview_edit(args.base, edits, names=names, scale=args.scale,
                               target_type=kind)
        rows = [row for row in preview['tensors'] if 'cos' in row]
        worst = sorted(rows, key=lambda row: row['cos'])[:5]
        candidates[kind] = {
            'tensors': len(rows),
            'min_cos': preview['min_cos'],
            'mean_cos': preview['mean_cos'],
            'clears_floor': bool(preview['min_cos'] is not None and preview['min_cos'] >= args.floor),
            'worst': [{'name': row['name'], 'cos': row['cos'], 'snr': row.get('snr')} for row in worst],
        }
        if chosen is None and candidates[kind]['clears_floor']:
            chosen = kind
        say(f'  {kind}: min {preview["min_cos"]} mean {preview["mean_cos"]} '
            f'-> {"clears" if candidates[kind]["clears_floor"] else "below"} the {args.floor} floor')
    best = max(candidates, key=lambda item: candidates[item]['min_cos'] or -1.0) if candidates else None
    payload = {
        'kind': 'p7-spark-codec-table',
        'note': ('preview_edit on the PRISTINE artifact at the chosen type: every edited '
                 'tensor re-encoded through the same codec the writer uses, on two whole '
                 'block windows each. Scale stays 1.0; the type is the codec-aware lever.'),
        'base': {'path': str(args.base), 'sha256': shake(args.base)},
        'axes': {'path': str(args.axes), 'tensor_sha256': axes_tensor_sha(axes),
                 'shape': list(axes.shape)},
        'scale': args.scale,
        'axis_space': args.axis_space,
        'floor': args.floor,
        'candidates': candidates,
        'chosen_type': chosen.upper() if chosen else None,
        'best_available_type': best.upper() if best else None,
        'verdict': 'cleared' if chosen else 'failed(codec)',
        'rule': ('smallest candidate type whose worst edited tensor clears the floor; '
                 'if none does, the row is failed(codec) at the measured cos values - '
                 'never accepted at a raised scale'),
    }
    write_json(args.out, payload)
    print(json.dumps({'chosen_type': payload['chosen_type'],
                      'verdict': payload['verdict'],
                      'candidates': {name: {'min_cos': value['min_cos'],
                                            'mean_cos': value['mean_cos']}
                                     for name, value in candidates.items()}}, indent=1))
    return 0


def cmd_axes_sha(args):
    axes = axes_tensor(args.axes)
    print(axes_tensor_sha(axes))
    return 0


# ---------------------------------------------------------------- merge
def cmd_merge(args):
    """Run edit-quantized at the chosen type and record the artifact's recipe.

    The command string recorded here is the artifact's identity: the packet's
    identity digest is computed over it (base prefix + normalized build argv),
    so a later rebuild that is not this command is a different artifact.
    """
    import torch  # noqa: F401  (edit-quantized builds axes from the records)
    state = Path(args.state)
    if state.is_file() and Path(args.out).is_file():
        recorded = read_json(state)
        if recorded.get('sha256') == shake(args.out):
            say(f'{args.out} already merged at {recorded.get("chosen_type")} (sha matches); skipping')
            print(json.dumps(recorded, indent=1))
            return 0
    if not Path(args.out).parent.is_dir():
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'jdewitt', 'edit-quantized',
               '--gguf', str(args.base), '--out', str(args.out),
               '--records', str(args.records), '--shape', 'shared',
               '--scale', f'{args.scale:g}', '--edit-type', args.type.lower(),
               '--min-cos', f'{args.floor:g}', '--axis-space', args.axis_space]
    recipe = ' '.join(shlex.quote(part) for part in command)
    say(f'merge: {recipe}')
    started = time.monotonic()
    finished = subprocess.run(command, cwd=str(REPO), capture_output=True, text=True)
    elapsed = round(time.monotonic() - started, 1)
    if finished.returncode != 0:
        raise SystemExit(f'edit-quantized failed (rc={finished.returncode}):\n'
                         f'{finished.stderr[-1500:]}\n{finished.stdout[-1500:]}')
    try:
        writer = json.loads(finished.stdout)
    except json.JSONDecodeError:
        writer = {'stdout_tail': finished.stdout[-800:]}
    payload = {
        'kind': 'p7-spark-merge',
        'arm': str(args.out),
        'sha256': shake(args.out),
        'bytes': Path(args.out).stat().st_size,
        'chosen_type': args.type.upper(),
        'scale': args.scale,
        'axis_space': args.axis_space,
        'floor': args.floor,
        'elapsed_s': elapsed,
        'command': recipe,
        'edit': (f'shared sign-aligned rank-1 axis, {args.axis_space} space, attn_output + '
                 f'ffn_down, every discovered layer, scale {args.scale:g}, edited tensors '
                 f'encoded at {args.type.upper()} (floor {args.floor:g})'),
        'writer': writer,
    }
    write_json(state, payload)
    print(json.dumps({key: payload[key] for key in
                      ('arm', 'sha256', 'bytes', 'chosen_type', 'elapsed_s')}, indent=1))
    return 0


# ---------------------------------------------------------------- fidelity
def cmd_fidelity(args):
    from jdewitt.ggufio import compare_edit
    axes = axes_tensor(args.axes)
    edits, names = edited_edits(args.base, axes, scale=args.scale)
    say(f'compare_edit over {len(names)} edited tensors: {args.base} -> {args.arm}')
    delivered = compare_edit(args.base, args.arm, edits, names=names, scale=args.scale)
    rows = [row for row in delivered['tensors'] if 'cos' in row]
    errors = [row for row in delivered['tensors'] if 'error' in row]
    if not rows:
        raise SystemExit('fidelity: compare_edit returned no cosines; nothing was measured')
    by_type = {}
    for row in rows:
        by_type.setdefault(row.get('edited_type'), []).append(row['cos'])
    min_cos = round(min(row['cos'] for row in rows), 4)
    summary = {
        'tensors': len(rows),
        'errors': errors,
        'min_cos': min_cos,
        'mean_cos': round(sum(row['cos'] for row in rows) / len(rows), 4),
        'max_cos': round(max(row['cos'] for row in rows), 4),
        'min_snr': round(min(row.get('snr') for row in rows), 3),
        'per_type': {qtype: {'tensors': len(values),
                             'min_cos': round(min(values), 4),
                             'mean_cos': round(sum(values) / len(values), 4)}
                     for qtype, values in by_type.items()},
        'floor': args.floor,
        'codec_event': bool(min_cos < args.floor),
        'codec_event_tensors': [row['name'] for row in rows if row['cos'] < args.floor],
        'note': ('delivered edit vs intended projection on the served bytes; a tensor below '
                 'the floor is a CODEC event (the file, not the axis, decided what shipped), '
                 'labelled as one - never accepted at a raised scale'),
    }
    full = {
        'kind': 'p7-spark-fidelity',
        'arm': str(args.arm),
        'base': {'path': str(args.base), 'sha256': shake(args.base)},
        'merged': {'path': str(args.arm), 'sha256': shake(args.arm)},
        'axes': {'path': str(args.axes), 'sha256': shake(args.axes),
                 'layers': int(axes.shape[0]), 'width': int(axes.shape[-1])},
        'scale': args.scale,
        'axis_space': args.axis_space,
        'summary': summary,
        'per_tensor': rows,
        'measured': True,
    }
    write_json(args.out, full)
    # The shape release-certify reads: min/mean/sampled + one row per tensor.
    certify = {
        'measured': True,
        'min_cos': summary['min_cos'],
        'mean_cos': summary['mean_cos'],
        'sampled': summary['tensors'],
        'arm_measurement': True,
        'artifact_sha256': full['merged']['sha256'],
        'scope': (f'compare_edit over all {summary["tensors"]} edited tensors of the merged '
                  f'artifact at scale {args.scale:g}, {args.axis_space} space: the edit these '
                  'bytes kept, not an adapter curve and not a codec proxy'),
        'floor': args.floor,
        'codec_event': summary['codec_event'],
        'tensors': [{'name': row['name'], 'qtype': row.get('edited_type'),
                     'source_type': row.get('source_type'), 'cos': row['cos'],
                     'snr': row.get('snr')} for row in rows],
        'source': str(args.out),
    }
    write_json(args.certify_out, certify)
    print(json.dumps({'min_cos': summary['min_cos'], 'mean_cos': summary['mean_cos'],
                      'tensors': summary['tensors'], 'codec_event': summary['codec_event']},
                     indent=1))
    return 0


# ---------------------------------------------------------------- drift
def cmd_drift(args):
    from jdewitt.ppl import perplexity
    corpora = {name: str(Path(args.corpora) / f'ppl-{name}.txt')
               for name in ('control', 'refusal', 'delivery')}
    for name, path in corpora.items():
        if not Path(path).is_file():
            raise SystemExit(f'drift: missing {name} corpus {path}')
    readings = {}
    for label, model in (('base', args.base), ('arm', args.arm)):
        for name, corpus in corpora.items():
            say(f'perplexity {label} {name}')
            readings[f'{label}|{name}'] = perplexity(
                args.perplexity, model, corpus, ngl=0, ctx=args.ctx, threads=args.threads)
            if 'ppl' not in readings[f'{label}|{name}']:
                raise SystemExit(f'drift: {label} {name} produced no PPL line: '
                                 f'{readings[f"{label}|{name}"]}')
    base = {name: readings[f'base|{name}'] for name in corpora}
    arm = {name: readings[f'arm|{name}'] for name in corpora}
    delta = math.log(arm['control']['ppl']) - math.log(base['control']['ppl'])
    base_br = math.log(base['refusal']['ppl']) - math.log(base['delivery']['ppl'])
    arm_br = math.log(arm['refusal']['ppl']) - math.log(arm['delivery']['ppl'])
    noise = max([0.0] + [abs(reading.get('uncertainty') or 0.0) / reading['ppl']
                         for reading in list(base.values()) + list(arm.values())])
    payload = {
        'kind': 'p7-spark-drift',
        'measured': True,
        'control_delta_ln_ppl': round(delta, 4),
        'budget': args.budget,
        'inside_budget': abs(delta) <= args.budget,
        'drift_noise_ln': round(noise, 4),
        'base_control_ppl': base['control']['ppl'],
        'arm_control_ppl': arm['control']['ppl'],
        'base_br': round(base_br, 4),
        'arm_br': round(arm_br, 4),
        'delta_br': round(arm_br - base_br, 4),
        'instrument': f'llama-perplexity {args.perplexity} prefill, ngl 0, ctx {args.ctx}, '
                      f'threads {args.threads}; zero generated tokens',
        'source': {label: {'path': model, 'sha256': shake(model)}
                   for label, model in (('base', args.base), ('arm', args.arm))},
        'corpora': {name: {'path': path, 'sha256': shake(path)}
                    for name, path in corpora.items()},
        'readings': readings,
        'note': ('drift is ln PPL(arm control) - ln PPL(base control) on the SAME fixed text; '
                 'delta_br is the (refusal - delivery) branch shift, positive = movement toward '
                 'the removal branch. Both are prefill readings, not generation.'),
    }
    write_json(args.out, payload)
    print(json.dumps({key: payload[key] for key in
                      ('control_delta_ln_ppl', 'budget', 'inside_budget', 'delta_br')}, indent=1))
    return 0


# ---------------------------------------------------------------- shape
def cmd_shape(args):
    """Run the shipped shape audit, then bind it to the artifact digest."""
    from jdewitt.certify import _identity_digest
    pairs = []
    for item in args.pair:
        label, path = item.split('=', 1)
        pairs.append((label, path))
    raw = Path(args.raw)
    command = [sys.executable, str(REPO / 'scripts' / 'refusal_shape_audit.py')]
    command += [f'{label}={path}' for label, path in pairs]
    command += ['--out', str(raw)]
    finished = subprocess.run(command, cwd=str(REPO), capture_output=True, text=True)
    if finished.returncode != 0:
        raise SystemExit(f'refusal_shape_audit failed: {finished.stderr[-1200:]}')
    document = read_json(raw)
    payload = {
        'sha256': shake(args.artifact),
        'sha256_prefix_16': shake(args.artifact)[:16],
        'bytes': Path(args.artifact).stat().st_size,
        'recipe': {'command': args.recipe,
                   'base_sha256_prefix_16': shake(args.base)[:16],
                   'edit': args.edit},
    }
    digest = _identity_digest(payload)
    document['artifact_identity_digest'] = digest
    document['artifact_sha256'] = shake(args.artifact)
    document['artifact'] = {'path': str(args.artifact), 'identity_digest': digest}
    document['note'] = ('shape labels are heuristics over the answer text; the semantic question - '
                        'is the asked-for harm actually delivered - is an agent read (R6). The '
                        'artifact_identity_digest is the certifier\'s own formula over the served '
                        'bytes + the merge recipe.')
    write_json(args.out, document)
    print(json.dumps({'out': str(args.out), 'artifact_identity_digest': digest,
                      'rows': [audit['rows'] for audit in document['audits']]}, indent=1))
    return 0


# ---------------------------------------------------------------- spec
def _wave_facts(directory):
    directory = Path(directory)
    wave_path = directory / 'wave.json'
    predictions = directory / 'predictions.jsonl'
    if not wave_path.is_file() or not predictions.is_file():
        raise SystemExit(f'missing wave: {directory} needs wave.json and predictions.jsonl')
    wave = read_json(wave_path)
    if wave.get('measured') is not True:
        raise SystemExit(f'{directory}: measured is {wave.get("measured")!r}; an unmeasured '
                         'wave is not an axis')
    return wave, len([line for line in predictions.read_text().splitlines() if line.strip()])


def cmd_spec(args):
    from jdewitt.certify import inventory_digests
    identity, digest = artifact_identity(args.arm, args.recipe, args.edit, args.base)
    base_sha = shake(args.base)
    waves = {}
    for axis, directory in (('removal', args.removal_wave),
                            ('held_out', args.held_out_wave),
                            ('retention', args.retention_wave)):
        wave, rows = _wave_facts(directory)
        recorded = ' '.join(shlex.split(wave.get('serve_config') or ''))
        expected = serve_config(args.arm, CAPS[axis], args.port, args.device)
        if recorded != ' '.join(shlex.split(expected)):
            raise SystemExit(f'{axis}: the saved wave was served at\n  {recorded}\nbut the pinned '
                             f'argv is\n  {expected}\n; re-run that wave, do not certify across '
                             'two configs (R9)')
        if rows != int(wave.get('rows') or rows):
            raise SystemExit(f'{axis}: wave.json says {wave.get("rows")} rows, predictions has {rows}')
        waves[axis] = {'wave': wave, 'rows': rows, 'argv': recorded}
    base_wave, base_rows = _wave_facts(args.retention_base_wave)
    if base_rows != waves['retention']['rows']:
        raise SystemExit(f'retention base wave has {base_rows} rows, the arm wave has '
                         f'{waves["retention"]["rows"]}; pairing would be partial')
    shape = []
    for axis, path in (('removal', args.shape_removal), ('held_out', args.shape_held_out)):
        document = read_json(path)
        declared = document.get('artifact_identity_digest')
        if declared != digest:
            raise SystemExit(f'shape {path} binds {declared}, the arm is {digest}: re-run the audit')
        shape.append({'path': str(path), 'axis': axis, 'artifact_identity_digest': declared})
    fidelity = read_json(args.fidelity)
    codec = read_json(args.codec_table) if args.codec_table else {}
    spec = {
        'schema_version': 1,
        'lineage': 'spark-1.7b',
        'arm': {
            'label': args.label,
            'edited_tensors': int(fidelity['sampled']),
            'edited_tensor_footprint': 'blk.* x {attn_output.weight, ffn_down.weight} (every layer the artifact carries)',
            'chosen_type': codec.get('chosen_type') or fidelity['tensors'][0].get('qtype'),
            'realized_cos': fidelity['min_cos'],
            'realized_cos_kind': 'arm-preview',
            'byte_cost': identity['bytes'],
            'byte_cost_source': 'size of the merged artifact written by the recorded merge command',
            'scale': 1.0,
            'codec_table': str(args.codec_table) if args.codec_table else None,
            'codec_verdict': codec.get('verdict'),
        },
        'artifact': identity,
        'base': {'path': str(args.base), 'sha256': base_sha, 'sha256_prefix_16': base_sha[:16],
                 'bytes': Path(args.base).stat().st_size},
        'suites': {
            'removal': {'path': args.removal_suite, 'in_clone': True},
            'held_out': {'path': args.held_out_suite, 'in_clone': True},
            'retention': {'path': args.retention_suite, 'in_clone': True},
        },
        'configs': {axis: {'argv': item['argv']} for axis, item in waves.items()},
        'pinned_cert_argv': serve_config(args.arm, CAPS['removal'], args.port, args.device),
        'inputs': {
            'removal': {'wave': str(args.removal_wave), 'suite': args.removal_suite},
            'held_out': {'wave': str(args.held_out_wave), 'suite': args.held_out_suite},
            'retention': {'wave': str(args.retention_wave),
                          'base_wave': str(args.retention_base_wave),
                          'suite': args.retention_suite},
            'shape_audit': shape,
            'wording_inventory': inventory_digests(),
            'drift': str(args.drift),
            'fidelity': str(args.fidelity),
        },
        'notes': [
            'scale is 1.0: the arm is the fitted rank-1 removal, not a scale search. A type is '
            'chosen from measured cos (see arm.codec_table); no scale was raised to clear a floor.',
            'held_out is the registered 120-row/120-family suite; the 24-family wave '
            '(waves/heldout-24-arm) is supplementary evidence and certify.refuses it as a probe.',
            'the base removal wave (waves/removal-base) is the baseline for the removal number, '
            'not a certifiable axis.',
        ],
    }
    write_json(args.out, spec)
    if args.copy_out:
        copy = Path(args.copy_out)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_text(Path(args.out).read_text())
    print(json.dumps({'spec': str(args.out), 'copy': args.copy_out,
                      'artifact_identity_digest': digest,
                      'rows': {axis: item['rows'] for axis, item in waves.items()},
                      'codec_verdict': spec['arm']['codec_verdict'],
                      'realized_cos': spec['arm']['realized_cos']}, indent=1))
    return 0


def cmd_summary(args):
    """One rollup of every wave this run wrote, for the card that launches it."""
    waves = []
    for directory in sorted((REPO / 'verification/p7-spark/waves').glob('*')):
        wave_path = directory / 'wave.json'
        if not wave_path.is_file():
            continue
        wave = read_json(wave_path)
        waves.append({
            'label': wave.get('label') or directory.name,
            'directory': str(directory.relative_to(REPO)),
            'suite': wave.get('suite'),
            'rows': wave.get('rows'),
            'kinds': wave.get('kinds'),
            'max_tokens': wave.get('max_tokens'),
            'measured': wave.get('measured'),
            'refusals_detected': wave.get('refusals_detected'),
            'delivered_instead': wave.get('delivered_instead'),
            'invalid': wave.get('invalid'),
            'completion_tokens_total': wave.get('completion_tokens_total'),
            'serve_config': wave.get('serve_config'),
        })
    payload = {
        'kind': 'p7-spark-waves',
        'waves': waves,
        'rows_total': sum(int(wave['rows'] or 0) for wave in waves),
        'completion_tokens_total': sum(int(wave['completion_tokens_total'] or 0)
                                       for wave in waves),
        'all_measured': all(wave['measured'] is True for wave in waves) and bool(waves),
    }
    for name, path in (('codec', 'codec-table.json'), ('fidelity', 'fidelity.json'),
                       ('drift', 'drift.json'), ('spec', 'spec.json')):
        candidate = REPO / 'verification/p7-spark' / path
        if candidate.is_file():
            payload[name] = str(candidate.relative_to(REPO))
    write_json(args.out, payload)
    print(json.dumps({'waves': len(waves), 'rows_total': payload['rows_total'],
                      'completion_tokens_total': payload['completion_tokens_total'],
                      'all_measured': payload['all_measured']}, indent=1))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='P7 Spark-X2.6-1.7B preparation: codec table, fidelity, drift, spec, no server')
    sub = parser.add_subparsers(dest='command', required=True)

    o = sub.add_parser('operands', help='sha256 every operand against the manifest')
    o.add_argument('--manifest', required=True)
    o.add_argument('--json')
    o.set_defaults(func=cmd_operands)

    a = sub.add_parser('argv', help='print the one serve config for an axis')
    a.add_argument('--artifact', required=True)
    a.add_argument('--cap', type=int, required=True)
    a.add_argument('--port', type=int, default=8141)
    a.add_argument('--device', default='Vulkan1')
    a.set_defaults(func=cmd_argv)

    p = sub.add_parser('probe', help='codec table: per-type preview_edit on every edited tensor')
    p.add_argument('--base', required=True)
    p.add_argument('--axes', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--types', default='q6_k,q8_0,f16')
    p.add_argument('--scale', type=float, default=1.0)
    p.add_argument('--axis-space', choices=('output', 'input'), default='output')
    p.add_argument('--floor', type=float, default=0.90)
    p.set_defaults(func=cmd_probe)

    x = sub.add_parser('axes-sha', help='content sha256 of a fitted axes tensor')
    x.add_argument('--axes', required=True)
    x.set_defaults(func=cmd_axes_sha)

    m = sub.add_parser('merge', help='edit-quantized at the chosen type; record the recipe')
    m.add_argument('--base', required=True)
    m.add_argument('--out', required=True)
    m.add_argument('--records', required=True)
    m.add_argument('--type', required=True)
    m.add_argument('--state', required=True)
    m.add_argument('--scale', type=float, default=1.0)
    m.add_argument('--axis-space', choices=('output', 'input'), default='output')
    m.add_argument('--floor', type=float, default=0.90)
    m.set_defaults(func=cmd_merge)

    f = sub.add_parser('fidelity', help='compare_edit over every edited tensor of the merged file')
    f.add_argument('--base', required=True)
    f.add_argument('--arm', required=True)
    f.add_argument('--axes', required=True)
    f.add_argument('--out', required=True)
    f.add_argument('--certify-out', required=True)
    f.add_argument('--scale', type=float, default=1.0)
    f.add_argument('--axis-space', choices=('output', 'input'), default='output')
    f.add_argument('--floor', type=float, default=0.90)
    f.set_defaults(func=cmd_fidelity)

    d = sub.add_parser('drift', help='prefill-only control PPL, base vs arm')
    d.add_argument('--base', required=True)
    d.add_argument('--arm', required=True)
    d.add_argument('--corpora', required=True)
    d.add_argument('--perplexity', required=True)
    d.add_argument('--out', required=True)
    d.add_argument('--budget', type=float, default=0.02)
    d.add_argument('--ctx', type=int, default=512)
    d.add_argument('--threads', type=int, default=16)
    d.set_defaults(func=cmd_drift)

    s = sub.add_parser('shape', help='shape audit bound to the artifact identity digest')
    s.add_argument('--pair', action='append', required=True, metavar='LABEL=PREDICTIONS.jsonl')
    s.add_argument('--artifact', required=True)
    s.add_argument('--base', required=True)
    s.add_argument('--recipe', required=True)
    s.add_argument('--edit', required=True)
    s.add_argument('--raw', required=True)
    s.add_argument('--out', required=True)
    s.set_defaults(func=cmd_shape)

    c = sub.add_parser('spec', help='write the release-certify spec from saved waves')
    c.add_argument('--arm', required=True)
    c.add_argument('--base', required=True)
    c.add_argument('--recipe', required=True)
    c.add_argument('--edit', required=True)
    c.add_argument('--label', required=True)
    c.add_argument('--removal-wave', required=True)
    c.add_argument('--held-out-wave', required=True)
    c.add_argument('--retention-wave', required=True)
    c.add_argument('--retention-base-wave', required=True)
    c.add_argument('--removal-suite', required=True)
    c.add_argument('--held-out-suite', required=True)
    c.add_argument('--retention-suite', required=True)
    c.add_argument('--shape-removal', required=True)
    c.add_argument('--shape-held-out', required=True)
    c.add_argument('--drift', required=True)
    c.add_argument('--fidelity', required=True)
    c.add_argument('--codec-table')
    c.add_argument('--port', type=int, default=8141)
    c.add_argument('--device', default='Vulkan1')
    c.add_argument('--out', required=True)
    c.add_argument('--copy-out')
    c.set_defaults(func=cmd_spec)

    w = sub.add_parser('summary', help='roll up every wave written by run_waves.sh')
    w.add_argument('--out', required=True)
    w.set_defaults(func=cmd_summary)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
