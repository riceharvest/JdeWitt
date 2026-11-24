# SPDX-License-Identifier: AGPL-3.0-or-later
"""The P7 Spark spec writer must produce a spec release-certify can read.

Synthetic fixtures only (a 4-byte "artifact" and hand-written wave dirs): the
test exercises the assembly rules — the per-axis argv must equal what the saved
wave recorded, every shape audit must carry the artifact identity digest the
certifier recomputes, and the wording-inventory digests must be the shipped
ones. It never claims to measure a model.
"""
import importlib.util
import json
from pathlib import Path

from jdewitt.certify import _identity_digest, inventory_digests

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        'p7_spark_waves', ROOT / 'scripts' / 'p7_spark_waves.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wave(directory, artifact, cap, rows):
    module = _module()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'predictions.jsonl').write_text(
        ''.join(json.dumps({'id': f'row-{index}', 'status': 'ok', 'answer': 'x'}) + '\n'
                for index in range(rows)))
    (directory / 'wave.json').write_text(json.dumps({
        'measured': True, 'rows': rows, 'suite': 'synthetic.jsonl',
        'serve_config': module.serve_config(str(artifact), cap, 8141, 'Vulkan1'),
    }))


def test_spec_binds_waves_identity_and_inventory(tmp_path, monkeypatch):
    module = _module()
    base = tmp_path / 'base.gguf'
    arm = tmp_path / 'arm.gguf'
    base.write_bytes(b'base')
    arm.write_bytes(b'arm!')
    recipe = 'python -m jdewitt edit-quantized --gguf base.gguf --out arm.gguf'
    identity, digest = module.artifact_identity(arm, recipe, 'synthetic edit', base)

    waves = {}
    for name, axis, rows in (('removal', 'removal', 3), ('heldout', 'held_out', 4),
                             ('retention', 'retention', 5)):
        directory = tmp_path / 'waves' / name
        cap = module.CAPS[axis]
        _wave(directory, arm, cap, rows)
        waves[axis] = directory
    _wave(tmp_path / 'waves' / 'retention-base', base, module.CAPS['retention'], 5)

    for name, axis in (('removal', 'removal'), ('heldout', 'held_out')):
        document = {'audits': [{'label': name, 'predictions': str(waves[axis] / 'predictions.jsonl'),
                                'rows': 3 if axis == 'removal' else 4,
                                'shapes': {'flat_decline': 3}}],
                    'artifact_identity_digest': digest}
        (tmp_path / f'shape-{name}.json').write_text(json.dumps(document))

    (tmp_path / 'drift.json').write_text(json.dumps(
        {'measured': True, 'control_delta_ln_ppl': 0.01, 'budget': 0.02}))
    (tmp_path / 'fidelity.json').write_text(json.dumps(
        {'measured': True, 'min_cos': 0.98, 'mean_cos': 0.99, 'sampled': 2,
         'tensors': [{'name': 'blk.0.attn_output.weight', 'qtype': 'Q8_0', 'cos': 0.98}]}))
    (tmp_path / 'codec-table.json').write_text(json.dumps(
        {'chosen_type': 'Q8_0', 'verdict': 'cleared'}))

    out = tmp_path / 'spec.json'
    copy = tmp_path / 'release' / 'spec.json'
    code = module.main([
        'spec', '--arm', str(arm), '--base', str(base), '--recipe', recipe,
        '--edit', 'synthetic edit', '--label', 'synthetic',
        '--removal-wave', str(waves['removal']),
        '--held-out-wave', str(waves['held_out']),
        '--retention-wave', str(waves['retention']),
        '--retention-base-wave', str(tmp_path / 'waves' / 'retention-base'),
        '--removal-suite', 'synthetic-removal.jsonl',
        '--held-out-suite', 'synthetic-heldout.jsonl',
        '--retention-suite', 'synthetic-retention.jsonl',
        '--shape-removal', str(tmp_path / 'shape-removal.json'),
        '--shape-held-out', str(tmp_path / 'shape-heldout.json'),
        '--drift', str(tmp_path / 'drift.json'),
        '--fidelity', str(tmp_path / 'fidelity.json'),
        '--codec-table', str(tmp_path / 'codec-table.json'),
        '--port', '8141', '--device', 'Vulkan1',
        '--out', str(out), '--copy-out', str(copy),
    ])
    assert code == 0
    written = json.loads(out.read_text())
    assert copy.read_text() == out.read_text()

    # The identity the packet will recompute, over the same payload.
    assert written['artifact'] == identity
    assert _identity_digest(written['artifact']) == digest
    assert written['arm']['chosen_type'] == 'Q8_0'
    assert written['arm']['realized_cos'] == 0.98
    assert written['arm']['byte_cost'] == 4
    assert written['arm']['scale'] == 1.0
    # One argv per axis, and it is the one the wave recorded.
    for axis, directory in waves.items():
        recorded = json.loads((directory / 'wave.json').read_text())['serve_config']
        assert written['configs'][axis]['argv'] == recorded
    assert written['inputs']['wording_inventory'] == inventory_digests()
    assert [entry['axis'] for entry in written['inputs']['shape_audit']] == ['removal', 'held_out']
    assert written['inputs']['removal']['wave'] == str(waves['removal'])


def test_spec_refuses_a_wave_at_another_config(tmp_path):
    module = _module()
    base = tmp_path / 'base.gguf'
    arm = tmp_path / 'arm.gguf'
    base.write_bytes(b'base')
    arm.write_bytes(b'arm!')
    directory = tmp_path / 'waves' / 'removal'
    directory.mkdir(parents=True)
    (directory / 'predictions.jsonl').write_text(
        json.dumps({'id': 'row', 'status': 'ok', 'answer': 'x'}) + '\n')
    (directory / 'wave.json').write_text(json.dumps({
        'measured': True, 'rows': 1,
        'serve_config': module.serve_config(str(arm), 1024, 8134, 'Vulkan0'),
    }))
    try:
        module.main([
            'spec', '--arm', str(arm), '--base', str(base), '--recipe', 'r', '--edit', 'e',
            '--label', 'x', '--removal-wave', str(directory),
            '--held-out-wave', str(directory), '--retention-wave', str(directory),
            '--retention-base-wave', str(directory),
            '--removal-suite', 'a', '--held-out-suite', 'b', '--retention-suite', 'c',
            '--shape-removal', 'd', '--shape-held-out', 'f', '--drift', 'g', '--fidelity', 'h',
            '--out', str(tmp_path / 'spec.json'),
        ])
    except SystemExit as error:
        assert 'R9' in str(error)
    else:  # pragma: no cover - the refusal is the point
        raise AssertionError('spec accepted a wave served at another config')
