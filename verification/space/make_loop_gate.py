#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rebuild `verification/space/loop.json` from a real dry run and the gate checks.

    python verification/space/make_loop_gate.py

It runs `space_loop.dry_run` on the committed fixture hub (no GPU, no network),
stages a tree into a temporary directory, runs the banned-word grep, `ruff` and the
Space's tests, and writes one JSON. Timings and temp paths are kept out of the
gate except for the per-step seconds the dry run itself measured.
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jdewitt import space_loop as loop  # noqa: E402

FIXTURE = ROOT / 'verification' / 'space' / 'fixtures' / 'loop'
OUT = ROOT / 'verification' / 'space' / 'loop.json'
MODEL, QUANT = 'jdewitt/loop-fixture', 'Q4_K_M'


def _stage_module():
    spec = importlib.util.spec_from_file_location('space_stage_gate',
                                                  ROOT / 'scripts' / 'space_stage.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage_check():
    stage = _stage_module()
    with tempfile.TemporaryDirectory(prefix='space-stage-gate-') as directory:
        out = Path(directory) / 'tree'
        stage.stage(out)
        files = [path.relative_to(out).as_posix() for path in sorted(out.rglob('*'))
                 if path.is_file()]
        vendored = sorted(name for name in files if name.startswith('jdewitt/'))
        app_bytes = (out / 'app.py').read_bytes()
        return {
            'files': len(files),
            'app_is_the_repo_file': app_bytes == (ROOT / 'spaces' / 'app.py').read_bytes(),
            'vendored': vendored,
            'quantize_vendored': 'jdewitt/quantize.py' in vendored,
            'requirements': (out / 'requirements.txt').read_text().split(),
        }


def _grep():
    result = subprocess.run(
        ['grep', '-rE', 'uncensored|refusals removed', 'spaces/', 'jdewitt/space_loop.py'],
        cwd=ROOT, capture_output=True, text=True)
    return {'patterns': 'uncensored|refusals removed', 'matches': result.stdout.strip(),
            'exit': result.returncode, 'empty': result.returncode == 1}


def _ruff():
    result = subprocess.run([sys.executable, '-m', 'ruff', 'check', 'jdewitt', 'scripts'],
                            cwd=ROOT, capture_output=True, text=True)
    return {'exit': result.returncode, 'tail': (result.stdout or result.stderr).strip().splitlines()[-1]}


def _pytest():
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/test_space_loop.py', 'tests/test_space_stage.py',
         '-q', '-o', 'addopts='],
        cwd=ROOT, capture_output=True, text=True)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    tail = lines[-1] if lines else (result.stdout + result.stderr).strip()[-300:]
    return {'exit': result.returncode, 'summary': tail}


def _case(case):
    result = case['result']
    record = {'expected': case['expected'], 'repo': case['repo'], 'revision': case['revision'],
              'file': case['file'], 'label': result['label'],
              'workspace_deleted': result['workspace_deleted'],
              'files_present': case['files_present'],
              'steps': [{key: step[key] for key in ('step', 'state', 'seconds', 'detail', 'code')}
                        for step in result['steps']]}
    if result['label'] == 'refused':
        record['refusal'] = result['refusal']
    else:
        record['card'] = result['card']
        record['readback'] = {key: result['readback'][key]
                              for key in ('input_sha256', 'output_sha256', 'bytes_in',
                                          'bytes_out', 'reopened', 'ok')}
        record['readback']['edited'] = {key: result['readback']['edited'][key]
                                        for key in ('count', 'min_cos', 'mean_cos', 'floor')}
        record['files'] = {key: Path(path).name for key, path in result['files'].items()
                           if key != 'bytes'}
    return record


def main():
    report = loop.dry_run(ROOT / 'spaces' / 'bank', MODEL, QUANT, FIXTURE)
    gate = {
        'schema': 'jdewitt-space-loop-gate-v1',
        'command': ('python -m jdewitt space-dry-run --bank spaces/bank --model '
                    f'{MODEL} --quant {QUANT} --fixture verification/space/fixtures/loop'),
        'dry_run': {
            'ok': report['ok'],
            'labels': report['labels'],
            'caveat_above_file': report['caveat_above_file'],
            'panel_order': report['panel_order'],
            'panel': report['panel'],
            'quant_control': report['quant_control'],
            'network': report['network'],
            'cases': [_case(case) for case in report['cases']],
        },
        'checks': {
            'banned_grep': _grep(),
            'ruff': _ruff(),
            'pytest': _pytest(),
            'stage': _stage_check(),
        },
        'claims': {
            'no_network': ('the fixture hub serves file lists, header prefixes and licence facts '
                           'from disk; the dry run imports no Hub client'),
            'header_only': ('each coverage step read a 512-byte prefix of a 295 488-byte artifact '
                            'and padded the copy to the declared size as a sparse file'),
            'labels': ('all three states come from entry.measurement.state narrowed to the '
                       'artifact class; the refusal sentence names the architecture'),
            'workspace': 'each job deleted its workspace; the four delivered files existed',
        },
    }
    OUT.write_text(json.dumps(gate, indent=2) + '\n')
    print(json.dumps({'out': str(OUT), 'ok': report['ok'],
                      'checks': {key: value.get('empty', value.get('exit'))
                                 for key, value in gate['checks'].items()}}, indent=2))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
