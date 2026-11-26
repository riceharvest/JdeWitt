#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Local CPU dry run of the Space editor path on the in-clone 0.5B IQ3_M.

No Space, no ZeroGPU, no quota, no capture, no serve: one command reproduces
the Phase-0 editor evidence and writes `verification/p2/space-dry-run.json`.

It exercises the five checks the CPU queue asks for:

  1. the editor report carries edited/untouched counts, delivered cos and the
     input/output sha256;
  2. a second run from the same pristine bytes is byte-identical (sha256 + cmp);
  3. running the editor on its own output answers with a byte copy and
     `status: already_applied` (the file carries `jdewitt.fingerprint`);
  4. an input above the size cap is refused from `stat()` before it is read —
     the probe is a zero-filled sparse file that is not a GGUF at all, so the
     refusal code proves the cap ran first;
  5. `jdewitt gguf-tensor-diff` on pristine vs edited reports every untouched
     tensor bit-identical and only the named tensors differing.

The axis is the synthetic fixture (`space_edit.synthetic_axis`): deterministic
unit vectors from a seed, not a fitted direction. Removal is not measured and
every served claim stays GATED.

    PYTHONPATH=. python scripts/space_dry_run.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GGUF = 'verification/iq05b/Qwen2.5-0.5B-Instruct-IQ3_M.gguf'
DEFAULT_REPORT = 'verification/p2/space-dry-run.json'
DEFAULT_WORK = 'runs/space-dry-run'

BLOCKER = '''import sys, importlib.abc
class NoModels(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in {'torch', 'transformers', 'bitsandbytes'}:
   raise AssertionError('model runtime imported: ' + fullname)
sys.meta_path.insert(0, NoModels())
import jdewitt.space_edit  # noqa: F401
print('ok')
'''

NO_MODEL_ROOTS = {'torch', 'transformers', 'bitsandbytes', 'sentencepiece'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _imports_model_runtime(paths):
    """Any import of a model runtime in the modules the editor path loads."""
    import ast

    hits = []
    for path in paths:
        tree = ast.parse(Path(path).read_text(), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split('.')[0] in NO_MODEL_ROOTS:
                    hits.append({'file': str(Path(path).resolve().relative_to(ROOT)),
                                 'import': name})
    return hits


def _diff_summary(a, b, edited_names):
    """Run the real CLI and fold its JSON into the checks this card names."""
    proc = subprocess.run([sys.executable, '-m', 'jdewitt', 'gguf-tensor-diff', str(a), str(b)],
                          capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f'gguf-tensor-diff exited {proc.returncode}: {proc.stderr[-2000:]}')
    diff = json.loads(proc.stdout)
    states = diff['summary']
    # Every row that is not byte-identical is "differing", whether or not the
    # tool could decode both sides: a row it cannot measure is still not equal.
    differing = sorted(row['name'] for row in diff['differing_tensors'])
    only_before = diff.get('only_in_before') or []
    only_after = diff.get('only_in_after') or []
    cross_grid = sorted(row['name'] for row in diff['differing_tensors']
                        if 'block grids differ' in str(row.get('note') or ''))
    edited = set(edited_names)
    return {
        'command': 'jdewitt gguf-tensor-diff <pristine> <edited>',
        'exit': proc.returncode,
        'tensor_count_before': diff['tensor_count_before'],
        'tensor_count_after': diff['tensor_count_after'],
        'states': states,
        'bit_identical': states['bit_identical'],
        'value_differs': states['value_differs'],
        'codec_unsupported': states['codec_unsupported'],
        'codec_unsupported_names': cross_grid,
        'codec_unsupported_note': (
            'gguf-tensor-diff compares rel_rms only when both sides share a block grid. The edited '
            'tensors whose source type has a different block width (Q4_K: 256 values/block -> Q8_0: '
            '32) are listed as differing but carry no rel_rms; the ones with the same grid (Q5_0 -> '
            'Q8_0) do. Bit identity is a byte compare and is unaffected by this.'),
        'only_in_before': only_before,
        'only_in_after': only_after,
        'differing_count_listed': len(diff['differing_tensors']),
        'accounted_all_tensors': states['bit_identical'] + len(differing) == states['shared'],
        'differing_only_edited': set(differing) <= edited,
        'edited_all_differ': edited <= set(differing),
        'differing_not_edited': sorted(set(differing) - edited),
        'rel_rms': diff['rel_rms'],
        'note': diff['note'],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gguf', default=DEFAULT_GGUF,
                        help=f'visitor artifact to edit (default: {DEFAULT_GGUF})')
    parser.add_argument('--work-dir', default=DEFAULT_WORK,
                        help=f'scratch for the run outputs (default: {DEFAULT_WORK}, gitignored)')
    parser.add_argument('--report', default=DEFAULT_REPORT,
                        help=f'committed evidence JSON (default: {DEFAULT_REPORT})')
    parser.add_argument('--edit-type', default='q8_0', help='storage type for edited tensors')
    parser.add_argument('--strength', type=float, default=1.0, help='edit strength')
    parser.add_argument('--min-cos', type=float, default=0.95, help='delivered-cos floor')
    parser.add_argument('--first', type=int, default=None, help='first layer (default: all)')
    parser.add_argument('--last', type=int, default=None, help='last layer (default: all)')
    parser.add_argument('--cap-bytes', type=int, default=None,
                        help='override the size cap for the refusal probe (default: module cap)')
    parser.add_argument('--keep', action='store_true',
                        help='keep the edited outputs after the checks (default: keep; runs/ is ignored)')
    args = parser.parse_args(argv)

    os.chdir(ROOT)
    from jdewitt.space_edit import (GATED, MAX_INPUT_BYTES, SpaceEditRefused,
                                    edit_artifact)

    src = Path(args.gguf)
    if not src.is_file():
        raise SystemExit(f'input {src} does not exist; the iq05b GGUFs are gitignored, fetch the '
                         f'fixture into verification/iq05b first')
    work = Path(args.work_dir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    started = time.monotonic()
    outputs = {}
    editor_reports = {}
    common = {'strength': args.strength, 'edit_type': args.edit_type,
              'min_cos': args.min_cos, 'first': args.first, 'last': args.last}

    # 1 + 2. Two runs from the same pristine bytes; the second must match byte for byte.
    for index in (1, 2):
        out = work / f'run{index}.gguf'
        report = edit_artifact(src, out, report_path=work / f'run{index}.report.json', **common)
        outputs[index] = out
        editor_reports[index] = report

    first = editor_reports[1]
    run1_sha, run2_sha = sha256(outputs[1]), sha256(outputs[2])
    second_identical = run1_sha == run2_sha
    cmp_proc = subprocess.run(['cmp', '-s', str(outputs[1]), str(outputs[2])])
    if not second_identical or cmp_proc.returncode != 0:
        raise SystemExit(f'second run is not byte-identical: {run1_sha} != {run2_sha} '
                         f'(cmp exit {cmp_proc.returncode})')

    # 3. The output carries jdewitt.fingerprint; editing it again is a byte copy.
    out3 = work / 'already_applied.gguf'
    already = edit_artifact(outputs[1], out3, report_path=work / 'already_applied.report.json',
                            **common)
    if already['status'] != 'already_applied' or not already.get('already_applied'):
        raise SystemExit(f'editing the output answered {already["status"]!r}, not already_applied')
    copy_sha = sha256(out3)
    copy_identical = copy_sha == run1_sha
    if not copy_identical:
        raise SystemExit(f'already_applied output is not a byte copy: {copy_sha} != {run1_sha}')

    # 4. Oversized input: a sparse non-GGUF file above the cap must be refused
    #    from stat() alone. A read would fail with "not a GGUF file" instead.
    cap = int(args.cap_bytes or MAX_INPUT_BYTES)
    oversize = work / 'oversize.gguf'
    with oversize.open('wb') as handle:
        handle.truncate(cap + 1)
    refused_out = work / 'refused.gguf'
    cap_check = {'cap_bytes': cap, 'probe_bytes': oversize.stat().st_size,
                 'probe_is_gguf': False,
                 'probe_note': 'zero-filled sparse file: any read or parse would fail with '
                               '"not a GGUF file", so the cap code proves the refusal ran first'}
    try:
        edit_artifact(oversize, refused_out, report_path=work / 'refused.report.json', **common)
    except SpaceEditRefused as error:
        cap_check.update({'code': error.code, 'message': str(error),
                          'refused': error.code == 'INPUT_TOO_LARGE',
                          'output_written': refused_out.exists()})
    else:
        cap_check.update({'code': None, 'refused': False, 'output_written': refused_out.exists()})
    if not cap_check['refused'] or cap_check['output_written']:
        raise SystemExit(f'size cap did not refuse before read: {cap_check}')
    oversize.unlink()

    # 5. The independent tensor diff.
    diff = _diff_summary(src, outputs[1], first['targets']['names'])
    untouched_ok = (diff['bit_identical'] == first['untouched']['count']
                    and diff['accounted_all_tensors']
                    and not diff['only_in_before'] and not diff['only_in_after']
                    and diff['differing_only_edited'] and diff['edited_all_differ'])

    # The editor path must stay importable with no model runtime (a Space handler
    # imports it in a container whose requirements carry no inference stack).
    import_check = subprocess.run([sys.executable, '-c', BLOCKER], capture_output=True, text=True,
                                  cwd=ROOT,
                                  env={**os.environ, 'PYTHONPATH': str(ROOT),
                                       'PYTHONNOUSERSITE': '1'})
    module_hits = _imports_model_runtime([
        'jdewitt/space_edit.py', 'jdewitt/rewrite.py', 'jdewitt/ggufio.py',
        'jdewitt/ggufwrite.py', 'jdewitt/ggufgeom.py', 'jdewitt/quants.py',
        'jdewitt/gguf_types.py',
    ])
    torch_free = (import_check.returncode == 0 and import_check.stdout.strip() == 'ok'
                  and not module_hits)

    from importlib.metadata import version

    report = {
        'what': ('Phase-0 Space editor path, local CPU dry run: visitor GGUF in, the same file '
                 'edited out, untouched tensors byte-identical. No Space was deployed and no GPU '
                 'second or quota was spent.'),
        'schema': 'jdewitt-space-dry-run-v1',
        'fixture': {
            'path': str(src),
            'bytes': src.stat().st_size,
            'sha256': sha256(src),
            'note': ('gitignored *.gguf; in-clone at verification/iq05b/. This sha256 is the one '
                     'docs/EVIDENCE-MAP.md quotes (0b2e1747...32099).'),
        },
        'editor': {
            'module': 'jdewitt.space_edit',
            'function': 'edit_artifact(src, out, axis=None, edit_type=..., ...)',
            'axis': 'synthetic fixture (deterministic sha256(seed:layer:hidden) unit vectors per '
                    'layer); NOT a fitted removal, no capture read',
            'args': {'edit_type': args.edit_type, 'strength': args.strength,
                     'min_cos': args.min_cos, 'first': args.first, 'last': args.last},
            'architecture': first['input']['architecture'],
            'hidden': first['input']['hidden'],
            'artifact_tensors': first['input']['tensors'],
            'source_types': first['input']['source_types'],
            'footprint_layers': first['footprint']['layers'],
            'target_tensors': first['targets']['tensor_count'],
            'target_kinds': first['targets']['by_kind'],
        },
        'checks': {
            'report_counts_cos_sha256': {
                'edited_count': first['edited']['count'],
                'untouched_count': first['untouched']['count'],
                'delivered_min_cos': first['edited']['min_cos'],
                'delivered_mean_cos': first['edited']['mean_cos'],
                'floor': first['edited']['floor'],
                'floor_met': first['edited']['floor_met'],
                'by_kind': first['edited']['by_kind'],
                'input_sha256': first['input_sha256'],
                'output_sha256': first['output_sha256'],
                'bytes_in': first['bytes_in'],
                'bytes_out': first['bytes_out'],
                'status': first['status'],
            },
            'second_run_byte_identical': {
                'run1_sha256': run1_sha,
                'run2_sha256': run2_sha,
                'identical': second_identical,
                'cmp_exit': cmp_proc.returncode,
                'method': 'sha256 of both files plus `cmp -s` (not a hash comparison alone)',
            },
            'already_applied_byte_copy': {
                'status': already['status'],
                'already_applied': already['already_applied'],
                'input_sha256': already['input_sha256'],
                'output_sha256': already['output_sha256'],
                'identical': copy_identical,
                'edited_count': already['edited']['count'],
                'note': already['edited']['note'],
            },
            'size_cap_refused_before_read': cap_check,
            'untouched_bit_identical': {
                'expected_untouched': first['untouched']['count'],
                'editor_verified': first['untouched']['verified'],
                'bit_identical': diff['bit_identical'],
                'value_differs': diff['value_differs'],
                'codec_unsupported': diff['codec_unsupported'],
                'codec_unsupported_names': diff['codec_unsupported_names'],
                'codec_unsupported_note': diff['codec_unsupported_note'],
                'accounted_all_tensors': diff['accounted_all_tensors'],
                'differing_only_edited': diff['differing_only_edited'],
                'edited_all_differ': diff['edited_all_differ'],
                'differing_not_edited': diff['differing_not_edited'],
                'ok': untouched_ok,
                'diff': diff,
            },
            'import_without_model_runtime': {
                'ok': torch_free,
                'blocker': 'importing torch/transformers/bitsandbytes/sentencepiece raises inside '
                           'a subprocess that imports jdewitt.space_edit',
                'subprocess_exit': import_check.returncode,
                'subprocess_stdout': import_check.stdout.strip()[:200],
                'model_imports_in_editor_path': module_hits,
            },
        },
        'all_checks_pass': bool(second_identical and copy_identical and cap_check['refused']
                                and not cap_check['output_written'] and untouched_ok and torch_free
                                and first['edited']['floor_met']),
        'gated': GATED,
        'gated_note': ('ZeroGPU quota, GPU throughput and every served/behavioural claim stay '
                       'GATED: no Space was deployed, no @spaces.GPU call was made, and this run '
                       'read bytes and measured codec fidelity on this CPU host only.'),
        'host': {
            'platform': platform.platform(),
            'cpu_count': os.cpu_count(),
            'python': platform.python_version(),
            'numpy': version('numpy'),
            'gguf': version('gguf'),
            'wall_seconds': round(time.monotonic() - started, 1),
            'note': 'this host\'s CPU wall clock, NOT a ZeroGPU container; do not quote as throughput',
        },
        'outputs': {
            'run1': str(outputs[1]), 'run2': str(outputs[2]),
            'already_applied': str(out3),
            'editor_reports': {name: str(path) for name, path in (
                ('run1', work / 'run1.report.json'),
                ('run2', work / 'run2.report.json'),
                ('already_applied', work / 'already_applied.report.json'))},
            'note': 'runs/ is gitignored; the numbers above are the committed evidence and the '
                    'whole run-1 editor report is embedded below',
        },
        'editor_report_run1': first,
        'reproduce': 'PYTHONPATH=. python scripts/space_dry_run.py',
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1) + '\n')
    if not args.keep:
        for path in outputs.values():
            path.unlink(missing_ok=True)
        out3.unlink(missing_ok=True)

    print(json.dumps({
        'report': str(report_path),
        'all_checks_pass': report['all_checks_pass'],
        'edited': first['edited']['count'],
        'untouched': first['untouched']['count'],
        'min_cos': first['edited']['min_cos'],
        'mean_cos': first['edited']['mean_cos'],
        'floor_met': first['edited']['floor_met'],
        'output_sha256': first['output_sha256'],
        'second_run_identical': second_identical,
        'already_applied_identical': copy_identical,
        'size_cap_refused': cap_check['refused'] and not cap_check['output_written'],
        'untouched_bit_identical': diff['bit_identical'],
        'torch_free': torch_free,
    }, indent=1))
    return 0 if report['all_checks_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
