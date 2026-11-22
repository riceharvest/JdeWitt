#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Audit the agent-facing CLI surface: every help path model-free, `commands` complete.

One JSON on stdout:

  subparsers            top-level subcommands in the live jdewitt.cli parser
  help_ok               top-level subcommands whose `<cmd> --help` exits 0 with non-empty text
  help_failed           [{command, exit, error}] for the rest, plus '(root)' for `jdewitt --help`
                        and any nested group (e.g. `spanfit build`) whose help path fails
  model_import_attempts modules a probe tried to import from torch/transformers/bitsandbytes
  commands_covered      parser subcommands named by `commands`
  commands_missing      parser subcommands absent from `commands`
  commands_extra        `commands` entries absent from the parser (impossible while both walk one tree; kept visible)
  next_documented       agent steps whose handler source emits a `next` report key
  next_missing          agent steps with no documented `next`

Every probe runs in a fresh subprocess with the same meta-path blocker
tests/test_agent_interface.py installs, so a model import on a help path raises
instead of loading, and a module already imported by an earlier probe cannot
mask it. Exit 0 iff help_ok == subparsers == commands_covered, nothing failed and
no model import was attempted; otherwise exit 1.
"""
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = 120
ATTEMPT = re.compile(r'Model runtime imported by metadata command: ([\w.]+)')

BLOCKER = '''import sys, importlib.abc
class NoModels(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in {'torch','transformers','bitsandbytes'}:
   raise AssertionError('Model runtime imported by metadata command: '+fullname)
sys.meta_path.insert(0,NoModels())
'''

CLI = BLOCKER + 'from jdewitt.cli import main\nmain()\n'

ENUMERATE = BLOCKER + '''import argparse, json
from jdewitt.cli import build_parser

def walk(parser, prefix):
    found = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, child in action.choices.items():
                path = prefix + [name]
                found.append(path)
                found.extend(walk(child, path))
    return found

paths = walk(build_parser(), [])
print(json.dumps({'paths': paths}))
'''

# Agent-facing entry points that must hand the next command back in their report.
# export-smoke is audited automatically once its subparser exists.
AGENT_STEPS = ('capture', 'tune', 'edit-quantized', 'hub-census')
FUTURE_STEPS = ('export-smoke',)


def probe(code, *arguments):
    """Run one probe in a fresh interpreter; never let a failure kill the audit."""
    try:
        result = subprocess.run([sys.executable, '-c', code, *map(str, arguments)],
                                cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired as error:
        return {'exit': None, 'stdout': error.stdout or '', 'stderr': f'timeout after {TIMEOUT}s'}
    return {'exit': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


def attempts(*probes):
    found = set()
    for result in probes:
        found.update(ATTEMPT.findall(result['stderr'] or ''))
    return sorted(found)


def handler_modules(source, step):
    """Modules imported by cli.main()'s dispatch branch for one command.

    Read from the live dispatch so a future command is audited without editing
    this script: the branch that handles the command names the module that
    builds its report.
    """
    tree = ast.parse(source)
    entry = next((node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'main'), None)
    modules = []
    for node in ast.walk(entry) if entry else []:
        if not isinstance(node, ast.If):
            continue
        if step not in commands_matched(node.test):
            continue
        for statement in node.body:
            if isinstance(statement, ast.ImportFrom) and statement.module:
                modules.append(statement.module)
    return modules


def commands_matched(test):
    names = set()
    if isinstance(test, ast.Compare) and isinstance(test.left, ast.Attribute) and test.left.attr == 'command':
        for operator, comparator in zip(test.ops, test.comparators):
            if isinstance(operator, ast.Eq) and isinstance(comparator, ast.Constant) \
                    and isinstance(comparator.value, str):
                names.add(comparator.value)
            elif isinstance(operator, ast.In) and isinstance(comparator, (ast.Set, ast.Tuple, ast.List)):
                names.update(element.value for element in comparator.elts
                             if isinstance(element, ast.Constant) and isinstance(element.value, str))
    return names


def documents_next(path):
    """True when the module emits a dict/subscript key literally named 'next'."""
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            if any(isinstance(key, ast.Constant) and key.value == 'next' for key in node.keys):
                return True
        elif isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Constant) and node.slice.value == 'next':
                return True
    return False


def main():
    enumeration = probe(ENUMERATE)
    if enumeration['exit'] != 0 or not enumeration['stdout'].strip():
        print(json.dumps({'subparsers': 0, 'help_ok': 0,
                          'help_failed': [{'command': '(parser)', 'exit': enumeration['exit'],
                                           'error': enumeration['stderr'].strip()[-2000:]}],
                          'model_import_attempts': attempts(enumeration),
                          'commands_covered': 0, 'commands_missing': [], 'commands_extra': [],
                          'next_documented': [], 'next_missing': []}, indent=2))
        raise SystemExit(1)
    paths = [tuple(path) for path in json.loads(enumeration['stdout'])['paths']]
    top = sorted({path[0] for path in paths if len(path) == 1})

    root_help = probe(CLI, '--help')
    help_failed = []
    if root_help['exit'] != 0 or not root_help['stdout'].strip():
        help_failed.append({'command': '(root)', 'exit': root_help['exit'],
                            'error': root_help['stderr'].strip()[-2000:]})
    help_ok = 0
    probed = [enumeration, root_help]
    for path in paths:
        result = probe(CLI, *path, '--help')
        probed.append(result)
        good = result['exit'] == 0 and bool(result['stdout'].strip())
        if good and len(path) == 1:
            help_ok += 1
        elif not good:
            help_failed.append({'command': ' '.join(path), 'exit': result['exit'],
                                'error': (result['stderr'] or result['stdout']).strip()[-2000:]})

    manifest_result = probe(CLI, 'commands')
    described = set()
    if manifest_result['exit'] == 0:
        try:
            described = {row['command'] for row in json.loads(manifest_result['stdout'])['commands']}
        except (ValueError, KeyError):
            help_failed.append({'command': 'commands', 'exit': manifest_result['exit'],
                                'error': 'commands did not print its JSON manifest'})
    else:
        help_failed.append({'command': 'commands', 'exit': manifest_result['exit'],
                            'error': manifest_result['stderr'].strip()[-2000:]})
    probed.append(manifest_result)

    source = (ROOT / 'jdewitt' / 'cli.py').read_text()
    steps = list(AGENT_STEPS) + [step for step in FUTURE_STEPS if step in top]
    next_documented, next_missing = [], []
    for step in steps:
        modules = handler_modules(source, step) or ['cli']
        emitted = [module for module in modules
                   if documents_next(ROOT / 'jdewitt' / (module.replace('.', '/') + '.py'))]
        (next_documented if emitted else next_missing).append(step)

    report = {'subparsers': len(top),
              'help_ok': help_ok,
              'help_failed': help_failed,
              'model_import_attempts': attempts(*probed),
              'commands_covered': len(set(top) & described),
              'commands_missing': sorted(set(top) - described),
              'commands_extra': sorted(described - set(top)),
              'next_documented': next_documented,
              'next_missing': next_missing}
    print(json.dumps(report, indent=2))
    clean = (report['help_ok'] == report['subparsers'] == report['commands_covered']
             and not report['help_failed'] and not report['commands_extra']
             and not report['model_import_attempts'])
    raise SystemExit(0 if clean else 1)


if __name__ == '__main__':
    main()
