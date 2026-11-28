# SPDX-License-Identifier: AGPL-3.0-or-later
"""The exact next command must run in the interpreter that produced this report.

Every report ends with the command an agent should run next. A hardcoded
``python`` resolves to whatever is first on PATH — on a workstation that is
often a system interpreter without this package, so the handoff fails at the
first step with an import error that looks like a broken tool. Emitting
``sys.executable`` makes the printed command the one that already worked.
"""
from __future__ import annotations

import shlex
import sys


def module_command(module, *args):
    """Interpreter + module + args, using the interpreter running this process."""
    interpreter = sys.executable or 'python'
    return [interpreter, '-m', module, *[str(arg) for arg in args]]


def module_line(module, *args):
    """Shell-quoted form of :func:`module_command`."""
    return shlex.join(module_command(module, *args))


def cli_command(*args):
    """The jdewitt CLI under the current interpreter."""
    return module_command('jdewitt.cli', *args)


def cli_line(*args):
    """Shell-quoted jdewitt CLI command under the current interpreter."""
    return shlex.join(cli_command(*args))
