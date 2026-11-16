# SPDX-License-Identifier: AGPL-3.0-or-later
"""Expected experiment outcomes, distinct from infrastructure/programming failures."""
import json
import sqlite3
import subprocess


class SearchRejected(RuntimeError):
    """No candidate passed development-selection gates."""


class UnsupportedMethod(ValueError):
    """The actual model layout cannot execute this configured method."""


class DirectionRejected(ValueError):
    """Observed directions cannot meet rank/stability gates for this method."""


class HoldoutRejected(RuntimeError):
    """The sealed winner failed; never use this as a fallback-selection signal."""


class CancelRequested(RuntimeError):
    """run-cancel stopped the scheduler before asking for a new trial."""


EXPLAINER = {
    'SEARCH_REJECTED': {'meaning': 'The searcher finished without any candidate passing selection.',
                        'agent_action': 'Do not lower gates to force a result; inspect suite and judge validity first.'},
    'UNSUPPORTED_METHOD': {'meaning': 'The requested method is not implemented in this build.',
                           'agent_action': 'Run `methods` and choose from the implemented list.'},
    'DIRECTION_REJECTED': {'meaning': 'Fitted directions failed validation (degenerate or rejected).',
                           'agent_action': 'Check suite family structure, capture mode and rank; refit.'},
    'HOLDOUT_REJECTED': {'meaning': 'The sealed holdout failed for the selected winner.',
                         'agent_action': 'Treat the release as unproven. Never rescue the run by re-picking a runner-up.'},
    'CANCEL_REQUESTED': {'meaning': 'run-cancel stopped the scheduler before asking for new trials.',
                         'agent_action': 'Resume the same output directory when ready; persisted state is reused.'},
    'BUDGET_EXHAUSTED': {'meaning': 'A per-job generation-token or judge-request budget ran out.',
                         'agent_action': 'Raise the budget or resume; saved generations and scores are not redone.'},
    'INVALID_JSON': {'meaning': 'A config or artifact file was not parseable JSON.',
                     'agent_action': 'Point at the offending path in the message; repair the file.'},
    'FILE_EXISTS': {'meaning': 'Refusing to overwrite an existing artifact.',
                    'agent_action': 'Choose another output path; nothing is ever silently clobbered.'},
    'FILE_NOT_FOUND': {'meaning': 'A referenced file or directory does not exist.',
                       'agent_action': 'Create or locate the required input; suites/score dirs must exist first.'},
    'PERMISSION_DENIED': {'meaning': 'Filesystem permissions blocked the operation.',
                          'agent_action': 'Fix ownership/mode; do not retry blindly.'},
    'STATE_DB_ERROR': {'meaning': 'The optuna study.db is unreadable or busy.',
                       'agent_action': 'Check run-status run_lock_held before retrying; a live worker may be writing.'},
    'SUBPROCESS_ERROR': {'meaning': 'A converter/server subprocess failed.',
                         'agent_action': 'Read the captured stderr in the message; this is not a schema or config problem.'},
    'INPUT_ERROR': {'meaning': 'Invalid argument or configuration value; the message names the field.',
                    'agent_action': 'Fix the named field; --set values are JSON-parsed, quote deliberately.'},
    'OPERATION_FAILED': {'meaning': 'Runtime failure with no more specific code.',
                         'agent_action': 'The message carries the failing stage; report it verbatim.'},
    'IO_ERROR': {'meaning': 'Filesystem-level I/O failure.', 'agent_action': 'Check mount and disk state; not a schema problem.'},
    'ARGUMENT_ERROR': {'meaning': 'The CLI arguments themselves were invalid (exit code 2).',
                       'agent_action': 'Run `commands` for the exact flag table instead of guessing.'},
}


def explain_error(code):
    """Single-row reference for a stable CLI error code."""
    if code not in EXPLAINER:
        raise ValueError(f'Unknown error code {code!r}; known codes: {", ".join(sorted(EXPLAINER))}')
    return {'code': code, **EXPLAINER[code],
            'envelope': 'Failures print {"error":{"code","type","message","command"}} and exit 1; argument errors exit 2.'}


class BudgetExhausted(RuntimeError):
    """A per-job budget (generated tokens or judge requests) ran out mid-run."""


def error_code(exception):
    """Stable machine-readable code for expected CLI/API failures."""
    for cls, code in ((SearchRejected, 'SEARCH_REJECTED'),
                      (UnsupportedMethod, 'UNSUPPORTED_METHOD'),
                      (DirectionRejected, 'DIRECTION_REJECTED'),
                      (HoldoutRejected, 'HOLDOUT_REJECTED'),
                      (CancelRequested, 'CANCEL_REQUESTED'),
                      (BudgetExhausted, 'BUDGET_EXHAUSTED'),
                      (json.JSONDecodeError, 'INVALID_JSON'),
                      (FileExistsError, 'FILE_EXISTS'),
                      (FileNotFoundError, 'FILE_NOT_FOUND'),
                      (PermissionError, 'PERMISSION_DENIED'),
                      (sqlite3.Error, 'STATE_DB_ERROR'),
                      (subprocess.SubprocessError, 'SUBPROCESS_ERROR'),
                      (ValueError, 'INPUT_ERROR'),
                      (RuntimeError, 'OPERATION_FAILED'),
                      (OSError, 'IO_ERROR')):
        if isinstance(exception, cls):
            return code
    return type(exception).__name__.upper()
