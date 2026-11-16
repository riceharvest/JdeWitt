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
