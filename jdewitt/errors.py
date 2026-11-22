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


class NoVerifiedAdapter(ValueError):
    """A layout this tool cannot name to an adapter it can verify, so nothing is edited.

    Distinct from INPUT_ERROR on purpose: a typo'd flag value and "this checkpoint's
    block carries no residual-write projection I can name" are different problems
    with different next moves, and the message carries the discovered facts (layer
    ids with no target, linear module / tensor names seen, width-matching names that
    would not be edited) so a caller can list what was seen instead of guessing a
    tensor name.

    The honest boundary: discovery proves a tensor NAME exists at the expected
    width, not that the architecture serves the edit. Certifying a layout is a
    separate gate (export smoke); this code never means "the layout is broken",
    only "no adapter for it could be verified here".
    """

    def __init__(self, message, **facts):
        super().__init__(message)
        self.facts = facts


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
    'NO_VERIFIED_ADAPTER': {
        'meaning': 'The layout has no adapter this tool can name and verify: no residual-write '
                   'projection at the expected width was found, or the source key has no tensor '
                   'the artifact carries. Nothing was edited.',
        'agent_action': 'List what was discovered and what could not be named instead of guessing a '
                        'tensor: `python -m jdewitt.spanfit targets --gguf-in <artifact>` and '
                        '`architectures`. Do not retry with another tensor name; the checkpoint and '
                        'the edit footprint have to agree on the hidden width first.'},
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
    'SYMLINK_REFUSED': {'meaning': 'A symlink appeared where plain bytes were required (directory or entry).',
                        'agent_action': 'Point --foreign at the real directory and unpack the release '
                                        'without links; nothing is hashed through a link.'},
    'NONREGULAR_REFUSED': {'meaning': 'A socket, fifo or device was found in the directory.',
                           'agent_action': 'Inspect the directory by hand; only regular files can be '
                                           'hashed and reported.'},
    'PATH_TRAVERSAL_REFUSED': {'meaning': 'A filename was not a plain name inside the directory.',
                               'agent_action': 'Treat the directory as hostile; do not retry with a '
                                               'resolved path that points outside it.'},
    'FOREIGN_AMBIGUOUS': {'meaning': '--foreign was pointed at a directory that carries files.json.',
                          'agent_action': 'Use the native path: verify-artifact DIR --all (or '
                                          'verify-release for a release bundle).'},
    'HEADER_INVALID': {'meaning': 'A file named .gguf does not carry a readable GGUF header '
                                  '(bad magic, truncated, or tensor data past end of file).',
                       'agent_action': 'Re-download or re-export that file; the message names it. '
                                       'Nothing else in the directory was certified.'},
    'NOT_A_DIRECTORY': {'meaning': '--foreign needs a directory, not a file.',
                        'agent_action': 'Pass the release directory; for one artifact use the '
                                        'native path verify-artifact FILE --name SCHEMA.'},
}


def explain_error(code):
    """Single-row reference for a stable CLI error code."""
    if code not in EXPLAINER:
        raise ValueError(f'Unknown error code {code!r}; known codes: {", ".join(sorted(EXPLAINER))}')
    return {'code': code, **EXPLAINER[code],
            'envelope': 'Failures print {"error":{"code","type","message","command"}} and exit 1; argument errors exit 2.'}


class BudgetExhausted(RuntimeError):
    """A per-job budget (generated tokens or judge requests) ran out mid-run."""


class ForeignArtifactRefused(ValueError):
    """A foreign release directory cannot be inspected under the stated rules.

    ``code`` names the rule that stopped it (SYMLINK_REFUSED, NONREGULAR_REFUSED,
    PATH_TRAVERSAL_REFUSED, FOREIGN_AMBIGUOUS, HEADER_INVALID, NOT_A_DIRECTORY)
    so a caller can branch on the envelope instead of parsing prose.
    """
    def __init__(self, code, message):
        self.code = code
        super().__init__(f'{code}: {message}')


class AdapterCardRefused(ValueError):
    """A shipped adapter cannot be labelled honestly (R8); ``code`` names the reason.

    Distinct from a generic ValueError so the CLI envelope carries the refusal
    name instead of "INPUT_ERROR" and a reader can grep one vocabulary.
    """
    def __init__(self, code, message):
        self.code = code
        super().__init__(f'{code}: {message}')


def error_code(exception):
    """Stable machine-readable code for expected CLI/API failures."""
    if isinstance(exception, (AdapterCardRefused, ForeignArtifactRefused)):
        return exception.code
    for cls, code in ((SearchRejected, 'SEARCH_REJECTED'),
                      (UnsupportedMethod, 'UNSUPPORTED_METHOD'),
                      (DirectionRejected, 'DIRECTION_REJECTED'),
                      (NoVerifiedAdapter, 'NO_VERIFIED_ADAPTER'),
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
