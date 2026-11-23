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
    'CERTIFY_MISSING_AXIS': {
        'meaning': 'A Phase 7 packet spec or its evidence is missing a required axis; no packet was written.',
        'agent_action': 'Read the axis named in the message (shape, drift, fidelity, wording inventory, a serve argv) and supply its evidence; never pad the packet.'},
    'CERTIFY_PROBE_ONLY': {
        'meaning': 'The row count or prompt kinds say this is a probe, not a certified row.',
        'agent_action': 'Run the full suite at its recorded row count; a probe result is a ranking input, not a row.'},
    'CERTIFY_CONFIG_MISMATCH': {
        'meaning': 'A wave was taken at a different serve_config than its spec slot (METHODOLOGY R9).',
        'agent_action': 'Either point the spec at the wave that was actually taken and report the mismatch as pinned_cert_match=false, or re-run the wave at the pinned argv.'},
    'CERTIFY_UNMEASURED': {
        'meaning': 'A wave records measured!=true, or zero rows generated; an all-invalid screen is UNMEASURED.',
        'agent_action': 'Re-run the wave; do not read a zero-valid screen as a removal or retention result.'},
    'CERTIFY_ONE_INVENTORY': {
        'meaning': 'A removal count carries one wording inventory, or the spec pins an inventory the shipped checker does not match.',
        'agent_action': 'Regrade the saved answers under both inventories (scripts/screen_agreement.py) and record both digests.'},
    'CERTIFY_RETENTION_INCOMPLETE': {
        'meaning': 'Retention has no paired rows or no usable median answer length.',
        'agent_action': 'Regenerate the missing side at the same suite and config; a rate without a length cannot separate a lost answer from a shorter one.'},
    'CERTIFY_ARTIFACT_MISMATCH': {
        'meaning': 'A shape audit or artifact identity does not belong to the packet artifact (METHODOLOGY R8).',
        'agent_action': 'Re-audit the packet artifact or fix the recorded identity; an audit of another artifact is not evidence.'},
    'MATRIX_MANIFEST_INVALID': {
        'meaning': 'The release-matrix manifest cannot be counted: bad version/kind, a lineage registered twice, a missing registered floor, or a suite file shorter than the floor it registers.',
        'agent_action': 'Fix the manifest field the message names; the matrix writes nothing on a bad manifest and no row is silently dropped.'},
    'MATRIX_LEDGER_INVALID': {
        'meaning': 'The registered loss ledger cannot be published: a measured number has no in-clone source or recompute command, its quoted value no longer recomputes, or a void number cites no evidence-map claim with a void status.',
        'agent_action': 'Fix the ledger entry the message names (source, locator, command, or claim status); a loss with no artifact does not get printed as if it had one.'},
    'BANK_MISS': {
        'meaning': 'The Space bank has no entry for the architecture the GGUF declares; nothing is edited.',
        'agent_action': 'Add a fitted entry for that architecture, or refuse the artifact with this code; never guess an axis from another model.'},
    'GEOMETRY_MISMATCH': {
        'meaning': 'A bank entry exists for the architecture but its hidden width, layer count or footprint does not match the artifact; the message names the missing layers.',
        'agent_action': 'Check the artifact\'s own tensor table (bank-check prints the discovered layers and widths); widen the entry only from a fit on that geometry.'},
    'UNSUPPORTED_QUANT': {
        'meaning': 'A target tensor carries a source quant type this build cannot decode; nothing is edited.',
        'agent_action': 'Convert the artifact to a covered type (q4_k/q5_k/q6_k/q8_0) or teach this build that codec; do not requantize blind.'},
    'BUNDLE_BAD_DIGEST': {
        'meaning': 'A bank entry\'s bundle is missing or does not hash to the digest the entry pins.',
        'agent_action': 'Restore the bundle from the bank release and re-verify its sha256; never apply an axis whose bytes are not the measured ones.'},
    'CODEC_FLOOR': {
        'meaning': 'The edit does not survive any storage type the entry allows at its own edit.min_cos; nothing is written.',
        'agent_action': 'Keep the artifact unedited or fit the entry for a higher-bit artifact class. Do not raise the strength or drop the floor to force a write: that ships an edit nobody measured.'},
    'EDIT_TYPE_NOT_ALLOWED': {
        'meaning': 'The requested --edit-type is outside the types the bank entry measured for this artifact class.',
        'agent_action': 'Pass auto (the ladder intersects the entry\'s types) or one of the entry\'s edit.types; do not widen the entry to the type you want.'},
    'BUNDLE_REFUSED': {
        'meaning': 'A bank entry\'s bundle did not install, verify, or carry the key/geometry the entry claims; nothing is written.',
        'agent_action': 'Re-import the bundle from the bank release and re-pin its sha256 in bank.json; a bundle that disagrees with its entry is not the axis that was measured.'},
    'OUT_IS_INPUT': {
        'meaning': 'The requested --out is the input artifact; bank-apply never edits the file it read.',
        'agent_action': 'Write to a new path; the input must stay readable for the delivered-cos measurement.'},
    'OUT_DIRECTORY_MISSING': {
        'meaning': 'The requested --out names a directory that does not exist.',
        'agent_action': 'Create the directory or write inside an existing one; nothing was written.'},
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


class CertifyRefused(ValueError):
    """A certification packet cannot be assembled honestly; ``code`` names the axis.

    Phase 7's packet is only a row when every axis is present and measured. A
    missing axis, a wave taken at another serve config, one wording inventory or an
    audit of a different artifact are refusals with their own names, so a caller
    can tell which axis to repair instead of reading prose.
    """
    def __init__(self, code, message):
        self.code = code
        super().__init__(f'{code}: {message}')


class MatrixRefused(ValueError):
    """The release matrix cannot be counted at all; ``code`` names the rule.

    Row-level shortcuts (a narrowed suite, a lowered cap, a scale past the grid, a
    head-to-head at two configs) are row statuses in the emitted matrix and exit
    non-zero with the row named. This exception is for the manifest itself: when
    the registration is contradictory there is no honest count to print, so
    nothing is written.
    """
    def __init__(self, code, message):
        self.code = code
        super().__init__(f'{code}: {message}')


class ReleaseRefused(ValueError):
    """A phase-6 release cannot be assembled honestly; ``code`` names the reason.

    The phase-6 release is the edit on the artifact the user already serves, so a
    directory that omits the adapter, the merged twin or the plan card, or that
    ships labels whose digests do not bind the shipped bytes, is refused before a
    staging directory becomes visible. Distinct from INPUT_ERROR so scripts can
    grep one vocabulary (jdewitt.schemas names the same shape problems in
    verify-artifact).
    """
    def __init__(self, code, message):
        self.code = code
        super().__init__(f'{code}: {message}')


class BankRefused(ValueError):
    """The Space direction bank cannot describe this artifact; ``code`` names why.

    BANK_MISS: no entry for the architecture the artifact declares.
    GEOMETRY_MISMATCH: hidden/layer/footprint disagree; the message names the
    missing layers. UNSUPPORTED_QUANT: a source tensor type this build cannot
    decode. BUNDLE_BAD_DIGEST: the entry's bundle does not hash to its pinned
    digest. Every one is raised before a byte is read from the bundle's axis, so
    a Space never edits with an axis it could not verify.
    """
    def __init__(self, code, message):
        self.code = code
        super().__init__(f'{code}: {message}')


class BankApplyRefused(ValueError):
    """A resolved bank entry still cannot be applied to this artifact; named why.

    Distinct from ``BankRefused`` (the bank cannot describe the artifact at all):
    here the entry and the bundle are the bytes the bank pins, and the refusal is
    about the write itself. CODEC_FLOOR: no type the entry allows stores the edit
    at ``edit.min_cos`` — raising the strength to hide that is exactly what this
    code exists to prevent. EDIT_TYPE_NOT_ALLOWED: a named type outside the
    entry's measured set. BUNDLE_REFUSED: the bundle does not install, verify, or
    carry the key/geometry the entry claims. OUT_IS_INPUT / OUT_DIRECTORY_MISSING:
    the requested output cannot be written without touching the input or a missing
    directory. Every one is raised before the writer opens ``out``.
    """
    def __init__(self, code, message):
        self.code = code
        super().__init__(f'{code}: {message}')


def error_code(exception):
    """Stable machine-readable code for expected CLI/API failures."""
    if isinstance(exception, (AdapterCardRefused, ForeignArtifactRefused, CertifyRefused,
                             MatrixRefused, ReleaseRefused, BankRefused, BankApplyRefused)):
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
