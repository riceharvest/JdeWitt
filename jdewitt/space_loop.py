# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Space's first honest loop: tell a visitor whether the bank covers them.

`inspect(repo, revision, quant)` answers the two questions that can be answered
before a byte is downloaded or read:

  * does the direction bank have an entry for this base, at a source quant this
    build claims to serve (`BANK_MISS` / `UNSUPPORTED_QUANT` when it does not);
  * is the licence verified (it is **not**, yet: the licence gate is another
    card, and until it lands this report says `UNVERIFIED`).

Nothing here downloads a model, reads tensor data, edits a tensor or generates
a token. `edit.state` is `none` and says so; the app renders that verbatim so a
visitor is never shown a download button that would lie about what happened.

Coverage is deliberately a bank question, not an artifact question: the bank
entry names the base it was fitted on (`base.repo`, `base.revision`), so the
answer is available from a few KB of JSON. The artifact-level checks — geometry,
footprint, decodable source type, bundle digest — are `jdewitt.bank.resolve`'s
job and run later, against the file's header, by the card that adds the download.

`import jdewitt.space_loop` must work without gradio and without torch; the
module imports only `jdewitt.bank` (torch-free by construction) and the standard
library. The UI lives in the staged `app.py`.
"""
import re
from pathlib import Path

from .bank import SCHEMA as BANK_SCHEMA, load_bank

SCHEMA = 'jdewitt-space-inspect-v1'

LICENCE = {
    'state': 'UNVERIFIED',
    'detail': ('the licence and access gate lands in another card; no licence '
               'decision has been made for any repository'),
}
EDIT = {
    'state': 'none',
    'detail': ('this build only answers coverage; it downloads nothing, reads no '
               'tensor and writes no file'),
}

# Q4_K_M / Q5_K_S / Q8_0 -> the GGUF type name a bank entry carries (q4_k, q5_k,
# q8_0). The bank speaks in gguf types; visitors speak in file names.
_QUANT_K_SUFFIX = re.compile(r'^(.*_k)_[sml]$')
_REPO = re.compile(r'^[^/\s]+/[^/\s]+$')


class _Refusal(Exception):
    """An answer the visitor gets instead of coverage; `code` names the reason."""

    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(f'{code}: {message}')


def _default_bank():
    """`<tree>/spaces/bank`, the same relative path in the clone and the Space."""
    return Path(__file__).resolve().parents[1] / 'spaces' / 'bank'


def quant_type(text):
    """The GGUF type name a visitor's quant string denotes, or None.

    `Q4_K_M` -> `q4_k`; `q8_0` -> `q8_0`; `IQ4_XS` -> `iq4_xs`. The `_S/_M/_L`
    suffix names a quant *variant* that shares the block type, and the bank
    covers the type. Anything that is not a plausible gguf type is None.
    """
    if not isinstance(text, str):
        return None
    name = text.strip().lower()
    if not name:
        return None
    if not re.fullmatch(r'[a-z0-9_]+', name):
        return None
    match = _QUANT_K_SUFFIX.match(name)
    return match.group(1) if match else name


def _revision_match(asked, pinned):
    """`exact`, `different` or `unstated` for a visitor's revision against a pin."""
    if not asked:
        return 'unstated'
    asked, pinned = str(asked).strip().lower(), str(pinned or '').strip().lower()
    if not asked:
        return 'unstated'
    if not pinned:
        return 'different'
    if asked == pinned:
        return 'exact'
    shorter, longer = sorted((asked, pinned), key=len)
    # A commit sha is routinely quoted truncated; only a prefix of 7+ characters
    # is treated as the same revision, never a coincidence of two letters.
    if len(shorter) >= 7 and longer.startswith(shorter):
        return 'exact'
    return 'different'


def _label(entry, revision_match):
    """The entry's measurement state, narrowed when the artifact is not its base."""
    measurement = entry.get('measurement') or {}
    state = str(measurement.get('state') or 'unstated')
    if revision_match == 'different':
        return 'transfer'
    return state


def _coverage(document, repo, revision, quant):
    """Pick the entry that covers this base/quant, or raise `_Refusal`."""
    by_repo = [entry for entry in document['entries']
               if str(entry['base'].get('repo', '')).casefold() == repo.casefold()]
    if not by_repo:
        covered = sorted({str(entry['base'].get('repo')) for entry in document['entries']})
        raise _Refusal(
            'BANK_MISS',
            f'the bank has no entry for {repo!r} (covered bases: '
            f'{", ".join(covered) if covered else "nothing"})')

    if quant is None:
        chosen = by_repo[0]
        return chosen, None, 'unstated'
    name = quant_type(quant)
    if name is None:
        raise _Refusal(
            'UNSUPPORTED_QUANT',
            f'{quant!r} does not name a GGUF quantization type; give the file name '
            f'tag (for example Q4_K_M) or the type itself (q4_k)')
    by_quant = [entry for entry in by_repo
                if name in {str(item).casefold() for item in entry['sources']['quant_types']}]
    if not by_quant:
        covered = sorted({str(item).casefold() for entry in by_repo
                          for item in entry['sources']['quant_types']})
        raise _Refusal(
            'UNSUPPORTED_QUANT',
            f'{repo!r} is banked, but no entry covers source type {name!r} '
            f'(covered: {", ".join(covered)})')
    # Prefer the entry fitted on the revision the visitor actually named.
    exact = [entry for entry in by_quant
             if _revision_match(revision, entry['base'].get('revision')) == 'exact']
    return (exact or by_quant)[0], name, None


def inspect(repo, revision=None, quant=None, *, bank_dir=None):
    """Coverage for one visitor request, as one JSON-shaped report.

    Returns a dict and never raises for visitor input: a malformed repository,
    an uncovered base, an uncovered quant or an unreadable bank each come back
    as a typed `refusal`, beside the licence and edit states, so the UI has one
    code path and cannot accidentally show an edit action.
    """
    repo_text = str(repo or '').strip()
    revision_text = str(revision or '').strip() or None
    report = {
        'schema': SCHEMA,
        'artifact': {'repo': repo_text, 'revision': revision_text,
                     'quant': {'input': quant, 'type': quant_type(quant)}},
        'bank': {'schema': BANK_SCHEMA, 'entries': None},
        'coverage': {'state': 'uncovered', 'entry': None, 'base': None,
                     'label': None, 'revision_match': None,
                     'quant_types': [], 'measurement': None},
        'licence': dict(LICENCE),
        'edit': dict(EDIT),
        'refusal': None,
    }
    try:
        if not _REPO.match(repo_text):
            raise _Refusal(
                'INPUT_ERROR',
                f'{repo_text!r} is not a repository id; name it as owner/name '
                f'(for example Qwen/Qwen3-4B-Instruct-2507)')
        try:
            document = load_bank(Path(bank_dir) if bank_dir is not None else _default_bank())
        except (OSError, ValueError) as error:
            raise _Refusal('BANK_INVALID', f'the direction bank cannot be read: {error}')
        report['bank']['entries'] = len(document['entries'])
        entry, quant_name, _ = _coverage(document, repo_text, revision_text, quant)
    except _Refusal as refusal:
        report['refusal'] = {'code': refusal.code, 'message': refusal.message}
        return report

    match = _revision_match(revision_text, entry['base'].get('revision'))
    report['artifact']['quant']['type'] = quant_name
    report['coverage'] = {
        'state': 'covered',
        'entry': str(entry['key']),
        'base': {'repo': str(entry['base'].get('repo')),
                 'revision': entry['base'].get('revision')},
        'label': _label(entry, match),
        'revision_match': match,
        'quant_types': [str(item) for item in entry['sources']['quant_types']],
        'measurement': entry.get('measurement'),
    }
    return report


def render(report):
    """The report as the handful of plain lines the Gradio page shows."""
    artifact = report['artifact']
    quant = artifact['quant']['type'] or artifact['quant']['input'] or 'unstated quant'
    lines = [f"Model: {artifact['repo']}  ·  {quant}", '']
    refusal = report.get('refusal')
    if refusal:
        lines += [f"Not covered — {refusal['code']}",
                  refusal['message'], '']
    else:
        coverage = report['coverage']
        lines += [f"Covered by bank entry {coverage['entry']}",
                  f"Revision match: {coverage['revision_match']}",
                  f"Label: {coverage['label']}",
                  f"Covered source types: {', '.join(coverage['quant_types'])}", '']
    lines += [f"Licence: {report['licence']['state']} — {report['licence']['detail']}",
              f"Edit: {report['edit']['state']} — {report['edit']['detail']}"]
    return '\n'.join(lines)
