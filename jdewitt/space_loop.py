# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Space's honest loop: coverage, licence, a banked edit, and the card.

`inspect(repo, revision, quant)` answers the coverage question from the bank alone,
before a byte is read. `start_job(...)` runs the whole loop (§7): header-only
coverage -> licence and access -> download -> `bank_apply` -> read-back -> the
generated card, its sidecar and `SHA256SUMS` -> the deliverable files.

Coverage is deliberately a bank question at preflight and an artifact question once
a file is named: the bank entry names the base it was fitted on, but a fine-tune of
that base is covered **geometrically**, so the job reads the artifact's header (and
tensor table) and resolves it against the bank. Nothing here downloads a model on
`inspect`, reads tensor data on `inspect`, edits a tensor, or generates a token.

The three honesty labels (§9) are derived from the entry's `measurement.state` and
the artifact in hand, never typed by hand:

  * `measured_artifact_class` — the artifact is exactly the class the entry was
    measured on (`measurement.artifact_class`: repo, revision, quant);
  * `banked_unmeasured` — the artifact is covered geometrically but is not that
    class (a fine-tune, another quant, another revision), so the card says in its
    first line that nothing was measured on this file;
  * `refused` — one typed code (`BANK_MISS`, `GEOMETRY_MISMATCH`,
    `UNSUPPORTED_QUANT`, `LICENCE_UNKNOWN`, `GATED_REPO`, ...) with one sentence
    naming the artifact's own architecture or file and what would cover it.

`import jdewitt.space_loop` must work without gradio and without torch. The live Hub
is imported only inside `LiveHub`'s methods, so a fixture run and the dry run need
neither a socket nor an inference stack.
"""
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .bank import SCHEMA as BANK_SCHEMA, load_bank

SCHEMA = 'jdewitt-space-inspect-v1'
LOOP_SCHEMA = 'jdewitt-space-loop-v1'
LABELS = ('measured_artifact_class', 'banked_unmeasured', 'refused')
# Order of the page's sections. `caveat` sits above `start` and `file` on purpose:
# the unmeasured sentence must be read before the download control.
PANEL_ORDER = ('model', 'quant', 'licence', 'coverage', 'caveat', 'start', 'file')
HEADER_READ_CAP = 16 * 1024 * 1024
JOB_LOCK_STALE_SECONDS = 3600

LICENCE = {
    'state': 'UNVERIFIED',
    'detail': ('this coverage preflight does not ask the Hub; the licence and access '
               'gate runs before the download when the job starts'),
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


def default_bank():
    """The bank directory this tree ships; the app and the CLI both start here."""
    return _default_bank()


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


def _same_class(cls, artifact):
    """Is this artifact the exact repo/revision/quant the entry was measured on?"""
    asked_repo = str(artifact.get('repo') or '').strip().casefold()
    held_repo = str(cls.get('repo') or '').strip().casefold()
    if not asked_repo or asked_repo != held_repo:
        return False
    if _revision_match(artifact.get('revision'), cls.get('revision')) != 'exact':
        return False
    held_quant = quant_type(cls.get('quant')) or str(cls.get('quant') or '').casefold()
    asked_quant = quant_type(artifact.get('quant')) or str(artifact.get('quant') or '').casefold()
    return bool(asked_quant) and asked_quant == held_quant


def artifact_label(entry, artifact):
    """One of §9's three states for this entry and this artifact.

    `measured_artifact_class` only when the entry's `measurement.state` says it was
    measured AND the artifact in hand is that exact class; anything else the entry
    covers is `banked_unmeasured`, because the bank may not lend a measured row to a
    file it never ran.
    """
    measurement = entry.get('measurement') or {}
    if str(measurement.get('state') or '') == 'measured_artifact_class':
        cls = measurement.get('artifact_class')
        if isinstance(cls, dict) and _same_class(cls, artifact):
            return 'measured_artifact_class'
    return 'banked_unmeasured'


def _label(entry, revision_match, artifact=None):
    """The entry's measurement state, narrowed to the artifact in hand."""
    return artifact_label(entry, artifact or {'revision': None})


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
        return chosen, None
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
    return (exact or by_quant)[0], name


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
        entry, quant_name = _coverage(document, repo_text, revision_text, quant)
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
        'label': _label(entry, match, {'repo': repo_text, 'revision': revision_text,
                                       'quant': quant}),
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


# ------------------------------------------------------------------ labels, card

def _pair(value, default='unstated'):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f'{value[0]}/{value[1]}'
    return default


def _strength_text(entry):
    strength = entry.get('strength') or {}
    return ', '.join(f'{kind} {float(value):g}' for kind, value in strength.items()) \
        or 'unstated'


def label_sentence(label, entry, artifact, delivered_cos=None, tensors=None):
    """§9's card line for one label, with the placeholders filled from the entry.

    The three sentences are the roadmap's, not this module's invention; the values
    come from the entry's ``measurement`` and the run's own read-back.
    """
    measurement = entry.get('measurement') or {}
    if label == 'measured_artifact_class':
        cls = measurement.get('artifact_class') or {}
        argv = cls.get('argv') or measurement.get('config') or 'unstated config'
        return (f"Measured on `{cls.get('repo')}@{cls.get('revision')}` "
                f"`{cls.get('quant')}` at `{argv}`: `{_pair(cls.get('harmful'))}` harmful "
                f"and `{_pair(cls.get('held_out'))}` held-out by answer shape, retention "
                f"`{_pair(cls.get('retention'))}`, drift `{cls.get('drift', 'unstated')}` "
                f"ln ppl. Shape read by an agent, wide inventory; no word-list count is used.")
    if label == 'banked_unmeasured':
        base = entry.get('base') or {}
        cos = 'not measured' if delivered_cos is None else f'{delivered_cos:.4f}'
        count = 'no' if tensors is None else str(int(tensors))
        return (f"This file was **not** measured. The direction was fitted on "
                f"`{base.get('repo')}@{base.get('revision')}` and applied as-is at strength "
                f"`{_strength_text(entry)}`. Codec fidelity only: delivered cos `{cos}` on "
                f"`{count}` tensors. No claim is made about this model's answers.")
    raise ValueError(f'{label!r} is not one of the card labels (measured/refused are built elsewhere)')


_NEEDS = {
    'BANK_MISS': 'covering it needs an axis fitted and measured on that architecture at its hidden width',
    'GEOMETRY_MISMATCH': 'covering it needs an entry fitted at that exact geometry',
    'UNSUPPORTED_QUANT': 'covering it needs an entry whose sources include that source type',
    'BUNDLE_BAD_DIGEST': 'the entry cannot be applied until its bundle matches its pinned digest',
    'LICENCE_UNKNOWN': "the allow-list is the bank's, so that licence has to join it first",
    'GATED_REPO': 'an anonymous Space cannot authenticate, so the file cannot be served here',
}


def refusal_copy(code, message, artifact=None):
    """§9's refused line: the typed code, one sentence, and what would cover it."""
    text = str(message or '').strip().rstrip('.')
    needs = _NEEDS.get(code)
    if needs and needs.rstrip('.') not in text:
        text = f'{text}; {needs}'
    return f'`{code}`: {text}.'


def _error_message(error):
    """`BankRefused.__str__` is "CODE: message"; the card wants the message alone."""
    text = str(error)
    prefix = f'{getattr(error, "code", "")}: '
    return text[len(prefix):] if text.startswith(prefix) else text


def _summary(result):
    """The five facts §9 requires on every card, whatever the label."""
    entry = result.get('entry') or {}
    artifact = result.get('artifact') or {}
    measurement = entry.get('measurement') or {}
    cls = measurement.get('artifact_class') or {}
    readback = result.get('readback') or {}
    edited = (readback.get('edited') or {}) if isinstance(readback, dict) else {}
    return {
        'base': (entry.get('base') or {}).get('repo'),
        'revision': (entry.get('base') or {}).get('revision'),
        'artifact_class': {'repo': artifact.get('repo'), 'revision': artifact.get('revision'),
                           'quant': artifact.get('quant')},
        'measured_class': cls or None,
        'strength': entry.get('strength') or {},
        'config': measurement.get('config') or cls.get('argv'),
        'delivered_cos': edited.get('min_cos'),
        'tensors': edited.get('count'),
        'licence': (result.get('licence') or {}).get('licence'),
        'output_sha256': (result.get('readback') or {}).get('output_sha256'),
    }


def card_markdown(result):
    """The generated card: §9's labels, then the five facts, then the pipeline."""
    label = result.get('label')
    entry = result.get('entry') or {}
    artifact = result.get('artifact') or {}
    readback = result.get('readback') or {}
    edited = readback.get('edited') or {}
    untouched = readback.get('untouched') or {}
    licence = (result.get('licence') or {}).get('licence')
    summary = _summary(result)
    base = entry.get('base') or {}
    if label == 'refused':
        return '\n'.join([
            '# Refused',
            '',
            (result.get('refusal') or {}).get('sentence', ''),
            '',
            f"File: `{artifact.get('file')}` · requested quant `{artifact.get('quant')}`",
            '',
            'No file was read beyond its header, nothing was downloaded, and nothing was edited.',
        ])
    first = label_sentence(label, entry, artifact,
                           delivered_cos=edited.get('min_cos'),
                           tensors=edited.get('count'))
    lines = [
        first,
        '',
        f"## Banked edit — {artifact.get('repo')}@{artifact.get('revision')}",
        '',
        f"- Label: `{label}`",
        f"- Base (the direction was fitted on): `{base.get('repo')}@{base.get('revision')}`",
        f"- Applied strength: {_strength_text(entry)} (per tensor, the banked arm's number; never a UI control)",
        f"- Artifact class: `{summary['artifact_class']['repo']}@{summary['artifact_class']['revision']}` "
        f"`{artifact.get('quant')}`",
        f"- Config: `{summary['config'] or 'unstated config'}`",
        f"- Delivered cos: `{edited.get('min_cos')}` min over `{edited.get('count')}` edited tensors "
        f"(codec fidelity of the stored edit, not an answer measurement)",
        f"- Untouched tensors: `{untouched.get('count')}` byte-identical on re-read of both files",
        f"- Licence: `{licence}` (verbatim from the Hub card)",
        "- Pipeline: header read -> byte copy -> tensor re-encode -> read-back",
        f"- Output sha256: `{summary['output_sha256']}`",
        "- No inference: this Space's requirements carry no transformers, no llama-server and no tokenizer.",
    ]
    if label == 'banked_unmeasured':
        lines += ['', 'The only measured facts on this card are the codec ones above; the bank '
                      'has no measured row for this file.']
    return '\n'.join(lines)


def sidecar_document(result):
    """The JSON beside the card: what ran, on which bytes, with which label."""
    summary = _summary(result)
    return {
        'schema': 'jdewitt-space-sidecar-v1',
        'job': result.get('job'),
        'label': result.get('label'),
        'label_source': 'entry.measurement.state narrowed to this artifact class',
        'facts': summary,
        'steps': result.get('steps'),
        'licence': result.get('licence'),
        'artifact': result.get('artifact'),
        'readback': result.get('readback'),
        'claims': {
            'removal': ('not measured for this file: the card carries a removal row only when '
                        'the entry was measured on this exact artifact class and the sidecar '
                        'names that row'),
            'delivered_cos': 'codec fidelity of the stored edit against the intended projection, '
                             'measured from the two GGUF files on CPU',
            'untouched_bytes': 'verified by re-reading both files, not asserted',
        },
    }


def panel_sections(result):
    """The page's sections in order; `caveat` must precede `start` and `file`."""
    entry = result.get('entry') or {}
    artifact = result.get('artifact') or {}
    readback = result.get('readback') or {}
    edited = readback.get('edited') or {}
    licence = result.get('licence') or {}
    if result.get('label') == 'refused':
        caveat = (result.get('refusal') or {}).get('sentence', '')
    elif result.get('label') == 'measured_artifact_class':
        caveat = (f"Measured artifact class: the row on the card is this exact file's class "
                  f"(`{artifact.get('repo')}@{artifact.get('revision')}` `{artifact.get('quant')}`).")
    else:
        caveat = label_sentence('banked_unmeasured', entry, artifact,
                                delivered_cos=edited.get('min_cos'), tensors=edited.get('count'))
    return [
        {'id': 'model', 'body': artifact.get('repo')},
        {'id': 'quant', 'body': artifact.get('file')},
        {'id': 'licence', 'body': f"{licence.get('licence')}: {licence.get('decision') or 'not asked'}"},
        {'id': 'coverage', 'body': result.get('label')},
        {'id': 'caveat', 'body': caveat},
        {'id': 'start'},
        {'id': 'file', 'body': (Path(files['artifact']).name
                                 if (files := result.get('files') or {}).get('artifact') else None)},
    ]


# ------------------------------------------------------------------------ the Hub

class LiveHub:
    """The real Hub: metadata, a header range read, then an anonymous download.

    Only ``hf_hub_download`` moves tensor bytes, and only after coverage and the
    licence gate have passed. ``hf_hub_download`` is called without a token, so a
    private repo cannot be reached by accident.
    """

    def _api(self):
        from huggingface_hub import HfApi
        return HfApi()

    def _info(self, repo, revision):
        return self._api().model_info(repo, revision=revision, files_metadata=True)

    def stat(self, repo, filename, revision=None):
        info = self._info(repo, revision)
        for sibling in getattr(info, 'siblings', None) or ():
            if sibling.rfilename == filename:
                return {'size': int(sibling.size or 0), 'sha': getattr(info, 'sha', None)}
        raise _Refusal('FILE_NOT_FOUND',
                       f'{repo}@{revision or "default"} lists no file {filename!r}')

    def files(self, repo, revision=None):
        info = self._info(repo, revision)
        return [{'name': sibling.rfilename, 'size': int(sibling.size or 0)}
                for sibling in getattr(info, 'siblings', None) or ()
                if sibling.rfilename.lower().endswith('.gguf')]

    def header(self, repo, filename, revision=None, dest=None, *, size=None):
        """Write the first ``HEADER_READ_CAP`` bytes, padded to the Hub's size.

        The padding is a sparse truncate, not a download: ``GGUFReader`` needs the
        file's declared length to map it, but the tensor data is left as zeroes and
        never transferred. Coverage reads names, types and shapes only.
        """
        from huggingface_hub import HfFileSystem
        path = f'{repo}/{filename}' if not revision else f'{repo}@{revision}/{filename}'
        with HfFileSystem().open(path, 'rb') as stream:
            data = stream.read(HEADER_READ_CAP)
        with open(dest, 'wb') as stream:
            stream.write(data)
            stream.truncate(int(size or len(data)))
        return len(data)

    def download(self, repo, filename, revision=None, dest=None):
        from huggingface_hub import hf_hub_download
        dest = Path(dest)
        staging = dest.parent / 'hub'
        got = hf_hub_download(repo_id=repo, filename=filename, revision=revision,
                              local_dir=str(staging))
        Path(got).replace(dest)
        return dest

    def licence(self, repo, revision=None, *, bank_dir=None):
        from .bank_licence import licence_gate
        return licence_gate(repo, revision, bank_dir=bank_dir)


class _FixtureApi:
    """The slice of ``HfApi`` the licence gate uses, served from a fixture."""

    def __init__(self, facts):
        self._facts = facts

    def model_info(self, repo, revision=None, files_metadata=False):
        from types import SimpleNamespace
        facts = self._facts
        return SimpleNamespace(gated=facts.get('gated', False),
                               card_data={'license': facts.get('licence')}
                               if facts.get('licence') else None,
                               tags=[], sha=facts.get('sha') or revision)


class FixtureHub:
    """A Hub served from a directory: no socket, no download, no token.

    ``hub.json`` names each repo's revision, licence, files and, per file, the
    header prefix that still parses. ``header`` writes that prefix and truncates
    the copy to the real size, so the coverage step provably reads no tensor data.
    """

    def __init__(self, fixture_dir, licence_bank=None):
        self.dir = Path(fixture_dir)
        self.document = json.loads((self.dir / 'hub.json').read_text())
        self.bank_dir = self.dir / 'bank'
        self.licence_bank = Path(licence_bank) if licence_bank is not None else self.bank_dir

    def _repo(self, repo, revision=None):
        try:
            return self.document['repos'][repo]
        except KeyError:
            raise _Refusal('FILE_NOT_FOUND', f'the fixture hub has no repo {repo!r}') from None

    def _file(self, repo, filename):
        record = self._repo(repo)
        for item in record['files']:
            if item['name'] == filename:
                return record, item
        raise _Refusal('FILE_NOT_FOUND',
                       f'{repo} lists no file {filename!r} (the fixture hub has '
                       f'{", ".join(item["name"] for item in record["files"])})')

    def stat(self, repo, filename, revision=None):
        record, item = self._file(repo, filename)
        return {'size': int(item['size']), 'sha': record.get('sha')}

    def files(self, repo, revision=None):
        return [{'name': item['name'], 'size': int(item['size'])}
                for item in self._repo(repo)['files']]

    def header(self, repo, filename, revision=None, dest=None, *, size=None):
        _record, item = self._file(repo, filename)
        source = self.dir / 'repos' / repo.replace('/', '__') / filename
        with source.open('rb') as stream:
            data = stream.read(int(item['header_bytes']))
        with open(dest, 'wb') as stream:
            stream.write(data)
            stream.truncate(int(size or item['size']))
        return len(data)

    def download(self, repo, filename, revision=None, dest=None):
        _record, item = self._file(repo, filename)
        source = self.dir / 'repos' / repo.replace('/', '__') / filename
        shutil.copyfile(source, dest)
        return Path(dest)

    def licence(self, repo, revision=None, *, bank_dir=None):
        from .bank_licence import licence_gate
        record = self._repo(repo)
        return licence_gate(repo, revision, bank_dir=self.licence_bank,
                            api=_FixtureApi(record))


# ---------------------------------------------------------------------- the loop

def _default_root():
    env = os.environ.get('JW_SPACE_WORK')
    return Path(env) if env else Path(tempfile.gettempdir()) / 'jdewitt-space'


@contextmanager
def _job_lock(root):
    """One job at a time, across processes as well as the Gradio queue."""
    path = Path(root) / '.job.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    for _attempt in (0, 1):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age <= JOB_LOCK_STALE_SECONDS:
                raise _Refusal(
                    'JOB_RUNNING',
                    'another job is running; this Space serves one job at a time and '
                    'deletes its workspace after each response') from None
            path.unlink(missing_ok=True)
    if fd is None:
        raise _Refusal('JOB_RUNNING', 'another job is running; try again when it finishes')
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield path
    finally:
        path.unlink(missing_ok=True)


def _sweep(root):
    """Delete deliverables and workspaces an earlier job left behind."""
    for entry in Path(root).iterdir():
        if entry.name.startswith('deliver-') or entry.name.startswith('job-'):
            shutil.rmtree(entry, ignore_errors=True)


def _mark(steps, name, started, detail=None, state='ok', code=None):
    steps.append({'step': name, 'state': state,
                  'seconds': round(time.monotonic() - started, 3),
                  'detail': detail, 'code': code})


def quant_tag(filename):
    """`...-Q4_K_M.gguf` -> `Q4_K_M`: the file's own quant tag, as the card spells it."""
    stem = Path(str(filename)).stem
    parts = stem.split('-')
    for index in range(len(parts)):
        candidate = '-'.join(parts[index:])
        lowered = candidate.casefold()
        if not (re.match(r'^(iq|q)\d', lowered) or lowered in ('bf16', 'f16', 'f32')):
            continue
        if quant_type(candidate) is not None:
            return candidate
    return None


def quant_from_filename(filename):
    """`...-Q4_K_M.gguf` -> `q4_k`; the longest dash-separated suffix naming a type."""
    tag = quant_tag(filename)
    return None if tag is None else quant_type(tag)


def covered_bases(bank_dir=None):
    """The short list of bases the bank carries, for the UI's repo field."""
    document = load_bank(Path(bank_dir) if bank_dir is not None else _default_bank())
    seen, bases = set(), []
    for entry in document['entries']:
        base = entry.get('base') or {}
        repo = str(base.get('repo') or '')
        if not repo or repo.casefold() in seen:
            continue
        seen.add(repo.casefold())
        measurement = entry.get('measurement') or {}
        bases.append({'repo': repo, 'revision': base.get('revision'),
                      'entry': entry.get('key'),
                      'state': measurement.get('state'),
                      'quant_types': [str(item) for item in entry['sources']['quant_types']]})
    return bases


def quant_choices(files, entry=None):
    """The quant control's options, with the reason attached to disabled ones.

    Gradio's Dropdown has no per-item disabled state, so a choice outside the
    entry's source types carries its reason in the label and the loop refuses it by
    code if it is picked anyway. It is never removed from the list: hiding it would
    make the picker lie about what the bank can read.
    """
    covered = None
    if entry is not None:
        covered = [str(item) for item in (entry.get('sources') or {}).get('quant_types', [])]
    choices = []
    for item in files:
        name = item['name'] if isinstance(item, dict) else str(item)
        size = item.get('size') if isinstance(item, dict) else None
        kind = quant_from_filename(name)
        enabled = True
        reason = None
        if covered is not None and kind is not None and kind not in {q.casefold() for q in covered}:
            enabled = False
            reason = (f'source type `{kind}` is not in this entry\'s covered source types '
                      f'({", ".join(covered)})')
        label = name if enabled else f'{name} — unavailable: {reason}'
        choices.append({'name': name, 'size': size, 'quant_type': kind,
                        'enabled': enabled, 'reason': reason, 'label': label})
    return choices


def _entry_view(resolved):
    return {
        'key': resolved['entry']['key'],
        'base': resolved['entry']['base'],
        'geometry': resolved['geometry'],
        'footprint': resolved['footprint'],
        'strength': resolved['strength'],
        'edit': resolved['edit'],
        'sources': resolved['sources'],
        'measurement': resolved['measurement'],
    }


def start_job(repo, revision=None, filename=None, quant=None, *, bank_dir=None,
              hub=None, root=None):
    """Run one visitor request end to end, or refuse it with one typed code.

    The order is the roadmap's: coverage from the header only, then the licence
    gate, then the download, then the banked edit, then the read-back and the
    package. A refusal before the download leaves the Hub with no tensor bytes
    moved. The job workspace is deleted when the deliverable exists; the four
    delivered files stay until the next job sweeps them.
    """
    started = time.monotonic()
    repo = str(repo or '').strip()
    revision = str(revision or '').strip() or None
    filename = str(filename or '').strip()
    quant = str(quant or '').strip() or None
    job_id = ('job-' + time.strftime('%Y%m%dT%H%M%S', time.gmtime())
              + f'-{os.getpid()}-{uuid.uuid4().hex[:6]}')
    result = {
        'schema': LOOP_SCHEMA,
        'job': {'id': job_id, 'repo': repo, 'revision': revision,
                'file': filename, 'quant': quant},
        'steps': [], 'label': None, 'refusal': None,
        'entry': None,
        'artifact': {'repo': repo, 'revision': revision, 'file': filename,
                     'quant': quant, 'quant_type': quant_type(quant)},
        'licence': None, 'apply': None, 'readback': None,
        'card': None, 'files': None,
        'workspace_deleted': False, 'wall_seconds': None,
    }

    def refuse(code, message):
        result['label'] = 'refused'
        result['refusal'] = {'code': code, 'message': message,
                             'sentence': refusal_copy(code, message, result['artifact'])}
        result['card'] = card_markdown(result)
        result['wall_seconds'] = round(time.monotonic() - started, 2)
        return result

    if not _REPO.match(repo):
        return refuse('INPUT_ERROR', f'{repo!r} is not a repository id; name it as owner/name')
    if Path(filename).suffix.casefold() != '.gguf' or not filename:
        return refuse('INPUT_ERROR', f'{filename!r} is not a .gguf file name')
    bank_path = Path(bank_dir) if bank_dir is not None else _default_bank()
    hub = hub or LiveHub()
    workspace_root = Path(root) if root is not None else _default_root()
    try:
        workspace_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        return refuse('WORKSPACE_FAILED', f'the job workspace cannot be made: {error}')
    try:
        with _job_lock(workspace_root):
            _sweep(workspace_root)
            workspace = Path(tempfile.mkdtemp(prefix=job_id + '-', dir=workspace_root))
            try:
                return _run_job(result, refuse, bank_path, hub, workspace,
                                workspace_root, job_id, started)
            finally:
                shutil.rmtree(workspace, ignore_errors=True)
                result['workspace_deleted'] = not workspace.exists()
                result['wall_seconds'] = round(time.monotonic() - started, 2)
    except _Refusal as refusal:
        return refuse(refusal.code, refusal.message)


def _run_job(result, refuse, bank_path, hub, workspace, workspace_root, job_id, started):
    from . import bank as bank_module
    from .bank import BankRefused
    from .errors import BankApplyRefused
    from .ggufgeom import gguf_weight_shapes
    from .storage import file_digest

    steps = result['steps']
    job = result['job']
    repo, revision, filename = job['repo'], job['revision'], job['file']

    # 1. coverage: the header and tensor table only, before any tensor bytes.
    step_at = time.monotonic()
    stat = hub.stat(repo, filename, revision)
    header_path = workspace / 'header.gguf'
    written = hub.header(repo, filename, revision, header_path, size=stat['size'])
    read_detail = f'read {written} of {stat["size"]} bytes (header and tensor table only)'
    try:
        resolved = bank_module.resolve(bank_path, header_path)
    except BankRefused as error:
        _mark(steps, 'coverage', step_at, detail=f'{read_detail}; {_error_message(error)}',
              state='refused', code=error.code)
        return refuse(error.code, _error_message(error))
    except Exception as error:  # a header slice that does not parse is a refusal, not a crash
        _mark(steps, 'coverage', step_at, detail=f'{read_detail}; {error}',
              state='refused', code='HEADER_UNREADABLE')
        return refuse('HEADER_UNREADABLE',
                      f'the first {written} bytes of {filename} do not contain a parseable '
                      f'GGUF header ({type(error).__name__}: {error})')
    _mark(steps, 'coverage', step_at, detail=read_detail)

    entry = _entry_view(resolved)
    result['entry'] = entry
    result['artifact'].update({
        'architecture': resolved['artifact']['architecture'],
        'layers': resolved['artifact']['layers'],
        'hidden': resolved['artifact']['hidden'],
        'tensors': resolved['artifact']['tensors'],
        'source_types': resolved['artifact']['source_types'],
        'size': int(stat['size']),
    })
    if not resolved['sources']['covered']:
        message = (f'{filename} carries source type(s) '
                   f'{", ".join(resolved["artifact"]["source_types"])}, and entry '
                   f'{resolved["entry"]["key"]} covers '
                   f'{", ".join(entry["sources"]["quant_types"])}')
        _mark(steps, 'coverage', step_at, detail=message, state='refused',
              code='UNSUPPORTED_QUANT')
        return refuse('UNSUPPORTED_QUANT', message)

    # 2. licence and access, from Hub metadata only, still before the download.
    step_at = time.monotonic()
    try:
        licence = hub.licence(repo, revision, bank_dir=bank_path)
    except Exception as error:
        _mark(steps, 'licence', step_at, detail=str(error), state='refused',
              code='LICENCE_UNREACHABLE')
        return refuse('LICENCE_UNREACHABLE',
                      f'the licence gate could not read {repo} metadata ({type(error).__name__}: {error})')
    result['licence'] = licence
    _mark(steps, 'licence', step_at,
          detail=f'{licence.get("reason")}: {licence.get("detail")}')
    if licence.get('decision') != 'allowed':
        return refuse(licence.get('reason') or 'LICENCE_UNKNOWN',
                      licence.get('detail') or f'{repo} is not allowed by the bank licence list')

    # 3. download, with the size the Hub declared.
    step_at = time.monotonic()
    input_path = workspace / 'input.gguf'
    try:
        hub.download(repo, filename, revision, input_path)
    except (_Refusal, OSError) as error:
        message = error.message if isinstance(error, _Refusal) else f'{filename}: {error}'
        _mark(steps, 'download', step_at, detail=message, state='refused',
              code='DOWNLOAD_FAILED')
        return refuse('DOWNLOAD_FAILED', message)
    size = input_path.stat().st_size
    if stat.get('size') and size != int(stat['size']):
        _mark(steps, 'download', step_at, detail=f'{size} bytes, Hub said {stat["size"]}',
              state='refused', code='DOWNLOAD_SIZE')
        return refuse('DOWNLOAD_SIZE',
                      f'{filename} downloaded {size} bytes but the Hub metadata says '
                      f'{stat["size"]}; the file is not the one coverage checked')
    input_sha = file_digest(input_path)
    _mark(steps, 'download', step_at, detail=f'{size} bytes, sha256 {input_sha[:16]}')

    # 4. the banked edit: bank-apply re-resolves the file, so its own typed refusals
    #    are the ones the visitor sees.
    step_at = time.monotonic()
    edited = workspace / 'edited.gguf'
    from .bank_apply import bank_apply
    try:
        report = bank_apply(bank_path, resolved['entry']['key'], input_path, edited)
    except (BankRefused, BankApplyRefused) as error:
        _mark(steps, 'edit', step_at, detail=_error_message(error), state='refused',
              code=error.code)
        return refuse(error.code, _error_message(error))
    except (OSError, ValueError) as error:
        message = f'{resolved["entry"]["key"]} could not be applied: {error}'
        _mark(steps, 'edit', step_at, detail=message, state='refused', code='EDIT_FAILED')
        return refuse('EDIT_FAILED', message)
    result['apply'] = report
    _mark(steps, 'edit', step_at,
          detail=f"{resolved['entry']['key']} at {_strength_text(entry)} -> "
                 f"{report['edit_type']['chosen']}: {report['edited']['count']} tensors, "
                 f"min cos {report['edited']['min_cos']}")

    # 5. read-back: reopen BOTH files and compare the tensor table, not the claim.
    step_at = time.monotonic()
    shapes_in = gguf_weight_shapes(input_path)
    shapes_out = gguf_weight_shapes(edited)
    if sorted(shapes_in) != sorted(shapes_out) or shapes_in != shapes_out:
        message = (f're-opening the edited file found a different tensor table '
                   f'({len(shapes_out)} tensors against {len(shapes_in)}, or a shape changed)')
        _mark(steps, 'readback', step_at, detail=message, state='refused',
              code='READBACK_FAILED')
        return refuse('READBACK_FAILED', message)
    output_sha = file_digest(edited)
    edited_facts = report.get('edited') or {}
    result['readback'] = {
        'input_sha256': input_sha,
        'output_sha256': output_sha,
        'bytes_in': size,
        'bytes_out': edited.stat().st_size,
        'reopened': {'tensors': len(shapes_out), 'shapes_match': True,
                     'source_types': report['artifact'].get('source_types')},
        'edited': {'count': edited_facts.get('count'),
                   'min_cos': edited_facts.get('min_cos'),
                   'mean_cos': edited_facts.get('mean_cos'),
                   'floor': edited_facts.get('floor'),
                   'floor_met': edited_facts.get('floor_met'),
                   'by_kind': edited_facts.get('by_kind'),
                   'tensors': edited_facts.get('tensors')},
        'untouched': report.get('untouched'),
        'ok': bool(report.get('ok')),
    }
    _mark(steps, 'readback', step_at,
          detail=f'{edited_facts.get("count")} edited, min cos '
                 f'{edited_facts.get("min_cos")}, untouched '
                 f'{report.get("untouched", {}).get("count")} verified, sha256 '
                 f'{output_sha[:16]}')

    # 6. package: the label comes from measurement.state, then the files exist.
    step_at = time.monotonic()
    result['label'] = artifact_label(entry, result['artifact'])
    result['card'] = card_markdown(result)
    deliver = workspace_root / f'deliver-{job_id}'
    deliver.mkdir(parents=True, exist_ok=True)
    artifact_name = f'{Path(filename).stem}-banked.gguf'
    shutil.copyfile(edited, deliver / artifact_name)
    (deliver / 'card.md').write_text(result['card'] + '\n')
    (deliver / 'sidecar.json').write_text(
        json.dumps(sidecar_document(result), indent=2, sort_keys=True) + '\n')
    names = (artifact_name, 'card.md', 'sidecar.json')
    (deliver / 'SHA256SUMS').write_text(''.join(
        f'{file_digest(deliver / name)}  {name}\n' for name in names))
    result['files'] = {
        'artifact': str(deliver / artifact_name),
        'card': str(deliver / 'card.md'),
        'sidecar': str(deliver / 'sidecar.json'),
        'sha256sums': str(deliver / 'SHA256SUMS'),
        'bytes': (deliver / artifact_name).stat().st_size,
    }
    _mark(steps, 'package', step_at,
          detail=f'{len(names)} files, {result["files"]["bytes"]} artifact bytes')
    return result


# -------------------------------------------------------------------- the dry run

def preview(repo, revision=None, *, bank_dir=None, hub=None):
    """The repo-level view the page shows before Start: files, licence, verdict.

    Metadata and the file list only: no header read, no download. A failure comes
    back as a refusal-shaped dict so the UI has one code path.
    """
    repo = str(repo or '').strip()
    bank_path = Path(bank_dir) if bank_dir is not None else _default_bank()
    view = {'repo': repo, 'revision': revision, 'ok': False, 'files': [],
            'quant_control': [], 'licence': None, 'entry': None,
            'refusal': None, 'verdict': None, 'licence_line': '', 'coverage_line': ''}
    if not _REPO.match(repo):
        view['refusal'] = {'code': 'INPUT_ERROR',
                           'message': f'{repo!r} is not a repository id; name it as owner/name'}
        return view
    hub = hub or LiveHub()
    try:
        document = load_bank(bank_path)
        files = hub.files(repo, revision)
    except _Refusal as refusal:
        view['refusal'] = {'code': refusal.code, 'message': refusal.message}
        return view
    except Exception as error:
        view['refusal'] = {'code': 'BANK_INVALID' if 'bank' in str(error).lower() else 'HUB_ERROR',
                           'message': str(error)}
        return view
    entry = None
    for candidate in document['entries']:
        if str((candidate.get('base') or {}).get('repo') or '').casefold() == repo.casefold():
            entry = candidate
            break
    view['entry'] = entry['key'] if entry else None
    view['files'] = files
    view['quant_control'] = quant_choices(files, entry)
    try:
        licence = hub.licence(repo, revision, bank_dir=bank_path)
        view['licence'] = licence
        stated = licence.get('licence') or 'none in the card'
        view['licence_line'] = f"Licence: `{stated}` — {licence.get('reason')}"
    except Exception as error:
        view['licence_line'] = f'Licence: not read ({type(error).__name__}: {error})'
    if entry is None:
        view['coverage_line'] = ('Coverage: this repo id is not a bank base. A fine-tune of a '
                                 'bank base is covered by its geometry; the header decides when '
                                 'you start.')
    else:
        measurement = entry.get('measurement') or {}
        view['coverage_line'] = (f"Coverage: bank entry `{entry['key']}` covers these source "
                                 f"types ({', '.join(entry['sources']['quant_types'])}); its "
                                 f"measurement state is `{measurement.get('state')}`.")
    view['ok'] = True
    view['verdict'] = (view['coverage_line'] + ' ' + view['licence_line'])
    return view


def render_preview(view):
    """The preview as the two plain lines the page shows."""
    lines = [view.get('coverage_line') or '', view.get('licence_line') or '']
    refusal = view.get('refusal')
    if refusal:
        lines.insert(0, refusal_copy(refusal['code'], refusal['message']))
    for choice in view.get('quant_control') or []:
        if not choice['enabled']:
            lines.append(f"Not servable: {choice['label']}")
    return '\n'.join(line for line in lines if line)


def dry_run(bank_dir, model, quant, fixture_dir):
    """Run the loop three times on a fixture hub: measured, unmeasured, refused.

    ``bank_dir`` is the bank the Space ships; the fixture supplies the entries when
    it carries a ``bank/`` directory (the shipped bank ships empty on purpose), and
    the licence allow-list is read from ``bank_dir`` when that directory has one.
    No socket, no GPU, no Hub download: the fixture hub serves every header, every
    file and every licence decision from disk.
    """
    fixture_dir = Path(fixture_dir)
    fixture = FixtureHub(fixture_dir)
    shipped = Path(bank_dir)
    entry_bank = fixture.bank_dir if (fixture.bank_dir / 'bank.json').is_file() else shipped
    licence_bank = shipped if (shipped / 'licences.json').is_file() else entry_bank
    hub = FixtureHub(fixture_dir, licence_bank=licence_bank)
    document = hub.document
    plan = document.get('dry_run') or {}
    unmeasured_repo = plan.get('unmeasured_repo')
    uncovered_repo = plan.get('uncovered_repo')

    def pick(repo):
        return next(item['name'] for item in hub.files(repo)
                    if quant_from_filename(item['name']) == quant_type(quant))

    requests = [
        ('measured', model, hub.document['repos'][model]['revision']),
        ('banked_unmeasured', unmeasured_repo,
         hub.document['repos'][unmeasured_repo]['revision']),
        ('refused', uncovered_repo, hub.document['repos'][uncovered_repo]['revision']),
    ]
    root = Path(tempfile.mkdtemp(prefix='jdewitt-space-dry-'))
    cases = []
    try:
        for expected, repo, revision in requests:
            filename = pick(repo)
            # A root per case: the loop sweeps old deliverables at the start of the
            # next job, and the dry run prints all three jobs' files afterwards.
            case_root = root / expected
            result = start_job(repo, revision, filename, quant,
                               bank_dir=entry_bank, hub=hub, root=case_root)
            files = result.get('files') or {}
            cases.append({'expected': expected, 'repo': repo, 'revision': revision,
                          'file': filename, 'result': result,
                          'files_present': {key: Path(path).is_file()
                                            for key, path in files.items()
                                            if key != 'bytes' and isinstance(path, str)}})
    finally:
        shutil.rmtree(root, ignore_errors=True)
    base = next((item for item in covered_bases(entry_bank)
                 if item['repo'].casefold() == str(model).casefold()), None)
    entry = None
    if base is not None:
        document_bank = load_bank(entry_bank)
        entry = next(item for item in document_bank['entries']
                     if str(item['key']) == str(base['entry']))
    quant_control = quant_choices(hub.files(model), entry)
    unmeasured = next(case['result'] for case in cases if case['expected'] == 'banked_unmeasured')
    # The caveat is the unmeasured sentence, so the panel proof is taken from that
    # case: it is the one whose text must sit above the download control.
    panel = panel_sections(unmeasured)
    labels = [case['result']['label'] for case in cases]
    return {
        'schema': 'jdewitt-space-dry-run-v1',
        'fixture': str(fixture_dir),
        'bank': str(entry_bank),
        'licence_bank': str(licence_bank),
        'network': 'none: the fixture hub serves headers, files and licence facts from disk',
        'requests': [{'expected': case['expected'], 'repo': case['repo'],
                      'file': case['file']} for case in cases],
        'cases': cases,
        'quant_control': quant_control,
        'panel': panel,
        'panel_order': [section['id'] for section in panel],
        'caveat_above_file': ([section['id'] for section in panel].index('caveat')
                              < [section['id'] for section in panel].index('file')),
        'labels': labels,
        'ok': (labels == ['measured_artifact_class', 'banked_unmeasured', 'refused']
               and all(case['result']['workspace_deleted'] for case in cases)
               and all(all(case['files_present'].values())
                       for case in cases if case['result']['label'] != 'refused')),
    }


def render_dry_run(report):
    """The dry run as plain text: the three labels, the refusal sentence, the order."""
    lines = [
        'Space dry run (fixture hub, no GPU, no network, no Hub download)',
        '=' * 72,
        f"fixture:       {report['fixture']}",
        f"bank entries:  {report['bank']}",
        f"licence list:  {report['licence_bank']}",
        f"network:       {report['network']}",
        '',
    ]
    for index, case in enumerate(report['cases'], 1):
        result = case['result']
        lines += [f"[{index}] expected {case['expected']} — {case['repo']} {case['file']}",
                  f"    label: {result['label']}  (workspace deleted: {result['workspace_deleted']})"]
        for step in result['steps']:
            state = 'refused' if step['state'] == 'refused' else 'ok'
            lines.append(f"    {step['step']:<9} {state:<7} {step['seconds']:>6.3f}s  "
                         f"{step['detail']}")
        if result['label'] == 'refused':
            lines += ['', '    ' + result['refusal']['sentence']]
        else:
            lines += ['', result['card'], '',
                      f"    files written: {case['files_present']}",
                      f"    artifact: {Path(result['files']['artifact']).name}",
                      f"    card:     {Path(result['files']['card']).name}",
                      f"    sidecar:  {Path(result['files']['sidecar']).name}",
                      f"    sums:     {Path(result['files']['sha256sums']).name}"]
        lines.append('')
    lines += ['Quant control for the covered base (source types outside the entry stay '
              'listed, disabled with the reason):']
    for choice in report['quant_control']:
        state = 'enabled ' if choice['enabled'] else 'DISABLED'
        lines.append(f"    {state}  {choice['label']}")
    if not report['quant_control']:
        lines.append('    (the fixture hub lists no files for this repo)')
    lines += ['', 'Panel as the page renders it (caveat above the download control):']
    for section in report['panel']:
        body = section.get('body')
        lines.append(f"    {section['id']:<9} {body}" if body else f"    {section['id']}")
    lines += ['',
              'Panel order: ' + ' -> '.join(report['panel_order']),
              f"Caveat above the download control: {report['caveat_above_file']}",
              '',
              'Labels rendered: ' + ', '.join(report['labels']),
              f"Dry run passed: {report['ok']}"]
    return '\n'.join(lines)
