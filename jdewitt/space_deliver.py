# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deliver the edited artifact to the visitor's own Hub account (§8 v1.1).

Direct download is v1 because it asks the visitor for nothing. This module is the
alternative that only runs when the request window cannot carry the file: the
visitor signs in with `hf_oauth` (`write-repos`), the Space creates
`jdewitt-<base>-<quant>` under that account, streams the artifact, the generated
card, the sidecar, `SHA256SUMS` and (when it exists and fits the cap) the
`.revert.npz` sidecar into it, and prints the repo URL and the commit sha.

Three properties are the reason this file exists rather than a call to
`HfApi.upload_file`:

  * **bytes, not a spinner.** Every file is PUT to the URL the Hub's git-lfs batch
    hands out, in chunks we read from disk, so `bytes_done` is what left this
    process — not a filesize, not the Hub's own figure, and not a Xet dedup ratio.
    The rate is a windowed measurement and the remaining time is `remaining/rate`.
  * **one upload at a time.** A process lock (Gradio's queue is not enough: a
    restart or a second worker would slip past it) plus an in-process semaphore.
  * **no half repo.** The repo is created PRIVATE, and any failure or cancel
    deletes it and re-reads the API to prove it is gone; only a complete upload is
    made public, at which point the URL and commit are handed back.

The flag is `JW_OAUTH_PUSH=1` (or `space_deliver.enabled()`): off means the module
is inert and the app keeps its v1 direct-download path. The 5.5 GB transfer number
is what decides it (§8), and the card in `verification/space/oauth-upload.json`
carries the verdict.

`import jdewitt.space_deliver` uses the standard library only: the Space's
`requirements.txt` ships no HTTP client, and the staged-tree check refuses an
import the tree cannot supply.
"""
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

SCHEMA = 'jdewitt-space-delivery-v1'
FLAG_ENV = 'JW_OAUTH_PUSH'
API = 'https://huggingface.co'
# Hard cap for the optional `.revert.npz` sidecar: the 4B footprint sidecar is
# 761 MB, so 1 GiB carries the models this Space serves and refuses the 27B class
# where the sidecar would double the request window on its own.
SIDECAR_MAX_BYTES = 1 << 30
CHUNK_BYTES = 8 << 20
RATE_WINDOW_SECONDS = 15.0
EMIT_EVERY_SECONDS = 0.5
LOCK_STALE_SECONDS = 2 * 60 * 60
SIDECAR_SUFFIXES = ('.revert.npz',)
SUMS_NAME = 'SHA256SUMS'


class DeliveryRefused(RuntimeError):
    """The upload cannot run; `code` is the typed reason the page shows."""

    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(f'{code}: {message}')


class Cancelled(DeliveryRefused):
    def __init__(self, message='the upload was cancelled and the repo was deleted'):
        super().__init__('UPLOAD_CANCELLED', message)


# ----------------------------------------------------------------- the direction
#
# The Space asks for `write-repos`; an OAuth token is a string on the page but a
# `gr.OAuthToken` dataclass on the server, so both are accepted and neither is
# logged.

def token_text(token):
    """A token from `gr.OAuthToken`, a dict, or the string itself."""
    if token is None:
        return None
    value = getattr(token, 'token', token)
    if isinstance(value, dict):
        value = value.get('token')
    value = str(value or '').strip()
    return value or None


def enabled(environ=None):
    """The delivery path is off unless `JW_OAUTH_PUSH=1` says otherwise."""
    environ = os.environ if environ is None else environ
    return str(environ.get(FLAG_ENV, '')).strip().casefold() in ('1', 'true', 'yes', 'on')


def repo_name(base_repo, quant):
    """`Qwen/Qwen3-4B-Instruct-2507` + `Q4_K_M` -> `jdewitt-qwen3-4b-instruct-2507-q4_k_m`.

    Lower-case, Hub-legal, and deterministic: the same source class always lands in
    the same repo name, so a second run is refused (`REPO_EXISTS`) instead of
    silently clobbering the visitor's first delivery.
    """
    base = str(base_repo or '').rstrip('/').rsplit('/', 1)[-1]
    parts = ['jdewitt', base]
    if quant:
        parts.append(str(quant))
    slug = '-'.join(parts).casefold()
    slug = re.sub(r'[^a-z0-9._-]+', '-', slug)
    slug = re.sub(r'-{2,}', '-', slug).strip('-._')
    return slug[:96].rstrip('-._') or 'jdewitt-delivery'


# ------------------------------------------------------------------ the progress

class Progress:
    """Per-file bytes actually moved, a windowed rate and a remaining time.

    One instance per delivery: `snapshot()` is what the page renders and what the
    verification JSON records, so the number on the page and the number in the
    evidence cannot disagree.
    """

    def __init__(self, files, clock=time.monotonic):
        self._clock = clock
        self._started = clock()
        self.files = [{'name': name, 'total': int(size), 'done': 0,
                       'state': 'pending'} for name, size in files]
        self._index = 0
        self._samples = [(self._started, 0)]
        self._last_emit = 0.0
        self._lock = threading.Lock()

    def start_file(self, index):
        with self._lock:
            self._index = index
            self.files[index]['state'] = 'uploading'
            self._samples = [(self._clock(), self.bytes_done)]

    def advance(self, count):
        with self._lock:
            entry = self.files[self._index]
            entry['done'] = min(entry['total'], entry['done'] + int(count))
            now = self._clock()
            self._samples.append((now, self.bytes_done))
            horizon = now - RATE_WINDOW_SECONDS
            while len(self._samples) > 2 and self._samples[0][0] < horizon:
                self._samples.pop(0)

    def finish_file(self, index, seconds):
        with self._lock:
            entry = self.files[index]
            entry['done'] = entry['total']
            entry['state'] = 'done'
            entry['seconds'] = round(seconds, 3)
            entry['bytes_per_second'] = round(entry['total'] / max(seconds, 1e-6), 1)

    def fail_file(self, index, state='failed'):
        with self._lock:
            self.files[index]['state'] = state

    @property
    def bytes_done(self):
        return sum(entry['done'] for entry in self.files)

    @property
    def bytes_total(self):
        return sum(entry['total'] for entry in self.files)

    def _rate_locked(self):
        if len(self._samples) < 2:
            return 0.0
        first_at, first_bytes = self._samples[0]
        last_at, last_bytes = self._samples[-1]
        span = last_at - first_at
        if span <= 0:
            return 0.0
        return max(0.0, (last_bytes - first_bytes) / span)

    def rate_bps(self):
        with self._lock:
            return self._rate_locked()

    def snapshot(self):
        with self._lock:
            done = self.bytes_done
            total = self.bytes_total
            rate = self._rate_locked()
            current = self.files[self._index]
            return {
                'file': current['name'],
                'file_index': self._index + 1,
                'file_count': len(self.files),
                'file_done': current['done'],
                'file_total': current['total'],
                'bytes_done': done,
                'bytes_total': total,
                'rate_bytes_per_second': round(rate, 1),
                'eta_seconds': (round((total - done) / rate, 1) if rate > 0 else None),
                'elapsed_seconds': round(self._clock() - self._started, 2),
                'files': [dict(entry) for entry in self.files],
            }

    def due(self, force=False):
        """Throttle: the page does not need 8 MiB-granularity updates."""
        now = self._clock()
        if force or now - self._last_emit >= EMIT_EVERY_SECONDS:
            self._last_emit = now
            return True
        return False


# ------------------------------------------------------------------ the lock(s)

_UPLOAD = threading.Lock()
_ACTIVE = {'cancel': None, 'repo': None, 'progress': None}
_ACTIVE_LOCK = threading.Lock()


def current_upload():
    """The page's Cancel button needs a handle on the upload it is cancelling."""
    with _ACTIVE_LOCK:
        return _ACTIVE.get('progress')


def cancel_current():
    """Set the active upload's cancel flag; returns the repo name or None."""
    with _ACTIVE_LOCK:
        cancel, repo = _ACTIVE.get('cancel'), _ACTIVE.get('repo')
    if cancel is None:
        return None
    cancel.set()
    return repo


def _acquire_process_slot(root):
    """Cross-process cap of one upload, with the same stale-lock rule as a job."""
    path = Path(root) / '.upload.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    for _attempt in (0, 1):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age <= LOCK_STALE_SECONDS:
                raise DeliveryRefused(
                    'UPLOAD_BUSY',
                    'another upload is running; this Space serves one upload at a time')
            path.unlink(missing_ok=True)
            continue
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return path
    raise DeliveryRefused('UPLOAD_BUSY', 'another upload is running; try again when it finishes')


# --------------------------------------------------------------------- the HTTP

def _headers(token, extra=None):
    head = {'Authorization': 'Bearer ' + token, 'User-Agent': 'jdewitt-space-deliver'}
    head.update(extra or {})
    return head


def _read_body(response, limit=1 << 20):
    try:
        return response.read(limit).decode('utf-8', 'replace')
    except OSError:
        return ''


def _call(method, url, *, token, body=None, raw=None, headers=None, timeout=120,
          expect=(200,)):
    """One call; a status outside `expect` raises `DeliveryRefused` with the Hub's text."""
    if body is not None:
        raw = json.dumps(body).encode()
    request = urllib.request.Request(url, data=raw, method=method,
                                     headers=_headers(token, headers))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = _read_body(response)
            status = response.status
    except urllib.error.HTTPError as error:
        text = _read_body(error)
        if error.code in expect:  # a 404 from an existence probe is an answer
            return error.code, text
        raise DeliveryRefused(
            'UPLOAD_HTTP',
            f'{method} {url} -> {error.code}: {text[:300]}') from None
    except (urllib.error.URLError, OSError) as error:
        raise DeliveryRefused('UPLOAD_HTTP', f'{method} {url} failed: {error}') from None
    if status not in expect:
        raise DeliveryRefused('UPLOAD_HTTP', f'{method} {url} -> {status}: {text[:300]}')
    return status, text


def _json_call(method, url, *, token, body=None, headers=None, timeout=120,
               expect=(200, 201)):
    head = {'Content-Type': 'application/json'}
    head.update(headers or {})
    status, text = _call(method, url, token=token, body=body, timeout=timeout,
                         headers=head, expect=expect)
    try:
        return json.loads(text) if text.strip() else {}
    except ValueError:
        raise DeliveryRefused('UPLOAD_HTTP',
                              f'{method} {url} returned non-JSON: {text[:200]}') from None


def whoami(token, *, api=API):
    """The account the token belongs to; the page prints it before the URL."""
    document = _json_call('GET', f'{api}/api/whoami-v2', token=token)
    name = document.get('name')
    if not name:
        raise DeliveryRefused('UPLOAD_AUTH',
                              'the Hub did not name the account this token belongs to')
    auth = document.get('auth') or {}
    return {'name': name, 'type': document.get('type'),
            'auth': auth.get('type'), 'expires_at': auth.get('expiresAt')}


def _digest(path):
    hasher = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b''):
            hasher.update(block)
    return hasher.hexdigest()


class _ChunkedBody:
    """A file read in chunks: this is where bytes, progress and cancel live.

    `http.client` calls `read(n)` until it returns empty, so raising here aborts the
    request before the remaining bytes leave the process — that is the cancel path.
    """

    def __init__(self, path, progress, chunk=CHUNK_BYTES, cancel=None, on_emit=None):
        self._handle = open(path, 'rb')
        self._progress = progress
        self._chunk = chunk
        self._cancel = cancel
        self._emit = on_emit

    def read(self, size=-1):
        if self._cancel is not None and self._cancel.is_set():
            raise Cancelled()
        if size is None or size < 0:
            size = self._chunk
        block = self._handle.read(size)
        if block:
            self._progress.advance(len(block))
            if self._emit is not None and self._progress.due():
                self._emit(self._progress.snapshot())
        return block

    def close(self):
        self._handle.close()


def _put_file(url, path, *, headers, size, progress, cancel=None, on_emit=None, timeout=300):
    """PUT the file to the batch's signed URL, counting what is actually sent."""
    body = _ChunkedBody(path, progress, cancel=cancel, on_emit=on_emit)
    request = urllib.request.Request(
        url, method='PUT', data=body,
        headers={**headers, 'Content-Length': str(size),
                 'Content-Type': 'application/octet-stream'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read(1 << 16)
            if response.status not in (200, 201, 204):
                raise DeliveryRefused('UPLOAD_HTTP',
                                      f'PUT {url.split("?")[0]} -> {response.status}')
    except urllib.error.HTTPError as error:
        raise DeliveryRefused(
            'UPLOAD_HTTP', f'PUT -> {error.code}: {_read_body(error)[:300]}') from None
    except (urllib.error.URLError, OSError) as error:
        raise DeliveryRefused('UPLOAD_HTTP',
                              f'PUT {url.split("?")[0]} failed: {error}') from None
    finally:
        body.close()


def _lfs_upload(path, *, repo_id, token, digest, api=API, progress=None, cancel=None,
                on_emit=None):
    """git-lfs batch -> signed PUT -> the object's oid. Returns (oid, size)."""
    size = Path(path).stat().st_size
    batch = {'operation': 'upload', 'transfers': ['basic'], 'hash_algo': 'sha256',
             'objects': [{'oid': digest, 'size': size}]}
    document = _json_call(
        'POST', f'{api}/{repo_id}.git/info/lfs/objects/batch', token=token, body=batch,
        headers={'Accept': 'application/vnd.git-lfs+json',
                 'Content-Type': 'application/vnd.git-lfs+json'})
    objects = document.get('objects') or []
    if not objects:
        raise DeliveryRefused('UPLOAD_LFS',
                              f'the batch response for {Path(path).name} named no object')
    actions = objects[0].get('actions') or {}
    if 'upload' not in actions:  # already at the destination: dedup, zero bytes moved
        return digest, size
    upload = actions['upload']
    _put_file(upload['href'], path, headers=upload.get('header') or {}, size=size,
              progress=progress, cancel=cancel, on_emit=on_emit)
    return digest, size


# --------------------------------------------------------------- the file set

def _delivery_files(files, sidecar_max_bytes=SIDECAR_MAX_BYTES):
    """Split the deliverable set into (uploaded, skipped) with the sidecar cap.

    `files` is a name -> path mapping in upload order (artifact first). A
    `.revert.npz` above the cap is not uploaded and the reason travels with the
    result; the card then says the sidecar died with the Space.
    """
    uploaded, skipped = [], []
    for name, path in files.items():
        path = Path(path)
        if not path.is_file() or name == SUMS_NAME:
            continue
        size = path.stat().st_size
        lowered = str(name).casefold()
        if any(lowered.endswith(suffix) for suffix in SIDECAR_SUFFIXES) and size > sidecar_max_bytes:
            skipped.append({'name': name, 'bytes': size, 'reason': 'over_sidecar_cap',
                            'cap_bytes': int(sidecar_max_bytes)})
            continue
        uploaded.append((name, path, size))
    return uploaded, skipped


def _commit(repo_id, entries, token, api=API, timeout=300):
    """The ndjson commit: one header, then one `lfsFile` per already-uploaded object."""
    body = ''.join(json.dumps(entry) + '\n' for entry in entries).encode()
    status, text = _call(
        'POST', f'{api}/api/models/{repo_id}/commit/main', token=token, raw=body,
        headers={'Content-Type': 'application/x-ndjson'}, timeout=timeout,
        expect=(200, 201))
    try:
        return json.loads(text)
    except ValueError:
        raise DeliveryRefused('UPLOAD_HTTP',
                              f'the commit answered {status} with non-JSON: {text[:200]}') from None


def _delete_repo(repo_id, token, api=API):
    """Delete a repo this call created and prove it is gone; returns an error or None."""
    try:
        _call('DELETE', f'{api}/api/repos/delete', token=token,
              body={'name': repo_id.split('/', 1)[1], 'type': 'model'},
              expect=(200, 201, 204))
    except DeliveryRefused as error:
        return error.message
    try:
        status, _ = _call('GET', f'{api}/api/models/{repo_id}', token=token, expect=(200, 404))
    except DeliveryRefused:
        return None  # an unreadable repo is not a repo we can prove exists
    return None if status == 404 else f'{repo_id} answered {status} after delete'


# ------------------------------------------------------------------ the delivery

def deliver(files, *, token, base_repo, quant=None, cancel=None,
            on_progress=None, root=None, api=API,
            sidecar_max_bytes=SIDECAR_MAX_BYTES, public=True, timeout=300):
    """Create the visitor's repo, upload every file, commit, publish, hand back.

    Returns the JSON the page prints and `verification/space/oauth-upload.json`
    records. Raises `DeliveryRefused` (typed) or `Cancelled`; in both cases a repo
    this call created is deleted and its absence is re-read from the API before the
    exception leaves.
    """
    token = token_text(token)
    if not token:
        raise DeliveryRefused('UPLOAD_AUTH',
                              'publishing needs a signed-in Hub account; the token is empty')
    account = whoami(token, api=api)
    uploaded, skipped = _delivery_files(files, sidecar_max_bytes)
    if not uploaded:
        raise DeliveryRefused('UPLOAD_EMPTY', 'there is nothing to upload')
    repo = repo_name(base_repo, quant)
    repo_id = f'{account["name"]}/{repo}'
    root = Path(root) if root is not None else (
        Path(os.environ.get('JW_SPACE_WORK', '/tmp')) / 'jdewitt-space')
    cancel = cancel or threading.Event()
    started = time.monotonic()

    # The existence check comes before the slot: a second click on an already
    # delivered class must refuse, not wait behind the first upload.
    status, _ = _call('GET', f'{api}/api/models/{repo_id}', token=token, expect=(200, 404))
    if status == 200:
        raise DeliveryRefused(
            'REPO_EXISTS',
            f'{repo_id} already exists; this Space will not overwrite a delivery — '
            f'rename the file or delete the repo first')

    if not _UPLOAD.acquire(blocking=False):
        raise DeliveryRefused('UPLOAD_BUSY',
                              'another upload is running; this Space serves one upload at a time')
    lock_path = None
    sums_path = None
    created = False
    committed = False
    progress = None
    try:
        # Hash and build SHA256SUMS before the repo exists, so the repo's lifetime
        # is the upload's lifetime and nothing about digests can fail afterwards.
        digests = {name: _digest(path) for name, path, _size in uploaded}
        sums_text = ''.join(f'{digests[name]}  {name}\n' for name, _path, _size in uploaded)
        sums_bytes = sums_text.encode()
        sums_digest = hashlib.sha256(sums_bytes).hexdigest()
        Path(root).mkdir(parents=True, exist_ok=True)
        sums_path = Path(root) / f'.{SUMS_NAME}.{os.getpid()}.{int(started * 1000)}.tmp'
        sums_path.write_bytes(sums_bytes)
        digests[SUMS_NAME] = sums_digest
        order = uploaded + [(SUMS_NAME, sums_path, len(sums_bytes))]

        progress = Progress([(name, size) for name, _path, size in order])
        with _ACTIVE_LOCK:
            _ACTIVE.update({'cancel': cancel, 'repo': repo_id, 'progress': progress})
        lock_path = _acquire_process_slot(root)
        _json_call('POST', f'{api}/api/repos/create', token=token,
                   body={'name': repo, 'type': 'model', 'private': True},
                   timeout=60, expect=(200, 201))
        created = True

        for index, (file_name, path, size) in enumerate(order):
            if cancel.is_set():
                raise Cancelled()
            progress.start_file(index)
            if on_progress is not None:
                on_progress(progress.snapshot())
            file_started = time.monotonic()
            _lfs_upload(path, repo_id=repo_id, token=token, digest=digests[file_name],
                        api=api, progress=progress, cancel=cancel, on_emit=on_progress)
            progress.finish_file(index, time.monotonic() - file_started)
            if on_progress is not None:
                on_progress(progress.snapshot())

        entries = [{'key': 'header', 'value': {
            'summary': f'Banked GGUF edit of {base_repo} delivered by the jdewitt Space',
            'description': ('Generated by the Space when the visitor signed in, from the '
                            'artefact and card that Space produced. The Space held a '
                            'temporary copy only; this repository is the delivery.')}}]
        entries += [{'key': 'lfsFile', 'value': {
            'path': name, 'algo': 'sha256', 'oid': digests[name],
            'size': size, 'mtime': 1}} for name, _path, size in order]
        commit = _commit(repo_id, entries, token, api=api, timeout=timeout)
        committed = True
    except BaseException:
        _cleanup(repo_id, token, api, created, committed)
        raise
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE.update({'cancel': None, 'repo': None, 'progress': None})
        _UPLOAD.release()
        if lock_path is not None:
            Path(lock_path).unlink(missing_ok=True)
        if sums_path is not None:
            Path(sums_path).unlink(missing_ok=True)

    visibility_error = None
    if public:
        try:
            _call('PUT', f'{api}/api/models/{repo_id}/settings', token=token,
                  body={'visibility': 'public'}, expect=(200, 201, 204))
        except DeliveryRefused as error:
            visibility_error = error.message
    wall = time.monotonic() - started
    upload_bytes = sum(size for _name, _path, size in order)
    return {
        'schema': SCHEMA,
        'account': account,
        'repo_id': repo_id,
        'repo_url': f'{api}/{repo_id}',
        'visibility': 'public' if public and not visibility_error else 'private',
        'visibility_error': visibility_error,
        'commit_sha': commit.get('commitOid'),
        'commit_url': commit.get('commitUrl'),
        'files': [dict(entry) for entry in progress.files],
        'skipped': skipped,
        'sha256': dict(digests),
        'bytes': upload_bytes,
        'wall_seconds': round(wall, 3),
        'upload_seconds': round(sum(entry.get('seconds') or 0 for entry in progress.files), 3),
        'bytes_per_second': round(upload_bytes / max(wall, 1e-6), 1),
        'flag': FLAG_ENV,
    }


def _cleanup(repo_id, token, api, created, committed):
    """A cancelled or failed job must not leave a repo behind."""
    if not created or committed:
        return
    error = _delete_repo(repo_id, token, api)
    if error:
        raise DeliveryRefused(
            'UPLOAD_LEFTOVER',
            f'the upload failed and {repo_id} could not be deleted ({error}); '
            f'it holds a partial copy and is named here on purpose') from None
