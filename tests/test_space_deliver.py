# SPDX-License-Identifier: AGPL-3.0-or-later
"""The OAuth delivery path, against a local Hub stand-in over real HTTP.

The stand-in is a `ThreadingHTTPServer` that speaks the four endpoints the delivery
uses (create, git-lfs batch, signed PUT, ndjson commit) plus whoami, settings and
delete. Going through a socket rather than monkeypatching the transport is the point:
the tested properties — bytes counted as they leave, a cancel that stops mid-file,
one upload at a time, a repo deleted on failure — are all properties of the wire.
"""
import hashlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from jdewitt import space_deliver as sd


class FakeHub:
    def __init__(self):
        self.repos = {}
        self.commit = {}
        self.visibility = {}
        self.deleted = []
        self.put_bytes = {}
        self.put_complete = {}
        self.completed = {}
        self.part_bytes = {}
        self.multipart_oids = set()
        self.chunk_size = 1 << 20
        self.fail_put = None       # (oid, after_bytes) -> drop the connection
        self.requested = []
        self.lock = threading.Lock()
        hub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *args):
                pass

            def _body(self):
                length = int(self.headers.get('Content-Length') or 0)
                return self.rfile.read(length) if length else b''

            def _json(self, status, document):
                payload = json.dumps(document).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                if self.path == '/api/whoami-v2':
                    return self._json(200, {'name': 'visitor', 'type': 'user',
                                            'auth': {'type': 'oauth',
                                                     'expiresAt': '2026-12-18T00:00:00Z'}})
                if self.path.startswith('/api/models/'):
                    repo = self.path[len('/api/models/'):]
                    return self._json(200 if repo in hub.repos else 404,
                                      {'id': repo} if repo in hub.repos else {'error': 'not found'})
                return self._json(404, {'error': 'no route'})

            def do_POST(self):
                raw = self._body()
                if self.path == '/api/repos/create':
                    document = json.loads(raw)
                    repo = f"visitor/{document['name']}"
                    with hub.lock:
                        if repo in hub.repos:
                            return self._json(409, {'error': 'exists'})
                        hub.repos[repo] = {'private': document.get('private', False),
                                           'puts': [], 'commit': None}
                    return self._json(200, {'url': repo, 'name': repo})
                if self.path.endswith('/info/lfs/objects/batch'):
                    document = json.loads(raw)
                    repo = self.path[:-len('.git/info/lfs/objects/batch')].lstrip('/')
                    with hub.lock:
                        hub.requested.extend(document['objects'])
                    objects = []
                    for obj in document['objects']:
                        if obj['oid'] in hub.multipart_oids:
                            parts = -(-obj['size'] // hub.chunk_size)
                            header = {'chunk_size': str(hub.chunk_size)}
                            header.update({f'{index:05d}': (
                                f'http://127.0.0.1:{hub.port}/part/{obj["oid"]}/{index}')
                                for index in range(1, parts + 1)})
                            objects.append({
                                'oid': obj['oid'], 'size': obj['size'], 'authenticated': True,
                                'actions': {'upload': {
                                    'href': f'http://127.0.0.1:{hub.port}/complete/{obj["oid"]}',
                                    'header': header}}})
                        else:
                            objects.append({
                                'oid': obj['oid'], 'size': obj['size'], 'authenticated': True,
                                'actions': {'upload': {
                                    'href': f'http://127.0.0.1:{hub.port}/put/{obj["oid"]}',
                                    'header': {'X-Test': '1'}}}})
                    return self._json(200, {'transfer': 'basic', 'objects': objects})
                if self.path.startswith('/complete/'):
                    oid = self.path[len('/complete/'):]
                    document = json.loads(raw)
                    with hub.lock:
                        hub.completed[oid] = document
                    return self._json(200, {})
                if self.path.endswith('/commit/main'):
                    repo = self.path[len('/api/models/'):-len('/commit/main')]
                    entries = [json.loads(line) for line in raw.decode().splitlines() if line]
                    oid = hashlib.sha1(raw).hexdigest()
                    with hub.lock:
                        hub.repos.setdefault(repo, {})['commit'] = entries
                        hub.commit[repo] = {'entries': entries, 'oid': oid}
                    return self._json(200, {'success': True, 'commitOid': oid,
                                            'commitUrl': f'{repo}/commit/{oid}'})
                return self._json(404, {'error': 'no route'})

            def do_PUT(self):
                if self.path.startswith('/part/'):
                    _, _, oid, number = self.path.split('/')
                    length = int(self.headers.get('Content-Length') or 0)
                    moved = 0
                    while moved < length:
                        block = self.rfile.read(min(1 << 20, length - moved))
                        if not block:
                            break
                        moved += len(block)
                    with hub.lock:
                        hub.part_bytes[(oid, int(number))] = moved
                    self.send_response(200)
                    self.send_header('ETag', f'"part-{number}"')
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                if self.path.startswith('/put/'):
                    oid = self.path[len('/put/'):]
                    length = int(self.headers.get('Content-Length') or 0)
                    if length != int(self.headers.get('Content-Length') or 0):
                        return self._json(400, {'error': 'length'})
                    limit = length
                    if hub.fail_put and hub.fail_put[0] == oid:
                        limit = hub.fail_put[1]
                    moved, digest = 0, hashlib.sha256()
                    while moved < limit:
                        block = self.rfile.read(min(1 << 20, limit - moved))
                        if not block:
                            break
                        digest.update(block)
                        moved += len(block)
                    with hub.lock:
                        hub.put_bytes[oid] = moved
                        hub.put_complete[oid] = moved == length
                    if hub.fail_put and hub.fail_put[0] == oid:
                        self.close_connection = True
                        return
                    if digest.hexdigest() != oid:
                        return self._json(400, {'error': 'bad digest'})
                    self.send_response(200)
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                if self.path.startswith('/api/models/') and self.path.endswith('/settings'):
                    document = json.loads(self._body())
                    repo = self.path[len('/api/models/'):-len('/settings')]
                    with hub.lock:
                        hub.visibility[repo] = document.get('visibility')
                    return self._json(200, {})
                return self._json(404, {'error': 'no route'})

            def do_DELETE(self):
                document = json.loads(self._body())
                repo = f"visitor/{document['name']}"
                with hub.lock:
                    hub.repos.pop(repo, None)
                    hub.deleted.append(repo)
                return self._json(200, {})

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.port = self.server.server_address[1]
        self.api = f'http://127.0.0.1:{self.port}'
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def hub():
    with FakeHub() as fake:
        yield fake


def _files(tmp_path, sizes=(200_000, 1_000, 500)):
    names = ('model-Q8_0-banked.gguf', 'card.md', 'sidecar.json')
    paths = {}
    for name, size in zip(names, sizes):
        path = tmp_path / name
        path.write_bytes(bytes((index * 7) % 251 for index in range(size)))
        paths[name] = path
    return paths


def test_flag_is_off_by_default_and_reads_the_environment():
    assert sd.enabled({}) is False
    assert sd.enabled({'JW_OAUTH_PUSH': ''}) is False
    assert sd.enabled({'JW_OAUTH_PUSH': '0'}) is False
    for value in ('1', 'true', 'YES', 'on'):
        assert sd.enabled({'JW_OAUTH_PUSH': value}) is True


def test_repo_name_is_deterministic_and_legal():
    assert sd.repo_name('Qwen/Qwen3-4B-Instruct-2507', 'Q4_K_M') == \
        'jdewitt-qwen3-4b-instruct-2507-q4_k_m'
    assert sd.repo_name('Qwen/Qwen3-4B-Instruct-2507', 'Q8_0') == \
        'jdewitt-qwen3-4b-instruct-2507-q8_0'
    assert sd.repo_name('a/b c', None) == 'jdewitt-b-c'


def test_sidecar_cap_skips_the_big_npz_and_keeps_the_rest(tmp_path):
    small = tmp_path / 'x.revert.npz'
    small.write_bytes(b'0' * 10)
    big = tmp_path / 'y.revert.npz'
    big.write_bytes(b'0' * 64)
    files = {'card.md': tmp_path / 'card.md', 'x.revert.npz': small, 'y.revert.npz': big}
    files['card.md'].write_bytes(b'card')
    uploaded, skipped = sd._delivery_files(files, sidecar_max_bytes=32)
    assert [name for name, _path, _size in uploaded] == ['card.md', 'x.revert.npz']
    assert skipped == [{'name': 'y.revert.npz', 'bytes': 64, 'reason': 'over_sidecar_cap',
                        'cap_bytes': 32}]


def test_progress_reports_bytes_rate_and_remaining_time_from_a_clock():
    now = [100.0]
    progress = sd.Progress([('a.gguf', 1_000), ('b.md', 100)], clock=lambda: now[0])
    progress.start_file(0)
    for _ in range(4):
        now[0] += 1.0
        progress.advance(100)
    snap = progress.snapshot()
    assert snap['bytes_done'] == 400 and snap['bytes_total'] == 1100
    assert snap['rate_bytes_per_second'] == pytest.approx(100.0, rel=0.05)
    assert snap['eta_seconds'] == pytest.approx(7.0, rel=0.1)
    progress.finish_file(0, 4.0)
    assert progress.files[0]['bytes_per_second'] == 250.0


def test_chunked_body_stops_reading_when_cancelled(tmp_path):
    path = tmp_path / 'big.bin'
    path.write_bytes(b'x' * (3 * sd.CHUNK_BYTES))
    cancel = threading.Event()
    progress = sd.Progress([('big.bin', path.stat().st_size)])
    progress.start_file(0)
    handle = open(path, 'rb')
    body = sd._ChunkedBody(handle, progress, cancel=cancel)
    body.read(1024)
    cancel.set()
    with pytest.raises(sd.Cancelled):
        body.read(1024)
    body.close()
    handle.close()


def test_deliver_uploads_every_file_commits_and_publishes(hub, tmp_path):
    files = _files(tmp_path)
    events = []
    result = sd.deliver(files, token='oauth-token', base_repo='Qwen/Qwen3-4B-Instruct-2507',
                        quant='Q8_0', api=hub.api, root=tmp_path, on_progress=events.append)
    repo = result['repo_id']
    assert result['visibility'] == 'public'
    assert hub.visibility[repo] == 'public'
    assert result['commit_sha'] and result['commit_sha'] in result['commit_url']
    assert result['account']['name'] == 'visitor'
    sums_len = len(''.join(f"{result['sha256'][name]}  {name}\n" for name in files))
    assert result['bytes'] == sum(path.stat().st_size for path in files.values()) + sums_len
    entries = hub.commit[repo]['entries']
    paths = [entry['value']['path'] for entry in entries if entry['key'] == 'lfsFile']
    assert paths == ['model-Q8_0-banked.gguf', 'card.md', 'sidecar.json', 'SHA256SUMS']
    sums = next(entry['value'] for entry in entries if entry['key'] == 'lfsFile'
                and entry['value']['path'] == 'SHA256SUMS')
    assert sums['oid'] == result['sha256']['SHA256SUMS']
    # Every oid the batch was asked for is the sha256 of the bytes, and the PUT
    # carried the whole file.
    for oid, moved in hub.put_bytes.items():
        assert hub.put_complete[oid], (oid, moved)
    assert result['files'][0]['bytes_per_second'] > 0
    assert result['files'][0]['state'] == 'done'
    assert events and events[-1]['bytes_done'] == events[-1]['bytes_total']
    assert result['skipped'] == []
    assert not hub.deleted


def test_cancel_after_create_deletes_the_repo_and_the_runs_leave_nothing(hub, tmp_path):
    files = _files(tmp_path, sizes=(4 * sd.CHUNK_BYTES, 100, 100))
    cancel = threading.Event()

    def on_progress(snapshot):
        if snapshot['bytes_done'] >= 2 * sd.CHUNK_BYTES:
            cancel.set()

    with pytest.raises(sd.Cancelled):
        sd.deliver(files, token='oauth-token', base_repo='Qwen/Qwen3-4B-Instruct-2507',
                   quant='Q8_0', api=hub.api, root=tmp_path, cancel=cancel,
                   on_progress=on_progress)
    repo = 'visitor/jdewitt-qwen3-4b-instruct-2507-q8_0'
    assert repo in hub.deleted and repo not in hub.repos
    assert not hub.commit


def test_a_second_upload_is_refused_while_one_runs(hub, tmp_path):
    files = _files(tmp_path, sizes=(4 * sd.CHUNK_BYTES, 10, 10))
    release = threading.Event()
    started = threading.Event()

    def on_progress(_snapshot):
        started.set()
        release.wait(timeout=10)

    thread = threading.Thread(target=lambda: sd.deliver(
        files, token='oauth-token', base_repo='a/one', quant='Q8_0', api=hub.api,
        root=tmp_path / 'first', on_progress=on_progress))
    thread.start()
    assert started.wait(timeout=10)
    try:
        with pytest.raises(sd.DeliveryRefused) as refusal:
            sd.deliver(files, token='oauth-token', base_repo='a/two', quant='Q8_0',
                       api=hub.api, root=tmp_path / 'second')
        assert refusal.value.code == 'UPLOAD_BUSY'
    finally:
        release.set()
        thread.join(timeout=20)
    assert sd.current_upload() is None


def test_a_file_over_the_basic_limit_goes_multipart_with_per_part_bytes(hub, tmp_path):
    """The Hub refuses one PUT above 5 GB; the multipart path is the 5.5 GB carrier."""
    files = _files(tmp_path, sizes=(2 * hub.chunk_size + 1234, 10, 10))
    oid = hashlib.sha256(files['model-Q8_0-banked.gguf'].read_bytes()).hexdigest()
    hub.multipart_oids.add(oid)
    events = []
    result = sd.deliver(files, token='oauth-token', base_repo='a/multipart', quant='Q8_0',
                        api=hub.api, root=tmp_path, on_progress=events.append)
    assert result['files'][0]['transfer'] == 'multipart'
    assert result['files'][0]['done'] == result['files'][0]['total']
    parts = hub.completed[oid]['parts']
    assert [part['partNumber'] for part in parts] == [1, 2, 3]
    assert hub.part_bytes[(oid, 1)] == hub.chunk_size
    assert hub.part_bytes[(oid, 3)] == 1234
    assert sum(hub.part_bytes[(oid, n)] for n in (1, 2, 3)) == files[
        'model-Q8_0-banked.gguf'].stat().st_size
    assert hub.commit[result['repo_id']]['entries'][1]['value']['oid'] == oid


def test_a_lock_left_by_a_dead_process_is_taken_over(hub, tmp_path):
    """A hard kill leaves the file, not a two-hour outage."""
    root = tmp_path / 'root'
    root.mkdir()
    (root / '.upload.lock').write_text('999999')
    path = sd._acquire_process_slot(root)
    assert path.read_text() == str(os.getpid())
    path.unlink()


def test_a_lock_held_by_a_live_process_refuses(hub, tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    (root / '.upload.lock').write_text(str(os.getpid()))
    with pytest.raises(sd.DeliveryRefused) as refusal:
        sd._acquire_process_slot(root)
    assert refusal.value.code == 'UPLOAD_BUSY'


def test_an_existing_repo_is_never_overwritten(hub, tmp_path):
    files = _files(tmp_path, sizes=(1000, 10, 10))
    with hub.lock:
        hub.repos['visitor/jdewitt-qwen3-4b-instruct-2507-q8_0'] = {'private': False}
    with pytest.raises(sd.DeliveryRefused) as refusal:
        sd.deliver(files, token='oauth-token', base_repo='Qwen/Qwen3-4B-Instruct-2507',
                   quant='Q8_0', api=hub.api, root=tmp_path)
    assert refusal.value.code == 'REPO_EXISTS'


def test_a_failed_put_deletes_the_partial_repo(hub, tmp_path):
    files = _files(tmp_path, sizes=(4 * sd.CHUNK_BYTES, 10, 10))
    digest = hashlib.sha256(files['model-Q8_0-banked.gguf'].read_bytes()).hexdigest()
    hub.fail_put = (digest, 1 * sd.CHUNK_BYTES)
    with pytest.raises(sd.DeliveryRefused):
        sd.deliver(files, token='oauth-token', base_repo='Qwen/Qwen3-4B-Instruct-2507',
                   quant='Q8_0', api=hub.api, root=tmp_path)
    repo = 'visitor/jdewitt-qwen3-4b-instruct-2507-q8_0'
    assert repo in hub.deleted and repo not in hub.repos
    assert not hub.commit


def test_an_empty_token_is_refused_before_any_request(hub, tmp_path):
    with pytest.raises(sd.DeliveryRefused) as refusal:
        sd.deliver(_files(tmp_path), token=None, base_repo='a/b', quant='Q8_0',
                   api=hub.api, root=tmp_path)
    assert refusal.value.code == 'UPLOAD_AUTH'
    assert hub.repos == {}
