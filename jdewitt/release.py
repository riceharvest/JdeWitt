# SPDX-License-Identifier: AGPL-3.0-or-later
import hashlib
import errno
import fcntl
import json
import uuid

import os
import re
import shutil
import socket
import subprocess
import time
import urllib.error
from pathlib import Path
import gguf
from .transport import ChatClient
from .scoring import search_rate
from .answers import atomic_json, atomic_text, exclusive
from .clock import now_iso as _now_iso
from .data import digest
from .errors import ReleaseRefused
from .storage import atomic_destination, sync_directory

QUANTS = {'Q4_K_M': gguf.LlamaFileType.MOSTLY_Q4_K_M, 'Q5_K_M': gguf.LlamaFileType.MOSTLY_Q5_K_M}


def load_mode_flags(server):
    """Pick the no-mmap spelling this llama-server actually accepts.

    Upstream replaced `--no-mmap` with `-lm/--load-mode none`. Probe the binary
    instead of guessing, so a pinned older build keeps working.
    """
    help_text = subprocess.run([str(server), '--help'], text=True, capture_output=True, check=True).stdout
    if re.search(r'^\s*-lm,\s+--load-mode\b', help_text, re.MULTILINE):
        return ['--load-mode', 'none']
    if re.search(r'--no-mmap\b', help_text):
        return ['--no-mmap']
    raise RuntimeError('llama-server exposes neither --load-mode nor --no-mmap; refusing to guess')


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_quant(path, expected):
    if expected not in QUANTS:
        raise ValueError('Only explicitly supported release quants are allowed')
    try:
        reader = gguf.GGUFReader(str(path))
        actual = reader.get_field('general.file_type').contents()
        architecture = reader.get_field('general.architecture').contents()
        if int(actual) != int(QUANTS[expected]) or not reader.tensors:
            raise ValueError('GGUF metadata does not match the requested quant')
        del reader
    except Exception as e:
        raise ValueError(f'Invalid {expected} GGUF: {path}') from e
    return {'path':str(Path(path).resolve()),'quant':expected,'architecture':architecture,'sha256':sha256_file(path),'bytes':Path(path).stat().st_size}


def run_logged(cmd, log, timeout=3600):
    with Path(log).open('w') as f:
        result = subprocess.run([str(x) for x in cmd], stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'Command exited {result.returncode}; see {log}')


def verify_bf16(path):
    try:
        reader=gguf.GGUFReader(str(path))
        kind=int(reader.get_field('general.file_type').contents())
        if kind!=int(gguf.LlamaFileType.MOSTLY_BF16) or not reader.tensors:
            raise ValueError('Expected an actual BF16 GGUF')
        architecture=reader.get_field('general.architecture').contents()
        del reader
    except Exception as e:
        raise ValueError('Invalid BF16 conversion artifact') from e
    return {'path':str(Path(path).resolve()),'quant':'BF16','architecture':architecture,
            'sha256':sha256_file(path),'bytes':Path(path).stat().st_size}


def conversion_identity(hf, converter, quantizer, python, imatrix=None):
    return {'source_files':{p.name:sha256_file(p) for p in sorted(Path(hf).iterdir()) if p.is_file()},
            'tools':{str(Path(p).resolve()):sha256_file(p) for p in (converter,quantizer,python)},
            'imatrix':sha256_file(imatrix) if imatrix is not None else None}


def cached_conversion(hf, output, converter, quantizer, python, imatrix=None, convert=None, verify=None):
    convert,verify = convert or export_gguf,verify or verify_quant
    directory=Path(output)
    identity=digest(conversion_identity(hf,converter,quantizer,python,imatrix))
    manifest=directory/'artifacts.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        if set(saved)!={'input_digest','artifacts'} or saved['input_digest']!=identity:
            raise ValueError('Completed conversion identity changed; preserve old artifacts and use a new directory')
        checked={q:verify(saved['artifacts'][q]['path'],q) for q in QUANTS}
        if saved['artifacts']!=checked:
            raise ValueError('Cached quant artifact changed')
        return checked
    if directory.exists() and not (directory/'conversion-state.json').exists():
        directory.rename(directory.with_name(directory.name+'-interrupted-'+uuid.uuid4().hex))
    results=convert(hf,directory,converter,quantizer,python,imatrix)
    atomic_json(manifest,{'input_digest':identity,'artifacts':results})
    return results


def export_gguf(hf, output, converter, quantizer, python, imatrix=None):
    hf, output = Path(hf), Path(output)
    for path in (hf,converter,quantizer,python):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    identity=conversion_identity(hf,converter,quantizer,python,imatrix)
    state_path=output/'conversion-state.json'
    if output.exists() and not state_path.exists():
        raise FileExistsError('Unidentified incomplete conversion directory; preserve it and use a new directory')
    output.mkdir(parents=True,exist_ok=True)
    if state_path.exists():
        state=json.loads(state_path.read_text())
        if state.get('record_digest')!=digest({k:v for k,v in state.items() if k!='record_digest'}) or state.get('identity')!=digest(identity):
            raise ValueError('Conversion state identity/integrity mismatch')
    else:
        state={'identity':digest(identity),'artifacts':{},'raw':None,'complete':False}
    def checkpoint():
        state['record_digest']=digest({k:v for k,v in state.items() if k!='record_digest'})
        atomic_json(state_path,state)
    checkpoint()
    for quant,artifact in state['artifacts'].items():
        checked=verify_quant(artifact['path'],quant)
        if checked!=artifact:
            raise ValueError('Previously verified quant artifact changed')
    if state['complete']:
        if set(state['artifacts'])!=set(QUANTS):
            raise ValueError('Completed conversion lacks both quants')
        return state['artifacts']
    weight_bytes = sum(p.stat().st_size for p in hf.glob('*.safetensors'))
    if shutil.disk_usage(output).free < weight_bytes * 2:
        raise RuntimeError('Insufficient disk for conversion and both quants')
    raw = output / 'model-BF16.gguf'
    if state['raw'] is not None:
        if verify_bf16(raw)!=state['raw']:
            raise ValueError('Previously verified BF16 artifact changed')
    else:
        with atomic_destination(raw) as temporary:
            run_logged([python,converter,hf,'--outfile',temporary,'--outtype','bf16'], output/'convert.log')
            proof=verify_bf16(temporary)
        state['raw']={**proof,'path':str(raw.resolve())}
        checkpoint()
    for quant in QUANTS:
        if quant in state['artifacts']:
            continue
        target = output / f'model-{quant}.gguf'
        args = [quantizer]
        if imatrix:
            if not Path(imatrix).is_file():
                raise FileNotFoundError(imatrix)
            args += ['--imatrix',imatrix]
        with atomic_destination(target) as temporary:
            run_logged(args + [raw,temporary,quant], output/f'{quant}.log')
            proof=verify_quant(temporary,quant)
        state['artifacts'][quant]={**proof,'path':str(target.resolve())}
        checkpoint()
    state['complete']=True
    checkpoint()
    raw.unlink()  # Only this invocation's successfully converted intermediate.
    return state['artifacts']


def rank_quantized(candidates, baseline, max_capability_drop, require_consistency=False):
    if set(baseline) != set(QUANTS) or max_capability_drop < 0:
        raise ValueError('Need both matched baseline quants and nonnegative damage limit')
    modes={v.get('scoring_mode','judge') for v in baseline.values()}
    if len(modes)!=1:
        raise ValueError('Matched baseline quants require one scoring mode')
    mode=next(iter(modes))
    counter='refusal_check_failures' if mode=='refusal-check' else 'noncompliant'
    if any(v['invalid'] or v['capability_pass_rate'] is None or search_rate(v) is None for v in baseline.values()):
        raise ValueError('Invalid quantized baseline')
    ranked = []
    for candidate in candidates:
        quants = candidate.get('quants', {})
        if set(quants) != set(QUANTS):
            continue
        if require_consistency and any(v.get('paired', {}).get('passes') is not True for v in quants.values()):
            continue
        if any(v.get('scoring_mode','judge')!=mode or v['invalid'] or v['capability_pass_rate'] is None or search_rate(v) is None for v in quants.values()):
            continue
        drops = [max(0.,baseline[q]['capability_pass_rate'] - quants[q]['capability_pass_rate']) for q in QUANTS]
        if max(drops) > max_capability_drop:
            continue
        if any(baseline[q].get('by_kind', {}).get(kind, {}).get('pass_rate', 0) - quants[q].get('by_kind', {}).get(kind, {}).get('pass_rate', 0) > max_capability_drop
               for q in QUANTS for kind in baseline[q].get('by_kind', {})):
            continue
        if any(quants[q].get('by_language', {}).get(lang, {}).get(counter, 0) / stats['total'] > stats[counter] / stats['total']
               for q in QUANTS for lang, stats in baseline[q].get('by_language', {}).items()):
            continue
        # Reject a quant regression even if the other precision got better.
        if any(search_rate(quants[q]) > search_rate(baseline[q]) for q in QUANTS):
            continue
        if not any(search_rate(quants[q]) < search_rate(baseline[q]) for q in QUANTS):
            continue
        key = (max(search_rate(v) for v in quants.values()),max(drops),max(v.get('disclaimer_rate') or 0 for v in quants.values()),max(v.get('mean_preamble_chars') or 0 for v in quants.values()))
        ranked.append((key,candidate))
    return [c for _,c in sorted(ranked,key=lambda pair:pair[0])]


GPU_LOCK_PATH = Path(os.environ.get('JDEWITT_GPU_LOCK', '/tmp/jdewitt-gpu.lock'))


def acquire_gpu_lock(timeout, secondary=False):
    """Serialise GPU jobs across processes. The B70 holds the primary lock; a
    second-rank test card uses its own lock so a test run cannot queue behind — or
    displace — a certified B70 job."""
    lock_path = Path(GPU_LOCK_PATH if not secondary else str(GPU_LOCK_PATH) + '.secondary')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open('a+')
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except OSError:
            if time.monotonic() >= deadline:
                handle.close()
                raise RuntimeError(
                    f'GPU ({lock_path.name}) is held by another jdewitt job; waited {timeout}s '
                    '(override with JDEWITT_GPU_WAIT or kill the holding job)')
            time.sleep(.5)


class ManagedGGUF:
    """Own one local Vulkan server. Refuse partial offload, changed artifacts or wrong models.
    The B70 is the preferred serving device and is used whenever llama-server can see it;
    secondary=True permits a non-B70 Vulkan device as a second-rank test backend, recorded
    on self.serving_backend so no caller can mistake it for a certified gate."""
    def __init__(self, artifact, server, ctx_size, max_tokens, log, startup_timeout=120, parallel=1, gpu_wait=None, secondary=False):
        if type(parallel) is not int or parallel<1:
            raise ValueError('Server parallel slot count must be a positive integer')
        self.parallel=parallel
        self.secondary=bool(secondary)
        self.artifact = Path(artifact).resolve()
        self.server, self.ctx_size, self.max_tokens = str(server), ctx_size, max_tokens
        self.log, self.startup_timeout = Path(log), startup_timeout
        self.process = None
        self.serving_backend = None
        wait = gpu_wait if gpu_wait is not None else float(os.environ.get('JDEWITT_GPU_WAIT', '300'))
        self.gpu_wait = 0.0 if wait is None else float(wait)

    def command(self,device,port,alias):
        # llama.cpp divides the context across slots. Preserve ctx_size for
        # EACH request rather than silently shrinking the admitted context.
        return [self.server,'--model',str(self.artifact),'--alias',alias,'--device',device,
            '--gpu-layers','999','--split-mode','none',*load_mode_flags(self.server),
            '--ctx-size',str(self.ctx_size*self.parallel),'--parallel',str(self.parallel),
            '--host','127.0.0.1','--port',str(port),'--jinja','--reasoning-format','deepseek',
            '--no-webui','--verbosity','4','--cors-origins','localhost','--no-cors-credentials']

    def select_device(self, devices):
        """Rank-aware Vulkan device choice: a B70 is always preferred when present."""
        candidates = []
        for line in devices.splitlines():
            found = re.match(r'\s*(Vulkan\d+):(.*)', line)
            if found:
                candidates.append((found.group(1), found.group(2).strip()))
        for device, label in candidates:
            if re.search(r'B70|BMG G31', label):
                return device, label, 'primary'
        if not candidates:
            raise RuntimeError('No Vulkan device in llama-server')
        if not self.secondary:
            raise RuntimeError('No verified B70 Vulkan device in llama-server')
        return candidates[0] + ('secondary',)

    def __enter__(self):
        self._gpu_lock = acquire_gpu_lock(self.gpu_wait, secondary=self.secondary)
        devices = subprocess.run([self.server,'--list-devices'], text=True,capture_output=True,check=True).stdout
        device, label, rank = self.select_device(devices)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0))
            port=sock.getsockname()[1]
        self.artifact_hash = sha256_file(self.artifact)
        alias = ('b70-' if rank=='primary' else 'test-') + self.artifact_hash
        self.serving_backend = {'device': device, 'name': label, 'rank': rank}
        env = {k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_', 'HF_TOKEN'))}
        self.log.parent.mkdir(parents=True,exist_ok=True)
        self.log_file = self.log.open('w')
        cmd = self.command(device,port,alias)
        self.process = subprocess.Popen(cmd, stdout=self.log_file,stderr=subprocess.STDOUT,env=env)
        self.client = ChatClient(f'http://127.0.0.1:{port}/v1',alias,self.max_tokens,ctx_size=self.ctx_size)
        deadline = time.monotonic() + self.startup_timeout
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f'llama-server exited; see {self.log}')
                try:
                    self.client.verify()
                    break
                except (urllib.error.URLError, ConnectionError, TimeoutError):
                    time.sleep(.1)
            else:
                raise RuntimeError(f'llama-server startup exceeded deadline; see {self.log}')
            log = self.log.read_text(errors='replace')
            offload = re.findall(r'offloaded (\d+)/(\d+) layers to GPU',log)
            if not offload or any(int(a) != int(b) for a,b in offload):
                raise RuntimeError('Could not prove full GPU offload')
            props = ChatClient(self.client.url.removesuffix('/v1'),alias,self.max_tokens)._request('/props')
            if Path(props['model_path']).resolve() != self.artifact:
                raise RuntimeError('Server loaded a different GGUF')
            actual_context = props.get('default_generation_settings',{}).get('n_ctx')
            if type(actual_context) is not int or actual_context < self.ctx_size:
                raise RuntimeError('Could not prove the requested server context budget')
            return self.client
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def __exit__(self,*args):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if hasattr(self,'log_file'):
            self.log_file.close()
        gpu_lock = getattr(self,'_gpu_lock',None)
        if gpu_lock is not None:
            try:
                fcntl.flock(gpu_lock,fcntl.LOCK_UN)
            finally:
                gpu_lock.close()
                self._gpu_lock = None
        if hasattr(self,'artifact_hash') and sha256_file(self.artifact) != self.artifact_hash:
            raise RuntimeError('Artifact changed during evaluation')


def copy_independent(source,target):
    """Reflink where available, otherwise stream-copy; never share an inode."""
    source,target=Path(source),Path(target)
    if source.is_symlink() or not source.is_file():
        raise ValueError('Release source must be a regular file, not a symlink')
    with atomic_destination(target) as temporary:
        with source.open('rb') as src, temporary.open('xb') as dst:
            try:
                fcntl.ioctl(dst.fileno(),0x40049409,src.fileno())  # Linux FICLONE
            except OSError as error:
                if error.errno not in {errno.EXDEV,errno.EINVAL,errno.ENOTTY,errno.EOPNOTSUPP,errno.ENOSYS}:
                    raise
                src.seek(0)
                dst.seek(0)
                dst.truncate(0)
                shutil.copyfileobj(src,dst,8*1024*1024)
            dst.flush()
    if source.stat().st_ino==target.stat().st_ino and source.stat().st_dev==target.stat().st_dev:
        raise RuntimeError('Release copy unexpectedly shares the working inode')


def release_inventory(directory):
    directory=Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('Release must be a regular directory')
    files={}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError('Release cannot contain symlinks')
        if path.is_file() and path!=directory/'files.json':
            files[str(path.relative_to(directory))]={'sha256':sha256_file(path),'bytes':path.stat().st_size}
    return files


def compute_evidence_digest(entries):
    """Fold the claim chain into one digest; each entry pins the previous state."""
    previous = ''
    for entry in entries:
        if entry.get('previous_digest') != previous:
            raise ValueError('Evidence ledger chain is broken at entry {}'.format(entry.get('seq')))
        previous = digest({k: v for k, v in entry.items() if k != 'digest'})
    return previous


def read_evidence(directory):
    """Load and integrity-check a release evidence ledger. Absent ledger returns None."""
    path = Path(directory) / 'evidence.json'
    if not path.is_file():
        return None
    if path.is_symlink():
        raise ValueError('Evidence ledger cannot be a symlink')
    chain = json.loads(path.read_text())
    if set(chain) != {'version', 'release_identity', 'entries', 'ledger_digest'} or chain['version'] != 1:
        raise ValueError('Evidence ledger shape mismatch')
    if chain['ledger_digest'] != compute_evidence_digest(chain['entries']):
        raise ValueError('Evidence ledger integrity mismatch')
    return chain


def claim_rate(claim, successes=None, total=None):
    """Attach uncertainty to a claim that states a rate.

    A claim like 'removal 31/40' is only falsifiable with its denominator and its
    interval: 31/40 and 31/310 mean different things and a reader has no way to tell.
    An explicit pair wins; otherwise the first `n/m` in the claim text is used and the
    interval is marked inferred, so nothing looks more measured than it is.
    """
    from .stats import wilson_interval
    if successes is None or total is None:
        match = re.search(r'(\d+)\s*/\s*(\d+)', str(claim))
        if not match:
            return None
        successes, total = int(match.group(1)), int(match.group(2))
        inferred = True
    else:
        successes, total = int(successes), int(total)
        inferred = False
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError(f'claim rate {successes}/{total} is not a valid fraction')
    interval = wilson_interval(successes, total)
    return {'successes': successes, 'total': total,
            'rate': round(successes / total, 4),
            'wilson_95_lower': round(interval['lower'], 4),
            'wilson_95_upper': round(interval['upper'], 4),
            'source': 'stated in the claim text' if inferred else 'given explicitly'}


def add_claim(directory, claim, evidence, verified_by, release_identity=None,
              successes=None, total=None):
    """Append an immutable claim to a release evidence ledger.

    Every entry records what is asserted, which artifact backs it, and which gate
    produced it. A tampered or swapped ledger is detected because the chain digest
    covers the release identity and the previous entries.
    """
    directory = Path(directory)
    if not (directory / 'files.json').is_file():
        raise FileNotFoundError(str(directory / 'files.json'))
    record = json.loads((directory / 'files.json').read_text())
    payload = {k: v for k, v in record.items() if k != 'record_digest'}
    if record.get('record_digest') != digest(payload):
        raise ValueError('Release index integrity mismatch')
    identity = record['release_identity']
    if release_identity is not None and release_identity != identity:
        raise ValueError('Release identity mismatch')
    chain = read_evidence(directory)
    if chain is None:
        chain = {'version': 1, 'release_identity': identity, 'entries': []}
    elif chain['release_identity'] != identity:
        raise ValueError('Evidence ledger belongs to a different release identity')
    entry = {'seq': len(chain['entries']) + 1, 'claim': str(claim), 'evidence': str(evidence),
             'verified_by': str(verified_by), 'at': _now_iso(),
             'previous_digest': chain['entries'][-1].get('digest', '') if chain['entries'] else ''}
    rate = claim_rate(claim, successes, total)
    if rate is not None:
        entry['rate'] = rate
    entry['digest'] = digest(entry)
    chain['entries'].append(entry)
    chain['ledger_digest'] = compute_evidence_digest(chain['entries'])
    atomic_json(directory / 'evidence.json', chain)
    return {'directory': str(directory.resolve()), 'entries': len(chain['entries']),
            'ledger_digest': chain['ledger_digest'], 'entry_digest': entry['digest']}


def evidence_summary(directory):
    chain = read_evidence(directory)
    if chain is None:
        return {'directory': str(Path(directory).resolve()), 'ledger_present': False}
    return {'directory': str(Path(directory).resolve()), 'ledger_present': True,
            'release_identity': chain['release_identity'], 'entries': chain['entries'],
            'ledger_digest': chain['ledger_digest']}


def _openssl(*args, timeout=20, **kwargs):
    try:
        return subprocess.run(['openssl', *args], capture_output=True, text=True,
                              check=True, timeout=timeout, **kwargs)
    except FileNotFoundError:
        raise RuntimeError('openssl is required for release signing but was not found on PATH')


def release_diff(a, b):
    """File-set and claim-chain differences between two release directories. Read-only."""
    a, b = Path(a), Path(b)

    def side(directory):
        record_file = directory / 'files.json'
        if not record_file.is_file():
            raise FileNotFoundError(str(record_file))
        record = json.loads(record_file.read_text())
        ledger = read_evidence(directory)
        signature_file = directory / 'signature.json'
        signature = None
        if signature_file.is_file():
            signature = json.loads(signature_file.read_text())
        return {'record': record,
                'files': record.get('files') or {},
                'identity': record.get('release_identity'),
                'entries': (ledger or {}).get('entries') or [],
                'ledger_digest': (ledger or {}).get('ledger_digest'),
                'signature': signature}
    left, right = side(a), side(b)
    only_a = sorted(set(left['files']) - set(right['files']))
    only_b = sorted(set(right['files']) - set(left['files']))
    changed = []
    for name in sorted(set(left['files']) & set(right['files'])):
        if left['files'][name].get('sha256') != right['files'][name].get('sha256'):
            changed.append({'name': name, 'left_sha256': left['files'][name].get('sha256'),
                            'right_sha256': right['files'][name].get('sha256')})
    left_claims = {entry.get('digest'): entry for entry in left['entries']}
    right_claims = {entry.get('digest'): entry for entry in right['entries']}
    shared_claims = sorted(set(left_claims) & set(right_claims), key=lambda d: left_claims[d].get('seq', 0))
    claims = {'identical': len(shared_claims),
              'added_on_right': [{'seq': e.get('seq'), 'claim': e.get('claim')}
                                 for d, e in sorted(right_claims.items(), key=lambda de: de[1].get('seq', 0)) if d not in left_claims],
              'added_on_left': [{'seq': e.get('seq'), 'claim': e.get('claim')}
                                for d, e in sorted(left_claims.items(), key=lambda de: de[1].get('seq', 0)) if d not in right_claims]}
    left_key = (left['signature'] or {}).get('public_key_pem')
    right_key = (right['signature'] or {}).get('public_key_pem')
    named, unnamed = _classify_differences(left, right, only_a, only_b, changed,
                                           left_key, right_key)
    return {'left': str(a.resolve()), 'right': str(b.resolve()),
            'release_identities': {'left': left['identity'], 'right': right['identity'],
                                   'same': left['identity'] == right['identity']},
            'files': {'only_in_left': only_a, 'only_in_right': only_b, 'changed': changed,
                      'identical': len(set(left['files']) & set(right['files'])) - len(changed)},
            'named_differences': named,
            'unnamed_differences': unnamed,
            'valid': not unnamed,
            'claims': claims,
            'ledger_digests': {'left': left['ledger_digest'], 'right': right['ledger_digest']},
            'signatures': {'left': bool(left['signature']), 'right': bool(right['signature']),
                           'same_public_key': left_key is not None and left_key == right_key},
            'note': 'Byte-level digests compared from each release record; contents of changed '
                    'files are never printed. Every difference is named by kind; an entry that '
                    'cannot be classified lands in unnamed_differences and makes valid false.'}


# File name -> the release claim whose absence/change the difference IS. A
# competitor-shaped release (a 16-bit model, no adapter, no plan card) differs
# from ours in exactly these named ways; anything this table cannot classify is
# reported as unnamed instead of being silently absorbed.
_RELEASE_DIFFERENCE_KINDS = {
    'adapter.gguf': 'adapter',
    'adapter.json': 'adapter',
    'model-merged.gguf': 'merged',
    'plan-card.tar': 'plan',
    'plan-card.json': 'plan',
    'twin.json': 'codec-realized',
    'evaluation.json': 'claims',
    'evidence.json': 'claims',
    'README.md': 'identity',
    'files.json': 'identity',
}


def _difference_kind(name):
    kind = _RELEASE_DIFFERENCE_KINDS.get(name)
    if kind:
        return kind
    if name.endswith('.gguf'):
        return 'merged'
    if name.endswith('.json'):
        return 'claims'
    return None


def _classify_differences(left, right, only_a, only_b, changed, left_key, right_key):
    """Name every difference by release claim; refuse to absorb the rest.

    ``named_differences`` groups file-set and digest differences under the claim
    they belong to (adapter / merged / plan / claims / identity / codec-realized),
    and adds the structural facts (a different release identity, a different
    signing key, a claim ledger that is not shared). ``unnamed_differences``
    carries whatever the name table cannot explain, so ``valid`` is true only
    when the comparison has accounted for every byte-level difference.
    """
    named, unnamed = [], []
    groups = {}
    for change, names in (('only_in_left', only_a), ('only_in_right', only_b),
                          ('changed', [row['name'] for row in changed])):
        for name in names:
            kind = _difference_kind(name)
            if kind is None:
                unnamed.append({'change': change, 'name': name})
                continue
            groups.setdefault((kind, change), []).append(name)
    for (kind, change), names in sorted(groups.items()):
        if change == 'only_in_left':
            detail = f'{kind} present in the left release and absent from the right'
        elif change == 'only_in_right':
            detail = f'{kind} present in the right release and absent from the left'
        else:
            detail = f'{kind} files share a name but not their sha256 across the two releases'
        named.append({'kind': kind, 'change': change, 'files': sorted(names), 'detail': detail})
    if left['identity'] != right['identity']:
        named.append({'kind': 'identity', 'change': 'changed', 'files': ['files.json'],
                      'detail': 'release identity differs: the two releases do not bind the same '
                                'evaluation record and README'})
    if left_key != right_key:
        named.append({'kind': 'identity', 'change': 'changed', 'files': ['signature.json'],
                      'detail': ('one release is unsigned' if left_key is None or right_key is None
                                 else 'the two releases are signed by different public keys')})
    if left['ledger_digest'] != right['ledger_digest']:
        added = len([e for e in right['entries'] if e.get('digest')
                     not in {x.get('digest') for x in left['entries']}])
        removed = len([e for e in left['entries'] if e.get('digest')
                       not in {x.get('digest') for x in right['entries']}])
        named.append({'kind': 'claims', 'change': 'changed', 'files': ['evidence.json'],
                      'detail': f'claim ledgers differ: {added} entry/entries only on the right, '
                                f'{removed} only on the left'})
    return named, unnamed



def keygen(path):
    """Create a new Ed25519 release-signing key; never overwrites."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _openssl('genpkey', '-algorithm', 'ed25519', '-out', str(path))
    os.chmod(path, 0o600)
    public = _openssl('pkey', '-pubout', '-in', str(path)).stdout
    return {'private_key': str(path.resolve()), 'public_key_pem': public,
            'note': 'Protect the private key file (0600). Publish only the public key.'}


# Seal files are written next to a finished release and must never be mistaken for
# release payload: `verify_release` re-hashes the inventory, and a seal that appeared
# in it would make the release invalid the moment it was signed.
REPRO_SEAL_FILES = ('repro.json', 'repro-spec.json')
_MISSING = object()

# Paths worth trying when an evaluation record pins no argv under `repro.serve_argv`.
_SERVE_ARGV_PATHS = ('serve_argv', 'argv', 'server.argv', 'behaviour.serve_argv',
                     'evaluation.serve_argv', 'merged_verification.serve_argv',
                     'winner.serve_argv', 'settings.serve_argv')


def _json_if(path):
    path = Path(path)
    if not path.is_file():
        return None
    if path.is_symlink():
        raise ValueError(f'{path.name} cannot be a symlink')
    return json.loads(path.read_text())


def _declared_path(value):
    """A path declared as a string or as a {path|file: ...} record, else None."""
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, dict):
        for key in ('path', 'file'):
            if isinstance(value.get(key), str) and value[key].strip():
                return value[key]
    return None


def _resolve(directory, text):
    path = Path(text)
    return path if path.is_absolute() else Path(directory) / path


def _portable_path(directory, path):
    """A path relative to the release when it lives inside it, absolute otherwise."""
    path = Path(path).resolve()
    directory = Path(directory).resolve()
    try:
        return str(path.relative_to(directory))
    except ValueError:
        return str(path)


def _dig_path(document, path):
    current = document
    for part in path.split('.'):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _gguf_facts(path):
    """Architecture, declared file type and observed per-tensor quant types; header reads only."""
    try:
        reader = gguf.GGUFReader(str(path))
        arch_field = reader.get_field('general.architecture')
        arch = str(arch_field.contents()) if arch_field is not None else None
        type_field = reader.get_field('general.file_type')
        file_type = None
        if type_field is not None:
            kind = int(type_field.contents())
            try:
                file_type = gguf.LlamaFileType(kind).name
            except ValueError:
                file_type = str(kind)
        counts = {}
        for tensor in reader.tensors:
            name = tensor.tensor_type.name
            counts[name] = counts.get(name, 0) + 1
        total = len(reader.tensors)
        del reader
    except Exception as error:
        raise ValueError(f'{Path(path).name} is not a readable GGUF: {error}') from error
    if not total:
        raise ValueError(f'{Path(path).name} carries no tensors')
    return {'arch': arch, 'file_type': file_type,
            'quant_types': dict(sorted(counts.items())), 'tensors': total}


def _gguf_tensor_table(path):
    """Per-tensor quant type and stored byte cost from the released bytes."""
    try:
        reader = gguf.GGUFReader(str(path))
        table = {tensor.name: {'quant': tensor.tensor_type.name,
                               'bytes': int(tensor.n_bytes)}
                 for tensor in reader.tensors}
        del reader
    except Exception as error:
        raise ValueError(f'{Path(path).name} is not a readable GGUF: {error}') from error
    if not table:
        raise ValueError(f'{Path(path).name} carries no tensors')
    return table


def _read_plan_bundle(path):
    """Manifest and payload digest of a plan bundle, refusing a card that lies about its bytes."""
    import tarfile
    from .registry import BUNDLE_FORMAT
    path = Path(path)
    try:
        with tarfile.open(path, 'r') as tar:
            names = tar.getnames()
            if len(names) != 2 or set(names) != {f'{BUNDLE_FORMAT}/manifest.json',
                                                 f'{BUNDLE_FORMAT}/plan.pt'}:
                raise ValueError(f'{path.name} members {names} do not match '
                                 f'{BUNDLE_FORMAT}/manifest.json + plan.pt')
            manifest_member = tar.extractfile(f'{BUNDLE_FORMAT}/manifest.json')
            payload_member = tar.extractfile(f'{BUNDLE_FORMAT}/plan.pt')
            if manifest_member is None or payload_member is None:
                raise ValueError(f'{path.name} bundle members are not regular files')
            manifest = json.loads(manifest_member.read())
            payload = payload_member.read()
    except (tarfile.TarError, json.JSONDecodeError) as error:
        raise ValueError(f'{path.name} is not a readable plan-bundle: {error}') from error
    if not isinstance(manifest, dict) or manifest.get('format') != BUNDLE_FORMAT:
        found = manifest.get('format') if isinstance(manifest, dict) else None
        raise ValueError(f'{path.name} manifest format {found!r} is not {BUNDLE_FORMAT!r}')
    payload_sha = hashlib.sha256(payload).hexdigest()
    if manifest.get('payload_sha256') != payload_sha:
        raise ValueError(f'{path.name} plan card payload digest disagrees with the bundle payload: '
                         f'manifest {str(manifest.get("payload_sha256"))[:12]}... '
                         f'payload {payload_sha[:12]}...')
    meta = manifest.get('meta')
    if not isinstance(meta, dict):
        raise ValueError(f'{path.name} manifest carries no signed meta')
    if meta.get('payload_sha256') != payload_sha:
        raise ValueError(f'{path.name} signed meta payload digest disagrees with the bundle payload')
    if meta.get('meta_digest') != digest({key: value for key, value in meta.items()
                                          if key != 'meta_digest'}):
        raise ValueError(f'{path.name} signed meta fails its own meta_digest')
    return manifest, payload_sha


def _repro_spec(directory):
    """The declared reproduction operands: repro-spec.json wins over evaluation.json's `repro` block."""
    directory = Path(directory)
    evaluation = _json_if(directory / 'evaluation.json')
    if not isinstance(evaluation, dict):
        raise ValueError(f'{directory}/evaluation.json is not a JSON object')
    spec = dict(evaluation.get('repro') or {}) if isinstance(evaluation.get('repro'), dict) else {}
    extra = _json_if(directory / 'repro-spec.json')
    if extra is not None:
        if not isinstance(extra, dict):
            raise ValueError('repro-spec.json must be a JSON object')
        spec.update(extra)
    return evaluation, spec


def _file_with_digest(directory, wanted):
    if not isinstance(wanted, str) or len(wanted) != 64:
        return None
    for path in sorted(Path(directory).rglob('*')):
        if path.name in REPRO_SEAL_FILES or path.is_symlink() or not path.is_file():
            continue
        if sha256_file(path) == wanted:
            return path
    return None


def _repro_base(directory, evaluation, spec):
    """The user's quant artifact: name, digest, size, GGUF header and observed quant types."""
    declared = spec.get('base', spec.get('base_artifact'))
    declared_path = _declared_path(declared)
    declared_sha = declared.get('sha256') if isinstance(declared, dict) else None
    twin = _json_if(Path(directory) / 'twin.json') or {}
    twin_base = twin.get('base')
    if not isinstance(twin_base, dict):
        twin_base = {}
    twin_sha = twin_base.get('sha256') if isinstance(twin_base.get('sha256'), str) else None
    source = 'repro spec'
    path = _resolve(directory, declared_path) if declared_path else None
    if path is None:
        for key in ('base_artifact', 'base_artifact_path'):
            if isinstance(evaluation.get(key), str) and evaluation[key].strip():
                path, source = _resolve(directory, evaluation[key]), f'evaluation.json {key}'
                break
    if path is None:
        match = _file_with_digest(directory, declared_sha or twin_sha)
        if match is not None:
            path, source = match, 'release directory digest'
    if path is None:
        fallback = Path(directory) / 'base.gguf'
        if fallback.is_file():
            path, source = fallback, 'base.gguf in release directory'
    if path is None:
        declared_digest = declared_sha or twin_sha
        if not declared_digest:
            raise ValueError('no base artifact: declare repro.base (evaluation.json or '
                             'repro-spec.json), ship base.gguf, or record twin.json.base.sha256')
        return {'name': twin_base.get('file') or twin_base.get('name'), 'path': None,
                'sha256': declared_digest, 'bytes': None, 'arch': twin_base.get('arch'),
                'file_type': None, 'quant_types': None,
                'tensors': twin_base.get('tensor_count'),
                'observed': False,
                'source': 'twin.json.base digest; the artifact is not shipped, so it cannot be re-hashed'}
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'base artifact {path} must be an existing regular file')
    facts = _gguf_facts(path)
    return {'name': path.name, 'path': _portable_path(directory, path),
            'sha256': sha256_file(path), 'bytes': path.stat().st_size,
            'arch': facts['arch'], 'file_type': facts['file_type'],
            'quant_types': facts['quant_types'], 'tensors': facts['tensors'],
            'observed': True, 'source': source}


def _repro_plan(directory, evaluation, spec):
    """The digest-pinned plan card: bundle digest, manifest payload digest and registry key."""
    declared = spec.get('plan')
    declared_path = _declared_path(declared)
    key_hint = declared.get('registry_key') if isinstance(declared, dict) else None
    card = _json_if(Path(directory) / 'plan-card.json')
    if declared_path is None:
        for name in ('plan-card.tar', 'plan.tar'):
            if (Path(directory) / name).is_file():
                declared_path = name
                break
    if declared_path is None and isinstance(card, dict):
        declared_path = card.get('bundle_file')
    if declared_path is None:
        return None
    path = _resolve(directory, declared_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'plan bundle {path} is missing')
    manifest, payload_sha = _read_plan_bundle(path)
    bundle_sha = sha256_file(path)
    card_problem = None
    if isinstance(card, dict):
        if card.get('bundle_sha256') not in (None, bundle_sha):
            card_problem = (f'plan card names bundle {str(card.get("bundle_sha256"))[:12]}... '
                            f'but the bundle is {bundle_sha[:12]}...')
        elif card.get('payload_sha256') not in (None, payload_sha):
            card_problem = 'plan card payload digest disagrees with the plan card payload'
    key = key_hint or manifest.get('key') or (card.get('key') if isinstance(card, dict) else None)
    if key_hint and manifest.get('key') not in (None, key_hint) and card_problem is None:
        card_problem = f'plan bundle key {manifest.get("key")!r} is not the declared {key_hint!r}'
    record = {'bundle': _portable_path(directory, path), 'bundle_sha256': bundle_sha,
              'registry_key': key, 'manifest_payload_digest': payload_sha,
              'meta_digest': (manifest.get('meta') or {}).get('meta_digest')}
    if card_problem is not None:
        record['card_problem'] = card_problem
    return record


def _repro_suites(directory, evaluation, spec):
    """Per-axis suite file: digest, row count and kind counts."""
    declared = spec.get('suites')
    if declared is None:
        declared = evaluation.get('suites')
    if declared is None:
        config = (evaluation.get('provenance') or {}).get('config') or {}
        declared = {name: config[name] for name in ('directions', 'selection', 'holdout')
                    if isinstance(config.get(name), str)}
    if declared is None:
        return None
    if not isinstance(declared, dict):
        raise ValueError('repro.suites must map an axis name to a suite path')
    out = {}
    for axis in sorted(declared):
        text = _declared_path(declared[axis])
        if text is None:
            raise ValueError(f'suite {axis!r} names no path')
        path = _resolve(directory, text)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'suite {axis!r} {path} is missing')
        rows, kinds, unparsed = 0, {}, 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A damaged row still counts: the digest is the operand, and a
                # check must name the suite rather than lose the whole axis.
                unparsed += 1
                continue
            kind = row.get('kind') if isinstance(row, dict) else None
            if isinstance(kind, str):
                kinds[kind] = kinds.get(kind, 0) + 1
        out[str(axis)] = {'path': _portable_path(directory, path), 'sha256': sha256_file(path),
                          'rows': rows, 'kinds': dict(sorted(kinds.items())),
                          'unparsed': unparsed}
    return out or None


def _repro_serve_argv(directory, evaluation, spec):
    """The full serve argv per axis, as the packet recorded it."""
    declared = spec.get('serve_argv')
    if declared is None:
        for path in _SERVE_ARGV_PATHS:
            value = _dig_path(evaluation, path)
            if isinstance(value, list) and value:
                declared = {'default': value}
                break
    if declared is None:
        twin = _json_if(Path(directory) / 'twin.json') or {}
        value = _dig_path(twin, 'behaviour.serve_argv')
        if isinstance(value, list) and value:
            declared = {'merged': value}
    if declared is None:
        return None
    if not isinstance(declared, dict):
        raise ValueError('repro.serve_argv must map an axis name to a full argv list')
    out = {}
    for axis in sorted(declared):
        argv = declared[axis]
        if (not isinstance(argv, list) or not argv
                or any(not isinstance(token, str) for token in argv)):
            raise ValueError(f'serve_argv {axis!r} must be a nonempty list of strings')
        out[str(axis)] = [str(token) for token in argv]
    return out or None


def _repro_codec(directory, evaluation, spec):
    """What the served artifact stores per edited tensor, and the bytes that cost."""
    declared = spec.get('codec')
    twin = _json_if(Path(directory) / 'twin.json') or {}
    artifact_hint, rows = None, []
    if isinstance(declared, dict):
        artifact_hint = _declared_path(declared) or _declared_path(declared.get('artifact'))
        raw = declared.get('tensors')
        if isinstance(raw, dict):
            raw = [dict(value, name=value.get('name', key)) for key, value in raw.items()]
        if isinstance(raw, list):
            rows = [dict(value) for value in raw if isinstance(value, dict)]
    elif isinstance(declared, list):
        rows = [dict(value) for value in declared if isinstance(value, dict)]
    if not rows and isinstance(twin, dict):
        artifact_hint = artifact_hint or _declared_path(twin.get('artifact'))
        types = twin.get('edit_type_per_tensor') or {}
        for row in ((twin.get('delivered') or {}).get('tensors') or []):
            if not isinstance(row, dict) or not isinstance(row.get('name'), str):
                continue
            rows.append({'name': row['name'], 'quant': types.get(row['name']),
                         'realized_cos': row.get('cos')})
    if not rows:
        return None
    artifact_hint = artifact_hint or 'model-merged.gguf'
    artifact = _resolve(directory, artifact_hint)
    if artifact.is_symlink() or not artifact.is_file():
        raise ValueError(f'codec artifact {artifact} is missing')
    table = _gguf_tensor_table(artifact)
    served = []
    for row in sorted(rows, key=lambda item: str(item.get('name'))):
        name = row.get('name')
        if not isinstance(name, str):
            raise ValueError('codec row has no tensor name')
        if name not in table:
            raise ValueError(f'codec names tensor {name!r}, which {artifact.name} does not carry')
        quant = row.get('quant') or table[name]['quant']
        if quant != table[name]['quant']:
            raise ValueError(f'{name} ships as {table[name]["quant"]} but the codec table '
                             f'says {quant}; the artifact and the codec table disagree')
        cos = row.get('realized_cos', row.get('cos'))
        if isinstance(cos, bool) or not isinstance(cos, (int, float)):
            raise ValueError(f'codec row {name!r} has no realized cos')
        served.append({'name': name, 'quant': quant,
                       'realized_cos': round(float(cos), 6),
                       'bytes': table[name]['bytes']})
    return {'artifact': {'name': artifact.name,
                         'path': _portable_path(directory, artifact),
                         'sha256': sha256_file(artifact)},
            'tensors': served}


def _repro_tool():
    """The tool that built the index: package version and the checkout it ran from."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        package_version = version('jdewitt')
    except PackageNotFoundError:
        from . import __version__ as package_version
    describe = None
    try:
        result = subprocess.run(['git', '-C', str(Path(__file__).resolve().parent.parent),
                                 'describe', '--always', '--dirty', '--tags'],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            describe = result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        describe = None
    return {'package': 'jdewitt', 'version': package_version, 'git_describe': describe}


def _repro_runtime(directory, evaluation, spec):
    """The binary, interpreter and backend a reproduction needs; a hash where one exists."""
    declared = spec.get('runtime') if isinstance(spec.get('runtime'), dict) else {}
    server_sha = declared.get('llama_server_sha256')
    server_hint = _declared_path(declared.get('llama_server'))
    if server_sha is None and server_hint is not None:
        server = _resolve(directory, server_hint)
        if server.is_file() and not server.is_symlink():
            server_sha = sha256_file(server)
    provenance = evaluation.get('provenance') or {}
    if server_sha is None:
        server_sha = provenance.get('server_sha256')
    backend = declared.get('backend')
    if backend is None:
        compute = provenance.get('compute_backend') or {}
        backend = compute.get('name') or compute.get('type') or 'cpu'
    import platform
    return {'llama_server_sha256': server_sha, 'python': platform.python_version(),
            'backend': backend or 'unknown'}


def build_repro(directory, tolerant=False):
    """Machine-readable reproduction index over every operand a release depends on.

    Version 2 keeps every version-1 field (release identity, index/evaluation/README
    digests, release file digests, ledger digest) and adds `operands`: the base GGUF,
    the plan card, each suite, each serve argv, the per-tensor codec table, and the
    tool/runtime that produced the index. `binding_digest` covers the portable
    operands only, so a signature can be checked on hardware whose Python or backend
    differs without pretending those are release bytes. With ``tolerant=True`` a
    missing or unreadable operand is recorded as ``{error: ...}`` instead of raising,
    which is how `release-repro --check` names every move at once.
    """
    directory = Path(directory)
    record = json.loads((directory / 'files.json').read_text())
    evaluation, spec = _repro_spec(directory)
    repro = {'version': 2,
             'release_identity': record['release_identity'],
             'files_index_digest': record['record_digest'],
             'evaluation_sha256': sha256_file(directory / 'evaluation.json'),
             'readme_sha256': sha256_file(directory / 'README.md'),
             'files': {name: metadata['sha256'] for name, metadata in record['files'].items()}}
    ledger = read_evidence(directory)
    if ledger is not None:
        repro['ledger_digest'] = ledger['ledger_digest']
    operands = {}
    for key, builder in (('base', _repro_base), ('plan', _repro_plan),
                         ('suites', _repro_suites), ('serve_argv', _repro_serve_argv),
                         ('codec', _repro_codec)):
        try:
            operands[key] = builder(directory, evaluation, spec)
        except Exception as error:
            if not tolerant:
                raise
            operands[key] = {'error': f'{type(error).__name__}: {error}'}
        value = operands.get(key)
        if isinstance(value, dict) and isinstance(value.get('card_problem'), str):
            if not tolerant:
                raise ValueError(value['card_problem'])
            operands[key] = dict(value)
    operands['tool'] = _repro_tool()
    operands['runtime'] = _repro_runtime(directory, evaluation, spec)
    repro['operands'] = operands
    repro['binding_digest'] = digest(_portable_repro(repro))
    repro['repro_digest'] = digest(repro)
    return repro


def _portable_repro(repro):
    """The part of a v2 index that describes release bytes rather than the host that read them."""
    view = {key: value for key, value in repro.items()
            if key not in {'repro_digest', 'binding_digest'}}
    operands = dict(view.get('operands') or {})
    operands.pop('tool', None)
    operands.pop('runtime', None)
    view['operands'] = operands
    return view


def _build_repro_v1(directory):
    """The exact version-1 payload, so signatures made before operands existed still verify."""
    directory = Path(directory)
    record = json.loads((directory / 'files.json').read_text())
    repro = {'version': 1,
             'release_identity': record['release_identity'],
             'files_index_digest': record['record_digest'],
             'evaluation_sha256': sha256_file(directory / 'evaluation.json'),
             'readme_sha256': sha256_file(directory / 'README.md'),
             'files': {name: metadata['sha256'] for name, metadata in record['files'].items()}}
    ledger = read_evidence(directory)
    if ledger is not None:
        repro['ledger_digest'] = ledger['ledger_digest']
    repro['repro_digest'] = digest(repro)
    return repro


def _operand_count(repro):
    files = len(repro.get('files') or {})
    operands = len(_flatten(repro.get('operands') or {}))
    return files + operands


def _flatten(value, prefix=''):
    """Every leaf of a JSON value under a dotted path: `suites.retention.sha256`, `argv[1]`."""
    leaves = {}
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            leaves.update(_flatten(value[key], f'{prefix}.{key}' if prefix else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            leaves.update(_flatten(item, f'{prefix}[{index}]'))
    elif prefix:
        leaves[prefix] = value
    return leaves


def _write_repro_file(path, repro):
    """Write the index with sorted keys, so identical operands produce identical bytes."""
    atomic_text(Path(path), json.dumps(repro, ensure_ascii=False, indent=2,
                                       sort_keys=True, allow_nan=False) + '\n')


def write_repro(directory, out=None):
    """Build the reproduction index and write it as repro.json (the build path)."""
    directory = Path(directory)
    repro = build_repro(directory)
    out = Path(out) if out is not None else directory / 'repro.json'
    if out.is_symlink():
        raise ValueError('repro.json cannot be a symlink')
    _write_repro_file(out, repro)
    return {'directory': str(directory.resolve()), 'repro': str(out.resolve()),
            'repro_digest': repro['repro_digest'], 'binding_digest': repro['binding_digest'],
            'operand_count': _operand_count(repro),
            'next': f'release-repro {directory} --check'}


def _signature_record(directory):
    """Load and cryptographically check signature.json; the payload is returned unverified."""
    import base64
    import tempfile
    directory = Path(directory)
    path = directory / 'signature.json'
    if path.is_symlink():
        raise ValueError('Release signature cannot be a symlink')
    if not path.is_file():
        raise FileNotFoundError(path)
    record = json.loads(path.read_text())
    if (record.get('version') != 1 or record.get('algorithm') != 'ed25519'
            or set(record) != {'version', 'algorithm', 'public_key_pem', 'payload', 'signature_base64'}):
        raise ValueError('Release signature shape mismatch')
    payload = record['payload']
    if not isinstance(payload, dict) or 'version' not in payload or 'repro_digest' not in payload:
        raise ValueError('Release signature payload is not a repro index')
    message = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    signature = base64.b64decode(record['signature_base64'])
    with tempfile.TemporaryDirectory() as tmp:
        message_path = Path(tmp) / 'message'
        signature_path = Path(tmp) / 'signature'
        key_path = Path(tmp) / 'public.pem'
        message_path.write_bytes(message)
        signature_path.write_bytes(signature)
        key_path.write_text(record['public_key_pem'])
        try:
            _openssl('pkeyutl', '-verify', '-pubin', '-inkey', str(key_path), '-rawin',
                     '-in', str(message_path), '-sigfile', str(signature_path))
        except subprocess.CalledProcessError:
            raise ValueError('Release signature verification failed')
    return record


def sign_release(directory, secret_path):
    """Sign the release reproduction index with an Ed25519 key. Refuses to guess or create a key."""
    import base64
    import tempfile
    directory = Path(directory)
    secret = Path(secret_path)
    if not secret.is_file():
        raise FileNotFoundError(f'Signing key not found: {secret} (create one with release-keygen)')
    verify_release(directory)
    repro = build_repro(directory)
    message = json.dumps(repro, sort_keys=True, separators=(',', ':'))
    with tempfile.TemporaryDirectory() as tmp:
        message_path = Path(tmp) / 'message'
        message_path.write_text(message)
        signature_path = Path(tmp) / 'signature'
        _openssl('pkeyutl', '-sign', '-inkey', str(secret), '-rawin',
                 '-in', str(message_path), '-out', str(signature_path))
        signature = base64.b64encode(signature_path.read_bytes()).decode()
    public = _openssl('pkey', '-pubout', '-in', str(secret)).stdout
    record = {'version': 1, 'algorithm': 'ed25519', 'public_key_pem': public,
              'payload': repro, 'signature_base64': signature}
    # The index lands before the seal: a signature.json without its repro.json is a
    # seal over nothing, while a repro.json without a signature is visibly unsigned.
    _write_repro_file(directory / 'repro.json', repro)
    atomic_json(directory / 'signature.json', record)
    return {'directory': str(directory.resolve()), 'repro': str((directory / 'repro.json').resolve()),
            'repro_digest': repro['repro_digest'], 'binding_digest': repro['binding_digest'],
            'signature_digest': repro['repro_digest'],
            'public_key_sha256': hashlib.sha256(public.encode()).hexdigest(),
            'operand_count': _operand_count(repro),
            'next': f'release-repro {directory} --check',
            'note': 'Signs byte reproducibility, not model quality; store the public key where buyers can pin it.'}


def verify_signature(directory, public_path=None):
    """Verify a release signature; with public_path also pin authenticity to that key."""
    directory = Path(directory)
    record = _signature_record(directory)
    payload = record['payload']
    if payload.get('version') == 1:
        repro = _build_repro_v1(directory)
        if payload != repro:
            raise ValueError('Signature payload does not match current release state')
    else:
        repro = build_repro(directory)
        if payload.get('binding_digest') != repro.get('binding_digest'):
            raise ValueError('Signature payload does not match current release state')
    if public_path is not None:
        pinned = Path(public_path).read_text().strip()
        if pinned != record['public_key_pem'].strip():
            raise ValueError('Signing key is not the pinned public key')
    return {'valid': True, 'directory': str(directory.resolve()),
            'repro_digest': payload['repro_digest'], 'release_identity': payload['release_identity'],
            'binding_digest': payload.get('binding_digest'),
            'authenticity': 'verified against pinned public key' if public_path else
                            'signature math verified; not checked against a pinned public key'}


def check_repro(directory, strict_provenance=False):
    """Re-hash every operand and compare it with the seal; name each moved operand by JSON path.

    The seal is signature.json's payload when present (the signed index), otherwise
    repro.json. Release files are re-hashed from disk rather than compared with
    files.json, so a changed artifact is a move even if its index entry was rewritten.
    Tool/runtime values describe the box that built the index, not release bytes: they
    are reported as `provenance_drift` and only fail with ``strict_provenance``.
    """
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('Release must be a regular directory')
    signature_path = directory / 'signature.json'
    repro_path = directory / 'repro.json'
    if signature_path.is_file():
        expected, seal = _signature_record(directory)['payload'], 'signature.json'
    elif repro_path.is_file():
        if repro_path.is_symlink():
            raise ValueError('repro.json cannot be a symlink')
        expected, seal = json.loads(repro_path.read_text()), 'repro.json'
        if (not isinstance(expected, dict)
                or expected.get('repro_digest') != digest({key: value for key, value in expected.items()
                                                          if key != 'repro_digest'})):
            raise ValueError('repro.json integrity mismatch')
    else:
        raise FileNotFoundError(f'{directory}/repro.json '
                                f'(build it: release-repro {directory} --out repro.json)')
    if expected.get('version') == 1:
        raise ValueError('repro index version 1 carries no operands; rebuild it with '
                         'release-repro DIR --out repro.json')
    actual = build_repro(directory, tolerant=True)
    moved, drift, problems = [], [], []
    files_checked = 0
    for name, metadata in sorted((expected.get('files') or {}).items()):
        files_checked += 1
        wanted = metadata if isinstance(metadata, str) else (metadata or {}).get('sha256')
        path = directory / name
        try:
            found = None if (path.is_symlink() or not path.is_file()) else sha256_file(path)
        except OSError:
            found = None
        if found != wanted:
            moved.append({'operand': f'files.{name}.sha256', 'expected': wanted, 'actual': found})
    for key in ('version', 'release_identity', 'files_index_digest', 'evaluation_sha256',
                'readme_sha256', 'ledger_digest'):
        if expected.get(key) != actual.get(key):
            moved.append({'operand': key, 'expected': expected.get(key), 'actual': actual.get(key)})
    expected_flat = _flatten(expected.get('operands') or {})
    actual_flat = _flatten(actual.get('operands') or {})
    broken = {key: value for key, value in (actual.get('operands') or {}).items()
              if isinstance(value, dict) and isinstance(value.get('error'), str)}
    for key in sorted(broken):
        moved.append({'operand': key,
                      'expected': (expected.get('operands') or {}).get(key),
                      'actual': broken[key]})
    for operand in sorted(set(expected_flat) | set(actual_flat)):
        root = operand.split('.', 1)[0].split('[', 1)[0]
        if root in broken:
            continue
        if expected_flat.get(operand, _MISSING) == actual_flat.get(operand, _MISSING):
            continue
        entry = {'operand': operand, 'expected': expected_flat.get(operand),
                 'actual': actual_flat.get(operand)}
        if operand.startswith(('tool.', 'runtime.')):
            drift.append(entry)
        else:
            moved.append(entry)
    for key in ('base', 'plan', 'suites', 'serve_argv', 'codec'):
        value = (actual.get('operands') or {}).get(key)
        if isinstance(value, dict) and isinstance(value.get('error'), str):
            problems.append({'operand': key, 'problem': value['error']})
        elif isinstance(value, dict) and isinstance(value.get('card_problem'), str):
            problems.append({'operand': key, 'problem': value['card_problem']})
    if seal == 'signature.json' and repro_path.is_file():
        shipped = json.loads(repro_path.read_text())
        if shipped.get('repro_digest') != expected.get('repro_digest'):
            moved.append({'operand': 'repro.repro_digest', 'expected': expected.get('repro_digest'),
                          'actual': shipped.get('repro_digest')})
    if strict_provenance:
        moved.extend(drift)
        drift = []
    valid = not moved and not problems
    return {'valid': valid, 'directory': str(directory.resolve()), 'seal': seal,
            'operand_count': files_checked + len(expected_flat),
            'checked': {'files': files_checked, 'operands': len(expected_flat)},
            'resolved': {key: (actual.get('operands') or {}).get(key)
                         for key in ('base', 'plan', 'suites', 'serve_argv', 'codec')},
            'moved': moved, 'provenance_drift': drift, 'problems': problems,
            'next': (f'release-repro {directory} --check' if not valid else
                     'release-verify-signature DIR --public PUB.pem; verify-release DIR'),
            'note': ('Every operand re-hashed from the release directory and the paths the '
                     'index names; tool/runtime differences are provenance drift, not release '
                     'bytes, and fail only with --strict-provenance.')}


def verify_release(directory,expected_identity=None):
    """Check exact file bytes; this does not certify a model's semantic quality."""
    directory=Path(directory)
    if directory.is_symlink() or (directory/'files.json').is_symlink():
        raise ValueError('Release/index cannot be symlinks')
    record=json.loads((directory/'files.json').read_text())
    payload={k:v for k,v in record.items() if k!='record_digest'}
    if set(record)!={'version','release_identity','files','record_digest'} or record['version']!=1 or record['record_digest']!=digest(payload):
        raise ValueError('Release index integrity mismatch')
    for name,metadata in record['files'].items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or str(p)!=name or not name or name in {'evidence.json','signature.json',*REPRO_SEAL_FILES}:
            raise ValueError('Invalid release index path')
        if set(metadata)!={'sha256','bytes'} or type(metadata['bytes']) is not int or metadata['bytes']<0:
            raise ValueError('Invalid release file metadata')
    if expected_identity is not None and record['release_identity']!=expected_identity:
        raise ValueError('Release identity mismatch')
    actual=release_inventory(directory)
    if {k:v for k,v in actual.items() if k not in {'evidence.json','signature.json',*REPRO_SEAL_FILES}}!=record['files']:
        raise ValueError('Release file set/hash mismatch')
    evaluation=json.loads((directory/'evaluation.json').read_text())
    readme=(directory/'README.md').read_text()
    if record['release_identity']!=digest({'evaluation':evaluation,'readme':readme}):
        raise ValueError('Release metadata identity mismatch')
    ledger=read_evidence(directory)
    if ledger is not None and ledger['release_identity']!=record['release_identity']:
        raise ValueError('Evidence ledger belongs to a different release identity')
    result={'valid':True,'directory':str(directory.resolve()),'files':len(actual),
            'bytes':sum(v['bytes'] for v in actual.values()),'release_identity':record['release_identity'],
            'ledger_present':ledger is not None,
            'scope':'File integrity only, not independent proof of model quality or authenticity'}
    if ledger is not None:
        result['evidence_entries']=len(ledger['entries'])
        result['ledger_digest']=ledger['ledger_digest']
    return result


def create_release(directory,hf,artifacts,metadata,readme):
    directory,hf=Path(directory),Path(hf)
    identity=digest({'evaluation':metadata,'readme':readme})
    with exclusive(directory.parent,'.'+directory.name+'.lock'):
        if directory.exists():
            return verify_release(directory,identity)
        stage=directory.with_name(directory.name+'-staging-'+uuid.uuid4().hex)
        stage.mkdir()
        if not hf.is_dir() or hf.is_symlink() or set(artifacts)!=set(QUANTS):
            raise ValueError('Release requires a regular HF export and both verified quants')
        (stage/'hf').mkdir()
        expected_hf=metadata.get('winner',{}).get('export_verification',{}).get('file_sha256',{})
        if expected_hf and set(expected_hf)!={p.name for p in hf.iterdir() if p.is_file()}:
            raise ValueError('HF export file set differs from the verified finalist')
        for source in sorted(hf.iterdir()):
            if not source.is_file() or source.is_symlink():
                raise ValueError('HF export contains unsupported nonregular entries')
            if source.name in expected_hf and sha256_file(source)!=expected_hf[source.name]:
                raise ValueError('HF export changed since resident verification')
            copy_independent(source,stage/'hf'/source.name)
            if sha256_file(source)!=sha256_file(stage/'hf'/source.name):
                raise ValueError('HF source changed during independent release copy')
        for quant,artifact in artifacts.items():
            source=Path(artifact['path'])
            if sha256_file(source)!=artifact['sha256']:
                raise ValueError('Quantized artifact changed before release copy')
            copy_independent(source,stage/f'model-{quant}.gguf')
            if sha256_file(stage/f'model-{quant}.gguf')!=artifact['sha256']:
                raise ValueError('Quantized release copy hash mismatch')
        atomic_json(stage/'evaluation.json',metadata)
        atomic_text(stage/'README.md',readme)
        record={'version':1,'release_identity':identity,'files':release_inventory(stage)}
        record['record_digest']=digest(record)
        atomic_json(stage/'files.json',record)
        verify_release(stage,identity)
        sync_directory(stage/'hf')
        sync_directory(stage)
        stage.rename(directory)
        sync_directory(directory.parent)
        add_claim(directory,'Release created: file set matches the verified finalist and evaluation record',
                  'files.json release index','jdewitt create_release')
        return verify_release(directory,identity)


# ---------------------------------------------------------------------------
# Phase 6 assembly: the edit on the artifact the user already serves.
#
# docs/ROADMAP.md "Phase 6 — Delivery formats that do not fight the codec": the
# release carries the adapter (a first-class artifact with its honest R8 label),
# the merged twin (the bytes the adapter curve is NOT), and the plan card a
# stranger replays on their own quant. `create_release` above stays the legacy
# HF-export shape and is untouched by this section.
#
# The plan card is distributed as the digest-pinned bundle (plan-card.tar, a
# `jdewitt plan-bundle`) AND its manifest beside it (plan-card.json), so
# `verify-artifact --all` can check the card without unpacking a tar and
# `plan-unbundle DIR/plan-card.tar` still works. Both are first-class files.
# ---------------------------------------------------------------------------

# Exactly what a phase-6 staging directory may contain before files.json exists.
V2_RELEASE_FILES = ('README.md', 'adapter.gguf', 'adapter.json', 'evaluation.json',
                    'model-merged.gguf', 'plan-card.json', 'plan-card.tar', 'twin.json')

NO_PREQUANTIZED_MODEL_RELEASE = 'NO_PREQUANTIZED_MODEL_RELEASE'
NON_GGUF_EXPORT_ROADMAP = 'NON_GGUF_EXPORT_ROADMAP'
MISSING_RELEASE_INPUT = 'MISSING_RELEASE_INPUT'
TWIN_HEADER_UNREADABLE = 'TWIN_HEADER_UNREADABLE'
ADAPTER_CARD_BASE_SHA_MISMATCH = 'ADAPTER_CARD_BASE_SHA_MISMATCH'
ADAPTER_ARTIFACT_SHA_MISMATCH = 'ADAPTER_ARTIFACT_SHA_MISMATCH'
TWIN_ARTIFACT_SHA_MISMATCH = 'TWIN_ARTIFACT_SHA_MISMATCH'
TWIN_TENSOR_COUNTS_DISAGREE = 'TWIN_TENSOR_COUNTS_DISAGREE'
TWIN_TENSOR_MISSING = 'TWIN_TENSOR_MISSING'
BEHAVIOUR_CLAIM_UNLABELLED = 'BEHAVIOUR_CLAIM_UNLABELLED'
PLAN_CARD_STRUCTURE = 'PLAN_CARD_STRUCTURE'
PLAN_CARD_PAYLOAD_MISMATCH = 'PLAN_CARD_PAYLOAD_MISMATCH'
UNEXPECTED_RELEASE_FILE = 'UNEXPECTED_RELEASE_FILE'

# Non-GGUF export stays a roadmap line until a fitted plan can be served and
# round-tripped through that runtime (docs/ROADMAP.md Phase 6, last bullet).
NON_GGUF_REASON = ('non-GGUF export is a roadmap line until a fitted plan can be served and '
                   'round-tripped through vLLM/SGLang')

# Where a pinned serve argv may live in an evaluation record. Nothing is invented:
# an absent argv is recorded as absent next to the behaviour numbers it belongs to.
_SERVE_ARGV_PATHS = ('serve_argv', 'argv', 'server.argv', 'behaviour.serve_argv',
                     'evaluation.serve_argv', 'merged_verification.serve_argv',
                     'winner.serve_argv', 'settings.serve_argv')


def _dig_path(document, path):
    current = document
    for part in path.split('.'):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _pinned_serve_argv(evaluation):
    """The serve argv an evaluation record pins, or None when it recorded none."""
    for path in _SERVE_ARGV_PATHS:
        value = _dig_path(evaluation, path)
        if isinstance(value, list) and value and all(isinstance(token, str) for token in value):
            return [str(token) for token in value]
    return None


def _gguf_twin_header(path):
    """Architecture and per-tensor codec type of the released bytes; header reads only."""
    try:
        reader = gguf.GGUFReader(str(path))
        field = reader.get_field('general.architecture')
        if field is None:
            raise ValueError('GGUF has no general.architecture')
        architecture = str(field.contents())
        tensors = {tensor.name: tensor.tensor_type.name for tensor in reader.tensors}
        del reader
    except Exception as error:
        raise ReleaseRefused(TWIN_HEADER_UNREADABLE,
                             f'{Path(path).name} is not a readable GGUF: {error}') from error
    if not tensors:
        raise ReleaseRefused(TWIN_HEADER_UNREADABLE, f'{Path(path).name} carries no tensors')
    return architecture, tensors


def _read_plan_card(path):
    """Verify a plan-bundle by digest and return (manifest, payload_sha256)."""
    import tarfile
    from .registry import BUNDLE_FORMAT
    path = Path(path)
    try:
        with tarfile.open(path, 'r') as tar:
            names = tar.getnames()
            if len(names) != 2 or set(names) != {f'{BUNDLE_FORMAT}/manifest.json',
                                                 f'{BUNDLE_FORMAT}/plan.pt'}:
                raise ReleaseRefused(
                    PLAN_CARD_STRUCTURE,
                    f'{path.name} members {names} do not match {BUNDLE_FORMAT}/manifest.json + plan.pt')
            manifest_member = tar.extractfile(f'{BUNDLE_FORMAT}/manifest.json')
            payload_member = tar.extractfile(f'{BUNDLE_FORMAT}/plan.pt')
            if manifest_member is None or payload_member is None:
                raise KeyError('bundle members are not regular files')
            manifest = json.loads(manifest_member.read())
            payload = payload_member.read()
    except (tarfile.TarError, KeyError, json.JSONDecodeError) as error:
        raise ReleaseRefused(PLAN_CARD_STRUCTURE,
                             f'{path.name} is not a readable plan-bundle: {error}') from error
    if not isinstance(manifest, dict) or manifest.get('format') != BUNDLE_FORMAT:
        found = manifest.get('format') if isinstance(manifest, dict) else None
        raise ReleaseRefused(PLAN_CARD_STRUCTURE,
                             f'{path.name} manifest format {found!r} is not {BUNDLE_FORMAT!r}')
    payload_sha = hashlib.sha256(payload).hexdigest()
    if manifest.get('payload_sha256') != payload_sha:
        raise ReleaseRefused(
            PLAN_CARD_PAYLOAD_MISMATCH,
            f'{path.name} manifest names payload {str(manifest.get("payload_sha256"))[:12]}… but the '
            f'bundle payload is {payload_sha[:12]}…; the card does not describe these bytes')
    meta = manifest.get('meta')
    if not isinstance(meta, dict):
        raise ReleaseRefused(PLAN_CARD_STRUCTURE, f'{path.name} manifest carries no signed meta')
    if meta.get('payload_sha256') != payload_sha:
        raise ReleaseRefused(
            PLAN_CARD_PAYLOAD_MISMATCH,
            f'{path.name} signed meta names payload {str(meta.get("payload_sha256"))[:12]}… but the '
            f'bundle payload is {payload_sha[:12]}…')
    if meta.get('meta_digest') != digest({key: value for key, value in meta.items()
                                          if key != 'meta_digest'}):
        raise ReleaseRefused(PLAN_CARD_STRUCTURE,
                             f'{path.name} signed meta fails its own meta_digest')
    return manifest, payload_sha


def _require_input(label, path):
    if path is None:
        raise ReleaseRefused(MISSING_RELEASE_INPUT, f'{label} is required for a phase-6 release')
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ReleaseRefused(MISSING_RELEASE_INPUT,
                             f'{label} {path} must be an existing regular file')
    return path


def build_twin_record(twin, architecture, header_types, card, evaluation, card_sha256,
                      plan_manifest, plan_payload_sha, plan_bundle_sha):
    """The codec-realized record of the shipped twin, labelled number by number.

    Every number is copied from a named upstream artifact or read from the released
    bytes here; this function measures no behaviour. cos/SNR are requant fidelity
    (CPU, full-tensor decode) and are labelled codec-realized; delta_br and the
    ranking verdict are behaviour claims, labelled with where they were measured and
    the pinned serve argv when the evaluation record carries one.
    """
    twin, header_types = Path(twin), dict(header_types)
    merged = card['merged_twin']
    delivered = merged['delivered']['tensors']
    fingerprint = merged.get('edit_fingerprint') or {}
    rows = []
    for row in delivered:
        name = row['name']
        if name not in header_types:
            raise ReleaseRefused(
                TWIN_TENSOR_MISSING,
                f'adapter card measures {name!r}, which the released twin does not carry')
        rows.append({'name': name, 'cos': row['cos'], 'snr': row.get('snr'),
                     'edited_type': header_types[name],
                     'edit': {'shape': _dig_path(fingerprint, 'shape'),
                              'scale': _dig_path(fingerprint, 'scale'),
                              'rank': _dig_path(fingerprint, 'rank')},
                     'measurement_class': 'codec-realized'})
    cosines = [row['cos'] for row in rows]
    snrs = [row['snr'] for row in rows if isinstance(row['snr'], (int, float))]
    argv = _pinned_serve_argv(evaluation)
    measured = card.get('merged_effect_status') == 'MEASURED'
    argv_note = ('recorded in evaluation.json and pinned here' if argv else
                 'evaluation.json records no serve argv, so none is claimed')
    behaviour = {
        'status': card.get('merged_effect_status'),
        'measurement_class': 'behaviour-claim',
        'measured_where': (
            f'merged bytes, reported by adapter.json as merged_measured_on='
            f'{card.get("merged_measured_on") or "merged-bytes"}; release-assemble does not '
            f'run generation, so these are behaviour claims, not codec-realized numbers'),
        'serve_argv': argv,
        'argv_recorded': argv is not None,
        'adapter_delta_br': card.get('adapter_delta_br'),
        'merged_delta_br': card.get('merged_delta_br'),
        'adapter_to_merged_ratio': card.get('adapter_to_merged_ratio'),
        'ranking_valid_on_this_artifact': card.get('ranking_valid_on_this_artifact'),
        'ranking_validity_ratio_floor': card.get('ranking_validity_ratio_floor'),
        'curve_is_not_merged_bytes': True,
        'curve_note': ('R8: the adapter curve is a screening number; merged_delta_br is the '
                       'edit\'s effect on the released bytes'),
    }
    if not measured and behaviour['merged_delta_br'] is not None:
        raise ReleaseRefused(
            BEHAVIOUR_CLAIM_UNLABELLED,
            'card reports a merged delta_br without a MEASURED status; refusing an unlabelled '
            'behaviour claim')
    return {
        'version': 1,
        'kind': 'merged-twin',
        'generated_by': 'jdewitt release-assemble',
        'artifact': {'file': 'model-merged.gguf', 'sha256': sha256_file(twin),
                     'bytes': twin.stat().st_size, 'arch': architecture},
        'base': {'sha256': card['base']['sha256'], 'arch': card['base'].get('arch'),
                 'tensor_count': card['base'].get('tensor_count')},
        'tensor_count': len(header_types),
        'edited_tensors': merged['edited_tensors'],
        'untouched_tensors': merged['untouched_tensors'],
        'edit_fingerprint': merged.get('edit_fingerprint') or {},
        'edit_type_per_tensor': {row['name']: row['edited_type'] for row in rows},
        'delivered': {'measurement_class': 'codec-realized',
                      'measured_where': ('CPU byte-level: cos/SNR are the requant-fidelity numbers '
                                         'the adapter card copied from a ggufio.compare_edit '
                                         'full-tensor decode that names this artifact by sha256; '
                                         'not a behaviour measurement'),
                      'tensors': rows, 'sampled': len(rows),
                      'min_cos': min(cosines) if cosines else None,
                      'min_snr': min(snrs) if snrs else None},
        'behaviour': behaviour,
        'provenance': {'adapter_file': 'adapter.gguf',
                       'adapter_sha256': card['adapter']['sha256'],
                       'adapter_card_file': 'adapter.json', 'adapter_card_sha256': card_sha256,
                       'adapter_card_twin_record_sha256': merged.get('record_sha256'),
                       'plan_card_file': 'plan-card.tar', 'plan_card_key': plan_manifest.get('key'),
                       'plan_card_payload_sha256': plan_payload_sha,
                       'plan_card_bundle_sha256': plan_bundle_sha},
        'measurement_ledger': [
            {'numbers': ['artifact.sha256', 'artifact.bytes', 'tensor_count', 'edited_tensors',
                         'untouched_tensors', 'delivered.tensors[].cos',
                         'delivered.tensors[].snr', 'delivered.tensors[].edited_type'],
             'class': 'codec-realized', 'basis': 'CPU byte-level',
             'where': ('released twin header + full-tensor decode of the shipped bytes; cos/SNR are '
                       'requant fidelity copied from the adapter card, which names this artifact '
                       'by sha256')},
            {'numbers': ['behaviour.adapter_delta_br', 'behaviour.merged_delta_br',
                         'behaviour.adapter_to_merged_ratio',
                         'behaviour.ranking_valid_on_this_artifact'],
             'class': 'behaviour-claim', 'basis': 'generation-scored',
             'where': f'{behaviour["measured_where"]}; serve argv: {argv_note}'}],
        'note': ('The adapter ranks, this twin ships. codec-realized numbers name these bytes by '
                 'sha256 and were measured on CPU; behaviour numbers are claims carried from '
                 'adapter.json and are marked as such.'),
    }


def assemble_release(directory, adapter, twin, plan, adapter_card, evaluation, readme):
    """Stage the phase-6 release: adapter + merged twin + plan card, with labels.

    All six arguments are file paths. The finished directory carries adapter.gguf,
    model-merged.gguf, plan-card.tar, plan-card.json, adapter.json (copied byte for
    byte), twin.json (built here, labelled number by number), evaluation.json,
    README.md and files.json. Copies go through `copy_independent`; files.json is
    written last; the staging directory is renamed only after `verify_release`
    passes, so a refusal never publishes a partial release.

    Refusals are named (jdewitt.errors.ReleaseRefused): a .gguf twin with no
    adapter and no plan card is a pre-quantized model release; a .safetensors twin
    is the non-GGUF roadmap line; an adapter card whose base digest does not bind
    the adapter, or which does not name the released twin, is refused; and a plan
    card whose manifest payload digest disagrees with the bundle payload is refused.
    """
    directory = Path(directory)
    twin_path = Path(twin)
    if not twin_path.is_file() or twin_path.is_symlink():
        raise ReleaseRefused(MISSING_RELEASE_INPUT, f'--twin {twin_path} must be a regular file')
    suffix = twin_path.suffix.lower()
    if suffix != '.gguf':
        raise ReleaseRefused(NON_GGUF_EXPORT_ROADMAP,
                             f'--twin {twin_path.name}: {NON_GGUF_REASON}')
    if adapter is None and plan is None:
        raise ReleaseRefused(
            NO_PREQUANTIZED_MODEL_RELEASE,
            f'--twin {twin_path.name} with no adapter and no plan card is a bare pre-quantized '
            f'model release; ship the adapter, the merged twin and the plan card together')
    adapter_path = _require_input('--adapter', adapter)
    plan_path = _require_input('--plan', plan)
    card_path = _require_input('--adapter-card', adapter_card)
    evaluation_path = _require_input('--evaluation', evaluation)
    readme_path = _require_input('--readme', readme)

    # Labels are checked against the shipped bytes before anything is staged.
    from .adapter_card import _adapter_identity
    from .schemas import validate_artifact
    adapter_info = _adapter_identity(adapter_path)
    card = json.loads(card_path.read_text())
    if not isinstance(card, dict):
        raise ReleaseRefused(ADAPTER_CARD_BASE_SHA_MISMATCH,
                             f'adapter card {card_path.name} is not a JSON object')
    base_sha = str((card.get('base') or {}).get('sha256') or '')
    bound_sha = str((card.get('adapter') or {}).get('bound_base_sha256') or '')
    if not base_sha or base_sha != bound_sha:
        raise ReleaseRefused(
            ADAPTER_CARD_BASE_SHA_MISMATCH,
            f'adapter card base sha256 {base_sha[:12] or "(missing)"}… does not equal the base the '
            f'adapter is bound to ({bound_sha[:12] or "(missing)"}…); the card would label bytes it '
            f'was not fitted on')
    card_check = validate_artifact(card_path, 'adapter.json')
    if not card_check['valid']:
        raise ReleaseRefused(card_check['problems'][0].split(':', 1)[0],
                             'adapter card fails its own R8 check: '
                             + '; '.join(card_check['problems'][:3]))
    if card.get('adapter', {}).get('sha256') != adapter_info['sha256']:
        raise ReleaseRefused(
            ADAPTER_ARTIFACT_SHA_MISMATCH,
            f'adapter card names adapter {str(card.get("adapter", {}).get("sha256"))[:12]}… but '
            f'--adapter is {adapter_info["sha256"][:12]}…')
    twin_sha = sha256_file(twin_path)
    if card['merged_twin']['artifact_sha256'].lower() != twin_sha:
        raise ReleaseRefused(
            TWIN_ARTIFACT_SHA_MISMATCH,
            f'adapter card names merged twin {card["merged_twin"]["artifact_sha256"][:12]}… but '
            f'--twin is {twin_sha[:12]}…; the codec-realized numbers were not measured on these bytes')

    architecture, header_types = _gguf_twin_header(twin_path)
    edited = card['merged_twin']['edited_tensors']
    untouched = card['merged_twin']['untouched_tensors']
    if edited + untouched != len(header_types):
        raise ReleaseRefused(
            TWIN_TENSOR_COUNTS_DISAGREE,
            f'adapter card counts {edited} edited + {untouched} untouched tensors but the released '
            f'twin carries {len(header_types)}; the card describes a different artifact')

    plan_manifest, plan_payload_sha = _read_plan_card(plan_path)
    plan_bundle_sha = sha256_file(plan_path)
    try:
        evaluation_document = json.loads(evaluation_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f'--evaluation {evaluation_path} is not valid JSON: {error}') from error
    if not isinstance(evaluation_document, dict):
        raise ValueError(f'--evaluation {evaluation_path} must be a JSON object')
    readme_text = readme_path.read_text()
    identity = digest({'evaluation': evaluation_document, 'readme': readme_text})

    with exclusive(directory.parent, '.' + directory.name + '.lock'):
        if directory.exists():
            return verify_release(directory, identity)
        stage = directory.with_name(directory.name + '-staging-' + uuid.uuid4().hex)
        stage.mkdir()
        copies = ((adapter_path, stage / 'adapter.gguf'),
                  (twin_path, stage / 'model-merged.gguf'),
                  (plan_path, stage / 'plan-card.tar'),
                  (card_path, stage / 'adapter.json'))
        for source, target in copies:
            copy_independent(source, target)
            if sha256_file(source) != sha256_file(target):
                raise ValueError(f'{source.name} changed during independent release copy')
        atomic_json(stage / 'evaluation.json', evaluation_document)
        atomic_text(stage / 'README.md', readme_text)
        atomic_json(stage / 'plan-card.json',
                    {**plan_manifest, 'bundle_file': 'plan-card.tar',
                     'bundle_sha256': plan_bundle_sha,
                     'note': 'manifest of plan-card.tar, shipped beside the bundle so '
                             'verify-artifact can check the card without unpacking it'})
        atomic_json(stage / 'twin.json',
                    build_twin_record(twin_path, architecture, header_types, card,
                                      evaluation_document, sha256_file(card_path), plan_manifest,
                                      plan_payload_sha, plan_bundle_sha))
        unexpected = sorted({path.name for path in stage.iterdir()} - set(V2_RELEASE_FILES))
        if unexpected:
            raise ReleaseRefused(
                UNEXPECTED_RELEASE_FILE,
                f'staging directory carries files that are not release artifacts: {unexpected}')
        missing = sorted(set(V2_RELEASE_FILES) - {path.name for path in stage.iterdir()})
        if missing:
            raise ReleaseRefused(MISSING_RELEASE_INPUT,
                                 f'staging directory is missing release files: {missing}')
        record = {'version': 1, 'release_identity': identity,
                  'files': release_inventory(stage)}
        record['record_digest'] = digest(record)
        atomic_json(stage / 'files.json', record)
        verify_release(stage, identity)
        sync_directory(stage)
        stage.rename(directory)
        sync_directory(directory.parent)
        add_claim(directory,
                  'Release assembled: adapter, merged twin, plan card and their labels bind these '
                  'exact bytes; codec-realized numbers are separated from behaviour claims',
                  'files.json release index + twin.json measurement ledger',
                  'jdewitt release-assemble')
        result = verify_release(directory, identity)
        result['plan_card_key'] = plan_manifest.get('key')
        result['plan_card_payload_sha256'] = plan_payload_sha
        result['twin_artifact_sha256'] = twin_sha
        result['adapter_sha256'] = adapter_info['sha256']
        return result
