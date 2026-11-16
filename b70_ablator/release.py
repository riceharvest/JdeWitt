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


GPU_LOCK_PATH = Path(os.environ.get('B70_ABLATOR_GPU_LOCK', '/tmp/b70-ablator-gpu.lock'))


def acquire_gpu_lock(timeout):
    """Serialise B70 GPU jobs across processes; there is exactly one GPU."""
    lock_path = Path(GPU_LOCK_PATH)
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
                    f'B70 GPU is held by another b70-ablator job; waited {timeout}s '
                    '(override with B70_ABLATOR_GPU_WAIT or kill the holding job)')
            time.sleep(.5)


class ManagedGGUF:
    """Own one local Vulkan server. Refuse partial offload, changed artifacts or wrong models."""
    def __init__(self, artifact, server, ctx_size, max_tokens, log, startup_timeout=120, parallel=1, gpu_wait=None):
        if type(parallel) is not int or parallel<1:
            raise ValueError('Server parallel slot count must be a positive integer')
        self.parallel=parallel
        self.artifact = Path(artifact).resolve()
        self.server, self.ctx_size, self.max_tokens = str(server), ctx_size, max_tokens
        self.log, self.startup_timeout = Path(log), startup_timeout
        self.process = None
        wait = gpu_wait if gpu_wait is not None else float(os.environ.get('B70_ABLATOR_GPU_WAIT', '300'))
        self.gpu_wait = 0.0 if wait is None else float(wait)

    def command(self,device,port,alias):
        # llama.cpp divides the context across slots. Preserve ctx_size for
        # EACH request rather than silently shrinking the admitted context.
        return [self.server,'--model',str(self.artifact),'--alias',alias,'--device',device,
            '--gpu-layers','999','--split-mode','none',*load_mode_flags(self.server),
            '--ctx-size',str(self.ctx_size*self.parallel),'--parallel',str(self.parallel),
            '--host','127.0.0.1','--port',str(port),'--jinja','--reasoning-format','deepseek',
            '--no-webui','--verbosity','4','--cors-origins','localhost','--no-cors-credentials']

    def __enter__(self):
        self._gpu_lock = acquire_gpu_lock(self.gpu_wait)
        devices = subprocess.run([self.server,'--list-devices'], text=True,capture_output=True,check=True).stdout
        match = re.search(r'^\s*(Vulkan\d+):.*(?:B70|BMG G31)',devices,re.MULTILINE)
        if not match:
            raise RuntimeError('No verified B70 Vulkan device in llama-server')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0))
            port=sock.getsockname()[1]
        alias = 'b70-' + sha256_file(self.artifact)
        env = {k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_', 'HF_TOKEN'))}
        self.log.parent.mkdir(parents=True,exist_ok=True)
        self.log_file = self.log.open('w')
        cmd = self.command(match.group(1),port,alias)
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
            self.artifact_hash = alias[4:]
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


def add_claim(directory, claim, evidence, verified_by, release_identity=None):
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
        if p.is_absolute() or '..' in p.parts or str(p)!=name or not name or name=='evidence.json':
            raise ValueError('Invalid release index path')
        if set(metadata)!={'sha256','bytes'} or type(metadata['bytes']) is not int or metadata['bytes']<0:
            raise ValueError('Invalid release file metadata')
    if expected_identity is not None and record['release_identity']!=expected_identity:
        raise ValueError('Release identity mismatch')
    actual=release_inventory(directory)
    if {k:v for k,v in actual.items() if k!='evidence.json'}!=record['files']:
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
                  'files.json release index','b70-ablator create_release')
        return verify_release(directory,identity)
