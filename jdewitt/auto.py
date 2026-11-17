# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-model method selection. Only the globally selected candidate sees holdout."""
import json
import re
from dataclasses import dataclass,asdict
from importlib.metadata import version
from pathlib import Path
from .search import RunConfig
from .data import digest
from .answers import atomic_json,exclusive
from .storage import file_digest
from .validation import validate_run_inputs
from .errors import SearchRejected,UnsupportedMethod,HoldoutRejected,DirectionRejected


@dataclass(frozen=True)
class AutoConfig:
    run: RunConfig
    candidates: tuple
    total_trials: int

    @classmethod
    def from_json(cls,obj):
        if not isinstance(obj,dict) or set(obj)!={'run','candidates','total_trials'}:
            raise ValueError('Auto configuration requires run, candidates and total_trials')
        base=RunConfig.from_json(obj['run'])
        if type(obj['total_trials']) is not int or obj['total_trials']<1:
            raise ValueError('Explicit positive integer total_trials required')
        if not isinstance(obj['candidates'],list) or not obj['candidates']:
            raise ValueError('Explicit nonempty method candidates required')
        candidates=[]
        ids=set()
        for item in obj['candidates']:
            if not isinstance(item,dict) or set(item)!={'id','method','rank','method_options','trials'}:
                raise ValueError('Each candidate requires id, method, rank, method_options and trials')
            name=item['id']
            if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',name) or name in ids:
                raise ValueError('Candidate ids must be unique safe path segments')
            ids.add(name)
            candidate=json.loads(json.dumps(item,allow_nan=False))
            RunConfig.from_json({**asdict(base),**{k:v for k,v in candidate.items() if k!='id'},
                                 'output':str(Path(base.output)/'candidates'/name)})
            candidates.append(candidate)
        if sum(c['trials'] for c in candidates)!=obj['total_trials']:
            raise ValueError('Candidate trial allocations must sum exactly to total_trials')
        return cls(base,tuple(candidates),obj['total_trials'])

    def runs(self):
        return [RunConfig.from_json({**asdict(self.run),**{k:v for k,v in c.items() if k!='id'},
                                    'output':str(Path(self.run.output)/'candidates'/c['id'])}) for c in self.candidates]


def validate_auto(config):
    source=Path(config.run.model)
    if not source.is_dir():
        raise ValueError('Automatic method selection requires a reviewed local original-safetensors model')
    output=Path(config.run.output).resolve()
    if output.is_relative_to(source.resolve()) or source.resolve().is_relative_to(output):
        raise ValueError('Automatic experiment output must not overlap original model files')
    from .architectures import ARCHITECTURES
    architecture=json.loads((source/'config.json').read_text()).get('model_type')
    if architecture not in ARCHITECTURES or not list(source.glob('*.safetensors')):
        raise ValueError('Automatic selection requires a registered architecture and original safetensors')
    reports=[]
    for candidate in config.runs():
        _,report=validate_run_inputs(candidate)
        reports.append({'id':Path(candidate.output).name,'trials':candidate.trials,**report})
    for key in ('converter','quantizer','converter_python','llama_server','imatrix')+(
        ('judge_artifact','calibration') if config.run.scoring_mode=='judge' else ()):
        value=getattr(config.run,key)
        if value is not None and not Path(value).is_file():
            raise FileNotFoundError(f'Missing {key}: {value}')
    return {'status':'metadata-valid','trial_budget':config.total_trials,'candidates':reports,
            'holdout_policy':'Only the global winner is evaluated; no fallback after holdout failure.',
            'models_loaded':False,'output_created':False}


def experiment_identity(config):
    base=config.run
    inputs={key:file_digest(Path(getattr(base,key))) for key in ('directions','selection','holdout',
        'converter','quantizer','converter_python','llama_server')}
    for key in ('imatrix','calibration','judge_artifact','warm_starts'):
        if getattr(base,key) is not None:
            inputs[key]=file_digest(Path(getattr(base,key)))
    source=Path(base.model)
    inputs['source']={p.name:file_digest(p) for p in sorted(source.iterdir())
        if p.is_file() and not p.name.startswith('.') and p.suffix in {'.safetensors','.json','.jinja','.py','.model','.txt'}}
    code={p.name:file_digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    assets={p.name:file_digest(p) for p in sorted((Path(__file__).parent/'assets').iterdir()) if p.is_file()}
    return {'config':json.loads(json.dumps(asdict(config),allow_nan=False)),'inputs':inputs,'code':code,'assets':assets,
            'dependencies':{name:version(name) for name in ('torch','transformers','optuna','numpy','safetensors','gguf')}}


def evidence(directory):
    directory=Path(directory).resolve()
    result={}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError('Auto-selection evidence cannot contain symlinks')
        if path.is_file() and path.suffix in {'.json','.jsonl','.gguf','.safetensors','.pt'}:
            result[str(path.relative_to(directory))]={'sha256':file_digest(path),'bytes':path.stat().st_size}
    return result


def checked_record(path,directory):
    record=json.loads(Path(path).read_text())
    payload={k:v for k,v in record.items() if k!='record_digest'}
    if set(record)!={'payload','evidence','record_digest'} or digest(payload)!=record['record_digest']:
        raise ValueError('Auto-selection record integrity mismatch')
    root=Path(directory).resolve()
    for name,expected in record['evidence'].items():
        path=root/name
        if Path(name).is_absolute() or '..' in Path(name).parts or not path.resolve().is_relative_to(root):
            raise ValueError('Invalid auto-selection evidence path')
        if path.is_symlink() or not path.is_file() or expected!={'sha256':file_digest(path),'bytes':path.stat().st_size}:
            raise ValueError('Auto-selection evidence integrity changed')
    return record['payload']


def store_record(path,payload,directory):
    record={'payload':payload,'evidence':evidence(directory)}
    record['record_digest']=digest(record)
    atomic_json(path,record)


def run_auto(config,*,runner=None):
    validate_auto(config)
    identity=experiment_identity(config)
    root=Path(config.run.output).resolve()
    runs={Path(c.output).name:c for c in config.runs()}
    if runner is None:
        from .pipeline import run
        runner=run
    from .release import rank_quantized,verify_release
    with exclusive(root,'.auto.lock'):
        manifest=root/'auto-identity.json'
        if manifest.exists():
            if json.loads(manifest.read_text())!=identity:
                raise ValueError('Auto-selection identity changed; use a new output directory')
        else:
            if any(root.glob('candidate-*.json')) or (root/'auto-winner.json').exists():
                raise ValueError('Auto-selection records lack a bound identity')
            atomic_json(manifest,identity)
        completed=root/'auto-report.json'
        if completed.exists():
            report=checked_record(completed,root/'candidates')
            if report['status']=='holdout-failed':
                raise HoldoutRejected(report['reason'])
            verify_release(report['release'],report['release_identity'])
            return report
        winner_path=root/'auto-winner.json'
        if winner_path.exists():
            winner=checked_record(winner_path,root/'candidates')
        else:
            selected=[]
            baseline=None
            baseline_hashes=None
            outcomes=[]
            for name,child in runs.items():
                checkpoint=root/f'candidate-{name}.json'
                atomic_json(root/'auto-progress.json',{'stage':'selection','candidate':name,'trial_budget':config.total_trials})
                if checkpoint.exists():
                    outcome=checked_record(checkpoint,child.output)
                else:
                    try:
                        result=runner(child,selection_only=True)
                        if result.get('status')!='selection-complete':
                            raise ValueError('Method returned no completed selection result')
                        outcome={'id':name,'status':'selected','result':result}
                    except (UnsupportedMethod,SearchRejected,DirectionRejected) as error:
                        outcome={'id':name,'status':'incompatible' if isinstance(error,UnsupportedMethod) else 'rejected',
                                 'reason':str(error)}
                    store_record(checkpoint,outcome,child.output)
                outcomes.append({k:v for k,v in outcome.items() if k!='result'})
                if outcome['status']!='selected':
                    continue
                result=outcome['result']
                hashes={q:a['sha256'] for q,a in result['baseline_artifacts'].items()}
                if baseline is None:
                    baseline,baseline_hashes=result['baseline'],hashes
                elif baseline!=result['baseline'] or baseline_hashes!=hashes:
                    raise ValueError('Methods produced different quantized baselines; ranking is incomparable')
                for candidate in result['candidates']:
                    for artifact in candidate['artifacts'].values():
                        path=Path(artifact['path'])
                        if not path.resolve().is_relative_to(Path(child.output).resolve()) or file_digest(path)!=artifact['sha256']:
                            raise ValueError('Candidate artifact identity/path changed')
                    selected.append({**candidate,'id':name+':'+str(candidate['id']),
                                     'candidate_id':name,'trial_id':str(candidate['id']),'method':child.method,'rank':child.rank,
                                     'method_options':child.method_options or {}})
            if baseline is None:
                raise SearchRejected('No compatible method produced a feasible selection candidate')
            ranked=rank_quantized(selected,baseline,config.run.max_capability_drop,require_consistency=True)
            if not ranked:
                raise SearchRejected('No method passed the common quantized selection gates')
            winner={**ranked[0],'outcomes':outcomes,'ranked_ids':[c['id'] for c in ranked]}
            # Persist the choice BEFORE any holdout request. It cannot be replaced
            # by a runner-up after failure or interruption.
            store_record(winner_path,winner,root/'candidates')
        if experiment_identity(config)!=identity:
            raise ValueError('Auto-selection inputs changed while methods were executing')
        child=runs[winner['candidate_id']]
        atomic_json(root/'auto-progress.json',{'stage':'holdout','candidate':winner['candidate_id'],
                                             'trial':winner['trial_id'],'trial_budget':config.total_trials})
        try:
            result=runner(child,expected_winner=winner['trial_id'])
        except HoldoutRejected as error:
            report={'status':'holdout-failed','winner':winner,'reason':str(error),'trial_budget':config.total_trials}
            store_record(completed,report,root/'candidates')
            raise
        if str(result.get('winner'))!=winner['trial_id']:
            raise ValueError('Certification returned a different trial than the sealed winner')
        release=Path(result['release']).resolve()
        if release!=Path(child.output).resolve()/'release':
            raise ValueError('Certification returned an unexpected release directory')
        verified=verify_release(release)
        report={'status':'complete','winner':winner,'release':str(release),
                'release_identity':verified['release_identity'],'trial_budget':config.total_trials,
                'note':'Best measured candidate within the configured methods/budget, not a global optimum.'}
        store_record(completed,report,root/'candidates')
        atomic_json(root/'auto-progress.json',{'stage':'complete','candidate':winner['candidate_id'],'release':str(release)})
        return report
