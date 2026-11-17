# SPDX-License-Identifier: AGPL-3.0-or-later

import math
import hashlib
from dataclasses import dataclass, fields
from pathlib import Path
import optuna
from .data import digest
from .architectures import EDIT_TARGETS, MAX_STRENGTH

# Metrics the trial objective can already report; every one is minimized.
MO_METRICS = ('search_rate', 'full_reply_kl', 'answer_kl', 'capability_drop')


def pareto_front(points):
    """Indices of non-dominated points; every coordinate is a minimized metric."""
    out = []
    for i, p in enumerate(points):
        dominated = any(all(b <= a for a, b in zip(q, p)) and any(b < a for a, b in zip(q, p))
                        for j, q in enumerate(points) if j != i)
        if not dominated:
            out.append(i)
    return out


@dataclass(frozen=True)
class RunConfig:
    model: str
    revision: str
    model_license: str
    directions: str
    selection: str
    holdout: str

    max_tokens: int
    ctx_size: int
    trials: int
    startup_trials: int
    finalists: int
    rank: int
    seed: int
    max_full_reply_kl: float
    max_capability_drop: float
    vram_headroom_gib: float
    converter: str
    quantizer: str
    converter_python: str
    llama_server: str
    sandbox_image: str
    output: str
    dtype: str = 'bfloat16'
    quantization: str = 'none'
    trust_remote_code: bool = False
    warm_starts: str | None = None
    imatrix: str | None = None
    scoring_mode: str = 'judge'
    consistency: dict | None = None
    calibration: str | None = None
    judge_url: str | None = None
    judge_model: str | None = None
    judge_artifact: str | None = None
    judge_max_tokens: int | None = None
    judge_min_confidence: float | None = None
    judge_min_accuracy: float | None = None
    method: str = 'directional'
    method_options: dict | None = None
    refusal_floor: int | None = None
    edit_targets: str = 'writeback'
    strength_range: list | None = None
    generation_batch: dict | None = None
    merged_cache_bytes: int = 0
    screening: dict | None = None
    schema_version: int | None = None
    objectives: list | None = None
    warm_start_registry: str | None = None
    # Second-rank CUDA backend, opt-in per run. Default False keeps the B70-only
    # contract: without it a missing B70 is still a hard refusal.
    allow_cuda_secondary: bool = False
    # Optional short-decode stage-1 funnel: a heuristic prefilter of trials
    # before full-budget answer generation. None = classic single-stage eval.
    search_stages: dict | None = None
    # Optional pre-search probe: sample N strength sets, score them with the
    # stage-1 short-decode signal in one model load, enqueue the best few as
    # first trials. Requires search_stages. None = no pre-search probe.
    candidate_probe: dict | None = None

    @classmethod
    def from_json(cls,obj):
        declared=obj.get('schema_version')
        if declared is not None and declared!=1:
            raise ValueError(f'Configuration declares schema_version {declared!r}; this build supports only 1 and never migrates old experiments silently')
        try:
            config = cls(**obj)
        except TypeError as e:
            raise ValueError(f'Incomplete/unknown run configuration: {e}') from e
        for field in fields(cls):
            value=getattr(config,field.name)
            if (field.type is str or field.type==str|None and value is not None) and not isinstance(value,str):
                raise ValueError(f'{field.name} must be a string')
        if config.scoring_mode not in {'judge','refusal-check'}:
            raise ValueError('scoring_mode must be judge or refusal-check')
        if not isinstance(config.allow_cuda_secondary,bool):
            raise ValueError('allow_cuda_secondary must be a JSON boolean (true or false)')
        judge_fields={'calibration','judge_url','judge_model','judge_artifact','judge_max_tokens','judge_min_confidence','judge_min_accuracy'}
        optional={'warm_starts','imatrix','consistency','method_options','refusal_floor',
                  'strength_range','generation_batch','screening','schema_version','objectives','warm_start_registry','search_stages','candidate_probe'} | (judge_fields if config.scoring_mode=='refusal-check' else set())
        for field in fields(cls):
            value = getattr(config,field.name)
            if field.name not in optional and value is None:
                raise ValueError(f'Missing {field.name}')
            if isinstance(value,str) and not value.strip():
                raise ValueError(f'Empty {field.name}')
        positive=('max_tokens','ctx_size','trials','startup_trials','finalists','rank')+ (('judge_max_tokens',) if config.scoring_mode=='judge' else ())
        for name in positive:
            if type(getattr(config,name)) is not int or getattr(config,name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if config.ctx_size <= config.max_tokens or config.finalists > config.trials:
            raise ValueError('Context must exceed output budget; finalists cannot exceed trials')
        if config.refusal_floor is not None and (type(config.refusal_floor) is not int or config.refusal_floor <= 0 or config.refusal_floor > config.max_tokens):
            raise ValueError('refusal_floor must be a positive integer within the output budget')
        if config.edit_targets not in EDIT_TARGETS:
            raise ValueError(f'edit_targets must be one of {list(EDIT_TARGETS)}')
        if config.strength_range is not None:
            if (not isinstance(config.strength_range,list) or len(config.strength_range)!=2
                    or any(type(v) not in (int,float) or not math.isfinite(v) for v in config.strength_range)
                    or not 0 <= config.strength_range[0] < config.strength_range[1] <= MAX_STRENGTH):
                raise ValueError(f'strength_range must satisfy 0 <= low < high <= {MAX_STRENGTH}')
        if config.objectives is not None:
            if (not isinstance(config.objectives,list) or not 2 <= len(config.objectives) <= 4
                    or len(set(config.objectives)) != len(config.objectives)
                    or any(m not in MO_METRICS for m in config.objectives)):
                raise ValueError(f'objectives must be 2-4 unique names from {sorted(MO_METRICS)}; the first is the primary objective used by warm starts and display')
        if config.warm_start_registry is not None and not (isinstance(config.warm_start_registry,str) and config.warm_start_registry.strip()):
            raise ValueError('warm_start_registry must be a non-empty path to a plan registry directory')
        thresholds=('max_capability_drop',)+ (('judge_min_confidence','judge_min_accuracy') if config.scoring_mode=='judge' else ())
        for name in thresholds:
            v=getattr(config,name)
            if type(v) not in (int,float) or not math.isfinite(v) or not 0 <= v <= 1:
                raise ValueError(f'Invalid {name}')
        for name in ('max_full_reply_kl','vram_headroom_gib'):
            v=getattr(config,name)
            if type(v) not in (int,float) or not math.isfinite(v) or v <= 0:
                raise ValueError(f'Positive finite {name} required')
        if config.dtype not in {'float16','bfloat16'} or config.quantization not in {'none','nf4'} or type(config.trust_remote_code) is not bool:
            raise ValueError('Invalid dtype/remote-code policy')
        if config.consistency is not None:
            from .scoring import ConsistencyPolicy
            ConsistencyPolicy.from_json(config.consistency)
        from .methods import IMPLEMENTED,validate_method_options
        if config.method not in IMPLEMENTED:
            raise ValueError(f'Method {config.method!r} is not implemented; see b70-ablator methods')
        if config.method_options is not None and not isinstance(config.method_options,dict):
            raise ValueError('method_options must be an object')
        validate_method_options(config.method,config.method_options or {})
        from .batching import validate_batch_policy
        validate_batch_policy(config.generation_batch)
        from .stages import validate_search_stages
        validate_search_stages(config.search_stages,config.max_tokens)
        from .stages import validate_candidate_probe
        if config.candidate_probe is not None and config.search_stages is None:
            raise ValueError('candidate_probe requires search_stages: the probe reuses the stage-1 short-decode signal')
        validate_candidate_probe(config.candidate_probe,config.trials)
        if config.search_stages is not None and config.refusal_floor is not None and config.refusal_floor > config.search_stages['probe_tokens']:
            raise ValueError('search_stages.probe_tokens must reach at least refusal_floor: the short decode cannot satisfy the floor')
        if type(config.merged_cache_bytes) is not int or config.merged_cache_bytes<0:
            raise ValueError('merged_cache_bytes must be a nonnegative integer')
        from .screening import validate_screening
        validate_screening(config.screening)
        if config.method=='leace' and config.rank!=1:
            raise ValueError('Binary LEACE requires rank=1')
        if type(config.seed) is not int or not 0<=config.seed<2**32:
            raise ValueError('seed must be an integer in [0,2**32)')
        return config


def compatible_warm_starts(records, fingerprint):
    keys=('architecture','rank','dtype','quantization','edit_targets')
    defaults={'method':'directional','method_options_digest':digest({}),'edit_targets':'writeback'}
    return [r['parameters'] for r in records if r.get('feasible') is True and isinstance(r.get('parameters'),dict)
            and all(r.get('fingerprint',{}).get(k)==fingerprint.get(k) for k in keys)
            and all(r.get('fingerprint',{}).get(k,v)==fingerprint.get(k,v) for k,v in defaults.items())]


def parameter_bounds(strength_range=None):
    """The parameter space suggest_parameters explores; candidate probing samples
    the same space. One literal table shared by both callers so they cannot drift."""
    low, high = (0., 1.5) if strength_range is None else (strength_range[0], strength_range[1])
    return {'strength': (low, high), 'center': (0., 1.), 'width': (.05, 1.), 'floor': (0., 1.)}


def unflatten_parameters(flat):
    """Optuna parameter names (``attn.strength``) back to the nested shape the
    resident editor consumes ({'attn': {'strength': ...}})."""
    out = {}
    for key, value in flat.items():
        group, name = key.split('.', 1)
        out.setdefault(group, {})[name] = value
    return out


def sample_probe_params(groups, strength_range, layer_profile, samples, seed):
    """Deterministic uniform sampling of the suggestion space for candidate
    probing. random.Random(seed) is reproducible across processes and
    platforms; returns optuna-shaped flat dicts keyed like the parameter names.
    """
    import random
    from .methods import LAYER_PROFILES
    if layer_profile not in LAYER_PROFILES:
        raise ValueError(f'layer_profile must be one of {list(LAYER_PROFILES)}')
    bounds = parameter_bounds(strength_range)
    rng = random.Random(seed)
    out = []
    for _ in range(samples):
        flat = {}
        for group in groups:
            flat[f'{group}.strength'] = rng.uniform(*bounds['strength'])
            if layer_profile == 'searched':
                flat[f'{group}.center'] = rng.uniform(*bounds['center'])
                flat[f'{group}.width'] = rng.uniform(*bounds['width'])
                flat[f'{group}.floor'] = rng.uniform(*bounds['floor'])
        out.append(flat)
    return out


class RestartableTPE(optuna.samplers.BaseSampler):
    """Stateless per-trial/per-parameter RNG; persisted history supplies TPE data.

    No pickle or volatile random cursor. A crash between sampling and saving a
    parameter repeats exactly that suggestion, not a previous trial's RNG stream.
    This contract is for the exclusively locked, serial search only.
    """
    def __init__(self, seed, startup_trials):
        self.seed, self.startup_trials = seed, startup_trials

    def infer_relative_search_space(self, study, trial):
        return {}

    def sample_relative(self, study, trial, search_space):
        return {}

    def sample_independent(self, study, trial, param_name, param_distribution):
        key = f'{self.seed}:{trial.number}:{param_name}'.encode()
        seed = int.from_bytes(hashlib.sha256(key).digest()[:4], 'big')
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=self.startup_trials)
        return sampler.sample_independent(study, trial, param_name, param_distribution)


def build_study(path, provenance, seed, startup_trials, objectives=None):
    sampler = RestartableTPE(seed, startup_trials)
    if objectives is None:
        study = optuna.create_study(study_name='b70-ablator', storage='sqlite:///'+str(Path(path).resolve()), load_if_exists=True, direction='minimize', sampler=sampler)
        contract = {'algorithm':'keyed-tpe-v1', 'seed':seed, 'startup_trials':startup_trials, 'optuna':optuna.__version__}
    else:
        study = optuna.create_study(study_name='b70-ablator', storage='sqlite:///'+str(Path(path).resolve()), load_if_exists=True, directions=['minimize']*len(objectives), sampler=sampler)
        contract = {'algorithm':'keyed-mo-tpe-v1', 'seed':seed, 'startup_trials':startup_trials, 'optuna':optuna.__version__, 'objectives':list(objectives)}
    prior = study.user_attrs.get('sampler_contract')
    if prior != contract and (prior is not None or study.trials):
        raise ValueError('Sampler contract/version changed or legacy study lacks reproducible state; use a new study')
    identity = digest(provenance)
    existing = study.user_attrs.get('provenance_digest')
    if existing is not None and existing != identity:
        raise ValueError('Cannot resume a study with different model, suite, judge or settings')
    study.set_user_attr('sampler_contract',contract)
    study.set_user_attr('provenance_digest',identity)
    study.set_user_attr('provenance',provenance)
    return study


def execute_trials(study,objective,total,cancel_path=None,events_path=None):
    """Resume RUNNING attempts before asking for new ones; never spend budget twice."""
    from .errors import CancelRequested
    from .events import emit
    if type(total) is not int or total<1:
        raise ValueError('Positive trial execution budget required')
    attempted=[t for t in study.trials if t.state!=optuna.trial.TrialState.WAITING]
    if len(attempted)>total:
        raise ValueError('Execution budget is smaller than already attempted trials')
    pending=[optuna.Trial(study,t._trial_id) for t in attempted if t.state==optuna.trial.TrialState.RUNNING]
    while pending or len(attempted)<total:
        if cancel_path is not None and Path(cancel_path).exists():
            raise CancelRequested('run-cancel requested: stopped before asking for a new trial; unfinished attempts stay resumable')
        if pending:
            trial=pending.pop(0)
            if events_path is not None:
                emit(events_path,'trial-resume',number=trial.number)
        else:
            trial=study.ask()
            attempted.append(trial)
            if events_path is not None:
                emit(events_path,'trial-ask',number=trial.number)
        try:
            value=objective(trial)
        except optuna.TrialPruned as pruned:
            study.tell(trial,state=optuna.trial.TrialState.PRUNED)
            if events_path is not None:
                emit(events_path,'trial-pruned',number=trial.number,reason=str(pruned)[:200] or None)
        except BaseException as error:
            if events_path is not None:
                emit(events_path,'trial-failed',number=trial.number,error=str(error)[:400])
            raise
        else:
            study.tell(trial,value)
            if events_path is not None:
                emit(events_path,'trial-complete',number=trial.number,value=value)
        # Unexpected errors leave this exact attempt RUNNING for explicit resume.


def suggest_parameters(trial, groups=('attn','mlp'), strength_range=None, layer_profile='searched'):
    """One parameter block per edit group actually present in the editor.

    With the searched profile each group gets a layer-relative center/width/floor
    shape. With the discriminative profile the shape comes from measured per-layer
    signal instead, so only a single strength is suggested -- searching the shape
    parameters there would spend budget on values the operator ignores.
    """
    from .methods import LAYER_PROFILES
    if layer_profile not in LAYER_PROFILES:
        raise ValueError(f'layer_profile must be one of {list(LAYER_PROFILES)}')
    if strength_range is not None:
        if not isinstance(strength_range, (list, tuple)) or len(strength_range) != 2:
            raise ValueError('strength_range must be a two-element range')
    bounds = parameter_bounds(strength_range)
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds['strength']):
        raise ValueError('strength_range must be finite numbers')
    if not 0 <= bounds['strength'][0] < bounds['strength'][1] <= MAX_STRENGTH:
        raise ValueError(f'strength_range must satisfy 0 <= low < high <= {MAX_STRENGTH}')
    groups = list(groups)
    if not groups or any(not isinstance(g, str) or not g.strip() for g in groups):
        raise ValueError('At least one named edit group is required')
    params = {}
    for group in groups:
        params[group] = {'strength': trial.suggest_float(group+'.strength', *bounds['strength'])}
        if layer_profile == 'searched':
            # Layer-relative coordinates permit transfer between related depths.
            params[group].update({
                'center':trial.suggest_float(group+'.center',*bounds['center']),
                'width':trial.suggest_float(group+'.width',*bounds['width']),
                'floor':trial.suggest_float(group+'.floor',*bounds['floor']),
            })
    return params
