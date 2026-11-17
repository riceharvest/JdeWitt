# SPDX-License-Identifier: AGPL-3.0-or-later
import gc
import json
import os
import shutil
import subprocess
from importlib.metadata import version
from contextlib import contextmanager

from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import optuna
import torch
from huggingface_hub import snapshot_download, hf_hub_download, model_info
from .data import digest
from .editor import ARCHITECTURES
from .evaluation import DominatedTrial
from .export import stream_merge, require_resident_precision, verify_resident_export
from .hf import HFModel, require_b70, resolve_backend, backend_device, device_memory_info, CUDA_SECONDARY_NOTE, reply_boundary, BoundaryContract, preflight_hf
from .plans import fit_plan, move_plan,direction_identity
from .release import export_gguf, rank_quantized, ManagedGGUF, sha256_file, verify_quant, create_release
from .scoring import summarize, extract_answer, search_rate, ConsistencyPolicy, paired_consistency
from .validation import validate_experiment as validate_experiment,validate_run_inputs
from .search import build_study, compatible_warm_starts, suggest_parameters, execute_trials, pareto_front
from .transport import ChatClient
from .grounded_judge import GroundedJudge, RUBRIC_SYSTEM, EVIDENCE_SYSTEM
from .style_judge import STYLE_SYSTEM
from .answers import AnswerStore, generate_answers, score_answers, completed_scores, exclusive, atomic_json
from .storage import save_checkpoint, load_checkpoint
from .phases import ModelPhases, PhasedHF, PhasedJudge
from .refusal_check import RefusalChecker
from .screening import validate_screening,margin_differential
from .stages import validate_search_stages,stratified_sample,stage1_score,stage1_pruned,validate_candidate_probe
from .errors import SearchRejected,UnsupportedMethod,HoldoutRejected


def cached_report(path,suite):
    path=Path(path)
    if not path.exists():
        return None
    obj=json.loads(path.read_text())
    predictions=path.parent/'predictions.jsonl'
    if obj['suite_digest']!=digest(suite) or not predictions.exists():
        raise ValueError('Cached report has different suite or missing predictions')
    rows=[json.loads(line) for line in predictions.read_text().splitlines()]
    calculated=summarize(rows,[r['id'] for r in suite])
    if calculated != obj['report']:
        raise ValueError('Cached report disagrees with actual predictions')
    return calculated


def configure_compiler_paths():
    # Process-local setup only, never mutate shell rc files or global oneAPI installs.
    root=Path.home()/'.local'
    if (root/'include/level_zero/ze_api.h').is_file():
        os.environ['CPATH']=str(root/'include')+(':'+os.environ['CPATH'] if os.environ.get('CPATH') else '')
    if (root/'lib/libze_loader.so').exists():
        os.environ['LIBRARY_PATH']=str(root/'lib')+(':'+os.environ['LIBRARY_PATH'] if os.environ.get('LIBRARY_PATH') else '')


def verify_judge(config):
    client=ChatClient(config.judge_url,config.judge_model,config.judge_max_tokens,ctx_size=config.ctx_size)
    client.verify()
    root=ChatClient(client.url.removesuffix('/v1'),config.judge_model,config.judge_max_tokens)
    props=root._request('/props')
    if Path(props['model_path']).resolve()!=Path(config.judge_artifact).resolve():
        raise ValueError('Judge endpoint does not serve the configured fixed GGUF')
    return GroundedJudge(client,config.judge_min_confidence,Path(config.output)/'judge-rubrics.json')


def eval_cached(suite,generator,judge,directory,sandbox_image,generation_identity=None,phases=None,scoring_identity=None,incumbent=None,refusal_floor=None,batch_policy=None):
    directory=Path(directory)
    if generation_identity is None:
        raise ValueError('Explicit generation identity required for resumable evaluation')
    if scoring_identity is None:
        if judge is not None or any(t['kind']=='refusal' for t in suite):
            raise ValueError('Explicit scorer/judge identity required for resumable evaluation')
        scoring_identity={'mode':'deterministic-tasks','scorer_sources':{n:sha256_file(Path(__file__).with_name(n)) for n in ('scoring.py','evaluation.py')}}
    store=AnswerStore(directory/'answers',suite,generation_identity)
    found=completed_scores(suite,store,directory,scoring_identity,sandbox_image)
    if found is not None:
        return found
    generate_answers(suite,generator,store,refusal_floor,batch_policy=batch_policy)
    if phases is not None and getattr(judge,'requires_gpu',True):
        phases.idle()
    return score_answers(suite,store,judge,directory,scoring_identity,sandbox_image,incumbent=incumbent)


def run(config,*,selection_only=False,expected_winner=None):
    """Lifecycle-wrapped pipeline run; events.jsonl is append-only telemetry."""
    from .events import emit
    events=Path(config.output)/'events.jsonl'
    emit(events,'run-start',model=str(config.model),method=getattr(config,'method','?'),
         trials=getattr(config,'trials','?'),selection_only=selection_only)
    try:
        result=_run_body(config,selection_only=selection_only,expected_winner=expected_winner)
    except BaseException as error:
        emit(events,'run-failed',error=str(error)[:500])
        raise
    emit(events,'run-complete')
    return result


def _run_body(config,*,selection_only=False,expected_winner=None):
    if getattr(config,'consistency',None) is None:
        raise ValueError('Explicit consistency policy required; legacy run settings must be reviewed')
    ConsistencyPolicy.from_json(config.consistency)
    if config.quantization!='none':
        raise ValueError('Resident export-consistent search requires original-precision weights; NF4 is not export-equivalent')
    source=Path(config.model)
    config_path=source/'config.json' if source.is_dir() else Path(hf_hub_download(config.model,'config.json',revision=config.revision))
    model_type=json.loads(config_path.read_text()).get('model_type')
    if model_type not in ARCHITECTURES:
        raise ValueError(f'Unsupported model architecture {model_type!r}; no judge or model weights were loaded')
    BoundaryContract.for_architecture(model_type)
    mode=getattr(config,'scoring_mode','judge')
    if mode=='judge' and config.judge_url != 'managed':
        raise ValueError('Automatic search requires judge_url="managed" so GPU models can be unloaded safely')
    @contextmanager
    def judge_factory():
        with ManagedGGUF(config.judge_artifact, config.llama_server, config.ctx_size,
                         config.judge_max_tokens, Path(config.output)/'judge-server.log',
                         secondary=config.allow_cuda_secondary) as client:
            yield verify_judge(replace(config,judge_url=client.url,judge_model=client.model))
    with exclusive(config.output,'.run.lock'), ModelPhases(judge_factory=judge_factory if mode=='judge' else None) as phases:
        if selection_only or expected_winner is not None:
            return _run(config,phases,selection_only=selection_only,expected_winner=expected_winner)
        return _run(config,phases)


def _warm_start_directions(config,backend,provenance,checkpoint_identity,direction_file,output):
    """Transfer a registered fitted plan into this run when weight geometry proves compatible.

    Geometry must match on every checkpoint-fingerprint key except model and
    revision (same family, different checkpoint); the suite digest, method and
    options must match exactly. Ineligible or corrupt candidates are recorded
    and skipped; the run then fits from scratch. The transferred plan is
    saved under this run's identity so trial provenance stays uniform.
    """
    from .registry import GEOMETRY_KEYS, RegistryError, load_plan, warm_start_candidates
    geometry={k:v for k,v in backend.fingerprint.items() if k in GEOMETRY_KEYS}
    notes=[]
    for cand in warm_start_candidates(config.warm_start_registry,config.method,config.method_options,
                                      provenance['directions_digest'],geometry=geometry,same_model=config.model):
        if not cand['eligible']:
            notes.append({'key':cand['key'],'skipped':cand['reason']})
            continue
        try:
            loaded=load_plan(config.warm_start_registry,cand['key'],expect={'rank':config.rank},device='cpu')
        except (RegistryError,FileNotFoundError) as e:
            notes.append({'key':cand['key'],'skipped':str(e)})
            continue
        directions=move_plan(loaded['plan'],backend.device)
        audit={'warm_start':{'key':cand['key'],'model':cand['model'],'revision':cand['revision'],
                             'measured':cand['measured'],'source_fit':loaded['audit']},
               'skipped_candidates':notes}
        save_checkpoint(direction_file,move_plan(directions,'cpu'),checkpoint_identity,
            {'warm_start':audit['warm_start'],'generation_fingerprint':backend.fingerprint})
        atomic_json(output/'direction-stability.json',audit)
        return directions,audit
    atomic_json(output/'direction-stability.json',{'warm_start':{'attempted':True,'candidates':notes}})
    return None,{}


def _run(config,phases,*,selection_only=False,expected_winner=None):
    configure_compiler_paths()
    output=Path(config.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    suites,_=validate_run_inputs(config)
    direction_tasks,selection,holdout=(suites[name] for name in ('directions','selection','holdout'))
    policy=ConsistencyPolicy.from_json(config.consistency)
    budget_report=preflight_hf(config.model,config.revision,{'directions':direction_tasks,'selection':selection,'holdout':holdout},
                              config.ctx_size,config.max_tokens,config.trust_remote_code)
    atomic_json(output/'prompt-budgets.json',budget_report)
    require_b70(config.allow_cuda_secondary)
    for tool in (config.converter,config.quantizer,config.converter_python,config.llama_server):
        if not Path(tool).is_file():
            raise FileNotFoundError(tool)
    calibration=[]
    judge_hash=None
    if config.scoring_mode=='judge':
        calibration=json.loads(Path(config.calibration).read_text())
        if not {r['language'] for r in selection+holdout} <= {r['language'] for r in calibration}:
            raise ValueError('Judge calibration does not cover evaluation languages')
        binder=GroundedJudge(None,config.judge_min_confidence,output/'judge-rubrics.json')
        judge=PhasedJudge(phases,binder)
        judge.bind_rubrics(direction_tasks+[r for r in selection+holdout if r['kind']=='refusal'])
        judge_hash=sha256_file(config.judge_artifact)
        from .job_cli import calibrate_cached
        check_identity={'judge_sha256':judge_hash,'server_sha256':sha256_file(config.llama_server),
                        'max_tokens':config.judge_max_tokens,'ctx_size':config.ctx_size,
                        'min_confidence':config.judge_min_confidence,
                        'code':{p.name:sha256_file(p) for p in Path(__file__).parent.glob('*.py')},
                        'prompts':digest({'rubric':RUBRIC_SYSTEM,'evidence':EVIDENCE_SYSTEM,'style':STYLE_SYSTEM})}
        check_args=SimpleNamespace(output=output/'judge-checks',judge_min_accuracy=config.judge_min_accuracy)
        calibrated=calibrate_cached(judge,calibration,check_args,check_identity)
        atomic_json(output/'judge-calibration.json',calibrated)
        phases.idle()
    else:
        judge=RefusalChecker()
    sandbox=subprocess.run(['docker','image','inspect',config.sandbox_image,'--format','{{.Id}}'],text=True,capture_output=True,check=True).stdout.strip()
    source=Path(config.model)
    resolved_revision=config.revision
    if not source.is_dir():
        info=model_info(config.model,revision=config.revision,files_metadata=True)
        resolved_revision=info.sha
        raw_config=json.loads(Path(hf_hub_download(config.model,'config.json',revision=resolved_revision)).read_text())
        if raw_config['model_type'] not in ARCHITECTURES:
            raise ValueError('Unsupported model architecture; weights were not downloaded')
        remote_bytes=sum(s.size or 0 for s in info.siblings if s.rfilename.endswith('.safetensors'))
        if not remote_bytes or shutil.disk_usage(output).free < remote_bytes*(1.7+config.finalists*1.7+1.1):
            raise RuntimeError('Insufficient disk for download, merged finalists and quantization')
        source=Path(snapshot_download(config.model,revision=resolved_revision,ignore_patterns=['*.bin','*.pt','*.pth','*.gguf','*.h5','*.msgpack','*.onnx']))
    model_config=json.loads((source/'config.json').read_text())
    if model_config['model_type'] not in ARCHITECTURES:
        raise ValueError('Unsupported model architecture; add and verify an adapter first')
    weights=sorted(source.glob('*.safetensors'))
    if not weights:
        raise ValueError('Original safetensors are required for streaming export')
    raw_bytes=sum(p.stat().st_size for p in weights)
    # Baseline quants + finalist merged weights/quants + one transient BF16 GGUF.
    needed=raw_bytes*(.7+config.finalists*1.7+1.1)
    if shutil.disk_usage(output).free<needed:
        raise RuntimeError(f'Insufficient disk: need approximately {int(needed)} free bytes for this finalist budget')
    compute_backend=resolve_backend(config.allow_cuda_secondary)
    device=backend_device(compute_backend)
    free,total=device_memory_info(device)
    approximate_weights=raw_bytes if config.quantization=='none' else raw_bytes*.35
    if free < approximate_weights+config.vram_headroom_gib*1024**3:
        raise RuntimeError(f'Model plus configured VRAM headroom will not fit the free {compute_backend["name"]} memory')
    provenance={
        'config':{k:v for k,v in asdict(config).items() if k not in {'trials','judge_url'}},
        'source_files':{p.name:sha256_file(p) for p in source.iterdir() if p.is_file() and p.suffix in {'.safetensors','.json','.jinja','.py'}},
        'resolved_revision':resolved_revision,
        'directions_digest':digest(direction_tasks),'selection_digest':digest(selection),'holdout_digest':digest(holdout),
        'judge_sha256':judge_hash,'judge_prompt_digest':digest({'rubric':RUBRIC_SYSTEM,'evidence':EVIDENCE_SYSTEM,'style':STYLE_SYSTEM}),'calibration_digest':digest(calibration),
        'sandbox_image_id':sandbox,'torch':torch.__version__,
        'dependency_versions':{name:_dependency_version(name) for name in ('transformers','optuna','numpy','accelerate','bitsandbytes','safetensors','gguf')},
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'prompt_budget_digest':digest(budget_report),
        'framework_sources':{p.name:sha256_file(p) for p in Path(__file__).parent.glob('*.py')},
        'converter_sha256':sha256_file(config.converter),'quantizer_sha256':sha256_file(config.quantizer),'server_sha256':sha256_file(config.llama_server),
    }
    if compute_backend['rank'] != 'primary':
        # Recorded only off the B70: provenance is digested and compared on resume, so an
        # always-present key would make every study created before this change unresumable.
        provenance['compute_backend'] = {'type':compute_backend['type'],'name':compute_backend['name'],
                                         'rank':compute_backend['rank'],'note':CUDA_SECONDARY_NOTE}
    study=build_study(output/'study.db',provenance,config.seed,config.startup_trials,getattr(config,'objectives',None))
    atomic_json(output/'provenance.json',provenance)
    torch.manual_seed(config.seed)
    phases.target_factory=lambda:HFModel(str(source),None,config.rank,config.max_tokens,config.dtype,config.trust_remote_code,config.quantization,'merged-resident',config.ctx_size,config.edit_targets,config.merged_cache_bytes,allow_secondary=config.allow_cuda_secondary)
    backend=PhasedHF(phases)
    try:
        fingerprint={**backend.fingerprint,'method':config.method,'method_options_digest':digest(config.method_options or {})}
        if config.edit_targets != backend.editor.edit_targets:
            raise ValueError('Configured edit surface differs from the loaded editor')
        if config.method=='ara' and any(a.base.weight.ndim!=2 for *_,a in backend.editor.targets):
            raise UnsupportedMethod('ARA fused-expert route-I/O is not implemented; no experts were skipped')
        workspace=require_resident_precision(source,backend.editor)
        if device_memory_info(device)[0] < workspace + config.merged_cache_bytes + config.vram_headroom_gib*1024**3:
            raise RuntimeError(f'Insufficient {compute_backend["name"]} headroom for the resident merged projection workspace')
    except BaseException:
        backend.release()
        raise
    generation_identity={'source_files':provenance['source_files'],'backend':fingerprint,
                         'generation_batch':config.generation_batch,
                         'max_tokens':config.max_tokens,'seed':config.seed,'refusal_floor':config.refusal_floor,
                         'hf_code_sha256':sha256_file(Path(__file__).with_name('hf.py'))}
    scoring_identity={'scoring_mode':config.scoring_mode,'judge_sha256':judge_hash,'judge_prompt_digest':provenance['judge_prompt_digest'],
                      'framework_sources':provenance['framework_sources'],'sandbox_image_id':sandbox,
                      'judge_max_tokens':config.judge_max_tokens,'ctx_size':config.ctx_size,
                      'min_confidence':config.judge_min_confidence}
    try:
        baseline=eval_cached(selection,backend,judge,output/'baseline-hf',config.sandbox_image,generation_identity,
                             phases,scoring_identity,refusal_floor=config.refusal_floor,batch_policy=config.generation_batch)
        if baseline['invalid']:
            raise ValueError(f"Invalid baseline outputs/judgments ({_invalid_row_detail(output/'baseline-hf')}): adjust budgets or judge, not model weights")
        direction_file=output/'directions.pt'
        checkpoint_identity=direction_identity(generation_identity,provenance['directions_digest'],config.method,
            config.method_options or {},scoring_identity,config.consistency,config.seed)
        saved=load_checkpoint(direction_file,checkpoint_identity)
        directions=direction_audit=None
        if saved is not None:
            directions,direction_audit=saved
            directions=move_plan(directions,backend.device)
        elif getattr(config,'warm_start_registry',None):
            directions,direction_audit=_warm_start_directions(config,backend,provenance,
                checkpoint_identity,direction_file,output)
        if directions is None:
            captures=[]
            captured_io={}
            capture=(getattr(config,'method_options',None) or {}).get('capture','last')
            # Capture all small activation records before loading the judge.
            for task in direction_tasks:
                response=backend.generate(task['prompt'])
                answer=extract_answer(response['content'],response['finish_reason'],response['thinking_open'],response.get('final_marker'))
                if answer.complete:
                    residual=(backend.residual_mean_at_answer(task['prompt'],response)
                              if capture=='answer-mean' else backend.residual_at_answer(task['prompt'],response))
                    captures.append((task,answer.text,residual))
                    if getattr(config,'method','directional')=='ara':
                        captured_io[task['id']]=backend.module_io_at_answer(task['prompt'],response)
            if getattr(judge,'requires_gpu',True):
                phases.idle()
            accepted=[]
            for task,text,residual in captures:
                verdict=judge.classify(task['prompt'],text,task['language']).verdict
                good_verdict='no_explicit_refusal' if config.scoring_mode=='refusal-check' else 'compliance'
                if task['label']=='bad' and verdict not in {'refusal','partial'} or task['label']=='good' and verdict!=good_verdict:
                    continue
                accepted.append((task,residual.cpu()))
            # Small activation statistics may use CPU SVD; model weight inference/editing never does.
            method=getattr(config,'method','directional')
            options=getattr(config,'method_options',None) or {}
            directions,direction_audit=fit_plan(accepted,config.rank,policy,config.seed,direction_tasks,method,options,
                [(task,captured_io[task['id']]) for task,_ in accepted] if method=='ara' else None,
                backend.editor if method=='ara' else None)
            directions=move_plan(directions,backend.device)
            direction_audit.update({'attempted':len(direction_tasks),'accepted':len(accepted)})
            save_checkpoint(direction_file,move_plan(directions,'cpu'),checkpoint_identity,
                dict(direction_audit,generation_fingerprint=fingerprint))
            atomic_json(output/'direction-stability.json',direction_audit)
        # Bind generated trial answers to the exact committed learned operator,
        # not merely rank/strength. Same-shaped plans from other methods differ.
        edit_identity={'derivation':checkpoint_identity,
                       'plan_sha256':json.loads(direction_file.with_suffix('.checkpoint.json').read_text())['sha256']}
        refs_file=output/'harmless-trajectories.pt'
        refs_identity={'generation':generation_identity,'selection_digest':provenance['selection_digest']}
        saved=load_checkpoint(refs_file,refs_identity)
        if saved is not None:
            references,_=saved
        else:
            references=[]
            for task in selection:
                if task['kind']!='harmless':
                    continue
                response=backend.generate(task['prompt'])
                answer=extract_answer(response['content'],response['finish_reason'],response['thinking_open'],response.get('final_marker'))
                if not answer.complete:
                    raise ValueError('Harmless baseline did not reach a complete final answer')
                boundary=reply_boundary(backend.tokenizer,response['generated_ids'],response['thinking_open'],response.get('final_marker'))
                references.append({'id':task['id'],'sequence':response['sequence'].cpu(),'prompt_length':response['prompt_length'],'answer_start':boundary})
            save_checkpoint(refs_file,references,refs_identity,{})
        screening = validate_screening(getattr(config,'screening',None))
        bad_prompts = [t['prompt'] for t in direction_tasks if t['label']=='bad']
        harmless_prompts = [t['prompt'] for t in selection if t['kind']=='harmless']
        if screening is not None:
            if not bad_prompts or not harmless_prompts:
                raise ValueError('Screening needs bad-direction tasks and a harmless selection slice')
            screen_identity = {'generation':generation_identity,'screening':screening,
                               'directions_digest':provenance['directions_digest'],
                               'selection_digest':provenance['selection_digest']}
            saved = load_checkpoint(output/'screen-baselines.pt',screen_identity)
            if saved is not None:
                screen_baseline = saved[0]
            else:
                screen_baseline = {'bad':backend.answer_start_margins(bad_prompts,screening['refusal_tokens'],screening['reference_tokens']),
                                   'harmless':backend.answer_start_margins(harmless_prompts,screening['refusal_tokens'],screening['reference_tokens'])}
                save_checkpoint(output/'screen-baselines.pt',screen_baseline,screen_identity,{})
        stages = validate_search_stages(getattr(config,'search_stages',None), config.max_tokens)
        stage_rows = stage_checker = None
        if stages is not None:
            stage_rows = stratified_sample([t for t in selection if t['kind']=='refusal'], stages['probe_fraction'])
            if not stage_rows:
                raise ValueError('search_stages needs refusal rows in the selection suite')
            stage_checker = RefusalChecker()
            from .batching import plan_batches
            def probe_decode():
                """Short-decode the probe slice under the run's own batch admission
                policy and the same refusal-floor semantics trial answers use —
                the probe never bypasses max_sequences/max_padded_tokens."""
                out = {}
                for batch, batch_floor in plan_batches(stage_rows, backend, config.refusal_floor,
                        config.generation_batch, output_budget=stages['probe_tokens']):
                    responses = backend.generate_batch([t['prompt'] for t in batch],
                        min_tokens=batch_floor, max_tokens=stages['probe_tokens'])
                    if len(responses) != len(batch):
                        raise ValueError('Probe batch returned the wrong cardinality')
                    for task, response in zip(batch, responses):
                        if batch_floor is not None and response.get('floor_enforced') is not True:
                            raise ValueError('Probe backend did not confirm the refusal floor')
                        out[task['id']] = response
                return [out[task['id']] for task in stage_rows]
            atomic_json(output/'search-stages.json',{'stages':stages,'probe_rows':[t['id'] for t in stage_rows],
                'batch_policy':config.generation_batch,'refusal_floor':config.refusal_floor,
                'note':'Stage-1 probe answers are heuristic wording verdicts on truncated short decodes; they never enter the answer store, the gates or any release report.'})
        if config.warm_starts:
            warm=json.loads(Path(config.warm_starts).read_text())
            for params in compatible_warm_starts(warm,fingerprint):
                study.enqueue_trial(params,skip_if_exists=True)
        probe = validate_candidate_probe(getattr(config,'candidate_probe',None), config.trials)
        if probe is not None:
            if stages is None:
                raise ValueError('candidate_probe requires search_stages: the probe reuses the stage-1 short-decode signal')
            from .events import emit
            from .search import sample_probe_params, unflatten_parameters
            layer_profile=(config.method_options or {}).get('layer_profile','searched')
            scored=[]
            for flat in sample_probe_params(backend.editor.groups, config.strength_range,
                                            layer_profile, probe['samples'], config.seed):
                # One model load for every probe: apply, short-decode the
                # harmful slice, word-level verdicts, reset. No judge, no
                # answer store, no KL/capability claim — see the artifact note.
                backend.editor.apply(directions, unflatten_parameters(flat))
                try:
                    responses = probe_decode()
                    refused = 0
                    for task, response in zip(stage_rows, responses):
                        answer = extract_answer(response['content'], response['finish_reason'], response['thinking_open'], response.get('final_marker'))
                        if stage_checker.classify(task['prompt'], answer.text, task['language']).verdict in {'refusal', 'partial'}:
                            refused += 1
                finally:
                    backend.editor.reset()
                scored.append({'parameters':flat,'score':stage1_score(refused,len(stage_rows))})
            scored.sort(key=lambda c:(c['score'], json.dumps(c['parameters'],sort_keys=True)))
            atomic_json(output/'candidate-probe.json',{'probe':stages,'probe_of':probe,
                'layer_profile':layer_profile,'probe_rows':[t['id'] for t in stage_rows],
                'ranking':scored,
                'note':'Stage-1 heuristic scores on truncated short decodes: KL, capability and semantic judging are NOT probed; every enqueued trial still pays every ordinary gate in full.'})
            for c in scored[:probe['enqueue_top']]:
                study.enqueue_trial(c['parameters'], skip_if_exists=True)
            emit(output/'events.jsonl','candidate-probe',samples=probe['samples'],
                 enqueued=min(probe['enqueue_top'],len(scored)),best=scored[0]['score'],worst=scored[-1]['score'])
        def objective(trial):
            params=suggest_parameters(trial,backend.editor.groups,config.strength_range,
                                      (config.method_options or {}).get('layer_profile','searched'))
            backend.editor.apply(directions,params)
            trial.set_user_attr('parameters',params)
            try:
                kl_sum=0.
                answer_sum=0.
                token_count=0
                answer_count=0
                for ref in references:
                    metric=backend.trajectory_kl(ref['sequence'].to(backend.device),ref['prompt_length'],ref['answer_start'])
                    kl_sum+=metric['full_reply_kl']*metric['tokens']
                    token_count+=metric['tokens']
                    answer_sum+=metric['answer_kl']*metric['answer_tokens']
                    answer_count+=metric['answer_tokens']
                    # KL is nonnegative. This lower bound permits only certain rejection.
                    final_tokens=sum(r['sequence'].shape[1]-r['prompt_length'] for r in references)
                    if kl_sum/final_tokens>config.max_full_reply_kl:
                        raise optuna.TrialPruned('Full-reply KL cannot meet the damage limit')
                kl=kl_sum/token_count
                if screening is not None:
                    # Forward-only refusal-margin screen; survivors still pay for
                    # full generation and judging. Screening never certifies removal.
                    statistic = margin_differential(
                        screen_baseline['bad'],
                        backend.answer_start_margins(bad_prompts,screening['refusal_tokens'],screening['reference_tokens']),
                        screen_baseline['harmless'],
                        backend.answer_start_margins(harmless_prompts,screening['refusal_tokens'],screening['reference_tokens']))
                    passed = statistic >= screening['min_margin_drop']
                    trial.set_user_attr('screening',{'status':'passed' if passed else 'rejected',
                                                     'statistic':statistic,'threshold':screening['min_margin_drop']})
                    if not passed:
                        raise optuna.TrialPruned('Screened out: refusal-margin change below threshold; no answers were generated')
                if stages is not None:
                    # Short-decode stage 1: generate probe_tokens on a stratified
                    # harmful slice, word-level refusal verdicts only. Heuristic
                    # prefilter, never a certification; probe answers never enter
                    # the answer store. Trials that reach stage 2 pay for their
                    # own full-budget generation, so total spend falls with the
                    # keep line, not with luck.
                    from .events import emit
                    stored = trial.user_attrs.get('stage1')
                    if stored is None:
                        responses = probe_decode()
                        refused = 0
                        rows_report = []
                        for task, response in zip(stage_rows, responses):
                            answer = extract_answer(response['content'], response['finish_reason'], response['thinking_open'], response.get('final_marker'))
                            verdict = stage_checker.classify(task['prompt'], answer.text, task['language']).verdict
                            if verdict in {'refusal', 'partial'}:
                                refused += 1
                            rows_report.append({'id': task['id'], 'verdict': verdict, 'finish_reason': response['finish_reason']})
                        stored = {'score': stage1_score(refused, len(stage_rows)), 'refused': refused,
                                  'rows': len(stage_rows), 'probe_tokens': stages['probe_tokens']}
                        atomic_json(output / f'trial-{trial.number}' / 'stage1.json', dict(stored, rows=rows_report))
                        trial.set_user_attr('stage1', stored)
                        emit(output / 'events.jsonl', 'trial-stage1', number=trial.number,
                             score=stored['score'], rows=stored['rows'], probe_tokens=stages['probe_tokens'])
                    # Keep line over trials that actually paid for stage 2.
                    prior = [t.user_attrs['stage1']['score'] for t in study.trials
                             if t.user_attrs.get('stage1') is not None
                             and (t.user_attrs.get('report') is not None or t.state == optuna.trial.TrialState.COMPLETE)]
                    if stage1_pruned(stored['score'], prior, stages['keep_top']):
                        raise optuna.TrialPruned('Stage-1 probe refusal rate worse than the keep line; no full-budget answers were generated')
                incumbent=min([v for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE and t.user_attrs.get('feasible') and (v:=t.values[0] if config.objectives else t.value) is not None],default=None)
                report=eval_cached(selection,backend,judge,output/f'trial-{trial.number}',config.sandbox_image,
                                   {**generation_identity,'edit_identity':edit_identity,'edit_parameters':params},phases,scoring_identity,incumbent,
                                   refusal_floor=config.refusal_floor,batch_policy=config.generation_batch)
                drop=max(0.,baseline['capability_pass_rate']-report['capability_pass_rate'])
                drop=max(drop,*(max(0.,baseline['by_kind'][kind]['pass_rate']-report['by_kind'][kind]['pass_rate']) for kind in ('code','math')))
                feasible=not report['invalid'] and drop<=config.max_capability_drop and kl<=config.max_full_reply_kl
                report['paired']=paired_consistency(selection,read_predictions(output/'baseline-hf'),read_predictions(output/f'trial-{trial.number}'),policy)
                feasible=feasible and report['paired']['regression_ok']
                report.update({'full_reply_kl':kl,'answer_kl':answer_sum/answer_count,'capability_drop':drop})
                metrics={'search_rate':search_rate(report),'full_reply_kl':kl,
                         'answer_kl':answer_sum/answer_count,'capability_drop':drop}
                trial.set_user_attr('metrics',metrics)
                atomic_json(output/f'trial-{trial.number}'/'trial-report.json',report)
                trial.set_user_attr('report',report)
                trial.set_user_attr('feasible',feasible)
                if not feasible:
                    raise optuna.TrialPruned('Capability or validity gate failed')
                if config.objectives:
                    return [metrics[name] for name in config.objectives]
                return metrics['search_rate']
            except DominatedTrial as e:
                raise optuna.TrialPruned(str(e)) from e
            finally:
                backend.editor.reset()
        execute_trials(study,objective,config.trials,output/'CANCEL',events_path=output/'events.jsonl')
        feasible=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE and t.user_attrs.get('feasible')]
        if config.objectives:
            front=[feasible[i] for i in pareto_front([list(t.values) for t in feasible])]
            primary=config.objectives.index('search_rate')
            tiebreak=[config.objectives.index(m) for m in ('capability_drop','full_reply_kl') if m in config.objectives]
            front.sort(key=lambda t:(t.values[primary],)+tuple(t.values[j] for j in tiebreak))
            study.set_user_attr('pareto_front',{'metrics':config.objectives,'primary':'search_rate',
                'front':[{'number':t.number,'values':list(t.values)} for t in front]})
            finalists=front[:config.finalists]
        else:
            feasible.sort(key=lambda t:(t.value,t.user_attrs['report']['capability_drop'],t.user_attrs['report']['full_reply_kl']))
            finalists=feasible[:config.finalists]
        if not finalists:
            raise SearchRejected('No feasible candidate; no release was produced')
        merged=[]
        for trial in finalists:
            directory=output/f'finalist-{trial.number}'
            hf=directory/'hf'
            backend.editor.apply(directions,trial.user_attrs['parameters'])
            if not (directory/'merge.json').exists():
                if hf.exists():
                    hf.rename(hf.with_name('hf-interrupted-'+str(len(list(directory.glob('hf-interrupted-*'))))))
                result=stream_merge(source,hf,backend.editor)
                atomic_json(directory/'merge.json',result)
            audit=verify_resident_export(source,hf,backend.editor)
            atomic_json(directory/'export-verification.json',audit)
            # Tensor equality proves that this resident operator is the exported
            # operator. Re-score without loading or offloading another model.
            report=eval_cached(selection,backend,judge,directory/'merged-selection',config.sandbox_image,
                               {**generation_identity,'edit_identity':edit_identity,'edit_parameters':trial.user_attrs['parameters'],'verified_export':audit},
                               phases,scoring_identity,refusal_floor=config.refusal_floor,batch_policy=config.generation_batch)
            report['paired']=paired_consistency(selection,read_predictions(output/'baseline-hf'),read_predictions(directory/'merged-selection'),policy)
            if report['invalid'] or not report['paired']['regression_ok']:
                continue
            drops=[baseline['by_kind'][k]['pass_rate']-report['by_kind'][k]['pass_rate'] for k in ('code','math')]
            if max(drops)>config.max_capability_drop:
                continue
            merged.append({'id':str(trial.number),'hf_path':str(hf),'hf':report,'search_hf':trial.user_attrs['report'],
                           'export_verification':audit,'parameters':trial.user_attrs['parameters']})
        atomic_json(output/'warm-starts.json',[{'fingerprint':fingerprint,'feasible':True,'parameters':t.params} for t in feasible])
    finally:
        backend.release()
        gc.collect()
    def convert_cached(hf,directory):
        from .release import cached_conversion
        return cached_conversion(hf,directory,config.converter,config.quantizer,config.converter_python,
                                 config.imatrix,convert=export_gguf,verify=verify_quant)
    def score_quants(artifacts,suite,directory):
        reports={}
        for q,artifact in artifacts.items():
            folder=Path(directory)/q
            identity={'artifact_sha256':artifact['sha256'],'server_sha256':provenance['server_sha256'],
                      'ctx_size':config.ctx_size,'max_tokens':config.max_tokens,'temperature':0,'seed':0,
                      'refusal_floor':config.refusal_floor,
                      'generation_batch':config.generation_batch,
                      'transport_sha256':sha256_file(Path(__file__).with_name('transport.py'))}
            store=AnswerStore(folder/'answers',suite,identity)
            cached=completed_scores(suite,store,folder,scoring_identity,config.sandbox_image)
            if cached is not None:
                reports[q]=cached
                continue
            phases.idle()
            if store.status()['missing']:
                slots={'parallel':config.generation_batch['max_sequences']} if config.generation_batch else {}
                with ManagedGGUF(artifact['path'],config.llama_server,config.ctx_size,config.max_tokens,folder/'server.log',secondary=config.allow_cuda_secondary,**slots) as client:
                    generate_answers(suite,client,store,refusal_floor=config.refusal_floor,batch_policy=config.generation_batch)
            reports[q]=score_answers(suite,store,judge,folder,scoring_identity,config.sandbox_image)
        return reports
    baseline_artifacts=convert_cached(source,output/'baseline-quants')
    quant_baseline=score_quants(baseline_artifacts,selection,output/'baseline-quant-selection')
    for candidate in merged:
        directory=output/f"finalist-{candidate['id']}"
        candidate['artifacts']=convert_cached(candidate['hf_path'],directory/'quants')
        candidate['quants']=score_quants(candidate['artifacts'],selection,directory/'selection')
        for q,report in candidate['quants'].items():
            report['paired']=paired_consistency(selection,read_predictions(output/'baseline-quant-selection'/q),read_predictions(directory/'selection'/q),policy)
    ranked=rank_quantized(merged,quant_baseline,config.max_capability_drop,require_consistency=True)
    selection_result={'baseline':quant_baseline,'baseline_artifacts':baseline_artifacts,
                      'candidates':merged,'ranked_ids':[c['id'] for c in ranked]}
    atomic_json(output/'quantized-ranking.json',selection_result)
    if not ranked:
        raise SearchRejected('No quantized candidate improved the matched baseline within the capability limit')
    if selection_only:
        return {'status':'selection-complete',**selection_result}
    winner=ranked[0]
    if expected_winner is not None and str(winner['id'])!=str(expected_winner):
        raise ValueError('Sealed winner changed before holdout; refusing to certify a different candidate')
    # Holdout is used ONCE for certification, never to pick another candidate on failure.
    baseline_holdout=score_quants(baseline_artifacts,holdout,output/'baseline-quant-holdout')
    winner_holdout=score_quants(winner['artifacts'],holdout,output/f"finalist-{winner['id']}"/'holdout')
    for q,report in winner_holdout.items():
        report['paired']=paired_consistency(holdout,read_predictions(output/'baseline-quant-holdout'/q),
                                           read_predictions(output/f"finalist-{winner['id']}"/'holdout'/q),policy)
    certified=rank_quantized([{'id':winner['id'],'quants':winner_holdout}],baseline_holdout,config.max_capability_drop,require_consistency=True)
    if not certified:
        raise HoldoutRejected('Selected candidate failed sealed holdout; no release produced and no holdout-driven retry')
    judge.verify()
    if config.scoring_mode=='judge' and sha256_file(config.judge_artifact)!=judge_hash:
        raise RuntimeError('Judge changed during the experiment')
    manifest={'base_model':config.model,'requested_revision':config.revision,'base_revision':resolved_revision,'model_license':config.model_license,'software_license':'AGPL-3.0-or-later','winner':winner,'holdout':winner_holdout,'baseline_holdout':baseline_holdout,'provenance':provenance,'limits':{'full_reply_kl':config.max_full_reply_kl,'capability_drop':config.max_capability_drop},'status':'certified' if config.scoring_mode=='judge' else 'refusal-screened'}
    release=output/'release'
    assessment='Judged on a disjoint holdout.' if config.scoring_mode=='judge' else 'Screened for explicit refusal wording on a disjoint holdout. Actual task fulfillment and disclaimers were NOT judged.'
    readme='# B70-ablator release\n\nBase: '+config.model+' @ '+config.revision+'\n\nModel license: '+config.model_license+' (base-model obligations remain).\n\nSelected on Q4_K_M/Q5_K_M results. '+assessment+'\nMetrics and exact settings: evaluation.json.\n\nFull-reply KL gates were measured in the search precision. Quantized damage was\nchecked through executed code tests and exact maths grading, not GGUF KL.\nNo claim of zero capability loss or universal uncensoring is made.\n'
    if provenance.get('compute_backend'):
        readme += '\nThis release was produced on a second-rank test backend ('+provenance['compute_backend']['name']+'): B70 kernels, the B70 VRAM envelope and B70 Vulkan serving gates were NOT exercised on it. Read every number here as unverified against the certified device.\n'
    create_release(release,winner['hf_path'],winner['artifacts'],manifest,readme)
    return {'release':str(release),'winner':winner['id'],'holdout':winner_holdout}


def _dependency_version(name):
    """Installed version of a dependency, or None when it is simply not present.
    bitsandbytes is only needed for the nf4 search path, so a run without it must still
    record honest provenance instead of crashing on a missing package metadata entry."""
    from importlib.metadata import PackageNotFoundError
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _invalid_row_detail(directory):
    """Name the rows that made a baseline unusable, from the predictions already on disk.
    The bare gate message forced an operator to open the JSON by hand; the ids and the token
    count at which a row stopped are exactly what separates 'raise the budget' from 'this model
    never terminates'. Read-only, and it can never make an invalid baseline pass."""
    try:
        rows = [json.loads(line) for line in (Path(directory) / 'predictions.jsonl').read_text().splitlines() if line.strip()]
    except OSError:
        return 'details unreadable'
    bad = [r for r in rows if r.get('status') != 'ok']
    if not bad:
        return 'no non-compliant rows recorded'
    listed = ', '.join(f"{r.get('id')} {r.get('language')} {r.get('status')}@{r.get('completion_tokens')}" for r in bad[:5])
    return f"{len(bad)} non-compliant rows: {listed}" + ('...' if len(bad) > 5 else '')


def read_predictions(directory):
    return [json.loads(line) for line in (Path(directory)/'predictions.jsonl').read_text().splitlines() if line.strip()]
