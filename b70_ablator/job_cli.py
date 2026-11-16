# SPDX-License-Identifier: AGPL-3.0-or-later
"""Private B70 job commands. Generation and judging own separate GPU lifetimes."""
import json
import math
from pathlib import Path
import subprocess

from .answers import AnswerStore, atomic_json, generate_answers, score_answers, score_status, scoring_manifest
from .data import load_suite, digest
from .pipeline import verify_judge
from .release import ManagedGGUF, sha256_file
from .refusal_check import RefusalChecker


def add_arguments(sub):
    generate=sub.add_parser('generate',help='Save model answers, resuming only missing prompts')
    generate.add_argument('--model',type=Path,required=True,help='Actual target GGUF')
    generate.add_argument('--server',type=Path,required=True)
    generate.add_argument('--suite',type=Path,required=True)
    generate.add_argument('--max-tokens',type=int,required=True)
    generate.add_argument('--ctx-size',type=int,required=True)
    generate.add_argument('--refusal-floor',type=int)
    generate.add_argument('--batch-max-sequences',type=int)
    generate.add_argument('--batch-token-budget',type=int)
    generate.add_argument('--max-generated-tokens',type=int,help='Stop cleanly after generating this many completion tokens; rerun resumes')
    generate.add_argument('--output',type=Path,required=True,help='Persistent answer directory')
    score=sub.add_parser('score',help='Judge saved answers without loading the target model')
    score.add_argument('--mode',dest='scoring_mode',choices=['judge','refusal-check'],default='judge')
    score.add_argument('--answers',type=Path,required=True)
    score.add_argument('--suite',type=Path,required=True)
    score.add_argument('--server',type=Path)
    score.add_argument('--judge-artifact',type=Path)
    score.add_argument('--calibration',type=Path)
    score.add_argument('--judge-max-tokens',type=int)
    score.add_argument('--ctx-size',type=int)
    score.add_argument('--judge-min-confidence',type=float)
    score.add_argument('--judge-min-accuracy',type=float)
    score.add_argument('--sandbox-image',default='python:3.12-slim')
    score.add_argument('--output',type=Path,required=True)
    score.add_argument('--retry-invalid',action='store_true',help='Retry invalid judge results; keep valid scores')
    score.add_argument('--max-judge-requests',type=int,help='Fresh judge calls allowed this run; raises BUDGET_EXHAUSTED with saved items intact')
    status=sub.add_parser('status',help='Read saved job progress without starting any model')
    status.add_argument('--answers',type=Path,required=True)
    status.add_argument('--scores',type=Path)


def sandbox_identity(image):
    return subprocess.run(['docker','image','inspect',image,'--format','{{.Id}}'],
                          capture_output=True,text=True,check=True).stdout.strip()


def source_identity():
    root=Path(__file__).parent
    return {str(p.relative_to(root)):sha256_file(p) for p in sorted(root.glob('*.py'))+sorted((root/'assets').glob('*.txt'))}


def generate_job(args):
    if not 0<args.max_tokens<args.ctx_size:
        raise ValueError('Context must exceed a positive output budget')
    floor=args.refusal_floor
    if floor is not None and not 0<floor<=args.max_tokens:
        raise ValueError('refusal-floor must be positive and within max-tokens')
    batch=None
    if args.batch_max_sequences is not None or args.batch_token_budget is not None:
        from .batching import validate_batch_policy
        batch={'max_sequences':args.batch_max_sequences,'max_padded_tokens':args.batch_token_budget}
        validate_batch_policy(batch)
    suite=load_suite(args.suite)
    identity={'artifact_sha256':sha256_file(args.model),'server_sha256':sha256_file(args.server),
              'refusal_floor':floor,'generation_batch':batch,
              'ctx_size':args.ctx_size,'max_tokens':args.max_tokens,'temperature':0,'seed':0,
              'transport_sha256':sha256_file(Path(__file__).with_name('transport.py'))}
    store=AnswerStore(args.output,suite,identity)
    if not store.status()['missing']:
        return generate_answers(suite,None,store,refusal_floor=floor,batch_policy=batch,budget=args.max_generated_tokens)
    slots={'parallel':batch['max_sequences']} if batch else {}
    with ManagedGGUF(args.model,args.server,args.ctx_size,args.max_tokens,args.output/'target-server.log',**slots) as client:
        return generate_answers(suite,client,store,refusal_floor=floor,batch_policy=batch,budget=args.max_generated_tokens)


def calibrate_cached(judge,records,args,identity):
    # A completed check is reusable ONLY with exactly the same judge and scorer.
    key=digest({'identity':identity,'records':records,'minimum_accuracy':args.judge_min_accuracy})
    path=args.output/'judge-calibration.json'
    if path.exists():
        record=json.loads(path.read_text())
        payload={k:v for k,v in record.items() if k!='record_digest'}
        if record.get('record_digest')!=digest(payload) or record.get('identity')!=key:
            raise ValueError('Judge calibration identity/integrity mismatch')
        report=record['report']
    else:
        if judge is None:
            raise ValueError('Saved judge check is missing; do not reuse scores without it')
        report=judge.calibrate(records,args.judge_min_accuracy)
        record={'identity':key,'report':report}
        record['record_digest']=digest(record)
        atomic_json(path,record)
    accuracy=report.get('accuracy')
    if report.get('total')!=len(records) or type(accuracy) not in (int,float) or not math.isfinite(accuracy) or accuracy<args.judge_min_accuracy:
        raise ValueError('Cached judge check does not meet the required accuracy')
    return report


def score_job(args):
    suite=load_suite(args.suite)
    store=AnswerStore.open(args.answers,suite)
    if store.status()['missing']:
        raise ValueError('Saved answers are missing; finish generation before loading the judge')
    if args.scoring_mode=='refusal-check':
        identity={'scoring_mode':'refusal-check','framework_sources':source_identity(),
                  'sandbox_image_id':sandbox_identity(args.sandbox_image) if any(r['kind']=='code' for r in suite) else None}
        return score_answers(suite,store,RefusalChecker(),args.output,identity,args.sandbox_image,max_judge_requests=args.max_judge_requests)
    required=('server','judge_artifact','calibration','judge_max_tokens','ctx_size','judge_min_confidence','judge_min_accuracy')
    missing=[key for key in required if getattr(args,key) is None]
    if missing:
        raise ValueError('Judge mode requires: '+', '.join(missing))
    if not 0<args.judge_max_tokens<args.ctx_size:
        raise ValueError('Context must exceed a positive judge output budget')
    for value in (args.judge_min_confidence,args.judge_min_accuracy):
        if not math.isfinite(value) or not 0<=value<=1:
            raise ValueError('Judge thresholds must be finite values in 0..1')
    calibration=json.loads(args.calibration.read_text())
    if not {r['language'] for r in suite}<={r['language'] for r in calibration}:
        raise ValueError('Judge checks do not cover all evaluation languages')
    before=sha256_file(args.judge_artifact)
    identity={'judge_sha256':before,'server_sha256':sha256_file(args.server),
              'judge_max_tokens':args.judge_max_tokens,'ctx_size':args.ctx_size,
              'min_confidence':args.judge_min_confidence,'min_accuracy':args.judge_min_accuracy,
              'calibration_digest':digest(calibration),'framework_sources':source_identity(),
              'sandbox_image_id':sandbox_identity(args.sandbox_image)}
    scoring_manifest(suite,store,args.output,identity,args.sandbox_image)
    status=score_status(args.output,store)
    if status.get('report_current') and not (args.retry_invalid and status['invalid']):
        calibrate_cached(None,calibration,args,identity)
        return score_answers(suite,store,None,args.output,identity,args.sandbox_image,max_judge_requests=args.max_judge_requests)
    with ManagedGGUF(args.judge_artifact,args.server,args.ctx_size,args.judge_max_tokens,args.output/'judge-server.log') as client:
        args.judge_url,args.judge_model=client.url,client.model
        judge=verify_judge(args)
        calibrate_cached(judge,calibration,args,identity)
        result=score_answers(suite,store,judge,args.output,identity,args.sandbox_image,args.retry_invalid,max_judge_requests=args.max_judge_requests)
        verify_judge(args)
        if before!=sha256_file(args.judge_artifact):
            raise ValueError('Judge changed during scoring')
    return result


def status_job(args):
    saved=json.loads((args.answers/'manifest.json').read_text())
    store=AnswerStore.open(args.answers,saved['inputs'])
    result=store.status()
    if args.scores is not None:
        result['scoring']=score_status(args.scores,store)
    return result
