# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
import subprocess
import sys
from pathlib import Path
import sqlite3


class AgentParser(argparse.ArgumentParser):
    def error(self,message):
        print(json.dumps({'error':{'code':'ARGUMENT_ERROR','type':'ArgumentError','message':message}}),file=sys.stderr)
        raise SystemExit(2)


def main():
    parser=AgentParser(description='B70-only ablation with real answer and quantized release gates')
    from . import __version__
    parser.add_argument('--version',action='version',version=f'b70-ablator {__version__}')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor',help='Run a real XPU kernel on the B70')
    sub.add_parser('about',help='Installation/provenance metadata as JSON, no model loads')
    sub.add_parser('methods',help='List implemented and researched methods as JSON, no model loads')
    sub.add_parser('architectures',help='List adapter layouts and verification limits as JSON')
    validate=sub.add_parser('validate-run',help='Read-only configuration/suite/file checks, no model loads')
    validate.add_argument('config',type=Path)
    validate.add_argument('--set',action='append',default=[],metavar='FIELD=VALUE',
                          help='Override a JSON field before validation (dot path; value parsed as JSON, else string)')
    describe=sub.add_parser('describe-run',help='Static suite/family/budget report; refuses what validate-run refuses, no model loads')
    describe.add_argument('config',type=Path)
    describe.add_argument('--set',action='append',default=[],metavar='FIELD=VALUE',
                          help='Override a JSON field before reporting (dot path; value parsed as JSON, else string)')
    diff=sub.add_parser('suite-diff',help='Compare two suite files by id, content and family; no model loads')
    diff.add_argument('before',type=Path)
    diff.add_argument('after',type=Path)
    gguf=sub.add_parser('gguf-inspect',help='GGUF metadata/tensor header inventory; optional second file for the diff, no model loads')
    gguf.add_argument('file',type=Path)
    gguf.add_argument('other',type=Path,nargs='?',default=None)
    near=sub.add_parser('suite-near-dup',help='Report reworded duplicate prompts across suite files; files are never modified')
    near.add_argument('suite',type=Path,nargs='+')
    near.add_argument('--threshold',type=float,default=0.8)
    progress=sub.add_parser('run-status',help='Read-only search state, no model loads or DB migrations')
    progress.add_argument('directory',type=Path)
    studies_p=sub.add_parser('studies',help='Summarize every study.db under a runs root; read-only sqlite, no DB creation')
    studies_p.add_argument('directory',type=Path)
    cancel_p=sub.add_parser('run-cancel',help='Ask a study to stop before its next trial request; no process signalling')
    cancel_p.add_argument('directory',type=Path)
    for name,help_text in (('auto','Select methods within one explicit trial budget, then certify only the global winner'),
                           ('validate-auto','Validate automatic method selection without model loads or output writes')):
        command=sub.add_parser(name,help=help_text)
        command.add_argument('config',type=Path)
        command.add_argument('--set',action='append',default=[],metavar='FIELD=VALUE',
                             help='Override a JSON field before validation (dot path; value parsed as JSON, else string)')
    for name,help_text in (('tree-search','Experimental J-space partial-forward beam search; B70 only, no certified export'),
                          ('validate-tree','Read-only tree configuration/provenance checks, no model runtime imports')):
        command=sub.add_parser(name,help=help_text)
        command.add_argument('config',type=Path)
    for name,help_text in (('tree-ranker-data','Join recorded trajectories to independent answer labels; enforce split isolation'),
                          ('tree-ranker-train','Train and calibrate the pairwise behavior ranker on B70; no test split'),
                          ('tree-ranker-audit','Held-out refusal/evasion/broken-output false-confidence audit on B70')):
        command=sub.add_parser(name,help=help_text)
        command.add_argument('config',type=Path)
    bind_lens=sub.add_parser('bind-tree-lens',help='Bind an upstream jlens file to operator-attested model provenance; no model inference')
    bind_lens.add_argument('lens',type=Path)
    bind_lens.add_argument('--model-identity',type=Path,required=True)
    bind_lens.add_argument('--output',type=Path,required=True)
    inspect=sub.add_parser('inspect',help='Inspect model architecture without loading weights')
    inspect.add_argument('model')
    inspect.add_argument('--revision',default=None)
    inspect.add_argument('--trust-remote-code',action='store_true')
    import_suite=sub.add_parser('import-suite',help='Build suites from public prompt corpora with recorded provenance')
    import_suite.add_argument('spec',type=Path)
    import_suite.add_argument('--output',type=Path,required=True)
    validate_import_parser=sub.add_parser('validate-import',help='Check an import spec and local files; no downloads, no writes')
    validate_import_parser.add_argument('spec',type=Path)
    run=sub.add_parser('run',help='Run a resumable experiment through sealed quantized certification')
    run.add_argument('config',type=Path)
    run.add_argument('--set',action='append',default=[],metavar='FIELD=VALUE',
                     help='Override a JSON field before the run (dot path; value parsed as JSON, else string)')
    train=sub.add_parser('train',help='Train local B70 SFT/DPO edit factors; output is unvalidated')
    train.add_argument('config',type=Path)
    verify=sub.add_parser('verify-release',help='Verify a finished release using only file reads, no models')
    verify.add_argument('directory',type=Path)
    verify.add_argument('--expected-identity')
    evaluate=sub.add_parser('evaluate',help='Score a local GGUF with the same answer-level harness')
    evaluate.add_argument('--mode',dest='scoring_mode',choices=['judge','refusal-check'],default='judge')
    evaluate.add_argument('--model',type=Path,required=True)
    evaluate.add_argument('--server',type=Path,required=True)
    evaluate.add_argument('--suite',type=Path,required=True)
    evaluate.add_argument('--judge-url')
    evaluate.add_argument('--judge-model')
    evaluate.add_argument('--judge-artifact',type=Path)
    evaluate.add_argument('--calibration',type=Path)
    evaluate.add_argument('--judge-min-confidence',type=float)
    evaluate.add_argument('--judge-min-accuracy',type=float)
    evaluate.add_argument('--judge-max-tokens',type=int)
    evaluate.add_argument('--max-tokens',type=int,required=True)
    evaluate.add_argument('--ctx-size',type=int,required=True)
    evaluate.add_argument('--sandbox-image',required=True)
    evaluate.add_argument('--output',type=Path,required=True)
    from .arguments import add_arguments
    add_arguments(sub)
    args=parser.parse_args()
    try:
        if args.command=='tree-ranker-data':
            from .tree_behavior_data import prepare_ranker_data
            result=prepare_ranker_data(args.config)
        elif args.command in {'tree-ranker-train','tree-ranker-audit'}:
            from .tree_ranker import train_ranker,audit_ranker
            result=(train_ranker if args.command=='tree-ranker-train' else audit_ranker)(args.config)
        elif args.command=='validate-tree':
            from .tree_config import validate_tree
            result=validate_tree(args.config)
        elif args.command=='tree-search':
            from .tree_cli import run_tree
            result=run_tree(args.config)
        elif args.command=='bind-tree-lens':
            from .tree_cli import bind_lens
            result=bind_lens(args.lens,json.loads(args.model_identity.read_text()),args.output)
        elif args.command in {'auto','validate-auto'}:
            from .auto import AutoConfig,run_auto,validate_auto
            obj=json.loads(args.config.read_text())
            if args.set:
                from .overrides import apply_overrides
                obj=apply_overrides(obj,args.set)
            config=AutoConfig.from_json(obj)
            result=validate_auto(config) if args.command=='validate-auto' else run_auto(config)
        elif args.command=='about':
            from .agent import about
            result=about()
        elif args.command=='validate-run':
            from .agent import validate_run
            result=validate_run(args.config,args.set)
        elif args.command=='run-status':
            from .agent import run_status
            result=run_status(args.directory)
        elif args.command=='describe-run':
            from .agent import describe_run
            result=describe_run(args.config,args.set)
        elif args.command=='gguf-inspect':
            from .agent import gguf_inspect
            result=gguf_inspect(args.file,args.other)
        elif args.command=='suite-near-dup':
            from .agent import suite_near_dup
            result=suite_near_dup(args.suite,args.threshold)
        elif args.command=='suite-diff':
            from .agent import suite_diff
            result=suite_diff(args.before,args.after)
        elif args.command in {'studies','run-cancel'}:
            from .agent import studies, request_cancel
            result=studies(args.directory) if args.command=='studies' else request_cancel(args.directory)
        elif args.command in {'import-suite','validate-import'}:
            from .sources import import_suites,validate_import
            spec=json.loads(args.spec.read_text())
            result=validate_import(spec) if args.command=='validate-import' else import_suites(spec,args.output)
        elif args.command=='methods':
            from .methods import method_catalog
            result=method_catalog()
        elif args.command=='architectures':
            from .architectures import architecture_catalog
            result=architecture_catalog()
        elif args.command in {'generate','score','status'}:
            from .job_cli import generate_job, score_job, status_job
            result={'generate':generate_job,'score':score_job,'status':status_job}[args.command](args)
        elif args.command=='verify-release':
            from .release import verify_release
            result=verify_release(args.directory,args.expected_identity)
        elif args.command=='doctor':
            from .pipeline import configure_compiler_paths
            configure_compiler_paths()
            import torch
            from .hf import require_b70
            device=require_b70()
            x=torch.arange(16,device=device,dtype=torch.float32).reshape(4,4)
            product=x@x.T
            torch.xpu.synchronize()
            result={'torch':torch.__version__,'device':torch.xpu.get_device_name(0),'kernel_sum':float(product.sum()),'kernel_finite':bool(torch.isfinite(product).all()),'cpu_fallback':False}
        elif args.command=='inspect':
            from transformers import AutoConfig
            from .editor import ARCHITECTURES
            config=AutoConfig.from_pretrained(args.model,revision=args.revision,trust_remote_code=args.trust_remote_code)
            result={'model_type':config.model_type,'architectures':config.architectures,'layers':getattr(config,'num_hidden_layers',None),'hidden_size':getattr(config,'hidden_size',None),'adapter_available':config.model_type in ARCHITECTURES,'revision':getattr(config,'_commit_hash',None),'note':'Availability is not evidence of a successful generation/quantization round trip.'}
        elif args.command=='run':
            from .search import RunConfig
            from .pipeline import run
            obj=json.loads(args.config.read_text())
            if args.set:
                from .overrides import apply_overrides
                obj=apply_overrides(obj,args.set)
            result=run(RunConfig.from_json(obj))
        elif args.command=='train':
            from .training import train_job
            result=train_job(args.config)
        else:
            if (args.output/'report.json').exists():
                raise FileExistsError('Use a new output directory for an independent evaluation')
            from .data import load_suite, digest
            from .pipeline import verify_judge, eval_cached, atomic_json
            from .release import ManagedGGUF, sha256_file
            suite=load_suite(args.suite)
            before=None
            if args.scoring_mode=='refusal-check':
                from .refusal_check import RefusalChecker
                judge=RefusalChecker()
            else:
                required=('judge_url','judge_model','judge_artifact','calibration','judge_min_confidence','judge_min_accuracy','judge_max_tokens')
                if any(getattr(args,key) is None for key in required):
                    raise ValueError('Judge mode requires all judge settings')
                judge=verify_judge(args)
                calibration=json.loads(args.calibration.read_text())
                judge.calibrate(calibration,args.judge_min_accuracy)
                if not {r['language'] for r in suite}<={r['language'] for r in calibration}:
                    raise ValueError('Calibration does not cover the evaluation languages')
                before=sha256_file(args.judge_artifact)
            with ManagedGGUF(args.model,args.server,args.ctx_size,args.max_tokens,args.output/'server.log') as client:
                identity={'artifact_sha256':sha256_file(args.model),'server_sha256':sha256_file(args.server),
                          'ctx_size':args.ctx_size,'max_tokens':args.max_tokens,'temperature':0,'seed':0}
                from .job_cli import source_identity, sandbox_identity
                scoring_identity={'judge_sha256':before,'framework_sources':source_identity(),
                                  'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                                  'sandbox_image_id':sandbox_identity(args.sandbox_image) if any(t['kind']=='code' for t in suite) else None}
                result=eval_cached(suite,client,judge,args.output,args.sandbox_image,identity,scoring_identity=scoring_identity)
            if args.scoring_mode=='judge':
                verify_judge(args)
                if before!=sha256_file(args.judge_artifact):
                    raise ValueError('Judge changed during evaluation')
            atomic_json(args.output/'evaluation-metadata.json',{'artifact_sha256':sha256_file(args.model),'judge_sha256':before,'suite_digest':digest(suite),'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'report':result})
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,RuntimeError,OSError,sqlite3.Error,subprocess.SubprocessError) as e:
        from .errors import error_code
        print(json.dumps({'error':{'code':error_code(e),'type':type(e).__name__,'message':str(e) or repr(e),'command':args.command}}),file=sys.stderr)
        raise SystemExit(1) from e


if __name__=='__main__':
    main()
