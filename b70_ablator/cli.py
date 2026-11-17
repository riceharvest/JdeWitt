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


def completion_script(parser, shell):
    """Generate bash/zsh completion from the same argparse tree `commands` introspects.

    The script is derived from the live parser, so it cannot drift from the
    real flag surface; regenerate with `commands --completions bash > ...`.
    """
    import argparse as _argparse
    prog = parser.prog or 'b70-ablator'
    if prog.endswith('__main__.py') or '/' in prog:
        prog = 'b70-ablator'
    safe_prog = prog.replace('-', '_').replace('.', '_')
    subactions = None
    for action in parser._actions:
        if isinstance(action, _argparse._SubParsersAction):
            subactions = action
    if subactions is None:
        raise ValueError('parser has no subcommands')
    commands = []
    per_command_opts = {}
    for subaction in subactions._get_subactions():
        child = subactions.choices[subaction.dest]
        opts = []
        for argument in child._actions:
            if argument.dest == 'help' or argument.help == _argparse.SUPPRESS:
                continue
            opts.extend(argument.option_strings or [])
            if argument.metavar and not argument.option_strings:
                opts.append('<' + argument.metavar.rstrip(']') + '>')
        per_command_opts[subaction.dest] = ' '.join(sorted(set(opts)))
        help_text = (subaction.help or child.description or '').replace(':', '-')
        commands.append((subaction.dest, help_text))
    command_names = ' '.join(sorted(name for name, _help in commands))
    if shell == 'bash':
        lines = [f'_{safe_prog}_completion() {{',
                 '    local cur subcmd opts',
                 '    cur="${COMP_WORDS[COMP_CWORD]}"',
                 '    if [[ $COMP_CWORD -eq 1 ]]; then',
                 f'        COMPREPLY=( $(compgen -W "{command_names}" -- "$cur") )',
                 '        return',
                 '    fi',
                 '    subcmd="${COMP_WORDS[1]}"',
                 '    opts="--help"',
                 '    case "$subcmd" in']
        for name, _help in sorted(commands):
            lines.append(f'        {name}) opts="--help {per_command_opts[name]}" ;;')
        lines += ['    esac',
                  '    COMPREPLY=( $(compgen -W "$opts" -- "$cur") )',
                  '}',
                  f'complete -o default -F _{safe_prog}_completion {prog}']
        return '\n'.join(lines) + '\n'
    if shell == 'zsh':
        lines = ['#compdef ' + prog,
                 f'_{safe_prog}() {{',
                 '    local -a commands',
                 '    commands=(']
        for name, help_text in sorted(commands):
            lines.append(f"        '{name}:{help_text}'")
        lines += ['    )',
                  '    if (( CURRENT == 2 )); then',
                  "        _describe 'command' commands",
                  '    else',
                  '        _arguments -w -s -S',
                  '        compadd -- --help',
                  '    fi',
                  '}',
                  f'_{safe_prog}']
        return '\n'.join(lines) + '\n'
    raise ValueError(f'unsupported shell {shell!r}')


def command_manifest(parser):
    """Self-description of the installed command surface, for driving agents."""
    import argparse as _argparse
    from .errors import EXPLAINER
    from .schemas import EMIT_SCHEMAS
    commands = []
    for action in parser._actions:
        if not isinstance(action, _argparse._SubParsersAction):
            continue
        for subaction in action._get_subactions():
            child = action.choices[subaction.dest]
            arguments = []
            for argument in child._actions:
                if argument.dest == 'help' or argument.help == _argparse.SUPPRESS:
                    continue
                choices = getattr(argument, 'choices', None)
                arguments.append({'flags': list(argument.option_strings or [argument.dest]),
                                  'dest': argument.dest, 'required': bool(argument.required),
                                  'choices': [str(choice) for choice in choices] if choices else None})
            commands.append({'command': subaction.dest, 'help': subaction.help or child.description,
                             'arguments': arguments, 'schema_enforced': subaction.dest in EMIT_SCHEMAS})
    commands.sort(key=lambda item: item['command'])
    return {'scope': 'Self-description of the installed CLI; anything marked schema_enforced fails its own command if the writer drifts.',
            'commands': commands,
            'exit_codes': {'success': 0, 'stable_failure': 1, 'argument_error': 2},
            'error_codes': [dict(row, code=code) for code, row in sorted(EXPLAINER.items())],
            'note': 'Failures print {"error":{"code","type","message","command"}}. explain-error CODE prints one row of the same table.'}


def main():
    parser=AgentParser(description='B70-only ablation with real answer and quantized release gates')
    from . import __version__
    parser.add_argument('--version',action='version',version=f'b70-ablator {__version__}')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor',help='Run a real XPU kernel on the B70')
    sub.add_parser('about',help='Installation/provenance metadata as JSON, no model loads')
    sub.add_parser('methods',help='List implemented and researched methods as JSON, no model loads')
    sub.add_parser('architectures',help='List adapter layouts and verification limits as JSON')
    hub=sub.add_parser('hub-census',help='Live Hub text-generation census joined to the adapter table; adapter worklist ordered by downloads, no model loads')
    hub.add_argument('--since',default='2026-08-01',help='Only repositories created on or after this UTC date')
    hub.add_argument('--author',action='append',help='Repeat for publishers to scan; default is the architecture inventory authors')
    hub.add_argument('--per-author',type=int,default=40,help='Cap newest text-generation repositories scanned per publisher')
    cmdm=sub.add_parser('commands',help='Self-description of the command surface, flags, exit codes and error-code reference; no model loads')
    cmdm.add_argument('--completions',choices=['bash','zsh'],help='Print a shell completion script generated from the live parser')
    explain=sub.add_parser('explain-error',help='What one stable CLI error code means and what to do about it; no model loads')
    explain.add_argument('code')
    runls=sub.add_parser('run-ls',help='Artifact contract inventory of a run output directory; no model loads')
    runls.add_argument('output',type=Path)
    rundoctor=sub.add_parser('run-doctor',help='One-shot read-only diagnosis of a run directory: why it looks stuck, broken or idle, with actions')
    rundoctor.add_argument('output',type=Path)
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
    editg=sub.add_parser('gguf-edit',help='Edit quantized GGUF tensors via dequant->edit->requant; everything else byte-verbatim, no model')
    editg.add_argument('input',type=Path)
    editg.add_argument('output',type=Path)
    editg.add_argument('--tensor',action='append',default=[],metavar='NAME=DELTA.NPY',
                       help='float32 delta npy matching the tensor shape; repeatable')
    editg.add_argument('--delta-gguf',type=Path,default=None,
                       help='GGUF delta file (gguf-delta-pack); float tensors named like target tensors')
    editg.add_argument('--strength',type=float,default=1.0,help='multiplier applied to --delta-gguf tensors')
    editm=sub.add_parser('gguf-meta',help='Edit GGUF metadata (chat templates, tokenizer fields); tensor data is copied verbatim, no model')
    dpack=sub.add_parser('gguf-delta-pack',help='Pack float32 delta npys into a GGUF delta file consumable by gguf-edit --delta-gguf; inspectable, signable')
    dpack.add_argument('output',type=Path)
    dpack.add_argument('specs',nargs='+',metavar='NAME=DELTA.NPY')
    dexport=sub.add_parser('direction-export',help='Export a registered fitted plan as a GGUF direction file (tensors + method/geometry/provenance metadata); no model')
    dexport.add_argument('registry',type=Path)
    dexport.add_argument('key')
    dexport.add_argument('output',type=Path)
    dexport.add_argument('--dtype',choices=['bf16','f32'],default='bf16')
    qaudit=sub.add_parser('quants-audit',help='Dequant/requant falsifier for the GGML codec ports over a real GGUF: byte identity + worst drift on bounded block samples; no model')
    qaudit.add_argument('file',type=Path)
    qaudit.add_argument('--blocks',type=int,default=4096,help='block samples per dtype')
    editm.add_argument('input',type=Path)
    editm.add_argument('output',type=Path)
    editm.add_argument('--set',action='append',default=[],metavar='KEY=JSON',
                       help='Set metadata key to a JSON value (string/number/bool/array); repeatable')
    editm.add_argument('--unset',action='append',default=[],metavar='KEY',
                       help='Remove a metadata key; repeatable')
    gdiff=sub.add_parser('gguf-tensor-diff',help='Compare two GGUF files tensor-by-tensor: byte identity plus dequant error, no model')
    gdiff.add_argument('before',type=Path)
    gdiff.add_argument('after',type=Path)
    sdiff=sub.add_parser('safetensors-diff',help='Compare two safetensors checkpoints tensor-by-tensor: byte identity plus relative RMS, no model')
    sdiff.add_argument('before',type=Path)
    sdiff.add_argument('after',type=Path)
    pdiff=sub.add_parser('pt-diff',help='Compare two torch .pt checkpoints tensor-by-tensor: byte identity, relative RMS and cosine, no model')
    pdiff.add_argument('before',type=Path)
    pdiff.add_argument('after',type=Path)
    pdiff.add_argument('--limit',type=int,default=100,help='max differing tensors to list')
    edit8=sub.add_parser('fp8-edit',help='Edit FP8 (E4M3/E5M2) safetensors tensors via dequant->edit->requant; everything else byte-verbatim, no model')
    edit8.add_argument('input',type=Path)
    edit8.add_argument('output',type=Path)
    edit8.add_argument('--format',choices=('e4m3','e5m2'),required=True)
    edit8.add_argument('--weights',required=True)
    edit8.add_argument('--delta',type=Path,required=True,help='float32 npy delta matching the tensor shape')
    edit8.add_argument('--scales',required=True,help='F32 scale tensor name')
    edit8.add_argument('--scale-mode',choices=('tensor','block'),default='block')
    edit8.add_argument('--axis',choices=('row','col'),default='row',help='block axis for block scales')
    nf4=sub.add_parser('nf4-edit',help='Edit bitsandbytes-style NF4 safetensors (U8 packed + absmax) via dequant->edit->requant; everything else byte-verbatim, no model')
    nf4.add_argument('input',type=Path)
    nf4.add_argument('output',type=Path)
    nf4.add_argument('--weights',required=True,help='U8 packed tensor name (two 4-bit codes per byte)')
    nf4.add_argument('--delta',type=Path,required=True,help='float32 npy delta matching the unpacked tensor shape')
    nf4.add_argument('--scales',required=True,help='F32 or F16 absmax tensor name (one per 64-element block)')
    nf4.add_argument('--blocksize',type=int,default=64)
    fp4=sub.add_parser('fp4-edit',help='Edit MXFP4/NVFP4 safetensors tensors via dequant->edit->requant; everything else byte-verbatim, no model')
    fp4.add_argument('input',type=Path)
    fp4.add_argument('output',type=Path)
    fp4.add_argument('--format',choices=['mxfp4','nvfp4'],required=True)
    fp4.add_argument('--weights',required=True,help='safetensors tensor name holding packed fp4 nibbles (U8)')
    fp4.add_argument('--delta',type=Path,required=True,help='float32 npy delta matching the logical element count')
    fp4.add_argument('--scales',required=True,help='safetensors tensor name holding block scales (U8: E8M0 or E4M3)')
    fp4.add_argument('--global-scale',type=float,default=None,help='NVFP4 global scale, if not stored in the file')
    fp4.add_argument('--global-scale-tensor',default=None,help='NVFP4 global scale stored as an F32/F64 tensor')
    near=sub.add_parser('suite-near-dup',help='Report reworded duplicate prompts across suite files; files are never modified')
    near.add_argument('suite',type=Path,nargs='+')
    near.add_argument('--threshold',type=float,default=0.8)
    stats=sub.add_parser('suite-stats',help='Descriptive statistics for one suite file; no model loads')
    stats.add_argument('suite',type=Path)
    sub.add_parser('presets',help='List evidence-gated preset profiles; presets without evidence are refused')
    abl=sub.add_parser('abliterate',help='One command: build a validated auto-search config from MODEL + --suites + policy and run it (--dry-run to only validate)')
    abl.add_argument('model',type=Path)
    abl.add_argument('--suites',type=Path,required=True,help='Directory containing directions.jsonl, selection.jsonl, holdout.jsonl')
    abl.add_argument('--output',type=Path,required=True)
    abl.add_argument('--policy',choices=['standard','strict'],default='standard')
    abl.add_argument('--total-trials',type=int)
    abl.add_argument('--rank',type=int,default=2)
    abl.add_argument('--revision',required=True,help='Pinned model revision or content hash')
    abl.add_argument('--license',required=True,dest='license_id',help='Model license identifier')
    abl.add_argument('--defaults',type=Path,help='JSON object of run fields (converter, quantizer, llama_server, ...)')
    abl.add_argument('--preset')
    abl.add_argument('--set',action='append',default=[],metavar='FIELD=VALUE',
                     help='Override dotted paths in the built {run, candidates, total_trials} document')
    abl.add_argument('--dry-run',action='store_true',help='Validate the built configuration instead of running the search')
    cache=sub.add_parser('corpus-cache',help='Inventory the dataset import cache; --prune-older-than N previews/deletes old unpinned revisions')
    cache.add_argument('--prune-older-than',type=int,metavar='DAYS')
    cache.add_argument('--execute',action='store_true',help='Actually delete prune candidates instead of previewing')
    runev=sub.add_parser('run-events',help='Tail the append-only lifecycle event log for a run directory')
    runev.add_argument('output',type=Path)
    runev.add_argument('--tail',type=int,default=50)
    smetrics=sub.add_parser('study-metrics',help='Per-trial parameter/value rollup from a study sqlite, read-only')
    smetrics.add_argument('run',type=Path)
    sub.add_parser('schemas',help='List published JSON Schemas for on-disk artifacts')
    art=sub.add_parser('verify-artifact',help='Check one artifact against its published schema; shape and version only')
    art.add_argument('file',type=Path)
    art.add_argument('--name',help='Schema name; inferred from the file name by default')
    art.add_argument('--all',action='store_true',help='Treat FILE as a directory and check every known artifact found in it')
    rdiff=sub.add_parser('release-diff',help='File-set, claim-chain and signature differences between two release directories; reads only')
    rdiff.add_argument('left',type=Path)
    rdiff.add_argument('right',type=Path)
    robust=sub.add_parser('robustness-report',help='Worst-case pass rate of saved scores across multiple suites (e.g. main + attacks)')
    robust.add_argument('--eval',action='append',nargs=3,required=True,metavar='LABEL SUITE SCORES_DIR')
    robust.add_argument('--pass-on',choices=['compliance','compliance+partial'],default='compliance')
    fdecay=sub.add_parser('family-decay',help='Pass-rate decay from fitted families to held-out families over saved scores')
    fdecay.add_argument('--fitted',nargs=2,required=True,metavar='SUITE SCORES_DIR')
    fdecay.add_argument('--held',nargs=2,required=True,metavar='SUITE SCORES_DIR')
    fdecay.add_argument('--pass-on',choices=['compliance','compliance+partial'],default='compliance')
    preg=sub.add_parser('plan-register',help='Bind a finished run\'s fitted plan into a local warm-start registry')
    preg.add_argument('registry',type=Path)
    preg.add_argument('run_output',type=Path)
    preg.add_argument('--gates',type=Path,help='Evidence JSON (verify-release / robustness-report output)')
    preg.add_argument('--allow-unmeasured',action='store_true',help='Register with an explicit UNMEASURED label instead of gates')
    plist=sub.add_parser('plan-list',help='Registered plans with provenance and gate state')
    plist.add_argument('registry',type=Path)
    pshow=sub.add_parser('plan-show',help='Metadata of one registered plan')
    pshow.add_argument('registry',type=Path)
    pshow.add_argument('key')
    pload=sub.add_parser('plan-load',help='Integrity-checked load of a registered plan; refuses geometry mismatches')
    pload.add_argument('registry',type=Path)
    pload.add_argument('key')
    pload.add_argument('--expect',nargs=3,type=int,default=None,metavar='LAYERS RANK HIDDEN')
    pload.add_argument('--device',default='cpu')
    pbundle=sub.add_parser('plan-bundle',help='Archive one registered plan (signed meta + payload) into a single digest-pinned file')
    pbundle.add_argument('registry',type=Path)
    pbundle.add_argument('key')
    pbundle.add_argument('-o','--output',type=Path,required=True)
    punbundle=sub.add_parser('plan-unbundle',help='Verify a plan bundle by digest and install it; refuses tampered or geometry-disagreeing entries')
    punbundle.add_argument('bundle',type=Path)
    punbundle.add_argument('--into',type=Path,required=True)
    pprune=sub.add_parser('plan-prune',help='Report registry rot (failed integrity, orphans); delete only with --execute')
    pprune.add_argument('registry',type=Path)
    pprune.add_argument('--execute',action='store_true')
    compare=sub.add_parser('score-compare',help='Paired offline comparison of two saved scoring dirs over one suite; no models')
    compare.add_argument('suite',type=Path)
    compare.add_argument('before',type=Path)
    compare.add_argument('after',type=Path)
    compare.add_argument('--pass-on',choices=['compliance','compliance+partial'],default='compliance')
    scmp2=sub.add_parser('suites-compare',help='Several score-compare families with Holm-adjusted sign-test p-values so a lucky suite cannot hide the family-wise cost; no models')
    scmp2.add_argument('families',type=Path,nargs='+',metavar='SUITE BEFORE AFTER ...')
    scmp2.add_argument('--pass-on',choices=['compliance','compliance+partial'],default='compliance')
    ensemble=sub.add_parser('judge-ensemble',help='Majority aggregation of independently saved judge verdict files over one suite; no models')
    ensemble.add_argument('suite',type=Path)
    ensemble.add_argument('directory',type=Path,nargs='+')
    ensemble.add_argument('--min-agreement',type=float,default=0.5)
    calibration=sub.add_parser('calibration-report',help='Agreement statistics of independently saved judge runs against majority consensus; no models, no accuracy claim')
    calibration.add_argument('suite',type=Path)
    calibration.add_argument('directory',type=Path,nargs='+')
    calibration.add_argument('--min-agreement',type=float,default=0.5)
    calibration.add_argument('--ledger',type=Path,default=None,help='Authored calibration ledger to report alongside')
    vaudit=sub.add_parser('verdict-audit',help='Saved judge verdicts vs deterministic wording check on the same saved answers; no models')
    vaudit.add_argument('directory',type=Path,help='Scores directory containing predictions.jsonl')
    vaudit.add_argument('--limit',type=int,default=25)
    sensitivity=sub.add_parser('score-sensitivity',help='Offline recount of saved verdicts under alternative confidence floors; no models')
    sensitivity.add_argument('directory',type=Path)
    sensitivity.add_argument('--floor',action='append',type=float,dest='min_confidences',
                             help='Confidence floor to recount under; repeatable (default 0.5 0.7 0.9)')
    progress=sub.add_parser('run-status',help='Read-only search state, no model loads or DB migrations')
    progress.add_argument('directory',type=Path)
    studies_p=sub.add_parser('studies',help='Summarize every study.db under a runs root; read-only sqlite, no DB creation')
    studies_p.add_argument('directory',type=Path)
    scmp=sub.add_parser('studies-compare',help='Cross-study rollup: budget to best, cost curves, per-parameter best-value shares; read-only sqlite')
    scmp.add_argument('paths',type=Path,nargs='+',help='study.db files or run directories containing one')
    cancel_p=sub.add_parser('run-cancel',help='Ask a study to stop before its next trial request; no process signalling')
    cancel_p.add_argument('directory',type=Path)
    for name,help_text in (('auto','Select methods within one explicit trial budget, then certify only the global winner'),
                           ('validate-auto','Validate automatic method selection without model loads or output writes')):
        command=sub.add_parser(name,help=help_text)
        command.add_argument('config',type=Path)
        command.add_argument('--set',action='append',default=[],metavar='FIELD=VALUE',
                             help='Override a JSON field before validation (dot path; value parsed as JSON, else string)')
        command.add_argument('--preset',help='Evidence-gated preset profile applied under the explicit config')
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
    claims=sub.add_parser('release-claims',help='Read a release evidence ledger (claims, evidence, chain digest)')
    claims.add_argument('directory',type=Path)
    claimadd=sub.add_parser('claim-add',help='Append an evidence-backed claim to a release evidence ledger')
    claimadd.add_argument('directory',type=Path)
    claimadd.add_argument('--claim',required=True,help='What is asserted')
    claimadd.add_argument('--evidence',required=True,help='Which artifact or run backs the assertion')
    claimadd.add_argument('--verified-by',required=True,help='Which gate or human produced it')
    claimadd.add_argument('--expected-identity')
    keygen=sub.add_parser('release-keygen',help='Create an Ed25519 release-signing key (openssl ed25519)')
    keygen.add_argument('key',type=Path)
    signr=sub.add_parser('release-sign',help='Sign a verified release reproduction index (repro.json payload)')
    signr.add_argument('directory',type=Path)
    signr.add_argument('--secret',type=Path,required=True,help='Ed25519 private key from release-keygen')
    verifyr=sub.add_parser('release-verify-signature',help='Verify a release signature; --public pins it to a known key')
    verifyr.add_argument('directory',type=Path)
    verifyr.add_argument('--public',type=Path,help='Pinned public key; without this only signature math is checked')
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
            if args.preset:
                from .presets import apply_preset
                obj=apply_preset(obj,args.preset)
            if args.set:
                from .overrides import apply_overrides
                obj=apply_overrides(obj,args.set)
            config=AutoConfig.from_json(obj)
            result=validate_auto(config) if args.command=='validate-auto' else run_auto(config)
        elif args.command=='presets':
            from .presets import available_presets
            result={'presets':available_presets(),
                    'rule':'Presets without an evidence reference are refused at load time.'}
        elif args.command=='abliterate':
            from .abliterate import build_config
            from .auto import AutoConfig,run_auto,validate_auto
            document=build_config(args.model,args.suites,args.output,args.policy,args.total_trials,args.rank,
                                  args.revision,args.license_id,args.defaults,args.preset,args.set)
            config=AutoConfig.from_json(document)
            result=validate_auto(config) if args.dry_run else run_auto(config)
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
        elif args.command=='judge-ensemble':
            from .agent import judge_ensemble
            result=judge_ensemble(args.suite,args.directory,args.min_agreement)
        elif args.command=='calibration-report':
            from .agent import calibration_report
            result=calibration_report(args.suite,args.directory,args.min_agreement,args.ledger)
        elif args.command=='verdict-audit':
            from .agent import verdict_audit
            result=verdict_audit(args.directory,args.limit)
        elif args.command=='score-sensitivity':
            from .agent import score_sensitivity
            result=score_sensitivity(args.directory,args.min_confidences or (0.5,0.7,0.9))
        elif args.command=='score-compare':
            from .agent import score_compare
            result=score_compare(args.suite,args.before,args.after,args.pass_on)
        elif args.command=='suites-compare':
            from .agent import suites_compare
            result=suites_compare(args.families,args.pass_on)
        elif args.command=='gguf-inspect':
            from .agent import gguf_inspect
            result=gguf_inspect(args.file,args.other)
        elif args.command=='gguf-edit':
            from .agent import gguf_edit
            if not args.tensor and args.delta_gguf is None:
                raise ValueError('provide at least one --tensor NAME=DELTA.NPY or a --delta-gguf')
            result=gguf_edit(args.input,args.output,args.tensor,args.delta_gguf,args.strength)
        elif args.command=='gguf-meta':
            from .agent import gguf_meta
            result=gguf_meta(args.input,args.output,args.set,args.unset)
        elif args.command=='gguf-delta-pack':
            from .agent import gguf_delta_pack
            result=gguf_delta_pack(args.output,args.specs)
        elif args.command=='direction-export':
            from .agent import direction_export
            result=direction_export(args.registry,args.key,args.output,args.dtype)
        elif args.command=='quants-audit':
            from .agent import quants_audit
            result=quants_audit(args.file,args.blocks)
        elif args.command=='gguf-tensor-diff':
            from .agent import gguf_tensor_diff
            result=gguf_tensor_diff(args.before,args.after)
        elif args.command=='safetensors-diff':
            from .agent import safetensors_tensor_diff
            result=safetensors_tensor_diff(args.before,args.after)
        elif args.command=='pt-diff':
            from .agent import pt_tensor_diff
            result=pt_tensor_diff(args.before,args.after,args.limit)
        elif args.command=='fp8-edit':
            from .fp8 import safetensors_fp8_edit
            result=safetensors_fp8_edit(args.input,args.output,getattr(args,'format'),args.weights,args.delta,
                                        args.scales,args.scale_mode,args.axis)
        elif args.command=='nf4-edit':
            from .nf4 import safetensors_nf4_edit
            result=safetensors_nf4_edit(args.input,args.output,args.weights,args.delta,args.scales,args.blocksize)
        elif args.command=='fp4-edit':
            from .fp4 import safetensors_fp4_edit
            if args.output.exists():
                raise FileExistsError(args.output)
            result=safetensors_fp4_edit(args.input,args.output,getattr(args,'format'),args.weights,args.delta,
                                        args.scales,args.global_scale_tensor,args.global_scale)
        elif args.command=='suite-near-dup':
            from .agent import suite_near_dup
            result=suite_near_dup(args.suite,args.threshold)
        elif args.command=='suite-diff':
            from .agent import suite_diff
            result=suite_diff(args.before,args.after)
        elif args.command=='suite-stats':
            from .agent import suite_stats
            result=suite_stats(args.suite)
        elif args.command=='corpus-cache':
            from .sources import corpus_cache_prune, corpus_cache_report
            result=corpus_cache_prune(args.prune_older_than,args.execute) if args.prune_older_than else corpus_cache_report()
        elif args.command=='run-events':
            from .events import read_events
            result={'events':read_events(args.output/'events.jsonl',args.tail)}
        elif args.command=='study-metrics':
            from .agent import study_metrics
            result=study_metrics(args.run)
        elif args.command=='schemas':
            from .schemas import SCHEMAS
            result={'schemas':[{'file':name,'description':schema['description']} for name,schema in sorted(SCHEMAS.items())]}
        elif args.command=='verify-artifact':
            from .schemas import validate_artifact, verify_all
            result=verify_all(args.file) if args.all else validate_artifact(args.file,args.name)
        elif args.command=='release-diff':
            from .release import release_diff
            result=release_diff(args.left,args.right)
        elif args.command=='robustness-report':
            from .agent import robustness_report
            result=robustness_report(args.eval,args.pass_on)
        elif args.command=='family-decay':
            from .agent import family_decay
            result=family_decay(tuple(args.fitted),tuple(args.held),args.pass_on)
        elif args.command=='plan-register':
            from .registry import register_plan
            result=register_plan(args.registry,args.run_output,args.gates,args.allow_unmeasured)
        elif args.command=='plan-list':
            from .registry import list_plans
            result=list_plans(args.registry)
        elif args.command=='plan-show':
            from .registry import _read_json
            result=_read_json(Path(args.registry)/'plans'/args.key/'meta.json')
        elif args.command=='plan-load':
            from .registry import load_plan
            loaded=load_plan(args.registry,args.key,dict(zip(('layers','rank','hidden'),args.expect)) if args.expect else None,args.device)
            result={'key':args.key,'geometry':loaded['geometry'],'measured':loaded['meta']['measured'],
                    'method':loaded['meta']['method'],'meta_digest':loaded['meta']['meta_digest']}
        elif args.command=='plan-bundle':
            from .registry import bundle_plan
            result=bundle_plan(args.registry,args.key,args.output)
        elif args.command=='plan-unbundle':
            from .registry import unbundle_plan
            result=unbundle_plan(args.bundle,args.into)
        elif args.command=='plan-prune':
            from .registry import prune_registry
            result=prune_registry(args.registry,args.execute)
        elif args.command=='studies-compare':
            from .agent import studies_compare
            result=studies_compare(args.paths)
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
        elif args.command=='hub-census':
            from .hubcensus import hub_census
            result=hub_census(authors=args.author or None,since=args.since,per_author=args.per_author)
        elif args.command=='commands':
            if args.completions:
                print(completion_script(parser,args.completions),end='')
                return
            result=command_manifest(parser)
        elif args.command=='explain-error':
            from .errors import explain_error
            result=explain_error(args.code)
        elif args.command=='run-ls':
            from .agent import run_ls
            result=run_ls(args.output)
        elif args.command=='run-doctor':
            from .agent import run_doctor
            result=run_doctor(args.output)
        elif args.command in {'generate','score','status'}:
            from .job_cli import generate_job, score_job, status_job
            result={'generate':generate_job,'score':score_job,'status':status_job}[args.command](args)
        elif args.command=='verify-release':
            from .release import verify_release
            result=verify_release(args.directory,args.expected_identity)
        elif args.command=='release-claims':
            from .release import evidence_summary
            result=evidence_summary(args.directory)
        elif args.command=='claim-add':
            from .release import add_claim
            result=add_claim(args.directory,args.claim,args.evidence,args.verified_by,args.expected_identity)
        elif args.command=='release-keygen':
            from .release import keygen
            result=keygen(args.key)
        elif args.command=='release-sign':
            from .release import sign_release
            result=sign_release(args.directory,args.secret)
        elif args.command=='release-verify-signature':
            from .release import verify_signature
            result=verify_signature(args.directory,args.public)
        elif args.command=='doctor':
            from .pipeline import configure_compiler_paths
            import os
            configure_compiler_paths()
            import torch
            from .hf import resolve_backend, backend_device, device_synchronize, CUDA_SECONDARY_NOTE
            backend=resolve_backend(allow_secondary=os.environ.get('B70ABLATOR_ALLOW_CUDA')=='1')
            device=backend_device(backend)
            x=torch.arange(16,device=device,dtype=torch.float32).reshape(4,4)
            product=x@x.T
            device_synchronize(device)
            result={'torch':torch.__version__,'device':backend['name'],
                    'compute_backend':{'type':backend['type'],'rank':backend['rank']},
                    'kernel_sum':float(product.sum()),'kernel_finite':bool(torch.isfinite(product).all()),
                    'cpu_fallback':False,
                    'note':CUDA_SECONDARY_NOTE if backend['rank']!='primary' else 'B70 primary backend; CUDA is second-rank and was not used here.'}
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
        from .schemas import EMIT_SCHEMAS, enforce
        if args.command in EMIT_SCHEMAS:
            enforce(args.command,result)
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,RuntimeError,OSError,sqlite3.Error,subprocess.SubprocessError) as e:
        from .errors import error_code
        print(json.dumps({'error':{'code':error_code(e),'type':type(e).__name__,'message':str(e) or repr(e),'command':args.command}}),file=sys.stderr)
        raise SystemExit(1) from e


if __name__=='__main__':
    main()
