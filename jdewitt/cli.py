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
    prog = parser.prog or 'jdewitt'
    if prog.endswith('__main__.py') or '/' in prog:
        prog = 'jdewitt'
    safe_prog = prog.replace('-', '_').replace('.', '_')
    subactions = None
    for action in parser._actions:
        if isinstance(action, _argparse._SubParsersAction):
            subactions = action
    if subactions is None:
        raise ValueError('parser has no subcommands')
    commands = []
    per_command_opts = {}
    helped = {subaction.dest: subaction.help for subaction in subactions._get_subactions()}
    for name, child in subactions.choices.items():
        opts = []
        for argument in child._actions:
            if argument.dest == 'help' or argument.help == _argparse.SUPPRESS:
                continue
            opts.extend(argument.option_strings or [])
            if argument.metavar and not argument.option_strings:
                opts.append('<' + argument.metavar.rstrip(']') + '>')
        per_command_opts[name] = ' '.join(sorted(set(opts)))
        help_text = (helped.get(name) or child.description or '').replace(':', '-')
        commands.append((name, help_text))
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
            # A single quote in the help text closes the zsh array element and makes
            # the whole completion file fail to parse; escape it the zsh way.
            lines.append("        '" + name + ':' + help_text.replace("'", "'\\''") + "'")
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
        helped = {subaction.dest: subaction.help for subaction in action._get_subactions()}
        for name, child in action.choices.items():
            arguments = []
            for argument in child._actions:
                if argument.dest == 'help' or argument.help == _argparse.SUPPRESS:
                    continue
                choices = getattr(argument, 'choices', None)
                arguments.append({'flags': list(argument.option_strings or [argument.dest]),
                                  'dest': argument.dest, 'required': bool(argument.required),
                                  'choices': [str(choice) for choice in choices] if choices else None})
            commands.append({'command': name, 'help': helped.get(name) or child.description,
                             'arguments': arguments, 'schema_enforced': name in EMIT_SCHEMAS})
    commands.sort(key=lambda item: item['command'])
    return {'scope': 'Self-description of the installed CLI; anything marked schema_enforced fails its own command if the writer drifts.',
            'commands': commands,
            'exit_codes': {'success': 0, 'stable_failure': 1, 'argument_error': 2},
            'error_codes': [dict(row, code=code) for code, row in sorted(EXPLAINER.items())],
            'note': 'Failures print {"error":{"code","type","message","command"}}. explain-error CODE prints one row of the same table.'}


def build_parser():
    """Construct the full argparse tree without importing any command's runtime.

    Kept separate from main() so auditing tools can walk the same parser the
    installed CLI uses; building it must stay model-free (see
    scripts/help_audit.py and tests/test_agent_interface.py).
    """
    parser=AgentParser(description='B70-only ablation with real answer and quantized release gates')
    from . import __version__
    parser.add_argument('--version',action='version',version=f'jdewitt {__version__}')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor',help='Run a real XPU kernel on the B70')
    sub.add_parser('about',help='Installation/provenance metadata as JSON, no model loads')
    sub.add_parser('methods',help='List implemented and researched methods as JSON, no model loads')
    sub.add_parser('architectures',help='List adapter layouts and verification limits as JSON')
    hub=sub.add_parser('hub-census',help='Live Hub text-generation census joined to the adapter table; adapter worklist ordered by downloads, no model loads')
    hub.add_argument('--since',default='2026-08-01',help='Only repositories created on or after this UTC date')
    hub.add_argument('--author',action='append',help='Repeat for publishers to scan; default is the architecture inventory authors')
    hub.add_argument('--per-author',type=int,default=40,help='Cap newest text-generation repositories scanned per publisher')
    hub.add_argument('--out',type=Path,help='Persist the same JSON to this path; refuses an existing file (FILE_EXISTS)')
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
    esmoke=sub.add_parser('export-smoke',help='Serve one exported GGUF with a real llama-server on a free port: header identity, /health, /props path check, one greedy completion, one JSON; exit 0 iff served (-ngl 0, CPU)')
    esmoke.add_argument('--gguf',type=Path,required=True)
    esmoke.add_argument('--server',type=Path,help='llama-server binary; required unless --no-server')
    esmoke.add_argument('--ctx',type=int,default=4096)
    esmoke.add_argument('--max-tokens',type=int,default=16)
    esmoke.add_argument('--out',type=Path,default=Path('export-smoke.json'),help='JSON report path; the server log is written beside it')
    esmoke.add_argument('--server-arg',action='append',default=[],metavar='ARG',help='Extra llama-server argv token, passed in order and recorded; use --server-arg=--flag for dash-leading values (repeatable)')
    esmoke.add_argument('--no-server',action='store_true',help='Header half only: servable is "unmeasured", never true')
    acard=sub.add_parser('adapter-card',help='Label a shipped LoRA adapter (R8): bind it to its base artifact, record the merged twin fidelity and BOTH delta_br readings')
    acard.add_argument('--adapter',type=Path,required=True,help='shipped LoRA adapter GGUF (must carry jdewitt.base_artifact_sha256)')
    acard.add_argument('--base',type=Path,required=True,help='base artifact the adapter was fitted on; its sha256 must match the binding')
    acard.add_argument('--twin',type=Path,required=True,help='merged twin fidelity record: per-tensor delivered cos/SNR, artifact sha256, edited/untouched counts, edit fingerprint, both delta_br readings')
    acard.add_argument('--out',type=Path,required=True,help='adapter.json to write beside the shipped adapter (refuses to overwrite)')
    acard.add_argument('--scale',type=float,default=None,help='ladder scale the adapter shipped at (default: scale/edit_fingerprint.scale from the twin record)')
    editm.add_argument('input',type=Path)
    editm.add_argument('output',type=Path)
    editm.add_argument('--set',action='append',default=[],metavar='KEY=JSON',
                       help='Set metadata key to a JSON value (string/number/bool/array); repeatable')
    editm.add_argument('--unset',action='append',default=[],metavar='KEY',
                       help='Remove a metadata key; repeatable')
    gdiff=sub.add_parser('gguf-tensor-diff',help='Compare two GGUF files tensor-by-tensor: byte identity plus dequant error, no model')
    gdiff.add_argument('before',type=Path)
    gdiff.add_argument('after',type=Path)
    vf=sub.add_parser('verify',help='Recompute an edit fingerprint from pristine+edited GGUF (or --other: same-edit test); exit 1 unless recomputes/same edit; no model')
    vf.add_argument('--pristine',type=Path,required=True)
    vf.add_argument('--edited',type=Path,required=True)
    vf.add_argument('--claim',type=Path,help='Fingerprint JSON block to check field by field')
    vf.add_argument('--other',type=Path,help='Second edited artifact on the same base: same-edit test')
    vf.add_argument('--rows',type=int,default=8,help='Sampled rows per changed tensor, plus row 0')
    vf.add_argument('--threshold',type=float,default=0.98,help='Median cos for same edit')
    vf.add_argument('--min-overlap',type=float,default=0.5,help='Changed-set overlap for same edit')
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
    nf4.add_argument('--scales',help='F32 or F16 absmax tensor name (one per 64-element block); required unless --quant-state is given')
    nf4.add_argument('--blocksize',type=int,default=64)
    nf4.add_argument('--quant-state',help='double-quantized: name of the packed quant_state.bitsandbytes__nf4 JSON blob; then --scales is the U8 quantized absmax and --nested-absmax the F32 nested absmax')
    nf4.add_argument('--nested-absmax',help='double-quantized: F32 nested absmax tensor name')
    nf4.add_argument('--nested-blocksize',type=int,default=256,help='double-quantized: block size of the absmax quantization (bnb default 256)')
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
    compare.add_argument('--allow-config-mismatch',action='store_true',
        help='compare waves whose recorded serve_config differs (R9 says they are not comparable)')
    compare.add_argument('suite',type=Path)
    compare.add_argument('before',type=Path)
    compare.add_argument('after',type=Path)
    compare.add_argument('--pass-on',choices=['compliance','compliance+partial'],default='compliance')
    compare.add_argument('--kind',choices=['all','refusal','math','code','harmless'],default='all',help='Restrict the paired set to one prompt kind; math/code is the retention axis')
    scmp2=sub.add_parser('suites-compare',help='Several score-compare families with Holm-adjusted sign-test p-values so a lucky suite cannot hide the family-wise cost; no models')
    scmp2.add_argument('--allow-config-mismatch',action='store_true',
        help='pass through to each score_compare (R9: mismatched serve configs are not comparable)')
    scmp2.add_argument('families',type=Path,nargs='+',metavar='SUITE BEFORE AFTER ...')
    scmp2.add_argument('--pass-on',choices=['compliance','compliance+partial'],default='compliance')
    scmp2.add_argument('--kind',choices=['all','refusal','math','code','harmless'],default='all',help='Restrict every family to one prompt kind; math/code is the retention axis')
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
    claimadd.add_argument('--successes',type=int,help='Numerator when the claim states a rate; without it the first n/m in the claim text is read and marked inferred')
    claimadd.add_argument('--total',type=int,help='Denominator; give it with --successes')
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
    evaluate.add_argument('--secondary-device',action='store_true',help='Serve on a non-B70 Vulkan device as a second-rank test backend; the B70 is still used whenever it is visible')
    evaluate.add_argument('--sandbox-image',required=True)
    evaluate.add_argument('--output',type=Path,required=True)
    capture=sub.add_parser('capture',help='Capture labelled residual records off any checkpoint: decoder stack and residual-write projections are discovered from the model itself, answers are labelled by their shape; writes spanfit records + one JSON report')
    capture.add_argument('--model',required=True,help='local path or Hub id of the base checkpoint')
    capture.add_argument('--revision',default=None)
    capture.add_argument('--suite',type=Path,required=True,help='jsonl with id/kind/prompt per row')
    capture.add_argument('--out',type=Path,required=True,help='records .pt for spanfit')
    capture.add_argument('--gguf',type=Path,default=None,help='artifact to cross-check the capture against; enables the exact spanfit merge command')
    capture.add_argument('--kinds',default=','.join(('refusal','harmless')),help='comma list of suite kinds to capture')
    capture.add_argument('--locus',choices=('answer','last-prompt','first-answer'),default='answer',help='where in the residual stream the axis is read (METHODOLOGY R4)')
    capture.add_argument('--max-tokens',type=int,default=512)
    capture.add_argument('--ctx-size',type=int,required=True)
    capture.add_argument('--hidden',type=int,default=None,help='residual width when the config does not state one')
    capture.add_argument('--limit',type=int,default=None)
    capture.add_argument('--system',default='You are a helpful assistant.')
    capture.add_argument('--raw-prompt',action='store_true',help='send the prompt untemplated instead of through the chat template')
    capture.add_argument('--trust-remote-code',action='store_true')
    capture.add_argument('--allow-cuda',action='store_true',help='allow the second-rank CUDA backend')
    capture.add_argument('--dtype',choices=('bfloat16','float16'),default='bfloat16')
    capture.add_argument('--settle-min-tokens',type=int,default=24,help='do not settle before this many generated tokens')
    capture.add_argument('--no-settle',dest='settle',action='store_false',help='always decode to --max-tokens (the token-expensive, fully-determined label path)')
    capture.add_argument('--answers',type=Path,default=None,help='jsonl with id+answer (a banked predictions.jsonl): reuse those answers and spend ZERO output tokens on those rows')
    tune=sub.add_parser('tune',help='Rank edit arms by perplexity and refusal-branch preference before generating anything: adapters + --lora-scaled, prefill only, then re-measure the winner on the merged bytes')
    tune.add_argument('--gguf',type=Path,required=True)
    tune.add_argument('--hf',default=None,help='HF checkpoint dir for the PRIMAL adapter basis: the adapter curve only predicts the merged bytes when its factors come from un-folded weights')
    tune.add_argument('--records',type=Path,required=True)
    tune.add_argument('--perplexity',default=None,help='llama-perplexity binary (default: the repo CPU build)')
    tune.add_argument('--control',default=None,help='benign corpus for generic drift (default: derived from the records)')
    tune.add_argument('--refusal',default=None,help='refusal-continuation corpus (default: derived from the model\'s own declining answers)')
    tune.add_argument('--delivery',default=None,help='delivery-continuation corpus (default: derived from the model\'s own compliant answers)')
    tune.add_argument('--corpus-chars',type=int,default=1500,help='characters per derived corpus: this is what keeps one reading at a few seconds, so the loop iterates in about a minute')
    tune.add_argument('--shapes',default='perlayer,shared')
    tune.add_argument('--footprints',default='all',help="layer footprints to search, 'all' or a-b ranges (e.g. 'all,10-27,16-27'): the footprint is the strongest damage lever and costs ZERO generated tokens to test")
    tune.add_argument('--ppl-ngl',type=int,default=None,help='GPU layers for the perplexity ladder (default: same as --ngl); the ladder is compute-bound, so a GPU makes the free part of the search fast too')
    tune.add_argument('--best-effort',action='store_true',help='when no arm leaves zero refusals, ship the arm with the lowest refusal RATE over the confirmed rows (ties: fewest undecided, then damage); the artifact records certified=false and the residual row ids; never on by default')
    tune.add_argument('--rank',type=int,default=1,help='directions per layer in the ladder: rank>1 searches the top-k contrast subspace instead of a single direction')
    tune.add_argument('--scales',default='1.0,1.5')
    tune.add_argument('--profile',choices=('flat','bell','manual'),default='flat',help='per-layer strength profile over each footprint; bell/manual also run flat and two bell widths at the same scales so the comparison is like-for-like')
    tune.add_argument('--profile-centre',type=float,default=0.5,help='bell peak position as a fraction of the footprint (0 first layer, 1 last)')
    tune.add_argument('--profile-width',type=float,default=0.5,help='bell: footprint fraction from the centre at which the weight reaches --profile-min')
    tune.add_argument('--profile-min',type=float,default=0.3,help='bell weight at and beyond --profile-width (before --profile-mean)')
    tune.add_argument('--profile-spec',default=None,help="manual weights 'LAYER=WEIGHT,...' e.g. '10=1.0,11=0.8'; unlisted footprint layers get 1.0")
    tune.add_argument('--profile-mean',choices=('keep','one'),default='one',help="'one' rescales weights to mean 1.0 over the footprint (same total edit as flat); 'keep' uses them as given")
    tune.add_argument('--drift-budget',type=float,default=0.02,help='max |control ln-PPL drift| an arm may cost')
    tune.add_argument('--min-edit-cos',type=float,default=0.9,help='reject an arm whose edit does not survive the artifact codec: cos(realized, intended) floor, measured by a real requant round-trip on sampled tensors (0 disables)')
    tune.add_argument('--max-scale',type=float,default=12.0,help='upper bound for the codec-floor bisection that runs when every arm is rejected by the quantizer')
    tune.add_argument('--axis-space',choices=('output','input'),default='output')
    tune.add_argument('--ctx',type=int,default=512)
    tune.add_argument('--threads',type=int,default=16)
    tune.add_argument('--parallel',type=int,default=3,help='perplexity readings run concurrently (mirrors the ladder: one run of wall clock instead of one per arm)')
    tune.add_argument('--out',type=Path,default=None)
    tune.add_argument('--no-verify',action='store_true',help='skip the merged re-measurement')
    tune.add_argument('--merged-screen',action='store_true',help='after the merge, screen the MERGED bytes (server reload, no adapter): the only verdict that belongs to the shipped file')
    tune.add_argument('--apply-quantized',action='store_true',help='DEFAULT deliverable: write the edit into the artifact the user runs (only touched tensors re-encoded, everything else byte-identical)')
    tune.add_argument('--out-gguf',type=Path,default=None,help='where --apply-quantized writes the shipped artifact (default: <out>/edited-<family>@<scale>-<artifact>)')
    tune.add_argument('--serve-type',default=None,help='faithful pipeline: merge into --merge-gguf (16-bit), then requantize to this serving type while the edited tensors are held at --edit-type')
    tune.add_argument('--merge-gguf',type=Path,default=None,help='16-bit artifact to merge into (default: --gguf); the edit is exact there, unlike at 4-bit')
    tune.add_argument('--edit-type',default='q8_0',help='type held for the tensors that carry the edit in the requantized serving file')
    tune.add_argument('--quantize-binary',default=None,help='llama-quantize binary (default: first known build or JDEWITT_QUANTIZE)')
    tune.add_argument('--renormalize',action='store_true',help='preserve each edited row magnitude (Heretic "full" mode): remove the direction, keep the output scale')
    tune.add_argument('--keep-merged',action='store_true',help='leave the winning edit on the artifact')
    tune.add_argument('--token-budget',type=int,default=0,help='output tokens the screen stage may spend (0 = prefill-only ladder; the ladder itself always costs zero tokens)')
    tune.add_argument('--serve',default=None,help='llama-server binary used by the screen stage (default: the repo CUDA build)')
    tune.add_argument('--serve-lib',default=None,help="colon-separated dirs prepended to the server's LD_LIBRARY_PATH (e.g. the nccl dir the build links against)")
    tune.add_argument('--screen-rows',type=int,default=8,help='harmful rows decoded per arm in the screen stage (the CONFIRMATION count; elimination uses --probe-rows)')
    tune.add_argument('--probe-rows',type=int,default=1,help='rows used to ELIMINATE an arm: a refusal settles on one row, so one row decides most arms and the full set is only paid for the arm that is about to ship')
    tune.add_argument('--screen-k',type=int,default=100,help='tokens decoded per screened row (K=100 measured zero false delivery-settles)')
    tune.add_argument('--screen-ctx',type=int,default=4096,help='context the screen server is started with')
    tune.add_argument('--escalate-to',type=int,default=600,help='window cap per row in tokens: undecided rows are continued (window doubling) until they settle or reach this cap')
    tune.add_argument('--ngl',type=int,default=99,help='gpu layers for the screen server')
    tune.add_argument('--front',type=int,default=3,help='how many Pareto arms the screen stage may decode')
    tune.add_argument('--rounds',type=int,default=1,help='ladder rounds; later rounds grow the strength grid by --grow')
    tune.add_argument('--grow',type=float,default=1.5,help='factor the scale grid is multiplied by on the next round')
    tune.add_argument('--refine-steps',type=int,default=0,help='after an arm clears the screen, walk its scale DOWN by --refine-factor and re-screen: lowest-perplexity strength that still clears (0 = off)')
    tune.add_argument('--refine-factor',type=float,default=0.75,help='multiplier for each strength-reduction step')
    tune.add_argument('--settle-chars',type=int,default=300,help='at the window cap, a non-declining row with at least this many characters becomes a delivery CANDIDATE (still agent-read, never a pass)')
    tune.add_argument('--refine-floor',type=float,default=0.1,help='the strength reduction stops here')
    requant=sub.add_parser('requant',help='Requantize an artifact, holding the tensors that carry an edit at a type where the edit survives (fidelity-driven per-tensor types)')
    requant.add_argument('--gguf',type=Path,required=True)
    requant.add_argument('--out',type=Path,required=True)
    requant.add_argument('--type',required=True,help='serving type for everything not overridden (q4_k_m, q6_k, q8_0, f16, ...)')
    requant.add_argument('--keep-tensor',action='append',default=[],help='NAME=TYPE for a tensor that must stay high-precision (repeatable)')
    requant.add_argument('--binary',default=None,help='llama-quantize binary (default: first known local build, or JDEWITT_QUANTIZE)')
    requant.add_argument('--threads',type=int,default=None)
    requant.add_argument('--imatrix',default=None,help='importance matrix for IQ-family targets (they refuse without one)')
    editq=sub.add_parser('edit-quantized',help="Apply a fitted axis to ANY quantized GGUF (any source quant type): only the edited tensors are re-encoded, every other tensor is copied byte for byte so your calibration survives")
    editq.add_argument('--gguf',type=Path,required=True)
    editq.add_argument('--out',type=Path,required=True)
    editq.add_argument('--records',type=Path,required=True)
    editq.add_argument('--shape',choices=('perlayer','shared'),default='perlayer')
    editq.add_argument('--rank',type=int,default=1,help='directions per layer: rank-k removes the layer top-k contrast subspace (each extra direction is one more thing the file must store)')
    editq.add_argument('--scale',type=float,default=1.0)
    editq.add_argument('--layers',default=None,help="footprint 'a-b' (default: every discovered layer)")
    editq.add_argument('--edit-type',default='auto',help='type the edited tensors are re-encoded at: auto picks the smallest type whose delivered cos clears --min-cos (q6_k -> q8_0 -> f16), or name one')
    editq.add_argument('--min-cos',type=float,default=0.95,help='delivered-edit fidelity floor used by --edit-type auto')
    editq.add_argument('--axis-space',choices=('output','input'),default='output')
    editq.add_argument('--renormalize',action='store_true')
    editq.add_argument('--repair',choices=('none','renorm','norms'),default='none',help="post-edit repair: renorm restores each edited row's pre-edit norm; norms also scales the edited layers' F32 norm weights by the median row ratio (clamped 0.9..1.1)")
    editq.add_argument('--no-verify',action='store_true',help='skip the byte-identity check of untouched tensors')
    from .board import TRUST
    board=sub.add_parser('board',help='Behaviour board: refusal_rate by family, capability delta in nats, coherence flags and token cost for one GGUF; needs llama-server',
                         description=TRUST,formatter_class=argparse.RawDescriptionHelpFormatter)
    board.add_argument('--artifact',type=Path,required=True,help='GGUF to score')
    board.add_argument('--records',type=Path,required=True,help='capture records: harmful rows with family_id, harmless text for the capability corpus')
    board.add_argument('--base',type=Path,default=None,help='pristine GGUF: capability_delta_nats = artifact minus base (no base, no clear)')
    board.add_argument('--adapter',type=Path,default=None,help='LoRA GGUF applied on the artifact; the board raises if the server does not apply it')
    board.add_argument('--adapter-scale',type=float,default=1.0)
    board.add_argument('--families',type=int,default=8,help='families on the board (first ids that hold --prompts rows)')
    board.add_argument('--prompts',type=int,default=8,help='prompts per family')
    board.add_argument('--tokens',type=int,default=128,help='tokens decoded per row')
    board.add_argument('--corpus-chars',type=int,default=20000,help='capability corpus size in characters, cut from harmless record text')
    board.add_argument('--corpus',type=Path,default=None,help='pinned capability corpus text file (replaces the records-derived one)')
    board.add_argument('--perplexity',default=None,help='llama-perplexity binary (default: the repo CPU build)')
    board.add_argument('--serve',default=None,help='llama-server binary (default: the repo CUDA build)')
    board.add_argument('--serve-lib',default=None,help="colon-separated dirs prepended to the server's LD_LIBRARY_PATH")
    board.add_argument('--ngl',type=int,default=99,help='gpu layers for the server')
    board.add_argument('--threads',type=int,default=16,help='threads for the perplexity readings')
    board.add_argument('--out',type=Path,required=True,help='report json path')
    revert=sub.add_parser('gguf-revert',help='Restore the pre-patch bytes an --in-place merge saved; refuses if the artifact changed since')
    revert.add_argument('artifact',type=Path)
    revert.add_argument('sidecar',type=Path)
    from .arguments import add_arguments
    add_arguments(sub)
    return parser


def main():
    parser=build_parser()
    args=parser.parse_args()
    try:
        exit_code=None
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
            result=score_compare(args.suite,args.before,args.after,args.pass_on,args.kind,args.allow_config_mismatch)
        elif args.command=='suites-compare':
            from .agent import suites_compare
            result=suites_compare(args.families,args.pass_on,args.kind,args.allow_config_mismatch)
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
        elif args.command=='export-smoke':
            from .export_smoke import export_smoke_command
            result=export_smoke_command(args)
            exit_code=0 if result['verdict']['servable'] is True else 1
        elif args.command=='adapter-card':
            from .adapter_card import build_adapter_card
            result=build_adapter_card(args.adapter,args.base,args.twin,args.out,args.scale)
        elif args.command=='verify':
            from .verify import verify_files
            result=verify_files(args.pristine,args.edited,args.claim,args.other,args.rows,args.threshold,args.min_overlap)
            if result['verdict'] not in ('recomputes','same edit'):
                print(json.dumps(result,indent=2,allow_nan=False))
                raise SystemExit(1)
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
            from .nf4 import safetensors_nf4_edit, safetensors_nf4_edit_double
            if args.quant_state:
                if not args.nested_absmax:
                    raise ValueError('--quant-state needs --nested-absmax (the F32 nested absmax tensor)')
                result=safetensors_nf4_edit_double(args.input,args.output,args.weights,args.delta,
                                                   args.scales,args.nested_absmax,args.quant_state,
                                                   args.blocksize,args.nested_blocksize)
            elif args.scales:
                result=safetensors_nf4_edit(args.input,args.output,args.weights,args.delta,args.scales,args.blocksize)
            else:
                raise ValueError('nf4-edit needs --scales (single quant) or --quant-state + --nested-absmax (double quant)')
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
            result=hub_census(authors=args.author or None,since=args.since,per_author=args.per_author,out=args.out)
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
            result=add_claim(args.directory,args.claim,args.evidence,args.verified_by,args.expected_identity,args.successes,args.total)
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
            backend=resolve_backend(allow_secondary=os.environ.get('JDEWITT_ALLOW_CUDA')=='1')
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
        elif args.command=='capture':
            from .capture import capture_command
            result=capture_command(args)
        elif args.command=='edit-quantized':
            from .spanfit import load_records,parse_layer_ids,per_layer_axes,per_layer_subspace,shared_axis
            from .gguf_adapters import discover_writeback_targets,gguf_weight_shapes,inferred_hidden_width
            from .errors import NoVerifiedAdapter
            from .ggufio import compare_edit
            from .rewrite import rewrite_with_edits
            records=load_records(args.records)
            if args.rank > 1:
                axes=per_layer_subspace(records,args.rank)
            else:
                axes=(shared_axis(records)[0] if args.shape=='shared' else per_layer_axes(records))
            width=int(axes.shape[-1])
            shapes=gguf_weight_shapes(str(args.gguf))
            discovered=discover_writeback_targets(shapes,width)
            layers=sorted(discovered['targets'])
            if not layers:
                raise NoVerifiedAdapter(f'{args.gguf} carries no residual-write projection at width {width}: '
                                        f'its own writeback names put the hidden width at {inferred_hidden_width(shapes)}, '
                                        f'and width-matching tensors it would not edit are '
                                        f'{discovered["unmapped_writeback_width"][:8] or "none"}. The records and the '
                                        f'artifact have to belong to the same model; list what the file carries with '
                                        f'`python -m jdewitt.spanfit targets --gguf-in {args.gguf}`',
                                        width=width, records_width=width,
                                        artifact_hidden_width=inferred_hidden_width(shapes),
                                        unmapped_writeback_width=discovered['unmapped_writeback_width'],
                                        artifact=str(args.gguf))
            if args.layers:
                wanted=set(parse_layer_ids(args.layers))
                layers=[layer for layer in layers if layer in wanted]
            edits={}
            for layer in layers[:int(axes.shape[0])]:
                for name in discovered['targets'][layer]:
                    edits[name]=(axes[layer].numpy(),args.axis_space)
            if not edits:
                raise NoVerifiedAdapter(f'the footprint selects no tensor this artifact carries: {len(layers)} '
                                        f'discovered layer(s) survive the --layers filter, so the axes '
                                        f'({int(axes.shape[0])} layer rows) and the footprint do not cover the same '
                                        f'layers. Discovered layers: {sorted(discovered["targets"])[:8]}; pass a '
                                        f'--layers range the artifact actually has, do not edit a partial model and '
                                        f'call it done',
                                        discovered_layers=sorted(discovered['targets']), width=width,
                                        artifact=str(args.gguf))
            sample=list(edits)[:1]+list(edits)[-1:]
            probe={name:edits[name] for name in sample}
            attempts=['q6_k','q8_0','f16'] if args.edit_type=='auto' else [args.edit_type]
            result,chosen=None,None
            for index,kind in enumerate(attempts):
                candidate=args.out if len(attempts)==1 else args.out.with_name(
                    f'{args.out.stem}-{kind}{args.out.suffix}')
                attempt=rewrite_with_edits(args.gguf,candidate,edits,scale=args.scale,
                                           axis_space=args.axis_space,edit_type=kind,
                                           renormalize=args.renormalize,repair=args.repair,
                                           verify_untouched=not args.no_verify)
                delivered=compare_edit(str(args.gguf),str(candidate),probe,
                                       renormalize=args.renormalize or args.repair!='none')
                attempt['delivered_edit']=delivered
                attempt['path']=str(candidate)
                if result is not None and candidate != args.out:
                    candidate.unlink(missing_ok=True)
                chosen=(attempt,candidate)
                if (delivered.get('min_cos') or 0.0)>=args.min_cos or index==len(attempts)-1:
                    break
            result,chosen_path=chosen
            if chosen_path != args.out:
                chosen_path.replace(args.out)
            result['path']=str(args.out)
            result['edit_type_chosen']=result['edit_type']
        elif args.command=='requant':
            from .quantize import quantize_file
            result=quantize_file(args.gguf,args.out,args.type,binary=args.binary,
                                 overrides=args.keep_tensor,threads=args.threads,
                                 imatrix=args.imatrix)
        elif args.command=='gguf-revert':
            from .ggufio import revert_inplace
            result=revert_inplace(args.artifact,args.sidecar)
        elif args.command=='tune':
            from .ppl import DEFAULT_BINARY
            from .tune import tune_command
            if args.ppl_ngl is None:
                args.ppl_ngl=args.ngl
            if args.perplexity is None:
                args.perplexity=DEFAULT_BINARY
            result=tune_command(args)
        elif args.command=='board':
            from .board import board_command
            result=board_command(args)
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
            with ManagedGGUF(args.model,args.server,args.ctx_size,args.max_tokens,args.output/'server.log',secondary=args.secondary_device) as client:
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
        if exit_code is not None:
            raise SystemExit(exit_code)
        # verify-artifact is a check, so an invalid artifact is a failing check, not
        # a report: scripts and release gates must see a non-zero exit. The body is
        # still printed first (valid:false + the named problems).
        if args.command=='verify-artifact' and not result.get('valid'):
            raise SystemExit(1)
    except (ValueError,RuntimeError,OSError,sqlite3.Error,subprocess.SubprocessError) as e:
        from .errors import error_code
        print(json.dumps({'error':{'code':error_code(e),'type':type(e).__name__,'message':str(e) or repr(e),'command':args.command}}),file=sys.stderr)
        raise SystemExit(1) from e


if __name__=='__main__':
    main()
