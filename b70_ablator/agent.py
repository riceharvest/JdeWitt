# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read-only agent operations plus a durable cancel flag. No model imports, device access or network calls."""
from collections import Counter
import json
from pathlib import Path
import sqlite3


def validate_run(path,overrides=()):
    from .search import RunConfig
    from .validation import validate_run_inputs
    from .architectures import ARCHITECTURES
    obj=json.loads(Path(path).read_text())
    if overrides:
        from .overrides import apply_overrides
        obj=apply_overrides(obj,overrides)
    config=RunConfig.from_json(obj)
    _,report=validate_run_inputs(config)
    for field in ('converter','quantizer','converter_python','llama_server'):
        if not Path(getattr(config,field)).is_file():
            raise FileNotFoundError(f'Missing {field}: {getattr(config,field)}')
    if config.imatrix and not Path(config.imatrix).is_file():
        raise FileNotFoundError(config.imatrix)
    source=Path(config.model)
    if source.is_dir():
        model_config=json.loads((source/'config.json').read_text())
        architecture=model_config.get('model_type')
        if architecture not in ARCHITECTURES:
            raise ValueError(f'No architecture adapter for {architecture!r}')
        weights=list(source.glob('*.safetensors'))
        if not weights:
            raise ValueError('Original safetensors are required for streaming export')
        report['source']={'architecture':architecture,'weight_bytes':sum(p.stat().st_size for p in weights)}
    else:
        report['source']={'model':config.model,'remote_metadata_checked':False}
    if config.scoring_mode=='judge':
        for field in ('judge_artifact','calibration'):
            if not Path(getattr(config,field)).is_file():
                raise FileNotFoundError(getattr(config,field))
    return report


def _trial_rows(db_path):
    if not db_path.exists():
        return []
    # mode=ro forbids accidental database creation or Optuna schema migration.
    with sqlite3.connect(db_path.as_uri()+'?mode=ro',uri=True,timeout=1) as connection:
        connection.execute('PRAGMA query_only=ON')
        rows=connection.execute('''SELECT t.number,t.state,v.value,a.value_json FROM trials t
            JOIN studies s ON s.study_id=t.study_id
            LEFT JOIN trial_values v ON v.trial_id=t.trial_id AND v.objective=0
            LEFT JOIN trial_user_attributes a ON a.trial_id=t.trial_id AND a.key='feasible'
            WHERE s.study_name='b70-ablator' ORDER BY t.number''').fetchall()
    return [{'number':n,'state':state,'value':value,'feasible':json.loads(feasible) is True if feasible else False}
            for n,state,value,feasible in rows]


def run_status(directory):
    root=Path(directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    trials=_trial_rows(root/'study.db')
    candidates=[t for t in trials if t['state']=='COMPLETE' and t['feasible'] and t['value'] is not None]
    return {'output':str(root),'trial_counts':dict(Counter(t['state'] for t in trials)),
            'trials':trials,'best_feasible':min(candidates,key=lambda t:t['value']) if candidates else None,
            'cancel_requested':(root/'CANCEL').is_file(),
            'release_present':(root/'release').is_dir(),'release_verified':False,
            'note':'RUNNING is an unfinished resumable trial, not proof of a live process. Use verify-release for bundle integrity.'}


def request_cancel(directory):
    root=Path(directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    path=root/'CANCEL'
    if path.exists():
        return {'output':str(root),'cancel':'already requested','file':str(path)}
    path.write_text('{}\n')
    return {'output':str(root),'cancel':'requested','file':str(path),
            'note':'The scheduler stops before asking for a new trial; a finished attempt is never restarted and unfinished attempts stay resumable. This does not signal any live process.'}


def studies(root):
    root=Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    summaries=[]
    for db in sorted(root.rglob('study.db')):
        run=db.parent
        rows=_trial_rows(db)
        feasible=[t for t in rows if t['state']=='COMPLETE' and t['feasible'] and t['value'] is not None]
        summaries.append({'output':str(run),'trial_counts':dict(Counter(t['state'] for t in rows)),
                          'best_feasible':min(feasible,key=lambda t:t['value']) if feasible else None,
                          'cancel_requested':(run/'CANCEL').is_file(),
                          'release_present':(run/'release').is_dir(),
                          'study_bytes':db.stat().st_size})
    return {'root':str(root),'study_count':len(summaries),'studies':summaries,
            'note':'Counts read from persisted state only; RUNNING does not prove a live process.'}


def describe_run(path,overrides=()):
    """Static budget/suite/family shape of a validated experiment. No model or GPU access."""
    from .search import RunConfig
    from .validation import validate_run_inputs
    obj=json.loads(Path(path).read_text())
    if overrides:
        from .overrides import apply_overrides
        obj=apply_overrides(obj,overrides)
    config=RunConfig.from_json(obj)
    suites,report=validate_run_inputs(config)

    def split_stats(rows):
        families=Counter(r['family_id'] for r in rows if isinstance(r.get('family_id'),str) and r['family_id'].strip())
        return {'tasks':len(rows),'kinds':dict(Counter(r['kind'] for r in rows)),
                'languages':dict(Counter(r['language'] for r in rows)),
                'family_count':len(families),
                'largest_families':sorted(families.items(),key=lambda kv:(-kv[1],kv[0]))[:5],
                'singletons':sum(1 for count in families.values() if count==1)}
    family_sets={name:{r['family_id'] for r in rows if isinstance(r.get('family_id'),str) and r['family_id'].strip()}
                 for name,rows in suites.items()}
    collisions=[]
    names=sorted(family_sets)
    for i,first in enumerate(names):
        for second in names[i+1:]:
            shared=family_sets[first]&family_sets[second]
            if shared:
                collisions.append({'splits':[first,second],'families':sorted(shared)})

    n_sel,n_hold=len(suites['selection']),len(suites['holdout'])
    estimates={'trial_attempts_max':config.trials,
               'selection_rows_per_attempt':n_sel,
               'baseline_selection_generations':n_sel,
               'selection_generations_upper_bound':(config.trials+1)*n_sel,
               'holdout_generations_only_if_winner':n_hold,
               'finalist_rescoring_attempts_max':config.finalists,
               'judge_calls_upper_bound':(config.trials+1)*n_sel+n_hold if config.scoring_mode=='judge' else 0}
    return {'config':str(path),'status':'described-only, not validated on hardware',
            'source_validated':'validate_run_inputs passed before this report was built',
            'suites':{name:split_stats(rows) for name,rows in suites.items()},
            'split_family_collisions':collisions,
            'budget_estimates_upper_bounds':estimates,
            'policy':{'method':config.method,'method_options':config.method_options,'rank':config.rank,
                      'trials':config.trials,'startup_trials':config.startup_trials,'finalists':config.finalists,
                      'strength_range':config.strength_range,'edit_targets':config.edit_targets,
                      'consistency':config.consistency,'screening':config.screening},
            'generation_settings':{'max_tokens':config.max_tokens,'ctx_size':config.ctx_size,'dtype':config.dtype,
                                   'generation_batch':config.generation_batch},
            'metadata_valid':report['status'],
            'note':'Estimates are counts for cost planning, not measurements or throughput claims. No model, judge or GPU access.'}


def suite_diff(before,after):
    """Id/content/family comparison of two suite files. No model or GPU access."""
    from .data import load_suite, canonical_prompt, digest
    list_cap=200
    a,b=load_suite(Path(before)),load_suite(Path(after))
    a_by,b_by={r['id']:r for r in a},{r['id']:r for r in b}
    if len(a_by)!=len(a) or len(b_by)!=len(b):
        raise ValueError('Suite ids must be unique for comparison')
    canonical_a={canonical_prompt(r['prompt']):r['id'] for r in a}
    canonical_b={canonical_prompt(r['prompt']):r['id'] for r in b}
    tracked=('prompt','kind','language','family_id','expected','tests')
    changed=[{'id':i,'fields':[field for field in tracked if a_by[i].get(field)!=b_by[i].get(field)]}
             for i in sorted(a_by.keys()&b_by.keys())
             if any(a_by[i].get(field)!=b_by[i].get(field) for field in tracked)]
    reprompted=[{'id':canonical_a[key],'becomes':canonical_b[key]} for key in canonical_a.keys()&canonical_b.keys()
                if canonical_a[key]!=canonical_b[key]]
    return {'before':str(before),'after':str(after),
            'digest_before':digest(a),'digest_after':digest(b),
            'counts':{'before':len(a),'after':len(b),
                      'removed':len(a_by.keys()-b_by.keys()),'added':len(b_by.keys()-a_by.keys()),'changed':len(changed)},
            'removed':sorted(a_by.keys()-b_by.keys())[:list_cap],
            'added':sorted(b_by.keys()-a_by.keys())[:list_cap],
            'changed':changed[:list_cap],
            'retitled_prompts':reprompted[:list_cap],
            'list_cap':list_cap,
            'note':'Counts are exact; id lists are capped. A retitled prompt keeps its id but changes content or vice versa.'}


def about():
    import hashlib
    import platform
    import re
    from importlib.metadata import version as package_version, PackageNotFoundError
    from . import __version__
    package_root=Path(__file__).parent
    def sha256_file(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    digest=hashlib.sha256()
    for source in sorted(p for p in package_root.rglob('*.py') if '__pycache__' not in p.parts):
        digest.update(source.relative_to(package_root).as_posix().encode())
        digest.update(sha256_file(source).encode())
    notice_path=package_root.parent/'NOTICE'
    revision=None
    if notice_path.is_file():
        match=re.search(r'Inspected revision: (\w+)',notice_path.read_text())
        revision=match.group(1) if match else None
    dependencies={}
    for name in ('torch','transformers','optuna','numpy','accelerate','bitsandbytes','safetensors','gguf','sentencepiece','protobuf'):
        try:
            dependencies[name]=package_version(name)
        except PackageNotFoundError:
            dependencies[name]=None
    return {'version':__version__,'python':platform.python_version(),'platform':platform.platform(),
            'tool_manifest_digest':digest.hexdigest(),
            'assets':{p.name:sha256_file(p) for p in sorted((package_root/'assets').iterdir())},
            'upstream':{'repository':'https://github.com/p-e-w/heretic','inspected_revision':revision},
            'dependencies':dependencies,
            'note':'Installation and provenance metadata only; certifies no runtime behaviour or model result.'}
