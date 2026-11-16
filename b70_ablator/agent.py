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


def _study_user_attr(db_path, key):
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path.as_uri()+'?mode=ro',uri=True,timeout=1) as connection:
        connection.execute('PRAGMA query_only=ON')
        row=connection.execute('''SELECT sa.value_json FROM study_user_attributes sa
            JOIN studies s ON s.study_id=sa.study_id
            WHERE s.study_name='b70-ablator' AND sa.key=?''', (key,)).fetchone()
    return json.loads(row[0]) if row else None


def _run_lock_held(root):
    import fcntl
    lock=root/'.run.lock'
    if not lock.is_file():
        return None
    try:
        with lock.open('r+') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            fcntl.flock(handle,fcntl.LOCK_UN)
        return False
    except OSError:
        return True


def run_status(directory):
    root=Path(directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    trials=_trial_rows(root/'study.db')
    candidates=[t for t in trials if t['state']=='COMPLETE' and t['feasible'] and t['value'] is not None]
    return {'output':str(root),'trial_counts':dict(Counter(t['state'] for t in trials)),
            'trials':trials,'best_feasible':min(candidates,key=lambda t:t['value']) if candidates else None,
            'pareto_front':_study_user_attr(root/'study.db','pareto_front'),
            'cancel_requested':(root/'CANCEL').is_file(),
            'run_lock_held':_run_lock_held(root),
            'release_present':(root/'release').is_dir(),'release_verified':False,
            'note':'RUNNING is an unfinished resumable trial, not proof of a live process; run_lock_held=true means a worker currently holds this study. Use verify-release for bundle integrity.'}


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


def suite_near_dup(paths,threshold):
    """Reworded-duplicate prompt pairs across suite files, importer semantics. Files are never modified."""
    from .sources import token_set, jaccard
    from .data import load_suite
    if not 0 < threshold <= 1:
        raise ValueError('jaccard threshold must be in (0,1]')
    seen=[]
    pairs=[]
    counts={}
    for file in paths:
        rows=load_suite(Path(file))
        counts[str(file)]=len(rows)
        for row in rows:
            tokens=token_set(row['prompt'])
            duplicate=next((entry for entry in seen if jaccard(tokens,entry['tokens'])>=threshold),None)
            if duplicate is not None:
                pairs.append({'file':str(file),'id':row['id'],'duplicate_of':duplicate['id'],
                              'jaccard':round(jaccard(tokens,duplicate['tokens']),4)})
                continue
            seen.append({'id':row['id'],'file':str(file),'tokens':tokens})
    return {'threshold':threshold,'first_occurrence_wins':True,'suite_counts':counts,
            'near_duplicate_pairs':pairs[:200],'pair_count':len(pairs),'list_cap':200,
            'note':'Near duplicates are reworded prompts, not family overlap; the importer would drop the later one at this threshold.'}


def score_compare(suite,before,after,pass_on='compliance'):
    """Paired offline comparison of two saved scoring directories over one suite. No models run."""
    from .data import load_suite
    from .stats import wilson_interval,family_sign_test,family_bootstrap,paired_permutation_test
    suite_rows={row['id']:row for row in load_suite(Path(suite))}

    def scoring(directory):
        path=Path(directory)/'predictions.jsonl'
        if not path.is_file():
            raise FileNotFoundError(path)
        out={}
        for line in path.read_text().splitlines():
            if line.strip():
                row=json.loads(line)
                out[row['id']]=row
        return out

    def passes(row):
        if row.get('scoring_mode')=='refusal-check':
            return row.get('refusal_check',{}).get('result')=='no_explicit_refusal'
        judgment=row.get('judgment')
        if judgment is None:
            return False
        verdict=judgment.get('verdict')
        return verdict=='compliance' if pass_on=='compliance' else verdict in {'compliance','partial'}

    def label(row):
        if row.get('scoring_mode')=='refusal-check':
            return str(row.get('refusal_check',{}).get('result'))
        return str((row.get('judgment') or {}).get('verdict'))

    a,b=scoring(before),scoring(after)
    shared=sorted(set(a)&set(b))
    if not shared:
        raise ValueError('The two scoring directories share no prompt ids')
    modes={a[i].get('scoring_mode') or 'judge' for i in shared} | {b[i].get('scoring_mode') or 'judge' for i in shared}
    flips=[]
    by_kind={}
    units={}
    for pid in shared:
        pa,pb=passes(a[pid]),passes(b[pid])
        kind=a[pid].get('kind','?')
        counter=by_kind.setdefault(kind,{'before':[0,0],'after':[0,0]})
        counter['before'][0]+=pa
        counter['before'][1]+=1
        counter['after'][0]+=pb
        counter['after'][1]+=1
        if pa!=pb:
            family=suite_rows.get(pid,{}).get('family_id') or pid
            direction='improved' if pb else 'regressed'
            units.setdefault(family,set()).add(direction)
            flips.append({'id':pid,'kind':kind,'from':label(a[pid]),'to':label(b[pid]),
                          'family':family if 'family_id' in suite_rows.get(pid,{}) else None})
    improved=sum(1 for directions in units.values() if directions=={'improved'})
    regressed=sum(1 for directions in units.values() if directions=={'regressed'})
    mixed=len(units)-improved-regressed
    family_rates={}
    for pid in shared:
        fam=suite_rows.get(pid,{}).get('family_id') or pid
        entry=family_rates.setdefault(fam,{'before':0,'after':0,'total':0})
        entry['before']+=int(passes(a[pid]))
        entry['after']+=int(passes(b[pid]))
        entry['total']+=1
    bootstrap={'before':family_bootstrap([e['before']/e['total'] for e in family_rates.values()]),
               'after':family_bootstrap([e['after']/e['total'] for e in family_rates.values()])}
    permutation=paired_permutation_test([int(passes(b[pid]))-int(passes(a[pid])) for pid in shared])
    def summarize(rows):
        ids=set(shared)
        passed=sum(1 for pid,row in rows.items() if pid in ids and passes(row))
        total=sum(1 for pid in rows if pid in ids)
        return {'passed':passed,'total':total,'pass_rate_wilson':wilson_interval(passed,total)} if total else \
            {'passed':0,'total':0,'pass_rate_wilson':None}
    return {'suite':str(suite),'before':str(before),'after':str(after),'pass_on':pass_on,
            'shared_prompts':len(shared),
            'before_summary':summarize(a),'after_summary':summarize(b),
            'per_kind':{kind:{'before':f'{c["before"][0]}/{c["before"][1]}',
                              'after':f'{c["after"][0]}/{c["after"][1]}'} for kind,c in sorted(by_kind.items())},
            'unit_level':{'unit':'authored family_id, falling back to prompt id when absent',
                          'improved':improved,'regressed':regressed,'mixed':mixed,
                          'sign_test':family_sign_test(improved,regressed)},
            'family_bootstrap':bootstrap,'paired_permutation':permutation,
            'flips':flips[:200],'flip_count':len(flips),'list_cap':200,
            'scoring_modes':sorted(modes),
            'note':'Comparison of saved verdict files only; no inference was run. Wording-only refusal-check is not semantic compliance.'}


def calibration_report(suite, directories, min_agreement=0.5, ledger=None):
    """Calibration view over independently saved judge runs on one suite. No models run.

    Reports agreement with majority consensus — consensus is not ground truth,
    and this function never rewrites the authored calibration ledger; it only
    hands a human the numbers to author from.
    """
    from .data import load_suite
    from .stats import wilson_interval
    from collections import Counter
    if not directories:
        raise ValueError('At least one scoring directory is required')
    rows_per_dir = {}
    for directory in directories:
        path = Path(directory) / 'predictions.jsonl'
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('scoring_mode') == 'refusal-check':
                raise ValueError(f'Refusal-check rows in {directory} are wording checks, not judge votes; compare them with score-compare')
            judgment = row.get('judgment') or {}
            if judgment.get('verdict') is None:
                continue
            rows[row['id']] = (judgment['verdict'], judgment.get('confidence'))
        rows_per_dir[str(directory)] = rows
    labels = sorted(rows_per_dir)
    shared = set.intersection(*(set(rows) for rows in rows_per_dir.values())) if len(rows_per_dir) > 1 else set()
    consensus = {}
    for pid in shared:
        verdicts = [rows_per_dir[label][pid][0] for label in labels]
        top, count = Counter(verdicts).most_common(1)[0]
        if count / len(verdicts) >= min_agreement:
            consensus[pid] = top
    per_judge = {}
    for label, rows in sorted(rows_per_dir.items()):
        common = sorted(set(rows) & shared)
        confident_rows = [pid for pid in common if consensus.get(pid)]
        agree = sum(1 for pid in confident_rows if rows[pid][0] == consensus[pid])
        interval = wilson_interval(agree, len(confident_rows)) if confident_rows else None
        confidences = [rows[pid][1] for pid in common if isinstance(rows[pid][1], (int, float))]
        per_judge[label] = {'verdicts': len(rows), 'rows_in_common': len(common),
                            'rows_with_consensus': len(confident_rows),
                            'agreement_with_consensus': agree,
                            'agreement_interval': {'lower': round(interval['lower'], 4),
                                                   'upper': round(interval['upper'], 4)} if interval else None,
                            'mean_confidence': round(sum(confidences) / len(confidences), 4) if confidences else None}
    pairwise = {}
    for index, first in enumerate(labels):
        for second in labels[index + 1:]:
            common = sorted(set(rows_per_dir[first]) & set(rows_per_dir[second]))
            same = sum(1 for pid in common if rows_per_dir[first][pid][0] == rows_per_dir[second][pid][0])
            pairwise[first + ' | ' + second] = {'common': len(common), 'same_verdict': same}
    result = {'scope': 'Agreement with majority consensus across independently saved judge runs; '
                       'consensus is not ground truth and no accuracy is claimed.',
              'suite': str(suite), 'rows_in_suite': len(load_suite(Path(suite))),
              'min_agreement': min_agreement, 'rows_with_consensus': len(consensus),
              'per_judge': per_judge, 'pairwise_agreement': pairwise,
              'note': 'Author the calibration ledger from these numbers; this report never rewrites it.'}
    if ledger is not None:
        from .storage import file_digest
        path = Path(ledger)
        if not path.is_file():
            raise FileNotFoundError(path)
        result['authored_ledger'] = {'path': str(path), 'sha256': file_digest(path),
                                     'contents': json.loads(path.read_text())}
    return result


def judge_ensemble(suite,directories,min_agreement=0.5):
    """Majority aggregation of independently saved judge verdict files over one suite. No models run."""
    from .data import load_suite
    from collections import Counter
    if not directories:
        raise ValueError('At least one scoring directory is required')
    suite_rows={row['id']:row for row in load_suite(Path(suite))}
    votes={}
    per_directory={}
    for directory in directories:
        directory=str(Path(directory))
        path=Path(directory)/'predictions.jsonl'
        if not path.is_file():
            raise FileNotFoundError(path)
        counted,skipped=0,0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row=json.loads(line)
            if row.get('scoring_mode')=='refusal-check':
                raise ValueError(f'Refusal-check rows in {directory} are wording checks, not judge votes; compare them with score-compare')
            judgment=row.get('judgment')
            if judgment is None or judgment.get('verdict') is None:
                skipped+=1
                continue
            votes.setdefault(row['id'],{})[directory]=judgment['verdict']
            counted+=1
        per_directory[directory]={'judge_verdicts':counted,'rows_without_verdict':skipped}
    consensus,contested=[],[]
    for pid in sorted(votes):
        verdicts=list(votes[pid].values())
        tally=Counter(verdicts)
        winner,win_count=tally.most_common(1)[0]
        agreement=win_count/len(verdicts)
        record={'id':pid,'kind':suite_rows.get(pid,{}).get('kind'),'verdicts':votes[pid],
                'agreement':round(agreement,4)}
        if len(tally)>1 and win_count==tally.most_common(2)[1][1]:
            record['verdict']='tie'
            contested.append(record)
        elif agreement>=min_agreement:
            record['verdict']=winner
            consensus.append(record)
        else:
            record['verdict']='below-minimum-agreement'
            contested.append(record)
    total=len(votes)
    return {'suite':str(suite),'directories':per_directory,'min_agreement':min_agreement,
            'prompts_scored_by_any_judge':total,
            'consensus_count':len(consensus),'contested_count':len(contested),
            'contested_rate':round(len(contested)/total,4) if total else 0.0,
            'consensus':consensus[:500],'contested':contested[:200],
            'list_cap':{'consensus':500,'contested':200},
            'note':'Majority voting across saved verdict files, not ground truth; independent judges can share blind spots.'}


def score_sensitivity(directory,min_confidences=(0.5,0.7,0.9)):
    """Recount saved judge verdicts under alternative confidence floors. No models run."""
    from .stats import wilson_interval
    path=Path(directory)/'predictions.jsonl'
    if not path.is_file():
        raise FileNotFoundError(path)
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('No saved verdict rows')
    judged=[row for row in rows if (row.get('judgment') or {}).get('verdict') is not None]
    report={}
    for floor in min_confidences:
        if not 0<=floor<=1:
            raise ValueError('confidence floors must be within [0,1]')
        kept=[row for row in judged
              if row['judgment']['verdict']!='uncertain' and (row['judgment'].get('confidence') or 0)>=floor]
        compliant=sum(1 for row in kept if row['judgment']['verdict']=='compliance')
        report[str(floor)]={'rows_kept':len(kept),'rows_excluded':len(judged)-len(kept),
                           'compliant':compliant,
                           'compliance_rate_wilson':wilson_interval(compliant,len(kept)) if kept else None}
    return {'directory':str(directory),'rows':len(rows),'judge_verdict_rows':len(judged),
            'compliance_rate_by_confidence_floor':report,
            'note':'Offline recount of saved verdicts under alternative decision floors; no rejudging happened.'}


def gguf_edit(path, out, specs):
    """Edit quantized GGUF tensors: dequantize -> add delta -> requantize in place.

    ``specs`` are ``NAME=DELTA.NPY`` pairs holding float32 deltas with the
    tensor's shape. Everything outside the edited tensors (header, metadata,
    tensor info, all other tensor bytes) is copied byte for byte; offsets do
    not move because each edited tensor keeps its exact block count.
    """
    import numpy as np
    from .ggufio import edit_tensors
    deltas = {}
    for spec in specs:
        name, sep, file = spec.partition('=')
        if not sep or not name or not file:
            raise ValueError('spec format is NAME=DELTA.NPY')
        with open(file, 'rb') as f:
            deltas[name] = np.load(f)
    return edit_tensors(path, out, deltas)


def gguf_meta(path, out, assignments, unsets):
    """Edit GGUF v3 metadata; ``assignments`` are ``KEY=JSON`` strings (JSON parsed, bare text = string)."""
    from .ggufio import edit_metadata
    sets = {}
    for spec in assignments:
        key, sep, text = spec.partition('=')
        if not sep or not key:
            raise ValueError('assignment format is KEY=VALUE')
        try:
            sets[key] = json.loads(text)
        except json.JSONDecodeError:
            sets[key] = text
    return edit_metadata(path, out, sets, list(unsets))


def gguf_inspect(path,other=None):
    """GGUF metadata and tensor inventory read from the file header; tensor data is never loaded."""
    from gguf import GGUFReader
    def report(file):
        file=Path(file)
        reader=GGUFReader(str(file))
        def value(field):
            try:
                contents=field.contents()
            except Exception:
                contents=field.data
            if isinstance(contents,(list,tuple)):
                return [str(item)[:256] for item in contents[:64]]
            return str(contents)[:512]
        keep=('general.architecture','general.quantization_version','general.alignment','general.name','tokenizer.chat_template')
        metadata={name:value(field) for name,field in reader.fields.items()
                  if name.startswith('GGUF.') or name in keep or name.startswith(('tokenizer.','evaluator.','splitting.','add_'))}
        by_dtype={}
        total=0
        for tensor in reader.tensors:
            dtype=tensor.tensor_type.name if hasattr(tensor.tensor_type,'name') else str(tensor.tensor_type)
            entry=by_dtype.setdefault(dtype,{'tensors':0,'bytes':0})
            entry['tensors']+=1
            entry['bytes']+=int(tensor.n_bytes)
            total+=int(tensor.n_bytes)
        largest=sorted(({'name':tensor.name,
                         'dtype':tensor.tensor_type.name if hasattr(tensor.tensor_type,'name') else str(tensor.tensor_type),
                         'shape':[int(x) for x in tensor.shape],'bytes':int(tensor.n_bytes)}
                        for tensor in reader.tensors),key=lambda tensor:tensor['bytes'],reverse=True)[:10]
        schema={tensor.name:((tensor.tensor_type.name if hasattr(tensor.tensor_type,'name') else str(tensor.tensor_type)),
                             [int(x) for x in tensor.shape]) for tensor in reader.tensors}
        return {'file':str(file),'size_bytes':int(file.stat().st_size),'tensor_count':len(reader.tensors),
                'data_bytes':total,'metadata':metadata,'tensors_by_dtype':by_dtype,'largest_tensors':largest}, schema
    def metadata_digests(file):
        # Every header key hashed from raw bytes; values never appear in output.
        import hashlib
        from .ggufio import parse_gguf_header
        parsed=parse_gguf_header(file)
        return {name.decode(): hashlib.sha256(raw).hexdigest() for name, _tag, raw in parsed['entries']}
    single,schema=report(path)
    if other is None:
        single['metadata_digests']=metadata_digests(path)
        single['metadata_note']='metadata_digests covers every header key; values are never printed.'
        return single
    compared,second_schema=report(other)
    before_digests,after_digests=metadata_digests(path),metadata_digests(other)
    shared_keys=set(before_digests)&set(after_digests)
    metadata_changed=[{'key':key,'before_sha256':before_digests[key],'after_sha256':after_digests[key]}
                      for key in sorted(shared_keys) if before_digests[key]!=after_digests[key]]
    shared=set(schema)&set(second_schema)
    dtype_changed=[{'name':name,'before':schema[name][0],'after':second_schema[name][0]}
                   for name in sorted(shared) if schema[name][0]!=second_schema[name][0]]
    shape_changed=[{'name':name,'before':schema[name][1],'after':second_schema[name][1]}
                   for name in shared if schema[name][1]!=second_schema[name][1]]
    return {'before':single,'after':compared,
            'diff':{'only_in_before':sorted(set(schema)-shared)[:200],
                    'only_in_after':sorted(set(second_schema)-shared)[:200],
                    'dtype_changed':dtype_changed[:200],'shape_changed':shape_changed[:200],
                    'metadata_changed':metadata_changed[:200],
                    'metadata_only_in_before':sorted(set(before_digests)-set(after_digests))[:50],
                    'metadata_only_in_after':sorted(set(after_digests)-set(before_digests))[:50],
                    'counts':{'shared_tensors':len(shared),'dtype_changed':len(dtype_changed),
                              'shape_changed':len(shape_changed),'metadata_changed':len(metadata_changed)}},
            'note':'Header inventory only; tensor contents were not compared and no inference was run. '
                   'Metadata is compared by digest for every header key; values are never printed.'}


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


def suite_stats(path):
    """Descriptive statistics for one suite file. Loads and validates, no models run."""
    from collections import Counter
    from .data import load_suite
    rows = load_suite(Path(path))
    kinds = Counter(row['kind'] for row in rows)
    languages = Counter(row['language'] for row in rows)
    families = Counter(row.get('family_id') or row['id'] for row in rows)
    sizes = sorted(families.values())
    lengths = sorted(len(row['prompt']) for row in rows)
    n = len(rows)
    direction_labels = Counter(row.get('label') for row in rows if row['kind'] == 'direction')
    return {'suite': str(path), 'prompts': n,
            'kinds': dict(kinds), 'languages': dict(languages),
            'families': {'total': len(families), 'singletons': sum(1 for s in sizes if s == 1),
                         'largest': sizes[-1], 'median_size': sizes[len(sizes) // 2]},
            'prompt_chars': {'min': lengths[0], 'median': lengths[n // 2],
                             'p90': lengths[min(n - 1, int(0.9 * n))], 'max': lengths[-1]},
            'direction_labels': dict(direction_labels),
            'note': 'Structure statistics only; says nothing about difficulty or judgeability.'}


def study_metrics(run):
    """Per-trial parameter/value rollup read straight from a study's sqlite. Read-only."""
    from optuna.trial import TrialState
    db = Path(run) / 'study.db'
    if not db.is_file():
        raise FileNotFoundError(db)
    states = {int(state): state.name for state in TrialState}
    connection = sqlite3.connect(db.as_uri() + '?mode=ro', uri=True, timeout=1)
    try:
        connection.execute('PRAGMA query_only=ON')
        trials = {}
        for number, state, value, name, param_value, distribution in connection.execute(
                "select t.number,t.state,tv.value,tp.param_name,tp.param_value,tp.distribution_json "
                "from trials t "
                "left join trial_values tv on tv.trial_id=t.trial_id and tv.objective=0 "
                "left join trial_params tp on tp.trial_id=t.trial_id "
                "order by t.number"):
            trial = trials.setdefault(number, {'number': number, 'state': states.get(state, str(state)),
                                               'value': value, 'params': {}})
            if trial['value'] is None and value is not None:
                trial['value'] = value
            if name is not None:
                trial['params'][name] = param_value
                try:
                    bounds = json.loads(distribution)
                    if 'low' in bounds and 'high' in bounds:
                        trial['params'][name] = {'value': param_value, 'low': bounds['low'], 'high': bounds['high']}
                except (TypeError, ValueError):
                    pass
        feasible = [t for t in trials.values() if t['state'] == 'COMPLETE' and t['value'] is not None]
        numeric = {}
        for trial in feasible:
            for key, v in trial['params'].items():
                value = v['value'] if isinstance(v, dict) else v
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric.setdefault(key, []).append(value)
        ranges = {key: {'min': min(vals), 'max': max(vals),
                        'mean': round(sum(vals) / len(vals), 6), 'n': len(vals)}
                  for key, vals in sorted(numeric.items())}
        best = min(feasible, key=lambda t: t['value']) if feasible else None
    finally:
        connection.close()
    return {'study': str(db), 'trials': len(trials),
            'state_counts': dict(Counter(t['state'] for t in trials.values())),
            'best': best, 'complete_value_min': (min(t['value'] for t in feasible) if feasible else None),
            'searched_ranges': ranges,
            'note': 'Lower objective value is better in this tool\u2019s study direction.'}


def robustness_report(evaluations, pass_on='compliance'):
    """Worst-case pass rate of one saved model across several attack suites. Verdict files only, no models run.

    ``evaluations`` is a list of (label, suite_path, scores_directory).
    """
    from .data import load_suite
    from .stats import wilson_interval
    if not evaluations:
        raise ValueError('At least one --eval LABEL SUITE SCORES_DIR triple is required')
    reports = []
    for label, suite_path, scores_directory in evaluations:
        load_suite(Path(suite_path))
        predictions = Path(scores_directory) / 'predictions.jsonl'
        if not predictions.is_file():
            raise FileNotFoundError(predictions)
        passed = total = 0
        for line in predictions.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('scoring_mode') == 'refusal-check':
                ok = row.get('refusal_check', {}).get('result') == 'no_explicit_refusal'
            else:
                verdict = (row.get('judgment') or {}).get('verdict')
                ok = verdict == 'compliance' if pass_on == 'compliance' else verdict in {'compliance', 'partial'}
            passed += int(ok)
            total += 1
        if not total:
            raise ValueError(f'{label}: no scored prompts in {scores_directory}')
        reports.append({'label': label, 'suite': str(suite_path), 'passed': passed, 'total': total,
                        'pass_rate': passed / total,
                        'pass_rate_wilson': wilson_interval(passed, total)})
    reports.sort(key=lambda r: r['pass_rate'])
    return {'pass_on': pass_on, 'worst_suite': reports[0], 'suites': reports,
            'note': 'Aggregation of saved verdicts only; the attack suites themselves decide what this measures.'}


def family_decay(fitted, held, pass_on='compliance'):
    """Pass-rate decay of saved scores from fitted families to held-out families. Verdict files only, no models run.

    ``fitted`` and ``held`` are each (suite_path, scores_directory). Families
    come from authored suite metadata; a decay number is only as honest as
    the family split, so leaking families are reported loudly, not merged.
    """
    from .data import load_suite
    from .stats import wilson_interval

    def side(label, suite_path, scores_directory):
        rows = {row['id']: row for row in load_suite(Path(suite_path))}
        predictions = Path(scores_directory) / 'predictions.jsonl'
        if not predictions.is_file():
            raise FileNotFoundError(predictions)
        families: dict = {}
        passed = total = 0
        for line in predictions.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            suite_row = rows.get(row['id'])
            if suite_row is None:
                raise ValueError(f'{label}: scored prompt {row["id"]} is not in {suite_path}')
            if row.get('scoring_mode') == 'refusal-check':
                ok = row.get('refusal_check', {}).get('result') == 'no_explicit_refusal'
            else:
                verdict = (row.get('judgment') or {}).get('verdict')
                ok = verdict == 'compliance' if pass_on == 'compliance' else verdict in {'compliance', 'partial'}
            family = suite_row.get('family_id') or row['id']
            stats = families.setdefault(family, [0, 0])
            stats[0] += int(ok)
            stats[1] += 1
            passed += int(ok)
            total += 1
        if not total:
            raise ValueError(f'{label}: no scored prompts in {scores_directory}')
        return families, passed, total

    fitted_families, fitted_passed, fitted_total = side('fitted', *fitted)
    held_families, held_passed, held_total = side('held', *held)
    fitted_rate = fitted_passed / fitted_total
    held_rate = held_passed / held_total

    shared = sorted(set(fitted_families) & set(held_families))
    novel = sorted(set(held_families) - set(fitted_families))

    def family_row(family, stats):
        p, t = stats
        return {'family_id': family, 'passed': p, 'total': t, 'pass_rate': p / t}

    held_rows = sorted((family_row(f, held_families[f]) for f in novel),
                       key=lambda r: r['pass_rate'])
    decayed = [dict(r, decay_vs_fitted=r['pass_rate'] - fitted_rate) for r in held_rows]
    result = {
        'pass_on': pass_on,
        'fitted': {'suite': str(fitted[0]), 'passed': fitted_passed, 'total': fitted_total,
                   'pass_rate': fitted_rate, 'families': len(fitted_families)},
        'held': {'suite': str(held[0]), 'passed': held_passed, 'total': held_total,
                 'pass_rate': held_rate, 'pass_rate_wilson': wilson_interval(held_passed, held_total),
                 'families': len(held_families)},
        'global_decay': held_rate - fitted_rate,
        'novel_families': decayed,
        'worst_novel_family': decayed[0] if decayed else None,
        'best_novel_family': decayed[-1] if decayed else None,
    }
    if shared:
        result['warning'] = (f'{len(shared)} families appear in both suites and were not held out; '
                             'their decay numbers are meaningless as generalisation evidence')
        result['shared_families'] = shared[:25]
    if not novel:
        result['warning'] = 'No family in the held suite is absent from the fitted suite; nothing here measures decay.'
    result['note'] = ('Family membership is authored suite metadata. Decay is pass-rate difference on '
                      'families the fitted run never saw; judge noise is not subtracted.')
    return result


RUN_ARTIFACTS = {
    'provenance.json': 'model/suite/revision digests binding everything the run produced',
    'direction-stability.json': 'fit audit including warm-start transfer records',
    'prompt-budgets.json': 'generation token budget accounting',
    'judge-calibration.json': 'judge agreement/accuracy checks recorded during the run',
    'judge-rubrics.json': 'rubric cache used for grounded judging',
    'warm-starts.json': 'hyperparameter warm-start records (not the plan registry)',
    'quantized-ranking.json': 'finalist ranking under exported quants',
    'events.jsonl': 'append-only lifecycle event log (read with run-events)',
    'study.db': 'optuna storage (read with study-metrics and run-status)',
    'directions.pt': 'fit checkpoint (payload name and sha in directions.checkpoint.json)',
    'directions.checkpoint.json': 'integrity pointer for the fit payload',
    'harmless-trajectories.pt': 'reference-answer checkpoint pointer',
    'screen-baselines.pt': 'screening baseline checkpoint pointer',
    'baseline-hf': 'unmodified-baseline answers directory',
    'baseline-quants': 'baseline answer generation artifacts per quant',
    'judge-checks': 'per-stage judge check outputs directory',
    'release': 'release directory (files.json, evidence.json, signature.json when signed)',
    'CANCEL': 'run-cancel flag; its presence explains a stopped scheduler',
}


def run_ls(output):
    """Artifact-contract inventory of a run output directory. File reads only."""
    from .storage import file_digest
    root = Path(output)
    if not root.is_dir():
        raise FileNotFoundError(output)
    entries = []
    for name, role in sorted(RUN_ARTIFACTS.items()):
        path = root / name
        if not path.exists():
            entries.append({'name': name, 'present': False, 'role': role})
            continue
        if path.is_dir():
            entries.append({'name': name, 'present': True, 'kind': 'dir', 'role': role,
                            'children': sorted(child.name for child in path.iterdir())})
            continue
        size = path.stat().st_size
        row = {'name': name, 'present': True, 'kind': 'file', 'size_bytes': size, 'role': role}
        if size <= 256 * 1024 * 1024:
            row['sha256'] = file_digest(path)
        else:
            row['sha256'] = None
            row['digest_note'] = 'not hashed: over 256 MiB'
        entries.append(row)
    unexpected = [entry.name for entry in root.iterdir() if entry.name not in RUN_ARTIFACTS]
    return {'output': str(root.resolve()), 'entries': entries, 'unexpected_entries': sorted(unexpected),
            'note': 'The contract lists artifacts the pipeline may write; absence is stage-dependent. '
                    'Anything outside it is reported by name, never hidden — UUID payloads included.'}


_ST_FLOAT_DTYPES = {'F32': 'float32', 'F16': 'float16', 'F64': 'float64'}


def safetensors_tensor_diff(a, b, limit=100):
    """Compare two safetensors checkpoints by content, no model loads."""
    import json
    import mmap
    import numpy as np
    from pathlib import Path
    a, b = Path(a), Path(b)
    for p in (a, b):
        if not p.is_file():
            raise FileNotFoundError(str(p))

    def header(path):
        with open(path, 'rb') as f:
            size = int.from_bytes(f.read(8), 'little')
            if size <= 0 or size > (1 << 28):
                raise ValueError(f'{path}: implausible safetensors header length {size}')
            return json.loads(f.read(size)), 8 + size

    head_a, base_a = header(a)
    head_b, base_b = header(b)
    names_a = {n for n in head_a if n != '__metadata__'}
    names_b = {n for n in head_b if n != '__metadata__'}
    shared = sorted(names_a & names_b)
    states = {'shared': len(shared), 'bit_identical': 0, 'value_differs': 0,
              'shape_changed': 0, 'byte_only_dtype': 0, 'missing_dtype': 0}
    rel = []
    differing = []
    chunk = 1 << 24
    with open(a, 'rb') as fa, open(b, 'rb') as fb:
        mm_a = mmap.mmap(fa.fileno(), 0, access=mmap.MAP_SHARED)
        mm_b = mmap.mmap(fb.fileno(), 0, access=mmap.MAP_SHARED)
        try:
            for name in shared:
                info_a, info_b = head_a[name], head_b[name]
                if list(info_a['shape']) != list(info_b['shape']):
                    states['shape_changed'] += 1
                    continue
                dtype_a, dtype_b = info_a['dtype'], info_b['dtype']
                begin_a, end_a = info_a['data_offsets']
                begin_b, end_b = info_b['data_offsets']
                n_bytes = end_a - begin_a
                if n_bytes != end_b - begin_a:
                    states['missing_dtype'] += 1  # byte size cannot match a shared shape/dtype
                    continue
                raw_a = np.frombuffer(mm_a, dtype=np.uint8, count=n_bytes, offset=base_a + begin_a).copy()
                raw_b = np.frombuffer(mm_b, dtype=np.uint8, count=n_bytes, offset=base_b + begin_b).copy()
                if np.array_equal(raw_a, raw_b):
                    states['bit_identical'] += 1
                    continue
                if dtype_a != dtype_b:
                    states['value_differs'] += 1
                    differing.append({'name': name, 'dtype_before': dtype_a, 'dtype_after': dtype_b,
                                      'rel_rms': None})
                    continue
                floats = _st_tensor_values(np, dtype_a, raw_a)
                if floats is None:
                    states['byte_only_dtype'] += 1
                    differing.append({'name': name, 'dtype_before': dtype_a, 'rel_rms': None,
                                      'byte_identical': False})
                    continue
                others = _st_tensor_values(np, dtype_b, raw_b)
                if others is None or floats.shape != others.shape:
                    states['byte_only_dtype'] += 1
                    continue
                sum_sq = diff_sq = 0.0
                for start in range(0, floats.shape[0], chunk):
                    va = floats[start:start + chunk].astype(np.float32, copy=False)
                    vb = others[start:start + chunk].astype(np.float32, copy=False)
                    diff = va - vb
                    sum_sq += float((va * va).sum())
                    diff_sq += float((diff * diff).sum())
                rel_rms = float(np.sqrt(diff_sq / sum_sq)) if sum_sq > 0 else float('inf')
                rel.append(rel_rms)
                states['value_differs'] += 1
                differing.append({'name': name, 'dtype_before': dtype_a,
                                  'rel_rms': round(rel_rms, 8)})
        finally:
            mm_a.close()
            mm_b.close()
    summary = {'n': 0}
    if rel:
        ordered = sorted(rel)
        summary = {'n': len(ordered), 'min': round(ordered[0], 8),
                   'median': round(ordered[len(ordered) // 2], 8),
                   'max': round(ordered[-1], 8)}
    differing.sort(key=lambda row: (row.get('rel_rms') or 0.0), reverse=True)
    return {'before': str(a), 'after': str(b),
            'tensor_count_before': len(names_a), 'tensor_count_after': len(names_b),
            'only_in_before': sorted(names_a - names_b)[:limit],
            'only_in_after': sorted(names_b - names_a)[:limit],
            'summary': states, 'rel_rms': summary,
            'differing_tensors': differing[:limit],
            'note': 'Content comparison only, no inference: bit_identical means every stored byte '
                    'matches; value_differs reports the RMS difference relative to the first file. '
                    'Integer and unsupported float storages are compared by bytes only.'}


def _st_tensor_values(np, dtype, raw):
    """Decode a safetensors float storage to a float array, or None if byte-only."""
    global _E5M2
    if dtype in _ST_FLOAT_DTYPES:
        return np.frombuffer(raw, dtype=_ST_FLOAT_DTYPES[dtype])
    if dtype == 'BF16':
        return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    if dtype == 'F8_E4M3':
        from .fp4 import _E4M3_DECODE
        return _E4M3_DECODE[raw]
    if dtype == 'F8_E5M2':
        from .fp8 import decode_e5m2
        return decode_e5m2(raw)
    return None


def gguf_tensor_diff(a, b, limit=100):
    """Compare two GGUF checkpoints tensor-by-tensor: byte identity plus dequant error."""
    import mmap
    import numpy as np
    from pathlib import Path
    from gguf import GGUFReader
    from .ggufio import CODECS
    a, b = Path(a), Path(b)
    for p in (a, b):
        if not p.is_file():
            raise FileNotFoundError(str(p))
    reader_a = GGUFReader(str(a))
    reader_b = GGUFReader(str(b))
    tensors_a = {t.name: t for t in reader_a.tensors}
    tensors_b = {t.name: t for t in reader_b.tensors}
    shared = sorted(set(tensors_a) & set(tensors_b))
    states = {'shared': len(shared), 'bit_identical': 0, 'value_differs': 0,
              'shape_changed': 0, 'codec_unsupported': 0, 'byte_size_changed': 0}
    rel = []
    differing = []
    chunk = 1 << 24
    with open(a, 'rb') as fa, open(b, 'rb') as fb:
        mm_a = mmap.mmap(fa.fileno(), 0, access=mmap.MAP_SHARED)
        mm_b = mmap.mmap(fb.fileno(), 0, access=mmap.MAP_SHARED)
        try:
            for name in shared:
                ta, tb = tensors_a[name], tensors_b[name]
                shape_a = tuple(int(x) for x in np.asarray(ta.shape).ravel())
                shape_b = tuple(int(x) for x in np.asarray(tb.shape).ravel())
                if shape_a != shape_b:
                    states['shape_changed'] += 1
                    continue
                dtype_a, dtype_b = ta.tensor_type.name, tb.tensor_type.name
                offset_a, offset_b = int(ta.data_offset), int(tb.data_offset)
                size_a, size_b = int(ta.n_bytes), int(tb.n_bytes)
                if size_a == size_b and size_a <= chunk:
                    if mm_a[offset_a:offset_a + size_a] == mm_b[offset_b:offset_b + size_b]:
                        states['bit_identical'] += 1
                        continue
                elif size_a == size_b:
                    identical = True
                    for start in range(0, size_a, chunk):
                        count = min(chunk, size_a - start)
                        if mm_a[offset_a + start:offset_a + start + count] != \
                                mm_b[offset_b + start:offset_b + start + count]:
                            identical = False
                            break
                    if identical:
                        states['bit_identical'] += 1
                        continue
                if size_a != size_b:
                    states['byte_size_changed'] += 1
                    differing.append({'name': name, 'dtype_before': dtype_a, 'dtype_after': dtype_b,
                                      'bytes_before': size_a, 'bytes_after': size_b, 'rel_rms': None})
                    continue
                if dtype_a not in CODECS or dtype_b not in CODECS:
                    states['codec_unsupported'] += 1
                    differing.append({'name': name, 'dtype_before': dtype_a, 'dtype_after': dtype_b,
                                      'rel_rms': None})
                    continue
                sum_sq = diff_sq = 0.0
                bs_a, els_a, _, deq_a = CODECS[dtype_a]
                bs_b, els_b, _, deq_b = CODECS[dtype_b]
                n_a = size_a // bs_a * els_a
                n_b = size_b // bs_b * els_b
                if n_a != n_b:
                    states['codec_unsupported'] += 1
                    differing.append({'name': name, 'dtype_before': dtype_a, 'dtype_after': dtype_b,
                                      'rel_rms': None})
                    continue
                blocks_per_tile = max(1, (chunk // max(els_a, els_b)))
                nb = size_a // bs_a
                for start in range(0, nb, blocks_per_tile):
                    tile = min(blocks_per_tile, nb - start)
                    byte0, byte1 = start * bs_a, (start + tile) * bs_a
                    raw_a = np.frombuffer(mm_a, dtype=np.uint8,
                                          count=byte1 - byte0, offset=offset_a + byte0).copy()
                    raw_b = np.frombuffer(mm_b, dtype=np.uint8,
                                          count=byte1 - byte0, offset=offset_b + byte0).copy()
                    va = deq_a(raw_a, tile * els_a).reshape(-1)
                    vb = deq_b(raw_b, tile * els_b).reshape(-1)
                    diff = va - vb
                    sum_sq += float((va * va).sum())
                    diff_sq += float((diff * diff).sum())
                rel_rms = float(np.sqrt(diff_sq / sum_sq)) if sum_sq > 0 else float('inf')
                rel.append(rel_rms)
                states['value_differs'] += 1
                row = {'name': name, 'dtype_before': dtype_a, 'dtype_after': dtype_b,
                       'rel_rms': round(rel_rms, 6)}
                if dtype_a == dtype_b:
                    row['same_dtype'] = True
                differing.append(row)
        finally:
            mm_a.close()
            mm_b.close()
    summary = {'n': 0}
    if rel:
        ordered = sorted(rel)
        summary = {'n': len(ordered), 'min': round(ordered[0], 6),
                   'median': round(ordered[len(ordered) // 2], 6),
                   'max': round(ordered[-1], 6)}
    differing.sort(key=lambda row: (row.get('rel_rms') or 0.0), reverse=True)
    return {'before': str(a), 'after': str(b),
            'tensor_count_before': len(tensors_a), 'tensor_count_after': len(tensors_b),
            'only_in_before': sorted(set(tensors_a) - set(tensors_b))[:limit],
            'only_in_after': sorted(set(tensors_b) - set(tensors_a))[:limit],
            'summary': states, 'rel_rms': summary,
            'differing_tensors': differing[:limit],
            'note': 'Header and dequant measurement only, no inference: bit_identical tensors match '
                    'byte-for-byte; value_differs reports RMS difference after full dequant, '
                    'relative to the first file, using the pinned GGML block codecs.'}
