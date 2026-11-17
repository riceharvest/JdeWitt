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


def _funnel_summary(root):
    """Tally the search funnel from events.jsonl (the rotated tail is skipped;
    these are efficiency signals, not claims). Returns None when no events."""
    path = root / 'events.jsonl'
    if not path.is_file():
        return None
    summary = {}
    tallies = {('trial-stage1', None): 'stage1_probed',
               ('trial-pruned', 'Stage-1'): 'stage1_pruned',
               ('trial-pruned', 'Confirm-subset'): 'confirm_pruned',
               ('trial-pruned', 'Screened out'): 'screen_pruned',
               ('trial-pruned', 'cannot meet the damage limit'): 'kl_pruned',
               ('trial-pruned', 'Capability or validity gate failed'): 'capability_pruned'}
    for key in tallies.values():
        summary[key] = 0
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return None
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        kind, reason = record.get('kind'), record.get('reason') or ''
        for (event_kind, needle), name in tallies.items():
            if kind == event_kind and (needle is None or needle in reason):
                summary[name] += 1
        if kind == 'candidate-probe':
            summary['candidate_probe'] = {'samples': record.get('samples'), 'enqueued': record.get('enqueued'),
                                          'best': record.get('best'), 'worst': record.get('worst')}
    return summary


def run_status(directory):
    root=Path(directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    trials=_trial_rows(root/'study.db')
    candidates=[t for t in trials if t['state']=='COMPLETE' and t['feasible'] and t['value'] is not None]
    return {'output':str(root),'trial_counts':dict(Counter(t['state'] for t in trials)),
            'trials':trials,'best_feasible':min(candidates,key=lambda t:t['value']) if candidates else None,
            'pareto_front':_study_user_attr(root/'study.db','pareto_front'),
            'search_funnel':_funnel_summary(root),
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
    from .stages import validate_search_stages,stratified_sample,validate_candidate_probe
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
    stages = validate_search_stages(getattr(config,'search_stages',None),config.max_tokens)
    policy_stages = stages
    probe = validate_candidate_probe(getattr(config,'candidate_probe',None),config.trials)
    if stages is not None:
        probe_rows = stratified_sample([r for r in suites['selection'] if r['kind']=='refusal'], stages['probe_fraction'])
        estimates.update({'stage1_probe_rows_per_attempt':len(probe_rows),
                          'stage1_probe_generations_upper_bound':config.trials*len(probe_rows),
                          'stage1_output_tokens_per_generation':stages['probe_tokens'],
                          'stage1_keep_top':stages['keep_top'],
                          'note_stage1':'upper bound: assumes every trial passes the short-decode probe; stage 1 generates at most probe_tokens on harmful rows and never certifies'})
        if probe is not None:
            estimates.update({'candidate_probe_samples':probe['samples'],
                              'candidate_probe_enqueue_top':probe['enqueue_top'],
                              'candidate_probe_probe_generations':probe['samples']*len(probe_rows),
                              'note_candidate_probe':'pre-search probe costs samples*probe_rows short decodes in one model load; the top enqueue_top enter the study as ordinary trials and pay every gate in full'})
        if 'confirm_fraction' in stages:
            cf_rows = stratified_sample(suites['selection'], stages['confirm_fraction'])
            estimates.update({'confirm_rows_per_attempt':len(cf_rows),
                              'confirm_evals_upper_bound':config.trials*len(cf_rows),
                              'note_confirm':'confirm rung evaluates this stratified subset at full budget before the full-suite pass; it only prunes, all gates stay on the full pass'})
    return {'config':str(path),'status':'described-only, not validated on hardware',
            'source_validated':'validate_run_inputs passed before this report was built',
            'suites':{name:split_stats(rows) for name,rows in suites.items()},
            'split_family_collisions':collisions,
            'budget_estimates_upper_bounds':estimates,
            'policy':{'method':config.method,'method_options':config.method_options,'rank':config.rank,
                      'trials':config.trials,'startup_trials':config.startup_trials,'finalists':config.finalists,
                      'strength_range':config.strength_range,'edit_targets':config.edit_targets,
                      'consistency':config.consistency,'screening':config.screening,'search_stages':policy_stages,'candidate_probe':probe},
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
    before_summary=summarize(a)
    after_summary=summarize(b)
    from .stats import cohens_h
    effect=cohens_h(before_summary['passed'],before_summary['total'],
                     after_summary['passed'],after_summary['total'])
    return {'suite':str(suite),'before':str(before),'after':str(after),'pass_on':pass_on,
            'shared_prompts':len(shared),
            'before_summary':before_summary,'after_summary':after_summary,
            'effect_size_cohens_h':effect,
            'per_kind':{kind:{'before':f'{c["before"][0]}/{c["before"][1]}',
                              'after':f'{c["after"][0]}/{c["after"][1]}'} for kind,c in sorted(by_kind.items())},
            'unit_level':{'unit':'authored family_id, falling back to prompt id when absent',
                          'improved':improved,'regressed':regressed,'mixed':mixed,
                          'sign_test':family_sign_test(improved,regressed)},
            'family_bootstrap':bootstrap,'paired_permutation':permutation,
            'flips':flips[:200],'flip_count':len(flips),'list_cap':200,
            'scoring_modes':sorted(modes),
            'note':'Comparison of saved verdict files only; no inference was run. Wording-only refusal-check is not semantic compliance.'}


def suites_compare(families, pass_on='compliance'):
    """Paired comparisons across several (suite, before, after) families with
    Holm-Bonferroni adjusted sign-test p-values.

    Surfacing every family's p-value together with the step-down adjustment
    means a favorable suite cannot be reported without the family-wise cost
    of having looked at all the others. Effect sizes use the same signed
    Cohen's h convention as score_compare. No models run.
    """
    from .stats import holm_adjust
    if not families or len(families) % 3:
        raise ValueError('suites-compare expects SUITE BEFORE AFTER triples')
    per_family = []
    for index in range(0, len(families), 3):
        suite, before, after = families[index:index + 3]
        result = score_compare(suite, before, after, pass_on)
        per_family.append({
            'suite': str(suite), 'before': str(before), 'after': str(after),
            'shared_prompts': result['shared_prompts'],
            'before_pass_rate': result['before_summary']['pass_rate_wilson']['point']
                if result['before_summary']['pass_rate_wilson'] else None,
            'after_pass_rate': result['after_summary']['pass_rate_wilson']['point']
                if result['after_summary']['pass_rate_wilson'] else None,
            'effect_size_cohens_h': result['effect_size_cohens_h'],
            'informative_units': result['unit_level']['improved'] + result['unit_level']['regressed'],
            'sign_test_p_two_sided': result['unit_level']['sign_test']['p_two_sided'],
        })
    raw_p = [f['sign_test_p_two_sided'] for f in per_family]
    adjusted = holm_adjust(raw_p)
    for family, adjusted_p in zip(per_family, adjusted):
        family['sign_test_p_holm_adjusted'] = adjusted_p
    return {'families': per_family, 'tests_in_family': len(per_family),
            'holm_min_adjusted_p': min(adjusted) if adjusted else None,
            'note': 'Sign-test p-values per family with Holm-Bonferroni step-down adjustment; '
                    'report the adjusted values, the raw minimum alone is a cherry-pick.'}


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


def gguf_edit(path, out, specs, delta_gguf=None, strength=1.0):
    """Edit quantized GGUF tensors: dequantize -> add delta -> requantize in place.

    ``specs`` are ``NAME=DELTA.NPY`` pairs holding float32 deltas with the
    tensor's shape. ``delta_gguf`` is a GGUF delta file (gguf-delta-pack output)
    whose float tensors are named like target tensors; each is added scaled by
    ``strength``. Everything outside the edited tensors (header, metadata,
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
    if delta_gguf is not None:
        from gguf import GGUFReader
        from gguf.constants import GGMLQuantizationType
        reader = GGUFReader(str(delta_gguf))
        float_types = {int(GGMLQuantizationType.F32), int(GGMLQuantizationType.F16),
                       int(GGMLQuantizationType.BF16)}
        packed = str(delta_gguf)
        names = []
        for tensor in reader.tensors:
            if tensor.tensor_type not in float_types:
                raise ValueError(f'delta gguf tensor {tensor.name} is not float; refusing')
            if tensor.tensor_type == GGMLQuantizationType.BF16:
                # The reader byte-views BF16 as uint8 (2 bytes per value);
                # pair bytes back into uint16 code units, then widen to f32.
                u16 = np.frombuffer(np.asarray(tensor.data, dtype=np.uint8).tobytes(), dtype=np.uint16)
                values = np.frombuffer((u16.astype(np.uint32) << 16).tobytes(), dtype=np.float32)
            elif tensor.tensor_type == GGMLQuantizationType.F16:
                values = np.asarray(tensor.data, dtype=np.float16).astype(np.float32).reshape(-1)
            else:
                values = np.asarray(tensor.data, dtype=np.float32).reshape(-1)
            # tensor.shape is GGUF order (innermost first); numpy is row-major, reversed.
            want = tuple(int(d) for d in reversed(tensor.shape))
            if values.size != int(np.prod(want)):
                raise ValueError(f'delta gguf tensor {tensor.name} has {values.size} values '
                                 f'for shape {want}; refusing')
            values = values.reshape(want)
            if tensor.name in deltas:
                raise ValueError(f'{tensor.name} appears in both --tensor specs and the delta gguf')
            deltas[tensor.name] = values * strength
            names.append(tensor.name)
        if not names:
            raise ValueError(f'delta gguf {packed} contains no tensors')
        delta_gguf_names = names
    else:
        delta_gguf_names = []
    result = edit_tensors(path, out, deltas)
    result['delta_gguf'] = str(delta_gguf) if delta_gguf else None
    result['delta_gguf_tensors'] = delta_gguf_names
    result['strength'] = strength
    return result


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


def pt_tensor_diff(a, b, limit=100):
    """Compare two torch .pt checkpoints by content, no model loads.

    Handles state_dicts and the pipeline's saved payloads (a tuple of dicts
    for directions + audit) by flattening containers into dotted leaf
    paths. Tensors report byte identity, then relative RMS and cosine for
    float leaves; scalars report equality with truncated repr. Cosine 1
    means the delta is a positive scalar multiple.
    """
    import torch
    a, b = Path(a), Path(b)
    for p_ in (a, b):
        if not p_.is_file():
            raise FileNotFoundError(str(p_))
    raw_a = torch.load(a, map_location='cpu', weights_only=True)
    raw_b = torch.load(b, map_location='cpu', weights_only=True)

    def flatten(obj, prefix, leaves):
        if isinstance(obj, torch.Tensor):
            leaves[prefix] = obj.detach().to('cpu')
        elif isinstance(obj, dict):
            for key in obj:
                flatten(obj[key], f'{prefix}.{key}' if prefix else str(key), leaves)
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                flatten(item, f'{prefix}.{i}' if prefix else str(i), leaves)
        else:
            leaves[prefix] = obj

    leaves_a, leaves_b = {}, {}
    flatten(raw_a, '', leaves_a)
    flatten(raw_b, '', leaves_b)
    names_a, names_b = set(leaves_a), set(leaves_b)
    shared = sorted(names_a & names_b)
    states = {'shared': len(shared), 'bit_identical': 0, 'value_differs': 0,
              'shape_changed': 0, 'not_comparable': 0}
    comparable = []
    differing = []
    for name in shared:
        va, vb = leaves_a[name], leaves_b[name]
        if isinstance(va, torch.Tensor) or isinstance(vb, torch.Tensor):
            if not (isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor)):
                states['not_comparable'] += 1
                if len(differing) < limit:
                    differing.append({'name': name, 'note': 'tensor on one side, scalar on the other'})
                continue
            ta, tb = va, vb
            if ta.shape != tb.shape:
                states['shape_changed'] += 1
                differing.append({'name': name, 'shapes': [list(ta.shape), list(tb.shape)]})
                continue
            if ta.dtype.is_floating_point and tb.dtype.is_floating_point:
                if torch.equal(ta, tb):
                    states['bit_identical'] += 1
                    continue
                states['value_differs'] += 1
                fa, fb = ta.reshape(-1).double(), tb.reshape(-1).double()
                denom = (fa * fa).sum().item()
                delta_sq = (fa - fb).pow(2).sum().item()
                rel = float((delta_sq / denom) ** 0.5) if denom else None
                nrm_a, nrm_b = fa.norm(), fb.norm()
                cos = float(fa @ fb / (nrm_a * nrm_b)) if float(nrm_a) and float(nrm_b) else None
                row = {'name': name, 'rel_rms': rel, 'cosine': cos,
                       'l2_delta': float(delta_sq ** 0.5)}
                if denom == 0:
                    row['note'] = 'before tensor is all zero; relative RMS and cosine are undefined'
                comparable.append(row)
                if len(differing) < limit:
                    differing.append(row)
            else:
                if torch.equal(ta, tb):
                    states['bit_identical'] += 1
                else:
                    states['not_comparable'] += 1
                    if len(differing) < limit:
                        differing.append({'name': name, 'note': f'{ta.dtype} tensors differ'})
        else:
            if va == vb:
                states['bit_identical'] += 1
            else:
                states['value_differs'] += 1
                if len(differing) < limit:
                    differing.append({'name': name,
                                      'before': repr(va)[:100], 'after': repr(vb)[:100]})
    summary = {'n': len(comparable)}
    rels = sorted(r['rel_rms'] for r in comparable if r['rel_rms'] is not None)
    cosines = sorted(r['cosine'] for r in comparable if r['cosine'] is not None)
    if rels:
        summary.update({'rel_rms_min': rels[0], 'rel_rms_median': rels[len(rels) // 2],
                        'rel_rms_max': rels[-1]})
    if cosines:
        summary.update({'cosine_min': cosines[0], 'cosine_median': cosines[len(cosines) // 2],
                        'cosine_max': cosines[-1]})
    return {'before': str(a), 'after': str(b), 'states': states, 'summary': summary,
            'differing': differing,
            'missing_in_after': sorted(names_a - names_b),
            'missing_in_before': sorted(names_b - names_a),
            'note': 'Containers are flattened to dotted leaf paths; tensors report byte identity, '
                    'then relative RMS and cosine for float leaves; scalars report equality with '
                    'truncated repr. A missing leaf means the two files disagree on structure.'}


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

def run_doctor(output):
    """Diagnose a run directory: what is wrong (or nothing), with actions.

    Read-only aggregation. It never starts, resumes or repairs anything; it
    answers 'why does this run look stuck/broken' in one machine-readable
    answer so a driving agent does not have to chain run-status, run-ls and
    run-events and interpret sqlite bytes itself.
    """
    from .storage import file_digest
    root = Path(output).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    problems = []

    def problem(code, detail, action):
        problems.append({'code': code, 'detail': detail, 'action': action})

    def try_json(path, label, code):
        try:
            raw = path.read_text()
            if not raw.strip():
                raise ValueError('file is empty')
            json.loads(raw)
        except Exception as exc:
            problem(code, f'{label} present but unusable: {exc}',
                    f'Inspect or delete {path.name}; the pipeline rewrites it when the stage reruns.')

    db = root / 'study.db'
    if db.is_file() and db.stat().st_size >= 16:
        magic = db.read_bytes()[:16]
        if magic != b'SQLite format 3\x00':
            problem('STUDY_DB_CORRUPT', f'{db.name} does not start with the SQLite magic header',
                    'The study storage is damaged; this run cannot continue. Move it aside and rerun with the same --output to refit.')
    trial_counts = None
    if db.is_file():
        try:
            trial_counts = dict(Counter(t['state'] for t in _trial_rows(db)))
        except Exception as exc:
            if not problems:
                problem('STUDY_DB_CORRUPT', f'{db.name} cannot be opened read-only: {exc}',
                        'The study storage is damaged; move it aside and rerun with the same --output to refit.')
    lock = _run_lock_held(root)
    if trial_counts is not None:
        running = trial_counts.get('RUNNING', 0)
        if running and lock is not True:
            problem('STALE_RUNNING_TRIALS',
                    f'{running} trial(s) marked RUNNING but no worker holds the run lock',
                    'A worker died mid-trial. Rerun the same command with the same --output: unfinished trials are resumable; no process is running now.')
    events = root / 'events.jsonl'
    if events.is_file() and events.stat().st_size > 0:
        tail = events.read_bytes()[-65536:]
        last = tail.split(b'\n')[-1]
        if tail[-1:] != b'\n':
            problem('EVENTS_TAIL_BROKEN', 'events.jsonl ends with an unparsed partial line',
                    'The append-only log was cut mid-write (kill or disk full); earlier lines stay valid. Safe to truncate the partial line.')
        elif last.strip():
            try:
                json.loads(last)
            except Exception:
                problem('EVENTS_TAIL_BROKEN', 'the final complete line of events.jsonl is not valid JSON',
                        'Earlier lines stay valid; delete the last line to keep the log appendable.')
    pointer = root / 'directions.checkpoint.json'
    if pointer.is_file():
        try:
            meta = json.loads(pointer.read_text())
            payload = meta.get('payload')
            digest = meta.get('sha256') or meta.get('digest')
            pline = root / str(payload)
            if not pline.is_file():
                problem('FIT_PAYLOAD_MISSING', f'checkpoint pointer names payload {payload}, which does not exist',
                        'The fit weights are gone; rerun the fitting stage (same --output) or drop the pointer to force a clean refit.')
            elif digest and file_digest(pline) != digest:
                problem('FIT_PAYLOAD_DIGEST_MISMATCH',
                        f'payload {payload} hash does not match the pointer ({digest[:12]}...) expected',
                        'Pointer and payload disagree: the checkpoint was altered. Rerun the fitting stage to rewrite both.')
        except Exception as exc:
            problem('CHECKPOINT_POINTER_CORRUPT', f'{pointer.name} present but unusable: {exc}',
                    f'Delete {pointer.name}; the next fit stage rewrites it.')
    if (root / 'provenance.json').is_file():
        try_json(root / 'provenance.json', 'provenance.json', 'PROVENANCE_CORRUPT')
    for name in ('prompt-budgets.json', 'judge-calibration.json', 'judge-rubrics.json',
                 'direction-stability.json', 'warm-starts.json', 'quantized-ranking.json'):
        if (root / name).is_file():
            try_json(root / name, name, 'ARTIFACT_JSON_CORRUPT')
    if (root / 'CANCEL').is_file() and lock is not True:
        problem('CANCELLED_IDLE', 'CANCEL flag present and no worker holds the lock',
                'The run was stopped by request and stays resumable: delete CANCEL and rerun with --output to continue.')
    if not problems and lock is not True and (not trial_counts or sum(trial_counts.values()) == 0) \
            and not (root / 'events.jsonl').is_file():
        problem('NOTHING_RECORDED_YET', 'the run directory holds no trials, no events and no artifacts',
                'Either the run never started or it was wiped before its first write; rerun the same command and watch run-events.')
    healthy = not problems
    return {'output': str(root), 'healthy': healthy, 'problems': problems,
            'trial_counts': trial_counts, 'run_lock_held': lock,
            'cancel_requested': (root / 'CANCEL').is_file(),
            'note': 'Findings are read-only observations; nothing here starts, resumes or repairs work. '
                    'healthy=true means every check that could run passed, not that results are good.'}


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
    from .quants import DECODE_ONLY
    deq = {k: (bs, els, dec) for k, (bs, els, _q, dec) in CODECS.items()}
    deq.update(DECODE_ONLY)

    port_cache = {}

    def pinned_port(t_enum, dtype_name):
        """Decode-only entry backed by the pinned gguf.quants port.

        Used ONLY for dtypes this package has not ported itself (the IQ
        codebook types). Rows decoded this way are labelled per row, and the
        dtype counts into port_decoded, so no reader mistakes a borrowed
        decoder for ours.
        """
        if dtype_name in port_cache:
            return port_cache[dtype_name]
        entry = None
        try:
            from gguf import quants as gq_port
            from gguf.constants import GGML_QUANT_SIZES
            els, bs = GGML_QUANT_SIZES[t_enum]

            def dec(raw, n, t_enum=t_enum, bs=bs):
                blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, bs)
                return np.asarray(gq_port.dequantize(blocks, t_enum),
                                  dtype=np.float32).reshape(-1)

            if int(gq_port.dequantize(np.zeros(bs, dtype=np.uint8).reshape(1, bs),
                                      t_enum).size) == els:
                entry = (bs, els, dec)
        except Exception:
            entry = None
        port_cache[dtype_name] = entry
        return entry
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
              'shape_changed': 0, 'codec_unsupported': 0, 'packed_size_changed': 0,
              'port_decoded': 0}
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
                decoder = 'ours'
                da = deq.get(dtype_a)
                db = deq.get(dtype_b)
                if da is None:
                    da = pinned_port(ta.tensor_type, dtype_a)
                if db is None:
                    db = pinned_port(tb.tensor_type, dtype_b)
                if da is None or db is None:
                    states['codec_unsupported'] += 1
                    differing.append({'name': name, 'dtype_before': dtype_a, 'dtype_after': dtype_b,
                                      'rel_rms': None,
                                      'note': 'neither our codec nor the pinned gguf.quants port '
                                              'can decode one of these dtypes'})
                    continue
                port_used = (da is not deq.get(dtype_a)) or (db is not deq.get(dtype_b))
                if port_used:
                    decoder = 'pinned gguf.quants port (not ours)'
                    states['port_decoded'] += 1
                bs_a, els_a, deq_a = da
                bs_b, els_b, deq_b = db
                nb_a, nb_b = size_a // bs_a, size_b // bs_b
                if els_a != els_b or nb_a != nb_b or nb_a * bs_a != size_a or nb_b * bs_b != size_b:
                    states['codec_unsupported'] += 1
                    differing.append({'name': name, 'dtype_before': dtype_a, 'dtype_after': dtype_b,
                                      'rel_rms': None,
                                      'note': f'block grids differ ({els_a}x{nb_a} vs {els_b}x{nb_b}) '
                                              f'or a tensor is not block-aligned'})
                    continue
                size_changed = size_a != size_b
                if size_changed:
                    states['packed_size_changed'] += 1
                sum_sq = diff_sq = 0.0
                blocks_per_tile = max(1, (chunk // max(els_a, els_b)))
                for start in range(0, nb_a, blocks_per_tile):
                    tile = min(blocks_per_tile, nb_a - start)
                    raw_a = np.frombuffer(mm_a, dtype=np.uint8, count=tile * bs_a,
                                          offset=offset_a + start * bs_a).copy()
                    raw_b = np.frombuffer(mm_b, dtype=np.uint8, count=tile * bs_b,
                                          offset=offset_b + start * bs_b).copy()
                    va = deq_a(raw_a, tile * els_a).reshape(-1)
                    vb = deq_b(raw_b, tile * els_b).reshape(-1)
                    diff = va - vb
                    sum_sq += float((va * va).sum())
                    diff_sq += float((diff * diff).sum())
                rel_rms = float(np.sqrt(diff_sq / sum_sq)) if sum_sq > 0 else float('inf')
                row = {'name': name, 'dtype_before': dtype_a, 'dtype_after': dtype_b}
                if np.isfinite(rel_rms):
                    rel.append(rel_rms)
                    row['rel_rms'] = round(rel_rms, 6)
                else:
                    row['rel_rms'] = None
                    row['note'] = 'relative RMS undefined: decoded data has no energy or has non-finite values'
                states['value_differs'] += 1
                if dtype_a == dtype_b:
                    row['same_dtype'] = True
                if size_changed:
                    row['packed_bytes'] = [size_a, size_b]
                if decoder != 'ours':
                    row['decoder'] = decoder
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
                    'relative to the first file, using the pinned GGML block codecs. Cross-dtype '
                    'tensors (packed_size_changed) are decoded per side with their own block size '
                    'and compared by value; rows carry packed_bytes when the stored widths differ.'}


def verdict_audit(directory, limit=25):
    """Compare saved LLM-judge verdicts against the deterministic wording
    refusal-check re-run on the same saved answers. CPU-only: no model, no
    server, no new generation. This measures judge agreement with a
    lexical heuristic — not ground truth.
    """
    from collections import Counter
    from .refusal_check import RefusalChecker, PATTERNS
    from .stats import wilson_interval
    path = Path(directory) / 'predictions.jsonl'
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    refusal_rows = [r for r in rows if r.get('kind') == 'refusal']
    if not refusal_rows or not any('judgment' in r or r.get('status') == 'judge_invalid' for r in refusal_rows):
        raise ValueError('predictions contain no judge-scored refusal rows to audit')
    checker = RefusalChecker()
    checked_lang = set(PATTERNS)
    states = Counter()
    kinds = Counter()
    dangerous = []
    other = []
    agree = comp = 0
    for r in refusal_rows:
        if r.get('status') != 'ok':
            states[r.get('status', 'no_status')] += 1
            continue
        lang = str(r.get('language', '')).lower().split('-')[0]
        if lang not in checked_lang:
            states['language_not_covered'] += 1
            continue
        if not isinstance(r.get('answer'), str):
            states['no_answer_text'] += 1
            continue
        res = checker.check(r['answer'], r.get('language', 'en'))
        verdict = (r.get('judgment') or {}).get('verdict', 'invalid')
        key = f'judge_{verdict}__check_{res["result"]}'
        if res['result'] == 'refusal_detected' and verdict in ('refusal', 'partial'):
            agree += 1
        elif res['result'] == 'no_explicit_refusal' and verdict == 'compliance':
            agree += 1
        else:
            kinds[key] += 1
        comp += 1
    for r in refusal_rows:
        if r.get('status') != 'ok' or not isinstance(r.get('answer'), str):
            continue
        lang = str(r.get('language', '')).lower().split('-')[0]
        if lang not in checked_lang:
            continue
        res = checker.check(r['answer'], r.get('language', 'en'))
        verdict = (r.get('judgment') or {}).get('verdict', 'invalid')
        if verdict == 'compliance' and res['result'] == 'refusal_detected':
            dangerous.append({'id': r.get('id'), 'category': r.get('category'), 'language': r.get('language'),
                              'judge_verdict': verdict, 'check_result': res['result'],
                              'evidence': res['evidence'][:5]})
        elif not (
            (res['result'] == 'refusal_detected' and verdict in ('refusal', 'partial')) or
            (res['result'] == 'no_explicit_refusal' and verdict == 'compliance')
        ):
            if len(other) < limit:
                other.append({'id': r.get('id'), 'category': r.get('category'),
                              'judge_verdict': verdict, 'check_result': res['result'],
                              'evidence': res['evidence'][:5]})
    interval = wilson_interval(agree, comp) if comp else None
    return {'directory': str(directory), 'rows_total': len(rows),
            'compared': comp, 'agreed': agree,
            'agreement_rate': agree / comp if comp else None,
            'agreement_wilson95': interval,
            'skipped_states': dict(states),
            'dangerous_disagreements': dangerous[:limit],
            'dangerous_count': len(dangerous),
            'other_disagreement_kinds': dict(kinds),
            'other_disagreement_examples': other,
            'check_method': 'explicit-wording-v1',
            'note': 'Judge agreement with a deterministic lexical wording check on saved answers; '
                    'the check is not a semantic oracle and agreement says nothing about correctness.'}


def gguf_delta_pack(out, specs, metadata=None):
    """Pack NAME=DELTA.NPY float32 deltas into a GGUF delta file (F32 tensors
    named exactly like the target tensors) that `gguf-edit --delta-gguf` consumes.
    The file is inspectable with gguf-inspect and signable like any other artifact.
    """
    import numpy as np
    from .ggufio import write_gguf
    tensors = []
    for spec in specs:
        name, sep, file = spec.partition('=')
        if not sep or not name or not file:
            raise ValueError('spec format is NAME=DELTA.NPY')
        delta = np.load(file)
        if delta.dtype not in (np.float32, np.float64):
            raise ValueError(f'delta for {name} is {delta.dtype}; pack float32/float64 npys only')
        # GGUF dims are innermost-first; np arrays are row-major, same order reversed.
        tensors.append((name, tuple(reversed(delta.shape)), 0,
                        np.ascontiguousarray(delta, dtype=np.float32).tobytes()))
    meta = dict(metadata or {})
    meta['b70ablator.format'] = 'delta-gguf-v1'
    return write_gguf(out, meta, tensors)


def direction_export(registry, key, out, dtype='bf16'):
    """Export a registered fitted plan to a GGUF direction file.

    Writes the plan tensors (F32 or BF16, exact bf16 round-to-nearest via
    torch's own cast) plus method/geometry/provenance metadata that
    gguf-inspect and gguf-meta already understand, so fitted directions
    become portable, signable, inspectable files. Mapping plan tensors to
    per-target-tensor deltas is architecture dependent and deliberately
    not done here; the format version is stamped in the header.
    """
    import torch
    from .registry import load_plan
    from .ggufio import write_gguf, FLOAT_DTYPE_IDS
    if dtype not in FLOAT_DTYPE_IDS:
        raise ValueError(f'unsupported export dtype {dtype}; pick one of {sorted(FLOAT_DTYPE_IDS)}')
    if Path(out).exists():
        raise ValueError(f'{out} already exists')
    loaded = load_plan(registry, key)
    directions = loaded['plan']
    meta = loaded['meta']
    geometry = loaded['geometry']
    tensors = []
    for name, value in directions.items():
        if not isinstance(value, torch.Tensor):
            continue
        if value.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f'plan tensor {name} has dtype {value.dtype}; only float exports supported')
        if dtype == 'bf16':
            raw = value.detach().to(torch.bfloat16).view(torch.uint16).cpu().numpy().tobytes()
        else:
            raw = value.detach().to(torch.float32).cpu().numpy().tobytes()
        gguf_dims = tuple(reversed(tuple(value.shape)))  # GGUF order: innermost first
        tensors.append((name, gguf_dims, FLOAT_DTYPE_IDS[dtype], raw))
    if not tensors:
        raise ValueError('plan contains no tensors to export')
    metadata = {
        'b70ablator.format': 'direction-gguf-v1',
        'b70ablator.plan.key': str(key),
        'b70ablator.export_dtype': dtype,
        'b70ablator.payload_sha256': meta.get('payload_sha256', ''),
    }
    method = directions.get('method')
    if isinstance(method, str):
        metadata['b70ablator.plan.method'] = method
    row_norm = directions.get('row_normalization')
    if isinstance(row_norm, str):
        metadata['b70ablator.plan.row_normalization'] = row_norm
    for field in ('layers', 'rank', 'hidden'):
        if field in geometry:
            metadata[f'b70ablator.geometry.{field}'] = int(geometry[field])
    audit = loaded['audit']
    for field, value in audit.items() if isinstance(audit, dict) else ():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metadata[f'b70ablator.audit.{field}'] = float(value)
    metadata['b70ablator.provenance'] = json.dumps(
        {k: v for k, v in meta.items() if k.endswith('_sha256') or k in
         ('architecture', 'layers', 'hidden_size', 'rank', 'dtype', 'quantization',
          'edit_groups', 'boundary', 'method', 'registered_at')},
        sort_keys=True)
    result = write_gguf(out, metadata, tensors)
    result['plan_key'] = str(key)
    result['format'] = 'direction-gguf-v1'
    result['note'] = ('Plan container: tensors are fitted directions, not per-target-tensor '
                      'deltas. Applying them to quantized weights needs an architecture-specific '
                      'tensor-name mapping, which is verified separately.')
    return result


def _study_trial_params(db_path):
    if not db_path.exists():
        return {}
    with sqlite3.connect(db_path.as_uri()+'?mode=ro',uri=True,timeout=1) as connection:
        connection.execute('PRAGMA query_only=ON')
        rows=connection.execute('''SELECT t.number,tp.param_name,tp.param_value FROM trials t
            JOIN studies s ON s.study_id=t.study_id
            LEFT JOIN trial_params tp ON tp.trial_id=t.trial_id
            WHERE s.study_name='b70-ablator' ''').fetchall()
    out={}
    for number,name,value in rows:
        if name is not None:
            out.setdefault(number,{})[name]=value
    return out


def studies_compare(paths):
    """Read-only rollup across several study databases: per-study budget
    spent to reach the best feasible trial, cost curves, and per-parameter
    best-value shares. SQLite opened mode=ro with query_only; nothing is
    written or migrated. No models, no statistics beyond counts."""
    databases=[]
    params_roll={}
    for path in paths:
        db=Path(path)
        if db.is_dir():
            db=db/'study.db'
        if not db.is_file():
            raise FileNotFoundError(db)
        rows=_trial_rows(db)
        params=_study_trial_params(db)
        feasible=[t for t in rows if t['state']=='COMPLETE' and t['feasible'] and t['value'] is not None]
        best=min((t['value'] for t in feasible), default=None)
        trials_to_best=None
        if best is not None:
            for t in sorted(feasible,key=lambda t:t['number']):
                if t['value']<=best:
                    trials_to_best=t['number']
                    break
        curve=[]
        running=None
        for t in sorted(rows,key=lambda t:t['number']):
            if t['state']=='COMPLETE' and t['feasible'] and t['value'] is not None:
                running=t['value'] if running is None else min(running,t['value'])
                curve.append({'trials_evaluated':t['number']+1,'best_so_far':running})
        stride=max(1,len(curve)//64)
        databases.append({'study':str(db),'trials':len(rows),
                          'state_counts':dict(Counter(t['state'] for t in rows)),
                          'feasible':len(feasible),'best':best,'trials_to_best':trials_to_best,
                          'cost_curve':curve[::stride]+([curve[-1]] if curve and (len(curve)-1)%stride else [])})
        for number,values in params.items():
            trial=next((t for t in rows if t['number']==number),None)
            if not (trial and trial['state']=='COMPLETE' and trial['feasible'] and trial['value'] is not None):
                continue
            for name,value in values.items():
                bucket=params_roll.setdefault(name,{}).setdefault(str(value),
                    {'feasible':0,'best_count':0,'value_sum':0.0})
                bucket['feasible']+=1
                bucket['value_sum']+=trial['value']
                if best is not None and trial['value']<=best:
                    bucket['best_count']+=1
    param_stats={name:{value:{'feasible':bucket['feasible'],
                              'best_count':bucket['best_count'],
                              'best_value_share':bucket['best_count']/bucket['feasible'] if bucket['feasible'] else None,
                              'mean_value':round(bucket['value_sum']/bucket['feasible'],6) if bucket['feasible'] else None}
                       for value,bucket in sorted(values.items())}
                for name,values in sorted(params_roll.items())}
    return {'databases':databases,'param_stats':param_stats,
            'note':'Read-only rollup of persisted Optuna state. best_value_share is the fraction of feasible trials with that parameter value that hit the study best; it is a count, not a statistical test.'}


def quants_audit(path, blocks_per_dtype=4096):
    """Standing falsifier for the GGML codec ports: for every tensor of a
    supported dtype in a real file, dequantize a bounded, evenly spaced block
    sample and requantize it with the same codec. Byte identity is expected
    except at rounding-tie edges, so any drift here means a codec broke.
    Decode-only dtypes are reported honestly as un-auditable for requant.
    """
    import numpy as np
    from gguf import GGUFReader
    from .ggufio import CODECS
    from .quants import DECODE_ONLY
    reader = GGUFReader(str(path))
    reports = {}
    decode_only = {}
    port_only = {}
    unsupported = {}
    port_cache = {}

    def port_supported(t_enum):
        if t_enum in port_cache:
            return port_cache[t_enum]
        ok = False
        try:
            from gguf import quants as gq_port
            from gguf.constants import GGML_QUANT_SIZES
            _els, bs = GGML_QUANT_SIZES[t_enum]
            gq_port.dequantize(np.zeros(bs, dtype=np.uint8).reshape(1, bs), t_enum)
            ok = True
        except Exception:
            ok = False
        port_cache[t_enum] = ok
        return ok

    with open(path, 'rb') as f:
        for tensor in reader.tensors:
            dtype = tensor.tensor_type.name
            n_bytes = int(tensor.n_bytes)
            if dtype in DECODE_ONLY:
                decode_only[dtype] = decode_only.get(dtype, {'tensors': 0, 'bytes': 0})
                decode_only[dtype]['tensors'] += 1
                decode_only[dtype]['bytes'] += n_bytes
                continue
            if dtype not in CODECS:
                if port_supported(tensor.tensor_type):
                    port_only[dtype] = port_only.get(dtype, {'tensors': 0, 'bytes': 0})
                    port_only[dtype]['tensors'] += 1
                    port_only[dtype]['bytes'] += n_bytes
                else:
                    unsupported[dtype] = unsupported.get(dtype, 0) + 1
                continue
            block_bytes, block_elements, quantize, dequantize = CODECS[dtype]
            nb = n_bytes // block_bytes
            if nb * block_bytes != n_bytes:
                raise ValueError(f'tensor {tensor.name}: {n_bytes} not a multiple of {block_bytes} for {dtype}')
            rep = reports.setdefault(dtype, {'tensors': 0, 'blocks_total': 0, 'blocks_sampled': 0,
                                             'bytes_per_block': block_bytes, 'elements_per_block': block_elements,
                                             'byte_identical_blocks': 0, 'max_abs_change': 0.0,
                                             'max_change_over_std': 0.0})
            rep['tensors'] += 1
            rep['blocks_total'] += nb
            if rep['blocks_sampled'] >= blocks_per_dtype:
                continue
            remaining = blocks_per_dtype - rep['blocks_sampled']
            stride = max(1, nb // remaining)
            indices = list(range(0, nb, stride))[:remaining]
            for index in indices:
                offset = int(tensor.data_offset) + index * block_bytes
                f.seek(offset)
                raw = np.frombuffer(f.read(block_bytes), dtype=np.uint8)
                values = dequantize(raw, block_elements)
                requant = quantize(values)
                if requant.size != block_bytes:
                    raise ValueError(f'{tensor.name}: requant produced {requant.size} bytes, expected {block_bytes}')
                back = dequantize(np.frombuffer(requant, dtype=np.uint8), block_elements)
                identical = bool(np.array_equal(np.frombuffer(requant, np.uint8).reshape(-1),
                                                raw.reshape(-1)))
                rep['blocks_sampled'] += 1
                rep['byte_identical_blocks'] += int(identical)
                change = float(np.abs(back - values).max())
                std = float(values.std())
                rep['max_abs_change'] = max(rep['max_abs_change'], change)
                if std > 0:
                    rep['max_change_over_std'] = max(rep['max_change_over_std'], change / std)
    def finish(rep):
        rep['byte_identity_rate'] = rep['byte_identical_blocks'] / rep['blocks_sampled'] if rep['blocks_sampled'] else None
        return rep
    return {'file': str(path), 'dtypes_audited': sorted(reports),
            'per_dtype': {dtype: finish(rep) for dtype, rep in sorted(reports.items())},
            'decode_only_dtypes': {dtype: dict(rep, note='dequant verified elsewhere; no reference quantizer ported, requant not auditable')
                                   for dtype, rep in sorted(decode_only.items())},
            'port_decode_only_dtypes': {dtype: dict(rep, note='dequantizable only via the pinned gguf.quants port (not ours); no quantizer in reach, requant not auditable')
                                        for dtype, rep in sorted(port_only.items())},
            'unsupported_dtypes': dict(sorted(unsupported.items())),
            'note': 'Bounded evenly spaced block sample per dtype: dequantize then requantize with the same codec; byte identity expected except at rounding-tie edges.'}
