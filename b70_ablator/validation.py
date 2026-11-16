# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared metadata-only experiment gates, before model/device initialization."""
from .data import load_suite,validate_splits,digest
from .scoring import ConsistencyPolicy,validate_consistency_suite


def validate_experiment(directions,selection,holdout):
    validate_splits({'directions':directions,'selection':selection,'holdout':holdout})
    if any(r['kind']!='direction' for r in directions):
        raise ValueError('Direction suite contains non-direction tasks')
    languages={r['language'] for r in directions}
    if len(languages)<2 or any({r['label'] for r in directions if r['language']==language}!={'good','bad'} for language in languages):
        raise ValueError('Directions need matched good/bad prompts in at least two languages')
    for suite in (selection,holdout):
        if not {'refusal','harmless','math','code'} <= {r['kind'] for r in suite}:
            raise ValueError('Evaluation needs refusal, harmless, code and maths slices')
        if any(r['kind']=='direction' for r in suite):
            raise ValueError('Direction examples cannot be evaluated as held-out tasks')


def validate_run_inputs(config):
    if config.consistency is None:
        raise ValueError('Explicit consistency policy required; legacy run settings must be reviewed')
    policy=ConsistencyPolicy.from_json(config.consistency)
    if config.quantization!='none':
        raise ValueError('Resident export-consistent search requires original-precision weights; NF4 is not export-equivalent')
    if config.scoring_mode=='judge' and config.judge_url!='managed':
        raise ValueError('Automatic search requires judge_url="managed" so GPU models can be unloaded safely')
    suites={name:load_suite(getattr(config,name)) for name in ('directions','selection','holdout')}
    validate_experiment(**suites)
    for name in ('selection','holdout'):
        validate_consistency_suite(suites[name],policy)
    groups={}
    for task in suites['directions']:
        if not isinstance(task.get('family_id'),str) or not task['family_id'].strip():
            raise ValueError('Direction tasks require authored matched good/bad family_id metadata')
        groups.setdefault(task['language'],{}).setdefault(task['family_id'],[]).append(task['label'])
    if (config.method=='directional' and config.rank>len(groups)) or any(
        len(families)<policy.min_direction_families or
        any(set(labels)!={'good','bad'} for labels in families.values()) or
        any(sum(labels.count(label) for labels in families.values())<policy.min_direction_examples for label in ('good','bad'))
        for families in groups.values()):
        raise ValueError('Insufficient authored direction coverage for the requested policy/rank')
    if config.scoring_mode=='judge':
        for task in suites['directions']+[t for name in ('selection','holdout') for t in suites[name] if t['kind']=='refusal']:
            requirements=task.get('requirements')
            if (not isinstance(requirements,list) or not 1<=len(requirements)<=16 or
                any(not isinstance(x,str) or not x.strip() for x in requirements) or len(set(requirements))!=len(requirements)):
                raise ValueError('Judge tasks need 1..16 unique explicit requirements')
    return suites,{'status':'metadata-valid','suites':{name:{'tasks':len(rows),'digest':digest(rows)} for name,rows in suites.items()},
                   'direction_families':{lang:len(families) for lang,families in groups.items()},
                   'method':config.method,'models_loaded':False,'output_created':False,
                   'not_checked':['token budgets','available B70 memory','XPU kernels','model quality','GGUF converter support']}
