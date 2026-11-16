# SPDX-License-Identifier: AGPL-3.0-or-later
"""Explicit padded-token budgets, not guessed GPU occupancy."""


def validate_batch_policy(policy):
    if policy is None:
        return
    if (not isinstance(policy,dict) or set(policy)!={'max_sequences','max_padded_tokens'}
            or any(type(v) is not int or v<1 for v in policy.values())):
        raise ValueError('batch policy needs positive integer max_sequences and max_padded_tokens')


def plan_batches(tasks, generator, refusal_floor, policy):
    """Preflight all lengths, pack nearby lengths, keep generation policies apart.

    Reserve the full output budget even when the request has a refusal floor.
    This is a conservative token admission limit, NOT a VRAM-fit guarantee.
    """
    validate_batch_policy(policy)
    if policy is None:
        return [([task], refusal_floor if task['kind']=='refusal' else None) for task in tasks]
    if not tasks:
        return []
    if not callable(getattr(generator,'generate_batch',None)) or not callable(getattr(generator,'prompt_tokens',None)):
        raise ValueError('Configured batch policy needs backend batch generation and exact prompt counts')
    output=getattr(generator,'max_tokens',None)
    if type(output) is not int or output<1:
        raise ValueError('Batch backend needs an explicit output budget')
    groups={}
    for task in tasks:
        length=generator.prompt_tokens(task['prompt'])
        if type(length) is not int or length<1 or length+output>policy['max_padded_tokens']:
            raise ValueError(f'Prompt {task["id"]!r} exceeds the batch token budget')
        floor=refusal_floor if task['kind']=='refusal' else None
        groups.setdefault(floor,[]).append((length,task))
    batches=[]
    for floor,entries in groups.items():
        batch=[]

        for length,task in sorted(entries,key=lambda pair:pair[0]):
            if batch and (len(batch)>=policy['max_sequences'] or
                          (len(batch)+1)*(length+output)>policy['max_padded_tokens']):
                batches.append((batch,floor))
                batch=[]

            batch.append(task)
        if batch:
            batches.append((batch,floor))
    return batches
