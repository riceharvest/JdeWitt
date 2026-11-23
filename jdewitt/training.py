# SPDX-License-Identifier: AGPL-3.0-or-later
"""Local B70 SFT/DPO of edit factors. Training output is never a certified release."""
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F


def completion_logprob(logits,ids,prompt_length,chunk=32):
    if ids.ndim!=2 or ids.shape[0]!=1 or logits.shape[:2]!=ids.shape or not 0<prompt_length<ids.shape[1]:
        raise ValueError('Need one untruncated prompt plus labeled completion')
    labels=ids[:,prompt_length:]
    logits=logits[:,prompt_length-1:-1]
    # sum logp over the labeled tokens, computed in position chunks: the fused
    # cross_entropy keeps a 32x152k float32 block instead of a full float32
    # log_softmax copy of the vocabulary at every completion position, which is
    # what lets a dense 4B fit on a 12 GiB card.
    total=logits.new_zeros((),dtype=torch.float32)
    for start in range(0,labels.shape[1],chunk):
        block=logits[:,start:start+chunk].float().reshape(-1,logits.shape[-1])
        total=total+F.cross_entropy(block,labels[:,start:start+chunk].reshape(-1),reduction='sum')
    return -total


def dpo_loss(chosen,rejected,reference_chosen,reference_rejected,beta):
    if type(beta) not in (int,float) or not math.isfinite(beta) or beta<=0:
        raise ValueError('Positive finite DPO beta required')
    return -F.logsigmoid(beta*((chosen-rejected)-(reference_chosen-reference_rejected)))


def train_adapters(model,editor,rows,mode,steps,learning_rate,beta,seed,progress=None):
    if mode not in {'sft','dpo'} or type(steps) is not int or steps<1 or not rows:
        raise ValueError('Explicit SFT/DPO mode, rows and positive step count required')
    if type(learning_rate) not in (int,float) or not math.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError('Positive finite learning_rate required')
    device=next(model.parameters()).device
    rng=torch.Generator().manual_seed(seed)
    parameters=[]
    for p in model.parameters():
        p.requires_grad_(False)
    for *_,adapter in editor.targets:
        if (adapter.base.weight.ndim!=2 or getattr(adapter.base.weight,'quant_state',None) is not None
                or adapter.operator not in {'live','merged-resident'}):
            raise ValueError('Training requires dense original-precision projections')
        adapter.reset()
        adapter.A=(torch.randn(adapter.A.shape,generator=rng)/math.sqrt(adapter.A.shape[-1])).to(device).requires_grad_()
        adapter.B=torch.zeros(adapter.B.shape,device=device,dtype=torch.float32,requires_grad=True)
        adapter.enabled=True
        parameters.extend([adapter.A,adapter.B])
    optimizer=torch.optim.Adam(parameters,lr=learning_rate)
    losses=[]
    # Fit mode (dropout-free configs; the base stays frozen) so checkpointing is
    # live: transformers only recomputes checkpointed blocks in training mode,
    # and retaining every block's activations does not fit a small card.
    model.train()
    # Adapter-only gradients: the base stays frozen, so recomputing each block in
    # the backward pass costs one extra forward and saves the retained-activation
    # footprint. Without it a dense 4B DPO fit does not fit a 12 GiB card.
    checkpointing=False
    if hasattr(model,'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        checkpointing=True
    def score(ids,prompt):
        ids=ids.to(device)
        return completion_logprob(model(ids,use_cache=False).logits,ids,prompt)
    try:
        for step in range(steps):
            row=rows[step%len(rows)]
            prompt=row['prompt_length']
            optimizer.zero_grad()
            if mode=='dpo':
                if 'rejected_ids' not in row:
                    raise ValueError('DPO requires rejected completions')
                try:
                    for *_,adapter in editor.targets:
                        adapter.enabled=False
                    with torch.no_grad():
                        rc=score(row['ids'],prompt)
                        rr=score(row['rejected_ids'],prompt)
                finally:
                    for *_,adapter in editor.targets:
                        adapter.enabled=True
                loss=dpo_loss(score(row['ids'],prompt),score(row['rejected_ids'],prompt),rc,rr,beta)
            else:
                loss=-score(row['ids'],prompt)/(row['ids'].shape[1]-prompt)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise ValueError('Missing or nonfinite edit-factor gradients')
            # A two-pair preference fit saturates fast; an unclipped Adam step on
            # the saturated margin can push the factors into a nonfinite regime.
            torch.nn.utils.clip_grad_norm_(parameters,1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            if progress is not None:
                progress({'stage':'training','completed_steps':step+1,'total_steps':steps,'loss':losses[-1]})
    finally:
        if checkpointing:
            model.gradient_checkpointing_disable()
        model.eval()
        for p in parameters:
            p.requires_grad_(False)
            p.grad=None
    return {'mode':mode,'steps':steps,'losses':losses,'status':'trained-unvalidated',
            'note':'Executed optimization mechanics, not evidence of refusal reduction or retained capability.'}


def validate_training_config(obj):
    required={'mode','model','revision','rank','ctx_size','max_tokens','dtype','trust_remote_code','data',
              'holdout','steps','learning_rate','beta','seed','output'}
    if not isinstance(obj,dict) or set(obj)!=required or not isinstance(obj['mode'],str) or obj['mode'] not in {'sft','dpo'}:
        raise ValueError('Training requires exactly the documented explicit training configuration fields')
    for key in ('rank','ctx_size','max_tokens','steps'):
        if type(obj[key]) is not int or obj[key]<=0:
            raise ValueError(f'Positive integer {key} required')
    if obj['ctx_size']<=obj['max_tokens']:
        raise ValueError('Context must exceed the completion budget')
    for key in ('learning_rate','beta'):
        if type(obj[key]) not in (int,float) or not math.isfinite(obj[key]) or obj[key]<=0:
            raise ValueError(f'Positive finite {key} required')
    if type(obj['seed']) is not int or not 0<=obj['seed']<2**32:
        raise ValueError('seed must be an integer in [0,2**32)')
    if not isinstance(obj['dtype'],str) or obj['dtype'] not in {'bfloat16','float16'} or type(obj['trust_remote_code']) is not bool:
        raise ValueError('Invalid training dtype/remote-code policy')
    for key in ('model','data','holdout','output'):
        if not isinstance(obj[key],str) or not obj[key].strip():
            raise ValueError(f'Explicit nonempty {key} path required')
    if obj['revision'] is not None and (not isinstance(obj['revision'],str) or not obj['revision'].strip()):
        raise ValueError('revision must be null or a nonempty string')


def train_job(path,allow_secondary=False):
    from .data import load_suite,validate_splits,digest
    from .hf import HFModel,preflight_hf
    from .release import sha256_file
    from .export import stream_merge,verify_resident_export
    from .answers import atomic_json,exclusive
    obj=json.loads(Path(path).read_text())
    validate_training_config(obj)
    if not Path(obj['model']).is_dir():
        raise ValueError('Training requires a reviewed local original-safetensors model directory')
    train,holdout=load_suite(obj['data']),load_suite(obj['holdout'])
    validate_splits({'train':train,'holdout':holdout})
    for task in train:
        fields=('chosen','rejected') if obj['mode']=='dpo' else ('completion',)
        if any(not isinstance(task.get(k),str) or not task[k].strip() for k in fields):
            raise ValueError('Missing explicit training completion/preference strings')
    preflight_hf(obj['model'],obj['revision'],{'train':train,'holdout':holdout},obj['ctx_size'],obj['max_tokens'],obj['trust_remote_code'])
    output=Path(obj['output'])
    if output.exists():
        raise FileExistsError('Training outputs are immutable; use a new output directory')
    with exclusive(output,'.training.lock'):
        backend=HFModel(obj['model'],obj['revision'],obj['rank'],obj['max_tokens'],obj['dtype'],
                        obj['trust_remote_code'],ctx_size=obj['ctx_size'],allow_secondary=allow_secondary,
                        attn_implementation='sdpa',trial_operator='live')
        try:
            rows=[]
            for task in train:
                prompt=backend.tokenizer.encode(backend._render(task['prompt']),add_special_tokens=False)
                row={'prompt_length':len(prompt)}
                for key,field in (('ids','chosen' if obj['mode']=='dpo' else 'completion'),('rejected_ids','rejected')):
                    if field not in task:
                        continue
                    answer=backend.tokenizer.encode(task[field],add_special_tokens=False)
                    if not answer or len(answer)>obj['max_tokens']:
                        raise ValueError('Training completion is empty or exceeds its explicit output budget')
                    ids=prompt+answer
                    if len(ids)>obj['ctx_size']:
                        raise ValueError('Training sequence exceeds context; truncation is forbidden')
                    row[key]=torch.tensor([ids])
                rows.append(row)
            identity={'config':obj,'train_digest':digest(train),'holdout_digest':digest(holdout),
                      'source_files':{p.name:sha256_file(p) for p in Path(obj['model']).iterdir() if p.is_file()}}
            atomic_json(output/'training-identity.json',identity)
            report=train_adapters(backend.model,backend.editor,rows,obj['mode'],obj['steps'],obj['learning_rate'],
                                  obj['beta'],obj['seed'],lambda r:atomic_json(output/'progress.json',r))
            report['backend']=dict(backend.backend)
            if backend.secondary_test_run:
                report['backend_note']=('Second-rank CUDA test backend: the B70 kernels, the B70 VRAM '
                                        'envelope and the llama.cpp Vulkan serving gates were not exercised '
                                        'on this fit; the numbers it enables are a test, never a certification.')
            # The fit ran the live low-rank forward (same function, no per-module
            # dense workspace). Export verifies against the resident operator, so
            # switch the adapters back to it now that no gradient is being taken.
            for *_,adapter in backend.editor.targets:
                adapter.operator='merged-resident'
            stream_merge(obj['model'],output/'hf',backend.editor)
            report['export']=verify_resident_export(obj['model'],output/'hf',backend.editor)
            atomic_json(output/'training-report.json',report)
            return {'hf':str((output/'hf').resolve()),**report}
        finally:
            backend.release()
