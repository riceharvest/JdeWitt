# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small portable edit plans; never contain full model weights."""
import torch
from .methods import fit_directions, layer_discriminability, matched_rows, fit_ara
from .numerics import stable_directions
from .erasure import leace_factors


def direction_identity(generation,suite_digest,method,options,scoring,consistency,seed):
    return {'generation':generation,'directions_digest':suite_digest,'method':method,
            'method_options':options,'scoring':scoring,'consistency':consistency,'seed':seed}


def move_plan(value, device):
    if isinstance(value,torch.Tensor):
        return value.detach().to(device)
    if isinstance(value,dict):
        return {k:move_plan(v,device) for k,v in value.items()}
    if isinstance(value,list):
        return [move_plan(v,device) for v in value]
    return value


def fit_plan(records, rank, policy, seed, expected_tasks, method='directional', options=None,
             module_records=None, editor=None):
    options = options or {}
    direction_method = 'svd' if method=='ara' else method
    direction_rank = 1 if method=='ara' else rank
    basis,audit = stable_directions(records,direction_rank,policy,seed,expected_tasks=expected_tasks,
        fitter=lambda rows,r:fit_directions(rows,r,direction_method,options,seed))
    plan = {'method':method,'basis':basis,'row_normalization':options.get('row_normalization','full' if method=='mpoa' else 'none')}
    if method=='nullspace':
        # Fit metadata: how much of each fitted direction is actually used by
        # the contrast, per layer. A direction with ~zero mean projection in
        # some layer carries no signal there and is a candidate for strength 0.
        _, good, bad = matched_rows(records,options)
        contrast = (bad-good).transpose(0,1)
        projected = torch.einsum('lch,lrh->lcr',contrast,plan['basis'])
        audit['nullspace'] = {'deflations':options['nullspace_deflations'],
            'mean_abs_projection':projected.abs().mean(-1).cpu().tolist(),
            'contrast_norm':contrast.norm(dim=-1).mean(-1).cpu().tolist()}
    if options.get('layer_profile') == 'discriminative':
        # Data-driven per-layer strength. The searched center/width/floor shape is
        # replaced, not multiplied, so the two are never double-counted.
        plan['layer_profile'] = layer_discriminability(records, options).cpu()
    if method=='leace':
        _,good,bad = matched_rows(records,options)
        plan['left'],plan['right'] = leace_factors(good,bad,options['ridge'])
    if method=='ara':
        if editor is None or not module_records:
            raise ValueError('ARA requires real captured dense-module I/O')
        plan['factors'] = {}
        audit['ara'] = {}
        for *_,adapter in editor.targets:
            if adapter.base.weight.ndim!=2:
                raise ValueError('ARA fused-expert per-route I/O adapter is not implemented; no experts were skipped')
            groups = {}
            for task,io in module_records:
                if adapter.source_key not in io:
                    raise ValueError(f'ARA module I/O missing: {adapter.source_key}')
                groups.setdefault(task['label'],[]).append(io[adapter.source_key])
            if set(groups)!={'good','bad'}:
                raise ValueError('ARA requires accepted good and bad module evidence')
            gx,gy = [torch.stack([r[i] for r in groups['good']]) for i in (0,1)]
            bx,by = [torch.stack([r[i] for r in groups['bad']]) for i in (0,1)]
            a,b,report = fit_ara(adapter.base.weight,gx,gy,bx,by,rank,options,seed)
            plan['factors'][adapter.source_key] = {'A':a.cpu(),'B':b.cpu()}
            audit['ara'][adapter.source_key] = report
    return plan,audit
