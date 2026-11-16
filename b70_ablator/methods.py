# SPDX-License-Identifier: AGPL-3.0-or-later
"""Method implementations. Importing the catalogue never imports PyTorch."""
import math
from .architectures import MAX_STRENGTH
from .errors import DirectionRejected

SOURCES = {
    'directional': ['https://arxiv.org/abs/2406.11717'],
    'projected': ['https://huggingface.co/blog/grimjim/projected-abliteration'],
    'mpoa': ['https://github.com/p-e-w/heretic/pull/52'],
    'svd': ['https://arxiv.org/abs/2406.11717'],
    'whitened-svd': ['https://github.com/elder-plinius/OBLITERATUS'],
    'som': ['https://arxiv.org/abs/2511.08379', 'https://github.com/pralab/som-refusal-directions'],
    'leace': ['https://arxiv.org/abs/2306.03819'],
    'ara': ['https://github.com/p-e-w/heretic/pull/211', 'https://github.com/p-e-w/heretic/pull/332',
            'https://github.com/p-e-w/heretic/pull/400', 'https://github.com/p-e-w/heretic/pull/478'],
    'sft': ['https://arxiv.org/abs/2406.04313'],
    'dpo': ['https://arxiv.org/abs/2305.18290'],
    'rdo': ['https://arxiv.org/abs/2502.17420'],
    'router-prebias': ['https://github.com/p-e-w/heretic/issues/322'],
    'sae': ['https://github.com/elder-plinius/OBLITERATUS'],
    'nullspace': ['https://arxiv.org/abs/2301.12627'],
}
DIRECTION_METHODS = {'directional','projected','mpoa','svd','whitened-svd','som','leace','nullspace'}
# Methods whose operator is a plain per-layer projection, so a per-layer
# strength profile can be attached. LEACE supplies its own left/right factors
# and ARA its own learned factors; neither consumes a layer profile.
PROFILE_METHODS = DIRECTION_METHODS - {'leace'}
LAYER_PROFILES = ('searched', 'discriminative')
IMPLEMENTED = DIRECTION_METHODS | {'ara'}
TRAINING_METHODS = {'sft','dpo'}


def method_catalog():
    return {'scope': 'Explicit implementations, not guarantees of refusal removal or exact paper reproduction.',
            'methods': {name: {'status': 'implemented' if name in IMPLEMENTED | TRAINING_METHODS else 'researched-not-implemented',
                               'sources': urls, 'full_model_b70_verified': False,
                               'entrypoint': 'run' if name in IMPLEMENTED else 'train' if name in TRAINING_METHODS else None,
                               'notes': {'som': 'Deterministic 1-D SOM over harmful family means, then subtract harmless centroid; orthonormalized directions.',
                                         'leace': 'Linear covariance erasure, no affine mean-restoration term.',
                                         'ara': 'Bounded KNN low-rank module-I/O optimization, then layer-strength search; dense modules only.'}.get(name,'')}
                        for name, urls in SOURCES.items()},
            'edit_surface': {
                'scope': 'Which projections are edited. Both sides remove the same direction; they differ in where it is removed.',
                'writeback': 'Default. o_proj/out_proj and down_proj: the module no longer writes the direction into the residual stream.',
                'full': 'Adds q_proj/k_proj/v_proj and gate_proj/up_proj: the module no longer reads the direction from the residual stream. '
                        'The read side is the only way to reach grouped-query k/v, whose output width is smaller than the residual width.',
                'note': 'Each side is tuned by its own searched strength, so read and write pressure are independent.'},
            'layer_profiles': {
                'searched': 'Default. Optuna searches a layer-relative center/width/floor strength shape per edit group.',
                'discriminative': 'Per-layer strength comes from the measured good/bad signal-to-noise of that layer, '
                                  'normalized to [0,1] and capped at 1; only one strength is searched. Replaces the searched shape '
                                  'rather than compounding with it. Not available to leace or ara, which carry their own factors.'},
            'strength': {'hard_bound': MAX_STRENGTH, 'default_search_range': [0., 1.5],
                         'note': 'The bound is not the search range; a run must ask for a wider range explicitly.'},
            'not_verified': ['that any of this removes refusals on a real checkpoint',
                             'that the read side is more effective than the write-back side, or by how much',
                             'that the discriminative profile beats a searched shape on a real model',
                             'expert-granular router profiling/suppression and independently learned expert directions; '
                             'basic shared-direction projection across fused experts is already implemented']}


def finite_option(options, name, low=0., high=float('inf'), integer=False):
    value = options.get(name)
    if type(value) not in ((int,) if integer else (int,float)) or not math.isfinite(value) or not low < value <= high:
        raise ValueError(f'Explicit valid {name} required in method_options')
    return value


def validate_method_options(method, options):
    common={'winsorization_quantile','massive_threshold','massive_ratio','row_normalization','layer_profile','capture','family_weighting'}
    specific={'whitened-svd':{'ridge'},'leace':{'ridge'},'som':{'som_steps','som_learning_rate'},
              'nullspace':{'nullspace_deflations'},
              'ara':{'steps','learning_rate','neighbors','preserve','steer','overcorrect','repulsion_cap'}}
    if not isinstance(options,dict) or set(options)-(common|specific.get(method,set())):
        raise ValueError('Unknown method_options or options for a different method')
    if options.get('capture','last') not in {'last','answer-mean'}:
        raise ValueError("capture must be 'last' or 'answer-mean'")
    if options.get('family_weighting','none') not in {'none','balanced'}:
        raise ValueError("family_weighting must be 'none' or 'balanced'")
    for key in specific.get(method,set())-{'overcorrect'}:
        finite_option(options,key,high=1. if key=='som_learning_rate' else float('inf'),integer=key in {'steps','neighbors','som_steps','nullspace_deflations'})
    if method=='ara':
        v=options.get('overcorrect')
        if type(v) not in (int,float) or not math.isfinite(v) or v<0:
            raise ValueError('Explicit nonnegative overcorrect required')
        if options.get('row_normalization','none')!='none':
            raise ValueError('ARA uses its preservation loss; row_normalization is unsupported')
    if options.get('row_normalization','none') not in {'none','pre','full'}:
        raise ValueError('Unknown row_normalization')
    if method=='leace' and options.get('row_normalization','none')!='none':
        raise ValueError('Row normalization would invalidate LEACE covariance erasure')
    if 'winsorization_quantile' in options:
        finite_option(options,'winsorization_quantile',high=1.)
    if 'massive_threshold' in options or 'massive_ratio' in options:
        finite_option(options,'massive_threshold')
        finite_option(options,'massive_ratio')
    if 'layer_profile' in options:
        if options['layer_profile'] not in LAYER_PROFILES:
            raise ValueError(f'layer_profile must be one of {list(LAYER_PROFILES)}')
        if method not in PROFILE_METHODS:
            raise ValueError(f'{method} does not consume a layer profile')


def layer_discriminability(records, options):
    """Per-layer signal-to-noise of the good/bad separation, scaled to [0,1].

    A layer whose class means barely differ, or whose spread is as large as the
    difference, carries no usable refusal signal. Weighting the edit by this
    profile spends strength where the signal actually is instead of spreading it
    uniformly across the depth, which is what a blind center/width search does.
    """
    import torch
    _, good, bad, _ = matched_rows(records, options)
    signal = (bad - good).norm(dim=-1).mean(0)
    spread = torch.cat([(good - good.mean(0, keepdim=True)).norm(dim=-1),
                        (bad - bad.mean(0, keepdim=True)).norm(dim=-1)], dim=0).mean(0)
    score = signal / spread.clamp_min(1e-8)
    peak = score.max()
    if not torch.isfinite(score).all() or float(peak) <= 1e-8:
        raise DirectionRejected('Layer discriminability is degenerate')
    return (score / peak).clamp(0., 1.)


def matched_rows(records, options):
    import torch
    groups = {}
    all_values = torch.stack([r.float() for _, r in records])
    mask = torch.zeros_like(all_values[0],dtype=torch.bool)
    if 'massive_threshold' in options or 'massive_ratio' in options:
        threshold = finite_option(options,'massive_threshold')
        ratio = finite_option(options,'massive_ratio')
        magnitudes = all_values.abs()
        mask = ((magnitudes > threshold) & (magnitudes > ratio*magnitudes.median(-1,keepdim=True).values.clamp_min(1e-8))).any(0)
    quantile = options.get('winsorization_quantile',1.)
    if type(quantile) not in (int,float) or not 0 < quantile <= 1:
        raise ValueError('winsorization_quantile must be in (0,1]')
    for task, value in records:
        value = value.float().clone()
        value.masked_fill_(mask,0)
        if quantile < 1:
            bound = torch.quantile(value.abs(),quantile,dim=-1,keepdim=True)
            value = value.clamp(-bound,bound)
        groups.setdefault((task['language'],task['family_id']),{}).setdefault(task['label'],[]).append(value)
    if any(set(v)!={'good','bad'} for v in groups.values()):
        raise ValueError('Matched good/bad families are required')
    labels = sorted(groups)
    good = torch.stack([torch.stack(groups[k]['good']).mean(0) for k in labels])
    bad = torch.stack([torch.stack(groups[k]['bad']).mean(0) for k in labels])
    counts = torch.tensor([len(groups[k]['good']) for k in labels],dtype=torch.int64)
    return labels,good,bad,counts


def fit_directions(records, rank, method='directional', options=None, seed=0):
    import torch
    from .numerics import directions_from_languages
    options = options or {}
    labels, good, bad, counts = matched_rows(records,options)
    if method == 'directional':
        languages = sorted({k[0] for k in labels})
        return directions_from_languages({lang:good[[i for i,k in enumerate(labels) if k[0]==lang]].mean(0) for lang in languages},
                                         {lang:bad[[i for i,k in enumerate(labels) if k[0]==lang]].mean(0) for lang in languages},rank)
    if method == 'leace':
        from .erasure import leace_factors
        if rank != 1:
            raise ValueError('Binary LEACE has rank one; set rank=1')
        left,_ = leace_factors(good,bad,finite_option(options,'ridge'))
        return torch.nn.functional.normalize(left.transpose(-1,-2),dim=-1)
    matrix = (bad-good).transpose(0,1)
    if method in {'projected','mpoa'}:
        harmless = torch.nn.functional.normalize(good.mean(0),dim=-1)
        matrix = matrix - (matrix*harmless[:,None,:]).sum(-1,keepdim=True)*harmless[:,None,:]
    elif method == 'whitened-svd':
        ridge = finite_option(options,'ridge')
        weighting = options.get('family_weighting','none')
        if weighting not in {'none','balanced'}:
            raise ValueError("family_weighting must be 'none' or 'balanced'")
        if weighting == 'balanced':
            weights = counts.float().clamp_min(1)/counts.float().clamp_min(1).sum()
        else:
            weights = torch.full((len(good),),1./len(good),device=good.device)
        mean_good = (weights.view(-1,1,1)*good).sum(0)
        centered = (good-mean_good).transpose(0,1)
        if weighting == 'balanced':
            centered = centered*weights.sqrt().view(1,-1,1)
        _,s,vh = torch.linalg.svd(centered,full_matrices=False)
        n_eff = float(counts.float().clamp_min(1).sum()) if weighting=='balanced' else float(len(good))
        variances = s.square()/max(n_eff-1,1.)
        # Thin-SVD inverse covariance square root; no residual-width-squared allocation.
        coefficients = (variances+ridge).rsqrt()-ridge**-.5
        matrix = matrix*ridge**-.5 + ((matrix@vh.transpose(-1,-2))*coefficients[:,None,:])@vh
        if weighting == 'balanced':
            matrix = matrix*weights.view(1,-1,1)
    elif method == 'som':
        steps = finite_option(options,'som_steps',integer=True)
        lr = finite_option(options,'som_learning_rate',high=1.)
        # Fit prototypes to harmful activations, as in the SOM paper, rather
        # than clustering good/bad differences (a different algorithm).
        matrix=bad.transpose(0,1)
        if rank > matrix.shape[1]:
            raise DirectionRejected('SOM rank exceeds matched-family contrast count')
        prototypes = matrix[:,torch.linspace(0,matrix.shape[1]-1,rank).long()].clone()
        topology = torch.arange(rank,device=matrix.device)
        for step in range(steps):
            sample = matrix[:,(step+seed)%matrix.shape[1]]
            winner = (prototypes-sample[:,None,:]).square().sum(-1).argmin(-1)
            radius = max(.5,rank*(1-step/steps)/2)
            neighborhood = torch.exp(-(topology[None,:]-winner[:,None]).square()/(2*radius**2))
            prototypes += lr*(1-step/steps)*neighborhood[:,:,None]*(sample[:,None,:]-prototypes)
        matrix = prototypes-good.mean(0)[:,None,:]
    elif method == 'nullspace':
        deflations = finite_option(options, 'nullspace_deflations', integer=True)
        return nullspace_directions(matrix, rank, deflations)
    elif method != 'svd':
        raise ValueError(f'Unknown direction method {method!r}')
    _,singular,vh = torch.linalg.svd(matrix,full_matrices=False)
    if rank < 1 or rank > vh.shape[1] or (singular[:,rank-1]<=1e-7).any():
        raise DirectionRejected('Insufficient independent direction rank')
    return vh[:,:rank,:].contiguous()


def nullspace_directions(matrix, rank, deflations):
    """Null-space projection family (RepEng-style): greedy deflated contrast basis.

    ``matrix`` is the per-layer batch of good/bad contrasts [layers, families,
    features]. Direction k is the mean contrast that survives after projecting
    out directions < k, sharpened by ``deflations`` power passes that
    re-weight each contrast by its own projection on the running direction.
    Deterministic, no optimizer, orthonormal by construction — a genuinely
    different basis from the singular vectors SVD would pick when contrast
    signs are mixed. Returns [layers, rank, features].
    """
    import torch
    residual = matrix
    directions = []
    for _ in range(rank):
        d = residual.mean(dim=-2)
        for _ in range(max(deflations - 1, 0)):
            weights = (residual * d.unsqueeze(-2)).sum(dim=-1)
            d = (residual * weights.unsqueeze(-1)).sum(dim=-2)
        for previous in directions:
            d = d - previous * (previous * d).sum(dim=-1, keepdim=True)
        norms = d.norm(dim=-1)
        if float(norms.min()) <= 1e-8:
            raise DirectionRejected('Null-space direction is degenerate')
        d = d / norms.unsqueeze(-1)
        directions.append(d)
        residual = residual - d.unsqueeze(-2) * (residual * d.unsqueeze(-2)).sum(dim=-1, keepdim=True)
    return torch.stack(directions, dim=1).contiguous()


def fit_ara(weight, good_x, good_y, bad_x, bad_y, rank, options, seed):
    """Optimize low-rank edits on the weight's device, never offload base weights."""
    import torch
    from torch.nn import functional as F
    steps = finite_option(options,'steps',integer=True)
    lr = finite_option(options,'learning_rate')
    neighbors = finite_option(options,'neighbors',integer=True)
    preserve = finite_option(options,'preserve')
    steer = finite_option(options,'steer')
    overcorrect = options.get('overcorrect')
    if type(overcorrect) not in (int,float) or not math.isfinite(overcorrect) or overcorrect < 0:
        raise ValueError('Explicit nonnegative overcorrect required')
    cap = finite_option(options,'repulsion_cap')
    if neighbors > min(len(good_x),len(bad_x)):
        raise ValueError('ARA neighbor count exceeds good/bad samples')
    if weight.ndim != 2 or not 0 < rank <= min(weight.shape):
        raise ValueError('ARA needs a dense projection and a valid rank')
    arrays = [good_x,good_y,bad_x,bad_y]
    if any(x.ndim!=2 or not torch.isfinite(x).all() for x in arrays):
        raise ValueError('Invalid ARA module I/O')
    # Original module I/O already contains W's contribution. Neither a full
    # FP32 copy of W nor an output-by-input delta matrix is needed here.
    device=weight.device
    gx,gy,bx,by = [x.to(device=device,dtype=torch.float32) for x in arrays]
    rng = torch.Generator().manual_seed(seed)
    a = (torch.randn(rank,weight.shape[-1],generator=rng)/math.sqrt(weight.shape[-1])).to(device).requires_grad_()
    b = torch.zeros(weight.shape[0],rank,device=device,requires_grad=True)
    def distances(x,y):
        # Explicit alignment and squared distances avoid XPU cdist kernel assumptions.
        d = (x[:,None,:]-y[None,:,:]).square().sum(-1)
        return d.topk(neighbors,largest=False).values.mean()
    def loss():
        ng,nb = gy+F.linear(F.linear(gx,a),b),by+F.linear(F.linear(bx,a),b)
        return preserve*(ng-gy).square().mean()+steer*(distances(nb,gy)-overcorrect*distances(nb,by).clamp(max=cap))
    with torch.enable_grad():
        optimizer = torch.optim.LBFGS([a,b],lr=lr,max_iter=10,history_size=5,line_search_fn='strong_wolfe')
        initial = float(loss().detach())
        def closure():
            optimizer.zero_grad()
            value = loss()
            if not torch.isfinite(value):
                raise ValueError('Nonfinite ARA loss')
            value.backward()
            return value
        for _ in range(steps):
            optimizer.step(closure)
        final = float(loss().detach())
    if not torch.isfinite(a).all() or not torch.isfinite(b).all() or final > initial:
        raise ValueError('ARA optimization diverged')
    return a.detach(),b.detach(),{'initial_loss':initial,'final_loss':final,'seed':seed,'bounded_repulsion':cap}
