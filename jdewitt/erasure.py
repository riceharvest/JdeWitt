# SPDX-License-Identifier: AGPL-3.0-or-later
"""Binary linear LEACE, fitted on small activation statistics, not model weights."""
import torch


def leace_factors(good, bad, ridge):
    if good.shape != bad.shape or good.ndim != 3 or ridge <= 0:
        raise ValueError('LEACE requires matched [samples,layers,width] data and positive ridge')
    samples = torch.cat([good,bad]).transpose(0,1).float()
    samples = samples - samples.mean(1,keepdim=True)
    _, singular, vh = torch.linalg.svd(samples,full_matrices=False)
    variance = singular.square()/max(samples.shape[1]-1,1)
    delta = (bad.mean(0)-good.mean(0)).float()
    def transform(x, inverse=False):
        base = ridge**(.5 if inverse else -.5)
        values = (variance+ridge).sqrt() if inverse else (variance+ridge).rsqrt()
        return x*base + ((x[:,None,:]@vh.transpose(-1,-2))*(values-base)[:,None,:]@vh).squeeze(1)
    whitened = transform(delta)
    norm = whitened.norm(dim=-1,keepdim=True)
    if (norm <= 1e-8).any():
        raise ValueError('Degenerate LEACE concept')
    u = whitened/norm
    return transform(u,True).unsqueeze(-1),transform(u).unsqueeze(-2)
