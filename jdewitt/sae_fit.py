# SPDX-License-Identifier: AGPL-3.0-or-later
"""Train a small sparse autoencoder on captured residual rows (P5 workstream b).

This is the fit half of the SAE lane whose edit half lives in ``jdewitt/sae.py``.
Nothing here removes anything or claims a behaviour: it trains a dictionary on
one family's captured residual rows, reports held-out FVU against a baseline,
and turns the trained dictionary into the feature table the ablation card
pre-registers from.

Shapes follow the repo convention: a captured record is ``[layers, hidden]`` per
row (``capture``'s answer-locus mean), so a family's training matrix is
``[rows, hidden]`` at one layer.

The model is a top-k autoencoder (Gao et al. 2024) rather than L1: the plan's
feature budget is a *compute* statement and top-k makes the per-row active-set
size constant, so the step FLOPs estimate in ``scripts/sae_budget.py`` is the
actual work rather than an expectation over an L1 penalty. The decoder is kept
unit-norm after every step (the standard reparameterisation) so a feature's
decoder row is a direction that ``sae.feature_ablation`` can consume directly.

Scaling: the dictionary sees standardised rows (per-dimension mean/std of the
TRAIN split only -- a held-out row never contributes to them). Reconstruction is
mapped back to raw units before FVU, so a reported FVU is directly comparable to
``1.0 = predicting the train mean``.

FVU: ``sum ||x - x_hat||^2 / sum ||x - mean_train||^2`` over the evaluated rows.
The plan's baseline rule also includes a rank-k PCA control computed on the same
train rows with the same number of active directions, because a top-k SAE with
k active features is a rank-at-most-k reconstruction: an SAE that cannot beat
that control bought nothing over a linear projection of the same rank.
"""
from __future__ import annotations

import json
import math
import time

import torch
from torch import nn


class TopKSAE(nn.Module):
    """``x_hat = topk(relu(W_enc (x - b_dec) + b_enc)) @ W_dec + b_dec``."""

    def __init__(self, hidden: int, features: int, k: int):
        super().__init__()
        if features < 1:
            raise ValueError(f'features is {features}; an SAE needs at least one')
        if not 1 <= k <= features:
            raise ValueError(f'k is {k} with {features} features; k must be in [1, features]')
        self.hidden, self.dict_size, self.k = int(hidden), int(features), int(k)
        self.W_enc = nn.Parameter(torch.empty(features, hidden))
        self.b_enc = nn.Parameter(torch.zeros(features))
        self.W_dec = nn.Parameter(torch.empty(features, hidden))
        self.b_dec = nn.Parameter(torch.zeros(hidden))
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.detach())
        self._unit_norm_decoder()

    @torch.no_grad()
    def _unit_norm_decoder(self) -> None:
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_dec) @ self.W_enc.T + self.b_enc
        values, indices = torch.topk(torch.relu(pre), self.k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, indices, values)
        return acts

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        acts = self.encode(x)
        return acts @ self.W_dec + self.b_dec, acts

    @torch.no_grad()
    def decoder_directions(self) -> torch.Tensor:
        """Unit-norm ``[features, hidden]`` directions, one per feature."""
        return self.W_dec.detach() / self.W_dec.detach().norm(dim=1, keepdim=True).clamp_min(1e-8)


def _sync(device) -> None:
    """Wait for queued device work so a wall-clock reading is real work, not launch."""
    kind = torch.device(device).type
    if kind == 'xpu':
        torch.xpu.synchronize()
    elif kind == 'cuda':
        torch.cuda.synchronize()


def _as_matrix(rows) -> torch.Tensor:
    """Stack captured rows (``[layers, hidden]`` tensors or ``[hidden]`` lists) into ``[n, hidden]``."""
    tensors = []
    for row in rows:
        tensor = row if torch.is_tensor(row) else torch.as_tensor(row)
        if tensor.ndim != 1:
            raise ValueError(f'row has shape {tuple(tensor.shape)}; expected one layer, [hidden]')
        tensors.append(tensor.float())
    if not tensors:
        raise ValueError('no rows to stack')
    widths = {tensor.numel() for tensor in tensors}
    if len(widths) != 1:
        raise ValueError(f'rows disagree on hidden width: {sorted(widths)}')
    return torch.stack(tensors)


def select_layer(records, layer: int) -> list[torch.Tensor]:
    """Pull one layer out of ``(meta, [layers, hidden])`` records, refusing a mismatched depth."""
    out = []
    for _, rec in records:
        tensor = rec if torch.is_tensor(rec) else torch.as_tensor(rec)
        if tensor.ndim != 2:
            raise ValueError(f'record has shape {tuple(tensor.shape)}; expected [layers, hidden]')
        if not 0 <= layer < tensor.shape[0]:
            raise ValueError(f'layer {layer} outside [0, {tensor.shape[0]})')
        out.append(tensor[layer].float())
    return out


class Standardiser:
    """Per-dimension standardisation fit on the train split only."""

    def __init__(self, train: torch.Tensor):
        self.mean = train.mean(dim=0, keepdim=True)
        self.scale = train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.scale

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.scale + self.mean

    def state(self) -> dict:
        return {'mean': self.mean.flatten().tolist(), 'scale': self.scale.flatten().tolist()}


def fvu(x: torch.Tensor, x_hat: torch.Tensor, mean: torch.Tensor | None = None) -> float:
    """Fraction of variance unexplained, in raw units; ``1.0`` is the mean predictor."""
    reference = x.mean(dim=0, keepdim=True) if mean is None else mean
    residual = float(((x - x_hat) ** 2).sum())
    total = float(((x - reference) ** 2).sum())
    if total <= 0:
        raise ValueError('zero variance in the evaluated rows; FVU is undefined')
    return residual / total


def pca_control_fvu(train: torch.Tensor, heldout: torch.Tensor, rank: int) -> float:
    """Held-out FVU of the best rank-``rank`` linear reconstruction fit on ``train``."""
    mean = train.mean(dim=0, keepdim=True)
    centred = train - mean
    # economy SVD: at most min(n, hidden) components exist.
    _, _, basis = torch.linalg.svd(centred, full_matrices=False)
    basis = basis[: min(rank, basis.shape[0])]
    reconstruction = (heldout - mean) @ basis.T @ basis + mean
    return fvu(heldout, reconstruction, mean)


def train_sae(rows, *, features: int, k: int, steps: int, batch: int, lr: float,
              seed: int, device: str | torch.device = 'cpu',
              record_every: int = 0,
              deterministic: bool = True) -> tuple[TopKSAE, Standardiser, dict]:
    """Train one top-k SAE on ``rows`` (train split). Deterministic for a given seed.

    Returns the model, the standardiser (needed to score held-out rows) and a
    history dict with per-record loss and the device seconds spent.

    ``deterministic`` asks torch for deterministic kernels. On XPU the default
    reduction order is not stable: two runs at the same seed differ by ~1e-6 in
    the decoder and a digest comparison fails. With the flag on, the same seed
    reproduces the weights bit-exactly on both CPU and XPU (measured), which is
    what makes a committed checkpoint's digest mean something.
    """
    matrix = _as_matrix(rows)
    standardiser = Standardiser(matrix)
    data = standardiser(matrix).to(device)
    if deterministic:
        torch.use_deterministic_algorithms(True)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    torch.manual_seed(seed)
    model = TopKSAE(matrix.shape[1], features, k).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    std = data.std().clamp_min(1e-6)

    _sync(device)
    started = time.perf_counter()
    history = []
    model.train()
    for step in range(steps):
        index = torch.randint(0, data.shape[0], (min(batch, data.shape[0]),), generator=generator).to(device)
        xb = data[index]
        x_hat, acts = model(xb)
        # MSE plus the standard sparsity-promoting term on active magnitudes.
        loss = ((x_hat - xb) ** 2).mean() + 1e-3 * acts.mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        # The decoder carries the dictionary; rescaled every step so a feature is a direction.
        with torch.no_grad():
            model.W_dec.div_(model.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))
            model.b_dec.clamp_(-std, std)
        if record_every and (step % record_every == 0 or step == steps - 1):
            history.append({'step': step, 'loss': float(loss.detach())})
    _sync(device)
    seconds = time.perf_counter() - started
    model.eval()
    return model, standardiser, {'seconds': seconds, 'steps': steps, 'loss_trace': history,
                                 'hidden': int(matrix.shape[1]), 'rows': int(matrix.shape[0])}


@torch.no_grad()
def score_heldout(model: TopKSAE, standardiser: Standardiser, heldout_rows, train_rows,
                  device: str | torch.device = 'cpu') -> dict:
    """Held-out FVU of the SAE and of the same-rank PCA control in raw units."""
    heldout = _as_matrix(heldout_rows)
    train = _as_matrix(train_rows)
    model.eval()
    x_hat, _ = model(standardiser(heldout).to(device))
    x_hat = (x_hat * standardiser.scale.to(x_hat.device) + standardiser.mean.to(x_hat.device)).cpu()
    mean = train.mean(dim=0, keepdim=True)
    return {
        'rows': int(heldout.shape[0]),
        'mse': float(((heldout - x_hat) ** 2).mean()),
        'fvu': fvu(heldout, x_hat, mean),
        'pca_control_fvu': pca_control_fvu(train, heldout, model.k),
        'baseline_rule': 'fvu < 1.0 and fvu < pca_control_fvu',
    }


@torch.no_grad()
def feature_table(model: TopKSAE, standardiser: Standardiser, refusal_rows, compliant_rows,
                  *, top_k: int, device: str | torch.device = 'cpu') -> list[dict]:
    """Top features by |activation contrast| between refusal and compliant rows.

    The contrast is ``mean(act | refusal) - mean(act | compliant)`` in feature
    space. Each row also carries ``cos_to_axis`` -- the cosine between the
    feature's unit decoder direction and the refusal-minus-compliant mean axis in
    the same standardised space -- because ``jdewitt/sae.py`` requires that
    number next to any feature a downstream arm ablates: a cosine near 1 means
    the feature is the contrast axis again, not a new thing.
    """
    refusal = _as_matrix(refusal_rows)
    compliant = _as_matrix(compliant_rows)
    model.eval()
    acts_ref = model.encode(standardiser(refusal).to(device)).cpu()
    acts_comp = model.encode(standardiser(compliant).to(device)).cpu()
    mean_ref, mean_comp = acts_ref.mean(dim=0), acts_comp.mean(dim=0)
    contrast = mean_ref - mean_comp
    axis = standardiser(refusal).mean(dim=0) - standardiser(compliant).mean(dim=0)
    axis = axis / axis.norm().clamp_min(1e-8)
    directions = model.decoder_directions().cpu()
    order = torch.argsort(contrast.abs(), descending=True)[:top_k]
    rows = []
    for feature in order.tolist():
        rows.append({
            'feature': int(feature),
            'mean_refusal': float(mean_ref[feature]),
            'mean_compliant': float(mean_comp[feature]),
            'contrast': float(contrast[feature]),
            'active_refusal': float((acts_ref[:, feature] > 0).float().mean()),
            'active_compliant': float((acts_comp[:, feature] > 0).float().mean()),
            'decoder_norm': float(model.W_dec.detach()[feature].norm()),
            'cos_to_axis': float(torch.dot(directions[feature], axis)),
        })
    return rows


def digest_tensor(tensor: torch.Tensor) -> str:
    """sha256 of a tensor's raw bytes, so a checkpoint's numbers are reproducible."""
    import hashlib
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def checkpoint_state(model: TopKSAE, standardiser: Standardiser) -> dict:
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    state.update({'standardiser_mean': standardiser.mean.flatten().cpu(),
                  'standardiser_scale': standardiser.scale.flatten().cpu()})
    return state


def config_json(config: dict) -> str:
    return json.dumps(config, indent=1, sort_keys=True)
