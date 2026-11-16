# SPDX-License-Identifier: AGPL-3.0-or-later
# Projection mechanics adapted from Heretic, Copyright (C) 2025-2026
# Philipp Emanuel Weidmann + contributors. See NOTICE for pinned provenance.
import torch
import math
from torch import nn
from types import SimpleNamespace
from .architectures import ARCHITECTURES, EDIT_TARGETS, MAX_STRENGTH

SIDES = ('output', 'input')


def merged_weight(weight, a, b, row_scale=None):
    """One same-device workspace, identical FP32 addition/cast for search and export."""
    if weight.device != a.device or weight.device != b.device:
        raise ValueError('Merged operator requires same-device weights and edits')
    result = weight.float() + b.float() @ a.float()
    if row_scale is not None:
        result = result * row_scale.float()
    return result.to(weight.dtype)


class AblatedLinear(nn.Module):
    """Delta W from a projection of the residual direction.

    ``side='output'`` removes the component the module WRITES into the residual
    stream: ``dW = -s * d.T @ (d @ W)``, so the module's output loses the
    direction. It needs ``out_features == hidden``.

    ``side='input'`` removes the component the module READS from the residual
    stream: ``dW = -s * (W @ d.T) @ d``, so the module no longer sees the
    direction in its input. It needs ``in_features == hidden``, which holds for
    q/k/v/gate/up even under grouped-query attention where ``out_features`` does
    not. Editing only the write-back matrices leaves the read side untouched.
    """
    def __init__(self, base, rank, operator='auto', side='output'):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank < 1:
            raise ValueError('Adapter needs a dense Linear with positive rank')
        if side not in SIDES:
            raise ValueError('Unknown projection side')
        self.side = side
        self.base = base
        quantized = getattr(base.weight, 'quant_state', None) is not None
        if operator == 'auto':
            operator = 'live' if quantized else 'merged-resident'
        if operator not in {'live', 'merged-resident'}:
            raise ValueError('Unknown trial operator')
        if quantized and operator == 'merged-resident':
            raise ValueError('Resident export parity needs original full-precision weights, not NF4')
        self.operator = operator
        dtype = getattr(base, 'compute_dtype', None) or base.weight.dtype
        if not dtype.is_floating_point:
            raise ValueError('Quantized layer needs an explicit floating compute dtype')
        self.register_buffer('A', torch.zeros(rank, base.in_features, device=base.weight.device, dtype=dtype))
        self.register_buffer('B', torch.zeros(base.out_features, rank, device=base.weight.device, dtype=dtype))
        self.register_buffer('row_scale', torch.ones(base.out_features, 1, device=base.weight.device, dtype=torch.float32))
        self.enabled = False

    def canonical_weight(self):
        return self.base.weight

    def merge_source(self, weight):
        return merged_weight(weight, self.A, self.B, self.row_scale)

    def effective_weight(self):
        return self.rounded_weight(self.base.weight) if self.enabled else self.base.weight

    def rounded_weight(self,weight):
        cache=getattr(self,'merge_cache',None)
        return cache.weight(self,weight) if cache is not None else self.merge_source(weight)

    def direction_shape(self):
        # Output side: the direction lives in the module's OUTPUT space.
        # Input side: it lives in the module's INPUT space (the residual stream).
        width = self.base.out_features if self.side == 'output' else self.base.in_features
        return (self.A.shape[0], width)

    def direction_width(self):
        return self.direction_shape()[1]

    def set_directions(self, directions, strength, row_normalization='none'):
        # One (rank, width) direction per target, in that target's own side space.
        if (not isinstance(directions, torch.Tensor) or tuple(directions.shape) != self.direction_shape()
                or not torch.isfinite(directions).all() or not 0 <= strength <= MAX_STRENGTH):
            raise ValueError('Invalid projection')
        d = directions.to(device=self.base.weight.device, dtype=torch.float32)
        if not torch.allclose(d @ d.T, torch.eye(d.shape[0], device=d.device), atol=1e-4):
            raise ValueError('Directions must be orthonormal')
        if row_normalization not in {'none', 'pre', 'full'}:
            raise ValueError('Unknown row normalization')

        with torch.no_grad():
            self.row_scale.fill_(1)
            weight = self.canonical_weight()
            if getattr(weight, 'quant_state', None) is not None:
                from bitsandbytes.functional import dequantize_4bit
                weight = dequantize_4bit(weight.data, weight.quant_state)
            original = weight.float()
            norms = original.norm(dim=-1, keepdim=True)
            if self.side == 'input':
                self.A.copy_(d.to(self.A.dtype))
                self.B.copy_((-strength * (original @ d.T)).to(self.B.dtype))
                # Input projection acts independently on each row: normalizing
                # first and restoring its norm cancels algebraically for 'pre'.
                # 'full' rescales the rounded edited rows below. It preserves
                # the read-side nullspace while restoring each row's magnitude.
            else:
                normalized = original / norms.clamp_min(1e-12) if row_normalization != 'none' else original
                a = d @ normalized
                self.A.copy_(a.to(self.A.dtype))
                b = -strength * d.T
                if row_normalization != 'none':
                    b = norms * b
                self.B.copy_(b.to(self.B.dtype))
            if row_normalization == 'full':
                changed = original + self.B.float() @ self.A.float()
                new_norms = changed.norm(dim=-1, keepdim=True)
                if ((new_norms < 1e-12) & (norms > 1e-12)).any():
                    self.reset()
                    raise ValueError('Norm-preserving edit collapsed a nonzero row')
                self.row_scale.copy_(torch.where(norms > 0, norms / new_norms.clamp_min(1e-12), torch.ones_like(norms)))
        self.enabled = strength != 0

    def forward(self, x):
        if not self.enabled:
            return self.base(x)
        if self.operator == 'merged-resident':
            # Base storage never moves or changes. The temporary lives only for
            # this projection; do not cache a second model's worth of weights.
            weight = self.effective_weight()
            return nn.functional.linear(x, weight, self.base.bias)
        return self.base(x) + nn.functional.linear(nn.functional.linear(x, self.A), self.B)

    def set_factors(self, a, b, strength):
        if tuple(a.shape) != tuple(self.A.shape) or tuple(b.shape) != tuple(self.B.shape) or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('Invalid learned edit factors')
        with torch.no_grad():
            self.row_scale.fill_(1)
            self.A.copy_(a.to(device=self.A.device, dtype=self.A.dtype))
            self.B.copy_((b * strength).to(device=self.B.device, dtype=self.B.dtype))
        self.enabled = strength != 0

    def reset(self):
        cache=getattr(self,'merge_cache',None)
        if cache is not None:
            cache.discard(id(self))
        self.A.zero_()
        self.B.zero_()
        self.row_scale.fill_(1)
        self.enabled = False

    def merge(self):
        if getattr(self.base.weight, 'quant_state', None) is not None:
            raise ValueError('NF4 bases require streaming merge into original safetensors')
        with torch.no_grad():
            self.base.weight.copy_(self.merge_source(self.base.weight))
        self.reset()
        return self.base


class AblatedTensor(AblatedLinear):
    """Parametrize fused experts in their native orientation, no weight clone."""
    def __init__(self, weight, rank, transposed=False):
        nn.Module.__init__(self)
        if weight.ndim != 3 or not weight.dtype.is_floating_point:
            raise ValueError('Fused experts need a floating [experts,output,input] tensor')
        self.side = 'output'
        self.base = SimpleNamespace(weight=weight)
        self.transposed = transposed
        experts, output, inputs = self.canonical_weight().shape
        self.register_buffer('A', torch.zeros(experts, rank, inputs, device=weight.device, dtype=weight.dtype))
        self.register_buffer('B', torch.zeros(experts, output, rank, device=weight.device, dtype=weight.dtype))
        self.register_buffer('row_scale', torch.ones(experts, output, 1, device=weight.device, dtype=torch.float32))
        self.enabled, self.operator = False, 'merged-resident'

    def canonical_weight(self):
        return self.base.weight.transpose(-1, -2) if self.transposed else self.base.weight

    def merge_source(self, weight):
        canonical = weight.transpose(-1, -2) if self.transposed else weight
        result = merged_weight(canonical, self.A, self.B, self.row_scale)
        return result.transpose(-1, -2).contiguous() if self.transposed else result

    def direction_shape(self):
        # Fused experts broadcast one direction across every expert: A holds a
        # per-expert copy, so the direction's own shape is (rank, output).
        return (self.A.shape[-2], self.B.shape[-2])

    def forward(self, weight):
        return self.rounded_weight(weight) if self.enabled else weight


def get_path(obj, path):
    for part in path.split('.'):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def expand_path(obj, parts, prefix=''):
    if not parts:
        return [(prefix, obj)]
    part, *rest = parts
    if part == '*':
        return [item for name, child in obj.named_children() for item in expand_path(child, rest, f'{prefix}.{name}'.strip('.'))]
    try:
        child = obj[int(part)] if part.isdigit() else getattr(obj, part)
    except (AttributeError, IndexError, KeyError, TypeError) as e:
        raise ValueError(f'Architecture adapter path missing: {prefix}.{part}') from e
    return expand_path(child, rest, f'{prefix}.{part}'.strip('.'))


def target_groups(spec, edit_targets):
    """(group, component, side, patterns) for the requested edit surface."""
    if edit_targets not in EDIT_TARGETS:
        raise ValueError(f'edit_targets must be one of {list(EDIT_TARGETS)}')
    groups = [('attn', 'attn', 'output', spec['attn']), ('mlp', 'mlp', 'output', spec['mlp'])]
    if edit_targets == 'full':
        groups += [('attn_in', 'attn', 'input', spec['attn_in']), ('mlp_in', 'mlp', 'input', spec['mlp_in'])]
    return groups


def discover_targets(model, edit_targets='writeback'):
    model_type = model.config.model_type
    if model_type not in ARCHITECTURES:
        raise ValueError(f'No architecture adapter for {model_type!r}')
    spec = ARCHITECTURES[model_type]
    layers = None
    layers_path = None
    for candidate in spec['layers']:
        try:
            layers = get_path(model, candidate)
            layers_path = candidate
            break
        except (AttributeError, IndexError, TypeError):
            layers = None
    if layers is None or layers_path is None:
        raise ValueError(f'No text-backbone layer path for {model_type!r}')
    text_config = getattr(model.config, 'text_config', model.config)
    groups = target_groups(spec, edit_targets)
    targets = []
    if not len(layers):
        raise ValueError('Architecture adapter found no layers')
    for index, layer in enumerate(layers):
        before = len(targets)
        for group, component, side, patterns in groups:
            if component == 'mlp' and 'mlp_bottleneck_output' in spec:
                try:
                    bottleneck = get_path(layer, spec['mlp_bottleneck_output'])
                except AttributeError:
                    bottleneck = None
                if isinstance(bottleneck, nn.Linear):
                    patterns = spec['mlp_bottleneck_targets']
            for pattern in patterns:
                try:
                    found = expand_path(layer, pattern.split('.'))
                except ValueError:
                    continue
                for path, module in found:
                    if isinstance(module, nn.Module) and not isinstance(module, nn.Linear) and isinstance(getattr(module, 'linear', None), nn.Linear):
                        path, module = path + '.linear', module.linear
                    if isinstance(module, nn.Linear):
                        width = module.in_features if side == 'input' else module.out_features
                        valid = width == text_config.hidden_size
                    elif side == 'input':
                        # A fused expert tensor has no separable read-side
                        # operator here; refuse rather than edit it wrongly.
                        raise ValueError(f'Input-side editing needs a dense Linear at layer {index}: {path}')
                    elif isinstance(module, nn.Parameter) and module.ndim == 3:
                        valid = module.shape[-1 if spec['fused_transposed'] else -2] == text_config.hidden_size
                    else:
                        raise ValueError(f'Unsupported output weight layout at layer {index}: {path}')
                    if not valid:
                        raise ValueError(f'Projection does not match residual width at layer {index}: {path}')
                    full = f'{layers_path}.{index}.{path}'
                    if not any(t[3] == full for t in targets):
                        targets.append((index, group, side, full, module))
        if len(targets) == before:
            raise ValueError(f'Architecture adapter has no targets at layer {index}; refusing silent layer skips')
    return layers, targets


class Editor:
    def __init__(self, model, rank, operator='auto', edit_targets='writeback', merged_cache_bytes=0):
        from .merged_cache import MergedCache
        self.merge_cache=MergedCache(merged_cache_bytes)
        self.edit_targets = edit_targets
        self.layers, raw = discover_targets(model, edit_targets)
        self.targets = []
        for index, group, side, full, module in raw:
            parent_path, name = full.rsplit('.', 1)
            parent = get_path(model, parent_path)
            if isinstance(module, nn.Parameter):
                from torch.nn.utils import parametrize
                adapter = AblatedTensor(module, rank, ARCHITECTURES[model.config.model_type]['fused_transposed'])
                adapter.source_key = full
                parametrize.register_parametrization(parent, name, adapter)
            else:
                adapter = AblatedLinear(module, rank, operator, side)
                adapter.source_key = full + '.weight'
                setattr(parent, name, adapter)
            adapter.source_path = full
            adapter.merge_cache = self.merge_cache
            self.targets.append((index, group, side, parent, name, adapter))
        self.groups = sorted({group for _, group, _, _, _, _ in self.targets})
        self.group_sides = {group: side for _, group, side, _, _, _ in self.targets}
        self.layers_with = {group: sorted({index for index, g, _, _, _, _ in self.targets if g == group})
                            for group in self.groups}

    def reset(self):
        for _, _, _, _, _, adapter in self.targets:
            adapter.reset()

    def plan_shape(self, rank, plan=None):
        first = self.targets[0][-1]
        return (len(self.layers), 1 if plan is not None and plan['method'] == 'ara' else rank,
                first.direction_width())

    def apply(self, directions, params):
        self.reset()
        plan = directions if isinstance(directions, dict) else None
        directions = plan['basis'] if plan is not None else directions
        profile = plan.get('layer_profile') if plan is not None else None
        if not isinstance(params, dict) or set(params) != set(self.groups):
            raise ValueError(f'Edit parameters must cover exactly {self.groups}')
        expected = {'strength'} if profile is not None else {'strength', 'center', 'width', 'floor'}
        for values in params.values():
            if not isinstance(values, dict) or set(values) != expected:
                raise ValueError('Incomplete edit parameters')
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in values.values()):
                raise ValueError('Edit parameters must be finite numbers')
            if not 0 <= values['strength'] <= MAX_STRENGTH:
                raise ValueError('Edit strength is outside the supported range')
            if profile is None and not (0 <= values['center'] <= 1 and 0 < values['width'] <= 1 and 0 <= values['floor'] <= 1):
                raise ValueError('Edit parameters are outside the supported ranges')
        first = self.targets[0][-1]
        shape = self.plan_shape(first.A.shape[-2], plan)
        if not isinstance(directions, torch.Tensor) or tuple(directions.shape) != shape or not torch.isfinite(directions).all():
            raise ValueError('Directions have invalid shape or nonfinite values')
        d = directions.float()
        identity = torch.eye(shape[1], device=d.device).expand(shape[0], -1, -1)
        if not torch.allclose(d @ d.transpose(-1, -2), identity, atol=1e-4):
            raise ValueError('Every layer needs orthonormal directions')
        last = max(len(self.layers) - 1, 1)
        try:
            for index, group, side, _, _, adapter in self.targets:
                p = params[group]
                if profile is not None:
                    strength = p['strength'] * float(profile[index])
                else:
                    distance = abs(index / last - p['center'])
                    if distance > p['width']:
                        continue
                    strength = p['strength'] * (1 - distance / p['width'] * (1 - p['floor']))
                if plan is not None and plan['method'] == 'ara':
                    factors = plan['factors'][adapter.source_key]
                    adapter.set_factors(factors['A'], factors['B'], strength)
                elif plan is not None and plan['method'] == 'leace':
                    weight = adapter.canonical_weight().float()
                    left = plan['left'][index].to(weight.device)
                    right = plan['right'][index].to(weight.device)
                    if side == 'input':
                        a = right
                        b = weight @ left
                    else:
                        a = right @ weight
                        b = left.expand(*adapter.B.shape)
                    adapter.set_factors(a, b, -strength)
                else:
                    adapter.set_directions(directions[index], strength,
                        row_normalization=plan.get('row_normalization', 'none') if plan is not None else 'none')
        except BaseException:
            self.reset()
            raise

    def merge(self):
        for _, _, _, parent, name, adapter in self.targets:
            if isinstance(adapter, AblatedTensor):
                from torch.nn.utils import parametrize
                parametrize.remove_parametrizations(parent, name, leave_parametrized=True)
            else:
                setattr(parent, name, adapter.merge())
