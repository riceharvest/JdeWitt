# SPDX-License-Identifier: AGPL-3.0-or-later
"""Explicit Transformers segment adapters. No arbitrary-architecture fallback.

All sequence positions travel through each segment; only the last position is
intervened on. Cache-free prefill means hybrid recurrent blocks start from zero
and finish their full sequence in one call. No branch borrows another branch's
KV/recurrent state. Finalist decoding replays the path once on the first prefill.
"""
import torch
from transformers.masking_utils import (
    create_causal_mask, create_sliding_window_causal_mask, create_recurrent_attention_mask,
)

from .tree_search import project_last
from .tree_config import SEGMENT_PATHS


class HFSegments:
    def __init__(self, hf, inputs):
        architecture = hf.config.model_type
        if architecture not in SEGMENT_PATHS:
            raise ValueError(f'No verified-shape segment adapter for {architecture}')
        if any(adapter.enabled for *_, adapter in hf.editor.targets):
            raise ValueError('Tree search requires unedited original weights, not active weight edits')
        if hf.model.training:
            raise ValueError('Tree search requires deterministic eval mode')
        root = hf.model
        for part in SEGMENT_PATHS[architecture].split('.'):
            root = getattr(root, part, None)
            if root is None:
                raise ValueError('Explicit tree text-backbone path is missing')
        if any(not hasattr(root, name) for name in ('layers', 'embed_tokens', 'rotary_emb', 'norm', 'config')):
            raise ValueError('Tree segment adapter is missing a required text-backbone component')
        self.root, self.layers, self.device = root, root.layers, hf.device
        self.config, self.inputs, self.architecture = root.config, inputs, architecture
        self.n_layers = self.config.num_hidden_layers
        if (len(self.layers) != self.n_layers or len(hf.editor.layers) != self.n_layers
                or any(a is not b for a, b in zip(self.layers, hf.editor.layers))):
            raise ValueError('Tree/editor layer identity mismatch')
        ids = inputs.get('input_ids')
        if (not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1
                or ids.shape[1] < 1 or ids.device != self.device
                or ids.dtype != torch.long or set(inputs) - {'input_ids', 'attention_mask'}):
            raise ValueError('Tree needs a single same-device text-only unpadded sequence')
        mask = inputs.get('attention_mask')
        if mask is not None and (mask.shape != ids.shape or mask.device != ids.device or not bool((mask == 1).all())):
            raise ValueError('Tree segment adapter accepts unpadded all-ones masks only')
        embedding = root.embed_tokens.weight
        if embedding.device != self.device:
            raise ValueError('Tree cannot offload embeddings or branch states')
        self.state_bytes = ids.shape[1] * self.config.hidden_size * embedding.element_size()
        self.positions = self.rotary = self.masks = None

    @torch.inference_mode()
    def prepare(self):
        if self.positions is not None:
            raise ValueError('Tree prompt context has already been prepared')
        hidden = self.root.embed_tokens(self.inputs['input_ids'])
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        mask_kwargs = dict(config=self.config, inputs_embeds=hidden,
                           attention_mask=self.inputs.get('attention_mask'), past_key_values=None,
                           position_ids=positions)
        types = getattr(self.config, 'layer_types', ['full_attention'] * self.n_layers)
        if len(types) != self.n_layers or set(types) - {'full_attention', 'sliding_attention', 'linear_attention'}:
            raise ValueError('Unverified tree layer/mask schedule')
        self.masks = {'full_attention': create_causal_mask(**mask_kwargs)}
        if 'sliding_attention' in types:
            self.masks['sliding_attention'] = create_sliding_window_causal_mask(**mask_kwargs)
        if 'linear_attention' in types:
            self.masks['linear_attention'] = create_recurrent_attention_mask(**mask_kwargs)
        if self.architecture.startswith('qwen3_5'):
            # Text-only mRoPE: upstream expands four channels and passes the last
            # three to rotary, while the first supplies ordinary text positions.
            rotary_positions = positions.unsqueeze(0).expand(3, 1, -1)
        else:
            rotary_positions = positions
        self.positions = positions
        self.rotary = self.root.rotary_emb(hidden, rotary_positions)
        self.types = types
        return hidden

    @torch.inference_mode()
    def advance(self, hidden, start, stop):
        if self.positions is None or not 0 <= start < stop <= self.n_layers:
            raise ValueError('Invalid/unprepared tree layer segment')
        if hidden.device != self.device or hidden.shape[:2] != self.inputs['input_ids'].shape:
            raise ValueError('Tree segment state shape/device mismatch')
        for index in range(start, stop):
            hidden = self.layers[index](hidden, attention_mask=self.masks[self.types[index]],
                position_embeddings=self.rotary, position_ids=self.positions,
                past_key_values=None, use_cache=False)
            if not isinstance(hidden, torch.Tensor):
                raise ValueError('Segment adapter expected a tensor block output')
        return hidden


class PrefillReplay:
    """Replay exact selected residual edits once, then leave cached decode alone."""
    def __init__(self, layers, actions, path, prompt_length):
        self.layers, self.path, self.prompt_length = layers, path, prompt_length
        self.actions = {a.name: a for a in actions}
        self.handles, self.applied = [], set()
        if type(prompt_length) is not int or prompt_length < 1:
            raise ValueError('Replay requires the exact positive prompt length')
        cps = [s['checkpoint'] for s in path]
        if cps != sorted(set(cps)) or any(type(i) is not int or not 0 <= i < len(layers) for i in cps):
            raise ValueError('Invalid replay checkpoint path')
        for step in path:
            if step['action'] == 'none':
                if step['strength'] != 0:
                    raise ValueError('No-edit replay must have zero strength')
            elif step['action'] not in self.actions or step['checkpoint'] not in self.actions[step['action']].bases:
                raise ValueError('Unknown replay action/direction')

    def _hook(self, step):
        checkpoint = step['checkpoint']
        def apply(module, args, output):
            if checkpoint in self.applied:
                return output
            if not isinstance(output, torch.Tensor) or output.shape[:2] != (1, self.prompt_length):
                raise ValueError('First replay call does not match the searched prefill')
            self.applied.add(checkpoint)
            if step['action'] == 'none':
                return output
            basis = self.actions[step['action']].bases[checkpoint]
            return project_last(output, basis, step['strength'])
        return apply

    def __enter__(self):
        if self.handles or self.applied:
            raise ValueError('Replay context cannot be reused')
        try:
            for step in self.path:
                self.handles.append(self.layers[step['checkpoint']].register_forward_hook(self._hook(step)))
        except BaseException:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if exc_type is None and self.applied != {s['checkpoint'] for s in self.path}:
            raise ValueError('Selected tree path was not replayed in full')
