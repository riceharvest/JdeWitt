"""Fixture-only adapter/replay tests; real transformer parity is XPU-marked."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, number):
        super().__init__()
        self.number, self.calls = number, []

    def forward(self, h, **kwargs):
        self.calls.append(kwargs)
        return h + self.number


def target_fixture(architecture='qwen3'):
    blocks = nn.ModuleList([Block(1), Block(2), Block(3)])
    class Rotary(nn.Module):
        def forward(self, h, position_ids):
            return (torch.ones_like(h), torch.zeros_like(h))
    root = SimpleNamespace(layers=blocks, embed_tokens=nn.Embedding(8, 2),
                           rotary_emb=Rotary(), norm=nn.Identity(), config=SimpleNamespace(
                               model_type=architecture, num_hidden_layers=3, hidden_size=2,
                               layer_types=['full_attention'] * 3))
    model = SimpleNamespace(model=root, config=root.config, training=False)
    hf = SimpleNamespace(model=model, config=root.config, device=torch.device('cpu'),
                         editor=SimpleNamespace(layers=blocks, targets=[]))
    return hf, root


def test_segment_adapter_calls_only_requested_layers_and_prepares_positions_once(monkeypatch):
    from b70_ablator.tree_hf import HFSegments
    hf, root = target_fixture()
    mask_calls = []
    def mask(**kwargs):
        mask_calls.append(kwargs)
        return torch.zeros(1, 1, 3, 3)
    monkeypatch.setattr('b70_ablator.tree_hf.create_causal_mask', mask)
    backend = HFSegments(hf, {'input_ids': torch.tensor([[1, 2, 3]]),
                              'attention_mask': torch.ones(1, 3, dtype=torch.long)})
    h = backend.prepare()
    expected = h + 6
    mid = backend.advance(h, 0, 1)
    final = backend.advance(mid, 1, 3)
    torch.testing.assert_close(final, expected)
    assert len(mask_calls) == 1
    assert all(len(b.calls) == 1 for b in root.layers)
    assert all(b.calls[0]['use_cache'] is False and b.calls[0]['past_key_values'] is None for b in root.layers)
    assert backend.state_bytes == h.numel() * h.element_size()
    assert backend.n_layers == 3


def test_prefill_replay_applies_exact_path_once_and_cleans_hooks_on_error():
    from b70_ablator.tree_hf import PrefillReplay
    from b70_ablator.tree_search import Action
    layers = [Block(1), Block(2)]
    action = Action('x', {0: torch.tensor([[1., 0.]])})
    path = [{'checkpoint': 0, 'action': 'x', 'strength': 1.}]
    with PrefillReplay(layers, [action], path, prompt_length=1):
        h = layers[0](torch.tensor([[[2., 4.]]]))
        torch.testing.assert_close(h, torch.tensor([[[0., 5.]]]))
        # One-token prompt must not cause every cached decode to be re-edited.
        h = layers[0](torch.tensor([[[2., 4.]]]))
        torch.testing.assert_close(h, torch.tensor([[[3., 5.]]]))
    assert not layers[0]._forward_hooks
    with pytest.raises(RuntimeError, match='interrupted'):
        with PrefillReplay(layers, [action], path, prompt_length=2):
            raise RuntimeError('interrupted')
    assert not any(layer._forward_hooks for layer in layers)
    with pytest.raises(ValueError, match='not replayed'):
        with PrefillReplay(layers, [action], path, prompt_length=2):
            pass
    assert not any(layer._forward_hooks for layer in layers)


def test_prefill_replay_matches_segmented_interventions(monkeypatch):
    from b70_ablator.tree_hf import HFSegments, PrefillReplay
    from b70_ablator.tree_search import Action, project_last
    hf, root = target_fixture()
    monkeypatch.setattr('b70_ablator.tree_hf.create_causal_mask', lambda **kwargs: None)
    backend = HFSegments(hf, {'input_ids': torch.tensor([[1, 2]])})
    initial = backend.prepare()
    action = Action('x', {0: torch.tensor([[1., 0.]]), 1: torch.tensor([[0., 1.]])})
    path = [{'checkpoint': 0, 'action': 'x', 'strength': .5},
            {'checkpoint': 1, 'action': 'x', 'strength': 1.}]
    hidden = backend.advance(initial.clone(), 0, 1)
    hidden = backend.advance(project_last(hidden, action.bases[0], .5), 1, 2)
    hidden = backend.advance(project_last(hidden, action.bases[1], 1.), 2, 3)
    with PrefillReplay(root.layers, [action], path, prompt_length=2):
        replay = backend.advance(initial.clone(), 0, 3)
    torch.testing.assert_close(replay, hidden)


def test_unknown_architecture_and_active_weight_edits_are_rejected():
    from b70_ablator.tree_hf import HFSegments
    hf, _ = target_fixture('unverified')
    with pytest.raises(ValueError, match='segment adapter'):
        HFSegments(hf, {'input_ids': torch.tensor([[1]])})
    hf, _ = target_fixture()
    hf.editor.targets = [(0, SimpleNamespace(enabled=True))]
    with pytest.raises(ValueError, match='weight edits'):
        HFSegments(hf, {'input_ids': torch.tensor([[1]])})


@pytest.mark.xpu
@pytest.mark.parametrize('architecture', ['llama', 'qwen3', 'qwen3_5_text'])
def test_real_b70_segment_and_replay_logits_match_full_forward(architecture):
    from b70_ablator.hf import HFModel, require_b70
    from b70_ablator.editor import Editor
    from b70_ablator.tree_hf import HFSegments, PrefillReplay
    from b70_ablator.tree_search import Action, project_last
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    constructors = {'llama': (LlamaConfig, LlamaForCausalLM),
                    'qwen3': (Qwen3Config, Qwen3ForCausalLM),
                    'qwen3_5_text': (Qwen3_5TextConfig, Qwen3_5ForCausalLM)}
    config_cls, model_cls = constructors[architecture]
    device = require_b70()
    kwargs = dict(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    if architecture == 'qwen3_5_text':
        kwargs.update(layer_types=['linear_attention', 'full_attention', 'linear_attention'],
                      linear_key_head_dim=8, linear_value_head_dim=8,
                      linear_num_key_heads=2, linear_num_value_heads=4)
    config = config_cls(**kwargs)
    config._attn_implementation = 'eager'
    model = model_cls(config).to(device=device, dtype=torch.bfloat16).eval()
    hf = HFModel.from_objects(model, None, Editor(model, 1), max_tokens=4, ctx_size=64)
    inputs = {'input_ids': torch.tensor([[1, 2, 3, 4]], device=device)}
    with torch.inference_mode():
        backend = HFSegments(hf, inputs)
        initial = backend.prepare()
        h = backend.advance(initial.clone(), 0, 1)
        h = backend.advance(h, 1, 3)
        expected = model(**inputs, use_cache=False).logits[:, -1]
        logits = model.lm_head(backend.root.norm(h[:, -1]))
        torch.testing.assert_close(logits, expected, atol=0, rtol=0)
        basis = torch.zeros(1, 32, device=device)
        basis[0, 0] = 1
        action = Action('x', {0: basis, 1: basis})
        path = [{'checkpoint': 0, 'action': 'x', 'strength': .5},
                {'checkpoint': 1, 'action': 'x', 'strength': 1.}]
        h = backend.advance(initial.clone(), 0, 1)
        h = backend.advance(project_last(h, basis, .5), 1, 2)
        h = backend.advance(project_last(h, basis, 1.), 2, 3)
        with PrefillReplay(backend.layers, [action], path, prompt_length=4):
            replay = model(**inputs, use_cache=False).logits[:, -1]
        torch.testing.assert_close(model.lm_head(backend.root.norm(h[:, -1])), replay, atol=0, rtol=0)
    hf.release()
