import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from jdewitt.hf import require_b70, reply_boundary, HFModel
from jdewitt.editor import Editor


def test_boundary_after_closing_think_not_before():
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return ''.join(ids)
    assert reply_boundary(Tokenizer(), list('reason</think>answer'), thinking_open=True) == len('reason</think>')
    with pytest.raises(ValueError):
        reply_boundary(Tokenizer(), list('reason'), thinking_open=True)
    assert reply_boundary(Tokenizer(), list('answer'), thinking_open=False) == 0


@pytest.mark.xpu
def test_actual_b70_weight_edit_generation_and_full_kl():
    device = require_b70()
    torch.manual_seed(7)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2)).to(device=device, dtype=torch.float16).eval()
    editor = Editor(model, rank=1)
    backend = HFModel.from_objects(model, None, editor, max_tokens=8)
    prompt = torch.tensor([[1, 7, 8]], device=device)
    with torch.inference_mode():
        tokens = model.generate(prompt, max_new_tokens=8, do_sample=False, pad_token_id=0)
    directions = torch.zeros(2, 1, 32, device=device)
    directions[:, :, 0] = 1
    params = {c: {'strength': 1., 'center': .5, 'width': 1., 'floor': 1.} for c in ('attn','mlp')}
    editor.apply(directions, params)
    metric = backend.trajectory_kl(tokens, prompt_length=3, answer_start=0, chunk_size=3)
    assert metric['tokens'] == tokens.shape[1] - 3
    assert metric['full_reply_kl'] > 0
    assert metric['answer_kl'] > 0
    assert all(p.device.type == 'xpu' for p in model.parameters())
    editor.reset()
    assert backend.trajectory_kl(tokens, 3, 0, chunk_size=3)['full_reply_kl'] < 1e-7
    torch.xpu.synchronize()
