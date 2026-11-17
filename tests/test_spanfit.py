"""Tests for b70_ablator.spanfit — answer-span per-layer rank-1 removal."""
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/mnt/ssd/dev workspace/llama.cpp-dspark/gguf-py')
from b70_ablator import spanfit
from b70_ablator.ggufio import dequantize_tensor

L, D, N = 4, 16, 8


def _records():
    g = torch.Generator().manual_seed(7)
    axis = torch.randn(L, D, generator=g)
    axis = axis / axis.norm(dim=1, keepdim=True)
    bad = axis.unsqueeze(0) + 0.05 * torch.randn(N, L, D, generator=g)
    good = 0.05 * torch.randn(N, L, D, generator=g)
    recs = ([({'label': 'bad'}, bad[i].float()) for i in range(N)] +
            [({'label': 'good'}, good[i].float()) for i in range(N)])
    return recs, axis


def test_per_layer_axes_recovers_direction():
    recs, axis = _records()
    fit = spanfit.per_layer_axes(recs)
    assert fit.shape == (L, D)
    cos = torch.nn.functional.cosine_similarity(fit, axis, dim=1)
    # finite-sample noise (0.05 sd, N=8/group) keeps cos a bit below 1
    assert (cos > 0.95).all(), cos


def test_per_layer_axes_unit_norm_rows():
    fit = spanfit.per_layer_axes(_records()[0])
    assert torch.allclose(fit.norm(dim=1), torch.ones(L), atol=1e-5)


def test_group_means_dict_records():
    recs = [('bad', {0: np.ones(D), 1: np.zeros(D)}),
            ('good', {0: np.zeros(D), 1: np.ones(D)})]
    m = spanfit.group_means(recs, 'bad')
    assert m.shape == (1, 2, D)
    assert torch.allclose(m[0, 0], torch.ones(D, dtype=m.dtype))
    assert torch.allclose(m[0, 1], torch.zeros(D, dtype=m.dtype))


def test_removal_delta_kills_output_component():
    g = torch.Generator().manual_seed(11)
    a = torch.randn(D, generator=g)
    a = a / a.norm()
    W = torch.randn(D, N, generator=g)  # logical [out, in] with out==D
    delta = -(a.reshape(-1, 1) @ (a.reshape(1, -1) @ W))
    Wn = W + delta
    assert float((a @ Wn).norm()) < 1e-4 * float((a @ W).norm())


def test_inplace_removal_apply_revert():
    g = torch.Generator().manual_seed(5)
    model = nn.Module()
    model.layers = nn.ModuleList()
    for i in range(L):
        block = nn.Module()
        block.o_proj = nn.Linear(D, D, bias=False)
        with torch.no_grad():
            block.o_proj.weight.copy_(torch.randn(D, D, generator=g))
        model.layers.append(block)
    axes = torch.nn.functional.normalize(torch.randn(L, D, generator=g), dim=1)
    before = [float((axes[i] @ model.layers[i].o_proj.weight.detach().float()).norm())
              for i in range(L)]
    ir = spanfit.InPlaceRemoval(model, axes, key_format='layers.{L}.{suffix}')
    n = ir.apply(['o_proj.weight'])
    assert n == L
    for i in range(L):
        w = model.layers[i].o_proj.weight.detach().float()
        assert float((axes[i] @ w).norm()) < 0.01 * before[i]
    ir.revert()
    for i in range(L):
        w = model.layers[i].o_proj.weight.detach().float()
        got = float((axes[i] @ w).norm())
        assert abs(got - before[i]) < 1e-3 * max(before[i], 1.0)


def test_build_merged_rewrites_targets_only(tmp_path):
    from gguf import GGUFWriter
    g = np.random.default_rng(3)
    tensors = {'blk.0.attn_output.weight': g.standard_normal((D, D)).astype(np.float32),
               'blk.0.ffn_down.weight': g.standard_normal((D, D)).astype(np.float32),
               'blk.1.attn_output.weight': g.standard_normal((D, D)).astype(np.float32),
               'token_embd.weight': g.standard_normal((32, D)).astype(np.float32)}
    src = tmp_path / 'in.gguf'
    w = GGUFWriter(str(src), 'ggmf')
    for name, arr in tensors.items():
        w.add_tensor(name, arr)
    w.open_output_file(); w.write_header_to_file()
    w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    axes = torch.nn.functional.normalize(
        torch.from_numpy(g.standard_normal((L, D)).astype(np.float32)), dim=1)
    keys = ['model.layers.0.self_attn.o_proj.weight',
            'model.layers.0.mlp.down_proj.weight',
            'model.layers.1.self_attn.o_proj.weight']
    name_map = {keys[0]: 'blk.0.attn_output.weight',
                keys[1]: 'blk.0.ffn_down.weight',
                keys[2]: 'blk.1.attn_output.weight'}
    out = tmp_path / 'out.gguf'
    rep = spanfit.build_merged(str(src), str(out), axes, keys, name_map=name_map)
    assert rep['tensors_edited'] == 3
    for hf, base in name_map.items():
        L0 = int([p for p in hf.split('.') if p.isdigit()][0])
        a = axes[L0].numpy()
        want = tensors[base] - np.outer(a, a @ tensors[base])
        got = dequantize_tensor(str(out), base)
        assert np.allclose(got, want, atol=1e-5), base
    assert np.allclose(dequantize_tensor(str(out), 'token_embd.weight'),
                       tensors['token_embd.weight'], atol=1e-6)
