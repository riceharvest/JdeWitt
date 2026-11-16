"""Merged cache checks exercise tensor arithmetic, never a model forward."""
import pytest
import torch
from torch import nn
from b70_ablator.editor import Editor, AblatedLinear
from test_runtime_regressions import tiny_modules


@torch.no_grad()
def test_cache_avoids_rebuilding_identical_rounded_weights_and_respects_budget(monkeypatch):
    editor=Editor(tiny_modules(),1,merged_cache_bytes=64)
    adapter=next(a for *_,a in editor.targets if a.base.weight.shape==(4,4))
    basis=torch.tensor([[[1.,0.,0.,0.]]])
    params={g:{'strength':.7,'center':.5,'width':1.,'floor':1.} for g in editor.groups}
    editor.apply(basis,params)
    calls=[]
    original=adapter.merge_source
    def merge(weight):
        calls.append(1)
        return original(weight)
    monkeypatch.setattr(adapter,'merge_source',merge)
    first=adapter.effective_weight()
    assert torch.equal(first,adapter.effective_weight()) and len(calls)==1
    assert first.data_ptr()==adapter.effective_weight().data_ptr()
    assert editor.merge_cache.used_bytes==64
    # An oversized projection never becomes a persistent allocation.
    large=next(a for *_,a in editor.targets if a.base.weight.numel()>16)
    large.effective_weight()
    assert editor.merge_cache.used_bytes<=64
    with torch.no_grad():
        adapter.B.add_(.1)
    changed=adapter.effective_weight()
    assert not torch.equal(first,changed) and len(calls)==2
    editor.reset()
    assert editor.merge_cache.used_bytes==0
    assert not any(a.enabled for *_,a in editor.targets)


@torch.no_grad()
def test_partial_cache_does_not_thrash_between_layers():
    editor=Editor(tiny_modules(),1,edit_targets='full',merged_cache_bytes=64)
    d=torch.tensor([[[1.,0,0,0]]])
    params={g:{'strength':.5,'center':.5,'width':1.,'floor':1.} for g in editor.groups}
    editor.apply(d,params)
    first=editor.targets[0][-1]
    pinned=first.effective_weight()
    for *_,adapter in editor.targets:
        before=adapter.base.weight.detach().clone()
        actual=adapter.effective_weight()
        expected=adapter.merge_source(adapter.base.weight)
        assert torch.equal(actual,expected)
        assert torch.equal(before,adapter.base.weight)
        assert editor.merge_cache.used_bytes<=64
    assert first.effective_weight().data_ptr()==pinned.data_ptr()
    editor.apply(d,{g:{**p,'strength':0.} for g,p in params.items()})
    assert editor.merge_cache.used_bytes==0


def test_cache_disabled_by_default_and_invalid_budgets_rejected():
    assert Editor(tiny_modules(),1).merge_cache.budget_bytes==0
    for bad in (-1,True,1.5):
        with pytest.raises(ValueError,match='cache'):
            Editor(tiny_modules(),1,merged_cache_bytes=bad)


def test_learned_factor_replacement_clears_previous_normalization():
    adapter=AblatedLinear(nn.Linear(4,4,bias=False),1)
    d=torch.tensor([[1.,0,0,0]])
    adapter.set_directions(d,.5,row_normalization='full')
    adapter.set_factors(torch.zeros_like(adapter.A),torch.zeros_like(adapter.B),1.)
    assert torch.equal(adapter.effective_weight(),adapter.base.weight)


def test_cache_does_not_detach_a_training_graph():
    editor=Editor(tiny_modules(),1,merged_cache_bytes=4096)
    adapter=editor.targets[0][-1]
    adapter.enabled=True
    adapter.A.requires_grad_(True)
    adapter.B.requires_grad_(True)
    adapter.effective_weight().sum().backward()
    assert adapter.A.grad is not None and adapter.B.grad is not None
    assert editor.merge_cache.used_bytes==0
