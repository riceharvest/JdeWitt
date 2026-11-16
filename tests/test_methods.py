"""Method mathematics and reachable edit operators; not uncensoring benchmarks."""
import pytest
import torch
from torch import nn
from b70_ablator.methods import fit_directions, method_catalog, fit_ara
from b70_ablator.editor import AblatedLinear


def records():
    rng = torch.Generator().manual_seed(7)
    rows = []
    for language in ('en','nl'):
        for f in range(6):
            baseline = torch.randn(2, 8, generator=rng) * .1
            for label in ('good','bad'):
                r = baseline.clone()
                if label == 'bad':
                    r[:, 0] += 2
                    r[:, 1] += f * .15
                    if language == 'nl':
                        r[:, 2] += .2
                rows.append(({'language':language,'family_id':str(f),'label':label},r))
    return rows


@pytest.mark.parametrize('method', ['directional','projected','svd','whitened-svd','som','mpoa'])
def test_direction_methods_are_finite_orthonormal_and_repeatable(method):
    options = {'ridge': .1, 'som_steps': 12, 'som_learning_rate': .2}
    a = fit_directions(records(), 2, method, options, 7)
    b = fit_directions(records(), 2, method, options, 7)
    assert a.shape == (2, 2, 8) and torch.equal(a, b)
    assert torch.allclose(a @ a.transpose(-1,-2), torch.eye(2).expand(2,-1,-1),atol=1e-5)


def test_mpoa_preserves_row_norms_and_merged_export_operator():
    torch.manual_seed(4)
    layer = AblatedLinear(nn.Linear(8,8,bias=False),2)
    before = layer.base.weight.detach().clone()
    directions = torch.eye(8)[:2]
    layer.set_directions(directions, .65, row_normalization='full')
    effective = layer.effective_weight()
    assert torch.allclose(effective.norm(dim=-1), before.norm(dim=-1),atol=1e-6)
    assert torch.equal(layer.merge_source(before), effective)
    assert torch.equal(before, layer.base.weight)


def test_robust_options_are_explicit_and_suppress_massive_activation_dimension():
    data = records()
    for _, x in data:
        x[:, 7] = 1000
    directions = fit_directions(data,1,'svd',{'massive_threshold':100.,'massive_ratio':100.},7)
    assert directions[:,:,7].abs().max() < 1e-5
    with pytest.raises(ValueError):
        fit_directions(data,1,'whitened-svd',{},7)


def test_ara_optimizes_real_factors_is_repeatable_and_handles_neighbor_limits():
    weight = torch.eye(3)
    good_x = torch.tensor([[1.,0.,0.],[1.,.1,0.]])
    bad_x = torch.tensor([[0.,1.,0.],[0.,1.,.1]])
    options = {'steps':3,'learning_rate':.3,'neighbors':2,'preserve':1.,'steer':1.,
               'overcorrect':.1,'repulsion_cap':1.}
    a,b,audit = fit_ara(weight,good_x,good_x,bad_x,bad_x,2,options,7)
    assert b.abs().sum()>0 and audit['final_loss'] < audit['initial_loss']
    aa,bb,_ = fit_ara(weight,good_x,good_x,bad_x,bad_x,2,options,7)
    assert torch.equal(a,aa) and torch.equal(b,bb)
    with pytest.raises(ValueError,match='neighbor'):
        fit_ara(weight,good_x,good_x,bad_x,bad_x,2,{**options,'neighbors':3},7)


def test_catalog_does_not_misrepresent_missing_methods_as_implemented():
    catalog = method_catalog()
    for name, item in catalog['methods'].items():
        assert item['status'] in {'implemented','researched-not-implemented'}
        assert item['sources'] and isinstance(item['full_model_b70_verified'],bool)


def test_som_fits_harmful_prototypes_then_centers_on_harmless_mean():
    rows=[]
    for family,good,bad in [('a',[0.,1.],[2.,1.]),('b',[0.,3.],[2.,3.])]:
        for label,vector in [('good',good),('bad',bad)]:
            rows.append(({'language':'en','family_id':family,'label':label},torch.tensor([vector])))
    d=fit_directions(rows,1,'som',{'som_steps':1,'som_learning_rate':.5},seed=0)[0]
    expected=torch.nn.functional.normalize(torch.tensor([[2.,-1.]]),dim=-1)
    assert torch.allclose(d.T@d,expected.T@expected,atol=1e-6)


def test_ara_never_copies_or_materializes_the_original_weight():
    from types import SimpleNamespace
    # Optimization uses captured original I/O; the resident base is metadata only.
    weight=SimpleNamespace(ndim=2,shape=(3,3),device=torch.device('cpu'))
    good=torch.tensor([[1.,0.,0.],[1.,.1,0.]])
    bad=torch.tensor([[0.,1.,0.],[0.,1.,.1]])
    options={'steps':2,'learning_rate':.3,'neighbors':2,'preserve':1.,'steer':1.,'overcorrect':.1,'repulsion_cap':1.}
    a,b,audit=fit_ara(weight,good,good,bad,bad,1,options,7)
    assert audit['final_loss']<audit['initial_loss']
    assert a.shape==(1,3) and b.shape==(3,1)
