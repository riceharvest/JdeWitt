"""The P5 train driver's helpers: the deterministic split and the record grouping."""
import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'p5_sae_train.py'


def load():
    spec = importlib.util.spec_from_file_location('p5_sae_train', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_is_deterministic_disjoint_and_holds_out_every_fourth_row():
    module = load()
    rows = [{'id': f'row-{index:03d}', 'vector': torch.zeros(2)} for index in range(10)]
    train, heldout = module.split_rows(rows, holdout_every=4)
    assert [row['id'] for row in heldout] == ['row-003', 'row-007']
    assert len(train) == 8
    assert not {row['id'] for row in train} & {row['id'] for row in heldout}
    # Order of the input does not change the split: it sorts by id.
    shuffled = list(reversed(rows))
    assert [row['id'] for row in module.split_rows(shuffled)[1]] == ['row-003', 'row-007']
    assert [row['id'] for row in module.split_rows(rows)[0]] == [row['id'] for row in train]


def test_split_refuses_a_set_too_small_to_split():
    module = load()
    with pytest.raises(SystemExit, match='cannot be split'):
        module.split_rows([{'id': 'a'}, {'id': 'b'}], holdout_every=4)


def test_group_puts_bad_rows_in_their_family_and_good_rows_on_the_compliant_side():
    module = load()
    records = [
        ({'id': 'a', 'label': 'bad', 'family_id': 'fam-1', 'prompt': 'p'}, torch.zeros(3, 4)),
        ({'id': 'b', 'label': 'good', 'family_id': None, 'prompt': 'q'}, torch.ones(3, 4)),
        ({'id': 'c', 'label': 'ambiguous', 'family_id': 'fam-1', 'prompt': 'r'}, torch.zeros(3, 4)),
    ]
    grouped = module.group(records, layer=1)
    assert list(grouped['refusal']) == ['fam-1']
    assert [row['id'] for row in grouped['refusal']['fam-1']] == ['a']
    assert [row['id'] for row in grouped['compliant']] == ['b']
    # The layer slice is the one asked for.
    assert torch.equal(grouped['compliant'][0]['vector'], torch.ones(4))


def test_group_refuses_a_record_that_is_not_two_dimensional():
    module = load()
    with pytest.raises(SystemExit, match='expected \\[layers, hidden\\]'):
        module.group([({'id': 'a', 'label': 'bad', 'family_id': 'f'}, torch.zeros(4))], layer=0)


def test_markdown_renders_every_family_row_and_feature_row():
    module = load()
    report = {'plan': {'layer': 10, 'features': 64, 'k': 2, 'seeds': [1, 2, 3]},
              'families': [{'family': 'fam-1', 'status': 'trained',
                            'heldout': {'fvu': 0.5, 'pca_control_fvu': 0.6},
                            'rows_train': 3, 'rows_heldout': 1, 'gpu': {'gpu_seconds': 1.2},
                            'features': [{'feature': 7, 'mean_refusal': 1.0, 'mean_compliant': 0.0,
                                          'contrast': 1.0, 'active_refusal': 1.0,
                                          'active_compliant': 0.0, 'cos_to_axis': 0.5}]}]}
    text = module._markdown(report, top_features=1)
    assert '| fam-1 | trained | 0.5000 | 0.6000 | 1 |' in text
    assert '| 7 | 1.000 | 0.000 | +1.000 | 1.00/0.00 | +0.500 |' in text
