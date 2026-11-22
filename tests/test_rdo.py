"""Tests for jdewitt.rdo — gradient-refined direction against a linear refusal probe."""
import json
import math

import pytest
import torch

from jdewitt import rdo

L, W = 3, 24


def _toy_records(*, families=None, offset=1.0, counts=(20, 20), seed=3):
    """Planted r (dim 0) plus a per-family nuisance offset along dim 1.

    ``offset`` may be a scalar or one value per family; ``counts`` is the
    (bad, good) row count per family, so a caller can make the label composition
    follow the family offset (the topic confound) rather than refusal.
    """
    gen = torch.Generator().manual_seed(seed)
    families = families or ['a', 'b']
    offsets = offset if isinstance(offset, (list, tuple)) else [offset] * len(families)
    per_family = counts if isinstance(counts[0], (list, tuple)) else [counts] * len(families)
    bad, good = [], []
    for family, family_offset, (n_bad, n_good) in zip(families, offsets, per_family):
        for label, rows, sign, count in (('bad', bad, 1.0, n_bad), ('good', good, -1.0, n_good)):
            for _ in range(count):
                layer_rows = []
                for layer in range(L):
                    row = 0.05 * torch.randn(W, generator=gen)
                    row[0] += 0.6 * sign * (1.0 - 0.1 * layer)
                    row[1] += family_offset
                    row[2:12] += 1.2 * torch.randn(10, generator=gen)
                    layer_rows.append(row)
                rows.append(({'label': label, 'family_id': family}, torch.stack(layer_rows)))
    return bad + good


def test_axes_shape_normalization_and_digest():
    axes, report = rdo.fit_records(_toy_records(), holdout='none', steps=50)
    assert report['axes'] == [L, W]
    assert len(axes) == L and all(len(row) == W for row in axes)
    for row in axes:
        assert math.isclose(math.sqrt(sum(value * value for value in row)), 1.0, rel_tol=1e-9)
    assert rdo.axes_digest(axes) == report['axes_sha256']
    assert report['centering'] == 'family'


def test_fit_is_deterministic():
    records = _toy_records()
    first, report_one = rdo.fit_records(records, steps=40)
    second, report_two = rdo.fit_records(records, steps=40)
    assert report_one['axes_sha256'] == report_two['axes_sha256']
    assert first == second


def test_centering_beats_raw_rows_on_a_shared_topic_offset():
    """Topic offset follows the family, and the label mix follows the family too.

    Families with a positive offset carry most of the bad rows, so the raw bad
    mean sits at +offset and the raw good mean at -offset: the contrast mean
    becomes the topic offset. Centering each family at its own mean removes it,
    because the label change inside a family is the only signal left.
    """
    families = ['fam0', 'fam1', 'fam2', 'fam3']
    records = _toy_records(families=families, offset=[3.0, -3.0, 3.0, -3.0],
                           counts=[(30, 10), (10, 30), (30, 10), (10, 30)])
    planted = torch.zeros(W)
    planted[0] = 1.0
    centred, report = rdo.fit_records(records, steps=80, protect_k=0)
    raw = [({'label': meta['label']}, rec) for meta, rec in records]
    uncentred, report_raw = rdo.fit_records(raw, holdout='none', steps=80, protect_k=0)
    assert report['centering'] == 'family' and report_raw['centering'] == 'none'
    cos_centred = min(abs(float(torch.dot(torch.tensor(row), planted))) for row in centred)
    cos_raw = min(abs(float(torch.dot(torch.tensor(row), planted))) for row in uncentred)
    assert cos_centred > cos_raw
    assert cos_centred > 0.9


def test_holdout_families_are_not_fitted_and_are_scored():
    records = _toy_records(families=['a', 'b', 'c', 'd'])
    _axes, report = rdo.fit_records(records, holdout='families', steps=40)
    assert report['holdout_families'] == ['d']
    assert 'd' not in report['fit_families']
    assert report['layers'][0]['n_holdout_bad'] == 20
    assert report['layers'][0]['auc_holdout_rdo'] is not None


def test_holdout_families_needs_family_ids():
    records = [({'label': meta['label']}, rec) for meta, rec in _toy_records()]
    with pytest.raises(SystemExit):
        rdo.fit_records(records, holdout='families')


def test_protection_reports_the_kept_fraction():
    records = _toy_records()
    _axes, unprotected = rdo.fit_records(records, steps=40, protect_k=0)
    _axes, protected = rdo.fit_records(records, steps=40, protect_k=8)
    assert unprotected['layers'][0]['kept_fraction'] == 1.0
    assert 0.0 < protected['layers'][0]['kept_fraction'] <= 1.0


def test_auc_is_one_for_a_separating_score_and_ties_are_half():
    assert rdo._auc([2.0, 3.0], [0.0, 1.0]) == 1.0
    assert rdo._auc([1.0, 1.0], [1.0, 1.0]) == 0.5


def test_cli_fit_writes_axes_and_report(tmp_path):
    records = _toy_records(families=['a', 'b', 'c', 'd'])
    source = tmp_path / 'records.pt'
    torch.save({'records': records}, source)
    axes_path = tmp_path / 'axes.pt'
    report_path = tmp_path / 'rdo.json'
    assert rdo.main(['fit', '--records', str(source), '--out', str(axes_path),
                     '--report', str(report_path), '--steps', '40']) == 0
    payload = torch.load(axes_path, map_location='cpu', weights_only=False)
    report = json.loads(report_path.read_text())
    assert tuple(payload['axes'].shape) == (L, W)
    assert payload['axes'].dtype == torch.float32
    assert payload['meta']['axes_sha256'] == report['axes_sha256']
    assert payload['meta']['centering'] == 'family'


def test_self_check_passes():
    assert rdo.self_check(steps=120, probe_steps=200) == 0
