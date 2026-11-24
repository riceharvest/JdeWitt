"""Suite and shape-audit binding: same evidence under another worktree root.

Cards run in per-task git worktrees, so a wave measured in one tree records the
suite by an absolute path under that tree. The certifier must bind that path to
the spec's file by its bytes, not by its root -- and must still refuse when the
bytes differ, otherwise a swapped suite would certify.
"""
import pytest
from jdewitt.certify import _check_shape, _check_suite, _same_evidence_file
from jdewitt.errors import CertifyRefused

ROWS = '\n'.join(['{"id": "r1", "prompt": "x", "family_id": "f1"}',
                  '{"id": "r2", "prompt": "y", "family_id": "f2"}']) + '\n'


def tree(root, payload=ROWS):
    other = root / 'other-worktree'
    other.mkdir()
    path = other / 'suite.jsonl'
    path.write_text(payload)
    return path


def test_same_content_accepts_a_relocated_suite(tmp_path):
    recorded = tree(tmp_path)
    here = tmp_path / 'suite.jsonl'
    here.write_text(ROWS)
    assert _same_evidence_file(recorded, here)
    _check_suite('removal', {'path': str(here)},
                 {'suite': str(recorded), 'rows': 40}, [{'kind': 'refusal'}] * 40)


def test_different_content_still_refuses(tmp_path):
    recorded = tree(tmp_path, ROWS.replace('r1', 'swapped'))
    here = tmp_path / 'suite.jsonl'
    here.write_text(ROWS)
    assert not _same_evidence_file(recorded, here)
    with pytest.raises(CertifyRefused) as caught:
        _check_suite('removal', {'path': str(here)},
                     {'suite': str(recorded), 'rows': 40}, [{'kind': 'refusal'}] * 40)
    assert caught.value.code == 'CERTIFY_CONFIG_MISMATCH'


def test_missing_declared_suite_refuses(tmp_path):
    here = tmp_path / 'suite.jsonl'
    here.write_text(ROWS)
    with pytest.raises(CertifyRefused) as caught:
        _check_suite('removal', {'path': str(here)},
                     {'suite': str(tmp_path / 'gone.jsonl'), 'rows': 40},
                     [{'kind': 'refusal'}] * 40)
    assert caught.value.code == 'CERTIFY_CONFIG_MISMATCH'


def shape_fixture(tmp_path, payload=ROWS):
    directory = tmp_path / 'waves' / 'removal-arm'
    directory.mkdir(parents=True)
    wave_predictions = directory / 'predictions.jsonl'
    wave_predictions.write_text(payload)
    elsewhere = tree(tmp_path)
    audit = tmp_path / 'shape-audit-removal.json'
    audit.write_text('{"audits": [{"label": "removal", "predictions": "%s", "rows": 2, '
                     '"shapes": {"declines_flatly": 2}}]}' % elsewhere)
    waves = {'removal': {'predictions': wave_predictions, 'rows': [{'id': 'r1'}, {'id': 'r2'}]}}
    return audit, waves


def test_shape_audit_binds_to_the_wave_by_bytes(tmp_path):
    audit, waves = shape_fixture(tmp_path)
    block = _check_shape([{'path': str(audit), 'axis': 'removal',
                           'artifact_identity_digest': 'a' * 64}], 'a' * 64, waves, {})
    assert block['verdict_counts'] == {'declines_flatly': 2}
    assert block['artifact_identity_digest'] == 'a' * 64


def test_shape_audit_of_another_wave_refuses(tmp_path):
    audit, waves = shape_fixture(tmp_path)
    waves['removal']['predictions'].write_text(ROWS.replace('r2', 'other'))
    with pytest.raises(CertifyRefused) as caught:
        _check_shape([{'path': str(audit), 'axis': 'removal',
                       'artifact_identity_digest': 'a' * 64}], 'a' * 64, waves, {})
    assert caught.value.code == 'CERTIFY_ARTIFACT_MISMATCH'


def test_shape_audit_without_a_declared_digest_refuses(tmp_path):
    audit, waves = shape_fixture(tmp_path)
    with pytest.raises(CertifyRefused) as caught:
        _check_shape([{'path': str(audit), 'axis': 'removal'},
                      {'path': str(audit), 'axis': 'held_out', 'artifact_identity_digest': ''}],
                     'a' * 64, waves, {})
    assert caught.value.code == 'CERTIFY_MISSING_AXIS'
