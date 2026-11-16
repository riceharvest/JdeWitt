"""Explicit fixtures for consistency gates; not real refusal-quality evidence."""
import pytest
import torch
from b70_ablator import numerics, data, scoring


def policy():
    return {'min_direction_examples': 2, 'min_direction_families': 2,
            'min_direction_overlap': .9, 'min_refusal_families': 2,
            'min_variants_per_family': 2, 'confidence': .9, 'max_task_regressions': 0}


def test_policy_rejects_missing_guessed_and_nonfinite_thresholds():
    assert scoring.ConsistencyPolicy.from_json(policy()).confidence == .9
    for change in ({'confidence': float('nan')}, {'min_direction_examples': 1},
                   {'max_task_regressions': True}, {'min_direction_overlap': -1}):
        with pytest.raises(ValueError):
            scoring.ConsistencyPolicy.from_json({**policy(), **change})
    with pytest.raises(ValueError):
        scoring.ConsistencyPolicy.from_json({})


def direction_records(unstable=False):
    records = []
    for language in ('en', 'nl'):
        for family in ('a', 'b'):
            for label in ('good', 'bad'):
                vector = torch.zeros(2, 3)
                if label == 'bad':
                    vector[:, 1 if unstable and family == 'b' else 0] = 1
                records.append(({'id': language+family+label, 'language': language,
                    'label': label, 'family_id': family}, vector))
    return records


def test_direction_stability_rejects_adequate_counts_but_inconsistent_subspaces():
    with pytest.raises(ValueError, match='stability'):
        numerics.stable_directions(direction_records(True), 1, scoring.ConsistencyPolicy.from_json(policy()), 7)


def test_direction_stability_reports_counts_and_sign_invariant_overlap():
    directions, report = numerics.stable_directions(direction_records(), 1, scoring.ConsistencyPolicy.from_json(policy()), 7)
    assert directions.shape == (2, 1, 3)
    assert report['minimum_overlap'] == pytest.approx(1.)
    assert report['accepted_counts']['en'] == {'good': 2, 'bad': 2}
    assert report['family_counts'] == {'en': 2, 'nl': 2}
    with pytest.raises(ValueError, match='coverage'):
        numerics.stable_directions(direction_records()[:2], 1, scoring.ConsistencyPolicy.from_json(policy()), 7)


def test_paraphrase_family_overlap_is_rejected_between_splits():
    a = [{'id': 'a', 'prompt': 'Original phrasing', 'family_id': 'same'}]
    b = [{'id': 'b', 'prompt': 'Different wording of that task', 'family_id': 'same'}]
    with pytest.raises(ValueError, match='family'):
        data.validate_splits({'selection': a, 'holdout': b})


def rows(families=4, improved=True):
    tasks, baseline, candidate = [], [], []
    for family in range(families):
        for variant in range(2):
            task = {'id': f'r{family}-{variant}', 'kind': 'refusal', 'language': 'en',
                    'prompt': f'fixture family {family} variant {variant}', 'family_id': f'family-{family}'}
            tasks.append(task)
            baseline.append({**task, 'status': 'ok', 'judgment': {'verdict': 'refusal'}})
            candidate.append({**task, 'status': 'ok', 'judgment': {'verdict': 'compliance' if improved else 'refusal'}})
    return tasks, baseline, candidate


def test_paired_family_evidence_does_not_count_paraphrases_as_independent_wins():
    tasks, baseline, candidate = rows(2)
    report = scoring.paired_consistency(tasks, baseline, candidate, scoring.ConsistencyPolicy.from_json(policy()))
    assert report['family_wins'] == 2 and report['confidence'] == pytest.approx(.75)
    assert not report['passes']
    tasks, baseline, candidate = rows(4)
    report = scoring.paired_consistency(tasks, baseline, candidate, scoring.ConsistencyPolicy.from_json(policy()))
    assert report['confidence'] == pytest.approx(.9375) and report['passes']


def test_task_regression_is_visible_even_when_aggregate_capability_is_unchanged():
    tasks, baseline, candidate = rows()
    for i, passed in enumerate((True, False)):
        task = {'id': f'c{i}', 'kind': 'code', 'language': 'en', 'prompt': f'fixture code {i}'}
        tasks.append(task)
        baseline.append({**task, 'status': 'ok', 'passed': passed})
        candidate.append({**task, 'status': 'ok', 'passed': not passed})
    report = scoring.paired_consistency(tasks, baseline, candidate, scoring.ConsistencyPolicy.from_json(policy()))
    assert report['capability_regressions'] == ['c0'] and not report['passes']


def test_paired_evidence_rejects_missing_duplicate_or_mismatched_predictions():
    tasks, baseline, candidate = rows()
    for changed in (candidate[:-1], candidate+[candidate[0]], [{**candidate[0], 'prompt': 'wrong'}]+candidate[1:]):
        with pytest.raises(ValueError):
            scoring.paired_consistency(tasks, baseline, changed, scoring.ConsistencyPolicy.from_json(policy()))


def test_coverage_requires_variants_in_each_language_not_translations_as_variants():
    tasks, _, _ = rows(2)
    scoring.validate_consistency_suite(tasks, scoring.ConsistencyPolicy.from_json(policy()))
    changed = [{**t, 'language': 'nl' if i % 2 else 'en'} for i, t in enumerate(tasks)]
    with pytest.raises(ValueError, match='variants'):
        scoring.validate_consistency_suite(changed, scoring.ConsistencyPolicy.from_json(policy()))
