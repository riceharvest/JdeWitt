from jdewitt.transport import JUDGE_SCHEMA


def test_judge_has_to_ground_reason_before_emitting_a_verdict():
    assert list(JUDGE_SCHEMA['properties'])[0]=='reason'
    assert JUDGE_SCHEMA['required'][0]=='reason'
