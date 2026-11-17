from jdewitt import timing


def test_record_load_roundtrip(tmp_path):
    p = str(tmp_path / 'led.jsonl')
    timing.record(p, lineage='x', phase='wave', seconds=10.0, kind='gpu')
    timing.record(p, lineage='x', phase='fit', seconds=0.1, kind='cpu')
    ev = timing.load(p)
    assert len(ev) == 2
    assert ev[0]['phase'] == 'wave' and ev[1]['kind'] == 'cpu'


def test_summarize_percents_and_pivot(tmp_path):
    events = [
        {'ts': 0, 'lineage': 'x', 'phase': 'wave', 'seconds': 70.0, 'kind': 'gpu', 'scope': {}},
        {'ts': 0, 'lineage': 'x', 'phase': 'serve_swap', 'seconds': 20.0, 'kind': 'gpu', 'scope': {}},
        {'ts': 0, 'lineage': 'x', 'phase': 'fit', 'seconds': 10.0, 'kind': 'cpu', 'scope': {}},
        {'ts': 0, 'lineage': 'x', 'phase': 'judge', 'seconds': 5.0, 'kind': 'proxy', 'scope': {}},
    ]
    rep = timing.summarize(events)
    gpu = rep['gpu']
    assert gpu['total_s'] == 90.0
    assert gpu['wave'] == round(100 * 70 / 90, 1)
    assert any('screening subset' in h for h in gpu['pivot_hints'])
    assert rep['proxy']['judge_proxy_s'] == 5.0


def test_record_rejects_bad_kind(tmp_path):
    import pytest
    p = str(tmp_path / 'led.jsonl')
    with pytest.raises(ValueError):
        timing.record(p, lineage='x', phase='wave', seconds=1.0, kind='vibes')
