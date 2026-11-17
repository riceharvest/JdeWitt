"""Cache failure controls on CPU fixtures; never starts an inference server."""
import json
from types import SimpleNamespace
import pytest
from jdewitt.pipeline import eval_cached
from jdewitt.data import digest


TASKS = [{'id':'m','kind':'math','language':'en','prompt':'Fixture: two plus two?','expected':'4'}]
GENERATION = {'fixture_model':'fixed','max_tokens':32}
SCORING = {'fixture_scorer':'fixed','version':1}


class Generator:
    def generate(self, prompt):
        return {'content':'4','finish_reason':'stop'}


def completed(directory):
    return eval_cached(TASKS, Generator(), None, directory, 'fixture-sandbox', GENERATION,
                       SimpleNamespace(idle=lambda:None), SCORING)


@pytest.mark.parametrize('change', ['scorer','sandbox'])
def test_completed_cache_rejects_changed_scoring_identity(tmp_path, change):
    completed(tmp_path)
    identity = {**SCORING, 'version':2} if change=='scorer' else SCORING
    sandbox = 'changed-fixture-sandbox' if change=='sandbox' else 'fixture-sandbox'
    with pytest.raises(ValueError, match='identity'):
        eval_cached(TASKS, None, None, tmp_path, sandbox, GENERATION,
                    SimpleNamespace(idle=lambda:pytest.fail('No model lifetime should start')), identity)


@pytest.mark.parametrize('change', ['answer','score','missing-answer','missing-score','predictions'])
def test_completed_cache_verifies_saved_records_not_just_summary(tmp_path, change):
    completed(tmp_path)
    answer = tmp_path/'answers'/'items'/(digest('m')+'.json')
    score = tmp_path/'items'/(digest('m')+'.json')
    if change=='answer':
        record=json.loads(answer.read_text())
        record['response']['content']='40'
        answer.write_text(json.dumps(record))
    elif change=='score':
        record=json.loads(score.read_text())
        record['row']['raw']='fabricated answer with the same pass count'
        score.write_text(json.dumps(record))
    elif change=='missing-answer':
        answer.unlink()
    elif change=='missing-score':
        score.unlink()
    else:
        path=tmp_path/'predictions.jsonl'
        row=json.loads(path.read_text())
        row['raw']='altered raw text, unchanged aggregate score'
        path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError):
        eval_cached(TASKS, None, None, tmp_path, 'fixture-sandbox', GENERATION,
                    SimpleNamespace(idle=lambda:pytest.fail('Cache validation cannot load a model')), SCORING)


def test_identical_completed_cache_needs_no_generator_or_phase_switch(tmp_path):
    expected=completed(tmp_path)
    phases=SimpleNamespace(idle=lambda:pytest.fail('Completed cache must remain read-only'))
    assert eval_cached(TASKS,None,None,tmp_path,'fixture-sandbox',GENERATION,phases,SCORING)==expected


def test_scored_row_is_bound_to_exact_saved_response(tmp_path):
    completed(tmp_path)
    path=tmp_path/'items'/(digest('m')+'.json')
    record=json.loads(path.read_text())
    assert record['answer_digest'] == digest(json.loads((tmp_path/'answers'/'items'/(digest('m')+'.json')).read_text())['response'])
