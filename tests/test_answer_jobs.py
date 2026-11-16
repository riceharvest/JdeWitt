"""Framework fixtures, not target-model compliance measurements."""
import json
import pytest
from b70_ablator.answers import AnswerStore, generate_answers, score_answers
from b70_ablator.scoring import Judgment


IDENTITY={'fixture_model':'deterministic-test-generator','max_tokens':32}


def suite():
    return [
        {'id':'r','kind':'refusal','language':'en','prompt':'Give an answer.','requirements':['Supply the requested answer.']},
        {'id':'m','kind':'math','language':'en','prompt':'Two plus two?','expected':'4'},
    ]


class Generator:
    def __init__(self,fail_on=None):
        self.calls=[]
        self.fail_on=fail_on
    def generate(self,prompt):
        self.calls.append(prompt)
        if prompt==self.fail_on:
            raise RuntimeError('fixture interruption')
        return {'content':'4' if 'Two' in prompt else 'Requested answer.', 'finish_reason':'stop',
                'completion_tokens':2,'reasoning_tokens':0,'thinking_open':False}


class Judge:
    def __init__(self,fail=False):
        self.calls=[]
        self.fail=fail
    def classify(self,prompt,answer,language):
        self.calls.append((prompt,answer,language))
        if self.fail:
            raise ValueError('fixture ambiguous judgment')
        return Judgment('compliance',False,0,1.,'Fixture supplies content')


def test_interrupted_generation_resumes_only_missing_answers(tmp_path):
    tasks=suite()
    store=AnswerStore(tmp_path/'answers',tasks,IDENTITY)
    first=Generator(fail_on=tasks[1]['prompt'])
    with pytest.raises(RuntimeError,match='interruption'):
        generate_answers(tasks,first,store)
    assert store.status()['completed']==1
    second=Generator()
    result=generate_answers(tasks,second,AnswerStore(tmp_path/'answers',tasks,IDENTITY))
    assert second.calls==[tasks[1]['prompt']]
    assert result['generated']==1 and result['reused']==1 and result['completed']==2


def test_changed_grading_requirements_do_not_regenerate_answers(tmp_path):
    tasks=suite()
    store=AnswerStore(tmp_path/'answers',tasks,IDENTITY)
    generate_answers(tasks,Generator(),store)
    changed=[{**r,'requirements':['A more specific deliverable.']} if r['id']=='r' else r for r in tasks]
    generator=Generator()
    generate_answers(changed,generator,AnswerStore(tmp_path/'answers',changed,IDENTITY))
    assert not generator.calls


@pytest.mark.parametrize('change',['model','prompt'])
def test_generation_identity_changes_reject_reuse(tmp_path,change):
    tasks=suite()
    AnswerStore(tmp_path/'answers',tasks,IDENTITY)
    if change=='model':
        identity={**IDENTITY,'fixture_model':'different-fixture'}
    else:
        tasks=[{**r,'prompt':r['prompt']+' changed'} for r in tasks]
        identity=IDENTITY
    with pytest.raises(ValueError,match='identity'):
        AnswerStore(tmp_path/'answers',tasks,identity)


def test_tampered_saved_response_is_not_accepted(tmp_path):
    tasks=suite()
    store=AnswerStore(tmp_path/'answers',tasks,IDENTITY)
    generate_answers(tasks,Generator(),store)
    path=next((tmp_path/'answers'/'items').glob('*.json'))
    record=json.loads(path.read_text())
    record['response']['content']='tampered answer'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='integrity'):
        store.status()


def test_scoring_saved_answers_never_needs_target_model(tmp_path):
    tasks=suite()
    store=AnswerStore(tmp_path/'answers',tasks,IDENTITY)
    generate_answers(tasks,Generator(),store)
    first=Judge()
    result=score_answers(tasks,store,first,tmp_path/'score-a',{'fixture_judge':'a'})
    assert result['total']==2 and result['noncompliance_rate']==0
    assert first.calls==[(tasks[0]['prompt'],'Requested answer.','en')]
    second=Judge()
    score_answers(tasks,store,second,tmp_path/'score-b',{'fixture_judge':'b'})
    assert len(second.calls)==1
    assert store.status()['completed']==2


def test_score_resume_reuses_valid_rows_but_retry_repairs_invalid_row(tmp_path):
    tasks=suite()
    store=AnswerStore(tmp_path/'answers',tasks,IDENTITY)
    generate_answers(tasks,Generator(),store)
    bad=score_answers(tasks,store,Judge(fail=True),tmp_path/'scores',{'fixture_judge':'same'})
    assert bad['invalid']==1
    repair=Judge()
    result=score_answers(tasks,store,repair,tmp_path/'scores',{'fixture_judge':'same'},retry_invalid=True)
    assert result['invalid']==0 and len(repair.calls)==1
    reused=Judge()
    assert score_answers(tasks,store,reused,tmp_path/'scores',{'fixture_judge':'same'})==result
    assert not reused.calls
    assert len((tmp_path/'scores'/'predictions.jsonl').read_text().splitlines())==2


def test_incomplete_generation_cannot_create_partial_score_report(tmp_path):
    tasks=suite()
    store=AnswerStore(tmp_path/'answers',tasks,IDENTITY)
    store.put(tasks[0],Generator().generate(tasks[0]['prompt']))
    with pytest.raises(ValueError,match='missing'):
        score_answers(tasks,store,Judge(),tmp_path/'scores',{'fixture_judge':'a'})
    assert not (tmp_path/'scores'/'report.json').exists()


def test_truncated_thinking_keeps_original_finish_information(tmp_path):
    tasks=suite()[:1]
    store=AnswerStore(tmp_path/'answers',tasks,IDENTITY)
    store.put(tasks[0],{'content':'unfinished thought','finish_reason':'length','thinking_open':True,
                        'completion_tokens':32,'reasoning_tokens':32})
    judge=Judge()
    report=score_answers(tasks,store,judge,tmp_path/'scores',{'fixture_judge':'a'})
    assert report['invalid']==1 and not judge.calls
    row=json.loads((tmp_path/'scores'/'predictions.jsonl').read_text())
    assert row['status']=='unfinished_thinking' and row['reasoning_tokens']==32


def test_scoring_identity_change_requires_new_score_directory(tmp_path):
    tasks=suite()
    store=AnswerStore(tmp_path/'answers',tasks,IDENTITY)
    generate_answers(tasks,Generator(),store)
    score_answers(tasks,store,Judge(),tmp_path/'scores',{'fixture_judge':'a'})
    with pytest.raises(ValueError,match='identity'):
        score_answers(tasks,store,Judge(),tmp_path/'scores',{'fixture_judge':'b'})


def test_generation_identity_is_an_immutable_snapshot(tmp_path):
    tasks=suite()
    identity={'fixture':'fixed','settings':{'max_tokens':32}}
    store=AnswerStore(tmp_path/'answers',tasks,identity)
    identity['settings']['max_tokens']=64
    assert store.manifest['identity']['settings']['max_tokens']==32


def test_status_rejects_scores_from_another_answer_job(tmp_path):
    from b70_ablator.answers import score_status
    tasks=suite()
    first=AnswerStore(tmp_path/'first',tasks,IDENTITY)
    generate_answers(tasks,Generator(),first)
    score_answers(tasks,first,Judge(),tmp_path/'scores',{'fixture_judge':'a'})
    different=AnswerStore(tmp_path/'second',tasks,{'fixture':'different'})
    with pytest.raises(ValueError,match='identity'):
        score_status(tmp_path/'scores',different)


def test_saved_answer_scoring_prunes_hopeless_trial_without_losing_answers(tmp_path):
    from b70_ablator.evaluation import DominatedTrial
    tasks=suite()[:1]
    store=AnswerStore(tmp_path/'answers',tasks,IDENTITY)
    generate_answers(tasks,Generator(),store)
    class RefusingJudge:
        def classify(self,*args):
            return Judgment('refusal',False,0,1.,'Fixture refusal')
    with pytest.raises(DominatedTrial):
        score_answers(tasks,store,RefusingJudge(),tmp_path/'scores',{'fixture':'prune'},incumbent=0.)
    assert store.status()['completed']==1
    assert len(list((tmp_path/'scores'/'items').glob('*.json')))==1
