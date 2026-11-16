import pytest
from b70_ablator.pipeline import validate_experiment, cached_report, atomic_json
from b70_ablator.data import digest


def suite(prefix):
    return [
        {'id':prefix+'r','kind':'refusal','language':'en','prompt':prefix+' controversial'},
        {'id':prefix+'h','kind':'harmless','language':'en','prompt':prefix+' ordinary'},
        {'id':prefix+'c','kind':'code','language':'en','prompt':prefix+' code','tests':'assert f(1)==2'},
        {'id':prefix+'m','kind':'math','language':'en','prompt':prefix+' math','expected':'2'},
    ]


def directions():
    return [{'id':lang+label,'kind':'direction','language':lang,'prompt':lang+label,'label':label} for lang in ('en','nl') for label in ('good','bad')]


def test_experiment_needs_multilingual_directions_and_all_capability_slices():
    validate_experiment(directions(),suite('s'),suite('h'))
    with pytest.raises(ValueError):
        validate_experiment(directions()[:2],suite('s'),suite('h'))
    with pytest.raises(ValueError):
        validate_experiment(directions(),suite('s')[:-1],suite('h'))


def test_cached_report_rejects_wrong_suite_and_missing_predictions(tmp_path):
    tasks=suite('s')
    report=tmp_path/'report.json'
    atomic_json(report,{'suite_digest':digest(tasks),'report':{'total':4}})
    with pytest.raises(ValueError):
        cached_report(report,tasks)


def test_cached_report_returns_only_exact_matching_completed_suite(tmp_path):
    import json
    from b70_ablator.scoring import summarize
    tasks=suite('s')
    rows=[]
    for t in tasks:
        r={k:t[k] for k in ('id','kind','language')}
        r['status']='ok'
        if r['kind']=='refusal':
            r['judgment']={'verdict':'compliance','disclaimer':False,'preamble_chars':0}
        elif r['kind'] in ('math','code'):
            r['passed']=True
        rows.append(r)
    report=tmp_path/'report.json'
    (tmp_path/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    atomic_json(report,{'suite_digest':digest(tasks),'report':summarize(rows,[t['id'] for t in tasks])})
    assert cached_report(report,tasks)['total']==4
    with pytest.raises(ValueError):
        cached_report(report,suite('different'))


def test_interrupted_eval_reuses_answer_saved_before_judge_disconnect(tmp_path):
    from b70_ablator.pipeline import eval_cached
    from b70_ablator.scoring import Judgment
    tasks=[{'id':'r','kind':'refusal','language':'en','prompt':'Give an answer.'}]
    calls=[]
    class Generator:
        def generate(self,prompt):
            calls.append(prompt)
            return {'content':'Requested answer.','finish_reason':'stop','thinking_open':False}
    class Disconnected:
        def classify(self,*args):
            raise ConnectionError('fixture judge disconnected')
    class Repaired:
        def classify(self,*args):
            return Judgment('compliance',False,0,1.,'Fixture answer')
    identity={'fixture_generator':'same-fixed-model'}
    with pytest.raises(ConnectionError):
        eval_cached(tasks,Generator(),Disconnected(),tmp_path,'fixture-sandbox',generation_identity=identity,scoring_identity={'fixture':'repaired-same-scorer'})
    result=eval_cached(tasks,Generator(),Repaired(),tmp_path,'fixture-sandbox',generation_identity=identity,scoring_identity={'fixture':'repaired-same-scorer'})
    assert result['invalid']==0 and calls==['Give an answer.']


def test_completed_eval_does_not_reuse_scores_from_different_model(tmp_path):
    from b70_ablator.pipeline import eval_cached
    tasks=[{'id':'m','kind':'math','language':'en','prompt':'Two plus two?','expected':'4'}]
    class Generator:
        def generate(self,prompt):
            return {'content':'4','finish_reason':'stop'}
    eval_cached(tasks,Generator(),None,tmp_path,'fixture-sandbox',{'fixture_model':'first'})
    with pytest.raises(ValueError,match='identity'):
        eval_cached(tasks,Generator(),None,tmp_path,'fixture-sandbox',{'fixture_model':'second'})


def test_phased_eval_generates_whole_batch_before_judging(tmp_path):
    from types import SimpleNamespace
    from b70_ablator.pipeline import eval_cached
    from b70_ablator.scoring import Judgment
    tasks=[{'id':str(i),'kind':'refusal','language':'en','prompt':'fixture '+str(i)} for i in range(2)]
    events=[]
    class Generator:
        def generate(self,prompt):
            events.append('generate-'+prompt)
            return {'content':'fixture answer','finish_reason':'stop'}
    class Judge:
        def classify(self,prompt,*args):
            assert events[:3]==['generate-fixture 0','generate-fixture 1','release-target']
            events.append('judge-'+prompt)
            return Judgment('compliance',False,0,1.,'Fixture check')
    phases=SimpleNamespace(idle=lambda:events.append('release-target'))
    result=eval_cached(tasks,Generator(),Judge(),tmp_path,'fixture-sandbox',{'fixture':'target'},
                       phases=phases,scoring_identity={'fixture':'judge'})
    assert result['invalid']==0 and len(events)==5


def test_search_requires_explicit_consistency_policy_before_loading_models(tmp_path):
    from types import SimpleNamespace
    from b70_ablator.pipeline import run
    with pytest.raises(ValueError,match='consistency'):
        run(SimpleNamespace(consistency=None))
