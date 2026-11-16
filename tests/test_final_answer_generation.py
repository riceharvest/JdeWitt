"""Final-answer policy fixtures; no model inference or GPU initialization."""

from b70_ablator.hf import HFModel
from b70_ablator.transport import ChatClient
from b70_ablator.answers import AnswerStore, portable_response


def test_hf_template_closes_thinking_instead_of_starting_it():
    calls=[]
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            calls.append(kwargs)
            return 'assistant</think>' if kwargs.get('enable_thinking') is False else 'assistant<think>'
    model=HFModel.__new__(HFModel)
    model.tokenizer=Tokenizer()
    assert model._render('fixture').endswith('</think>')
    assert calls[0]['enable_thinking'] is False


def test_gguf_default_disables_thinking_with_zero_budget_for_count_and_generation():
    calls=[]
    client=ChatClient('http://127.0.0.1:9/v1','fixture',20,ctx_size=64)
    def request(path,payload):
        calls.append((path,payload))
        if path.endswith('/input_tokens'):
            return {'input_tokens':6}
        return {'model':'fixture','choices':[{'message':{'content':'answer','reasoning_content':'unexpected fixture reasoning'},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':6,'completion_tokens':2}}
    client._request=request
    response=client.generate('fixture')
    assert calls[0][1]==calls[1][1]
    assert calls[1][1]['chat_template_kwargs']['enable_thinking'] is False
    assert calls[1][1]['reasoning_budget_tokens']==0
    assert response['content']=='answer'
    assert response['reasoning_content']=='unexpected fixture reasoning'


def test_no_thinking_preserves_other_template_options():
    client=ChatClient('http://127.0.0.1:9/v1','fixture',20)
    calls=[]
    def request(path,payload):
        calls.append(payload)
        return {'model':'fixture','choices':[{'message':{'content':'answer'},'finish_reason':'stop'}]}
    client._request=request
    client.generate('fixture',template_kwargs={'fixture_option':'unchanged'})
    assert calls[0]['chat_template_kwargs']=={'enable_thinking':False,'fixture_option':'unchanged'}
    assert calls[0]['reasoning_budget_tokens']==0


def test_unexpected_reasoning_is_saved_separately_and_old_answer_records_stay_readable(tmp_path):
    tasks=[{'id':'a','prompt':'fixture'}]
    store=AnswerStore(tmp_path,tasks,{'policy':'final-answer-only-fixture'})
    response={'content':'answer','finish_reason':'stop','reasoning_content':'unexpected fixture reasoning'}
    store.put(tasks[0],response)
    saved=store.get(tasks[0])
    assert saved['content']=='answer'
    assert saved['reasoning_content']=='unexpected fixture reasoning'
    old=portable_response({'content':'old answer','finish_reason':'stop'})
    assert 'reasoning_content' not in old
    assert portable_response(old)==old
