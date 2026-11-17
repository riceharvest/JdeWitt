"""Token/template budget controls using CPU tokenizers and transport fixtures."""
from types import SimpleNamespace
import pytest
import torch
from jdewitt.hf import HFModel, BoundaryContract
from jdewitt.transport import ChatClient


class Tokens:
    chat_template='explicit test template'
    def apply_chat_template(self, messages, **kwargs):
        return 'system: '+messages[0]['content']+' user: '+messages[1]['content']+' assistant:'
    def encode(self, text, **kwargs):
        return list(range(len(text.split())))
    def __call__(self, text, **kwargs):
        return {'input_ids':torch.tensor([self.encode(text)]), 'attention_mask':torch.ones(1,len(self.encode(text)),dtype=torch.long)}


def backend(ctx_size=12, max_tokens=4):
    target=HFModel.__new__(HFModel)
    target.tokenizer=Tokens()
    target.device=torch.device('cpu')
    target.boundary_contract=BoundaryContract.for_architecture('llama')
    target.max_tokens=max_tokens
    target.ctx_size=ctx_size
    target.config=SimpleNamespace(max_position_embeddings=128)
    return target


def test_hf_rejects_prompt_plus_output_before_model_forward_or_device_transfer():
    target=backend()
    target.device='invalid-device-proves-transfer-was-not-reached'
    with pytest.raises(ValueError,match='context'):
        target.encode('one two three four five six seven eight nine')


def test_budget_counts_system_template_and_generation_prefix_not_only_user_words():
    target=backend()
    with pytest.raises(ValueError,match='context'):
        target.encode('one two three')  # 11 rendered tokens plus 4 output tokens
    target.ctx_size=16
    ids,_=target.encode('one two three')
    assert ids['input_ids'].shape[1]==11


def test_full_suite_preflight_reports_lengths_without_model_weights():
    target=backend(ctx_size=20)
    report=target.preflight([{'id':'a','prompt':'one'},{'id':'b','prompt':'one two three'}])
    assert report['prompt_tokens']=={'a':9,'b':11}
    assert report['output_reserve']==4 and report['ctx_size']==20


def client(tokens=5, ctx_size=12, output=4):
    obj=ChatClient('http://127.0.0.1:9/v1','fixture',output,ctx_size=ctx_size)
    calls=[]
    def request(path, payload=None):
        calls.append((path,payload))
        if path=='/chat/completions/input_tokens':
            return {'input_tokens':tokens}
        return {'model':'fixture','choices':[{'message':{'content':'fixture reply'},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':tokens,'completion_tokens':1}}
    obj._request=request
    return obj,calls


def test_gguf_budget_uses_actual_server_template_tokens_before_generation():
    obj,calls=client(tokens=9)
    with pytest.raises(ValueError,match='context'):
        obj.generate('short user message with a long template')
    assert [p for p,_ in calls]==['/chat/completions/input_tokens']


def test_gguf_counter_and_generation_use_identical_template_settings():
    obj,calls=client()
    obj.generate('fixture',template_kwargs={'enable_thinking':False})
    assert calls[0][1]==calls[1][1]
    assert calls[0][1]['chat_template_kwargs']=={'enable_thinking':False}


def test_gguf_missing_or_malformed_token_count_cannot_silently_skip_budget():
    obj,_=client()
    obj._request=lambda *args,**kwargs:{'input_tokens':True}
    with pytest.raises(ValueError,match='token count'):
        obj.generate('fixture')


def test_legacy_counter_uses_actual_native_template_and_special_tokens(monkeypatch):
    from urllib.error import HTTPError
    calls=[]
    def request(self,path,payload=None):
        calls.append((path,payload))
        if path=='/chat/completions/input_tokens':
            raise HTTPError(self.url,404,'Fixture older llama.cpp',{},None)
        if path=='/apply-template':
            return {'prompt':'<bos> system fixture user fixture assistant prefix'}
        if path=='/tokenize':
            assert payload['add_special'] is True and payload['parse_special'] is True
            return {'tokens':[1,2,3,4,5]}
        return {'model':'fixture','choices':[{'message':{'content':'fixture'},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':5,'completion_tokens':1}}
    monkeypatch.setattr(ChatClient,'_request',request)
    obj=ChatClient('http://127.0.0.1:9/v1','fixture',4,ctx_size=12)
    obj.generate('fixture',template_kwargs={'enable_thinking':False})
    assert [p for p,_ in calls]==['/chat/completions/input_tokens','/apply-template','/tokenize','/chat/completions']
    assert calls[1][1]==calls[-1][1]
