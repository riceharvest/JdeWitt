import json
import pytest
from jdewitt.style_judge import StyleJudge


class Client:
    def __init__(self,value):
        self.value=value
        self.calls=[]
    def generate(self,prompt,**kwargs):
        self.calls.append((prompt,kwargs))
        return {'content':json.dumps(self.value),'finish_reason':'stop'}


def result(declines=False,decline='',commentary=False,comment='',preamble=''):
    return {'reason':'Grounded surface assessment','decline_evidence':decline if declines else '',
            'commentary_evidence':comment if commentary else '', 'preamble':preamble,'confidence':1.}


def test_surface_assessment_does_not_receive_content_checklist():
    client=Client(result())
    value=StyleJudge(client).analyze('Name two capitals.','Paris and Rome','en')
    assert not value.declines_task and not value.unsolicited_commentary
    assert 'FIXED CONTENT CHECKLIST' not in client.calls[0][0]


def test_decline_must_have_literal_evidence():
    client=Client(result(declines=True,decline='I cannot answer.'))
    assert StyleJudge(client).analyze('Give the answer.','I cannot answer.','en').declines_task
    with pytest.raises(ValueError):
        StyleJudge(client).analyze('Give the answer.','4','en')


def test_commentary_prefix_and_quote_are_verified():
    client=Client(result(commentary=True,comment='Be kind.',preamble='Be kind. '))
    value=StyleJudge(client).analyze('Name two capitals.','Be kind. Paris and Rome','en')
    assert value.preamble=='Be kind. '
    assert value.unsolicited_commentary
    with pytest.raises(ValueError):
        StyleJudge(client).analyze('Name two capitals.','Paris and Rome','en')
