# SPDX-License-Identifier: AGPL-3.0-or-later
import math
from dataclasses import dataclass, asdict
from pathlib import Path

STYLE_SYSTEM=(Path(__file__).parent/'assets/style-system.txt').read_text()
STYLE_SCHEMA={'type':'object','additionalProperties':False,'required':['reason','decline_evidence','commentary_evidence','preamble','confidence'],'properties':{
    'reason':{'type':'string'},'decline_evidence':{'type':'string'},'commentary_evidence':{'type':'string'},
    'preamble':{'type':'string'},'confidence':{'type':'number','minimum':0,'maximum':1}}}


@dataclass(frozen=True)
class StyleAssessment:
    reason:str
    decline_evidence:str
    commentary_evidence:str
    preamble:str
    confidence:float

    @property
    def declines_task(self):
        return bool(self.decline_evidence)

    @property
    def unsolicited_commentary(self):
        return bool(self.commentary_evidence)

    def to_dict(self):
        return asdict(self)


class StyleJudge:
    def __init__(self,client):
        self.client=client

    def analyze(self,prompt,answer,language):
        from .grounded_judge import response_json
        obj=response_json(self.client,{'request':prompt,'answer':answer,'language':language},STYLE_SYSTEM,STYLE_SCHEMA)
        if not isinstance(obj,dict) or set(obj)!=set(STYLE_SCHEMA['required']):
            raise ValueError('Invalid surface-assessment schema')
        for key in ('reason','decline_evidence','commentary_evidence','preamble'):
            if not isinstance(obj[key],str):
                raise ValueError('Surface evidence must be text')
        for key in ('decline_evidence','commentary_evidence'):
            if obj[key] and obj[key] not in answer:
                raise ValueError('Fabricated surface evidence')
        if not answer.startswith(obj['preamble']):
            raise ValueError('Preamble is not an exact answer prefix')
        confidence=obj['confidence']
        if type(confidence) not in (int,float) or not math.isfinite(confidence) or not 0<=confidence<=1:
            raise ValueError('Invalid surface confidence')
        return StyleAssessment(obj['reason'],obj['decline_evidence'],obj['commentary_evidence'],obj['preamble'],float(confidence))
