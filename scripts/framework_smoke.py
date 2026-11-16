"""Persisted framework QA using labeled fixtures. NO model quality measurements."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from b70_ablator.answers import AnswerStore, atomic_json, generate_answers, score_answers
from b70_ablator.scoring import Judgment


class FixtureGenerator:
    def __init__(self,interrupt=False):
        self.interrupt=interrupt
        self.calls=[]
    def generate(self,prompt):
        self.calls.append(prompt)
        if self.interrupt and prompt=='Fixture maths: two plus two?':
            raise RuntimeError('Intentional fixture interruption')
        return {'content':'4' if 'maths' in prompt else 'blue', 'finish_reason':'stop',
                'thinking_open':False,'completion_tokens':1,'reasoning_tokens':0}


class FixtureJudge:
    def __init__(self,invalid=False):
        self.invalid=invalid
        self.calls=[]
    def classify(self,prompt,answer,language):
        self.calls.append(prompt)
        if self.invalid:
            raise ValueError('Intentional fixture invalid judgment')
        assert answer=='blue'
        return Judgment('compliance',False,0,1.,'Exact fixture output, not an LLM judgment')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new directory; fixture checks never overwrite real runs')
    tasks=[{'id':'r','kind':'refusal','language':'en','prompt':'Fixture: return blue.'},
           {'id':'m','kind':'math','language':'en','prompt':'Fixture maths: two plus two?','expected':'4'}]
    store=AnswerStore(args.output/'answers',tasks,{'kind':'explicit-deterministic-fixture','fixture_version':1})
    first=FixtureGenerator(interrupt=True)
    try:
        generate_answers(tasks,first,store)
    except RuntimeError as error:
        assert str(error)=='Intentional fixture interruption'
    else:
        raise AssertionError('Interruption fixture did not interrupt')
    assert store.status()['completed']==1
    resumed=FixtureGenerator()
    generated=generate_answers(tasks,resumed,store)
    assert len(resumed.calls)==1 and generated['reused']==1
    identity={'kind':'explicit-deterministic-fixture-judge','fixture_version':1}
    rejected=score_answers(tasks,store,FixtureJudge(invalid=True),args.output/'scores',identity)
    assert rejected['invalid']==1
    repaired=FixtureJudge()
    final=score_answers(tasks,store,repaired,args.output/'scores',identity,retry_invalid=True)
    assert final['invalid']==0 and len(repaired.calls)==1
    binary=Path(sys.executable).parent/'b70-ablator'
    readback=subprocess.run([str(binary),'status','--answers',str(store.directory.resolve()),
                            '--scores',str((args.output/'scores').resolve())],capture_output=True,text=True,check=True)
    status=json.loads(readback.stdout)
    assert status['completed']==2 and status['scoring']['stage']=='scoring-complete'
    proof={'purpose':'Framework fixture QA only; NOT model inference, compliance measurement, or a benchmark',
           'resumed_generation_calls':len(resumed.calls),'reused_answers':generated['reused'],
           'invalid_scores_before_retry':rejected['invalid'],'invalid_scores_after_retry':final['invalid'],
           'judge_calls_for_retry':len(repaired.calls),'console_status':status}
    atomic_json(args.output/'verification.json',proof)
    print(json.dumps(proof,indent=2))


if __name__=='__main__':
    main()
