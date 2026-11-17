"""Measure an actual local judge on authored calibration cases, not target-model quality."""
import argparse
import json
from pathlib import Path
from jdewitt.release import ManagedGGUF
from jdewitt.grounded_judge import GroundedJudge
from jdewitt.data import append_jsonl

parser=argparse.ArgumentParser()
parser.add_argument('--judge',type=Path,required=True)
parser.add_argument('--server',type=Path,required=True)
parser.add_argument('--cases',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
args.output.mkdir(parents=True)
cases=json.loads(args.cases.read_text())
rows=[]
with ManagedGGUF(args.judge,args.server,4096,2048,args.output/'server.log') as client:
    judge=GroundedJudge(client,min_confidence=0.,rubric_cache=args.output/'rubrics.json')
    judge.bind_rubrics(cases)
    for case in cases:
        row={'id':case['id'],'language':case['language'],'expected_verdict':case['expected_verdict'],'expected_disclaimer':case['expected_disclaimer']}
        try:
            verdict=judge.classify(case['prompt'],case['answer'],case['language'])
            row['judgment']=verdict.to_dict()
            row['correct']=verdict.verdict==case['expected_verdict'] and verdict.disclaimer==case['expected_disclaimer']
        except (ValueError,json.JSONDecodeError) as error:
            row['correct']=False
            row['error']=str(error)
        append_jsonl(args.output/'predictions.jsonl',row)
        rows.append(row)
        print(json.dumps(row),flush=True)
report={'purpose':'judge calibration on hand-authored cases; not an ablation benchmark','correct':sum(r['correct'] for r in rows),'total':len(rows),'by_language':{lang:{'correct':sum(r['correct'] for r in rows if r['language']==lang),'total':sum(r['language']==lang for r in rows)} for lang in sorted({r['language'] for r in rows})}}
(args.output/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
