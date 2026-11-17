"""Real conversion/quantization/B70 generation smoke. Not an ablation benchmark."""
import argparse
import json
from pathlib import Path
from jdewitt.release import export_gguf, ManagedGGUF, verify_quant, QUANTS
from jdewitt.evaluation import grade_math
from jdewitt.scoring import extract_answer

parser = argparse.ArgumentParser()
parser.add_argument('--hf',type=Path,required=True)
parser.add_argument('--converter',type=Path,required=True)
parser.add_argument('--quantizer',type=Path,required=True)
parser.add_argument('--python',type=Path,required=True)
parser.add_argument('--server',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--reuse-verified-quants',action='store_true')
args = parser.parse_args()
if args.reuse_verified_quants:
    artifacts = {q:verify_quant(args.output/'quants'/f'model-{q}.gguf',q) for q in QUANTS}
else:
    artifacts = export_gguf(args.hf,args.output/'quants',args.converter,args.quantizer,args.python)
results = {'purpose':'real infrastructure smoke, NOT a model-quality or ablation result','artifacts':artifacts,'responses':{}}
for quant,artifact in artifacts.items():
    with ManagedGGUF(artifact['path'],args.server,4096,2048,args.output/f'{quant}-server.log') as client:
        response = client.generate('What is 2 + 2? Finish your answer with just the resulting integer.')
        answer = extract_answer(response['content'],response['finish_reason'])
        results['responses'][quant] = {'answer':answer.text,'complete':answer.complete,'passed':answer.complete and grade_math(answer.text,'4'),'completion_tokens':response['completion_tokens'],'reasoning_tokens':response['reasoning_tokens']}
(args.output/'smoke.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
if not all(r['passed'] for r in results['responses'].values()):
    raise SystemExit('Real quantized generation smoke failed; inspect saved responses')
