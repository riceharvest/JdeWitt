"""Real B70 resident/edit/export/quant smoke; not a certified ablation benchmark."""
import argparse
import gc
import json
import signal
from pathlib import Path
import time
import torch
from b70_ablator.answers import AnswerStore, atomic_json, generate_answers
from b70_ablator.data import load_suite
from b70_ablator.evaluation import score_response
from b70_ablator.export import stream_merge, verify_resident_export
from b70_ablator.hf import HFModel
from b70_ablator.numerics import directions_from_languages
from b70_ablator.refusal_check import RefusalChecker
from b70_ablator.release import export_gguf, ManagedGGUF, create_release, sha256_file
from b70_ablator.scoring import extract_answer, summarize
from b70_ablator.storage import save_checkpoint


def main():
    def terminate(signum,frame):
        raise KeyboardInterrupt('Smoke stopped; owned models/servers are released by their finally blocks')
    signal.signal(signal.SIGTERM,terminate)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    output=args.output
    if output.exists():
        raise FileExistsError('Use a new output directory for an independent hardware smoke')
    output.mkdir(parents=True)
    tasks=load_suite(cfg['selection'])
    direction_tasks=load_suite(cfg['directions'])
    source=Path(cfg['model'])
    checker=RefusalChecker()
    started=time.monotonic()
    results={'purpose':'Real infrastructure smoke, NOT a certified ablation benchmark or a full optimization search',
             'source':str(source),'settings':cfg,'scoring_mode':'refusal-check',
             'limitations':['Small legacy suites have no independent-family release evidence.',
                            'Explicit refusal wording is not semantic task fulfillment.',
                            'Fixed smoke edit strengths are not optimized release settings.'],
             'stages':{}}
    def checkpoint(stage,value):
        results['stages'][stage]=value
        results['elapsed_seconds']=time.monotonic()-started
        atomic_json(output/'verification.json',results)
        print(stage, json.dumps(value,ensure_ascii=False),flush=True)
    source_hashes={p.name:sha256_file(p) for p in source.iterdir() if p.suffix in {'.safetensors','.json','.jinja','.py'}}
    identity={'source_files':source_hashes,'max_tokens':cfg['max_tokens'],'ctx_size':cfg['ctx_size'],
              'framework_sources':{p.name:sha256_file(p) for p in (Path(__file__).resolve().parents[1]/'b70_ablator').glob('*.py')}}
    model=HFModel(str(source),None,1,cfg['max_tokens'],cfg['dtype'],cfg['trust_remote_code'],ctx_size=cfg['ctx_size'])
    try:
        checkpoint('loaded',{'all_weights_on_b70':all(p.device.type=='xpu' for p in model.model.parameters()),
                             'weights_allocated_bytes':torch.xpu.memory_allocated(),'fingerprint':model.fingerprint})
        model.preflight(tasks+direction_tasks)
        pointers=[p.data_ptr() for p in model.model.parameters()]
        baseline_store=AnswerStore(output/'baseline-hf'/'answers',tasks,identity)
        baseline_rows=[]
        baselines={}
        inference_seconds=0.
        token_count=0
        for task in tasks:
            torch.xpu.synchronize()
            t=time.monotonic()
            response=model.generate(task['prompt'])
            torch.xpu.synchronize()
            inference_seconds+=time.monotonic()-t
            token_count+=response['completion_tokens']
            baseline_store.put(task,response)
            baseline_rows.append(score_response(task,response,checker,cfg['sandbox_image']))
            baselines[task['id']]={**response,'sequence':response['sequence'].detach().cpu()}
        baseline_report=summarize(baseline_rows,[t['id'] for t in tasks])
        atomic_json(output/'baseline-hf'/'rows.json',baseline_rows)
        checkpoint('baseline_hf',{'report':baseline_report,'tokens_per_second':token_count/inference_seconds,
                                  'completion_tokens':token_count,'inference_seconds':inference_seconds})
        repeats={}
        for task in [r for r in tasks if r['kind']=='math']:
            repeated=model.generate(task['prompt'])
            repeats[task['id']]=torch.equal(repeated['sequence'].cpu(),baselines[task['id']]['sequence'])
        checkpoint('repeatability',repeats)
        if not all(repeats.values()):
            raise RuntimeError('Real greedy baseline generation did not reproduce')
        accepted={label:{} for label in ('good','bad')}
        counts={}
        for task in direction_tasks:
            response=model.generate(task['prompt'])
            answer=extract_answer(response['content'],response['finish_reason'],response['thinking_open'])
            if not answer.complete:
                continue
            verdict=checker.classify(task['prompt'],answer.text,task['language']).verdict
            keep=verdict=='no_explicit_refusal' if task['label']=='good' else verdict in {'refusal','partial'}
            if keep:
                residual=model.residual_at_answer(task['prompt'],response).cpu()
                accepted[task['label']].setdefault(task['language'],[]).append(residual)
            counts.setdefault(task['language'],{'good':0,'bad':0})[task['label']]+=int(keep)
        checkpoint('direction_counts',counts)
        languages={r['language'] for r in direction_tasks}
        if any(not accepted[label].get(language) for label in accepted for language in languages):
            raise RuntimeError('No confirmed good/bad direction evidence in every language; no fabricated direction fallback')
        means={label:{language:torch.stack(rows).mean(0) for language,rows in group.items()} for label,group in accepted.items()}
        directions=directions_from_languages(means['good'],means['bad'],1).to(model.device)
        save_checkpoint(output/'smoke-directions.pt',directions.cpu(),identity,counts)
        params={c:{'strength':.5,'center':.5,'width':1.,'floor':1.} for c in ('attn','mlp')}
        zero={c:{**v,'strength':0.} for c,v in params.items()}
        model.editor.apply(directions,zero)
        first=next(t for t in tasks if t['kind']=='math')
        zero_response=model.generate(first['prompt'])
        zero_equal=torch.equal(zero_response['sequence'].cpu(),baselines[first['id']]['sequence'])
        checkpoint('zero_edit',{'identical_tokens':zero_equal,'original_storage_unchanged':pointers==[p.data_ptr() for p in model.model.parameters()]})
        if not zero_equal:
            raise RuntimeError('Zero-edit control changed the actual generated tokens')
        model.editor.apply(directions,params)
        edited_rows=[]
        candidate_store=AnswerStore(output/'edited-hf'/'answers',tasks,{**identity,'edit_parameters':params})
        inference_seconds=0.
        token_count=0
        for task in tasks:
            torch.xpu.synchronize()
            t=time.monotonic()
            response=model.generate(task['prompt'])
            torch.xpu.synchronize()
            inference_seconds+=time.monotonic()-t
            token_count+=response['completion_tokens']
            candidate_store.put(task,response)
            edited_rows.append(score_response(task,response,checker,cfg['sandbox_image']))
        edited_report=summarize(edited_rows,[t['id'] for t in tasks])
        atomic_json(output/'edited-hf'/'rows.json',edited_rows)
        checkpoint('edited_hf',{'report':edited_report,'edit_parameters':params,'tokens_per_second':token_count/inference_seconds,
                               'completion_tokens':token_count,'inference_seconds':inference_seconds,
                               'original_storage_unchanged':pointers==[p.data_ptr() for p in model.model.parameters()],
                               'peak_allocated_bytes':torch.xpu.max_memory_allocated()})
        harmless=next(t for t in tasks if t['kind']=='harmless')
        reference=baselines[harmless['id']]
        from b70_ablator.hf import reply_boundary
        boundary=reply_boundary(model.tokenizer,reference['generated_ids'],reference['thinking_open'])
        checkpoint('full_reply_kl',model.trajectory_kl(reference['sequence'].to(model.device),reference['prompt_length'],boundary))
        encoded,_=model.encode(first['prompt'])
        with torch.inference_mode():
            expected_logits=model.model(**encoded,use_cache=False,logits_to_keep=1).logits.cpu()
        hf=output/'edited-safetensors'
        stream_merge(source,hf,model.editor)
        audit=verify_resident_export(source,hf,model.editor)
        checkpoint('resident_export_verification',audit)
    finally:
        model.release()
        gc.collect()
    reloaded=HFModel(str(hf),None,1,cfg['max_tokens'],cfg['dtype'],cfg['trust_remote_code'],ctx_size=cfg['ctx_size'])
    try:
        encoded,_=reloaded.encode(first['prompt'])
        with torch.inference_mode():
            actual=reloaded.model(**encoded,use_cache=False,logits_to_keep=1).logits.cpu()
        equal=torch.equal(expected_logits,actual)
        checkpoint('reloaded_bf16',{'exact_logits_equal':equal,'maximum_absolute_logit_difference':float((expected_logits-actual).abs().max())})
        if not equal:
            raise RuntimeError('Real stored BF16 logits differ from the resident candidate operator')
    finally:
        reloaded.release()
        gc.collect()
    quantized={}
    for name,hf_source in (('baseline',source),('edited',hf)):
        artifacts=export_gguf(hf_source,output/(name+'-quants'),cfg['converter'],cfg['quantizer'],cfg['converter_python'])
        quantized[name]={'artifacts':artifacts,'reports':{}}
        for quant,artifact in artifacts.items():
            directory=output/(name+'-quant-answers')/quant
            store=AnswerStore(directory,tasks,{'artifact_sha256':artifact['sha256'],'settings':identity})
            with ManagedGGUF(artifact['path'],cfg['llama_server'],cfg['ctx_size'],cfg['max_tokens'],output/f'{name}-{quant}-server.log') as client:
                generate_answers(tasks,client,store)
            rows=[score_response(task,store.get(task),checker,cfg['sandbox_image']) for task in tasks]
            atomic_json(directory/'rows.json',rows)
            report=summarize(rows,[t['id'] for t in tasks])
            quantized[name]['reports'][quant]=report
            checkpoint(name+'-'+quant,report)
    results['quantized']=quantized
    results['status']='infrastructure-smoke-complete-not-certified'
    results['winner']={'export_verification':audit}
    results['elapsed_seconds']=time.monotonic()-started
    create_release(output/'roundtrip-bundle',hf,quantized['edited']['artifacts'],results,
                   '# Real B70 round-trip smoke\n\nNot a certified ablation release. See evaluation.json for actual controls and limited explicit-wording results.\n')
    atomic_json(output/'verification.json',results)
    print('COMPLETE',str(output/'verification.json'),flush=True)


if __name__=='__main__':
    main()
