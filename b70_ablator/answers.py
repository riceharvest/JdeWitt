# SPDX-License-Identifier: AGPL-3.0-or-later
"""Durable answer generation and independent, resumable scoring jobs."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import uuid

from .data import digest
from .evaluation import score_response, DominatedTrial
from .scoring import summarize, cannot_beat, response_failed


@contextmanager
def exclusive(directory,name='.lock'):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/name).open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(f'Another job owns {directory}') from e
        try:
            yield
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN)


def atomic_text(path,text):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.new')
    try:
        with temporary.open('x') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,path)
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path,value):
    atomic_text(path,json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def inputs_for(suite):
    rows=[{k:r[k] for k in ('id','prompt')} for r in suite]
    if not rows or len({r['id'] for r in rows})!=len(rows) or any(not isinstance(r[k],str) or not r[k] for r in rows for k in ('id','prompt')):
        raise ValueError('Answer inputs need nonempty unique IDs and prompts')
    return sorted(rows,key=lambda r:r['id'])


def portable_response(response):
    if not isinstance(response,dict):
        raise ValueError('Model response must be an object')
    result={k:response.get(k) for k in ('content','finish_reason','completion_tokens','reasoning_tokens','final_marker')}
    result['thinking_open']=response.get('thinking_open',False)
    # The floor changes how a terminal state must be read, so it is part of the
    # saved answer, not a re-derivable setting. Absent is fine; wrong is not.
    floor = response.get('floor_enforced', False)
    if type(floor) is not bool:
        raise ValueError('Invalid floor flag in saved response')
    result['floor_enforced'] = floor
    # Preserve unexpected server reasoning separately; never grade it as an answer.
    # Conditional inclusion keeps historical records readable without rewriting them.
    if 'reasoning_content' in response:
        result['reasoning_content']=response['reasoning_content']
        if result['reasoning_content'] is not None and not isinstance(result['reasoning_content'],str):
            raise ValueError('Invalid reasoning text field in saved response')
    for key in ('content','finish_reason','final_marker'):
        if result[key] is not None and not isinstance(result[key],str):
            raise ValueError('Invalid text field in saved response')
    for key in ('completion_tokens','reasoning_tokens'):
        if result[key] is not None and (type(result[key]) is not int or result[key]<0):
            raise ValueError('Invalid token count in saved response')
    if type(result['thinking_open']) is not bool:
        raise ValueError('Invalid thinking flag in saved response')
    return result


class AnswerStore:
    def __init__(self,directory,suite,identity,create=True):
        if not isinstance(identity,dict) or not identity:
            raise ValueError('Explicit model and generation identity required')
        self.directory=Path(directory)
        self.inputs=inputs_for(suite)
        self.tasks={r['id']:r for r in self.inputs}
        self.manifest={'version':1,'identity':json.loads(json.dumps(identity,allow_nan=False)),'inputs':self.inputs}
        self.manifest['generation_digest']=digest(self.manifest)
        path=self.directory/'manifest.json'
        if create:
            with exclusive(self.directory):
                if path.exists():
                    if json.loads(path.read_text())!=self.manifest:
                        raise ValueError('Saved answer identity differs from requested model, settings or prompts')
                else:
                    if any((self.directory/'items').glob('*.json')):
                        raise ValueError('Saved answers without identity manifest')
                    atomic_json(path,self.manifest)
        elif not path.exists() or json.loads(path.read_text())!=self.manifest:
            raise ValueError('Saved answer identity manifest missing or inconsistent')

    @classmethod
    def open(cls,directory,suite):
        saved=json.loads((Path(directory)/'manifest.json').read_text())
        return cls(directory,suite,saved['identity'],create=False)

    def _path(self,task):
        if self.tasks.get(task['id'])!={k:task[k] for k in ('id','prompt')}:
            raise ValueError('Answer task identity mismatch')
        return self.directory/'items'/(digest(task['id'])+'.json')

    def get(self,task):
        path=self._path(task)
        if not path.exists():
            return None
        record=json.loads(path.read_text())
        expected={'id':task['id'],'prompt':task['prompt'],'generation_digest':self.manifest['generation_digest']}
        if set(record)!={*expected,'response','record_digest'} or any(record[k]!=v for k,v in expected.items()):
            raise ValueError('Saved answer integrity/identity mismatch')
        payload={k:v for k,v in record.items() if k!='record_digest'}
        if record['record_digest']!=digest(payload) or portable_response(record['response'])!=record['response']:
            raise ValueError('Saved answer integrity check failed')
        return record['response']

    def put(self,task,response):
        snapshot=portable_response(response)
        path=self._path(task)
        with exclusive(self.directory):
            existing=self.get(task)
            if existing is not None:
                if existing!=snapshot:
                    raise ValueError('Never overwrite a saved model answer')
                return
            record={'id':task['id'],'prompt':task['prompt'],'generation_digest':self.manifest['generation_digest'],'response':snapshot}
            record['record_digest']=digest(record)
            atomic_json(path,record)

    def status(self):
        completed=sum(self.get(r) is not None for r in self.inputs)
        return {'answers':str(self.directory.resolve()),'total':len(self.inputs),'completed':completed,
                'missing':len(self.inputs)-completed,'generation_digest':self.manifest['generation_digest']}


def generate_answers(suite,generator,store,refusal_floor=None,batch_policy=None,budget=None):
    from .batching import plan_batches,validate_batch_policy
    if budget is not None and (type(budget) is not int or budget < 1):
        raise ValueError('generation budget must be a positive integer token count')
    validate_batch_policy(batch_policy)
    generated,reused,batches=0,0,0
    budget_used=0
    exhausted=False
    with exclusive(store.directory,'.generation.lock'):
        pending=[task for task in suite if store.get(task) is None]
        reused=len(suite)-len(pending)
        if pending and generator is None:
            raise ValueError('Saved answer missing; no target generator provided')
        pending_ids={t['id'] for t in pending}
        planned=plan_batches(suite if batch_policy is not None and pending else pending,
                             generator,refusal_floor,batch_policy)
        for batch,floor in planned:
            if not any(t['id'] in pending_ids for t in batch):
                continue
            batches+=1
            if batch_policy is None:
                prompt=batch[0]['prompt']
                responses=iter([generator.generate(prompt) if floor is None else generator.generate(prompt,min_tokens=floor)])
            else:
                responses=iter(generator.generate_batch([t['prompt'] for t in batch],min_tokens=floor))
            for task in batch:
                try:
                    response=next(responses)
                except StopIteration as error:
                    raise ValueError('Batch backend returned too few answers') from error
                if task['id'] not in pending_ids:
                    # Retain original batch companions on interrupted replay:
                    # changing batch shape can change low-precision arithmetic.
                    # Completed records remain immutable and are never replaced.
                    continue
                if floor is not None and response.get('floor_enforced') is not True:
                    raise ValueError('Backend did not confirm the requested refusal floor')
                store.put(task,response)
                generated+=1
                if budget is not None:
                    tokens=response.get('completion_tokens')
                    budget_used+=int(tokens or 0)
                    if budget_used>=budget:
                        exhausted=True
                        break
                # One verified ledger scan up front, not a full reread after
                # every prompt. The generation lock owns this completed count.
                atomic_json(store.directory/'progress.json',{'stage':'generation','total':len(suite),
                    'completed':reused+generated,'missing':len(suite)-reused-generated,
                    'generation_digest':store.manifest['generation_digest']})
            if not exhausted:
                sentinel=object()
                if next(responses,sentinel) is not sentinel:
                    raise ValueError('Batch backend returned too many answers')
            if exhausted:
                break
    result={**store.status(),'generated':generated,'reused':reused,'batches':batches}
    if budget is not None:
        result['budget']=budget
        result['budget_used']=budget_used
        result['budget_exhausted']=exhausted
    atomic_json(store.directory/'progress.json',{'stage':'budget-exhausted' if exhausted else 'generation-complete',**result})
    return result


def scoring_manifest(suite,store,directory,identity,sandbox_image):
    if not isinstance(identity,dict) or not identity:
        raise ValueError('Explicit scoring identity required')
    directory=Path(directory)
    manifest={'version':2,'suite_digest':digest(suite),'generation_digest':store.manifest['generation_digest'],
              'identity':json.loads(json.dumps(identity,allow_nan=False)),'sandbox_image':sandbox_image}
    manifest['scoring_digest']=digest(manifest)
    path=directory/'scoring-manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise ValueError('Scoring identity changed; use a new scoring directory')
    if not path.exists() and ((directory/'report.json').exists() or (directory/'predictions.jsonl').exists()):
        raise ValueError('Unidentified older scores cannot be resumed')
    return manifest


def saved_score(task, store, directory, scoring_digest):
    path=Path(directory)/'items'/(digest(task['id'])+'.json')
    if not path.exists():
        return None
    record=json.loads(path.read_text())
    payload={k:v for k,v in record.items() if k!='record_digest'}
    if set(record)!={'scoring_digest','answer_digest','row','record_digest'} or record.get('record_digest')!=digest(payload) or record.get('scoring_digest')!=scoring_digest:
        raise ValueError('Saved score integrity/identity check failed')
    response=store.get(task)
    if response is None or record['answer_digest']!=digest(response):
        raise ValueError('Saved score is not bound to the exact saved answer')
    row=record['row']
    if any(row.get(k)!=task[k] for k in ('id','prompt','kind','language') if k in task) or row.get('raw')!=response['content']:
        raise ValueError('Saved score task/answer identity mismatch')
    return row


def completed_scores(suite, store, directory, identity, sandbox_image):
    """Validate the entire completed ledger, including raw records, before reuse."""
    directory=Path(directory)
    if not (directory/'report.json').exists():
        return None
    manifest=scoring_manifest(suite,store,directory,identity,sandbox_image)
    report=json.loads((directory/'report.json').read_text())
    if report.get('suite_digest')!=manifest['suite_digest'] or report.get('scoring_digest')!=manifest['scoring_digest'] or report.get('generation_digest')!=manifest['generation_digest']:
        raise ValueError('Completed report identity mismatch')
    expected={digest(t['id'])+'.json' for t in suite}
    if {p.name for p in (directory/'items').glob('*.json')}!=expected:
        raise ValueError('Completed score records are missing or unexpected')
    rows=[saved_score(t,store,directory,manifest['scoring_digest']) for t in suite]
    if any(r is None for r in rows):
        raise ValueError('Completed score record missing')
    predictions=directory/'predictions.jsonl'
    if not predictions.exists():
        raise ValueError('Completed predictions missing')
    actual=[json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
    if actual!=rows or report['report']!=summarize(rows,[t['id'] for t in suite]):
        raise ValueError('Completed predictions/report disagree with the verified records')
    return report['report']


def score_status(directory,store):
    directory=Path(directory)
    path=directory/'scoring-manifest.json'
    if not path.exists():
        return {'stage':'not-started','completed':0,'total':len(store.inputs),'invalid':0}
    manifest=json.loads(path.read_text())
    if manifest.get('generation_digest')!=store.manifest['generation_digest'] or manifest.get('scoring_digest')!=digest({k:v for k,v in manifest.items() if k!='scoring_digest'}):
        raise ValueError('Score/answer identity mismatch')
    rows=[]
    for task in store.inputs:
        path=directory/'items'/(digest(task['id'])+'.json')
        if not path.exists():
            continue
        row=saved_score(task,store,directory,manifest['scoring_digest'])
        rows.append(row)
    invalid=sum(r['status']!='ok' for r in rows)
    complete=False
    path=directory/'report.json'
    if len(rows)==len(store.inputs) and path.exists():
        report=json.loads(path.read_text())
        complete=report.get('scoring_digest')==manifest['scoring_digest'] and report.get('report')==summarize(rows,[r['id'] for r in store.inputs])
    return {'stage':('needs-review' if invalid else 'scoring-complete') if complete else 'scoring-pending',
            'completed':len(rows),'total':len(store.inputs),'invalid':invalid,'report_current':complete}


def score_answers(suite,store,judge,directory,identity,sandbox_image='python:3.12-slim',retry_invalid=False,incumbent=None,max_judge_requests=None):
    # Check all inputs BEFORE writing reports or calling a judge.
    from .errors import BudgetExhausted
    if max_judge_requests is not None and (type(max_judge_requests) is not int or max_judge_requests < 1):
        raise ValueError('max_judge_requests must be a positive integer')
    responses={r['id']:store.get(r) for r in suite}
    if any(r is None for r in responses.values()):
        raise ValueError('Cannot score: saved answers are missing')
    directory=Path(directory)
    manifest=scoring_manifest(suite,store,directory,identity,sandbox_image)
    scoring_digest=manifest['scoring_digest']
    found=completed_scores(suite,store,directory,identity,sandbox_image)
    if found is not None and not (retry_invalid and found['invalid']):
        return found
    rows=[]
    bad=0
    total_refusal=sum(r['kind']=='refusal' for r in suite)
    with exclusive(directory,'.scoring.lock'):
        if found is not None and retry_invalid and found['invalid']:
            history=directory/'history'
            history.mkdir(parents=True,exist_ok=True)
            prefix=uuid.uuid4().hex+'-'
            # Retire the old complete report BEFORE modifying any of its rows.
            # A disconnect during retry then resumes from the valid item ledger.
            for name in ('report.json','predictions.jsonl'):
                old=directory/name
                if old.exists():
                    os.replace(old,history/(prefix+name))
        path=directory/'scoring-manifest.json'
        if path.exists():
            if json.loads(path.read_text())!=manifest:
                raise ValueError('Scoring identity changed; use a new scoring directory')
        else:
            if (directory/'report.json').exists() or (directory/'predictions.jsonl').exists():
                raise ValueError('Unidentified older scores cannot be resumed')
            atomic_json(path,manifest)
        if hasattr(judge,'bind_rubrics'):
            judge.bind_rubrics([r for r in suite if r['kind']=='refusal'])
        fresh=0
        for task_index,task in enumerate(suite):
            path=directory/'items'/(digest(task['id'])+'.json')
            row=None
            if path.exists():
                row=saved_score(task,store,directory,scoring_digest)
            if row is None or retry_invalid and row['status']=='judge_invalid':
                if row is not None:
                    history=directory/'history'/(path.stem+'-'+uuid.uuid4().hex+'.json')
                    history.parent.mkdir(parents=True,exist_ok=True)
                    os.replace(path,history)
                row=score_response(task,responses[task['id']],judge,sandbox_image)
                record={'scoring_digest':scoring_digest,'answer_digest':digest(responses[task['id']]),'row':row}
                record['record_digest']=digest(record)
                atomic_json(path,record)
                fresh+=1
                if max_judge_requests is not None and fresh>=max_judge_requests and task_index<len(suite)-1:
                    raise BudgetExhausted(f'Judge request budget {max_judge_requests} exhausted after {fresh} '
                                          'fresh scores; saved items persist and rerunning continues from them')
            rows.append(row)
            atomic_json(directory/'progress.json',{'stage':'scoring','completed':len(rows),'total':len(suite),
                                                   'invalid':sum(r['status']!='ok' for r in rows)})
            if task['kind']=='refusal':
                bad+=response_failed(row)
            if incumbent is not None and total_refusal and cannot_beat(bad,total_refusal,incumbent):
                raise DominatedTrial('Saved answers cannot beat the current best trial')
        result=summarize(rows,[r['id'] for r in suite])
        atomic_text(directory/'predictions.jsonl',''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
        atomic_json(directory/'report.json',{'suite_digest':digest(suite),'report':result,
                                            'scoring_digest':scoring_digest,'generation_digest':store.manifest['generation_digest']})
        atomic_json(directory/'progress.json',{'stage':'needs-review' if result['invalid'] else 'scoring-complete',
                                               'completed':len(rows),'total':len(suite),'invalid':result['invalid']})
    return result
