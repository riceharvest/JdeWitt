# SPDX-License-Identifier: AGPL-3.0-or-later
"""Own one GPU model at a time. Reload from disk, never CPU-offload weights."""
import copy
import gc
import torch


class ModelPhases:
    def __init__(self,target_factory=None,judge_factory=None):
        self.target_factory,self.judge_factory=target_factory,judge_factory
        self.current_target=None
        self.current_judge=None
        self.judge_context=None
        self.metadata={}
        self.edit=None
        self.closed=False

    def __enter__(self):
        return self

    def _check(self):
        if self.closed:
            raise RuntimeError('Model phase owner is closed')

    def idle(self):
        target,self.current_target=self.current_target,None
        if target is not None:
            try:
                target.release()
            except BaseException:
                self.closed=True
                raise
            finally:
                gc.collect()
        context,self.judge_context=self.judge_context,None
        self.current_judge=None
        if context is not None:
            try:
                context.__exit__(None,None,None)
            except BaseException:
                self.closed=True
                raise

    def target(self):
        self._check()
        if self.current_target is not None:
            return self.current_target
        self.idle()
        if self.target_factory is None:
            raise RuntimeError('Target factory is not configured')
        target=self.target_factory()
        try:
            fingerprint=getattr(target,'fingerprint',None)
            if self.metadata and fingerprint!=self.metadata['fingerprint']:
                raise ValueError('Reloaded target identity changed')
            if not self.metadata:
                self.metadata={'fingerprint':copy.deepcopy(fingerprint),'device':target.device,'tokenizer':target.tokenizer}
            if self.edit is not None:
                directions,params=self.edit
                from .plans import move_plan
                target.editor.apply(move_plan(directions,target.device),copy.deepcopy(params))
        except BaseException:
            target.release()
            raise
        self.current_target=target
        return target

    def judge(self):
        self._check()
        if self.current_judge is not None:
            return self.current_judge
        self.idle()
        if self.judge_factory is None:
            raise RuntimeError('Judge factory is not configured')
        context=self.judge_factory()
        judge=context.__enter__()
        self.judge_context,self.current_judge=context,judge
        return judge

    def __exit__(self,*args):
        try:
            self.idle()
        finally:
            self.closed=True


class PhasedEditor:
    def __init__(self,phases):
        self.phases=phases

    def apply(self,directions,params):
        from .plans import move_plan
        snapshot=copy.deepcopy(move_plan(directions,'cpu'))
        parameters=copy.deepcopy(params)
        target=self.phases.target()
        target.editor.apply(move_plan(snapshot,target.device),parameters)
        self.phases.edit=(snapshot,parameters)

    def reset(self):
        self.phases.edit=None
        if self.phases.current_target is not None:
            self.phases.current_target.editor.reset()

    @property
    def targets(self):
        return self.phases.target().editor.targets

    @property
    def layers(self):
        return self.phases.target().editor.layers

    @property
    def groups(self):
        return self.phases.target().editor.groups

    @property
    def edit_targets(self):
        return self.phases.target().editor.edit_targets


class PhasedHF:
    def __init__(self,phases):
        self.phases=phases
        self.editor=PhasedEditor(phases)

    def _metadata(self,key):
        if not self.phases.metadata:
            self.phases.target()
        return self.phases.metadata[key]

    @property
    def fingerprint(self):
        return copy.deepcopy(self._metadata('fingerprint'))

    @property
    def device(self):
        return self._metadata('device')

    @property
    def tokenizer(self):
        return self._metadata('tokenizer')

    def generate(self,prompt,min_tokens=None):
        target=self.phases.target()
        response=target.generate(prompt) if min_tokens is None else target.generate(prompt,min_tokens=min_tokens)
        # Token IDs are small storage records, not model weights. Do not leave
        # response-held GPU tensors alive when the judge takes ownership.
        if isinstance(response.get('sequence'),torch.Tensor):
            response['sequence']=response['sequence'].detach().cpu()
        return response

    @property
    def max_tokens(self):
        return self.phases.target().max_tokens

    def prompt_tokens(self,prompt):
        return self.phases.target().prompt_tokens(prompt)

    def generate_batch(self,prompts,min_tokens=None,max_tokens=None):
        responses=self.phases.target().generate_batch(prompts,min_tokens=min_tokens,max_tokens=max_tokens)
        for response in responses:
            if isinstance(response.get('sequence'),torch.Tensor):
                response['sequence']=response['sequence'].detach().cpu()
        return responses

    def residual_at_answer(self,prompt,response):
        target=self.phases.target()
        copied={**response,'sequence':response['sequence'].to(target.device)}
        return target.residual_at_answer(prompt,copied).detach().cpu()

    def residual_mean_at_answer(self,prompt,response):
        target=self.phases.target()
        copied={**response,'sequence':response['sequence'].to(target.device)}
        return target.residual_mean_at_answer(prompt,copied).detach().cpu()

    def module_io_at_answer(self,prompt,response):
        target=self.phases.target()
        copied={**response,'sequence':response['sequence'].to(target.device)}
        return target.module_io_at_answer(prompt,copied)

    def trajectory_kl(self,sequence,prompt_length,answer_start):
        target=self.phases.target()
        return target.trajectory_kl(sequence.to(target.device),prompt_length,answer_start)

    def answer_start_margins(self,prompts,refusal_tokens,reference_tokens):
        return self.phases.target().answer_start_margins(prompts,refusal_tokens,reference_tokens)

    def release(self):
        self.phases.idle()


class PhasedJudge:
    def __init__(self,phases,rubric_binder):
        self.phases,self.rubric_binder=phases,rubric_binder

    def bind_rubrics(self,records):
        self.rubric_binder.bind_rubrics(records)
        if self.phases.current_judge is not None:
            self.phases.current_judge.bind_rubrics(records)

    def classify(self,*args):
        return self.phases.judge().classify(*args)

    def calibrate(self,*args):
        return self.phases.judge().calibrate(*args)

    def verify(self):
        self.phases.judge().client.verify()
