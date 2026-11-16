# SPDX-License-Identifier: AGPL-3.0-or-later
import inspect

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig
from .editor import ARCHITECTURES, Editor
from .numerics import token_kl
from .compat import compatible_source
from .scoring import answer_boundary_chars
from .data import digest
from .chat import render_chat


class BoundaryContract:
    # Explicit channel spellings; never infer a reasoning prefix from prose.
    FINAL_MARKERS = ('<\u007cchannel\u007c>final<\u007cmessage\u007c>', '<\uff5cchannel\uff5c>final<\uff5cmessage\uff5c>')
    MODES = {**{name:'optional-think' for name in ARCHITECTURES},
             'gpt_oss':'harmony', 'llama': 'plain', 'mistral': 'plain', 'qwen2': 'plain',
             'qwen3': 'optional-think', 'qwen3_moe': 'optional-think',
             'qwen3_5': 'optional-think', 'qwen3_5_text': 'optional-think', 'spark2_5': 'optional-think'}

    def __init__(self, architecture, mode):
        self.architecture, self.mode = architecture, mode

    @classmethod
    def for_architecture(cls, architecture):
        if architecture not in cls.MODES:
            raise ValueError(f'No verified boundary adapter for {architecture!r}')
        return cls(architecture, cls.MODES[architecture])

    def boundary(self, tokenizer, ids, thinking_open, final_marker=None):
        text = tokenizer.decode(ids, skip_special_tokens=False)
        if self.mode == 'plain' and (thinking_open or text.lstrip().startswith('<think>')):
            raise ValueError('Reasoning output contradicts plain boundary adapter')
        if self.mode!='harmony' and text.lstrip().startswith(('[meta_sep]', '<｜channel｜>', '<|channel|>', '[im_start]assistant[meta_sep]')):
            raise ValueError('Unverified final-channel format needs a boundary adapter')
        if self.mode=='harmony' and final_marker is None:
            final_marker=next((marker for marker in self.FINAL_MARKERS if marker in text),None)
            if final_marker is None and any(text.lstrip().startswith(m.split('final')[0]+'analysis') for m in self.FINAL_MARKERS):
                raise ValueError('missing_final_channel')
        return reply_boundary(tokenizer, ids, thinking_open,final_marker)

    def identity(self):
        return {'architecture': self.architecture, 'mode': self.mode, 'version': 1}


def require_b70():
    if not torch.xpu.is_available():
        raise RuntimeError('PyTorch XPU unavailable; CPU/CUDA fallback is forbidden')
    name = torch.xpu.get_device_name(0)
    if 'B70' not in name:
        raise RuntimeError(f'Expected B70, found {name}')
    return torch.device('xpu:0')


def reply_boundary(tokenizer, generated_ids, thinking_open=False, final_marker=None):
    text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    boundary = answer_boundary_chars(text, thinking_open, final_marker)
    if not boundary:
        return 0
    for length in range(1, len(generated_ids) + 1):
        decoded = tokenizer.decode(generated_ids[:length], skip_special_tokens=False)
        if len(decoded) >= boundary:
            if decoded != text[:boundary]:
                raise ValueError('Answer boundary crosses a token; adapter needs explicit boundary handling')
            return length
    raise ValueError('Answer boundary not found')


class HFModel:
    def __init__(self, model_id, revision, rank, max_tokens, dtype='bfloat16', trust_remote_code=False, quantization='none', trial_operator='merged-resident', ctx_size=None, edit_targets='writeback', merged_cache_bytes=0):
        self.device = require_b70()
        if dtype not in {'bfloat16','float16'} or max_tokens <= 0:
            raise ValueError('Invalid B70 dtype or generation budget')
        self.config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=trust_remote_code)
        if self.config.model_type not in ARCHITECTURES:
            raise ValueError(f'No architecture adapter for {self.config.model_type}')
        self.boundary_contract = BoundaryContract.for_architecture(self.config.model_type)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=trust_remote_code)
        self.max_tokens = max_tokens
        text_config = getattr(self.config, 'text_config', self.config)
        self.ctx_size = ctx_size if ctx_size is not None else getattr(text_config, 'max_position_embeddings', None)
        self._check_context(1)
        if quantization not in {'none', 'nf4'}:
            raise ValueError('Unsupported search quantization')
        kwargs = {}
        if quantization == 'nf4':
            kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=getattr(torch,dtype), bnb_4bit_use_double_quant=True)
        factory = AutoModelForImageTextToText if hasattr(self.config, 'vision_config') else AutoModelForCausalLM
        self._source_context = compatible_source(model_id)
        prepared_source = self._source_context.__enter__()
        try:
            self.model = factory.from_pretrained(str(prepared_source), revision=revision, dtype=getattr(torch,dtype), device_map={'': self.device}, trust_remote_code=trust_remote_code, attn_implementation='eager', **kwargs).eval()
        except BaseException:
            self._source_context.__exit__(None,None,None)
            raise
        if any(p.device.type != 'xpu' for p in self.model.parameters()):
            raise RuntimeError('Model weights escaped the B70')
        if 'logits_to_keep' not in inspect.signature(self.model.forward).parameters:
            raise ValueError('Architecture needs chunked-logits adapter for full-reply KL')
        self.editor = Editor(self.model, rank, operator=trial_operator, edit_targets=edit_targets,merged_cache_bytes=merged_cache_bytes)
        self.max_tokens = max_tokens
        self.final_marker = None
        text_config = getattr(self.config, 'text_config', self.config)
        self.fingerprint = {'model':model_id,'revision':revision,'architecture':self.config.model_type,'layers':len(self.editor.layers),'hidden_size':text_config.hidden_size,'rank':rank,'dtype':dtype,'quantization':quantization,
                            'trial_operator': trial_operator, 'edit_targets': self.editor.edit_targets,
                            'edit_groups': self.editor.groups, 'boundary': self.boundary_contract.identity(),
                            'generation_policy': 'final-answer-only-v1', 'chat_template_kwargs': {'enable_thinking': False},
                            'template_digest': digest(self.tokenizer.chat_template),
                            'generation_config': self.model.generation_config.to_dict()}

    @classmethod
    def from_objects(cls, model, tokenizer, editor, max_tokens, ctx_size=None):
        self = cls.__new__(cls)
        self.device, self.model, self.tokenizer = next(model.parameters()).device, model, tokenizer
        self.editor, self.max_tokens, self.final_marker = editor, max_tokens, None
        self.config = model.config
        self.ctx_size = ctx_size if ctx_size is not None else getattr(getattr(model.config,'text_config',model.config),'max_position_embeddings',None)
        self.boundary_contract = BoundaryContract.for_architecture(model.config.model_type)
        return self

    def _render(self, prompt):
        return render_chat(self.tokenizer,prompt)

    def _check_context(self, prompt_tokens):
        if type(self.ctx_size) is not int or type(self.max_tokens) is not int or not 0 < self.max_tokens < self.ctx_size:
            raise ValueError('Explicit valid context and output budgets required')
        limit = getattr(getattr(self.config,'text_config',self.config),'max_position_embeddings',None)
        effective = min(self.ctx_size,limit) if type(limit) is int and limit > 0 else self.ctx_size
        if type(prompt_tokens) is not int or prompt_tokens < 1 or prompt_tokens+self.max_tokens > effective:
            raise ValueError(f'Prompt plus output reserve exceeds context: {prompt_tokens} + {self.max_tokens} > {effective}')
        return effective

    def preflight(self, suite):
        counts = {}
        for task in suite:
            if task['id'] in counts:
                raise ValueError('Duplicate prompt IDs in token-budget preflight')
            n = len(self.tokenizer.encode(self._render(task['prompt']),add_special_tokens=False))
            self._check_context(n)
            counts[task['id']] = n
        return {'prompt_tokens':counts,'output_reserve':self.max_tokens,'ctx_size':self._check_context(1),
                'template_digest':digest(self.tokenizer.chat_template), 'chat_template_kwargs':{'enable_thinking':False}}

    def _encode_cpu(self, prompt):
        text = self._render(prompt)
        if self.boundary_contract.mode=='harmony':
            marker=next((m for m in self.boundary_contract.FINAL_MARKERS if m.split('final')[0] in text),None)
            if marker is None:
                raise ValueError('Unverified Harmony template token spelling')
            self.final_marker=None if text.rstrip().endswith(marker) else marker
        thinking_open = text.rfind('<think>') > text.rfind('</think>')
        if thinking_open and self.boundary_contract.mode == 'plain':
            raise ValueError('Chat template contradicts plain boundary adapter')
        inputs = self.tokenizer(text, return_tensors='pt', add_special_tokens=False, return_token_type_ids=False, truncation=False)
        self._check_context(inputs['input_ids'].shape[1])
        return inputs, thinking_open

    def encode(self, prompt):
        inputs,thinking_open=self._encode_cpu(prompt)
        return {key:value.to(self.device) for key,value in inputs.items()},thinking_open

    def prompt_tokens(self,prompt):
        inputs,_=self._encode_cpu(prompt)
        return inputs['input_ids'].shape[1]

    @torch.inference_mode()
    def generate(self, prompt, min_tokens=None):
        return self.generate_batch([prompt],min_tokens=min_tokens)[0]

    @torch.inference_mode()
    def generate_batch(self, prompts, min_tokens=None):
        if not prompts:
            raise ValueError('A generation batch cannot be empty')
        # Validate every rendered row before transferring even the first row.
        prepared=[]
        for prompt in prompts:
            inputs,thinking_open=self._encode_cpu(prompt)
            prepared.append((inputs,thinking_open,self.final_marker))
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        if isinstance(pad,list):
            pad=pad[0]
        if type(pad) is not int:
            raise ValueError('Batched decoder generation needs an explicit padding token')
        width=max(row['input_ids'].shape[1] for row,_,_ in prepared)
        padded={key:torch.cat([torch.nn.functional.pad(row[key],(width-row[key].shape[1],0),
                        value=pad if key=='input_ids' else 0) for row,_,_ in prepared],dim=0).to(self.device)
                for key in ('input_ids','attention_mask')}
        kwargs = {}
        floor_enforced = False
        if min_tokens is not None:
            if type(min_tokens) is not int or not 0 < min_tokens <= self.max_tokens:
                raise ValueError('min_tokens must be a positive integer within the output budget')
            # Real positional floor: the stop token is suppressed for the first
            # min_tokens, so a delayed refusal still appears inside the window.
            kwargs['min_new_tokens'] = min_tokens
            floor_enforced = True
        sequence = self.model.generate(**padded, max_new_tokens=self.max_tokens, do_sample=False, pad_token_id=pad, **kwargs)
        if sequence.shape[0]!=len(prepared):
            raise ValueError('Model returned the wrong batch cardinality')
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos,list) else [eos]
        results=[]
        for index,(inputs,thinking_open,marker) in enumerate(prepared):
            generated=sequence[index,width:].tolist()
            stop=next((i for i,token in enumerate(generated) if token in eos),None)
            stopped=stop is not None
            generated=generated[:stop+1] if stopped else generated
            trimmed=generated[:-1] if stopped else generated
            if stopped:
                self.boundary_contract.boundary(self.tokenizer,trimmed,thinking_open,marker)
            n=inputs['input_ids'].shape[1]
            # Do not persist left padding or the finished rows' trailing pads:
            # replay/KL trajectories must still be a single unpadded sequence.
            unpadded=torch.cat([inputs['input_ids'].to(self.device),sequence[index:index+1,width:width+len(generated)]],dim=1)
            results.append({'content':self.tokenizer.decode(trimmed,skip_special_tokens=False),
                'finish_reason':'stop' if stopped else 'length','completion_tokens':len(generated),
                'thinking_open':thinking_open,'final_marker':marker,'sequence':unpadded,
                'prompt_length':n,'generated_ids':trimmed,'floor_enforced':floor_enforced})
        return results

    @torch.inference_mode()
    def residual_at_answer(self, prompt, response=None):
        response = response if response is not None else self.generate(prompt)
        boundary = self.boundary_contract.boundary(self.tokenizer, response['generated_ids'], response['thinking_open'],response.get('final_marker'))
        if len(response['generated_ids']) <= boundary:
            raise ValueError('No generated answer after thinking')
        prefix = response['sequence'][:, :response['prompt_length'] + boundary]
        captured, handles = [], []
        for layer in self.editor.layers:
            def hook(module, args, output):
                hidden = output[0] if isinstance(output,tuple) else output
                captured.append(residual_vector(hidden,self.config).detach().float().clone())
            handles.append(layer.register_forward_hook(hook))
        try:
            self.model(prefix, use_cache=False, logits_to_keep=1)
        finally:
            for handle in handles:
                handle.remove()
        if len(captured) != len(self.editor.layers):
            raise ValueError('Residual hooks did not cover every layer')
        return torch.stack(captured)

    @torch.inference_mode()
    def module_io_at_answer(self, prompt, response):
        boundary=self.boundary_contract.boundary(self.tokenizer,response['generated_ids'],response['thinking_open'],response.get('final_marker'))
        prefix=response['sequence'][:,:response['prompt_length']+boundary]
        captured,handles={},[]
        for *_,adapter in self.editor.targets:
            if adapter.base.weight.ndim!=2:
                raise ValueError('ARA requires a verified fused-expert route-I/O adapter')
            def hook(module,args,output,key=adapter.source_key):
                captured[key]=(args[0][0,-1].detach().float().cpu(),output[0,-1].detach().float().cpu())
            handles.append(adapter.register_forward_hook(hook))
        try:
            self.model(prefix,use_cache=False,logits_to_keep=1)
        finally:
            for handle in handles:
                handle.remove()
        return captured

    @torch.inference_mode()
    def trajectory_kl(self, sequence, prompt_length, answer_start, chunk_size=32):
        total = sequence.shape[1] - prompt_length
        if total <= 0 or not 0 <= answer_start < total or chunk_size <= 0:
            raise ValueError('Empty reply or invalid KL trajectory')
        enabled = [a.enabled for *_,a in self.editor.targets]
        values = []
        def forward(active, ids, cache=None, keep=0):
            for configured, (*_, adapter) in zip(enabled, self.editor.targets):
                adapter.enabled = configured if active else False
            return self.model(ids, past_key_values=cache, use_cache=True, logits_to_keep=keep)
        try:
            # Two independent KV/recurrent caches: never re-prefill growing prefixes.
            prompt = sequence[:, :prompt_length]
            base_output = forward(False, prompt, keep=1)
            candidate_output = forward(True, prompt, keep=1)
            base_cache, candidate_cache = base_output.past_key_values, candidate_output.past_key_values
            if base_cache is None or candidate_cache is None or base_cache is candidate_cache:
                raise ValueError('Model needs an independent-cache adapter for full-reply KL')
            values.append(token_kl(base_output.logits[0], candidate_output.logits[0]))
            del base_output, candidate_output
            # Input reply token i predicts reply token i+1. The EOS is scored too.
            for start in range(0, total-1, chunk_size):
                stop = min(total-1, start+chunk_size)
                ids = sequence[:, prompt_length+start:prompt_length+stop]
                base_output = forward(False, ids, base_cache)
                candidate_output = forward(True, ids, candidate_cache)
                base_cache, candidate_cache = base_output.past_key_values, candidate_output.past_key_values
                values.append(token_kl(base_output.logits[0], candidate_output.logits[0]))
                del base_output, candidate_output
        finally:
            for active,(*_,adapter) in zip(enabled,self.editor.targets):
                adapter.enabled = active
        values = torch.cat(values)
        return {'full_reply_kl':float(values.mean()), 'answer_kl':float(values[answer_start:].mean()), 'tokens':total, 'answer_tokens':total-answer_start}

    def release(self):
        self.editor = None
        self.model = None
        import gc
        gc.collect()
        torch.xpu.empty_cache()
        if hasattr(self, '_source_context'):
            self._source_context.__exit__(None,None,None)


def residual_vector(hidden, config):
    text_config=getattr(config,'text_config',config)
    width=text_config.hidden_size
    vector=hidden[0,-1]
    if config.model_type in {'qwen4_exp','qwen4_exp_text'}:
        streams=getattr(text_config,'hc_count',None)
        if type(streams) is not int or streams<2 or vector.numel()!=streams*width:
            raise ValueError('Qwen4 hyperconnection residual width differs from the explicit adapter')
        vector=vector.reshape(streams,width).mean(0)
    if vector.shape!=(width,):
        raise ValueError('Residual width needs an explicit architecture adapter')
    return vector


def preflight_hf(model_id, revision, suites, ctx_size, max_tokens, trust_remote_code=False):
    """Only configuration/tokenizer metadata; no model construction or device access."""
    probe=HFModel.__new__(HFModel)
    probe.config=AutoConfig.from_pretrained(model_id,revision=revision,trust_remote_code=trust_remote_code)
    probe.tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,trust_remote_code=trust_remote_code)
    probe.ctx_size,probe.max_tokens=ctx_size,max_tokens
    return {name:probe.preflight(tasks) for name,tasks in suites.items()}
