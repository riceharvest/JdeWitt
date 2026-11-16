"""Architecture-definition checks, not full checkpoint/B70 certification."""
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from b70_ablator.editor import Editor, discover_targets


@pytest.mark.parametrize('name', ['qwen3_moe', 'qwen3_next', 'qwen3_5_moe_text', 'qwen4_exp_text',
                                  'glm_moe_dsa', 'glm5_next_text', 'gemma4_text', 'gemma4_unified_text',
                                  'muse_glimmer_text', 'gpt_oss', 'deepseek_v4', 'lfm2', 'lfm2_moe',
                                  'olmo3', 'minimax_m3_vl_text', 'step3p5', 'mistral4',
                                  'muse_spark_text','hy_v3','hy_v4','bailing_hybrid','mimo_v2_flash'])
def test_real_transformers_definitions_have_editable_targets(name):
    from transformers import AutoConfig, AutoModel
    config = AutoConfig.for_model(name, vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4 if name=='glm5_next_text' else 2, head_dim=16,
        global_head_dim=16, num_local_experts=4, num_experts=4, n_routed_experts=4,
        num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=64,
        n_shared_experts=1, first_k_dense_replace=1, num_dense_layers=1,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=4, layer_types=['sliding_attention']*2 if name=='deepseek_v4' else
        ['linear_attention', 'full_attention'] if name in
        {'qwen3_next','qwen3_5_moe_text','qwen4_exp_text','glm5_next_text'} else ['full_attention']*2,
        num_kv_shared_layers=0, num_kv_shared_layers_global=0, q_lora_rank=16, kv_lora_rank=16,
        qk_nope_head_dim=8, qk_rope_head_dim=0 if name=='glm5_next_text' else 8, v_head_dim=16, o_lora_rank=16, o_groups=2,
        partial_rotary_factor=1., conv_dim=64, conv_L_cache=3, block_dim=64,
        dense_intermediate_size=128, shared_intermediate_size=64, num_nextn_predict_layers=0,
        indexer_n_heads=4,indexer_kv_heads=1,indexer_head_dim=16,indexer_budget=8,indexer_compress_ratio=2,ple_embed_dim=64,
        ngram_vocab_size_base=128,bos_token_id=1,eos_token_id=2,pad_token_id=0)
    # Meta construction uses the actual implementation without allocating large optional defaults.
    if name=='qwen4_exp_text':
        config.layer_types=['linear_attention','indexed_attention']
    with torch.device('meta'):
        if name=='step3p5':
            from transformers.models.step3p7.modeling_step3p7 import Step3p7TextModel
            model=Step3p7TextModel(config)
        else:
            model = AutoModel.from_config(config)
    layers, targets = discover_targets(model)
    assert len(layers) == 2 and targets
    assert all('visual' not in t[2] and 'vision' not in t[2] and 'mtp' not in t[2] for t in targets)


@pytest.mark.parametrize('transposed', [False, True])
def test_fused_experts_are_edited_without_mutating_original_storage(transposed):
    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            shape = (3, 5, 4) if transposed else (3, 4, 5)
            self.down_proj = nn.Parameter(torch.randn(shape))
        def forward(self, x):
            w = self.down_proj if transposed else self.down_proj.transpose(-1, -2)
            return torch.einsum('ei,eio->eo', x, w)
    model = nn.Module()
    model.config = SimpleNamespace(model_type='gpt_oss' if transposed else 'qwen3_moe', hidden_size=4)
    model.model = nn.Module()
    block = nn.Module()
    block.self_attn = nn.Module()
    block.self_attn.o_proj = nn.Linear(4, 4, bias=False)
    block.mlp = nn.Module()
    block.mlp.experts = Experts()
    model.model.layers = nn.ModuleList([block])
    experts = block.mlp.experts
    original = experts.down_proj.detach().clone()
    pointer = experts.down_proj.data_ptr()
    x = torch.randn(3, 5)
    baseline = experts(x)
    editor = Editor(model, 1)
    directions = torch.tensor([[[1., 0., 0., 0.]]])
    params = {k: {'strength': .7, 'center': .5, 'width': 1., 'floor': 1.} for k in ('attn','mlp')}
    editor.apply(directions, params)
    edited = experts(x)
    assert not torch.equal(edited, baseline)
    assert torch.allclose(edited[:, 0], baseline[:, 0] * .3, atol=1e-5)
    assert torch.equal(edited[:, 1:], baseline[:, 1:])
    adapter = next(a for *_, a in editor.targets if a.source_path.endswith('experts.down_proj'))
    assert torch.equal(adapter.base.weight, original) and adapter.base.weight.data_ptr() == pointer
    editor.reset()
    assert torch.equal(experts(x), baseline)


def test_systemless_templates_fold_system_content_without_hiding_other_errors():
    from b70_ablator.chat import render_chat
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            if messages[0]['role'] == 'system':
                from jinja2.exceptions import TemplateError
                raise TemplateError('System role not supported')
            return messages[0]['content']
    assert render_chat(Tokenizer(), 'hello', 'rules') == 'rules\n\nhello'
    class Broken(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            raise RuntimeError('unrelated failure')
    with pytest.raises(RuntimeError, match='unrelated'):
        render_chat(Broken(), 'hello', 'rules')


@pytest.mark.parametrize('latent',[None,8])
@pytest.mark.parametrize('wrapper',[False,True])
def test_nemotron_native_blocks_and_latent_experts_use_residual_coordinates(latent,wrapper):
    from transformers import AutoConfig,AutoModel
    config=AutoConfig.for_model('nemotron_h',vocab_size=32,hidden_size=16,intermediate_size=32,
        layers_block_type=['linear_attention','moe','full_attention','mlp'],
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,mamba_num_heads=4,mamba_head_dim=8,
        n_groups=1,ssm_state_size=4,n_routed_experts=2,num_experts_per_tok=1,
        moe_latent_size=latent,moe_intermediate_size=16,moe_shared_expert_intermediate_size=16)
    with torch.device('meta'):
        model=AutoModel.from_config(config)
    if wrapper:
        outer=nn.Module()
        outer.config=SimpleNamespace(model_type='nemotron_h_omni',text_config=config)
        outer.language_model=nn.Module()
        outer.language_model.model=model
        model=outer
    layers,targets=discover_targets(model)
    assert len(layers)==4
    assert {t[0] for t in targets}==set(range(4))
    paths=[t[3] for t in targets]
    assert any(p.endswith('mixer.shared_experts.down_proj') for p in paths)
    assert any(p.endswith('mixer.fc2_latent_proj' if latent else 'mixer.experts.down_proj') for p in paths)
    if latent:
        assert not any(p.endswith('mixer.experts.down_proj') for p in paths)


def test_minicpm_v47_native_wrapper_edits_only_qwen_text_projections():
    from transformers import AutoConfig,AutoModelForImageTextToText
    config=AutoConfig.for_model('minicpmv4_7',
        text_config={'model_type':'qwen3_5_text','vocab_size':128,'hidden_size':32,'intermediate_size':64,
            'num_hidden_layers':2,'num_attention_heads':2,'num_key_value_heads':1,'head_dim':16,
            'linear_key_head_dim':8,'linear_value_head_dim':8,'linear_num_key_heads':2,'linear_num_value_heads':2,
            'layer_types':['linear_attention','full_attention']},
        vision_config={'model_type':'minicpmv4_7_vision','hidden_size':32,'intermediate_size':64,
            'num_hidden_layers':2,'num_attention_heads':2,'image_size':28,'patch_size':14},insert_layer_id=1)
    with torch.device('meta'):
        model=AutoModelForImageTextToText.from_config(config)
    layers,targets=discover_targets(model)
    assert len(layers)==2 and len(targets)==4
    assert all(path.startswith('model.language_model.layers.') for _,_,_,path,_ in targets)
    assert any(path.endswith('linear_attn.out_proj') for _,_,_,path,_ in targets)
    assert all('vision_tower' not in path and 'merger' not in path for _,_,_,path,_ in targets)
