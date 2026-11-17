# SPDX-License-Identifier: AGPL-3.0-or-later
"""Explicit text-backbone layouts. Registration is not checkpoint certification."""

# The largest strength any plan may request. Published near-zero-refusal releases
# sit well above the old 1.5 search ceiling, so this bound must not be the thing
# that caps a run; the search range is a separate, explicit choice.
MAX_STRENGTH = 6.
EDIT_TARGETS = ('writeback', 'full')
COMMON = ('self_attn.o_proj', 'self_attn.out_proj', 'self_attn.o_b_proj',
          'linear_attn.out_proj', 'conv.out_proj', 'mixer.o_proj', 'mixer.out_proj')
# Input-side (read) projections. These consume the residual stream, so their
# in_features is the residual width even when out_features is not (grouped-query
# attention makes k_proj/v_proj narrower). Editing only the write-back matrices
# leaves the read of the refusal direction in place.
COMMON_IN = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
             'attn.q_proj', 'attn.k_proj', 'attn.v_proj',
             'linear_attn.q_proj', 'linear_attn.k_proj', 'linear_attn.v_proj',
             'mixer.q_proj', 'mixer.k_proj', 'mixer.v_proj')
MLP_IN = ('mlp.gate_proj', 'mlp.up_proj', 'feed_forward.w1', 'feed_forward.w3',
          'mixer.gate_proj', 'mixer.up_proj')
MLP = ('mlp.down_proj', 'feed_forward.w2', 'mlp.experts.*.down_proj',
       'mlp.experts.down_proj', 'mlp.shared_expert.down_proj', 'mlp.shared_experts.down_proj',
       'experts.down_proj', 'feed_forward.experts.*.w2', 'feed_forward.experts.down_proj',
       'block_sparse_moe.experts.*.w2', 'shared_mlp.output_linear', 'moe.experts.*.output_linear',
       'mixer.down_proj', 'mixer.experts.down_proj', 'mixer.shared_experts.down_proj')
LAYERS = ('model.language_model.layers', 'language_model.model.layers', 'language_model.layers',
          'model.layers', 'backbone.layers', 'layers')
# These model_type names and paths are from installed Transformers definitions,
# not marketing-version guesses. VLM editing is text-backbone only.
NAMES = ('llama mistral ministral ministral3 qwen2 qwen2_moe qwen3 qwen3_moe qwen3_next '
         'qwen3_5 qwen3_5_text qwen3_5_moe qwen3_5_moe_text qwen3_vl qwen3_vl_text '
         'qwen3_vl_moe qwen3_vl_moe_text qwen4_exp qwen4_exp_text '
         'gemma gemma2 gemma3 gemma3_text gemma4 gemma4_text gemma4_unified gemma4_unified_text '
         'muse_glimmer muse_glimmer_text glm4_moe glm4_moe_lite glm_moe_dsa glm5_next glm5_next_text '
         'deepseek_v2 deepseek_v3 deepseek_v32 deepseek_v4 '
         'lfm2 lfm2_moe lfm2_vl nemotron_h nemotron_h_omni gpt_oss olmo2 olmo3 seed_oss smollm3 '
         'minimax_m2 minimax_m3_vl minimax_m3_vl_text step3p7 step3p5 mistral4 spark2_5 '
         'muse_spark muse_spark_text hy_v3 hy_v4 bailing_hybrid mimo_v2_flash minicpmv4_7').split()
ARCHITECTURES = {name: {'layers': LAYERS, 'attn': COMMON, 'mlp': MLP,
                        'attn_in': COMMON_IN, 'mlp_in': MLP_IN,
                        'fused_transposed': name == 'gpt_oss'} for name in NAMES}
# Muse Spark experts operate in a bottleneck, not residual coordinates.
# Edit their aggregate output only after its explicit residual-space projection.
for name in ('muse_spark','muse_spark_text'):
    ARCHITECTURES[name]={**ARCHITECTURES[name],'mlp':('mlp.down_proj','mlp.post_expert_proj')}
for name in ('nemotron_h','nemotron_h_omni'):
    ARCHITECTURES[name]={**ARCHITECTURES[name],
        'mlp_bottleneck_output':'mixer.fc2_latent_proj',
        'mlp_bottleneck_targets':('mixer.fc2_latent_proj','mixer.shared_experts.down_proj')}
# Spark2.5 (and the X2.6 line) fuse Q, K and V into one read-side matrix and gate each
# head separately: the generic self_attn.q_proj/k_proj/v_proj do not exist in its
# checkpoints, so the read side has to name the fused tensor or it would be skipped.
# The headwise g_proj (out_features == num_heads) is not a residual-width target and is
# deliberately left unedited.
ARCHITECTURES['spark2_5'] = {**ARCHITECTURES['spark2_5'], 'attn_in': ('self_attn.q_k_v_proj',)}


def architecture_catalog():
    import json
    from importlib.resources import files
    inventory=json.loads(files('b70_ablator').joinpath('assets/architecture-inventory.json').read_text())
    return {'scope': 'Text-backbone adapter layouts; not proof of XPU kernels, model fit, chat boundaries or GGUF conversion.',
            'recent_releases':inventory,
            'architectures': {name: {'adapter': True, 'full_checkpoint_b70_verified': False,
                                    'fused_experts': True, 'layers': list(spec['layers'])}
                              for name, spec in sorted(ARCHITECTURES.items())}}
