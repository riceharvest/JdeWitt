"""Data-driven per-layer strength, and the explicit strength range."""
import math
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from jdewitt.editor import Editor, MAX_STRENGTH
from jdewitt.errors import DirectionRejected
from jdewitt.methods import (LAYER_PROFILES, layer_discriminability, method_catalog,
                                 validate_method_options)
from jdewitt.plans import fit_plan
from jdewitt.scoring import ConsistencyPolicy
from jdewitt.search import RunConfig, suggest_parameters


def records(signal=(10., .01)):
    """Two matched families, two languages, one layer per requested signal."""
    rows = []
    for language in ('en', 'nl'):
        for family in ('f1', 'f2'):
            for label in ('good', 'bad'):
                value = torch.zeros(2, 4)
                if label == 'bad':
                    value[0, 0] = signal[0]
                    value[1, 0] = signal[1]
                rows.append(({'language': language, 'family_id': family, 'label': label}, value))
    return rows


def test_discriminability_follows_the_per_layer_separation():
    profile = layer_discriminability(records(), {})
    assert tuple(profile.shape) == (2,)
    assert float(profile.max()) == 1.
    assert profile[0] > profile[1]
    assert 0 <= float(profile.min())


def test_a_layer_that_does_not_separate_scores_near_zero():
    profile = layer_discriminability(records(signal=(10., 0.)), {})
    assert float(profile[1]) < 1e-6
    assert float(profile[0]) == 1.


def test_degenerate_separation_is_rejected():
    with pytest.raises(DirectionRejected, match='discriminability'):
        layer_discriminability(records(signal=(0., 0.)), {})


def test_layer_profile_option_is_validated_per_method():
    assert LAYER_PROFILES == ('searched', 'discriminative')
    validate_method_options('directional', {'layer_profile': 'discriminative'})
    with pytest.raises(ValueError, match='layer_profile'):
        validate_method_options('directional', {'layer_profile': 'clever'})
    # LEACE and ARA carry their own factors and never consume a layer profile.
    for method, required in (('leace', {'ridge': .1}),
                             ('ara', {'steps': 1, 'learning_rate': .1, 'neighbors': 1,
                                      'preserve': 1., 'steer': 1., 'overcorrect': 0.,
                                      'repulsion_cap': 1.})):
        with pytest.raises(ValueError, match='does not consume'):
            validate_method_options(method, {'layer_profile': 'discriminative', **required})


def policy():
    return ConsistencyPolicy.from_json({'min_direction_examples': 2, 'min_direction_families': 2,
                                        'min_direction_overlap': 0.01, 'min_refusal_families': 2,
                                        'min_variants_per_family': 2, 'confidence': .9,
                                        'max_task_regressions': 0})


def test_plan_carries_the_measured_profile_only_when_requested():
    tasks = [{'language': t['language'], 'family_id': t['family_id'], 'label': t['label'], 'id': str(i)}
             for i, (t, _) in enumerate(records())]
    plain, _ = fit_plan(records(), 1, policy(), 0, tasks, 'directional', {})
    assert 'layer_profile' not in plain
    measured, _ = fit_plan(records(), 1, policy(), 0, tasks, 'directional',
                           {'layer_profile': 'discriminative'})
    assert tuple(measured['layer_profile'].shape) == (2,)
    assert float(measured['layer_profile'][0]) > float(measured['layer_profile'][1])


def llama():
    torch.manual_seed(3)
    return LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=64, eos_token_id=2, pad_token_id=0)).eval()


def test_profile_replaces_the_searched_shape_instead_of_compounding_it():
    model = llama()
    editor = Editor(model, 1)
    directions = torch.zeros(2, 1, 32)
    directions[:, :, 0] = 1
    plan = {'method': 'directional', 'basis': directions, 'row_normalization': 'none',
            'layer_profile': torch.tensor([1., .25])}
    params = {g: {'strength': 1.} for g in editor.groups}
    editor.apply(plan, params)
    # B magnitude is proportional to the applied strength, so the ratio of the
    # two layers must reproduce the profile rather than the searched profile.
    norms = {}
    for index, group, _, _, _, adapter in editor.targets:
        if group == 'attn':
            norms[index] = float(adapter.B.abs().sum())
    assert norms[1] / norms[0] == pytest.approx(.25, rel=1e-4)


def test_a_profiled_plan_refuses_searched_shape_parameters():
    model = llama()
    editor = Editor(model, 1)
    directions = torch.zeros(2, 1, 32)
    directions[:, :, 0] = 1
    plan = {'method': 'directional', 'basis': directions, 'row_normalization': 'none',
            'layer_profile': torch.tensor([1., 1.])}
    with pytest.raises(ValueError, match='Incomplete edit parameters'):
        editor.apply(plan, {g: {'strength': 1., 'center': .5, 'width': 1., 'floor': 1.}
                            for g in editor.groups})


def test_suggest_parameters_suggests_only_what_the_operator_consumes():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    searched = optuna.create_study().ask()
    params = suggest_parameters(searched, ['attn', 'mlp'])
    assert set(params) == {'attn', 'mlp'}
    assert all(set(v) == {'strength', 'center', 'width', 'floor'} for v in params.values())
    assert all(0 <= v['strength'] <= 1.5 for v in params.values())
    profiled = optuna.create_study().ask()
    params = suggest_parameters(profiled, ['attn', 'mlp'], None, 'discriminative')
    assert all(set(v) == {'strength'} for v in params.values())


def test_strength_range_is_explicit_and_bounded():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trial = optuna.create_study().ask()
    params = suggest_parameters(trial, ['attn'], [1., 6.])
    assert 1. <= params['attn']['strength'] <= 6.
    for bad in ([0., 7.], [2., 1.], [0.], [0., 1., 2.], [-1., 1.], [0., float('inf')], 'wide'):
        with pytest.raises(ValueError, match='strength_range|two-element'):
            suggest_parameters(trial, ['attn'], bad)
    with pytest.raises(ValueError, match='edit group'):
        suggest_parameters(trial, [])


def base_config(**overrides):
    return {'model': 'm', 'revision': 'r', 'model_license': 'apache-2.0', 'directions': 'd',
            'selection': 's', 'holdout': 'h', 'max_tokens': 128, 'ctx_size': 512, 'trials': 1,
            'startup_trials': 1, 'finalists': 1, 'rank': 1, 'seed': 0, 'max_full_reply_kl': .1,
            'max_capability_drop': .1, 'vram_headroom_gib': 1., 'converter': 'c', 'quantizer': 'q',
            'converter_python': 'p', 'llama_server': 'l', 'sandbox_image': 'i', 'output': 'o',
            'scoring_mode': 'refusal-check', **overrides}


def test_catalogue_states_what_is_not_verified():
    catalog = method_catalog()
    assert catalog['edit_surface']['scope']
    assert set(catalog['edit_surface']) >= {'writeback', 'full'}
    assert catalog['layer_profiles']['discriminative']
    assert catalog['strength']['hard_bound'] == MAX_STRENGTH
    assert catalog['strength']['default_search_range'] == [0., 1.5]
    gaps = ' '.join(catalog['not_verified'])
    assert 'real checkpoint' in gaps and 'expert-granular' in gaps
    # Every catalogued method must still declare whether it was run on a real model.
    assert all(m['full_model_b70_verified'] is False for m in catalog['methods'].values())


def test_run_config_accepts_the_new_knobs_and_rejects_bad_values():
    config = RunConfig.from_json(base_config(edit_targets='full', strength_range=[0., 6.]))
    assert config.edit_targets == 'full' and config.strength_range == [0., 6.]
    default = RunConfig.from_json(base_config())
    assert default.edit_targets == 'writeback' and default.strength_range is None
    for bad in ({'edit_targets': 'everything'}, {'strength_range': [0., 7.]},
                {'strength_range': [2., 1.]}, {'strength_range': [0.]}, {'strength_range': 'wide'}):
        with pytest.raises(ValueError, match='edit_targets|strength_range'):
            RunConfig.from_json(base_config(**bad))
    assert MAX_STRENGTH == 6. and math.isfinite(MAX_STRENGTH)
