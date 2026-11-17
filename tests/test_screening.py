"""Screening configuration, statistics and run-config surface (CPU fixtures)."""
import math
import pytest

from jdewitt.search import RunConfig
from jdewitt.screening import validate_screening, margin_differential


def good_spec(**overrides):
    spec = {'refusal_tokens': [5, 9], 'reference_tokens': [11, 13], 'min_margin_drop': 0.25}
    spec.update(overrides)
    return spec


def test_validate_screening_disabled_by_default():
    assert validate_screening(None) is None


def test_validate_screening_accepts_explicit_spec():
    assert validate_screening(good_spec()) == good_spec()


@pytest.mark.parametrize('bad', [
    'not-a-dict',
    'missing-key', 'extra-key',
    'empty-refusal', 'duplicate-tokens', 'negative-token', 'nonint-token',
    'overlapping-tokens', 'zero-drop', 'negative-drop', 'nan-drop', 'string-drop'])
def test_validate_screening_rejects(bad):
    if bad == 'not-a-dict':
        spec, match = [1, 2, 3], 'exactly'
    elif bad == 'missing-key':
        spec = {'refusal_tokens': [5], 'min_margin_drop': 1.}
        match = 'exactly'
    elif bad == 'extra-key':
        spec = good_spec(method_options={})
        match = 'exactly'
    elif bad == 'empty-refusal':
        spec, match = good_spec(refusal_tokens=[]), 'nonempty'
    elif bad == 'duplicate-tokens':
        spec, match = good_spec(reference_tokens=[11, 11]), 'distinct'
    elif bad == 'negative-token':
        spec, match = good_spec(refusal_tokens=[-1]), 'nonnegative'
    elif bad == 'nonint-token':
        spec, match = good_spec(refusal_tokens=['5']), 'nonnegative'
    elif bad == 'overlapping-tokens':
        spec, match = good_spec(reference_tokens=[5]), 'disjoint'
    elif bad == 'zero-drop':
        spec, match = good_spec(min_margin_drop=0.), 'positive'
    elif bad == 'negative-drop':
        spec, match = good_spec(min_margin_drop=-0.1), 'positive'
    elif bad == 'nan-drop':
        spec, match = good_spec(min_margin_drop=math.nan), 'finite'
    else:
        spec, match = good_spec(min_margin_drop='0.1'), 'finite'
    with pytest.raises(ValueError, match=match):
        validate_screening(spec)


def test_margin_differential_subtracts_harmless_drift():
    # Harmful margins fall by 1.0; harmless also fall by 1.0 -> no differential.
    assert margin_differential([2., 4.], [1., 3.], [0.5, 0.7], [-0.5, -0.3]) == pytest.approx(0.)
    assert margin_differential([2., 4.], [0., 2.], [0.5, 0.7], [0.5, 0.7]) == pytest.approx(2.)


@pytest.mark.parametrize('args', [([], []), ([1.], []), ([1.], [1., 2.])])
def test_margin_differential_requires_equal_nonempty_records(args):
    with pytest.raises(ValueError, match='equal nonempty'):
        margin_differential(args[0], args[1], [1.], [1.])


def base_config(**overrides):
    return {'model': 'm', 'revision': 'r', 'model_license': 'apache-2.0', 'directions': 'd',
            'selection': 's', 'holdout': 'h', 'max_tokens': 128, 'ctx_size': 512, 'trials': 1,
            'startup_trials': 1, 'finalists': 1, 'rank': 1, 'seed': 0, 'max_full_reply_kl': .1,
            'max_capability_drop': .1, 'vram_headroom_gib': 1., 'converter': 'c', 'quantizer': 'q',
            'converter_python': 'p', 'llama_server': 'l', 'sandbox_image': 'i', 'output': 'o',
            'scoring_mode': 'refusal-check', **overrides}


def test_run_config_carries_explicit_screening():
    config = RunConfig.from_json(base_config(screening=good_spec()))
    assert config.screening == good_spec()
    assert RunConfig.from_json(base_config()).screening is None


@pytest.mark.parametrize('spec', [{'refusal_tokens': [5]}, {'refusal_tokens': [5], 'reference_tokens': [5],
                                   'min_margin_drop': 1.}, {'refusal_tokens': [5], 'reference_tokens': [7],
                                                              'min_margin_drop': 0}])
def test_run_config_rejects_malformed_screening(spec):
    with pytest.raises(ValueError):
        RunConfig.from_json(base_config(screening=spec))
