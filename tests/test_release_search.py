
from pathlib import Path
import pytest
from jdewitt.release import rank_quantized, verify_quant, export_gguf
from jdewitt.search import build_study, compatible_warm_starts, RunConfig


def result(rate, capability=1., invalid=0):
    return {'noncompliance_rate': rate, 'capability_pass_rate': capability, 'invalid': invalid, 'mean_preamble_chars': 0., 'disclaimer_rate': 0., 'full_reply_kl': .01}


def test_final_ranking_uses_worst_actual_quant_not_bf16():
    baseline = {'Q4_K_M': result(.8), 'Q5_K_M': result(.8)}
    candidates = [
        {'id': 'bf16_winner_quant_loser', 'hf': result(0), 'quants': {'Q4_K_M': result(.6), 'Q5_K_M': result(.1)}},
        {'id': 'quant_winner', 'hf': result(.2), 'quants': {'Q4_K_M': result(.2), 'Q5_K_M': result(.2)}},
    ]
    ranked = rank_quantized(candidates, baseline, max_capability_drop=0)
    assert ranked[0]['id'] == 'quant_winner'


def test_missing_quant_or_damage_or_invalid_never_release():
    baseline = {'Q4_K_M': result(.8), 'Q5_K_M': result(.8)}
    candidates = [
        {'id':'missing', 'quants':{'Q4_K_M':result(0)}},
        {'id':'damage', 'quants':{'Q4_K_M':result(0,.5), 'Q5_K_M':result(0)}},
        {'id':'invalid', 'quants':{'Q4_K_M':result(0,invalid=1),'Q5_K_M':result(0)}},
    ]
    assert rank_quantized(candidates, baseline, max_capability_drop=.05) == []


def test_warm_starts_use_only_related_architecture_and_real_results():
    fingerprint = {'architecture':'qwen3', 'rank':1, 'dtype':'bfloat16'}
    records = [
        {'fingerprint':fingerprint, 'feasible':True,'parameters':{'attn.strength':1.}},
        {'fingerprint':{'architecture':'llama','rank':1,'dtype':'bfloat16'},'feasible':True,'parameters':{'attn.strength':.5}},
        {'fingerprint':fingerprint,'feasible':False,'parameters':{'attn.strength':2.}},
    ]
    assert compatible_warm_starts(records,fingerprint) == [{'attn.strength':1.}]


def test_resume_rejects_changed_eval_or_judge(tmp_path):
    p = tmp_path / 'study.db'
    build_study(p, {'suite':'a','judge':'x'}, seed=7,startup_trials=2)
    build_study(p, {'suite':'a','judge':'x'}, seed=7,startup_trials=2)
    with pytest.raises(ValueError):
        build_study(p, {'suite':'b','judge':'x'}, seed=7,startup_trials=2)


def test_config_rejects_guessed_or_unbounded_limits():
    with pytest.raises(ValueError):
        RunConfig.from_json({})


def test_consistency_ranking_cannot_ignore_paired_failure():
    baseline = {'Q4_K_M': result(.8), 'Q5_K_M': result(.8)}
    candidate = {'id': 'aggregate-only', 'quants': {q: {**result(0), 'paired': {'passes': False}} for q in baseline}}
    assert rank_quantized([candidate], baseline, 0, require_consistency=True) == []
    for report in candidate['quants'].values():
        report['paired']['passes'] = True
    assert rank_quantized([candidate], baseline, 0, require_consistency=True) == [candidate]
    del candidate['quants']['Q4_K_M']['paired']
    assert rank_quantized([candidate], baseline, 0, require_consistency=True) == []


def test_mislabeled_or_non_gguf_cannot_be_ranked(tmp_path):
    p = tmp_path / 'model-Q4_K_M.gguf'
    p.write_text('not a GGUF')
    with pytest.raises(ValueError):
        verify_quant(p, 'Q4_K_M')


def test_export_checks_tool_failures_and_no_fake_artifacts(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        export_gguf(tmp_path / 'hf', tmp_path / 'out', converter=Path('/nonexistent'), quantizer=Path('/nonexistent'), python=Path('/nonexistent'))
