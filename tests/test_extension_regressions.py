"""Real regression checks; CPU mechanics are not model-quality evidence."""
import json
import subprocess
import sys
from pathlib import Path
import tomllib
import pytest
import torch
from b70_ablator.numerics import stable_directions
from b70_ablator.scoring import ConsistencyPolicy


def test_direction_coverage_keeps_entirely_rejected_languages_visible():
    policy = ConsistencyPolicy(2, 2, .9, 2, 2, .9, 0)
    authored = [{'language': lang, 'label': label, 'family_id': family}
                for lang in ('en', 'nl', 'de') for family in ('a', 'b') for label in ('good', 'bad')]
    records = [(t, torch.tensor([[float(t['label'] == 'bad'), 0., 0.]]))
               for t in authored if t['language'] != 'de']
    with pytest.raises(ValueError, match='de'):
        stable_directions(records, 1, policy, 7, expected_tasks=authored)


def test_help_and_catalog_do_not_import_ml_runtime():
    root = Path(__file__).resolve().parents[1]
    code = """import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in {'torch','transformers','optuna','bitsandbytes'}:
   raise AssertionError('heavy import: '+fullname)
sys.meta_path.insert(0,Block())
from b70_ablator.cli import main
main()
"""
    for args in (['--help'], ['generate', '--help'], ['train','--help'], ['methods'], ['architectures']):
        result = subprocess.run([sys.executable, '-c', code, *args], cwd=root, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        if args[0] in {'methods', 'architectures'}:
            assert isinstance(json.loads(result.stdout), dict)


def test_completed_conversion_binds_mutable_imatrix_contents(tmp_path, monkeypatch):
    from b70_ablator import release
    source = tmp_path / 'hf'
    source.mkdir()
    (source / 'config.json').write_text('{}')
    tool = tmp_path / 'tool'
    tool.write_bytes(b'fixture tool')
    imatrix = tmp_path / 'imatrix'
    imatrix.write_bytes(b'original')
    artifacts = {q: {'path': q, 'sha256': 'fixture'} for q in release.QUANTS}
    monkeypatch.setattr(release, 'export_gguf', lambda *a: artifacts)
    monkeypatch.setattr(release, 'verify_quant', lambda path, q: artifacts[q])
    args = (source, tmp_path / 'quants', tool, tool, tool, imatrix)
    assert release.cached_conversion(*args) == artifacts
    imatrix.write_bytes(b'changed in place')
    with pytest.raises(ValueError, match='identity'):
        release.cached_conversion(*args)


def test_uv_routes_xpu_triton_to_same_index_as_torch():
    config = tomllib.loads((Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text())
    assert config['tool']['uv']['sources']['triton-xpu']['index'] == 'pytorch-xpu'


def test_harmless_refusals_are_scored_instead_of_silently_ignored():
    from b70_ablator.evaluation import score_response
    from b70_ablator.refusal_check import RefusalChecker
    from b70_ablator.scoring import summarize
    task = {'id': 'ordinary', 'kind': 'harmless', 'language': 'en', 'prompt': 'Write a poem about a tree.'}
    row = score_response(task, {'content': 'I cannot help with that request.', 'finish_reason': 'stop'}, RefusalChecker())
    report = summarize([row], [task['id']])
    assert report['harmless_refusals'] == 1
    assert report['harmless_total'] == 1


def test_method_options_fail_before_any_model_is_loaded():
    from b70_ablator.search import RunConfig
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'examples/first-test/run-refusal-check.json').read_text())
    with pytest.raises(ValueError,match='ridge'):
        RunConfig.from_json({**config,'method':'whitened-svd'})
    with pytest.raises(ValueError,match='Unknown'):
        RunConfig.from_json({**config,'method_options':{'typo_option':True}})


def test_direction_checkpoint_identity_binds_method_options_judge_and_policy():
    from b70_ablator.plans import direction_identity
    base=dict(generation={'model':'same'},suite_digest='same',method='svd',options={},
              scoring={'judge':'first'},consistency={'overlap':.9},seed=7)
    original=direction_identity(**base)
    for changed in ({'method':'mpoa'},{'options':{'row_normalization':'full'}},
                    {'scoring':{'judge':'second'}},{'consistency':{'overlap':.5}},{'seed':8}):
        assert direction_identity(**(base|changed))!=original


def test_warmstarts_do_not_cross_method_or_method_options():
    from b70_ablator.search import compatible_warm_starts
    fp={'architecture':'qwen3','rank':1,'dtype':'bfloat16','quantization':'none',
        'method':'svd','method_options_digest':'original'}
    records=[{'fingerprint':fp|change,'feasible':True,'parameters':{'attn.strength':.5}}
             for change in ({'method':'ara'},{'method_options_digest':'different'})]
    assert compatible_warm_starts(records,fp)==[]


@pytest.mark.parametrize('changed',[{'method':[]},{'dtype':{}},{'scoring_mode':[]},{'model':7}])
def test_run_config_rejects_wrong_setting_types_as_validation_errors(changed):
    from b70_ablator.search import RunConfig
    config=json.loads((Path(__file__).resolve().parents[1]/'examples/first-test/run-refusal-check.json').read_text())
    with pytest.raises(ValueError):
        RunConfig.from_json(config|changed)
