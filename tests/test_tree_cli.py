"""CLI and interrupted-job fixtures, not model-quality evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def config_fixture(tmp_path):
    identity = {'model': 'fixture-only', 'revision': 'fixture-v1', 'architecture': 'qwen3',
                'layers': 3, 'hidden_size': 2}
    lens = tmp_path / 'lens.pt'
    torch.save({'J': {0: torch.eye(2), 1: torch.eye(2), 2: torch.eye(2)},
                'd_model': 2, 'source_layers': [0, 1, 2], 'n_prompts': 10}, lens)
    manifest = {'format': 'upstream-jlens-v1', 'lens_file': 'lens.pt',
                'sha256': hashlib.sha256(lens.read_bytes()).hexdigest(), 'model_identity': identity,
                'source_layers': [0, 1, 2], 'n_prompts': 10}
    (tmp_path / 'lens.json').write_text(json.dumps(manifest))
    task = {'id': 'fixture', 'prompt': 'fixture prompt', 'kind': 'refusal', 'language': 'en'}
    (tmp_path / 'suite.jsonl').write_text(json.dumps(task) + '\n')
    config = {'model_identity': identity, 'dtype': 'bfloat16', 'lens_manifest': 'lens.json',
              'suite': 'suite.jsonl', 'suite_role': 'development', 'output': 'out', 'ctx_size': 64,
              'max_tokens': 8, 'refusal_floor': 4, 'checkpoints': [0, 1, 2], 'beam_width': 2,
              'finalists': 1, 'max_expansions': 30, 'max_state_bytes': 10000, 'max_lens_bytes': 10000,
              'strengths': [.5, 1.], 'refusal_tokens': [0], 'reference_tokens': [1],
              'protected_tokens': [2, 3], 'actions': {'narration': [0], 'other': [2]},
              'max_protected_kl': 100., 'max_norm_change': 100., 'drift_weight': .1}
    path = tmp_path / 'tree.json'
    path.write_text(json.dumps(config))
    return path, config, task


def invoke(*args):
    code = """import sys, importlib.abc
class NoModels(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in {'torch','transformers','bitsandbytes'}:
   raise AssertionError('model runtime imported: '+fullname)
sys.meta_path.insert(0, NoModels())
from b70_ablator.cli import main
main()
"""
    return subprocess.run([sys.executable, '-c', code, *map(str, args)], cwd=ROOT, capture_output=True, text=True)


def test_tree_metadata_cli_is_reachable_without_runtime_imports_or_output_writes(tmp_path):
    path, _, _ = config_fixture(tmp_path)
    result = invoke('validate-tree', path)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['status'] == 'metadata-valid' and report['experimental'] is True
    assert report['required_expansions_per_prompt'] == 15
    assert not (tmp_path / 'out').exists()
    assert invoke('tree-search', '--help').returncode == 0
    assert invoke('bind-tree-lens', '--help').returncode == 0


@pytest.mark.parametrize('change', ['identity', 'hash', 'unknown', 'budget', 'holdout', 'overlap'])
def test_tree_metadata_rejects_mismatched_or_incomplete_inputs(tmp_path, change):
    path, config, _ = config_fixture(tmp_path)
    if change == 'identity':
        manifest = json.loads((tmp_path / 'lens.json').read_text())
        manifest['model_identity']['revision'] = 'wrong-model'
        (tmp_path / 'lens.json').write_text(json.dumps(manifest))
    elif change == 'hash':
        (tmp_path / 'lens.pt').write_bytes(b'changed fixture')
    elif change == 'unknown':
        config['beam_wdith'] = 2
    elif change == 'budget':
        config['max_lens_bytes'] = 1
    elif change == 'holdout':
        config['suite_role'] = 'holdout'
    else:
        config['output'] = '.'
    path.write_text(json.dumps(config))
    result = invoke('validate-tree', path)
    assert result.returncode == 1, result.stderr
    assert json.loads(result.stderr)['error']['type'] == 'ValueError'
    assert not (tmp_path / 'out').exists()


def test_lens_binding_reads_actual_upstream_tensor_format_and_records_attestation(tmp_path):
    from b70_ablator.tree_cli import bind_lens
    path, config, _ = config_fixture(tmp_path)
    output = tmp_path / 'bound.json'
    result = bind_lens(tmp_path / 'lens.pt', config['model_identity'], output)
    assert result['model_origin'] == 'operator-attested'
    assert json.loads(output.read_text())['sha256'] == hashlib.sha256((tmp_path / 'lens.pt').read_bytes()).hexdigest()
    torch.save({'J': {0: torch.eye(3)}, 'd_model': 3, 'source_layers': [0], 'n_prompts': 10}, tmp_path / 'bad.pt')
    with pytest.raises(ValueError, match='width'):
        bind_lens(tmp_path / 'bad.pt', config['model_identity'], tmp_path / 'bad.json')
    assert not (tmp_path / 'bad.json').exists()


def test_learned_direction_bank_can_replace_token_pullbacks_and_is_identity_bound(tmp_path):
    from b70_ablator.tree_cli import load_action_bank
    from b70_ablator.tree_config import validate_tree
    path, config, _ = config_fixture(tmp_path)
    bank = tmp_path / 'bank.pt'
    basis = torch.tensor([[0., 1.]])
    torch.save({'model_identity': config['model_identity'],
                'actions': {'mean-difference': {0: basis, 1: basis}}}, bank)
    config['direction_bank'] = 'bank.pt'
    config['actions']['contrast'] = {'bank': 'mean-difference'}
    path.write_text(json.dumps(config))
    assert validate_tree(path)['status'] == 'metadata-valid'
    actions = load_action_bank(bank, config['model_identity'], (0, 1))
    torch.testing.assert_close(actions['mean-difference'].bases[0], basis)
    wrong = {**config['model_identity'], 'revision': 'different'}
    with pytest.raises(ValueError, match='identity'):
        load_action_bank(bank, wrong, (0, 1))


def test_completed_tree_checks_runtime_provenance_before_model_free_reuse(tmp_path, monkeypatch):
    from b70_ablator.tree_cli import run_tree
    path, _, _ = config_fixture(tmp_path)
    monkeypatch.setattr('b70_ablator.tree_hf.create_causal_mask', lambda **kwargs: None)
    run_tree(path, model_factory=lambda raw: fake_target_factory([]))
    runtime_path = tmp_path / 'out' / 'runtime.json'
    data = json.loads(runtime_path.read_text())
    data['payload'] = {'damaged': True}
    runtime_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='integrity'):
        run_tree(path, model_factory=lambda raw: pytest.fail('no reload on corrupt completed job'))


def fake_target_factory(calls, interrupt=False):
    from test_tree_hf import target_fixture
    hf, root = target_fixture()
    hf.config.max_position_embeddings = 64
    root.embed_tokens.weight.data.fill_(1.)
    head = torch.nn.Linear(2, 4, bias=False)
    head.weight.data.copy_(torch.tensor([[1., 0.], [0., 1.], [1., -1.], [-1., 1.]]))
    hf.model.get_output_embeddings = lambda: head
    hf.preflight = lambda suite: {'prompt_tokens': {r['id']: 2 for r in suite}}
    hf.encode = lambda prompt: ({'input_ids': torch.tensor([[1, 2]])}, False)
    hf.fingerprint = {'fixture_only': True}
    hf.release = lambda: calls.append('release')
    def generate(prompt, min_tokens=None):
        calls.append('decode')
        if interrupt and calls.count('decode') == 2:
            raise RuntimeError('fixture decode interruption')
        hidden = root.embed_tokens(torch.tensor([[1, 2]]))
        for block in root.layers:
            hidden = block(hidden)
        return {'content': str(hidden[0, -1].tolist()), 'finish_reason': 'stop', 'completion_tokens': 4,
                'floor_enforced': min_tokens is not None}
    hf.generate = generate
    return hf


def test_run_tree_persists_search_before_decode_and_resumes_without_search_or_completed_reload(tmp_path, monkeypatch):
    from b70_ablator.tree_cli import run_tree
    path, _, task = config_fixture(tmp_path)
    monkeypatch.setattr('b70_ablator.tree_hf.create_causal_mask', lambda **kwargs: None)
    calls = []
    with pytest.raises(RuntimeError, match='interruption'):
        run_tree(path, model_factory=lambda raw: fake_target_factory(calls, interrupt=True))
    assert calls[-1] == 'release'
    assert len(list((tmp_path / 'out' / 'trees').glob('*.json'))) == 1
    assert len(list((tmp_path / 'out' / 'answers' / 'baseline' / 'items').glob('*.json'))) == 1
    assert not (tmp_path / 'out' / 'report.json').exists()
    monkeypatch.setattr('b70_ablator.tree_search.search_tree', lambda *a, **k: pytest.fail('saved search must be reused'))
    result = run_tree(path, model_factory=lambda raw: fake_target_factory(calls))
    assert result['status'] == 'experimental-unvalidated' and result['generated'] == 1
    from b70_ablator.answers import AnswerStore
    winner = AnswerStore.open(tmp_path / 'out' / 'answers' / 'rank-0', [task])
    assert winner.get(task)['floor_enforced'] is True
    replay = run_tree(path, model_factory=lambda raw: pytest.fail('completed run must remain model-free'))
    assert replay['generated'] == 0 and replay['reused'] == 2
    trace_path = next((tmp_path / 'out' / 'trees').glob('*.json'))
    saved = json.loads(trace_path.read_text())
    saved['payload']['result']['finalists'][0]['path'][0]['strength'] = .001
    trace_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='integrity'):
        run_tree(path, model_factory=lambda raw: pytest.fail('corruption must fail before model loading'))
