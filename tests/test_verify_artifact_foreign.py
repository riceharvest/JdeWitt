"""verify-artifact --foreign: what is checkable in someone else's release directory.

Fixtures are tiny GGUFs written by this repo's own writer, never model weights:
the command reads headers and hashes bytes, so the tests measure that contract
rather than a model's quality.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from gguf import GGMLQuantizationType

from jdewitt.foreign import verify_foreign_artifact
from jdewitt.ggufio import write_gguf

ROOT = Path(__file__).resolve().parents[1]
INVOKE = """
import sys
from jdewitt.cli import main
main()
"""


def invoke(*args):
    return subprocess.run([sys.executable, '-c', INVOKE, *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True)


def _write_gguf(path, arch='qwen2', layers=(0, 1), tokenizer=True):
    """A GGUF whose tensor table spans two layers of one family, 256 values each."""
    tensors = [(f'blk.{layer}.attn_q.weight', [256], int(GGMLQuantizationType.IQ4_XS), b'\x00' * 136)
               for layer in layers]
    metadata: dict = {'general.architecture': arch, 'general.name': 'foreign-fixture'}
    if tokenizer:
        metadata.update({'tokenizer.ggml.model': 'gpt2', 'tokenizer.ggml.tokens': ['a', 'b'],
                         'tokenizer.chat_template': '{{ .Prompt }}'})
    write_gguf(path, metadata, tensors)
    return path


def _fixture(tmp_path, name='heretic-release'):
    directory = tmp_path / name
    directory.mkdir()
    _write_gguf(directory / 'model-IQ4_XS.gguf')
    (directory / 'README.md').write_text('# Heretic-style release\nAbliterated Qwen2.5-0.5B, IQ4_XS.\n')
    (directory / 'results.json').write_text(json.dumps({'refusal_rate': 0.02, 'harness': 'lm-eval'}))
    return directory


def test_foreign_fixture_reports_what_the_bytes_support(tmp_path):
    directory = _fixture(tmp_path)
    result = verify_foreign_artifact(directory)
    assert result['mode'] == 'foreign'
    assert 'valid' not in result  # readable is not the same as certified
    assert result['certified_claims'] == 0
    verifiable = result['verifiable']
    assert verifiable['file_count'] == 3
    assert [row['path'] for row in verifiable['files']] == sorted(
        ['README.md', 'model-IQ4_XS.gguf', 'results.json'])
    assert verifiable['other_files'] == ['README.md', 'results.json']
    row = verifiable['gguf'][0]
    assert row['architecture'] == 'qwen2'
    assert row['tensor_count'] == 2
    assert row['quant_types'] == {'IQ4_XS': {'tensors': 2, 'declared_bytes': 272}}
    # Layer numbers are folded into one family; a mixed family would show both types.
    assert row['quant_by_tensor_family'] == {'blk.*.attn_q.weight': {'IQ4_XS': 2}}
    assert row['tokenizer']['present'] is True
    assert row['tokenizer']['chat_template'] is True
    assert 'tokenizer.ggml.tokens' in row['tokenizer']['keys']
    assert verifiable['tokenizer_present'] is True
    assert verifiable['adapter_card_present'] is False
    assert verifiable['plan_card_present'] is False
    assert len(result['not_verifiable']) == 6
    assert {entry['needs_file'] for entry in result['not_verifiable']} >= {
        'files.json', 'evidence.json', 'signature.json (public_key_pem + signature_base64)'}
    assert result['checks_our_claims_against']
    assert 'release-diff' in result['next']


def test_verified_bytes_are_the_regular_files_and_digests_are_recomputed(tmp_path):
    directory = _fixture(tmp_path)
    result = verify_foreign_artifact(directory)
    regular = sorted(path for path in directory.rglob('*') if path.is_file())
    assert result['verifiable']['verified_bytes'] == sum(path.stat().st_size for path in regular)
    for row in result['verifiable']['files']:
        content = (directory / row['path']).read_bytes()
        assert row['bytes'] == len(content)
        assert row['sha256'] == hashlib.sha256(content).hexdigest()


def test_nested_directories_are_hashed_and_never_printed(tmp_path):
    directory = _fixture(tmp_path)
    (directory / 'hf').mkdir()
    (directory / 'hf' / 'config.json').write_text('{"hidden_marker": "never printed"}')
    result = verify_foreign_artifact(directory)
    assert 'hf/config.json' in [row['path'] for row in result['verifiable']['files']]
    assert 'never printed' not in json.dumps(result)
    assert 'Abliterated Qwen2.5-0.5B' not in json.dumps(result)


def test_cards_shipped_beside_the_artifact_are_named_not_read(tmp_path):
    directory = _fixture(tmp_path)
    (directory / 'adapter.json').write_text('{"not": "a valid card"}')
    (directory / 'plan.json').write_text('{"not": "a plan either"}')
    result = verify_foreign_artifact(directory)
    assert result['verifiable']['adapter_card_present'] is True
    assert result['verifiable']['adapter_card_files'] == ['adapter.json']
    assert result['verifiable']['plan_card_present'] is True
    assert result['verifiable']['plan_card_files'] == ['plan.json']


def test_cli_exits_zero_on_a_foreign_fixture(tmp_path):
    directory = _fixture(tmp_path)
    result = invoke('verify-artifact', directory, '--foreign')
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload['verifiable']['gguf'][0]['quant_types']
    assert payload['certified_claims'] == 0
    assert payload['not_verifiable'] and payload['checks_our_claims_against']


def test_corrupted_gguf_magic_is_header_invalid_and_names_the_file(tmp_path):
    directory = _fixture(tmp_path)
    broken = directory / 'model-IQ3_M.gguf'
    broken.write_bytes(b'XGUF' + b'\x00' * 32)
    result = invoke('verify-artifact', directory, '--foreign')
    assert result.returncode != 0
    error = json.loads(result.stderr)['error']
    assert error['code'] == 'HEADER_INVALID'
    assert str(broken) in error['message']


def test_truncated_gguf_header_is_header_invalid(tmp_path):
    directory = _fixture(tmp_path)
    good = (directory / 'model-IQ4_XS.gguf').read_bytes()
    (directory / 'model-IQ4_XS.gguf').write_bytes(good[:48])  # magic+counts, no KV/tensor table
    result = invoke('verify-artifact', directory, '--foreign')
    assert result.returncode != 0
    assert json.loads(result.stderr)['error']['code'] == 'HEADER_INVALID'


def test_symlinked_directory_is_refused(tmp_path):
    directory = _fixture(tmp_path)
    link = tmp_path / 'link-to-release'
    link.symlink_to(directory)
    result = invoke('verify-artifact', link, '--foreign')
    assert result.returncode != 0
    assert json.loads(result.stderr)['error']['code'] == 'SYMLINK_REFUSED'


def test_nested_symlink_is_refused(tmp_path):
    directory = _fixture(tmp_path)
    (directory / 'model-alias.gguf').symlink_to(directory / 'model-IQ4_XS.gguf')
    result = invoke('verify-artifact', directory, '--foreign')
    assert result.returncode != 0
    error = json.loads(result.stderr)['error']
    assert error['code'] == 'SYMLINK_REFUSED'
    assert 'model-alias.gguf' in error['message']


def test_fifo_is_refused(tmp_path):
    directory = _fixture(tmp_path)
    os.mkfifo(directory / 'daemon.pipe')
    result = invoke('verify-artifact', directory, '--foreign')
    assert result.returncode != 0
    assert json.loads(result.stderr)['error']['code'] == 'NONREGULAR_REFUSED'


def test_directory_with_files_json_is_ambiguous_not_foreign(tmp_path):
    directory = _fixture(tmp_path)
    (directory / 'files.json').write_text('{}')
    result = invoke('verify-artifact', directory, '--foreign')
    assert result.returncode != 0
    error = json.loads(result.stderr)['error']
    assert error['code'] == 'FOREIGN_AMBIGUOUS'
    assert 'files.json' in error['message']


def test_foreign_flag_cannot_be_combined_with_native_flags(tmp_path):
    directory = _fixture(tmp_path)
    result = invoke('verify-artifact', directory, '--foreign', '--all')
    assert result.returncode == 1
    assert json.loads(result.stderr)['error']['type'] == 'ValueError'


def test_name_that_tries_to_traverse_is_refused(tmp_path):
    directory = _fixture(tmp_path)
    (directory / '..\\evil.gguf').write_bytes(b'GGUF')
    result = invoke('verify-artifact', directory, '--foreign')
    assert result.returncode != 0
    assert json.loads(result.stderr)['error']['code'] == 'PATH_TRAVERSAL_REFUSED'


def test_file_path_is_not_a_foreign_directory(tmp_path):
    path = _write_gguf(tmp_path / 'single.gguf')
    result = invoke('verify-artifact', path, '--foreign')
    assert result.returncode != 0
    assert json.loads(result.stderr)['error']['code'] == 'NOT_A_DIRECTORY'


def test_native_path_still_checks_nothing_extra_on_such_a_directory(tmp_path):
    """--all stays the native path: a foreign dir has no schema artifacts to check."""
    directory = _fixture(tmp_path)
    result = invoke('verify-artifact', directory, '--all')
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload['checked'] == []
    assert 'files.json' in payload['missing_schemas']
