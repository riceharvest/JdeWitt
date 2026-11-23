# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Space staging script and the coverage-only loop it ships."""
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from jdewitt import space_loop as loop

REPO = Path(__file__).resolve().parents[1]
BANK = REPO / 'spaces' / 'bank'
FIXTURE_BANK = REPO / 'verification' / 'space' / 'fixtures' / 'bank'
BANNED = re.compile(r'/mnt/|/home/|t_[0-9a-f]{8}|uncensored')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage_mod = _load('space_stage_under_test', REPO / 'scripts' / 'space_stage.py')


def _stage(out):
    return stage_mod.stage(out)


def _tree(out):
    return {path.relative_to(out).as_posix(): path.read_bytes()
            for path in sorted(Path(out).rglob('*')) if path.is_file()}


def _frontmatter(text):
    assert text.startswith('---\n'), 'README must open with YAML frontmatter'
    block = text[4:].split('\n---\n', 1)[0]
    data, models = {}, []
    for line in block.splitlines():
        if line.startswith('  - '):
            models.append(line[4:].strip())
        else:
            key, _, value = line.partition(':')
            value = value.strip()
            if key == 'models':
                data['models'] = [] if value == '[]' else models
            elif value:
                data[key] = value.strip('"')
    return data


# --------------------------------------------------------------------------- staging


def test_two_stages_are_byte_identical(tmp_path):
    """Done gate: two runs, `diff -r` empty."""
    first, second = tmp_path / 'first', tmp_path / 'second'
    _stage(first)
    _stage(second)
    assert _tree(first) == _tree(second)
    assert len(_tree(first)) > 10


def test_tree_carries_the_space_contract(tmp_path):
    _stage(tmp_path / 'out')
    for name in ('README.md', 'app.py', 'requirements.txt', '.gitignore',
                 'jdewitt/__init__.py', 'jdewitt/bank.py', 'jdewitt/space_loop.py',
                 'spaces/bank/bank.json', 'spaces/bank/README.md'):
        assert (tmp_path / 'out' / name).is_file(), name


def test_readme_frontmatter_names_the_sdk_and_the_app(tmp_path):
    _stage(tmp_path / 'out')
    front = _frontmatter((tmp_path / 'out' / 'README.md').read_text())
    assert front['title']
    assert front['sdk'] == 'gradio'
    assert re.fullmatch(r'\d+\.\d+\.\d+', front['sdk_version'])
    assert front['app_file'] == 'app.py'
    assert front['short_description'] and len(front['short_description']) <= 60
    assert front['models'] == []  # the bank ships empty, so no base is claimed


def test_readme_models_lists_only_the_bank_bases(tmp_path, monkeypatch):
    bank = tmp_path / 'bank'
    bank.mkdir()
    document = {'schema': stage_mod.BANK_SCHEMA, 'entries': [
        {'key': 'one', 'base': {'repo': 'owner/one'}},
        {'key': 'two', 'base': {'repo': 'owner/two'}},
        {'key': 'one-again', 'base': {'repo': 'owner/one'}},
    ]}
    (bank / 'bank.json').write_text(json.dumps(document))
    monkeypatch.setattr(stage_mod, 'BANK', bank)
    files, _ = stage_mod.build()
    front = _frontmatter(files['README.md'].decode())
    assert front['models'] == ['owner/one', 'owner/two']


def test_requirements_are_the_four_and_carry_no_torch(tmp_path):
    _stage(tmp_path / 'out')
    lines = [line.strip() for line in (tmp_path / 'out' / 'requirements.txt').read_text().splitlines() if line.strip()]
    names = {re.split(r'[<>=!~]', line, 1)[0].strip() for line in lines}
    assert names == {'numpy', 'gguf', 'huggingface_hub', 'gradio'}
    text = ' '.join(lines)
    for absent in ('torch', 'transformers', 'llama'):
        assert absent not in text


def test_bank_directory_is_copied_verbatim(tmp_path):
    _stage(tmp_path / 'out')
    for path in sorted(item for item in BANK.rglob('*') if item.is_file()):
        relative = path.relative_to(BANK).as_posix()
        assert (tmp_path / 'out' / 'spaces' / 'bank' / relative).read_bytes() == path.read_bytes()


def test_vendored_package_files_are_byte_identical(tmp_path):
    _stage(tmp_path / 'out')
    vendored = [path for path in (tmp_path / 'out' / 'jdewitt').rglob('*.py')]
    assert vendored
    for path in vendored:
        assert path.read_bytes() == (REPO / 'jdewitt' / path.relative_to(tmp_path / 'out' / 'jdewitt')).read_bytes()


def test_no_host_path_board_id_or_banned_word_survives_staging(tmp_path):
    """Done gate: the sanitisation grep, run over every staged text file."""
    _stage(tmp_path / 'out')
    hits = []
    for path in sorted((tmp_path / 'out').rglob('*')):
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        hits += [f'{path.relative_to(tmp_path / "out")}:{match.group(0)}'
                 for match in BANNED.finditer(text)]
    assert hits == []


def test_staged_loop_imports_without_gradio_or_torch(tmp_path):
    """Done gate: the staged tree imports with neither Gradio nor torch."""
    out = tmp_path / 'out'
    _stage(out)
    result = subprocess.run(
        [sys.executable, '-c',
         'import sys\n'
         f'sys.path.insert(0, {str(out)!r})\n'
         'import jdewitt.space_loop as loop\n'
         'assert "gradio" not in sys.modules, gradio\n'
         'assert "torch" not in sys.modules, torch\n'
         'print(loop.SCHEMA)\n'],
        capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert loop.SCHEMA in result.stdout


@pytest.mark.parametrize('payload, label', [
    (b'a host path /mnt/ssd/models\n', 'absolute path'),
    (b'a home path /home/someone/x\n', 'absolute path'),
    (b'see t_0a1b2c3d for the card\n', 'board card id'),
    (b'the uncensored arm\n', 'banned word'),
])
def test_sanitisation_refuses_each_banned_shape(payload, label):
    with pytest.raises(stage_mod.StageRefused) as caught:
        stage_mod._check_sanitised({'note.md': payload})
    assert caught.value.code == 'SANITISATION'
    assert label in caught.value.message


def test_sanitisation_ignores_binary_files():
    assert stage_mod._check_sanitised({'bundle.tar': b'\x00/mnt/\xff'}) is True


def test_a_missing_import_is_refused():
    with pytest.raises(stage_mod.StageRefused) as caught:
        stage_mod._closure('from jdewitt.not_here import nothing\n')
    assert caught.value.code == 'MISSING_IMPORT'
    assert 'jdewitt.not_here' in caught.value.message


def test_an_import_the_requirements_do_not_ship_is_refused():
    with pytest.raises(stage_mod.StageRefused) as caught:
        stage_mod._check_external({'torch'}, 'app.py')
    assert caught.value.code == 'UNDECLARED_IMPORT'
    assert 'torch' in caught.value.message


def test_an_oversized_tree_is_refused(monkeypatch):
    monkeypatch.setattr(stage_mod, 'MAX_OUTSIDE_BANK', 10)
    with pytest.raises(stage_mod.StageRefused) as caught:
        stage_mod.build()
    assert caught.value.code == 'TOO_LARGE'


def test_staging_over_an_existing_tree_is_refused(tmp_path):
    _stage(tmp_path / 'out')
    with pytest.raises(stage_mod.StageRefused) as caught:
        _stage(tmp_path / 'out')
    assert caught.value.code == 'OUT_NOT_EMPTY'


# --------------------------------------------------------------------------- inspect


def test_inspect_refuses_the_empty_bank_with_bank_miss():
    report = loop.inspect('Qwen/Qwen3-4B-Instruct-2507', None, 'Q4_K_M')
    assert report['coverage']['state'] == 'uncovered'
    assert report['refusal']['code'] == 'BANK_MISS'
    assert 'Qwen/Qwen3-4B-Instruct-2507' in report['refusal']['message']
    assert report['licence']['state'] == 'UNVERIFIED'
    assert report['edit']['state'] == 'none'
    assert 'BANK_MISS' in loop.render(report)


def test_inspect_covers_the_fixture_base():
    report = loop.inspect('jdewitt/bank-fixture', 'fixture', 'F16', bank_dir=FIXTURE_BANK)
    assert report['coverage']['state'] == 'covered'
    assert report['coverage']['entry'] == 'bankfixture/f00l01-fixture'
    assert report['coverage']['revision_match'] == 'exact'
    assert report['coverage']['label'] == 'banked_unmeasured'
    assert report['coverage']['quant_types'] == ['f16']
    assert report['refusal'] is None
    assert report['licence']['state'] == 'UNVERIFIED'


def test_inspect_refuses_an_unbanked_quant():
    report = loop.inspect('jdewitt/bank-fixture', 'fixture', 'Q4_K_M', bank_dir=FIXTURE_BANK)
    assert report['refusal']['code'] == 'UNSUPPORTED_QUANT'
    assert 'q4_k' in report['refusal']['message']
    assert 'f16' in report['refusal']['message']


def test_inspect_refuses_a_malformed_repository():
    report = loop.inspect('not a repo', None, 'Q4_K_M')
    assert report['refusal']['code'] == 'INPUT_ERROR'
    assert report['coverage']['state'] == 'uncovered'


def test_inspect_labels_a_different_revision_as_transfer():
    report = loop.inspect('jdewitt/bank-fixture', 'deadbeef', 'F16', bank_dir=FIXTURE_BANK)
    assert report['coverage']['state'] == 'covered'
    assert report['coverage']['revision_match'] == 'different'
    assert report['coverage']['label'] == 'transfer'


@pytest.mark.parametrize('asked, expected', [
    ('Q4_K_M', 'q4_k'), ('q5_k_s', 'q5_k'), ('Q8_0', 'q8_0'), ('IQ4_XS', 'iq4_xs'),
    ('F16', 'f16'), ('bogus!', None), ('', None), (None, None),
])
def test_quant_type_maps_the_variant_suffixes(asked, expected):
    assert loop.quant_type(asked) == expected
