"""The reproduction index observes the buyer's base gguf; one moved byte is named."""
import json

import numpy as np

from jdewitt import release
from jdewitt.data import digest


def write_gguf(path, value=1.0):
    from gguf import GGUFWriter
    writer = GGUFWriter(str(path), 'fixturearch')
    writer.add_tensor('blk.0.attn_q.weight',
                      np.array([[value, 0.0], [0.0, value]], dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def release_dir(tmp_path, base_path):
    directory = tmp_path / 'release'
    directory.mkdir()
    write_gguf(directory / 'model-merged.gguf')
    evaluation = {'version': 1, 'kind': 'fixture', 'base': {
        'path': str(base_path), 'sha256': release.sha256_file(base_path),
        'bytes': base_path.stat().st_size}}
    (directory / 'evaluation.json').write_text(json.dumps(evaluation))
    (directory / 'README.md').write_text('fixture release, not a quality claim\n')
    record = {'version': 1,
              'release_identity': digest({'evaluation': evaluation,
                                          'readme': (directory / 'README.md').read_text()}),
              'files': release.release_inventory(directory)}
    record['record_digest'] = digest(record)
    (directory / 'files.json').write_text(json.dumps(record))
    return directory


def test_base_is_observed_from_the_declared_path(tmp_path):
    base = write_gguf(tmp_path / 'base.gguf')
    directory = release_dir(tmp_path, base)
    built = release.write_repro(directory)
    assert built['base']['observed'] is True
    assert built['base']['sha256'] == release.sha256_file(base)
    assert built['base']['tensors'] == 1
    payload = json.loads((directory / 'repro.json').read_text())
    assert payload['version'] == release.REPRO_VERSION
    checked = release.check_repro(directory)
    assert checked['valid'] is True, checked
    assert checked['moved'] == [] and checked['problems'] == []
    assert checked['base_file']['sha256'] == release.sha256_file(base)


def test_same_bytes_under_another_name_keep_the_binding(tmp_path):
    base = write_gguf(tmp_path / 'base.gguf')
    directory = release_dir(tmp_path, base)
    first = release.build_repro(directory)
    elsewhere = tmp_path / 'elsewhere' / 'Spark-renamed.gguf'
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(base.read_bytes())
    second = release.build_repro(directory, base=elsewhere)
    # The operand is the digest, not the file name or where it sat.
    assert first['binding_digest'] == second['binding_digest']
    assert first['repro_digest'] != second['repro_digest']


def test_one_flipped_byte_in_the_base_is_named_with_both_digests(tmp_path):
    base = write_gguf(tmp_path / 'base.gguf')
    directory = release_dir(tmp_path, base)
    release.write_repro(directory)
    spoiled = tmp_path / 'spoiled.gguf'
    data = bytearray(base.read_bytes())
    data[-1] ^= 0xFF
    spoiled.write_bytes(bytes(data))
    result = release.check_repro(directory, base=spoiled)
    assert result['valid'] is False
    rows = [row for row in result['moved'] if row['operand'] == 'base.sha256']
    assert rows == [{'operand': 'base.sha256', 'expected': release.sha256_file(base),
                     'actual': release.sha256_file(spoiled)}]
    assert release.check_repro(directory)['valid'] is True


def test_missing_explicit_base_is_named_not_silently_replaced(tmp_path):
    base = write_gguf(tmp_path / 'base.gguf')
    directory = release_dir(tmp_path, base)
    release.write_repro(directory)
    result = release.check_repro(directory, base=tmp_path / 'nope.gguf')
    assert result['valid'] is False
    assert 'nope.gguf' in json.dumps(result)


def test_version_2_seal_still_checks_without_reading_the_base(tmp_path):
    base = write_gguf(tmp_path / 'base.gguf')
    directory = release_dir(tmp_path, base)
    payload = release.build_repro(directory, version=2, observe_base=False)
    assert payload['operands']['base']['observed'] is False
    (directory / 'repro.json').write_text(json.dumps(dict(payload, repro_digest=digest(
        {key: value for key, value in payload.items() if key != 'repro_digest'}))))
    checked = release.check_repro(directory)
    assert checked['valid'] is True, checked
    assert checked['resolved']['base']['observed'] is False
