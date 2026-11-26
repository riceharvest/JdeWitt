"""gguf-inspect's tensor digests: metadata-only edits keep every tensor's sha256.

The experiment they exist for is a chat-template edit; the claim it needs is
"no tensor byte changed", checked per tensor rather than asserted.
"""
import hashlib

import numpy as np

from jdewitt.agent import gguf_inspect, tensor_data_digests
from jdewitt.ggufio import edit_metadata, edit_tensors, write_gguf


def _fixture(path):
    tensors = []
    raws = {}
    for name, dims, seed in (('a.weight', (3, 4), 1), ('b.weight', (2, 5), 2)):
        array = np.random.RandomState(seed).randn(*dims).astype(np.float32)
        raw = np.ascontiguousarray(array).reshape(-1).tobytes()
        tensors.append((name, dims, 0, raw))
        raws[name] = raw
    write_gguf(path, {'general.architecture': 'test', 'general.name': 'pytest',
                      'tokenizer.chat_template': 'before'}, tensors)
    return raws


def test_tensor_data_digests_are_sha256_of_stored_bytes(tmp_path):
    source = tmp_path / 'm.gguf'
    raws = _fixture(source)
    digests = tensor_data_digests(source)
    assert digests == {name: hashlib.sha256(raw).hexdigest() for name, raw in raws.items()}


def test_metadata_only_edit_reports_every_tensor_digest_identical(tmp_path):
    source, edited = tmp_path / 'm.gguf', tmp_path / 'm.edited.gguf'
    _fixture(source)
    edit_metadata(source, edited, {'tokenizer.chat_template': 'after'}, [])

    report = gguf_inspect(source, edited)
    assert report['diff']['counts']['metadata_changed'] == 1
    assert [row['key'] for row in report['diff']['metadata_changed']] == ['tokenizer.chat_template']
    assert report['diff']['counts']['tensor_digests_identical'] == 2
    assert report['diff']['counts']['tensor_digests_changed'] == 0
    assert sorted(report['diff']['tensor_digests']['identical']) == ['a.weight', 'b.weight']
    assert report['diff']['tensor_digests']['changed'] == []
    assert report['before']['tensor_data_sha256'] == report['after']['tensor_data_sha256']


def test_a_changed_tensor_is_named_not_hidden(tmp_path):
    source, edited = tmp_path / 'm.gguf', tmp_path / 'm.edited.gguf'
    _fixture(source)
    edit_tensors(source, edited, {'b.weight': np.full((5, 2), 0.25, dtype=np.float32)})

    report = gguf_inspect(source, edited)
    assert report['diff']['counts']['tensor_digests_identical'] == 1
    assert report['diff']['counts']['tensor_digests_changed'] == 1
    assert [row['name'] for row in report['diff']['tensor_digests']['changed']] == ['b.weight']
    assert report['diff']['tensor_digests']['identical'] == ['a.weight']
    assert report['before']['tensor_data_sha256'] != report['after']['tensor_data_sha256']
    assert report['diff']['counts']['metadata_changed'] == 0
