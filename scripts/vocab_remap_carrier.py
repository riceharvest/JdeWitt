#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build a tokenizer-carrier GGUF from a vocabulary-bearing artifact.

``llama-server`` refuses to load llama.cpp's vocab-only test files (there is no
``token_embd.weight``), so a tokenizer that is only available as a fixture
cannot be served for ``POST /tokenize``. This writes a one-block llama-arch
container whose tokenizer metadata (tokens, token types, merges, model, pre,
bos/eos) is copied verbatim from the source artifact and whose tensors are tiny
zero/one arrays. The served tokenizer is therefore the artifact's own
tokenizer; the tensors carry no information and no model is run for a
tokenize request.

Usage:
    python scripts/vocab_remap_carrier.py SOURCE.gguf DEST.gguf
    llama-server -m DEST.gguf -ngl 0 -c 256 --parallel 1 --port N --no-webui

The output prints one JSON summary. Never commit the carrier: it is derived.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFWriter

EMBEDDING = 32
FEED_FORWARD = 64
HEADS = 4
ROPE_DIMS = 8
TOKENIZER_IDS = ('tokenizer.ggml.bos_token_id', 'tokenizer.ggml.eos_token_id',
                 'tokenizer.ggml.padding_token_id')


def _field(reader, name):
    field = reader.fields.get(name)
    return field.contents() if field is not None else None


def build(source, destination):
    reader = GGUFReader(str(source))
    tokens = _field(reader, 'tokenizer.ggml.tokens')
    if not tokens:
        raise ValueError(f'{source.name}: no tokenizer.ggml.tokens array')
    model = _field(reader, 'tokenizer.ggml.model')
    pre = _field(reader, 'tokenizer.ggml.pre')
    if not model or not pre:
        raise ValueError(f'{source.name}: tokenizer.ggml.model/pre missing; this is not a BPE artifact')
    writer = GGUFWriter(str(destination), 'llama')
    writer.add_string('general.name', 'vocab-carrier')
    writer.add_uint32('general.file_type', 1)
    writer.add_uint32('llama.block_count', 1)
    writer.add_uint32('llama.context_length', 256)
    writer.add_uint32('llama.embedding_length', EMBEDDING)
    writer.add_uint32('llama.feed_forward_length', FEED_FORWARD)
    writer.add_uint32('llama.attention.head_count', HEADS)
    writer.add_uint32('llama.attention.head_count_kv', HEADS)
    writer.add_uint32('llama.rope.dimension_count', ROPE_DIMS)
    writer.add_float32('llama.attention.layer_norm_rms_epsilon', 1e-5)
    writer.add_string('tokenizer.ggml.model', model)
    writer.add_string('tokenizer.ggml.pre', pre)
    writer.add_array('tokenizer.ggml.tokens', list(tokens))
    token_types = _field(reader, 'tokenizer.ggml.token_type')
    if token_types:
        writer.add_array('tokenizer.ggml.token_type', [int(value) for value in token_types])
    merges = _field(reader, 'tokenizer.ggml.merges')
    if merges:
        writer.add_array('tokenizer.ggml.merges', list(merges))
    for name in TOKENIZER_IDS:
        value = _field(reader, name)
        if value is not None:
            writer.add_uint32(name, int(value))
    shape = (len(tokens), EMBEDDING)
    writer.add_tensor('token_embd.weight', np.zeros(shape, dtype=np.float32))
    writer.add_tensor('output_norm.weight', np.ones(EMBEDDING, dtype=np.float32))
    writer.add_tensor('output.weight', np.zeros(shape, dtype=np.float32))
    writer.add_tensor('blk.0.attn_norm.weight', np.ones(EMBEDDING, dtype=np.float32))
    writer.add_tensor('blk.0.attn_q.weight', np.zeros((EMBEDDING, EMBEDDING), dtype=np.float32))
    writer.add_tensor('blk.0.attn_k.weight', np.zeros((EMBEDDING, EMBEDDING), dtype=np.float32))
    writer.add_tensor('blk.0.attn_v.weight', np.zeros((EMBEDDING, EMBEDDING), dtype=np.float32))
    writer.add_tensor('blk.0.attn_output.weight', np.zeros((EMBEDDING, EMBEDDING), dtype=np.float32))
    writer.add_tensor('blk.0.ffn_norm.weight', np.ones(EMBEDDING, dtype=np.float32))
    writer.add_tensor('blk.0.ffn_gate.weight', np.zeros((FEED_FORWARD, EMBEDDING), dtype=np.float32))
    writer.add_tensor('blk.0.ffn_down.weight', np.zeros((EMBEDDING, FEED_FORWARD), dtype=np.float32))
    writer.add_tensor('blk.0.ffn_up.weight', np.zeros((FEED_FORWARD, EMBEDDING), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'destination': destination.name, 'n_vocab': len(tokens), 'pre': pre,
            'bytes': destination.stat().st_size,
            'note': 'derived carrier; tokenizer metadata copied, tensors carry no information'}


def main():
    parser = argparse.ArgumentParser(description='Build a tokenizer-carrier GGUF from a vocabulary-bearing artifact.')
    parser.add_argument('source')
    parser.add_argument('destination')
    args = parser.parse_args()
    print(json.dumps(build(Path(args.source), Path(args.destination)), indent=2))


if __name__ == '__main__':
    main()
