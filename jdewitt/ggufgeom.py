# SPDX-License-Identifier: AGPL-3.0-or-later
"""Artifact geometry read from a GGUF's own tensor table; no torch, no model.

`gguf_adapters` also writes LoRA adapters, which needs torch, so every caller
that only needs to *read* geometry (the Space bank loader, coverage checks,
header range requests) would otherwise pull a multi-hundred-MB import into the
process. These three functions lived there and are moved here verbatim;
`gguf_adapters` re-exports them so existing callers do not change.

Nothing in this module reads tensor data: `GGUFReader` parses the header and the
tensor info array, and every answer below comes from the names and dimensions
that array states.
"""
import re

from gguf import GGUFReader

# Canonical residual-write suffixes llama.cpp uses across architectures. Every
# one of these is a projection whose OUTPUT is the residual stream: attention
# output, SSM out on hybrid/recurrent blocks, FFN down. Discovery is by name
# because two projections can share the hidden width (q_proj on MHA models) and
# an axis captured in residual space is meaningless in query space, so a pure
# dimension rule would edit the wrong matrix while passing every shape check.
WRITEBACK_TENSOR = re.compile(
    r'^(attn_output|attn_out|ssm_out|ffn_down|w2|o_proj|down_proj)$')


def discover_writeback_targets(shapes, width):
    """{layer: [gguf tensor names]} discovered from the artifact itself.

    ``shapes`` is gguf_weight_shapes() output: name -> (in, out), so a tensor
    writes the residual stream when shape[1] == hidden width. This replaces the
    per-model source-key table: the artifact says which writeback projections it
    carries and where they live, on any architecture, with nothing typed by
    hand. Names that write a hidden-wide vector but are not residual-write
    projections are reported separately rather than edited.
    """
    by_layer, unmapped = {}, []
    for name, dims in shapes.items():
        match = re.match(r'^blk\.(\d+)\.(.+)\.weight$', name)
        if match is None or len(dims) != 2:
            continue
        layer, suffix = int(match.group(1)), match.group(2).split('.')[-1]
        if int(dims[1]) != int(width):
            continue
        if not WRITEBACK_TENSOR.match(suffix):
            unmapped.append(name)
            continue
        by_layer.setdefault(layer, []).append(name)
    return {'targets': {layer: sorted(names) for layer, names in sorted(by_layer.items())},
            'unmapped_writeback_width': sorted(unmapped)}

def gguf_weight_shapes(artifact, reader=None):
    """name -> (in, out) in ggml/gguf dimension order for every 2D weight.

    GGUF stores linear weights transposed from torch, so GGUFReader .shape is
    [in, out]: blk.5.ffn_down = (10240, 2560) = (ffn, hidden) on spark2_5, and
    qwen3-4b blk.9.attn_output = (4096, 2560) = (heads*head_dim, hidden). The
    torch layout is the reversed pair, which is what ggufio.dequantize_tensor
    hands back. ``reader`` lets a caller that already parsed the header share it.
    """
    reader = reader or GGUFReader(str(artifact))
    shapes = {}
    for t in reader.tensors:
        if len(t.shape) == 2:
            shapes[t.name] = tuple(t.shape)  # (out, in)
    return shapes

def inferred_hidden_width(shapes):
    """Modal output width of the residual-write projections the artifact carries.

    The artifact never states its hidden size; its writeback projections do, and
    the majority width is the residual stream's (a partial file still has more
    agreeing tensors than disagreeing ones). None when no writeback name is
    present at all — then there is nothing to infer from, which is itself the
    finding.
    """
    widths = {}
    for name, dims in shapes.items():
        match = re.match(r'^blk\.\d+\.(.+)\.weight$', name)
        if match is None or len(dims) != 2:
            continue
        if not WRITEBACK_TENSOR.match(match.group(1).split('.')[-1]):
            continue
        widths[int(dims[1])] = widths.get(int(dims[1]), 0) + 1
    if not widths:
        return None
    return max(widths.items(), key=lambda item: item[1])[0]
