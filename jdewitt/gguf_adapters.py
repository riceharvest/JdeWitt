# SPDX-License-Identifier: AGPL-3.0-or-later
"""Serve in-memory edits to a running llama.cpp server as GGUF LoRA adapters.

The merged-resident operator keeps one model in VRAM and recomputes each trial's
weights from scratch, so trial throughput is bound by inference kernels. The
same edits already have exactly the form ``W <- W + B @ A``: AblatedLinear stores
A [rank,in] and B [out,rank] in torch orientation. llama.cpp applies that delta
when it loads an adapter, so a trial can keep the model's quantized weights
untouched in VRAM and swap a small adapter instead.

Conventions pinned by reading this fork's code, not assumed:
  * llama-adapter.cpp load-time shape check (ggml ne order, ne[0]=fast dim):
    base W ne=[in,out]; lora_a ne=[in,rank]; lora_b ne=[rank,out].
    GGUF tensor dims are reversed numpy shape, so torch A [rank,in] and
    B [out,rank] are written with numpy dims unchanged (no transpose).
  * get_scale(): with adapter.lora.alpha omitted (==0) the applied scale is
    exactly the requested adapter_scale; POST scale 1.0 gives delta = B @ A.
  * llama-adapter.cpp requires general.type == 'adapter' and adapter.type == 'lora',
    and names adapter tensors '<base-tensor>.lora_a'/'.lora_b' with '.weight'
    intact (it strips the suffix and looks the remainder up verbatim).
  * Runtime lifecycle, PROVEN live on this fork (2026-11-17): adapters are listed
    at startup only (--lora comma list, ids in load order); POST /lora-adapters
    sets {id,scale} in ~1 ms; scale 0 disables; several adapter files touching
    the SAME tensor apply additively (a+b differs from either alone). There is
    NO dynamic file loading — new files never reach a running server.

A delta that serves the truth must be derived from the served artifact's own
weights: A = d @ W and the served W is quantized, so write adapters computed
against dequantized served weights (serve_factors), never against the
full-precision originals.
"""
import re
import torch
from pathlib import Path
from gguf import GGUFReader, GGUFWriter

from .errors import NoVerifiedAdapter

# editor source_key component -> llama.cpp tensor suffix. Deliberately narrow:
# an unmapped or absent tensor raises, because a silently skipped target means a
# trial whose served edit differs from the searched edit.
HF_TO_LLAMA = (
    ('self_attn.o_proj', 'attn_output'),
    ('mlp.down_proj', 'ffn_down'),
)

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

def gguf_weight_shapes(artifact):
    """name -> (in, out) in ggml/gguf dimension order for every 2D weight.

    GGUF stores linear weights transposed from torch, so GGUFReader .shape is
    [in, out]: blk.5.ffn_down = (10240, 2560) = (ffn, hidden) on spark2_5, and
    qwen3-4b blk.9.attn_output = (4096, 2560) = (heads*head_dim, hidden). The
    torch layout is the reversed pair, which is what ggufio.dequantize_tensor
    hands back.
    """
    reader = GGUFReader(str(artifact))
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


def artifact_adapter_view(shapes):
    """What discovery saw in this artifact, as facts a refusal can carry."""
    width = inferred_hidden_width(shapes)
    discovered = (discover_writeback_targets(shapes, width) if width is not None
                  else {'targets': {}, 'unmapped_writeback_width': []})
    layers = sorted(discovered['targets'])
    named = [name for layer in layers[:2] for name in discovered['targets'][layer]]
    return {'hidden_width': width, 'writeback_layers': layers,
            'writeback_names': named, 'unmapped_writeback_width':
                discovered['unmapped_writeback_width']}


def _unresolved_name(key, base, shapes, reason):
    """The refusal for a source_key this artifact cannot be named against."""
    view = artifact_adapter_view(shapes)
    layers = view['writeback_layers']
    looked = f' (looked for {base})' if base else ''
    if layers:
        carried = (f'this artifact carries residual-write projections at hidden width '
                   f'{view["hidden_width"]} for layers {layers[:8]}'
                   f'{"..." if len(layers) > 8 else ""}')
    else:
        carried = ('this artifact carries no residual-write projection name at all '
                   f'(known suffixes: {WRITEBACK_TENSOR.pattern})')
    unmapped = view['unmapped_writeback_width'][:8]
    raise NoVerifiedAdapter(
        f'{reason} for source_key {key}{looked}: {carried}; width-matching tensors it would '
        f'not edit: {unmapped or "none"}. A tensor that happens to exist at this width is not '
        f'an adapter, so this refuses instead of guessing one; list what the file carries with '
        f'`python -m jdewitt.spanfit targets --gguf-in <artifact>`',
        source_key=key, looked_for=base, **view)


def resolve_gguf_names(source_keys, artifact_shapes):
    """Every editor source_key to its artifact tensor name, or a loud refusal."""
    out = {}
    for key in source_keys:
        stem = key.removesuffix('.weight')
        layer = [p for p in stem.split('.') if p.isdigit()]
        if not layer:
            raise NoVerifiedAdapter(f'source_key has no layer index: {key}', source_key=key)
        suffix = next((s for c, s in HF_TO_LLAMA if c in stem), None)
        if suffix is None:
            _unresolved_name(key, None, artifact_shapes,
                             'no llama.cpp LoRA name mapping')
        base = f'blk.{layer[0]}.{suffix}.weight'
        if base not in artifact_shapes:
            _unresolved_name(key, base, artifact_shapes, 'artifact has no tensor')
        out[key] = base
    return out

def serve_factors(direction, weight, strength):
    """A [rank,in] and B [out,rank] derived from the served weight.

    Mirrors AblatedLinear.set_directions row_normalization='none':
    A = d @ W, B = -strength * d.T, so B @ A = -strength * d.T @ d @ W removes
    the projected direction from the served rows. ``weight`` is the dequantized
    served weight in torch [out,in] orientation; ``direction`` is [rank, out].
    """
    w = weight.to(device=direction.device, dtype=torch.float32)
    if tuple(direction.shape[1:]) != tuple(w.shape[:1]):
        raise ValueError(f'direction width {direction.shape[1:]} does not match served out_features {w.shape[0]}')
    d = direction.to(dtype=torch.float32)
    with torch.no_grad():
        a = d @ w
        b = -strength * d.T
    return a, b

def per_direction_adapters(basis, serve_weights, names_by_index, arch, out_dir, rank_cap=None):
    """One rank-1 adapter file per basis direction: file i = -d_i^T (d_i W_q).

    basis: [R, hidden] float, rows are the orthonormal directions the search
    mixes; names_by_index maps basis row i (or a subset) to an artifact tensor
    name; serve_weights maps tensor name -> served weight in torch [out,in].
    The server preloads all files (--lora comma list, ids in this order); a
    trial is then one POST /lora-adapters carrying per-direction scales, which
    is exactly the AblatedLinear strength/selection mix at row_normalization
    'none'. Returns per-direction file reports.

    The served LoRA curve is a screening approximation only: on tensors the
    server graph transforms, it is NOT the export delta (docs/METHODOLOGY.md
    R8). Shape verdicts are taken on the merged artifact from served_deltas.
    """
    files = []
    for i in range(basis.shape[0] if rank_cap is None else min(rank_cap, basis.shape[0])):
        pairs = []
        for name in names_by_index.get(i, []):
            w = serve_weights[name]
            a, b = serve_factors(basis[i:i + 1], w, strength=1.0)  # rank-1 slice [1, out]
            pairs.append((name, a, b))
        if not pairs:
            continue
        path = Path(out_dir) / f'dir{i:03d}.gguf'
        path.parent.mkdir(parents=True, exist_ok=True)
        report = write_adapter(path, pairs, arch)
        files.append({'direction': i, 'tensors': len(pairs), **report})
    return files

def write_adapter(path, pairs, arch):
    """pairs: iterable of (gguf_base_name, A_torch[rank,in], B_torch[out,rank]).

    Tensors are written float16 with numpy dims unchanged; the fork's load-time
    shape check then reads ne=[in,rank] and [rank,out] as required. With
    adapter.lora.alpha omitted the effective scale is the requested adapter_scale.
    """
    ranks = set()
    writer = GGUFWriter(str(path), arch)
    for name, A, B in pairs:
        A = A.detach().cpu().float().to(torch.float16).numpy()
        B = B.detach().cpu().float().to(torch.float16).numpy()
        if A.shape[0] != B.shape[1] or A.shape[1] == 0:
            raise ValueError(f'adapter factors disagree: {name}')
        ranks.add(A.shape[0])
        base = name  # llama strips '.lora_a' and looks the remainder up verbatim,
        # so the adapter name must be '<tensor>.lora_a' with '.weight' intact.
        writer.add_tensor(f'{base}.lora_a', A)
        writer.add_tensor(f'{base}.lora_b', B)
    writer.add_string('general.type', 'adapter')  # llama-adapter.cpp:206 requires this
    writer.add_string('adapter.type', 'lora')     # llama-adapter.cpp:217 requires this
    # This gguf-py opens the file lazily; close() only closes handles. Without
    # the explicit write sequence below the writer returns "success" and no
    # file ever lands on disk — proven by ls this exact function.
    writer.open_output_file()
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f'adapter writer returned success but wrote no file: {path}')
    return {'path': str(path), 'bytes': path.stat().st_size, 'tensors': len(pairs), 'ranks': sorted(ranks)}

def served_deltas(path, directions, tensor_names, strength=1.0):
    """Ground-truth edit deltas for the gguf at ``path``: {name: float32 delta}.

    Feed the result to ggufio.edit_tensors to produce the merged gguf; that is
    the artifact shape certification must run on (METHODOLOGY R8). ``directions``
    are [R, out_features] float rows (orthonormal; enforced by projection_delta).
    Weights are read back in gguf dimension order and the output-side projection
    algebra runs in that order; a tensor whose width matches the other dim is
    transposed through the algebra so both ggml layout conventions work.
    """
    import numpy as np
    from .ggufio import dequantize_tensor, projection_delta
    d = directions.detach().float().cpu().numpy() if hasattr(directions, 'detach') else np.asarray(directions, dtype=np.float32)
    d = np.atleast_2d(d).astype(np.float32)
    from gguf import GGUFReader
    reader = GGUFReader(str(path))
    deltas = {}
    for name in tensor_names:
        w = dequantize_tensor(path, name, reader=reader)
        if d.shape[1] == w.shape[0]:
            deltas[name] = projection_delta(w, d, strength, side='output')
        elif d.shape[1] == w.shape[1]:
            deltas[name] = projection_delta(w.T.copy(), d, strength, side='output').T.copy()
        else:
            raise ValueError(f'direction width {d.shape[1]} matches neither dim of {name} {w.shape}')
    return deltas
