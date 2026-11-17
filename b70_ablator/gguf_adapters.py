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
  * POST /lora-adapters (server-common.cpp parse_lora_request) accepts only
    {"id":N,"scale":S} entries for adapters the server knows; new files are
    picked up from a dynamic --lora-path directory, not by request path.

A delta that serves the truth must be derived from the served artifact's own
weights: A = d @ W and the served W is quantized, so write adapters computed
against dequantized served weights (serve_factors), never against the
full-precision originals.
"""
import torch
from pathlib import Path
from gguf import GGUFReader, GGUFWriter

# editor source_key component -> llama.cpp tensor suffix. Deliberately narrow:
# an unmapped or absent tensor raises, because a silently skipped target means a
# trial whose served edit differs from the searched edit.
HF_TO_LLAMA = (
    ('self_attn.o_proj', 'attn_output'),
    ('mlp.down_proj', 'ffn_down'),
)

def gguf_weight_shapes(artifact):
    """name -> (in, out) in ggml/gguf dimension order for every 2D weight.

    GGUF stores linear weights transposed from torch: GGUFReader .shape comes
    back as [in, out], verified against blk.5.ffn_down = (10240, 2560) =
    (ffn, hidden) for spark2_5. torch orientation is the reversed pair.
    """
    reader = GGUFReader(str(artifact))
    shapes = {}
    for t in reader.tensors:
        if len(t.shape) == 2:
            shapes[t.name] = tuple(t.shape)  # (out, in)
    return shapes

def resolve_gguf_names(source_keys, artifact_shapes):
    """Every editor source_key to its artifact tensor name, or a loud refusal."""
    out = {}
    for key in source_keys:
        stem = key.removesuffix('.weight')
        layer = [p for p in stem.split('.') if p.isdigit()]
        if not layer:
            raise ValueError(f'source_key has no layer index: {key}')
        suffix = next((s for c, s in HF_TO_LLAMA if c in stem), None)
        if suffix is None:
            raise ValueError(f'no llama.cpp LoRA name mapping for source_key {key}')
        base = f'blk.{layer[0]}.{suffix}.weight'
        if base not in artifact_shapes:
            raise ValueError(f'artifact has no tensor for source_key {key} (looked for {base})')
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
