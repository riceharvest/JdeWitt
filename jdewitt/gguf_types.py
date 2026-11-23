# SPDX-License-Identifier: AGPL-3.0-or-later
"""GGUF quantization type names: one alias table and one canonical-naming function.

Split out of `jdewitt.quantize` for the same reason `ggufgeom` was split out of
`gguf_adapters`: a caller that only needs to name a type must not pull a module
that shells out to `llama-quantize` and carries local binary paths. The Space
vendors this module, never the CLI wrapper.

`canonical_type` is deliberately permissive: anything gguf-py can name is
accepted, because the encoder is the authority on what it can write and refusing
here would silently cap callers at the K-quant family.
"""
from __future__ import annotations

TYPE_ALIASES = {
    'q2_k': 'Q2_K', 'q3_k_s': 'Q3_K_S', 'q3_k_m': 'Q3_K_M', 'q3_k_l': 'Q3_K_L',
    'q4_0': 'Q4_0', 'q4_1': 'Q4_1', 'q4_k_s': 'Q4_K_S', 'q4_k_m': 'Q4_K_M',
    'q5_0': 'Q5_0', 'q5_1': 'Q5_1', 'q5_k_s': 'Q5_K_S', 'q5_k_m': 'Q5_K_M',
    'q6_k': 'Q6_K', 'q8_0': 'Q8_0', 'f16': 'F16', 'bf16': 'BF16', 'f32': 'F32',
    'iq1_s': 'IQ1_S', 'iq1_m': 'IQ1_M', 'iq2_xxs': 'IQ2_XXS', 'iq2_xs': 'IQ2_XS',
    'iq2_s': 'IQ2_S', 'iq2_m': 'IQ2_M', 'iq3_xxs': 'IQ3_XXS', 'iq3_xs': 'IQ3_XS',
    'iq3_s': 'IQ3_S', 'iq3_m': 'IQ3_M', 'iq4_xs': 'IQ4_XS', 'iq4_nl': 'IQ4_NL',
    'tq1_0': 'TQ1_0', 'tq2_0': 'TQ2_0', 'mxfp4': 'MXFP4',
}


def canonical_type(name):
    key = str(name).strip().lower()
    if key in TYPE_ALIASES:
        return TYPE_ALIASES[key]
    upper = str(name).strip().upper()
    if upper in set(TYPE_ALIASES.values()):
        return upper
    if upper == 'Q2_0':
        return upper  # newer than published gguf; a codec here, not an enum member
    try:
        from gguf import GGMLQuantizationType

        if upper in GGMLQuantizationType.__members__:
            return upper
    except Exception:
        pass
    raise ValueError(f'unknown quantization type {name!r}; known: {", ".join(sorted(TYPE_ALIASES))}')
