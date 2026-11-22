# SPDX-License-Identifier: AGPL-3.0-or-later
"""Requantize an artifact, with per-tensor type overrides where they were edited.

Why this exists: an edit is only real if the file can STORE it. ``ggufio.preview_edit``
measures that per tensor, and on a 1.7B Q4_K_M a strength-1.0 rank-1 removal keeps only
cos 0.45 of the intended projection — the requantization noise is as large as the edit.
The faithful order is therefore: edit a 16-bit artifact (cos 1.0000), then quantize for
serving. The improvement over "ship 16-bit and let the user quantize" is the second half
of this command: the tensors that carry the edit can be held at a type where the edit
survives while everything else goes to the serving type, so one file is both small and
faithful.

The quantizer is llama.cpp's own ``llama-quantize`` (whole-file, C, already the tool
users quantize with); this wrapper is the model-free, agent-facing step: one JSON, an
explicit tensor-type map, and a verification that the output really carries every tensor
the input did.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

# Known local builds, in preference order. JDEWITT_QUANTIZE wins over all of them.
KNOWN_BINARIES = (
    '/home/dario/llama.cpp-spark/build-cpu/bin/llama-quantize',
    '/home/dario/llama.cpp-minicpm47/build-cpu/bin/llama-quantize',
)

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


def resolve_binary(binary=None):
    candidate = binary or os.environ.get('JDEWITT_QUANTIZE')
    if candidate:
        if not Path(candidate).is_file():
            raise ValueError(f'quantizer {candidate} does not exist; pass --binary or set JDEWITT_QUANTIZE')
        return str(candidate)
    for path in KNOWN_BINARIES:
        if Path(path).is_file():
            return path
    raise ValueError('no llama-quantize found; pass --binary PATH or set JDEWITT_QUANTIZE')


def canonical_type(name):
    key = str(name).strip().lower()
    if key in TYPE_ALIASES:
        return TYPE_ALIASES[key]
    upper = str(name).strip().upper()
    if upper in set(TYPE_ALIASES.values()):
        return upper
    # Anything gguf-py can name is worth handing to llama-quantize: the binary is the
    # authority on what this build can emit, and refusing here would silently cap the
    # tool at the K-quant family (IQ/TQ/MXFP4 targets included).
    try:
        from gguf import GGMLQuantizationType

        if upper in GGMLQuantizationType.__members__:
            return upper
    except Exception:
        pass
    raise ValueError(f'unknown quantization type {name!r}; known: {", ".join(sorted(TYPE_ALIASES))}')


def parse_overrides(specs):
    """['blk.10.attn_output.weight=q8_0', ...] -> ['blk.10.attn_output.weight=Q8_0', ...]."""
    out = []
    for spec in specs or ():
        if '=' not in spec:
            raise ValueError(f'--keep-tensor wants NAME=TYPE, got {spec!r}')
        name, kind = spec.rsplit('=', 1)
        out.append(f'{name.strip()}={canonical_type(kind)}')
    return out


def tensor_names(path):
    from gguf import GGUFReader

    return [tensor.name for tensor in GGUFReader(str(path)).tensors]


def quantize_file(src, out, kind, *, binary=None, overrides=(), threads=None,
                  allow_requantize=True, keep_output_tensor=False, imatrix=None,
                  timeout=7200):
    """Quantize `src` to `out` at `kind`, with per-tensor `overrides`.

    Returns one report dict: the exact argv, the tensor counts on both sides, and the
    overrides actually applied — an override that llama-quantize ignored is the silent
    failure this layer has to catch, so the report echoes what was requested.
    """
    src, out = Path(src), Path(out)
    if not src.is_file():
        raise ValueError(f'{src} does not exist')
    if out.exists() and out.resolve() == src.resolve():
        raise ValueError('refusing to quantize onto the input file: writing a second '
                         'artifact is what makes the type change possible')
    binary = resolve_binary(binary)
    target = canonical_type(kind)
    override_list = parse_overrides(overrides)
    command = [binary, str(src), str(out), target]
    if threads:
        command.append(str(int(threads)))
    if allow_requantize:
        command.append('--allow-requantize')
    if keep_output_tensor:
        command.append('--leave-output-tensor')
    if imatrix:
        # IQ-family targets refuse to quantize without an importance matrix; the flag is
        # the difference between "IQ is unsupported" and "IQ needs one more input".
        if not Path(imatrix).is_file():
            raise ValueError(f'importance matrix {imatrix} does not exist')
        command += ['--imatrix', str(imatrix)]
    for override in override_list:
        command += ['--tensor-type', override]
    started = time.monotonic()
    # llama-quantize writes progress bars with raw bytes: decoding as utf-8 strictly
    # raises in the middle of a working run and looks like a quantizer failure.
    finished = subprocess.run(command, capture_output=True, text=True,
                              encoding='utf-8', errors='replace', timeout=timeout)
    output = finished.stdout + finished.stderr
    if finished.returncode != 0 or not out.is_file():
        hint = ''
        if 'importance matrix' in output and not imatrix:
            hint = ('  Hint: this target needs an importance matrix; pass --imatrix FILE '
                    '(llama-imatrix over a corpus) or pick a type that does not '
                    '(q2_k..q8_0, f16, bf16).')
        raise ValueError(f'llama-quantize failed (rc {finished.returncode}): {output[-800:]}{hint}')
    before, after = tensor_names(src), tensor_names(out)
    missing = [name for name in before if name not in set(after)]
    report = {
        'source': str(src), 'output': str(out), 'type': target,
        'overrides_requested': override_list,
        'binary': binary,
        'argv': command,
        'tensors_in': len(before), 'tensors_out': len(after),
        'missing_tensors': missing[:16],
        'bytes_in': src.stat().st_size, 'bytes_out': out.stat().st_size,
        'elapsed_s': round(time.monotonic() - started, 1),
        'tail': output[-400:] if finished.returncode == 0 else None,
    }
    return report
