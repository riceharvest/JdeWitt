# ggml scalar-quantizer reference harnesses

ref_q4k.c / ref_q5k.c / ref_q6k.c contain the upstream scalar quantizers
(quantize_row_q4_K_ref / q5_K_ref / q6_K_ref plus make_qx_quants /
make_qkx2_quants / nearest_int / get_scale_min_k4) copied VERBATIM from
ggml-quants.c @ https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-quants.c
sha256 bcda33625707fdfcb1367cdd2bb1a2f017430162ec9623abc84ad17459698f73
with only these harness substitutions: GGML_FP16_TO_FP32/GGML_FP32_TO_FP16 via
_Float16 unions (IEEE RNE, bit-identical to ggml's fallback), struct definitions
from ggml-common.h, and a stdin/stdout main().

Build and falsify:
    gcc -O1 -o ref_q4k scripts/ggml_ref/ref_q4k.c -lm
    ./ref_q4k < weights.f32 > c.bin       # weights.f32: raw float32, multiple of 256 elements
    python -c "import numpy as np; from b70_ablator.quants import quantize_q4_k; \
        x=np.frombuffer(open('weights.f32','rb').read(), np.float32); \
        c=np.frombuffer(open('c.bin','rb').read(), np.uint8); \
        print('byte-exact:', (np.frombuffer(quantize_q4_k(x), np.uint8)==c).all())"

Landed byte-exact (100%) on 11 distributions x 3 codecs: gaussian, shifted
gaussian, zeros+impulse, 1e6 dynamic range, 1e-4 band, subnormal mixes,
GROUP_MAX_EPS boundary, negative-dominant, and six real F32 tensors of
verification/minicpm-v47-Q4_K_M.gguf. These harnesses caught four real port
bugs (stale running min, missing f16 round-trip on requant scales, int64->
float64 promotion, C-order left-fold sums) — keep them when touching quants.py.
