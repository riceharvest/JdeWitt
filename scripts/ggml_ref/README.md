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
    python -c "import numpy as np; from jdewitt.quants import quantize_q4_k; \
        x=np.frombuffer(open('weights.f32','rb').read(), np.float32); \
        c=np.frombuffer(open('c.bin','rb').read(), np.uint8); \
        print('byte-exact:', (np.frombuffer(quantize_q4_k(x), np.uint8)==c).all())"

Landed byte-exact (100%) on 11 distributions x 3 codecs: gaussian, shifted
gaussian, zeros+impulse, 1e6 dynamic range, 1e-4 band, subnormal mixes,
GROUP_MAX_EPS boundary, negative-dominant, and six real F32 tensors of
verification/minicpm-v47-Q4_K_M.gguf. These harnesses caught four real port
bugs (stale running min, missing f16 round-trip on requant scales, int64->
float64 promotion, C-order left-fold sums) — keep them when touching quants.py.

ref_dq.c: dequant reference for q4_0/q8_0/q4_K/q5_K/q6_K (`./ref_dq q4k < blocks.bin > f32.bin`).
ref_q.c: q4_0/q8_0 quantizer reference (`./ref_q q4_0 < weights.f32 > blocks.bin`).
Proven bit-exact (dequant, random packed bytes incl. NaN/inf scale bits and real
tensor blocks) and byte-exact (quantize, 11 distributions incl. exact-half ties
and real tensors). These caught two more real semantic bugs: q4_0 needs C's
truncation toward zero on (x0+8.5), not rounding, and q8_0 needs roundf's
half-away-from-zero, not round-half-to-even.
Every quantize/dequantize pair shipped in CODECS now has compiled-C proof.

ref_4150.c: q4_1/q5_0/q5_1 quantize+dequant reference (`./ref_4150 q5_0 [quantize]`).
Proven bit-exact (dequant, random packed bytes) and byte-exact (quantize, 14
distributions incl. exact-half ties, all-equal blocks and real tensors). The
falsifier caught a dequantizer bias we had missed: ggml dequantize_row_q5_0
stores an offset code and subtracts 16 (its quantizer adds 16.5 before
truncating), which our first port omitted.

ref_iq4.c: IQ4_NL / IQ4_XS quantizer reference (`./ref_iq4 iq4_nl|iq4_xs`), copied verbatim
from ggml-quants.c @ master sha256 1665bffa9c0ef0e692f8799523a729879960a6aa07adaa32f9efb41f7076a3da
with the kvalues_iq4nl table from ggml-common.h at the same revision (sha256
0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c).
quantize_iq4_nl and quantize_iq4_xs (jdewitt/quants.py) are byte-exact vs it on
10 distributions x 64 blocks each and on real F32 checkpoint tensors; the only
divergence is the upstream-undefined all-zero-block path of IQ4_NL, where the C
reference leaves the code buffer untouched and we write zero codes.
The IQ grid tables (iq2xxs/iq2xs/iq2s, iq3xxs/iq3s, iq1s, ksigns/kmask) are
vendored into jdewitt/iq_tables.py by scripts/vendor_iq_tables.py, which records
the ggml-common.h sha256 above and refuses a size mismatch.

ref_23k.c: q2_K/q3_K scalar quantizer reference (`./ref_23k q2_K < weights.f32 > blocks.bin`).
quantize_row_q2_K_ref byte-exact (13 distributions incl. 4 real F32 checkpoint tensors)
— the first falsifier run exposed that make_qkx2_quants uses the CALLER's weights:
q2_K passes plain |x| while q4_K/q5_K pass av_x+|x| (the port had hardcoded the latter,
which is why q2_K quantized 0.0% byte-exact initially).
quantize_row_q3_K_ref: byte-exact (100%) on 12 distributions (gaussian x2, zeros+impulse,
uniform-tiny, subnormal-mix, neg-dominant, near-eps, 1e6-range, all-equal-neg, bimodal,
flat-subblocks, exact-half) and 6 real F32 tensors of the minicpm Q4_K_M. The first port
diverged ~16% of gaussian blocks because its running sums were accumulated in double;
keeping every product in f32 order fixed it. Q3_K is in CODECS, so gguf-edit can requant it.
