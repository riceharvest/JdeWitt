// Q2_0 quantizer + dequantizer reference.
// Copied VERBATIM from ggml-quants.c @ master sha256 1665bffa9c0ef0e692f8799523a729879960a6aa07adaa32f9efb41f7076a3da
// (quantize_row_q2_0_ref, dequantize_row_q2_0); block struct from ggml-common.h sha256
// 0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c (QK2_0 = 64, 18-byte block).
// Harness substitutions only: ggml_half via _Float16, GGML_FP32_TO_FP16/FP16_TO_FP32, GGML_RESTRICT, main().
// usage: ref_q2_0 q < weights.f32 > blocks.bin ; ref_q2_0 dq < blocks.bin > f32.bin
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint16_t ggml_half;
#define GGML_RESTRICT
#define QK2_0 64
static inline float GGML_FP16_TO_FP32(ggml_half h) { union { uint16_t u; _Float16 f; } b = { .u = h }; return (float)b.f; }
static inline ggml_half GGML_FP32_TO_FP16(float f) { union { _Float16 f; uint16_t u; } b = { .f = (_Float16)f }; return b.u; }
typedef struct {
    ggml_half d;              // delta (scale)
    uint8_t qs[QK2_0 / 4];   // 2 bits per element
} block_q2_0;
static_assert(sizeof(block_q2_0) == sizeof(ggml_half) + QK2_0 / 4, "wrong q2_0 block size/padding");

void quantize_row_q2_0_ref(const float * GGML_RESTRICT x, block_q2_0 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK2_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        // Compute scale as max absolute value in the block
        float amax = 0.0f;
        for (int j = 0; j < qk; j++) {
            const float a = fabsf(x[i*qk + j]);
            if (a > amax) amax = a;
        }
        const float d = amax;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        // Clear quant bytes
        for (int j = 0; j < qk / 4; ++j) {
            y[i].qs[j] = 0;
        }

        // Encode 2-bit values: round(w/d) clamped to [-1, 2], then add 1
        // 00 (-1) = -scale, 01 (0) = 0, 10 (+1) = +scale, 11 (+2) = 2*scale
        for (int j = 0; j < qk; ++j) {
            const float w = x[i*qk + j];
            int q = (int)roundf(w * id) + 1;
            if (q < 0) q = 0;
            if (q > 3) q = 3;
            const int byte_index = j / 4;
            const int bit_offset = (j % 4) * 2;
            y[i].qs[byte_index] |= ((uint8_t)q << bit_offset);
        }
    }
}

void dequantize_row_q2_0(const block_q2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK2_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            const int byte_index = j / 4;
            const int bit_offset = (j % 4) * 2;
            const uint8_t q = (x[i].qs[byte_index] >> bit_offset) & 0x03;
            // 00=-1, 01=0, 10=+1, 11=+2
            y[i*qk + j] = ((int)q - 1) * d;
        }
    }
}

static uint8_t * slurp(size_t * n) {
    size_t cap = 1 << 20, len = 0; uint8_t * b = malloc(cap); size_t r;
    while ((r = fread(b + len, 1, cap - len, stdin)) > 0) { len += r; if (len == cap) { cap *= 2; b = realloc(b, cap); } }
    *n = len; return b;
}

int main(int argc, char ** argv) {
    if (argc != 2) { fprintf(stderr, "mode q|dq\n"); return 2; }
    size_t n; uint8_t * in = slurp(&n);
    if (!strcmp(argv[1], "q")) {
        if (n % (4 * QK2_0)) return 2;
        int64_t k = n / 4; block_q2_0 * y = calloc(1, k / QK2_0 * sizeof(block_q2_0));
        quantize_row_q2_0_ref((const float *)in, y, k);
        fwrite(y, sizeof(block_q2_0), k / QK2_0, stdout);
    } else if (!strcmp(argv[1], "dq")) {
        if (n % sizeof(block_q2_0)) return 2;
        size_t nb = n / sizeof(block_q2_0); float * y = malloc(nb * QK2_0 * 4);
        dequantize_row_q2_0((const block_q2_0 *)in, y, nb * QK2_0);
        fwrite(y, 4, nb * QK2_0, stdout);
    } else return 2;
    return 0;
}
