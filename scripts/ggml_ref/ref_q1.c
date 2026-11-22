// Q1_0 (1-bit sign + f16 mean-|x| scale, 128-element blocks) reference.
// Copied from ggml-quants.c @ master sha256 1665bffa9c0ef0e692f8799523a729879960a6aa07adaa32f9efb41f7076a3da
// (quantize_row_q1_0_ref, dequantize_row_q1_0), with harness substitutions only:
// ggml_fp16_t via _Float16, block_q1_0/QK1_0 from ggml-common.h (sha 0061131b...),
// GGML_RESTRICT, and a stdin/stdout main().
//
// usage: ref_q1 q < weights.f32 > blocks.bin
//        ref_q1 d < blocks.bin  > weights.f32
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGML_RESTRICT
#define QK1_0 128
typedef uint16_t ggml_half;
typedef uint16_t ggml_fp16_t;
typedef struct {
    ggml_half d;           // delta
    uint8_t qs[QK1_0 / 8]; // bits / quants
} block_q1_0;
static_assert(sizeof(block_q1_0) == sizeof(ggml_half) + QK1_0 / 8, "wrong q1_0 block size/padding");

static inline ggml_fp16_t GGML_FP32_TO_FP16(float f) {
    _Float16 h = (_Float16)f; ggml_fp16_t out; memcpy(&out, &h, sizeof(out)); return out;
}
static inline float GGML_FP16_TO_FP32(ggml_fp16_t v) {
    _Float16 h; memcpy(&h, &v, sizeof(h)); return (float)h;
}

// reference implementation for deterministic creation of model files
void quantize_row_q1_0_ref(const float * GGML_RESTRICT x, block_q1_0 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK1_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float sum_abs = 0.0f;
        for (int j = 0; j < qk; j++) {
            sum_abs += fabsf(x[i*qk + j]);
        }
        const float d = sum_abs / qk;

        y[i].d = GGML_FP32_TO_FP16(d);

        // Clear all bits first
        for (int j = 0; j < qk / 8; ++j) {
            y[i].qs[j] = 0;
        }

        // Just store sign of each weight directly (no normalization)
        for (int j = 0; j < qk; ++j) {
            const int bit_index = j;
            const int byte_index = bit_index / 8;
            const int bit_offset = bit_index % 8;

            if (x[i*qk + j] >= 0.0f) {
                y[i].qs[byte_index] |= (1 << bit_offset);
            }
        }
    }
}

void dequantize_row_q1_0(const block_q1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK1_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float neg_d = -d;

        for (int j = 0; j < qk; ++j) {
            const int byte_index = j / 8;
            const int bit_offset = j % 8;
            const uint8_t bit = (x[i].qs[byte_index] >> bit_offset) & 1;
            y[i*qk + j] = bit ? d : neg_d;
        }
    }
}

static unsigned char * slurp(size_t * n) {
    size_t cap = 1 << 20, len = 0;
    unsigned char * buf = malloc(cap);
    size_t r;
    while ((r = fread(buf + len, 1, cap - len, stdin)) > 0) {
        len += r;
        if (len == cap) { cap *= 2; buf = realloc(buf, cap); }
    }
    *n = len;
    return buf;
}

int main(int argc, char ** argv) {
    if (argc < 2 || (argv[1][0] != 'q' && argv[1][0] != 'd')) { fprintf(stderr, "usage: ref_q1 q|d\n"); return 2; }
    size_t n; unsigned char * in = slurp(&n);
    if (argv[1][0] == 'q') {
        int64_t k = n / 4;
        if (k % QK1_0) { fprintf(stderr, "need multiple of %d floats\n", QK1_0); return 2; }
        block_q1_0 * y = malloc((k / QK1_0) * sizeof(block_q1_0));
        quantize_row_q1_0_ref((const float *)in, y, k);
        fwrite(y, sizeof(block_q1_0), k / QK1_0, stdout);
    } else {
        int64_t nb = n / sizeof(block_q1_0);
        float * y = malloc(nb * QK1_0 * sizeof(float));
        dequantize_row_q1_0((const block_q1_0 *)in, y, nb * QK1_0);
        fwrite(y, sizeof(float), nb * QK1_0, stdout);
    }
    return 0;
}
