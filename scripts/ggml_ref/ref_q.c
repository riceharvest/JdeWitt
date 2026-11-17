// Q4_0/Q8_0 quantizer reference: ggml-quants.c master bcda336 verbatim.
// ./ref_q q4_0|q8_0 ; stdin float32, stdout blocks.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>
typedef uint16_t ggml_half;
#define GGML_RESTRICT
#define QK_K 256
#define QK4_0 32
#define QK4N QK4_0
#define QK8_0 32
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define GROUP_MAX_EPS 1e-15f
static inline float GGML_FP16_TO_FP32(ggml_half h) { union { uint16_t u; _Float16 f; } b = { .u = h }; return (float)b.f; }
static inline ggml_half GGML_FP32_TO_FP16(float f) { union { _Float16 f; uint16_t u; } b = { .f = (_Float16)f }; return b.u; }
typedef struct { ggml_half d; uint8_t qs[QK4_0/2]; } block_q4_0;
typedef struct { ggml_half d; int8_t qs[QK8_0]; } block_q8_0;
typedef struct { ggml_half d, dmin; uint8_t scales[12]; uint8_t qs[128]; } block_q4_K;
typedef struct { ggml_half d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } block_q5_K;
typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; ggml_half d; } block_q6_K;
static_assert(sizeof(block_q4_0) == 18, "q4_0");
static_assert(sizeof(block_q8_0) == 34, "q8_0");
static_assert(sizeof(block_q4_K) == 144, "q4_K");
static_assert(sizeof(block_q5_K) == 176, "q5_K");
static_assert(sizeof(block_q6_K) == 210, "q6_K");
void quantize_row_q4_0_ref(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}
void quantize_row_q8_0_ref(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i*QK8_0 + j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = x[i*QK8_0 + j]*id;

            y[i].qs[j] = roundf(x0);
        }
    }
}
int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: ref_q MODE\n"); return 2; }
    const char * m = argv[1];
    int qk = 32, bs = strcmp(m, "q4_0") ? 34 : 18;
    size_t cap = 1 << 20;
    float * in = malloc(cap * sizeof(float));
    uint8_t * out = malloc(cap / qk * bs);
    size_t n;
    while ((n = fread(in, sizeof(float), cap, stdin)) > 0) {
        if (n % qk) { fprintf(stderr, "not multiple of 32\n"); return 2; }
        if (!strcmp(m, "q4_0")) quantize_row_q4_0_ref(in, (block_q4_0 *) out, (int64_t)n);
        else quantize_row_q8_0_ref(in, (block_q8_0 *) out, (int64_t)n);
        size_t want = n / qk * bs;
        if (fwrite(out, 1, want, stdout) != want) return 2;
    }
    return 0;
}
