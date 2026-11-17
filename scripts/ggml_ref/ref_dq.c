// Combined dequant reference: ggml-quants.c master bcda336 verbatim (functions + get_scale_min_k4).
// ./ref_dq q4_0|q8_0|q4k|q5k|q6k ; stdin packed blocks, stdout float32.
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
static inline void get_scale_min_k4(int j, const uint8_t * GGML_RESTRICT q, uint8_t * GGML_RESTRICT d, uint8_t * GGML_RESTRICT m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
void dequantize_row_q4_0(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}
void dequantize_row_q8_0(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK8_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}
void dequantize_row_q4_K(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t * q = x[i].qs;

        const float d   = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}
void dequantize_row_q5_K(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t * ql = x[i].qs;
        const uint8_t * qh = x[i].qh;

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}
void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT ql = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const int8_t  * GGML_RESTRICT sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}
int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: ref_dq MODE\n"); return 2; }
    const char * m = argv[1];
    size_t bs = 0; int64_t els = 32;
    if (!strcmp(m, "q4_0")) bs = 18;
    else if (!strcmp(m, "q8_0")) bs = 34;
    else if (!strcmp(m, "q4k")) { bs = 144; els = 256; }
    else if (!strcmp(m, "q5k")) { bs = 176; els = 256; }
    else if (!strcmp(m, "q6k")) { bs = 210; els = 256; }
    else { fprintf(stderr, "bad mode\n"); return 2; }
    size_t cap = 1 << 19;
    uint8_t * in = malloc(cap * bs);
    float * out = malloc(cap * els * sizeof(float));
    size_t n;
    while ((n = fread(in, bs, cap, stdin)) > 0) {
        int64_t k = (int64_t)n * els;
        if (!strcmp(m, "q4_0")) dequantize_row_q4_0((const block_q4_0 *) in, out, k);
        else if (!strcmp(m, "q8_0")) dequantize_row_q8_0((const block_q8_0 *) in, out, k);
        else if (!strcmp(m, "q4k")) dequantize_row_q4_K((const block_q4_K *) in, out, k);
        else if (!strcmp(m, "q5k")) dequantize_row_q5_K((const block_q5_K *) in, out, k);
        else dequantize_row_q6_K((const block_q6_K *) in, out, k);
        if (fwrite(out, sizeof(float), n * els, stdout) != n * els) return 2;
    }
    return 0;
}
