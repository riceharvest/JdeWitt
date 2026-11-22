// TQ1_0 / TQ2_0 scalar reference (quantize + dequantize).
// Copied from ggml-quants.c @ master sha256 1665bffa9c0ef0e692f8799523a729879960a6aa07adaa32f9efb41f7076a3da
// (quantize_row_tq1_0_ref, quantize_row_tq2_0_ref, dequantize_row_tq1_0,
// dequantize_row_tq2_0), with harness substitutions only: ggml_fp16_t via
// _Float16, block structs from ggml-common.h, GGML_RESTRICT, MAX, and a
// stdin/stdout main().
//
// usage: ref_tq MODE < in > out
//   MODE tq1_0 | tq2_0       : f32 stream -> blocks
//   MODE dq_tq1_0 | dq_tq2_0 : blocks -> f32 stream
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QK_K 256
#define GGML_RESTRICT
#define MAX(a, b) ((a) > (b) ? (a) : (b))

typedef uint16_t ggml_fp16_t;
static inline ggml_fp16_t GGML_FP32_TO_FP16(float f) {
    _Float16 h = (_Float16)f; ggml_fp16_t out; memcpy(&out, &h, sizeof(out)); return out;
}
static inline float GGML_FP16_TO_FP32(ggml_fp16_t v) {
    _Float16 h; memcpy(&h, &v, sizeof(h)); return (float)h;
}

typedef struct { uint8_t qs[(QK_K - 4 * QK_K / 64) / 5]; uint8_t qh[QK_K/64]; ggml_fp16_t d; } block_tq1_0;
static_assert(sizeof(block_tq1_0) == 54, "tq1_0");
typedef struct { uint8_t qs[QK_K/4]; ggml_fp16_t d; } block_tq2_0;
static_assert(sizeof(block_tq2_0) == 66, "tq2_0");

void quantize_row_tq1_0_ref(const float * GGML_RESTRICT x, block_tq1_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK_K; j++) {
            const float v = x[j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        // 5 elements per byte, along 32 bytes
        for (size_t j = 0; j < sizeof(y->qs) - sizeof(y->qs) % 32; j += 32) {
            for (size_t m = 0; m < 32; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 5; ++n) {
                    int xi = lroundf(x[m + n*32] * id) + 1; // -1, 0, 1 -> 0, 1, 2
                    q *= 3;
                    q += xi;
                }
                // ceiling division (243 == pow(3, 5))
                q = ((uint16_t)q * 256 + (243 - 1)) / 243;
                y[i].qs[j + m] = q;
            }
            x += 5*32;
        }
        // along 16 bytes
        for (size_t j = sizeof(y->qs) - sizeof(y->qs) % 32; j < sizeof(y->qs); j += 16) {
            for (size_t m = 0; m < 16; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 5; ++n) {
                    int xi = lroundf(x[m + n*16] * id) + 1; // -1, 0, 1 -> 0, 1, 2
                    q *= 3;
                    q += xi;
                }
                // ceiling division (243 == pow(3, 5))
                q = ((uint16_t)q * 256 + (243 - 1)) / 243;
                y[i].qs[j + m] = q;
            }
            x += 5*16;
        }
        // 4 elements per byte
        for (size_t j = 0; j < sizeof(y->qh); ++j) {
            uint8_t q = 0;
            for (size_t m = 0; m < 4; ++m) {
                // -1, 0, 1 -> 0, 1, 2
                int xi = lroundf(x[j + m*sizeof(y->qh)] * id) + 1;
                q *= 3;
                q += xi;
            }
            // shift the first value to the most significant trit
            q *= 3;
            // ceiling division (243 == pow(3, 5))
            q = ((uint16_t)q * 256 + (243 - 1)) / 243;
            y[i].qh[j] = q;
        }
        x += 4*sizeof(y->qh);
    }
}

void quantize_row_tq2_0_ref(const float * GGML_RESTRICT x, block_tq2_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK_K; j++) {
            const float v = x[j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (size_t j = 0; j < sizeof(y->qs); j += 32) {
            for (size_t m = 0; m < 32; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 4; ++n) {
                    // -1, 0, 1 -> 0, 1, 2
                    int xi = lroundf(x[m + n*32] * id) + 1;
                    q += (xi & 3) << (2*n);
                }
                y[i].qs[j + m] = q;
            }
            x += 4*32;
        }
    }
}

void dequantize_row_tq1_0(const block_tq1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};

    for (int64_t i = 0; i < nb; ++i) {

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (size_t j = 0; j < sizeof(x->qs) - sizeof(x->qs) % 32; j += 32) {
            for (size_t n = 0; n < 5; ++n) {
                for (size_t m = 0; m < 32; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[n];
                    int16_t xi = ((uint16_t) q * 3) >> 8;
                    *y++ = (float) (xi - 1) * d;
                }
            }
        }
        for (size_t j = sizeof(x->qs) - sizeof(x->qs) % 32; j < sizeof(x->qs); j += 16) {
            for (size_t n = 0; n < 5; ++n) {
                for (size_t m = 0; m < 16; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[n];
                    int16_t xi = ((uint16_t) q * 3) >> 8;
                    *y++ = (float) (xi - 1) * d;
                }
            }
        }

        for (size_t n = 0; n < 4; ++n) {
            for (size_t j = 0; j < sizeof(x->qh); ++j) {
                uint8_t q = x[i].qh[j] * pow3[n];
                int16_t xi = ((uint16_t) q * 3) >> 8;
                *y++ = (float) (xi - 1) * d;
            }
        }
    }
}

void dequantize_row_tq2_0(const block_tq2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; ++i) {

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (size_t j = 0; j < sizeof(x->qs); j += 32) {
            for (size_t l = 0; l < 4; ++l) {
                for (size_t m = 0; m < 32; ++m) {
                    int8_t q = (x[i].qs[j + m] >> (l*2)) & 3;
                    *y++ = (float) (q - 1) * d;
                }
            }
        }
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: ref_tq tq1_0|tq2_0|dq_tq1_0|dq_tq2_0\n"); return 2; }
    const char * mode = argv[1];
    int dq = strncmp(mode, "dq_", 3) == 0;
    int t1 = strstr(mode, "tq1_0") != NULL;
    int t2 = strstr(mode, "tq2_0") != NULL;
    if (!t1 && !t2) { fprintf(stderr, "bad mode\n"); return 2; }
    size_t cap = 1 << 20, n = 0;
    uint8_t * buf = malloc(cap);
    size_t r;
    while ((r = fread(buf + n, 1, cap - n, stdin)) > 0) {
        n += r;
        if (n == cap) { cap *= 2; buf = realloc(buf, cap); }
    }
    size_t bsz = t1 ? sizeof(block_tq1_0) : sizeof(block_tq2_0);
    if (!dq) {
        if (n % (QK_K * 4)) { fprintf(stderr, "input not multiple of 1024 bytes\n"); return 2; }
        int64_t k = n / 4, nb = k / QK_K;
        float * x = malloc(n);
        memcpy(x, buf, n);
        uint8_t * out = calloc(nb, bsz);
        if (t1) quantize_row_tq1_0_ref(x, (block_tq1_0 *) out, k);
        else    quantize_row_tq2_0_ref(x, (block_tq2_0 *) out, k);
        fwrite(out, 1, nb * bsz, stdout);
    } else {
        if (n % bsz) { fprintf(stderr, "input not multiple of block size\n"); return 2; }
        int64_t nb = n / bsz, k = nb * QK_K;
        float * y = malloc(k * sizeof(float));
        if (t1) dequantize_row_tq1_0((const block_tq1_0 *) buf, y, k);
        else    dequantize_row_tq2_0((const block_tq2_0 *) buf, y, k);
        fwrite(y, sizeof(float), k, stdout);
    }
    return 0;
}
