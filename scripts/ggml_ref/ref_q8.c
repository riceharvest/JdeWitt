// Q8_1 / Q8_K quantizer + Q8_K dequantizer reference.
// Copied VERBATIM from ggml-quants.c @ master sha256 1665bffa9c0ef0e692f8799523a729879960a6aa07adaa32f9efb41f7076a3da
// (quantize_row_q8_1_ref, quantize_row_q8_K_ref, dequantize_row_q8_K, nearest_int); block structs from
// ggml-common.h sha256 0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c.
// Harness substitutions only: ggml_half as uint16 via _Float16, GGML_FP32_TO_FP16, MIN/MAX, and main().
// Upstream has NO dequantize_row_q8_1; mode dq8_1 is the obvious harness decode d*qs (not upstream code).
// NOTE: at this pin block_q8_1 is {half d; half s; int8 qs[32]} = 36 bytes (not f32 d,s / 40).
//
// (y is calloc-ed: upstream leaves bsums uninitialised for all-zero Q8_K blocks; the codec emits zeros.)
// usage: ref_q8 q8_1|q8_K < weights.f32 > blocks.bin ; ref_q8 dq8_1|dq8_K < blocks.bin > f32.bin
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint16_t ggml_half;
#define GGML_RESTRICT
#define QK_K 256
#define QK8_1 32
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
static inline float GGML_FP16_TO_FP32(ggml_half h) { union { uint16_t u; _Float16 f; } b = { .u = h }; return (float)b.f; }
static inline ggml_half GGML_FP32_TO_FP16(float f) { union { _Float16 f; uint16_t u; } b = { .f = (_Float16)f }; return b.u; }
typedef struct { ggml_half d; ggml_half s; int8_t qs[QK8_1]; } block_q8_1;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K;
static_assert(sizeof(block_q8_1) == 36, "q8_1");
static_assert(sizeof(block_q8_K) == 292, "q8_K");

static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

void quantize_row_q8_1_ref(const float * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k) {
    assert(QK8_1 == 32);
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_1; j++) {
            const float v = x[i*QK8_1 + j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        int sum = 0;

        for (int j = 0; j < QK8_1/2; ++j) {
            const float v0 = x[i*QK8_1           + j]*id;
            const float v1 = x[i*QK8_1 + QK8_1/2 + j]*id;

            y[i].qs[          j] = roundf(v0);
            y[i].qs[QK8_1/2 + j] = roundf(v1);

            sum += y[i].qs[          j];
            sum += y[i].qs[QK8_1/2 + j];
        }

        y[i].s = GGML_FP32_TO_FP16(sum*d);
    }
}

void quantize_row_q8_K_ref(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        float max = 0;
        float amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax; max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            x += QK_K;
            continue;
        }
        //const float iscale = -128.f/max;
        // We need this change for IQ2_XXS, else the AVX implementation becomes very awkward
        const float iscale = -127.f/max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale*x[j]);
            y[i].qs[j] = MIN(127, v);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j*16 + ii];
            }
            y[i].bsums[j] = sum;
        }
        y[i].d = 1/iscale;
        x += QK_K;
    }
}

void dequantize_row_q8_K(const block_q8_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < QK_K; ++j) {
            *y++ = x[i].d * x[i].qs[j];
        }
    }
}

static uint8_t * slurp(size_t * n) {
    size_t cap = 1 << 20, len = 0; uint8_t * b = malloc(cap); size_t r;
    while ((r = fread(b + len, 1, cap - len, stdin)) > 0) { len += r; if (len == cap) { cap *= 2; b = realloc(b, cap); } }
    *n = len; return b;
}

int main(int argc, char ** argv) {
    if (argc != 2) { fprintf(stderr, "mode q8_1|q8_K|dq8_1|dq8_K\n"); return 2; }
    const char * m = argv[1]; size_t n; uint8_t * in = slurp(&n);
    if (!strcmp(m, "q8_1")) {
        if (n % (4 * QK8_1)) return 2;
        int64_t k = n / 4; block_q8_1 * y = calloc(1, k / QK8_1 * sizeof(block_q8_1));
        quantize_row_q8_1_ref((const float *)in, y, k);
        fwrite(y, sizeof(block_q8_1), k / QK8_1, stdout);
    } else if (!strcmp(m, "q8_K")) {
        if (n % (4 * QK_K)) return 2;
        int64_t k = n / 4; block_q8_K * y = calloc(1, k / QK_K * sizeof(block_q8_K));
        quantize_row_q8_K_ref((const float *)in, y, k);
        fwrite(y, sizeof(block_q8_K), k / QK_K, stdout);
    } else if (!strcmp(m, "dq8_1")) {
        size_t nb = n / sizeof(block_q8_1); const block_q8_1 * x = (const block_q8_1 *)in;
        for (size_t i = 0; i < nb; i++) for (int j = 0; j < QK8_1; j++) { float v = GGML_FP16_TO_FP32(x[i].d) * x[i].qs[j]; fwrite(&v, 4, 1, stdout); }
    } else if (!strcmp(m, "dq8_K")) {
        size_t nb = n / sizeof(block_q8_K); float * y = malloc(nb * QK_K * 4);
        dequantize_row_q8_K((const block_q8_K *)in, y, nb * QK_K);
        fwrite(y, 4, nb * QK_K, stdout);
    } else return 2;
    return 0;
}
