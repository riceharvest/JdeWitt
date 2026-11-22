// MXFP4 / NVFP4 scalar quantizer + dequantizer reference.
// Copied from ggml-quants.c @ master sha256 1665bffa9c0ef0e692f8799523a729879960a6aa07adaa32f9efb41f7076a3da
// (best_index_mxfp4, quantize_row_mxfp4_ref, quantize_row_nvfp4_ref,
// dequantize_row_mxfp4, dequantize_row_nvfp4), with harness substitutions only: block
// structs + kvalues_fp4 from ggml-common.h (sha 0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c),
// ggml_e8m0_to_fp32_half / ggml_ue4m3_to_fp32 / ggml_fp32_to_ue4m3 copied from ggml-impl.h
// (llama.cpp-upstream b10f9ca58; ggml-impl.h is not in the pinned set), GGML_RESTRICT,
// and a stdin/stdout main().
//
// usage: ref_fp4 mxfp4|nvfp4 < weights.f32 > blocks.bin
//        ref_fp4 dq-mxfp4|dq-nvfp4 < blocks.bin > weights.f32
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGML_RESTRICT

#define QK_MXFP4 32
typedef struct { uint8_t e; uint8_t qs[QK_MXFP4/2]; } block_mxfp4;
static_assert(sizeof(block_mxfp4) == 17, "mxfp4");
#define QK_NVFP4 64
#define QK_NVFP4_SUB 16
typedef struct { uint8_t d[QK_NVFP4/QK_NVFP4_SUB]; uint8_t qs[QK_NVFP4/2]; } block_nvfp4;
static_assert(sizeof(block_nvfp4) == 36, "nvfp4");

static const int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

// Equal to ggml_e8m0_to_fp32/2
// Useful with MXFP4 quantization since the E0M2 values are doubled
static inline float ggml_e8m0_to_fp32_half(uint8_t x) {
    uint32_t bits;

    // For x < 2: use precomputed denormal patterns
    if (x < 2) {
        // 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
        bits = 0x00200000 << x;
    }
    // For x >= 2: normalized exponent adjustment
    else {
        // 0.5 * 2^(x-127) = 2^(x-128) = normalized with exponent (x-1)
        bits = (uint32_t)(x - 1) << 23;
    }
    // Note: NaNs are not handled here

    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

#define GGML_E8M0_TO_FP32(x) ggml_e8m0_to_fp32(x)
#define GGML_E8M0_TO_FP32_HALF(x) ggml_e8m0_to_fp32_half(x)

// UE4M3: unsigned, 4 exp bits (bias=7), 3 mantissa bits
// Returns value * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 * E2M1_float)
static inline float ggml_ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7F) {
        return 0.0f;
    }
    int   exp = (x >> 3) & 0xF;
    int   man = x & 0x7;
    float raw;
    if (exp == 0) {
        raw = ldexpf((float) man, -9);
    } else {
        raw = ldexpf(1.0f + (float) man / 8.0f, exp - 7);
    }
    return raw * 0.5f;
}

static inline uint8_t ggml_fp32_to_ue4m3(float x) {
    if (!(x > 0.0f)) {
        return 0;
    }
    if (x > 448.0f) {
        x = 448.0f;
    }
    uint32_t bits;
    memcpy(&bits, &x, 4);
    int fp32_exp  = ((bits >> 23) & 0xFF) - 127;
    int fp32_man  = (bits >> 20) & 0x7;
    int ue4m3_exp = fp32_exp + 7;
    if (ue4m3_exp <= 0) {
        // subnormal: value = man * 2^-9, man = round(x * 2^9)
        int man = (int) (x * 512.0f + 0.5f);
        if (man > 7) {
            man = 7;
        }
        if (man < 1) {
            return 0;
        }
        return (uint8_t) man;
    }
    if (ue4m3_exp >= 15) {
        return 0x7E;
    }
    int round_bit = (bits >> 19) & 1;
    int ue4m3_man = fp32_man + round_bit;
    if (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        if (ue4m3_exp >= 15) {
            return 0x7E;
        }
    }
    return (uint8_t) ((ue4m3_exp << 3) | ue4m3_man);
}

static inline int best_index_mxfp4(float x, float e) {
    int best_index = 0;
    float best_err = fabsf(kvalues_mxfp4[0]*e - x);
    for (int i = 1; i < 16; i++) {
        float err = fabsf(kvalues_mxfp4[i]*e - x);
        if (err < best_err) {
            best_index = i;
            best_err = err;
        }
    }
    return best_index;
}

void quantize_row_mxfp4_ref(const float * GGML_RESTRICT x, block_mxfp4 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK_MXFP4;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];

            if (amax < fabsf(v)) {
                amax = fabsf(v);
            }
        }

        const uint8_t e = amax > 0.0f ? (uint8_t) (floorf(log2f(amax)) - 2 + 127) : 0;

        const float d = GGML_E8M0_TO_FP32_HALF(e);

        y[i].e = e;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t x0 = best_index_mxfp4(x[i*qk + 0    + j], d);
            const uint8_t x1 = best_index_mxfp4(x[i*qk + qk/2 + j], d);

            y[i].qs[j]  = x0;
            y[i].qs[j] |= x1 << 4;
        }
    }
}

void quantize_row_nvfp4_ref(const float * GGML_RESTRICT x, block_nvfp4 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK_NVFP4;
    static const int qk_sub = QK_NVFP4_SUB;
    static const int n_sub = QK_NVFP4 / QK_NVFP4_SUB;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        for (int s = 0; s < n_sub; s++) {
            const float * xb = x + i*qk + s*qk_sub;

            float amax = 0.0f;
            for (int j = 0; j < qk_sub; j++) {
                if (amax < fabsf(xb[j])) {
                    amax = fabsf(xb[j]);
                }
            }

            // UE4M3 scale: amax / 6.0 maps the max E2M1 value (6.0) to amax
            const uint8_t ue = ggml_fp32_to_ue4m3(amax / 6.0f);
            y[i].d[s] = ue;
            const float d = ggml_ue4m3_to_fp32(ue);

            for (int j = 0; j < qk_sub/2; ++j) {
                const uint8_t x0 = best_index_mxfp4(xb[0        + j], d);
                const uint8_t x1 = best_index_mxfp4(xb[qk_sub/2 + j], d);

                y[i].qs[s*(qk_sub/2) + j] = x0 | (x1 << 4);
            }
        }
    }
}

void dequantize_row_mxfp4(const block_mxfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK_MXFP4;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_E8M0_TO_FP32_HALF(x[i].e);

        for (int j = 0; j < qk/2; ++j) {
            const int8_t x0 = kvalues_mxfp4[x[i].qs[j] & 0x0F];
            const int8_t x1 = kvalues_mxfp4[x[i].qs[j] >>   4];

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

void dequantize_row_nvfp4(const block_nvfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK_NVFP4;
    static const int qk_sub = QK_NVFP4_SUB;
    static const int n_sub = QK_NVFP4 / QK_NVFP4_SUB;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        for (int s = 0; s < n_sub; s++) {
            const float d = ggml_ue4m3_to_fp32(x[i].d[s]);
            float * yb = y + i*qk + s*qk_sub;

            for (int j = 0; j < qk_sub/2; ++j) {
                const int8_t v0 = kvalues_mxfp4[x[i].qs[s*(qk_sub/2) + j] & 0x0F];
                const int8_t v1 = kvalues_mxfp4[x[i].qs[s*(qk_sub/2) + j] >>   4];

                yb[j + 0       ] = v0*d;
                yb[j + qk_sub/2] = v1*d;
            }
        }
    }
}

int main(int argc, char ** argv) {
    if (argc != 2) { fprintf(stderr, "usage: ref_fp4 mxfp4|nvfp4|dq-mxfp4|dq-nvfp4\n"); return 2; }
    const int mx = !strcmp(argv[1], "mxfp4") || !strcmp(argv[1], "dq-mxfp4");
    const int dq = !strncmp(argv[1], "dq-", 3);
    if (!mx && strcmp(argv[1], "nvfp4") && strcmp(argv[1], "dq-nvfp4")) { fprintf(stderr, "bad mode\n"); return 2; }
    const size_t qk = mx ? QK_MXFP4 : QK_NVFP4, bs = mx ? sizeof(block_mxfp4) : sizeof(block_nvfp4);
    const size_t cap = 1 << 14;
    float * fbuf = malloc(cap * qk * sizeof(float));
    uint8_t * bbuf = malloc(cap * bs);
    size_t n;
    if (!dq) {
        while ((n = fread(fbuf, sizeof(float), cap * qk, stdin)) > 0) {
            if (n % qk) { fprintf(stderr, "not multiple of %zu\n", qk); return 2; }
            if (mx) quantize_row_mxfp4_ref(fbuf, (block_mxfp4 *) bbuf, (int64_t) n);
            else    quantize_row_nvfp4_ref(fbuf, (block_nvfp4 *) bbuf, (int64_t) n);
            size_t want = n / qk * bs;
            if (fwrite(bbuf, 1, want, stdout) != want) return 2;
        }
    } else {
        while ((n = fread(bbuf, 1, cap * bs, stdin)) > 0) {
            if (n % bs) { fprintf(stderr, "not multiple of %zu\n", bs); return 2; }
            size_t k = n / bs * qk;
            if (mx) dequantize_row_mxfp4((const block_mxfp4 *) bbuf, fbuf, (int64_t) k);
            else    dequantize_row_nvfp4((const block_nvfp4 *) bbuf, fbuf, (int64_t) k);
            if (fwrite(fbuf, sizeof(float), k, stdout) != k) return 2;
        }
    }
    return 0;
}
