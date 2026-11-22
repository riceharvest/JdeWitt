// IQ4_NL / IQ4_XS scalar quantizer reference.
// quantize_row_iq4_nl_impl copied VERBATIM from ggml-quants.c @
// https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-quants.c
// sha256 1665bffa9c0ef0e692f8799523a729879960a6aa07adaa32f9efb41f7076a3da,
// with harness substitutions only: ggml_fp16_t via _Float16, block structs from
// ggml-common.h, and a stdin/stdout main().  kvalues_iq4nl is the table from
// ggml-common.h at the same revision.
//
// usage: ref_iq4 iq4_nl|iq4_xs < weights.f32 > blocks.bin
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGML_RESTRICT
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define QK4_NL 32
#define QK_K 256
#define GROUP_MAX_EPS 1e-15f

typedef uint16_t ggml_fp16_t;
static inline ggml_fp16_t GGML_FP32_TO_FP16(float f) {
    _Float16 h = (_Float16)f;
    ggml_fp16_t out;
    memcpy(&out, &h, sizeof(out));
    return out;
}

typedef struct { ggml_fp16_t d; uint8_t qs[QK4_NL/2]; } block_iq4_nl;
typedef struct { ggml_fp16_t d; uint16_t scales_h; uint8_t scales_l[QK_K/64]; uint8_t qs[QK_K/2]; } block_iq4_xs;
static_assert(sizeof(block_iq4_nl) == 18, "iq4_nl");
static_assert(sizeof(block_iq4_xs) == 136, "iq4_xs");

static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static inline int best_index_iq4nl(const float * GGML_RESTRICT mid, float x) {
    int l = 15;
    for (int k = 0; k < 15; ++k) l -= x < mid[k];
    return l;
}

static void quantize_row_iq4_nl_impl(const int super_block_size, const int block_size, const float * GGML_RESTRICT x,
        ggml_fp16_t * dh, uint8_t * q4, uint16_t * scales_h, uint8_t * scales_l,
        float * scales, float * weight, uint8_t * L,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    float sigma2 = 0;
    for (int j = 0; j < super_block_size; ++j) sigma2 += x[j]*x[j];
    sigma2 *= 2.f/super_block_size;

    memset(q4, 0, super_block_size/2);
    dh[0] = GGML_FP32_TO_FP16(0.f);

    float mid[15];
    float fval[16];
    for (int k = 0; k < 15; ++k) mid[k] = 0.5f*(values[k] + values[k+1]);
    for (int k = 0; k < 16; ++k) fval[k] = values[k];

    float max_scale = 0, amax_scale = 0;
    for (int ib = 0; ib < super_block_size/block_size; ++ib) {
        const float * xb = x + ib*block_size;
        uint8_t * Lb = L + ib*block_size;
        if (quant_weights) {
            const float * qw = quant_weights + ib*block_size;
            for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        } else {
            for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
        }
        float amax = 0, max = 0;
        for (int j = 0; j < block_size; ++j) {
            float ax = fabsf(xb[j]);
            if (ax > amax) {
                amax = ax; max = xb[j];
            }
        }
        if (amax < GROUP_MAX_EPS) {
            scales[ib] = 0;
            continue;
        }
        float d = ntry > 0 ? -max/values[0] : max/values[0];
        const int nt = ntry >= 0 ? 2*ntry + 2 : 1;
        float idt[16], sqx[16], sq2[16];
        idt[0] = 1/d;
        for (int t = 1; t < nt; ++t) idt[t] = (t - 1 - ntry + values[0])/max;
        for (int t = 0; t < nt; ++t) sqx[t] = sq2[t] = 0;
        for (int j = 0; j < block_size; ++j) {
            Lb[j] = best_index_iq4nl(mid, idt[0]*xb[j]);
        }
        for (int j = 0; j < block_size; ++j) {
            float w = weight[j];
            for (int t = 0; t < nt; ++t) {
                float q = fval[best_index_iq4nl(mid, idt[t]*xb[j])];
                sqx[t] += w*q*xb[j];
                sq2[t] += w*q*q;
            }
        }
        d = sq2[0] > 0 ? sqx[0]/sq2[0] : 0.f;
        float best = d*sqx[0];
        for (int t = 1; t < nt; ++t) {
            if (sq2[t] > 0 && sqx[t]*sqx[t] > best*sq2[t]) {
                d = sqx[t]/sq2[t]; best = d * sqx[t];
            }
        }
        scales[ib] = d;
        float abs_d = fabsf(d);
        if (abs_d > amax_scale) {
            amax_scale = abs_d; max_scale = d;
        }
    }

    if (super_block_size/block_size > 1) {
        int nb = super_block_size/block_size;
        memset(scales_h, 0, ((nb+7)/8)*sizeof(uint16_t));
        float d = -max_scale/32;
        dh[0] = GGML_FP32_TO_FP16(d);
        float id = d ? 1/d : 0.f;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            int l = nearest_int(id*scales[ib]);
            l = MAX(-32, MIN(31, l));
            float dl = d * l;
            float idl = dl ? 1/dl : 0.f;
            uint8_t * Lb = L + ib*block_size;
            const float * xb = x + ib*block_size;
            for (int j = 0; j < block_size; ++j) {
                Lb[j] = best_index_iq4nl(mid, idl*xb[j]);
            }
            l += 32;
            uint8_t l_l = l & 0xf;
            uint8_t l_h = l >>  4;
            if (ib%2 == 0) scales_l[ib/2] = l_l;
            else scales_l[ib/2] |= (l_l << 4);
            scales_h[ib/8] |= (l_h << 2*(ib%8));
        }
    } else {
        dh[0] = GGML_FP32_TO_FP16(scales[0]);
        if (ntry > 0) {
            float id = scales[0] ? 1/scales[0] : 0;
            for (int j = 0; j < super_block_size; ++j) {
                L[j] = best_index_iq4nl(mid, id*x[j]);
            }
        }
    }

    for (int i = 0; i < super_block_size/32; ++i) {
        for (int j = 0; j < 16; ++j) {
            q4[16*i + j] = L[32*i + j] | (L[32*i + 16 + j] << 4);
        }
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: ref_iq4 iq4_nl|iq4_xs\n"); return 2; }
    int xs = !strcmp(argv[1], "iq4_xs");
    size_t els = xs ? QK_K : QK4_NL;
    size_t bs = xs ? sizeof(block_iq4_xs) : sizeof(block_iq4_nl);
    size_t cap = 1 << 15;
    float * fbuf = malloc(cap * els * sizeof(float));
    uint8_t * bbuf = malloc(cap * bs);
    uint8_t L[QK_K];
    float scales[QK_K/32];
    float weight[32];
    size_t n;
    while ((n = fread(fbuf, sizeof(float), cap * els, stdin)) > 0) {
        if (n % els) { fprintf(stderr, "not multiple of %zu\n", els); return 2; }
        size_t blocks = n / els;
        for (size_t b = 0; b < blocks; ++b) {
            if (xs) {
                block_iq4_xs * y = (block_iq4_xs *)bbuf + b;
                quantize_row_iq4_nl_impl(QK_K, 32, fbuf + b*els, &y->d, y->qs, &y->scales_h,
                                         y->scales_l, scales, weight, L, kvalues_iq4nl, NULL, 7);
            } else {
                block_iq4_nl * y = (block_iq4_nl *)bbuf + b;
                uint16_t unused_h; uint8_t * unused_l = NULL; float scale;
                quantize_row_iq4_nl_impl(QK4_NL, 32, fbuf + b*els, &y->d, y->qs, &unused_h,
                                         unused_l, &scale, weight, L, kvalues_iq4nl, NULL, -1);
            }
        }
        size_t want = blocks * bs;
        if (fwrite(bbuf, 1, want, stdout) != want) return 2;
    }
    return 0;
}
