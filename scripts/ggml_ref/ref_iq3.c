// IQ3_XXS / IQ3_S scalar quantizer reference.
// Copied from ggml-quants.c @ master sha256 1665bffa9c0ef0e692f8799523a729879960a6aa07adaa32f9efb41f7076a3da
// (iq3 data structs + iq3xs_init_impl + iq3_find_best_neighbour + the IQ3_XXS and
// IQ3_S reference quantizers), with harness substitutions only: ggml_fp16_t via
// _Float16, block structs from ggml-common.h, GGML_ASSERT/GGML_ABORT, and a
// stdin/stdout main().
//
// usage: ref_iq3 iq3_xxs|iq3_s < weights.f32 > blocks.bin
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
#define GGML_ASSERT(x) assert(x)
#define GGML_ABORT(msg) do { fprintf(stderr, "%s\n", msg); abort(); } while (0)
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f

typedef uint16_t ggml_fp16_t;
static inline ggml_fp16_t GGML_FP32_TO_FP16(float f) {
    _Float16 h = (_Float16)f; ggml_fp16_t out; memcpy(&out, &h, sizeof(out)); return out;
}
static inline float GGML_FP16_TO_FP32(ggml_fp16_t h) {
    _Float16 v; memcpy(&v, &h, sizeof(v)); return (float)v;
}

typedef struct { ggml_fp16_t d; uint8_t qs[3*QK_K/8]; } block_iq3_xxs;
typedef struct { ggml_fp16_t d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32]; uint8_t signs[QK_K/8]; uint8_t scales[QK_K/64]; } block_iq3_s;
static_assert(sizeof(block_iq3_xxs) == 98, "iq3_xxs");
static_assert(sizeof(block_iq3_s) == 110, "iq3_s");

static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

// ============================================= 3-bit using D4 lattice
//

typedef struct {
    uint32_t * grid;
    int      * map;
    uint16_t * neighbours;
} iq3_entry_t;

static iq3_entry_t iq3_data[2] = {
    {NULL, NULL, NULL},
    {NULL, NULL, NULL},
};

static inline int iq3_data_index(int grid_size) {
    (void)grid_size;
    GGML_ASSERT(grid_size == 256 || grid_size == 512);
    return grid_size == 256 ? 0 : 1;
}

static int iq3_compare_func(const void * left, const void * right) {
    const int * l = (const int *)left;
    const int * r = (const int *)right;
    return l[0] < r[0] ? -1 : l[0] > r[0] ? 1 : l[1] < r[1] ? -1 : l[1] > r[1] ? 1 : 0;
}

void iq3xs_init_impl(int grid_size) {
    const int gindex = iq3_data_index(grid_size);
    if (iq3_data[gindex].grid) {
        return;
    }
    static const uint16_t kgrid_256[256] = {
            0,     2,     4,     9,    11,    15,    16,    18,    25,    34,    59,    61,    65,    67,    72,    74,
           81,    85,    88,    90,    97,   108,   120,   128,   130,   132,   137,   144,   146,   153,   155,   159,
          169,   175,   189,   193,   199,   200,   202,   213,   248,   267,   287,   292,   303,   315,   317,   321,
          327,   346,   362,   413,   436,   456,   460,   462,   483,   497,   513,   515,   520,   522,   529,   531,
          536,   538,   540,   551,   552,   576,   578,   585,   592,   594,   641,   643,   648,   650,   657,   664,
          698,   704,   706,   720,   729,   742,   758,   769,   773,   808,   848,   852,   870,   889,   901,   978,
          992,  1024,  1026,  1033,  1035,  1040,  1042,  1046,  1049,  1058,  1089,  1091,  1093,  1096,  1098,  1105,
         1112,  1139,  1143,  1144,  1152,  1154,  1161,  1167,  1168,  1170,  1183,  1184,  1197,  1217,  1224,  1228,
         1272,  1276,  1309,  1323,  1347,  1367,  1377,  1404,  1473,  1475,  1486,  1509,  1537,  1544,  1546,  1553,
         1555,  1576,  1589,  1594,  1600,  1602,  1616,  1625,  1636,  1638,  1665,  1667,  1672,  1685,  1706,  1722,
         1737,  1755,  1816,  1831,  1850,  1856,  1862,  1874,  1901,  1932,  1950,  1971,  2011,  2032,  2052,  2063,
         2077,  2079,  2091,  2095,  2172,  2192,  2207,  2208,  2224,  2230,  2247,  2277,  2308,  2345,  2356,  2389,
         2403,  2424,  2501,  2504,  2506,  2520,  2570,  2593,  2616,  2624,  2630,  2646,  2669,  2700,  2714,  2746,
         2754,  2795,  2824,  2835,  2839,  2874,  2882,  2905,  2984,  3028,  3042,  3092,  3108,  3110,  3124,  3153,
         3185,  3215,  3252,  3288,  3294,  3364,  3397,  3434,  3483,  3523,  3537,  3587,  3589,  3591,  3592,  3610,
         3626,  3670,  3680,  3722,  3749,  3754,  3776,  3789,  3803,  3824,  3857,  3873,  3904,  3906,  3924,  3992,
    };
    static const uint16_t kgrid_512[512] = {
            0,     1,     2,     5,     7,     8,     9,    10,    12,    14,    16,    17,    21,    27,    32,    34,
           37,    39,    41,    43,    48,    50,    57,    60,    63,    64,    65,    66,    68,    72,    73,    77,
           80,    83,    87,    89,    93,   100,   113,   117,   122,   128,   129,   133,   135,   136,   139,   142,
          145,   149,   152,   156,   162,   165,   167,   169,   171,   184,   187,   195,   201,   205,   208,   210,
          217,   219,   222,   228,   232,   234,   247,   249,   253,   256,   267,   271,   273,   276,   282,   288,
          291,   297,   312,   322,   324,   336,   338,   342,   347,   353,   357,   359,   374,   379,   390,   393,
          395,   409,   426,   441,   448,   450,   452,   464,   466,   470,   475,   488,   492,   512,   513,   514,
          516,   520,   521,   523,   525,   527,   528,   530,   537,   540,   542,   556,   558,   561,   570,   576,
          577,   579,   582,   584,   588,   593,   600,   603,   609,   616,   618,   632,   638,   640,   650,   653,
          655,   656,   660,   666,   672,   675,   685,   688,   698,   705,   708,   711,   712,   715,   721,   727,
          728,   732,   737,   754,   760,   771,   773,   778,   780,   793,   795,   802,   806,   808,   812,   833,
          840,   843,   849,   856,   858,   873,   912,   916,   919,   932,   934,   961,   963,   968,   970,   977,
          989,   993,  1010,  1016,  1024,  1025,  1027,  1029,  1031,  1032,  1034,  1036,  1038,  1041,  1043,  1047,
         1048,  1050,  1057,  1059,  1061,  1064,  1066,  1079,  1080,  1083,  1085,  1088,  1090,  1096,  1099,  1103,
         1106,  1109,  1113,  1116,  1122,  1129,  1153,  1156,  1159,  1169,  1171,  1176,  1183,  1185,  1195,  1199,
         1209,  1212,  1216,  1218,  1221,  1225,  1234,  1236,  1241,  1243,  1250,  1256,  1270,  1281,  1287,  1296,
         1299,  1306,  1309,  1313,  1338,  1341,  1348,  1353,  1362,  1375,  1376,  1387,  1400,  1408,  1410,  1415,
         1425,  1453,  1457,  1477,  1481,  1494,  1496,  1507,  1512,  1538,  1545,  1547,  1549,  1551,  1554,  1561,
         1563,  1565,  1570,  1572,  1575,  1577,  1587,  1593,  1601,  1603,  1605,  1612,  1617,  1619,  1632,  1648,
         1658,  1662,  1664,  1674,  1680,  1690,  1692,  1704,  1729,  1736,  1740,  1745,  1747,  1751,  1752,  1761,
         1763,  1767,  1773,  1787,  1795,  1801,  1806,  1810,  1817,  1834,  1840,  1844,  1857,  1864,  1866,  1877,
         1882,  1892,  1902,  1915,  1934,  1953,  1985,  1987,  2000,  2002,  2013,  2048,  2052,  2058,  2064,  2068,
         2071,  2074,  2081,  2088,  2104,  2114,  2119,  2121,  2123,  2130,  2136,  2141,  2147,  2153,  2157,  2177,
         2179,  2184,  2189,  2193,  2203,  2208,  2223,  2226,  2232,  2244,  2249,  2251,  2256,  2258,  2265,  2269,
         2304,  2306,  2324,  2335,  2336,  2361,  2373,  2375,  2385,  2418,  2443,  2460,  2480,  2504,  2509,  2520,
         2531,  2537,  2562,  2568,  2572,  2578,  2592,  2596,  2599,  2602,  2614,  2620,  2625,  2627,  2629,  2634,
         2641,  2650,  2682,  2688,  2697,  2707,  2712,  2718,  2731,  2754,  2759,  2760,  2775,  2788,  2793,  2805,
         2811,  2817,  2820,  2832,  2842,  2854,  2890,  2902,  2921,  2923,  2978,  3010,  3012,  3026,  3081,  3083,
         3085,  3097,  3099,  3120,  3136,  3152,  3159,  3188,  3210,  3228,  3234,  3245,  3250,  3256,  3264,  3276,
         3281,  3296,  3349,  3363,  3378,  3392,  3395,  3420,  3440,  3461,  3488,  3529,  3531,  3584,  3588,  3591,
         3600,  3602,  3614,  3616,  3628,  3634,  3650,  3657,  3668,  3683,  3685,  3713,  3716,  3720,  3726,  3729,
         3736,  3753,  3778,  3802,  3805,  3819,  3841,  3845,  3851,  3856,  3880,  3922,  3938,  3970,  3993,  4032,
    };

    const int kmap_size = 4096;
    const int nwant = grid_size == 256 ? 2 : 3;
    const uint16_t * kgrid = grid_size == 256 ? kgrid_256 : kgrid_512;
    uint32_t * kgrid_q3xs;
    int      * kmap_q3xs;
    uint16_t * kneighbors_q3xs;

    //printf("================================================================= %s(grid_size = %d)\n", __func__, grid_size);
    uint32_t * the_grid = (uint32_t *)malloc(grid_size*sizeof(uint32_t));
    for (int k = 0; k < grid_size; ++k) {
        int8_t * pos = (int8_t *)(the_grid + k);
        for (int i = 0; i < 4; ++i) {
            int l = (kgrid[k] >> 3*i) & 0x7;
            pos[i] = 2*l + 1;
        }
    }
    kgrid_q3xs = the_grid;
    iq3_data[gindex].grid = the_grid;
    kmap_q3xs = (int *)malloc(kmap_size*sizeof(int));
    iq3_data[gindex].map = kmap_q3xs;
    for (int i = 0; i < kmap_size; ++i) kmap_q3xs[i] = -1;
    uint32_t aux32;
    uint8_t * aux8 = (uint8_t *)&aux32;
    for (int i = 0; i < grid_size; ++i) {
        aux32 = kgrid_q3xs[i];
        uint16_t index = 0;
        for (int k=0; k<4; ++k) {
            uint16_t q = (aux8[k] - 1)/2;
            index |= (q << 3*k);
        }
        kmap_q3xs[index] = i;
    }
    // See explanation of parallelism in iq2xs_init_impl
    int * n_per_i = (int *)malloc(kmap_size*sizeof(int));
    GGML_ASSERT(n_per_i);
    int num_neighbors = 0, num_not_in_map = 0;
#ifdef GGML_USE_OPENMP
    #pragma omp parallel reduction(+:num_neighbors,num_not_in_map)
#endif
    {
        int * dist2 = (int *)malloc(2*grid_size*sizeof(int));
        GGML_ASSERT(dist2);
        int8_t pos[4];
        int i;
#ifdef GGML_USE_OPENMP
        #pragma omp for schedule(dynamic, 64)
#endif
        for (i = 0; i < kmap_size; ++i) {
            if (kmap_q3xs[i] >= 0) {
                n_per_i[i] = 0;
                continue;
            }
            ++num_not_in_map;
            for (int k = 0; k < 4; ++k) {
                int l = (i >> 3*k) & 0x7;
                pos[k] = 2*l + 1;
            }
            for (int j = 0; j < grid_size; ++j) {
                const int8_t * pg = (const int8_t *)(kgrid_q3xs + j);
                int d2 = 0;
                for (int k = 0; k < 4; ++k) d2 += (pg[k] - pos[k])*(pg[k] - pos[k]);
                dist2[2*j+0] = d2;
                dist2[2*j+1] = j;
            }
            qsort(dist2, grid_size, 2*sizeof(int), iq3_compare_func);
            int n = 0; int d2 = dist2[0];
            int nhave = 1;
            for (int j = 0; j < grid_size; ++j) {
                if (dist2[2*j] > d2) {
                    if (nhave == nwant) break;
                    d2 = dist2[2*j];
                    ++nhave;
                }
                ++n;
            }
            n_per_i[i] = n;
            num_neighbors += n;
        }
        free(dist2);
    }
    //printf("%s: %d neighbours in total\n", __func__, num_neighbors);
    kneighbors_q3xs = (uint16_t *)malloc((num_neighbors + num_not_in_map)*sizeof(uint16_t));
    iq3_data[gindex].neighbours = kneighbors_q3xs;

    int * offsets = (int *)malloc(kmap_size*sizeof(int));
    GGML_ASSERT(offsets);
    int counter = 0;
    for (int i = 0; i < kmap_size; ++i) {
        if (kmap_q3xs[i] >= 0) {
            offsets[i] = -1;
            continue;
        }
        offsets[i] = counter;
        counter += 1 + n_per_i[i];
    }

#ifdef GGML_USE_OPENMP
    #pragma omp parallel
#endif
    {
        int * dist2 = (int *)malloc(2*grid_size*sizeof(int));
        GGML_ASSERT(dist2);
        int8_t pos[4];
        int i;
#ifdef GGML_USE_OPENMP
        #pragma omp for schedule(dynamic, 64)
#endif
        for (i = 0; i < kmap_size; ++i) {
            if (kmap_q3xs[i] >= 0) continue;
            for (int k = 0; k < 4; ++k) {
                int l = (i >> 3*k) & 0x7;
                pos[k] = 2*l + 1;
            }
            for (int j = 0; j < grid_size; ++j) {
                const int8_t * pg = (const int8_t *)(kgrid_q3xs + j);
                int d2 = 0;
                for (int k = 0; k < 4; ++k) d2 += (pg[k] - pos[k])*(pg[k] - pos[k]);
                dist2[2*j+0] = d2;
                dist2[2*j+1] = j;
            }
            qsort(dist2, grid_size, 2*sizeof(int), iq3_compare_func);
            int local_counter = offsets[i];
            kmap_q3xs[i] = -(local_counter + 1);
            int d2 = dist2[0];
            uint16_t * start = &kneighbors_q3xs[local_counter++];
            int n = 0, nhave = 1;
            for (int j = 0; j < grid_size; ++j) {
                if (dist2[2*j] > d2) {
                    if (nhave == nwant) break;
                    d2 = dist2[2*j];
                    ++nhave;
                }
                kneighbors_q3xs[local_counter++] = dist2[2*j+1];
                ++n;
            }
            *start = n;
        }
        free(dist2);
    }
    free(offsets);
    free(n_per_i);
}

void iq3xs_free_impl(int grid_size) {
    GGML_ASSERT(grid_size == 256 || grid_size == 512);
    const int gindex = iq3_data_index(grid_size);
    if (iq3_data[gindex].grid) {
        free(iq3_data[gindex].grid);       iq3_data[gindex].grid = NULL;
        free(iq3_data[gindex].map);        iq3_data[gindex].map  = NULL;
        free(iq3_data[gindex].neighbours); iq3_data[gindex].neighbours = NULL;
    }
}

static int iq3_find_best_neighbour(const uint16_t * GGML_RESTRICT neighbours, const uint32_t * GGML_RESTRICT grid,
        const float * GGML_RESTRICT xval, const float * GGML_RESTRICT weight, float scale, int8_t * GGML_RESTRICT L) {
    int num_neighbors = neighbours[0];
    GGML_ASSERT(num_neighbors > 0);
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 4; ++i) {
            float q = pg[i];
            float diff = scale*q - xval[i];
            d2 += weight[i]*diff*diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2; grid_index = neighbours[j];
        }
    }
    GGML_ASSERT(grid_index >= 0);
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 4; ++i) L[i] = (pg[i] - 1)/2;
    return grid_index;
}

static void quantize_row_iq3_xxs_impl(int grid_size, const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n,
        const float * GGML_RESTRICT quant_weights) {

    const int gindex = iq3_data_index(grid_size);

    const uint32_t * kgrid_q3xs      = iq3_data[gindex].grid;
    const int      * kmap_q3xs       = iq3_data[gindex].map;
    const uint16_t * kneighbors_q3xs = iq3_data[gindex].neighbours;

    //GGML_ASSERT(quant_weights   && "missing quantization weights");
    GGML_ASSERT(kgrid_q3xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kmap_q3xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q3xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    const int kMaxQ = 8;

    const int64_t nbl = n/QK_K;

    ggml_fp16_t * dh;
    uint8_t * qs;
    int block_size;
    if (grid_size == 256) {
        block_iq3_xxs * y = vy;
        dh = &y->d;
        qs = y->qs;
        block_size = sizeof(block_iq3_xxs);
    } else {
        block_iq3_s * y = vy;
        dh = &y->d;
        qs = y->qs;
        block_size = sizeof(block_iq3_s);
    }
    int quant_size = block_size - sizeof(ggml_fp16_t);

    float scales[QK_K/32];
    float weight[32];
    float xval[32];
    int8_t L[32];
    int8_t Laux[32];
    float  waux[32];
    bool   is_on_grid[8];
    bool   is_on_grid_aux[8];
    uint8_t block_signs[8];
    uint8_t q3[3*(QK_K/8)+QK_K/32];
    uint32_t * scales_and_signs = (uint32_t *)(q3 + QK_K/4);
    uint8_t  * qh = q3 + 3*(QK_K/8);

    for (int ibl = 0; ibl < nbl; ++ibl) {

        dh[0] = GGML_FP32_TO_FP16(0.f);
        memset(q3, 0, 3*QK_K/8+QK_K/32);

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = 2*sumx2/QK_K;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const float * xb = xbl + 32*ib;
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*ibl + 32*ib;
                for (int i = 0; i < 32; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } else {
                for (int i = 0; i < 32; ++i) weight[i] = xb[i]*xb[i];
            }
            for (int i = 0; i < 32; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < 4; ++k) {
                int nflip = 0;
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    else {
                        xval[8*k + i] = -xb[8*k + i]; ++nflip; s |= (1 << i);
                    }
                }
                if (nflip%2) {
                    int imin = 0; float min = weight[8*k+imin]*xb[8*k+imin]*xb[8*k+imin];
                    for (int i = 1; i < 8; ++i) {
                        float ax = weight[8*k+i]*xb[8*k+i]*xb[8*k+i];
                        if (ax < min) {
                            min = ax; imin = i;
                        }
                    }
                    xval[8*k+imin] = -xval[8*k+imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }
            float max = xval[0];
            for (int i = 1; i < 32; ++i) max = MAX(max, xval[i]);
            memset(L, 0, 32);
            if (max < GROUP_MAX_EPS_IQ3_XXS) {
                scales[ib] = 0;
                continue;
            }
            float best = 0;
            float scale = max/(2*kMaxQ-1);
            for (int k = 0; k < 8; ++k) is_on_grid[k] = true;
            for (int is = -15; is <= 15; ++is) {
                float id = (2*kMaxQ-1+is*0.2f)/max;
                float this_scale = 1/id;
                for (int k = 0; k < 8; ++k) {
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        Laux[4*k+i] = MAX(0, MIN(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) u |= (Laux[4*k+i] << 3*i);
                    int grid_index = kmap_q3xs[u];
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, this_scale, Laux + 4*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float w = weight[i];
                    float q = 2*Laux[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    for (int i = 0; i < 32; ++i) L[i] = Laux[i];
                    for (int k = 0; k <  8; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            int n_not_ongrid = 0;
            for (int k = 0; k < 8; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < 8; ++k) {
                    if (is_on_grid[k]) continue;
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        l = MAX(0, MIN(kMaxQ-1, l));
                        u |= (l << 3*i);
                    }
                    int grid_index = kmap_q3xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, scale, L + 4*k);
                    }
                    const int8_t * pg = (const int8_t *)(kgrid_q3xs + grid_index);
                    for (int i = 0; i < 4; ++i) L[4*k+i] = (pg[i] - 1)/2;
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float w = weight[i];
                    float q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            if (scale < 0) {
                // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the scale)
                // and correspondingly flip quant signs.
                scale = -scale;
                for (int k = 0; k < 4; ++k) block_signs[k] = (~block_signs[k]) & 127;
            }
            for (int k = 0; k < 8; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) u |= (L[4*k+i] << 3*i);
                int grid_index = kmap_q3xs[u];
                if (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 4; ++i) printf(" %d", L[4*k+i]);
                    printf("\n");
                    GGML_ABORT("fatal error");
                }
                if (grid_size == 256) {
                    q3[8*ib+k] = grid_index;
                } else {
                    q3[8*ib+k] = grid_index & 255;
                    qh[ib] |= ((grid_index >> 8) << k);
                }

            }
            scales_and_signs[ib] = block_signs[0] | (block_signs[1] << 7) | (block_signs[2] << 14) | (block_signs[3] << 21);
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
            max_scale = MAX(max_scale, scale);
        }

        if (!max_scale) {
            memset(qs, 0, quant_size);
            dh += block_size/sizeof(ggml_fp16_t);
            qs += block_size;
            continue;
        }

        float d = max_scale/31;
        dh[0] = GGML_FP32_TO_FP16(d * 1.0125f);  // small improvement via this fudge factor
        float id = 1/d;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            int l = nearest_int(0.5f*(id*scales[ib]-1));
            l = MAX(0, MIN(15, l));
            scales_and_signs[ib] |= ((uint32_t)l << 28);
        }
        memcpy(qs, q3, quant_size);

        dh += block_size/sizeof(ggml_fp16_t);
        qs += block_size;

    }
}

size_t quantize_iq3_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    int64_t nblock = n_per_row/QK_K;
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq3_xxs_impl(256, src, qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += nblock*sizeof(block_iq3_xxs);
    }
    return nrow * nblock * sizeof(block_iq3_xxs);
}

void quantize_row_iq3_xxs_ref(const float * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_row_iq3_xxs_impl(256, x, y, k, NULL);
}

static void quantize_row_iq3_s_impl(int block_size, const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int n,
        const float * GGML_RESTRICT quant_weights,
        float   * scales,
        float   * weight,
        float   * xval,
        int8_t  * L,
        int8_t  * Laux,
        float   * waux,
        bool    * is_on_grid,
        bool    * is_on_grid_aux,
        uint8_t * block_signs) {

    const int gindex = iq3_data_index(512);

    const uint32_t * kgrid_q3xs      = iq3_data[gindex].grid;
    const int      * kmap_q3xs       = iq3_data[gindex].map;
    const uint16_t * kneighbors_q3xs = iq3_data[gindex].neighbours;

    //GGML_ASSERT(quant_weights   && "missing quantization weights");
    GGML_ASSERT(kgrid_q3xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kmap_q3xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q3xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    const int kMaxQ = 8;

    const int64_t nbl = n/QK_K;

    block_iq3_s * y = vy;

    const int bs4 = block_size/4;
    const int bs8 = block_size/8;

    for (int ibl = 0; ibl < nbl; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq3_s));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        uint8_t * qs = y[ibl].qs;
        uint8_t * qh = y[ibl].qh;
        uint8_t * signs = y[ibl].signs;

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = 2*sumx2/QK_K;

        for (int ib = 0; ib < QK_K/block_size; ++ib) {
            const float * xb = xbl + block_size*ib;
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*ibl + block_size*ib;
                for (int i = 0; i < block_size; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } else {
                for (int i = 0; i < block_size; ++i) weight[i] = xb[i]*xb[i];
            }
            for (int i = 0; i < block_size; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < bs8; ++k) {
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    else {
                        xval[8*k + i] = -xb[8*k + i]; s |= (1 << i);
                    }
                }
                block_signs[k] = s;
            }
            float max = xval[0];
            for (int i = 1; i < block_size; ++i) max = MAX(max, xval[i]);
            memset(L, 0, block_size);
            if (!max) {
                scales[ib] = 0;
                continue;
            }
            float best = 0;
            float scale = max/(2*kMaxQ-1);
            for (int k = 0; k < bs4; ++k) is_on_grid[k] = false;
            for (int is = -9; is <= 9; ++is) {
                float id = (2*kMaxQ-1+is*0.2f)/max;
                float this_scale = 1/id;
                for (int k = 0; k < bs4; ++k) {
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        Laux[4*k+i] = MAX(0, MIN(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) u |= (Laux[4*k+i] << 3*i);
                    int grid_index = kmap_q3xs[u];
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, this_scale, Laux + 4*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < block_size; ++i) {
                    float w = weight[i];
                    float q = 2*Laux[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    for (int i = 0; i < block_size; ++i) L[i] = Laux[i];
                    for (int k = 0; k < bs4; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            int n_not_ongrid = 0;
            for (int k = 0; k < bs4; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < bs4; ++k) {
                    //if (is_on_grid[k]) continue;
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        l = MAX(0, MIN(kMaxQ-1, l));
                        u |= (l << 3*i);
                    }
                    int grid_index = kmap_q3xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, scale, L + 4*k);
                    }
                    const int8_t * pg = (const int8_t *)(kgrid_q3xs + grid_index);
                    for (int i = 0; i < 4; ++i) L[4*k+i] = (pg[i] - 1)/2;
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < block_size; ++i) {
                    float w = weight[i];
                    float q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            if (scale < 0) {
                // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the scale)
                // and correspondingly flip quant signs.
                scale = -scale;
                for (int k = 0; k < bs8; ++k) block_signs[k] = ~block_signs[k];
            }
            for (int k = 0; k < bs4; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) u |= (L[4*k+i] << 3*i);
                int grid_index = kmap_q3xs[u];
                if (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 4; ++i) printf(" %d", L[4*k+i]);
                    printf("\n");
                    GGML_ABORT("fatal error");
                }
                qs[k] = grid_index & 255;
                qh[(ib*bs4+k)/8] |= ((grid_index >> 8) << ((ib*bs4+k)%8));
            }
            qs += bs4;
            for (int k = 0; k < bs8; ++k) signs[k] = block_signs[k];
            signs += bs8;
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
            max_scale = MAX(max_scale, scale);
        }

        if (!max_scale) {
            continue;
        }

        float d = max_scale/31;
        y[ibl].d = GGML_FP32_TO_FP16(d * 1.033f);
        float id = 1/d;
        for (int ib = 0; ib < QK_K/block_size; ib += 2) {
            int l1 = nearest_int(0.5f*(id*scales[ib+0]-1));
            l1 = MAX(0, MIN(15, l1));
            int l2 = nearest_int(0.5f*(id*scales[ib+1]-1));
            l2 = MAX(0, MIN(15, l2));
            y[ibl].scales[ib/2] = l1 | (l2 << 4);
        }

    }
}

#define IQ3S_BLOCK_SIZE 32
size_t quantize_iq3_s(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    int64_t nblock = n_per_row/QK_K;
    float scales[QK_K/IQ3S_BLOCK_SIZE];
    float weight[IQ3S_BLOCK_SIZE];
    float xval[IQ3S_BLOCK_SIZE];
    int8_t L[IQ3S_BLOCK_SIZE];
    int8_t Laux[IQ3S_BLOCK_SIZE];
    float  waux[IQ3S_BLOCK_SIZE];
    bool   is_on_grid[IQ3S_BLOCK_SIZE/4];
    bool   is_on_grid_aux[IQ3S_BLOCK_SIZE/4];
    uint8_t block_signs[IQ3S_BLOCK_SIZE/8];
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq3_s_impl(IQ3S_BLOCK_SIZE, src, qrow, n_per_row, quant_weights,
                scales, weight, xval, L, Laux, waux, is_on_grid, is_on_grid_aux, block_signs);
        src += n_per_row;
        qrow += nblock*sizeof(block_iq3_s);
    }
    return nrow * nblock * sizeof(block_iq3_s);
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: ref_iq3 iq3_xxs|iq3_s\n"); return 2; }
    int is_s = !strcmp(argv[1], "iq3_s");
    size_t bs = is_s ? 110 : 98;
    size_t cap = 1 << 15;
    float * fbuf = malloc(cap * QK_K * sizeof(float));
    uint8_t * bbuf = malloc(cap * bs);
    iq3xs_init_impl(is_s ? 512 : 256);
    size_t n;
    while ((n = fread(fbuf, sizeof(float), cap * QK_K, stdin)) > 0) {
        if (n % QK_K) { fprintf(stderr, "not multiple of 256\n"); return 2; }
        if (is_s) quantize_iq3_s(fbuf, bbuf, 1, (int64_t)n, NULL);
        else quantize_row_iq3_xxs_impl(256, fbuf, bbuf, (int64_t)n, NULL);
        size_t want = n / QK_K * bs;
        if (fwrite(bbuf, 1, want, stdout) != want) return 2;
    }
    return 0;
}
