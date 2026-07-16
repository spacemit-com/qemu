/*
 * XSmtsfu helper implementations
 *
 * Copyright (c) 2026
 *
 * Aligned with the AMEv0.6 cmodel SFU implementation:
 * - mfex2.s / mflg2.s / mfrcp.s use 64/64/128-segment LUTs with
 *   2nd-order polynomial evaluation in double precision.
 * - mftanh.s uses per-exponent-bucket LUTs (exp in [-10, 2], 8
 *   subsegments per bucket) with 2nd-order polynomial evaluation.
 * - mfrcp.s falls back to an exact softfloat division for zero,
 *   infinity and NaN inputs.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2, as published by the Free Software Foundation.
 */

#include "qemu/osdep.h"
#include "qemu/host-utils.h"
#include "qemu/bitops.h"
#include "cpu.h"
#include "exec/helper-proto.h"
#include "fpu/softfloat.h"
#include <math.h>

#define xsmtsfu_env_tlenb(env)  ame_cfg_tlenb(xsmtsfu_env_cfg(env))
#define xsmtsfu_env_cfg(env)    (&env_archcpu(env)->cfg)

static inline uint8_t *xsmtsfu_tile_ptr(CPURISCVState *env, uint32_t id)
{
    g_assert(id < AME_NR_TILES);
    return (uint8_t *)env->ame_tile + id * xsmtsfu_env_tlenb(env);
}

static inline float xsmtsfu_f32_to_host(uint32_t raw)
{
    union {
        uint32_t i;
        float f;
    } u = { .i = raw };

    return u.f;
}

static inline uint32_t xsmtsfu_host_to_f32(float f)
{
    union {
        float f;
        uint32_t i;
    } u = { .f = f };

    return u.i;
}

typedef struct amev06_poly2_t {
    double c0;
    double c1;
    double c2;
} amev06_poly2_t;

enum {
    AMEV06_SFU_EXP2_SEGMENTS = 64,
    AMEV06_SFU_LOG2_SEGMENTS = 64,
    AMEV06_SFU_RCP_SEGMENTS = 128,
    AMEV06_SFU_TANH_SUBSEGMENTS = 8,
    AMEV06_SFU_TANH_MIN_EXP = -10,
    AMEV06_SFU_TANH_MAX_EXP = 2,
    AMEV06_SFU_TANH_EXP_BUCKETS =
        AMEV06_SFU_TANH_MAX_EXP - AMEV06_SFU_TANH_MIN_EXP + 1,
};

#include "xsmtsfu_lut_tables.c.inc"

typedef uint32_t (*xsmtsfu_tile32_unary_fn)(CPURISCVState *env, uint32_t raw);

static double xsmtsfu_poly2_eval(const amev06_poly2_t *poly, double x)
{
    return (poly->c2 * x + poly->c1) * x + poly->c0;
}

static void xsmtsfu_unpack_positive_f32(uint32_t raw, int *exp2,
                                        uint32_t *mant23)
{
    uint32_t exp_field = (raw >> 23) & 0xffu;
    uint32_t frac = raw & 0x7fffffu;

    g_assert((raw & 0x80000000u) == 0);
    g_assert((raw & 0x7fffffffu) != 0);
    g_assert(exp_field != 0xffu);

    if (exp_field != 0) {
        *exp2 = (int)exp_field - 127;
        *mant23 = frac;
        return;
    }

    {
        uint32_t tmp = frac;
        uint32_t highest = 0;

        while (tmp >>= 1) {
            ++highest;
        }

        *exp2 = (int)highest - 149;
        *mant23 = (frac << (23 - highest)) & 0x7fffffu;
    }
}

/*
 * Exact 1/x via softfloat for zero / infinity / NaN inputs.  The cmodel
 * only propagates softfloat NX/UF/OF/NV flags (no DZ); mirror that by
 * merging the same subset into env->fp_status.
 */
static uint32_t xsmtsfu_mfrcp_s_exact(CPURISCVState *env, uint32_t raw)
{
    float_status scratch = env->fp_status;
    float32 res;

    scratch.float_exception_flags = 0;
    res = float32_div(make_float32(0x3f800000u), make_float32(raw), &scratch);
    env->fp_status.float_exception_flags |=
        scratch.float_exception_flags &
        (float_flag_inexact | float_flag_underflow |
         float_flag_overflow | float_flag_invalid);
    return float32_val(res);
}

static uint32_t xsmtsfu_mfex2_s_scalar(CPURISCVState *env, uint32_t raw)
{
    float x = xsmtsfu_f32_to_host(raw);
    double frac;
    double poly;
    double result;
    double n_d;
    uint32_t frac23;
    uint32_t idx;
    uint32_t low;

    (void)env;

    if (!isfinite(x)) {
        return xsmtsfu_host_to_f32(exp2f(x));
    }

    n_d = floor((double)x);
    frac = (double)x - n_d;
    frac23 = (uint32_t)llround(ldexp(frac, 23));
    if (frac23 >= (1u << 23)) {
        frac23 = 0;
        n_d += 1.0;
    }

    idx = frac23 >> 17;
    low = frac23 & ((1u << 17) - 1u);
    poly = xsmtsfu_poly2_eval(&amev06_sfu_exp2_lut[idx],
                              ldexp((double)low, -23));
    result = ldexp(poly, (int)n_d);
    return xsmtsfu_host_to_f32((float)result);
}

static uint32_t xsmtsfu_mftanh_s_scalar(CPURISCVState *env, uint32_t raw)
{
    uint32_t mag = raw & 0x7fffffffu;
    int exp2;
    uint32_t mant23;
    uint32_t subidx;
    uint32_t low;
    double offset;
    double result;

    (void)env;

    if (((raw >> 23) & 0xffu) == 0xffu) {
        if (mag > 0x7f800000u) {
            return xsmtsfu_host_to_f32(tanhf(xsmtsfu_f32_to_host(raw)));
        }
        return (raw & 0x80000000u) ? 0xbf800000u : 0x3f800000u;
    }
    if (mag <= 0x3a800000u) {
        return raw;
    }
    if (mag >= 0x41000000u) {
        return (raw & 0x80000000u) ? 0xbf800000u : 0x3f800000u;
    }

    xsmtsfu_unpack_positive_f32(mag, &exp2, &mant23);
    g_assert(exp2 >= AMEV06_SFU_TANH_MIN_EXP &&
             exp2 <= AMEV06_SFU_TANH_MAX_EXP);

    subidx = mant23 >> 20;
    low = mant23 & ((1u << 20) - 1u);
    offset = ldexp((double)low, exp2 - 23);
    result = xsmtsfu_poly2_eval(
        &amev06_sfu_tanh_lut[exp2 - AMEV06_SFU_TANH_MIN_EXP][subidx], offset);
    if (raw & 0x80000000u) {
        result = -result;
    }
    return xsmtsfu_host_to_f32((float)result);
}

static uint32_t xsmtsfu_mflg2_s_scalar(CPURISCVState *env, uint32_t raw)
{
    int exp2;
    uint32_t mant23;
    uint32_t idx;
    uint32_t low;
    double result;

    (void)env;

    if ((raw & 0x80000000u) != 0 || (raw & 0x7fffffffu) == 0 ||
        ((raw >> 23) & 0xffu) == 0xffu) {
        return xsmtsfu_host_to_f32(log2f(xsmtsfu_f32_to_host(raw)));
    }

    xsmtsfu_unpack_positive_f32(raw, &exp2, &mant23);

    idx = mant23 >> 17;
    low = mant23 & ((1u << 17) - 1u);
    result = (double)exp2 +
             xsmtsfu_poly2_eval(&amev06_sfu_log2_lut[idx],
                                ldexp((double)low, -23));
    return xsmtsfu_host_to_f32((float)result);
}

static uint32_t xsmtsfu_mfrcp_s_scalar(CPURISCVState *env, uint32_t raw)
{
    uint32_t sign = raw & 0x80000000u;
    uint32_t mag = raw & 0x7fffffffu;
    int exp2;
    uint32_t mant23;
    uint32_t idx;
    uint32_t low;
    double result;

    if (mag == 0 || ((raw >> 23) & 0xffu) == 0xffu) {
        return xsmtsfu_mfrcp_s_exact(env, raw);
    }

    xsmtsfu_unpack_positive_f32(mag, &exp2, &mant23);
    idx = mant23 >> 16;
    low = mant23 & ((1u << 16) - 1u);
    result = ldexp(xsmtsfu_poly2_eval(&amev06_sfu_rcp_lut[idx],
                                      ldexp((double)low, -23)),
                   -exp2);
    if (sign != 0) {
        result = -result;
    }

    return xsmtsfu_host_to_f32((float)result);
}

static void xsmtsfu_tile32(CPURISCVState *env, uint32_t md,
                           uint32_t ms2, xsmtsfu_tile32_unary_fn op)
{
    uint32_t *dst = (uint32_t *)xsmtsfu_tile_ptr(env, md);
    const uint32_t *src = (const uint32_t *)xsmtsfu_tile_ptr(env, ms2);
    uint32_t elems = xsmtsfu_env_tlenb(env) / sizeof(uint32_t);
    uint32_t i;

    for (i = 0; i < elems; i++) {
        dst[i] = op(env, src[i]);
    }
}

void HELPER(xsmtsfu_mfex2_s)(CPURISCVState *env, uint32_t md, uint32_t ms2)
{
    xsmtsfu_tile32(env, md, ms2, xsmtsfu_mfex2_s_scalar);
}

void HELPER(xsmtsfu_mftanh_s)(CPURISCVState *env, uint32_t md, uint32_t ms2)
{
    xsmtsfu_tile32(env, md, ms2, xsmtsfu_mftanh_s_scalar);
}

void HELPER(xsmtsfu_mflg2_s)(CPURISCVState *env, uint32_t md, uint32_t ms2)
{
    xsmtsfu_tile32(env, md, ms2, xsmtsfu_mflg2_s_scalar);
}

void HELPER(xsmtsfu_mfrcp_s)(CPURISCVState *env, uint32_t md, uint32_t ms2)
{
    xsmtsfu_tile32(env, md, ms2, xsmtsfu_mfrcp_s_scalar);
}
