/*
 * AME (Attached Matrix Extension) translation functions
 *
 * Copyright (c) 2025
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2, as published by the Free Software Foundation.
 */

#include "tcg/tcg-op-gvec.h"

static bool ame_check_shape_limit(target_ulong val, target_ulong max)
{
    return val <= max;
}

static bool ame_check_tile_reg_2(uint32_t reg)
{
    return reg < AME_NR_TILES;
}

static bool ame_check_acc_reg_2(uint32_t reg)
{
    return reg < AME_NR_ACCS;
}

static bool ame_check_matrix_reg_3(uint32_t reg)
{
    return reg < AME_NR_TILES + AME_NR_ACCS;
}

static bool ame_is_tile_reg_3(uint32_t reg)
{
    return reg < AME_NR_TILES;
}

static bool ame_is_acc_reg_3(uint32_t reg)
{
    return reg >= AME_NR_TILES && reg < AME_NR_TILES + AME_NR_ACCS;
}

static bool ame_same_reg_class_3(uint32_t a, uint32_t b)
{
    return (ame_is_tile_reg_3(a) && ame_is_tile_reg_3(b)) ||
           (ame_is_acc_reg_3(a) && ame_is_acc_reg_3(b));
}

static bool ame_require(bool cond, DisasContext *s)
{
    if (!cond) {
        gen_exception_illegal(s);
    }

    return cond;
}

#define GEN_XSMTAME_WHOLE_LOAD_TRANS(NAME, ARGTYPE, HELPER)                 \
    static bool NAME(DisasContext *s, ARGTYPE *a)                           \
    {                                                                       \
        if (!require_xsmtame(s) ||                                          \
            !ame_require(ame_check_matrix_reg_3(a->mx_md3), s)) {           \
            return false;                                                   \
        }                                                                   \
        gen_helper_##HELPER(tcg_env, tcg_constant_i32(a->mx_md3),           \
                            get_gpr(s, a->mx_rs1, EXT_NONE));               \
        mark_ms_dirty(s);                                                   \
        return true;                                                        \
    }

#define GEN_XSMTAME_WHOLE_STORE_TRANS(NAME, ARGTYPE, HELPER)                \
    static bool NAME(DisasContext *s, ARGTYPE *a)                           \
    {                                                                       \
        if (!require_xsmtame(s) ||                                          \
            !ame_require(ame_check_matrix_reg_3(a->mx_md3), s)) {           \
            return false;                                                   \
        }                                                                   \
        gen_helper_##HELPER(tcg_env, tcg_constant_i32(a->mx_md3),           \
                            get_gpr(s, a->mx_rs1, EXT_NONE));               \
        mark_ms_dirty(s);                                                   \
        return true;                                                        \
    }

GEN_XSMTAME_WHOLE_LOAD_TRANS(trans_mlme8, arg_ame_load_matrix,
                             xsmtame_mlme8)
GEN_XSMTAME_WHOLE_LOAD_TRANS(trans_mlme16, arg_ame_load_matrix,
                             xsmtame_mlme16)

static bool trans_mlae8(DisasContext *s, arg_ame_load_tile_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_tile_reg_2(a->mx_td2), s)) {
        return false;
    }

    TCGv addr = get_gpr(s, a->mx_rs1, EXT_NONE);
    TCGv stride = get_gpr(s, a->mx_rs2, EXT_NONE);
    gen_helper_xsmtame_mlae8(tcg_env, tcg_constant_i32(a->mx_td2),
                                  addr, stride);
    mark_ms_dirty(s);
    return true;
}

static bool trans_mlae16(DisasContext *s, arg_ame_load_tile_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_tile_reg_2(a->mx_td2), s)) {
        return false;
    }
    gen_helper_xsmtame_mlae16(tcg_env,
                          tcg_constant_i32(a->mx_td2),
                          get_gpr(s, a->mx_rs1, EXT_NONE),
                          get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mlae32(DisasContext *s, arg_ame_load_tile_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_tile_reg_2(a->mx_td2), s)) {
        return false;
    }
    gen_helper_xsmtame_mlae32(tcg_env,
                          tcg_constant_i32(a->mx_td2),
                          get_gpr(s, a->mx_rs1, EXT_NONE),
                          get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mlbe8(DisasContext *s, arg_ame_load_tile_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_tile_reg_2(a->mx_td2), s)) {
        return false;
    }

    TCGv addr = get_gpr(s, a->mx_rs1, EXT_NONE);
    TCGv stride = get_gpr(s, a->mx_rs2, EXT_NONE);
    gen_helper_xsmtame_mlbe8(tcg_env, tcg_constant_i32(a->mx_td2),
                                  addr, stride);
    mark_ms_dirty(s);
    return true;
}

static bool trans_mlbe16(DisasContext *s, arg_ame_load_tile_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_tile_reg_2(a->mx_td2), s)) {
        return false;
    }
    gen_helper_xsmtame_mlbe16(tcg_env,
                          tcg_constant_i32(a->mx_td2),
                          get_gpr(s, a->mx_rs1, EXT_NONE),
                          get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mlbe32(DisasContext *s, arg_ame_load_tile_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_tile_reg_2(a->mx_td2), s)) {
        return false;
    }
    gen_helper_xsmtame_mlbe32(tcg_env,
                          tcg_constant_i32(a->mx_td2),
                          get_gpr(s, a->mx_rs1, EXT_NONE),
                          get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mlate8(DisasContext *s, arg_ame_load_tile_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_tile_reg_2(a->mx_td2), s)) {
        return false;
    }

    gen_helper_xsmtame_mlate8(tcg_env, tcg_constant_i32(a->mx_td2),
                                   get_gpr(s, a->mx_rs1, EXT_NONE),
                                   get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mlate16(DisasContext *s, arg_ame_load_tile_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_tile_reg_2(a->mx_td2), s)) {
        return false;
    }
    gen_helper_xsmtame_mlate16(tcg_env,
                          tcg_constant_i32(a->mx_td2),
                          get_gpr(s, a->mx_rs1, EXT_NONE),
                          get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mlate32(DisasContext *s, arg_ame_load_tile_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_tile_reg_2(a->mx_td2), s)) {
        return false;
    }
    gen_helper_xsmtame_mlate32(tcg_env,
                          tcg_constant_i32(a->mx_td2),
                          get_gpr(s, a->mx_rs1, EXT_NONE),
                          get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mlbte8(DisasContext *s, arg_ame_load_tile_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_tile_reg_2(a->mx_td2), s)) {
        return false;
    }

    gen_helper_xsmtame_mlbte8(tcg_env, tcg_constant_i32(a->mx_td2),
                                   get_gpr(s, a->mx_rs1, EXT_NONE),
                                   get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mlbte32(DisasContext *s, arg_ame_load_tile_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_tile_reg_2(a->mx_td2), s)) {
        return false;
    }
    gen_helper_xsmtame_mlbte32(tcg_env,
                          tcg_constant_i32(a->mx_td2),
                          get_gpr(s, a->mx_rs1, EXT_NONE),
                          get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mlce8(DisasContext *s, arg_ame_load_acc_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_acc_reg_2(a->mx_ad2), s)) {
        return false;
    }

    gen_helper_xsmtame_mlce8(tcg_env, tcg_constant_i32(a->mx_ad2),
                                  get_gpr(s, a->mx_rs1, EXT_NONE),
                                  get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mlce16(DisasContext *s, arg_ame_load_acc_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_acc_reg_2(a->mx_ad2), s)) {
        return false;
    }

    gen_helper_xsmtame_mlce16(tcg_env, tcg_constant_i32(a->mx_ad2),
                                   get_gpr(s, a->mx_rs1, EXT_NONE),
                                   get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mlce32(DisasContext *s, arg_ame_load_acc_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_acc_reg_2(a->mx_ad2), s)) {
        return false;
    }

    gen_helper_xsmtame_mlce32(tcg_env, tcg_constant_i32(a->mx_ad2),
                                   get_gpr(s, a->mx_rs1, EXT_NONE),
                                   get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mlcte8(DisasContext *s, arg_ame_load_acc_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_acc_reg_2(a->mx_ad2), s)) {
        return false;
    }

    gen_helper_xsmtame_mlcte8(tcg_env, tcg_constant_i32(a->mx_ad2),
                                   get_gpr(s, a->mx_rs1, EXT_NONE),
                                   get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mlcte16(DisasContext *s, arg_ame_load_acc_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_acc_reg_2(a->mx_ad2), s)) {
        return false;
    }

    gen_helper_xsmtame_mlcte16(tcg_env, tcg_constant_i32(a->mx_ad2),
                                    get_gpr(s, a->mx_rs1, EXT_NONE),
                                    get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mlcte32(DisasContext *s, arg_ame_load_acc_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_acc_reg_2(a->mx_ad2), s)) {
        return false;
    }

    gen_helper_xsmtame_mlcte32(tcg_env, tcg_constant_i32(a->mx_ad2),
                                    get_gpr(s, a->mx_rs1, EXT_NONE),
                                    get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_msae16(DisasContext *s, arg_ame_store_tile_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_tile_reg_2(a->mx_td2), s)) {
        return false;
    }
    gen_helper_xsmtame_msae16(tcg_env,
                          tcg_constant_i32(a->mx_td2),
                          get_gpr(s, a->mx_rs1, EXT_NONE),
                          get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_msae32(DisasContext *s, arg_ame_store_tile_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_tile_reg_2(a->mx_td2), s)) {
        return false;
    }
    gen_helper_xsmtame_msae32(tcg_env,
                          tcg_constant_i32(a->mx_td2),
                          get_gpr(s, a->mx_rs1, EXT_NONE),
                          get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_msae8(DisasContext *s, arg_ame_store_tile_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_tile_reg_2(a->mx_td2), s)) {
        return false;
    }

    TCGv addr = get_gpr(s, a->mx_rs1, EXT_NONE);
    TCGv stride = get_gpr(s, a->mx_rs2, EXT_NONE);
    gen_helper_xsmtame_msae8(tcg_env, tcg_constant_i32(a->mx_td2),
                                  addr, stride);
    mark_ms_dirty(s);
    return true;
}

static bool trans_msbe16(DisasContext *s, arg_ame_store_tile_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_tile_reg_2(a->mx_td2), s)) {
        return false;
    }
    gen_helper_xsmtame_msbe16(tcg_env,
                          tcg_constant_i32(a->mx_td2),
                          get_gpr(s, a->mx_rs1, EXT_NONE),
                          get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_msbe32(DisasContext *s, arg_ame_store_tile_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_tile_reg_2(a->mx_td2), s)) {
        return false;
    }
    gen_helper_xsmtame_msbe32(tcg_env,
                          tcg_constant_i32(a->mx_td2),
                          get_gpr(s, a->mx_rs1, EXT_NONE),
                          get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_msbe8(DisasContext *s, arg_ame_store_tile_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_tile_reg_2(a->mx_td2), s)) {
        return false;
    }

    TCGv addr = get_gpr(s, a->mx_rs1, EXT_NONE);
    TCGv stride = get_gpr(s, a->mx_rs2, EXT_NONE);
    gen_helper_xsmtame_msbe8(tcg_env, tcg_constant_i32(a->mx_td2),
                                  addr, stride);
    mark_ms_dirty(s);
    return true;
}

static bool trans_msce8(DisasContext *s, arg_ame_store_acc_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_acc_reg_2(a->mx_ad2), s)) {
        return false;
    }

    gen_helper_xsmtame_msce8(tcg_env, tcg_constant_i32(a->mx_ad2),
                                  get_gpr(s, a->mx_rs1, EXT_NONE),
                                  get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_msce16(DisasContext *s, arg_ame_store_acc_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_acc_reg_2(a->mx_ad2), s)) {
        return false;
    }

    gen_helper_xsmtame_msce16(tcg_env, tcg_constant_i32(a->mx_ad2),
                                   get_gpr(s, a->mx_rs1, EXT_NONE),
                                   get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_msce32(DisasContext *s, arg_ame_store_acc_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_acc_reg_2(a->mx_ad2), s)) {
        return false;
    }

    gen_helper_xsmtame_msce32(tcg_env, tcg_constant_i32(a->mx_ad2),
                                   get_gpr(s, a->mx_rs1, EXT_NONE),
                                   get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_msate8(DisasContext *s, arg_ame_store_tile_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_tile_reg_2(a->mx_td2), s)) {
        return false;
    }

    gen_helper_xsmtame_msate8(tcg_env, tcg_constant_i32(a->mx_td2),
                                   get_gpr(s, a->mx_rs1, EXT_NONE),
                                   get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_msate32(DisasContext *s, arg_ame_store_tile_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_tile_reg_2(a->mx_td2), s)) {
        return false;
    }
    gen_helper_xsmtame_msate32(tcg_env,
                          tcg_constant_i32(a->mx_td2),
                          get_gpr(s, a->mx_rs1, EXT_NONE),
                          get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_msbte8(DisasContext *s, arg_ame_store_tile_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_tile_reg_2(a->mx_td2), s)) {
        return false;
    }

    gen_helper_xsmtame_msbte8(tcg_env, tcg_constant_i32(a->mx_td2),
                                   get_gpr(s, a->mx_rs1, EXT_NONE),
                                   get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mscte8(DisasContext *s, arg_ame_store_acc_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_acc_reg_2(a->mx_ad2), s)) {
        return false;
    }

    gen_helper_xsmtame_mscte8(tcg_env, tcg_constant_i32(a->mx_ad2),
                                   get_gpr(s, a->mx_rs1, EXT_NONE),
                                   get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mscte16(DisasContext *s, arg_ame_store_acc_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_acc_reg_2(a->mx_ad2), s)) {
        return false;
    }

    gen_helper_xsmtame_mscte16(tcg_env, tcg_constant_i32(a->mx_ad2),
                                    get_gpr(s, a->mx_rs1, EXT_NONE),
                                    get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mscte32(DisasContext *s, arg_ame_store_acc_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_acc_reg_2(a->mx_ad2), s)) {
        return false;
    }

    gen_helper_xsmtame_mscte32(tcg_env, tcg_constant_i32(a->mx_ad2),
                                    get_gpr(s, a->mx_rs1, EXT_NONE),
                                    get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

GEN_XSMTAME_WHOLE_STORE_TRANS(trans_msme8, arg_ame_store_matrix,
                              xsmtame_msme8)
GEN_XSMTAME_WHOLE_STORE_TRANS(trans_msme16, arg_ame_store_matrix,
                              xsmtame_msme16)

static bool trans_msbte16(DisasContext *s, arg_ame_store_tile_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_tile_reg_2(a->mx_td2), s)) {
        return false;
    }
    gen_helper_xsmtame_msbte16(tcg_env,
                          tcg_constant_i32(a->mx_td2),
                          get_gpr(s, a->mx_rs1, EXT_NONE),
                          get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_msbte32(DisasContext *s, arg_ame_store_tile_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_tile_reg_2(a->mx_td2), s)) {
        return false;
    }
    gen_helper_xsmtame_msbte32(tcg_env,
                          tcg_constant_i32(a->mx_td2),
                          get_gpr(s, a->mx_rs1, EXT_NONE),
                          get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_msate16(DisasContext *s, arg_ame_store_tile_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_tile_reg_2(a->mx_td2), s)) {
        return false;
    }
    gen_helper_xsmtame_msate16(tcg_env,
                          tcg_constant_i32(a->mx_td2),
                          get_gpr(s, a->mx_rs1, EXT_NONE),
                          get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mlbte16(DisasContext *s, arg_ame_load_tile_str *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_tile_reg_2(a->mx_td2), s)) {
        return false;
    }
    gen_helper_xsmtame_mlbte16(tcg_env,
                          tcg_constant_i32(a->mx_td2),
                          get_gpr(s, a->mx_rs1, EXT_NONE),
                          get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

GEN_XSMTAME_WHOLE_LOAD_TRANS(trans_mlme32, arg_ame_load_matrix,
                             xsmtame_mlme32)
GEN_XSMTAME_WHOLE_STORE_TRANS(trans_msme32, arg_ame_store_matrix,
                              xsmtame_msme32)

/*
 * ──────────────────────────────────────────
 *  GEMM
 * ──────────────────────────────────────────
 *
 * mmacc.w.b  ad, ms2, ms1 : acc[ad] += A(ms1) * B_T(ms2) (INT8→INT32)
 * mfmacc.s.h md, ms2, ms1 : acc[md] += A(ms1) * B_T(ms2) (FP16→FP32)
 *
 * Since acc = 4096B exceeds TCG gvec maxsz, we use direct helper calls
 * passing register indices rather than tcg_gen_gvec_3_ptr.
 */

static bool trans_mmacc_w_b(DisasContext *s, arg_ame_matmul_acc *a)
{
    if (!require_xsmtame(s)) {
        return false;
    }
    if (!ame_require(ame_check_acc_reg_2(a->mx_ad2) &&
                     ame_is_tile_reg_3(a->mx_ms1_3) &&
                     ame_is_tile_reg_3(a->mx_ms2_3), s)) {
        return false;
    }
    gen_helper_xsmtame_mmacc_w_b(tcg_env,
                              tcg_constant_i32(a->mx_ad2),
                              tcg_constant_i32(a->mx_ms2_3),
                              tcg_constant_i32(a->mx_ms1_3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mmaccu_w_b(DisasContext *s, arg_ame_matmul_acc *a)
{
    if (!require_xsmtame(s)) {
        return false;
    }
    if (!ame_require(ame_check_acc_reg_2(a->mx_ad2) &&
                     ame_is_tile_reg_3(a->mx_ms1_3) &&
                     ame_is_tile_reg_3(a->mx_ms2_3), s)) {
        return false;
    }
    gen_helper_xsmtame_mmaccu_w_b(tcg_env,
                               tcg_constant_i32(a->mx_ad2),
                               tcg_constant_i32(a->mx_ms2_3),
                               tcg_constant_i32(a->mx_ms1_3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mmaccus_w_b(DisasContext *s, arg_ame_matmul_acc *a)
{
    if (!require_xsmtame(s)) {
        return false;
    }
    if (!ame_require(ame_check_acc_reg_2(a->mx_ad2) &&
                     ame_is_tile_reg_3(a->mx_ms1_3) &&
                     ame_is_tile_reg_3(a->mx_ms2_3), s)) {
        return false;
    }
    gen_helper_xsmtame_mmaccus_w_b(tcg_env,
                                tcg_constant_i32(a->mx_ad2),
                                tcg_constant_i32(a->mx_ms2_3),
                                tcg_constant_i32(a->mx_ms1_3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mmaccsu_w_b(DisasContext *s, arg_ame_matmul_acc *a)
{
    if (!require_xsmtame(s)) {
        return false;
    }
    if (!ame_require(ame_check_acc_reg_2(a->mx_ad2) &&
                     ame_is_tile_reg_3(a->mx_ms1_3) &&
                     ame_is_tile_reg_3(a->mx_ms2_3), s)) {
        return false;
    }
    gen_helper_xsmtame_mmaccsu_w_b(tcg_env,
                                tcg_constant_i32(a->mx_ad2),
                                tcg_constant_i32(a->mx_ms2_3),
                                tcg_constant_i32(a->mx_ms1_3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mfmacc_h_e5(DisasContext *s, arg_ame_matmul_acc *a)
{
    if (!require_xsmtame(s)) {
        return false;
    }
    if (!ame_require(ame_check_acc_reg_2(a->mx_ad2) &&
                     ame_is_tile_reg_3(a->mx_ms1_3) &&
                     ame_is_tile_reg_3(a->mx_ms2_3), s)) {
        return false;
    }
    gen_set_rm(s, RISCV_FRM_DYN);
    gen_helper_xsmtame_mfmacc_h_e5(tcg_env,
                                tcg_constant_i32(a->mx_ad2),
                                tcg_constant_i32(a->mx_ms2_3),
                                tcg_constant_i32(a->mx_ms1_3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mfmacc_h_e4(DisasContext *s, arg_ame_matmul_acc *a)
{
    if (!require_xsmtame(s)) {
        return false;
    }
    if (!ame_require(ame_check_acc_reg_2(a->mx_ad2) &&
                     ame_is_tile_reg_3(a->mx_ms1_3) &&
                     ame_is_tile_reg_3(a->mx_ms2_3), s)) {
        return false;
    }
    gen_set_rm(s, RISCV_FRM_DYN);
    gen_helper_xsmtame_mfmacc_h_e4(tcg_env,
                                tcg_constant_i32(a->mx_ad2),
                                tcg_constant_i32(a->mx_ms2_3),
                                tcg_constant_i32(a->mx_ms1_3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mfmacc_bf16_e5(DisasContext *s, arg_ame_matmul_acc *a)
{
    if (!require_xsmtame(s)) {
        return false;
    }
    if (!ame_require(ame_check_acc_reg_2(a->mx_ad2) &&
                     ame_is_tile_reg_3(a->mx_ms1_3) &&
                     ame_is_tile_reg_3(a->mx_ms2_3), s)) {
        return false;
    }
    gen_set_rm(s, RISCV_FRM_DYN);
    gen_helper_xsmtame_mfmacc_bf16_e5(tcg_env,
                                   tcg_constant_i32(a->mx_ad2),
                                   tcg_constant_i32(a->mx_ms2_3),
                                   tcg_constant_i32(a->mx_ms1_3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mfmacc_bf16_e4(DisasContext *s, arg_ame_matmul_acc *a)
{
    if (!require_xsmtame(s)) {
        return false;
    }
    if (!ame_require(ame_check_acc_reg_2(a->mx_ad2) &&
                     ame_is_tile_reg_3(a->mx_ms1_3) &&
                     ame_is_tile_reg_3(a->mx_ms2_3), s)) {
        return false;
    }
    gen_set_rm(s, RISCV_FRM_DYN);
    gen_helper_xsmtame_mfmacc_bf16_e4(tcg_env,
                                   tcg_constant_i32(a->mx_ad2),
                                   tcg_constant_i32(a->mx_ms2_3),
                                   tcg_constant_i32(a->mx_ms1_3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mfmacc_h(DisasContext *s, arg_ame_matmul_acc *a)
{
    if (!require_xsmtame(s)) {
        return false;
    }
    if (!ame_require(ame_check_acc_reg_2(a->mx_ad2) &&
                     ame_is_tile_reg_3(a->mx_ms1_3) &&
                     ame_is_tile_reg_3(a->mx_ms2_3), s)) {
        return false;
    }
    gen_set_rm(s, RISCV_FRM_DYN);
    gen_helper_xsmtame_mfmacc_h(tcg_env,
                             tcg_constant_i32(a->mx_ad2),
                             tcg_constant_i32(a->mx_ms2_3),
                             tcg_constant_i32(a->mx_ms1_3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mfmacc_s_h(DisasContext *s, arg_ame_matmul_acc *a)
{
    if (!require_xsmtame(s)) {
        return false;
    }
    if (!ame_require(ame_check_acc_reg_2(a->mx_ad2) &&
                     ame_is_tile_reg_3(a->mx_ms1_3) &&
                     ame_is_tile_reg_3(a->mx_ms2_3), s)) {
        return false;
    }
    gen_set_rm(s, RISCV_FRM_DYN);
    gen_helper_xsmtame_mfmacc_s_h(tcg_env,
                              tcg_constant_i32(a->mx_ad2),
                              tcg_constant_i32(a->mx_ms2_3),
                              tcg_constant_i32(a->mx_ms1_3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mfmacc_s_bf16(DisasContext *s, arg_ame_matmul_acc *a)
{
    if (!require_xsmtame(s)) {
        return false;
    }
    if (!ame_require(ame_check_acc_reg_2(a->mx_ad2) &&
                     ame_is_tile_reg_3(a->mx_ms1_3) &&
                     ame_is_tile_reg_3(a->mx_ms2_3), s)) {
        return false;
    }
    gen_set_rm(s, RISCV_FRM_DYN);
    gen_helper_xsmtame_mfmacc_s_bf16(tcg_env,
                                  tcg_constant_i32(a->mx_ad2),
                                  tcg_constant_i32(a->mx_ms2_3),
                                  tcg_constant_i32(a->mx_ms1_3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mfmacc_s_e4(DisasContext *s, arg_ame_matmul_acc *a)
{
    if (!require_xsmtame(s)) {
        return false;
    }
    if (!ame_require(ame_check_acc_reg_2(a->mx_ad2) &&
                     ame_is_tile_reg_3(a->mx_ms1_3) &&
                     ame_is_tile_reg_3(a->mx_ms2_3), s)) {
        return false;
    }
    gen_set_rm(s, RISCV_FRM_DYN);
    gen_helper_xsmtame_mfmacc_s_e4(tcg_env,
                                tcg_constant_i32(a->mx_ad2),
                                tcg_constant_i32(a->mx_ms2_3),
                                tcg_constant_i32(a->mx_ms1_3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mfmacc_s_e5(DisasContext *s, arg_ame_matmul_acc *a)
{
    if (!require_xsmtame(s)) {
        return false;
    }
    if (!ame_require(ame_check_acc_reg_2(a->mx_ad2) &&
                     ame_is_tile_reg_3(a->mx_ms1_3) &&
                     ame_is_tile_reg_3(a->mx_ms2_3), s)) {
        return false;
    }
    gen_set_rm(s, RISCV_FRM_DYN);
    gen_helper_xsmtame_mfmacc_s_e5(tcg_env,
                                tcg_constant_i32(a->mx_ad2),
                                tcg_constant_i32(a->mx_ms2_3),
                                tcg_constant_i32(a->mx_ms1_3));
    mark_ms_dirty(s);
    return true;
}

/*
 * ──────────────────────────────────────────
 *  Config: msettile*i
 * ──────────────────────────────────────────
 *
 * msettilemi imm10 : mtilem = imm10, illegal if imm10 > HW_MAX_M
 * msettileni imm10 : mtilen = imm10, illegal if imm10 > HW_MAX_N
 * msettileki imm10 : mtilek = imm10, illegal if imm10 > TRLEN / 8
 *
 * These are pure TCG ops, no helper needed.
 */

static bool trans_msettilemi(DisasContext *s, arg_msettilemi *a)
{
    if (!require_xsmtame(s)) {
        return false;
    }
    target_ulong val = (target_ulong)a->mx_uimm10;
    if (!ame_require(ame_check_shape_limit(val, ame_cfg_rownum(s->cfg_ptr)), s)) {
        return false;
    }
    tcg_gen_st_tl(tcg_constant_tl(val),
                  tcg_env,
                  offsetof(CPURISCVState, mtilem));
    mark_ms_dirty(s);
    return true;
}

static bool trans_msettileni(DisasContext *s, arg_msettileni *a)
{
    if (!require_xsmtame(s)) {
        return false;
    }
    target_ulong val = (target_ulong)a->mx_uimm10;
    if (!ame_require(ame_check_shape_limit(val, ame_cfg_rownum(s->cfg_ptr)), s)) {
        return false;
    }
    tcg_gen_st_tl(tcg_constant_tl(val),
                  tcg_env,
                  offsetof(CPURISCVState, mtilen));
    mark_ms_dirty(s);
    return true;
}

static bool trans_msettileki(DisasContext *s, arg_msettileki *a)
{
    if (!require_xsmtame(s)) {
        return false;
    }
    target_ulong val = (target_ulong)a->mx_uimm10;
    if (!ame_require(ame_check_shape_limit(val, ame_cfg_kmax(s->cfg_ptr)), s)) {
        return false;
    }
    tcg_gen_st_tl(tcg_constant_tl(val),
                  tcg_env,
                  offsetof(CPURISCVState, mtilek));
    mark_ms_dirty(s);
    return true;
}

static bool trans_msettilem(DisasContext *s, arg_ame_settile_reg *a)
{
    if (!require_xsmtame(s)) {
        return false;
    }
    TCGv val = get_gpr(s, a->mx_rs1, EXT_NONE);
    TCGv limit = tcg_constant_tl(ame_cfg_rownum(s->cfg_ptr));
    TCGv ok = tcg_temp_new();
    TCGLabel *label_ok = gen_new_label();

    tcg_gen_setcond_tl(TCG_COND_LEU, ok, val, limit);
    tcg_gen_brcondi_tl(TCG_COND_NE, ok, 0, label_ok);
    gen_exception_illegal(s);
    gen_set_label(label_ok);
    tcg_gen_st_tl(val, tcg_env, offsetof(CPURISCVState, mtilem));
    mark_ms_dirty(s);
    return true;
}

static bool trans_msettilen(DisasContext *s, arg_ame_settile_reg *a)
{
    if (!require_xsmtame(s)) {
        return false;
    }
    TCGv val = get_gpr(s, a->mx_rs1, EXT_NONE);
    TCGv limit = tcg_constant_tl(ame_cfg_rownum(s->cfg_ptr));
    TCGv ok = tcg_temp_new();
    TCGLabel *label_ok = gen_new_label();

    tcg_gen_setcond_tl(TCG_COND_LEU, ok, val, limit);
    tcg_gen_brcondi_tl(TCG_COND_NE, ok, 0, label_ok);
    gen_exception_illegal(s);
    gen_set_label(label_ok);
    tcg_gen_st_tl(val, tcg_env, offsetof(CPURISCVState, mtilen));
    mark_ms_dirty(s);
    return true;
}

static bool trans_msettilek(DisasContext *s, arg_ame_settile_reg *a)
{
    if (!require_xsmtame(s)) {
        return false;
    }
    TCGv val = get_gpr(s, a->mx_rs1, EXT_NONE);
    TCGv limit = tcg_constant_tl(ame_cfg_kmax(s->cfg_ptr));
    TCGv ok = tcg_temp_new();
    TCGLabel *label_ok = gen_new_label();

    tcg_gen_setcond_tl(TCG_COND_LEU, ok, val, limit);
    tcg_gen_brcondi_tl(TCG_COND_NE, ok, 0, label_ok);
    gen_exception_illegal(s);
    gen_set_label(label_ok);
    tcg_gen_st_tl(val, tcg_env, offsetof(CPURISCVState, mtilek));
    mark_ms_dirty(s);
    return true;
}

/*
 * ──────────────────────────────────────────
 *  MISC: mmov
 * ──────────────────────────────────────────
 */

static bool trans_mmov_mm(DisasContext *s, arg_ame_misc_mm *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms1_3), s)) {
        return false;
    }

    gen_helper_xsmtame_mmov_mm(tcg_env,
                                    tcg_constant_i32(a->mx_md3),
                                    tcg_constant_i32(a->mx_ms1_3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mmovb_x_m(DisasContext *s, arg_ame_misc_to_gpr *a)
{
    TCGv dest;

    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms2_3), s)) {
        return false;
    }

    dest = tcg_temp_new();
    gen_helper_xsmtame_mmovb_x_m(dest, tcg_env,
                                      tcg_constant_i32(a->mx_ms2_3),
                                      get_gpr(s, a->mx_rs1, EXT_NONE));
    gen_set_gpr(s, a->rd, dest);
    return true;
}

static bool trans_mmovh_x_m(DisasContext *s, arg_ame_misc_to_gpr *a)
{
    TCGv dest;

    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms2_3), s)) {
        return false;
    }

    dest = tcg_temp_new();
    gen_helper_xsmtame_mmovh_x_m(dest, tcg_env,
                                      tcg_constant_i32(a->mx_ms2_3),
                                      get_gpr(s, a->mx_rs1, EXT_NONE));
    gen_set_gpr(s, a->rd, dest);
    return true;
}

static bool trans_mmovw_x_m(DisasContext *s, arg_ame_misc_to_gpr *a)
{
    TCGv dest;

    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms2_3), s)) {
        return false;
    }

    dest = tcg_temp_new();
    gen_helper_xsmtame_mmovw_x_m(dest, tcg_env,
                                      tcg_constant_i32(a->mx_ms2_3),
                                      get_gpr(s, a->mx_rs1, EXT_NONE));
    gen_set_gpr(s, a->rd, dest);
    return true;
}

static bool trans_mmovd_x_m(DisasContext *s, arg_ame_misc_to_gpr *a)
{
    TCGv dest;

    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms2_3), s)) {
        return false;
    }

    dest = tcg_temp_new();
    gen_helper_xsmtame_mmovd_x_m(dest, tcg_env,
                                      tcg_constant_i32(a->mx_ms2_3),
                                      get_gpr(s, a->mx_rs1, EXT_NONE));
    gen_set_gpr(s, a->rd, dest);
    return true;
}

static bool trans_mmovb_m_x(DisasContext *s, arg_ame_misc_from_gpr *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s)) {
        return false;
    }

    gen_helper_xsmtame_mmovb_m_x(tcg_env,
                                      tcg_constant_i32(a->mx_md3),
                                      get_gpr(s, a->mx_rs1, EXT_NONE),
                                      get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mmovh_m_x(DisasContext *s, arg_ame_misc_from_gpr *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s)) {
        return false;
    }

    gen_helper_xsmtame_mmovh_m_x(tcg_env,
                                      tcg_constant_i32(a->mx_md3),
                                      get_gpr(s, a->mx_rs1, EXT_NONE),
                                      get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mmovw_m_x(DisasContext *s, arg_ame_misc_from_gpr *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s)) {
        return false;
    }

    gen_helper_xsmtame_mmovw_m_x(tcg_env,
                                      tcg_constant_i32(a->mx_md3),
                                      get_gpr(s, a->mx_rs1, EXT_NONE),
                                      get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mmovd_m_x(DisasContext *s, arg_ame_misc_from_gpr *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s)) {
        return false;
    }

    gen_helper_xsmtame_mmovd_m_x(tcg_env,
                                      tcg_constant_i32(a->mx_md3),
                                      get_gpr(s, a->mx_rs1, EXT_NONE),
                                      get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mdupb_m_x(DisasContext *s, arg_ame_misc_dup *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s)) {
        return false;
    }

    gen_helper_xsmtame_mdupb_m_x(tcg_env,
                                      tcg_constant_i32(a->mx_md3),
                                      get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mduph_m_x(DisasContext *s, arg_ame_misc_dup *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s)) {
        return false;
    }

    gen_helper_xsmtame_mduph_m_x(tcg_env,
                                      tcg_constant_i32(a->mx_md3),
                                      get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mdupw_m_x(DisasContext *s, arg_ame_misc_dup *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s)) {
        return false;
    }

    gen_helper_xsmtame_mdupw_m_x(tcg_env,
                                      tcg_constant_i32(a->mx_md3),
                                      get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mdupd_m_x(DisasContext *s, arg_ame_misc_dup *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s)) {
        return false;
    }

    gen_helper_xsmtame_mdupd_m_x(tcg_env,
                                      tcg_constant_i32(a->mx_md3),
                                      get_gpr(s, a->mx_rs2, EXT_NONE));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mpack_mm(DisasContext *s, arg_ame_misc_rrr *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms2_3), s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms1_3), s) ||
        !ame_require(ame_same_reg_class_3(a->mx_md3, a->mx_ms2_3), s) ||
        !ame_require(ame_same_reg_class_3(a->mx_md3, a->mx_ms1_3), s)) {
        return false;
    }

    gen_helper_xsmtame_mpack_mm(tcg_env,
                                     tcg_constant_i32(a->mx_md3),
                                     tcg_constant_i32(a->mx_ms2_3),
                                     tcg_constant_i32(a->mx_ms1_3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mpackhl_mm(DisasContext *s, arg_ame_misc_rrr *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms2_3), s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms1_3), s) ||
        !ame_require(ame_same_reg_class_3(a->mx_md3, a->mx_ms2_3), s) ||
        !ame_require(ame_same_reg_class_3(a->mx_md3, a->mx_ms1_3), s)) {
        return false;
    }

    gen_helper_xsmtame_mpackhl_mm(tcg_env,
                                       tcg_constant_i32(a->mx_md3),
                                       tcg_constant_i32(a->mx_ms2_3),
                                       tcg_constant_i32(a->mx_ms1_3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mpackhh_mm(DisasContext *s, arg_ame_misc_rrr *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms2_3), s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms1_3), s) ||
        !ame_require(ame_same_reg_class_3(a->mx_md3, a->mx_ms2_3), s) ||
        !ame_require(ame_same_reg_class_3(a->mx_md3, a->mx_ms1_3), s)) {
        return false;
    }

    gen_helper_xsmtame_mpackhh_mm(tcg_env,
                                       tcg_constant_i32(a->mx_md3),
                                       tcg_constant_i32(a->mx_ms2_3),
                                       tcg_constant_i32(a->mx_ms1_3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mrslidedown(DisasContext *s, arg_ame_misc_slide *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms1_3), s) ||
        !ame_require(ame_same_reg_class_3(a->mx_md3, a->mx_ms1_3), s)) {
        return false;
    }

    gen_helper_xsmtame_mrslidedown(tcg_env,
                                        tcg_constant_i32(a->mx_md3),
                                        tcg_constant_i32(a->mx_ms1_3),
                                        tcg_constant_i32(a->mx_uimm3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mrslideup(DisasContext *s, arg_ame_misc_slide *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms1_3), s) ||
        !ame_require(ame_same_reg_class_3(a->mx_md3, a->mx_ms1_3), s)) {
        return false;
    }

    gen_helper_xsmtame_mrslideup(tcg_env,
                                      tcg_constant_i32(a->mx_md3),
                                      tcg_constant_i32(a->mx_ms1_3),
                                      tcg_constant_i32(a->mx_uimm3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mcslidedown_b(DisasContext *s, arg_ame_misc_slide *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms1_3), s) ||
        !ame_require(ame_same_reg_class_3(a->mx_md3, a->mx_ms1_3), s)) {
        return false;
    }

    gen_helper_xsmtame_mcslidedown_b(tcg_env,
                                          tcg_constant_i32(a->mx_md3),
                                          tcg_constant_i32(a->mx_ms1_3),
                                          tcg_constant_i32(a->mx_uimm3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mcslidedown_h(DisasContext *s, arg_ame_misc_slide *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms1_3), s) ||
        !ame_require(ame_same_reg_class_3(a->mx_md3, a->mx_ms1_3), s)) {
        return false;
    }

    gen_helper_xsmtame_mcslidedown_h(tcg_env,
                                          tcg_constant_i32(a->mx_md3),
                                          tcg_constant_i32(a->mx_ms1_3),
                                          tcg_constant_i32(a->mx_uimm3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mcslidedown_w(DisasContext *s, arg_ame_misc_slide *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms1_3), s) ||
        !ame_require(ame_same_reg_class_3(a->mx_md3, a->mx_ms1_3), s)) {
        return false;
    }

    gen_helper_xsmtame_mcslidedown_w(tcg_env,
                                          tcg_constant_i32(a->mx_md3),
                                          tcg_constant_i32(a->mx_ms1_3),
                                          tcg_constant_i32(a->mx_uimm3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mcslidedown_d(DisasContext *s, arg_ame_misc_slide *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms1_3), s) ||
        !ame_require(ame_same_reg_class_3(a->mx_md3, a->mx_ms1_3), s)) {
        return false;
    }

    gen_helper_xsmtame_mcslidedown_d(tcg_env,
                                          tcg_constant_i32(a->mx_md3),
                                          tcg_constant_i32(a->mx_ms1_3),
                                          tcg_constant_i32(a->mx_uimm3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mcslideup_b(DisasContext *s, arg_ame_misc_slide *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms1_3), s) ||
        !ame_require(ame_same_reg_class_3(a->mx_md3, a->mx_ms1_3), s)) {
        return false;
    }

    gen_helper_xsmtame_mcslideup_b(tcg_env,
                                        tcg_constant_i32(a->mx_md3),
                                        tcg_constant_i32(a->mx_ms1_3),
                                        tcg_constant_i32(a->mx_uimm3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mcslideup_h(DisasContext *s, arg_ame_misc_slide *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms1_3), s) ||
        !ame_require(ame_same_reg_class_3(a->mx_md3, a->mx_ms1_3), s)) {
        return false;
    }

    gen_helper_xsmtame_mcslideup_h(tcg_env,
                                        tcg_constant_i32(a->mx_md3),
                                        tcg_constant_i32(a->mx_ms1_3),
                                        tcg_constant_i32(a->mx_uimm3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mcslideup_w(DisasContext *s, arg_ame_misc_slide *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms1_3), s) ||
        !ame_require(ame_same_reg_class_3(a->mx_md3, a->mx_ms1_3), s)) {
        return false;
    }

    gen_helper_xsmtame_mcslideup_w(tcg_env,
                                        tcg_constant_i32(a->mx_md3),
                                        tcg_constant_i32(a->mx_ms1_3),
                                        tcg_constant_i32(a->mx_uimm3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mcslideup_d(DisasContext *s, arg_ame_misc_slide *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms1_3), s) ||
        !ame_require(ame_same_reg_class_3(a->mx_md3, a->mx_ms1_3), s)) {
        return false;
    }

    gen_helper_xsmtame_mcslideup_d(tcg_env,
                                        tcg_constant_i32(a->mx_md3),
                                        tcg_constant_i32(a->mx_ms1_3),
                                        tcg_constant_i32(a->mx_uimm3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mrbca_mv_i(DisasContext *s, arg_ame_misc_slide *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms1_3), s) ||
        !ame_require(ame_same_reg_class_3(a->mx_md3, a->mx_ms1_3), s)) {
        return false;
    }

    gen_helper_xsmtame_mrbca_mv_i(tcg_env,
                                       tcg_constant_i32(a->mx_md3),
                                       tcg_constant_i32(a->mx_ms1_3),
                                       tcg_constant_i32(a->mx_uimm3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mcbcab_mv_i(DisasContext *s, arg_ame_misc_slide *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms1_3), s) ||
        !ame_require(ame_same_reg_class_3(a->mx_md3, a->mx_ms1_3), s)) {
        return false;
    }

    gen_helper_xsmtame_mcbcab_mv_i(tcg_env,
                                        tcg_constant_i32(a->mx_md3),
                                        tcg_constant_i32(a->mx_ms1_3),
                                        tcg_constant_i32(a->mx_uimm3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mcbcah_mv_i(DisasContext *s, arg_ame_misc_slide *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms1_3), s) ||
        !ame_require(ame_same_reg_class_3(a->mx_md3, a->mx_ms1_3), s)) {
        return false;
    }

    gen_helper_xsmtame_mcbcah_mv_i(tcg_env,
                                        tcg_constant_i32(a->mx_md3),
                                        tcg_constant_i32(a->mx_ms1_3),
                                        tcg_constant_i32(a->mx_uimm3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mcbcaw_mv_i(DisasContext *s, arg_ame_misc_slide *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms1_3), s) ||
        !ame_require(ame_same_reg_class_3(a->mx_md3, a->mx_ms1_3), s)) {
        return false;
    }

    gen_helper_xsmtame_mcbcaw_mv_i(tcg_env,
                                        tcg_constant_i32(a->mx_md3),
                                        tcg_constant_i32(a->mx_ms1_3),
                                        tcg_constant_i32(a->mx_uimm3));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mcbcad_mv_i(DisasContext *s, arg_ame_misc_slide *a)
{
    if (!require_xsmtame(s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_md3), s) ||
        !ame_require(ame_check_matrix_reg_3(a->mx_ms1_3), s) ||
        !ame_require(ame_same_reg_class_3(a->mx_md3, a->mx_ms1_3), s)) {
        return false;
    }

    gen_helper_xsmtame_mcbcad_mv_i(tcg_env,
                                        tcg_constant_i32(a->mx_md3),
                                        tcg_constant_i32(a->mx_ms1_3),
                                        tcg_constant_i32(a->mx_uimm3));
    mark_ms_dirty(s);
    return true;
}

/*
 * ──────────────────────────────────────────
 *  Control: mzero{,2,4,8}r / mrelease
 * ──────────────────────────────────────────
 */

static bool trans_mzero_common(DisasContext *s, uint32_t mx_md3,
                               uint32_t expected_count)
{
    if (!require_xsmtame(s)) {
        return false;
    }

    if (!ame_require(ame_check_matrix_reg_3(mx_md3), s)) {
        return false;
    }

    if (!ame_require((mx_md3 & (expected_count - 1)) == 0, s)) {
        return false;
    }

    if (!ame_require(mx_md3 + expected_count <= AME_NR_TILES + AME_NR_ACCS, s)) {
        return false;
    }

    gen_helper_xsmtame_mzero(tcg_env,
                                  tcg_constant_i32(mx_md3),
                                  tcg_constant_i32(expected_count));
    mark_ms_dirty(s);
    return true;
}

static bool trans_mzero(DisasContext *s, arg_mzero *a)
{
    return trans_mzero_common(s, a->mx_md3, 1);
}

static bool trans_mzero2r(DisasContext *s, arg_mzero2r *a)
{
    return trans_mzero_common(s, a->mx_md3, 2);
}

static bool trans_mzero4r(DisasContext *s, arg_mzero4r *a)
{
    return trans_mzero_common(s, a->mx_md3, 4);
}

static bool trans_mzero8r(DisasContext *s, arg_mzero8r *a)
{
    return trans_mzero_common(s, a->mx_md3, 8);
}

static bool trans_mrelease(DisasContext *s, arg_ame_release *a)
{
    if (!require_xsmtame(s)) {
        return false;
    }
    gen_helper_xsmtame_mrelease(tcg_env);
    /* mrelease sets MS = Initial (01), NOT dirty.
     * We do NOT call mark_ms_dirty here. The helper
     * directly modifies env->mstatus. */
    return true;
}
