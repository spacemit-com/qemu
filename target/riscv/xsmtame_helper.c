/*
 * AME (Attached Matrix Extension) helper implementations
 *
 * Copyright (c) 2025
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2, as published by the Free Software Foundation.
 */

#include "qemu/osdep.h"
#include "qemu/host-utils.h"
#include "qemu/bitops.h"
#include "cpu.h"
#include "accel/tcg/cpu-ldst.h"
#include "exec/memop.h"
#include "exec/page-protection.h"
#include "exec/helper-proto.h"
#include "fpu/softfloat.h"
#include "internals.h"

/*
 * ──────────────────────────────────────────
 *  AME runtime dimension helpers
 * ──────────────────────────────────────────
 * Use these instead of the compile-time AME_TILE_LEN_B / AME_ACC_LEN_B
 * constants whenever the actual (possibly narrower) register size is needed.
 */
#define ame_env_tlenb(env)      ame_cfg_tlenb(ame_env_cfg(env))
#define ame_env_trlenb(env)     ame_cfg_trlenb(ame_env_cfg(env))
#define ame_env_rownum(env)     ame_cfg_rownum(ame_env_cfg(env))
#define ame_env_acc_len_b(env)  ame_cfg_acc_len_b(ame_env_cfg(env))
#define ame_env_cfg(env)        (&env_archcpu(env)->cfg)

/*
 * ──────────────────────────────────────────
 *  Internal helpers: register base pointers
 * ──────────────────────────────────────────
 */

/* Return pointer to the start of tile[id] inside CPURISCVState */
static inline uint8_t *xsmtame_tile_ptr(CPURISCVState *env, uint32_t id)
{
    g_assert(id < AME_NR_TILES);
    return (uint8_t *)env->ame_tile + id * ame_env_tlenb(env);
}

/* Return pointer to the start of acc[id] inside CPURISCVState */
static inline uint8_t *xsmtame_acc_ptr(CPURISCVState *env, uint32_t id)
{
    g_assert(id < AME_NR_ACCS);
    return (uint8_t *)env->ame_acc + id * ame_env_acc_len_b(env);
}

typedef struct AMEShapeInfo {
    uint32_t m;
    uint32_t n;
    uint32_t k;
} AMEShapeInfo;

typedef struct AMEMatrixLayout {
    size_t row_bytes;
    size_t cols;
} AMEMatrixLayout;

static G_NORETURN void xsmtame_raise_illegal(CPURISCVState *env)
{
    riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
}

static inline void xsmtame_validate_shape_eew(CPURISCVState *env,
                                                   uint32_t eew_bits)
{
    RISCVCPUConfig *cfg = ame_env_cfg(env);
    uint32_t rownum = ame_cfg_rownum(cfg);
    uint32_t kmax = eew_bits ? ame_cfg_kmax_eew(cfg, eew_bits) : 0;

    if (env->mtilem > rownum || env->mtilen > rownum || env->mtilek > kmax) {
        xsmtame_raise_illegal(env);
    }
}

static inline void xsmtame_validate_shape(CPURISCVState *env)
{
    xsmtame_validate_shape_eew(env, AME_MIN_EEW_BITS);
}

static inline AMEShapeInfo xsmtame_shape(CPURISCVState *env)
{
    xsmtame_validate_shape(env);

    return (AMEShapeInfo) {
        .m = env->mtilem,
        .n = env->mtilen,
        .k = env->mtilek,
    };
}

static inline uint16_t *xsmtame_tile16_ptr(CPURISCVState *env, uint32_t id)
{
    xsmtame_validate_shape_eew(env, 16);

    return (uint16_t *)xsmtame_tile_ptr(env, id);
}

static inline uint32_t *xsmtame_tile32_ptr(CPURISCVState *env, uint32_t id)
{
    xsmtame_validate_shape_eew(env, 32);

    return (uint32_t *)xsmtame_tile_ptr(env, id);
}

static inline int8_t *xsmtame_tile8s_ptr(CPURISCVState *env, uint32_t id)
{
    return (int8_t *)xsmtame_tile_ptr(env, id);
}

static inline uint8_t *xsmtame_acc8_ptr(CPURISCVState *env, uint32_t id)
{
    return xsmtame_acc_ptr(env, id);
}

static inline uint16_t *xsmtame_acc16_ptr(CPURISCVState *env, uint32_t id)
{
    return (uint16_t *)xsmtame_acc_ptr(env, id);
}

static inline uint32_t *xsmtame_acc32_ptr(CPURISCVState *env, uint32_t id)
{
    return (uint32_t *)xsmtame_acc_ptr(env, id);
}

static inline uint8_t *xsmtame_matrix_ptr(CPURISCVState *env, uint32_t reg,
                                               size_t *size)
{
    xsmtame_validate_shape(env);

    if (reg < AME_NR_TILES) {
        if (size) {
            *size = ame_env_tlenb(env);
        }
        return xsmtame_tile_ptr(env, reg);
    }

    g_assert(reg < AME_NR_TILES + AME_NR_ACCS);
    if (size) {
        *size = ame_env_acc_len_b(env);
    }
    return xsmtame_acc_ptr(env, reg - AME_NR_TILES);
}

static inline void xsmtame_load_bytes(CPURISCVState *env, uint8_t *dst,
                                           target_ulong addr, size_t size)
{
    size_t i;

    for (i = 0; i < size; i++) {
        dst[i] = cpu_ldub_data(env, addr + i);
    }
}

static inline void xsmtame_store_bytes(CPURISCVState *env, target_ulong addr,
                                            const uint8_t *src, size_t size)
{
    size_t i;

    for (i = 0; i < size; i++) {
        cpu_stb_data(env, addr + i, src[i]);
    }
}

static inline void xsmtame_load_matrix_full(CPURISCVState *env, uint32_t md,
                                                 target_ulong addr)
{
    size_t reg_size;
    uint8_t *dst = xsmtame_matrix_ptr(env, md, &reg_size);

    xsmtame_load_bytes(env, dst, addr, reg_size);
}

static inline void xsmtame_store_matrix_full(CPURISCVState *env, uint32_t ms,
                                                  target_ulong addr)
{
    size_t reg_size;
    const uint8_t *src = xsmtame_matrix_ptr(env, ms, &reg_size);

    xsmtame_store_bytes(env, addr, src, reg_size);
}

#define GEN_XSMTAME_WHOLE_LOAD_HELPER(NAME)                                 \
    void HELPER(NAME)(CPURISCVState *env, uint32_t md, target_ulong addr)   \
    {                                                                       \
        xsmtame_load_matrix_full(env, md, addr);                            \
    }

#define GEN_XSMTAME_WHOLE_STORE_HELPER(NAME)                                \
    void HELPER(NAME)(CPURISCVState *env, uint32_t ms, target_ulong addr)   \
    {                                                                       \
        xsmtame_store_matrix_full(env, ms, addr);                           \
    }

#define GEN_XSMTAME_STRIDE_LOAD_HELPER(NAME, ELEM_TYPE, REGVAR, PTR_FN,     \
                                       LOAD_FN, ROW_EXPR, COL_EXPR,         \
                                       TRANSPOSE)                           \
    void HELPER(NAME)(CPURISCVState *env, uint32_t REGVAR,                  \
                      target_ulong addr, target_ulong stride)               \
    {                                                                       \
        AMEShapeInfo shape = xsmtame_shape(env);                            \
        ELEM_TYPE *data = PTR_FN(env, REGVAR);                              \
                                                                            \
        LOAD_FN(data, env, REGVAR, addr, stride,                            \
                ROW_EXPR, COL_EXPR, TRANSPOSE);                             \
    }

#define GEN_XSMTAME_STRIDE_STORE_HELPER(NAME, ELEM_TYPE, REGVAR, PTR_FN,    \
                                        STORE_FN, ROW_EXPR, COL_EXPR,       \
                                        TRANSPOSE)                          \
    void HELPER(NAME)(CPURISCVState *env, uint32_t REGVAR,                  \
                      target_ulong addr, target_ulong stride)               \
    {                                                                       \
        AMEShapeInfo shape = xsmtame_shape(env);                            \
        const ELEM_TYPE *data = PTR_FN(env, REGVAR);                        \
                                                                            \
        STORE_FN(data, env, REGVAR, addr, stride,                           \
                 ROW_EXPR, COL_EXPR, TRANSPOSE);                            \
    }

static inline size_t xsmtame_mmov_elem_offset(size_t reg_size,
                                                   size_t elem_size,
                                                   target_ulong idx)
{
    size_t elem_count = reg_size / elem_size;
    size_t elem_idx = (size_t)(idx % elem_count);

    return elem_idx * elem_size;
}

static inline AMEMatrixLayout xsmtame_matrix_layout(CPURISCVState *env,
                                                    uint32_t reg,
                                                    size_t elem_size)
{
    size_t row_bytes = reg < AME_NR_TILES ?
                       ame_env_trlenb(env) :
                       ame_env_acc_len_b(env) / ame_env_rownum(env);
    size_t cols = 0;

    if (elem_size != 0) {
        g_assert(row_bytes % elem_size == 0);
        cols = row_bytes / elem_size;
    }

    return (AMEMatrixLayout) {
        .row_bytes = row_bytes,
        .cols = cols,
    };
}

static inline bool xsmtame_matrix_is_same_class(uint32_t lhs, uint32_t rhs)
{
    return (lhs < AME_NR_TILES) == (rhs < AME_NR_TILES);
}

static inline size_t xsmtame_matrix_size(CPURISCVState *env, uint32_t reg)
{
    return reg < AME_NR_TILES ? ame_env_tlenb(env) : ame_env_acc_len_b(env);
}

static inline void xsmtame_validate_same_matrix_class(CPURISCVState *env,
                                                      uint32_t lhs,
                                                      uint32_t rhs)
{
    if (!xsmtame_matrix_is_same_class(lhs, rhs)) {
        xsmtame_raise_illegal(env);
    }
}

static inline void xsmtame_validate_same_matrix_len(CPURISCVState *env,
                                                    uint32_t lhs,
                                                    uint32_t rhs)
{
    if (xsmtame_matrix_size(env, lhs) != xsmtame_matrix_size(env, rhs)) {
        xsmtame_raise_illegal(env);
    }
}

static inline void xsmtame_validate_same_matrix_row_layout(CPURISCVState *env,
                                                           uint32_t lhs,
                                                           uint32_t rhs)
{
    if (xsmtame_matrix_layout(env, lhs, 0).row_bytes !=
        xsmtame_matrix_layout(env, rhs, 0).row_bytes) {
        xsmtame_raise_illegal(env);
    }
}

static inline void xsmtame_validate_same_matrix_layout(CPURISCVState *env,
                                                       uint32_t lhs,
                                                       uint32_t rhs)
{
    xsmtame_validate_same_matrix_class(env, lhs, rhs);
    xsmtame_validate_same_matrix_len(env, lhs, rhs);
    xsmtame_validate_same_matrix_row_layout(env, lhs, rhs);
}

static void xsmtame_zero_acc_inactive_region(CPURISCVState *env,
                                             uint32_t ad,
                                             size_t elem_size)
{
    AMEShapeInfo shape = xsmtame_shape(env);
    size_t reg_size = ame_env_acc_len_b(env);
    uint8_t *acc = xsmtame_acc_ptr(env, ad);
    size_t row_bytes = reg_size / ame_env_rownum(env);
    size_t rows = row_bytes ? reg_size / row_bytes : 0;
    size_t cols;
    size_t row;

    g_assert(ad < AME_NR_ACCS);
    g_assert(elem_size != 0);
    g_assert(row_bytes % elem_size == 0);

    cols = row_bytes / elem_size;

    for (row = 0; row < rows; row++) {
        uint8_t *row_ptr = acc + row * row_bytes;

        if (row >= shape.m) {
            memset(row_ptr, 0, row_bytes);
        } else if (shape.n < cols) {
            memset(row_ptr + shape.n * elem_size, 0,
                   (cols - shape.n) * elem_size);
        }
    }
}

static void xsmtame_mmov_m_x_common(CPURISCVState *env, uint32_t md,
                                         target_ulong idx,
                                         target_ulong value,
                                         size_t elem_size)
{
    size_t reg_size;
    uint8_t *dst = xsmtame_matrix_ptr(env, md, &reg_size);
    size_t offset = xsmtame_mmov_elem_offset(reg_size, elem_size, idx);

    switch (elem_size) {
    case 1:
        stb_p(dst + offset, value);
        break;
    case 2:
        stw_le_p(dst + offset, value);
        break;
    case 4:
        stl_le_p(dst + offset, value);
        break;
    case 8:
        stq_le_p(dst + offset, value);
        break;
    default:
        g_assert_not_reached();
    }
}

static void xsmtame_zero_tile_load_inactive_region(CPURISCVState *env,
                                                   uint32_t td,
                                                   size_t elem_size,
                                                   uint32_t rows,
                                                   uint32_t cols)
{
    size_t reg_size = ame_env_tlenb(env);
    size_t row_bytes = xsmtame_matrix_layout(env, td, elem_size).row_bytes;
    size_t total_rows = row_bytes ? reg_size / row_bytes : 0;
    size_t loaded_bytes = (size_t)cols * elem_size;
    uint8_t *tile = xsmtame_tile_ptr(env, td);
    size_t row;

    g_assert(elem_size != 0);
    g_assert(rows <= total_rows);
    g_assert(loaded_bytes <= row_bytes);

    if (loaded_bytes < row_bytes) {
        for (row = 0; row < rows; row++) {
            memset(tile + row * row_bytes + loaded_bytes, 0,
                   row_bytes - loaded_bytes);
        }
    }
    if (rows < total_rows) {
        memset(tile + rows * row_bytes, 0,
               (total_rows - rows) * row_bytes);
    }
}

static void xsmtame_zero_acc_load_inactive_region(CPURISCVState *env,
                                                  uint32_t ad,
                                                  size_t elem_size,
                                                  uint32_t rows,
                                                  uint32_t cols,
                                                  bool transpose)
{
    size_t reg_size = ame_env_acc_len_b(env);
    size_t row_bytes = xsmtame_matrix_layout(env, ad + AME_NR_TILES,
                                             elem_size).row_bytes;
    size_t total_rows = row_bytes ? reg_size / row_bytes : 0;
    size_t loaded_bytes = (size_t)cols * elem_size;
    uint8_t *acc = xsmtame_acc_ptr(env, ad);
    size_t row;

    g_assert(elem_size != 0);
    g_assert(rows <= total_rows);
    g_assert(loaded_bytes <= row_bytes);

    if (transpose) {
        for (row = rows; row < total_rows; row++) {
            memset(acc + row * row_bytes, 0, loaded_bytes);
        }
    } else if (loaded_bytes < row_bytes) {
        for (row = 0; row < rows; row++) {
            memset(acc + row * row_bytes + loaded_bytes, 0,
                   row_bytes - loaded_bytes);
        }
    }
}

static inline void xsmtame_load_tile8_stride(uint8_t *tile,
                                             CPURISCVState *env,
                                             uint32_t reg,
                                             target_ulong addr,
                                             target_ulong stride,
                                             uint32_t rows,
                                             uint32_t cols,
                                             bool transpose)
{
    size_t row_bytes = xsmtame_matrix_layout(env, reg, 0).row_bytes;
    uint32_t row, col;

    for (row = 0; row < rows; row++) {
        for (col = 0; col < cols; col++) {
            tile[row * row_bytes + col] =
                cpu_ldub_data(env, addr + (transpose ?
                              col * stride + row :
                              row * stride + col));
        }
    }

    xsmtame_zero_tile_load_inactive_region(env, reg, sizeof(*tile),
                                           rows, cols);
}

static inline void xsmtame_store_tile8_stride(const uint8_t *tile,
                                              CPURISCVState *env,
                                              uint32_t reg,
                                              target_ulong addr,
                                              target_ulong stride,
                                              uint32_t rows,
                                              uint32_t cols,
                                              bool transpose)
{
    size_t row_bytes = xsmtame_matrix_layout(env, reg, 0).row_bytes;
    uint32_t row, col;

    for (row = 0; row < rows; row++) {
        for (col = 0; col < cols; col++) {
            cpu_stb_data(env, addr + (transpose ?
                         col * stride + row :
                         row * stride + col),
                         tile[row * row_bytes + col]);
        }
    }
}

static inline void xsmtame_load_tile16_stride(uint16_t *tile16,
                                              CPURISCVState *env,
                                              uint32_t reg,
                                              target_ulong addr,
                                              target_ulong stride,
                                              uint32_t rows,
                                              uint32_t cols,
                                              bool transpose)
{
    size_t cols_per_row = xsmtame_matrix_layout(env, reg, sizeof(*tile16)).cols;
    uint32_t row, col;

    for (row = 0; row < rows; row++) {
        for (col = 0; col < cols; col++) {
            tile16[row * cols_per_row + col] =
                cpu_lduw_data(env, addr + (transpose ?
                              col * stride + row * sizeof(*tile16) :
                              row * stride + col * sizeof(*tile16)));
        }
    }

    xsmtame_zero_tile_load_inactive_region(env, reg, sizeof(*tile16),
                                           rows, cols);
}

static inline void xsmtame_store_tile16_stride(const uint16_t *tile16,
                                               CPURISCVState *env,
                                               uint32_t reg,
                                               target_ulong addr,
                                               target_ulong stride,
                                               uint32_t rows,
                                               uint32_t cols,
                                               bool transpose)
{
    size_t cols_per_row = xsmtame_matrix_layout(env, reg, sizeof(*tile16)).cols;
    uint32_t row, col;

    for (row = 0; row < rows; row++) {
        for (col = 0; col < cols; col++) {
            cpu_stw_data(env, addr + (transpose ?
                         col * stride + row * sizeof(*tile16) :
                         row * stride + col * sizeof(*tile16)),
                         tile16[row * cols_per_row + col]);
        }
    }
}

static inline void xsmtame_load_tile32_stride(uint32_t *tile32,
                                              CPURISCVState *env,
                                              uint32_t reg,
                                              target_ulong addr,
                                              target_ulong stride,
                                              uint32_t rows,
                                              uint32_t cols,
                                              bool transpose)
{
    size_t cols_per_row = xsmtame_matrix_layout(env, reg, sizeof(*tile32)).cols;
    uint32_t row, col;

    for (row = 0; row < rows; row++) {
        for (col = 0; col < cols; col++) {
            tile32[row * cols_per_row + col] =
                cpu_ldl_data(env, addr + (transpose ?
                             col * stride + row * sizeof(*tile32) :
                             row * stride + col * sizeof(*tile32)));
        }
    }

    xsmtame_zero_tile_load_inactive_region(env, reg, sizeof(*tile32),
                                           rows, cols);
}

static inline void xsmtame_store_tile32_stride(const uint32_t *tile32,
                                               CPURISCVState *env,
                                               uint32_t reg,
                                               target_ulong addr,
                                               target_ulong stride,
                                               uint32_t rows,
                                               uint32_t cols,
                                               bool transpose)
{
    size_t cols_per_row = xsmtame_matrix_layout(env, reg, sizeof(*tile32)).cols;
    uint32_t row, col;

    for (row = 0; row < rows; row++) {
        for (col = 0; col < cols; col++) {
            cpu_stl_data(env, addr + (transpose ?
                         col * stride + row * sizeof(*tile32) :
                         row * stride + col * sizeof(*tile32)),
                         tile32[row * cols_per_row + col]);
        }
    }
}

static inline void xsmtame_load_acc8_stride(uint8_t *acc8,
                                            CPURISCVState *env,
                                            uint32_t reg,
                                            target_ulong addr,
                                            target_ulong stride,
                                            uint32_t rows,
                                            uint32_t cols,
                                            bool transpose)
{
    size_t row_bytes = xsmtame_matrix_layout(env, reg + AME_NR_TILES,
                                             0).row_bytes;
    uint32_t row, col;

    for (row = 0; row < rows; row++) {
        for (col = 0; col < cols; col++) {
            acc8[row * row_bytes + col] =
                cpu_ldub_data(env, addr + (transpose ?
                              col * stride + row :
                              row * stride + col));
        }
    }

    xsmtame_zero_acc_load_inactive_region(env, reg, sizeof(*acc8),
                                          rows, cols, transpose);
}

static inline void xsmtame_store_acc8_stride(const uint8_t *acc8,
                                             CPURISCVState *env,
                                             uint32_t reg,
                                             target_ulong addr,
                                             target_ulong stride,
                                             uint32_t rows,
                                             uint32_t cols,
                                             bool transpose)
{
    size_t row_bytes = xsmtame_matrix_layout(env, reg + AME_NR_TILES,
                                             0).row_bytes;
    uint32_t row, col;

    for (row = 0; row < rows; row++) {
        for (col = 0; col < cols; col++) {
            cpu_stb_data(env, addr + (transpose ?
                         col * stride + row :
                         row * stride + col),
                         acc8[row * row_bytes + col]);
        }
    }
}

static inline void xsmtame_load_acc16_stride(uint16_t *acc16,
                                             CPURISCVState *env,
                                             uint32_t reg,
                                             target_ulong addr,
                                             target_ulong stride,
                                             uint32_t rows,
                                             uint32_t cols,
                                             bool transpose)
{
    size_t cols_per_row = xsmtame_matrix_layout(env, reg + AME_NR_TILES,
                                                sizeof(*acc16)).cols;
    uint32_t row, col;

    for (row = 0; row < rows; row++) {
        for (col = 0; col < cols; col++) {
            acc16[row * cols_per_row + col] =
                cpu_lduw_data(env, addr + (transpose ?
                              col * stride + row * sizeof(*acc16) :
                              row * stride + col * sizeof(*acc16)));
        }
    }

    xsmtame_zero_acc_load_inactive_region(env, reg, sizeof(*acc16),
                                          rows, cols, transpose);
}

static inline void xsmtame_store_acc16_stride(const uint16_t *acc16,
                                              CPURISCVState *env,
                                              uint32_t reg,
                                              target_ulong addr,
                                              target_ulong stride,
                                              uint32_t rows,
                                              uint32_t cols,
                                              bool transpose)
{
    size_t cols_per_row = xsmtame_matrix_layout(env, reg + AME_NR_TILES,
                                                sizeof(*acc16)).cols;
    uint32_t row, col;

    for (row = 0; row < rows; row++) {
        for (col = 0; col < cols; col++) {
            cpu_stw_data(env, addr + (transpose ?
                         col * stride + row * sizeof(*acc16) :
                         row * stride + col * sizeof(*acc16)),
                         acc16[row * cols_per_row + col]);
        }
    }
}

static inline void xsmtame_load_acc32_stride(uint32_t *acc32,
                                             CPURISCVState *env,
                                             uint32_t reg,
                                             target_ulong addr,
                                             target_ulong stride,
                                             uint32_t rows,
                                             uint32_t cols,
                                             bool transpose)
{
    size_t cols_per_row = xsmtame_matrix_layout(env, reg + AME_NR_TILES,
                                                sizeof(*acc32)).cols;
    uint32_t row, col;

    for (row = 0; row < rows; row++) {
        for (col = 0; col < cols; col++) {
            acc32[row * cols_per_row + col] =
                cpu_ldl_data(env, addr + (transpose ?
                              col * stride + row * sizeof(*acc32) :
                              row * stride + col * sizeof(*acc32)));
        }
    }

    xsmtame_zero_acc_load_inactive_region(env, reg, sizeof(*acc32),
                                          rows, cols, transpose);
}

static inline void xsmtame_store_acc32_stride(const uint32_t *acc32,
                                              CPURISCVState *env,
                                              uint32_t reg,
                                              target_ulong addr,
                                              target_ulong stride,
                                              uint32_t rows,
                                              uint32_t cols,
                                              bool transpose)
{
    size_t cols_per_row = xsmtame_matrix_layout(env, reg + AME_NR_TILES,
                                                sizeof(*acc32)).cols;
    uint32_t row, col;

    for (row = 0; row < rows; row++) {
        for (col = 0; col < cols; col++) {
            cpu_stl_data(env, addr + (transpose ?
                         col * stride + row * sizeof(*acc32) :
                         row * stride + col * sizeof(*acc32)),
                         acc32[row * cols_per_row + col]);
        }
    }
}

/*
 * ──────────────────────────────────────────
 *  Whole matrix load/store helpers
 * ──────────────────────────────────────────
 */

GEN_XSMTAME_WHOLE_LOAD_HELPER(xsmtame_mlme8)
GEN_XSMTAME_WHOLE_LOAD_HELPER(xsmtame_mlme16)
GEN_XSMTAME_WHOLE_LOAD_HELPER(xsmtame_mlme32)

GEN_XSMTAME_WHOLE_STORE_HELPER(xsmtame_msme8)
GEN_XSMTAME_WHOLE_STORE_HELPER(xsmtame_msme16)
GEN_XSMTAME_WHOLE_STORE_HELPER(xsmtame_msme32)

/*
 * ──────────────────────────────────────────
 *  Stride load/store helpers
 * ──────────────────────────────────────────
 *
 * Legacy helpers still support stride where required by old instructions.
 * mlae/mlbe use explicit stride without transpose.
 * mlate/mlbte use explicit stride and transpose while loading.
 * msae/msbe use explicit stride without transpose while storing.
 * msate/msbte use explicit stride and transpose while storing.
 */

GEN_XSMTAME_STRIDE_LOAD_HELPER(xsmtame_mlae8, uint8_t, td,
                               xsmtame_tile_ptr, xsmtame_load_tile8_stride,
                               shape.m, shape.k, false)
GEN_XSMTAME_STRIDE_LOAD_HELPER(xsmtame_mlae16, uint16_t, td,
                               xsmtame_tile16_ptr, xsmtame_load_tile16_stride,
                               shape.m, shape.k, false)
GEN_XSMTAME_STRIDE_LOAD_HELPER(xsmtame_mlae32, uint32_t, td,
                               xsmtame_tile32_ptr, xsmtame_load_tile32_stride,
                               shape.m, shape.k, false)
GEN_XSMTAME_STRIDE_LOAD_HELPER(xsmtame_mlbe8, uint8_t, td,
                               xsmtame_tile_ptr, xsmtame_load_tile8_stride,
                               shape.n, shape.k, false)
GEN_XSMTAME_STRIDE_LOAD_HELPER(xsmtame_mlbe16, uint16_t, td,
                               xsmtame_tile16_ptr, xsmtame_load_tile16_stride,
                               shape.n, shape.k, false)
GEN_XSMTAME_STRIDE_LOAD_HELPER(xsmtame_mlbe32, uint32_t, td,
                               xsmtame_tile32_ptr, xsmtame_load_tile32_stride,
                               shape.n, shape.k, false)
GEN_XSMTAME_STRIDE_LOAD_HELPER(xsmtame_mlce8, uint8_t, ad,
                               xsmtame_acc8_ptr, xsmtame_load_acc8_stride,
                               shape.m, shape.n, false)
GEN_XSMTAME_STRIDE_LOAD_HELPER(xsmtame_mlce16, uint16_t, ad,
                               xsmtame_acc16_ptr, xsmtame_load_acc16_stride,
                               shape.m, shape.n, false)
GEN_XSMTAME_STRIDE_LOAD_HELPER(xsmtame_mlce32, uint32_t, ad,
                               xsmtame_acc32_ptr, xsmtame_load_acc32_stride,
                               shape.m, shape.n, false)
GEN_XSMTAME_STRIDE_STORE_HELPER(xsmtame_msae8, uint8_t, td,
                                xsmtame_tile_ptr, xsmtame_store_tile8_stride,
                                shape.m, shape.k, false)
GEN_XSMTAME_STRIDE_STORE_HELPER(xsmtame_msae16, uint16_t, td,
                                xsmtame_tile16_ptr, xsmtame_store_tile16_stride,
                                shape.m, shape.k, false)
GEN_XSMTAME_STRIDE_STORE_HELPER(xsmtame_msae32, uint32_t, td,
                                xsmtame_tile32_ptr, xsmtame_store_tile32_stride,
                                shape.m, shape.k, false)
GEN_XSMTAME_STRIDE_STORE_HELPER(xsmtame_msbe8, uint8_t, td,
                                xsmtame_tile_ptr, xsmtame_store_tile8_stride,
                                shape.n, shape.k, false)
GEN_XSMTAME_STRIDE_STORE_HELPER(xsmtame_msbe16, uint16_t, td,
                                xsmtame_tile16_ptr, xsmtame_store_tile16_stride,
                                shape.n, shape.k, false)
GEN_XSMTAME_STRIDE_STORE_HELPER(xsmtame_msbe32, uint32_t, td,
                                xsmtame_tile32_ptr, xsmtame_store_tile32_stride,
                                shape.n, shape.k, false)
GEN_XSMTAME_STRIDE_STORE_HELPER(xsmtame_msce8, uint8_t, ad,
                                xsmtame_acc8_ptr, xsmtame_store_acc8_stride,
                                shape.m, shape.n, false)
GEN_XSMTAME_STRIDE_STORE_HELPER(xsmtame_msce16, uint16_t, ad,
                                xsmtame_acc16_ptr, xsmtame_store_acc16_stride,
                                shape.m, shape.n, false)
GEN_XSMTAME_STRIDE_STORE_HELPER(xsmtame_msce32, uint32_t, ad,
                                xsmtame_acc32_ptr, xsmtame_store_acc32_stride,
                                shape.m, shape.n, false)
GEN_XSMTAME_STRIDE_LOAD_HELPER(xsmtame_mlate8, uint8_t, td,
                               xsmtame_tile_ptr, xsmtame_load_tile8_stride,
                               shape.m, shape.k, true)
GEN_XSMTAME_STRIDE_LOAD_HELPER(xsmtame_mlate16, uint16_t, td,
                               xsmtame_tile16_ptr, xsmtame_load_tile16_stride,
                               shape.m, shape.k, true)
GEN_XSMTAME_STRIDE_LOAD_HELPER(xsmtame_mlate32, uint32_t, td,
                               xsmtame_tile32_ptr, xsmtame_load_tile32_stride,
                               shape.m, shape.k, true)
GEN_XSMTAME_STRIDE_LOAD_HELPER(xsmtame_mlbte8, uint8_t, td,
                               xsmtame_tile_ptr, xsmtame_load_tile8_stride,
                               shape.n, shape.k, true)
GEN_XSMTAME_STRIDE_LOAD_HELPER(xsmtame_mlbte16, uint16_t, td,
                               xsmtame_tile16_ptr, xsmtame_load_tile16_stride,
                               shape.n, shape.k, true)
GEN_XSMTAME_STRIDE_LOAD_HELPER(xsmtame_mlbte32, uint32_t, td,
                               xsmtame_tile32_ptr, xsmtame_load_tile32_stride,
                               shape.n, shape.k, true)
GEN_XSMTAME_STRIDE_LOAD_HELPER(xsmtame_mlcte8, uint8_t, ad,
                               xsmtame_acc8_ptr, xsmtame_load_acc8_stride,
                               shape.m, shape.n, true)
GEN_XSMTAME_STRIDE_LOAD_HELPER(xsmtame_mlcte16, uint16_t, ad,
                               xsmtame_acc16_ptr, xsmtame_load_acc16_stride,
                               shape.m, shape.n, true)
GEN_XSMTAME_STRIDE_LOAD_HELPER(xsmtame_mlcte32, uint32_t, ad,
                               xsmtame_acc32_ptr, xsmtame_load_acc32_stride,
                               shape.m, shape.n, true)
GEN_XSMTAME_STRIDE_STORE_HELPER(xsmtame_msate8, uint8_t, td,
                                xsmtame_tile_ptr, xsmtame_store_tile8_stride,
                                shape.m, shape.k, true)
GEN_XSMTAME_STRIDE_STORE_HELPER(xsmtame_msate16, uint16_t, td,
                                xsmtame_tile16_ptr, xsmtame_store_tile16_stride,
                                shape.m, shape.k, true)
GEN_XSMTAME_STRIDE_STORE_HELPER(xsmtame_msate32, uint32_t, td,
                                xsmtame_tile32_ptr, xsmtame_store_tile32_stride,
                                shape.m, shape.k, true)
GEN_XSMTAME_STRIDE_STORE_HELPER(xsmtame_msbte8, uint8_t, td,
                                xsmtame_tile_ptr, xsmtame_store_tile8_stride,
                                shape.n, shape.k, true)
GEN_XSMTAME_STRIDE_STORE_HELPER(xsmtame_msbte16, uint16_t, td,
                                xsmtame_tile16_ptr, xsmtame_store_tile16_stride,
                                shape.n, shape.k, true)
GEN_XSMTAME_STRIDE_STORE_HELPER(xsmtame_msbte32, uint32_t, td,
                                xsmtame_tile32_ptr, xsmtame_store_tile32_stride,
                                shape.n, shape.k, true)
GEN_XSMTAME_STRIDE_STORE_HELPER(xsmtame_mscte8, uint8_t, ad,
                                xsmtame_acc8_ptr, xsmtame_store_acc8_stride,
                                shape.m, shape.n, true)
GEN_XSMTAME_STRIDE_STORE_HELPER(xsmtame_mscte16, uint16_t, ad,
                                xsmtame_acc16_ptr, xsmtame_store_acc16_stride,
                                shape.m, shape.n, true)
GEN_XSMTAME_STRIDE_STORE_HELPER(xsmtame_mscte32, uint32_t, ad,
                                xsmtame_acc32_ptr, xsmtame_store_acc32_stride,
                                shape.m, shape.n, true)

/*
 * ──────────────────────────────────────────
 *  GEMM helpers
 * ──────────────────────────────────────────
 *
 * INT8 → INT32 GEMM  (mmacc.w.b):
 *   acc[ad][m][n] += Σ_{k} (int32)tile_A[ms1][m][k] * (int32)tile_B_T[ms2][n][k]
 *
 * tile_B is stored transposed: shape [N][K], so tile_B_T[ms2][n][k] is natural.
 *
 * FP16 → FP32 GEMM  (mfmacc.s.h):
 *   mfmacc.s.h md, ms2, ms1
 *   acc[md][m][n] += Σ_{k} fp32(A[ms1][m][k]) * fp32(B_T[ms2][n][k])
 */

static inline int32_t xsmtame_acc_add_i32(CPURISCVState *env,
                                          int32_t acc,
                                          int32_t sum)
{
    int64_t wide = (int64_t)acc + (int64_t)sum;

    if (!(env->xmsaten & 0x1)) {
        return (int32_t)wide;
    }

    if (wide > INT32_MAX) {
        env->xmsat = 1;
        return INT32_MAX;
    }
    if (wide < INT32_MIN) {
        env->xmsat = 1;
        return INT32_MIN;
    }

    return (int32_t)wide;
}

static void xsmtame_mmacc_w_b_common(CPURISCVState *env, uint32_t ad,
                                     uint32_t ms2, uint32_t ms1,
                                     bool ms1_unsigned,
                                     bool ms2_unsigned)
{
    AMEShapeInfo shape = xsmtame_shape(env);
    const uint8_t *tA = (const uint8_t *)xsmtame_tile8s_ptr(env, ms1);
    const uint8_t *tBT = (const uint8_t *)xsmtame_tile8s_ptr(env, ms2);
    int32_t *acc = (int32_t *)xsmtame_acc32_ptr(env, ad);
    size_t a_row_bytes = xsmtame_matrix_layout(env, ms1, 0).row_bytes;
    size_t b_row_bytes = xsmtame_matrix_layout(env, ms2, 0).row_bytes;
    size_t acc_cols = xsmtame_matrix_layout(env, ad + AME_NR_TILES,
                                            sizeof(*acc)).cols;
    uint32_t m, n, k;

    for (m = 0; m < shape.m; m++) {
        for (n = 0; n < shape.n; n++) {
            int32_t sum = 0;
            for (k = 0; k < shape.k; k++) {
                uint32_t a_raw = tA[m * a_row_bytes + k];
                uint32_t b_raw = tBT[n * b_row_bytes + k];
                int32_t a = ms1_unsigned ? (int32_t)a_raw : (int32_t)(int8_t)a_raw;
                int32_t b = ms2_unsigned ? (int32_t)b_raw : (int32_t)(int8_t)b_raw;

                sum += a * b;
            }
            acc[m * acc_cols + n] = xsmtame_acc_add_i32(env,
                                                       acc[m * acc_cols + n],
                                                       sum);
        }
    }

    xsmtame_zero_acc_inactive_region(env, ad, sizeof(*acc));
}

void HELPER(xsmtame_mmaccu_w_b)(CPURISCVState *env, uint32_t ad,
                            uint32_t ts2, uint32_t ts1)
{
    xsmtame_mmacc_w_b_common(env, ad, ts2, ts1, true, true);
}

void HELPER(xsmtame_mmaccus_w_b)(CPURISCVState *env, uint32_t ad,
                             uint32_t ts2, uint32_t ts1)
{
    xsmtame_mmacc_w_b_common(env, ad, ts2, ts1, true, false);
}

void HELPER(xsmtame_mmaccsu_w_b)(CPURISCVState *env, uint32_t ad,
                             uint32_t ts2, uint32_t ts1)
{
    xsmtame_mmacc_w_b_common(env, ad, ts2, ts1, false, true);
}

void HELPER(xsmtame_mmacc_w_b)(CPURISCVState *env, uint32_t ad,
                           uint32_t ts2, uint32_t ts1)
{
    xsmtame_mmacc_w_b_common(env, ad, ts2, ts1, false, false);
}

static inline float32 xsmtame_mfmacc_fp16_to_f32(uint16_t raw,
                                                       float_status *fpst)
{
    return float16_to_float32(make_float16(raw), true, fpst);
}

static inline float32 xsmtame_mfmacc_bf16_to_f32(uint16_t raw,
                                                       float_status *fpst)
{
    return bfloat16_to_float32((bfloat16)raw, fpst);
}

static inline uint16_t xsmtame_mfmacc_f32_to_f16_bits(float32 raw,
                                                           float_status *fpst)
{
    return float16_val(float32_to_float16(raw, true, fpst));
}

static inline uint16_t xsmtame_mfmacc_f32_to_bf16_bits(float32 raw,
                                                            float_status *fpst)
{
    return (uint16_t)float32_to_bfloat16(raw, fpst);
}

static inline float32 xsmtame_mfmacc_e4_to_f32(uint8_t raw,
                                                     float_status *fpst)
{
    bool sign = raw >> 7;
    uint8_t exp = (raw >> 3) & 0x0f;
    uint8_t frac = raw & 0x07;
    int16_t unbiased_exp;
    int32_t sig;

    if (exp == 0x0f && frac == 0x07) {
        return float32_default_nan(fpst);
    }

    if (!exp) {
        if (!frac) {
            return make_float32(sign ? 0x80000000u : 0);
        }
        unbiased_exp = -6;
        sig = frac;
    } else {
        unbiased_exp = exp - 7;
        sig = 0x08 | frac;
    }

    return int32_to_float32_scalbn(sign ? -sig : sig,
                                  unbiased_exp - 3, fpst);
}

static inline float32 xsmtame_mfmacc_e5_to_f32(uint8_t raw,
                                                     float_status *fpst)
{
    bool sign = raw >> 7;
    uint8_t exp = (raw >> 2) & 0x1f;
    uint8_t frac = raw & 0x03;
    int16_t unbiased_exp;
    int32_t sig;

    if (exp == 0x1f) {
        if (frac) {
            return float32_default_nan(fpst);
        }
        return make_float32((sign ? 0x80000000u : 0) | 0x7f800000u);
    }

    if (!exp) {
        if (!frac) {
            return make_float32(sign ? 0x80000000u : 0);
        }
        unbiased_exp = -14;
        sig = frac;
    } else {
        unbiased_exp = exp - 15;
        sig = 0x04 | frac;
    }

    return int32_to_float32_scalbn(sign ? -sig : sig,
                                  unbiased_exp - 2, fpst);
}

typedef struct AMEMfmaccInternal30 {
    bool sign;
    int16_t exp;
    uint32_t sig;
    bool is_zero;
} AMEMfmaccInternal30;

typedef struct AMEMfmaccDecodedFloat {
    bool sign;
    int16_t exp;
    uint16_t sig;
    bool is_zero;
} AMEMfmaccDecodedFloat;

typedef struct AMEMfmaccSpecial {
    bool sign;
    bool is_zero;
    bool is_inf;
    bool is_qnan;
    bool is_snan;
} AMEMfmaccSpecial;

typedef struct AMEMfmaccSpecialEval {
    bool invalid;
    bool qnan_seen;
    bool pos_inf_seen;
    bool neg_inf_seen;
    bool all_zero_addends;
    bool pos_zero_seen;
    bool neg_zero_seen;
} AMEMfmaccSpecialEval;

enum {
    XSMTAME_MFMACC_MAX_INTERNAL_K_16 = 2,
    XSMTAME_MFMACC_MAX_INTERNAL_K_8 = 4,
};

static inline uint32_t xsmtame_mfmacc_shrjam32(uint32_t a, uint8_t dist)
{
    if (!dist) {
        return a;
    }
    if (dist < 32) {
        return (a >> dist) | ((uint32_t)(a << ((32 - dist) & 31)) != 0);
    }
    return a ? 1 : 0;
}

static inline uint64_t xsmtame_mfmacc_shrjam64(uint64_t a, uint8_t dist)
{
    if (!dist) {
        return a;
    }
    if (dist < 64) {
        return (a >> dist) | ((uint64_t)(a << ((64 - dist) & 63)) != 0);
    }
    return a ? 1 : 0;
}

static AMEMfmaccSpecial xsmtame_mfmacc_classify_f32(float32 a)
{
    uint32_t ui = float32_val(a);
    uint32_t exp = (ui >> 23) & 0xff;
    uint32_t frac = ui & 0x007fffff;

    return (AMEMfmaccSpecial) {
        .sign = ui >> 31,
        .is_zero = exp == 0 && frac == 0,
        .is_inf = exp == 0xff && frac == 0,
        .is_qnan = exp == 0xff && frac != 0 && (frac & 0x00400000) != 0,
        .is_snan = exp == 0xff && frac != 0 && (frac & 0x00400000) == 0,
    };
}

static AMEMfmaccSpecial xsmtame_mfmacc_classify_f16(uint16_t ui)
{
    uint16_t exp = (ui >> 10) & 0x1f;
    uint16_t frac = ui & 0x03ff;

    return (AMEMfmaccSpecial) {
        .sign = ui >> 15,
        .is_zero = exp == 0 && frac == 0,
        .is_inf = exp == 0x1f && frac == 0,
        .is_qnan = exp == 0x1f && frac != 0 && (frac & 0x0200) != 0,
        .is_snan = exp == 0x1f && frac != 0 && (frac & 0x0200) == 0,
    };
}

static AMEMfmaccSpecial xsmtame_mfmacc_classify_bf16(uint16_t ui)
{
    uint16_t exp = (ui >> 7) & 0xff;
    uint16_t frac = ui & 0x007f;

    return (AMEMfmaccSpecial) {
        .sign = ui >> 15,
        .is_zero = exp == 0 && frac == 0,
        .is_inf = exp == 0xff && frac == 0,
        .is_qnan = exp == 0xff && frac != 0 && (frac & 0x0040) != 0,
        .is_snan = exp == 0xff && frac != 0 && (frac & 0x0040) == 0,
    };
}

static AMEMfmaccSpecial xsmtame_mfmacc_classify_e4(uint8_t ui)
{
    uint8_t exp = (ui >> 3) & 0x0f;
    uint8_t frac = ui & 0x07;

    return (AMEMfmaccSpecial) {
        .sign = ui >> 7,
        .is_zero = exp == 0 && frac == 0,
        .is_inf = false,
        .is_qnan = exp == 0x0f && frac == 0x07,
        .is_snan = false,
    };
}

static AMEMfmaccSpecial xsmtame_mfmacc_classify_e5(uint8_t ui)
{
    uint8_t exp = (ui >> 2) & 0x1f;
    uint8_t frac = ui & 0x03;

    return (AMEMfmaccSpecial) {
        .sign = ui >> 7,
        .is_zero = exp == 0 && frac == 0,
        .is_inf = exp == 0x1f && frac == 0,
        .is_qnan = exp == 0x1f && frac != 0,
        .is_snan = false,
    };
}

static void xsmtame_mfmacc_special_init(AMEMfmaccSpecialEval *eval)
{
    eval->invalid = false;
    eval->qnan_seen = false;
    eval->pos_inf_seen = false;
    eval->neg_inf_seen = false;
    eval->all_zero_addends = true;
    eval->pos_zero_seen = false;
    eval->neg_zero_seen = false;
}

static void xsmtame_mfmacc_special_note_zero(AMEMfmaccSpecialEval *eval,
                                             bool sign)
{
    if (sign) {
        eval->neg_zero_seen = true;
    } else {
        eval->pos_zero_seen = true;
    }
}

static void xsmtame_mfmacc_special_note_inf(AMEMfmaccSpecialEval *eval,
                                            bool sign)
{
    eval->all_zero_addends = false;
    if (sign) {
        eval->neg_inf_seen = true;
    } else {
        eval->pos_inf_seen = true;
    }
}

static void xsmtame_mfmacc_special_scan_c(AMEMfmaccSpecialEval *eval,
                                          AMEMfmaccSpecial c)
{
    if (c.is_snan) {
        eval->invalid = true;
    } else if (c.is_qnan) {
        eval->qnan_seen = true;
    } else if (c.is_inf) {
        xsmtame_mfmacc_special_note_inf(eval, c.sign);
    } else if (c.is_zero) {
        xsmtame_mfmacc_special_note_zero(eval, c.sign);
    } else {
        eval->all_zero_addends = false;
    }
}

static void xsmtame_mfmacc_special_scan_product(AMEMfmaccSpecialEval *eval,
                                                AMEMfmaccSpecial a,
                                                AMEMfmaccSpecial b)
{
    if (a.is_snan || b.is_snan) {
        eval->invalid = true;
    } else if (a.is_qnan || b.is_qnan) {
        eval->qnan_seen = true;
    } else if ((a.is_zero && b.is_inf) || (a.is_inf && b.is_zero)) {
        eval->invalid = true;
    } else if (a.is_inf || b.is_inf) {
        xsmtame_mfmacc_special_note_inf(eval, a.sign ^ b.sign);
    } else if (a.is_zero || b.is_zero) {
        xsmtame_mfmacc_special_note_zero(eval, a.sign ^ b.sign);
    } else {
        eval->all_zero_addends = false;
    }
}

static bool xsmtame_mfmacc_special_finish_f32(const AMEMfmaccSpecialEval *eval,
                                             float32 *out,
                                             float_status *fpst)
{
    if (eval->invalid || (eval->pos_inf_seen && eval->neg_inf_seen)) {
        float_raise(float_flag_invalid, fpst);
        *out = make_float32(0x7fc00000u);
        return true;
    }
    if (eval->qnan_seen) {
        *out = make_float32(0x7fc00000u);
        return true;
    }
    if (eval->pos_inf_seen || eval->neg_inf_seen) {
        *out = make_float32((eval->neg_inf_seen ? 0x80000000u : 0) |
                            0x7f800000u);
        return true;
    }
    if (eval->all_zero_addends) {
        bool sign;

        if (eval->neg_zero_seen && !eval->pos_zero_seen) {
            sign = true;
        } else if (!eval->neg_zero_seen && eval->pos_zero_seen) {
            sign = false;
        } else {
            sign = get_float_rounding_mode(fpst) == float_round_down;
        }
        *out = make_float32(sign ? 0x80000000u : 0);
        return true;
    }
    return false;
}

static inline uint32_t xsmtame_mfmacc_round_to_odd32(uint32_t a,
                                                           uint8_t dist)
{
    uint32_t z;

    if (!dist) {
        return a;
    }
    z = xsmtame_mfmacc_shrjam32(a, dist);
    if (z && (a & ((((uint32_t)1) << (dist < 32 ? dist : 31)) - 1))) {
        z |= 1;
    }
    return z;
}

static uint32_t xsmtame_mfmacc_round_pack_subnormal_frac32(bool sign,
                                                          uint32_t sig,
                                                          uint16_t dist,
                                                          float_status *fpst)
{
    FloatRoundMode rounding_mode = get_float_rounding_mode(fpst);
    uint64_t sig_wide = sig;
    uint64_t discarded;
    uint64_t halfway;
    uint32_t frac;
    int flags = 0;

    g_assert(dist != 0);

    if (dist < 64) {
        frac = sig_wide >> dist;
        discarded = sig_wide & ((UINT64_C(1) << dist) - 1);
        halfway = UINT64_C(1) << (dist - 1);
    } else {
        frac = 0;
        discarded = sig_wide;
        halfway = UINT64_MAX;
    }

    if (discarded != 0) {
        switch (rounding_mode) {
        case float_round_nearest_even:
            if (discarded > halfway ||
                (discarded == halfway && (frac & 1))) {
                frac++;
            }
            break;
        case float_round_ties_away:
            if (discarded >= halfway) {
                frac++;
            }
            break;
        case float_round_down:
            if (sign) {
                frac++;
            }
            break;
        case float_round_up:
            if (!sign) {
                frac++;
            }
            break;
        case float_round_to_zero:
            break;
        default:
            g_assert_not_reached();
        }

        flags |= float_flag_inexact;
        if (get_float_detect_tininess(fpst) ==
                float_tininess_before_rounding ||
            frac < UINT32_C(0x00800000)) {
            flags |= float_flag_underflow;
        }
        float_raise(flags, fpst);
    }

    return frac;
}

static bool xsmtame_mfmacc_decode_float(uint16_t ui,
                                             uint8_t exp_bits,
                                             uint8_t frac_bits,
                                             int16_t exp_bias,
                                             AMEMfmaccDecodedFloat *out)
{
    uint16_t exp_mask = (((uint16_t)1) << exp_bits) - 1;
    uint16_t frac_mask = (((uint16_t)1) << frac_bits) - 1;
    uint16_t exp = (ui >> frac_bits) & exp_mask;
    uint16_t frac = ui & frac_mask;
    int16_t shift_dist;

    out->sign = (ui >> (exp_bits + frac_bits)) & 1;
    out->exp = 0;
    out->sig = 0;
    out->is_zero = false;

    if (exp == exp_mask) {
        return false;
    }
    if (!exp) {
        if (!frac) {
            out->is_zero = true;
            return true;
        }
        shift_dist = 0;
        while (frac < (((uint16_t)1) << frac_bits)) {
            frac <<= 1;
            ++shift_dist;
        }
        out->exp = 1 - exp_bias - shift_dist;
        out->sig = frac;
        return true;
    }

    out->exp = (int16_t)exp - exp_bias;
    out->sig = (((uint16_t)1) << frac_bits) | frac;
    return true;
}

static bool xsmtame_mfmacc_decode_e4(uint8_t ui,
                                          AMEMfmaccDecodedFloat *out)
{
    uint16_t exp = (ui >> 3) & 0x0f;
    uint16_t frac = ui & 0x07;
    int16_t shift_dist;

    out->sign = ui >> 7;
    out->exp = 0;
    out->sig = 0;
    out->is_zero = false;

    if (exp == 0x0f && frac == 0x07) {
        return false;
    }
    if (!exp) {
        if (!frac) {
            out->is_zero = true;
            return true;
        }
        shift_dist = 0;
        while (frac < 0x08) {
            frac <<= 1;
            ++shift_dist;
        }
        out->exp = -6 - shift_dist;
        out->sig = frac;
        return true;
    }

    out->exp = (int16_t)exp - 7;
    out->sig = 0x08 | frac;
    return true;
}

static bool xsmtame_mfmacc_mul_decoded_to_internal30(
    const AMEMfmaccDecodedFloat *a,
    uint8_t frac_bits_a,
    const AMEMfmaccDecodedFloat *b,
    uint8_t frac_bits_b,
    AMEMfmaccInternal30 *out)
{
    uint64_t sig_prod;
    uint8_t frac_bits_prod;

    out->sign = a->sign ^ b->sign;
    out->exp = 0;
    out->sig = 0;
    out->is_zero = false;

    if (a->is_zero || b->is_zero) {
        out->is_zero = true;
        return true;
    }

    sig_prod = (uint64_t)a->sig * (uint64_t)b->sig;
    out->exp = a->exp + b->exp;
    frac_bits_prod = frac_bits_a + frac_bits_b;

    if (sig_prod & (((uint64_t)1) << (frac_bits_prod + 1))) {
        /*
         * The product is already normalized into [2, 4), so bump the
         * exponent and account for the extra leading bit in the later
         * internal30 scaling. Do not right shift `sig_prod` here: that would
         * quantize an otherwise exact product (for example 1479 * 1461) one
         * step too early and can produce a 0x20 FP32 error in `mfmacc.s.h`.
         */
        ++out->exp;
        ++frac_bits_prod;
    }

    if (frac_bits_prod < 26) {
        sig_prod <<= (26 - frac_bits_prod);
    } else if (frac_bits_prod > 26) {
        sig_prod = xsmtame_mfmacc_shrjam64(sig_prod,
                                                frac_bits_prod - 26);
    }

    out->sig = (uint32_t)sig_prod;
    return true;
}

static bool xsmtame_mfmacc_mul_float_to_internal30(uint16_t ui_a,
                                                        uint8_t exp_bits_a,
                                                        uint8_t frac_bits_a,
                                                        int16_t exp_bias_a,
                                                        uint16_t ui_b,
                                                        uint8_t exp_bits_b,
                                                        uint8_t frac_bits_b,
                                                        int16_t exp_bias_b,
                                                        AMEMfmaccInternal30 *out)
{
    AMEMfmaccDecodedFloat a;
    AMEMfmaccDecodedFloat b;

    if (!xsmtame_mfmacc_decode_float(ui_a, exp_bits_a, frac_bits_a,
                                          exp_bias_a, &a) ||
        !xsmtame_mfmacc_decode_float(ui_b, exp_bits_b, frac_bits_b,
                                          exp_bias_b, &b)) {
        return false;
    }

    return xsmtame_mfmacc_mul_decoded_to_internal30(&a, frac_bits_a,
                                                         &b, frac_bits_b,
                                                         out);
}

static inline bool xsmtame_mfmacc_mul_f16_to_internal30(uint16_t ui_a,
                                                              uint16_t ui_b,
                                                              AMEMfmaccInternal30 *out)
{
    return xsmtame_mfmacc_mul_float_to_internal30(ui_a, 5, 10, 15,
                                                       ui_b, 5, 10, 15,
                                                       out);
}

static inline bool xsmtame_mfmacc_mul_bf16_to_internal30(uint16_t ui_a,
                                                               uint16_t ui_b,
                                                               AMEMfmaccInternal30 *out)
{
    return xsmtame_mfmacc_mul_float_to_internal30(ui_a, 8, 7, 127,
                                                       ui_b, 8, 7, 127,
                                                       out);
}

static inline bool xsmtame_mfmacc_mul_e4_to_internal30(uint8_t ui_a,
                                                             uint8_t ui_b,
                                                             AMEMfmaccInternal30 *out)
{
    AMEMfmaccDecodedFloat a;
    AMEMfmaccDecodedFloat b;

    if (!xsmtame_mfmacc_decode_e4(ui_a, &a) ||
        !xsmtame_mfmacc_decode_e4(ui_b, &b)) {
        return false;
    }

    return xsmtame_mfmacc_mul_decoded_to_internal30(&a, 3,
                                                    &b, 3, out);
}

static inline bool xsmtame_mfmacc_mul_e5_to_internal30(uint8_t ui_a,
                                                             uint8_t ui_b,
                                                             AMEMfmaccInternal30 *out)
{
    return xsmtame_mfmacc_mul_float_to_internal30(ui_a, 5, 2, 15,
                                                       ui_b, 5, 2, 15,
                                                       out);
}

static void xsmtame_mfmacc_pack_normalized_internal30(bool sign,
                                                           int16_t exp,
                                                           uint64_t sig,
                                                           AMEMfmaccInternal30 *out)
{
    out->sign = false;
    out->exp = 0;
    out->sig = 0;
    out->is_zero = true;

    if (!sig) {
        return;
    }

    while (sig >= UINT64_C(0x8000000)) {
        sig = xsmtame_mfmacc_shrjam64(sig, 1);
        ++exp;
    }
    while (sig < UINT64_C(0x4000000)) {
        sig <<= 1;
        --exp;
    }

    out->sign = sign;
    out->exp = exp;
    out->sig = (uint32_t)sig;
    out->is_zero = false;
}

static void xsmtame_mfmacc_add_internal30_unified(const AMEMfmaccInternal30 *terms,
                                                        uint8_t term_count,
                                                        AMEMfmaccInternal30 *out)
{
    bool have_non_zero = false;
    int16_t exp_z = 0;
    int64_t sig_sum = 0;
    uint8_t k;

    for (k = 0; k < term_count; ++k) {
        if (terms[k].is_zero) {
            continue;
        }
        if (!have_non_zero || terms[k].exp > exp_z) {
            exp_z = terms[k].exp;
            have_non_zero = true;
        }
    }

    if (!have_non_zero) {
        out->sign = false;
        out->exp = 0;
        out->sig = 0;
        out->is_zero = true;
        return;
    }

    for (k = 0; k < term_count; ++k) {
        uint32_t aligned_sig;

        if (terms[k].is_zero) {
            continue;
        }
        aligned_sig = terms[k].sig;
        if (terms[k].exp < exp_z) {
            aligned_sig = xsmtame_mfmacc_round_to_odd32(aligned_sig,
                                (uint8_t)(exp_z - terms[k].exp));
        }
        sig_sum += terms[k].sign ? -(int64_t)aligned_sig : (int64_t)aligned_sig;
    }

    if (sig_sum < 0) {
        xsmtame_mfmacc_pack_normalized_internal30(true, exp_z,
                                                       (uint64_t)(-sig_sum),
                                                       out);
    } else {
        xsmtame_mfmacc_pack_normalized_internal30(false, exp_z,
                                                       (uint64_t)sig_sum,
                                                       out);
    }
}

static float32 xsmtame_mfmacc_internal30_to_f32(const AMEMfmaccInternal30 *a,
                                                     float_status *fpst)
{
    uint32_t ui_z;
    int16_t exp;
    uint16_t shift_dist;
    uint32_t frac;

    if (a->is_zero) {
        ui_z = (((uint32_t)a->sign) << 31);
        return make_float32(ui_z);
    }

    exp = a->exp + 127;
    if (exp <= 0) {
        /* Subnormals need one extra shift beyond the three guard bits. */
        shift_dist = (uint16_t)(4 - exp);
        frac = xsmtame_mfmacc_round_pack_subnormal_frac32(a->sign, a->sig,
                                                         shift_dist, fpst);
        if (frac >= UINT32_C(0x00800000)) {
            ui_z = (((uint32_t)a->sign) << 31) | (1u << 23);
        } else {
            ui_z = (((uint32_t)a->sign) << 31) | frac;
        }
        return make_float32(ui_z);
    }
    if (exp >= 0xFF) {
        float_raise(float_flag_overflow | float_flag_inexact, fpst);
        ui_z = (((uint32_t)a->sign) << 31) | (0xFFu << 23);
        return make_float32(ui_z);
    }

    if (a->sig & 0x00000007u) {
        float_raise(float_flag_inexact, fpst);
    }
    frac = xsmtame_mfmacc_round_to_odd32(a->sig, 3) & 0x007fffff;
    ui_z = (((uint32_t)a->sign) << 31) | ((uint32_t)exp << 23) | frac;
    return make_float32(ui_z);
}

static float32 xsmtame_mfmacc_add_internal30_to_f32_final(
        const AMEMfmaccInternal30 *acc_int, float32 c, float_status *fpst)
{
    int old_flags = get_float_exception_flags(fpst);
    int product_flags;
    int final_flags;
    float32 product;
    float32 z;

    set_float_exception_flags(0, fpst);
    product = xsmtame_mfmacc_internal30_to_f32(acc_int, fpst);
    product_flags = get_float_exception_flags(fpst);

    set_float_exception_flags(0, fpst);
    z = float32_add(product, c, fpst);
    final_flags = get_float_exception_flags(fpst);
    if ((float32_val(c) & 0x7fffffffu) == 0) {
        final_flags |= product_flags;
    }
    set_float_exception_flags(old_flags | final_flags, fpst);
    return z;
}

/*
 * Large-k fallback only. Special values are screened before these helpers are
 * reached; the remaining fallback is for dot products longer than the direct
 * internal30 finite path models.
 */
static float32 xsmtame_mfmacc_reference_dot16(const uint16_t *lhs,
                                                   const uint16_t *rhs,
                                                   uint8_t k_cols,
                                                   float32 c,
                                                   float32 (*convert)(uint16_t,
                                                                      float_status *),
                                                   float_status *fpst)
{
    uint8_t k;

    for (k = 0; k < k_cols; ++k) {
        c = float32_add(float32_mul(convert(lhs[k], fpst), convert(rhs[k], fpst), fpst),
                        c, fpst);
    }
    return c;
}

static float32 xsmtame_mfmacc_reference_dot8(const uint8_t *lhs,
                                                  const uint8_t *rhs,
                                                  uint8_t k_cols,
                                                  float32 c,
                                                  float32 (*convert)(uint8_t,
                                                                     float_status *),
                                                  float_status *fpst)
{
    uint8_t k;

    for (k = 0; k < k_cols; ++k) {
        c = float32_add(float32_mul(convert(lhs[k], fpst), convert(rhs[k], fpst), fpst),
                        c, fpst);
    }
    return c;
}

static float32 xsmtame_mfmacc_cell16(const uint16_t *lhs,
                                     const uint16_t *rhs,
                                     uint8_t k_cols,
                                     uint8_t max_internal_k,
                                     float32 c,
                                     bool (*mul_to_internal)(uint16_t,
                                                             uint16_t,
                                                             AMEMfmaccInternal30 *),
                                     float32 (*fallback_convert)(uint16_t,
                                                                 float_status *),
                                     AMEMfmaccSpecial (*classify)(uint16_t),
                                     float_status *fpst)
{
    AMEMfmaccInternal30 acc_int;
    AMEMfmaccInternal30 prod_list[4];
    AMEMfmaccSpecialEval eval;
    float32 special_z;
    uint8_t k;

    acc_int.sign = false;
    acc_int.exp = 0;
    acc_int.sig = 0;
    acc_int.is_zero = true;

    xsmtame_mfmacc_special_init(&eval);
    xsmtame_mfmacc_special_scan_c(&eval,
                                  xsmtame_mfmacc_classify_f32(c));
    for (k = 0; k < k_cols; ++k) {
        xsmtame_mfmacc_special_scan_product(&eval, classify(lhs[k]),
                                            classify(rhs[k]));
    }
    if (xsmtame_mfmacc_special_finish_f32(&eval, &special_z, fpst)) {
        return special_z;
    }

    if (k_cols > max_internal_k) {
        return xsmtame_mfmacc_reference_dot16(lhs, rhs, k_cols, c,
                                                   fallback_convert, fpst);
    }

    for (k = 0; k < k_cols; ++k) {
        mul_to_internal(lhs[k], rhs[k], &prod_list[k]);
    }

    xsmtame_mfmacc_add_internal30_unified(prod_list, k_cols, &acc_int);
    return xsmtame_mfmacc_add_internal30_to_f32_final(&acc_int, c, fpst);
}

static float32 xsmtame_mfmacc_cell8(const uint8_t *lhs,
                                    const uint8_t *rhs,
                                    uint8_t k_cols,
                                    uint8_t max_internal_k,
                                    float32 c,
                                    bool (*mul_to_internal)(uint8_t,
                                                            uint8_t,
                                                            AMEMfmaccInternal30 *),
                                    float32 (*fallback_convert)(uint8_t,
                                                                float_status *),
                                    AMEMfmaccSpecial (*classify)(uint8_t),
                                    float_status *fpst)
{
    AMEMfmaccInternal30 acc_int;
    AMEMfmaccInternal30 prod_list[4];
    AMEMfmaccSpecialEval eval;
    float32 special_z;
    uint8_t k;

    acc_int.sign = false;
    acc_int.exp = 0;
    acc_int.sig = 0;
    acc_int.is_zero = true;

    xsmtame_mfmacc_special_init(&eval);
    xsmtame_mfmacc_special_scan_c(&eval,
                                  xsmtame_mfmacc_classify_f32(c));
    for (k = 0; k < k_cols; ++k) {
        xsmtame_mfmacc_special_scan_product(&eval, classify(lhs[k]),
                                            classify(rhs[k]));
    }
    if (xsmtame_mfmacc_special_finish_f32(&eval, &special_z, fpst)) {
        return special_z;
    }

    if (k_cols > max_internal_k) {
        return xsmtame_mfmacc_reference_dot8(lhs, rhs, k_cols, c,
                                                  fallback_convert, fpst);
    }

    for (k = 0; k < k_cols; ++k) {
        mul_to_internal(lhs[k], rhs[k], &prod_list[k]);
    }

    xsmtame_mfmacc_add_internal30_unified(prod_list, k_cols, &acc_int);
    return xsmtame_mfmacc_add_internal30_to_f32_final(&acc_int, c, fpst);
}

static void xsmtame_mfmacc16_common(CPURISCVState *env, uint32_t md,
                                         uint32_t ms2, uint32_t ms1,
                                         bool (*mul_to_internal)(uint16_t,
                                                                 uint16_t,
                                                                 AMEMfmaccInternal30 *),
                                         float32 (*fallback_convert)(uint16_t,
                                                                     float_status *),
                                         AMEMfmaccSpecial (*classify)(uint16_t))
{
    AMEShapeInfo shape = xsmtame_shape(env);
    const uint16_t *tA = xsmtame_tile16_ptr(env, ms1);
    const uint16_t *tBT = xsmtame_tile16_ptr(env, ms2);
    uint32_t *acc = xsmtame_acc32_ptr(env, md);
    size_t a_cols = xsmtame_matrix_layout(env, ms1, sizeof(*tA)).cols;
    size_t b_cols = xsmtame_matrix_layout(env, ms2, sizeof(*tBT)).cols;
    size_t acc_cols = xsmtame_matrix_layout(env, md + AME_NR_TILES,
                                            sizeof(*acc)).cols;
    float_status *fpst = &env->fp_status;
    uint32_t m, n;

    for (m = 0; m < shape.m; m++) {
        for (n = 0; n < shape.n; n++) {
            float32 c = make_float32(acc[m * acc_cols + n]);
            c = xsmtame_mfmacc_cell16(&tA[m * a_cols],
                                      &tBT[n * b_cols],
                                      shape.k,
                                      XSMTAME_MFMACC_MAX_INTERNAL_K_16,
                                      c,
                                      mul_to_internal,
                                      fallback_convert,
                                      classify,
                                      fpst);
            acc[m * acc_cols + n] = float32_val(c);
        }
    }

    xsmtame_zero_acc_inactive_region(env, md, sizeof(*acc));
}

static void xsmtame_mfmacc8_common(CPURISCVState *env, uint32_t md,
                                        uint32_t ms2, uint32_t ms1,
                                        bool (*mul_to_internal)(uint8_t,
                                                                uint8_t,
                                                                AMEMfmaccInternal30 *),
                                        float32 (*fallback_convert)(uint8_t,
                                                                    float_status *),
                                        AMEMfmaccSpecial (*classify)(uint8_t))
{
    AMEShapeInfo shape = xsmtame_shape(env);
    const uint8_t *tA = (const uint8_t *)xsmtame_tile_ptr(env, ms1);
    const uint8_t *tBT = (const uint8_t *)xsmtame_tile_ptr(env, ms2);
    uint32_t *acc = xsmtame_acc32_ptr(env, md);
    size_t a_row_bytes = xsmtame_matrix_layout(env, ms1, 0).row_bytes;
    size_t b_row_bytes = xsmtame_matrix_layout(env, ms2, 0).row_bytes;
    size_t acc_cols = xsmtame_matrix_layout(env, md + AME_NR_TILES,
                                            sizeof(*acc)).cols;
    float_status *fpst = &env->fp_status;
    uint32_t m, n;

    for (m = 0; m < shape.m; m++) {
        for (n = 0; n < shape.n; n++) {
            float32 c = make_float32(acc[m * acc_cols + n]);
            c = xsmtame_mfmacc_cell8(&tA[m * a_row_bytes],
                                     &tBT[n * b_row_bytes],
                                     shape.k,
                                     XSMTAME_MFMACC_MAX_INTERNAL_K_8,
                                     c,
                                     mul_to_internal,
                                     fallback_convert,
                                     classify,
                                     fpst);
            acc[m * acc_cols + n] = float32_val(c);
        }
    }

    xsmtame_zero_acc_inactive_region(env, md, sizeof(*acc));
}

static void xsmtame_mfmacc16_acc16_common(CPURISCVState *env, uint32_t md,
                                               uint32_t ms2, uint32_t ms1,
                                               bool (*mul_to_internal)(uint16_t,
                                                                       uint16_t,
                                                                       AMEMfmaccInternal30 *),
                                               float32 (*acc_to_f32)(uint16_t,
                                                                     float_status *),
                                               uint16_t (*f32_to_acc)(float32,
                                                                      float_status *))
{
    AMEShapeInfo shape = xsmtame_shape(env);
    const uint16_t *tA = xsmtame_tile16_ptr(env, ms1);
    const uint16_t *tBT = xsmtame_tile16_ptr(env, ms2);
    uint16_t *acc = xsmtame_acc16_ptr(env, md);
    size_t a_cols = xsmtame_matrix_layout(env, ms1, sizeof(*tA)).cols;
    size_t b_cols = xsmtame_matrix_layout(env, ms2, sizeof(*tBT)).cols;
    size_t acc_cols = xsmtame_matrix_layout(env, md + AME_NR_TILES,
                                            sizeof(*acc)).cols;
    float_status *fpst = &env->fp_status;
    uint32_t m, n;

    for (m = 0; m < shape.m; m++) {
        for (n = 0; n < shape.n; n++) {
            float32 c = acc_to_f32(acc[m * acc_cols + n], fpst);
            c = xsmtame_mfmacc_cell16(&tA[m * a_cols],
                                      &tBT[n * b_cols],
                                      shape.k,
                                      XSMTAME_MFMACC_MAX_INTERNAL_K_16,
                                      c,
                                      mul_to_internal,
                                      xsmtame_mfmacc_fp16_to_f32,
                                      xsmtame_mfmacc_classify_f16,
                                      fpst);
            acc[m * acc_cols + n] = f32_to_acc(c, fpst);
        }
    }

    xsmtame_zero_acc_inactive_region(env, md, sizeof(*acc));
}

static void xsmtame_mfmacc8_acc16_common(CPURISCVState *env, uint32_t md,
                                              uint32_t ms2, uint32_t ms1,
                                              bool (*mul_to_internal)(uint8_t,
                                                                      uint8_t,
                                                                      AMEMfmaccInternal30 *),
                                              float32 (*acc_to_f32)(uint16_t,
                                                                    float_status *),
                                              uint16_t (*f32_to_acc)(float32,
                                                                     float_status *),
                                              float32 (*fallback_convert)(uint8_t,
                                                                          float_status *),
                                              AMEMfmaccSpecial (*classify)(uint8_t))
{
    AMEShapeInfo shape = xsmtame_shape(env);
    const uint8_t *tA = (const uint8_t *)xsmtame_tile_ptr(env, ms1);
    const uint8_t *tBT = (const uint8_t *)xsmtame_tile_ptr(env, ms2);
    uint16_t *acc = xsmtame_acc16_ptr(env, md);
    size_t a_row_bytes = xsmtame_matrix_layout(env, ms1, 0).row_bytes;
    size_t b_row_bytes = xsmtame_matrix_layout(env, ms2, 0).row_bytes;
    size_t acc_cols = xsmtame_matrix_layout(env, md + AME_NR_TILES,
                                            sizeof(*acc)).cols;
    float_status *fpst = &env->fp_status;
    uint32_t m, n;

    for (m = 0; m < shape.m; m++) {
        for (n = 0; n < shape.n; n++) {
            float32 c = acc_to_f32(acc[m * acc_cols + n], fpst);
            c = xsmtame_mfmacc_cell8(&tA[m * a_row_bytes],
                                     &tBT[n * b_row_bytes],
                                     shape.k,
                                     XSMTAME_MFMACC_MAX_INTERNAL_K_8,
                                     c,
                                     mul_to_internal,
                                     fallback_convert,
                                     classify,
                                     fpst);
            acc[m * acc_cols + n] = f32_to_acc(c, fpst);
        }
    }

    xsmtame_zero_acc_inactive_region(env, md, sizeof(*acc));
}

void HELPER(xsmtame_mfmacc_h_e5)(CPURISCVState *env, uint32_t md,
                             uint32_t ms2, uint32_t ms1)
{
    xsmtame_mfmacc8_acc16_common(env, md, ms2, ms1,
                                      xsmtame_mfmacc_mul_e5_to_internal30,
                                      xsmtame_mfmacc_fp16_to_f32,
                                      xsmtame_mfmacc_f32_to_f16_bits,
                                      xsmtame_mfmacc_e5_to_f32,
                                      xsmtame_mfmacc_classify_e5);
}

void HELPER(xsmtame_mfmacc_h_e4)(CPURISCVState *env, uint32_t md,
                             uint32_t ms2, uint32_t ms1)
{
    xsmtame_mfmacc8_acc16_common(env, md, ms2, ms1,
                                      xsmtame_mfmacc_mul_e4_to_internal30,
                                      xsmtame_mfmacc_fp16_to_f32,
                                      xsmtame_mfmacc_f32_to_f16_bits,
                                      xsmtame_mfmacc_e4_to_f32,
                                      xsmtame_mfmacc_classify_e4);
}

void HELPER(xsmtame_mfmacc_bf16_e5)(CPURISCVState *env, uint32_t md,
                                uint32_t ms2, uint32_t ms1)
{
    xsmtame_mfmacc8_acc16_common(env, md, ms2, ms1,
                                      xsmtame_mfmacc_mul_e5_to_internal30,
                                      xsmtame_mfmacc_bf16_to_f32,
                                      xsmtame_mfmacc_f32_to_bf16_bits,
                                      xsmtame_mfmacc_e5_to_f32,
                                      xsmtame_mfmacc_classify_e5);
}

void HELPER(xsmtame_mfmacc_bf16_e4)(CPURISCVState *env, uint32_t md,
                                uint32_t ms2, uint32_t ms1)
{
    xsmtame_mfmacc8_acc16_common(env, md, ms2, ms1,
                                      xsmtame_mfmacc_mul_e4_to_internal30,
                                      xsmtame_mfmacc_bf16_to_f32,
                                      xsmtame_mfmacc_f32_to_bf16_bits,
                                      xsmtame_mfmacc_e4_to_f32,
                                      xsmtame_mfmacc_classify_e4);
}

void HELPER(xsmtame_mfmacc_h)(CPURISCVState *env, uint32_t md,
                          uint32_t ms2, uint32_t ms1)
{
    xsmtame_mfmacc16_acc16_common(env, md, ms2, ms1,
                                       xsmtame_mfmacc_mul_f16_to_internal30,
                                       xsmtame_mfmacc_fp16_to_f32,
                                       xsmtame_mfmacc_f32_to_f16_bits);
}

void HELPER(xsmtame_mfmacc_s_e5)(CPURISCVState *env, uint32_t md,
                             uint32_t ms2, uint32_t ms1)
{
    xsmtame_mfmacc8_common(env, md, ms2, ms1,
                                xsmtame_mfmacc_mul_e5_to_internal30,
                                xsmtame_mfmacc_e5_to_f32,
                                xsmtame_mfmacc_classify_e5);
}

void HELPER(xsmtame_mfmacc_s_e4)(CPURISCVState *env, uint32_t md,
                             uint32_t ms2, uint32_t ms1)
{
    xsmtame_mfmacc8_common(env, md, ms2, ms1,
                                xsmtame_mfmacc_mul_e4_to_internal30,
                                xsmtame_mfmacc_e4_to_f32,
                                xsmtame_mfmacc_classify_e4);
}

void HELPER(xsmtame_mfmacc_s_h)(CPURISCVState *env, uint32_t md,
                            uint32_t ms2, uint32_t ms1)
{
    xsmtame_mfmacc16_common(env, md, ms2, ms1,
                                 xsmtame_mfmacc_mul_f16_to_internal30,
                                 xsmtame_mfmacc_fp16_to_f32,
                                 xsmtame_mfmacc_classify_f16);
}

void HELPER(xsmtame_mfmacc_s_bf16)(CPURISCVState *env, uint32_t md,
                               uint32_t ms2, uint32_t ms1)
{
    xsmtame_mfmacc16_common(env, md, ms2, ms1,
                                 xsmtame_mfmacc_mul_bf16_to_internal30,
                                 xsmtame_mfmacc_bf16_to_f32,
                                 xsmtame_mfmacc_classify_bf16);
}
/*
 * ──────────────────────────────────────────
 *  MISC helpers: mmov
 * ──────────────────────────────────────────
 */

void HELPER(xsmtame_mmov_mm)(CPURISCVState *env, uint32_t md,
                                  uint32_t ms1)
{
    size_t dst_size;
    size_t src_size;
    uint8_t *dst = xsmtame_matrix_ptr(env, md, &dst_size);
    uint8_t *src = xsmtame_matrix_ptr(env, ms1, &src_size);

    if (dst == src) {
        return;
    }

    {
        size_t dst_row_bytes = xsmtame_matrix_layout(env, md, 0).row_bytes;
        size_t src_row_bytes = xsmtame_matrix_layout(env, ms1, 0).row_bytes;
        size_t copy_bytes = MIN(dst_row_bytes, src_row_bytes);
        size_t dst_rows = dst_row_bytes ? dst_size / dst_row_bytes : 0;
        size_t src_rows = src_row_bytes ? src_size / src_row_bytes : 0;
        size_t rows = MIN(dst_rows, src_rows);
        size_t row;

        for (row = 0; row < rows; row++) {
            memmove(dst + row * dst_row_bytes,
                    src + row * src_row_bytes,
                    copy_bytes);
        }
    }
}

target_ulong HELPER(xsmtame_mmovb_x_m)(CPURISCVState *env, uint32_t ms2,
                                            target_ulong idx)
{
    size_t reg_size;
    uint8_t *src = xsmtame_matrix_ptr(env, ms2, &reg_size);
    size_t offset = xsmtame_mmov_elem_offset(reg_size, 1, idx);

    return (target_ulong)(target_long)(int8_t)src[offset];
}

target_ulong HELPER(xsmtame_mmovh_x_m)(CPURISCVState *env, uint32_t ms2,
                                            target_ulong idx)
{
    size_t reg_size;
    uint8_t *src = xsmtame_matrix_ptr(env, ms2, &reg_size);
    size_t offset = xsmtame_mmov_elem_offset(reg_size, 2, idx);

    return (target_ulong)(target_long)(int16_t)lduw_le_p(src + offset);
}

target_ulong HELPER(xsmtame_mmovw_x_m)(CPURISCVState *env, uint32_t ms2,
                                            target_ulong idx)
{
    size_t reg_size;
    uint8_t *src = xsmtame_matrix_ptr(env, ms2, &reg_size);
    size_t offset = xsmtame_mmov_elem_offset(reg_size, 4, idx);

    return (target_ulong)(target_long)(int32_t)ldl_le_p(src + offset);
}

target_ulong HELPER(xsmtame_mmovd_x_m)(CPURISCVState *env, uint32_t ms2,
                                            target_ulong idx)
{
    size_t reg_size;
    uint8_t *src = xsmtame_matrix_ptr(env, ms2, &reg_size);
    size_t offset = xsmtame_mmov_elem_offset(reg_size, 8, idx);

    return ldq_le_p(src + offset);
}

void HELPER(xsmtame_mmovb_m_x)(CPURISCVState *env, uint32_t md,
                                    target_ulong idx,
                                    target_ulong value)
{
    xsmtame_mmov_m_x_common(env, md, idx, value, 1);
}

void HELPER(xsmtame_mmovh_m_x)(CPURISCVState *env, uint32_t md,
                                    target_ulong idx,
                                    target_ulong value)
{
    xsmtame_mmov_m_x_common(env, md, idx, value, 2);
}

void HELPER(xsmtame_mmovw_m_x)(CPURISCVState *env, uint32_t md,
                                    target_ulong idx,
                                    target_ulong value)
{
    xsmtame_mmov_m_x_common(env, md, idx, value, 4);
}

void HELPER(xsmtame_mmovd_m_x)(CPURISCVState *env, uint32_t md,
                                    target_ulong idx,
                                    target_ulong value)
{
    xsmtame_mmov_m_x_common(env, md, idx, value, 8);
}

static void xsmtame_mdup_common(CPURISCVState *env, uint32_t md,
                                target_ulong value, size_t elem_size)
{
    size_t reg_size;
    uint8_t *dst = xsmtame_matrix_ptr(env, md, &reg_size);
    size_t off;

    g_assert(elem_size != 0);
    g_assert(reg_size % elem_size == 0);

    for (off = 0; off < reg_size; off += elem_size) {
        switch (elem_size) {
        case 1:
            stb_p(dst + off, value);
            break;
        case 2:
            stw_le_p(dst + off, value);
            break;
        case 4:
            stl_le_p(dst + off, value);
            break;
        case 8:
            stq_le_p(dst + off, value);
            break;
        default:
            g_assert_not_reached();
        }
    }
}

void HELPER(xsmtame_mdupb_m_x)(CPURISCVState *env, uint32_t md,
                               target_ulong value)
{
    xsmtame_mdup_common(env, md, value, 1);
}

void HELPER(xsmtame_mduph_m_x)(CPURISCVState *env, uint32_t md,
                               target_ulong value)
{
    xsmtame_mdup_common(env, md, value, 2);
}

void HELPER(xsmtame_mdupw_m_x)(CPURISCVState *env, uint32_t md,
                               target_ulong value)
{
    xsmtame_mdup_common(env, md, value, 4);
}

void HELPER(xsmtame_mdupd_m_x)(CPURISCVState *env, uint32_t md,
                               target_ulong value)
{
    xsmtame_mdup_common(env, md, value, 8);
}

static void xsmtame_mpack_common(CPURISCVState *env, uint32_t md,
                                      uint32_t ms2, uint32_t ms1,
                                      bool high1, bool high2)
{
    size_t reg_size;
    uint8_t tmp[AME_ACC_LEN_B];
    uint8_t *dst = xsmtame_matrix_ptr(env, md, &reg_size);
    uint8_t *src2 = xsmtame_matrix_ptr(env, ms2, NULL);
    uint8_t *src1 = xsmtame_matrix_ptr(env, ms1, NULL);
    size_t row_bytes = xsmtame_matrix_layout(env, md, 0).row_bytes;
    size_t half = row_bytes / 2;
    size_t rows = reg_size / row_bytes;
    size_t row;

    xsmtame_validate_same_matrix_layout(env, md, ms1);
    xsmtame_validate_same_matrix_layout(env, md, ms2);

    g_assert(row_bytes % 2 == 0);

    for (row = 0; row < rows; row++) {
        size_t off = row * row_bytes;

        memcpy(tmp + off,
               src1 + off + (high1 ? half : 0),
               half);
        memcpy(tmp + off + half,
               src2 + off + (high2 ? half : 0),
               half);
    }

    memcpy(dst, tmp, reg_size);
}

static void xsmtame_mrbc_common(CPURISCVState *env, uint32_t md,
                                uint32_t ms1, uint32_t amount)
{
    size_t reg_size;
    uint8_t tmp[AME_ACC_LEN_B];
    uint8_t *dst = xsmtame_matrix_ptr(env, md, &reg_size);
    uint8_t *src = xsmtame_matrix_ptr(env, ms1, NULL);
    size_t row_bytes = xsmtame_matrix_layout(env, md, 0).row_bytes;
    size_t rows = reg_size / row_bytes;
    size_t row;

    xsmtame_validate_same_matrix_layout(env, md, ms1);

    if (rows != 0) {
        amount &= rows - 1;
    }

    for (row = 0; row < rows; row++) {
        memcpy(tmp + row * row_bytes, src + amount * row_bytes, row_bytes);
    }

    memcpy(dst, tmp, reg_size);
}

static void xsmtame_mcbc_common(CPURISCVState *env, uint32_t md,
                                uint32_t ms1, uint32_t amount,
                                size_t elem_size)
{
    size_t reg_size;
    uint8_t tmp[AME_ACC_LEN_B];
    uint8_t *dst = xsmtame_matrix_ptr(env, md, &reg_size);
    uint8_t *src = xsmtame_matrix_ptr(env, ms1, NULL);
    size_t row_bytes = xsmtame_matrix_layout(env, md, 0).row_bytes;
    size_t rows = reg_size / row_bytes;
    size_t cols = xsmtame_matrix_layout(env, md, elem_size).cols;
    size_t row;
    size_t col;

    xsmtame_validate_same_matrix_layout(env, md, ms1);

    if (cols != 0) {
        amount &= cols - 1;
    }

    for (row = 0; row < rows; row++) {
        uint8_t *dst_row = tmp + row * row_bytes;
        uint8_t *src_elem = src + row * row_bytes + amount * elem_size;

        for (col = 0; col < cols; col++) {
            memcpy(dst_row + col * elem_size, src_elem, elem_size);
        }
    }

    memcpy(dst, tmp, reg_size);
}

static void xsmtame_mrslide_common(CPURISCVState *env, uint32_t md,
                                        uint32_t ms1, uint32_t amount,
                                        bool up)
{
    size_t reg_size;
    uint8_t tmp[AME_ACC_LEN_B];
    uint8_t *dst = xsmtame_matrix_ptr(env, md, &reg_size);
    uint8_t *src = xsmtame_matrix_ptr(env, ms1, NULL);
    size_t row_bytes = xsmtame_matrix_layout(env, md, 0).row_bytes;
    size_t rows = reg_size / row_bytes;
    size_t row;

    xsmtame_validate_same_matrix_layout(env, md, ms1);

    if (rows != 0) {
        amount &= rows - 1;
    }

    memset(tmp, 0, reg_size);
    for (row = 0; row < rows; row++) {
        size_t src_row;

        if (up) {
            if (row < amount) {
                continue;
            }
            src_row = row - amount;
        } else {
            src_row = row + amount;
            if (src_row >= rows) {
                continue;
            }
        }

        memcpy(tmp + row * row_bytes, src + src_row * row_bytes, row_bytes);
    }

    memcpy(dst, tmp, reg_size);
}

static void xsmtame_mcslide_common(CPURISCVState *env, uint32_t md,
                                        uint32_t ms1, uint32_t amount,
                                        size_t elem_size, bool up)
{
    size_t reg_size;
    uint8_t tmp[AME_ACC_LEN_B];
    uint8_t *dst = xsmtame_matrix_ptr(env, md, &reg_size);
    uint8_t *src = xsmtame_matrix_ptr(env, ms1, NULL);
    size_t row_bytes = xsmtame_matrix_layout(env, md, 0).row_bytes;
    size_t rows = reg_size / row_bytes;
    size_t cols = xsmtame_matrix_layout(env, md, elem_size).cols;
    size_t row;
    size_t col;

    xsmtame_validate_same_matrix_layout(env, md, ms1);

    if (cols != 0) {
        amount &= cols - 1;
    }

    memset(tmp, 0, reg_size);
    for (row = 0; row < rows; row++) {
        for (col = 0; col < cols; col++) {
            size_t src_col;

            if (up) {
                if (col < amount) {
                    continue;
                }
                src_col = col - amount;
            } else {
                src_col = col + amount;
                if (src_col >= cols) {
                    continue;
                }
            }

            memcpy(tmp + row * row_bytes + col * elem_size,
                   src + row * row_bytes + src_col * elem_size,
                   elem_size);
        }
    }

    memcpy(dst, tmp, reg_size);
}

void HELPER(xsmtame_mpack_mm)(CPURISCVState *env, uint32_t md,
                                   uint32_t ms2, uint32_t ms1)
{
    xsmtame_mpack_common(env, md, ms2, ms1, false, false);
}

void HELPER(xsmtame_mpackhl_mm)(CPURISCVState *env, uint32_t md,
                                     uint32_t ms2, uint32_t ms1)
{
    xsmtame_mpack_common(env, md, ms2, ms1, true, false);
}

void HELPER(xsmtame_mpackhh_mm)(CPURISCVState *env, uint32_t md,
                                     uint32_t ms2, uint32_t ms1)
{
    xsmtame_mpack_common(env, md, ms2, ms1, true, true);
}

void HELPER(xsmtame_mrslidedown)(CPURISCVState *env, uint32_t md,
                                      uint32_t ms1, uint32_t amount)
{
    xsmtame_mrslide_common(env, md, ms1, amount, false);
}

void HELPER(xsmtame_mrslideup)(CPURISCVState *env, uint32_t md,
                                    uint32_t ms1, uint32_t amount)
{
    xsmtame_mrslide_common(env, md, ms1, amount, true);
}

void HELPER(xsmtame_mcslidedown_b)(CPURISCVState *env, uint32_t md,
                                        uint32_t ms1, uint32_t amount)
{
    xsmtame_mcslide_common(env, md, ms1, amount, 1, false);
}

void HELPER(xsmtame_mcslidedown_h)(CPURISCVState *env, uint32_t md,
                                        uint32_t ms1, uint32_t amount)
{
    xsmtame_mcslide_common(env, md, ms1, amount, 2, false);
}

void HELPER(xsmtame_mcslidedown_w)(CPURISCVState *env, uint32_t md,
                                        uint32_t ms1, uint32_t amount)
{
    xsmtame_mcslide_common(env, md, ms1, amount, 4, false);
}

void HELPER(xsmtame_mcslidedown_d)(CPURISCVState *env, uint32_t md,
                                        uint32_t ms1, uint32_t amount)
{
    xsmtame_mcslide_common(env, md, ms1, amount, 8, false);
}

void HELPER(xsmtame_mcslideup_b)(CPURISCVState *env, uint32_t md,
                                      uint32_t ms1, uint32_t amount)
{
    xsmtame_mcslide_common(env, md, ms1, amount, 1, true);
}

void HELPER(xsmtame_mcslideup_h)(CPURISCVState *env, uint32_t md,
                                      uint32_t ms1, uint32_t amount)
{
    xsmtame_mcslide_common(env, md, ms1, amount, 2, true);
}

void HELPER(xsmtame_mcslideup_w)(CPURISCVState *env, uint32_t md,
                                      uint32_t ms1, uint32_t amount)
{
    xsmtame_mcslide_common(env, md, ms1, amount, 4, true);
}

void HELPER(xsmtame_mcslideup_d)(CPURISCVState *env, uint32_t md,
                                      uint32_t ms1, uint32_t amount)
{
    xsmtame_mcslide_common(env, md, ms1, amount, 8, true);
}

void HELPER(xsmtame_mrbca_mv_i)(CPURISCVState *env, uint32_t md,
                                     uint32_t ms1, uint32_t amount)
{
    xsmtame_mrbc_common(env, md, ms1, amount);
}

void HELPER(xsmtame_mcbcab_mv_i)(CPURISCVState *env, uint32_t md,
                                      uint32_t ms1, uint32_t amount)
{
    xsmtame_mcbc_common(env, md, ms1, amount, 1);
}

void HELPER(xsmtame_mcbcah_mv_i)(CPURISCVState *env, uint32_t md,
                                      uint32_t ms1, uint32_t amount)
{
    xsmtame_mcbc_common(env, md, ms1, amount, 2);
}

void HELPER(xsmtame_mcbcaw_mv_i)(CPURISCVState *env, uint32_t md,
                                      uint32_t ms1, uint32_t amount)
{
    xsmtame_mcbc_common(env, md, ms1, amount, 4);
}

void HELPER(xsmtame_mcbcad_mv_i)(CPURISCVState *env, uint32_t md,
                                      uint32_t ms1, uint32_t amount)
{
    xsmtame_mcbc_common(env, md, ms1, amount, 8);
}

/*
 * ──────────────────────────────────────────
 *  Control helpers
 * ──────────────────────────────────────────
 */

/* mzero{,2,4,8}r : zero count unified matrix register slots starting at md */
void HELPER(xsmtame_mzero)(CPURISCVState *env, uint32_t md,
                                uint32_t count)
{
    uint32_t i;

    for (i = 0; i < count; i++) {
        uint32_t reg = md + i;

        if (reg < AME_NR_TILES) {
            memset((uint8_t *)env->ame_tile + reg * ame_env_tlenb(env), 0,
                   ame_env_tlenb(env));
        } else if (reg < AME_NR_TILES + AME_NR_ACCS) {
            memset((uint8_t *)env->ame_acc +
                   (reg - AME_NR_TILES) * ame_env_acc_len_b(env), 0,
                   ame_env_acc_len_b(env));
        }
    }
}

/* mrelease : set mstatus.MS → 01 (Initial) */
void HELPER(xsmtame_mrelease)(CPURISCVState *env)
{
#ifndef CONFIG_USER_ONLY
    /*
     * Clear MS bits [26:25], then set to 01 (Initial).
     * MSTATUS_MS = 0x06000000 covers both bits.
     */
    env->mstatus = (env->mstatus & ~MSTATUS_MS)
                   | (1ULL << 25);   /* 01 in bits [26:25] */
#endif
}
