/*
 * AME mmacc regression tests.
 *
 * Verifies `mmacc.w.b` operand orientation follows `A(ms1) * B_T(ms2)`.
 *
 * Copyright (c) 2026
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "ame_insn.h"

#define AME_STRIDE_TILE_LOAD_ENC(func4, size, td, rs1, rs2) \
    (((uint32_t)(func4) << 28) | ((uint32_t)0x1 << 26) | \
     (((uint32_t)(rs2) & 0x1fu) << 20) | (((uint32_t)(rs1) & 0x1fu) << 15) | \
     (((uint32_t)(size) & 0x3u) << 10) | (((uint32_t)(td) & 0x3u) << 7) | \
     AME_OP)

#define MLAE8(td, rs1, rs2) AME_STRIDE_TILE_LOAD_ENC(0x0, 0x0, td, rs1, rs2)
#define MLATE8(td, rs1, rs2) \
    AME_STRIDE_TILE_LOAD_ENC(0x4, 0x0, td, rs1, rs2)
#define MLCE8(ad, rs1, rs2) \
    (AME_STRIDE_TILE_LOAD_ENC(0x2, 0x0, ad, rs1, rs2) | (1u << 9))
#define MLCTE8(ad, rs1, rs2) \
    (AME_STRIDE_TILE_LOAD_ENC(0x6, 0x0, ad, rs1, rs2) | (1u << 9))

enum {
    AME_TILE_BYTES = 128,
    AME_TILE_ROW_BYTES = 4,
    AME_ROW_COUNT = AME_TILE_BYTES / AME_TILE_ROW_BYTES,
    AME_ACC_BYTES = 4096,
    AME_ACC_WORDS = AME_ACC_BYTES / sizeof(int32_t),
};

static inline void clear_fflags(void)
{
    __asm__ volatile ("csrw fflags, zero" : : : "memory");
}

static inline unsigned long read_fflags(void)
{
    unsigned long value;

    __asm__ volatile ("csrr %0, fflags" : "=r"(value));
    return value;
}

static inline void load_tile8(unsigned reg, const void *ptr)
{
    register const void *a0 __asm__("a0") = ptr;

    switch (reg) {
    case 0:
        __asm__ volatile (".4byte %0" :: "i"(MLME8(0, 10)), "r"(a0) : "memory");
        break;
    case 1:
        __asm__ volatile (".4byte %0" :: "i"(MLME8(1, 10)), "r"(a0) : "memory");
        break;
    default:
        TEST_ASSERT(0, "unsupported tile load register");
        break;
    }
}

static inline void load_tile8_stride(unsigned reg, const void *ptr,
                                     ptrdiff_t stride)
{
    register const void *a0 __asm__("a0") = ptr;
    register ptrdiff_t a1 __asm__("a1") = stride;

    switch (reg) {
    case 0:
        __asm__ volatile (".4byte %0" :: "i"(MLAE8(0, 10, 11)),
                          "r"(a0), "r"(a1) : "memory");
        break;
    default:
        TEST_ASSERT(0, "unsupported tile stride load register");
        break;
    }
}

static inline void load_tile8_stride_transpose(unsigned reg, const void *ptr,
                                               ptrdiff_t stride)
{
    register const void *a0 __asm__("a0") = ptr;
    register ptrdiff_t a1 __asm__("a1") = stride;

    switch (reg) {
    case 0:
        __asm__ volatile (".4byte %0" :: "i"(MLATE8(0, 10, 11)),
                          "r"(a0), "r"(a1) : "memory");
        break;
    default:
        TEST_ASSERT(0, "unsupported tile transpose load register");
        break;
    }
}

static inline void load_acc8_stride(unsigned reg, const void *ptr,
                                    ptrdiff_t stride)
{
    register const void *a0 __asm__("a0") = ptr;
    register ptrdiff_t a1 __asm__("a1") = stride;

    switch (reg) {
    case 4:
        __asm__ volatile (".4byte %0" :: "i"(MLCE8(0, 10, 11)),
                          "r"(a0), "r"(a1) : "memory");
        break;
    default:
        TEST_ASSERT(0, "unsupported acc stride load register");
        break;
    }
}

static inline void load_acc8_stride_transpose(unsigned reg, const void *ptr,
                                              ptrdiff_t stride)
{
    register const void *a0 __asm__("a0") = ptr;
    register ptrdiff_t a1 __asm__("a1") = stride;

    switch (reg) {
    case 4:
        __asm__ volatile (".4byte %0" :: "i"(MLCTE8(0, 10, 11)),
                          "r"(a0), "r"(a1) : "memory");
        break;
    default:
        TEST_ASSERT(0, "unsupported acc transpose load register");
        break;
    }
}

static inline void store_tile8(unsigned reg, void *ptr)
{
    register void *a0 __asm__("a0") = ptr;

    switch (reg) {
    case 0:
        __asm__ volatile (".4byte %0" :: "i"(MSME8(0, 10)), "r"(a0) : "memory");
        break;
    default:
        TEST_ASSERT(0, "unsupported tile store register");
        break;
    }
}

static inline void load_tile16(unsigned reg, const void *ptr)
{
    register const void *a0 __asm__("a0") = ptr;

    switch (reg) {
    case 0:
        __asm__ volatile (".4byte %0" :: "i"(MLME16(0, 10)), "r"(a0) : "memory");
        break;
    case 1:
        __asm__ volatile (".4byte %0" :: "i"(MLME16(1, 10)), "r"(a0) : "memory");
        break;
    default:
        TEST_ASSERT(0, "unsupported tile16 load register");
        break;
    }
}

static inline void load_acc32(unsigned reg, const void *ptr)
{
    register const void *a0 __asm__("a0") = ptr;

    switch (reg) {
    case 4:
        __asm__ volatile (".4byte %0" :: "i"(MLME32(4, 10)), "r"(a0) : "memory");
        break;
    default:
        TEST_ASSERT(0, "unsupported acc load register");
        break;
    }
}

static inline void store_acc32(unsigned reg, void *ptr)
{
    register void *a0 __asm__("a0") = ptr;

    switch (reg) {
    case 4:
        __asm__ volatile (".4byte %0" :: "i"(MSME32(4, 10)), "r"(a0) : "memory");
        break;
    default:
        TEST_ASSERT(0, "unsupported acc store register");
        break;
    }
}

static inline void load_acc16(unsigned reg, const void *ptr)
{
    register const void *a0 __asm__("a0") = ptr;

    switch (reg) {
    case 4:
        __asm__ volatile (".4byte %0" :: "i"(MLME16(4, 10)), "r"(a0) : "memory");
        break;
    default:
        TEST_ASSERT(0, "unsupported acc16 load register");
        break;
    }
}

static inline void store_acc16(unsigned reg, void *ptr)
{
    register void *a0 __asm__("a0") = ptr;

    switch (reg) {
    case 4:
        __asm__ volatile (".4byte %0" :: "i"(MSME16(4, 10)), "r"(a0) : "memory");
        break;
    default:
        TEST_ASSERT(0, "unsupported acc16 store register");
        break;
    }
}

static void test_mmacc_operand_order(void)
{
    int8_t tile_a[AME_TILE_BYTES] = {
        [0] = 1, [1] = 2,
        [AME_TILE_ROW_BYTES] = 3, [AME_TILE_ROW_BYTES + 1] = 4,
    };
    int8_t tile_bt[AME_TILE_BYTES] = {
        [0] = 5, [1] = 6,
        [AME_TILE_ROW_BYTES] = 7, [AME_TILE_ROW_BYTES + 1] = 8,
    };
    int32_t acc_out[AME_ACC_WORDS];

    memset(acc_out, 0, sizeof(acc_out));

    AME_INSN(MZERO8R(0));
    AME_INSN(MSETTILEMI(2));
    AME_INSN(MSETTILENI(2));
    AME_INSN(MSETTILEKI(2));

    load_tile8(0, tile_a);
    load_tile8(1, tile_bt);

    AME_INSN(MMACC_W_B(0, 1, 0));

    store_acc32(4, acc_out);

    TEST_ASSERT(acc_out[0] == 17, "mmacc cell[0,0] matches A * B_T");
    TEST_ASSERT(acc_out[1] == 23, "mmacc cell[0,1] uses ms1 as A");
    TEST_ASSERT(acc_out[AME_ROW_COUNT] == 39,
                "mmacc cell[1,0] uses ms2 as B_T");
    TEST_ASSERT(acc_out[AME_ROW_COUNT + 1] == 53,
                "mmacc cell[1,1] matches A * B_T");
}

static void test_strided_load_uses_physical_row_stride(void)
{
    const unsigned long row_bytes = AME_TILE_ROW_BYTES;
    const unsigned long total_rows = AME_TILE_BYTES / row_bytes;
    uint8_t mem_src[4] = { 1, 2, 3, 4 };
    uint8_t tile_init[AME_TILE_BYTES];
    uint8_t mem_dst[AME_TILE_BYTES];
    bool active_tails_zero = true;
    bool inactive_rows_zero = true;
    unsigned long row;
    unsigned long col;

    memset(tile_init, 0xa5, sizeof(tile_init));
    memset(mem_dst, 0xff, sizeof(mem_dst));

    TEST_ASSERT(row_bytes > 2, "xtrlenb exposes physical row padding");

    AME_INSN(MSETTILEMI(2));
    AME_INSN(MSETTILENI(1));
    AME_INSN(MSETTILEKI(2));

    load_tile8(0, tile_init);
    load_tile8_stride(0, mem_src, 2);
    store_tile8(0, mem_dst);

    TEST_ASSERT(mem_dst[0] == 1, "mlae8 keeps row0 col0");
    TEST_ASSERT(mem_dst[1] == 2, "mlae8 keeps row0 col1");
    TEST_ASSERT(mem_dst[row_bytes] == 3, "mlae8 uses physical row stride for row1 col0");
    TEST_ASSERT(mem_dst[row_bytes + 1] == 4, "mlae8 uses physical row stride for row1 col1");

    for (row = 0; row < 2; row++) {
        for (col = 2; col < row_bytes; col++) {
            active_tails_zero &= mem_dst[row * row_bytes + col] == 0;
        }
    }
    for (row = 2; row < total_rows; row++) {
        for (col = 0; col < row_bytes; col++) {
            inactive_rows_zero &= mem_dst[row * row_bytes + col] == 0;
        }
    }
    TEST_ASSERT(active_tails_zero, "mlae8 clears active-row tails");
    TEST_ASSERT(inactive_rows_zero, "mlae8 clears inactive rows");
}

static void test_transpose_tile_load_clears_inactive_region(void)
{
    const unsigned long row_bytes = AME_TILE_ROW_BYTES;
    uint8_t mem_src[2] = { 0x11, 0x22 };
    uint8_t tile_init[AME_TILE_BYTES];
    uint8_t mem_dst[AME_TILE_BYTES];
    bool inactive_region_zero = true;
    unsigned long i;

    memset(tile_init, 0xa5, sizeof(tile_init));
    memset(mem_dst, 0xff, sizeof(mem_dst));

    TEST_ASSERT(row_bytes > 2, "xtrlenb exposes transpose-load padding");

    AME_INSN(MSETTILEMI(1));
    AME_INSN(MSETTILENI(1));
    AME_INSN(MSETTILEKI(2));

    load_tile8(0, tile_init);
    load_tile8_stride_transpose(0, mem_src, 1);
    store_tile8(0, mem_dst);

    TEST_ASSERT(mem_dst[0] == mem_src[0], "mlate8 keeps row0 col0");
    TEST_ASSERT(mem_dst[1] == mem_src[1], "mlate8 keeps row0 col1");
    for (i = 2; i < AME_TILE_BYTES; i++) {
        inactive_region_zero &= mem_dst[i] == 0;
    }
    TEST_ASSERT(inactive_region_zero,
                "mlate8 clears active-row tail and inactive rows");
}

static void test_acc_load_zero_fill(void)
{
    const unsigned long total_rows = AME_ROW_COUNT;
    const unsigned long acc_row_bytes = AME_ACC_BYTES / total_rows;
    uint8_t mem_src[2] = { 0x31, 0x32 };
    uint8_t acc_init[AME_ACC_BYTES];
    uint8_t acc_out[AME_ACC_BYTES];
    bool active_tail_zero = true;
    bool inactive_rows_preserved = true;
    bool active_tail_preserved = true;
    bool inactive_prefix_zero = true;
    bool inactive_tails_preserved = true;
    unsigned long row;
    unsigned long col;

    memset(acc_init, 0xa5, sizeof(acc_init));
    memset(acc_out, 0xff, sizeof(acc_out));

    TEST_ASSERT(acc_row_bytes > 2,
                "accumulator layout exposes inactive columns");

    AME_INSN(MSETTILEMI(1));
    AME_INSN(MSETTILENI(2));
    AME_INSN(MSETTILEKI(1));

    load_acc32(4, acc_init);
    load_acc8_stride(4, mem_src, 2);
    store_acc32(4, acc_out);

    TEST_ASSERT(acc_out[0] == mem_src[0], "mlce8 keeps row0 col0");
    TEST_ASSERT(acc_out[1] == mem_src[1], "mlce8 keeps row0 col1");
    for (col = 2; col < acc_row_bytes; col++) {
        active_tail_zero &= acc_out[col] == 0;
    }
    for (row = 1; row < total_rows; row++) {
        for (col = 0; col < acc_row_bytes; col++) {
            inactive_rows_preserved &=
                acc_out[row * acc_row_bytes + col] == 0xa5;
        }
    }
    TEST_ASSERT(active_tail_zero, "mlce8 clears active-row tail");
    TEST_ASSERT(inactive_rows_preserved, "mlce8 preserves inactive rows");

    load_acc32(4, acc_init);
    load_acc8_stride_transpose(4, mem_src, 1);
    store_acc32(4, acc_out);

    TEST_ASSERT(acc_out[0] == mem_src[0], "mlcte8 keeps row0 col0");
    TEST_ASSERT(acc_out[1] == mem_src[1], "mlcte8 keeps row0 col1");
    for (col = 2; col < acc_row_bytes; col++) {
        active_tail_preserved &= acc_out[col] == 0xa5;
    }
    for (row = 1; row < total_rows; row++) {
        for (col = 0; col < 2; col++) {
            inactive_prefix_zero &=
                acc_out[row * acc_row_bytes + col] == 0;
        }
        for (col = 2; col < acc_row_bytes; col++) {
            inactive_tails_preserved &=
                acc_out[row * acc_row_bytes + col] == 0xa5;
        }
    }
    TEST_ASSERT(active_tail_preserved, "mlcte8 preserves active-row tail");
    TEST_ASSERT(inactive_prefix_zero,
                "mlcte8 clears loaded columns in inactive rows");
    TEST_ASSERT(inactive_tails_preserved,
                "mlcte8 preserves inactive-row tails");
}

static void test_mmacc_whole_load_uses_physical_row_stride(void)
{
    const unsigned long row_bytes = AME_TILE_ROW_BYTES;
    int8_t tile_a[AME_TILE_BYTES];
    int8_t tile_bt[AME_TILE_BYTES];
    int32_t acc_out[AME_ACC_WORDS];

    memset(tile_a, 0, sizeof(tile_a));
    memset(tile_bt, 0, sizeof(tile_bt));
    memset(acc_out, 0, sizeof(acc_out));

    TEST_ASSERT(row_bytes > 2, "xtrlenb exposes multi-column physical rows");

    tile_a[0] = 1;
    tile_a[1] = 2;
    tile_a[row_bytes] = 3;
    tile_a[row_bytes + 1] = 4;

    tile_bt[0] = 5;
    tile_bt[1] = 6;
    tile_bt[row_bytes] = 7;
    tile_bt[row_bytes + 1] = 8;

    AME_INSN(MZERO8R(0));
    AME_INSN(MSETTILEMI(2));
    AME_INSN(MSETTILENI(2));
    AME_INSN(MSETTILEKI(2));

    load_tile8(0, tile_a);
    load_tile8(1, tile_bt);

    AME_INSN(MMACC_W_B(0, 1, 0));

    store_acc32(4, acc_out);

    TEST_ASSERT(acc_out[0] == 17, "whole-load mmacc cell[0,0] honors physical row stride");
    TEST_ASSERT(acc_out[1] == 23, "whole-load mmacc cell[0,1] honors physical row stride");
    TEST_ASSERT(acc_out[AME_ROW_COUNT] == 39,
                "whole-load mmacc cell[1,0] honors physical row stride");
    TEST_ASSERT(acc_out[AME_ROW_COUNT + 1] == 53,
                "whole-load mmacc cell[1,1] honors physical row stride");
}

static void test_mmacc_clears_inactive_accumulator_region(void)
{
    const unsigned long rownum = AME_ROW_COUNT;
    int8_t tile_a[AME_TILE_BYTES] = { 2 };
    int8_t tile_bt[AME_TILE_BYTES] = { 3 };
    int32_t acc_init[AME_ACC_WORDS];
    int32_t acc_out[AME_ACC_WORDS];
    unsigned long i;

    memset(acc_out, 0, sizeof(acc_out));
    for (i = 0; i < AME_ACC_WORDS; i++) {
        acc_init[i] = 0x55aa0000u + (int32_t)i;
    }

    TEST_ASSERT(rownum >= 2, "configuration has inactive accumulator region");

    AME_INSN(MZERO8R(0));
    AME_INSN(MSETTILEMI(1));
    AME_INSN(MSETTILENI(1));
    AME_INSN(MSETTILEKI(1));

    load_tile8(0, tile_a);
    load_tile8(1, tile_bt);
    load_acc32(4, acc_init);

    AME_INSN(MMACC_W_B(0, 1, 0));

    store_acc32(4, acc_out);

    TEST_ASSERT(acc_out[0] == acc_init[0] + 6,
                "mmacc keeps active accumulator cell and accumulates into it");

    for (i = 1; i < rownum; i++) {
        TEST_ASSERT(acc_out[i] == 0,
                    "mmacc clears inactive accumulator columns");
    }

    for (i = rownum; i < rownum * rownum; i++) {
        TEST_ASSERT(acc_out[i] == 0,
                    "mmacc clears inactive accumulator rows");
    }
}

static void test_mfmacc_s_h_clears_inactive_accumulator_region(void)
{
    const unsigned long rownum = AME_ROW_COUNT;
    uint16_t tile_a[AME_TILE_BYTES / sizeof(uint16_t)] = { 0x3c00 };
    uint16_t tile_bt[AME_TILE_BYTES / sizeof(uint16_t)] = { 0x4000 };
    uint32_t acc_init[AME_ACC_WORDS];
    uint32_t acc_out[AME_ACC_WORDS];
    unsigned long i;

    memset(acc_out, 0, sizeof(acc_out));
    for (i = 0; i < AME_ACC_WORDS; i++) {
        acc_init[i] = 0x3f800000u + (uint32_t)i;
    }

    TEST_ASSERT(rownum >= 2, "configuration has inactive fp32 accumulator region");

    AME_INSN(MZERO8R(0));
    AME_INSN(MSETTILEMI(1));
    AME_INSN(MSETTILENI(1));
    AME_INSN(MSETTILEKI(1));

    load_tile16(0, tile_a);
    load_tile16(1, tile_bt);
    load_acc32(4, acc_init);

    AME_INSN(MFMACC_S_H(0, 1, 0));

    store_acc32(4, acc_out);

    TEST_ASSERT(acc_out[0] == 0x40400000u,
                "mfmacc.s.h updates active fp32 accumulator cell");

    for (i = 1; i < rownum; i++) {
        TEST_ASSERT(acc_out[i] == 0,
                    "mfmacc.s.h clears inactive fp32 accumulator columns");
    }

    for (i = rownum; i < rownum * rownum; i++) {
        TEST_ASSERT(acc_out[i] == 0,
                    "mfmacc.s.h clears inactive fp32 accumulator rows");
    }
}

static void test_mfmacc_h_clears_inactive_accumulator_region(void)
{
    const unsigned long rownum = AME_ROW_COUNT;
    uint16_t tile_a[AME_TILE_BYTES / sizeof(uint16_t)] = { 0x3c00 };
    uint16_t tile_bt[AME_TILE_BYTES / sizeof(uint16_t)] = { 0x4000 };
    uint16_t acc_init[AME_ACC_WORDS * (sizeof(uint32_t) / sizeof(uint16_t))];
    uint16_t acc_out[AME_ACC_WORDS * (sizeof(uint32_t) / sizeof(uint16_t))];
    unsigned long i;

    memset(acc_out, 0, sizeof(acc_out));
    for (i = 0; i < sizeof(acc_init) / sizeof(acc_init[0]); i++) {
        acc_init[i] = (uint16_t)(0x3c00u + i);
    }

    TEST_ASSERT(rownum >= 2, "configuration has inactive fp16 accumulator region");

    AME_INSN(MZERO8R(0));
    AME_INSN(MSETTILEMI(1));
    AME_INSN(MSETTILENI(1));
    AME_INSN(MSETTILEKI(1));

    load_tile16(0, tile_a);
    load_tile16(1, tile_bt);
    load_acc16(4, acc_init);

    AME_INSN(MFMACC_H(0, 1, 0));

    store_acc16(4, acc_out);

    TEST_ASSERT(acc_out[0] == 0x4200u,
                "mfmacc.h updates active fp16 accumulator cell");

    for (i = 1; i < rownum; i++) {
        TEST_ASSERT(acc_out[i] == 0,
                    "mfmacc.h clears inactive fp16 accumulator columns");
    }

    for (i = rownum; i < rownum * rownum; i++) {
        TEST_ASSERT(acc_out[i] == 0,
                    "mfmacc.h clears inactive fp16 accumulator rows");
    }
}

static void test_mfmacc_s_bf16_subnormal_result(void)
{
    uint16_t tile_a[AME_TILE_BYTES / sizeof(uint16_t)] = {
        0x036d, 0x18b0,
    };
    uint16_t tile_bt[AME_TILE_BYTES / sizeof(uint16_t)] = {
        0xb731, 0x267b,
    };
    uint32_t acc_init[AME_ACC_WORDS] = { 0 };
    uint32_t acc_out[AME_ACC_WORDS] = { 0 };

    AME_INSN(MSETTILEMI(1));
    AME_INSN(MSETTILENI(1));
    AME_INSN(MSETTILEKI(2));

    load_tile16(0, tile_a);
    load_tile16(1, tile_bt);
    load_acc32(4, acc_init);

    clear_fflags();
    AME_INSN(MFMACC_S_BF16(0, 1, 0));
    store_acc32(4, acc_out);

    TEST_ASSERT(acc_out[0] == 0x002b0f84u,
                "mfmacc.s.bf16 preserves a subnormal fp32 result");
    TEST_ASSERT((read_fflags() & 0x1fu) == 0x03u,
                "mfmacc.s.bf16 raises underflow and inexact");
}

int main(void)
{
    printf("=== AME MMACC Test ===\n");

    test_mmacc_operand_order();
    test_strided_load_uses_physical_row_stride();
    test_transpose_tile_load_clears_inactive_region();
    test_acc_load_zero_fill();
    test_mmacc_whole_load_uses_physical_row_stride();
    test_mmacc_clears_inactive_accumulator_region();
    test_mfmacc_s_h_clears_inactive_accumulator_region();
    test_mfmacc_h_clears_inactive_accumulator_region();
    test_mfmacc_s_bf16_subnormal_result();

    TEST_SUMMARY();
}
