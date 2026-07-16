/*
 * AME instruction encoding helpers for TCG regression tests.
 *
 * Copyright (c) 2025
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef TCG_RISCV64_AME_INSN_H
#define TCG_RISCV64_AME_INSN_H

#include <stdint.h>
#include <stdio.h>

#define AME_OP 0x2b

#define AME_INSN(enc) __asm__ volatile (".4byte %0" :: "i"(enc) : "memory")

#define AME_SETI_ENC(func4, imm10) \
    (((uint32_t)(func4) << 28) | (((uint32_t)(imm10) & 0x3ffu) << 15) | AME_OP)

#define AME_MEM_NEW_ENC(store, size, md, rs1) \
    (((uint32_t)0x3 << 28) | ((uint32_t)0x1 << 26) | \
     (((uint32_t)(store) & 0x1u) << 25) | (((uint32_t)(rs1) & 0x1fu) << 15) | \
     (((uint32_t)(size) & 0x3u) << 10) | (((uint32_t)(md) & 0x7u) << 7) | AME_OP)

#define AME_ZERO_ENC(count_enc, md) \
    (((uint32_t)0x0 << 28) | ((uint32_t)0x3 << 26) | \
     (((uint32_t)(count_enc) & 0x7u) << 23) | (((uint32_t)(md) & 0x7u) << 7) | \
     AME_OP)

#define AME_MISC_MM_ENC(func4, misc3, md, ms2, ms1) \
    (((uint32_t)(func4) << 28) | ((uint32_t)0x3 << 26) | \
     (((uint32_t)(misc3) & 0x7u) << 23) | (((uint32_t)(ms2) & 0x7u) << 20) | \
     (((uint32_t)(ms1) & 0x7u) << 15) | (((uint32_t)(md) & 0x7u) << 7) | \
     AME_OP)

#define AME_MISC_TO_GPR_ENC(size, rd, ms2, rs1) \
    (((uint32_t)0x2 << 28) | ((uint32_t)0x3 << 26) | \
     (((uint32_t)(size) & 0x7u) << 23) | (((uint32_t)(ms2) & 0x7u) << 20) | \
     (((uint32_t)(rs1) & 0x1fu) << 15) | (((uint32_t)(rd) & 0x1fu) << 7) | \
     AME_OP)

#define AME_MISC_FROM_GPR_ENC(size, md, rs1, rs2) \
    (((uint32_t)0x3 << 28) | ((uint32_t)0x3 << 26) | ((uint32_t)1 << 25) | \
     (((uint32_t)(rs2) & 0x1fu) << 20) | (((uint32_t)(rs1) & 0x1fu) << 15) | \
     (((uint32_t)(size) & 0x3u) << 10) | (((uint32_t)(md) & 0x7u) << 7) | \
     AME_OP)

#define AME_MISC_SLIDE_ENC(func4, size, md, ms1, imm3) \
    (((uint32_t)(func4) << 28) | ((uint32_t)0x3 << 26) | \
     (((uint32_t)(imm3) & 0x7u) << 23) | (((uint32_t)(size) & 0x3u) << 18) | \
     (((uint32_t)(ms1) & 0x7u) << 15) | (((uint32_t)(size) & 0x3u) << 10) | \
     (((uint32_t)(md) & 0x7u) << 7) | AME_OP)

#define AME_MATMUL_ENC(func4, uop3, ad, ms2, ms1) \
    (((uint32_t)(func4) << 28) | ((uint32_t)0x1 << 26) | \
     (((uint32_t)(uop3) & 0x7u) << 23) | (((uint32_t)(ms2) & 0x7u) << 20) | \
     (((uint32_t)(ms1) & 0x7u) << 15) | ((uint32_t)0x2 << 10) | \
     ((uint32_t)0x1 << 9) | (((uint32_t)(ad) & 0x3u) << 7) | AME_OP)

#define AME_MATMUL_SRC_ENC(func4, uop3, s_size, ad, ms2, ms1) \
    (((uint32_t)(func4) << 28) | ((uint32_t)0x1 << 26) | \
     (((uint32_t)(uop3) & 0x7u) << 23) | (((uint32_t)(ms2) & 0x7u) << 20) | \
     (((uint32_t)(s_size) & 0x3u) << 18) | (((uint32_t)(ms1) & 0x7u) << 15) | \
     ((uint32_t)0x2 << 10) | ((uint32_t)0x1 << 9) | \
     (((uint32_t)(ad) & 0x3u) << 7) | AME_OP)

#define AME_MATMUL_ACC16_ENC(func4, uop3, s_size, ad, ms2, ms1) \
    (((uint32_t)(func4) << 28) | ((uint32_t)0x1 << 26) | \
     (((uint32_t)(uop3) & 0x7u) << 23) | (((uint32_t)(ms2) & 0x7u) << 20) | \
     (((uint32_t)(s_size) & 0x3u) << 18) | (((uint32_t)(ms1) & 0x7u) << 15) | \
     ((uint32_t)0x1 << 10) | ((uint32_t)0x1 << 9) | \
     (((uint32_t)(ad) & 0x3u) << 7) | AME_OP)

#define AME_SFU_ENC(funct3, vd, vs2) \
    (((uint32_t)0x13 << 26) | (((uint32_t)(vs2) & 0x1fu) << 20) | \
     ((uint32_t)0x7 << 15) | (((uint32_t)(funct3) & 0x7u) << 12) | \
     (((uint32_t)(vd) & 0x1fu) << 7) | AME_OP)

#define CSR_XMSAT 0x807
#define CSR_XMSATEN 0x80a

#define MSETTILEKI(imm) AME_SETI_ENC(0x1, imm)
#define MSETTILEMI(imm) AME_SETI_ENC(0x2, imm)
#define MSETTILENI(imm) AME_SETI_ENC(0x3, imm)

#define MZERO8R(md) AME_ZERO_ENC(0x7, md)

#define MLME8(md, rs1) AME_MEM_NEW_ENC(0, 0x0, md, rs1)
#define MSME8(md, rs1) AME_MEM_NEW_ENC(1, 0x0, md, rs1)
#define MLME16(md, rs1) AME_MEM_NEW_ENC(0, 0x1, md, rs1)
#define MSME16(md, rs1) AME_MEM_NEW_ENC(1, 0x1, md, rs1)
#define MLME32(md, rs1) AME_MEM_NEW_ENC(0, 0x2, md, rs1)
#define MSME32(md, rs1) AME_MEM_NEW_ENC(1, 0x2, md, rs1)

#define MMACCU_W_B(ad, ms2, ms1) AME_MATMUL_ENC(0x1, 0x0, ad, ms2, ms1)
#define MMACCUS_W_B(ad, ms2, ms1) AME_MATMUL_ENC(0x1, 0x1, ad, ms2, ms1)
#define MMACCSU_W_B(ad, ms2, ms1) AME_MATMUL_ENC(0x1, 0x2, ad, ms2, ms1)
#define MMACC_W_B(ad, ms2, ms1) AME_MATMUL_ENC(0x1, 0x3, ad, ms2, ms1)

#define MFMACC_S_H(ad, ms2, ms1) AME_MATMUL_SRC_ENC(0x0, 0x0, 0x1, ad, ms2, ms1)
#define MFMACC_S_BF16(ad, ms2, ms1) \
    AME_MATMUL_SRC_ENC(0x0, 0x1, 0x1, ad, ms2, ms1)
#define MFMACC_H(ad, ms2, ms1) AME_MATMUL_ACC16_ENC(0x0, 0x0, 0x1, ad, ms2, ms1)

#define MMOV_MM(md, ms1) AME_MISC_MM_ENC(0x1, 0x0, md, 0, ms1)
#define MMOVB_X_M(rd, ms2, rs1) AME_MISC_TO_GPR_ENC(0x0, rd, ms2, rs1)
#define MMOVH_X_M(rd, ms2, rs1) AME_MISC_TO_GPR_ENC(0x1, rd, ms2, rs1)
#define MMOVW_X_M(rd, ms2, rs1) AME_MISC_TO_GPR_ENC(0x2, rd, ms2, rs1)
#define MMOVD_X_M(rd, ms2, rs1) AME_MISC_TO_GPR_ENC(0x3, rd, ms2, rs1)
#define MMOVB_M_X(md, rs1, rs2) AME_MISC_FROM_GPR_ENC(0x0, md, rs1, rs2)
#define MMOVH_M_X(md, rs1, rs2) AME_MISC_FROM_GPR_ENC(0x1, md, rs1, rs2)
#define MMOVW_M_X(md, rs1, rs2) AME_MISC_FROM_GPR_ENC(0x2, md, rs1, rs2)
#define MMOVD_M_X(md, rs1, rs2) AME_MISC_FROM_GPR_ENC(0x3, md, rs1, rs2)

#define MPACK(md, ms2, ms1) AME_MISC_MM_ENC(0x4, 0x0, md, ms2, ms1)
#define MPACKHL(md, ms2, ms1) AME_MISC_MM_ENC(0x4, 0x2, md, ms2, ms1)
#define MPACKHH(md, ms2, ms1) AME_MISC_MM_ENC(0x4, 0x3, md, ms2, ms1)

#define MRSLIDEDOWN(md, ms1, imm3) AME_MISC_SLIDE_ENC(0x5, 0x0, md, ms1, imm3)
#define MRSLIDEUP(md, ms1, imm3) AME_MISC_SLIDE_ENC(0x6, 0x0, md, ms1, imm3)
#define MCSLIDEDOWN_B(md, ms1, imm3) AME_MISC_SLIDE_ENC(0x7, 0x0, md, ms1, imm3)
#define MCSLIDEDOWN_H(md, ms1, imm3) AME_MISC_SLIDE_ENC(0x7, 0x1, md, ms1, imm3)
#define MCSLIDEDOWN_W(md, ms1, imm3) AME_MISC_SLIDE_ENC(0x7, 0x2, md, ms1, imm3)
#define MCSLIDEUP_B(md, ms1, imm3) AME_MISC_SLIDE_ENC(0x8, 0x0, md, ms1, imm3)
#define MCSLIDEUP_H(md, ms1, imm3) AME_MISC_SLIDE_ENC(0x8, 0x1, md, ms1, imm3)
#define MCSLIDEUP_W(md, ms1, imm3) AME_MISC_SLIDE_ENC(0x8, 0x2, md, ms1, imm3)

#define VFEX2_V(vd, vs2) AME_SFU_ENC(0x1, vd, vs2)
#define VFTANH_V(vd, vs2) AME_SFU_ENC(0x2, vd, vs2)
#define VFLG2_V(vd, vs2) AME_SFU_ENC(0x3, vd, vs2)
#define VFRCP_V(vd, vs2) AME_SFU_ENC(0x4, vd, vs2)

static int test_pass_count;
static int test_fail_count;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
        test_fail_count++; \
    } else { \
        test_pass_count++; \
    } \
} while (0)

#define TEST_SUMMARY() do { \
    printf("---\n%d passed, %d failed\n", test_pass_count, test_fail_count); \
    return test_fail_count ? 1 : 0; \
} while (0)

#endif
