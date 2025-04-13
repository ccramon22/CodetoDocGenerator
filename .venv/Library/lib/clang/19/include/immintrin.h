/*===---- immintrin.h - Intel intrinsics -----------------------------------=== */
/*
 * Modifications, Copyright (C) 2021 Intel Corporation
 *
 * This software and the related documents are Intel copyrighted materials, and
 * your use of them is governed by the express license under which they were
 * provided to you ("License"). Unless the License provides otherwise, you may not
 * use, modify, copy, publish, distribute, disclose or transmit this software or
 * the related documents without Intel's prior written permission.
 *
 * This software and the related documents are provided as is, with no express
 * or implied warranties, other than those that are expressly stated in the
 * License.
 */
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __IMMINTRIN_H
#define __IMMINTRIN_H

/* Turn fp precise on for intrinsics, push state to restore at end. */
#pragma float_control(push)
#pragma float_control(precise, on)

/* Enable fast contract if it was enabled on the command line */
#pragma STDC FP_CONTRACT DEFAULT

/// TODO: Move it into clang/lib/Headers/amxintrin.h when all amx disclosed.

/// AMX tile register size can be configured, the maximum size is 16x64=1024
/// bytes. Since there is no 2D type in llvm IR, we use vector type to
/// represent 2D tile and the fixed size is maximum amx tile register size.
typedef int _tile1024i __attribute__((__vector_size__(1024), __aligned__(64)));
typedef int _tile1024i_1024a __attribute__((__vector_size__(1024), __aligned__(1024)));

/// This struct pack the shape and tile data together for user. We suggest
/// initializing the struct as early as possible, because compiler depends
/// on the shape information to do configure. The constant value is preferred
/// for optimization by compiler.
typedef struct __tile1024i_str {
  const unsigned short row;
  const unsigned short col;
  _tile1024i tile;
} __tile1024i;

#if !defined(__i386__) && !defined(__x86_64__)
#error "This header is only meant to be used on x86 and x64 architecture"
#endif

#include <x86gprintrin.h>

#include <ia32intrin.h>

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__MMX__) ||       \
    defined(__M_INTRINSIC_PROMOTE__)
#include <mmintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__SSE__) ||       \
    defined(__M_INTRINSIC_PROMOTE__)
#include <xmmintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__SSE2__) ||      \
    defined(__M_INTRINSIC_PROMOTE__)
#include <emmintrin.h>
#endif


#if !(defined(__SCE__)) || __has_feature(modules) || defined(__SSE3__) ||      \
    defined(__M_INTRINSIC_PROMOTE__)
#include <pmmintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__SSSE3__) ||     \
    defined(__M_INTRINSIC_PROMOTE__) || defined(__DSPV1__)
#include <tmmintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    (defined(__SSE4_2__) || defined(__SSE4_1__)) ||                            \
    defined(__M_INTRINSIC_PROMOTE__)
#include <smmintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    (defined(__AES__) || defined(__PCLMUL__)) ||                               \
    defined(__M_INTRINSIC_PROMOTE__)
#include <wmmintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    defined(__CLFLUSHOPT__) || defined(__M_INTRINSIC_PROMOTE__)
#include <clflushoptintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__CLWB__) ||      \
    defined(__M_INTRINSIC_PROMOTE__)
#include <clwbintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__AVX__) ||       \
    defined(__M_INTRINSIC_PROMOTE__)
#include <avxintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__AVX2__) ||      \
    defined(__M_INTRINSIC_PROMOTE__)
#include <avx2intrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__F16C__) ||      \
    defined(__M_INTRINSIC_PROMOTE__)
#include <f16cintrin.h>
#endif

/* No feature check desired due to internal checks */
#include <bmiintrin.h>

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__BMI2__) ||      \
    defined(__M_INTRINSIC_PROMOTE__)
#include <bmi2intrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__LZCNT__) ||     \
    defined(__M_INTRINSIC_PROMOTE__)
#include <lzcntintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__POPCNT__) ||    \
    defined(__M_INTRINSIC_PROMOTE__)
#include <popcntintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__FMA__) ||       \
    defined(__M_INTRINSIC_PROMOTE__)
#include <fmaintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__AVX512F__) ||   \
    defined(__M_INTRINSIC_PROMOTE__)
#include <avx512fintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__AVX512VL__) ||  \
    defined(__M_INTRINSIC_PROMOTE__)
#include <avx512vlintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__AVX512BW__) ||  \
    defined(__M_INTRINSIC_PROMOTE__)
#include <avx512bwintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    defined(__AVX512BITALG__) || defined(__M_INTRINSIC_PROMOTE__)
#include <avx512bitalgintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__AVX512CD__) ||  \
    defined(__M_INTRINSIC_PROMOTE__)
#include <avx512cdintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    defined(__AVX512VPOPCNTDQ__) || defined(__M_INTRINSIC_PROMOTE__)
#include <avx512vpopcntdqintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    (defined(__AVX512VL__) && defined(__AVX512VPOPCNTDQ__)) ||                 \
    defined(__M_INTRINSIC_PROMOTE__)
#include <avx512vpopcntdqvlintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    defined(__AVX512VNNI__) || defined(__M_INTRINSIC_PROMOTE__)
#include <avx512vnniintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    (defined(__AVX512VL__) && defined(__AVX512VNNI__)) ||                      \
    defined(__M_INTRINSIC_PROMOTE__)
#include <avx512vlvnniintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__AVXVNNI__) ||   \
    defined(__M_INTRINSIC_PROMOTE__)
#include <avxvnniintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__AVX512DQ__) ||  \
    defined(__M_INTRINSIC_PROMOTE__)
#include <avx512dqintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    (defined(__AVX512VL__) && defined(__AVX512BITALG__)) ||                    \
    defined(__M_INTRINSIC_PROMOTE__)
#include <avx512vlbitalgintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    (defined(__AVX512VL__) && defined(__AVX512BW__)) ||                        \
    defined(__M_INTRINSIC_PROMOTE__)
#include <avx512vlbwintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    (defined(__AVX512VL__) && defined(__AVX512CD__)) ||                        \
    defined(__M_INTRINSIC_PROMOTE__)
#include <avx512vlcdintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    (defined(__AVX512VL__) && defined(__AVX512DQ__)) ||                        \
    defined(__M_INTRINSIC_PROMOTE__)
#include <avx512vldqintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    defined(__AVX512IFMA__) || defined(__M_INTRINSIC_PROMOTE__)
#include <avx512ifmaintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    (defined(__AVX512IFMA__) && defined(__AVX512VL__)) ||                      \
    defined(__M_INTRINSIC_PROMOTE__)
#include <avx512ifmavlintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__AVXIFMA__) ||   \
    defined(__M_INTRINSIC_PROMOTE__)
#include <avxifmaintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    defined(__AVX512VBMI__) || defined(__M_INTRINSIC_PROMOTE__)
#include <avx512vbmiintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    (defined(__AVX512VBMI__) && defined(__AVX512VL__)) ||                      \
    defined(__M_INTRINSIC_PROMOTE__)
#include <avx512vbmivlintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    defined(__AVX512VBMI2__) || defined(__M_INTRINSIC_PROMOTE__)
#include <avx512vbmi2intrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    (defined(__AVX512VBMI2__) && defined(__AVX512VL__)) ||                     \
    defined(__M_INTRINSIC_PROMOTE__)
#include <avx512vlvbmi2intrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    defined(__AVX512FP16__) || defined(__M_INTRINSIC_PROMOTE__)
#include <avx512fp16intrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    (defined(__AVX512VL__) && defined(__AVX512FP16__)) ||                      \
    defined(__M_INTRINSIC_PROMOTE__)
#include <avx512vlfp16intrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    defined(__AVX512BF16__) || defined(__M_INTRINSIC_PROMOTE__)
#include <avx512bf16intrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    (defined(__AVX512VL__) && defined(__AVX512BF16__)) ||                      \
    defined(__M_INTRINSIC_PROMOTE__)
#include <avx512vlbf16intrin.h>
#endif


#if !(defined(__SCE__)) || __has_feature(modules) || defined(__PKU__) ||       \
    defined(__M_INTRINSIC_PROMOTE__)
#include <pkuintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    defined(__VPCLMULQDQ__) || defined(__M_INTRINSIC_PROMOTE__)
#include <vpclmulqdqintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__VAES__) ||      \
    defined(__M_INTRINSIC_PROMOTE__)
#include <vaesintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__GFNI__) ||      \
    defined(__M_INTRINSIC_PROMOTE__)
#include <gfniintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    defined(__AVXVNNIINT8__) || defined(__M_INTRINSIC_PROMOTE__)
#include <avxvnniint8intrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    defined(__AVXNECONVERT__) || defined(__M_INTRINSIC_PROMOTE__)
#include <avxneconvertintrin.h>
#endif
















#if !(defined(__SCE__)) || __has_feature(modules) || defined(__SHA512__) ||    \
    defined(__M_INTRINSIC_PROMOTE__)
#include <sha512intrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__SM3__) ||       \
    defined(__M_INTRINSIC_PROMOTE__)
#include <sm3intrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__SM4__) ||       \
    defined(__M_INTRINSIC_PROMOTE__)
#include <sm4intrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    defined(__AVXVNNIINT16__) || defined(__M_INTRINSIC_PROMOTE__)
#include <avxvnniint16intrin.h>
#endif

#if !defined(__SCE__) || __has_feature(modules) || defined(__RDPID__)
/// Reads the value of the IA32_TSC_AUX MSR (0xc0000103).
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> RDPID </c> instruction.
///
/// \returns The 32-bit contents of the MSR.
static __inline__ unsigned int __attribute__((__always_inline__, __nodebug__, __target__("rdpid")))
_rdpid_u32(void) {
  return __builtin_ia32_rdpid();
}
#endif // __RDPID__

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__RDRND__) ||     \
    defined(__M_INTRINSIC_PROMOTE__)
/// Returns a 16-bit hardware-generated random value.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> RDRAND </c> instruction.
///
/// \param __p
///    A pointer to a 16-bit memory location to place the random value.
/// \returns 1 if the value was successfully generated, 0 otherwise.
static __inline__ int __attribute__((__always_inline__, __nodebug__, __target__("rdrnd")))
_rdrand16_step(unsigned short *__p)
{
  return (int)__builtin_ia32_rdrand16_step(__p);
}

/// Returns a 32-bit hardware-generated random value.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> RDRAND </c> instruction.
///
/// \param __p
///    A pointer to a 32-bit memory location to place the random value.
/// \returns 1 if the value was successfully generated, 0 otherwise.
static __inline__ int __attribute__((__always_inline__, __nodebug__, __target__("rdrnd")))
_rdrand32_step(unsigned int *__p)
{
  return (int)__builtin_ia32_rdrand32_step(__p);
}

/// Returns a 64-bit hardware-generated random value.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> RDRAND </c> instruction.
///
/// \param __p
///    A pointer to a 64-bit memory location to place the random value.
/// \returns 1 if the value was successfully generated, 0 otherwise.
static __inline__ int __attribute__((__always_inline__, __nodebug__, __target__("rdrnd")))
_rdrand64_step(unsigned long long *__p)
{
#ifdef __x86_64__
  return (int)__builtin_ia32_rdrand64_step(__p);
#else
  // We need to emulate the functionality of 64-bit rdrand with 2 32-bit
  // rdrand instructions.
  unsigned int __lo, __hi;
  unsigned int __res_lo = __builtin_ia32_rdrand32_step(&__lo);
  unsigned int __res_hi = __builtin_ia32_rdrand32_step(&__hi);
  if (__res_lo && __res_hi) {
    *__p = ((unsigned long long)__hi << 32) | (unsigned long long)__lo;
    return 1;
  } else {
    *__p = 0;
    return 0;
  }
#endif
}
#endif /* __RDRND__ */

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__FSGSBASE__) ||  \
    defined(__M_INTRINSIC_PROMOTE__)
#ifdef __x86_64__
/// Reads the FS base register.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> RDFSBASE </c> instruction.
///
/// \returns The lower 32 bits of the FS base register.
static __inline__ unsigned int __attribute__((__always_inline__, __nodebug__, __target__("fsgsbase")))
_readfsbase_u32(void)
{
  return __builtin_ia32_rdfsbase32();
}

/// Reads the FS base register.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> RDFSBASE </c> instruction.
///
/// \returns The contents of the FS base register.
static __inline__ unsigned long long __attribute__((__always_inline__, __nodebug__, __target__("fsgsbase")))
_readfsbase_u64(void)
{
  return __builtin_ia32_rdfsbase64();
}

/// Reads the GS base register.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> RDGSBASE </c> instruction.
///
/// \returns The lower 32 bits of the GS base register.
static __inline__ unsigned int __attribute__((__always_inline__, __nodebug__, __target__("fsgsbase")))
_readgsbase_u32(void)
{
  return __builtin_ia32_rdgsbase32();
}

/// Reads the GS base register.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> RDGSBASE </c> instruction.
///
/// \returns The contents of the GS base register.
static __inline__ unsigned long long __attribute__((__always_inline__, __nodebug__, __target__("fsgsbase")))
_readgsbase_u64(void)
{
  return __builtin_ia32_rdgsbase64();
}

/// Modifies the FS base register.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> WRFSBASE </c> instruction.
///
/// \param __V
///    Value to use for the lower 32 bits of the FS base register.
static __inline__ void __attribute__((__always_inline__, __nodebug__, __target__("fsgsbase")))
_writefsbase_u32(unsigned int __V)
{
  __builtin_ia32_wrfsbase32(__V);
}

/// Modifies the FS base register.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> WRFSBASE </c> instruction.
///
/// \param __V
///    Value to use for the FS base register.
static __inline__ void __attribute__((__always_inline__, __nodebug__, __target__("fsgsbase")))
_writefsbase_u64(unsigned long long __V)
{
  __builtin_ia32_wrfsbase64(__V);
}

/// Modifies the GS base register.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> WRGSBASE </c> instruction.
///
/// \param __V
///    Value to use for the lower 32 bits of the GS base register.
static __inline__ void __attribute__((__always_inline__, __nodebug__, __target__("fsgsbase")))
_writegsbase_u32(unsigned int __V)
{
  __builtin_ia32_wrgsbase32(__V);
}

/// Modifies the GS base register.
///
/// \headerfile <immintrin.h>
///
/// This intrinsic corresponds to the <c> WRFSBASE </c> instruction.
///
/// \param __V
///    Value to use for GS base register.
static __inline__ void __attribute__((__always_inline__, __nodebug__, __target__("fsgsbase")))
_writegsbase_u64(unsigned long long __V)
{
  __builtin_ia32_wrgsbase64(__V);
}

#endif
#endif /* __FSGSBASE__ */

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__MOVBE__) ||     \
    defined(__M_INTRINSIC_PROMOTE__)

/* The structs used below are to force the load/store to be unaligned. This
 * is accomplished with the __packed__ attribute. The __may_alias__ prevents
 * tbaa metadata from being generated based on the struct and the type of the
 * field inside of it.
 */

/// Load a 16-bit value from memory and swap its bytes.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the MOVBE instruction.
///
/// \param __P
///    A pointer to the 16-bit value to load.
/// \returns The byte-swapped value.
static __inline__ short __attribute__((__always_inline__, __nodebug__, __target__("movbe")))
_loadbe_i16(void const * __P) {
  struct __loadu_i16 {
    unsigned short __v;
  } __attribute__((__packed__, __may_alias__));
  return (short)__builtin_bswap16(((const struct __loadu_i16*)__P)->__v);
}

/// Swap the bytes of a 16-bit value and store it to memory.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the MOVBE instruction.
///
/// \param __P
///    A pointer to the memory for storing the swapped value.
/// \param __D
///    The 16-bit value to be byte-swapped.
static __inline__ void __attribute__((__always_inline__, __nodebug__, __target__("movbe")))
_storebe_i16(void * __P, short __D) {
  struct __storeu_i16 {
    unsigned short __v;
  } __attribute__((__packed__, __may_alias__));
  ((struct __storeu_i16*)__P)->__v = __builtin_bswap16((unsigned short)__D);
}

/// Load a 32-bit value from memory and swap its bytes.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the MOVBE instruction.
///
/// \param __P
///    A pointer to the 32-bit value to load.
/// \returns The byte-swapped value.
static __inline__ int __attribute__((__always_inline__, __nodebug__, __target__("movbe")))
_loadbe_i32(void const * __P) {
  struct __loadu_i32 {
    unsigned int __v;
  } __attribute__((__packed__, __may_alias__));
  return (int)__builtin_bswap32(((const struct __loadu_i32*)__P)->__v);
}

/// Swap the bytes of a 32-bit value and store it to memory.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the MOVBE instruction.
///
/// \param __P
///    A pointer to the memory for storing the swapped value.
/// \param __D
///    The 32-bit value to be byte-swapped.
static __inline__ void __attribute__((__always_inline__, __nodebug__, __target__("movbe")))
_storebe_i32(void * __P, int __D) {
  struct __storeu_i32 {
    unsigned int __v;
  } __attribute__((__packed__, __may_alias__));
  ((struct __storeu_i32*)__P)->__v = __builtin_bswap32((unsigned int)__D);
}

#ifdef __x86_64__
/// Load a 64-bit value from memory and swap its bytes.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the MOVBE instruction.
///
/// \param __P
///    A pointer to the 64-bit value to load.
/// \returns The byte-swapped value.
static __inline__ long long __attribute__((__always_inline__, __nodebug__, __target__("movbe")))
_loadbe_i64(void const * __P) {
  struct __loadu_i64 {
    unsigned long long __v;
  } __attribute__((__packed__, __may_alias__));
  return (long long)__builtin_bswap64(((const struct __loadu_i64*)__P)->__v);
}

/// Swap the bytes of a 64-bit value and store it to memory.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the MOVBE instruction.
///
/// \param __P
///    A pointer to the memory for storing the swapped value.
/// \param __D
///    The 64-bit value to be byte-swapped.
static __inline__ void __attribute__((__always_inline__, __nodebug__, __target__("movbe")))
_storebe_i64(void * __P, long long __D) {
  struct __storeu_i64 {
    unsigned long long __v;
  } __attribute__((__packed__, __may_alias__));
  ((struct __storeu_i64*)__P)->__v = __builtin_bswap64((unsigned long long)__D);
}
#endif
#endif /* __MOVBE */

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__RTM__) ||       \
    defined(__M_INTRINSIC_PROMOTE__)
#include <rtmintrin.h>
#include <xtestintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__SHA__) ||       \
    defined(__M_INTRINSIC_PROMOTE__)
#include <shaintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__FXSR__) ||      \
    defined(__M_INTRINSIC_PROMOTE__)
#include <fxsrintrin.h>
#endif

/* No feature check desired due to internal MSC_VER checks */
#include <xsaveintrin.h>

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__XSAVEOPT__) ||  \
    defined(__M_INTRINSIC_PROMOTE__)
#include <xsaveoptintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__XSAVEC__) ||    \
    defined(__M_INTRINSIC_PROMOTE__)
#include <xsavecintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__XSAVES__) ||    \
    defined(__M_INTRINSIC_PROMOTE__)
#include <xsavesintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__SHSTK__) ||     \
    defined(__M_INTRINSIC_PROMOTE__)
#include <cetintrin.h>
#endif

/* Intrinsics inside adcintrin.h are available at all times. */
#include <adcintrin.h>

#if !defined(__SCE__) || __has_feature(modules) || defined(__ADX__)
#include <adxintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__RDSEED__) ||    \
    defined(__M_INTRINSIC_PROMOTE__)
#include <rdseedintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__WBNOINVD__) ||  \
    defined(__M_INTRINSIC_PROMOTE__)
#include <wbnoinvdintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__CLDEMOTE__) ||  \
    defined(__M_INTRINSIC_PROMOTE__)
#include <cldemoteintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__WAITPKG__) ||   \
    defined(__M_INTRINSIC_PROMOTE__)
#include <waitpkgintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__MOVDIRI__) ||   \
    defined(__MOVDIR64B__) || defined(__M_INTRINSIC_PROMOTE__)
#include <movdirintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__PCONFIG__) ||   \
    defined(__M_INTRINSIC_PROMOTE__)
#include <pconfigintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__SGX__) ||       \
    defined(__M_INTRINSIC_PROMOTE__)
#include <sgxintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__PTWRITE__) ||   \
    defined(__M_INTRINSIC_PROMOTE__)
#include <ptwriteintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__INVPCID__) ||   \
    defined(__M_INTRINSIC_PROMOTE__)
#include <invpcidintrin.h>
#endif
#if !defined(__SCE__) || __has_feature(modules) || defined(__AMX_FP16__)
#include <amxfp16intrin.h>
#endif






























#if !(defined(__SCE__)) || __has_feature(modules) || defined(__KL__) ||        \
    defined(__WIDEKL__)
#include <keylockerintrin.h>
#endif

#if !defined(__SCE__) || __has_feature(modules) || defined(__AMX_TILE__) ||    \
    defined(__AMX_INT8__) || defined(__AMX_BF16__)
#include <amxintrin.h>
#endif

#if !defined(__SCE__) || __has_feature(modules) || defined(__AMX_COMPLEX__)
#include <amxcomplexintrin.h>
#endif

#if !defined(__SCE__) || __has_feature(modules) ||                             \
    defined(__AVX512VP2INTERSECT__)
#include <avx512vp2intersectintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) ||                           \
    (defined(__AVX512VL__) && defined(__AVX512VP2INTERSECT__)) ||              \
    defined(__M_INTRINSIC_PROMOTE__)
#include <avx512vlvp2intersectintrin.h>
#endif


#if !(defined(_MSC_VER) || defined(__SCE__)) || __has_feature(modules) ||      \
    defined(__ENQCMD__)
#include <enqcmdintrin.h>
#endif

#if !defined(__SCE__) || __has_feature(modules) || defined(__SERIALIZE__)
#include <serializeintrin.h>
#endif

#if !defined(__SCE__) || __has_feature(modules) || defined(__TSXLDTRK__)
#include <tsxldtrkintrin.h>
#endif

#if defined(_MSC_VER) && __has_extension(gnu_asm)
/* Define the default attributes for these intrinsics */
#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__))
#ifdef __cplusplus
extern "C" {
#endif
/*----------------------------------------------------------------------------*\
|* Interlocked Exchange HLE
\*----------------------------------------------------------------------------*/
#if defined(__i386__) || defined(__x86_64__)
static __inline__ long __DEFAULT_FN_ATTRS
_InterlockedExchange_HLEAcquire(long volatile *_Target, long _Value) {
  __asm__ __volatile__(".byte 0xf2 ; lock ; xchg {%0, %1|%1, %0}"
                       : "+r" (_Value), "+m" (*_Target) :: "memory");
  return _Value;
}
static __inline__ long __DEFAULT_FN_ATTRS
_InterlockedExchange_HLERelease(long volatile *_Target, long _Value) {
  __asm__ __volatile__(".byte 0xf3 ; lock ; xchg {%0, %1|%1, %0}"
                       : "+r" (_Value), "+m" (*_Target) :: "memory");
  return _Value;
}
#endif
#if defined(__x86_64__)
static __inline__ __int64 __DEFAULT_FN_ATTRS
_InterlockedExchange64_HLEAcquire(__int64 volatile *_Target, __int64 _Value) {
  __asm__ __volatile__(".byte 0xf2 ; lock ; xchg {%0, %1|%1, %0}"
                       : "+r" (_Value), "+m" (*_Target) :: "memory");
  return _Value;
}
static __inline__ __int64 __DEFAULT_FN_ATTRS
_InterlockedExchange64_HLERelease(__int64 volatile *_Target, __int64 _Value) {
  __asm__ __volatile__(".byte 0xf3 ; lock ; xchg {%0, %1|%1, %0}"
                       : "+r" (_Value), "+m" (*_Target) :: "memory");
  return _Value;
}
#endif
/*----------------------------------------------------------------------------*\
|* Interlocked Compare Exchange HLE
\*----------------------------------------------------------------------------*/
#if defined(__i386__) || defined(__x86_64__)
static __inline__ long __DEFAULT_FN_ATTRS
_InterlockedCompareExchange_HLEAcquire(long volatile *_Destination,
                              long _Exchange, long _Comparand) {
  __asm__ __volatile__(".byte 0xf2 ; lock ; cmpxchg {%2, %1|%1, %2}"
                       : "+a" (_Comparand), "+m" (*_Destination)
                       : "r" (_Exchange) : "memory");
  return _Comparand;
}
static __inline__ long __DEFAULT_FN_ATTRS
_InterlockedCompareExchange_HLERelease(long volatile *_Destination,
                              long _Exchange, long _Comparand) {
  __asm__ __volatile__(".byte 0xf3 ; lock ; cmpxchg {%2, %1|%1, %2}"
                       : "+a" (_Comparand), "+m" (*_Destination)
                       : "r" (_Exchange) : "memory");
  return _Comparand;
}
#endif
#if defined(__x86_64__)
static __inline__ __int64 __DEFAULT_FN_ATTRS
_InterlockedCompareExchange64_HLEAcquire(__int64 volatile *_Destination,
                              __int64 _Exchange, __int64 _Comparand) {
  __asm__ __volatile__(".byte 0xf2 ; lock ; cmpxchg {%2, %1|%1, %2}"
                       : "+a" (_Comparand), "+m" (*_Destination)
                       : "r" (_Exchange) : "memory");
  return _Comparand;
}
static __inline__ __int64 __DEFAULT_FN_ATTRS
_InterlockedCompareExchange64_HLERelease(__int64 volatile *_Destination,
                              __int64 _Exchange, __int64 _Comparand) {
  __asm__ __volatile__(".byte 0xf3 ; lock ; cmpxchg {%2, %1|%1, %2}"
                       : "+a" (_Comparand), "+m" (*_Destination)
                       : "r" (_Exchange) : "memory");
  return _Comparand;
}
#endif
#ifdef __cplusplus
}
#endif

#undef __DEFAULT_FN_ATTRS

#endif /* defined(_MSC_VER) && __has_extension(gnu_asm) */


#include <svmlintrin.h>

#if __has_extension(gnu_asm)

#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__))

static __inline__ void __DEFAULT_FN_ATTRS
_clac(void) {
  __asm__ __volatile__ ("clac" : : : "memory");
}

static __inline__ void __DEFAULT_FN_ATTRS
_stac(void) {
  __asm__ __volatile__ ("stac" : : : "memory");
}

static __inline__ void __DEFAULT_FN_ATTRS
_lgdt(void *__ptr) {
  __asm__ __volatile__("lgdt %0" : : "m"(*(short *)(__ptr)) : "memory");
}

static __inline__ void __DEFAULT_FN_ATTRS
_sgdt(void *__ptr) {
  __asm__ __volatile__("sgdt %0" : "=m"(*(short *)(__ptr)) : : "memory");
}

#endif /* __has_extension(gnu_asm) */

#if !defined(_MSC_VER) && !defined(__cpuid) && __has_extension(gnu_asm)
#ifdef __cplusplus
extern "C" {
#endif
void __cpuid(int[4], int);
#ifdef __cplusplus
}
#endif

static __inline__ unsigned long long __DEFAULT_FN_ATTRS
__readmsr(unsigned int __register) {
  // Loads the contents of a 64-bit model specific register (MSR) specified in
  // the ECX register into registers EDX:EAX. The EDX register is loaded with
  // the high-order 32 bits of the MSR and the EAX register is loaded with the
  // low-order 32 bits. If less than 64 bits are implemented in the MSR being
  // read, the values returned to EDX:EAX in unimplemented bit locations are
  // undefined.
  unsigned int __edx;
  unsigned int __eax;
  __asm__ ("rdmsr" : "=d"(__edx), "=a"(__eax) : "c"(__register));
  return (((unsigned long long)__edx) << 32) | (unsigned long long)__eax;
}

static __inline__ void __DEFAULT_FN_ATTRS
__writemsr(unsigned int __register, unsigned long long __data) {
  __asm__ ("wrmsr" : : "d"((unsigned)(__data >> 32)), "a"((unsigned)__data), "c"(__register));
}

#define _readmsr(R) __readmsr(R)
#define _writemsr(R, D) __writemsr(R, D)

#endif /* !defined(_MSC_VER) && __has_extension(gnu_asm) */

#undef __DEFAULT_FN_ATTRS

/* Definitions of feature list to be used by feature select intrinsics */
#define _FEATURE_GENERIC_IA32        (1ULL     )
#define _FEATURE_FPU                 (1ULL << 1)
#define _FEATURE_CMOV                (1ULL << 2)
#define _FEATURE_MMX                 (1ULL << 3)
#define _FEATURE_FXSAVE              (1ULL << 4)
#define _FEATURE_SSE                 (1ULL << 5)
#define _FEATURE_SSE2                (1ULL << 6)
#define _FEATURE_SSE3                (1ULL << 7)
#define _FEATURE_SSSE3               (1ULL << 8)
#define _FEATURE_SSE4_1              (1ULL << 9)
#define _FEATURE_SSE4_2              (1ULL << 10)
#define _FEATURE_MOVBE               (1ULL << 11)
#define _FEATURE_POPCNT              (1ULL << 12)
#define _FEATURE_PCLMULQDQ           (1ULL << 13)
#define _FEATURE_AES                 (1ULL << 14)
#define _FEATURE_F16C                (1ULL << 15)
#define _FEATURE_AVX                 (1ULL << 16)
#define _FEATURE_RDRND               (1ULL << 17)
#define _FEATURE_FMA                 (1ULL << 18)
#define _FEATURE_BMI                 (1ULL << 19)
#define _FEATURE_LZCNT               (1ULL << 20)
#define _FEATURE_HLE                 (1ULL << 21)
#define _FEATURE_RTM                 (1ULL << 22)
#define _FEATURE_AVX2                (1ULL << 23)
#define _FEATURE_AVX512DQ            (1ULL << 24)
#define _FEATURE_PTWRITE             (1ULL << 25)
#define _FEATURE_AVX512F             (1ULL << 27)
#define _FEATURE_ADX                 (1ULL << 28)
#define _FEATURE_RDSEED              (1ULL << 29)
#define _FEATURE_AVX512IFMA52        (1ULL << 30)
#define _FEATURE_AVX512ER            (1ULL << 32)
#define _FEATURE_AVX512PF            (1ULL << 33)
#define _FEATURE_AVX512CD            (1ULL << 34)
#define _FEATURE_SHA                 (1ULL << 35)
#define _FEATURE_MPX                 (1ULL << 36)
#define _FEATURE_AVX512BW            (1ULL << 37)
#define _FEATURE_AVX512VL            (1ULL << 38)
#define _FEATURE_AVX512VBMI          (1ULL << 39)
#define _FEATURE_AVX512_4FMAPS       (1ULL << 40)
#define _FEATURE_AVX512_4VNNIW       (1ULL << 41)
#define _FEATURE_AVX512_VPOPCNTDQ    (1ULL << 42)
#define _FEATURE_AVX512_BITALG       (1ULL << 43)
#define _FEATURE_AVX512_VBMI2        (1ULL << 44)
#define _FEATURE_GFNI                (1ULL << 45)
#define _FEATURE_VAES                (1ULL << 46)
#define _FEATURE_VPCLMULQDQ          (1ULL << 47)
#define _FEATURE_AVX512_VNNI         (1ULL << 48)
#define _FEATURE_CLWB                (1ULL << 49)
#define _FEATURE_RDPID               (1ULL << 50)
#define _FEATURE_IBT                 (1ULL << 51)
#define _FEATURE_SHSTK               (1ULL << 52)
#define _FEATURE_SGX                 (1ULL << 53)
#define _FEATURE_WBNOINVD            (1ULL << 54)
#define _FEATURE_PCONFIG             (1ULL << 55)
#define _FEATURE_AXV512_VP2INTERSECT (1ULL << 56)
#define _FEATURE_AXV512_FP16         (1ULL << 60)

/* NOTE: Features with bit_pos >= 64 are defined in Page2       */
/* Example of how allow_cpu_features uses those features:       */
/*    __attribute__((allow_cpu_features(0, _FEATURE_CLDEMOTE))) */
#define _FEATURE_CLDEMOTE            (1ULL     )
#define _FEATURE_MOVDIRI             (1ULL << 1)
#define _FEATURE_MOVDIR64B           (1ULL << 2)
#define _FEATURE_WAITPKG             (1ULL << 3)
#define _FEATURE_AVX512_Bf16         (1ULL << 4)
#define _FEATURE_ENQCMD              (1ULL << 5)
#define _FEATURE_AVX_VNNI            (1ULL << 6)
#define _FEATURE_AMX_TILE            (1ULL << 7)
#define _FEATURE_AMX_INT8            (1ULL << 8)
#define _FEATURE_AMX_BF16            (1ULL << 9)
#define _FEATURE_KL                  (1ULL << 10)
#define _FEATURE_WIDE_KL             (1ULL << 11)
#define _FEATURE_HRESET              (1ULL << 12)
#define _FEATURE_UINTR               (1ULL << 13)
#define _FEATURE_PREFETCHI           (1ULL << 14)
#define _FEATURE_AVXVNNIINT8         (1ULL << 15)
#define _FEATURE_CMPCCXADD           (1ULL << 16)
#define _FEATURE_AVXIFMA             (1ULL << 17)
#define _FEATURE_AVXNECONVERT        (1ULL << 18)
#define _FEATURE_RAOINT              (1ULL << 19)
#define _FEATURE_AMX_FP16            (1ULL << 20)
#define _FEATURE_AMX_COMPLEX         (1ULL << 21)
#define _FEATURE_SHA512              (1ULL << 22)
#define _FEATURE_SM3                 (1ULL << 23)
#define _FEATURE_SM4                 (1ULL << 24)
#define _FEATURE_AVXVNNIINT16        (1ULL << 25)
#define _FEATURE_USERMSR             (1ULL << 26)
#define _FEATURE_AVX10_1_256         (1ULL << 27)
#define _FEATURE_AVX10_1_512         (1ULL << 28)
#define _FEATURE_APXF                (1ULL << 29)
#define _FEATURE_MSRLIST             (1ULL << 30)
#define _FEATURE_WRMSRNS             (1ULL << 31)
#define _FEATURE_PBNDKB              (1ULL << 32)

#pragma float_control(pop)

#endif /* __IMMINTRIN_H */
