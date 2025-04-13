/*===---- x86intrin.h - X86 intrinsics -------------------------------------=== */
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

#ifndef __X86INTRIN_H
#define __X86INTRIN_H

/* Turn fp precise on for intrinsics, push state to restore at end. */
#pragma float_control(push)
#pragma float_control(precise, on)

/* Enable fast contract if it was enabled on the command line */
#pragma STDC FP_CONTRACT DEFAULT

/* Moved to immintrin.h */
/*#include <ia32intrin.h>*/

#include <immintrin.h>

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__PRFCHW__) ||    \
    defined(__M_INTRINSIC_PROMOTE__)
#include <prfchwintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__SSE4A__) ||     \
    defined(__M_INTRINSIC_PROMOTE__)
#include <ammintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__FMA4__) ||      \
    defined(__M_INTRINSIC_PROMOTE__)
#include <fma4intrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__XOP__) ||       \
    defined(__M_INTRINSIC_PROMOTE__)
#include <xopintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__TBM__) ||       \
    defined(__M_INTRINSIC_PROMOTE__)
#include <tbmintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__LWP__) ||       \
    defined(__M_INTRINSIC_PROMOTE__)
#include <lwpintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__MWAITX__) ||    \
    defined(__M_INTRINSIC_PROMOTE__)
#include <mwaitxintrin.h>
#endif

#if !(defined(__SCE__)) || __has_feature(modules) || defined(__CLZERO__) ||    \
    defined(__M_INTRINSIC_PROMOTE__)
#include <clzerointrin.h>
#endif

#if !defined(__SCE__) || __has_feature(modules) || defined(__RDPRU__)
#include <rdpruintrin.h>
#endif

#pragma float_control(pop)

#endif /* __X86INTRIN_H */
