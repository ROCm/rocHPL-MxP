/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    HPL - 2.2 - February 24, 2016
 *    Antoine P. Petitet
 *    University of Tennessee, Knoxville
 *    Innovative Computing Laboratory
 *    (C) Copyright 2000-2008 All Rights Reserved
 *
 *    Modified by: Noel Chalmers
 *    (C) 2018-2022 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */
#ifndef HPLMXP_AUXIL_HPP
#define HPLMXP_AUXIL_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hplmxp_misc.hpp"
#include "hplmxp_blas.hpp"
/*
 * ---------------------------------------------------------------------
 * typedef definitions
 * ---------------------------------------------------------------------
 */
typedef enum {
  HPLMXP_NORM_A = 800,
  HPLMXP_NORM_1 = 801,
  HPLMXP_NORM_I = 802
} HPLMXP_T_NORM;

typedef enum {
  HPLMXP_MACH_EPS   = 900, /* relative machine precision */
  HPLMXP_MACH_SFMIN = 901, /* safe minimum st 1/sfmin does not overflow */
  HPLMXP_MACH_BASE  = 902, /* base = base of the machine */
  HPLMXP_MACH_PREC  = 903, /* prec  = eps*base */
  HPLMXP_MACH_MLEN  = 904, /* number of (base) digits in the mantissa */
  HPLMXP_MACH_RND   = 905, /* 1.0 if rounding occurs in addition */
  HPLMXP_MACH_EMIN  = 906, /* min exponent before (gradual) underflow */
  HPLMXP_MACH_RMIN  = 907, /* underflow threshold base**(emin-1) */
  HPLMXP_MACH_EMAX  = 908, /* largest exponent before overflow */
  HPLMXP_MACH_RMAX  = 909  /* overflow threshold - (base**emax)*(1-eps) */

} HPLMXP_T_MACH;
/*
 * ---------------------------------------------------------------------
 * Function prototypes
 * ---------------------------------------------------------------------
 */
void HPLMXP_fprintf(FILE*, const char*, ...);
void HPLMXP_warn(FILE*, int, const char*, const char*, ...);
void HPLMXP_abort(int, const char*, const char*, ...);

template <typename T, typename U>
void HPLMXP_lacpy(const int m,
                  const int n,
                  const T*  A,
                  const int lda,
                  U*        B,
                  const int ldb);

template <typename T, typename U>
void HPLMXP_latcpy(const int m,
                   const int n,
                   const T*  A,
                   const int lda,
                   U*        B,
                   const int ldb);

template <typename T, typename U>
void HPLMXP_lacpyL(const int M,
                   const int N,
                   const T*  A,
                   const int LDA,
                   U*        B,
                   const int LDB);

template <typename T, typename U>
void HPLMXP_lacpyU(const int M,
                   const int N,
                   const T*  A,
                   const int LDA,
                   U*        B,
                   const int LDB);

template <typename T, typename U>
void HPLMXP_latcpyU(const int M,
                    const int N,
                    const T*  A,
                    const int LDA,
                    U*        B,
                    const int LDB);

template <typename T>
void HPLMXP_identity(const int M, T* A, const int LDA);

template <typename T>
void HPLMXP_copy(const int N, const T* x, T* y);

template <typename T>
void HPLMXP_axpy(const int N, const T alpha, const T* x, T* y);

template <typename T>
void HPLMXP_set(const int N, const T alpha, T* x);

template <typename T>
void HPLMXP_scale(const int N, const T alpha, T* x);

double HPLMXP_dlange(const HPLMXP_T_NORM,
                     const int,
                     const int,
                     const double*,
                     const int);

template <typename T>
T HPLMXP_lamch(const HPLMXP_T_MACH CMACH);

#endif
/*
 * End of hpl_auxil.hpp
 */
