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
#ifndef HPLMXP_PGESV_HPP
#define HPLMXP_PGESV_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hplmxp_grid.hpp"

/*
 * ---------------------------------------------------------------------
 * #typedefs and data structures
 * ---------------------------------------------------------------------
 */
typedef struct HPLMXP_S_palg {
  HPLMXP_T_TOP btopo; /* row broadcast topology */
  fp64_t       epsil; /* epsilon machine */
  fp64_t       thrsh; /* threshold */
} HPLMXP_T_palg;

template <typename T>
struct HPLMXP_T_pmat {
  T*     A;     /* pointer to local piece of A */
  int    n;     /* global problem size */
  int    nb;    /* blocking factor */
  int    ld;    /* local leading dimension */
  int    mp;    /* local number of rows */
  int    nq;    /* local number of columns */
  int    nbrow; /* local number of row panels */
  int    nbcol; /* local number of column panels */
  T*     d;     /* pointer to diagonal of A */
  T*     b;     /* pointer to rhs vector */
  T*     x;     /* pointer to solution vector */
  fp64_t norma; /* matrix norm */
  fp64_t normb; /* rhs vector norm */
  fp64_t res;   /* residual norm */
};

/*
 * ---------------------------------------------------------------------
 * Function prototypes
 * ---------------------------------------------------------------------
 */
void HPLMXP_pgesv(HPLMXP_T_grid&         grid,
                  HPLMXP_T_palg&         algo,
                  HPLMXP_T_pmat<fp64_t>& A,
                  HPLMXP_T_pmat<fp32_t>& LU);

void HPLMXP_pgetrf(HPLMXP_T_grid&         grid,
                   HPLMXP_T_palg&         algo,
                   HPLMXP_T_pmat<fp32_t>& A);

void HPLMXP_iterative_refinement(HPLMXP_T_grid&         grid,
                                 HPLMXP_T_palg&         algo,
                                 HPLMXP_T_pmat<fp64_t>& A,
                                 HPLMXP_T_pmat<fp32_t>& LU);

void HPLMXP_pgemv(HPLMXP_T_grid&         grid,
                  HPLMXP_T_pmat<fp64_t>& A,
                  fp64_t                 alpha,
                  fp64_t*                x,
                  fp64_t                 beta,
                  fp64_t*                y);

void HPLMXP_ptrsvL(HPLMXP_T_grid&         grid,
                   HPLMXP_T_pmat<fp32_t>& LU,
                   fp64_t*                x,
                   fp64_t*                work);

void HPLMXP_ptrsvU(HPLMXP_T_grid&         grid,
                   HPLMXP_T_pmat<fp32_t>& LU,
                   fp64_t*                x,
                   fp64_t*                work);

#endif
/*
 * End of hpl_pgesv.hpp
 */
