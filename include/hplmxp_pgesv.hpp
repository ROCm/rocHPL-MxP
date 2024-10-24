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
#include "hplmxp_panel.hpp"

/*
 * ---------------------------------------------------------------------
 * #typedefs and data structures
 * ---------------------------------------------------------------------
 */
typedef struct HPLMXP_S_palg {
  HPLMXP_T_TOP btopo; /* row broadcast topology */
  fp64_t       epsil; /* epsilon machine */
  fp64_t       thrsh; /* threshold */
  int          its;   /* iterations */
} HPLMXP_T_palg;

template <typename A_t, typename C_t>
struct HPLMXP_T_pmat {
  A_t*   A;     /* pointer to local piece of A */
  int    n;     /* global problem size */
  int    nb;    /* blocking factor */
  int    ld;    /* local leading dimension */
  int    mp;    /* local number of rows */
  int    nq;    /* local number of columns */
  int    nbrow; /* local number of row panels */
  int    nbcol; /* local number of column panels */
  fp64_t*   d;     /* pointer to diagonal of A */
  fp64_t*   b;     /* pointer to rhs vector */
  fp64_t*   x;     /* pointer to solution vector */
  fp64_t norma; /* matrix norm */
  fp64_t normb; /* rhs vector norm */
  fp64_t res;   /* residual norm */

  // Accumulation type in GEMMs, and precision for getrf
  using factType_t = typename gemmTypes<C_t>::computeType;
  factType_t* piv;  /* pointer to diagonal panel */
  C_t*        pivL; /* pointer to diagonal panel */
  C_t*        pivU; /* pointer to diagonal panel */

  fp64_t*   work; /* workspace */

  HPLMXP_T_panel<A_t, C_t> panels[2];

  size_t totalMem;
};

/*
 * ---------------------------------------------------------------------
 * Function prototypes
 * ---------------------------------------------------------------------
 */
void HPLMXP_pgesv(HPLMXP_T_grid&                 grid,
                  HPLMXP_T_palg&                 algo,
                  HPLMXP_T_pmat<approx_type_t,
                                compute_type_t>& A);

void HPLMXP_pgetrf(HPLMXP_T_grid&                 grid,
                   HPLMXP_T_palg&                 algo,
                   HPLMXP_T_pmat<approx_type_t,
                                 compute_type_t>& A);

void HPLMXP_iterative_refinement(HPLMXP_T_grid&                 grid,
                                 HPLMXP_T_palg&                 algo,
                                 HPLMXP_T_pmat<approx_type_t,
                                               compute_type_t>& A);

void HPLMXP_pgemv(HPLMXP_T_grid&                         grid,
                  HPLMXP_T_pmat<approx_type_t,
                                compute_type_t>&         A,
                  fp64_t                         alpha,
                  fp64_t*                        x,
                  fp64_t                         beta,
                  fp64_t*                        y);

void HPLMXP_ptrsvL(HPLMXP_T_grid&                         grid,
                   HPLMXP_T_pmat<approx_type_t,
                                 compute_type_t>&         A,
                   fp64_t*                        x,
                   fp64_t*                        work);

void HPLMXP_ptrsvU(HPLMXP_T_grid&                         grid,
                   HPLMXP_T_pmat<approx_type_t,
                                 compute_type_t>&         A,
                   fp64_t*                        x,
                   fp64_t*                        work);

#endif
/*
 * End of hpl_pgesv.hpp
 */
