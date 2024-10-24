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

#include "hplmxp.hpp"

void HPLMXP_pgesv(HPLMXP_T_grid&                 grid,
                  HPLMXP_T_palg&                 algo,
                  HPLMXP_T_pmat<approx_type_t,
                                compute_type_t>& A) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdgesv solve a N-by-N linear system using by LU factoring a lower
   * precision proxy, and using the LU factorization in an iterative
   * refinement method.
   *
   * Arguments
   * =========
   *
   * GRID    (local input)                 HPL_T_grid *
   *         On entry,  GRID  points  to the data structure containing the
   *         process grid information.
   *
   * ALGO    (global input)                HPL_T_palg *
   *         On entry,  ALGO  points to  the data structure containing the
   *         algorithmic parameters.
   *
   * A       (local input/output)          HPL_T_pmat *
   *         On entry, A points to the data structure containing the local
   *         array information.
   *
   * ---------------------------------------------------------------------
   */

  HPLMXP_ptimer(HPLMXP_TIMING_PGETRF);
  HPLMXP_pgetrf(grid, algo, A);
  HPLMXP_ptimer(HPLMXP_TIMING_PGETRF);

  HPLMXP_ptimer(HPLMXP_TIMING_IR);
  HPLMXP_iterative_refinement(grid, algo, A);
  HPLMXP_ptimer(HPLMXP_TIMING_IR);
}
