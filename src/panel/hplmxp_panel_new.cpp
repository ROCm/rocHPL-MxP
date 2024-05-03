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

void HPLMXP_pdpanel_new(HPLMXP_T_grid&         grid,
                        HPLMXP_T_pmat<fp32_t>& A,
                        const int              N,
                        const int              NB,
                        const int              IA,
                        const int              JA,
                        const int              II,
                        const int              JJ,
                        HPLMXP_T_panel&        P) {
  /*
   * Purpose
   * =======
   *
   * HPLMXP_pdpanel_new creates and initializes a panel data structure.
   *
   *
   * Arguments
   * =========
   *
   * grid    (local input)                 HPLMXP_T_grid &
   *         On entry,  grid  points  to the data structure containing the
   *         process grid information.
   *
   * A       (local input/output)          HPLMXP_T_pmat &
   *         On entry, A points to the data structure containing the local
   *         array information.
   *
   * N       (local input)                 const int
   *         On entry,  N  specifies  the  global number of columns of the
   *         panel and trailing submatrix. N must be at least zero.
   *
   * NB      (global input)                const int
   *         On entry, NB specifies is the number of columns of the panel.
   *         NB must be at least zero.
   *
   * IA      (global input)                const int
   *         On entry,  IA  is  the global row index identifying the panel
   *         and trailing submatrix. IA must be at least zero.
   *
   * JA      (global input)                const int
   *         On entry, JA is the global column index identifying the panel
   *         and trailing submatrix. JA must be at least zero.
   *
   * II      (local input)                 const int
   *         On entry, II  specifies the  local  starting  row index of the
   *         submatrix.
   *
   * JJ      (local input)                 const int
   *         On entry, JJ  specifies the local starting column index of the
   *         submatrix.
   *
   * PANEL   (local input/output)          HPLMXP_T_panel &
   *         On entry,  PANEL  points to the data structure containing the
   *         panel information.
   *
   * ---------------------------------------------------------------------
   */

  P.max_lwork_size = 0;
  P.max_uwork_size = 0;
  P.grid           = nullptr;
  P.A              = nullptr;
  P.L              = nullptr;
  P.U              = nullptr;
  HPLMXP_pdpanel_init(grid, A, N, NB, IA, JA, II, JJ, P);
}
