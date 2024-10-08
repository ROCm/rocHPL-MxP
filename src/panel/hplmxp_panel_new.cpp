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

static int deviceMalloc(HPLMXP_T_grid&  grid,
                        void**          ptr,
                        const size_t    bytes) {

  hipError_t err = hipMalloc(ptr, bytes);

  /*Check allocation is valid*/
  int error = (err != hipSuccess);
  HPLMXP_all_reduce(&error, 1, HPLMXP_MAX, grid.all_comm);
  if(error != 0) {
    return HPLMXP_FAILURE;
  } else {
    return HPLMXP_SUCCESS;
  }
}

template<typename T>
int HPLMXP_pdpanel_new(HPLMXP_T_grid&      grid,
                       HPLMXP_T_pmat<T>&   A,
                       const int           N,
                       const int           NB,
                       const int           IA,
                       const int           JA,
                       const int           II,
                       const int           JJ,
                       HPLMXP_T_panel<T>&  P,
                       size_t&             totalMem) {
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

  P.ldl = (((sizeof(fp16_t) * A.mp + 767) / 1024) * 1024 + 256) / sizeof(fp16_t);
  size_t numbytes = sizeof(fp16_t) * A.nb * P.ldl;
  totalMem += numbytes;
  if(deviceMalloc(grid, reinterpret_cast<void**>(&(P.L)), numbytes) != HPLMXP_SUCCESS) {
    if(grid.iam == 0)
      HPLMXP_pwarn(stderr,
                   __LINE__,
                   "HPLMXP_pdpanel_new",
                   "Device memory allocation failed for L. Requested %g GiBs total. Test Skiped.",
                   ((double)totalMem) / (1024 * 1024 * 1024));
    return HPLMXP_FAILURE;
  }

  P.ldu = (((sizeof(fp16_t) * A.nq + 767) / 1024) * 1024 + 256) / sizeof(fp16_t);
  numbytes = sizeof(fp16_t) * A.nb * P.ldu;
  totalMem += numbytes;
  if(deviceMalloc(grid, reinterpret_cast<void**>(&(P.U)), numbytes) != HPLMXP_SUCCESS) {
    if(grid.iam == 0)
      HPLMXP_pwarn(stderr,
                   __LINE__,
                   "HPLMXP_pdpanel_new",
                   "Device memory allocation failed for U. Requested %g GiBs total. Test Skiped.",
                   ((double)totalMem) / (1024 * 1024 * 1024));
    return HPLMXP_FAILURE;
  }

  return HPLMXP_SUCCESS;
}

template
int HPLMXP_pdpanel_new(HPLMXP_T_grid&           grid,
                       HPLMXP_T_pmat<double>&   A,
                       const int                N,
                       const int                NB,
                       const int                IA,
                       const int                JA,
                       const int                II,
                       const int                JJ,
                       HPLMXP_T_panel<double>&  P,
                       size_t&                  totalMem);

template
int HPLMXP_pdpanel_new(HPLMXP_T_grid&          grid,
                       HPLMXP_T_pmat<float>&   A,
                       const int               N,
                       const int               NB,
                       const int               IA,
                       const int               JA,
                       const int               II,
                       const int               JJ,
                       HPLMXP_T_panel<float>&  P,
                       size_t&                 totalMem);
