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

template<typename A_t, typename C_t>
int HPLMXP_pdpanel_new(HPLMXP_T_grid&                  grid,
                       HPLMXP_T_pmat<A_t, C_t>&   A,
                       HPLMXP_T_panel<A_t, C_t>&       P) {
  /*
   * Purpose
   * =======
   *
   * HPLMXP_pdpanel_new creates a panel data structure.
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
   * P       (local input/output)          HPLMXP_T_panel &
   *         On entry,  PANEL  points to the data structure containing the
   *         panel information.
   *
   * ---------------------------------------------------------------------
   */

  P.ldl = (((sizeof(C_t) * A.mp + 767) / 1024) * 1024 + 256) / sizeof(C_t);
  size_t numbytes = sizeof(C_t) * A.nb * P.ldl;
  A.totalMem += numbytes;
  if(deviceMalloc(grid, reinterpret_cast<void**>(&(P.L)), numbytes) != HPLMXP_SUCCESS) {
    if(grid.iam == 0)
      HPLMXP_pwarn(stderr,
                   __LINE__,
                   "HPLMXP_pdpanel_new",
                   "Device memory allocation failed for L. Requested %g GiBs total. Test Skiped.",
                   ((double)A.totalMem) / (1024 * 1024 * 1024));
    return HPLMXP_FAILURE;
  }

  P.ldu = (((sizeof(C_t) * A.nq + 767) / 1024) * 1024 + 256) / sizeof(C_t);
  numbytes = sizeof(C_t) * A.nb * P.ldu;
  A.totalMem += numbytes;
  if(deviceMalloc(grid, reinterpret_cast<void**>(&(P.U)), numbytes) != HPLMXP_SUCCESS) {
    if(grid.iam == 0)
      HPLMXP_pwarn(stderr,
                   __LINE__,
                   "HPLMXP_pdpanel_new",
                   "Device memory allocation failed for U. Requested %g GiBs total. Test Skiped.",
                   ((double)A.totalMem) / (1024 * 1024 * 1024));
    return HPLMXP_FAILURE;
  }

  return HPLMXP_SUCCESS;
}

template
int HPLMXP_pdpanel_new(HPLMXP_T_grid&           grid,
                       HPLMXP_T_pmat<approx_type_t,
                                     compute_type_t>&   A,
                       HPLMXP_T_panel<approx_type_t,
                                      compute_type_t>&  P);
