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

template<typename T>
int HPLMXP_pdpanel_free(HPLMXP_T_panel<T>& P) {
  /*
   * Purpose
   * =======
   *
   * HPLMXP_pdpanel_free deallocates  the panel resources  and  stores the error
   * code returned by the panel factorization.
   *
   * Arguments
   * =========
   *
   * PANEL   (local input/output)          HPLMXP_T_panel &
   *         On entry,  PANEL  points  to  the  panel data  structure from
   *         which the resources should be deallocated.
   *
   * ---------------------------------------------------------------------
   */

  if(P.L) {
    HIP_CHECK(hipFree(P.L));
    P.L = nullptr;
  }
  if(P.U) {
    HIP_CHECK(hipFree(P.U));
    P.U = nullptr;
  }

  return (HPLMXP_SUCCESS);
}

template
int HPLMXP_pdpanel_free(HPLMXP_T_panel<double>& P);

template
int HPLMXP_pdpanel_free(HPLMXP_T_panel<float>& P);
