/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    Noel Chalmers
 *    (C) 2018-2022 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

#include "hplmxp.hpp"

template <typename A_t, typename C_t>
void HPLMXP_pmat_init(HPLMXP_T_grid&                grid,
                      const int                     N,
                      const int                     NB,
                      HPLMXP_T_pmat<A_t, C_t>& A) {

  const int nblocks = N / NB;

  A.n     = N;
  A.nb    = NB;
  A.nbrow = (nblocks - grid.myrow + grid.nprow - 1) / grid.nprow;
  A.nbcol = (nblocks - grid.mycol + grid.npcol - 1) / grid.npcol;

  A.mp = A.nbrow * NB;
  A.nq = A.nbcol * NB;

  A.A = nullptr;
  A.d = nullptr;
  A.b = nullptr;
  A.x = nullptr;

  A.ld = 0;

  A.norma = 0.0;
  A.normb = 0.0;
  A.res   = 0.0;

  A.piv  = nullptr;
  A.pivL = nullptr;
  A.pivU = nullptr;
  A.work = nullptr;

  A.panels[0].L = nullptr;
  A.panels[0].U = nullptr;
  A.panels[1].L = nullptr;
  A.panels[1].U = nullptr;
}

template void HPLMXP_pmat_init(HPLMXP_T_grid&         grid,
                               const int              N,
                               const int              NB,
                               HPLMXP_T_pmat<approx_type_t,
                                             compute_type_t>& A);

template <typename A_t, typename C_t>
void HPLMXP_pmat_free(HPLMXP_T_pmat<A_t, C_t>& A) {

  if(A.work) {
    HIP_CHECK(hipFree(A.work));
    A.work = nullptr;
  }

  HPLMXP_pdpanel_free(A.panels[1]);
  HPLMXP_pdpanel_free(A.panels[0]);

  if(A.pivU) {
    HIP_CHECK(hipFree(A.pivU));
    A.pivU = nullptr;
  }
  if(A.pivL) {
    HIP_CHECK(hipFree(A.pivL));
    A.pivL = nullptr;
  }
  if(A.piv) {
    HIP_CHECK(hipFree(A.piv));
    A.piv = nullptr;
  }

  if(A.b) {
    HIP_CHECK(hipFree(A.b));
    A.b = nullptr;
  }
  if(A.d) {
    HIP_CHECK(hipFree(A.d));
    A.d = nullptr;
  }
  if(A.x) {
    HIP_CHECK(hipFree(A.x));
    A.x = nullptr;
  }
  if(A.A) {
    HIP_CHECK(hipFree(A.A));
    A.A = nullptr;
  }
}

template void HPLMXP_pmat_free(HPLMXP_T_pmat<approx_type_t,
                                             compute_type_t>& A);
