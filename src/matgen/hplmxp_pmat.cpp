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

template <typename T>
void HPLMXP_pmat_init(HPLMXP_T_grid&    grid,
                      const int         N,
                      const int         NB,
                      HPLMXP_T_pmat<T>& A) {

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
}

template void HPLMXP_pmat_init(HPLMXP_T_grid&         grid,
                               const int              N,
                               const int              NB,
                               HPLMXP_T_pmat<double>& A);

template void HPLMXP_pmat_init(HPLMXP_T_grid&        grid,
                               const int             N,
                               const int             NB,
                               HPLMXP_T_pmat<float>& A);

template <typename T>
void HPLMXP_pmat_free(HPLMXP_T_pmat<T>& A) {
  if(A.A) {
    HIP_CHECK(hipFree(A.A));
    A.A = nullptr;
  }
  if(A.x) {
    HIP_CHECK(hipFree(A.x));
    A.x = nullptr;
  }
  if(A.d) {
    HIP_CHECK(hipFree(A.d));
    A.d = nullptr;
  }
  if(A.b) {
    HIP_CHECK(hipFree(A.b));
    A.b = nullptr;
  }
}

template void HPLMXP_pmat_free(HPLMXP_T_pmat<double>& A);

template void HPLMXP_pmat_free(HPLMXP_T_pmat<float>& A);
