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
#include <hip/hip_runtime.h>

#define BLOCK_DIM 256

template <typename T>
static __launch_bounds__(BLOCK_DIM) __global__
    void paydx_knl(const int n,
                   const int b,
                   const int myrow,
                   const int mycol,
                   const int nprow,
                   const int npcol,
                   const T   alpha,
                   T const* __restrict__ x,
                   T* __restrict__ y) {

  const int id = blockIdx.x * blockDim.x + threadIdx.x;

  if(id < n) {
    const int i    = id / b;
    const int k    = id % b;
    const int ipos = myrow + i * nprow;

    if(ipos % npcol == mycol) {
      y[b * i + k] = alpha * y[b * i + k] / x[b * i + k];
    } else {
      y[b * i + k] = 0.0;
    }
  }
}

// Compute y = alpha*y/x for distributed vectors x and y over an
// nprow x npcol process grid with blocking factor NB
template <typename T>
void HPLMXP_paydx(HPLMXP_T_grid& grid,
                  const int      N,
                  const int      NB,
                  const T        alpha,
                  const T*       x,
                  T*             y) {

  if(N <= 0) return;

  dim3 grid_size((N + BLOCK_DIM - 1) / BLOCK_DIM);
  dim3 block_size(BLOCK_DIM);

  paydx_knl<<<grid_size, block_size, 0, computeStream>>>(
      N, NB, grid.myrow, grid.mycol, grid.nprow, grid.npcol, alpha, x, y);
  HIP_CHECK(hipGetLastError());
}

template void HPLMXP_paydx(HPLMXP_T_grid& grid,
                           const int      N,
                           const int      NB,
                           const double   alpha,
                           const double*  x,
                           double*        y);

template void HPLMXP_paydx(HPLMXP_T_grid& grid,
                           const int      N,
                           const int      NB,
                           const float    alpha,
                           const float*   x,
                           float*         y);

template void HPLMXP_paydx(HPLMXP_T_grid& grid,
                           const int      N,
                           const int      NB,
                           const __half   alpha,
                           const __half*  x,
                           __half*        y);
