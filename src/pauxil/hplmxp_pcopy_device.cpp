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
    void pcopy_knl(const int n,
                   const int b,
                   const int myrow,
                   const int mycol,
                   const int nprow,
                   const int npcol,
                   T const* __restrict__ x,
                   T* __restrict__ y) {

  const int id = blockIdx.x * blockDim.x + threadIdx.x;

  if(id < n) {
    const int i    = id / b;
    const int k    = id % b;
    const int ipos = myrow + i * nprow;

    if(ipos % npcol == mycol) { y[b * i + k] = x[b * i + k]; }
  }
}

// Copy a vector distributed over an nprow x npcol process grid with
// blocking factor NB
template <typename T>
void HPLMXP_pcopy(HPLMXP_T_grid& grid,
                  const int      N,
                  const int      NB,
                  const T*       x,
                  T*             y) {

  if(N <= 0) return;

  dim3 grid_size((N + BLOCK_DIM - 1) / BLOCK_DIM);
  dim3 block_size(BLOCK_DIM);

  pcopy_knl<<<grid_size, block_size, 0, computeStream>>>(
      N, NB, grid.myrow, grid.mycol, grid.nprow, grid.npcol, x, y);
  HIP_CHECK(hipGetLastError());
}

template void HPLMXP_pcopy(HPLMXP_T_grid& grid,
                           const int      N,
                           const int      NB,
                           const double*  x,
                           double*        y);

template void HPLMXP_pcopy(HPLMXP_T_grid& grid,
                           const int      N,
                           const int      NB,
                           const float*   x,
                           float*         y);

template void HPLMXP_pcopy(HPLMXP_T_grid& grid,
                           const int      N,
                           const int      NB,
                           const __half*  x,
                           __half*        y);
