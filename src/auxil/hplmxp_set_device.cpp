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
    void hpl_set_knl(const int N, const T alpha, T* __restrict__ x) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < N) { x[i] = alpha; }
}

template <typename T>
void HPLMXP_set(const int N, const T alpha, T* x) {

  if(N <= 0) return;

  dim3 grid_size((N + BLOCK_DIM - 1) / BLOCK_DIM);
  dim3 block_size(BLOCK_DIM);

  hpl_set_knl<<<grid_size, block_size, 0, computeStream>>>(N, alpha, x);
  HIP_CHECK(hipGetLastError());
}

template void HPLMXP_set(const int n, const double alpha, double* x);

template void HPLMXP_set(const int n, const float alpha, float* x);

template void HPLMXP_set(const int n, const __half alpha, __half* x);
