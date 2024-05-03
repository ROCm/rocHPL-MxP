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

#define BLOCK_SIZE 512

template <typename T>
__launch_bounds__(BLOCK_SIZE) __global__ void norminf_1(const int N,
                                                        const int NB,
                                                        const int myrow,
                                                        const int mycol,
                                                        const int nprow,
                                                        const int npcol,
                                                        const T* __restrict__ x,
                                                        T* __restrict__ norm) {

  __shared__ T s_norm[BLOCK_SIZE];

  const int t = threadIdx.x;
  const int b = blockIdx.x;

  T r_norm = 0.0;
  for(int id = b * BLOCK_SIZE + t; id < N; id += gridDim.x * BLOCK_SIZE) {
    const int i    = id / NB;
    const int ipos = myrow + i * nprow;

    if(ipos % npcol == mycol) { // diagonal panel
      const T xi = fabs(x[id]);
      r_norm     = std::max(xi, r_norm);
    }
  }
  s_norm[t] = r_norm;

  __syncthreads();

  for(int k = BLOCK_SIZE / 2; k > 0; k /= 2) {
    if(t < k) { s_norm[t] = std::max(s_norm[t + k], s_norm[t]); }
    __syncthreads();
  }

  if(t == 0) norm[b] = s_norm[0];
}

template <typename T>
__launch_bounds__(BLOCK_SIZE) __global__
    void norminf_2(const int N, T* __restrict__ norm) {

  __shared__ T s_norm[BLOCK_SIZE];

  const int t = threadIdx.x;

  T r_norm = 0.0;
  for(size_t id = t; id < N; id += BLOCK_SIZE) {
    r_norm = std::max(norm[id], r_norm);
  }
  s_norm[t] = r_norm;

  __syncthreads();

  for(int k = BLOCK_SIZE / 2; k > 0; k /= 2) {
    if(t < k) { s_norm[t] = std::max(s_norm[t + k], s_norm[t]); }
    __syncthreads();
  }

  if(t == 0) norm[0] = s_norm[0];
}

template <typename T>
T HPLMXP_plange(const HPLMXP_T_grid& GRID,
                const int            N,
                const int            NB,
                const T*             x) {
  /*
   * Purpose
   * =======
   *
   * HPLMXP_plange returns the infinity norm, of a distributed vector x
   *
   * Arguments
   * =========
   *
   * GRID    (local input)                 const HPLMXP_T_grid *
   *         On entry,  GRID  points  to the data structure containing the
   *         process grid information.
   *
   * N       (global input)                const int
   *         On entry,  N specifies the number of columns of the matrix A.
   *         N must be at least zero.
   *
   * NB      (global input)                const int
   *         On entry,  NB specifies the blocking factor used to partition
   *         and distribute the matrix. NB must be larger than one.
   *
   * x       (local input)                 const T *
   *         On entry,  x  points to an vector of dimension N,
   *         that contains the local pieces of the distributed vector x.
   *
   * ---------------------------------------------------------------------
   */

  int mycol, myrow, npcol, nprow;
  (void)HPLMXP_grid_info(GRID, nprow, npcol, myrow, mycol);

  if(N <= 0) return 0.0;

  /*
   * Find norm_inf( x )
   */
  const int grid_size =
      std::min((N + BLOCK_SIZE - 1) / BLOCK_SIZE, REDUCTION_SCRATCH_SIZE);

  T* norm_scratch = reinterpret_cast<T*>(reduction_scratch);
  T* h_norm       = reinterpret_cast<T*>(h_reduction_scratch);

  norminf_1<<<grid_size, BLOCK_SIZE, 0, computeStream>>>(
      N, NB, myrow, mycol, nprow, npcol, x, norm_scratch);
  HIP_CHECK(hipGetLastError());
  norminf_2<<<1, BLOCK_SIZE, 0, computeStream>>>(grid_size, norm_scratch);
  HIP_CHECK(hipGetLastError());

  T norm = 0.0;
  HIP_CHECK(hipMemcpyAsync(h_norm,
                           norm_scratch,
                           1 * sizeof(T),
                           hipMemcpyDeviceToHost,
                           computeStream));
  HIP_CHECK(hipDeviceSynchronize());

  HPLMXP_all_reduce(h_norm, 1, HPLMXP_MAX, GRID.all_comm);

  return h_norm[0];
}

template double HPLMXP_plange(const HPLMXP_T_grid& GRID,
                              const int            N,
                              const int            NB,
                              const double*        x);

template float HPLMXP_plange(const HPLMXP_T_grid& GRID,
                             const int            N,
                             const int            NB,
                             const float*         x);
