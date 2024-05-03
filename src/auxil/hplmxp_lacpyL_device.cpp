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

template <typename T, typename U>
__launch_bounds__(BLOCK_DIM) __global__
    void HPLMXP_lacpyL_kernel(const int M,
                              const int N,
                              const T* __restrict__ A,
                              const int LDA,
                              U* __restrict__ B,
                              const int LDB) {

  const int I = threadIdx.x + blockIdx.x * BLOCK_DIM;
  const int J = blockIdx.y;

  if(4 * (blockIdx.x + 1) * BLOCK_DIM <= J) return;

  const bool complete_block =
      (4 * blockIdx.x * BLOCK_DIM > J) && (M / (4 * BLOCK_DIM) != blockIdx.x);

  const T* __restrict__ Aj = A + J * static_cast<size_t>(LDA);
  U* __restrict__ Bj       = B + J * static_cast<size_t>(LDB);

  if(complete_block) {
    Bj[4 * I + 0] = Aj[4 * I + 0];
    Bj[4 * I + 1] = Aj[4 * I + 1];
    Bj[4 * I + 2] = Aj[4 * I + 2];
    Bj[4 * I + 3] = Aj[4 * I + 3];
  } else {
    if((4 * I + 0 > J) && (4 * I + 0 < M)) Bj[4 * I + 0] = Aj[4 * I + 0];
    if((4 * I + 1 > J) && (4 * I + 1 < M)) Bj[4 * I + 1] = Aj[4 * I + 1];
    if((4 * I + 2 > J) && (4 * I + 2 < M)) Bj[4 * I + 2] = Aj[4 * I + 2];
    if((4 * I + 3 > J) && (4 * I + 3 < M)) Bj[4 * I + 3] = Aj[4 * I + 3];
  }
}

union fp32or16x2 {
  float  fp32;
  __half fp16[2];
};

// Specialization for converting from float to __half
template <>
__launch_bounds__(BLOCK_DIM) __global__
    void HPLMXP_lacpyL_kernel(const int M,
                              const int N,
                              const float* __restrict__ A,
                              const int LDA,
                              __half* __restrict__ B,
                              const int LDB) {

  const int I = threadIdx.x + blockIdx.x * BLOCK_DIM;
  const int J = blockIdx.y;

  if(4 * (blockIdx.x + 1) * BLOCK_DIM <= J) return;

  const bool complete_block =
      (4 * blockIdx.x * BLOCK_DIM > J) && (M / (4 * BLOCK_DIM) != blockIdx.x);

  const float* __restrict__ Aj = A + J * static_cast<size_t>(LDA);

  if(complete_block) {
    // Make the compiler emit dwordx2 load/stores
    float* __restrict__ Bj =
        reinterpret_cast<float*>(B + J * static_cast<size_t>(LDB));

    fp32or16x2 out[2];
    out[0].fp16[0] = Aj[4 * I + 0];
    out[0].fp16[1] = Aj[4 * I + 1];
    out[1].fp16[0] = Aj[4 * I + 2];
    out[1].fp16[1] = Aj[4 * I + 3];

    Bj[2 * I + 0] = out[0].fp32;
    Bj[2 * I + 1] = out[1].fp32;
  } else {
    __half* __restrict__ Bj = B + J * static_cast<size_t>(LDB);

    if((4 * I + 0 > J) && (4 * I + 0 < M)) Bj[4 * I + 0] = Aj[4 * I + 0];
    if((4 * I + 1 > J) && (4 * I + 1 < M)) Bj[4 * I + 1] = Aj[4 * I + 1];
    if((4 * I + 2 > J) && (4 * I + 2 < M)) Bj[4 * I + 2] = Aj[4 * I + 2];
    if((4 * I + 3 > J) && (4 * I + 3 < M)) Bj[4 * I + 3] = Aj[4 * I + 3];
  }
}

template <typename T, typename U>
void HPLMXP_lacpyL(const int M,
                   const int N,
                   const T*  A,
                   const int LDA,
                   U*        B,
                   const int LDB) {
  /*
   * Purpose
   * =======
   *
   * HPLMXP_lacpyL copies the lower-triangular piece of an array A into an array
   * B.
   *
   *
   * Arguments
   * =========
   *
   * M       (local input)                 const int
   *         On entry,  M specifies the number of rows of the arrays A and
   *         B. M must be at least zero.
   *
   * N       (local input)                 const int
   *         On entry,  N specifies  the number of columns of the arrays A
   *         and B. N must be at least zero.
   *
   * A       (local input)                 const T *
   *         On entry, A points to an array of dimension (LDA,N).
   *
   * LDA     (local input)                 const int
   *         On entry, LDA specifies the leading dimension of the array A.
   *         LDA must be at least MAX(1,M).
   *
   * B       (local output)                U *
   *         On entry, B points to an array of dimension (LDB,N). On exit,
   *         B is overwritten with A.
   *
   * LDB     (local input)                 const int
   *         On entry, LDB specifies the leading dimension of the array B.
   *         LDB must be at least MAX(1,M).
   *
   * ---------------------------------------------------------------------
   */

  if((M <= 0) || (N <= 0)) return;

  dim3 grid_size((M + 4 * BLOCK_DIM - 1) / (4 * BLOCK_DIM), N);
  dim3 block_size(BLOCK_DIM);

  HPLMXP_lacpyL_kernel<<<grid_size, block_size, 0, computeStream>>>(
      M, N, A, LDA, B, LDB);
  HIP_CHECK(hipGetLastError());
}

template void HPLMXP_lacpyL(const int     m,
                            const int     n,
                            const double* A,
                            const int     lda,
                            double*       B,
                            const int     ldb);

template void HPLMXP_lacpyL(const int     m,
                            const int     n,
                            const double* A,
                            const int     lda,
                            float*        B,
                            const int     ldb);

template void HPLMXP_lacpyL(const int     m,
                            const int     n,
                            const double* A,
                            const int     lda,
                            __half*       B,
                            const int     ldb);

template void HPLMXP_lacpyL(const int    m,
                            const int    n,
                            const float* A,
                            const int    lda,
                            double*      B,
                            const int    ldb);

template void HPLMXP_lacpyL(const int    m,
                            const int    n,
                            const float* A,
                            const int    lda,
                            float*       B,
                            const int    ldb);

template void HPLMXP_lacpyL(const int    m,
                            const int    n,
                            const float* A,
                            const int    lda,
                            __half*      B,
                            const int    ldb);

template void HPLMXP_lacpyL(const int     m,
                            const int     n,
                            const __half* A,
                            const int     lda,
                            double*       B,
                            const int     ldb);

template void HPLMXP_lacpyL(const int     m,
                            const int     n,
                            const __half* A,
                            const int     lda,
                            float*        B,
                            const int     ldb);

template void HPLMXP_lacpyL(const int     m,
                            const int     n,
                            const __half* A,
                            const int     lda,
                            __half*       B,
                            const int     ldb);
