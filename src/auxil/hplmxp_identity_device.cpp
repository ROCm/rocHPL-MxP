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
__launch_bounds__(BLOCK_DIM) __global__
    void HPLMXP_identity_kernel(const int M, T* __restrict__ A, const int LDA) {

  const int I = threadIdx.x + blockIdx.x * BLOCK_DIM;
  const int J = blockIdx.y;

  const bool complete_block = (M / (4 * BLOCK_DIM) != blockIdx.x);

  T* __restrict__ Aj = A + J * static_cast<size_t>(LDA);

  T Aij[4];
  Aij[0] = 0.0;
  Aij[1] = 0.0;
  Aij[2] = 0.0;
  Aij[3] = 0.0;

  if(4 * I + 0 == J) Aij[0] = 1.0;
  if(4 * I + 1 == J) Aij[1] = 1.0;
  if(4 * I + 2 == J) Aij[2] = 1.0;
  if(4 * I + 3 == J) Aij[3] = 1.0;

  if(complete_block) {
    Aj[4 * I + 0] = Aij[0];
    Aj[4 * I + 1] = Aij[1];
    Aj[4 * I + 2] = Aij[2];
    Aj[4 * I + 3] = Aij[3];
  } else {
    if(4 * I + 0 < M) Aj[4 * I + 0] = Aij[0];
    if(4 * I + 1 < M) Aj[4 * I + 1] = Aij[1];
    if(4 * I + 2 < M) Aj[4 * I + 2] = Aij[2];
    if(4 * I + 3 < M) Aj[4 * I + 3] = Aij[3];
  }
}

union fp32or16x2 {
  float  fp32;
  __half fp16[2];
};

// Specialization for converting from float to __half
template <>
__launch_bounds__(BLOCK_DIM) __global__
    void HPLMXP_identity_kernel(const int M,
                                __half* __restrict__ A,
                                const int LDA) {

  const int I = threadIdx.x + blockIdx.x * BLOCK_DIM;
  const int J = blockIdx.y;

  const bool complete_block = (M / (4 * BLOCK_DIM) != blockIdx.x);

  if(complete_block) {
    // Make the compiler emit dwordx2 load/stores
    float* __restrict__ Aj =
        reinterpret_cast<float*>(A + J * static_cast<size_t>(LDA));

    fp32or16x2 out[2];
    out[0].fp16[0] = 0.0;
    out[0].fp16[1] = 0.0;
    out[1].fp16[0] = 0.0;
    out[1].fp16[1] = 0.0;

    if(4 * I + 0 == J) out[0].fp16[0] = 1.0;
    if(4 * I + 1 == J) out[0].fp16[1] = 1.0;
    if(4 * I + 2 == J) out[1].fp16[0] = 1.0;
    if(4 * I + 3 == J) out[1].fp16[1] = 1.0;

    Aj[2 * I + 0] = out[0].fp32;
    Aj[2 * I + 1] = out[1].fp32;
  } else {
    __half* __restrict__ Aj = A + J * static_cast<size_t>(LDA);

    __half Aij[4];
    Aij[0] = 0.0;
    Aij[1] = 0.0;
    Aij[2] = 0.0;
    Aij[3] = 0.0;

    if(4 * I + 0 == J) Aij[0] = 1.0;
    if(4 * I + 1 == J) Aij[1] = 1.0;
    if(4 * I + 2 == J) Aij[2] = 1.0;
    if(4 * I + 3 == J) Aij[3] = 1.0;

    if(4 * I + 0 < M) Aj[4 * I + 0] = Aij[0];
    if(4 * I + 1 < M) Aj[4 * I + 1] = Aij[1];
    if(4 * I + 2 < M) Aj[4 * I + 2] = Aij[2];
    if(4 * I + 3 < M) Aj[4 * I + 3] = Aij[3];
  }
}

template <typename T>
void HPLMXP_identity(const int M, T* A, const int LDA) {
  /*
   * Purpose
   * =======
   *
   * HPLMXP_identity write an MxM identity matrix into an array A
   *
   *
   * Arguments
   * =========
   *
   * M       (local input)                 const int
   *         On entry,  M specifies the number of rows of the array A.
   *         M must be at least zero.
   *
   * A       (local output)                 const T *
   *         On entry, A points to an array of dimension (LDA,M).
   *
   * LDA     (local input)                 const int
   *         On entry, LDA specifies the leading dimension of the array A.
   *         LDA must be at least MAX(1,M).
   *
   * ---------------------------------------------------------------------
   */

  if(M <= 0) return;

  dim3 grid_size((M + 4 * BLOCK_DIM - 1) / (4 * BLOCK_DIM), M);
  dim3 block_size(BLOCK_DIM);

  HPLMXP_identity_kernel<<<grid_size, block_size, 0, computeStream>>>(
      M, A, LDA);
  HIP_CHECK(hipGetLastError());
}

template void HPLMXP_identity(const int m, double* A, const int lda);

template void HPLMXP_identity(const int m, float* A, const int lda);

template void HPLMXP_identity(const int m, __half* A, const int lda);
