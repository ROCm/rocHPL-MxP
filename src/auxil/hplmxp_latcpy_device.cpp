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

template <typename T,
          typename U,
          int BLOCK_DIM_X,
          int BLOCK_DIM_Y,
          int TILE_DIM_X,
          int TILE_DIM_Y>
__launch_bounds__(BLOCK_DIM_X* BLOCK_DIM_Y) __global__
    void HPLMXP_latcpy_kernel(const int M,
                              const int N,
                              const T* __restrict__ A,
                              const int LDA,
                              U* __restrict__ B,
                              const int LDB) {

  __shared__ U s_tile[TILE_DIM_X][TILE_DIM_Y];

  const bool complete_block =
      (M / (TILE_DIM_X) != blockIdx.y) && (N / (TILE_DIM_Y) != blockIdx.x);

  if(complete_block) {

    int si = (TILE_DIM_Y / BLOCK_DIM_X) * threadIdx.x;
    int sj = (TILE_DIM_X / BLOCK_DIM_Y) * threadIdx.y;
    int I  = sj + blockIdx.y * TILE_DIM_X;
    int J  = si + blockIdx.x * TILE_DIM_Y;

    const T* __restrict__ Aj = A + J + I * static_cast<size_t>(LDA);

    for(int j = 0; j < (TILE_DIM_X / BLOCK_DIM_Y); ++j) {
      for(int i = 0; i < (TILE_DIM_Y / BLOCK_DIM_X); ++i) {
        s_tile[sj + j][si + i] = Aj[i + j * LDA];
      }
    }

    si = (TILE_DIM_X / BLOCK_DIM_X) * threadIdx.x;
    sj = (TILE_DIM_Y / BLOCK_DIM_Y) * threadIdx.y;
    I  = si + blockIdx.y * TILE_DIM_X;
    J  = sj + blockIdx.x * TILE_DIM_Y;

    U* __restrict__ Bj = B + I + J * static_cast<size_t>(LDB);

    __syncthreads();

    for(int j = 0; j < (TILE_DIM_Y / BLOCK_DIM_Y); ++j) {
      for(int i = 0; i < (TILE_DIM_X / BLOCK_DIM_X); ++i) {
        Bj[i + j * LDB] = s_tile[si + i][sj + j];
      }
    }

  } else {

    int si = (TILE_DIM_Y / BLOCK_DIM_X) * threadIdx.x;
    int sj = (TILE_DIM_X / BLOCK_DIM_Y) * threadIdx.y;
    int I  = sj + blockIdx.y * TILE_DIM_X;
    int J  = si + blockIdx.x * TILE_DIM_Y;

    const T* __restrict__ Aj = A + J + I * static_cast<size_t>(LDA);

    for(int j = 0; j < (TILE_DIM_X / BLOCK_DIM_Y); ++j) {
      for(int i = 0; i < (TILE_DIM_Y / BLOCK_DIM_X); ++i) {
        if((J + i < N) && (I + j < M)) s_tile[sj + j][si + i] = Aj[i + j * LDA];
      }
    }

    si = (TILE_DIM_X / BLOCK_DIM_X) * threadIdx.x;
    sj = (TILE_DIM_Y / BLOCK_DIM_Y) * threadIdx.y;
    I  = si + blockIdx.y * TILE_DIM_X;
    J  = sj + blockIdx.x * TILE_DIM_Y;

    U* __restrict__ Bj = B + I + J * static_cast<size_t>(LDB);

    __syncthreads();

    for(int j = 0; j < (TILE_DIM_Y / BLOCK_DIM_Y); ++j) {
      for(int i = 0; i < (TILE_DIM_X / BLOCK_DIM_X); ++i) {
        if((I + i < M) && (J + j < N)) Bj[i + j * LDB] = s_tile[si + i][sj + j];
      }
    }
  }
}

union fp32or16x2 {
  float  fp32;
  __half fp16[2];
};

// Specialization for converting from float to __half
template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int TILE_DIM_X, int TILE_DIM_Y>
__launch_bounds__(BLOCK_DIM_X* BLOCK_DIM_Y) __global__
    void HPLMXP_latcpy_kernel_half(const int M,
                                   const int N,
                                   const float* __restrict__ A,
                                   const int LDA,
                                   __half* __restrict__ B,
                                   const int LDB) {

  __shared__ __half s_tile[TILE_DIM_X][TILE_DIM_Y];

  const bool complete_block =
      (M / (TILE_DIM_X) != blockIdx.y) && (N / (TILE_DIM_Y) != blockIdx.x);

  if(complete_block) {

    int si = (TILE_DIM_Y / BLOCK_DIM_X) * threadIdx.x;
    int sj = (TILE_DIM_X / BLOCK_DIM_Y) * threadIdx.y;
    int I  = sj + blockIdx.y * TILE_DIM_X;
    int J  = si + blockIdx.x * TILE_DIM_Y;

    const float* __restrict__ Aj = A + J + I * static_cast<size_t>(LDA);

    for(int j = 0; j < (TILE_DIM_X / BLOCK_DIM_Y); ++j) {
      for(int i = 0; i < (TILE_DIM_Y / BLOCK_DIM_X); ++i) {
        s_tile[sj + j][si + i] = Aj[i + j * LDA];
      }
    }

    si = (TILE_DIM_X / BLOCK_DIM_X) * threadIdx.x;
    sj = (TILE_DIM_Y / BLOCK_DIM_Y) * threadIdx.y;
    I  = si + blockIdx.y * TILE_DIM_X;
    J  = sj + blockIdx.x * TILE_DIM_Y;

    __half* __restrict__ Bj = B + I + J * static_cast<size_t>(LDB);

    __syncthreads();

    fp32or16x2 out[(TILE_DIM_Y / BLOCK_DIM_Y)][(TILE_DIM_X / BLOCK_DIM_X) / 2];

    for(int j = 0; j < (TILE_DIM_Y / BLOCK_DIM_Y); ++j) {
      for(int i = 0; i < (TILE_DIM_X / BLOCK_DIM_X) / 2; ++i) {
        out[j][i].fp16[0] = s_tile[si + 2 * i + 0][sj + j];
        out[j][i].fp16[1] = s_tile[si + 2 * i + 1][sj + j];
      }
    }

    // Make the compiler emit dwordx2 load/stores
    for(int j = 0; j < (TILE_DIM_Y / BLOCK_DIM_Y); ++j) {
      for(int i = 0; i < (TILE_DIM_X / BLOCK_DIM_X) / 2; ++i) {
        ((float*)(Bj + 2 * i + j * LDB))[0] = out[j][i].fp32;
      }
    }

  } else {

    int si = (TILE_DIM_Y / BLOCK_DIM_X) * threadIdx.x;
    int sj = (TILE_DIM_X / BLOCK_DIM_Y) * threadIdx.y;
    int I  = sj + blockIdx.y * TILE_DIM_X;
    int J  = si + blockIdx.x * TILE_DIM_Y;

    const float* __restrict__ Aj = A + J + I * static_cast<size_t>(LDA);

    for(int j = 0; j < (TILE_DIM_X / BLOCK_DIM_Y); ++j) {
      for(int i = 0; i < (TILE_DIM_Y / BLOCK_DIM_X); ++i) {
        if((J + i < N) && (I + j < M)) s_tile[sj + j][si + i] = Aj[i + j * LDA];
      }
    }

    si = (TILE_DIM_X / BLOCK_DIM_X) * threadIdx.x;
    sj = (TILE_DIM_Y / BLOCK_DIM_Y) * threadIdx.y;
    I  = si + blockIdx.y * TILE_DIM_X;
    J  = sj + blockIdx.x * TILE_DIM_Y;

    __half* __restrict__ Bj = B + I + J * static_cast<size_t>(LDB);

    __syncthreads();

    for(int j = 0; j < (TILE_DIM_Y / BLOCK_DIM_Y); ++j) {
      for(int i = 0; i < (TILE_DIM_X / BLOCK_DIM_X); ++i) {
        if((I + i < M) && (J + j < N)) Bj[i + j * LDB] = s_tile[si + i][sj + j];
      }
    }
  }
}

union fp32or8x4 {
  float  fp32;
  hipblaslt_f8_fnuz fp8[8];
};

// Specialization for converting from float to hipblaslt_f8_fnuz
template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int TILE_DIM_X, int TILE_DIM_Y>
__launch_bounds__(BLOCK_DIM_X* BLOCK_DIM_Y) __global__
    void HPLMXP_latcpy_kernel_f8(const int M,
                                   const int N,
                                   const float* __restrict__ A,
                                   const int LDA,
                                   hipblaslt_f8_fnuz* __restrict__ B,
                                   const int LDB) {

  __shared__ hipblaslt_f8_fnuz s_tile[TILE_DIM_X][TILE_DIM_Y];

  const bool complete_block =
      (M / (TILE_DIM_X) != blockIdx.y) && (N / (TILE_DIM_Y) != blockIdx.x);

  if(complete_block) {

    int si = (TILE_DIM_Y / BLOCK_DIM_X) * threadIdx.x;
    int sj = (TILE_DIM_X / BLOCK_DIM_Y) * threadIdx.y;
    int I  = sj + blockIdx.y * TILE_DIM_X;
    int J  = si + blockIdx.x * TILE_DIM_Y;

    const float* __restrict__ Aj = A + J + I * static_cast<size_t>(LDA);

    for(int j = 0; j < (TILE_DIM_X / BLOCK_DIM_Y); ++j) {
      for(int i = 0; i < (TILE_DIM_Y / BLOCK_DIM_X); ++i) {
        s_tile[sj + j][si + i] = hipblaslt_f8_fnuz{Aj[i + j * LDA]};
      }
    }

    si = (TILE_DIM_X / BLOCK_DIM_X) * threadIdx.x;
    sj = (TILE_DIM_Y / BLOCK_DIM_Y) * threadIdx.y;
    I  = si + blockIdx.y * TILE_DIM_X;
    J  = sj + blockIdx.x * TILE_DIM_Y;

    hipblaslt_f8_fnuz* __restrict__ Bj = B + I + J * static_cast<size_t>(LDB);

    __syncthreads();

    fp32or8x4 out[(TILE_DIM_Y / BLOCK_DIM_Y)][(TILE_DIM_X / BLOCK_DIM_X) / 4];

    for(int j = 0; j < (TILE_DIM_Y / BLOCK_DIM_Y); ++j) {
      for(int i = 0; i < (TILE_DIM_X / BLOCK_DIM_X) / 4; ++i) {
        out[j][i].fp8[0] = s_tile[si + 4 * i + 0][sj + j];
        out[j][i].fp8[1] = s_tile[si + 4 * i + 1][sj + j];
        out[j][i].fp8[2] = s_tile[si + 4 * i + 2][sj + j];
        out[j][i].fp8[3] = s_tile[si + 4 * i + 3][sj + j];
      }
    }

    // Make the compiler emit dwordx2 load/stores
    for(int j = 0; j < (TILE_DIM_Y / BLOCK_DIM_Y); ++j) {
      for(int i = 0; i < (TILE_DIM_X / BLOCK_DIM_X) / 4; ++i) {
        ((float*)(Bj + 4 * i + j * LDB))[0] = out[j][i].fp32;
      }
    }

  } else {

    int si = (TILE_DIM_Y / BLOCK_DIM_X) * threadIdx.x;
    int sj = (TILE_DIM_X / BLOCK_DIM_Y) * threadIdx.y;
    int I  = sj + blockIdx.y * TILE_DIM_X;
    int J  = si + blockIdx.x * TILE_DIM_Y;

    const float* __restrict__ Aj = A + J + I * static_cast<size_t>(LDA);

    for(int j = 0; j < (TILE_DIM_X / BLOCK_DIM_Y); ++j) {
      for(int i = 0; i < (TILE_DIM_Y / BLOCK_DIM_X); ++i) {
        if((J + i < N) && (I + j < M)) s_tile[sj + j][si + i] = hipblaslt_f8_fnuz{Aj[i + j * LDA]};
      }
    }

    si = (TILE_DIM_X / BLOCK_DIM_X) * threadIdx.x;
    sj = (TILE_DIM_Y / BLOCK_DIM_Y) * threadIdx.y;
    I  = si + blockIdx.y * TILE_DIM_X;
    J  = sj + blockIdx.x * TILE_DIM_Y;

    hipblaslt_f8_fnuz* __restrict__ Bj = B + I + J * static_cast<size_t>(LDB);

    __syncthreads();

    for(int j = 0; j < (TILE_DIM_Y / BLOCK_DIM_Y); ++j) {
      for(int i = 0; i < (TILE_DIM_X / BLOCK_DIM_X); ++i) {
        if((I + i < M) && (J + j < N)) Bj[i + j * LDB] = s_tile[si + i][sj + j];
      }
    }
  }
}


template <typename T, typename U>
void HPLMXP_latcpy(const int M,
                   const int N,
                   const T*  A,
                   const int LDA,
                   U*        B,
                   const int LDB) {
  /*
   * Purpose
   * =======
   *
   * HPLMXP_latcpy copies the transpose of an array A into an array B.
   *
   *
   * Arguments
   * =========
   *
   * M       (local input)                 const int
   *         On entry,  M specifies the number of  rows of the array B and
   *         the number of columns of A. M must be at least zero.
   *
   * N       (local input)                 const int
   *         On entry,  N specifies the number of  rows of the array A and
   *         the number of columns of B. N must be at least zero.
   *
   * A       (local input)                 const T *
   *         On entry, A points to an array of dimension (LDA,M).
   *
   * LDA     (local input)                 const int
   *         On entry, LDA specifies the leading dimension of the array A.
   *         LDA must be at least MAX(1,N).
   *
   * B       (local output)                U *
   *         On entry, B points to an array of dimension (LDB,N). On exit,
   *         B is overwritten with the transpose of A.
   *
   * LDB     (local input)                 const int
   *         On entry, LDB specifies the leading dimension of the array B.
   *         LDB must be at least MAX(1,M).
   *
   * ---------------------------------------------------------------------
   */

  if((M <= 0) || (N <= 0)) return;

  constexpr int BLOCK_DIM_X = 32;
  constexpr int BLOCK_DIM_Y = 8;
  constexpr int TILE_DIM_X  = 64;
  constexpr int TILE_DIM_Y  = 64;

  dim3 grid_size((N + TILE_DIM_Y - 1) / (TILE_DIM_Y),
                 (M + TILE_DIM_X - 1) / (TILE_DIM_X));
  dim3 block_size(BLOCK_DIM_X, BLOCK_DIM_Y);

  HPLMXP_latcpy_kernel<T, U, BLOCK_DIM_X, BLOCK_DIM_Y, TILE_DIM_X, TILE_DIM_Y>
      <<<grid_size, block_size, 0, computeStream>>>(M, N, A, LDA, B, LDB);
  HIP_CHECK(hipGetLastError());
}

// Specialization for converting from float to __half
template <>
void HPLMXP_latcpy(const int    M,
                   const int    N,
                   const float* A,
                   const int    LDA,
                   __half*      B,
                   const int    LDB) {

  if((M <= 0) || (N <= 0)) return;

  constexpr int BLOCK_DIM_X = 32;
  constexpr int BLOCK_DIM_Y = 8;
  constexpr int TILE_DIM_X  = 64;
  constexpr int TILE_DIM_Y  = 64;

  dim3 grid_size((N + TILE_DIM_Y - 1) / (TILE_DIM_Y),
                 (M + TILE_DIM_X - 1) / (TILE_DIM_X));
  dim3 block_size(BLOCK_DIM_X, BLOCK_DIM_Y);

  HPLMXP_latcpy_kernel_half<BLOCK_DIM_X, BLOCK_DIM_Y, TILE_DIM_X, TILE_DIM_Y>
      <<<grid_size, block_size, 0, computeStream>>>(M, N, A, LDA, B, LDB);
  HIP_CHECK(hipGetLastError());
}

// Specialization for converting from float to hipblaslt_f8_fnuz
template <>
void HPLMXP_latcpy(const int    M,
                   const int    N,
                   const float* A,
                   const int    LDA,
                   hipblaslt_f8_fnuz*      B,
                   const int    LDB) {

  if((M <= 0) || (N <= 0)) return;

  constexpr int BLOCK_DIM_X = 16;
  constexpr int BLOCK_DIM_Y = 8;
  constexpr int TILE_DIM_X  = 64;
  constexpr int TILE_DIM_Y  = 64;

  dim3 grid_size((N + TILE_DIM_Y - 1) / (TILE_DIM_Y),
                 (M + TILE_DIM_X - 1) / (TILE_DIM_X));
  dim3 block_size(BLOCK_DIM_X, BLOCK_DIM_Y);

  HPLMXP_latcpy_kernel_f8<BLOCK_DIM_X, BLOCK_DIM_Y, TILE_DIM_X, TILE_DIM_Y>
      <<<grid_size, block_size, 0, computeStream>>>(M, N, A, LDA, B, LDB);
  HIP_CHECK(hipGetLastError());
}

template void HPLMXP_latcpy(const int     m,
                            const int     n,
                            const double* A,
                            const int     lda,
                            double*       B,
                            const int     ldb);

template void HPLMXP_latcpy(const int    m,
                            const int    n,
                            const float* A,
                            const int    lda,
                            float*       B,
                            const int    ldb);

template void HPLMXP_latcpy(const int     m,
                            const int     n,
                            const __half* A,
                            const int     lda,
                            __half*       B,
                            const int     ldb);

template void HPLMXP_latcpy(const int     m,
                            const int     n,
                            const hipblaslt_f8_fnuz* A,
                            const int     lda,
                            hipblaslt_f8_fnuz*       B,
                            const int     ldb);
