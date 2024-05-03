#include "hplmxp.hpp"
#include <hip/hip_runtime.h>

/* config */
#define GEMV_USE_MED_KNL 1

/* gemv kernel for mixed types */
template <int DIM_X, int DIM_Y, typename T, typename U>
static __launch_bounds__(DIM_X* DIM_Y) __global__
    void gemv_kernel(const int m,
                     const int n,
                     const U   alpha,
                     const T* __restrict__ A,
                     const int lda,
                     const U* __restrict__ x,
                     const int incx,
                     const U   beta,
                     U* __restrict__ y,
                     const int incy) {
  const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;

  if(!alpha) {
    if(thread_id < DIM_X * 4) {
      const int ind = blockIdx.x * DIM_X * 4 + thread_id;

      if(ind < m) y[ind * incy] = beta ? beta * y[ind * incy] : 0;
    }
    return;
  }

  // threads are all configurated locally
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  int ind;

  __shared__ U sdata[DIM_X * 4 * DIM_Y];

  U res_A[4];
  U res_x[4];

  res_A[0] = res_A[1] = res_A[2] = res_A[3] = U{0};

  ind = blockIdx.x * DIM_X * 4 + tx;

  const int n_tail = n % (4 * DIM_Y);
  int       col;

  for(col = ty * 4; col < (n - n_tail); col += 4 * DIM_Y) {
    res_x[0] = x[(col + 0) * incx];
    res_x[1] = x[(col + 1) * incx];
    res_x[2] = x[(col + 2) * incx];
    res_x[3] = x[(col + 3) * incx];

    if(ind < m) {
      res_A[0] += static_cast<U>(A[ind + (col + 0) * lda]) * res_x[0];
      res_A[0] += static_cast<U>(A[ind + (col + 1) * lda]) * res_x[1];
      res_A[0] += static_cast<U>(A[ind + (col + 2) * lda]) * res_x[2];
      res_A[0] += static_cast<U>(A[ind + (col + 3) * lda]) * res_x[3];

      if(ind + DIM_X < m) {
        res_A[1] += static_cast<U>(A[ind + DIM_X + (col + 0) * lda]) * res_x[0];
        res_A[1] += static_cast<U>(A[ind + DIM_X + (col + 1) * lda]) * res_x[1];
        res_A[1] += static_cast<U>(A[ind + DIM_X + (col + 2) * lda]) * res_x[2];
        res_A[1] += static_cast<U>(A[ind + DIM_X + (col + 3) * lda]) * res_x[3];

        if(ind + 2 * DIM_X < m) {
          res_A[2] +=
              static_cast<U>(A[ind + 2 * DIM_X + (col + 0) * lda]) * res_x[0];
          res_A[2] +=
              static_cast<U>(A[ind + 2 * DIM_X + (col + 1) * lda]) * res_x[1];
          res_A[2] +=
              static_cast<U>(A[ind + 2 * DIM_X + (col + 2) * lda]) * res_x[2];
          res_A[2] +=
              static_cast<U>(A[ind + 2 * DIM_X + (col + 3) * lda]) * res_x[3];

          if(ind + 3 * DIM_X < m) {
            res_A[3] +=
                static_cast<U>(A[ind + 3 * DIM_X + (col + 0) * lda]) * res_x[0];
            res_A[3] +=
                static_cast<U>(A[ind + 3 * DIM_X + (col + 1) * lda]) * res_x[1];
            res_A[3] +=
                static_cast<U>(A[ind + 3 * DIM_X + (col + 2) * lda]) * res_x[2];
            res_A[3] +=
                static_cast<U>(A[ind + 3 * DIM_X + (col + 3) * lda]) * res_x[3];
          }
        }
      }
    }
  }

  // if n is not multiple of (DIM_Y * 4)
  if(n_tail > 0) {
    res_x[0] = res_x[1] = res_x[2] = res_x[3] = U{0};

    if(col + 0 < n) {
      res_x[0] = x[(col + 0) * incx];

      if(col + 1 < n) {
        res_x[1] = x[(col + 1) * incx];

        if(col + 2 < n) {
          res_x[2] = x[(col + 2) * incx];

          if(col + 3 < n) res_x[3] = x[(col + 3) * incx];
        }
      }
    }

    if(ind < m) {
      res_A[0] +=
          static_cast<U>(A[ind + (col + 0) * lda * (col + 0 < n)]) * res_x[0];
      res_A[0] +=
          static_cast<U>(A[ind + (col + 1) * lda * (col + 1 < n)]) * res_x[1];
      res_A[0] +=
          static_cast<U>(A[ind + (col + 2) * lda * (col + 2 < n)]) * res_x[2];
      res_A[0] +=
          static_cast<U>(A[ind + (col + 3) * lda * (col + 3 < n)]) * res_x[3];

      if(ind + DIM_X < m) {
        res_A[1] +=
            static_cast<U>(A[ind + DIM_X + (col + 0) * lda * (col + 0 < n)]) *
            res_x[0];
        res_A[1] +=
            static_cast<U>(A[ind + DIM_X + (col + 1) * lda * (col + 1 < n)]) *
            res_x[1];
        res_A[1] +=
            static_cast<U>(A[ind + DIM_X + (col + 2) * lda * (col + 2 < n)]) *
            res_x[2];
        res_A[1] +=
            static_cast<U>(A[ind + DIM_X + (col + 3) * lda * (col + 3 < n)]) *
            res_x[3];

        if(ind + 2 * DIM_X < m) {
          res_A[2] +=
              static_cast<U>(
                  A[ind + 2 * DIM_X + (col + 0) * lda * (col + 0 < n)]) *
              res_x[0];
          res_A[2] +=
              static_cast<U>(
                  A[ind + 2 * DIM_X + (col + 1) * lda * (col + 1 < n)]) *
              res_x[1];
          res_A[2] +=
              static_cast<U>(
                  A[ind + 2 * DIM_X + (col + 2) * lda * (col + 2 < n)]) *
              res_x[2];
          res_A[2] +=
              static_cast<U>(
                  A[ind + 2 * DIM_X + (col + 3) * lda * (col + 3 < n)]) *
              res_x[3];

          if(ind + 3 * DIM_X < m) {
            res_A[3] +=
                static_cast<U>(
                    A[ind + 3 * DIM_X + (col + 0) * lda * (col + 0 < n)]) *
                res_x[0];
            res_A[3] +=
                static_cast<U>(
                    A[ind + 3 * DIM_X + (col + 1) * lda * (col + 1 < n)]) *
                res_x[1];
            res_A[3] +=
                static_cast<U>(
                    A[ind + 3 * DIM_X + (col + 2) * lda * (col + 2 < n)]) *
                res_x[2];
            res_A[3] +=
                static_cast<U>(
                    A[ind + 3 * DIM_X + (col + 3) * lda * (col + 3 < n)]) *
                res_x[3];
          }
        }
      }
    }
  }

  sdata[tx + ty * DIM_X * 4]             = res_A[0];
  sdata[tx + DIM_X + ty * DIM_X * 4]     = res_A[1];
  sdata[tx + 2 * DIM_X + ty * DIM_X * 4] = res_A[2];
  sdata[tx + 3 * DIM_X + ty * DIM_X * 4] = res_A[3];

  __syncthreads();

  if(thread_id < DIM_X * 4) {
    for(int i = 1; i < DIM_Y; i++) {
      sdata[thread_id] += sdata[thread_id + DIM_X * 4 * i];
    }

    ind = blockIdx.x * DIM_X * 4 + thread_id;

    if(ind < m)
      y[ind * incy] = beta ? alpha * sdata[thread_id] + beta * y[ind * incy]
                           : alpha * sdata[thread_id];
  }
}

template <typename T, typename U>
void HPLMXP_gemv(const int m,
                 const int n,
                 const T   alpha,
                 const U*  A,
                 const int lda,
                 const T*  x,
                 const T   beta,
                 T*        y) {

#if GEMV_USE_MED_KNL
  static constexpr int GEMVN_DIM_X = 32;
  static constexpr int GEMVN_DIM_Y = 16;
#else
  static constexpr int GEMVN_DIM_X = 64;
  static constexpr int GEMVN_DIM_Y = 16;
#endif

  const int blocks = (m + GEMVN_DIM_X * 4 - 1) / (GEMVN_DIM_X * 4);
  dim3      gemvn_grid(blocks);
  dim3      gemvn_threads(GEMVN_DIM_X, GEMVN_DIM_Y);

  gemv_kernel<GEMVN_DIM_X, GEMVN_DIM_Y, U, T>
      <<<gemvn_grid, gemvn_threads, 0, computeStream>>>(
          m, n, alpha, A, lda, x, 1, beta, y, 1);
  HIP_CHECK(hipGetLastError());
}

template void HPLMXP_gemv(const int     m,
                          const int     n,
                          const double  alpha,
                          const float*  A,
                          const int     lda,
                          const double* x,
                          const double  beta,
                          double*       y);

template void HPLMXP_gemv(const int     m,
                          const int     n,
                          const float   alpha,
                          const __half* A,
                          const int     lda,
                          const float*  x,
                          const float   beta,
                          float*        y);

template <>
void HPLMXP_gemv(const int     m,
                 const int     n,
                 const double  alpha,
                 const double* A,
                 const int     lda,
                 const double* x,
                 const double  beta,
                 double*       y) {
  ROCBLAS_CHECK(rocblas_dgemv(blas_hdl,
                              rocblas_operation_none,
                              m,
                              n,
                              &alpha,
                              A,
                              lda,
                              x,
                              1,
                              &beta,
                              y,
                              1));
}

template <>
void HPLMXP_gemv(const int    m,
                 const int    n,
                 const float  alpha,
                 const float* A,
                 const int    lda,
                 const float* x,
                 const float  beta,
                 float*       y) {
  ROCBLAS_CHECK(rocblas_sgemv(blas_hdl,
                              rocblas_operation_none,
                              m,
                              n,
                              &alpha,
                              A,
                              lda,
                              x,
                              1,
                              &beta,
                              y,
                              1));
}
