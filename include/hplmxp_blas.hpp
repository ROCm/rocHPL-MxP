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
#ifndef HPLMXP_BLAS_HPP
#define HPLMXP_BLAS_HPP

#include "rocblas/rocblas.h"
#include "rocsolver/rocsolver.h"
#include "hipblas/hipblas.h"
#include "hipblaslt/hipblaslt.h"

#define HIP_CHECK(val) hipCheck((val), #val, __FILE__, __LINE__)
inline void hipCheck(hipError_t        err,
                     const char* const func,
                     const char* const file,
                     const int         line) {
  if(err != hipSuccess) {
    fprintf(
        stderr,
        "Error: HIP runtime error in file %s, line %d, error code: %s, %s\n",
        file,
        line,
        hipGetErrorString(err),
        func);
    exit(err);
  }
}

#define ROCBLAS_CHECK(val) rocBLASCheck((val), #val, __FILE__, __LINE__)
inline void rocBLASCheck(rocblas_status    err,
                         const char* const func,
                         const char* const file,
                         const int         line) {
  if(err != rocblas_status_success) {
    fprintf(stderr,
            "Error: rocblas error in file %s, line %d, error code: %s, %s\n",
            file,
            line,
            rocblas_status_to_string(err),
            func);
    exit(err);
  }
}

#define HIPBLAS_CHECK(val) hipBLASCheck((val), #val, __FILE__, __LINE__)
inline void hipBLASCheck(hipblasStatus_t   err,
                         const char* const func,
                         const char* const file,
                         const int         line) {
  if(err != HIPBLAS_STATUS_SUCCESS) {
    fprintf(stderr,
            "Error: hipblas error in file %s, line %d, error code: %s, %s\n",
            file,
            line,
            hipblasStatusToString(err),
            func);
    exit(err);
  }
}

extern rocblas_handle    blas_hdl;
extern hipblasLtHandle_t hipblaslt_handle;
extern hipStream_t       computeStream;
extern rocblas_int*      blas_info;

extern hipblasLtMatrixLayout_t a_layout;
extern hipblasLtMatrixLayout_t b_layout;
extern hipblasLtMatrixLayout_t c_layout;

extern hipblasLtMatmulDesc_t            matmul64;
extern hipblasLtMatmulDesc_t            matmul32;
extern hipblasLtMatmulPreference_t      pref;
extern hipblasLtMatmulHeuristicResult_t heuristicResult;

extern hipEvent_t getrf, lbcast, ubcast;
extern hipEvent_t piv;
extern hipEvent_t DgemmStart, DgemmEnd;
extern hipEvent_t LgemmStart, LgemmEnd;
extern hipEvent_t UgemmStart, UgemmEnd;
extern hipEvent_t TgemmStart, TgemmEnd;

#define REDUCTION_SCRATCH_SIZE 512
extern fp64_t* reduction_scratch;
extern fp64_t* h_reduction_scratch;

template <typename T>
struct gemmTypes { using computeType = fp32_t; };

template <>
struct gemmTypes<fp64_t> { using computeType = fp64_t; };


template <typename T, typename U>
void HPLMXP_gemv(const int m,
                 const int n,
                 const T   alpha,
                 const U*  A,
                 const int lda,
                 const T*  x,
                 const T   beta,
                 T*        y);

template <typename T>
void HPLMXP_trsvU(const int m, const T* A, const int lda, T* x);

template <typename T>
void HPLMXP_trsvL(const int m, const T* A, const int lda, T* x);

template <typename T>
void HPLMXP_getrf(const int m, const int n, T* a, const int lda);

template <typename T>
void HPLMXP_trtriU(const int m, T* A, const int lda);

template <typename T>
void HPLMXP_trtriL(const int m, T* A, const int lda);

template <typename T>
void HPLMXP_trsmR(const int m,
                  const int n,
                  const T   alpha,
                  const T*  a,
                  const int lda,
                  T*        b,
                  const int ldb);

template <typename T>
void HPLMXP_trsmL(const int m,
                  const int n,
                  const T   alpha,
                  const T*  a,
                  const int lda,
                  T*        b,
                  const int ldb);

template <typename AB_type, typename C_type, typename compute_type>
void HPLMXP_gemmNT(const int            m,
                   const int            n,
                   const int            k,
                   const compute_type   alpha,
                   const AB_type*       a,
                   const int            lda,
                   const AB_type*       b,
                   const int            ldb,
                   const compute_type   beta,
                         C_type*        c,
                   const int            ldc);

#endif
