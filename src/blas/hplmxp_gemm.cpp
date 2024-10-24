
#include "hplmxp.hpp"

template <>
void HPLMXP_gemmNT(const int     m,
                   const int     n,
                   const int     k,
                   const double  alpha,
                   const double* a,
                   const int     lda,
                   const double* b,
                   const int     ldb,
                   const double  beta,
                         double* c,
                   const int     ldc) {
  ROCBLAS_CHECK(rocblas_gemm_ex(blas_hdl,
                                rocblas_operation_none,
                                rocblas_operation_transpose,
                                m,
                                n,
                                k,
                                &alpha,
                                a,
                                rocblas_datatype_f64_r,
                                lda,
                                b,
                                rocblas_datatype_f64_r,
                                ldb,
                                &beta,
                                c,
                                rocblas_datatype_f64_r,
                                ldc,
                                c,
                                rocblas_datatype_f64_r,
                                ldc,
                                rocblas_datatype_f64_r,
                                rocblas_gemm_algo_standard,
                                0,
                                rocblas_gemm_flags_none));
}

template <>
void HPLMXP_gemmNT(const int    m,
                   const int    n,
                   const int    k,
                   const float  alpha,
                   const float* a,
                   const int    lda,
                   const float* b,
                   const int    ldb,
                   const float  beta,
                         float* c,
                   const int    ldc) {
  ROCBLAS_CHECK(rocblas_gemm_ex(blas_hdl,
                                rocblas_operation_none,
                                rocblas_operation_transpose,
                                m,
                                n,
                                k,
                                &alpha,
                                a,
                                rocblas_datatype_f32_r,
                                lda,
                                b,
                                rocblas_datatype_f32_r,
                                ldb,
                                &beta,
                                c,
                                rocblas_datatype_f32_r,
                                ldc,
                                c,
                                rocblas_datatype_f32_r,
                                ldc,
                                rocblas_datatype_f32_r,
                                rocblas_gemm_algo_standard,
                                0,
                                rocblas_gemm_flags_none));
}

template <>
void HPLMXP_gemmNT(const int     m,
                   const int     n,
                   const int     k,
                   const float   alpha,
                   const __half* a,
                   const int     lda,
                   const __half* b,
                   const int     ldb,
                   const float   beta,
                         float*  c,
                   const int     ldc) {
  ROCBLAS_CHECK(rocblas_gemm_ex(blas_hdl,
                                rocblas_operation_none,
                                rocblas_operation_transpose,
                                m,
                                n,
                                k,
                                &alpha,
                                a,
                                rocblas_datatype_f16_r,
                                lda,
                                b,
                                rocblas_datatype_f16_r,
                                ldb,
                                &beta,
                                c,
                                rocblas_datatype_f32_r,
                                ldc,
                                c,
                                rocblas_datatype_f32_r,
                                ldc,
                                rocblas_datatype_f32_r,
                                rocblas_gemm_algo_standard,
                                0,
                                rocblas_gemm_flags_none));
}

template <>
void HPLMXP_gemmNT(const int     m,
                   const int     n,
                   const int     k,
                   const float   alpha,
                   const __half* a,
                   const int     lda,
                   const __half* b,
                   const int     ldb,
                   const float   beta,
                         __half* c,
                   const int     ldc) {

  // ROCBLAS_CHECK(rocblas_gemm_ex(blas_hdl,
  //                               rocblas_operation_none,
  //                               rocblas_operation_transpose,
  //                               m,
  //                               n,
  //                               k,
  //                               &alpha,
  //                               a,
  //                               rocblas_datatype_f16_r,
  //                               lda,
  //                               b,
  //                               rocblas_datatype_f16_r,
  //                               ldb,
  //                               &beta,
  //                               c,
  //                               rocblas_datatype_f16_r,
  //                               ldc,
  //                               c,
  //                               rocblas_datatype_f16_r,
  //                               ldc,
  //                               rocblas_datatype_f32_r,
  //                               rocblas_gemm_algo_standard,
  //                               0,
  //                               rocblas_gemm_flags_none));

  if (m==0 || n==0 || k==0) return;

  int64_t M = m;
  int64_t N = n;
  int64_t K = k;
  int64_t LDA = lda;
  int64_t LDB = ldb;
  int64_t LDC = ldc;

  hipDataType a_type = HIP_R_16F;
  hipDataType b_type = HIP_R_16F;
  hipDataType c_type = HIP_R_16F;
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(a_layout, HIPBLASLT_MATRIX_LAYOUT_ROWS, &M, sizeof(M)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(a_layout, HIPBLASLT_MATRIX_LAYOUT_COLS, &K, sizeof(K)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(a_layout, HIPBLASLT_MATRIX_LAYOUT_LD,   &LDA, sizeof(LDA)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(a_layout, HIPBLASLT_MATRIX_LAYOUT_TYPE, &a_type, sizeof(a_type)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(b_layout, HIPBLASLT_MATRIX_LAYOUT_ROWS, &N, sizeof(N)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(b_layout, HIPBLASLT_MATRIX_LAYOUT_COLS, &K, sizeof(K)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(b_layout, HIPBLASLT_MATRIX_LAYOUT_LD,   &LDB, sizeof(LDA)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(b_layout, HIPBLASLT_MATRIX_LAYOUT_TYPE, &b_type, sizeof(b_type)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(c_layout, HIPBLASLT_MATRIX_LAYOUT_ROWS, &M, sizeof(M)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(c_layout, HIPBLASLT_MATRIX_LAYOUT_COLS, &N, sizeof(N)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(c_layout, HIPBLASLT_MATRIX_LAYOUT_LD,   &LDC, sizeof(LDC)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(c_layout, HIPBLASLT_MATRIX_LAYOUT_TYPE, &c_type, sizeof(c_type)));

  hipblasOperation_t op_a = HIPBLAS_OP_N;
  hipblasOperation_t op_b = HIPBLAS_OP_T;
  HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(matmul32, HIPBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
  HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(matmul32, HIPBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)));
  HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(matmul32, HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT, &a_type, sizeof(a_type)));
  HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(matmul32, HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT, &b_type, sizeof(b_type)));

  int returnedAlgoCount = 0;
  HIPBLAS_CHECK(hipblasLtMatmulAlgoGetHeuristic(hipblaslt_handle,
                                               matmul32,
                                               a_layout,
                                               b_layout,
                                               c_layout,
                                               c_layout,
                                               pref,
                                               1 /*requested_solutions*/,
                                               &heuristicResult,
                                               &returnedAlgoCount));

  if(returnedAlgoCount == 0) {
    fprintf(stderr, "Error: hipblasLtMatmulAlgoGetHeuristic returned no algo.\n");
    exit(-1);
  }

  HIPBLAS_CHECK(hipblasLtMatmul(hipblaslt_handle,
                               matmul32,
                               &alpha,
                               a,
                               a_layout,
                               b,
                               b_layout,
                               &beta,
                               c,
                               c_layout,
                               c,
                               c_layout,
                               &heuristicResult.algo,
                               NULL /*workspace*/,
                               0 /*workspace_size*/,
                               computeStream));

}

template <>
void HPLMXP_gemmNT(const int     m,
                   const int     n,
                   const int     k,
                   const float   alpha,
                   const hipblaslt_f8_fnuz* a,
                   const int     lda,
                   const hipblaslt_f8_fnuz* b,
                   const int     ldb,
                   const float   beta,
                         float*  c,
                   const int     ldc) {

  if (m==0 || n==0 || k==0) return;

  int64_t M = m;
  int64_t N = n;
  int64_t K = k;
  int64_t LDA = lda;
  int64_t LDB = ldb;
  int64_t LDC = ldc;

  hipDataType a_type = HIP_R_8F_E4M3_FNUZ;
  hipDataType b_type = HIP_R_8F_E4M3_FNUZ;
  hipDataType c_type = HIP_R_32F;
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(a_layout, HIPBLASLT_MATRIX_LAYOUT_ROWS, &M, sizeof(M)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(a_layout, HIPBLASLT_MATRIX_LAYOUT_COLS, &K, sizeof(K)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(a_layout, HIPBLASLT_MATRIX_LAYOUT_LD,   &LDA, sizeof(LDA)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(a_layout, HIPBLASLT_MATRIX_LAYOUT_TYPE, &a_type, sizeof(a_type)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(b_layout, HIPBLASLT_MATRIX_LAYOUT_ROWS, &N, sizeof(N)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(b_layout, HIPBLASLT_MATRIX_LAYOUT_COLS, &K, sizeof(K)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(b_layout, HIPBLASLT_MATRIX_LAYOUT_LD,   &LDB, sizeof(LDA)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(b_layout, HIPBLASLT_MATRIX_LAYOUT_TYPE, &b_type, sizeof(b_type)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(c_layout, HIPBLASLT_MATRIX_LAYOUT_ROWS, &M, sizeof(M)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(c_layout, HIPBLASLT_MATRIX_LAYOUT_COLS, &N, sizeof(N)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(c_layout, HIPBLASLT_MATRIX_LAYOUT_LD,   &LDC, sizeof(LDC)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(c_layout, HIPBLASLT_MATRIX_LAYOUT_TYPE, &c_type, sizeof(c_type)));

  hipblasOperation_t op_a = HIPBLAS_OP_N;
  hipblasOperation_t op_b = HIPBLAS_OP_T;
  HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(matmul32, HIPBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
  HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(matmul32, HIPBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)));
  HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(matmul32, HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT, &a_type, sizeof(a_type)));
  HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(matmul32, HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT, &b_type, sizeof(b_type)));

  int returnedAlgoCount = 0;
  HIPBLAS_CHECK(hipblasLtMatmulAlgoGetHeuristic(hipblaslt_handle,
                                               matmul32,
                                               a_layout,
                                               b_layout,
                                               c_layout,
                                               c_layout,
                                               pref,
                                               1 /*requested_solutions*/,
                                               &heuristicResult,
                                               &returnedAlgoCount));

  if(returnedAlgoCount == 0) {
    fprintf(stderr, "Error: hipblasLtMatmulAlgoGetHeuristic returned no algo.\n");
    exit(-1);
  }

  HIPBLAS_CHECK(hipblasLtMatmul(hipblaslt_handle,
                               matmul32,
                               &alpha,
                               a,
                               a_layout,
                               b,
                               b_layout,
                               &beta,
                               c,
                               c_layout,
                               c,
                               c_layout,
                               &heuristicResult.algo,
                               NULL /*workspace*/,
                               0 /*workspace_size*/,
                               computeStream));

}

template <>
void HPLMXP_gemmNT(const int     m,
                   const int     n,
                   const int     k,
                   const float   alpha,
                   const hipblaslt_f8_fnuz* a,
                   const int     lda,
                   const hipblaslt_f8_fnuz* b,
                   const int     ldb,
                   const float   beta,
                         hipblaslt_f8_fnuz* c,
                   const int     ldc) {

  if (m==0 || n==0 || k==0) return;

  int64_t M = m;
  int64_t N = n;
  int64_t K = k;
  int64_t LDA = lda;
  int64_t LDB = ldb;
  int64_t LDC = ldc;

  hipDataType a_type = HIP_R_8F_E4M3_FNUZ;
  hipDataType b_type = HIP_R_8F_E4M3_FNUZ;
  hipDataType c_type = HIP_R_8F_E4M3_FNUZ;
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(a_layout, HIPBLASLT_MATRIX_LAYOUT_ROWS, &M, sizeof(M)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(a_layout, HIPBLASLT_MATRIX_LAYOUT_COLS, &K, sizeof(K)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(a_layout, HIPBLASLT_MATRIX_LAYOUT_LD,   &LDA, sizeof(LDA)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(a_layout, HIPBLASLT_MATRIX_LAYOUT_TYPE, &a_type, sizeof(a_type)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(b_layout, HIPBLASLT_MATRIX_LAYOUT_ROWS, &N, sizeof(N)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(b_layout, HIPBLASLT_MATRIX_LAYOUT_COLS, &K, sizeof(K)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(b_layout, HIPBLASLT_MATRIX_LAYOUT_LD,   &LDB, sizeof(LDA)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(b_layout, HIPBLASLT_MATRIX_LAYOUT_TYPE, &b_type, sizeof(b_type)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(c_layout, HIPBLASLT_MATRIX_LAYOUT_ROWS, &M, sizeof(M)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(c_layout, HIPBLASLT_MATRIX_LAYOUT_COLS, &N, sizeof(N)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(c_layout, HIPBLASLT_MATRIX_LAYOUT_LD,   &LDC, sizeof(LDC)));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(c_layout, HIPBLASLT_MATRIX_LAYOUT_TYPE, &c_type, sizeof(c_type)));

  hipblasOperation_t op_a = HIPBLAS_OP_N;
  hipblasOperation_t op_b = HIPBLAS_OP_T;
  HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(matmul32, HIPBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
  HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(matmul32, HIPBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)));
  HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(matmul32, HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT, &a_type, sizeof(a_type)));
  HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(matmul32, HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT, &b_type, sizeof(b_type)));

  int returnedAlgoCount = 0;
  HIPBLAS_CHECK(hipblasLtMatmulAlgoGetHeuristic(hipblaslt_handle,
                                               matmul32,
                                               a_layout,
                                               b_layout,
                                               c_layout,
                                               c_layout,
                                               pref,
                                               1 /*requested_solutions*/,
                                               &heuristicResult,
                                               &returnedAlgoCount));

  if(returnedAlgoCount == 0) {
    fprintf(stderr, "Error: hipblasLtMatmulAlgoGetHeuristic returned no algo.\n");
    exit(-1);
  }

  HIPBLAS_CHECK(hipblasLtMatmul(hipblaslt_handle,
                               matmul32,
                               &alpha,
                               a,
                               a_layout,
                               b,
                               b_layout,
                               &beta,
                               c,
                               c_layout,
                               c,
                               c_layout,
                               &heuristicResult.algo,
                               NULL /*workspace*/,
                               0 /*workspace_size*/,
                               computeStream));

}
