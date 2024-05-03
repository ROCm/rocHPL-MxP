
#include "hplmxp.hpp"

template <>
void HPLMXP_trsmR(const int     m,
                  const int     n,
                  const double  alpha,
                  const double* a,
                  const int     lda,
                  double*       b,
                  const int     ldb) {
  ROCBLAS_CHECK(rocblas_dtrsm(blas_hdl,
                              rocblas_side_right,
                              rocblas_fill_upper,
                              rocblas_operation_none,
                              rocblas_diagonal_non_unit,
                              m,
                              n,
                              &alpha,
                              a,
                              lda,
                              b,
                              ldb));
}

template <>
void HPLMXP_trsmR(const int    m,
                  const int    n,
                  const float  alpha,
                  const float* a,
                  const int    lda,
                  float*       b,
                  const int    ldb) {
  ROCBLAS_CHECK(rocblas_strsm(blas_hdl,
                              rocblas_side_right,
                              rocblas_fill_upper,
                              rocblas_operation_none,
                              rocblas_diagonal_non_unit,
                              m,
                              n,
                              &alpha,
                              a,
                              lda,
                              b,
                              ldb));
}

template <>
void HPLMXP_trsmL(const int     m,
                  const int     n,
                  const double  alpha,
                  const double* a,
                  const int     lda,
                  double*       b,
                  const int     ldb) {
  ROCBLAS_CHECK(rocblas_dtrsm(blas_hdl,
                              rocblas_side_left,
                              rocblas_fill_lower,
                              rocblas_operation_none,
                              rocblas_diagonal_unit,
                              m,
                              n,
                              &alpha,
                              a,
                              lda,
                              b,
                              ldb));
}

template <>
void HPLMXP_trsmL(const int    m,
                  const int    n,
                  const float  alpha,
                  const float* a,
                  const int    lda,
                  float*       b,
                  const int    ldb) {
  ROCBLAS_CHECK(rocblas_strsm(blas_hdl,
                              rocblas_side_left,
                              rocblas_fill_lower,
                              rocblas_operation_none,
                              rocblas_diagonal_unit,
                              m,
                              n,
                              &alpha,
                              a,
                              lda,
                              b,
                              ldb));
}
