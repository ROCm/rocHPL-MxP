
#include "hplmxp.hpp"

template <>
void HPLMXP_trsvU(const int m, const double* A, const int lda, double* x) {
  ROCBLAS_CHECK(rocblas_dtrsv(blas_hdl,
                              rocblas_fill_upper,
                              rocblas_operation_none,
                              rocblas_diagonal_non_unit,
                              m,
                              A,
                              lda,
                              x,
                              1));
}

template <>
void HPLMXP_trsvL(const int m, const double* A, const int lda, double* x) {
  ROCBLAS_CHECK(rocblas_dtrsv(blas_hdl,
                              rocblas_fill_lower,
                              rocblas_operation_none,
                              rocblas_diagonal_unit,
                              m,
                              A,
                              m,
                              x,
                              1));
}

template <>
void HPLMXP_trsvU(const int m, const float* A, const int lda, float* x) {
  ROCBLAS_CHECK(rocblas_strsv(blas_hdl,
                              rocblas_fill_upper,
                              rocblas_operation_none,
                              rocblas_diagonal_non_unit,
                              m,
                              A,
                              lda,
                              x,
                              1));
}

template <>
void HPLMXP_trsvL(const int m, const float* A, const int lda, float* x) {
  ROCBLAS_CHECK(rocblas_strsv(blas_hdl,
                              rocblas_fill_lower,
                              rocblas_operation_none,
                              rocblas_diagonal_unit,
                              m,
                              A,
                              m,
                              x,
                              1));
}
