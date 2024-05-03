
#include "hplmxp.hpp"

template <>
void HPLMXP_trtriU(const int m, float* A, const int lda) {
  ROCBLAS_CHECK(rocsolver_strtri(blas_hdl,
                                 rocblas_fill_upper,
                                 rocblas_diagonal_non_unit,
                                 m,
                                 A,
                                 lda,
                                 blas_info));
}

template <>
void HPLMXP_trtriU(const int m, double* A, const int lda) {
  ROCBLAS_CHECK(rocsolver_dtrtri(blas_hdl,
                                 rocblas_fill_upper,
                                 rocblas_diagonal_non_unit,
                                 m,
                                 A,
                                 lda,
                                 blas_info));
}

template <>
void HPLMXP_trtriL(const int m, float* A, const int lda) {
  ROCBLAS_CHECK(rocsolver_strtri(blas_hdl,
                                 rocblas_fill_lower,
                                 rocblas_diagonal_unit,
                                 m,
                                 A,
                                 lda,
                                 blas_info));
}

template <>
void HPLMXP_trtriL(const int m, double* A, const int lda) {
  ROCBLAS_CHECK(rocsolver_dtrtri(blas_hdl,
                                 rocblas_fill_lower,
                                 rocblas_diagonal_unit,
                                 m,
                                 A,
                                 lda,
                                 blas_info));
}
