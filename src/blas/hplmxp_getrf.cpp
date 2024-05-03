
#include "hplmxp.hpp"

template <>
void HPLMXP_getrf(const int m, const int n, float* a, const int lda) {
  ROCBLAS_CHECK(rocsolver_sgetrf_npvt(blas_hdl, m, n, a, lda, blas_info));
}

template <>
void HPLMXP_getrf(const int m, const int n, double* a, const int lda) {
  ROCBLAS_CHECK(rocsolver_dgetrf_npvt(blas_hdl, m, n, a, lda, blas_info));
}
