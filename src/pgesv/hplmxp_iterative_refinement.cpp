#include "hplmxp.hpp"

void HPLMXP_iterative_refinement(HPLMXP_T_grid&         grid,
                                 HPLMXP_T_palg&         algo,
                                 HPLMXP_T_pmat<fp64_t>& A,
                                 HPLMXP_T_pmat<fp32_t>& LU) {
  // do IR with approximated LU factors in p and the accurate initial matrix
  int const n  = A.n;
  int const nb = A.nb;

  const int maxits = 50;

  fp64_t* b = A.b;
  fp64_t* x = A.x;

  /* workspaces */
  fp64_t* r = LU.work;
  fp64_t* v = LU.work + nb * LU.nbrow;
  fp64_t* work = LU.work + nb * LU.nbrow + nb * LU.nbcol;

  for(int iter = 0; iter < maxits; ++iter) {
    HPLMXP_pcopy(grid, nb * A.nbrow, nb, b, r);

    fp64_t normx = HPLMXP_plange(grid, nb * A.nbrow, nb, x);

    HPLMXP_ptranspose(grid, nb * A.nbrow, nb, x, v);

    // compute residual, r_i = b - A x_i
    HPLMXP_pgemv(grid, A, -1., v, 1., r);

    fp64_t normr = HPLMXP_plange(grid, nb * A.nbrow, nb, r);

    // residual := \|b-Ax\|_\infty / (\|A\|_\infty \|x\|_\infty + \|b\|_\infty)
    // * (n * \epsilon)^{-1}
    A.res = normr / (A.norma * normx + A.normb) * 1. / (n * algo.epsil);

#ifdef HPLMXP_PROGRESS_REPORT
    if(grid.myrow == 0 && grid.mycol == 0) {
      printf("# refinement: step=%3d, ||r||=%.8e, residual=%g\n",
             iter + 1,
             normr,
             A.res);
    }
#endif

    if(A.res < algo.thrsh || iter == maxits - 1) break;

    // x_1 = x_0 + (LU)^{-1} r
    HPLMXP_ptrsvL(grid, LU, r, work);
    HPLMXP_ptrsvU(grid, LU, r, work);
    HPLMXP_paxpy(grid, nb * A.nbrow, nb, 1.0, r, x);
  }
}
