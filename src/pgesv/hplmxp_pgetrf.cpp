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

void HPLMXP_pgetrf(HPLMXP_T_grid&         grid,
                   HPLMXP_T_palg&         algo,
                   HPLMXP_T_pmat<fp32_t>& A) {
  /*
   * Purpose
   * =======
   *
   * HPLMXP_pgetrf factors a N-by-N matrix using LU factorization.  The
   * main algorithm  is the "right looking" variant with  look-ahead.
   *
   * Arguments
   * =========
   *
   * GRID    (local input)                 HPLMXP_T_grid *
   *         On entry,  GRID  points  to the data structure containing the
   *         process grid information.
   *
   * ALGO    (global input)                HPLMXP_T_palg *
   *         On entry,  ALGO  points to  the data structure containing the
   *         algorithmic parameters.
   *
   * A       (local input/output)          HPLMXP_T_pmat *
   *         On entry, A points to the data structure containing the local
   *         array information.
   *
   * ---------------------------------------------------------------------
   */

  fp32_t*   Ap      = A.A;
  int const N       = A.n;
  int const b       = A.nb;
  int const nblocks = N / b;
  int const nbrow   = A.nbrow;
  int const nbcol   = A.nbcol;
  int const lda     = A.ld;

  int const myrow = grid.myrow;
  int const mycol = grid.mycol;
  int const nprow = grid.nprow;
  int const npcol = grid.npcol;

  const fp32_t one   = 1.0;
  const fp32_t alpha = -1.0;
  const fp32_t beta  = 1.0;

  /* piv */
  fp32_t*   piv   = nullptr;
  const int ldpiv = b;
  if(hipMalloc(&piv, sizeof(fp32_t) * b * ldpiv) != hipSuccess) {
    HPLMXP_pabort(
        __LINE__, "HPLMXP_pgetrf", "Memory allocation failed for piv.");
  }
  fp16_t* pivL = nullptr;
  if(hipMalloc(&pivL, sizeof(fp16_t) * b * ldpiv) != hipSuccess) {
    HPLMXP_pabort(
        __LINE__, "HPLMXP_pgetrf", "Memory allocation failed for pivL.");
  }
  fp16_t* pivU = nullptr;
  if(hipMalloc(&pivU, sizeof(fp16_t) * b * ldpiv) != hipSuccess) {
    HPLMXP_pabort(
        __LINE__, "HPLMXP_pgetrf", "Memory allocation failed for pivU.");
  }

  HPLMXP_identity(b, pivL, ldpiv);
  HPLMXP_identity(b, pivU, ldpiv);

  int i = 0, j = 0;
  int ip1 = i + (((1 % grid.nprow) == grid.myrow) ? 1 : 0);
  int jp1 = j + (((1 % grid.npcol) == grid.mycol) ? 1 : 0);

  HPLMXP_T_panel panels[2];
  HPLMXP_pdpanel_new(grid, A, N, b, 0, 0, i * b, j * b, panels[0]);
  HPLMXP_pdpanel_new(grid, A, N - b, b, b, b, ip1 * b, jp1 * b, panels[1]);

#ifdef HPLMXP_PROGRESS_REPORT
#ifdef HPLMXP_DETAILED_TIMING
  float  DgemmTime, LgemmTime, UgemmTime, TgemmTime;
  double DgemmGflops, LgemmGflops, UgemmGflops, TgemmGflops;

  if(grid.myrow == 0 && grid.mycol == 0) {
    printf("-------------------------------------------------------------------"
           "-------------------------------------------------------------------"
           "------------------------------\n");
    printf("   %%   | Column    | Step Time (s) ||               GEMM GFLOPS   "
           "            || Dbcast (s) | Lbcast (s) | Ubcast (s) |  GPU Sync "
           "(s) | Step GFLOPS | Overall GFLOPS\n");
    printf("       |           |               | Diagonal |    L     |    U    "
           " | Trailing |"
           "            |            |            |               |            "
           " |               \n");
    printf("-------------------------------------------------------------------"
           "-------------------------------------------------------------------"
           "------------------------------\n");
  }
#else
  if(grid.myrow == 0 && grid.mycol == 0) {
    printf("---------------------------------------------------\n");
    printf("   %%   | Column    | Step Time (s) | Overall GFLOPS\n");
    printf("       |           |               |               \n");
    printf("---------------------------------------------------\n");
  }
#endif
#endif

  /* start time */
  double start_time = MPI_Wtime();

  i = 0, j = 0;

  for(int k = 0; k < nblocks; ++k) {
    HPLMXP_ptimer_stepReset(HPLMXP_TIMING_N, HPLMXP_TIMING_BEG);

    const int n = N - k * b;

    double stepStart = MPI_Wtime();

    HPLMXP_T_panel& prev = panels[k % 2];
    HPLMXP_T_panel& next = panels[(k + 1) % 2];

    int const rootrow = k % grid.nprow;
    int const rootcol = k % grid.npcol;

    const bool icurrow = (rootrow == grid.myrow);
    const bool icurcol = (rootcol == grid.mycol);

    ip1 = i + ((icurrow) ? 1 : 0);
    jp1 = j + ((icurcol) ? 1 : 0);

    HPLMXP_pdpanel_init(
        grid, A, n - b, b, (k + 1) * b, (k + 1) * b, ip1 * b, jp1 * b, next);

    if(icurrow && icurcol) {
      /* Update look-ahead */
      if(k > 0) {
        HIP_CHECK(hipEventRecord(DgemmStart, computeStream));
        HPLMXP_gemmNT(b,
                      b,
                      b,
                      alpha,
                      prev.L,
                      prev.ldl,
                      prev.U,
                      prev.ldu,
                      beta,
                      Mptr(Ap, i * b, j * b, lda),
                      lda);
        HIP_CHECK(hipEventRecord(DgemmEnd, computeStream));
      }

      /* Factor panel */
      HPLMXP_lacpy(b, b, Mptr(Ap, i * b, j * b, lda), lda, piv, ldpiv);
      HPLMXP_getrf(b, b, piv, ldpiv);
      HPLMXP_lacpy(b, b, piv, ldpiv, Mptr(Ap, i * b, j * b, lda), lda);
      HPLMXP_trtriU(b, piv, ldpiv);
      HPLMXP_trtriL(b, piv, ldpiv);

      /* Record */
      HIP_CHECK(hipEventRecord(getrf, computeStream));
    }

    if(icurcol) {
      /* gemm L */
      if(k > 0) {
        HIP_CHECK(hipEventRecord(LgemmStart, computeStream));
        HPLMXP_gemmNT(A.mp - ip1 * b,
                      b,
                      b,
                      alpha,
                      Mptr(prev.L, (ip1 - i) * b, 0, prev.ldl),
                      prev.ldl,
                      prev.U,
                      prev.ldu,
                      beta,
                      Mptr(Ap, ip1 * b, j * b, lda),
                      lda);
        HIP_CHECK(hipEventRecord(LgemmEnd, computeStream));
      }

      HPLMXP_lacpy(A.mp - ip1 * b,
                   b,
                   Mptr(Ap, ip1 * b, j * b, lda),
                   lda,
                   next.L,
                   next.ldl);

      /*recieve piv*/
      if(!icurrow) {
        HPLMXP_TracingPush("D Column Bcast");
        HPLMXP_ptimer(HPLMXP_TIMING_DBCAST);
        HPLMXP_bcast(piv, ldpiv * b, rootrow, grid.col_comm, algo.btopo);
        HPLMXP_ptimer(HPLMXP_TIMING_DBCAST);
        HPLMXP_TracingPop("D Column Bcast");
      }

      HPLMXP_latcpyU(b, b, piv, ldpiv, pivU, ldpiv);

      HPLMXP_gemmNT(A.mp - ip1 * b,
                    b,
                    b,
                    one,
                    next.L,
                    next.ldl,
                    pivU,
                    ldpiv,
                    fp32_t{0.0},
                    Mptr(Ap, ip1 * b, j * b, lda),
                    lda);

      HPLMXP_lacpy(A.mp - ip1 * b,
                   b,
                   Mptr(Ap, ip1 * b, j * b, lda),
                   lda,
                   next.L,
                   next.ldl);

      /* Record */
      HIP_CHECK(hipEventRecord(lbcast, computeStream));
    }

    if(icurrow) {
      /* gemm U */
      if(k > 0) {
        HIP_CHECK(hipEventRecord(UgemmStart, computeStream));
        HPLMXP_gemmNT(b,
                      A.nq - jp1 * b,
                      b,
                      alpha,
                      prev.L,
                      prev.ldl,
                      Mptr(prev.U, (jp1 - j) * b, 0, prev.ldu),
                      prev.ldu,
                      beta,
                      Mptr(Ap, i * b, jp1 * b, lda),
                      lda);
        HIP_CHECK(hipEventRecord(UgemmEnd, computeStream));
      }

      HPLMXP_latcpy(A.nq - jp1 * b,
                    b,
                    Mptr(Ap, i * b, jp1 * b, lda),
                    lda,
                    next.U,
                    next.ldu);

      /*recieve piv*/
      if(!icurcol) {
        HPLMXP_TracingPush("D Row Bcast");
        HPLMXP_ptimer(HPLMXP_TIMING_DBCAST);
        HPLMXP_bcast(piv, ldpiv * b, rootcol, grid.row_comm, algo.btopo);
        HPLMXP_ptimer(HPLMXP_TIMING_DBCAST);
        HPLMXP_TracingPop("D Row Bcast");
      }

      HPLMXP_lacpyL(b, b, piv, ldpiv, pivL, ldpiv);

      HPLMXP_gemmNT(b,
                    A.nq - jp1 * b,
                    b,
                    one,
                    pivL,
                    ldpiv,
                    next.U,
                    next.ldu,
                    fp32_t{0.0},
                    Mptr(Ap, i * b, jp1 * b, lda),
                    lda);

      HPLMXP_latcpy(A.nq - jp1 * b,
                    b,
                    Mptr(Ap, i * b, jp1 * b, lda),
                    lda,
                    next.U,
                    next.ldu);

      /* Record */
      HIP_CHECK(hipEventRecord(ubcast, computeStream));
    }

    /* Trailing update */
    if(k > 0) {
      HIP_CHECK(hipEventRecord(TgemmStart, computeStream));
      HPLMXP_gemmNT(A.mp - ip1 * b,
                    A.nq - jp1 * b,
                    b,
                    alpha,
                    Mptr(prev.L, (ip1 - i) * b, 0, prev.ldl),
                    prev.ldl,
                    Mptr(prev.U, (jp1 - j) * b, 0, prev.ldu),
                    prev.ldu,
                    beta,
                    Mptr(Ap, ip1 * b, jp1 * b, lda),
                    lda);
      HIP_CHECK(hipEventRecord(TgemmEnd, computeStream));
    }

    if(icurrow && icurcol) {
      HPLMXP_ptimer(HPLMXP_TIMING_UPDATE);
      HIP_CHECK(hipEventSynchronize(getrf));
      HPLMXP_ptimer(HPLMXP_TIMING_UPDATE);

      /* broadcast piv */
      HPLMXP_TracingPush("D Row Bcast");
      HPLMXP_ptimer(HPLMXP_TIMING_DBCAST);
      HPLMXP_bcast(piv, ldpiv * b, rootcol, grid.row_comm, algo.btopo);
      HPLMXP_ptimer(HPLMXP_TIMING_DBCAST);
      HPLMXP_TracingPop("D Row Bcast");

      /* broadcast piv */
      HPLMXP_TracingPush("D Column Bcast");
      HPLMXP_ptimer(HPLMXP_TIMING_DBCAST);
      HPLMXP_bcast(piv, ldpiv * b, rootrow, grid.col_comm, algo.btopo);
      HPLMXP_ptimer(HPLMXP_TIMING_DBCAST);
      HPLMXP_TracingPop("D Column Bcast");
    }

    /* broadcast left panel */
    if(icurcol) {
      HPLMXP_ptimer(HPLMXP_TIMING_UPDATE);
      HIP_CHECK(hipEventSynchronize(lbcast));
      HPLMXP_ptimer(HPLMXP_TIMING_UPDATE);
    }
    HPLMXP_TracingPush("L Bcast");
    HPLMXP_ptimer(HPLMXP_TIMING_LBCAST);
    HPLMXP_bcast(next.L, next.ldl * b, rootcol, grid.row_comm, algo.btopo);
    HPLMXP_ptimer(HPLMXP_TIMING_LBCAST);
    HPLMXP_TracingPop("L Bcast");

    /* broadcast right panel */
    if(icurrow) {
      HPLMXP_ptimer(HPLMXP_TIMING_UPDATE);
      HIP_CHECK(hipEventSynchronize(ubcast));
      HPLMXP_ptimer(HPLMXP_TIMING_UPDATE);
    }
    HPLMXP_TracingPush("U Bcast");
    HPLMXP_ptimer(HPLMXP_TIMING_UBCAST);
    HPLMXP_bcast(next.U, next.ldu * b, rootrow, grid.col_comm, algo.btopo);
    HPLMXP_ptimer(HPLMXP_TIMING_UBCAST);
    HPLMXP_TracingPop("U Bcast");

    /* wait here for the updates to compete */
    HPLMXP_ptimer(HPLMXP_TIMING_UPDATE);
    HIP_CHECK(hipDeviceSynchronize());
    HPLMXP_ptimer(HPLMXP_TIMING_UPDATE);

    double stepEnd = MPI_Wtime();

#ifdef HPLMXP_PROGRESS_REPORT
#ifdef HPLMXP_DETAILED_TIMING
    DgemmTime = 0.0;
    if(k > 0 && icurrow && icurcol) {
      HIP_CHECK(hipEventElapsedTime(&DgemmTime, DgemmStart, DgemmEnd));
      DgemmGflops = (2.0 * b * b * b) / (1.0e6 * (DgemmTime));
    }
    LgemmTime = 0.0;
    if(k > 0 && icurcol && (nbrow - ip1 > 0)) {
      HIP_CHECK(hipEventElapsedTime(&LgemmTime, LgemmStart, LgemmEnd));
      LgemmGflops = (2.0 * b * b * b * (nbrow - ip1)) / (1.0e6 * (LgemmTime));
    }
    UgemmTime = 0.0;
    if(k > 0 && icurrow && (nbcol - jp1 > 0)) {
      HIP_CHECK(hipEventElapsedTime(&UgemmTime, UgemmStart, UgemmEnd));
      UgemmGflops = (2.0 * b * b * b * (nbcol - jp1)) / (1.0e6 * (UgemmTime));
    }
    TgemmTime = 0.0;
    if(k > 0 && (nbrow - ip1 > 0) && (nbcol - jp1 > 0)) {
      HIP_CHECK(hipEventElapsedTime(&TgemmTime, TgemmStart, TgemmEnd));
      TgemmGflops = (2.0 * b * b * (nbrow - ip1) * b * (nbcol - jp1)) /
                    (1.0e6 * (TgemmTime));
    }
#endif
    /* if this is process 0,0 and not the first panel */
    if(grid.myrow == 0 && grid.mycol == 0 && k > 0) {
      double time      = HPLMXP_ptimer_walltime() - start_time;
      double step_time = stepEnd - stepStart;
      /*
      Step FLOP count is (2/3)NB^3 - (1/2)NB^2 - (1/6)NB
                          + 2*n*NB^2 - n*NB + 2*NB*n^2

      Overall FLOP count is (2/3)(N^3-n^3) - (1/2)(N^2-n^2) - (1/6)(N-n)
      */
      double step_gflops =
          ((2.0 / 3.0) * b * b * b - (1.0 / 2.0) * b * b - (1.0 / 6.0) * b +
           2.0 * n * b * b - b * n + 2.0 * b * n * n) /
          (step_time > 0.0 ? step_time : 1.e-6) / 1.e9;
      double gflops = ((2.0 / 3.0) * (N * (double)N * N - n * (double)n * n) -
                       (1.0 / 2.0) * (N * (double)N - n * (double)n) -
                       (1.0 / 6.0) * ((double)N - (double)n)) /
                      (time > 0.0 ? time : 1.e-6) / 1.e9;
      printf("%5.1f%% | %09d | ", k * b * 100.0 / N, k * b);
      printf("   %9.7f  |", stepEnd - stepStart);

#ifdef HPLMXP_DETAILED_TIMING
      if(icurrow && icurcol) {
        printf(" %9.3e|", DgemmGflops);
      } else {
        printf("          |");
      }
      if(icurcol && (nbrow - ip1 > 0)) {
        printf(" %9.3e|", LgemmGflops);
      } else {
        printf("          |");
      }
      if(icurrow && (nbcol - jp1 > 0)) {
        printf(" %9.3e|", UgemmGflops);
      } else {
        printf("          |");
      }
      if((nbrow - ip1 > 0) && (nbcol - jp1 > 0)) {
        printf(" %9.3e|", TgemmGflops);
      } else {
        printf("          |");
      }

      if(icurrow || icurcol) {
        printf("  %9.3e |", HPLMXP_ptimer_getStep(HPLMXP_TIMING_DBCAST));
      } else {
        printf("            |");
      }

      printf("  %9.3e |  %9.3e |    %9.3e  |",
             HPLMXP_ptimer_getStep(HPLMXP_TIMING_LBCAST),
             HPLMXP_ptimer_getStep(HPLMXP_TIMING_UBCAST),
             HPLMXP_ptimer_getStep(HPLMXP_TIMING_UPDATE));

      printf("  %9.3e  |", step_gflops);
#endif

      printf("    %9.3e   \n", gflops);
    }
#endif

    i = ip1;
    j = jp1;
  }

  HPLMXP_pdpanel_free(panels[1]);
  HPLMXP_pdpanel_free(panels[0]);

  HIP_CHECK(hipFree(pivU));
  HIP_CHECK(hipFree(pivL));
  HIP_CHECK(hipFree(piv));
}
