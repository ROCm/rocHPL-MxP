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

void HPLMXP_ptest(HPLMXP_T_test& test,
                  HPLMXP_T_grid& grid,
                  HPLMXP_T_palg& algo,
                  const int      N,
                  const int      NB) {
/*
 * Purpose
 * =======
 *
 * HPLMXP_pdtest performs  one  test  given a set of parameters such as the
 * process grid, the  problem size, the distribution blocking factor ...
 * This function generates  the data, calls  and times the linear system
 * solver,  checks  the  accuracy  of the  obtained vector solution  and
 * writes this information to the file pointed to by test.outfp.
 *
 * Arguments
 * =========
 *
 * test    (global input)                HPLMXP_T_test *
 *         On entry,  test  points  to a testing data structure:  outfp
 *         specifies the output file where the results will be printed.
 *         It is only defined and used by the process  0  of the  grid.
 *         thrsh  specifies  the  threshhold value  for the test ratio.
 *         Concretely, a test is declared "PASSED"  if and only if the
 *         following inequality is satisfied:
 *         ||Ax-b||_oo / ( epsil *
 *                         ( || x ||_oo * || A ||_oo + || b ||_oo ) *
 *                          N )  < thrsh.
 *         epsil  is the  relative machine precision of the distributed
 *         computer. Finally the test counters, kfail, kpass, kskip and
 *         ktest are updated as follows:  if the test passes,  kpass is
 *         incremented by one;  if the test fails, kfail is incremented
 *         by one; if the test is skipped, kskip is incremented by one.
 *         ktest is left unchanged.
 *
 * grid    (local input)                 HPLMXP_T_grid *
 *         On entry,  grid  points  to the data structure containing the
 *         process grid information.
 *
 * algo    (global input)                HPLMXP_T_palg *
 *         On entry,  algo  points to  the data structure containing the
 *         algorithmic parameters to be used for this test.
 *
 * N       (global input)                const int
 *         On entry,  N specifies the order of the coefficient matrix A.
 *         N must be at least zero.
 *
 * NB      (global input)                const int
 *         On entry,  NB specifies the blocking factor used to partition
 *         and distribute the matrix A. NB must be larger than one.
 *
 * ---------------------------------------------------------------------
 */
/*
 * .. Local Variables ..
 */
#ifdef HPLMXP_DETAILED_TIMING
  double HPLMXP_w[HPLMXP_TIMING_N];
#endif
  double     wtime[1];
  double     Gflops;
  static int first = 1;
  char       ctop, cpfact, crfact;
  time_t     current_time_start, current_time_end;
  int        ierr;

  int mycol, myrow, npcol, nprow;
  HPLMXP_grid_info(grid, nprow, npcol, myrow, mycol);

  /* Create an fp64 Matrix */
  HPLMXP_T_pmat<fp64_t> A;
  HPLMXP_pmat_init(grid, N, NB, A);

  /* Create an fp32 proxy matrix */
  HPLMXP_T_pmat<fp32_t> LU;
  HPLMXP_pmat_init(grid, N, NB, LU);

  /* generate fp32 matrix on device */
  size_t totalMem = 0;
  ierr = HPLMXP_pmatgen(grid, LU, totalMem);
  if(ierr != HPLMXP_SUCCESS) {
    (test.kskip)++;
    HPLMXP_pmat_free(LU);
    HPLMXP_pmat_free(A);
    return;
  }

  /* generate fp64 rhs vector and initial guess */
  ierr = HPLMXP_pmatgen_rhs(grid, A, totalMem);
  if(ierr != HPLMXP_SUCCESS) {
    (test.kskip)++;
    HPLMXP_pmat_free(LU);
    HPLMXP_pmat_free(A);
    return;
  }
  ierr = HPLMXP_pmatgen_x(grid, A, totalMem);
  if(ierr != HPLMXP_SUCCESS) {
    (test.kskip)++;
    HPLMXP_pmat_free(LU);
    HPLMXP_pmat_free(A);
    return;
  }

#ifdef HPLMXP_VERBOSE_PRINT
  if((myrow == 0) && (mycol == 0)) {
    printf("Total device memory use = %g GiBs\n",
           ((double)totalMem) / (1024 * 1024 * 1024));
  }
#endif

  /* warm up the GPU to make sure library workspaces are allocated */
  HPLMXP_Warmup(grid, algo, A, LU);

  /* Generate problem */
  HPLMXP_prandmat(grid, LU);
  HPLMXP_prandmat_rhs(grid, A);
  HPLMXP_prandmat_x(grid, A);
  HIP_CHECK(hipDeviceSynchronize());

  /*
   * Solve linear system
   */
  HPLMXP_ptimer_boot();
  HPLMXP_barrier(grid.all_comm);
  time(&current_time_start);
  HPLMXP_ptimer(0);
  HPLMXP_pgesv(grid, algo, A, LU);
  HPLMXP_ptimer(0);
  time(&current_time_end);

  /*
   * Gather max of all CPU and WALL clock timings and print timing results
   */
  HPLMXP_ptimer_combine(
      grid.all_comm, HPLMXP_AMAX_PTIME, HPLMXP_WALL_PTIME, 1, 0, wtime);

  if((myrow == 0) && (mycol == 0)) {
    if(first) {
      HPLMXP_fprintf(test.outfp,
                     "%s%s\n",
                     "========================================",
                     "========================================");
      HPLMXP_fprintf(test.outfp,
                     "%s%s\n",
                     "T/V                N    NB     P     Q",
                     "               Time                 Gflops");
      HPLMXP_fprintf(test.outfp,
                     "%s%s\n",
                     "----------------------------------------",
                     "----------------------------------------");
      if(test.thrsh <= HPLMXP_rzero) first = 0;
    }
    /*
     * 2/3 N^3 - 1/2 N^2 flops for LU factorization + 2 N^2 flops for solve.
     * Print WALL time
     */
    Gflops = (((double)(N) / 1.0e+9) * ((double)(N) / wtime[0])) *
             ((2.0 / 3.0) * (double)(N) + (3.0 / 2.0));

    if(algo.btopo == HPLMXP_1RING)
      ctop = '0';
    else if(algo.btopo == HPLMXP_1RING_M)
      ctop = '1';
    else if(algo.btopo == HPLMXP_2RING)
      ctop = '2';
    else if(algo.btopo == HPLMXP_2RING_M)
      ctop = '3';
    else if(algo.btopo == HPLMXP_BLONG)
      ctop = '4';
    else /* if( algo.btopo == HPLMXP_BLONG_M ) */
      ctop = '5';

    if(wtime[0] > HPLMXP_rzero) {
      HPLMXP_fprintf(test.outfp,
                     "W%c%c%17d %5d %5d %5d %18.2f     %18.3e\n",
                     (grid.order == HPLMXP_ROW_MAJOR ? 'R' : 'C'),
                     ctop,
                     N,
                     NB,
                     nprow,
                     npcol,
                     wtime[0],
                     Gflops);
      HPLMXP_fprintf(test.outfp,
                     "HPLMXP_pdgesv() start time %s\n",
                     ctime(&current_time_start));
      HPLMXP_fprintf(test.outfp,
                     "HPLMXP_pdgesv() end time   %s\n",
                     ctime(&current_time_end));
    }
#ifdef HPLMXP_PROGRESS_REPORT
    printf("Final Score:    %7.4e GFLOPS \n", Gflops);
#endif
  }
#ifdef HPLMXP_DETAILED_TIMING
  HPLMXP_ptimer_combine(grid.all_comm,
                        HPLMXP_AMAX_PTIME,
                        HPLMXP_WALL_PTIME,
                        HPLMXP_TIMING_N,
                        HPLMXP_TIMING_BEG,
                        HPLMXP_w);
  if((myrow == 0) && (mycol == 0)) {
    HPLMXP_fprintf(test.outfp,
                   "%s%s\n",
                   "--VVV--VVV--VVV--VVV--VVV--VVV--VVV--V",
                   "VV--VVV--VVV--VVV--VVV--VVV--VVV--VVV-");
    /*
     * Dbcast
     */
    HPLMXP_fprintf(test.outfp,
                   "Max aggregated wall time D bcast . . : %18.2f\n",
                   HPLMXP_w[HPLMXP_TIMING_DBCAST - HPLMXP_TIMING_BEG]);
    /*
     * Lbcast
     */
    HPLMXP_fprintf(test.outfp,
                   "Max aggregated wall time L bcast . . : %18.2f\n",
                   HPLMXP_w[HPLMXP_TIMING_LBCAST - HPLMXP_TIMING_BEG]);
    /*
     * Ubcast
     */
    HPLMXP_fprintf(test.outfp,
                   "Max aggregated wall time U bcast . . : %18.2f\n",
                   HPLMXP_w[HPLMXP_TIMING_UBCAST - HPLMXP_TIMING_BEG]);
    /*
     * Update
     */
    HPLMXP_fprintf(test.outfp,
                   "Max aggregated wall time update  . . : %18.2f\n",
                   HPLMXP_w[HPLMXP_TIMING_UPDATE - HPLMXP_TIMING_BEG]);
    /*
     * Iterative Refinement
     */
    HPLMXP_fprintf(test.outfp,
                   "Max aggregated wall time Iter Refine : %18.2f\n",
                   HPLMXP_w[HPLMXP_TIMING_IR - HPLMXP_TIMING_BEG]);

    if(test.thrsh <= HPLMXP_rzero)
      HPLMXP_fprintf(test.outfp,
                     "%s%s\n",
                     "========================================",
                     "========================================");
  }
#endif

  /*
   * Computes and displays norms, residuals ...
   */
  double resid1;
  if(N <= 0) {
    resid1 = HPLMXP_rzero;
  } else {
    resid1 = A.res;
  }

  if(resid1 < test.thrsh)
    (test.kpass)++;
  else
    (test.kfail)++;

  if((myrow == 0) && (mycol == 0)) {
    HPLMXP_fprintf(test.outfp,
                   "%s%s\n",
                   "----------------------------------------",
                   "----------------------------------------");
    HPLMXP_fprintf(test.outfp,
                   "%s%16.7f%s%s\n",
                   "||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)= ",
                   resid1,
                   " ...... ",
                   (resid1 < test.thrsh ? "PASSED" : "FAILED"));

    if(resid1 >= test.thrsh) {
      HPLMXP_fprintf(test.outfp,
                     "%s%18.6f\n",
                     "||A||_oo . . . . . . . . . . . . . . . . . . . = ",
                     A.norma);
      HPLMXP_fprintf(test.outfp,
                     "%s%18.6f\n",
                     "||b||_oo . . . . . . . . . . . . . . . . . . . = ",
                     A.normb);
    }

#ifdef HPLMXP_PROGRESS_REPORT
    if(resid1 < test.thrsh)
      printf("Residual Check: PASSED \n");
    else
      printf("Residual Check: FAILED \n");
#endif
  }

  HPLMXP_pmat_free(LU);
  HPLMXP_pmat_free(A);
}
