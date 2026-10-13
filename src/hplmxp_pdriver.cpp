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
#include <numeric>

int main(int ARGC, char** ARGV) {
  /*
   * Purpose
   * =======
   *
   * main is the main driver program for testing the HPL routines.
   * This  program is  driven  by  a short data file named  "HPL.dat".
   *
   * ---------------------------------------------------------------------
   */

  int nval[HPLMXP_MAX_PARAM], nbval[HPLMXP_MAX_PARAM];

  HPLMXP_T_TOP topval[HPLMXP_MAX_PARAM];

  HPLMXP_T_grid grid;
  HPLMXP_T_palg algo;
  HPLMXP_T_test test;

  int            P, Q, p, q, ns, nbs, ntps, rank, size, its;
  HPLMXP_T_ORDER pmapping;

  /* A run killed from outside (step timeout) keeps every line it printed */
  setvbuf(stdout, nullptr, _IOLBF, 0);

  HPLMXP_Comm world;
  HPLMXP_comm_world_init(&ARGC, &ARGV, &world);
  rank = HPLMXP_comm_rank(world);
  size = HPLMXP_comm_size(world);

  /*
   * Read and check validity of test parameters from input file
   *
   * HPL-MxP Version 1.0, Linpack benchmark input file
   * Your message here
   * HPL-MxP.out  output file name (if any)
   * 6            device out (6=stdout,7=stderr,file)
   * 4            # of problems sizes (N)
   * 29 30 34 35  Ns
   * 4            # of NBs
   * 1 2 3 4      NBs
   * 0            PMAP process mapping (0=Row-,1=Column-major)
   * 2            P
   * 2            Q
   * 16.0         threshold
   * 1            # of broadcast
   * 0            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
   */
  HPLMXP_pinfo(world,
               ARGC,
               ARGV,
               &test,
               &ns,
               nval,
               &nbs,
               nbval,
               &pmapping,
               &P,
               &Q,
               &p,
               &q,
               &ntps,
               topval,
               &its);

  HPLMXP_grid_init(world, pmapping, P, Q, p, q, grid);

  int mycol, myrow, nprow, npcol;
  HPLMXP_grid_info(grid, nprow, npcol, myrow, mycol);

  /* Processes beyond P*Q are not part of the grid: they only finalize */
  const bool in_grid = (grid.all_comm != HPLMXP_COMM_NULL);

  // Initialize GPU
  if(in_grid) HPLMXP_InitGPU(grid);

  for(int in = 0; in_grid && in < ns;
      in++) { /* Loop over various problem sizes */
    for(int inb = 0; inb < nbs;
        inb++) { /* Loop over various blocking factors */
      for(int itop = 0; itop < ntps;
          itop++) { /* Loop over various broadcast topologies */
        /*
         * Set up the algorithm parameters
         */
        algo.btopo = topval[itop];
        algo.epsil = test.epsil;
        algo.thrsh = test.thrsh;
        algo.its   = its;

        int n  = nval[in];
        int nb = nbval[inb];

        /* Need NB to evenly divide N */
        if(n % nb != 0) { n = ((n + nb - 1) / nb) * nb; }

        /* load balance - makes all processes have the same size local matrix*/
        int epoch_size = nb * std::lcm(P, Q);
        if(n % epoch_size) {
          n = ((n + epoch_size - 1) / epoch_size) * epoch_size;
        }

        HPLMXP_ptest(test, grid, algo, n, nb);
      }
    }
  }

  if(in_grid) HPLMXP_FreeGPU();

  HPLMXP_grid_exit(grid);

  /*
   * Print ending messages, close output file, exit.
   */
  if(rank == 0) {
    test.ktest = test.kpass + test.kfail + test.kskip;
#ifndef HPLMXP_DETAILED_TIMING
    HPLMXP_fprintf(test.outfp,
                   "%s%s\n",
                   "========================================",
                   "========================================");
#else
    if(test.thrsh > HPLMXP_rzero)
      HPLMXP_fprintf(test.outfp,
                     "%s%s\n",
                     "========================================",
                     "========================================");
#endif

    HPLMXP_fprintf(test.outfp,
                   "\n%s %6d %s\n",
                   "Finished",
                   test.ktest,
                   "tests with the following results:");
    if(test.thrsh > HPLMXP_rzero) {
      HPLMXP_fprintf(test.outfp,
                     "         %6d %s\n",
                     test.kpass,
                     "tests completed and passed residual checks,");
      HPLMXP_fprintf(test.outfp,
                     "         %6d %s\n",
                     test.kfail,
                     "tests completed and failed residual checks,");
      HPLMXP_fprintf(test.outfp,
                     "         %6d %s\n",
                     test.kskip,
                     "tests skipped because of illegal input values.");
    } else {
      HPLMXP_fprintf(test.outfp,
                     "         %6d %s\n",
                     test.kpass,
                     "tests completed without checking,");
      HPLMXP_fprintf(test.outfp,
                     "         %6d %s\n",
                     test.kskip,
                     "tests skipped because of illegal input values.");
    }

    HPLMXP_fprintf(test.outfp,
                   "%s%s\n",
                   "----------------------------------------",
                   "----------------------------------------");
    HPLMXP_fprintf(test.outfp, "\nEnd of Tests.\n");
    HPLMXP_fprintf(test.outfp,
                   "%s%s\n",
                   "========================================",
                   "========================================");

    if((test.outfp != stdout) && (test.outfp != stderr))
      (void)fclose(test.outfp);
  }

  HPLMXP_comm_finalize();

  return (0);
}
