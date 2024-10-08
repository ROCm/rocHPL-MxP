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
#ifndef HPLMXP_PTEST_HPP
#define HPLMXP_PTEST_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hplmxp_misc.hpp"
#include "hplmxp_blas.hpp"
#include "hplmxp_auxil.hpp"

#include "hplmxp_pmisc.hpp"
#include "hplmxp_pauxil.hpp"
#include "hplmxp_panel.hpp"
#include "hplmxp_pgesv.hpp"

#include "hplmxp_ptimer.hpp"
#include "hplmxp_pmatgen.hpp"

/*
 * ---------------------------------------------------------------------
 * Data Structures
 * ---------------------------------------------------------------------
 */
typedef struct HPLMXP_S_test {
  double epsil; /* epsilon machine */
  double thrsh; /* threshold */
  FILE*  outfp; /* output stream (only in proc 0) */
  int    kfail; /* # of tests failed */
  int    kpass; /* # of tests passed */
  int    kskip; /* # of tests skipped */
  int    ktest; /* total number of tests */
} HPLMXP_T_test;

/*
 * ---------------------------------------------------------------------
 * #define macro constants for testing only
 * ---------------------------------------------------------------------
 */
#define HPLMXP_LINE_MAX 256
#define HPLMXP_MAX_PARAM 20
#define HPLMXP_SEED 42

/*
 * ---------------------------------------------------------------------
 * global timers for timing analysis only
 * ---------------------------------------------------------------------
 */
#define HPLMXP_TIMING_BEG 11    /* timer 0 reserved, used by main */
#define HPLMXP_TIMING_N 6       /* number of timers defined below */
#define HPLMXP_TIMING_DBCAST 11 /* starting from here, contiguous */
#define HPLMXP_TIMING_LBCAST 12
#define HPLMXP_TIMING_UBCAST 13
#define HPLMXP_TIMING_UPDATE 14
#define HPLMXP_TIMING_PGETRF 15
#define HPLMXP_TIMING_IR 16

/*
 * ---------------------------------------------------------------------
 * Function prototypes
 * ---------------------------------------------------------------------
 */
void HPLMXP_pinfo(int             ARGC,
                  char**          ARGV,
                  HPLMXP_T_test*  TEST,
                  int*            NS,
                  int*            N,
                  int*            NBS,
                  int*            NB,
                  HPLMXP_T_ORDER* PMAPPIN,
                  int*            P,
                  int*            Q,
                  int*            p,
                  int*            q,
                  int*            NTPS,
                  HPLMXP_T_TOP*   TP);

void HPLMXP_ptest(HPLMXP_T_test& test,
                  HPLMXP_T_grid& grid,
                  HPLMXP_T_palg& algo,
                  const int      N,
                  const int      NB);
void HPLMXP_Warmup(HPLMXP_T_grid&         grid,
                   HPLMXP_T_palg&         algo,
                   HPLMXP_T_pmat<fp64_t>& A,
                   HPLMXP_T_pmat<fp32_t>& LU);
void HPLMXP_InitGPU(const HPLMXP_T_grid& grid);
void HPLMXP_FreeGPU();

#endif
/*
 * End of hplmxp_ptest.hpp
 */
