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
#ifndef HPLMXP_PTIMER_HPP
#define HPLMXP_PTIMER_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hplmxp_pmisc.hpp"

/*
 * ---------------------------------------------------------------------
 * #define macro constants
 * ---------------------------------------------------------------------
 */
#define HPLMXP_NPTIMER 64
#define HPLMXP_PTIMER_STARTFLAG 5.0
#define HPLMXP_PTIMER_ERROR -1.0
/*
 * ---------------------------------------------------------------------
 * type definitions
 * ---------------------------------------------------------------------
 */
typedef enum { HPLMXP_WALL_PTIME = 101, HPLMXP_CPU_PTIME = 102 } HPLMXP_T_PTIME;

typedef enum {
  HPLMXP_AMAX_PTIME = 201,
  HPLMXP_AMIN_PTIME = 202,
  HPLMXP_SUM_PTIME  = 203
} HPLMXP_T_PTIME_OP;
/*
 * ---------------------------------------------------------------------
 * Function prototypes
 * ---------------------------------------------------------------------
 */
double HPLMXP_ptimer_cputime(void);
double HPLMXP_ptimer_walltime(void);
void   HPLMXP_ptimer(const int);
void   HPLMXP_ptimer_boot(void);

void HPLMXP_ptimer_combine(MPI_Comm comm,
                           const HPLMXP_T_PTIME_OP,
                           const HPLMXP_T_PTIME,
                           const int,
                           const int,
                           double*);

void   HPLMXP_ptimer_disable(void);
void   HPLMXP_ptimer_enable(void);
double HPLMXP_ptimer_inquire(const HPLMXP_T_PTIME, const int);
void   HPLMXP_ptimer_stepReset(const int, const int);
double HPLMXP_ptimer_getStep(const int);

#endif
/*
 * End of hpl_ptimer.hpp
 */
