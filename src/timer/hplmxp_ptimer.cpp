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
/*
 * ---------------------------------------------------------------------
 * Static variables
 * ---------------------------------------------------------------------
 */
static int    HPLMXP_ptimer_disabled;
static double HPLMXP_ptimer_cpusec[HPLMXP_NPTIMER],
    HPLMXP_ptimer_cpustart[HPLMXP_NPTIMER];
static double HPLMXP_ptimer_wallsec[HPLMXP_NPTIMER],
    HPLMXP_ptimer_wallstart[HPLMXP_NPTIMER];
static double HPLMXP_ptimer_wallstep[HPLMXP_NPTIMER];
/*
 * ---------------------------------------------------------------------
 * User callable functions
 * ---------------------------------------------------------------------
 */
void HPLMXP_ptimer_boot() {
  /*
   * HPLMXP_ptimer_boot (re)sets all timers to 0, and enables HPLMXP_ptimer.
   */

  int i;

  HPLMXP_ptimer_disabled = 0;

  for(i = 0; i < HPLMXP_NPTIMER; i++) {
    HPLMXP_ptimer_cpusec[i] = HPLMXP_ptimer_wallsec[i] = HPLMXP_rzero;
    HPLMXP_ptimer_wallstep[i]                          = HPLMXP_rzero;
    HPLMXP_ptimer_cpustart[i] = HPLMXP_ptimer_wallstart[i] =
        HPLMXP_PTIMER_STARTFLAG;
  }
}

void HPLMXP_ptimer(const int I) {
  /*
   * Purpose
   * =======
   *
   * HPLMXP_ptimer provides a  "stopwatch"  functionality  cpu/wall  timer in
   * seconds.  Up to  64  separate timers can be functioning at once.  The
   * first call starts the timer,  and the second stops it.  This  routine
   * can be disenabled  by calling HPLMXP_ptimer_disable(),  so that calls to
   * the timer are ignored.  This feature can be used to make sure certain
   * sections of code do not affect timings,  even  if  they call routines
   * which have HPLMXP_ptimer calls in them. HPLMXP_ptimer_enable()  will enable
   * the  timer  functionality.  One  can retrieve  the current value of a
   * timer by calling
   *
   * t0 = HPLMXP_ptimer_inquire( HPLMXP_WALL_TIME | HPLMXP_CPU_TIME, I )
   *
   * where  I  is the timer index in  [0..64).  To  inititialize the timer
   * functionality, one must have called HPLMXP_ptimer_boot() prior to any of
   * the functions mentioned above.
   *
   * Arguments
   * =========
   *
   * I       (global input)                const int
   *         On entry, I specifies the timer to stop/start.
   *
   * ---------------------------------------------------------------------
   */

  if(HPLMXP_ptimer_disabled) return;
  /*
   * If timer has not been started, start it.  Otherwise,  stop it and add
   * interval to count
   */
  if(HPLMXP_ptimer_wallstart[I] == HPLMXP_PTIMER_STARTFLAG) {
    HPLMXP_ptimer_wallstart[I] = HPLMXP_ptimer_walltime();
    HPLMXP_ptimer_cpustart[I]  = HPLMXP_ptimer_cputime();
  } else {
    HPLMXP_ptimer_cpusec[I] +=
        HPLMXP_ptimer_cputime() - HPLMXP_ptimer_cpustart[I];
    const double walltime =
        HPLMXP_ptimer_walltime() - HPLMXP_ptimer_wallstart[I];
    HPLMXP_ptimer_wallstep[I] += walltime;
    HPLMXP_ptimer_wallsec[I] += walltime;
    HPLMXP_ptimer_wallstart[I] = HPLMXP_PTIMER_STARTFLAG;
  }
}

void HPLMXP_ptimer_enable(void) {
  /*
   * HPLMXP_ptimer_enable sets it so calls to HPLMXP_ptimer are not ignored.
   */

  HPLMXP_ptimer_disabled = 0;
  return;
}

void HPLMXP_ptimer_disable(void) {
  /*
   * HPLMXP_ptimer_disable sets it so calls to HPLMXP_ptimer are ignored.
   */

  HPLMXP_ptimer_disabled = 1;
  return;
}

void HPLMXP_ptimer_stepReset(const int N, const int IBEG) {
  for(int i = 0; i < N; i++) {
    HPLMXP_ptimer_wallstep[IBEG + i] = HPLMXP_rzero;
  }
}

double HPLMXP_ptimer_getStep(const int I) {

  double time;

  /*
   * If wall-time are not available on this machine, return
   * HPLMXP_PTIMER_ERROR
   */
  if(HPLMXP_ptimer_walltime() == HPLMXP_PTIMER_ERROR)
    time = HPLMXP_PTIMER_ERROR;
  else
    time = HPLMXP_ptimer_wallstep[I];

  return (time);
}

double HPLMXP_ptimer_inquire(const HPLMXP_T_PTIME TMTYPE, const int I) {
  /*
   * Purpose
   * =======
   *
   * HPLMXP_ptimer_inquire returns wall- or cpu- time that has accumulated in
   * timer I.
   *
   * Arguments
   * =========
   *
   * TMTYPE  (global input)              const HPLMXP_T_PTIME
   *         On entry, TMTYPE specifies what time will be returned as fol-
   *         lows
   *            = HPLMXP_WALL_PTIME : wall clock time is returned,
   *            = HPLMXP_CPU_PTIME  : CPU time is returned (default).
   *
   * I       (global input)              const int
   *         On entry, I specifies the timer to return.
   *
   * ---------------------------------------------------------------------
   */

  double time;

  /*
   * If wall- or cpu-time are not available on this machine, return
   * HPLMXP_PTIMER_ERROR
   */
  if(TMTYPE == HPLMXP_WALL_PTIME) {
    if(HPLMXP_ptimer_walltime() == HPLMXP_PTIMER_ERROR)
      time = HPLMXP_PTIMER_ERROR;
    else
      time = HPLMXP_ptimer_wallsec[I];
  } else {
    if(HPLMXP_ptimer_cputime() == HPLMXP_PTIMER_ERROR)
      time = HPLMXP_PTIMER_ERROR;
    else
      time = HPLMXP_ptimer_cpusec[I];
  }
  return (time);
}

void HPLMXP_ptimer_combine(MPI_Comm                COMM,
                           const HPLMXP_T_PTIME_OP OPE,
                           const HPLMXP_T_PTIME    TMTYPE,
                           const int               N,
                           const int               IBEG,
                           double*                 TIMES) {
  /*
   * Purpose
   * =======
   *
   * HPLMXP_ptimer_combine  combines the timing information stored on a scope
   * of processes into the user TIMES array.
   *
   * Arguments
   * =========
   *
   * COMM    (global/local input)        MPI_Comm
   *         The MPI communicator  identifying  the process  collection on
   *         which the timings are taken.
   *
   * OPE     (global input)              const HPLMXP_T_PTIME_OP
   *         On entry, OP  specifies what combine operation should be done
   *         as follows:
   *            = HPLMXP_AMAX_PTIME get max. time on any process (default),
   *            = HPLMXP_AMIN_PTIME get min. time on any process,
   *            = HPLMXP_SUM_PTIME  get sum of times across processes.
   *
   * TMTYPE  (global input)              const HPLMXP_T_PTIME
   *         On entry, TMTYPE specifies what time will be returned as fol-
   *         lows
   *            = HPLMXP_WALL_PTIME : wall clock time is returned,
   *            = HPLMXP_CPU_PTIME  : CPU time is returned (default).
   *
   * N       (global input)              const int
   *         On entry, N specifies the number of timers to combine.
   *
   * IBEG    (global input)              const int
   *         On entry, IBEG specifies the first timer to be combined.
   *
   * TIMES   (global output)             double *
   *         On entry, TIMES is an array of dimension at least N. On exit,
   *         this array contains the requested timing information.
   *
   * ---------------------------------------------------------------------
   */

  int i, tmpdis;

  tmpdis                 = HPLMXP_ptimer_disabled;
  HPLMXP_ptimer_disabled = 1;
  /*
   * Timer has been disabled for combine operation -  copy timing informa-
   * tion into user times array.  If  wall- or  cpu-time are not available
   * on this machine, fill in times with HPLMXP_PTIMER_ERROR flag and return.
   */
  if(TMTYPE == HPLMXP_WALL_PTIME) {
    if(HPLMXP_ptimer_walltime() == HPLMXP_PTIMER_ERROR) {
      for(i = 0; i < N; i++) TIMES[i] = HPLMXP_PTIMER_ERROR;
      return;
    } else {
      for(i = 0; i < N; i++) TIMES[i] = HPLMXP_ptimer_wallsec[IBEG + i];
    }
  } else {
    if(HPLMXP_ptimer_cputime() == HPLMXP_PTIMER_ERROR) {
      for(i = 0; i < N; i++) TIMES[i] = HPLMXP_PTIMER_ERROR;
      return;
    } else {
      for(i = 0; i < N; i++) TIMES[i] = HPLMXP_ptimer_cpusec[IBEG + i];
    }
  }
  /*
   * Combine all nodes information, restore HPLMXP_ptimer_disabled, and return
   */
  for(i = 0; i < N; i++) TIMES[i] = Mmax(HPLMXP_rzero, TIMES[i]);

  if(OPE == HPLMXP_AMAX_PTIME)
    (void)HPLMXP_all_reduce(TIMES, N, HPLMXP_MAX, COMM);
  else if(OPE == HPLMXP_AMIN_PTIME)
    (void)HPLMXP_all_reduce(TIMES, N, HPLMXP_MIN, COMM);
  else if(OPE == HPLMXP_SUM_PTIME)
    (void)HPLMXP_all_reduce(TIMES, N, HPLMXP_SUM, COMM);
  else
    (void)HPLMXP_all_reduce(TIMES, N, HPLMXP_MAX, COMM);

  HPLMXP_ptimer_disabled = tmpdis;
}
