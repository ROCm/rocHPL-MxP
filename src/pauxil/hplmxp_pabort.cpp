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

void HPLMXP_pabort(int LINE, const char* SRNAME, const char* FORM, ...) {
  /*
   * Purpose
   * =======
   *
   * HPLMXP_pabort displays an error message on stderr and halts execution.
   *
   *
   * Arguments
   * =========
   *
   * LINE    (local input)                 int
   *         On entry,  LINE  specifies the line  number in the file where
   *         the  error  has  occured.  When  LINE  is not a positive line
   *         number, it is ignored.
   *
   * SRNAME  (local input)                 const char *
   *         On entry, SRNAME  should  be the name of the routine  calling
   *         this error handler.
   *
   * FORM    (local input)                 const char *
   *         On entry, FORM specifies the format, i.e., how the subsequent
   *         arguments are converted for output.
   *
   *         (local input)                 ...
   *         On entry,  ...  is the list of arguments to be printed within
   *         the format string.
   *
   * ---------------------------------------------------------------------
   */

  va_list argptr;
  int     rank;
  char    cline[512];

  va_start(argptr, FORM);
  (void)vsnprintf(cline, sizeof(cline), FORM, argptr);
  va_end(argptr);

  rank = (HPLMXP_comm_world() == HPLMXP_COMM_NULL
              ? -1
              : HPLMXP_comm_rank(HPLMXP_comm_world()));
  /*
   * Display an error message
   */
  if(LINE <= 0)
    HPLMXP_fprintf(stderr,
                   "%s %s %d, %s %s:\n>>> %s <<< Abort ...\n\n",
                   "HPL ERROR",
                   "from process #",
                   rank,
                   "in function",
                   SRNAME,
                   cline);
  else
    HPLMXP_fprintf(stderr,
                   "%s %s %d, %s %d %s %s:\n>>> %s <<< Abort ...\n\n",
                   "HPL ERROR",
                   "from process #",
                   rank,
                   "on line",
                   LINE,
                   "of function",
                   SRNAME,
                   cline);

  HPLMXP_comm_abort(HPLMXP_COMM_NULL, -1);
}
