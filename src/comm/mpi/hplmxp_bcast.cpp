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

#include "hplmxp_comm_mpi.hpp"
#include <climits>

template <typename T>
int HPLMXP_bcast(T*           SBUF,
                 size_t       COUNT,
                 int          ROOT,
                 HPLMXP_Comm  HCOMM,
                 hipStream_t  STREAM,
                 HPLMXP_T_TOP top) {
  /*
   * Purpose
   * =======
   *
   * HPLMXP_bcast is a simple wrapper around  MPI_Bcast.  Its  main  purpose is
   * to  allow for some  experimentation / tuning  of this simple routine.
   * Successful  completion  is  indicated  by  the  returned  error  code
   * HPLMXP_SUCCESS.  In the case of messages of length less than or equal to
   * zero, this function returns immediately.
   *
   * Arguments
   * =========
   *
   * SBUF    (local input)                 T *
   *         On entry, SBUF specifies the starting address of buffer to be
   *         broadcast.
   *
   * COUNT   (local input)                 size_t
   *         On entry,  COUNT  specifies  the number of  type T
   *         entries in SBUF.
   *
   * ROOT    (local input)                 int
   *         On entry, ROOT specifies the rank of the origin process in
   *         the communication space defined by COMM.
   *
   * HCOMM   (local input)                 HPLMXP_Comm
   *         The communicator identifying the communication space.
   *
   * STREAM  (local input)                 hipStream_t
   *         Consumer stream; unused here (MPI is host-blocking).
   *
   * ---------------------------------------------------------------------
   */

  if(COUNT == 0) return (HPLMXP_SUCCESS);
  const int SCOUNT = HPLMXP_mpi_count(COUNT, "HPLMXP_bcast");
  MPI_Comm  COMM   = HPLMXP_mpi(HCOMM);

  int ierr;

  // roctxRangePush("HPLMXP_Bcast");

#ifdef HPLMXP_USE_COLLECTIVES

  ierr = MPI_Bcast(SBUF, SCOUNT, T2MPI<T>::type, ROOT, COMM);

#else

  switch(top) {
    case HPLMXP_1RING_M:
      ierr = HPLMXP_bcast_1rinM(SBUF, SCOUNT, ROOT, COMM);
      break;
    case HPLMXP_1RING:
      ierr = HPLMXP_bcast_1ring(SBUF, SCOUNT, ROOT, COMM);
      break;
    case HPLMXP_2RING_M:
      ierr = HPLMXP_bcast_2rinM(SBUF, SCOUNT, ROOT, COMM);
      break;
    case HPLMXP_2RING:
      ierr = HPLMXP_bcast_2ring(SBUF, SCOUNT, ROOT, COMM);
      break;
    case HPLMXP_BLONG_M:
      ierr = HPLMXP_bcast_blonM(SBUF, SCOUNT, ROOT, COMM);
      break;
    case HPLMXP_BLONG:
      ierr = HPLMXP_bcast_blong(SBUF, SCOUNT, ROOT, COMM);
      break;
    case HPLMXP_COLL:
      ierr = MPI_Bcast(SBUF, SCOUNT, T2MPI<T>::type, ROOT, COMM);
      break;
    default: ierr = HPLMXP_FAILURE;
  }

#endif

  // roctxRangePop();

  return ((ierr == MPI_SUCCESS ? HPLMXP_SUCCESS : HPLMXP_FAILURE));
}

template int HPLMXP_bcast(double*,
                          size_t,
                          int,
                          HPLMXP_Comm,
                          hipStream_t,
                          HPLMXP_T_TOP);
template int HPLMXP_bcast(float*,
                          size_t,
                          int,
                          HPLMXP_Comm,
                          hipStream_t,
                          HPLMXP_T_TOP);
template int HPLMXP_bcast(__half*,
                          size_t,
                          int,
                          HPLMXP_Comm,
                          hipStream_t,
                          HPLMXP_T_TOP);
