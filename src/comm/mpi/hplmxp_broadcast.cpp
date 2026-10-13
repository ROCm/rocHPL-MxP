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

template <typename T>
int HPLMXP_broadcast_host(T*          BUFFER,
                          const int   COUNT,
                          const int   ROOT,
                          HPLMXP_Comm COMM) {
  /*
   * Purpose
   * =======
   *
   * HPLMXP_broadcast broadcasts  a message from the process with rank ROOT to
   * all processes in the group.
   *
   * Arguments
   * =========
   *
   * BUFFER  (local input/output)          void *
   *         On entry,  BUFFER  points to  the  buffer to be broadcast. On
   *         exit, this array contains the broadcast data and is identical
   *         on all processes in the group.
   *
   * COUNT   (global input)                const int
   *         On entry,  COUNT  indicates the number of entries in  BUFFER.
   *         COUNT must be at least zero.
   *
   * ROOT    (global input)                const int
   *         On entry, ROOT is the coordinate of the source process.
   *
   * COMM    (global/local input)          HPLMXP_Comm
   *         The communicator identifying the process collection.
   *
   * ---------------------------------------------------------------------
   */

  int ierr = MPI_Bcast(BUFFER, COUNT, T2MPI<T>::type, ROOT, HPLMXP_mpi(COMM));

  return ((ierr == MPI_SUCCESS ? HPLMXP_SUCCESS : HPLMXP_FAILURE));
}

template int HPLMXP_broadcast_host(double*, const int, const int, HPLMXP_Comm);
template int HPLMXP_broadcast_host(float*, const int, const int, HPLMXP_Comm);
template int HPLMXP_broadcast_host(int*, const int, const int, HPLMXP_Comm);
