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

template <typename T>
int HPLMXP_all_reduce(T*                BUFFER,
                      const int         COUNT,
                      const HPLMXP_T_OP OP,
                      MPI_Comm          COMM) {
  /*
   * Purpose
   * =======
   *
   * HPL_all_reduce performs   a   global   reduce  operation  across  all
   * processes of a group leaving the results on all processes.
   *
   * Arguments
   * =========
   *
   * BUFFER  (local input/global output)   T *
   *         On entry,  BUFFER  points to  the  buffer to be combined.  On
   *         exit, this array contains the combined data and  is identical
   *         on all processes in the group.
   *
   * COUNT   (global input)                const int
   *         On entry,  COUNT  indicates the number of entries in  BUFFER.
   *         COUNT must be at least zero.
   *
   * OP      (global input)                const HPL_T_OP
   *         On entry, OP is a pointer to the local combine function.
   *
   * COMM    (global/local input)          MPI_Comm
   *         The MPI communicator identifying the process collection.
   *
   * ---------------------------------------------------------------------
   */

  int ierr =
      MPI_Allreduce(MPI_IN_PLACE, BUFFER, COUNT, T2MPI<T>::type, OP, COMM);

  return ((ierr == MPI_SUCCESS ? HPLMXP_SUCCESS : HPLMXP_FAILURE));
}

template int HPLMXP_all_reduce(double*           BUFFER,
                               const int         COUNT,
                               const HPLMXP_T_OP OP,
                               MPI_Comm          COMM);

template int HPLMXP_all_reduce(float*            BUFFER,
                               const int         COUNT,
                               const HPLMXP_T_OP OP,
                               MPI_Comm          COMM);

template int HPLMXP_all_reduce(int*              BUFFER,
                               const int         COUNT,
                               const HPLMXP_T_OP OP,
                               MPI_Comm          COMM);
