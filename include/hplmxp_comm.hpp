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
#ifndef HPLMXP_COMM_HPP
#define HPLMXP_COMM_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hplmxp_pmisc.hpp"

/*
 * ---------------------------------------------------------------------
 * #typedefs and data structures
 * ---------------------------------------------------------------------
 */
typedef enum {
  HPLMXP_1RING   = 401, /* Unidirectional ring */
  HPLMXP_1RING_M = 402, /* Unidirectional ring (modified) */
  HPLMXP_2RING   = 403, /* Bidirectional ring */
  HPLMXP_2RING_M = 404, /* Bidirectional ring (modified) */
  HPLMXP_BLONG   = 405, /* long broadcast */
  HPLMXP_BLONG_M = 406, /* long broadcast (modified) */
} HPLMXP_T_TOP;

typedef MPI_Op HPLMXP_T_OP;

#define HPLMXP_SUM MPI_SUM
#define HPLMXP_MAX MPI_MAX
#define HPLMXP_MIN MPI_MIN

template <typename T>
struct Mpi_type_wrappe {};

template <>
struct Mpi_type_wrappe<int> {
  operator MPI_Datatype() { return MPI_INT; }
};

template <>
struct Mpi_type_wrappe<hipblaslt_f8_fnuz> {
  operator MPI_Datatype() { return MPI_CHAR; }
};

template <>
struct Mpi_type_wrappe<__half> {
  operator MPI_Datatype() { return MPI_SHORT; }
};

template <>
struct Mpi_type_wrappe<float> {
  operator MPI_Datatype() { return MPI_FLOAT; }
};

template <>
struct Mpi_type_wrappe<double> {
  operator MPI_Datatype() { return MPI_DOUBLE; }
};

template <typename F>
struct T2MPI {
  static Mpi_type_wrappe<F> type;
};

template <typename F>
Mpi_type_wrappe<F> T2MPI<F>::type;

/*
 * ---------------------------------------------------------------------
 * #define macro constants
 * ---------------------------------------------------------------------
 */
#define HPLMXP_FAILURE 0
#define HPLMXP_SUCCESS 1
/*
 * ---------------------------------------------------------------------
 * comm function prototypes
 * ---------------------------------------------------------------------
 */

template <typename T>
int HPLMXP_bcast(T*, int, int, MPI_Comm, HPLMXP_T_TOP top);
template <typename T>
int HPLMXP_bcast_1ring(T* SBUF, int SCOUNT, int ROOT, MPI_Comm COMM);
template <typename T>
int HPLMXP_bcast_1rinM(T* SBUF, int SCOUNT, int ROOT, MPI_Comm COMM);
template <typename T>
int HPLMXP_bcast_2ring(T* SBUF, int SCOUNT, int ROOT, MPI_Comm COMM);
template <typename T>
int HPLMXP_bcast_2rinM(T* SBUF, int SCOUNT, int ROOT, MPI_Comm COMM);
template <typename T>
int HPLMXP_bcast_blong(T* SBUF, int SCOUNT, int ROOT, MPI_Comm COMM);
template <typename T>
int HPLMXP_bcast_blonM(T* SBUF, int SCOUNT, int ROOT, MPI_Comm COMM);

template <typename T>
int HPLMXP_all_reduce(T*, const int, const HPLMXP_T_OP, MPI_Comm);

template <typename T>
int HPLMXP_broadcast(T* BUFFER, const int COUNT, const int ROOT, MPI_Comm COMM);

int HPLMXP_barrier(MPI_Comm COMM);

#endif
/*
 * End of hpl_comm.hpp
 */
