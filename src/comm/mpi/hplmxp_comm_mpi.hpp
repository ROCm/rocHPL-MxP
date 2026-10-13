/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    (C) 2026 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */
#ifndef HPLMXP_COMM_MPI_HPP
#define HPLMXP_COMM_MPI_HPP
/*
 * Private to the MPI backend (src/comm/mpi/). The rest of the code base
 * only sees the opaque HPLMXP_Comm from hplmxp_comm.hpp.
 */
#include "hplmxp.hpp"
#include "mpi.h"
#include <climits>
#include <cstdio>

struct HPLMXP_CommImpl {
  MPI_Comm comm;
};

static inline MPI_Comm HPLMXP_mpi(HPLMXP_Comm COMM) { return COMM->comm; }

static inline MPI_Op HPLMXP_mpi_op(HPLMXP_T_OP OP) {
  switch(OP) {
    case HPLMXP_MAX: return MPI_MAX;
    case HPLMXP_MIN: return MPI_MIN;
    default: return MPI_SUM;
  }
}

/*
 * MPI counts are int. A larger COUNT aborts the job with a message: callers
 * do not check return codes, so returning HPLMXP_FAILURE would go unnoticed.
 */
static inline int HPLMXP_mpi_count(size_t COUNT, const char* WHAT) {
  if(COUNT > (size_t)INT_MAX) {
    fprintf(stderr,
            "HPLMXP MPI backend: %s of %zu elements exceeds the MPI count "
            "limit (INT_MAX)\n",
            WHAT,
            COUNT);
    HPLMXP_comm_abort(HPLMXP_COMM_NULL, 1);
  }
  return (int)COUNT;
}

template <typename T>
struct Mpi_type_wrappe {};

template <>
struct Mpi_type_wrappe<int> {
  operator MPI_Datatype() { return MPI_INT; }
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

/* Hand-rolled broadcast topologies (upstream rocHPL-MxP) */
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

#endif
