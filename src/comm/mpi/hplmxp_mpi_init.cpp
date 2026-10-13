/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    (C) 2026 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

/*
 * MPI backend: control plane. Thin wrappers around the MPI calls that
 * upstream rocHPL-MxP made directly from pdriver, pinfo, grid_* and the
 * pauxil error routines.
 */

#include "hplmxp_comm_mpi.hpp"

static HPLMXP_CommImpl world_impl = {MPI_COMM_NULL};
static int             local_rank = -1;
static int             local_size = -1;

int HPLMXP_comm_world_init(int* argc, char*** argv, HPLMXP_Comm* world) {
  MPI_Init(argc, argv);
  world_impl.comm = MPI_COMM_WORLD;
  *world          = &world_impl;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /*Node-local rank and size*/
  MPI_Comm nodeComm;
  MPI_Comm_split_type(
      MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
  MPI_Comm_rank(nodeComm, &local_rank);
  MPI_Comm_size(nodeComm, &local_size);
  MPI_Comm_free(&nodeComm);

  return HPLMXP_SUCCESS;
}

int HPLMXP_comm_local_rank() { return local_rank; }

int HPLMXP_comm_local_size() { return local_size; }

int HPLMXP_comm_device() { return -1; }

HPLMXP_Comm HPLMXP_comm_world() {
  return (world_impl.comm == MPI_COMM_NULL ? HPLMXP_COMM_NULL : &world_impl);
}

int HPLMXP_comm_split(HPLMXP_Comm COMM, int color, int key, HPLMXP_Comm* OUT) {
  MPI_Comm newcomm;
  int      ierr = MPI_Comm_split(
      HPLMXP_mpi(COMM), (color < 0 ? MPI_UNDEFINED : color), key, &newcomm);

  if(newcomm == MPI_COMM_NULL) {
    *OUT = HPLMXP_COMM_NULL;
  } else {
    *OUT         = new HPLMXP_CommImpl;
    (*OUT)->comm = newcomm;
  }
  return ((ierr == MPI_SUCCESS ? HPLMXP_SUCCESS : HPLMXP_FAILURE));
}

int HPLMXP_comm_rank(HPLMXP_Comm COMM) {
  int rank;
  MPI_Comm_rank(HPLMXP_mpi(COMM), &rank);
  return rank;
}

int HPLMXP_comm_size(HPLMXP_Comm COMM) {
  int size;
  MPI_Comm_size(HPLMXP_mpi(COMM), &size);
  return size;
}

void HPLMXP_comm_free(HPLMXP_Comm* COMM) {
  if(*COMM == HPLMXP_COMM_NULL || *COMM == &world_impl) return;
  MPI_Comm_free(&((*COMM)->comm));
  delete *COMM;
  *COMM = HPLMXP_COMM_NULL;
}

void HPLMXP_comm_finalize() { MPI_Finalize(); }

void HPLMXP_comm_abort(HPLMXP_Comm COMM, int code) {
  MPI_Abort((COMM == HPLMXP_COMM_NULL ? MPI_COMM_WORLD : HPLMXP_mpi(COMM)),
            code);
  exit(code);
}

double HPLMXP_wtime() { return (MPI_Wtime()); }

void HPLMXP_comm_sync(hipStream_t STREAM, const char* WHAT) {
  HIP_CHECK(STREAM ? hipStreamSynchronize(STREAM) : hipDeviceSynchronize());
}
