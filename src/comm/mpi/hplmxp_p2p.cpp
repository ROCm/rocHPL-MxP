/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    (C) 2026 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

/*
 * MPI backend: point-to-point on device buffers (GPU-aware MPI). A single
 * tag is used for all messages, so matching is FIFO per (comm, peer) as
 * the contract requires. Inside a group, operations are posted
 * non-blocking and completed together at HPLMXP_group_end.
 *
 * HPLMXP_isend/HPLMXP_irecv/HPLMXP_wait are MPI_Isend/MPI_Irecv/MPI_Wait.
 * Stream ordering is done on the host: a post first synchronizes STREAM,
 * and the host-blocking wait completes the operation before any later work
 * is enqueued on STREAM.
 */

#include "hplmxp_comm_mpi.hpp"
#include <climits>
#include <vector>

#define HPLMXP_P2P_TAG 0

static int                      group_depth = 0;
static std::vector<MPI_Request> group_reqs;

template <typename T>
int HPLMXP_send(const T*    SBUF,
                size_t      COUNT,
                int         PEER,
                HPLMXP_Comm COMM,
                hipStream_t STREAM) {
  if(PEER < 0) return (HPLMXP_SUCCESS);
  HPLMXP_mpi_count(COUNT, "HPLMXP_send");

  int ierr;
  if(group_depth > 0) {
    MPI_Request req;
    ierr = MPI_Isend(SBUF,
                     (int)COUNT,
                     T2MPI<T>::type,
                     PEER,
                     HPLMXP_P2P_TAG,
                     HPLMXP_mpi(COMM),
                     &req);
    group_reqs.push_back(req);
  } else {
    ierr = MPI_Send(SBUF,
                    (int)COUNT,
                    T2MPI<T>::type,
                    PEER,
                    HPLMXP_P2P_TAG,
                    HPLMXP_mpi(COMM));
  }
  return ((ierr == MPI_SUCCESS ? HPLMXP_SUCCESS : HPLMXP_FAILURE));
}

template <typename T>
int HPLMXP_recv(T*          RBUF,
                size_t      COUNT,
                int         PEER,
                HPLMXP_Comm COMM,
                hipStream_t STREAM) {
  if(PEER < 0) return (HPLMXP_SUCCESS);
  HPLMXP_mpi_count(COUNT, "HPLMXP_recv");

  int ierr;
  if(group_depth > 0) {
    MPI_Request req;
    ierr = MPI_Irecv(RBUF,
                     (int)COUNT,
                     T2MPI<T>::type,
                     PEER,
                     HPLMXP_P2P_TAG,
                     HPLMXP_mpi(COMM),
                     &req);
    group_reqs.push_back(req);
  } else {
    ierr = MPI_Recv(RBUF,
                    (int)COUNT,
                    T2MPI<T>::type,
                    PEER,
                    HPLMXP_P2P_TAG,
                    HPLMXP_mpi(COMM),
                    MPI_STATUS_IGNORE);
  }
  return ((ierr == MPI_SUCCESS ? HPLMXP_SUCCESS : HPLMXP_FAILURE));
}

void HPLMXP_group_start() { group_depth++; }

void HPLMXP_group_end() {
  if(--group_depth > 0) return;
  if(!group_reqs.empty()) {
    MPI_Waitall((int)group_reqs.size(), group_reqs.data(), MPI_STATUSES_IGNORE);
    group_reqs.clear();
  }
}

template int HPLMXP_send(const double*, size_t, int, HPLMXP_Comm, hipStream_t);
template int HPLMXP_recv(double*, size_t, int, HPLMXP_Comm, hipStream_t);

static std::vector<MPI_Request> reqs; /* request slots */
static std::vector<int>         reqs_free;

static HPLMXP_Request p2p_slot() {
  if(reqs_free.empty()) {
    reqs.push_back(MPI_REQUEST_NULL);
    return (HPLMXP_Request)reqs.size() - 1;
  }
  HPLMXP_Request r = reqs_free.back();
  reqs_free.pop_back();
  return r;
}

template <typename T>
int HPLMXP_isend(const T*        SBUF,
                 size_t          COUNT,
                 int             PEER,
                 HPLMXP_Comm     COMM,
                 hipStream_t     STREAM,
                 HPLMXP_Request* REQ) {
  *REQ = HPLMXP_REQUEST_NULL;
  if(PEER < 0) return (HPLMXP_SUCCESS);
  HPLMXP_mpi_count(COUNT, "HPLMXP_isend");
  if(STREAM != nullptr) HIP_CHECK(hipStreamSynchronize(STREAM));

  HPLMXP_Request r    = p2p_slot();
  int            ierr = MPI_Isend(SBUF,
                       (int)COUNT,
                       T2MPI<T>::type,
                       PEER,
                       HPLMXP_P2P_TAG,
                       HPLMXP_mpi(COMM),
                       &reqs[r]);
  *REQ                = r;
  return ((ierr == MPI_SUCCESS ? HPLMXP_SUCCESS : HPLMXP_FAILURE));
}

template <typename T>
int HPLMXP_irecv(T*              RBUF,
                 size_t          COUNT,
                 int             PEER,
                 HPLMXP_Comm     COMM,
                 hipStream_t     STREAM,
                 HPLMXP_Request* REQ) {
  *REQ = HPLMXP_REQUEST_NULL;
  if(PEER < 0) return (HPLMXP_SUCCESS);
  HPLMXP_mpi_count(COUNT, "HPLMXP_irecv");
  if(STREAM != nullptr) HIP_CHECK(hipStreamSynchronize(STREAM));

  HPLMXP_Request r    = p2p_slot();
  int            ierr = MPI_Irecv(RBUF,
                       (int)COUNT,
                       T2MPI<T>::type,
                       PEER,
                       HPLMXP_P2P_TAG,
                       HPLMXP_mpi(COMM),
                       &reqs[r]);
  *REQ                = r;
  return ((ierr == MPI_SUCCESS ? HPLMXP_SUCCESS : HPLMXP_FAILURE));
}

void HPLMXP_wait(HPLMXP_Request* REQ, hipStream_t STREAM) {
  if(*REQ == HPLMXP_REQUEST_NULL) return;
  MPI_Wait(&reqs[*REQ], MPI_STATUS_IGNORE);
  reqs_free.push_back(*REQ);
  *REQ = HPLMXP_REQUEST_NULL;
}

template int HPLMXP_isend(const double*,
                          size_t,
                          int,
                          HPLMXP_Comm,
                          hipStream_t,
                          HPLMXP_Request*);
template int HPLMXP_irecv(double*,
                          size_t,
                          int,
                          HPLMXP_Comm,
                          hipStream_t,
                          HPLMXP_Request*);
