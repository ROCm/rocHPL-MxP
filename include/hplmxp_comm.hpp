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
 * Communication backend contract.
 *
 * Everything outside src/comm/<backend>/ talks to the network through
 * this header only. Two backends implement it, selected at configure
 * time with -DHPLMXP_COMM_BACKEND=MPI|RCCL (one directory is compiled):
 *
 *   src/comm/mpi/   MPI (GPU-aware), the upstream reference behaviour
 *   src/comm/rccl/  RCCL, MPI-free: ranks from the launcher environment,
 *                   bootstrap through NCCL_COMM_ID
 *
 * This header must not include mpi.h or rccl.h.
 * ---------------------------------------------------------------------
 */
#include <stddef.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_fp16.h>

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
  HPLMXP_COLL    = 407, /* library collective (MPI_Bcast / ncclBroadcast) */
} HPLMXP_T_TOP;

typedef enum { HPLMXP_SUM, HPLMXP_MAX, HPLMXP_MIN } HPLMXP_T_OP;

/* Opaque communicator handle; wraps an MPI_Comm or an ncclComm_t. */
typedef struct HPLMXP_CommImpl* HPLMXP_Comm;
#define HPLMXP_COMM_NULL ((HPLMXP_Comm)0)

/*
 * ---------------------------------------------------------------------
 * #define macro constants
 * ---------------------------------------------------------------------
 */
#define HPLMXP_FAILURE 0
#define HPLMXP_SUCCESS 1
/*
 * ---------------------------------------------------------------------
 * Control plane
 * ---------------------------------------------------------------------
 *
 * HPLMXP_comm_world_init   Initializes the backend and returns the world
 *                          communicator. RCCL: reads rank/size/local rank
 *                          from the launcher environment, binds the GPU
 *                          (local rank % device count) and creates the
 *                          world communicator from NCCL_COMM_ID.
 * HPLMXP_comm_world        The world communicator (HPLMXP_COMM_NULL before
 *                          world_init and after finalize).
 * HPLMXP_comm_local_rank   Rank within the node (valid after world_init).
 * HPLMXP_comm_local_size   Number of ranks on this node.
 * HPLMXP_comm_device       HIP device bound by the backend, or -1 if the
 *                          backend does not bind one (MPI).
 * HPLMXP_comm_split        Collective over COMM. color < 0 means "not a
 *                          member": *OUT is set to HPLMXP_COMM_NULL.
 * HPLMXP_comm_abort        Tears the whole job down; never returns.
 * HPLMXP_wtime             Wall clock in seconds.
 * HPLMXP_comm_sync         Host wait for STREAM (the whole device if STREAM
 *                          is null). Use it instead of hipStreamSynchronize/
 *                          hipDeviceSynchronize wherever the wait can cover
 *                          communication (stream-ordered isend/irecv, or
 *                          work queued behind it): the RCCL backend puts it
 *                          under the HPLMXP_RCCL_TIMEOUT deadline, with WHAT
 *                          (a string literal) naming the site.
 */
int         HPLMXP_comm_world_init(int* argc, char*** argv, HPLMXP_Comm* world);
HPLMXP_Comm HPLMXP_comm_world();
int         HPLMXP_comm_local_rank();
int         HPLMXP_comm_local_size();
int         HPLMXP_comm_device();
int  HPLMXP_comm_split(HPLMXP_Comm COMM, int color, int key, HPLMXP_Comm* OUT);
int  HPLMXP_comm_rank(HPLMXP_Comm COMM);
int  HPLMXP_comm_size(HPLMXP_Comm COMM);
void HPLMXP_comm_free(HPLMXP_Comm* COMM);
void HPLMXP_comm_finalize();
[[noreturn]] void HPLMXP_comm_abort(HPLMXP_Comm COMM, int code);
double            HPLMXP_wtime();
void              HPLMXP_comm_sync(hipStream_t STREAM, const char* WHAT);

/*
 * ---------------------------------------------------------------------
 * Device-buffer collectives
 * ---------------------------------------------------------------------
 *
 * Host-blocking: on return the operation is complete on this rank. The
 * caller guarantees that the source data is ready before the call (as
 * upstream does with event/stream synchronizations); the backend does not
 * synchronize STREAM. STREAM is the stream that consumes the result. The
 * MPI backend ignores it; the RCCL backend currently runs on a private
 * high-priority stream per communicator so the operation can overlap
 * work already queued on STREAM (e.g. the trailing update in pgetrf).
 *
 * HPLMXP_bcast types: __half, float, double. HPLMXP_all_reduce types:
 * int, float, double.
 */
template <typename T>
int HPLMXP_bcast(T*           SBUF,
                 size_t       COUNT,
                 int          ROOT,
                 HPLMXP_Comm  COMM,
                 hipStream_t  STREAM,
                 HPLMXP_T_TOP TOP);

template <typename T>
int HPLMXP_all_reduce(T*                BUFFER,
                      size_t            COUNT,
                      const HPLMXP_T_OP OP,
                      HPLMXP_Comm       COMM,
                      hipStream_t       STREAM);

/*
 * ---------------------------------------------------------------------
 * Host-buffer collectives (small scalars and arrays). Types: int, float,
 * double. The RCCL backend stages them through device memory.
 * ---------------------------------------------------------------------
 */
template <typename T>
int HPLMXP_all_reduce_host(T*                BUFFER,
                           const int         COUNT,
                           const HPLMXP_T_OP OP,
                           HPLMXP_Comm       COMM);

template <typename T>
int HPLMXP_broadcast_host(T*          BUFFER,
                          const int   COUNT,
                          const int   ROOT,
                          HPLMXP_Comm COMM);

int HPLMXP_barrier(HPLMXP_Comm COMM);

/*
 * ---------------------------------------------------------------------
 * Point-to-point on device buffers. Types: double.
 * ---------------------------------------------------------------------
 *
 * PEER < 0 makes the call a no-op (replaces MPI_PROC_NULL). Messages
 * between a (COMM, PEER) pair are matched in FIFO order; there are no
 * tags, so a caller needing two independent logical channels uses two
 * communicators. Inside a group_start/group_end pair, sends and receives
 * are posted together and complete at group_end; outside a group each
 * call is host-blocking.
 */
template <typename T>
int HPLMXP_send(const T*    SBUF,
                size_t      COUNT,
                int         PEER,
                HPLMXP_Comm COMM,
                hipStream_t STREAM);

template <typename T>
int HPLMXP_recv(T*          RBUF,
                size_t      COUNT,
                int         PEER,
                HPLMXP_Comm COMM,
                hipStream_t STREAM);

void HPLMXP_group_start();
void HPLMXP_group_end();

/*
 * Non-blocking, stream-ordered point-to-point on device buffers. Types:
 * double.
 *
 * HPLMXP_isend / HPLMXP_irecv post an operation and return a request. The
 * operation touches its buffer only after all work enqueued on STREAM
 * before the call. A null STREAM orders it against no stream: the caller
 * guarantees the buffer is ready, e.g. it was filled by an earlier receive
 * on the same COMM whose request has already been waited on. PEER < 0
 * makes the call a no-op and returns HPLMXP_REQUEST_NULL. Matching is FIFO
 * per (COMM, PEER). Operations on one COMM may execute one at a time in
 * call order (RCCL), and a send or receive does not complete before its
 * peer posts the matching operation, so the call order on every COMM must
 * be deadlock-free under that serialization; independent channels use
 * separate communicators. Not for use inside a group_start/group_end pair,
 * nor with HPLMXP_send/HPLMXP_recv on a COMM that has requests in flight.
 *
 * HPLMXP_wait: work enqueued on STREAM after the call sees the operation
 * completed (its buffer may then be reused). RCCL does not block the host;
 * MPI blocks in MPI_Wait. Every request must be waited on. Resets *REQ.
 * Under RCCL the host first sees completion at its next wait on STREAM or
 * the device; make that wait an HPLMXP_comm_sync so a lost peer hits the
 * progress deadline.
 */
typedef int HPLMXP_Request;
#define HPLMXP_REQUEST_NULL (-1)

template <typename T>
int HPLMXP_isend(const T*        SBUF,
                 size_t          COUNT,
                 int             PEER,
                 HPLMXP_Comm     COMM,
                 hipStream_t     STREAM,
                 HPLMXP_Request* REQ);

template <typename T>
int HPLMXP_irecv(T*              RBUF,
                 size_t          COUNT,
                 int             PEER,
                 HPLMXP_Comm     COMM,
                 hipStream_t     STREAM,
                 HPLMXP_Request* REQ);

void HPLMXP_wait(HPLMXP_Request* REQ, hipStream_t STREAM);

#endif
/*
 * End of hplmxp_comm.hpp
 */
