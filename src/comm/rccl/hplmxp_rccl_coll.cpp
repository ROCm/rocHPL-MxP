/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    (C) 2026 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

/*
 * RCCL backend: device-buffer collectives.
 *
 * Host-blocking (hplmxp_comm.hpp contract). The operation is enqueued on
 * the communicator's private high-priority stream, so it runs alongside
 * work already queued on the caller's stream (the trailing update in
 * pgetrf), and the host then waits on that stream through
 * HPLMXP_rccl_wait. Callers make the source data ready before the call;
 * STREAM is not synchronized. The broadcast topology argument is ignored:
 * RCCL picks its own algorithm.
 *
 * Environment (read once):
 *   HPLMXP_RCCL_MAX_COUNT  elements per RCCL call (default 2^30); larger
 *                          operations are split into contiguous chunks
 */

#include "hplmxp_comm_rccl.hpp"
#include <algorithm>
#include <cstdlib>

namespace {

/* Elements per RCCL call, from HPLMXP_RCCL_MAX_COUNT */
size_t max_count() {
  static const size_t count = [] {
    const size_t fallback = (size_t)1 << 30;
    const char*  v        = getenv("HPLMXP_RCCL_MAX_COUNT");
    if(v == nullptr || v[0] == '\0') return fallback;
    char*     end = nullptr;
    long long x   = strtoll(v, &end, 10);
    if(end == v || *end != '\0' || x < 1) {
      fprintf(stderr,
              "HPLMXP RCCL: HPLMXP_RCCL_MAX_COUNT=%s is not a positive "
              "integer\n",
              v);
      HPLMXP_comm_abort(HPLMXP_COMM_NULL, 1);
    }
    return (size_t)x;
  }();
  return count;
}

/* Bad arguments are programming errors: report with context and abort */
void coll_args(const char* op, size_t count, int root, HPLMXP_Comm COMM) {
  if(COMM == HPLMXP_COMM_NULL) {
    fprintf(stderr,
            "HPLMXP RCCL: %s of %zu elements on rank %d called with a null "
            "communicator\n",
            op,
            count,
            HPLMXP_rccl_world_rank());
    HPLMXP_comm_abort(HPLMXP_COMM_NULL, 1);
  }
  if(root < 0 || root >= COMM->size) {
    fprintf(stderr,
            "HPLMXP RCCL: %s of %zu elements on rank %d: root %d outside "
            "communicator of size %d\n",
            op,
            count,
            HPLMXP_rccl_world_rank(),
            root,
            COMM->size);
    HPLMXP_comm_abort(HPLMXP_COMM_NULL, 1);
  }
}

} // namespace

template <typename T>
int HPLMXP_bcast(T*           SBUF,
                 size_t       COUNT,
                 int          ROOT,
                 HPLMXP_Comm  COMM,
                 hipStream_t  STREAM,
                 HPLMXP_T_TOP TOP) {
  coll_args("bcast", COUNT, ROOT, COMM);
  if(COUNT == 0 || COMM->size <= 1) return (HPLMXP_SUCCESS);

  const size_t chunk = max_count();
  for(size_t off = 0; off < COUNT; off += chunk) {
    const size_t n = std::min(chunk, COUNT - off);
    HPLMXP_NCCL_CHECK(ncclBroadcast(SBUF + off,
                                    SBUF + off,
                                    n,
                                    HPLMXP_nccl_type<T>::value,
                                    ROOT,
                                    COMM->nccl,
                                    COMM->stream));
  }
  HPLMXP_rccl_wait(COMM);
  return (HPLMXP_SUCCESS);
}

template <typename T>
int HPLMXP_all_reduce(T*                BUFFER,
                      size_t            COUNT,
                      const HPLMXP_T_OP OP,
                      HPLMXP_Comm       COMM,
                      hipStream_t       STREAM) {
  coll_args("all_reduce", COUNT, 0, COMM);
  if(COUNT == 0 || COMM->size <= 1) return (HPLMXP_SUCCESS);

  const size_t chunk = max_count();
  for(size_t off = 0; off < COUNT; off += chunk) {
    const size_t n = std::min(chunk, COUNT - off);
    HPLMXP_NCCL_CHECK(ncclAllReduce(BUFFER + off,
                                    BUFFER + off,
                                    n,
                                    HPLMXP_nccl_type<T>::value,
                                    HPLMXP_nccl_op(OP),
                                    COMM->nccl,
                                    COMM->stream));
  }
  HPLMXP_rccl_wait(COMM);
  return (HPLMXP_SUCCESS);
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

template int HPLMXP_all_reduce(double*,
                               size_t,
                               const HPLMXP_T_OP,
                               HPLMXP_Comm,
                               hipStream_t);
template int HPLMXP_all_reduce(float*,
                               size_t,
                               const HPLMXP_T_OP,
                               HPLMXP_Comm,
                               hipStream_t);
template int HPLMXP_all_reduce(int*,
                               size_t,
                               const HPLMXP_T_OP,
                               HPLMXP_Comm,
                               hipStream_t);
