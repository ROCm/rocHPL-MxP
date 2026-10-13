/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    (C) 2026 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */
#ifndef HPLMXP_COMM_RCCL_HPP
#define HPLMXP_COMM_RCCL_HPP
/*
 * Private to the RCCL backend (src/comm/rccl/). MPI-free: no mpi.h here
 * or anywhere in an RCCL build.
 */
#include "hplmxp.hpp"
#include <rccl/rccl.h>

struct HPLMXP_CommImpl {
  ncclComm_t  nccl;
  hipStream_t stream; /* private high-priority stream for this comm */
  int         rank;
  int         size;
  char        name[48]; /* for messages, e.g. "world" or "split 2 of world" */
};

/*
 * Runs an RCCL call under the watchdog (an enqueue can block, e.g. while
 * connecting to a peer) and aborts the job if it fails. Statement only.
 */
#define HPLMXP_NCCL_CHECK(cmd)                                             \
  do {                                                                     \
    HPLMXP_rccl_watch_begin(#cmd, HPLMXP_COMM_NULL, HPLMXP_RCCL_PROGRESS); \
    ncclResult_t hplmxp_nccl_res_ = (cmd);                                 \
    HPLMXP_rccl_watch_end();                                               \
    HPLMXP_rccl_check(hplmxp_nccl_res_, #cmd, __FILE__, __LINE__);         \
  } while(0)

/* Prints the failing call and aborts the job (hplmxp_rccl_abort.cpp). */
void HPLMXP_rccl_check(ncclResult_t      res,
                       const char* const call,
                       const char* const file,
                       const int         line);

/*
 * Prints "HPLMXP RCCL rank R: <message>" on stderr and tears the job down
 * like HPLMXP_comm_abort(CODE) (hplmxp_rccl_abort.cpp).
 */
[[noreturn]] void HPLMXP_rccl_fatal(int code, const char* fmt, ...)
    __attribute__((format(printf, 2, 3)));

/*
 * Blocks until all work queued on COMM->stream has completed
 * (hplmxp_rccl_init.cpp). Every host-blocking operation of the backend
 * goes through here, so error/timeout detection lives in one place.
 */
void HPLMXP_rccl_wait(HPLMXP_Comm COMM);

/*
 * Watchdog (hplmxp_rccl_init.cpp). Communicators are created blocking; a
 * background thread polls ncclCommGetAsyncError on every live communicator
 * and enforces a deadline on the blocking call the main thread has marked
 * with watch_begin/watch_end. On an asynchronous error or a missed
 * deadline it prints one line and aborts the job (ncclCommAbort + _exit),
 * so srun --kill-on-bad-exit takes the other ranks down.
 *
 *   HPLMXP_RCCL_INIT_TIMEOUT  s, deadline for HPLMXP_RCCL_SETUP calls
 *                             (bootstrap, split, destroy); default 300
 *   HPLMXP_RCCL_TIMEOUT       s, deadline for HPLMXP_RCCL_PROGRESS calls
 *                             (waiting for a collective or p2p); default 300
 *   HPLMXP_RCCL_WATCHDOG_MS   polling interval; default 100. 0 disables the
 *                             watchdog thread (and both deadlines).
 * A timeout of 0 disables that deadline. COMM may be HPLMXP_COMM_NULL.
 * Calls nest; the outermost one sets the deadline and the message. Main
 * thread only. watch_end returns the seconds since the matching begin.
 *
 *   HPLMXP_RCCL_WATCHDOG_VERBOSE=1  each rank reports its longest wait at
 *                                   finalize (to choose HPLMXP_RCCL_TIMEOUT)
 */
enum HPLMXP_rccl_phase { HPLMXP_RCCL_SETUP, HPLMXP_RCCL_PROGRESS };
void   HPLMXP_rccl_watch_begin(const char*       what,
                               HPLMXP_Comm       COMM,
                               HPLMXP_rccl_phase phase);
double HPLMXP_rccl_watch_end();

/* Aborts every live communicator; used by the teardown path only. */
void HPLMXP_rccl_abort_comms();

/* True once some thread has started tearing the job down. */
bool HPLMXP_rccl_aborting();

/* Device scratch of at least BYTES for staging host collectives. */
void* HPLMXP_rccl_scratch(size_t bytes);

/* World rank of this process (for messages), -1 before init. */
int HPLMXP_rccl_world_rank();

/* Live communicators, for HPLMXP_comm_abort. */
int         HPLMXP_rccl_num_comms();
HPLMXP_Comm HPLMXP_rccl_comm(int i);

template <typename T>
struct HPLMXP_nccl_type {};
template <>
struct HPLMXP_nccl_type<int> {
  static constexpr ncclDataType_t value = ncclInt32;
};
template <>
struct HPLMXP_nccl_type<__half> {
  static constexpr ncclDataType_t value = ncclFloat16;
};
template <>
struct HPLMXP_nccl_type<float> {
  static constexpr ncclDataType_t value = ncclFloat32;
};
template <>
struct HPLMXP_nccl_type<double> {
  static constexpr ncclDataType_t value = ncclFloat64;
};

static inline ncclRedOp_t HPLMXP_nccl_op(HPLMXP_T_OP OP) {
  switch(OP) {
    case HPLMXP_MAX: return ncclMax;
    case HPLMXP_MIN: return ncclMin;
    default: return ncclSum;
  }
}

#endif
