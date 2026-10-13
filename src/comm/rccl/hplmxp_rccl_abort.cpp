/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    (C) 2026 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

/*
 * RCCL backend: error handling and job teardown. There is no MPI_Abort;
 * this process aborts its communicators and exits non-zero, and the
 * launcher takes the rest of the job down (srun --kill-on-bad-exit=1).
 *
 * The first thread to abort wins; any other thread that fails or aborts
 * meanwhile (typically the main thread, whose operation the abort has just
 * broken) parks until the process exits, so only the root cause is
 * reported. ncclCommAbort can block on a wedged GPU or network, so the
 * teardown is bounded by HPLMXP_RCCL_ABORT_TIMEOUT seconds (default 10)
 * after which the process exits regardless.
 */

#include "hplmxp_comm_rccl.hpp"
#include <atomic>
#include <climits>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

static std::atomic<bool> aborting{false};
static volatile int      abort_code = 1;

bool HPLMXP_rccl_aborting() { return aborting.load(); }

[[noreturn]] static void park() {
  for(;;) pause();
}

static void abort_timed_out(int) { _exit(abort_code); }

void HPLMXP_rccl_check(ncclResult_t      res,
                       const char* const call,
                       const char* const file,
                       const int         line) {
  if(res == ncclSuccess || res == ncclInProgress) return;
  if(aborting.load()) park();
  HPLMXP_rccl_fatal(1,
                    "%s failed at %s:%d: %s (%s)",
                    call,
                    file,
                    line,
                    ncclGetErrorString(res),
                    ncclGetLastError(nullptr));
}

void HPLMXP_rccl_fatal(int code, const char* fmt, ...) {
  if(aborting.load()) park();

  char host[HOST_NAME_MAX + 1];
  if(gethostname(host, sizeof(host)) != 0) strcpy(host, "unknown");
  host[HOST_NAME_MAX] = '\0';

  char    msg[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(msg, sizeof(msg), fmt, ap);
  va_end(ap);
  fprintf(stderr,
          "HPLMXP RCCL rank %d (%s): %s\n",
          HPLMXP_rccl_world_rank(),
          host,
          msg);
  HPLMXP_comm_abort(HPLMXP_COMM_NULL, code);
}

void HPLMXP_comm_abort(HPLMXP_Comm COMM, int code) {
  if(aborting.exchange(true)) park();
  abort_code = (code == 0 ? 1 : code);
  fflush(stdout);
  fflush(stderr);

  int         limit = 10;
  const char* v     = getenv("HPLMXP_RCCL_ABORT_TIMEOUT");
  if(v != nullptr && atoi(v) > 0) limit = atoi(v);
  signal(SIGALRM, abort_timed_out);
  alarm(limit);

  HPLMXP_rccl_abort_comms();
  _exit(abort_code);
}
