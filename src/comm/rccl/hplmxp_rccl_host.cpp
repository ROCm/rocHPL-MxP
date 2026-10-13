/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    (C) 2026 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

/*
 * RCCL backend: host-buffer collectives. RCCL only moves device memory,
 * so these stage through a device scratch buffer on the communicator's
 * stream. They carry small control data (error flags, input parameters,
 * timers) and are not performance critical.
 *
 * The whole call is one watch: the device-to-host copy into a pageable
 * BUFFER blocks the host until the collective has finished, before
 * HPLMXP_rccl_wait is reached, and growing the scratch buffer (hipFree)
 * waits for the whole device.
 */

#include "hplmxp_comm_rccl.hpp"
#include <unistd.h>

template <typename T>
int HPLMXP_all_reduce_host(T*                BUFFER,
                           const int         COUNT,
                           const HPLMXP_T_OP OP,
                           HPLMXP_Comm       COMM) {
  if(COUNT <= 0) return (HPLMXP_SUCCESS);
  const size_t bytes = sizeof(T) * COUNT;
  HPLMXP_rccl_watch_begin(
      "a host-buffer collective", COMM, HPLMXP_RCCL_PROGRESS);
  T* d = (T*)HPLMXP_rccl_scratch(bytes);

  HIP_CHECK(
      hipMemcpyAsync(d, BUFFER, bytes, hipMemcpyHostToDevice, COMM->stream));
  HPLMXP_NCCL_CHECK(ncclAllReduce(d,
                                  d,
                                  COUNT,
                                  HPLMXP_nccl_type<T>::value,
                                  HPLMXP_nccl_op(OP),
                                  COMM->nccl,
                                  COMM->stream));
  hipError_t err =
      hipMemcpyAsync(BUFFER, d, bytes, hipMemcpyDeviceToHost, COMM->stream);
  if(err == hipSuccess) HPLMXP_rccl_wait(COMM);
  HPLMXP_rccl_watch_end();
  /* An abort by the watchdog can break the copy; let it finish */
  if(HPLMXP_rccl_aborting())
    for(;;) pause();
  HIP_CHECK(err);
  return (HPLMXP_SUCCESS);
}

template <typename T>
int HPLMXP_broadcast_host(T*          BUFFER,
                          const int   COUNT,
                          const int   ROOT,
                          HPLMXP_Comm COMM) {
  if(COUNT <= 0) return (HPLMXP_SUCCESS);
  const size_t bytes = sizeof(T) * COUNT;
  HPLMXP_rccl_watch_begin(
      "a host-buffer collective", COMM, HPLMXP_RCCL_PROGRESS);
  T* d = (T*)HPLMXP_rccl_scratch(bytes);

  if(COMM->rank == ROOT)
    HIP_CHECK(
        hipMemcpyAsync(d, BUFFER, bytes, hipMemcpyHostToDevice, COMM->stream));
  HPLMXP_NCCL_CHECK(ncclBroadcast(
      d, d, COUNT, HPLMXP_nccl_type<T>::value, ROOT, COMM->nccl, COMM->stream));
  hipError_t err =
      hipMemcpyAsync(BUFFER, d, bytes, hipMemcpyDeviceToHost, COMM->stream);
  if(err == hipSuccess) HPLMXP_rccl_wait(COMM);
  HPLMXP_rccl_watch_end();
  /* An abort by the watchdog can break the copy; let it finish */
  if(HPLMXP_rccl_aborting())
    for(;;) pause();
  HIP_CHECK(err);
  return (HPLMXP_SUCCESS);
}

int HPLMXP_barrier(HPLMXP_Comm COMM) {
  int flag = 0;
  return HPLMXP_all_reduce_host(&flag, 1, HPLMXP_SUM, COMM);
}

template int HPLMXP_all_reduce_host(double*,
                                    const int,
                                    const HPLMXP_T_OP,
                                    HPLMXP_Comm);
template int HPLMXP_all_reduce_host(float*,
                                    const int,
                                    const HPLMXP_T_OP,
                                    HPLMXP_Comm);
template int HPLMXP_all_reduce_host(int*,
                                    const int,
                                    const HPLMXP_T_OP,
                                    HPLMXP_Comm);

template int HPLMXP_broadcast_host(double*, const int, const int, HPLMXP_Comm);
template int HPLMXP_broadcast_host(float*, const int, const int, HPLMXP_Comm);
template int HPLMXP_broadcast_host(int*, const int, const int, HPLMXP_Comm);
