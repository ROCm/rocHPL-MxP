/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    (C) 2026 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

/*
 * RCCL backend: point-to-point on device buffers. Each operation runs on
 * the communicator's private stream. Outside a group the call blocks until
 * it completes; inside a group the operations are fused by RCCL and
 * HPLMXP_group_end waits for every communicator touched by the group.
 *
 * HPLMXP_isend/HPLMXP_irecv/HPLMXP_wait never block the host. The
 * operation is enqueued on a second, normal-priority stream of the
 * communicator, behind an event recorded on the caller's STREAM; a request
 * is an event recorded on that stream after the operation, and waiting on
 * it makes the caller's stream wait for that event. A posted receive is a
 * kernel that spins until its data arrives, possibly for many blocks of a
 * solve. On a high-priority stream it takes CUs from the small solve
 * kernels on the caller's stream and slows them down several times, so
 * these streams have normal priority. The blocking calls above
 * must not be issued on a communicator while it has requests in flight.
 */

#include "hplmxp_comm_rccl.hpp"
#include <vector>

static int                      group_depth = 0;
static std::vector<HPLMXP_Comm> group_comms;

static void p2p_done(HPLMXP_Comm COMM) {
  if(group_depth == 0) {
    HPLMXP_rccl_wait(COMM);
  } else {
    for(HPLMXP_Comm c : group_comms)
      if(c == COMM) return;
    group_comms.push_back(COMM);
  }
}

template <typename T>
int HPLMXP_send(const T*    SBUF,
                size_t      COUNT,
                int         PEER,
                HPLMXP_Comm COMM,
                hipStream_t STREAM) {
  if(PEER < 0) return (HPLMXP_SUCCESS);
  HPLMXP_NCCL_CHECK(ncclSend(
      SBUF, COUNT, HPLMXP_nccl_type<T>::value, PEER, COMM->nccl, COMM->stream));
  p2p_done(COMM);
  return (HPLMXP_SUCCESS);
}

template <typename T>
int HPLMXP_recv(T*          RBUF,
                size_t      COUNT,
                int         PEER,
                HPLMXP_Comm COMM,
                hipStream_t STREAM) {
  if(PEER < 0) return (HPLMXP_SUCCESS);
  HPLMXP_NCCL_CHECK(ncclRecv(
      RBUF, COUNT, HPLMXP_nccl_type<T>::value, PEER, COMM->nccl, COMM->stream));
  p2p_done(COMM);
  return (HPLMXP_SUCCESS);
}

void HPLMXP_group_start() {
  group_depth++;
  HPLMXP_NCCL_CHECK(ncclGroupStart());
}

void HPLMXP_group_end() {
  HPLMXP_NCCL_CHECK(ncclGroupEnd());
  if(--group_depth > 0) return;
  for(HPLMXP_Comm c : group_comms) HPLMXP_rccl_wait(c);
  group_comms.clear();
}

template int HPLMXP_send(const double*, size_t, int, HPLMXP_Comm, hipStream_t);
template int HPLMXP_recv(double*, size_t, int, HPLMXP_Comm, hipStream_t);

static std::vector<hipEvent_t> req_events; /* one event per request slot */
static std::vector<int>        req_free;
static hipEvent_t              dep_event = nullptr;

/*
 * The normal-priority stream of COMM, created on first use. Streams are
 * kept until the process exits, so a handle reused by a later
 * communicator at the same address only shares a (valid) stream.
 */
static std::vector<std::pair<HPLMXP_Comm, hipStream_t>> p2p_streams;

static hipStream_t p2p_stream(HPLMXP_Comm COMM) {
  for(auto& p : p2p_streams)
    if(p.first == COMM) return p.second;
  hipStream_t s;
  HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  p2p_streams.push_back({COMM, s});
  return s;
}

static hipEvent_t p2p_event() {
  hipEvent_t e;
  HIP_CHECK(hipEventCreateWithFlags(&e, hipEventDisableTiming));
  return e;
}

/*
 * Orders the next operation on COMM's p2p stream after the work enqueued
 * on STREAM so far. A stream wait binds to the event's latest record at the
 * time of the call, so one event can be recorded again right away.
 */
static void p2p_after(HPLMXP_Comm COMM, hipStream_t STREAM) {
  if(STREAM == nullptr) return;
  if(dep_event == nullptr) dep_event = p2p_event();
  HIP_CHECK(hipEventRecord(dep_event, STREAM));
  HIP_CHECK(hipStreamWaitEvent(p2p_stream(COMM), dep_event, 0));
}

static HPLMXP_Request p2p_request(HPLMXP_Comm COMM) {
  HPLMXP_Request r;
  if(req_free.empty()) {
    r = (HPLMXP_Request)req_events.size();
    req_events.push_back(p2p_event());
  } else {
    r = req_free.back();
    req_free.pop_back();
  }
  HIP_CHECK(hipEventRecord(req_events[r], p2p_stream(COMM)));
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
  p2p_after(COMM, STREAM);
  HPLMXP_NCCL_CHECK(ncclSend(SBUF,
                             COUNT,
                             HPLMXP_nccl_type<T>::value,
                             PEER,
                             COMM->nccl,
                             p2p_stream(COMM)));
  *REQ = p2p_request(COMM);
  return (HPLMXP_SUCCESS);
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
  p2p_after(COMM, STREAM);
  HPLMXP_NCCL_CHECK(ncclRecv(RBUF,
                             COUNT,
                             HPLMXP_nccl_type<T>::value,
                             PEER,
                             COMM->nccl,
                             p2p_stream(COMM)));
  *REQ = p2p_request(COMM);
  return (HPLMXP_SUCCESS);
}

void HPLMXP_wait(HPLMXP_Request* REQ, hipStream_t STREAM) {
  if(*REQ == HPLMXP_REQUEST_NULL) return;
  HIP_CHECK(hipStreamWaitEvent(STREAM, req_events[*REQ], 0));
  req_free.push_back(*REQ);
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
