/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    (C) 2026 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

/*
 * Unit test for the communication layer (include/hplmxp_comm.hpp). Builds
 * against whichever backend was configured and exercises every entry
 * point on a P x Q grid split the same way HPLMXP_grid_init does it.
 *
 *   comm_test [-P p] [-Q q] [-n large_count] [--abort-rank r]
 *             [--fault mode --fault-rank r [--fault-at init|loop]]
 *
 * Exit status 0 means every check passed on every rank. With
 * --abort-rank r, rank r calls HPLMXP_comm_abort after the tests while the
 * others wait in a barrier; the launcher should then report a non-zero
 * exit for the job within seconds.
 *
 * --fault injects a failure on rank r, for testing that the job always
 * ends promptly with a non-zero status. With --fault-at loop (default)
 * every rank runs back-to-back world all-reduces of large_count doubles
 * after the tests, and rank r fails 1 s into the loop from a second
 * thread, i.e. while its main thread is inside a collective. With
 * --fault-at init rank r (from the launcher environment) fails before
 * HPLMXP_comm_world_init. Modes:
 *   exit1   exit(1)              exit0   exit(0)
 *   _exit0  _exit(0), no hooks   kill    raise(SIGKILL)
 *   abort   HPLMXP_comm_abort    hang    stop participating (sleep)
 */

#include "hplmxp_comm.hpp"
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <cstring>
#include <thread>
#include <unistd.h>
#include <vector>

#define CHECK_HIP(cmd)                        \
  do {                                        \
    hipError_t e = (cmd);                     \
    if(e != hipSuccess) {                     \
      fprintf(stderr,                         \
              "HIP error %s at %s:%d\n",      \
              hipGetErrorString(e),           \
              __FILE__,                       \
              __LINE__);                      \
      HPLMXP_comm_abort(HPLMXP_COMM_NULL, 2); \
    }                                         \
  } while(0)

static int         world_rank = 0;
static int         failures   = 0;
static hipStream_t stream;

/*
 * Two device buffers allocated once and reused by every test. Freeing and
 * re-allocating between tests leaves stale IPC handles in UCX's rocm_ipc
 * cache on the MPI backend (and HPL itself never does that).
 */
static void*  arena[2]    = {nullptr, nullptr};
static size_t arena_bytes = 0;

static void* buffer(int slot, size_t bytes) {
  if(bytes > arena_bytes) {
    fprintf(stderr, "comm_test: buffer of %zu bytes exceeds arena\n", bytes);
    HPLMXP_comm_abort(HPLMXP_COMM_NULL, 2);
  }
  return arena[slot];
}

static void expect(bool ok, const char* what, long a, long b) {
  if(ok) return;
  if(failures < 20)
    fprintf(stderr, "rank %d: FAIL %s (%ld vs %ld)\n", world_rank, what, a, b);
  failures++;
}

/* Exactly representable small integers, distinct per (seed, i) */
static long pattern(int seed, size_t i) {
  return (long)((i * 7 + (size_t)seed * 13) % 1021);
}

template <typename T>
struct Bits;
template <>
struct Bits<double> {
  static double make(long v) { return (double)v; }
  static long   get(double v) { return (long)v; }
};
template <>
struct Bits<float> {
  static float make(long v) { return (float)v; }
  static long  get(float v) { return (long)v; }
};
template <>
struct Bits<int> {
  static int  make(long v) { return (int)v; }
  static long get(int v) { return (long)v; }
};
/* Broadcast moves bits only, so __half is tested through its raw pattern */
template <>
struct Bits<uint16_t> {
  static uint16_t make(long v) { return (uint16_t)v; }
  static long     get(uint16_t v) { return (long)v; }
};

template <typename T>
struct DevType {
  typedef T type;
};
template <>
struct DevType<uint16_t> {
  typedef __half type;
};

/* Device broadcast from every root, for several sizes and topologies */
template <typename T>
static void test_bcast(HPLMXP_Comm         comm,
                       const char*         name,
                       const size_t        large,
                       const HPLMXP_T_TOP* tops,
                       int                 ntops) {
  typedef typename DevType<T>::type D;
  const int                         rank     = HPLMXP_comm_rank(comm);
  const int                         size     = HPLMXP_comm_size(comm);
  const size_t                      counts[] = {0, 1, 1000, large};

  for(size_t count : counts) {
    std::vector<T> h(count + 1);
    D*             d = (D*)buffer(0, sizeof(T) * (count + 1));
    for(int t = 0; t < ntops; ++t) {
      for(int root = 0; root < size; ++root) {
        for(size_t i = 0; i <= count; ++i)
          h[i] = Bits<T>::make(rank == root ? pattern(root + t, i) : 9999);
        CHECK_HIP(hipMemcpy(
            d, h.data(), sizeof(T) * (count + 1), hipMemcpyHostToDevice));

        HPLMXP_bcast(d, count, root, comm, stream, tops[t]);

        CHECK_HIP(hipMemcpy(
            h.data(), d, sizeof(T) * (count + 1), hipMemcpyDeviceToHost));
        bool ok = true;
        for(size_t i = 0; i < count && ok; ++i)
          ok = Bits<T>::get(h[i]) == pattern(root + t, i);
        expect(ok, name, (long)count, root);
        /* One past the end must be untouched */
        long sentinel = (rank == root ? pattern(root + t, count) : 9999);
        expect(
            Bits<T>::get(h[count]) == sentinel, "bcast overrun", count, root);
      }
    }
  }
}

template <typename T>
static long reduce_ref(HPLMXP_T_OP op, int size, size_t i) {
  long r = pattern(0, i);
  for(int p = 1; p < size; ++p) {
    long v = pattern(p, i);
    r      = (op == HPLMXP_SUM   ? r + v
              : op == HPLMXP_MAX ? std::max(r, v)
                                 : std::min(r, v));
  }
  return r;
}

/* Device all-reduce for every op */
template <typename T>
static void test_all_reduce(HPLMXP_Comm comm, const char* name, size_t large) {
  const int         rank     = HPLMXP_comm_rank(comm);
  const int         size     = HPLMXP_comm_size(comm);
  const size_t      counts[] = {0, 1, 1000, large};
  const HPLMXP_T_OP ops[]    = {HPLMXP_SUM, HPLMXP_MAX, HPLMXP_MIN};

  for(size_t count : counts) {
    std::vector<T> h(count + 1);
    T*             d = (T*)buffer(0, sizeof(T) * (count + 1));
    for(HPLMXP_T_OP op : ops) {
      for(size_t i = 0; i <= count; ++i) h[i] = Bits<T>::make(pattern(rank, i));
      CHECK_HIP(hipMemcpy(
          d, h.data(), sizeof(T) * (count + 1), hipMemcpyHostToDevice));

      HPLMXP_all_reduce(d, count, op, comm, stream);

      CHECK_HIP(hipMemcpy(
          h.data(), d, sizeof(T) * (count + 1), hipMemcpyDeviceToHost));
      bool ok = true;
      for(size_t i = 0; i < count && ok; ++i)
        ok = Bits<T>::get(h[i]) == reduce_ref<T>(op, size, i);
      expect(ok, name, (long)count, (long)op);
      expect(Bits<T>::get(h[count]) == pattern(rank, count),
             "all_reduce overrun",
             count,
             (long)op);
    }
  }
}

/* Host-buffer collectives */
template <typename T>
static void test_host(HPLMXP_Comm comm, const char* name) {
  const int         rank  = HPLMXP_comm_rank(comm);
  const int         size  = HPLMXP_comm_size(comm);
  const int         count = 37;
  const HPLMXP_T_OP ops[] = {HPLMXP_SUM, HPLMXP_MAX, HPLMXP_MIN};
  std::vector<T>    h(count);

  for(HPLMXP_T_OP op : ops) {
    for(int i = 0; i < count; ++i) h[i] = Bits<T>::make(pattern(rank, i));
    HPLMXP_all_reduce_host(h.data(), count, op, comm);
    bool ok = true;
    for(int i = 0; i < count && ok; ++i)
      ok = Bits<T>::get(h[i]) == reduce_ref<T>(op, size, i);
    expect(ok, name, count, (long)op);
  }

  for(int root = 0; root < size; ++root) {
    for(int i = 0; i < count; ++i)
      h[i] = Bits<T>::make(rank == root ? pattern(root, i) : -1);
    HPLMXP_broadcast_host(h.data(), count, root, comm);
    bool ok = true;
    for(int i = 0; i < count && ok; ++i)
      ok = Bits<T>::get(h[i]) == pattern(root, i);
    expect(ok, name, count, root);
  }
}

/*
 * Point-to-point: a grouped ring shift (every rank sends right and receives
 * from left in one group), then an ungrouped pairwise exchange, then
 * negative peers as no-ops.
 */
static void test_p2p(HPLMXP_Comm comm, size_t count) {
  const int rank  = HPLMXP_comm_rank(comm);
  const int size  = HPLMXP_comm_size(comm);
  const int right = (rank + 1) % size;
  const int left  = (rank + size - 1) % size;

  std::vector<double> h(count);
  double*             s = (double*)buffer(0, sizeof(double) * count);
  double*             r = (double*)buffer(1, sizeof(double) * count);
  for(size_t i = 0; i < count; ++i) h[i] = (double)pattern(rank, i);
  CHECK_HIP(
      hipMemcpy(s, h.data(), sizeof(double) * count, hipMemcpyHostToDevice));

  if(size > 1) {
    /* Two messages to the same peer must arrive in order */
    HPLMXP_group_start();
    HPLMXP_send(s, count / 2, right, comm, stream);
    HPLMXP_send(s + count / 2, count - count / 2, right, comm, stream);
    HPLMXP_recv(r, count / 2, left, comm, stream);
    HPLMXP_recv(r + count / 2, count - count / 2, left, comm, stream);
    HPLMXP_group_end();
    CHECK_HIP(
        hipMemcpy(h.data(), r, sizeof(double) * count, hipMemcpyDeviceToHost));
    bool ok = true;
    for(size_t i = 0; i < count && ok; ++i) ok = (long)h[i] == pattern(left, i);
    expect(ok, "p2p ring", (long)count, left);

    /* Pairwise exchange without a group: even ranks send first */
    const int partner = (rank % 2 == 0 ? rank + 1 : rank - 1);
    if(partner < size) {
      if(rank % 2 == 0) {
        HPLMXP_send(s, count, partner, comm, stream);
        HPLMXP_recv(r, count, partner, comm, stream);
      } else {
        HPLMXP_recv(r, count, partner, comm, stream);
        HPLMXP_send(s, count, partner, comm, stream);
      }
      CHECK_HIP(hipMemcpy(
          h.data(), r, sizeof(double) * count, hipMemcpyDeviceToHost));
      ok = true;
      for(size_t i = 0; i < count && ok; ++i)
        ok = (long)h[i] == pattern(partner, i);
      expect(ok, "p2p pairwise", (long)count, partner);
    }
  }

  /* Negative peers are no-ops, inside and outside a group */
  HPLMXP_send(s, count, -1, comm, stream);
  HPLMXP_recv(r, count, -1, comm, stream);
  HPLMXP_group_start();
  HPLMXP_send(s, count, -1, comm, stream);
  HPLMXP_recv(r, count, -1, comm, stream);
  HPLMXP_group_end();
}

static void run_suite(HPLMXP_Comm comm, const char* label, size_t large) {
  const HPLMXP_T_TOP all_tops[] = {HPLMXP_1RING,
                                   HPLMXP_1RING_M,
                                   HPLMXP_2RING,
                                   HPLMXP_2RING_M,
                                   HPLMXP_BLONG,
                                   HPLMXP_BLONG_M,
                                   HPLMXP_COLL};
  const HPLMXP_T_TOP coll_top[] = {HPLMXP_COLL};
  const int          before     = failures;
  const double       t0         = HPLMXP_wtime();

  test_host<int>(comm, "host int");
  test_host<float>(comm, "host float");
  test_host<double>(comm, "host double");

  /* Every topology on the type HPL broadcasts most; HPLMXP_COLL elsewhere */
  test_bcast<float>(comm, "bcast float", large, all_tops, 7);
  test_bcast<double>(comm, "bcast double", large / 2, coll_top, 1);
  test_bcast<uint16_t>(comm, "bcast half", large, coll_top, 1);

  test_all_reduce<int>(comm, "all_reduce int", large);
  test_all_reduce<float>(comm, "all_reduce float", large);
  test_all_reduce<double>(comm, "all_reduce double", large / 2);

  test_p2p(comm, large / 2 + 3);

  HPLMXP_barrier(comm);
  if(world_rank == 0)
    printf("  %-6s size %3d: %s (%.2f s)\n",
           label,
           HPLMXP_comm_size(comm),
           failures == before ? "ok" : "FAILURES on rank 0",
           HPLMXP_wtime() - t0);
}

static const char* fault_mode = nullptr;

static void inject_fault(HPLMXP_Comm world) {
  fprintf(stderr, "rank %d: injecting fault '%s'\n", world_rank, fault_mode);
  fflush(stderr);
  if(!strcmp(fault_mode, "exit1")) exit(1);
  if(!strcmp(fault_mode, "exit0")) exit(0);
  if(!strcmp(fault_mode, "_exit0")) _exit(0);
  if(!strcmp(fault_mode, "kill")) raise(SIGKILL);
  if(!strcmp(fault_mode, "abort")) HPLMXP_comm_abort(world, 3);
  for(;;) sleep(1000); /* hang */
}

/* Rank of this process before the backend is up, -1 if unknown */
static int launcher_rank() {
  const char* vars[] = {"HPLMXP_RANK",
                        "OMPI_COMM_WORLD_RANK",
                        "PMI_RANK",
                        "SLURM_PROCID",
                        "RANK"};
  for(const char* v : vars)
    if(getenv(v) != nullptr) return atoi(getenv(v));
  return -1;
}

int main(int argc, char** argv) {
  int    P = 0, Q = 0, abort_rank = -1, fault_rank = -1;
  bool   fault_init = false;
  size_t large      = (size_t)3 << 20; /* > the 2 MB chunks of the MPI rings */
  for(int i = 1; i < argc; ++i) {
    if(!strcmp(argv[i], "-P") && i + 1 < argc)
      P = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-Q") && i + 1 < argc)
      Q = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-n") && i + 1 < argc)
      large = atol(argv[++i]);
    else if(!strcmp(argv[i], "--abort-rank") && i + 1 < argc)
      abort_rank = atoi(argv[++i]);
    else if(!strcmp(argv[i], "--fault") && i + 1 < argc)
      fault_mode = argv[++i];
    else if(!strcmp(argv[i], "--fault-rank") && i + 1 < argc)
      fault_rank = atoi(argv[++i]);
    else if(!strcmp(argv[i], "--fault-at") && i + 1 < argc)
      fault_init = !strcmp(argv[++i], "init");
  }
  if(fault_mode != nullptr) {
    const char* modes[] = {"exit1", "exit0", "_exit0", "kill", "abort", "hang"};
    bool        known   = false;
    for(const char* m : modes) known = known || !strcmp(m, fault_mode);
    if(!known || fault_rank < 0) {
      fprintf(stderr,
              "comm_test: --fault needs a mode (exit1 exit0 _exit0 kill abort "
              "hang) and --fault-rank\n");
      return 2;
    }
  }

  if(fault_mode != nullptr && fault_init && launcher_rank() == fault_rank) {
    world_rank = fault_rank;
    inject_fault(HPLMXP_COMM_NULL);
  }

  HPLMXP_Comm world;
  HPLMXP_comm_world_init(&argc, &argv, &world);
  world_rank     = HPLMXP_comm_rank(world);
  const int size = HPLMXP_comm_size(world);
  if(P < 1 && Q < 1) {
    P = (int)std::sqrt((double)size);
    while(size % P) P--;
  }
  if(P < 1) P = size / Q;
  if(Q < 1) Q = size / P;

  /* Backends that do not bind a GPU (MPI) get the HPL default mapping */
  if(HPLMXP_comm_device() < 0) {
    int count = 0;
    CHECK_HIP(hipGetDeviceCount(&count));
    CHECK_HIP(hipSetDevice(HPLMXP_comm_local_rank() % count));
  }

  char host[256];
  gethostname(host, sizeof(host));
  int dev = -1;
  CHECK_HIP(hipGetDevice(&dev));
  printf(
      "rank %3d/%d on %s: local %d/%d, backend device %d, current device %d\n",
      world_rank,
      size,
      host,
      HPLMXP_comm_local_rank(),
      HPLMXP_comm_local_size(),
      HPLMXP_comm_device(),
      dev);
  fflush(stdout);
  HPLMXP_barrier(world);
  if(HPLMXP_comm_device() >= 0)
    expect(
        HPLMXP_comm_device() == dev, "bound device", HPLMXP_comm_device(), dev);
  expect(HPLMXP_comm_world() == world, "comm_world", 0, 0);

  CHECK_HIP(hipStreamCreate(&stream));
  arena_bytes = sizeof(double) * (large + 8);
  CHECK_HIP(hipMalloc(&arena[0], arena_bytes));
  CHECK_HIP(hipMalloc(&arena[1], arena_bytes));

  /* Grid split exactly as HPLMXP_grid_init (column-major) */
  const int   nprocs = P * Q;
  HPLMXP_Comm all, row = HPLMXP_COMM_NULL, col = HPLMXP_COMM_NULL;
  HPLMXP_comm_split(world, world_rank < nprocs ? 0 : -1, world_rank, &all);
  if(world_rank == 0)
    printf("comm_test: %d ranks, grid %d x %d, large count %zu\n",
           size,
           P,
           Q,
           large);

  expect(world_rank < nprocs ? all != HPLMXP_COMM_NULL
                             : all == HPLMXP_COMM_NULL,
         "split membership",
         world_rank,
         nprocs);

  run_suite(world, "world", large);

  if(all != HPLMXP_COMM_NULL) {
    const int myrow = world_rank % P;
    const int mycol = world_rank / P;
    HPLMXP_comm_split(all, myrow, mycol, &row);
    HPLMXP_comm_split(all, mycol, myrow, &col);
    expect(HPLMXP_comm_size(all) == nprocs,
           "all size",
           HPLMXP_comm_size(all),
           nprocs);
    expect(HPLMXP_comm_size(row) == Q, "row size", HPLMXP_comm_size(row), Q);
    expect(HPLMXP_comm_size(col) == P, "col size", HPLMXP_comm_size(col), P);
    expect(HPLMXP_comm_rank(row) == mycol,
           "row rank",
           HPLMXP_comm_rank(row),
           mycol);
    expect(HPLMXP_comm_rank(col) == myrow,
           "col rank",
           HPLMXP_comm_rank(col),
           myrow);

    /* Row and column suites run on disjoint groups concurrently */
    run_suite(row, "row", large);
    run_suite(col, "col", large);

    HPLMXP_comm_free(&row);
    HPLMXP_comm_free(&col);
    HPLMXP_comm_free(&all);
    expect(row == HPLMXP_COMM_NULL && all == HPLMXP_COMM_NULL, "free", 0, 0);
  }

  int total = failures;
  HPLMXP_all_reduce_host(&total, 1, HPLMXP_SUM, world);
  if(world_rank == 0)
    printf("comm_test: %s (%d failures)\n",
           total == 0 ? "PASSED" : "FAILED",
           total);
  fflush(stdout);

  if(fault_mode != nullptr) {
    if(world_rank == fault_rank && strcmp(fault_mode, "hang") != 0)
      std::thread([world]() {
        sleep(1);
        inject_fault(world);
      }).detach();
    const double t0 = HPLMXP_wtime();
    double*      d  = (double*)arena[0];
    for(long it = 0;; ++it) {
      if(world_rank == fault_rank && !strcmp(fault_mode, "hang") &&
         HPLMXP_wtime() - t0 > 1.0)
        inject_fault(world);
      HPLMXP_all_reduce(d, large, HPLMXP_MAX, world, stream);
      if(world_rank == 0 && (it & (it - 1)) == 0) {
        printf("comm_test: fault loop iteration %ld at %.1f s\n",
               it,
               HPLMXP_wtime() - t0);
        fflush(stdout);
      }
    }
  }

  if(abort_rank >= 0) {
    if(world_rank == abort_rank) {
      fprintf(stderr,
              "rank %d: calling HPLMXP_comm_abort as requested\n",
              world_rank);
      HPLMXP_comm_abort(world, 3);
    }
    HPLMXP_barrier(world); /* never completes */
    fprintf(stderr, "rank %d: barrier returned after abort\n", world_rank);
    return 4;
  }

  CHECK_HIP(hipFree(arena[0]));
  CHECK_HIP(hipFree(arena[1]));
  CHECK_HIP(hipStreamDestroy(stream));
  HPLMXP_comm_finalize();
  return total == 0 ? 0 : 1;
}
