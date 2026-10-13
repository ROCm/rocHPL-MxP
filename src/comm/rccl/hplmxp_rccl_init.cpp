/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    (C) 2026 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

/*
 * RCCL backend: control plane (bootstrap, communicators, watchdog, timing).
 *
 * Ranks come from the launcher environment. The world communicator is
 * bootstrapped through NCCL_COMM_ID=<host>:<port>, which every rank must
 * see; rank 0 opens the bootstrap root on that address. Every problem
 * found before RCCL is called (launcher environment, NCCL_COMM_ID, GPUs)
 * is reported as one line from the rank that sees it, followed by
 * exit(1).
 *
 * Communicators are blocking (the default config): a non-blocking one
 * would make every enqueue return ncclInProgress whenever RCCL connects
 * lazily, and all callers would have to poll before touching the stream.
 * Hangs are bounded by the watchdog thread instead (see
 * hplmxp_comm_rccl.hpp for the knobs).
 */

#include "hplmxp_comm_rccl.hpp"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

static HPLMXP_Comm              world      = HPLMXP_COMM_NULL;
static int                      world_rank = -1;
static int                      local_rank = -1;
static int                      local_size = -1;
static int                      device     = -1;
static std::vector<HPLMXP_Comm> comms; /* guarded by comms_lock */
static std::mutex               comms_lock;
static void*                    scratch       = nullptr;
static size_t                   scratch_bytes = 0;

/*
 * ---------------------------------------------------------------------
 * Launcher environment
 * ---------------------------------------------------------------------
 */
[[noreturn]] static void boot_error(const char* fmt, ...)
    __attribute__((format(printf, 1, 2)));

static void boot_error(const char* fmt, ...) {
  char host[HOST_NAME_MAX + 1];
  if(gethostname(host, sizeof(host)) != 0) strcpy(host, "unknown");
  host[HOST_NAME_MAX] = '\0';

  char    msg[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(msg, sizeof(msg), fmt, ap);
  va_end(ap);
  if(world_rank >= 0)
    fprintf(stderr, "HPLMXP RCCL rank %d (%s): %s\n", world_rank, host, msg);
  else
    fprintf(stderr, "HPLMXP RCCL (%s pid %d): %s\n", host, (int)getpid(), msg);
  fflush(stdout);
  exit(1);
}

/*
 * For errors that every rank sees alike (the environment is the same
 * everywhere): rank 0 reports, the others leave without repeating it.
 * The delay lets rank 0's line out before the launcher kills the step.
 */
[[noreturn]] static void quiet_exit() {
  sleep(2);
  exit(1);
}

/* Value of NAME as a non-negative int; -1 if unset or empty */
static int env_count(const char* name) {
  const char* v = getenv(name);
  if(v == nullptr || v[0] == '\0') return -1;
  char* end;
  errno  = 0;
  long n = strtol(v, &end, 10);
  if(*end != '\0' || end == v || errno != 0 || n < 0 || n > INT_MAX)
    boot_error("%s='%s' is not a non-negative integer", name, v);
  return (int)n;
}

/*
 * Slurm's SLURM_TASKS_PER_NODE is a compressed per-node list such as
 * "8(x2),7". Returns the entry for node NODEID, or -1.
 */
static int slurm_tasks_on_node(const char* spec, int nodeid) {
  if(spec == nullptr || nodeid < 0) return -1;
  int         node = 0;
  const char* p    = spec;
  while(*p != '\0') {
    char* end;
    int   n   = (int)strtol(p, &end, 10);
    int   rep = 1;
    if(end == p) return -1;
    p = end;
    if(p[0] == '(' && p[1] == 'x') { /* "(xK)" */
      rep = (int)strtol(p + 2, &end, 10);
      if(*end != ')') return -1;
      p = end + 1;
    }
    if(nodeid < node + rep) return n;
    node += rep;
    if(*p == ',') p++;
  }
  return -1;
}

/*
 * Launchers in priority order: an explicit HPLMXP_* override, then the
 * innermost launcher (mpirun/mpiexec started inside a Slurm allocation
 * also inherit the allocation's SLURM_* variables), then Slurm, then
 * torchrun-style variables. No launcher at all means a single process.
 */
struct Launcher {
  const char* name;
  const char* rank;
  const char* size;
  const char* local_rank;
  const char* local_size; /* nullptr: Slurm, derived from tasks per node */
};
static const Launcher launchers[] = {
    {"HPLMXP",
     "HPLMXP_RANK",
     "HPLMXP_SIZE",
     "HPLMXP_LOCAL_RANK",
     "HPLMXP_LOCAL_SIZE"},
    {"Open MPI",
     "OMPI_COMM_WORLD_RANK",
     "OMPI_COMM_WORLD_SIZE",
     "OMPI_COMM_WORLD_LOCAL_RANK",
     "OMPI_COMM_WORLD_LOCAL_SIZE"},
    {"PMI", "PMI_RANK", "PMI_SIZE", "MPI_LOCALRANKID", "MPI_LOCALNRANKS"},
    {"Slurm", "SLURM_PROCID", "SLURM_STEP_NUM_TASKS", "SLURM_LOCALID", nullptr},
    {"torchrun", "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"},
};

/*
 * True inside an srun step. A batch script also has SLURM_PROCID, but its
 * SLURM_STEP_ID is unset or negative (-5, SLURM_BATCH_SCRIPT).
 */
static bool in_slurm_step() {
  const char* v = getenv("SLURM_STEP_ID");
  if(v == nullptr) return false;
  char* end;
  long  n = strtol(v, &end, 10);
  return end != v && *end == '\0' && n >= 0;
}

static void launcher_env(int* rank, int* size, int* lrank, int* lsize) {
  *rank = 0, *size = 1, *lrank = 0, *lsize = 1;

  const Launcher* L = nullptr;
  for(const Launcher& l : launchers) {
    if(l.local_size == nullptr && !in_slurm_step()) continue;
    if(env_count(l.rank) >= 0) {
      L = &l;
      break;
    }
  }
  if(L == nullptr) return;

  *rank      = env_count(L->rank);
  world_rank = *rank;
  *size      = env_count(L->size);
  if(*size < 0 && L->local_size == nullptr) *size = env_count("SLURM_NTASKS");
  if(*size < 1)
    boot_error("%s launch: %s is set but %s is missing or 0",
               L->name,
               L->rank,
               L->size);
  *lrank = env_count(L->local_rank);
  if(*lrank < 0)
    boot_error(
        "%s launch: %s is set but %s is not", L->name, L->rank, L->local_rank);

  if(L->local_size != nullptr) {
    *lsize = env_count(L->local_size);
    if(*lsize < 1)
      boot_error("%s launch: %s is missing or 0", L->name, L->local_size);
  } else {
    const char* spec = getenv("SLURM_STEP_TASKS_PER_NODE");
    if(spec == nullptr) spec = getenv("SLURM_TASKS_PER_NODE");
    *lsize = slurm_tasks_on_node(spec, env_count("SLURM_NODEID"));
    if(*lsize < 1)
      boot_error("Slurm launch: cannot get the tasks on this node from "
                 "SLURM_STEP_TASKS_PER_NODE='%s' and SLURM_NODEID='%s'",
                 spec ? spec : "",
                 getenv("SLURM_NODEID") ? getenv("SLURM_NODEID") : "");
  }

  if(*rank >= *size || *lrank >= *lsize || *lsize > *size)
    boot_error("%s launch: inconsistent environment: rank %d of %d, local "
               "rank %d of %d (%s, %s, %s, %s)",
               L->name,
               *rank,
               *size,
               *lrank,
               *lsize,
               L->rank,
               L->size,
               L->local_rank,
               L->local_size ? L->local_size : "SLURM_STEP_TASKS_PER_NODE");
}

/* errno of binding a TCP socket to A (0 if it works or cannot be tried) */
static int test_bind(const struct addrinfo* a, bool reuseport) {
  int fd = socket(a->ai_family, SOCK_STREAM, 0);
  if(fd < 0) return 0;
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#ifdef SO_REUSEPORT
  if(reuseport) setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &one, sizeof(one));
#endif
  int err = (bind(fd, a->ai_addr, a->ai_addrlen) == 0 ? 0 : errno);
  close(fd);
  return err;
}

/*
 * NCCL_COMM_ID must be <host>:<port> or [<ipv6>]:<port>, resolvable on
 * every rank, and rank 0 must be able to listen on it.
 */
static void check_comm_id(int rank) {
  const char* id = getenv("NCCL_COMM_ID");
  if(id == nullptr || id[0] == '\0') {
    if(rank != 0) quiet_exit();
    boot_error("NCCL_COMM_ID is not set; multi-process runs need "
               "NCCL_COMM_ID=<host>:<port> (rank 0's host, a free port)");
  }

  std::string s(id), host, port;
  if(s[0] == '[') {
    size_t e = s.find(']');
    if(e != std::string::npos && e + 1 < s.size() && s[e + 1] == ':') {
      host = s.substr(1, e - 1);
      port = s.substr(e + 2);
    }
  } else if(s.find(':') == s.rfind(':') && s.find(':') != std::string::npos) {
    host = s.substr(0, s.find(':'));
    port = s.substr(s.find(':') + 1);
  }
  char* end   = nullptr;
  long  portn = port.empty() ? 0 : strtol(port.c_str(), &end, 10);
  if(host.empty() || port.empty() || *end != '\0' || portn < 1 ||
     portn > 65535) {
    if(rank != 0) quiet_exit();
    boot_error("NCCL_COMM_ID='%s' is malformed; use <host>:<port> or "
               "[<ipv6>]:<port> with a port in 1..65535",
               id);
  }

  struct addrinfo hints, *res = nullptr;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family   = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  int rv            = getaddrinfo(host.c_str(), port.c_str(), &hints, &res);
  if(rv != 0 || res == nullptr)
    boot_error("NCCL_COMM_ID='%s': cannot resolve host '%s': %s",
               id,
               host.c_str(),
               gai_strerror(rv));

  if(rank == 0) {
    /*
     * First with SO_REUSEADDR only, so that a live listener of another run
     * is caught (RCCL's root also sets SO_REUSEPORT and would share the
     * port with it); then with RCCL's options, to tell that case apart
     * from a port RCCL cannot bind either.
     */
    int err = test_bind(res, false);
    int rcc = (err == EADDRINUSE ? test_bind(res, true) : err);
    freeaddrinfo(res);
    if(err == EADDRNOTAVAIL)
      boot_error("NCCL_COMM_ID='%s' is not an address of this host, but "
                 "rank 0 opens the bootstrap there; point it at rank 0's "
                 "node",
                 id);
    if(err == EADDRINUSE && rcc == 0)
      boot_error("NCCL_COMM_ID='%s': another program (another run with the "
                 "same NCCL_COMM_ID?) is listening on port %s of rank 0's "
                 "host; pick another port",
                 id,
                 port.c_str());
    if(err == EADDRINUSE) {
      int lo = 0, hi = -1;
      if(FILE* f = fopen("/proc/sys/net/ipv4/ip_local_port_range", "r")) {
        if(fscanf(f, "%d %d", &lo, &hi) != 2) hi = -1;
        fclose(f);
      }
      if(portn >= lo && portn <= hi)
        boot_error("NCCL_COMM_ID='%s': port %s is in use on rank 0's host; "
                   "it is in the ephemeral range %d-%d that the kernel "
                   "gives to connections, pick a port outside it",
                   id,
                   port.c_str(),
                   lo,
                   hi);
      boot_error("NCCL_COMM_ID='%s': port %s is in use on rank 0's host; "
                 "pick another port",
                 id,
                 port.c_str());
    }
    if(err != 0)
      boot_error("NCCL_COMM_ID='%s': cannot bind the bootstrap address: %s",
                 id,
                 strerror(err));
    return;
  }
  freeaddrinfo(res);
}

/*
 * ---------------------------------------------------------------------
 * Watchdog
 * ---------------------------------------------------------------------
 */
static double now() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

static double env_seconds(const char* name, double fallback) {
  const char* v = getenv(name);
  if(v == nullptr || v[0] == '\0') return fallback;
  char*  end;
  double t = strtod(v, &end);
  if(*end != '\0' || t < 0)
    boot_error("%s='%s' is not a non-negative number of seconds", name, v);
  return t;
}

static double setup_timeout    = 300.0; /* HPLMXP_RCCL_INIT_TIMEOUT */
static double progress_timeout = 300.0; /* HPLMXP_RCCL_TIMEOUT */

/*
 * The blocking call in progress on the main thread. start == 0 means none;
 * it is written last by watch_begin, so a reader that sees the same start
 * before and after reading the other fields has a consistent snapshot.
 */
static std::atomic<double>      watch_start{0.0};
static std::atomic<const char*> watch_what{""};
static std::atomic<int>         watch_phase{HPLMXP_RCCL_SETUP};
static std::atomic<int>         watch_rank{-1}, watch_size{-1};
static char                     watch_name[48];
static int                      watch_depth = 0; /* main thread only */

static std::thread             watchdog;
static std::mutex              watchdog_lock;
static std::condition_variable watchdog_cv;
static bool                    watchdog_stop = false;

/* HPLMXP_RCCL_WATCHDOG_VERBOSE=1: report the longest wait at finalize */
static bool   verbose  = false;
static double wait_max = 0.0, wait_total = 0.0;
static long   wait_count = 0;

void HPLMXP_rccl_watch_begin(const char*       what,
                             HPLMXP_Comm       COMM,
                             HPLMXP_rccl_phase phase) {
  if(watch_depth++ > 0) return;
  watch_what.store(what, std::memory_order_relaxed);
  watch_phase.store(phase, std::memory_order_relaxed);
  watch_rank.store(COMM ? COMM->rank : -1, std::memory_order_relaxed);
  watch_size.store(COMM ? COMM->size : -1, std::memory_order_relaxed);
  if(COMM)
    memcpy(watch_name, COMM->name, sizeof(watch_name));
  else
    watch_name[0] = '\0';
  watch_start.store(now(), std::memory_order_release);
}

double HPLMXP_rccl_watch_end() {
  double t = now() - watch_start.load(std::memory_order_relaxed);
  if(--watch_depth == 0) watch_start.store(0.0, std::memory_order_release);
  return t;
}

static void watchdog_tick() {
  /* Asynchronous errors (network failures reported by the proxy) */
  {
    std::unique_lock<std::mutex> g(comms_lock);
    for(HPLMXP_Comm c : comms) {
      ncclResult_t err = ncclSuccess;
      if(ncclCommGetAsyncError(c->nccl, &err) != ncclSuccess) continue;
      if(err != ncclSuccess && err != ncclInProgress) {
        char name[48];
        memcpy(name, c->name, sizeof(name));
        int r = c->rank, s = c->size;
        g.unlock(); /* HPLMXP_rccl_fatal aborts every comm */
        HPLMXP_rccl_fatal(1,
                          "asynchronous RCCL error on communicator '%s' "
                          "(rank %d of %d): %s; last RCCL warning: %s",
                          name,
                          r,
                          s,
                          ncclGetErrorString(err),
                          ncclGetLastError(nullptr));
      }
    }
  }

  /* Deadline of the blocking call in progress */
  double start = watch_start.load(std::memory_order_acquire);
  if(start == 0.0) return;
  int         phase = watch_phase.load(std::memory_order_relaxed);
  const char* what  = watch_what.load(std::memory_order_relaxed);
  int         r = watch_rank.load(), s = watch_size.load();
  char        name[48];
  memcpy(name, watch_name, sizeof(name));
  name[sizeof(name) - 1] = '\0';
  if(watch_start.load(std::memory_order_acquire) != start) return;

  double limit =
      (phase == HPLMXP_RCCL_SETUP ? setup_timeout : progress_timeout);
  double t = now() - start;
  if(limit <= 0.0 || t < limit) return;
  const char* knob = (phase == HPLMXP_RCCL_SETUP ? "HPLMXP_RCCL_INIT_TIMEOUT"
                                                 : "HPLMXP_RCCL_TIMEOUT");

  if(name[0] != '\0')
    HPLMXP_rccl_fatal(1,
                      "no progress for %.0f s in %s on communicator '%s' "
                      "(rank %d of %d); a peer is dead or stuck. Raise %s "
                      "(0 = no limit) if this is legitimate",
                      t,
                      what,
                      name,
                      r,
                      s,
                      knob);
  if(phase == HPLMXP_RCCL_SETUP)
    HPLMXP_rccl_fatal(1,
                      "no progress for %.0f s in %s; are all ranks running "
                      "and can they reach NCCL_COMM_ID=%s? Raise %s (0 = no "
                      "limit) if this is legitimate",
                      t,
                      what,
                      getenv("NCCL_COMM_ID") ? getenv("NCCL_COMM_ID") : "",
                      knob);
  HPLMXP_rccl_fatal(1,
                    "no progress for %.0f s in %s; a peer is dead or stuck. "
                    "Raise %s (0 = no limit) if this is legitimate",
                    t,
                    what,
                    knob);
}

static void watchdog_start() {
  setup_timeout        = env_seconds("HPLMXP_RCCL_INIT_TIMEOUT", setup_timeout);
  progress_timeout     = env_seconds("HPLMXP_RCCL_TIMEOUT", progress_timeout);
  double      interval = env_seconds("HPLMXP_RCCL_WATCHDOG_MS", 100.0);
  const char* v        = getenv("HPLMXP_RCCL_WATCHDOG_VERBOSE");
  verbose              = (v != nullptr && atoi(v) != 0);
  if(interval <= 0.0) return;

  const auto period = std::chrono::microseconds((long)(interval * 1000.0));
  watchdog          = std::thread([period]() {
    std::unique_lock<std::mutex> lk(watchdog_lock);
    while(!watchdog_cv.wait_for(lk, period, [] { return watchdog_stop; })) {
      lk.unlock();
      watchdog_tick();
      lk.lock();
    }
  });
}

static void watchdog_join() {
  if(!watchdog.joinable()) return;
  {
    std::lock_guard<std::mutex> g(watchdog_lock);
    watchdog_stop = true;
  }
  watchdog_cv.notify_all();
  watchdog.join();
}

/*
 * A process that exits (exit() or return from main) while communicators
 * are live leaves its peers blocked in RCCL. Status 0 would not trigger
 * srun --kill-on-bad-exit, so turn it into 1. _exit avoids running static
 * destructors and RCCL teardown under the still-running watchdog.
 */
static std::atomic<bool> session{false}; /* world_init .. finalize */

static void exit_hook(int status, void*) {
  if(HPLMXP_rccl_aborting())
    for(;;) pause();
  if(!session.load()) return;
  fflush(nullptr);
  if(status == 0) {
    fprintf(stderr,
            "HPLMXP RCCL rank %d: process exiting with status 0 between "
            "HPLMXP_comm_world_init and HPLMXP_comm_finalize; exiting with 1 "
            "so that the launcher stops the other ranks\n",
            world_rank);
    status = 1;
  }
  _exit(status);
}

/*
 * ---------------------------------------------------------------------
 * Communicators
 * ---------------------------------------------------------------------
 */
static HPLMXP_Comm new_comm(ncclComm_t nccl, const char* name) {
  HPLMXP_Comm c = new HPLMXP_CommImpl;
  c->nccl       = nccl;
  HPLMXP_NCCL_CHECK(ncclCommUserRank(nccl, &c->rank));
  HPLMXP_NCCL_CHECK(ncclCommCount(nccl, &c->size));
  snprintf(c->name, sizeof(c->name), "%s", name);

  /*
   * High priority, so a collective is not starved by the trailing update
   * already queued on the compute stream
   */
  int least, greatest;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&least, &greatest));
  HIP_CHECK(
      hipStreamCreateWithPriority(&c->stream, hipStreamNonBlocking, greatest));

  std::lock_guard<std::mutex> g(comms_lock);
  comms.push_back(c);
  return c;
}

int HPLMXP_comm_world_init(int* argc, char*** argv, HPLMXP_Comm* out) {
  int rank, size;
  launcher_env(&rank, &size, &local_rank, &local_size);
  world_rank = rank;

  if(size > 1) check_comm_id(rank);

  /* Bind the GPU before any RCCL call */
  int        deviceCount = 0;
  hipError_t herr        = hipGetDeviceCount(&deviceCount);
  if(herr != hipSuccess || deviceCount < 1)
    boot_error(
        "no usable GPU (hipGetDeviceCount: %s, %d devices; "
        "HIP_VISIBLE_DEVICES='%s', ROCR_VISIBLE_DEVICES='%s')",
        hipGetErrorString(herr),
        deviceCount,
        getenv("HIP_VISIBLE_DEVICES") ? getenv("HIP_VISIBLE_DEVICES") : "",
        getenv("ROCR_VISIBLE_DEVICES") ? getenv("ROCR_VISIBLE_DEVICES") : "");
  if(deviceCount >= local_size)
    device = local_rank;
  else if(deviceCount == 1)
    device = 0; /* the launcher gave every task its own GPU */
  else
    boot_error("%d ranks on this node but only %d GPUs visible; RCCL needs "
               "one GPU per rank",
               local_size,
               deviceCount);
  herr = hipSetDevice(device);
  if(herr != hipSuccess)
    boot_error("hipSetDevice(%d) failed: %s", device, hipGetErrorString(herr));

  watchdog_start();
  session.store(true);
  on_exit(exit_hook, nullptr);

  /* With NCCL_COMM_ID set every rank derives the same id from it */
  ncclUniqueId id;
  HPLMXP_NCCL_CHECK(ncclGetUniqueId(&id));

  ncclComm_t nccl;
  HPLMXP_rccl_watch_begin("world communicator setup (ncclCommInitRank)",
                          HPLMXP_COMM_NULL,
                          HPLMXP_RCCL_SETUP);
  HPLMXP_NCCL_CHECK(ncclCommInitRank(&nccl, size, id, rank));
  HPLMXP_rccl_watch_end();

  world = new_comm(nccl, "world");
  *out  = world;
  return HPLMXP_SUCCESS;
}

int HPLMXP_comm_local_rank() { return local_rank; }

int HPLMXP_comm_local_size() { return local_size; }

int HPLMXP_comm_device() { return device; }

HPLMXP_Comm HPLMXP_comm_world() { return world; }

int HPLMXP_rccl_world_rank() { return world_rank; }

int HPLMXP_comm_split(HPLMXP_Comm COMM, int color, int key, HPLMXP_Comm* OUT) {
  ncclComm_t nccl = nullptr;
  HPLMXP_rccl_watch_begin("ncclCommSplit", COMM, HPLMXP_RCCL_SETUP);
  HPLMXP_NCCL_CHECK(ncclCommSplit(COMM->nccl,
                                  (color < 0 ? NCCL_SPLIT_NOCOLOR : color),
                                  key,
                                  &nccl,
                                  nullptr));
  HPLMXP_rccl_watch_end();

  char name[48];
  snprintf(name, sizeof(name), "color %d of %.32s", color, COMM->name);
  *OUT = (nccl == nullptr ? HPLMXP_COMM_NULL : new_comm(nccl, name));
  return HPLMXP_SUCCESS;
}

int HPLMXP_comm_rank(HPLMXP_Comm COMM) { return COMM->rank; }

int HPLMXP_comm_size(HPLMXP_Comm COMM) { return COMM->size; }

static void destroy_comm(HPLMXP_Comm c) {
  HPLMXP_rccl_wait(c);
  {
    std::lock_guard<std::mutex> g(comms_lock);
    comms.erase(std::remove(comms.begin(), comms.end(), c), comms.end());
  }
  HPLMXP_rccl_watch_begin("ncclCommDestroy", c, HPLMXP_RCCL_SETUP);
  HPLMXP_NCCL_CHECK(ncclCommDestroy(c->nccl));
  HPLMXP_rccl_watch_end();
  HIP_CHECK(hipStreamDestroy(c->stream));
  delete c;
}

void HPLMXP_comm_free(HPLMXP_Comm* COMM) {
  if(*COMM == HPLMXP_COMM_NULL || *COMM == world) return;
  destroy_comm(*COMM);
  *COMM = HPLMXP_COMM_NULL;
}

/*
 * Every rank leaves together: ranks outside the P x Q grid reach finalize
 * at once, and srun --wait would stop the job soon after they exit. They
 * may legitimately wait for the whole run, so there is no deadline here;
 * the watchdog still reports asynchronous errors.
 */
static void world_barrier() {
  if(world == HPLMXP_COMM_NULL || world->size < 2) return;
  void* buf = HPLMXP_rccl_scratch(1);
  HPLMXP_NCCL_CHECK(ncclAllReduce(
      buf, buf, 1, ncclInt8, ncclMax, world->nccl, world->stream));
  hipError_t err = hipStreamSynchronize(world->stream);
  if(HPLMXP_rccl_aborting())
    for(;;) pause();
  HIP_CHECK(err);
}

void HPLMXP_comm_finalize() {
  world_barrier();
  for(;;) {
    HPLMXP_Comm c;
    {
      std::lock_guard<std::mutex> g(comms_lock);
      if(comms.empty()) break;
      c = comms.back();
    }
    if(c == world) world = HPLMXP_COMM_NULL;
    destroy_comm(c);
  }
  if(scratch != nullptr) HIP_CHECK(hipFree(scratch));
  scratch       = nullptr;
  scratch_bytes = 0;

  watchdog_join();
  session.store(false);
  if(verbose)
    fprintf(stderr,
            "HPLMXP RCCL rank %d: %ld waits, %.3f s total, longest %.3f s\n",
            world_rank,
            wait_count,
            wait_total,
            wait_max);
}

int HPLMXP_rccl_num_comms() {
  std::lock_guard<std::mutex> g(comms_lock);
  return (int)comms.size();
}

HPLMXP_Comm HPLMXP_rccl_comm(int i) {
  std::lock_guard<std::mutex> g(comms_lock);
  return comms[i];
}

void HPLMXP_rccl_abort_comms() {
  /* The main thread never blocks while holding the lock; give up after a
   * short while rather than risk a deadlock on the way out */
  std::unique_lock<std::mutex> g(comms_lock, std::defer_lock);
  for(int i = 0; i < 100 && !g.try_lock(); ++i) usleep(10000);
  if(!g.owns_lock()) return;
  for(auto it = comms.rbegin(); it != comms.rend(); ++it)
    (void)ncclCommAbort((*it)->nccl);
}

void HPLMXP_rccl_wait(HPLMXP_Comm COMM) {
  HPLMXP_rccl_watch_begin(
      "a collective or point-to-point operation", COMM, HPLMXP_RCCL_PROGRESS);
  hipError_t err = hipStreamSynchronize(COMM->stream);
  double     t   = HPLMXP_rccl_watch_end();
  if(verbose) {
    wait_max = std::max(wait_max, t);
    wait_total += t;
    wait_count++;
  }
  /* An abort by the watchdog completes or breaks the wait; let it finish */
  if(HPLMXP_rccl_aborting())
    for(;;) pause();
  HIP_CHECK(err);
}

/*
 * A host wait that may cover communication RCCL_wait does not drain:
 * stream-ordered isend/irecv on the p2p streams, and anything queued
 * behind them. No communicator to name; WHAT names the call site.
 */
void HPLMXP_comm_sync(hipStream_t STREAM, const char* WHAT) {
  HPLMXP_rccl_watch_begin(WHAT, HPLMXP_COMM_NULL, HPLMXP_RCCL_PROGRESS);
  hipError_t err =
      STREAM ? hipStreamSynchronize(STREAM) : hipDeviceSynchronize();
  HPLMXP_rccl_watch_end();
  if(HPLMXP_rccl_aborting())
    for(;;) pause();
  HIP_CHECK(err);
}

void* HPLMXP_rccl_scratch(size_t bytes) {
  if(bytes > scratch_bytes) {
    if(scratch != nullptr) HIP_CHECK(hipFree(scratch));
    scratch_bytes = std::max(bytes, (size_t)4096);
    HIP_CHECK(hipMalloc(&scratch, scratch_bytes));
  }
  return scratch;
}

double HPLMXP_wtime() { return now(); }
