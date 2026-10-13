# rocHPL-MxP
rocHPL-MxP is a benchmark based on the [HPL-MxP][] benchmark application, implemented on top of AMD's Radeon Open Compute [ROCm][] Platform, runtime, and toolchains. rocHPL-MxP is created using the [HIP][] programming language and optimized for AMD's latest discrete GPUs.

## Requirements
* Git
* CMake (3.10 or later)
* MPI (MPI communication backend, the default) or [RCCL][] (RCCL communication backend)
* AMD [ROCm] platform (3.5 or later)
* [rocBLAS][]
* [rocSOLVER][]

## Quickstart rocHPL-MxP build and install

#### Install script
You can build rocHPL-MxP using the `install.sh` script
```
# Clone rocHPL-MxP using git
git clone https://github.com/ROCmSoftwarePlatform/rocHPL-MxP.git

# Go to rocHPL directory
cd rocHPL-MxP

# Run install.sh script
# Command line options:
#    -h|--help              - prints this help message
#    -g|--debug             - Set build type to Debug (otherwise build Release)
#    --prefix=<dir>         - Path to rocHPL install location (Default: build/rocHPL-MxP)
#    --with-rocm=<dir>      - Path to ROCm install (Default: /opt/rocm)
#    --with-rocblas=<dir>   - Path to rocBLAS library (Default: /opt/rocm/rocblas)
#    --with-rocsolver=<dir> - Path to rocSOLVER library (Default: /opt/rocm/rocsolver)
#    --comm-backend=<b>     - Communication backend, mpi or rccl (Default: mpi)
#    --with-mpi=<dir>       - Path to external MPI install (Default: clone+build OpenMPI; mpi backend only)
#    --verbose-print        - Verbose output during HPL setup (Default: true)
#    --progress-report      - Print progress report to terminal during HPL-MxP run (Default: true)
#    --detailed-timing      - Record detailed timers during HPL-MxP run (Default: true)
./install.sh
```
By default, [UCX] v1.12.1, and [OpenMPI] v4.1.4 will be cloned and built in rocHPL-MxP/tpl. After building, the `rochplmxp` executable is placed in build/rocHPL-MxP/bin.
With `--comm-backend=rccl` no MPI is cloned or built; see [RCCL communication backend](#rccl-communication-backend).

## Running rocHPL-MxP benchmark application
rocHPL-MxP provides some helpful wrapper scripts. A wrapper script for launching via `mpirun` is provided in `mpirun_rochplmxp`. This script has two distinct run modes:
```
mpirun_rochplmxp -P <P> -Q <P> -N <N> --NB <NB>
# where
# P       - is the number of rows in the MPI grid
# Q       - is the number of columns in the MPI grid
# N       - is the total number of rows/columns of the global matrix
# NB      - is the panel size in the blocking algorithm
```
This run script will launch a total of np=PxQ MPI processes.

The second runmode takes an input file together with a number of MPI processes:
```
mpirun_rochplmxp -P <p> -Q <q> -i <input>
# where
# P       - is the number of rows in the MPI grid
# Q       - is the number of columns in the MPI grid
# input   - is the input filename (default HPL-MxP.dat)
```

The input file accpted by the `rochplmxp` executable follows the format below:
```
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL-MxP.out  output file name (if any)
0            device out (6=stdout,7=stderr,file)
1            # of problems sizes (N)
61440        Ns
1            # of NBs
2560         NBs
1            PMAP process mapping (0=Row-,1=Column-major)
1            P
1            Q
16.0         threshold
1            # of broadcast
0            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
```

The `mpirun_rochplmxp` wraps a second script, `run_rochplmxp`, wherein some library paths are set. Users wishing to launch rocHPL-MxP via a workload manager such as slurm may directly use this run script. For example,
```
srun -N 2 -n 16 run_rochplmxp -P 4 -Q 4 -N 128000 --NB 2560
```
When launching to multiple compute nodes, it can be useful to specify the local MPI grid layout on each node. To specify this, the `-p` and `-q` input parameters are used. For example, the srun line above is launching to two compute nodes, each with 8 GPUs. The local MPI grid layout can be specifed as either:
```
srun -N 2 -n 16 run_rochplmxp -P 4 -Q 4 -p 2 -q 4 -N 128000 --NB 512
```
or 
```
srun -N 2 -n 16 run_rochplmxp -P 4 -Q 4 -p 4 -q 2 -N 128000 --NB 512
```
This helps to control where/how much inter-node communication is occuring. 

## RCCL communication backend
All communication in rocHPL-MxP goes through one interface, `include/hplmxp_comm.hpp`, with two implementations selected when configuring:
```
cmake -DHPLMXP_COMM_BACKEND=MPI  ...   # src/comm/mpi, GPU-aware MPI (default)
cmake -DHPLMXP_COMM_BACKEND=RCCL ...   # src/comm/rccl, RCCL only
```
The RCCL build does not link or call MPI. Panel broadcasts and reductions are `ncclBroadcast`/`ncclAllReduce`, the triangular solves use `ncclSend`/`ncclRecv`, and the process row and column communicators are created with `ncclCommSplit`. The `BCASTs` line of the input file is ignored, since RCCL chooses its own algorithm.

#### Launching
Without MPI, each process learns its rank from the launcher environment. The first of these that is set is used:

| Launcher | Variables |
|---|---|
| explicit | `HPLMXP_RANK`, `HPLMXP_SIZE`, `HPLMXP_LOCAL_RANK`, `HPLMXP_LOCAL_SIZE` |
| Open MPI `mpirun` | `OMPI_COMM_WORLD_RANK`, `_SIZE`, `_LOCAL_RANK`, `_LOCAL_SIZE` |
| PMI (e.g. MPICH Hydra) | `PMI_RANK`, `PMI_SIZE`, `MPI_LOCALRANKID`, `MPI_LOCALNRANKS` |
| Slurm `srun` | `SLURM_PROCID`, `SLURM_STEP_NUM_TASKS`, `SLURM_LOCALID`, `SLURM_STEP_TASKS_PER_NODE` |

Other launchers, such as Flux, can be used by setting the explicit variables.

Each process binds GPU `local rank % visible GPUs`. Use block task distribution (consecutive ranks on the same node) with p*q processes per node, so that this agrees with the node-local grid; the run stops with an error otherwise.

The processes find each other through `NCCL_COMM_ID=<host>:<port>`, which must be identical on every process. Rank 0 opens the bootstrap socket there, so `<host>` must be rank 0's node. Choose a port below the kernel's ephemeral range (`/proc/sys/net/ipv4/ip_local_port_range`, often 32768-60999). Every run leaves many sockets in TIME_WAIT on ephemeral ports, and a port in that range can then fail to bind with "Address already in use". A Slurm example on 2 nodes with 8 GPUs each:
```
export NCCL_COMM_ID=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1):$((20000 + SLURM_JOB_ID % 10000))
srun -N 2 -n 16 --ntasks-per-node=8 --mpi=none --kill-on-bad-exit=1 \
     run_rochplmxp -P 4 -Q 4 -p 2 -q 4 -N 256000 --NB 2560
```
`--kill-on-bad-exit=1` matters: when one process fails, the RCCL backend aborts its communicators and exits with a nonzero code, and the launcher has to take the remaining processes down. A single process needs no `NCCL_COMM_ID`.

#### Failure detection
RCCL calls block, so a lost peer would otherwise hang the run. A watchdog thread checks every communicator for asynchronous errors and puts a deadline on each blocking wait. On an error or a missed deadline, the process prints one line naming the operation it was waiting in, aborts its communicators and exits. Note that RCCL does not detect a peer that exits with status 0 in the middle of a run; only the deadline catches that case.

| Variable | Default | Meaning |
|---|---|---|
| `HPLMXP_RCCL_TIMEOUT` | 300 | Seconds a process may wait for a collective or point-to-point operation; 0 disables. |
| `HPLMXP_RCCL_INIT_TIMEOUT` | 300 | Seconds allowed for bootstrap, communicator split and destroy; 0 disables. |
| `HPLMXP_RCCL_ABORT_TIMEOUT` | 10 | Seconds allowed for aborting the communicators before the process exits anyway. |
| `HPLMXP_RCCL_WATCHDOG_MS` | 100 | Watchdog polling interval; 0 disables the watchdog and both deadlines. |
| `HPLMXP_RCCL_WATCHDOG_VERBOSE` | unset | If 1, each process reports its longest wait at the end, to help choose `HPLMXP_RCCL_TIMEOUT`. |
| `HPLMXP_RCCL_MAX_COUNT` | 2^30 | Largest element count per RCCL call; larger operations are split. |

The usual `NCCL_*`/`RCCL_*` variables configure RCCL itself.

#### Notes
* Collectives run on a high-priority stream per communicator, so a panel broadcast overlaps the trailing update already queued on the compute stream. The overlapping RCCL kernels do take compute units from that update, which can make the RCCL build slightly slower than a GPU-aware MPI on one or two nodes.
* Point-to-point operations in the triangular solves run on normal-priority streams. A posted receive is a kernel that waits for its data, and at high priority it slows the small solve kernels it overlaps.
* The iterative refinement residuals can differ from those of the MPI build in the last digits, because the two backends add the contributions of an all-reduce in a different order.
* `comm_test` (in the build's `bin` directory) checks every operation of the communication interface on a P x Q process grid, for whichever backend was built. Run it with the same launcher, e.g. `srun -n 16 --mpi=none comm_test -P 4 -Q 4`.

#### Example: MI355X nodes with AMD Pollara 400 NICs
The settings below were used on nodes with 8 MI355X GPUs and one AMD Pollara 400 NIC (`ionic` driver, RoCE) per GPU, with ROCm 7.2.4 and RCCL 2.27.7. Run 8 processes per node.
```
export NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
export NCCL_IB_TC=104                    # traffic class of the RoCE lossless queue; site specific
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_NCHANNELS_PER_NET_PEER=4
export NCCL_GDR_FLUSH_DISABLE=1
```
`NCCL_IB_QPS_PER_CONNECTION=4` and `NCCL_NCHANNELS_PER_NET_PEER=4` are needed to reach full NIC bandwidth. `NCCL_GDR_FLUSH_DISABLE=1` lowers small-message latency between nodes, and is required with the ANP plugin.

To use the [AMD ANP][] RCCL network plugin instead of RCCL's built-in transport, also set:
```
export NCCL_NET_PLUGIN=/path/to/librccl-anp.so
export RCCL_ANP_COMM_NGROUPS=0
```
The ANP plugin had to be run with `NCCL_GDR_FLUSH_DISABLE=1`: with GDR flushes enabled, the first flush never completed. Its queue-pair sharing between communicators (`RCCL_ANP_COMM_NGROUPS` > 0) gives full NIC bandwidth, but on 8 or more nodes HPL-MxP failed with remote access errors. With `RCCL_ANP_COMM_NGROUPS=0` it ran at every scale tested, a few percent slower than the built-in transport. These issues may depend on the NIC firmware and plugin version.

## Performance evaluation
rocHPL-MxP is typically weak scaled so that the global matrix fills all available VRAM on all GPUs. The matrix size N is usually selected to be a multiple of the blocksize NB. Some sample runs on 32GB MI210 GPUs include:
* 1 MI210: `mpirun_rochplmxp -P 1 -Q 1 -N 125440 --NB 2560`
* 2 MI210: `mpirun_rochplmxp -P 2 -Q 1 -N 179200 --NB 2560`
* 4 MI210: `mpirun_rochplmxp -P 2 -Q 2 -N 250880 --NB 2560`
* 8 MI210: `mpirun_rochplmxp -P 4 -Q 2 -N 358400 --NB 2560`

Overall performance of the benchmark is measured in floating point operations (FLOPs) per second. Performance is reported at the end of the run to the user's specified output (by default the performance is printed to stdout and a results file HPL-MxP.out).

See [the Wiki](../../wiki/Common-rocHPL-MxP-run-configurations) for some common run configurations for various AMD Instinct GPUs.

## Testing rocHPL-MxP
In the final phase of the benchmark, an iterative solve is performed. The iteration must coverge to tolerance within 50 iterations in order for the test to pass. A PASS or FAIL is printed to output.

The simplest suite of tests should run configurations from 1 to 4 GPUs to exercise different communcation code paths. For example the tests:
```
mpirun_rochplmxp -P 1 -Q 1 -N 56320
mpirun_rochplmxp -P 1 -Q 2 -N 56320
mpirun_rochplmxp -P 2 -Q 1 -N 56320
mpirun_rochplmxp -P 2 -Q 2 -N 56320
```
should all report PASSED.

Please note that for successful testing, a device with at least 16GB of device memory is required.

## Support
Please use [the issue tracker][] for bugs and feature requests.

## License
The [license file][] can be found in the main repository.

[HPL-MxP]: https://hpl-mxp.org/
[ROCm]: https://github.com/RadeonOpenCompute/ROCm
[HIP]: https://github.com/ROCm-Developer-Tools/HIP
[rocBLAS]: https://github.com/ROCmSoftwarePlatform/rocBLAS
[rocSOLVER]: https://github.com/ROCmSoftwarePlatform/rocSOLVER
[OpenMPI]: https://github.com/open-mpi/ompi
[UCX]: https://github.com/openucx/ucx
[RCCL]: https://github.com/ROCm/rccl
[AMD ANP]: https://github.com/ROCm/amd-anp
[the issue tracker]: https://github.com/ROCmSoftwarePlatform/rocHPL-MxP/issues
[license file]: https://github.com/ROCmSoftwarePlatform/rocHPL-MxP
