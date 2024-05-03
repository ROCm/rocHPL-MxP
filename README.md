# rocHPL-MxP
rocHPL-MxP is a benchmark based on the [HPL-MxP][] benchmark application, implemented on top of AMD's Radeon Open Compute [ROCm][] Platform, runtime, and toolchains. rocHPL-MxP is created using the [HIP][] programming language and optimized for AMD's latest discrete GPUs.

## Requirements
* Git
* CMake (3.10 or later)
* MPI
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
#    --with-mpi=<dir>       - Path to external MPI install (Default: clone+build OpenMPI)
#    --verbose-print        - Verbose output during HPL setup (Default: true)
#    --progress-report      - Print progress report to terminal during HPL-MxP run (Default: true)
#    --detailed-timing      - Record detailed timers during HPL-MxP run (Default: true)
./install.sh
```
By default, [UCX] v1.12.1, and [OpenMPI] v4.1.4 will be cloned and built in rocHPL-MxP/tpl. After building, the `rochplmxp` executable is placed in build/rocHPL-MxP/bin.

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
[the issue tracker]: https://github.com/ROCmSoftwarePlatform/rocHPL-MxP/issues
[license file]: https://github.com/ROCmSoftwarePlatform/rocHPL-MxP
