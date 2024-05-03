/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    Noel Chalmers
 *    (C) 2018-2022 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

#include "hplmxp.hpp"
#include <random>

rocblas_handle blas_hdl;
hipStream_t    computeStream;
rocblas_int*   blas_info;
fp64_t*        reduction_scratch;
fp64_t*        h_reduction_scratch;

hipEvent_t getrf, lbcast, ubcast;
hipEvent_t piv;
hipEvent_t DgemmStart, DgemmEnd, LgemmStart, LgemmEnd, UgemmStart, UgemmEnd,
    TgemmStart, TgemmEnd;

static char host_name[MPI_MAX_PROCESSOR_NAME];

/*
  This function finds out how many MPI processes are running on the same node
  and assigns a local rank that can be used to map a process to a device.
  This function needs to be called by all the MPI processes.
*/
void HPLMXP_InitGPU(const HPLMXP_T_grid& grid) {
  char host_name[MPI_MAX_PROCESSOR_NAME];

  int i, n, namelen, rank, nprocs;
  int dev;

  int nprow, npcol, myrow, mycol;
  HPLMXP_grid_info(grid, nprow, npcol, myrow, mycol);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  MPI_Get_processor_name(host_name, &namelen);

  int localRank = grid.local_mycol + grid.local_myrow * grid.local_npcol;
  int localSize = grid.local_npcol * grid.local_nprow;

  /* Find out how many GPUs are in the system and their device number */
  int deviceCount;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));

  if(deviceCount < 1) {
    if(localRank == 0)
      HPLMXP_pabort(__LINE__,
                    "HPLMXP_InitGPU",
                    "Node %s found no GPUs. Is the ROCm kernel module loaded?",
                    host_name);
    MPI_Finalize();
    exit(1);
  }

  dev = localRank % deviceCount;

#ifdef HPLMXP_VERBOSE_PRINT
  if(rank < localSize) {
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, dev));

    printf("GPU  Binding: Process %d [(p,q)=(%d,%d)] GPU: %d, pciBusID %x \n",
           rank,
           grid.local_myrow,
           grid.local_mycol,
           dev,
           props.pciBusID);
  }
#endif

  /* Assign device to MPI process, initialize BLAS and probe device properties
   */
  HIP_CHECK(hipSetDevice(dev));

  /* gpu */
  HIP_CHECK(hipMalloc(&blas_info, sizeof(rocblas_int)));
  HIP_CHECK(
      hipMalloc(&reduction_scratch, sizeof(double) * REDUCTION_SCRATCH_SIZE));
  HIP_CHECK(hipHostMalloc(&h_reduction_scratch, sizeof(double)));

  HIP_CHECK(hipStreamCreate(&computeStream));

  HIP_CHECK(hipEventCreateWithFlags(&getrf, hipEventDisableTiming));
  HIP_CHECK(hipEventCreateWithFlags(&lbcast, hipEventDisableTiming));
  HIP_CHECK(hipEventCreateWithFlags(&ubcast, hipEventDisableTiming));
  HIP_CHECK(hipEventCreate(&DgemmStart));
  HIP_CHECK(hipEventCreate(&DgemmEnd));
  HIP_CHECK(hipEventCreate(&LgemmStart));
  HIP_CHECK(hipEventCreate(&LgemmEnd));
  HIP_CHECK(hipEventCreate(&UgemmStart));
  HIP_CHECK(hipEventCreate(&UgemmEnd));
  HIP_CHECK(hipEventCreate(&TgemmStart));
  HIP_CHECK(hipEventCreate(&TgemmEnd));
  HIP_CHECK(hipEventCreate(&piv));

  rocblas_initialize();
}

void HPLMXP_WarmupGPU(const int NB) {

  std::mt19937                     rng(0);
  std::uniform_real_distribution<> dist(-1., 1.);

  // Make a diagonally dominate matrix
  fp32_t* work32 = (fp32_t*)malloc(sizeof(fp32_t) * NB * NB);
  for(int j = 0; j < NB; ++j) {
    for(int i = 0; i < NB; ++i) { work32[i + j * NB] = dist(rng); }
    work32[j + j * NB] = NB;
  }

  fp64_t* work64 = (fp64_t*)malloc(sizeof(fp64_t) * NB * (NB + 1));
  for(int j = 0; j < NB; ++j) {
    for(int i = 0; i < NB; ++i) { work64[i + j * NB] = dist(rng); }
    work64[j + j * NB] = NB;
  }
  for(int i = 0; i < NB; ++i) { work64[i + NB * NB] = dist(rng); }

  fp32_t* d_work32 = nullptr;
  if(hipMalloc((void**)&(d_work32), sizeof(fp32_t) * NB * NB) != hipSuccess) {
    HPLMXP_pabort(__LINE__,
                  "HPLMXP_WarmupGPU",
                  "Device memory allocation failed for warmup workspace.");
  }

  fp64_t* d_work64 = nullptr;
  if(hipMalloc((void**)&(d_work64), sizeof(fp64_t) * NB * (NB + 1)) !=
     hipSuccess) {
    HPLMXP_pabort(__LINE__,
                  "HPLMXP_WarmupGPU",
                  "Device memory allocation failed for warmup workspace.");
  }

  HIP_CHECK(hipMemcpy(
      d_work32, work32, sizeof(fp32_t) * NB * NB, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(
      d_work64, work64, sizeof(fp64_t) * NB * (NB + 1), hipMemcpyHostToDevice));

  // Call some rocsovler routines to warm up
  HPLMXP_getrf(NB, NB, d_work32, NB);
  HPLMXP_trtriU(NB, d_work32, NB);
  HPLMXP_trtriL(NB, d_work32, NB);

  // And call some rocblas routines as well
  HPLMXP_trsvL(NB, d_work64, NB, d_work64 + NB * NB);
  HPLMXP_trsvU(NB, d_work64, NB, d_work64 + NB * NB);

  HIP_CHECK(hipFree(d_work64));
  HIP_CHECK(hipFree(d_work32));
  free(work64);
  free(work32);
}

void HPLMXP_FreeGPU() {
  HIP_CHECK(hipEventDestroy(getrf));
  HIP_CHECK(hipEventDestroy(lbcast));
  HIP_CHECK(hipEventDestroy(ubcast));
  HIP_CHECK(hipEventDestroy(DgemmStart));
  HIP_CHECK(hipEventDestroy(DgemmEnd));
  HIP_CHECK(hipEventDestroy(LgemmStart));
  HIP_CHECK(hipEventDestroy(LgemmEnd));
  HIP_CHECK(hipEventDestroy(UgemmStart));
  HIP_CHECK(hipEventDestroy(UgemmEnd));
  HIP_CHECK(hipEventDestroy(TgemmStart));
  HIP_CHECK(hipEventDestroy(TgemmEnd));
  HIP_CHECK(hipEventDestroy(piv));

  HIP_CHECK(hipFree(reduction_scratch));
  HIP_CHECK(hipFree(blas_info));

  HIP_CHECK(hipStreamDestroy(computeStream));
}
