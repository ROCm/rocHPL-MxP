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

rocblas_handle    blas_hdl;
hipblasLtHandle_t hipblaslt_handle;
hipStream_t    computeStream;
rocblas_int*   blas_info;
fp64_t*        reduction_scratch;
fp64_t*        h_reduction_scratch;

hipblasLtMatrixLayout_t a_layout;
hipblasLtMatrixLayout_t b_layout;
hipblasLtMatrixLayout_t c_layout;

hipblasLtMatmulDesc_t            matmul32;
hipblasLtMatmulDesc_t            matmul64;
hipblasLtMatmulPreference_t      pref;
hipblasLtMatmulHeuristicResult_t heuristicResult;

hipEvent_t getrf, lbcast, ubcast;
hipEvent_t piv;
hipEvent_t DgemmStart, DgemmEnd;
hipEvent_t LgemmStart, LgemmEnd;
hipEvent_t UgemmStart, UgemmEnd;
hipEvent_t TgemmStart, TgemmEnd;

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


  /* Create a rocBLAS handle */
  ROCBLAS_CHECK(rocblas_create_handle(&blas_hdl));
  ROCBLAS_CHECK(rocblas_set_pointer_mode(blas_hdl, rocblas_pointer_mode_host));
  ROCBLAS_CHECK(rocblas_set_stream(blas_hdl, computeStream));

#ifdef HPLMXP_ROCBLAS_ALLOW_ATOMICS
  ROCBLAS_CHECK(rocblas_set_atomics_mode(blas_hdl, rocblas_atomics_allowed));
#else
  ROCBLAS_CHECK(
      rocblas_set_atomics_mode(blas_hdl, rocblas_atomics_not_allowed));
#endif

  rocblas_initialize();

  HIPBLAS_CHECK(hipblasLtCreate(&hipblaslt_handle));

  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&a_layout, HIP_R_16F, 1, 1, 1));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&b_layout, HIP_R_16F, 1, 1, 1));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&c_layout, HIP_R_32F, 1, 1, 1));

  HIPBLAS_CHECK(hipblasLtMatmulDescCreate(&matmul32, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  HIPBLAS_CHECK(hipblasLtMatmulDescCreate(&matmul64, HIPBLAS_COMPUTE_64F, HIP_R_64F));

  // Set User Preference attributes
  int64_t max_workspace_size = 0;
  HIPBLAS_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
  HIPBLAS_CHECK(hipblasLtMatmulPreferenceSetAttribute(pref,
                                                     HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     &max_workspace_size,
                                                     sizeof(max_workspace_size)));
}

void HPLMXP_FreeGPU() {
  ROCBLAS_CHECK(rocblas_destroy_handle(blas_hdl));

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
