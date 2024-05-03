
#include "hplmxp.hpp"
#include "hplmxp_rand.hpp"

/* 128, 256, 512, 1024 */
#define MATGEN_DIAG_DIM 256

template <typename T>
__global__ void matgen_diag(const int n,
                            const int nb,
                            const int nbcol,
                            const int myrow,
                            const int mycol,
                            const int nprow,
                            const int npcol,
                            RandStat  stat_ij,
                            RandCoeff jump_i,
                            RandCoeff jump_j,
                            RandCoeff jump_pnb,
                            RandCoeff jump_qnb,
                            T* __restrict__ d) {

  __shared__ double s_d[MATGEN_DIAG_DIM];

  const int il = blockIdx.x; // local row index
  const int bi = il / nb;    // row block
  const int i  = il % nb;    // row index in block

  stat_ij *= pow(jump_pnb, bi); // shift state down to row block
  stat_ij *= pow(jump_i, i);    // shift state down to row i

  RandStat stat_d = stat_ij;

  stat_ij *= pow(jump_j, threadIdx.x); // each thread shifts to a distinct
                                       // column

  RandCoeff jump_tb = pow(jump_j, MATGEN_DIAG_DIM);

  double dj = 0.0;

  // loop through panels of columns
  for(int b = 0; b < nbcol; ++b) {
    RandStat stat_j = stat_ij;
    // loop through columns in panel
    for(int j = threadIdx.x; j < nb; j += MATGEN_DIAG_DIM) {
      dj += std::abs(stat_j.toDouble());
      stat_j *= jump_tb; // shift by MATGEN_DIAG_DIM columns
    }
    stat_ij *= jump_qnb; // shift by Q*NB columns to next panel
  }

  // Save the partial result in shmem and reduce
  const int t = threadIdx.x;

  // Check for diagonal
  const int ipos = myrow + bi * nprow;
  if(ipos % npcol == mycol) {
    if(t == 0) {
      const int bj = ipos / npcol;
      stat_d *= pow(jump_qnb, bj); // shift state right to column block
      stat_d *= pow(jump_j, i);    // shift state right to diagonal entry
      dj -= std::abs(stat_d.toDouble());
    }
  }

  s_d[t] = dj;
  __syncthreads();

#if MATGEN_DIAG_DIM > 512
  if(t < 512) s_d[t] += s_d[t + 512];
  __syncthreads();
#endif
#if MATGEN_DIAG_DIM > 256
  if(t < 256) s_d[t] += s_d[t + 256];
  __syncthreads();
#endif
#if MATGEN_DIAG_DIM > 128
  if(t < 128) s_d[t] += s_d[t + 128];
  __syncthreads();
#endif
  if(t < 64) s_d[t] += s_d[t + 64];
  __syncthreads();
  if(t < 32) s_d[t] += s_d[t + 32];
  __syncthreads();
  if(t < 16) s_d[t] += s_d[t + 16];
  __syncthreads();
  if(t < 8) s_d[t] += s_d[t + 8];
  __syncthreads();
  if(t < 4) s_d[t] += s_d[t + 4];
  __syncthreads();
  if(t < 2) s_d[t] += s_d[t + 2];
  __syncthreads();
  if(t < 1) d[il] = s_d[0] + s_d[1];
}

/* 128, 256, 512, 1024 */
#define MATGEN_RHS_DIM 256

template <typename T>
__global__ void matgen_rhs(const int n,
                           const int nb,
                           const int myrow,
                           const int mycol,
                           const int nprow,
                           const int npcol,
                           RandStat  stat_rhs,
                           RandCoeff jump_i,
                           RandCoeff jump_pnb,
                           T* __restrict__ b) {

  const int id = threadIdx.x + blockIdx.x * blockDim.x; // local row index

  if(id < n) {
    const int bi   = id / nb;            // row block
    const int i    = id % nb;            // row index in block
    const int ipos = myrow + bi * nprow; // global block number

    if(ipos % npcol == mycol) {      // I own the diagonal block
      stat_rhs *= pow(jump_pnb, bi); // shift state down to row block
      stat_rhs *= pow(jump_i, i);    // shift state down to row i

      b[id] = stat_rhs.toDouble();
    } else {
      b[id] = 0.0;
    }
  }
}

#define MATGEN_DIM 256

template <typename T>
__global__ void matgen(const int n,
                       const int nb,
                       const int nbrow,
                       RandStat  stat_ij,
                       RandCoeff jump_i,
                       RandCoeff jump_j,
                       RandCoeff jump_pnb,
                       RandCoeff jump_qnb,
                       T* __restrict__ A,
                       const int lda) {

  const int j  = blockIdx.x;
  const int bj = blockIdx.y;

  // column pointer
  T* __restrict__ Aj = A + (j + bj * nb) * static_cast<size_t>(lda);

  stat_ij *= pow(jump_qnb, bj); // shift to column block
  stat_ij *= pow(jump_j, j);    // shift to column in block

  RandCoeff jump_tb = pow(jump_i, MATGEN_DIM);

  stat_ij *= pow(jump_i, threadIdx.x); // each thread shift to distinct row

  for(int bi = 0; bi < nbrow; ++bi) { // loop through row blocks

    RandStat stat_i = stat_ij;

    // loop through rows in panel
    for(int i = threadIdx.x; i < nb; i += MATGEN_DIM) {
      Aj[i] = stat_i.toDouble();
      stat_i *= jump_tb; // shift by MATGEN_DIM rows
    }

    stat_ij *= jump_pnb; // shift by P*NB rows to next panel
    Aj += nb;
  }
}

template <typename T>
__global__ void matgen_write_diag(const int n,
                                  const int nb,
                                  const int myrow,
                                  const int mycol,
                                  const int nprow,
                                  const int npcol,
                                  const double* __restrict__ d,
                                  T* __restrict__ A,
                                  const int lda) {

  const int id = threadIdx.x + blockIdx.x * blockDim.x; // local row index

  if(id < n) {
    const int bi   = id / nb;            // row block
    const int i    = id % nb;            // row index in block
    const int ipos = myrow + bi * nprow; // global block number

    if(ipos % npcol == mycol) { // I own the diagonal block
      const int bj = ipos / npcol;
      A[(i + bi * nb) + (i + bj * nb) * static_cast<size_t>(lda)] = d[id];
    }
  }
}

template <typename T>
void HPLMXP_pmatgen(HPLMXP_T_grid& grid, HPLMXP_T_pmat<T>& A) {

  int const n     = A.n;
  int const b     = A.nb;
  int const nbrow = A.nbrow;
  int const nbcol = A.nbcol;

  int const myrow = grid.myrow;
  int const mycol = grid.mycol;
  int const nprow = grid.nprow;
  int const npcol = grid.npcol;

  // Allocate matrix on device
  A.ld      = (((sizeof(T) * b * nbrow + 767) / 1024) * 1024 + 256) / sizeof(T);
  size_t sz = sizeof(T) * b * nbcol * A.ld;

#ifdef HPLMXP_VERBOSE_PRINT
  if((myrow == 0) && (mycol == 0)) {
    printf("Local matrix size = %g GBs\n", ((double)sz) / (1024 * 1024 * 1024));
  }
#endif

  if(hipMalloc(&(A.A), sz) != hipSuccess) {
    HPLMXP_pabort(
        __LINE__, "HPLMXP_pmatgen", "Memory allocation failed for A.");
  }

  double* d = nullptr;
  if(hipMalloc(&d, sizeof(double) * b * nbrow) != hipSuccess) {
    HPLMXP_pabort(
        __LINE__, "HPLMXP_pmatgen", "Memory allocation failed for d.");
  }

  RandCoeff jump_i = RandCoeff::default_vals();
  RandCoeff jump_j = pow(jump_i, n);

  // Starting state of this process's panels
  RandStat stat_ij = RandStat::initialize(HPLMXP_SEED);
  stat_ij *= pow(jump_i, b * myrow);
  stat_ij *= pow(jump_j, b * mycol);

  RandCoeff jump_pnb = pow(jump_i, b * nprow);
  RandCoeff jump_qnb = pow(jump_j, b * npcol);

  // generate diagonal in double precision
  dim3 gs = b * nbrow;
  dim3 bs = MATGEN_DIAG_DIM;
  matgen_diag<<<gs, bs, 0, computeStream>>>(n,
                                            b,
                                            nbcol,
                                            myrow,
                                            mycol,
                                            nprow,
                                            npcol,
                                            stat_ij,
                                            jump_i,
                                            jump_j,
                                            jump_pnb,
                                            jump_qnb,
                                            d);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(computeStream));

  // generate matrix
  gs = dim3(b, nbcol);
  bs = dim3(MATGEN_DIM);
  matgen<<<gs, bs, 0, computeStream>>>(
      n, b, nbrow, stat_ij, jump_i, jump_j, jump_pnb, jump_qnb, A.A, A.ld);
  HIP_CHECK(hipGetLastError());

  // assemble diagonal
  HPLMXP_all_reduce(d, b * nbrow, HPLMXP_SUM, grid.row_comm);

  // write diagonal into matrix
  gs = (b * nbrow + MATGEN_RHS_DIM - 1) / MATGEN_RHS_DIM;
  bs = MATGEN_RHS_DIM;
  matgen_write_diag<<<gs, bs, 0, computeStream>>>(
      n, b, myrow, mycol, nprow, npcol, d, A.A, A.ld);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipFree(d));
}

template void HPLMXP_pmatgen(HPLMXP_T_grid& grid, HPLMXP_T_pmat<double>& A);

template void HPLMXP_pmatgen(HPLMXP_T_grid& grid, HPLMXP_T_pmat<float>& A);

template <typename T>
void HPLMXP_pmatgen_rhs(HPLMXP_T_grid& grid, HPLMXP_T_pmat<T>& A) {

  int n     = A.n;
  int nb    = A.nb;
  int nbrow = A.nbrow;
  int myrow = grid.myrow;
  int mycol = grid.mycol;
  int nprow = grid.nprow;
  int npcol = grid.npcol;

  if(hipMalloc(&(A.b), sizeof(T) * nb * nbrow) != hipSuccess) {
    HPLMXP_pabort(
        __LINE__, "HPLMXP_pmatgen_rhs", "Memory allocation failed for b.");
  }

  RandCoeff jump_i = RandCoeff::default_vals();
  RandCoeff jump_j = pow(jump_i, n);

  // Starting state of the rhs in this process
  RandStat stat_rhs = RandStat::initialize(HPLMXP_SEED);
  stat_rhs *= pow(jump_j, n);
  stat_rhs *= pow(jump_i, nb * myrow);

  RandCoeff jump_pnb = pow(jump_i, nb * nprow);

  int gs = (nb * nbrow + MATGEN_RHS_DIM - 1) / MATGEN_RHS_DIM;
  int bs = MATGEN_RHS_DIM;
  matgen_rhs<<<gs, bs, 0, computeStream>>>(
      n, nb, myrow, mycol, nprow, npcol, stat_rhs, jump_i, jump_pnb, A.b);
  HIP_CHECK(hipGetLastError());

  A.normb = HPLMXP_plange(grid, nb * nbrow, nb, A.b);
}

template void HPLMXP_pmatgen_rhs(HPLMXP_T_grid& grid, HPLMXP_T_pmat<double>& A);

template void HPLMXP_pmatgen_rhs(HPLMXP_T_grid& grid, HPLMXP_T_pmat<float>& A);

template <typename T>
void HPLMXP_pmatgen_x(HPLMXP_T_grid& grid, HPLMXP_T_pmat<T>& A) {

  int n     = A.n;
  int nb    = A.nb;
  int nbrow = A.nbrow;
  int nbcol = A.nbcol;
  int myrow = grid.myrow;
  int mycol = grid.mycol;
  int nprow = grid.nprow;
  int npcol = grid.npcol;

  if(hipMalloc(&(A.x), sizeof(T) * nb * nbrow) != hipSuccess) {
    HPLMXP_pabort(
        __LINE__, "HPLMXP_pmatgen_x", "Memory allocation failed for x.");
  }
  if(hipMalloc(&(A.d), sizeof(T) * nb * nbrow) != hipSuccess) {
    HPLMXP_pabort(
        __LINE__, "HPLMXP_pmatgen_x", "Memory allocation failed for d.");
  }

  RandCoeff jump_i = RandCoeff::default_vals();
  RandCoeff jump_j = pow(jump_i, n);

  // Starting state of this process's panels
  RandStat stat_ij = RandStat::initialize(HPLMXP_SEED);
  stat_ij *= pow(jump_i, nb * myrow);
  stat_ij *= pow(jump_j, nb * mycol);

  RandCoeff jump_pnb = pow(jump_i, nb * nprow);
  RandCoeff jump_qnb = pow(jump_j, nb * npcol);

  int gs = nb * nbrow;
  int bs = MATGEN_DIAG_DIM;
  matgen_diag<<<gs, bs, 0, computeStream>>>(n,
                                            nb,
                                            nbcol,
                                            myrow,
                                            mycol,
                                            nprow,
                                            npcol,
                                            stat_ij,
                                            jump_i,
                                            jump_j,
                                            jump_pnb,
                                            jump_qnb,
                                            A.d);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipStreamSynchronize(computeStream));

  HPLMXP_all_reduce(A.d, nb * nbrow, HPLMXP_SUM, grid.row_comm);

  // initial approximation, x_0 = diag(A)^{-1} b
  HPLMXP_pcopy(grid, nb * nbrow, nb, A.b, A.x);
  HPLMXP_paydx(grid, nb * nbrow, nb, T{1.0}, A.d, A.x);

  // the diagonal of the hpl-mxp matrix is the sum of the absolute values of the
  // off-diagonals on the same row. therefore, twice of the diagonal is the
  // l1-norm of that row.
  A.norma = 2. * HPLMXP_plange(grid, nb * nbrow, nb, A.d);
}

template void HPLMXP_pmatgen_x(HPLMXP_T_grid& grid, HPLMXP_T_pmat<double>& A);

template void HPLMXP_pmatgen_x(HPLMXP_T_grid& grid, HPLMXP_T_pmat<float>& A);
