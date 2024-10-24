
#include "hplmxp.hpp"
#include "hplmxp_rand.hpp"

/* 128, 256, 512 or 1024 */
#define GEMV_DIM_KNL 256

__global__ void otf_gemv_knl(const int n,
                             const int nb,
                             const int jpos,
                             const int myrow,
                             const int nprow,
                             RandStat  stat_ij,
                             RandCoeff jump_i,
                             RandCoeff jump_j,
                             RandCoeff jump_pnb,
                             fp64_t const* __restrict__ diag,
                             fp64_t alpha,
                             fp64_t const* __restrict__ x,
                             fp64_t* __restrict__ y) {

  __shared__ fp64_t sh[GEMV_DIM_KNL];

  const int i  = blockIdx.x; // row number in block
  const int bi = blockIdx.y; // local row block number

  RandCoeff jump_tb = pow(jump_j, GEMV_DIM_KNL);

  stat_ij *= pow(jump_pnb, bi); // shift down to row block
  stat_ij *= pow(jump_i, i);    // shift down to row

  RandStat stat_d = stat_ij;

  stat_ij *=
      pow(jump_j, threadIdx.x); // each thread shifts right to distinct column

  fp64_t Axi = 0;
  for(int j = threadIdx.x; j < nb; j += blockDim.x) {
    Axi += stat_ij.toDouble() * x[j];
    stat_ij *= jump_tb;
  }

  // check for diagonal
  const int id   = threadIdx.x;
  const int ipos = myrow + bi * nprow; // global row block number
  if(ipos == jpos) {
    if(id == 0) {
      stat_d *= pow(jump_j, i); // shift state right to diagonal entry
      Axi -= stat_d.toDouble() * x[i];
      Axi += diag[i + bi * nb] * x[i];
    }
  }

  /* init */
  sh[id] = Axi;
  __syncthreads();

#if GEMV_DIM_KNL > 512
  if(id < 512) sh[id] += sh[id + 512];
  __syncthreads();
#endif
#if GEMV_DIM_KNL > 256
  if(id < 256) sh[id] += sh[id + 256];
  __syncthreads();
#endif
#if GEMV_DIM_KNL > 128
  if(id < 128) sh[id] += sh[id + 128];
  __syncthreads();
#endif
  if(id < 64) sh[id] += sh[id + 64];
  __syncthreads();
  if(id < 32) sh[id] += sh[id + 32];
  __syncthreads();
  if(id < 16) sh[id] += sh[id + 16];
  __syncthreads();
  if(id < 8) sh[id] += sh[id + 8];
  __syncthreads();
  if(id < 4) sh[id] += sh[id + 4];
  __syncthreads();
  if(id < 2) sh[id] += sh[id + 2];
  __syncthreads();
  if(id == 0) y[i + bi * nb] += alpha * (sh[0] + sh[1]);
}

static __launch_bounds__(256) __global__
    void pgemv_init_knl(const int    nbrow,
                        const int    b,
                        const int    myrow,
                        const int    mycol,
                        const int    nprow,
                        const int    npcol,
                        const fp64_t beta,
                        fp64_t* __restrict__ y) {

  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < b * nbrow) {
    const int n    = i / b;
    const int ipos = myrow + n * nprow;

    if((ipos % npcol) == mycol) {
      // diagonal block
      y[i] = beta * y[i];
    } else {
      y[i] = 0.0;
    }
  }
}

void HPLMXP_pgemv(HPLMXP_T_grid&         grid,
                  HPLMXP_T_pmat<approx_type_t,
                                compute_type_t>& A,
                  fp64_t                 alpha,
                  fp64_t*                x,
                  fp64_t                 beta,
                  fp64_t*                y) {

  int const n     = A.n;
  int const nb    = A.nb;
  int const nbrow = A.nbrow;
  int const nbcol = A.nbcol;

  int const myrow = grid.myrow;
  int const mycol = grid.mycol;
  int const nprow = grid.nprow;
  int const npcol = grid.npcol;

  RandCoeff jump_i = RandCoeff::default_vals();
  RandCoeff jump_j = pow(jump_i, n);

  RandCoeff jump_pnb = pow(jump_i, nb * nprow);
  RandCoeff jump_qnb = pow(jump_j, nb * npcol);

  // Starting state of this process's panels
  RandStat stat_ij = RandStat::initialize(HPLMXP_SEED);
  stat_ij *= pow(jump_i, nb * myrow);
  stat_ij *= pow(jump_j, nb * mycol);

  /* sync to ensure x is ready */
  HIP_CHECK(hipStreamSynchronize(computeStream));

  // first: initialize y data
  dim3 bs(256);
  dim3 gs((nb * nbrow + 256 - 1) / 256);
  pgemv_init_knl<<<gs, bs, 0, computeStream>>>(
      nbrow, nb, myrow, mycol, nprow, npcol, beta, y);
  HIP_CHECK(hipGetLastError());

  bs = dim3(GEMV_DIM_KNL);
  gs = dim3(nb, nbrow);

  // loop through column blocks
  for(int bj = 0; bj < nbcol; ++bj) {

    const int jpos    = mycol + bj * npcol; // global column block number
    const int rootrow = jpos % nprow;

    if(nprow > 1 && myrow != rootrow)
      MPI_Bcast(x + bj * nb, nb, T2MPI<fp64_t>::type, rootrow, grid.col_comm);

    otf_gemv_knl<<<gs, bs, 0, computeStream>>>(n,
                                               nb,
                                               jpos,
                                               myrow,
                                               nprow,
                                               stat_ij,
                                               jump_i,
                                               jump_j,
                                               jump_pnb,
                                               A.d,
                                               alpha,
                                               x + bj * nb,
                                               y);
    HIP_CHECK(hipGetLastError());

    stat_ij *= jump_qnb; // shift by Q*NB columns to next panel

    if(nprow > 1 && myrow == rootrow)
      MPI_Bcast(x + bj * nb, nb, T2MPI<fp64_t>::type, rootrow, grid.col_comm);
  }

  /* sync */
  HIP_CHECK(hipStreamSynchronize(computeStream));

  MPI_Allreduce(
      MPI_IN_PLACE, y, nb * nbrow, T2MPI<fp64_t>::type, MPI_SUM, grid.row_comm);
}
