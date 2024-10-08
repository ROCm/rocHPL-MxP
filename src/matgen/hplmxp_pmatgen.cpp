
#include "hplmxp.hpp"

static int deviceMalloc(HPLMXP_T_grid&  grid,
                        void**          ptr,
                        const size_t    bytes) {

  hipError_t err = hipMalloc(ptr, bytes);

  /*Check allocation is valid*/
  int error = (err != hipSuccess);
  HPLMXP_all_reduce(&error, 1, HPLMXP_MAX, grid.all_comm);
  if(error != 0) {
    return HPLMXP_FAILURE;
  } else {
    return HPLMXP_SUCCESS;
  }
}

template <typename T>
int HPLMXP_pmatgen(HPLMXP_T_grid& grid,
                   HPLMXP_T_pmat<T>& A,
                   size_t& totalMem) {

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
  size_t numbytes = sizeof(T) * b * nbcol * A.ld;
  totalMem += numbytes;
  if(deviceMalloc(grid, reinterpret_cast<void**>(&(A.A)), numbytes) != HPLMXP_SUCCESS) {
    if(grid.iam == 0)
      HPLMXP_pwarn(stderr,
                   __LINE__,
                   "HPLMXP_pdmatgen",
                   "Device memory allocation failed for A. Requested %g GiBs total. Test Skiped.",
                   ((double)totalMem) / (1024 * 1024 * 1024));
    return HPLMXP_FAILURE;
  }

#ifdef HPLMXP_VERBOSE_PRINT
  if((myrow == 0) && (mycol == 0)) {
    printf("Local matrix size       = %g GiBs\n", ((double)numbytes) / (1024 * 1024 * 1024));
  }
#endif

  /* piv */
  const int ldpiv = b;

  numbytes = sizeof(fp32_t) * b * ldpiv;
  totalMem += numbytes;
  if(deviceMalloc(grid, reinterpret_cast<void**>(&(A.piv)), numbytes) != HPLMXP_SUCCESS) {
    if(grid.iam == 0)
      HPLMXP_pwarn(stderr,
                   __LINE__,
                   "HPLMXP_pdmatgen",
                   "Device memory allocation failed for piv. Requested %g GiBs total. Test Skiped.",
                   ((double)totalMem) / (1024 * 1024 * 1024));
    return HPLMXP_FAILURE;
  }
  numbytes = sizeof(fp16_t) * b * ldpiv;
  totalMem += numbytes;
  if(deviceMalloc(grid, reinterpret_cast<void**>(&(A.pivL)), numbytes) != HPLMXP_SUCCESS) {
    if(grid.iam == 0)
      HPLMXP_pwarn(stderr,
                   __LINE__,
                   "HPLMXP_pdmatgen",
                   "Device memory allocation failed for pivL. Requested %g GiBs total. Test Skiped.",
                   ((double)totalMem) / (1024 * 1024 * 1024));
    return HPLMXP_FAILURE;
  }
  totalMem += numbytes;
  if(deviceMalloc(grid, reinterpret_cast<void**>(&(A.pivU)), numbytes) != HPLMXP_SUCCESS) {
    if(grid.iam == 0)
      HPLMXP_pwarn(stderr,
                   __LINE__,
                   "HPLMXP_pdmatgen",
                   "Device memory allocation failed for pivL. Requested %g GiBs total. Test Skiped.",
                   ((double)totalMem) / (1024 * 1024 * 1024));
    return HPLMXP_FAILURE;
  }

  int ierr = 0;
  ierr = HPLMXP_pdpanel_new(grid, A, n, b, 0, 0, 0, 0, A.panels[0], totalMem);
  if (ierr == HPLMXP_FAILURE) return HPLMXP_FAILURE;
  ierr = HPLMXP_pdpanel_new(grid, A, n, b, 0, 0, 0, 0, A.panels[1], totalMem);
  if (ierr == HPLMXP_FAILURE) return HPLMXP_FAILURE;

  numbytes = sizeof(fp64_t) * (3 * b * nbrow + b * nbcol + b * b + b);
  totalMem += numbytes;
  if(deviceMalloc(grid, reinterpret_cast<void**>(&(A.work)), numbytes) != HPLMXP_SUCCESS) {
    if(grid.iam == 0)
      HPLMXP_pwarn(stderr,
                   __LINE__,
                   "HPLMXP_pdmatgen",
                   "Device memory allocation failed for workspace. Requested %g GiBs total. Test Skiped.",
                   ((double)totalMem) / (1024 * 1024 * 1024));
    return HPLMXP_FAILURE;
  }

  // initialize panel spaces to identity
  HPLMXP_identity(b, A.pivL, ldpiv);
  HPLMXP_identity(b, A.pivU, ldpiv);

  return HPLMXP_SUCCESS;
}

template int HPLMXP_pmatgen(HPLMXP_T_grid& grid,
                            HPLMXP_T_pmat<double>& A,
                            size_t& totalMem);

template int HPLMXP_pmatgen(HPLMXP_T_grid& grid,
                            HPLMXP_T_pmat<float>& A,
                            size_t& totalMem);

template <typename T>
int HPLMXP_pmatgen_rhs(HPLMXP_T_grid& grid,
                       HPLMXP_T_pmat<T>& A,
                       size_t& totalMem) {

  int n     = A.n;
  int nb    = A.nb;
  int nbrow = A.nbrow;
  int myrow = grid.myrow;
  int mycol = grid.mycol;
  int nprow = grid.nprow;
  int npcol = grid.npcol;

  size_t numbytes = sizeof(T) * nb * nbrow;
  totalMem += numbytes;
  if(deviceMalloc(grid, reinterpret_cast<void**>(&(A.b)), numbytes) != HPLMXP_SUCCESS) {
    if(grid.iam == 0)
      HPLMXP_pwarn(stderr,
                   __LINE__,
                   "HPLMXP_pmatgen_rhs",
                   "Device memory allocation failed for b. Requested %g GiBs total. Test Skiped.",
                   ((double)totalMem) / (1024 * 1024 * 1024));
    return HPLMXP_FAILURE;
  }

  return HPLMXP_SUCCESS;
}

template int HPLMXP_pmatgen_rhs(HPLMXP_T_grid& grid,
                                HPLMXP_T_pmat<double>& A,
                                size_t& totalMem);

template int HPLMXP_pmatgen_rhs(HPLMXP_T_grid& grid,
                                HPLMXP_T_pmat<float>& A,
                                size_t& totalMem);

template <typename T>
int HPLMXP_pmatgen_x(HPLMXP_T_grid& grid,
                     HPLMXP_T_pmat<T>& A,
                     size_t& totalMem) {

  int n     = A.n;
  int nb    = A.nb;
  int nbrow = A.nbrow;
  int nbcol = A.nbcol;
  int myrow = grid.myrow;
  int mycol = grid.mycol;
  int nprow = grid.nprow;
  int npcol = grid.npcol;

  size_t numbytes = sizeof(T) * nb * nbrow;
  totalMem += numbytes;
  if(deviceMalloc(grid, reinterpret_cast<void**>(&(A.x)), numbytes) != HPLMXP_SUCCESS) {
    if(grid.iam == 0)
      HPLMXP_pwarn(stderr,
                   __LINE__,
                   "HPLMXP_pmatgen_x",
                   "Device memory allocation failed for x. Requested %g GiBs total. Test Skiped.",
                   ((double)totalMem) / (1024 * 1024 * 1024));
    return HPLMXP_FAILURE;
  }
  totalMem += numbytes;
  if(deviceMalloc(grid, reinterpret_cast<void**>(&(A.d)), numbytes) != HPLMXP_SUCCESS) {
    if(grid.iam == 0)
      HPLMXP_pwarn(stderr,
                   __LINE__,
                   "HPLMXP_pmatgen_x",
                   "Device memory allocation failed for d. Requested %g GiBs total. Test Skiped.",
                   ((double)totalMem) / (1024 * 1024 * 1024));
    return HPLMXP_FAILURE;
  }

  return HPLMXP_SUCCESS;
}

template int HPLMXP_pmatgen_x(HPLMXP_T_grid& grid,
                              HPLMXP_T_pmat<double>& A,
                              size_t& totalMem);

template int HPLMXP_pmatgen_x(HPLMXP_T_grid& grid,
                              HPLMXP_T_pmat<float>& A,
                              size_t& totalMem);

void HPLMXP_Warmup(HPLMXP_T_grid&         grid,
                   HPLMXP_T_palg&         algo,
                   HPLMXP_T_pmat<fp64_t>& A,
                   HPLMXP_T_pmat<fp32_t>& LU) {

  /* Generate problem */
  HPLMXP_prandmat(grid, LU);
  HPLMXP_prandmat_rhs(grid, A);
  HPLMXP_prandmat_x(grid, A);

  fp32_t*   Ap      = LU.A;
  int const b       = std::min(LU.nb, LU.mp);
  int const lda     = LU.ld;

  const int ldpiv = b;

  HPLMXP_pdpanel_init(grid, LU, LU.n, b, 0, 0, 0, 0, LU.panels[0]);
  HPLMXP_pdpanel_init(grid, LU, LU.n, b, 0, 0, 0, 0, LU.panels[1]);

  HPLMXP_lacpy(b, b, Ap, lda, LU.piv, ldpiv);
  HIP_CHECK(hipDeviceSynchronize());
  HPLMXP_bcast(LU.piv, ldpiv * b, 0, grid.col_comm, algo.btopo);
  HPLMXP_bcast(LU.piv, ldpiv * b, 0, grid.row_comm, algo.btopo);

  HPLMXP_getrf(b, b, LU.piv, ldpiv);
  HPLMXP_lacpy(b, b, LU.piv, ldpiv, Ap, lda);
  HPLMXP_trtriU(b, LU.piv, ldpiv);
  HPLMXP_trtriL(b, LU.piv, ldpiv);

  HPLMXP_latcpyU(b, b, LU.piv, ldpiv, LU.pivU, ldpiv);
  HPLMXP_lacpyL(b, b, LU.piv, ldpiv, LU.pivL, ldpiv);

  HPLMXP_lacpy(LU.mp,
               b,
               Ap,
               lda,
               LU.panels[0].L,
               LU.panels[0].ldl);
  HPLMXP_lacpy(LU.mp,
               b,
               Ap,
               lda,
               LU.panels[1].L,
               LU.panels[1].ldl);

  HPLMXP_latcpy(LU.nq,
                b,
                Ap,
                lda,
                LU.panels[0].U,
                LU.panels[0].ldu);
  HPLMXP_latcpy(LU.nq,
                b,
                Ap,
                lda,
                LU.panels[1].U,
                LU.panels[1].ldu);

  HIP_CHECK(hipDeviceSynchronize());
  HPLMXP_bcast(LU.panels[0].L, LU.panels[0].ldl * b, 0, grid.row_comm, algo.btopo);
  HPLMXP_bcast(LU.panels[1].L, LU.panels[1].ldl * b, 0, grid.row_comm, algo.btopo);
  HPLMXP_bcast(LU.panels[0].U, LU.panels[0].ldu * b, 0, grid.col_comm, algo.btopo);
  HPLMXP_bcast(LU.panels[1].U, LU.panels[1].ldu * b, 0, grid.col_comm, algo.btopo);

  const fp32_t one   = 1.0;
  const fp32_t alpha = -1.0;
  const fp32_t beta  = 1.0;

  HPLMXP_gemmNT(b,
                b,
                b,
                alpha,
                LU.panels[0].L,
                LU.panels[0].ldl,
                LU.panels[0].U,
                LU.panels[0].ldu,
                beta,
                Ap,
                lda);

  HPLMXP_gemmNT(LU.mp,
                b,
                b,
                one,
                LU.panels[0].L,
                LU.panels[0].ldl,
                LU.pivU,
                ldpiv,
                fp32_t{0.0},
                Ap,
                lda);

  HPLMXP_gemmNT(b,
                LU.nq,
                b,
                one,
                LU.pivL,
                ldpiv,
                LU.panels[0].U,
                LU.panels[0].ldu,
                fp32_t{0.0},
                Ap,
                lda);

  HPLMXP_gemmNT(LU.mp,
                LU.nq,
                b,
                alpha,
                LU.panels[0].L,
                LU.panels[0].ldl,
                LU.panels[0].U,
                LU.panels[0].ldu,
                beta,
                Ap,
                lda);

  fp64_t* r = LU.work;
  fp64_t* v = LU.work + b * LU.nbrow;
  fp64_t* work = LU.work + b * LU.nbrow + b * LU.nbcol;

  fp64_t* x = A.x;

  HPLMXP_pcopy(grid, b * A.nbrow, b, A.b, r);

  fp64_t normx = HPLMXP_plange(grid, b * A.nbrow, b, x);

  HPLMXP_ptranspose(grid, b * A.nbrow, b, x, v);

  HPLMXP_pgemv(grid, A, -1., v, 1., r);

  fp64_t normr = HPLMXP_plange(grid, b * A.nbrow, b, r);

  HPLMXP_ptrsvL(grid, LU, r, work);
  HPLMXP_ptrsvU(grid, LU, r, work);
  HPLMXP_paxpy(grid, b * A.nbrow, b, 1.0, r, x);
}
