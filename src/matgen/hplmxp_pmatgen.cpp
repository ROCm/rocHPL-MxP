
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

template <typename A_t, typename C_t>
int HPLMXP_pmatgen(HPLMXP_T_grid&                grid,
                   HPLMXP_T_pmat<A_t, C_t>& A) {

  int const n     = A.n;
  int const b     = A.nb;
  int const nbrow = A.nbrow;
  int const nbcol = A.nbcol;

  int const myrow = grid.myrow;
  int const mycol = grid.mycol;
  int const nprow = grid.nprow;
  int const npcol = grid.npcol;

  A.totalMem = 0;

  // Allocate matrix on device
  A.ld      = (((sizeof(A_t) * b * nbrow + 767) / 1024) * 1024 + 256) / sizeof(A_t);
  size_t numbytes = sizeof(A_t) * b * nbcol * A.ld;
  A.totalMem += numbytes;
  if(deviceMalloc(grid, reinterpret_cast<void**>(&(A.A)), numbytes) != HPLMXP_SUCCESS) {
    if(grid.iam == 0)
      HPLMXP_pwarn(stderr,
                   __LINE__,
                   "HPLMXP_pmatgen",
                   "Device memory allocation failed for A. Requested %g GiBs total. Test Skiped.",
                   ((double)A.totalMem) / (1024 * 1024 * 1024));
    return HPLMXP_FAILURE;
  }

#ifdef HPLMXP_VERBOSE_PRINT
  if((myrow == 0) && (mycol == 0)) {
    printf("Local matrix size       = %g GiBs\n", ((double)numbytes) / (1024 * 1024 * 1024));
  }
#endif

  /* piv */
  const int ldpiv = b;
  using factType_t = typename HPLMXP_T_pmat<A_t, C_t>::factType_t;

  numbytes = sizeof(factType_t) * b * ldpiv;
  A.totalMem += numbytes;
  if(deviceMalloc(grid, reinterpret_cast<void**>(&(A.piv)), numbytes) != HPLMXP_SUCCESS) {
    if(grid.iam == 0)
      HPLMXP_pwarn(stderr,
                   __LINE__,
                   "HPLMXP_pmatgen",
                   "Device memory allocation failed for piv. Requested %g GiBs total. Test Skiped.",
                   ((double)A.totalMem) / (1024 * 1024 * 1024));
    return HPLMXP_FAILURE;
  }
  numbytes = sizeof(C_t) * b * ldpiv;
  A.totalMem += numbytes;
  if(deviceMalloc(grid, reinterpret_cast<void**>(&(A.pivL)), numbytes) != HPLMXP_SUCCESS) {
    if(grid.iam == 0)
      HPLMXP_pwarn(stderr,
                   __LINE__,
                   "HPLMXP_pmatgen",
                   "Device memory allocation failed for pivL. Requested %g GiBs total. Test Skiped.",
                   ((double)A.totalMem) / (1024 * 1024 * 1024));
    return HPLMXP_FAILURE;
  }
  A.totalMem += numbytes;
  if(deviceMalloc(grid, reinterpret_cast<void**>(&(A.pivU)), numbytes) != HPLMXP_SUCCESS) {
    if(grid.iam == 0)
      HPLMXP_pwarn(stderr,
                   __LINE__,
                   "HPLMXP_pmatgen",
                   "Device memory allocation failed for pivL. Requested %g GiBs total. Test Skiped.",
                   ((double)A.totalMem) / (1024 * 1024 * 1024));
    return HPLMXP_FAILURE;
  }

  int ierr = 0;
  ierr = HPLMXP_pdpanel_new(grid, A, A.panels[0]);
  if (ierr == HPLMXP_FAILURE) return HPLMXP_FAILURE;
  ierr = HPLMXP_pdpanel_new(grid, A, A.panels[1]);
  if (ierr == HPLMXP_FAILURE) return HPLMXP_FAILURE;

  numbytes = sizeof(fp64_t) * b * nbrow;
  A.totalMem += numbytes;
  if(deviceMalloc(grid, reinterpret_cast<void**>(&(A.b)), numbytes) != HPLMXP_SUCCESS) {
    if(grid.iam == 0)
      HPLMXP_pwarn(stderr,
                   __LINE__,
                   "HPLMXP_pmatgen",
                   "Device memory allocation failed for b. Requested %g GiBs total. Test Skiped.",
                   ((double)A.totalMem) / (1024 * 1024 * 1024));
    return HPLMXP_FAILURE;
  }

  numbytes = sizeof(fp64_t) * b * nbrow;
  A.totalMem += numbytes;
  if(deviceMalloc(grid, reinterpret_cast<void**>(&(A.x)), numbytes) != HPLMXP_SUCCESS) {
    if(grid.iam == 0)
      HPLMXP_pwarn(stderr,
                   __LINE__,
                   "HPLMXP_pmatgen",
                   "Device memory allocation failed for x. Requested %g GiBs total. Test Skiped.",
                   ((double)A.totalMem) / (1024 * 1024 * 1024));
    return HPLMXP_FAILURE;
  }
  A.totalMem += numbytes;
  if(deviceMalloc(grid, reinterpret_cast<void**>(&(A.d)), numbytes) != HPLMXP_SUCCESS) {
    if(grid.iam == 0)
      HPLMXP_pwarn(stderr,
                   __LINE__,
                   "HPLMXP_pmatgen",
                   "Device memory allocation failed for d. Requested %g GiBs total. Test Skiped.",
                   ((double)A.totalMem) / (1024 * 1024 * 1024));
    return HPLMXP_FAILURE;
  }

  numbytes = sizeof(fp64_t) * (3 * b * nbrow + b * nbcol + b * b + b);
  A.totalMem += numbytes;
  if(deviceMalloc(grid, reinterpret_cast<void**>(&(A.work)), numbytes) != HPLMXP_SUCCESS) {
    if(grid.iam == 0)
      HPLMXP_pwarn(stderr,
                   __LINE__,
                   "HPLMXP_pmatgen",
                   "Device memory allocation failed for workspace. Requested %g GiBs total. Test Skiped.",
                   ((double)A.totalMem) / (1024 * 1024 * 1024));
    return HPLMXP_FAILURE;
  }

#ifdef HPLMXP_VERBOSE_PRINT
  if((myrow == 0) && (mycol == 0)) {
    printf("Total device memory use = %g GiBs\n",
           ((double)A.totalMem) / (1024 * 1024 * 1024));
  }
#endif

  // initialize panel spaces to identity
  HPLMXP_identity(b, A.pivL, ldpiv);
  HPLMXP_identity(b, A.pivU, ldpiv);

  return HPLMXP_SUCCESS;
}

template int HPLMXP_pmatgen(HPLMXP_T_grid& grid,
                            HPLMXP_T_pmat<approx_type_t,
                                          compute_type_t>& A);

template <typename A_t, typename C_t>
void HPLMXP_Warmup(HPLMXP_T_grid&                grid,
                   HPLMXP_T_palg&                algo,
                   HPLMXP_T_pmat<A_t, C_t>& A) {

  /* Generate problem */
  HPLMXP_prandmat(grid, A);
  HPLMXP_prandmat_rhs(grid, A);
  HPLMXP_prandmat_x(grid, A);

  A_t*      Ap  = A.A;
  int const b   = std::min(A.nb, A.mp);
  int const lda = A.ld;

  const int ldpiv = b;

  HPLMXP_pdpanel_init(grid, A, A.n, b, 0, 0, 0, 0, A.panels[0]);
  HPLMXP_pdpanel_init(grid, A, A.n, b, 0, 0, 0, 0, A.panels[1]);

  HPLMXP_lacpy(b, b, Ap, lda, A.piv, ldpiv);
  HIP_CHECK(hipDeviceSynchronize());
  HPLMXP_bcast(A.piv, ldpiv * b, 0, grid.col_comm, algo.btopo);
  HPLMXP_bcast(A.piv, ldpiv * b, 0, grid.row_comm, algo.btopo);

  HPLMXP_getrf(b, b, A.piv, ldpiv);
  HPLMXP_lacpy(b, b, A.piv, ldpiv, Ap, lda);
  HPLMXP_trtriU(b, A.piv, ldpiv);
  HPLMXP_trtriL(b, A.piv, ldpiv);

  HPLMXP_latcpyU(b, b, A.piv, ldpiv, A.pivU, ldpiv);
  HPLMXP_lacpyL(b, b, A.piv, ldpiv, A.pivL, ldpiv);

  HPLMXP_lacpy(A.mp,
               b,
               Ap,
               lda,
               A.panels[0].L,
               A.panels[0].ldl);
  HPLMXP_lacpy(A.mp,
               b,
               Ap,
               lda,
               A.panels[1].L,
               A.panels[1].ldl);

  HPLMXP_latcpy(A.nq,
                b,
                Ap,
                lda,
                A.panels[0].U,
                A.panels[0].ldu);
  HPLMXP_latcpy(A.nq,
                b,
                Ap,
                lda,
                A.panels[1].U,
                A.panels[1].ldu);

  HIP_CHECK(hipDeviceSynchronize());
  HPLMXP_bcast(A.panels[0].L, A.panels[0].ldl * b, 0, grid.row_comm, algo.btopo);
  HPLMXP_bcast(A.panels[1].L, A.panels[1].ldl * b, 0, grid.row_comm, algo.btopo);
  HPLMXP_bcast(A.panels[0].U, A.panels[0].ldu * b, 0, grid.col_comm, algo.btopo);
  HPLMXP_bcast(A.panels[1].U, A.panels[1].ldu * b, 0, grid.col_comm, algo.btopo);

  using T = typename gemmTypes<C_t>::computeType;

  const T one   = 1.0;
  const T alpha = -1.0;
  const T beta  = 1.0;

  HPLMXP_gemmNT(b,
                b,
                b,
                alpha,
                A.panels[0].L,
                A.panels[0].ldl,
                A.panels[0].U,
                A.panels[0].ldu,
                beta,
                Ap,
                lda);

  HPLMXP_gemmNT(A.mp,
                b,
                b,
                one,
                A.panels[0].L,
                A.panels[0].ldl,
                A.pivU,
                ldpiv,
                T{0.0},
                Ap,
                lda);

  HPLMXP_gemmNT(b,
                A.nq,
                b,
                one,
                A.pivL,
                ldpiv,
                A.panels[0].U,
                A.panels[0].ldu,
                T{0.0},
                Ap,
                lda);

  HPLMXP_gemmNT(A.mp,
                A.nq,
                b,
                alpha,
                A.panels[0].L,
                A.panels[0].ldl,
                A.panels[0].U,
                A.panels[0].ldu,
                beta,
                Ap,
                lda);

  fp64_t* r = A.work;
  fp64_t* v = A.work + b * A.nbrow;
  fp64_t* work = A.work + b * A.nbrow + b * A.nbcol;

  fp64_t* x = A.x;

  HPLMXP_pcopy(grid, b * A.nbrow, b, A.b, r);

  fp64_t normx = HPLMXP_plange(grid, b * A.nbrow, b, x);

  HPLMXP_ptranspose(grid, b * A.nbrow, b, x, v);

  HPLMXP_pgemv(grid, A, fp64_t{-1.}, v, fp64_t{1.}, r);

  fp64_t normr = HPLMXP_plange(grid, b * A.nbrow, b, r);

  HPLMXP_ptrsvL(grid, A, r, work);
  HPLMXP_ptrsvU(grid, A, r, work);
  HPLMXP_paxpy(grid, b * A.nbrow, b, fp64_t{1.0}, r, x);
}

template
void HPLMXP_Warmup(HPLMXP_T_grid&                grid,
                   HPLMXP_T_palg&                algo,
                   HPLMXP_T_pmat<approx_type_t,
                                 compute_type_t>& A);
