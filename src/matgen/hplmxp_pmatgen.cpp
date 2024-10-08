
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
