/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    HPL - 2.2 - February 24, 2016
 *    Antoine P. Petitet
 *    University of Tennessee, Knoxville
 *    Innovative Computing Laboratory
 *    (C) Copyright 2000-2008 All Rights Reserved
 *
 *    Modified by: Noel Chalmers
 *    (C) 2018-2022 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

#include "hplmxp.hpp"

template <typename T>
void HPLMXP_pdpanel_init(HPLMXP_T_grid&      grid,
                         HPLMXP_T_pmat<T>&   A,
                         const int           N,
                         const int           NB,
                         const int           IA,
                         const int           JA,
                         const int           II,
                         const int           JJ,
                         HPLMXP_T_panel<T>&  P) {
  /*
   * Purpose
   * =======
   *
   * HPLMXP_pdpanel_init initializes a panel data structure.
   *
   *
   * Arguments
   * =========
   *
   * grid    (local input)                 HPLMXP_T_grid &
   *         On entry,  grid  points  to the data structure containing the
   *         process grid information.
   *
   * A       (local input/output)          HPLMXP_T_pmat &
   *         On entry, A points to the data structure containing the local
   *         array information.
   *
   * N       (local input)                 const int
   *         On entry,  N  specifies  the  global number of columns of the
   *         panel and trailing submatrix. N must be at least zero.
   *
   * NB      (global input)                const int
   *         On entry, NB specifies is the number of columns of the panel.
   *         NB must be at least zero.
   *
   * IA      (global input)                const int
   *         On entry,  IA  is  the global row index identifying the panel
   *         and trailing submatrix. IA must be at least zero.
   *
   * JA      (global input)                const int
   *         On entry, JA is the global column index identifying the panel
   *         and trailing submatrix. JA must be at least zero.
   *
   * II      (local input)                 const int
   *         On entry, II  specifies the  local  starting  row index of the
   *         submatrix.
   *
   * JJ      (local input)                 const int
   *         On entry, JJ  specifies the local starting column index of the
   *         submatrix.
   *
   * PANEL   (local input/output)          HPLMXP_T_panel &
   *         On entry,  PANEL  points to the data structure containing the
   *         panel information.
   *
   * ---------------------------------------------------------------------
   */

  P.grid = &grid; /* ptr to the process grid */
  P.pmat = &A;    /* ptr to the local array info */

  /* ptr to trailing part of A */
  P.A = Mptr(A.A, II, JJ, A.ld);

  /*
   * Local lengths, indexes process coordinates
   */
  P.nb  = NB;        /* distribution blocking factor */
  P.n   = N;         /* global # of cols of trailing part of A */
  P.ia  = IA;        /* global row index of trailing part of A */
  P.ja  = JA;        /* global col index of trailing part of A */
  P.ii  = II;        /* local row index of trailing part of A */
  P.jj  = JJ;        /* local col index of trailing part of A */
  P.mp  = A.mp - II; /* local # of rows of trailing part of A */
  P.nq  = A.nq - JJ; /* local # of cols of trailing part of A */
  P.lda = A.ld;      /* local leading dim of array A */

  const int icurrow = (IA / NB) % grid.nprow;
  const int icurcol = (JA / NB) % grid.npcol;
  P.prow            = icurrow; /* proc row owning 1st row of trailing A */
  P.pcol            = icurcol; /* proc col owning 1st col of trailing A */

  P.ldl = (((sizeof(fp16_t) * P.mp + 767) / 1024) * 1024 + 256) / sizeof(fp16_t);
  P.ldu = (((sizeof(fp16_t) * P.nq + 767) / 1024) * 1024 + 256) / sizeof(fp16_t);
}

template
void HPLMXP_pdpanel_init(HPLMXP_T_grid&          grid,
                         HPLMXP_T_pmat<double>&  A,
                         const int               N,
                         const int               NB,
                         const int               IA,
                         const int               JA,
                         const int               II,
                         const int               JJ,
                         HPLMXP_T_panel<double>& P);

template
void HPLMXP_pdpanel_init(HPLMXP_T_grid&          grid,
                         HPLMXP_T_pmat<float>&   A,
                         const int               N,
                         const int               NB,
                         const int               IA,
                         const int               JA,
                         const int               II,
                         const int               JJ,
                         HPLMXP_T_panel<float>&  P);
