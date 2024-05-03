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
#ifndef HPLMXP_PANEL_HPP
#define HPLMXP_PANEL_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hplmxp_pmisc.hpp"
#include "hplmxp_grid.hpp"
#include "hplmxp_pgesv.hpp"

/*
 * ---------------------------------------------------------------------
 * Data Structures
 * ---------------------------------------------------------------------
 */
typedef struct HPLMXP_S_panel {
  HPLMXP_T_grid*         grid; /* ptr to the process grid */
  HPLMXP_T_pmat<fp32_t>* pmat; /* ptr to the local array info */
  fp32_t*                A;    /* ptr to trailing part of A */
  fp16_t*                L;    /* ptr to L */
  fp16_t*                U;    /* ptr to U */
  int                    lda;  /* local leading dim of array A */
  int                    ldl;  /* local leading dim of array L */
  int                    ldu;  /* local leading dim of array L */
  int                    nb;   /* distribution blocking factor */
  int                    n;    /* global # of cols of trailing part of A */
  int                    ia;   /* global row index of trailing part of A */
  int                    ja;   /* global col index of trailing part of A */
  int                    mp;   /* local # of rows of trailing part of A */
  int                    nq;   /* local # of cols of trailing part of A */
  int                    ii;   /* local row index of trailing part of A */
  int                    jj;   /* local col index of trailing part of A */
  int                    prow; /* proc. row owning 1st row of trail. A */
  int                    pcol; /* proc. col owning 1st col of trail. A */
  size_t                 max_lwork_size; /* largest size of L allocation */
  size_t                 max_uwork_size; /* largest size of U allocation */
} HPLMXP_T_panel;

/*
 * ---------------------------------------------------------------------
 * panel function prototypes
 * ---------------------------------------------------------------------
 */
void HPLMXP_pdpanel_new(HPLMXP_T_grid&         grid,
                        HPLMXP_T_pmat<fp32_t>& A,
                        const int              N,
                        const int              NB,
                        const int              IA,
                        const int              JA,
                        const int              II,
                        const int              JJ,
                        HPLMXP_T_panel&        P);

void HPLMXP_pdpanel_init(HPLMXP_T_grid&         grid,
                         HPLMXP_T_pmat<fp32_t>& A,
                         const int              N,
                         const int              NB,
                         const int              IA,
                         const int              JA,
                         const int              II,
                         const int              JJ,
                         HPLMXP_T_panel&        P);

int HPLMXP_pdpanel_free(HPLMXP_T_panel& P);

#endif
/*
 * End of hpl_panel.hpp
 */
