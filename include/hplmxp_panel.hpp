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

/*
 * ---------------------------------------------------------------------
 * Data Structures
 * ---------------------------------------------------------------------
 */
template<typename A_t, typename C_t>
struct HPLMXP_T_panel {
  HPLMXP_T_grid*    grid; /* ptr to the process grid */
  A_t*              A;    /* ptr to trailing part of A */
  C_t*              L;    /* ptr to L */
  C_t*              U;    /* ptr to U */
  int               lda;  /* local leading dim of array A */
  int               ldl;  /* local leading dim of array L */
  int               ldu;  /* local leading dim of array L */
  int               nb;   /* distribution blocking factor */
  int               n;    /* global # of cols of trailing part of A */
  int               ia;   /* global row index of trailing part of A */
  int               ja;   /* global col index of trailing part of A */
  int               mp;   /* local # of rows of trailing part of A */
  int               nq;   /* local # of cols of trailing part of A */
  int               ii;   /* local row index of trailing part of A */
  int               jj;   /* local col index of trailing part of A */
  int               prow; /* proc. row owning 1st row of trail. A */
  int               pcol; /* proc. col owning 1st col of trail. A */
};

template<typename A_t, typename C_t>
struct HPLMXP_T_pmat;

/*
 * ---------------------------------------------------------------------
 * panel function prototypes
 * ---------------------------------------------------------------------
 */
template <typename A_t, typename C_t>
int HPLMXP_pdpanel_new(HPLMXP_T_grid&                 grid,
                       HPLMXP_T_pmat<A_t, C_t>&  A,
                       HPLMXP_T_panel<A_t, C_t>&      P);

template <typename A_t, typename C_t>
void HPLMXP_pdpanel_init(HPLMXP_T_grid&                  grid,
                         HPLMXP_T_pmat<A_t, C_t>&   A,
                         const int                       N,
                         const int                       NB,
                         const int                       IA,
                         const int                       JA,
                         const int                       II,
                         const int                       JJ,
                         HPLMXP_T_panel<A_t, C_t>&       P);

template<typename A_t, typename C_t>
void HPLMXP_pdpanel_free(HPLMXP_T_panel<A_t, C_t>& P);

#endif
/*
 * End of hpl_panel.hpp
 */
