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
#ifndef HPLMXP_PAUXIL_HPP
#define HPLMXP_PAUXIL_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hplmxp_misc.hpp"
#include "hplmxp_blas.hpp"
#include "hplmxp_auxil.hpp"

#include "hplmxp_pmisc.hpp"
#include "hplmxp_grid.hpp"

void HPLMXP_pabort(int, const char*, const char*, ...);
void HPLMXP_pwarn(FILE*, int, const char*, const char*, ...);

void HPLMXP_pdlaprnt(const HPLMXP_T_grid*,
                     const int,
                     const int,
                     const int,
                     double*,
                     const int,
                     const int,
                     const int,
                     const char*);

template <typename T>
T HPLMXP_plamch(MPI_Comm COMM, const HPLMXP_T_MACH CMACH);

template <typename T>
void HPLMXP_pcopy(HPLMXP_T_grid& grid,
                  const int      N,
                  const int      NB,
                  const T*       x,
                  T*             y);

template <typename T>
void HPLMXP_paxpy(HPLMXP_T_grid& grid,
                  const int      N,
                  const int      NB,
                  const T        alpha,
                  const T*       x,
                  T*             y);

template <typename T>
void HPLMXP_paydx(HPLMXP_T_grid& grid,
                  const int      N,
                  const int      NB,
                  const T        alpha,
                  const T*       x,
                  T*             y);

template <typename T>
void HPLMXP_ptranspose(HPLMXP_T_grid& grid,
                       const int      N,
                       const int      NB,
                       const T*       x,
                       T*             y);

template <typename T>
T HPLMXP_plange(const HPLMXP_T_grid&, const int, const int, const T*);

#endif
/*
 * End of hpl_pauxil.hpp
 */
