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
#ifndef HPLMXP_PMATGEN_HPP
#define HPLMXP_PMATGEN_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hplmxp_pgesv.hpp"

/*
 * ---------------------------------------------------------------------
 * Function prototypes
 * ---------------------------------------------------------------------
 */
template <typename T>
void HPLMXP_pmat_init(HPLMXP_T_grid&    grid,
                      const int         N,
                      const int         NB,
                      HPLMXP_T_pmat<T>& A);

template <typename T>
void HPLMXP_pmatgen(HPLMXP_T_grid& grid, HPLMXP_T_pmat<T>& A);

template <typename T>
void HPLMXP_pmatgen_rhs(HPLMXP_T_grid& grid, HPLMXP_T_pmat<T>& A);

template <typename T>
void HPLMXP_pmatgen_x(HPLMXP_T_grid& grid, HPLMXP_T_pmat<T>& A);

template <typename T>
void HPLMXP_pmat_free(HPLMXP_T_pmat<T>& A);

#endif
/*
 * End of hpl_pmatgen.hpp
 */
