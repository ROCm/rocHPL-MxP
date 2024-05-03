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
T HPLMXP_plamch(MPI_Comm COMM, const HPLMXP_T_MACH CMACH) {
  /*
   * Purpose
   * =======
   *
   * HPLMXP_plamch determines  machine-specific  arithmetic  constants  such  as
   * the relative machine precision (eps),  the safe minimum(sfmin) such that
   * 1/sfmin does not overflow, the base of the machine (base), the precision
   * (prec),  the  number  of  (base)  digits in the  mantissa  (t),  whether
   * rounding occurs in addition (rnd = 1.0 and 0.0 otherwise),  the  minimum
   * exponent before  (gradual)  underflow (emin),  the  underflow  threshold
   * (rmin)- base**(emin-1), the largest exponent before overflow (emax), the
   * overflow threshold (rmax)  - (base**emax)*(1-eps).
   *
   * Arguments
   * =========
   *
   * COMM    (global/local input)          MPI_Comm
   *         The MPI communicator identifying the process collection.
   *
   * CMACH   (global input)                const HPLMXP_T_MACH
   *         Specifies the value to be returned by HPLMXP_plamch
   *            = HPLMXP_MACH_EPS,   HPLMXP_plamch := eps (default)
   *            = HPLMXP_MACH_SFMIN, HPLMXP_plamch := sfmin
   *            = HPLMXP_MACH_BASE,  HPLMXP_plamch := base
   *            = HPLMXP_MACH_PREC,  HPLMXP_plamch := eps*base
   *            = HPLMXP_MACH_MLEN,  HPLMXP_plamch := t
   *            = HPLMXP_MACH_RND,   HPLMXP_plamch := rnd
   *            = HPLMXP_MACH_EMIN,  HPLMXP_plamch := emin
   *            = HPLMXP_MACH_RMIN,  HPLMXP_plamch := rmin
   *            = HPLMXP_MACH_EMAX,  HPLMXP_plamch := emax
   *            = HPLMXP_MACH_RMAX,  HPLMXP_plamch := rmax
   *
   *         where
   *
   *            eps   = relative machine precision,
   *            sfmin = safe minimum,
   *            base  = base of the machine,
   *            prec  = eps*base,
   *            t     = number of digits in the mantissa,
   *            rnd   = 1.0 if rounding occurs in addition,
   *            emin  = minimum exponent before underflow,
   *            rmin  = underflow threshold,
   *            emax  = largest exponent before overflow,
   *            rmax  = overflow threshold.
   *
   * ---------------------------------------------------------------------
   */

  T param;

  param = HPLMXP_lamch<T>(CMACH);

  switch(CMACH) {
    case HPLMXP_MACH_EPS:
    case HPLMXP_MACH_SFMIN:
    case HPLMXP_MACH_EMIN:
    case HPLMXP_MACH_RMIN:
      (void)HPLMXP_all_reduce(&param, 1, HPLMXP_MAX, COMM);
      break;
    case HPLMXP_MACH_EMAX:
    case HPLMXP_MACH_RMAX:
      (void)HPLMXP_all_reduce(&param, 1, HPLMXP_MIN, COMM);
      break;
    default: break;
  }

  return (param);
}

template float HPLMXP_plamch(MPI_Comm COMM, const HPLMXP_T_MACH CMACH);

template double HPLMXP_plamch(MPI_Comm COMM, const HPLMXP_T_MACH CMACH);
