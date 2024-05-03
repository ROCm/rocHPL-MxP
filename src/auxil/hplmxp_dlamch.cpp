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
/*
 * ---------------------------------------------------------------------
 * Static function prototypes
 * ---------------------------------------------------------------------
 */
template <typename T>
static void HPLMXP_lamc1(int*, int*, int*, int*);

template <typename T>
static void HPLMXP_lamc2(int*, int*, int*, T*, int*, T*, int*, T*);

template <typename T>
static T HPLMXP_lamc3(const T, const T);

template <typename T>
static void HPLMXP_lamc4(int*, const T, const int);

template <typename T>
static void HPLMXP_lamc5(const int, const int, const int, const int, int*, T*);

template <typename T>
static double HPLMXP_ipow(const T, const int);

template <typename T>
T HPLMXP_lamch(const HPLMXP_T_MACH CMACH) {
  /*
   * Purpose
   * =======
   *
   * HPLMXP_lamch determines  machine-specific  arithmetic constants such as
   * the relative machine precision  (eps),  the safe minimum (sfmin) such
   * that 1 / sfmin does not overflow, the base of the machine (base), the
   * precision (prec), the  number of (base) digits  in the  mantissa (t),
   * whether rounding occurs in addition (rnd=1.0 and 0.0 otherwise),  the
   * minimum exponent before  (gradual)  underflow (emin),  the  underflow
   * threshold (rmin) base**(emin-1), the largest exponent before overflow
   * (emax), the overflow threshold (rmax) (base**emax)*(1-eps).
   *
   * Notes
   * =====
   *
   * This function has been manually translated from the Fortran 77 LAPACK
   * auxiliary function dlamch.f  (version 2.0 -- 1992), that  was  itself
   * based on the function ENVRON  by Malcolm and incorporated suggestions
   * by Gentleman and Marovich. See
   *
   * Malcolm M. A.,  Algorithms  to  reveal  properties  of floating-point
   * arithmetic.,  Comms. of the ACM, 15, 949-951 (1972).
   *
   * Gentleman W. M. and Marovich S. B.,  More  on algorithms  that reveal
   * properties of  floating point arithmetic units.,  Comms. of  the ACM,
   * 17, 276-277 (1974).
   *
   * Arguments
   * =========
   *
   * CMACH   (local input)                 const HPLMXP_T_MACH
   *         Specifies the value to be returned by HPLMXP_lamch
   *            = HPLMXP_MACH_EPS,   HPLMXP_lamch := eps (default)
   *            = HPLMXP_MACH_SFMIN, HPLMXP_lamch := sfmin
   *            = HPLMXP_MACH_BASE,  HPLMXP_lamch := base
   *            = HPLMXP_MACH_PREC,  HPLMXP_lamch := eps*base
   *            = HPLMXP_MACH_MLEN,  HPLMXP_lamch := t
   *            = HPLMXP_MACH_RND,   HPLMXP_lamch := rnd
   *            = HPLMXP_MACH_EMIN,  HPLMXP_lamch := emin
   *            = HPLMXP_MACH_RMIN,  HPLMXP_lamch := rmin
   *            = HPLMXP_MACH_EMAX,  HPLMXP_lamch := emax
   *            = HPLMXP_MACH_RMAX,  HPLMXP_lamch := rmax
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

  static T   eps, sfmin, base, t, rnd, emin, rmin, emax, rmax, prec;
  T          small;
  static int first = 1;
  int        beta = 0, imax = 0, imin = 0, it = 0, lrnd = 0;

  if(first != 0) {
    first = 0;
    HPLMXP_lamc2(&beta, &it, &lrnd, &eps, &imin, &rmin, &imax, &rmax);
    base = (T)(beta);
    t    = (T)(it);
    if(lrnd != 0) {
      rnd = T{HPLMXP_rone};
      eps = HPLMXP_ipow(base, 1 - it) / T{HPLMXP_rtwo};
    } else {
      rnd = T{HPLMXP_rzero};
      eps = HPLMXP_ipow(base, 1 - it);
    }
    prec  = eps * base;
    emin  = (T)(imin);
    emax  = (T)(imax);
    sfmin = rmin;
    small = T{HPLMXP_rone} / rmax;
    /*
     * Use  SMALL  plus a bit,  to avoid the possibility of rounding causing
     * overflow when computing  1/sfmin.
     */
    if(small >= sfmin) sfmin = small * (T{HPLMXP_rone} + eps);
  }

  if(CMACH == HPLMXP_MACH_EPS) return (eps);
  if(CMACH == HPLMXP_MACH_SFMIN) return (sfmin);
  if(CMACH == HPLMXP_MACH_BASE) return (base);
  if(CMACH == HPLMXP_MACH_PREC) return (prec);
  if(CMACH == HPLMXP_MACH_MLEN) return (t);
  if(CMACH == HPLMXP_MACH_RND) return (rnd);
  if(CMACH == HPLMXP_MACH_EMIN) return (emin);
  if(CMACH == HPLMXP_MACH_RMIN) return (rmin);
  if(CMACH == HPLMXP_MACH_EMAX) return (emax);
  if(CMACH == HPLMXP_MACH_RMAX) return (rmax);

  return (eps);
}

template <typename T>
static void HPLMXP_lamc1(int* BETA, int* TT, int* RND, int* IEEE1) {
  /*
   * Purpose
   * =======
   *
   * HPLMXP_lamc1  determines  the machine parameters given by BETA, TT, RND,
   * and IEEE1.
   *
   * Notes
   * =====
   *
   * This function has been manually translated from the Fortran 77 LAPACK
   * auxiliary function dlamc1.f  (version 2.0 -- 1992), that  was  itself
   * based on the function ENVRON  by Malcolm and incorporated suggestions
   * by Gentleman and Marovich. See
   *
   * Malcolm M. A.,  Algorithms  to  reveal  properties  of floating-point
   * arithmetic.,  Comms. of the ACM, 15, 949-951 (1972).
   *
   * Gentleman W. M. and Marovich S. B.,  More  on algorithms  that reveal
   * properties of  floating point arithmetic units.,  Comms. of  the ACM,
   * 17, 276-277 (1974).
   *
   * Arguments
   * =========
   *
   * BETA    (local output)              int *
   *         The base of the machine.
   *
   * TT      (local output)              int *
   *         The number of ( BETA ) digits in the mantissa.
   *
   * RND     (local output)              int *
   *         Specifies whether proper rounding (RND=1) or chopping (RND=0)
   *         occurs in addition.  This may not be a  reliable guide to the
   *         way in which the machine performs its arithmetic.
   *
   * IEEE1   (local output)              int *
   *         Specifies  whether  rounding  appears  to be done in the IEEE
   *         `round to nearest' style (IEEE1=1), (IEEE1=0) otherwise.
   *
   * ---------------------------------------------------------------------
   */

  T          a, b, c, f, one, qtr, savec, t1, t2;
  static int first = 1, lbeta, lieee1, lrnd, lt;

  if(first != 0) {
    first = 0;
    one   = T{HPLMXP_rone};
    /*
     * lbeta, lieee1, lt and lrnd are the local values of BETA, IEEE1, TT and
     * RND. Throughout this routine we use the function HPLMXP_lamc3 to ensure
     * that relevant values are stored and not held in registers, or are not
     * affected by optimizers.
     *
     * Compute  a = 2.0**m  with the  smallest  positive integer m such that
     * fl( a + 1.0 ) == a.
     */
    a = T{HPLMXP_rone};
    c = T{HPLMXP_rone};
    do {
      a *= T{HPLMXP_rtwo};
      c = HPLMXP_lamc3(a, one);
      c = HPLMXP_lamc3(c, -a);
    } while(c == T{HPLMXP_rone});
    /*
     * Now compute b = 2.0**m with the smallest positive integer m such that
     * fl( a + b ) > a.
     */
    b = T{HPLMXP_rone};
    c = HPLMXP_lamc3(a, b);
    while(c == a) {
      b *= T{HPLMXP_rtwo};
      c = HPLMXP_lamc3(a, b);
    }
    /*
     * Now compute the base.  a and c  are  neighbouring floating point num-
     * bers in the interval ( BETA**TT, BETA**( TT + 1 ) ) and so their diffe-
     * rence is BETA.  Adding 0.25 to c is to ensure that it is truncated to
     * BETA and not (BETA-1).
     */
    qtr   = one / 4.0;
    savec = c;
    c     = HPLMXP_lamc3(c, -a);
    lbeta = (int)(c + qtr);
    /*
     * Now  determine  whether  rounding or chopping occurs, by adding a bit
     * less than BETA/2 and a bit more than BETA/2 to a.
     */
    b = (T)(lbeta);
    f = HPLMXP_lamc3(b / T{HPLMXP_rtwo}, -b / T{100.0});
    c = HPLMXP_lamc3(f, a);
    if(c == a) {
      lrnd = 1;
    } else {
      lrnd = 0;
    }
    f = HPLMXP_lamc3(b / T{HPLMXP_rtwo}, b / T{100.0});
    c = HPLMXP_lamc3(f, a);
    if((lrnd != 0) && (c == a)) lrnd = 0;
    /*
     * Try  and decide whether rounding is done in the  IEEE  round to nea-
     * rest style.  b/2 is half a unit in the last place of the two numbers
     * a  and savec. Furthermore, a is even, i.e. has last bit zero, and sa-
     * vec is odd.  Thus adding b/2 to a should not change a, but adding b/2
     * to savec should change savec.
     */
    t1 = HPLMXP_lamc3(b / T{HPLMXP_rtwo}, a);
    t2 = HPLMXP_lamc3(b / T{HPLMXP_rtwo}, savec);
    if((t1 == a) && (t2 > savec) && (lrnd != 0))
      lieee1 = 1;
    else
      lieee1 = 0;
    /*
     * Now find the mantissa, TT. It should be the integer part of log to the
     * base BETA of a, however it is safer to determine TT by powering. So we
     * find TT as the smallest positive integer for which fl( beta**t + 1.0 )
     * is equal to 1.0.
     */
    lt = 0;
    a  = T{HPLMXP_rone};
    c  = T{HPLMXP_rone};

    do {
      lt++;
      a *= (T)(lbeta);
      c = HPLMXP_lamc3(a, one);
      c = HPLMXP_lamc3(c, -a);
    } while(c == T{HPLMXP_rone});
  }

  *BETA  = lbeta;
  *TT    = lt;
  *RND   = lrnd;
  *IEEE1 = lieee1;
}

template <typename T>
static void HPLMXP_lamc2(int* BETA,
                         int* TT,
                         int* RND,
                         T*   EPS,
                         int* EMIN,
                         T*   RMIN,
                         int* EMAX,
                         T*   RMAX) {
  /*
   * Purpose
   * =======
   *
   * HPLMXP_lamc2  determines the machine  parameters specified in its argu-
   * ment list.
   *
   * Notes
   * =====
   *
   * This function has been manually translated from the Fortran 77 LAPACK
   * auxiliary function  dlamc2.f (version 2.0 -- 1992), that  was  itself
   * based on a function PARANOIA  by  W. Kahan of the University of Cali-
   * fornia at Berkeley for the computation of the  relative machine epsi-
   * lon eps.
   *
   * Arguments
   * =========
   *
   * BETA    (local output)              int *
   *         The base of the machine.
   *
   * TT      (local output)              int *
   *         The number of ( BETA ) digits in the mantissa.
   *
   * RND     (local output)              int *
   *         Specifies whether proper rounding (RND=1) or chopping (RND=0)
   *         occurs in addition. This may not be a reliable  guide to  the
   *         way in which the machine performs its arithmetic.
   *
   * EPS     (local output)              T *
   *         The smallest positive number such that fl( 1.0 - EPS ) < 1.0,
   *         where fl denotes the computed value.
   *
   * EMIN    (local output)              int *
   *         The minimum exponent before (gradual) underflow occurs.
   *
   * RMIN    (local output)              T *
   *         The smallest  normalized  number  for  the  machine, given by
   *         BASE**( EMIN - 1 ), where  BASE  is the floating  point value
   *         of BETA.
   *
   * EMAX    (local output)              int *
   *         The maximum exponent before overflow occurs.
   *
   * RMAX    (local output)              T *
   *         The  largest  positive  number  for  the  machine,  given  by
   *         BASE**EMAX * ( 1 - EPS ), where  BASE  is the floating  point
   *         value of BETA.
   *
   * ---------------------------------------------------------------------
   */

  static T   leps, lrmax, lrmin;
  T          a, b, c, half, one, rbase, sixth, small, third, two, zero;
  static int first = 1, iwarn = 0, lbeta = 0, lemax, lemin, lt = 0;
  int        gnmin = 0, gpmin = 0, i, ieee, lieee1 = 0, lrnd = 0, ngnmin = 0,
      ngpmin = 0;

  if(first != 0) {
    first = 0;
    zero  = T{HPLMXP_rzero};
    one   = T{HPLMXP_rone};
    two   = T{HPLMXP_rtwo};
    /*
     * lbeta, lt, lrnd, leps, lemin and lrmin are the local values of  BETA,
     * TT, RND, EPS, EMIN and RMIN.
     *
     * Throughout this routine we use the function HPLMXP_lamc3 to ensure that
     * relevant values are stored and not held in registers,  or are not af-
     * fected by optimizers.
     *
     * HPLMXP_lamc1 returns the parameters  lbeta, lt, lrnd and lieee1.
     */
    HPLMXP_lamc1<T>(&lbeta, &lt, &lrnd, &lieee1);
    /*
     * Start to find eps.
     */
    b    = (T)(lbeta);
    a    = HPLMXP_ipow(b, -lt);
    leps = a;
    /*
     * Try some tricks to see whether or not this is the correct  EPS.
     */
    b     = two / 3.0;
    half  = one / T{HPLMXP_rtwo};
    sixth = HPLMXP_lamc3(b, -half);
    third = HPLMXP_lamc3(sixth, sixth);
    b     = HPLMXP_lamc3(third, -half);
    b     = HPLMXP_lamc3(b, sixth);
    b     = Mabs(b);
    if(b < leps) b = leps;

    leps = T{HPLMXP_rone};

    while((leps > b) && (b > zero)) {
      leps = b;
      c    = HPLMXP_lamc3(
          half * leps,
          static_cast<T>(HPLMXP_ipow(two, 5) * HPLMXP_ipow(leps, 2)));
      c = HPLMXP_lamc3(half, -c);
      b = HPLMXP_lamc3(half, c);
      c = HPLMXP_lamc3(half, -b);
      b = HPLMXP_lamc3(half, c);
    }
    if(a < leps) leps = a;
    /*
     * Computation of EPS complete.
     *
     * Now find  EMIN.  Let a = + or - 1, and + or - (1 + BASE**(-3)).  Keep
     * dividing a by BETA until (gradual) underflow occurs. This is detected
     * when we cannot recover the previous a.
     */
    rbase = one / (T)(lbeta);
    small = one;
    for(i = 0; i < 3; i++) small = HPLMXP_lamc3(small * rbase, zero);
    a = HPLMXP_lamc3(one, small);
    HPLMXP_lamc4(&ngpmin, one, lbeta);
    HPLMXP_lamc4(&ngnmin, -one, lbeta);
    HPLMXP_lamc4(&gpmin, a, lbeta);
    HPLMXP_lamc4(&gnmin, -a, lbeta);

    ieee = 0;

    if((ngpmin == ngnmin) && (gpmin == gnmin)) {
      if(ngpmin == gpmin) {
        /*
         * Non twos-complement machines, no gradual underflow; e.g.,  VAX )
         */
        lemin = ngpmin;
      } else if((gpmin - ngpmin) == 3) {
        /*
         * Non twos-complement machines with gradual underflow; e.g., IEEE stan-
         * dard followers
         */
        lemin = ngpmin - 1 + lt;
        ieee  = 1;
      } else {
        /*
         * A guess; no known machine
         */
        lemin = Mmin(ngpmin, gpmin);
        iwarn = 1;
      }
    } else if((ngpmin == gpmin) && (ngnmin == gnmin)) {
      if(Mabs(ngpmin - ngnmin) == 1) {
        /*
         * Twos-complement machines, no gradual underflow; e.g., CYBER 205
         */
        lemin = Mmax(ngpmin, ngnmin);
      } else {
        /*
         * A guess; no known machine
         */
        lemin = Mmin(ngpmin, ngnmin);
        iwarn = 1;
      }
    } else if((Mabs(ngpmin - ngnmin) == 1) && (gpmin == gnmin)) {
      if((gpmin - Mmin(ngpmin, ngnmin)) == 3) {
        /*
         * Twos-complement machines with gradual underflow; no known machine
         */
        lemin = Mmax(ngpmin, ngnmin) - 1 + lt;
      } else {
        /*
         * A guess; no known machine
         */
        lemin = Mmin(ngpmin, ngnmin);
        iwarn = 1;
      }
    } else {
      /*
       * A guess; no known machine
       */
      lemin = Mmin(ngpmin, ngnmin);
      lemin = Mmin(lemin, gpmin);
      lemin = Mmin(lemin, gnmin);
      iwarn = 1;
    }
    /*
     * Comment out this if block if EMIN is ok
     */
    if(iwarn != 0) {
      first = 1;
      HPLMXP_fprintf(
          stderr,
          "\n %s %8d\n%s\n%s\n%s\n",
          "WARNING. The value EMIN may be incorrect:- EMIN =",
          lemin,
          "If, after inspection, the value EMIN looks acceptable, "
          "please comment ",
          "out the  if  block  as marked within the code of routine  "
          "HPLMXP_lamc2, ",
          "otherwise supply EMIN explicitly.");
    }
    /*
     * Assume IEEE arithmetic if we found denormalised  numbers above, or if
     * arithmetic seems to round in the  IEEE style,  determined  in routine
     * HPLMXP_lamc1.  A true  IEEE  machine should have both things true; how-
     * ever, faulty machines may have one or the other.
     */
    if((ieee != 0) || (lieee1 != 0))
      ieee = 1;
    else
      ieee = 0;
    /*
     * Compute  RMIN by successive division by  BETA. We could compute  RMIN
     * as BASE**( EMIN - 1 ), but some machines underflow during this compu-
     * tation.
     */
    lrmin = T{HPLMXP_rone};
    for(i = 0; i < 1 - lemin; i++) lrmin = HPLMXP_lamc3(lrmin * rbase, zero);
    /*
     * Finally, call HPLMXP_lamc5 to compute emax and rmax.
     */
    HPLMXP_lamc5(lbeta, lt, lemin, ieee, &lemax, &lrmax);
  }
  *BETA = lbeta;
  *TT   = lt;
  *RND  = lrnd;
  *EPS  = leps;
  *EMIN = lemin;
  *RMIN = lrmin;
  *EMAX = lemax;
  *RMAX = lrmax;
}

template <typename T>
static T HPLMXP_lamc3(const T A, const T B) {
  /*
   * Purpose
   * =======
   *
   * HPLMXP_lamc3  is intended to force a and b  to be stored prior to doing
   * the addition of  a  and  b,  for  use  in situations where optimizers
   * might hold one of these in a register.
   *
   * Notes
   * =====
   *
   * This function has been manually translated from the Fortran 77 LAPACK
   * auxiliary function dlamc3.f (version 2.0 -- 1992).
   *
   * Arguments
   * =========
   *
   * A, B    (local input)               T
   *         The values a and b.
   *
   * ---------------------------------------------------------------------
   */

  return (A + B);
}

template <typename T>
static void HPLMXP_lamc4(int* EMIN, const T START, const int BASE) {
  /*
   * Purpose
   * =======
   *
   * HPLMXP_lamc4 is a service function for HPLMXP_lamc2.
   *
   * Notes
   * =====
   *
   * This function has been manually translated from the Fortran 77 LAPACK
   * auxiliary function dlamc4.f (version 2.0 -- 1992).
   *
   * Arguments
   * =========
   *
   * EMIN    (local output)              int *
   *         The minimum exponent before  (gradual) underflow, computed by
   *         setting A = START and dividing  by  BASE until the previous A
   *         can not be recovered.
   *
   * START   (local input)               T
   *         The starting point for determining EMIN.
   *
   * BASE    (local input)               int
   *         The base of the machine.
   *
   * ---------------------------------------------------------------------
   */

  T   a, b1, b2, c1, c2, d1, d2, one, rbase, zero;
  int i;

  a     = START;
  one   = T{HPLMXP_rone};
  rbase = one / (T)(BASE);
  zero  = T{HPLMXP_rzero};
  *EMIN = 1;
  b1    = HPLMXP_lamc3(a * rbase, zero);
  c1 = c2 = d1 = d2 = a;

  do {
    (*EMIN)--;
    a  = b1;
    b1 = HPLMXP_lamc3(a / BASE, zero);
    c1 = HPLMXP_lamc3(b1 * BASE, zero);
    d1 = zero;
    for(i = 0; i < BASE; i++) d1 = d1 + b1;
    b2 = HPLMXP_lamc3(a * rbase, zero);
    c2 = HPLMXP_lamc3(b2 / rbase, zero);
    d2 = zero;
    for(i = 0; i < BASE; i++) d2 = d2 + b2;
  } while((c1 == a) && (c2 == a) && (d1 == a) && (d2 == a));
}

template <typename T>
static void HPLMXP_lamc5(const int BETA,
                         const int P,
                         const int EMIN,
                         const int IEEE,
                         int*      EMAX,
                         T*        RMAX) {
  /*
   * Purpose
   * =======
   *
   * HPLMXP_lamc5  attempts  to compute RMAX, the largest machine  floating-
   * point number, without overflow.  It assumes that EMAX + abs(EMIN) sum
   * approximately to a power of 2.  It will fail  on machines where  this
   * assumption does not hold, for example, the  Cyber 205 (EMIN = -28625,
   * EMAX = 28718).  It will also fail if  the value supplied for  EMIN is
   * too large (i.e. too close to zero), probably with overflow.
   *
   * Notes
   * =====
   *
   * This function has been manually translated from the Fortran 77 LAPACK
   * auxiliary function dlamc5.f (version 2.0 -- 1992).
   *
   * Arguments
   * =========
   *
   * BETA    (local input)               int
   *         The base of floating-point arithmetic.
   *
   * P       (local input)               int
   *         The number of base BETA digits in the mantissa of a floating-
   *         point value.
   *
   * EMIN    (local input)               int
   *         The minimum exponent before (gradual) underflow.
   *
   * IEEE    (local input)               int
   *         A logical flag specifying whether or not  the arithmetic sys-
   *         tem is thought to comply with the IEEE standard.
   *
   * EMAX    (local output)              int *
   *         The largest exponent before overflow.
   *
   * RMAX    (local output)              T *
   *         The largest machine floating-point number.
   *
   * ---------------------------------------------------------------------
   */

  T   oldy   = T{HPLMXP_rzero}, recbas, y, z;
  int exbits = 1, expsum, i, lexp = 1, nbits, ttry, uexp;
/* ..
 * .. Executable Statements ..
 */
/*
 * First compute  lexp  and  uexp, two powers of 2 that bound abs(EMIN).
 * We then assume that  EMAX + abs( EMIN ) will sum approximately to the
 * bound that  is closest to abs( EMIN ). (EMAX  is the  exponent of the
 * required number RMAX).
 */
l_10:
  ttry = (int)((unsigned int)(lexp) << 1);
  if(ttry <= (-EMIN)) {
    lexp = ttry;
    exbits++;
    goto l_10;
  }

  if(lexp == -EMIN) {
    uexp = lexp;
  } else {
    uexp = ttry;
    exbits++;
  }
  /*
   * Now -lexp is less than or equal to EMIN, and -uexp is greater than or
   * equal to EMIN. exbits is the number of bits needed to store the expo-
   * nent.
   */
  if((uexp + EMIN) > (-lexp - EMIN)) {
    expsum = (int)((unsigned int)(lexp) << 1);
  } else {
    expsum = (int)((unsigned int)(uexp) << 1);
  }
  /*
   * expsum is the exponent range, approximately equal to EMAX - EMIN + 1.
   */
  *EMAX = expsum + EMIN - 1;
  /*
   * nbits  is  the total number of bits needed to store a  floating-point
   * number.
   */
  nbits = 1 + exbits + P;

  if((nbits % 2 == 1) && (BETA == 2)) {
    /*
     * Either there are an odd number of bits used to store a floating-point
     * number, which is unlikely, or some bits are not used in the represen-
     * tation of numbers,  which is possible,  (e.g. Cray machines)  or  the
     * mantissa has an implicit bit, (e.g. IEEE machines, Dec Vax machines),
     * which is perhaps the most likely. We have to assume the last alterna-
     * tive.  If this is true,  then we need to reduce  EMAX  by one because
     * there must be some way of representing zero  in an  implicit-bit sys-
     * tem. On machines like Cray we are reducing EMAX by one unnecessarily.
     */
    (*EMAX)--;
  }

  if(IEEE != 0) {
    /*
     * Assume we are on an IEEE  machine which reserves one exponent for in-
     * finity and NaN.
     */
    (*EMAX)--;
  }
  /*
   * Now create RMAX, the largest machine number, which should be equal to
   * (1.0 - BETA**(-P)) * BETA**EMAX . First compute 1.0-BETA**(-P), being
   * careful that the result is less than 1.0.
   */
  recbas = T{HPLMXP_rone} / (T)(BETA);
  z      = (T)(BETA)-T{HPLMXP_rone};
  y      = T{HPLMXP_rzero};

  for(i = 0; i < P; i++) {
    z *= recbas;
    if(y < T{HPLMXP_rone}) oldy = y;
    y = HPLMXP_lamc3(y, z);
  }

  if(y >= T{HPLMXP_rone}) y = oldy;
  /*
   * Now multiply by BETA**EMAX to get RMAX.
   */
  for(i = 0; i < *EMAX; i++) y = HPLMXP_lamc3(y * BETA, T{HPLMXP_rzero});

  *RMAX = y;
}

template <typename T>
static double HPLMXP_ipow(const T X, const int N) {
  /*
   * Purpose
   * =======
   *
   * HPLMXP_ipow computes the integer n-th power of a real scalar x.
   *
   * Arguments
   * =========
   *
   * X       (local input)               const double
   *         The real scalar x.
   *
   * N       (local input)               const int
   *         The integer power to raise x to.
   *
   * ---------------------------------------------------------------------
   */

  double r, y = HPLMXP_rone;
  int    k, n;

  if(X == T{HPLMXP_rzero}) return (HPLMXP_rzero);
  if(N < 0) {
    n = -N;
    r = HPLMXP_rone / X;
  } else {
    n = N;
    r = X;
  }
  for(k = 0; k < n; k++) y *= r;

  return (y);
}

template float HPLMXP_lamch(const HPLMXP_T_MACH CMACH);

template double HPLMXP_lamch(const HPLMXP_T_MACH CMACH);
