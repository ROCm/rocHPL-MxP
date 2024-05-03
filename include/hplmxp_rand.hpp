#ifndef HPL_RAND_HPP
#define HPL_RAND_HPP

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#define LCG_A 6364136223846793005ULL
#define LCG_C 1ULL

#if 0 /* original */
#define LCG_MUL 5.4210108624275222e-20
#else /* scaling, allowed */
#define LCG_MUL 5.4210108624275222e-24
#endif

#define HPLMXP_SEED 42

struct RandCoeff {
  uint64_t a;
  uint64_t c;

  __host__ __device__ static RandCoeff default_vals() { return {LCG_A, LCG_C}; }

  __host__ __device__ RandCoeff operator*(const RandCoeff& rhs) const {
    return {a * rhs.a, a * rhs.c + c};
  }

  __host__ __device__ void operator*=(const RandCoeff& rhs) {
    c = a * rhs.c + c;
    a = a * rhs.a;
  }
};

__host__ __device__ inline RandCoeff pow(RandCoeff base, uint32_t n) {
  RandCoeff result{1, 0};
  while(n != 0) {
    if(n & 1) result *= base;
    n >>= 1;
    base *= base;
  }
  return result;
}

struct RandStat {
  uint64_t x;

  __host__ __device__ static RandStat initialize(
      uint64_t  seed,
      RandCoeff coef = RandCoeff::default_vals()) {
    return coef * RandStat{seed};
  }

  __host__ __device__ void operator*=(RandCoeff coef) {
    x = coef.a * x + coef.c;
  }

  __host__ __device__ double toDouble() const {
    return static_cast<int64_t>(x) * LCG_MUL;
  }

  __host__ __device__ friend RandStat operator*(RandCoeff coef, RandStat stat) {
    return {coef.a * stat.x + coef.c};
  }
};

#endif
