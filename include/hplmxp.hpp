
#ifndef HPLMXP_HPP
#define HPLMXP_HPP

#include <hip/hip_runtime_api.h>
#include <hip/hip_fp16.h>

#ifdef HPLMXP_TRACING
#include <roctracer.h>
#include <roctx.h>
#endif

/*
 * ---------------------------------------------------------------------
 * #define types
 * ---------------------------------------------------------------------
 */
#define fp64_t double
#define fp32_t float
#define fp16_t __half

//#define HPLMXP_USE_COLLECTIVES 1

/*
Enabling atomics will potentially allow more performance optimization
but will potentailly lead to residual values which vary from run-to-run
*/
#undef HPLMXP_ROCBLAS_ALLOW_ATOMICS
// #define HPLMXP_ROCBLAS_ALLOW_ATOMICS

#include "hplmxp_version.hpp"
#include "hplmxp_misc.hpp"
#include "hplmxp_blas.hpp"
#include "hplmxp_auxil.hpp"
#include "hplmxp_comm.hpp"
#include "hplmxp_pauxil.hpp"
#include "hplmxp_grid.hpp"
#include "hplmxp_panel.hpp"
#include "hplmxp_pgesv.hpp"
#include "hplmxp_pmatgen.hpp"
#include "hplmxp_ptimer.hpp"
#include "hplmxp_ptest.hpp"

#endif
