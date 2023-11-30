/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2022 Blender Foundation */

/* Constant Globals */

#pragma once

#include "kernel/types.h"

#include "kernel/integrator/state.h"

#include "kernel/util/profiling.h"

CCL_NAMESPACE_BEGIN

/* Not actually used, just a NULL pointer that gets passed everywhere, which we
 * hope gets optimized out by the compiler. */
struct KernelGlobalsGPU {
  int unused[1];
};
typedef ccl_global const KernelGlobalsGPU *ccl_restrict KernelGlobals;

struct KernelParamsHIP {
  /* Global scene data and textures */
  KernelData data;
#define KERNEL_DATA_ARRAY(type, name) const type *name;
#include "kernel/data_arrays.h"

  /* Integrator state */
  IntegratorStateGPU integrator_state;
};

#ifdef __KERNEL_GPU__
__constant__ KernelParamsHIP kernel_params;

#  ifdef WITH_CLIENT_SHOW_STAT
#    define KERNEL_DATA_ARRAY(type, name) __device__ int *name##_counter_gpu = NULL;
#    include "kernel/data_arrays.h"
#  endif

#  ifdef __KERNEL_HIP_STAT__
#    define KERNEL_DATA_ARRAY(type, name) __device__ unsigned long long int *name##_counter = NULL;
#    include "kernel/data_arrays.h"

#    define KERNEL_DATA_ARRAY(type, name) \
      const __constant__ __device__ double *name##_counter_mul = NULL;
#    include "kernel/data_arrays.h"

#    define KERNEL_DATA_ARRAY(type, name) \
      const __constant__ __device__ unsigned int *name##_counter_size = NULL;
#    include "kernel/data_arrays.h"

#    define KERNEL_DATA_ARRAY(type, name) \
      __device__ const type &name##_func(const type &data, \
                                         unsigned long long int *counter, \
                                         const unsigned long long int index, \
                                         const double *counter_mul, \
                                         const unsigned int *counter_size) \
      { \
        unsigned long long int id = (unsigned long long int)((double)index * counter_mul[0]); \
        atomicAdd(&counter[id], 1ULL); \
        return data; \
      }
#    include "kernel/data_arrays.h"
#  endif

#endif

/* Abstraction macros */
#define kernel_data kernel_params.data

#ifdef __KERNEL_HIP_STAT__
#  define kernel_data_fetch(t, index) \
    (t##_func(kernel_params.t[(index)], t##_counter, (index), t##_counter_mul, t##_counter_size))
#else
#  define kernel_data_fetch(name, index) kernel_params.name[(index)]
#endif

#define kernel_data_array(name) (kernel_params.name)
#define kernel_integrator_state kernel_params.integrator_state

/* Mapping table */
#ifdef __KERNEL_GPU__
__constant__ float linear_to_srgb_table[1001];
#endif

CCL_NAMESPACE_END
