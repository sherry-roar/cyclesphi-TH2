/*
 * Copyright 2011-2021 Blender Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

//#include "kernel_camera.h"
//#include "kernel_cuda.h"
//#include "kernel_util.h"
#include "client_api.h"

//#ifdef WITH_CLIENT_CUDA

#include <cuda.h>
#include <cuda_runtime_api.h>

//#  define __KERNEL_GPU__
//#  define __KERNEL_CUDA__

#include <algorithm>
#include <math.h>
#include <omp.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <time.h>
#include <vector>

//#  define KERNEL_ARCH cuda
//#  include "kernel_cpu_impl.h"
//
//#  include "kernel/filter/filter.h"
//#  define KERNEL_ARCH cuda
//#  include "filter_cpu_impl.h"
//
//#  include "util/util_boundbox.h"

#if defined(WITH_CLIENT_CUDA_CPU_STAT)
#  include "kernel_omp.h"
#endif  // WITH_CLIENT_CUDA_CPU_STAT

#if 0  // def WITH_OPTIX_DENOISER
#  include "kernel_denoiser.h"
//#    include "util_math.h"
#endif

#define CU_DEVICE_PTR void *

//#  ifdef WITH_CUDA_STAT
//#    include <thrust/sort.h>
//#    define CUDA_CHUNK_SIZE (64 * 1024)
//#    define CUDA_CHUNK_SIZE (1)

///////////////////////////////

// moana - 16GB, 2MBchunk: RR=826
// moana - 16GB, size/GPUs: C=1387
//#  define UNITEST1_ROUND_ROBIN
//#define UNITEST1_RANDOM
//#define UNITEST1_COUNTINOUS

//#  define UNITEST1_CPU

//#  define UNITEST2_ROUND_ROBIN
//#define UNITEST2_RANDOM
//#  define UNITEST2_CREDITS
//#define UNITEST2_CREDITS_CPU

//# define UNITEST3_PENALTY
//# define UNITEST3_PENALTYv2

////////////////////////////
//
namespace cyclesphi {
namespace kernel {

#if 0 
const char *cuewErrorString(CUresult result);

//void check_exit();
//
//#  define cu_assert(stmt) \
//    { \
//      CUresult result = stmt; \
//      if (result != CUDA_SUCCESS) { \
//        char err[1024]; \
//        sprintf(err, "CUDA error: %s in %s, %s, line %d", cuewErrorString(result), __FILE__, #stmt, __LINE__); \
//        std::string message(err); \
//        fprintf(stderr, "%s\n", message.c_str()); \
//        check_exit(); \
//      } \
//    } \
//    (void)0

//#  define cuda_assert(stmt) \
//    { \
//      if (stmt != cudaSuccess) { \
//        char err[1024]; \
//        sprintf(err, \
//                "CUDA error: %s: %s in %s, %s, line %d", \
//                cudaGetErrorName(stmt), \
//                cudaGetErrorString(stmt), \
//                __FILE__, \
//                #stmt, \
//                __LINE__); \
//        std::string message(err); \
//        fprintf(stderr, "%s\n", message.c_str()); \
//        check_exit(); \
//      } \
//    } \
//    (void)0

#endif

bool cuda_error_(cudaError_t result, const std::string &stmt);

#define cuda_error(stmt) cuda_error_(stmt, #stmt)

void cuda_error_message(const std::string &message);

/////////////////////////////////////
size_t cuda_align_up(size_t offset, size_t alignment);

bool cuda_check_writable_mem(const char *_name, int dev);

bool cuda_check_unimem(const char *_name, int dev);

/////////////////////////////////////

////////////////////////WITH_CUDA_STATv2/////////////
extern int cuda_set_unimem_flag_round_robin_dev;
void cuda_set_unimem_flag(DEVICE_PTR map_id);

DEVICE_PTR generic_alloc(const char *name,
                         size_t memory_size,
                         size_t pitch_padding,
                         bool unimem_flag,
                         bool alloc_host = false);

void generic_copy_to(char *host_pointer, DEVICE_PTR map_id, size_t memory_size);

void generic_copy_to_async(char *host_pointer, DEVICE_PTR map_id, size_t memory_size);

void generic_free(DEVICE_PTR map_id, size_t memory_size);

}  // namespace kernel
}  // namespace cyclesphi
