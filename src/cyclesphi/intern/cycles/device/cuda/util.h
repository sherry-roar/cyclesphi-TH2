/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2022 Blender Foundation */

#pragma once

#ifdef WITH_CUDA

#  ifdef WITH_CUDA_DYNLOAD
#    include "cuew.h"
#  else
#    include <cuda.h>
#  endif

CCL_NAMESPACE_BEGIN

class CUDADevice;
class CUDADevices;

#ifdef BLENDER_CLIENT

#    ifdef WITH_CLIENT_OPTIX
class OptiXDevice;
class OptiXDevices;
extern OptiXDevices cuda_devices;
#    else
extern CUDADevices cuda_devices;
#    endif

#endif

/* Utility to push/pop CUDA context. */
class CUDAContextScope {
 public:
  CUDAContextScope(CUDADevice *device);
  ~CUDAContextScope();

#ifdef BLENDER_CLIENT
  CUDAContextScope(int id);
  CUDADevice &get();
#endif

 private:
  CUDADevice *device;
};

/* Utility for checking return values of CUDA function calls. */
#  define cuda_device_assert(cuda_device, stmt) \
    { \
      CUresult result = stmt; \
      if (result != CUDA_SUCCESS) { \
        const char *name = cuewErrorString(result); \
        cuda_device->set_error( \
            string_printf("%s in %s (%s:%d)", name, #stmt, __FILE__, __LINE__)); \
      } \
    } \
    (void)0

#  define cuda_assert(stmt) cuda_device_assert(this, stmt)

#ifdef BLENDER_CLIENT

void cuda_util_check_exit();


#    define cu_assert(stmt) \
      { \
        CUresult result = stmt; \
        if (result != CUDA_SUCCESS) { \
          char err[1024]; \
          sprintf(err, \
                  "CUDA error: %d in %s, %s, line %d", \
                  result, \
                  __FILE__, \
                  #stmt, \
                  __LINE__); \
          std::string message(err); \
          fprintf(stderr, "%s\n", message.c_str()); \
          ccl::cuda_util_check_exit(); \
        } \
      } \
      (void)0

#    define cuda_assert2(stmt) \
          { \
      if (stmt != cudaSuccess) { \
        char err[1024]; \
        sprintf(err, \
                "CUDA error: %s: %s in %s, %s, line %d", \
                cudaGetErrorName(stmt), \
                cudaGetErrorString(stmt), \
                __FILE__, \
                #stmt, \
                __LINE__); \
        std::string message(err); \
        fprintf(stderr, "%s\n", message.c_str()); \
        ccl::cuda_util_check_exit(); \
      } \
    } \
    (void)0
#  endif

#  ifndef WITH_CUDA_DYNLOAD
/* Transparently implement some functions, so majority of the file does not need
 * to worry about difference between dynamically loaded and linked CUDA at all. */
const char *cuewErrorString(CUresult result);
const char *cuewCompilerPath();
int cuewCompilerVersion();
#  endif /* WITH_CUDA_DYNLOAD */

CCL_NAMESPACE_END

#endif /* WITH_CUDA */
