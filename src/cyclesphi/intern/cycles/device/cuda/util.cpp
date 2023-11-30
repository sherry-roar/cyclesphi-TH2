/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2022 Blender Foundation */

#ifdef WITH_CUDA

#  include "device/cuda/util.h"
#  include "device/cuda/device_impl.h"

#  ifdef BLENDER_CLIENT

#    ifdef WITH_CLIENT_OPTIX
#      include "device/optix/device_impl.h"
#    endif

#define CYCLES_CUDA_NVCC_EXECUTABLE ""
#  endif

CCL_NAMESPACE_BEGIN

#  ifdef BLENDER_CLIENT
# ifdef WITH_CLIENT_OPTIX
OptiXDevices cuda_devices;
# else
CUDADevices cuda_devices;
# endif
#  endif

CUDAContextScope::CUDAContextScope(CUDADevice *device) : device(device)
{
#ifdef BLENDER_CLIENT
  cudaSetDevice(device->cuDevice);
#else
  cuda_device_assert(device, cuCtxPushCurrent(device->cuContext));
#endif
}

CUDAContextScope::~CUDAContextScope()
{
#ifndef BLENDER_CLIENT
  cuda_device_assert(device, cuCtxPopCurrent(NULL));
#endif
}

#  ifdef BLENDER_CLIENT
void cuda_util_check_exit()
{
  fflush(0);
  assert(false);
  exit(-1);
}

CUDAContextScope::CUDAContextScope(int id) : CUDAContextScope(&cuda_devices[id])
{
}

CUDADevice &CUDAContextScope::get()
{
  return *device;
}
#  endif

#  ifndef WITH_CUDA_DYNLOAD
const char *cuewErrorString(CUresult result)
{
  /* We can only give error code here without major code duplication, that
   * should be enough since dynamic loading is only being disabled by folks
   * who knows what they're doing anyway.
   *
   * NOTE: Avoid call from several threads.
   */
  static string error;
  error = string_printf("%d", result);
  return error.c_str();
}

const char *cuewCompilerPath()
{
  return CYCLES_CUDA_NVCC_EXECUTABLE;
}

int cuewCompilerVersion()
{
  return (CUDA_VERSION / 100) + (CUDA_VERSION % 100 / 10);
}
#  endif

CCL_NAMESPACE_END

#endif /* WITH_CUDA */
