/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2022 Blender Foundation */

#ifdef WITH_HIP

#  include "device/hip/util.h"
#  include "device/hip/device_impl.h"

CCL_NAMESPACE_BEGIN

#  ifdef BLENDER_CLIENT
HIPDevices hip_devices;
#  endif

HIPContextScope::HIPContextScope(HIPDevice *device) : device(device)
{
#ifdef BLENDER_CLIENT
  hipSetDevice(device->hipDevice);
#  else
  hip_device_assert(device, hipCtxPushCurrent(device->hipContext));
#  endif
}

HIPContextScope::~HIPContextScope()
{
#  ifndef BLENDER_CLIENT
  hip_device_assert(device, hipCtxPopCurrent(NULL));
#  endif
}

#  ifdef BLENDER_CLIENT
void hip_util_check_exit()
{
  fflush(0);
  assert(false);
  exit(-1);
}

HIPContextScope::HIPContextScope(int id) : HIPContextScope(&hip_devices[id])
{
}

HIPDevice &HIPContextScope::get()
{
  return *device;
}
#  endif

#  ifndef WITH_HIP_DYNLOAD
const char *hipewErrorString(hipError_t result)
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

const char *hipewCompilerPath()
{
  return CYCLES_HIP_HIPCC_EXECUTABLE;
}

int hipewCompilerVersion()
{
  return (HIP_VERSION / 100) + (HIP_VERSION % 100 / 10);
}
#  endif

CCL_NAMESPACE_END

#endif /* WITH_HIP */
