/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2022 Blender Foundation */

#pragma once

#ifdef WITH_HIP

#  ifdef WITH_HIP_DYNLOAD
#    include "hipew.h"
#  endif

CCL_NAMESPACE_BEGIN

class HIPDevice;
class HIPDevices;

#  ifdef BLENDER_CLIENT
extern HIPDevices hip_devices;
#  endif

/* Utility to push/pop HIP context. */
class HIPContextScope {
 public:
  HIPContextScope(HIPDevice *device);
  ~HIPContextScope();

#  ifdef BLENDER_CLIENT
  HIPContextScope(int id);
  HIPDevice &get();
#  endif

 private:
  HIPDevice *device;
};

/* Utility for checking return values of HIP function calls. */
#  define hip_device_assert(hip_device, stmt) \
    { \
      hipError_t result = stmt; \
      if (result != hipSuccess) { \
        const char *name = hipewErrorString(result); \
        hip_device->set_error( \
            string_printf("%s in %s (%s:%d)", name, #stmt, __FILE__, __LINE__)); \
      } \
    } \
    (void)0

#  define hip_assert(stmt) hip_device_assert(this, stmt)

#  ifdef BLENDER_CLIENT

void hip_util_check_exit();

#    define hip_assert2(stmt) \
      { \
        if (stmt != hipSuccess) { \
          char err[1024]; \
          sprintf(err, \
                  "HIP error: %s in %s, %s, line %d", \
                  hipewErrorString(stmt), \
                  __FILE__, \
                  #stmt, \
                  __LINE__); \
          std::string message(err); \
          fprintf(stderr, "%s\n", message.c_str()); \
          ccl::hip_util_check_exit(); \
        } \
      } \
      (void)0
#  endif

#  ifndef WITH_HIP_DYNLOAD
/* Transparently implement some functions, so majority of the file does not need
 * to worry about difference between dynamically loaded and linked HIP at all. */
const char *hipewErrorString(hipError_t result);
const char *hipewCompilerPath();
int hipewCompilerVersion();
#  endif /* WITH_HIP_DYNLOAD */

static inline bool hipSupportsDevice(const int hipDevId)
{
  int major, minor;
  hipDeviceGetAttribute(&major, hipDeviceAttributeComputeCapabilityMajor, hipDevId);
  hipDeviceGetAttribute(&minor, hipDeviceAttributeComputeCapabilityMinor, hipDevId);

  return (major >= 9);
}

CCL_NAMESPACE_END

#endif /* WITH_HIP */
