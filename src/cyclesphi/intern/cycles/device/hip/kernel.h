/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2022 Blender Foundation */

#pragma once

#ifdef WITH_HIP

#  include "device/kernel.h"

#  ifdef WITH_HIP_DYNLOAD
#    include "hipew.h"
#  endif

CCL_NAMESPACE_BEGIN

class HIPDevice;

/* HIP kernel and associate occupancy information. */
class HIPDeviceKernel {
 public:
  hipFunction_t function = nullptr;

  int num_threads_per_block = 0;
  int min_blocks = 0;
};

/* Cache of HIP kernels for each DeviceKernel. */
class HIPDeviceKernels {
 public:
  void load(HIPDevice *device);
  const HIPDeviceKernel &get(DeviceKernel kernel) const;
  bool available(DeviceKernel kernel) const;

#  ifdef BLENDER_CLIENT
  bool is_loaded()
  {
    return loaded;
  }
#  endif

#  ifdef WITH_HIP_STAT
  void set_use_stat(bool us)
  {
    use_stat = us;
  }
#  endif
 protected:
  HIPDeviceKernel kernels_[DEVICE_KERNEL_NUM];

#  ifdef WITH_HIP_STAT
  HIPDeviceKernel kernels_stat_[DEVICE_KERNEL_NUM];
  bool use_stat = false;
#  endif

  bool loaded = false;
};

CCL_NAMESPACE_END

#endif /* WITH_HIP */
