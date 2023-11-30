/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2022 Blender Foundation */

/* CUDA kernel entry points */

#ifdef __CUDA_ARCH__

#  include "kernel/device/cuda/compat.h"
#  include "kernel/device/cuda/config.h"
#  include "kernel/device/cuda/globals.h"

#  if defined(WITH_CUDA_CPUIMAGE)
#    include "kernel/device/cpu/image.h"
#  else
#    include "kernel/device/gpu/image.h"
#  endif

#  include "kernel/device/gpu/kernel.h"

#endif
