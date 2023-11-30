/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2022 Blender Foundation */

#pragma once

/* So ImathMath is included before our kernel_cpu_compat. */
#ifdef WITH_OSL
/* So no context pollution happens from indirectly included windows.h */
#  include "util/windows.h"
#  include <OSL/oslexec.h>
#endif

#include "device/cpu/kernel.h"
#include "device/device.h"
#include "device/memory.h"

// clang-format off
#include "kernel/device/cpu/compat.h"
#include "kernel/device/cpu/kernel.h"
#include "kernel/device/cpu/globals.h"

#include "kernel/osl/shader.h"
#include "kernel/osl/globals.h"
// clang-format on

#include "device/client/queue.h"

#ifdef WITH_CLIENT_OPTIX
#  ifdef WITH_CUDA_DYNLOAD
#    include "cuew.h"
#  else
#    include <cuda.h>
#    include <cudaGL.h>
#  endif
#  define OPTIX_DONT_INCLUDE_CUDA
#  include <optix.h>
#endif

#  include "device/cuda/device_impl.h"

class DeviceQueue;

CCL_NAMESPACE_BEGIN

#ifdef WITH_CLIENT_OPTIX
class BVHOptiX;
// enum OptixBuildOperation;
#endif

class CLIENTDevice : public CUDADevice {

  //friend class CUDAContextScope;

 public:
  //CUdevice cuDevice;
  //CUcontext cuContext;
  //CUmodule cuModule;
  //int cuDevId;
  //int cuDevArchitecture;

  KernelGlobalsCPU kernel_globals;

  //device_vector<TextureInfo> texture_info;
  //bool need_texture_info;

  //#ifdef WITH_CLIENT_RENDERENGINE_SENDER
  //  int custom_width;
  //  int custom_height;
  //  int custom_pix_size;
  //#endif

  int tile_num_samples;
  double fps_time;
  double fps_count;

  //CPUKernels kernels;
  //static int instance_count;

  CLIENTDevice(const DeviceInfo &info_, Stats &stats_, Profiler &profiler_);
  ~CLIENTDevice();

  virtual BVHLayoutMask get_bvh_layout_mask() const override;

  /* Returns true if the texture info was copied to the device (meaning, some more
   * re-initialization might be needed). */
  bool load_texture_info();

  virtual void generic_copy_dtod(device_ptr mem1, device_ptr mem2, size_t memory_size) override;

  virtual void mem_alloc(device_memory &mem) override;
  virtual void mem_copy_to(device_memory &mem) override;
  virtual void mem_copy_from(
      device_memory &mem, size_t y, size_t w, size_t h, size_t elem) override;
  virtual void mem_zero(device_memory &mem) override;
  virtual void mem_free(device_memory &mem) override;
  virtual device_ptr mem_alloc_sub_ptr(device_memory &mem,
                                       size_t offset,
                                       size_t /*size*/) override;

  virtual void const_copy_to(const char *name, void *host, size_t size) override;

  void global_alloc(device_memory &mem);
  void global_free(device_memory &mem);

  void tex_alloc(device_texture &mem);
  void tex_free(device_texture &mem);

  //virtual bool should_use_graphics_interop() override;

#ifdef WITH_CLIENT_OPTIX
  //  CUcontext cuContext;
  //  OptixDeviceContext context;
  vector<OptixBuildInput *> delayed_optix_build_input;
  bool motion_blur;
  bool build_optix_bvh(BVHOptiX *bvh,
                       OptixBuildOperation operation,
                       OptixBuildInput *build_input,
                       uint16_t num_motion_steps);

  void release_optix_bvh(BVH *bvh) override;
  void free_bvh_memory_delayed();
  vector<unique_ptr<device_only_memory<char>>> delayed_free_bvh_memory;
#endif

  void build_bvh(BVH *bvh, Progress &progress, bool refit) override;

  // virtual const CPUKernels *get_cpu_kernels() const override;
  virtual void get_cpu_kernel_thread_globals(
      vector<CPUKernelThreadGlobals> &kernel_thread_globals) override;
  virtual void *get_cpu_osl_memory() override;

  // virtual unique_ptr<DeviceQueue> gpu_queue_create() override;

 protected:
  virtual bool load_kernels(uint /*kernel_features*/) override;
};

CCL_NAMESPACE_END
