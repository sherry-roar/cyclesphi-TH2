/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2022 Blender Foundation */

#pragma once

#ifdef WITH_CUDA

#  include "device/cuda/kernel.h"
#  include "device/cuda/queue.h"
#  include "device/cuda/util.h"
#  include "device/device.h"

#  include "util/map.h"

#  ifdef WITH_CUDA_DYNLOAD
#    include "cuew.h"
#  else
#    include <cuda.h>
#    ifndef BLENDER_CLIENT
#      include <cudaGL.h>
#    endif
#  endif

#  ifdef BLENDER_CLIENT
#    include <cuda_runtime.h>

#    define CUDA_DEVICE_POINTER_MAP_ID 0
#    define CUDA_DEVICE_POINTER_SORT 1
#    define CUDA_DEVICE_POINTER_COUNT 4

#    define STREAM_PATH1 0
#    define STREAM_PATH2 1
#    define STREAM_PATH1_MEMCPY 2
#    define STREAM_PATH2_MEMCPY 3
#    define STREAM_COUNT 4
#    define EVENT_COUNT (2 * STREAM_COUNT)

#  endif

CCL_NAMESPACE_BEGIN

class DeviceQueue;

class CUDADevice : public Device {

  friend class CUDAContextScope;

 public:
  CUdevice cuDevice;
#  ifndef BLENDER_CLIENT
  CUcontext cuContext;
#  endif
  CUmodule cuModule;
  size_t device_texture_headroom;
  size_t device_working_headroom;
  bool move_texture_to_host;
  size_t map_host_used;
  size_t map_host_limit;
  int can_map_host;
  int pitch_alignment;
  int cuDevId;
  int cuDevArchitecture;
  bool first_error;

#  ifdef BLENDER_CLIENT
  struct CUDAMem {
    CUDAMem()
        : texobject(0),
          array(0),
          free_map_host(false),
          map_host_pointer(NULL),          
          host_pointer(NULL),
          //param2D(NULL),
          tex_type(-1),
          //data_count(0),
          use_mapped_host(false)
    // data_count(0)
    {
    }

    CUtexObject texobject;
    CUarray array;
    // std::vector<char> map_host_pointer;
    char *map_host_pointer;
    bool free_map_host;
    bool uni_mem;
    //std::string name;
    //size_t size;

    device_memory mem;
    device_memory dpointers[CUDA_DEVICE_POINTER_COUNT];
    //DEVICE_PTR device_pointer;
    //DEVICE_PTR device_pointers[CUDA_DEVICE_POINTER_COUNT];

    char *host_pointer;
    CUDA_MEMCPY2D param2D;
    CUDA_MEMCPY3D param3D;
    CUDA_TEXTURE_DESC texDesc;
    CUDA_RESOURCE_DESC resDesc;

    int tex_type;
    size_t counter_pitch;
    //size_t data_count;

    bool use_mapped_host;
  };
  typedef map<device_ptr, CUDAMem> CUDAMemMap;
#  else
  struct CUDAMem {
    CUDAMem() : texobject(0), array(0), use_mapped_host(false)
    {
    }

    CUtexObject texobject;
    CUarray array;

    /* If true, a mapped host memory in shared_pointer is being used. */
    bool use_mapped_host;
  };
  typedef map<device_memory *, CUDAMem> CUDAMemMap;
#  endif
  CUDAMemMap cuda_mem_map;
  thread_mutex cuda_mem_map_mutex;

#  ifdef BLENDER_CLIENT
  cudaStream_t stream[STREAM_COUNT];
  cudaEvent_t event[EVENT_COUNT];
  device_ptr texture_info_dmem;
  std::vector<char> texture_info_mem;
  float running_time[STREAM_COUNT];
  size_t time_count;
  ccl::KernelWorkTile wtile;
  int wtile_h;
  int path_stat_is_done;

  std::map<std::string, CUDAMem> cuda_stat_map;
  CUmodule cuModuleStat;

  int used_buffer;
  int start_sample;
  int num_samples;
#  endif

  /* Bindless Textures */
#  ifdef BLENDER_CLIENT
  std::vector<TextureInfo> texture_info;
#  else
  device_vector<TextureInfo> texture_info;
#  endif

  bool need_texture_info;

  CUDADeviceKernels kernels;

#  ifndef BLENDER_CLIENT
  static bool have_precompiled_kernels();

  virtual BVHLayoutMask get_bvh_layout_mask() const override;

  void set_error(const string &error) override;

  CUDADevice(const DeviceInfo &info, Stats &stats, Profiler &profiler);

  virtual ~CUDADevice();

  bool support_device(const uint /*kernel_features*/);

  bool check_peer_access(Device *peer_device) override;

  bool use_adaptive_compilation();

  virtual string compile_kernel_get_common_cflags(const uint kernel_features);

  string compile_kernel(const uint kernel_features,
                        const char *name,
                        const char *base = "cuda",
                        bool force_ptx = false);

  virtual bool load_kernels(const uint kernel_features) override;

  void reserve_local_memory(const uint kernel_features);

  void init_host_memory();

  void load_texture_info();

  void move_textures_to_host(size_t size, bool for_texture);

#  endif

  CUDAMem *generic_alloc(device_memory &mem, size_t pitch_padding = 0);

  void generic_copy_to(device_memory &mem);

  void generic_free(device_memory &mem);

  void mem_alloc(device_memory &mem) override;

  void mem_copy_to(device_memory &mem) override;

  void mem_copy_from(device_memory &mem, size_t y, size_t w, size_t h, size_t elem) override;

  void mem_zero(device_memory &mem) override;

  void mem_free(device_memory &mem) override;

  device_ptr mem_alloc_sub_ptr(device_memory &mem, size_t offset, size_t /*size*/) override;

  virtual void const_copy_to(const char *name, void *host, size_t size) override;

  void global_alloc(device_memory &mem);

  void global_free(device_memory &mem);

  void tex_alloc(device_texture &mem);

  void tex_free(device_texture &mem);

#  ifndef BLENDER_CLIENT
  virtual bool should_use_graphics_interop() override;
#  endif

  virtual unique_ptr<DeviceQueue> gpu_queue_create() override;

  int get_num_multiprocessors();
  int get_max_num_threads_per_multiprocessor();

 protected:
  bool get_device_attribute(CUdevice_attribute attribute, int *value);
  int get_device_default_attribute(CUdevice_attribute attribute, int default_value);
};

#  ifdef BLENDER_CLIENT
class CUDADevices {
 public:
  CUDADevices() : size_(0)
  {
  }

  int size()
  {
    return size_;
  }

  void resize(int s)
  {
    size_ = s;
  }

  CUDADevice &operator[](int64_t index)
  {
    return devices_[index];
  }

 private:
  CUDADevice devices_[16];
  int size_;
};
#  endif

CCL_NAMESPACE_END

#endif
