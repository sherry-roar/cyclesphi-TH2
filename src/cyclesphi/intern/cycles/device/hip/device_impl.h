/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2022 Blender Foundation */

#ifdef BLENDER_CLIENT
#  pragma once
#endif

#ifdef WITH_HIP

#  include "device/device.h"
#  include "device/hip/kernel.h"
#  include "device/hip/queue.h"
#  include "device/hip/util.h"

#  include "util/map.h"

#  ifdef WITH_HIP_DYNLOAD
#    include "hipew.h"
#  endif

#  ifdef BLENDER_CLIENT
#    define HIP_DEVICE_POINTER_MAP_ID 0
#    define HIP_DEVICE_POINTER_SORT 1
#    define HIP_DEVICE_POINTER_COUNT 4

#    define STREAM_PATH1 0
#    define STREAM_PATH2 1
#    define STREAM_PATH1_MEMCPY 2
#    define STREAM_PATH2_MEMCPY 3
#    define STREAM_COUNT 4

#    define EVENT_COUNT (2 * STREAM_COUNT)
#  endif

CCL_NAMESPACE_BEGIN

class DeviceQueue;

class HIPDevice : public Device {

  friend class HIPContextScope;

 public:
  hipDevice_t hipDevice;
#  ifndef BLENDER_CLIENT
  hipCtx_t hipContext;
#  endif
  hipModule_t hipModule;
  hipModule_t hipModuleIC;
  size_t device_texture_headroom;
  size_t device_working_headroom;
  bool move_texture_to_host;
  size_t map_host_used;
  size_t map_host_limit;
  int can_map_host;
  int pitch_alignment;
  int hipDevId;
  int hipDevArchitecture;
  bool first_error;

#  ifdef BLENDER_CLIENT
  struct HIPMem {
    HIPMem()
        : texobject(0),
          array(0),
          free_map_host(false),
          map_host_pointer(NULL),
          host_pointer(NULL),
          // param2D(NULL),
          tex_type(-1),
          // data_count(0),
          use_mapped_host(false)
    // data_count(0)
    {
    }

    hipTextureObject_t texobject;
    hArray array;
    // std::vector<char> map_host_pointer;
    char *map_host_pointer;
    bool free_map_host;
    bool uni_mem;
    // std::string name;
    // size_t size;

    device_memory mem;
    device_memory dpointers[HIP_DEVICE_POINTER_COUNT];
    // DEVICE_PTR device_pointer;
    // DEVICE_PTR device_pointers[HIP_DEVICE_POINTER_COUNT];

    char *host_pointer;
    hip_Memcpy2D param2D;
    HIP_MEMCPY3D param3D;
    hipTextureDesc texDesc;
    hipResourceDesc resDesc;

    int tex_type;
    size_t counter_pitch;
    // size_t data_count;

    bool use_mapped_host;
  };
  typedef map<device_ptr, HIPMem> HIPMemMap;
#  else

  struct HIPMem {
    HIPMem() : texobject(0), array(0), use_mapped_host(false)
    {
    }

    hipTextureObject_t texobject;
    hArray array;

    /* If true, a mapped host memory in shared_pointer is being used. */
    bool use_mapped_host;
  };
  typedef map<device_memory *, HIPMem> HIPMemMap;
#  endif
  HIPMemMap hip_mem_map;
  thread_mutex hip_mem_map_mutex;

#  ifdef BLENDER_CLIENT
  hipStream_t stream[STREAM_COUNT];
  hipEvent_t event[EVENT_COUNT];
  device_ptr texture_info_dmem;
  std::vector<char> texture_info_mem;
  float running_time[STREAM_COUNT];
  size_t time_count;
  ccl::KernelWorkTile wtile;
  int wtile_h;
  int path_stat_is_done;

  std::map<std::string, HIPMem> hip_stat_map;
  hipModule_t hipModuleStat;

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

  HIPDeviceKernels kernels;

#  ifndef BLENDER_CLIENT
  static bool have_precompiled_kernels();

  virtual BVHLayoutMask get_bvh_layout_mask() const override;

  void set_error(const string &error) override;

  HIPDevice(const DeviceInfo &info, Stats &stats, Profiler &profiler);

  virtual ~HIPDevice();

  bool support_device(const uint /*kernel_features*/);

  bool check_peer_access(Device *peer_device) override;

  bool use_adaptive_compilation();

  virtual string compile_kernel_get_common_cflags(const uint kernel_features);

  string compile_kernel(const uint kernel_features, const char *name, const char *base = "hip");

  virtual bool load_kernels(const uint kernel_features) override;
  void reserve_local_memory(const uint kernel_features);

  void init_host_memory();

  void load_texture_info();

  void move_textures_to_host(size_t size, bool for_texture);
#  endif

  HIPMem *generic_alloc(device_memory &mem, size_t pitch_padding = 0);

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

  /* Graphics resources interoperability. */
#  ifndef BLENDER_CLIENT
  virtual bool should_use_graphics_interop() override;
#  endif

  virtual unique_ptr<DeviceQueue> gpu_queue_create() override;

  int get_num_multiprocessors();
  int get_max_num_threads_per_multiprocessor();

 protected:
  bool get_device_attribute(hipDeviceAttribute_t attribute, int *value);
  int get_device_default_attribute(hipDeviceAttribute_t attribute, int default_value);
};

#  ifdef BLENDER_CLIENT
class HIPDevices {
 public:
  HIPDevices() : size_(0)
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

  HIPDevice &operator[](int64_t index)
  {
    return devices_[index];
  }

 private:
  HIPDevice devices_[16];
  int size_;
};
#  endif

CCL_NAMESPACE_END

#  endif
