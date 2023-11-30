/*
 * Copyright 2011-2021 Blender Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License atf
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "device/client/device_impl.h"

#include <stdlib.h>
#include <string.h>

#include "device/cpu/kernel.h"
#include "device/cpu/kernel_thread_globals.h"

#include "device/device.h"

// clang-format off
#include "kernel/device/cpu/compat.h"
#include "kernel/device/cpu/globals.h"
#include "kernel/device/cpu/kernel.h"
#include "kernel/types.h"

#include "kernel/osl/shader.h"
#include "kernel/osl/globals.h"
// clang-format on

#include "bvh/embree.h"

#include "device/cuda/util.h"

#ifdef WITH_CLIENT_OPTIX
#  include "bvh/bvh.h"
#  include "bvh/optix.h"
#  include "device/optix/util.h"

#  include "scene/hair.h"
#  include "scene/mesh.h"
#  include "scene/pointcloud.h"
#  include "scene/object.h"
#  include "scene/pass.h"
#  include "scene/scene.h"
#endif

#include "session/buffers.h"

#include "util/debug.h"
#include "util/foreach.h"
#include "util/function.h"
#include "util/log.h"
#include "util/map.h"
#include "util/openimagedenoise.h"
#include "util/optimization.h"
#include "util/progress.h"
#include "util/system.h"
#include "util/task.h"
#include "util/thread.h"

#include "kernel/device/client/kernel_client.h"

#define DEBUG_PRINT(size) //printf("%s: %lld\n", __FUNCTION__, size);
#define DEBUG_PRINT_MEM(mem) //printf("%s, %s: %lld\n", __FUNCTION__, mem.name, mem.device_size);
#define CHECK_CLIENT_ERROR \
  if (cyclesphi::kernel::client::is_error()) \
    set_error(string_printf("error in %s", __FUNCTION__));

CCL_NAMESPACE_BEGIN

// class CLIENTContextScope {
// public:
//  CLIENTContextScope(CLIENTDevice *device) : device_(device)
//  {
//    cuda_device_assert(device, cuCtxPushCurrent(device->cuContext));
//  }
//  ~CLIENTContextScope()
//  {
//    cuda_device_assert(device_, cuCtxPopCurrent(NULL));
//  }
//
// private:
//  CLIENTDevice *device_;
//};

//int CLIENTDevice::instance_count = 0;

CLIENTDevice::CLIENTDevice(const DeviceInfo &info_, Stats &stats_, Profiler &profiler_)
    : CUDADevice(info_, stats_, profiler_)  //, texture_info(this, "texture_info", MEM_GLOBAL)
{
  /* Pick any kernel, all of them are supposed to have same level of microarchitecture
   * optimization. */
  // VLOG(1) << "Will be using " << kernels.integrator_init_from_camera.get_uarch_name()
  //        << " kernels.";

  if (info.cpu_threads == 0) {
    info.cpu_threads = TaskScheduler::max_concurrency();
  }

  //#ifdef WITH_CLIENT_RENDERENGINE_SENDER
  //  custom_width = 1920;
  //    custom_height = 1080;
  //    custom_pix_size = 4;
  //
  //    const char *w_env = std::getenv("DEBUG_RES_W");
  //    if (w_env != NULL)
  //      custom_width = atoi(w_env);
  //
  //    const char *h_env = std::getenv("DEBUG_RES_H");
  //    if (h_env != NULL)
  //      custom_height = atoi(h_env);
  //#endif

  need_texture_info = false;

#ifdef WITH_CLIENT_OPTIX
  // cuda_assert(cuInit(0));

  // OptixDeviceContextOptions options = {};
  // optix_assert(optixDeviceContextCreate(cuContext, &options, &context));
  motion_blur = false;
#endif

  //if (CLIENTDevice::instance_count > 0) {
  //  CLIENTDevice::instance_count++;
  //  set_error(string_printf("error in %s: CLIENT_IS_RUNNING", __FUNCTION__));
  //  return;
  //}

  //CLIENTDevice::instance_count++;

  DEBUG_PRINT(0)
  cyclesphi::kernel::client::alloc_kg();
  CHECK_CLIENT_ERROR;

  // cuda_assert(cuInit(0));
  // CUdevice cuDevice;
  // cuda_assert(cuDeviceGet(&cuDevice, 0));
  // cuda_assert(cuCtxCreate(&g_cuContext, CU_CTX_LMEM_RESIZE_TO_MAX, cuDevice));
  /* Initialize CUDA. */
  // CUresult result = cuInit(0);
  // if (result != CUDA_SUCCESS) {
  //  set_error(string_printf("Failed to initialize CUDA runtime (%s)", cuewErrorString(result)));
  //  return;
  //}

  ///* Setup device and context. */
  // result = cuDeviceGet(&cuDevice, cuDevId);
  // if (result != CUDA_SUCCESS) {
  //  set_error(string_printf("Failed to get CUDA device handle from ordinal (%s)",
  //                          cuewErrorString(result)));
  //  return;
  //}

  ///* Create context. */
  // unsigned int ctx_flags = CU_CTX_LMEM_RESIZE_TO_MAX;
  // result = cuCtxCreate(&cuContext, ctx_flags, cuDevice);

  // if (result != CUDA_SUCCESS) {
  //  set_error(string_printf("Failed to create CUDA context (%s)", cuewErrorString(result)));
  //  return;
  //}

  // int major, minor;
  // cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevId);
  // cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevId);
  // cuDevArchitecture = major * 100 + minor * 10;

  ///* Pop context set by cuCtxCreate. */
  // cuCtxPopCurrent(NULL);
}

CLIENTDevice::~CLIENTDevice()
{
  //CLIENTDevice::instance_count--;

  //if (CLIENTDevice::instance_count > 0) {
  //  return;
  //}

  texture_info.free();

  //#ifdef WITH_CLIENT_OPTIX
  //  free_bvh_memory_delayed();
  //
  //  cuda_assert(cuCtxPushCurrent(cuContext));
  //  optixDeviceContextDestroy(context);
  //  cuda_assert(cuCtxPopCurrent(NULL));
  //
  //  cuCtxDestroy(cuContext);
  //#endif

#ifdef WITH_CLIENT_OPTIX
  for (int i = 0; i < delayed_optix_build_input.size(); i++) {
    delete delayed_optix_build_input[i];
  }
  delayed_optix_build_input.clear();
#endif

  DEBUG_PRINT(0)
  cyclesphi::kernel::client::free_kg();
  CHECK_CLIENT_ERROR;

  //cuda_assert(cuCtxDestroy(cuContext));
}

BVHLayoutMask CLIENTDevice::get_bvh_layout_mask() const
{
#ifdef WITH_CLIENT_OPTIX
  /* OptiX has its own internal acceleration structure format. */
  return BVH_LAYOUT_OPTIX;
#else
  return BVH_LAYOUT_BVH2;
#endif
}

bool CLIENTDevice::load_texture_info()
{
  if (!need_texture_info) {
    return false;
  }

  texture_info.copy_to_device();
  DEBUG_PRINT(texture_info.size())
  cyclesphi::kernel::client::load_textures(texture_info.size(), NULL /*client_data*/);
  CHECK_CLIENT_ERROR;

  need_texture_info = false;

  return true;
}

void CLIENTDevice::mem_alloc(device_memory &mem)
{
  if (mem.type == MEM_TEXTURE) {
    assert(!"mem_alloc not supported for textures.");
  }
  else if (mem.type == MEM_GLOBAL) {
    assert(!"mem_alloc not supported for global memory.");
  }
  else {
    if (mem.name) {
      VLOG_WORK << "Buffer allocate: " << mem.name << ", "
              << string_human_readable_number(mem.memory_size()) << " bytes. ("
              << string_human_readable_size(mem.memory_size()) << ")";
    }

    if (mem.type == MEM_DEVICE_ONLY || !mem.host_pointer) {
      size_t alignment = MIN_ALIGNMENT_CPU_DATA_TYPES;
      void *data = util_aligned_malloc(mem.memory_size(), alignment);
      mem.device_pointer = (device_ptr)data;
    }
    else {
      mem.device_pointer = (device_ptr)mem.host_pointer;
    }

    mem.device_size = mem.memory_size();
    stats.mem_alloc(mem.device_size);

    DEBUG_PRINT_MEM(mem)
    size_t mem_device_size = mem.device_size;

    //#ifdef WITH_CLIENT_RENDERENGINE_SENDER
    //    if (!strcmp(mem.name, "RenderBuffers")) {
    //        int pass_stride = 4;
    ////        mem_device_size = custom_width * custom_height * sizeof(float) * pass_stride * 2;
    //      }
    //#endif

    if (!strcmp(mem.name, "pixels")) {
      CUDAContextScope scope(this);
      cuda_assert(cuMemAlloc((CUdeviceptr*)&mem.device_pointer, mem_device_size));
    }

    cyclesphi::kernel::client::mem_alloc(
        mem.name, mem.device_pointer, mem_device_size, NULL, NULL /*client_data*/);
    CHECK_CLIENT_ERROR;
  }
}

void CLIENTDevice::generic_copy_dtod(device_ptr mem1, device_ptr mem2, size_t memory_size)
{
  if (!mem1 || !mem2) {
    return;
  }

  //const CUDAContextScope scope(this);
  cuda_assert(cuMemcpyDtoD((CUdeviceptr)mem1, (CUdeviceptr)mem2, memory_size));
}

void CLIENTDevice::mem_copy_to(device_memory &mem)
{
  if (mem.type == MEM_GLOBAL) {
    global_free(mem);
    global_alloc(mem);
  }
  else if (mem.type == MEM_TEXTURE) {
    tex_free((device_texture &)mem);
    tex_alloc((device_texture &)mem);
  }
  else {
    if (!mem.device_pointer) {
      mem_alloc(mem);
    }

    if (!strcmp(mem.name, "RenderBuffers"))
      return;

    DEBUG_PRINT_MEM(mem)
    cyclesphi::kernel::client::mem_copy_to(mem.name,
        mem.device_pointer, mem.device_size, 0, NULL, NULL /*cyclesphi::kernel::client::data*/);
    CHECK_CLIENT_ERROR;
  }
}

void CLIENTDevice::mem_copy_from(device_memory &mem, size_t y, size_t w, size_t h, size_t elem)
{
  if (!strcmp(mem.name, "pixels")) {
    CUDAContextScope scope(this);
    const size_t size = elem * w * h;
    const size_t offset = elem * y * w;

    cuda_assert(cuMemcpyDtoH(
        (char *)mem.host_pointer + offset, (CUdeviceptr)mem.device_pointer + offset, size));
  }
}

void CLIENTDevice::mem_zero(device_memory &mem)
{
  if (!mem.device_pointer) {
    mem_alloc(mem);
  }

  if (mem.device_pointer /* && strcmp(mem.name, "RenderBuffers") */) {

    DEBUG_PRINT_MEM(mem)

    size_t mem_device_size = mem.device_size;

    //#ifdef WITH_CLIENT_RENDERENGINE_SENDER
    //    if (!strcmp(mem.name, "RenderBuffers")) {
    //        int pass_stride = 4;
    //        mem_device_size = custom_width * custom_height * sizeof(float) * pass_stride * 2;
    //      }
    //#endif

    if (!strcmp(mem.name, "pixels")) {
      CUDAContextScope scope(this);
      cuda_assert(cuMemsetD8(mem.device_pointer, 0, mem_device_size));
    }
    else {
      memset((void *)mem.device_pointer, 0, mem.memory_size());
    }

    cyclesphi::kernel::client::mem_zero(
        mem.name,
        mem.device_pointer, mem_device_size, 0, NULL, NULL /*client_data*/);
    CHECK_CLIENT_ERROR;
  }
}

void CLIENTDevice::mem_free(device_memory &mem)
{
  //  if (mem.type == MEM_PIXELS) {
  //    pixels = 0;
  //    cyclesphi::kernel::client::mem_free(pixels, 0, NULL, NULL /*client_data*/);
  //  }
  //  else
  if (mem.type == MEM_GLOBAL) {
    global_free(mem);
  }
  else if (mem.type == MEM_TEXTURE) {
    tex_free((device_texture &)mem);
  }
  else if (mem.device_pointer) {
    DEBUG_PRINT_MEM(mem)
    cyclesphi::kernel::client::mem_free(
        mem.name,
        mem.device_pointer, mem.device_size, NULL, NULL /*client_data*/);
    CHECK_CLIENT_ERROR;

    if (!strcmp(mem.name, "pixels")) {
      CUDAContextScope scope(this);
      cuda_assert(cuMemFree(mem.device_pointer));
    }
    else {
      if (mem.type == MEM_DEVICE_ONLY) {
        util_aligned_free((void *)mem.device_pointer);
      }
    }

    mem.device_pointer = 0;
    stats.mem_free(mem.device_size);
    mem.device_size = 0;
  }
}

device_ptr CLIENTDevice::mem_alloc_sub_ptr(device_memory &mem, size_t offset, size_t /*size*/)
{
  // return (device_ptr)(((char *)mem.device_pointer) + mem.memory_elements_size(offset));
  device_ptr mem_sub = (device_ptr)(((char *)mem.device_pointer) +
                                    mem.memory_elements_size(offset));
  DEBUG_PRINT_MEM(mem)
  cyclesphi::kernel::client::mem_alloc_sub_ptr(mem.name,
      mem.device_pointer, mem.memory_elements_size(offset), mem_sub, NULL /*client_data*/);
  CHECK_CLIENT_ERROR;

  return mem_sub;
}

void CLIENTDevice::const_copy_to(const char *name, void *host, size_t size)
{
  kernel_const_copy(&kernel_globals, name, host, size);
  DEBUG_PRINT(size)
  cyclesphi::kernel::client::const_copy(name, (char *)host, size, NULL, NULL /*client_data*/);
  CHECK_CLIENT_ERROR;
}

void CLIENTDevice::global_alloc(device_memory &mem)
{
  VLOG(1) << "Global memory allocate: " << mem.name << ", "
          << string_human_readable_number(mem.memory_size()) << " bytes. ("
          << string_human_readable_size(mem.memory_size()) << ")";

  kernel_global_memory_copy(&kernel_globals, mem.name, mem.host_pointer, mem.data_size);

  DEBUG_PRINT_MEM(mem)
  cyclesphi::kernel::client::tex_copy(
      mem.name, (void *)mem.host_pointer, mem.data_size, mem.memory_size(), NULL, NULL);
  CHECK_CLIENT_ERROR;

  mem.device_pointer = (device_ptr)mem.host_pointer;
  mem.device_size = mem.memory_size();
  stats.mem_alloc(mem.device_size);
}

void CLIENTDevice::global_free(device_memory &mem)
{
  if (mem.device_pointer) {
    DEBUG_PRINT_MEM(mem)
    cyclesphi::kernel::client::tex_free(
        mem.name,
        mem.device_pointer, mem.device_size, NULL, NULL /*client_data*/);
    CHECK_CLIENT_ERROR;

    mem.device_pointer = 0;
    stats.mem_free(mem.device_size);
    mem.device_size = 0;
  }
}

void CLIENTDevice::tex_alloc(device_texture &mem)
{
  VLOG_WORK << "Texture allocate: " << mem.name << ", "
          << string_human_readable_number(mem.memory_size()) << " bytes. ("
          << string_human_readable_size(mem.memory_size()) << ")";

  mem.device_pointer = (device_ptr)mem.host_pointer;
  mem.device_size = mem.memory_size();
  stats.mem_alloc(mem.device_size);

  const uint slot = mem.slot;
  if (slot >= texture_info.size()) {
    /* Allocate some slots in advance, to reduce amount of re-allocations. */
    texture_info.resize(slot + 128);

    for (int i = slot; i < slot + 128; i++) {
      texture_info[i].data = 0;
    }
  }

  texture_info[slot] = mem.info;
  texture_info[slot].data = (uint64_t)mem.host_pointer;
  need_texture_info = true;

  DEBUG_PRINT_MEM(mem)
  cyclesphi::kernel::client::tex_info((char *)mem.host_pointer,
                                     mem.memory_size(),
                                     mem.name,
                                     mem.data_type,
                                     mem.data_elements,
                                     mem.info.interpolation,
                                     mem.info.extension,
                                     mem.data_width,
                                     mem.data_height,
                                     mem.data_depth);
  CHECK_CLIENT_ERROR;
}

void CLIENTDevice::tex_free(device_texture &mem)
{
  if (mem.device_pointer) {
    mem.device_pointer = 0;
    stats.mem_free(mem.device_size);
    mem.device_size = 0;
    need_texture_info = true;
  }
}

// unique_ptr<DeviceQueue> CLIENTDevice::gpu_queue_create()
//{
//  return make_unique<CLIENTDeviceQueue>(this);
//}

// bool CLIENTDevice::should_use_graphics_interop()
//{
//  /* Check whether this device is part of OpenGL context.
//   *
//   * Using CUDA device for graphics interoperability which is not part of the OpenGL context is
//   * possible, but from the empiric measurements it can be considerably slower than using naive
//   * pixels copy. */
//
//  CLIENTContextScope scope(this);
//
//  int num_all_devices = 0;
//  cuda_assert(cuDeviceGetCount(&num_all_devices));
//
//  if (num_all_devices == 0) {
//    return false;
//  }
//
//  vector<CUdevice> gl_devices(num_all_devices);
//  uint num_gl_devices;
//  cuGLGetDevices(&num_gl_devices, gl_devices.data(), num_all_devices, CU_GL_DEVICE_LIST_ALL);
//
//  for (CUdevice gl_device : gl_devices) {
//    if (gl_device == cuDevice) {
//      return true;
//    }
//  }
//
//  return false;
//}

#if 0 //def WITH_CLIENT_OPTIX
bool CLIENTDevice::build_optix_bvh(BVHOptiX *bvh,
                                   OptixBuildOperation operation,
                                   OptixBuildInput *build_input,
                                   uint16_t num_motion_steps)
{
  // const CUDAContextScope scope(this);
  // cudaSetDevice();

  const bool use_fast_trace_bvh = (bvh->params.bvh_type == BVH_TYPE_STATIC);

  /* Compute memory usage. */
  OptixAccelBufferSizes sizes = {};
  OptixAccelBuildOptions options = {};
  options.operation = operation;
  if (use_fast_trace_bvh ||
      /* The build flags have to match the ones used to query the built-in curve intersection
         program (see optixBuiltinISModuleGet above) */
      build_input.type == OPTIX_BUILD_INPUT_TYPE_CURVES) {
    VLOG_INFO << "Using fast to trace OptiX BVH";
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  }
  else {
    VLOG_INFO << "Using fast to update OptiX BVH";
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
  }

  options.motionOptions.numKeys = num_motion_steps;
  options.motionOptions.flags = OPTIX_MOTION_FLAG_START_VANISH | OPTIX_MOTION_FLAG_END_VANISH;
  options.motionOptions.timeBegin = 0.0f;
  options.motionOptions.timeEnd = 1.0f;

  optix_assert(optixAccelComputeMemoryUsage(context, &options, &build_input, 1, &sizes));

  /* Allocate required output buffers. */
  // device_only_memory<char> temp_mem(this, "optix temp as build mem");
  // temp_mem.alloc_to_device(align_up(sizes.tempSizeInBytes, 8) + 8);
  // if (!temp_mem.device_pointer) {
  //  /* Make sure temporary memory allocation succeeded. */
  //  return false;
  //}
  CUdeviceptr temp_mem_d = NULL;
  cuda_assert(cuMemAlloc(&temp_mem_d, align_up(sizes.tempSizeInBytes, 8) + 8));

   device_only_memory<char> &out_data = *bvh->as_data;
   if (operation == OPTIX_BUILD_OPERATION_BUILD) {
    assert(out_data.device == this);
    out_data.alloc_to_device(sizes.outputSizeInBytes);
    if (!out_data.device_pointer) {
      return false;
    }
  }
   else {
    assert(out_data.device_pointer && out_data.device_size >= sizes.outputSizeInBytes);
  }
  CUdeviceptr out_data_d = NULL;
  //std::vector<char> out_data_h(sizes.outputSizeInBytes);
  cuda_assert(cuMemAlloc(&out_data_d, sizes.outputSizeInBytes));

  /* Finally build the acceleration structure. */
  OptixAccelEmitDesc compacted_size_prop = {};
  compacted_size_prop.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  /* A tiny space was allocated for this property at the end of the temporary buffer above.
   * Make sure this pointer is 8-byte aligned. */
  compacted_size_prop.result = align_up(temp_mem_d + sizes.tempSizeInBytes, 8);

  OptixTraversableHandle out_handle = 0;
  optix_assert(optixAccelBuild(context,
                               NULL,
                               &options,
                               &build_input,
                               1,
                               temp_mem_d,
                               sizes.tempSizeInBytes,
                               out_data_d,
                               sizes.outputSizeInBytes,
                               &out_handle,
                               use_fast_trace_bvh ? &compacted_size_prop : NULL,
                               use_fast_trace_bvh ? 1 : 0));

  bvh->traversable_handle = static_cast<uint64_t>(out_handle);

  /* Wait for all operations to finish. */
  cuda_assert(cuStreamSynchronize(NULL));

#  if 0
  /* Compact acceleration structure to save memory (do not do this in viewport for faster builds).
   */
  if (use_fast_trace_bvh) {
    uint64_t compacted_size = sizes.outputSizeInBytes;
    cuda_assert(cuMemcpyDtoH(&compacted_size, compacted_size_prop.result, sizeof(compacted_size)));

    /* Temporary memory is no longer needed, so free it now to make space. */
    //temp_mem.free();
    //cuMemFree(temp_mem_d);

    /* There is no point compacting if the size does not change. */
    if (compacted_size < sizes.outputSizeInBytes) {
      //device_only_memory<char> compacted_data(this, "optix compacted as");
      //compacted_data.alloc_to_device(compacted_size);
      CUdeviceptr compacted_data_device_pointer;
      cuda_assert(cuMemAllocManaged(&compacted_data_device_pointer, compacted_size, CU_MEM_ATTACH_GLOBAL));

      if (!compacted_data_device_pointer)
        /* Do not compact if memory allocation for compacted acceleration structure fails.
         * Can just use the uncompacted one then, so succeed here regardless. */
        return !have_error();

      optix_assert(optixAccelCompact(
          context, NULL, out_handle, compacted_data_device_pointer, compacted_size, &out_handle));
      bvh->traversable_handle = static_cast<uint64_t>(out_handle);

      /* Wait for compaction to finish. */
      cuda_assert(cuStreamSynchronize(NULL));

      //std::swap(out_data.device_size, compacted_data.device_size);
      //std::swap(out_data_d, compacted_data.device_pointer);
    }
  }
#  endif

  cuda_assert(cuMemcpyDtoH((char*)out_data.device_pointer, out_data_d, sizes.outputSizeInBytes));
  cyclesphi::kernel::client::mem_copy_to(out_data.device_pointer,
                                        out_data.device_size,
                                        0,
                                        NULL,
                                        NULL /*cyclesphi::kernel::client::data*/);
  CHECK_CLIENT_ERROR;

  if (temp_mem_d!=NULL)
    cuda_assert(cuMemFree(temp_mem_d));

  if (out_data_d != NULL)
    cuda_assert(cuMemFree(out_data_d));

  return !have_error();
}
#endif

#ifdef WITH_CLIENT_OPTIX

bool CLIENTDevice::build_optix_bvh(BVHOptiX *bvh,
                                   OptixBuildOperation operation,
                                   OptixBuildInput *build_input,
                                   uint16_t num_motion_steps)
{
  DEBUG_PRINT(0)
  cyclesphi::kernel::client::build_optix_bvh(
      (int)operation, (char *)build_input, sizeof(OptixBuildInput), (int)num_motion_steps);
  CHECK_CLIENT_ERROR;

  // TODO - it is not unique
  bvh->traversable_handle = (uint64_t)build_input;

  return true;
}

void CLIENTDevice::build_bvh(BVH *bvh, Progress &progress, bool refit)
{
  const bool use_fast_trace_bvh = (bvh->params.bvh_type == BVH_TYPE_STATIC);

  free_bvh_memory_delayed();

  BVHOptiX *const bvh_optix = static_cast<BVHOptiX *>(bvh);

  progress.set_substatus("Building OptiX acceleration structure");

  if (!bvh->params.top_level) {
    assert(bvh->objects.size() == 1 && bvh->geometry.size() == 1);

    /* Refit is only possible in viewport for now (because AS is built with
     * OPTIX_BUILD_FLAG_ALLOW_UPDATE only there, see above). */
    OptixBuildOperation operation = OPTIX_BUILD_OPERATION_BUILD;
    if (refit && !use_fast_trace_bvh) {
      assert(bvh_optix->traversable_handle != 0);
      operation = OPTIX_BUILD_OPERATION_UPDATE;
    }
    else {
      bvh_optix->as_data->free();
      bvh_optix->traversable_handle = 0;
    }

    /* Build bottom level acceleration structures (BLAS). */
    Geometry *const geom = bvh->geometry[0];
    if (geom->geometry_type == Geometry::HAIR) {
      /* Build BLAS for curve primitives. */
      Hair *const hair = static_cast<Hair *const>(geom);
      if (hair->num_curves() == 0) {
        return;
      }

      const size_t num_segments = hair->num_segments();

      size_t num_motion_steps = 1;
      Attribute *motion_keys = hair->attributes.find(ATTR_STD_MOTION_VERTEX_POSITION);
      if (motion_blur && hair->get_use_motion_blur() && motion_keys) {
        num_motion_steps = hair->get_motion_steps();
      }

      device_vector<OptixAabb> aabb_data(this, "optix temp aabb data", MEM_READ_ONLY);
      device_vector<int> index_data(this, "optix temp index data", MEM_READ_ONLY);
      device_vector<float4> vertex_data(this, "optix temp vertex data", MEM_READ_ONLY);
      /* Four control points for each curve segment. */
      const size_t num_vertices = num_segments * 4;
      if (hair->curve_shape == CURVE_THICK) {
        index_data.alloc(num_segments);
        vertex_data.alloc(num_vertices * num_motion_steps);
      }
      else
        aabb_data.alloc(num_segments * num_motion_steps);

      /* Get AABBs for each motion step. */
      for (size_t step = 0; step < num_motion_steps; ++step) {
        /* The center step for motion vertices is not stored in the attribute. */
        const float3 *keys = hair->get_curve_keys().data();
        size_t center_step = (num_motion_steps - 1) / 2;
        if (step != center_step) {
          size_t attr_offset = (step > center_step) ? step - 1 : step;
          /* Technically this is a float4 array, but sizeof(float3) == sizeof(float4). */
          keys = motion_keys->data_float3() + attr_offset * hair->get_curve_keys().size();
        }

        for (size_t j = 0, i = 0; j < hair->num_curves(); ++j) {
          const Hair::Curve curve = hair->get_curve(j);
          const array<float> &curve_radius = hair->get_curve_radius();

          for (int segment = 0; segment < curve.num_segments(); ++segment, ++i) {
            if (hair->curve_shape == CURVE_THICK) {
              int k0 = curve.first_key + segment;
              int k1 = k0 + 1;
              int ka = max(k0 - 1, curve.first_key);
              int kb = min(k1 + 1, curve.first_key + curve.num_keys - 1);

              index_data[i] = i * 4;
              float4 *const v = vertex_data.data() + step * num_vertices + index_data[i];

#  if OPTIX_ABI_VERSION >= 55
              v[0] = make_float4(keys[ka].x, keys[ka].y, keys[ka].z, curve_radius[ka]);
              v[1] = make_float4(keys[k0].x, keys[k0].y, keys[k0].z, curve_radius[k0]);
              v[2] = make_float4(keys[k1].x, keys[k1].y, keys[k1].z, curve_radius[k1]);
              v[3] = make_float4(keys[kb].x, keys[kb].y, keys[kb].z, curve_radius[kb]);
#  else
              const float4 px = make_float4(keys[ka].x, keys[k0].x, keys[k1].x, keys[kb].x);
              const float4 py = make_float4(keys[ka].y, keys[k0].y, keys[k1].y, keys[kb].y);
              const float4 pz = make_float4(keys[ka].z, keys[k0].z, keys[k1].z, keys[kb].z);
              const float4 pw = make_float4(
                  curve_radius[ka], curve_radius[k0], curve_radius[k1], curve_radius[kb]);

              /* Convert Catmull-Rom data to B-spline. */
              static const float4 cr2bsp0 = make_float4(+7, -4, +5, -2) / 6.f;
              static const float4 cr2bsp1 = make_float4(-2, 11, -4, +1) / 6.f;
              static const float4 cr2bsp2 = make_float4(+1, -4, 11, -2) / 6.f;
              static const float4 cr2bsp3 = make_float4(-2, +5, -4, +7) / 6.f;

              v[0] = make_float4(
                  dot(cr2bsp0, px), dot(cr2bsp0, py), dot(cr2bsp0, pz), dot(cr2bsp0, pw));
              v[1] = make_float4(
                  dot(cr2bsp1, px), dot(cr2bsp1, py), dot(cr2bsp1, pz), dot(cr2bsp1, pw));
              v[2] = make_float4(
                  dot(cr2bsp2, px), dot(cr2bsp2, py), dot(cr2bsp2, pz), dot(cr2bsp2, pw));
              v[3] = make_float4(
                  dot(cr2bsp3, px), dot(cr2bsp3, py), dot(cr2bsp3, pz), dot(cr2bsp3, pw));
#  endif
            }
            else {
              BoundBox bounds = BoundBox::empty;
              curve.bounds_grow(segment, keys, hair->get_curve_radius().data(), bounds);

              const size_t index = step * num_segments + i;
              aabb_data[index].minX = bounds.min.x;
              aabb_data[index].minY = bounds.min.y;
              aabb_data[index].minZ = bounds.min.z;
              aabb_data[index].maxX = bounds.max.x;
              aabb_data[index].maxY = bounds.max.y;
              aabb_data[index].maxZ = bounds.max.z;
            }
          }
        }
      }

      /* Upload AABB data to GPU. */
      aabb_data.copy_to_device();
      index_data.copy_to_device();
      vertex_data.copy_to_device();

      // vector<device_ptr> aabb_ptrs;
      // aabb_ptrs.reserve(num_motion_steps);
      // vector<device_ptr> width_ptrs;
      // vector<device_ptr> vertex_ptrs;
      // width_ptrs.reserve(num_motion_steps);
      // vertex_ptrs.reserve(num_motion_steps);
      // for (size_t step = 0; step < num_motion_steps; ++step) {
      //  aabb_ptrs.push_back(aabb_data.device_pointer + step * num_segments * sizeof(OptixAabb));
      //  const device_ptr base_ptr = vertex_data.device_pointer +
      //                              step * num_vertices * sizeof(float4);
      //  width_ptrs.push_back(base_ptr + 3 * sizeof(float)); /* Offset by vertex size. */
      //  vertex_ptrs.push_back(base_ptr);
      //}

      /* Force a single any-hit call, so shadow record-all behavior works correctly. */
      unsigned int build_flags = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
      OptixBuildInput *build_input = new OptixBuildInput();
      delayed_optix_build_input.push_back(build_input);
      if (hair->curve_shape == CURVE_THICK) {
        build_input->type = OPTIX_BUILD_INPUT_TYPE_CURVES;
#  if OPTIX_ABI_VERSION >= 55
        build_input->curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM;
#  else
        build_input->curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
#  endif
        build_input->curveArray.numPrimitives = num_segments;
        build_input->curveArray.vertexBuffers =
            (CUdeviceptr *)vertex_data.device_pointer;  // vertex_ptrs.data();
        build_input->curveArray.numVertices = num_vertices;
        build_input->curveArray.vertexStrideInBytes = sizeof(float4);
        build_input->curveArray.widthBuffers =
            (CUdeviceptr *)vertex_data.device_pointer;  // width_ptrs.data();
        build_input->curveArray.widthStrideInBytes = sizeof(float4);
        build_input->curveArray.indexBuffer = (CUdeviceptr)index_data.device_pointer;
        build_input->curveArray.indexStrideInBytes = sizeof(int);
        build_input->curveArray.flag = build_flags;
        build_input->curveArray.primitiveIndexOffset = hair->curve_segment_offset;
      }
      else {
        /* Disable visibility test any-hit program, since it is already checked during
         * intersection. Those trace calls that require any-hit can force it with a ray flag. */
        build_flags |= OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

        build_input->type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        build_input->customPrimitiveArray.aabbBuffers =
            (CUdeviceptr *)aabb_data.device_pointer;  // aabb_ptrs.data();
        build_input->customPrimitiveArray.numPrimitives = num_segments;
        build_input->customPrimitiveArray.strideInBytes = sizeof(OptixAabb);
        build_input->customPrimitiveArray.flags = &build_flags;
        build_input->customPrimitiveArray.numSbtRecords = 1;
        build_input->customPrimitiveArray.primitiveIndexOffset = hair->curve_segment_offset;
      }

      if (!build_optix_bvh(bvh_optix, operation, build_input, num_motion_steps)) {
        progress.set_error("Failed to build OptiX acceleration structure");
      }
    }
    else if (geom->geometry_type == Geometry::MESH || geom->geometry_type == Geometry::VOLUME) {
      /* Build BLAS for triangle primitives. */
      Mesh *const mesh = static_cast<Mesh *const>(geom);
      if (mesh->num_triangles() == 0) {
        return;
      }

      const size_t num_verts = mesh->get_verts().size();

      size_t num_motion_steps = 1;
      Attribute *motion_keys = mesh->attributes.find(ATTR_STD_MOTION_VERTEX_POSITION);
      if (motion_blur && mesh->get_use_motion_blur() && motion_keys) {
        num_motion_steps = mesh->get_motion_steps();
      }

      device_vector<int> index_data(this, "optix temp index data", MEM_READ_ONLY);
      index_data.alloc(mesh->get_triangles().size());
      memcpy(index_data.data(),
             mesh->get_triangles().data(),
             mesh->get_triangles().size() * sizeof(int));
      device_vector<float4> vertex_data(this, "optix temp vertex data", MEM_READ_ONLY);
      vertex_data.alloc(num_verts * num_motion_steps);

      for (size_t step = 0; step < num_motion_steps; ++step) {
        const float3 *verts = mesh->get_verts().data();

        size_t center_step = (num_motion_steps - 1) / 2;
        /* The center step for motion vertices is not stored in the attribute. */
        if (step != center_step) {
          verts = motion_keys->data_float3() + (step > center_step ? step - 1 : step) * num_verts;
        }

        memcpy(vertex_data.data() + num_verts * step, verts, num_verts * sizeof(float3));
      }

      /* Upload triangle data to GPU. */
      index_data.copy_to_device();
      vertex_data.copy_to_device();

      vector<device_ptr> vertex_ptrs;
      vertex_ptrs.reserve(num_motion_steps);
      for (size_t step = 0; step < num_motion_steps; ++step) {
        vertex_ptrs.push_back(vertex_data.device_pointer + num_verts * step * sizeof(float3));
      }

      /* Force a single any-hit call, so shadow record-all behavior works correctly. */
      unsigned int build_flags = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
      OptixBuildInput *build_input = new OptixBuildInput();
      delayed_optix_build_input.push_back(build_input);

      build_input->type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
      build_input->triangleArray.vertexBuffers =
          (CUdeviceptr *)vertex_data.device_pointer;  // vertex_ptrs.data();
      build_input->triangleArray.numVertices = num_verts;
      build_input->triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
      build_input->triangleArray.vertexStrideInBytes = sizeof(float4);
      build_input->triangleArray.indexBuffer = index_data.device_pointer;
      build_input->triangleArray.numIndexTriplets = mesh->num_triangles();
      build_input->triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
      build_input->triangleArray.indexStrideInBytes = 3 * sizeof(int);
      build_input->triangleArray.flags = &build_flags;
      /* The SBT does not store per primitive data since Cycles already allocates separate
       * buffers for that purpose. OptiX does not allow this to be zero though, so just pass in
       * one and rely on that having the same meaning in this case. */
      build_input->triangleArray.numSbtRecords = 1;
      build_input->triangleArray.primitiveIndexOffset = mesh->prim_offset;

      if (!build_optix_bvh(bvh_optix, operation, build_input, num_motion_steps)) {
        progress.set_error("Failed to build OptiX acceleration structure");
      }
    }
    else if (geom->geometry_type == Geometry::POINTCLOUD) {
      /* Build BLAS for points primitives. */
      PointCloud *const pointcloud = static_cast<PointCloud *const>(geom);

      const size_t num_points = pointcloud->num_points();
      if (num_points == 0) {
        return;
      }

      size_t num_motion_steps = 1;
      Attribute *motion_points = pointcloud->attributes.find(ATTR_STD_MOTION_VERTEX_POSITION);
      if (motion_blur && pointcloud->get_use_motion_blur() && motion_points) {
        num_motion_steps = pointcloud->get_motion_steps();
      }

      device_vector<OptixAabb> aabb_data(this, "optix temp aabb data", MEM_READ_ONLY);
      aabb_data.alloc(num_points * num_motion_steps);

      /* Get AABBs for each motion step. */
      for (size_t step = 0; step < num_motion_steps; ++step) {
        /* The center step for motion vertices is not stored in the attribute. */
        const float3 *points = pointcloud->get_points().data();
        const float *radius = pointcloud->get_radius().data();
        size_t center_step = (num_motion_steps - 1) / 2;
        if (step != center_step) {
          size_t attr_offset = (step > center_step) ? step - 1 : step;
          /* Technically this is a float4 array, but sizeof(float3) == sizeof(float4). */
          points = motion_points->data_float3() + attr_offset * num_points;
        }

        for (size_t i = 0; i < num_points; ++i) {
          const PointCloud::Point point = pointcloud->get_point(i);
          BoundBox bounds = BoundBox::empty;
          point.bounds_grow(points, radius, bounds);

          const size_t index = step * num_points + i;
          aabb_data[index].minX = bounds.min.x;
          aabb_data[index].minY = bounds.min.y;
          aabb_data[index].minZ = bounds.min.z;
          aabb_data[index].maxX = bounds.max.x;
          aabb_data[index].maxY = bounds.max.y;
          aabb_data[index].maxZ = bounds.max.z;
        }
      }

      /* Upload AABB data to GPU. */
      aabb_data.copy_to_device();

      //vector<device_ptr> aabb_ptrs;
      //aabb_ptrs.reserve(num_motion_steps);
      //for (size_t step = 0; step < num_motion_steps; ++step) {
      //  aabb_ptrs.push_back(aabb_data.device_pointer + step * num_points * sizeof(OptixAabb));
      //}

      /* Disable visibility test any-hit program, since it is already checked during
       * intersection. Those trace calls that require anyhit can force it with a ray flag.
       * For those, force a single any-hit call, so shadow record-all behavior works correctly. */
      unsigned int build_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT |
                                 OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
      OptixBuildInput *build_input = new OptixBuildInput();
      delayed_optix_build_input.push_back(build_input);

      build_input->type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
#  if OPTIX_ABI_VERSION < 23
      build_input->aabbArray.aabbBuffers = 
          (CUdeviceptr *)aabb_data.device_pointer;  // aabb_ptrs.data();
      build_input->aabbArray.numPrimitives = num_points;
      build_input->aabbArray.strideInBytes = sizeof(OptixAabb);
      build_input->aabbArray.flags = &build_flags;
      build_input->aabbArray.numSbtRecords = 1;
      build_input->aabbArray.primitiveIndexOffset = pointcloud->prim_offset;
#  else
      build_input->customPrimitiveArray.aabbBuffers =
          (CUdeviceptr *)aabb_data.device_pointer;  // aabb_ptrs.data();
      build_input->customPrimitiveArray.numPrimitives = num_points;
      build_input->customPrimitiveArray.strideInBytes = sizeof(OptixAabb);
      build_input->customPrimitiveArray.flags = &build_flags;
      build_input->customPrimitiveArray.numSbtRecords = 1;
      build_input->customPrimitiveArray.primitiveIndexOffset = pointcloud->prim_offset;
#  endif

      if (!build_optix_bvh(bvh_optix, operation, build_input, num_motion_steps)) {
        progress.set_error("Failed to build OptiX acceleration structure");
      }
    }
  }
  else {
    unsigned int num_instances = 0;
    unsigned int max_num_instances = 0xFFFFFFFF;

    bvh_optix->as_data->free();
    bvh_optix->traversable_handle = 0;
    bvh_optix->motion_transform_data->free();

    // optixDeviceContextGetProperty(context,
    //                              OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID,
    //                              &max_num_instances,
    //                              sizeof(max_num_instances));
    ///* Do not count first bit, which is used to distinguish instanced and non-instanced objects.
    ///*/
    // max_num_instances >>= 1;
    // if (bvh->objects.size() > max_num_instances) {
    //  progress.set_error(
    //      "Failed to build OptiX acceleration structure because there are too many instances");
    //  return;
    //}

    /* Fill instance descriptions. */
    device_vector<OptixInstance> instances(this, "optix tlas instances", MEM_READ_ONLY);
    instances.alloc(bvh->objects.size());

    /* Calculate total motion transform size and allocate memory for them. */
    size_t motion_transform_offset = 0;
    if (motion_blur) {
      size_t total_motion_transform_size = 0;
      for (Object *const ob : bvh->objects) {
        if (ob->is_traceable() && ob->use_motion()) {
          total_motion_transform_size = align_up(total_motion_transform_size,
                                                 OPTIX_TRANSFORM_BYTE_ALIGNMENT);
          const size_t motion_keys = max(ob->get_motion().size(), (size_t)2) - 2;
          total_motion_transform_size = total_motion_transform_size +
                                        sizeof(OptixSRTMotionTransform) +
                                        motion_keys * sizeof(OptixSRTData);
        }
      }

      assert(bvh_optix->motion_transform_data->device == this);
      bvh_optix->motion_transform_data->alloc_to_device(total_motion_transform_size);
    }

    for (Object *ob : bvh->objects) {
      /* Skip non-traceable objects. */
      if (!ob->is_traceable()) {
        continue;
      }

      BVHOptiX *const blas = static_cast<BVHOptiX *>(ob->get_geometry()->bvh);
      OptixTraversableHandle handle = blas->traversable_handle;

      OptixInstance &instance = instances[num_instances++];
      memset(&instance, 0, sizeof(instance));

      /* Clear transform to identity matrix. */
      instance.transform[0] = 1.0f;
      instance.transform[5] = 1.0f;
      instance.transform[10] = 1.0f;

      /* Set user instance ID to object index. */
      instance.instanceId = ob->get_device_index();

      /* Add some of the object visibility bits to the mask.
       * __prim_visibility contains the combined visibility bits of all instances, so is not
       * reliable if they differ between instances. But the OptiX visibility mask can only contain
       * 8 bits, so have to trade-off here and select just a few important ones.
       */
      instance.visibilityMask = ob->visibility_for_tracing() & 0xFF;

      /* Have to have at least one bit in the mask, or else instance would always be culled. */
      if (0 == instance.visibilityMask) {
        instance.visibilityMask = 0xFF;
      }

      if (ob->get_geometry()->geometry_type == Geometry::HAIR &&
          static_cast<const Hair *>(ob->get_geometry())->curve_shape == CURVE_THICK) {
        if (motion_blur && ob->get_geometry()->has_motion_blur()) {
          /* Select between motion blur and non-motion blur built-in intersection module. */
          // instance.sbtOffset = PG_HITD_MOTION - PG_HITD;
        }
      }
      else if (ob->get_geometry()->geometry_type == Geometry::POINTCLOUD) {
        /* Use the hit group that has an intersection program for point clouds. */
        //instance.sbtOffset = PG_HITD_POINTCLOUD - PG_HITD;

        /* Also skip point clouds in local trace calls. */
        instance.visibilityMask |= 4;
      }

#  if OPTIX_ABI_VERSION < 55
      /* Cannot disable any-hit program for thick curves, since it needs to filter out end-caps. */
      else
#  endif
      {
        /* Can disable __anyhit__kernel_optix_visibility_test by default (except for thick curves,
         * since it needs to filter out end-caps there).
         *
         * It is enabled where necessary (visibility mask exceeds 8 bits or the other any-hit
         * programs like __anyhit__kernel_optix_shadow_all_hit) via OPTIX_RAY_FLAG_ENFORCE_ANYHIT.
         */
        instance.flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
      }

      /* Insert motion traversable if object has motion. */
      if (motion_blur && ob->use_motion()) {
        size_t motion_keys = max(ob->get_motion().size(), (size_t)2) - 2;
        size_t motion_transform_size = sizeof(OptixSRTMotionTransform) +
                                       motion_keys * sizeof(OptixSRTData);

        // const CUDAContextScope scope(this);

        motion_transform_offset = align_up(motion_transform_offset,
                                           OPTIX_TRANSFORM_BYTE_ALIGNMENT);
        CUdeviceptr motion_transform_gpu = bvh_optix->motion_transform_data->device_pointer +
                                           motion_transform_offset;
        motion_transform_offset += motion_transform_size;

        /* Allocate host side memory for motion transform and fill it with transform data. */
        OptixSRTMotionTransform &motion_transform = *reinterpret_cast<OptixSRTMotionTransform *>(
            new uint8_t[motion_transform_size]);
        motion_transform.child = handle;
        motion_transform.motionOptions.numKeys = ob->get_motion().size();
        motion_transform.motionOptions.flags = OPTIX_MOTION_FLAG_NONE;
        motion_transform.motionOptions.timeBegin = 0.0f;
        motion_transform.motionOptions.timeEnd = 1.0f;

        OptixSRTData *const srt_data = motion_transform.srtData;
        array<DecomposedTransform> decomp(ob->get_motion().size());
        transform_motion_decompose(
            decomp.data(), ob->get_motion().data(), ob->get_motion().size());

        for (size_t i = 0; i < ob->get_motion().size(); ++i) {
          /* Scale. */
          srt_data[i].sx = decomp[i].y.w; /* scale.x.x */
          srt_data[i].sy = decomp[i].z.w; /* scale.y.y */
          srt_data[i].sz = decomp[i].w.w; /* scale.z.z */

          /* Shear. */
          srt_data[i].a = decomp[i].z.x; /* scale.x.y */
          srt_data[i].b = decomp[i].z.y; /* scale.x.z */
          srt_data[i].c = decomp[i].w.x; /* scale.y.z */
          assert(decomp[i].z.z == 0.0f); /* scale.y.x */
          assert(decomp[i].w.y == 0.0f); /* scale.z.x */
          assert(decomp[i].w.z == 0.0f); /* scale.z.y */

          /* Pivot point. */
          srt_data[i].pvx = 0.0f;
          srt_data[i].pvy = 0.0f;
          srt_data[i].pvz = 0.0f;

          /* Rotation. */
          srt_data[i].qx = decomp[i].x.x;
          srt_data[i].qy = decomp[i].x.y;
          srt_data[i].qz = decomp[i].x.z;
          srt_data[i].qw = decomp[i].x.w;

          /* Translation. */
          srt_data[i].tx = decomp[i].y.x;
          srt_data[i].ty = decomp[i].y.y;
          srt_data[i].tz = decomp[i].y.z;
        }

        /* Upload motion transform to GPU. */
        // TODO
        // cuMemcpyHtoD(motion_transform_gpu, &motion_transform, motion_transform_size);
        // delete[] reinterpret_cast<uint8_t *>(&motion_transform);

        /* Get traversable handle to motion transform. */
        // optixConvertPointerToTraversableHandle(context,
        //                                       motion_transform_gpu,
        //                                       OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM,
        //                                       &instance.traversableHandle);
      }
      else {
        instance.traversableHandle = handle;

        if (ob->get_geometry()->is_instanced()) {
          /* Set transform matrix. */
          memcpy(instance.transform, &ob->get_tfm(), sizeof(instance.transform));
        }
      }
    }

    /* Upload instance descriptions. */
    instances.resize(num_instances);
    instances.copy_to_device();

    /* Build top-level acceleration structure (TLAS) */
    OptixBuildInput *build_input = new OptixBuildInput();
    delayed_optix_build_input.push_back(build_input);

    build_input->type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    build_input->instanceArray.instances = instances.device_pointer;
    build_input->instanceArray.numInstances = num_instances;

    if (!build_optix_bvh(bvh_optix, OPTIX_BUILD_OPERATION_BUILD, build_input, 0)) {
      progress.set_error("Failed to build OptiX acceleration structure");
    }
    // tlas_handle = bvh_optix->traversable_handle;
  }
}

// TODO remove
void CLIENTDevice::release_optix_bvh(BVH *bvh)
{
  // thread_scoped_lock lock(delayed_free_bvh_mutex);
  /* Do delayed free of BVH memory, since geometry holding BVH might be deleted
   * while GPU is still rendering. */
  BVHOptiX *const bvh_optix = static_cast<BVHOptiX *>(bvh);

  // delayed_free_bvh_memory.emplace_back(std::move(bvh_optix->as_data));
  // delayed_free_bvh_memory.emplace_back(std::move(bvh_optix->motion_transform_data));
  // delayed_free_bvh_memory.free_memory();

  bvh_optix->traversable_handle = 0;
}

void CLIENTDevice::free_bvh_memory_delayed()
{
  // thread_scoped_lock lock(delayed_free_bvh_mutex);
  // delayed_free_bvh_memory.free_memory();
}

#else

void CLIENTDevice::build_bvh(BVH *bvh, Progress &progress, bool refit)
{
  Device::build_bvh(bvh, progress, refit);
}

#endif

void CLIENTDevice::get_cpu_kernel_thread_globals(
    vector<CPUKernelThreadGlobals> &kernel_thread_globals)
{
  /* Ensure latest texture info is loaded into kernel globals before returning. */
  load_texture_info();

  kernel_thread_globals.clear();
  void *osl_memory = get_cpu_osl_memory();
  for (int i = 0; i < info.cpu_threads; i++) {
    kernel_thread_globals.emplace_back(kernel_globals, osl_memory, profiler);
  }
}

void *CLIENTDevice::get_cpu_osl_memory()
{
  return NULL;
}

bool CLIENTDevice::load_kernels(uint /*kernel_features*/)
{
  return true;
}

CCL_NAMESPACE_END
