/*
 * Copyright 2011-2013 Blender Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Contributors: Milan Jaros, IT4Innovations, VSB - Technical University of Ostrava
 *
 */

#include "kernel_optix.h"

#include "device/cuda/device_impl.h"
#include "device/optix/device_impl.h"

#include "kernel_cuda_util.h"
#include "kernel_util.h"

#undef __KERNEL_CPU__
#define __KERNEL_OPTIX__
#include "kernel/device/optix/globals.h"

#include <optix_denoiser_tiling.h>
#include <optix_function_table_definition.h>

#include <fstream>
#include <iostream>

namespace cyclesphi {
namespace kernel {
namespace optix {

void update_launch_params(ccl::OptiXDevice *dev, size_t offset, void *data, size_t data_size)
{
  cu_assert(cuMemcpyHtoD(dev->launch_params.device_pointer + offset, data, data_size));
}

DEVICE_PTR get_kernel_globals_cpu()
{
  return cuda::get_kernel_globals_cpu();
}

DEVICE_PTR get_ptr_map(DEVICE_PTR id)
{
  return cuda::get_ptr_map(id);
}

void set_ptr_map(const char *name, DEVICE_PTR id, DEVICE_PTR value)
{
  cuda::set_ptr_map(name, id, value);
}

void path_trace_init(int numDevice,
                     DEVICE_PTR map_kg_bin,
                     DEVICE_PTR map_buffer_bin,
                     DEVICE_PTR map_pixel_bin,
                     int start_sample,
                     int end_sample,
                     int tile_x,
                     int tile_y,
                     int offset,
                     int stride,
                     int tile_h,
                     int tile_w,
                     int h,
                     int w,
                     char *sample_finished_omp,
                     char *reqFinished_omp,
                     int nprocs_cpu,
                     char *signal_value)
{
  cuda::path_trace_init(numDevice,
                        map_kg_bin,
                        map_buffer_bin,
                        map_pixel_bin,
                        start_sample,
                        end_sample,
                        tile_x,
                        tile_y,
                        offset,
                        stride,
                        tile_h,
                        tile_w,
                        h,
                        w,
                        sample_finished_omp,
                        reqFinished_omp,
                        nprocs_cpu,
                        signal_value);
}

void path_trace_finish(int numDevice,
                       DEVICE_PTR map_kg_bin,
                       DEVICE_PTR map_buffer_bin,
                       DEVICE_PTR map_pixel_bin,
                       int start_sample,
                       int end_sample,
                       int tile_x,
                       int tile_y,
                       int offset,
                       int stride,
                       int tile_h,
                       int tile_w,
                       int h,
                       int w,
                       char *sample_finished_omp,
                       char *reqFinished_omp,
                       int nprocs_cpu,
                       char *signal_value)
{
  cuda::path_trace_finish(numDevice,
                          map_kg_bin,
                          map_buffer_bin,
                          map_pixel_bin,
                          start_sample,
                          end_sample,
                          tile_x,
                          tile_y,
                          offset,
                          stride,
                          tile_h,
                          tile_w,
                          h,
                          w,
                          sample_finished_omp,
                          reqFinished_omp,
                          nprocs_cpu,
                          signal_value);
}

void path_trace(int numDevice,
                DEVICE_PTR kg_bin,
                DEVICE_PTR buffer_bin,
                DEVICE_PTR pixels_bin,
                int start_sample,
                int end_sample,
                int tile_x,
                int tile_y,
                int offset,
                int stride,
                int tile_h,
                int tile_w,
                int h,
                int w,
                char *sample_finished_omp,
                char *reqFinished_omp,
                int nprocs_cpu,
                char *signal_value)
{
  cuda::path_trace(numDevice,
                   kg_bin,
                   buffer_bin,
                   pixels_bin,
                   start_sample,
                   end_sample,
                   tile_x,
                   tile_y,
                   offset,
                   stride,
                   tile_h,
                   tile_w,
                   h,
                   w,
                   sample_finished_omp,
                   reqFinished_omp,
                   nprocs_cpu,
                   signal_value);
}

void path_trace_time(DEVICE_PTR kg_bin,
                     DEVICE_PTR buffer_bin,
                     int start_sample,
                     int end_sample,
                     int tile_x,
                     int tile_y,
                     int offset,
                     int stride,
                     int tile_h,
                     int tile_w,
                     int nprocs_cpu,
                     double *row_times)
{
  cuda::path_trace_time(kg_bin,
                        buffer_bin,
                        start_sample,
                        end_sample,
                        tile_x,
                        tile_y,
                        offset,
                        stride,
                        tile_h,
                        tile_w,
                        nprocs_cpu,
                        row_times);
}

void path_trace_lb(int numDevice,
                   DEVICE_PTR map_buffer_bin,
                   int start_sample,
                   int end_sample,
                   int tile_x,
                   int tile_y,
                   int offset,
                   int stride,
                   int tile_h,
                   int tile_w,
                   int nprocs_cpu)
{
  cuda::path_trace_lb(numDevice,
                      map_buffer_bin,
                      start_sample,
                      end_sample,
                      tile_x,
                      tile_y,
                      offset,
                      stride,
                      tile_h,
                      tile_w,
                      nprocs_cpu);
}

void alloc_kg(int numDevice)
{
  cuda::alloc_kg(numDevice);
}

void free_kg(int numDevice, DEVICE_PTR kg)
{
  cuda::free_kg(numDevice, kg);
}

void host_register(const char *name, char *mem, size_t memSize)
{
  cuda::host_register(name, mem, memSize);
}

char *host_alloc(const char *name,
                 DEVICE_PTR mem,
                 size_t memSize,
                 int type)  // 1-managed, 2-pinned
{
  return cuda::host_alloc(name, mem, memSize, type);
}

void host_free(const char *name, DEVICE_PTR mem, char *dmem, int type)
{
  cuda::host_free(name, mem, dmem, type);
}

DEVICE_PTR get_host_get_device_pointer(char *mem)
{
  return cuda::get_host_get_device_pointer(mem);
}

// std::vector<char> g_optix_instances;
// DEVICE_PTR g_optix_instance_d;

DEVICE_PTR mem_alloc(
    int numDevice, const char *name, char *mem, size_t memSize, bool spec_setts, bool alloc_host)
{
  DEVICE_PTR map_id = cuda::mem_alloc(numDevice, name, mem, memSize, spec_setts, alloc_host);

  // if (!strcmp("optix tlas instances", name)) {
  //   g_optix_instances.resize(memSize);
  //   g_optix_instance_d = map_id;
  // }

  return map_id;
}

void mem_copy_to(int numDevice, char *mem, DEVICE_PTR dmem, size_t memSize, char *signal_value)
{
  cuda::mem_copy_to(numDevice, mem, dmem, memSize, signal_value);

  // if (g_optix_instance_d == dmem) {
  //   memcpy(g_optix_instances.data(), mem, memSize);
  // }
}

void mem_copy_from(
    int numDevice, DEVICE_PTR dmem, char *mem, size_t offset, size_t memSize, char *signal_value)
{
  cuda::mem_copy_from(numDevice, dmem, mem, offset, memSize, signal_value);
}

void mem_zero(const char *name, int numDevice, DEVICE_PTR dmem, size_t memSize)
{
  cuda::mem_zero(name, numDevice, dmem, memSize);
}

void mem_free(const char *name, int numDevice, DEVICE_PTR dmem, size_t memSize)
{
  cuda::mem_free(name, numDevice, dmem, memSize);
}

void tex_free(int numDevice, DEVICE_PTR kg_bin, const char *name, DEVICE_PTR dmem, size_t memSize)
{
  cuda::tex_free(numDevice, kg_bin, name, dmem, memSize);
}

DEVICE_PTR tex_info_alloc(int numDevice,
                          char *mem,
                          size_t memSize,
                          const char *name,
                          int data_type,
                          int data_elements,
                          int interpolation,
                          int extension,
                          size_t data_width,
                          size_t data_height,
                          size_t data_depth,
                          bool unimem_flag)
{
  return cuda::tex_info_alloc(numDevice,
                              mem,
                              memSize,
                              name,
                              data_type,
                              data_elements,
                              interpolation,
                              extension,
                              data_width,
                              data_height,
                              data_depth,
                              unimem_flag);
}

void tex_info_copy(const char *name,
                   char *mem,
                   DEVICE_PTR map_id,
                   size_t memSize,
                   int data_type,
                   bool check_uniname)
{
  cuda::tex_info_copy(name, mem, map_id, memSize, data_type, check_uniname);
}

void tex_info_copy_async(
    const char *name, char *mem, DEVICE_PTR map_id, size_t memSize, bool check_uniname)
{
  cuda::tex_info_copy_async(name, mem, map_id, memSize, check_uniname);
}

void const_copy(int numDevice, DEVICE_PTR kg, const char *name, char *host, size_t size, bool save)
{
  cuda::const_copy(numDevice, kg, name, host, size, save);
  //
  //  if (strcmp(name, "data") == 0) {
  //    //assert(size <= sizeof(KernelData));
  //
  //    /* Update traversable handle (since it is different for each device on multi devices). */
  //    ccl::KernelData *const data = (ccl::KernelData *)host;
  //    *(OptixTraversableHandle *)&data->bvh.scene = tlas_handle;
  //
  //    update_launch_params(offsetof(ccl::KernelParamsOptiX, data), host, size);
  //    return;
  //  }
  //
  //  /* Update data storage pointers in launch parameters. */
  //#define KERNEL_DATA_ARRAY(data_type, tex_name) \
//  if (strcmp(name, #tex_name) == 0) { \
//    update_launch_params(offsetof(ccl::KernelParamsOptiX, tex_name), host, size); \
//    return; \
//  }
  //  KERNEL_DATA_ARRAY(ccl::IntegratorStateGPU, integrator_state)
  //#include "kernel/textures.h"
  //#undef KERNEL_DATA_ARRAY

  // printf("omp_const_copy: %s, %.3f, %zu\n", name, (float)size / (1024.0f * 1024.0f));
  // if (strcmp(name, "data") == 0) {
  //  for (int id = 0; id < ccl::cuda_devices.size(); id++) {
  //    ccl::CUDAContextScope scope(id);
  //    ccl::OptiXDevice *dev = (ccl::OptiXDevice *)&scope.get();

  //    ccl::KernelData *const data = (ccl::KernelData *)host;
  //    *(OptixTraversableHandle *)&data->bvh.scene = dev->tlas_handle;

  //    update_launch_params(dev, offsetof(ccl::KernelParamsOptiX, data), host, size);
  //  }
  //}

  // if (save) {
  //  cuda::dev_kernel_data.resize(size);
  //  memcpy(&cuda::dev_kernel_data[0], host_bin, size);

  //#if defined(WITH_CLIENT_RENDERENGINE_VR) || \
    //    (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
  //  cuda::kernel_data_right.resize(size);
  //  memcpy(&cuda::kernel_data_right[0], host_bin, size);
  //#endif
  //}
}

void cam_recalc(char *data)
{
  cuda::cam_recalc(data);
}

// int cmp_data(DEVICE_PTR kg_bin, char *host_bin, size_t size);
size_t get_size_data(DEVICE_PTR kg_bin)
{
  return cuda::get_size_data(kg_bin);
}

DEVICE_PTR get_data(DEVICE_PTR kg_bin)
{
  return cuda::get_data(kg_bin);
}

#if defined(WITH_CLIENT_RENDERENGINE_VR) || \
    (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
DEVICE_PTR get_data_right(DEVICE_PTR kg_bin)
{
  return cuda::get_data_right(kg_bin);
}
#endif

float *get_camera_matrix(DEVICE_PTR kg_bin)
{
  return cuda::get_camera_matrix(kg_bin);
}

void tex_copy(int numDevice,
              DEVICE_PTR kg_bin,
              char *name_bin,
              DEVICE_PTR dmem,
              char *mem,
              size_t data_count,
              size_t mem_size)
{
  cuda::tex_copy(numDevice, kg_bin, name_bin, dmem, mem, data_count, mem_size);

  for (int id = 0; id < ccl::cuda_devices.size(); id++) {
    ccl::CUDAContextScope scope(id);
    ccl::OptiXDevice *dev = (ccl::OptiXDevice *)&scope.get();

    void *dp = &scope.get().cuda_mem_map[dmem].mem.device_pointer;
    size_t ds = sizeof(DEVICE_PTR);

    /* Update data storage pointers in launch parameters. */
#define KERNEL_DATA_ARRAY(data_type, tex_name) \
  if (strcmp(name_bin, #tex_name) == 0) { \
    update_launch_params(dev, offsetof(ccl::KernelParamsOptiX, tex_name), dp, ds); \
    continue; \
  }
    KERNEL_DATA_ARRAY(ccl::IntegratorStateGPU, integrator_state)
#include "kernel/data_arrays.h"
#undef KERNEL_DATA_ARRAY
  }
}

void load_textures(int numDevice, DEVICE_PTR kg_bin, size_t texture_info_size
                   /*std::map<DEVICE_PTR, DEVICE_PTR> &ptr_map*/)
{
  cuda::load_textures(numDevice, kg_bin, texture_info_size /*ptr_map*/);
}

void blender_camera(DEVICE_PTR mem, char *temp_data, size_t mem_size)
{
  cuda::blender_camera(mem, temp_data, mem_size);
}

int get_pass_stride(int numDevice, DEVICE_PTR kg)
{
  return cuda::get_pass_stride(numDevice, kg);
}

void anim_step(int numDevice, DEVICE_PTR kg_bin, char *data_bin, int s)
{
  cuda::anim_step(numDevice, kg_bin, data_bin, s);
}

void socket_step(int numDevice, DEVICE_PTR kg_bin, char *data_bin, char *cd)
{
  cuda::socket_step(numDevice, kg_bin, data_bin, cd);
}

void set_bounces(int numDevice,
                 DEVICE_PTR kg_bin,
                 char *data_bin,
                 int min_bounce,
                 int max_bounce,
                 int max_diffuse_bounce,
                 int max_glossy_bounce,
                 int max_transmission_bounce,
                 int max_volume_bounce,
                 int max_volume_bounds_bounce,
                 int transparent_min_bounce,
                 int transparent_max_bounce,
                 int use_mis_lamp)
{
  cuda::set_bounces(numDevice,
                    kg_bin,
                    data_bin,
                    min_bounce,
                    max_bounce,
                    max_diffuse_bounce,
                    max_glossy_bounce,
                    max_transmission_bounce,
                    max_volume_bounce,
                    max_volume_bounds_bounce,
                    transparent_min_bounce,
                    transparent_max_bounce,
                    use_mis_lamp);
}

//////////////
#ifdef BLENDER_CLIENT_OPTIX_LOGGING
bool g_optix_use_debug = true;
#else
bool g_optix_use_debug = false;
#endif

std::string readPTX(std::string const &filename)
{
  std::ifstream inputPtx(filename);

  if (!inputPtx) {
    std::cerr << "ERROR: readPTX() Failed to open file " << filename << '\n';
    return std::string();
  }

  std::stringstream ptx;

  ptx << inputPtx.rdbuf();

  if (inputPtx.fail()) {
    std::cerr << "ERROR: readPTX() Failed to read file " << filename << '\n';
    return std::string();
  }

  return ptx.str();
}

bool load_kernels(ccl::OptiXDevice *dev, const unsigned int kernel_features)
{
  // if (have_error()) {
  //  /* Abort early if context creation failed already. */
  //  return false;
  //}

  ///* Load CUDA modules because we need some of the utility kernels. */
  // if (!CUDADevice::load_kernels(kernel_features)) {
  //  return false;
  //}

  ///* Skip creating OptiX module if only doing denoising. */
  // if (!(kernel_features & (KERNEL_FEATURE_PATH_TRACING | KERNEL_FEATURE_BAKING))) {
  //  return true;
  //}

  // ccl::CUDAContextScope scope();

  /* Unload existing OptiX module and pipelines first. */
  if (dev->optix_module != NULL) {
    optixModuleDestroy(dev->optix_module);
    dev->optix_module = NULL;
  }
  for (unsigned int i = 0; i < 2; ++i) {
    if (dev->builtin_modules[i] != NULL) {
      optixModuleDestroy(dev->builtin_modules[i]);
      dev->builtin_modules[i] = NULL;
    }
  }
  for (unsigned int i = 0; i < ccl::NUM_PIPELINES; ++i) {
    if (dev->pipelines[i] != NULL) {
      optixPipelineDestroy(dev->pipelines[i]);
      dev->pipelines[i] = NULL;
    }
  }

  OptixModuleCompileOptions module_options = {};
  module_options.maxRegisterCount = 0; /* Do not set an explicit register limit. */

  if (g_optix_use_debug) {
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
  }
  else {
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
  }

  module_options.boundValues = nullptr;
  module_options.numBoundValues = 0;
#if OPTIX_ABI_VERSION >= 55
  module_options.payloadTypes = nullptr;
  module_options.numPayloadTypes = 0;
#endif

  OptixPipelineCompileOptions pipeline_options = {};
  /* Default to no motion blur and two-level graph, since it is the fastest option. */
  pipeline_options.usesMotionBlur = false;
  pipeline_options.traversableGraphFlags =
      OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
  pipeline_options.numPayloadValues = 8;
  pipeline_options.numAttributeValues = 2; /* u, v */
  pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pipeline_options.pipelineLaunchParamsVariableName = "kernel_params"; /* See globals.h */

  pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
  if (kernel_features & ccl::KERNEL_FEATURE_HAIR) {
    if (kernel_features & ccl::KERNEL_FEATURE_HAIR_THICK) {
#if OPTIX_ABI_VERSION >= 55
      pipeline_options.usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CATMULLROM;
#else
      pipeline_options.usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE;
#endif
    }
    else
      pipeline_options.usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
  }
  if (kernel_features & ccl::KERNEL_FEATURE_POINTCLOUD) {
    pipeline_options.usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
  }

  /* Keep track of whether motion blur is enabled, so to enable/disable motion in BVH builds
   * This is necessary since objects may be reported to have motion if the Vector pass is
   * active, but may still need to be rendered without motion blur if that isn't active as well. */
  dev->motion_blur = (kernel_features & ccl::KERNEL_FEATURE_OBJECT_MOTION) != 0;

  if (dev->motion_blur) {
    pipeline_options.usesMotionBlur = true;
    /* Motion blur can insert motion transforms into the traversal graph.
     * It is no longer a two-level graph then, so need to set flags to allow any configuration. */
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
  }

  { /* Load and compile PTX module with OptiX kernels. */
    // string ptx_data, ptx_filename = path_get((kernel_features & KERNEL_FEATURE_NODE_RAYTRACE) ?
    //                                             "lib/kernel_optix_shader_raytrace.ptx" :
    //                                             "lib/kernel_optix.ptx");
    // if (use_adaptive_compilation() || path_file_size(ptx_filename) == -1) {
    //  if (!getenv("OPTIX_ROOT_DIR")) {
    //    set_error(
    //        "Missing OPTIX_ROOT_DIR environment variable (which must be set with the path to "
    //        "the Optix SDK to be able to compile Optix kernels on demand).");
    //    return false;
    //  }
    //  ptx_filename = compile_kernel(
    //      kernel_features,
    //      (kernel_features & KERNEL_FEATURE_NODE_RAYTRACE) ? "kernel_shader_raytrace" : "kernel",
    //      "optix",
    //      true);
    //}
    // if (ptx_filename.empty() || !path_read_text(ptx_filename, ptx_data)) {
    //  set_error(string_printf("Failed to load OptiX kernel from '%s'", ptx_filename.c_str()));
    //  return false;
    //}

    const char *env_cubin = getenv("KERNEL_OPTIX_PTX");

    if (env_cubin == NULL) {
      printf("ASSERT: KERNEL_OPTIX_PTX is empty\n");
      exit(-1);
    }
    else {
      printf("PTX: %s\n", env_cubin);
    }

#ifdef WITH_CUDA_STAT
    const char *env_cubin_stat = getenv("KERNEL_OPTIX_STAT_PTX");

    if (env_cubin_stat == NULL) {
      printf("ASSERT: KERNEL_OPTIX_STAT_PTX is empty\n");
      exit(-1);
    }
    else {
      printf("PTX_STAT: %s\n", env_cubin_stat);
    }
#endif

    std::string ptx_data = readPTX(std::string(env_cubin));

    const OptixResult result = optixModuleCreateFromPTX(dev->context,
                                                        &module_options,
                                                        &pipeline_options,
                                                        ptx_data.data(),
                                                        ptx_data.size(),
                                                        nullptr,
                                                        0,
                                                        &dev->optix_module);
    if (result != OPTIX_SUCCESS) {
      printf("Failed to load OptiX kernel from '%s' (%s)",
             ptx_data.c_str(),
             optixGetErrorName(result));
      return false;
    }
  }

  /* Create program groups. */
  OptixProgramGroup groups[ccl::NUM_PROGRAM_GROUPS] = {};
  OptixProgramGroupDesc group_descs[ccl::NUM_PROGRAM_GROUPS] = {};
  OptixProgramGroupOptions group_options = {}; /* There are no options currently. */
  group_descs[ccl::PG_RGEN_INTERSECT_CLOSEST].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  group_descs[ccl::PG_RGEN_INTERSECT_CLOSEST].raygen.module = dev->optix_module;
  group_descs[ccl::PG_RGEN_INTERSECT_CLOSEST].raygen.entryFunctionName =
      "__raygen__kernel_optix_integrator_intersect_closest";
  group_descs[ccl::PG_RGEN_INTERSECT_SHADOW].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  group_descs[ccl::PG_RGEN_INTERSECT_SHADOW].raygen.module = dev->optix_module;
  group_descs[ccl::PG_RGEN_INTERSECT_SHADOW].raygen.entryFunctionName =
      "__raygen__kernel_optix_integrator_intersect_shadow";
  group_descs[ccl::PG_RGEN_INTERSECT_SUBSURFACE].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  group_descs[ccl::PG_RGEN_INTERSECT_SUBSURFACE].raygen.module = dev->optix_module;
  group_descs[ccl::PG_RGEN_INTERSECT_SUBSURFACE].raygen.entryFunctionName =
      "__raygen__kernel_optix_integrator_intersect_subsurface";
  group_descs[ccl::PG_RGEN_INTERSECT_VOLUME_STACK].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  group_descs[ccl::PG_RGEN_INTERSECT_VOLUME_STACK].raygen.module = dev->optix_module;
  group_descs[ccl::PG_RGEN_INTERSECT_VOLUME_STACK].raygen.entryFunctionName =
      "__raygen__kernel_optix_integrator_intersect_volume_stack";
  group_descs[ccl::PG_MISS].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  group_descs[ccl::PG_MISS].miss.module = dev->optix_module;
  group_descs[ccl::PG_MISS].miss.entryFunctionName = "__miss__kernel_optix_miss";
  group_descs[ccl::PG_HITD].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  group_descs[ccl::PG_HITD].hitgroup.moduleCH = dev->optix_module;
  group_descs[ccl::PG_HITD].hitgroup.entryFunctionNameCH = "__closesthit__kernel_optix_hit";
  group_descs[ccl::PG_HITD].hitgroup.moduleAH = dev->optix_module;
  group_descs[ccl::PG_HITD].hitgroup.entryFunctionNameAH =
      "__anyhit__kernel_optix_visibility_test";
  group_descs[ccl::PG_HITS].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  group_descs[ccl::PG_HITS].hitgroup.moduleAH = dev->optix_module;
  group_descs[ccl::PG_HITS].hitgroup.entryFunctionNameAH = "__anyhit__kernel_optix_shadow_all_hit";
  group_descs[ccl::PG_HITV].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  group_descs[ccl::PG_HITV].hitgroup.moduleCH = dev->optix_module;
  group_descs[ccl::PG_HITV].hitgroup.entryFunctionNameCH = "__closesthit__kernel_optix_hit";
  group_descs[ccl::PG_HITV].hitgroup.moduleAH = dev->optix_module;
  group_descs[ccl::PG_HITV].hitgroup.entryFunctionNameAH = "__anyhit__kernel_optix_volume_test";

  if (kernel_features & ccl::KERNEL_FEATURE_HAIR) {
    if (kernel_features & ccl::KERNEL_FEATURE_HAIR_THICK) {
      /* Built-in thick curve intersection. */
      OptixBuiltinISOptions builtin_options = {};
#if OPTIX_ABI_VERSION >= 55
      builtin_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM;
      builtin_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
                                   OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
      builtin_options.curveEndcapFlags = OPTIX_CURVE_ENDCAP_DEFAULT; /* Disable end-caps. */
#else
      builtin_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
#endif
      builtin_options.usesMotionBlur = false;

      optix_assert2(optixBuiltinISModuleGet(dev->context,
                                            &module_options,
                                            &pipeline_options,
                                            &builtin_options,
                                            &dev->builtin_modules[0]));

      group_descs[ccl::PG_HITD].hitgroup.moduleIS = dev->builtin_modules[0];
      group_descs[ccl::PG_HITD].hitgroup.entryFunctionNameIS = nullptr;
      group_descs[ccl::PG_HITS].hitgroup.moduleIS = dev->builtin_modules[0];
      group_descs[ccl::PG_HITS].hitgroup.entryFunctionNameIS = nullptr;

      if (dev->motion_blur) {
        builtin_options.usesMotionBlur = true;

        optix_assert2(optixBuiltinISModuleGet(dev->context,
                                              &module_options,
                                              &pipeline_options,
                                              &builtin_options,
                                              &dev->builtin_modules[1]));

        group_descs[ccl::PG_HITD_MOTION] = group_descs[ccl::PG_HITD];
        group_descs[ccl::PG_HITD_MOTION].hitgroup.moduleIS = dev->builtin_modules[1];
        group_descs[ccl::PG_HITS_MOTION] = group_descs[ccl::PG_HITS];
        group_descs[ccl::PG_HITS_MOTION].hitgroup.moduleIS = dev->builtin_modules[1];
      }
    }
    else {
      /* Custom ribbon intersection. */
      group_descs[ccl::PG_HITD].hitgroup.moduleIS = dev->optix_module;
      group_descs[ccl::PG_HITS].hitgroup.moduleIS = dev->optix_module;
      group_descs[ccl::PG_HITD].hitgroup.entryFunctionNameIS = "__intersection__curve_ribbon";
      group_descs[ccl::PG_HITS].hitgroup.entryFunctionNameIS = "__intersection__curve_ribbon";
    }
  }

  /* Pointclouds */
  if (kernel_features & ccl::KERNEL_FEATURE_POINTCLOUD) {
    group_descs[ccl::PG_HITD_POINTCLOUD] = group_descs[ccl::PG_HITD];
    group_descs[ccl::PG_HITD_POINTCLOUD].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    group_descs[ccl::PG_HITD_POINTCLOUD].hitgroup.moduleIS = dev->optix_module;
    group_descs[ccl::PG_HITD_POINTCLOUD].hitgroup.entryFunctionNameIS = "__intersection__point";
    group_descs[ccl::PG_HITS_POINTCLOUD] = group_descs[ccl::PG_HITS];
    group_descs[ccl::PG_HITS_POINTCLOUD].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    group_descs[ccl::PG_HITS_POINTCLOUD].hitgroup.moduleIS = dev->optix_module;
    group_descs[ccl::PG_HITS_POINTCLOUD].hitgroup.entryFunctionNameIS = "__intersection__point";
  }

  if (kernel_features & (ccl::KERNEL_FEATURE_SUBSURFACE | ccl::KERNEL_FEATURE_NODE_RAYTRACE)) {
    /* Add hit group for local intersections. */
    group_descs[ccl::PG_HITL].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    group_descs[ccl::PG_HITL].hitgroup.moduleAH = dev->optix_module;
    group_descs[ccl::PG_HITL].hitgroup.entryFunctionNameAH = "__anyhit__kernel_optix_local_hit";
  }

  /* Shader raytracing replaces some functions with direct callables. */
  if (kernel_features & ccl::KERNEL_FEATURE_NODE_RAYTRACE) {
    group_descs[ccl::PG_RGEN_SHADE_SURFACE_RAYTRACE].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    group_descs[ccl::PG_RGEN_SHADE_SURFACE_RAYTRACE].raygen.module = dev->optix_module;
    group_descs[ccl::PG_RGEN_SHADE_SURFACE_RAYTRACE].raygen.entryFunctionName =
        "__raygen__kernel_optix_integrator_shade_surface_raytrace";
    group_descs[ccl::PG_CALL_SVM_AO].kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    group_descs[ccl::PG_CALL_SVM_AO].callables.moduleDC = dev->optix_module;
    group_descs[ccl::PG_CALL_SVM_AO].callables.entryFunctionNameDC =
        "__direct_callable__svm_node_ao";
    group_descs[ccl::PG_CALL_SVM_BEVEL].kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    group_descs[ccl::PG_CALL_SVM_BEVEL].callables.moduleDC = dev->optix_module;
    group_descs[ccl::PG_CALL_SVM_BEVEL].callables.entryFunctionNameDC =
        "__direct_callable__svm_node_bevel";
  }

  /* MNEE. */
  if (kernel_features & ccl::KERNEL_FEATURE_MNEE) {
    group_descs[ccl::PG_RGEN_SHADE_SURFACE_MNEE].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    group_descs[ccl::PG_RGEN_SHADE_SURFACE_MNEE].raygen.module = dev->optix_module;
    group_descs[ccl::PG_RGEN_SHADE_SURFACE_MNEE].raygen.entryFunctionName =
        "__raygen__kernel_optix_integrator_shade_surface_mnee";
  }

  optix_assert2(optixProgramGroupCreate(
      dev->context, group_descs, ccl::NUM_PROGRAM_GROUPS, &group_options, nullptr, 0, groups));

  /* Get program stack sizes. */
  OptixStackSizes stack_size[ccl::NUM_PROGRAM_GROUPS] = {};
  /* Set up SBT, which in this case is used only to select between different programs. */
  dev->sbt_data.alloc(ccl::NUM_PROGRAM_GROUPS);
  memset(dev->sbt_data.host_pointer, 0, sizeof(ccl::SbtRecord) * ccl::NUM_PROGRAM_GROUPS);
  for (unsigned int i = 0; i < ccl::NUM_PROGRAM_GROUPS; ++i) {
    optix_assert2(optixSbtRecordPackHeader(groups[i], &dev->sbt_data[i]));
    optix_assert2(optixProgramGroupGetStackSize(groups[i], &stack_size[i]));
  }
  dev->sbt_data.copy_to_device(); /* Upload SBT to device. */

  /* Calculate maximum trace continuation stack size. */
  unsigned int trace_css = stack_size[ccl::PG_HITD].cssCH;
  /* This is based on the maximum of closest-hit and any-hit/intersection programs. */
  trace_css = std::max(trace_css, stack_size[ccl::PG_HITD].cssIS + stack_size[ccl::PG_HITD].cssAH);
  trace_css = std::max(trace_css, stack_size[ccl::PG_HITS].cssIS + stack_size[ccl::PG_HITS].cssAH);
  trace_css = std::max(trace_css, stack_size[ccl::PG_HITL].cssIS + stack_size[ccl::PG_HITL].cssAH);
  trace_css = std::max(trace_css, stack_size[ccl::PG_HITV].cssIS + stack_size[ccl::PG_HITV].cssAH);
  trace_css = std::max(trace_css, stack_size[ccl::PG_HITD_MOTION].cssIS + stack_size[ccl::PG_HITD_MOTION].cssAH);
  trace_css = std::max(trace_css, stack_size[ccl::PG_HITS_MOTION].cssIS + stack_size[ccl::PG_HITS_MOTION].cssAH);
  trace_css = std::max(trace_css,
                       stack_size[ccl::PG_HITD_POINTCLOUD].cssIS +
                           stack_size[ccl::PG_HITD_POINTCLOUD].cssAH);
  trace_css = std::max(trace_css,
                       stack_size[ccl::PG_HITS_POINTCLOUD].cssIS +
                           stack_size[ccl::PG_HITS_POINTCLOUD].cssAH);

  OptixPipelineLinkOptions link_options = {};
  link_options.maxTraceDepth = 1;

  if (g_optix_use_debug) {
    link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
  }
  else {
    link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
  }

  if (kernel_features & ccl::KERNEL_FEATURE_NODE_RAYTRACE) {
    /* Create shader raytracing pipeline. */
    std::vector<OptixProgramGroup> pipeline_groups;
    pipeline_groups.reserve(ccl::NUM_PROGRAM_GROUPS);
    pipeline_groups.push_back(groups[ccl::PG_RGEN_SHADE_SURFACE_RAYTRACE]);
    pipeline_groups.push_back(groups[ccl::PG_MISS]);
    pipeline_groups.push_back(groups[ccl::PG_HITD]);
    pipeline_groups.push_back(groups[ccl::PG_HITS]);
    pipeline_groups.push_back(groups[ccl::PG_HITL]);
    pipeline_groups.push_back(groups[ccl::PG_HITV]);
    if (dev->motion_blur) {
      pipeline_groups.push_back(groups[ccl::PG_HITD_MOTION]);
      pipeline_groups.push_back(groups[ccl::PG_HITS_MOTION]);
    }
    if (kernel_features & ccl::KERNEL_FEATURE_POINTCLOUD) {
      pipeline_groups.push_back(groups[ccl::PG_HITD_POINTCLOUD]);
      pipeline_groups.push_back(groups[ccl::PG_HITS_POINTCLOUD]);
    }
    pipeline_groups.push_back(groups[ccl::PG_CALL_SVM_AO]);
    pipeline_groups.push_back(groups[ccl::PG_CALL_SVM_BEVEL]);

    optix_assert2(optixPipelineCreate(dev->context,
                                      &pipeline_options,
                                      &link_options,
                                      pipeline_groups.data(),
                                      pipeline_groups.size(),
                                      nullptr,
                                      0,
                                      &dev->pipelines[ccl::PIP_SHADE_RAYTRACE]));

    /* Combine ray generation and trace continuation stack size. */
    const unsigned int css = stack_size[ccl::PG_RGEN_SHADE_SURFACE_RAYTRACE].cssRG +
                             link_options.maxTraceDepth * trace_css;
    const unsigned int dss = std::max(stack_size[ccl::PG_CALL_SVM_AO].dssDC,
                                      stack_size[ccl::PG_CALL_SVM_BEVEL].dssDC);

    /* Set stack size depending on pipeline options. */
    optix_assert2(optixPipelineSetStackSize(
        dev->pipelines[ccl::PIP_SHADE_RAYTRACE], 0, dss, css, dev->motion_blur ? 3 : 2));
  }

  if (kernel_features & ccl::KERNEL_FEATURE_MNEE) {
    /* Create MNEE pipeline. */
    std::vector<OptixProgramGroup> pipeline_groups;
    pipeline_groups.reserve(ccl::NUM_PROGRAM_GROUPS);
    pipeline_groups.push_back(groups[ccl::PG_RGEN_SHADE_SURFACE_MNEE]);
    pipeline_groups.push_back(groups[ccl::PG_MISS]);
    pipeline_groups.push_back(groups[ccl::PG_HITD]);
    pipeline_groups.push_back(groups[ccl::PG_HITS]);
    pipeline_groups.push_back(groups[ccl::PG_HITL]);
    pipeline_groups.push_back(groups[ccl::PG_HITV]);
    if (dev->motion_blur) {
      pipeline_groups.push_back(groups[ccl::PG_HITD_MOTION]);
      pipeline_groups.push_back(groups[ccl::PG_HITS_MOTION]);
    }
    if (kernel_features & ccl::KERNEL_FEATURE_POINTCLOUD) {
      pipeline_groups.push_back(groups[ccl::PG_HITD_POINTCLOUD]);
      pipeline_groups.push_back(groups[ccl::PG_HITS_POINTCLOUD]);
    }
    pipeline_groups.push_back(groups[ccl::PG_CALL_SVM_AO]);
    pipeline_groups.push_back(groups[ccl::PG_CALL_SVM_BEVEL]);

    optix_assert2(optixPipelineCreate(dev->context,
                                     &pipeline_options,
                                     &link_options,
                                     pipeline_groups.data(),
                                     pipeline_groups.size(),
                                     nullptr,
                                     0,
                                      &dev->pipelines[ccl::PIP_SHADE_MNEE]));

    /* Combine ray generation and trace continuation stack size. */
    const unsigned int css = stack_size[ccl::PG_RGEN_SHADE_SURFACE_MNEE].cssRG +
                             link_options.maxTraceDepth * trace_css;
    const unsigned int dss = 0;

    /* Set stack size depending on pipeline options. */
    optix_assert2(optixPipelineSetStackSize(
        dev->pipelines[ccl::PIP_SHADE_MNEE], 0, dss, css, dev->motion_blur ? 3 : 2));
  }

  { /* Create intersection-only pipeline. */
    std::vector<OptixProgramGroup> pipeline_groups;
    pipeline_groups.reserve(ccl::NUM_PROGRAM_GROUPS);
    pipeline_groups.push_back(groups[ccl::PG_RGEN_INTERSECT_CLOSEST]);
    pipeline_groups.push_back(groups[ccl::PG_RGEN_INTERSECT_SHADOW]);
    pipeline_groups.push_back(groups[ccl::PG_RGEN_INTERSECT_SUBSURFACE]);
    pipeline_groups.push_back(groups[ccl::PG_RGEN_INTERSECT_VOLUME_STACK]);
    pipeline_groups.push_back(groups[ccl::PG_MISS]);
    pipeline_groups.push_back(groups[ccl::PG_HITD]);
    pipeline_groups.push_back(groups[ccl::PG_HITS]);
    pipeline_groups.push_back(groups[ccl::PG_HITL]);
    pipeline_groups.push_back(groups[ccl::PG_HITV]);
    if (dev->motion_blur) {
      pipeline_groups.push_back(groups[ccl::PG_HITD_MOTION]);
      pipeline_groups.push_back(groups[ccl::PG_HITS_MOTION]);
    }
    if (kernel_features & ccl::KERNEL_FEATURE_POINTCLOUD) {
      pipeline_groups.push_back(groups[ccl::PG_HITD_POINTCLOUD]);
      pipeline_groups.push_back(groups[ccl::PG_HITS_POINTCLOUD]);
    }

    optix_assert2(optixPipelineCreate(dev->context,
                                      &pipeline_options,
                                      &link_options,
                                      pipeline_groups.data(),
                                      pipeline_groups.size(),
                                      nullptr,
                                      0,
                                      &dev->pipelines[ccl::PIP_INTERSECT]));

    /* Calculate continuation stack size based on the maximum of all ray generation stack sizes. */
    const unsigned int css =
        std::max(stack_size[ccl::PG_RGEN_INTERSECT_CLOSEST].cssRG,
                 std::max(stack_size[ccl::PG_RGEN_INTERSECT_SHADOW].cssRG,
                          std::max(stack_size[ccl::PG_RGEN_INTERSECT_SUBSURFACE].cssRG,
                                   stack_size[ccl::PG_RGEN_INTERSECT_VOLUME_STACK].cssRG))) +
        link_options.maxTraceDepth * trace_css;

    optix_assert2(optixPipelineSetStackSize(
        dev->pipelines[ccl::PIP_INTERSECT], 0, 0, css, dev->motion_blur ? 3 : 2));
  }

  /* Clean up program group objects. */
  for (unsigned int i = 0; i < ccl::NUM_PROGRAM_GROUPS; ++i) {
    optixProgramGroupDestroy(groups[i]);
  }

  return true;
}
//////////////

void frame_info(int current_frame, int current_frame_preview, int caching_enabled)
{
  cuda::frame_info(current_frame, current_frame_preview, caching_enabled);
}

void set_device(int device, int rank, int world_rank, int world_size)
{
  // cu_assert(cuInit(0));
#if 0
  int dev_count = 0;

  if (cuda::error(cudaGetDeviceCount(&dev_count))) {
    exit(-1);
  }

  if (device == -1) {

    const char *MPI_CUDA_VISIBLE_DEVICES = getenv("MPI_CUDA_VISIBLE_DEVICES");
    if (MPI_CUDA_VISIBLE_DEVICES != NULL) {
      int dev_start = util_get_int_from_env_array(MPI_CUDA_VISIBLE_DEVICES, rank * 2);
      dev_count = util_get_int_from_env_array(MPI_CUDA_VISIBLE_DEVICES, rank * 2 + 1);
      printf("rank: %d, dev_start: %d, dev_count:%d\n", rank, dev_start, dev_count);

      ccl::cuda_devices.resize(dev_count);

      for (int id = 0; id < ccl::cuda_devices.size(); id++) {
        /* Setup device and context. */
        ccl::cuda_devices[id].cuDevice = id + dev_start;
      }
    }
    else {
      ccl::cuda_devices.resize(dev_count);

      for (int id = 0; id < ccl::cuda_devices.size(); id++) {
        /* Setup device and context. */
        ccl::cuda_devices[id].cuDevice = id;
      }
    }
  }
  else {
    ccl::cuda_devices.resize(1);

    int id = 0;
    ccl::cuda_devices[id].cuDevice = device % dev_count;
  }

  for (int id = 0; id < ccl::cuda_devices.size(); id++) {
    ccl::CUDAContextScope scope(id);

    for (int s = 0; s < STREAM_COUNT; s++) {
      // printf("dev: %d, stream:%d\n", id, s);
      cuda::optix_assert2(cudaStreamCreate(&scope.get().stream[s]));
    }

    for (int s = 0; s < EVENT_COUNT; s++) {
      cuda::optix_assert2(cudaEventCreate(&scope.get().event[s]));
    }

    //cu_assert(cuModuleLoad(&scope.get().cuModule, env_cubin));

#  ifdef WITH_CUDA_STAT
    //cu_assert(cuModuleLoad(&scope.get().cuModuleStat, env_cubin_stat));
#  endif

    // scope.get().cuFilterModule = 0;
    scope.get().path_stat_is_done = 0;
    scope.get().texture_info_dmem = 0;
    // scope.get().min_num_active_paths_ = 0;
    // scope.get().max_active_path_index_ = 0;
    // scope.get().texture_info_mem_size = 0;
    // scope.get().d_work_tiles = 0;
    memset(scope.get().running_time, 0, sizeof(scope.get().running_time));
    // scope.get().time_old = 0;
    scope.get().time_count = 0;
    scope.get().wtile_h = 0;
    // scope.get().num_threads_per_block = 0;
    // scope.get().kerneldata = 0;
  }
  printf("\n========CUDA devices: %zu\n", ccl::cuda_devices.size());
#endif
  cuda::set_device(device, rank, world_rank, world_size);

  for (int id = 0; id < ccl::cuda_devices.size(); id++) {
    ccl::CUDAContextScope scope(id);
    ccl::OptiXDevice *dev = (ccl::OptiXDevice *)&scope.get();
    dev->init();

    load_kernels(dev, 0);
  }
}

int get_cpu_threads()
{
  return cuda::get_cpu_threads();
}

bool check_unimem(const char *_name)
{
  return cuda::check_unimem(_name);
}

void init_execution(int has_shadow_catcher_,
                    int max_shaders_,
                    int pass_stride_,
                    unsigned int kernel_features_,
                    unsigned int volume_stack_size_,
                    bool init)
{
  cuda::init_execution(
      has_shadow_catcher_, max_shaders_, pass_stride_, kernel_features_, volume_stack_size_, init);
}

/////////////////////////////////////////////////////
bool build_bvh_internal(ccl::CUDAContextScope &scope,
                        OptixBuildOperation operation,
                        OptixBuildInput *build_input,
                        uint16_t num_motion_steps)
{
  ccl::OptiXDevice *dev = (ccl::OptiXDevice *)&scope.get();
  // const bool use_fast_trace_bvh = (bvh->params.bvh_type == BVH_TYPE_STATIC);
  const bool use_fast_trace_bvh = true;

  /* Compute memory usage. */
  OptixAccelBufferSizes sizes = {};
  OptixAccelBuildOptions options = {};
  options.operation = operation;
  if (use_fast_trace_bvh) {
    // VLOG(2) << "Using fast to trace OptiX BVH";
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  }
  else {
    // VLOG(2) << "Using fast to update OptiX BVH";
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
  }

  options.motionOptions.numKeys = num_motion_steps;
  options.motionOptions.flags = OPTIX_MOTION_FLAG_START_VANISH | OPTIX_MOTION_FLAG_END_VANISH;
  options.motionOptions.timeBegin = 0.0f;
  options.motionOptions.timeEnd = 1.0f;

  optix_assert2(optixAccelComputeMemoryUsage(dev->context, &options, build_input, 1, &sizes));

  /* Allocate required output buffers. */
  // device_only_memory<char> temp_mem(this, "optix temp as build mem");
  // temp_mem.alloc_to_device(align_up(sizes.tempSizeInBytes, 8) + 8);
  // if (!temp_mem.device_pointer) {
  //  /* Make sure temporary memory allocation succeeded. */
  //  return false;
  //}
  CUdeviceptr temp_mem_d = NULL;
  cu_assert(cuMemAlloc(&temp_mem_d, ccl::align_up(sizes.tempSizeInBytes, 8) + 8));

  // device_only_memory<char> &out_data = *bvh->as_data;
  // if (operation == OPTIX_BUILD_OPERATION_BUILD) {
  //  assert(out_data.device == this);
  //  out_data.alloc_to_device(sizes.outputSizeInBytes);
  //  if (!out_data.device_pointer) {
  //    return false;
  //  }
  //}
  // else {
  //  assert(out_data.device_pointer && out_data.device_size >= sizes.outputSizeInBytes);
  //}

  CUdeviceptr out_data_d = NULL;
  cu_assert(cuMemAlloc(&out_data_d, sizes.outputSizeInBytes));

  /* Finally build the acceleration structure. */
  OptixAccelEmitDesc compacted_size_prop = {};
  compacted_size_prop.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  /* A tiny space was allocated for this property at the end of the temporary buffer above.
   * Make sure this pointer is 8-byte aligned. */
  compacted_size_prop.result = ccl::align_up(temp_mem_d + sizes.tempSizeInBytes, 8);

  OptixTraversableHandle out_handle = 0;
  optix_assert2(optixAccelBuild(dev->context,
                                NULL,
                                &options,
                                build_input,
                                1,
                                temp_mem_d,
                                sizes.tempSizeInBytes,
                                out_data_d,
                                sizes.outputSizeInBytes,
                                &out_handle,
                                use_fast_trace_bvh ? &compacted_size_prop : NULL,
                                use_fast_trace_bvh ? 1 : 0));
  dev->tlas_handle = static_cast<uint64_t>(out_handle);

  /* Wait for all operations to finish. */
  cu_assert(cuStreamSynchronize(NULL));

  /* Compact acceleration structure to save memory (do not do this in viewport for faster builds).
   */
  if (use_fast_trace_bvh) {
    uint64_t compacted_size = sizes.outputSizeInBytes;
    cu_assert(cuMemcpyDtoH(&compacted_size, compacted_size_prop.result, sizeof(compacted_size)));

    /* Temporary memory is no longer needed, so free it now to make space. */
    // temp_mem.free();
    cu_assert(cuMemFree(temp_mem_d));

    /* There is no point compacting if the size does not change. */
    if (compacted_size < sizes.outputSizeInBytes) {
      // device_only_memory<char> compacted_data(this, "optix compacted as");
      // compacted_data.alloc_to_device(compacted_size);
      // if (!compacted_data.device_pointer)
      //  /* Do not compact if memory allocation for compacted acceleration structure fails.
      //   * Can just use the uncompacted one then, so succeed here regardless. */
      //  return !have_error();

      CUdeviceptr compacted_data_d = NULL;
      cu_assert(cuMemAlloc(&compacted_data_d, compacted_size));

      optix_assert2(optixAccelCompact(
          dev->context, NULL, out_handle, compacted_data_d, compacted_size, &out_handle));

      dev->tlas_handle = static_cast<uint64_t>(out_handle);

      /* Wait for compaction to finish. */
      cu_assert(cuStreamSynchronize(NULL));

      // std::swap(out_data.device_size, compacted_data.device_size);
      // std::swap(out_data.device_pointer, compacted_data.device_pointer);
    }
  }

  // return !have_error();
  return true;
}

void build_bvh(int operation,
               char *build_input,
               DEVICE_PTR build_input_orig,
               size_t build_size,
               int num_motion_steps)
{
  for (int id = 0; id < ccl::cuda_devices.size(); id++) {
    ccl::CUDAContextScope scope(id);
    ccl::OptiXDevice *dev = (ccl::OptiXDevice *)&scope.get();

    OptixBuildInput obi_instance = *(OptixBuildInput *)build_input;
    OptixBuildInput *obi = &obi_instance;

    if (obi->type == OPTIX_BUILD_INPUT_TYPE_CURVES) {
      unsigned int build_flags = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
      std::vector<DEVICE_PTR> width_ptrs;
      std::vector<DEVICE_PTR> vertex_ptrs;
      width_ptrs.reserve(num_motion_steps);
      vertex_ptrs.reserve(num_motion_steps);
      for (size_t step = 0; step < num_motion_steps; ++step) {
        const DEVICE_PTR base_ptr =
            scope.get()
                .cuda_mem_map[get_ptr_map((DEVICE_PTR)obi->curveArray.vertexBuffers)]
                .mem.device_pointer +
            step * obi->curveArray.numVertices * sizeof(float4);
        width_ptrs.push_back(base_ptr + 3 * sizeof(float)); /* Offset by vertex size. */
        vertex_ptrs.push_back(base_ptr);
      }

      //#if OPTIX_ABI_VERSION >= 55
      //      obi->curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM;
      //#else
      //      obi->curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
      //#endif
      //      obi->curveArray.numPrimitives = num_segments;
      obi->curveArray.vertexBuffers = (CUdeviceptr *)vertex_ptrs.data();

      //      obi->curveArray.numVertices = num_vertices;
      //    obi->curveArray.vertexStrideInBytes = sizeof(float4);
      obi->curveArray.widthBuffers = (CUdeviceptr *)width_ptrs.data();

      //  obi->curveArray.widthStrideInBytes = sizeof(float4);
      // obi->curveArray.indexBuffer = (CUdeviceptr)index_data.device_pointer;
      obi->curveArray.indexBuffer =
          (CUdeviceptr)scope.get()
              .cuda_mem_map[get_ptr_map((DEVICE_PTR)obi->curveArray.indexBuffer)]
              .mem.device_pointer;

      // obi->curveArray.indexStrideInBytes = sizeof(int);
      obi->curveArray.flag = build_flags;
      // obi->curveArray.primitiveIndexOffset = hair->curve_segment_offset;

      build_bvh_internal(scope, (OptixBuildOperation)operation, obi, num_motion_steps);
      dev->map_instances_buildinout[build_input_orig] = dev->tlas_handle;
    }

    if (obi->type == OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES) {
      unsigned int build_flags = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
      std::vector<DEVICE_PTR> aabb_ptrs;
      aabb_ptrs.reserve(num_motion_steps);
      for (size_t step = 0; step < num_motion_steps; ++step) {
        aabb_ptrs.push_back(
            scope.get()
                .cuda_mem_map[get_ptr_map((DEVICE_PTR)obi->customPrimitiveArray.aabbBuffers)]
                .mem.device_pointer +
            step * obi->customPrimitiveArray.numPrimitives * sizeof(OptixAabb));
      }

      /* Disable visibility test any-hit program, since it is already checked during
       * intersection. Those trace calls that require any-hit can force it with a ray flag. */
      build_flags |= OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

      obi->customPrimitiveArray.aabbBuffers = (CUdeviceptr *)aabb_ptrs.data();
      // obi->customPrimitiveArray.numPrimitives = num_segments;
      // obi->customPrimitiveArray.strideInBytes = sizeof(OptixAabb);
      obi->customPrimitiveArray.flags = &build_flags;
      // obi->customPrimitiveArray.numSbtRecords = 1;
      // obi->customPrimitiveArray.primitiveIndexOffset = hair->curve_segment_offset;

      build_bvh_internal(scope, (OptixBuildOperation)operation, obi, num_motion_steps);
      dev->map_instances_buildinout[build_input_orig] = dev->tlas_handle;
    }

    if (obi->type == OPTIX_BUILD_INPUT_TYPE_TRIANGLES) {
      unsigned int build_flags = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;

      std::vector<DEVICE_PTR> vertex_ptrs;
      vertex_ptrs.reserve(num_motion_steps);
      for (size_t step = 0; step < num_motion_steps; ++step) {
        vertex_ptrs.push_back(
            scope.get()
                .cuda_mem_map[get_ptr_map((DEVICE_PTR)obi->triangleArray.vertexBuffers)]
                .mem.device_pointer +
            obi->triangleArray.numVertices * step * sizeof(float3));
      }

      obi->triangleArray.vertexBuffers = (CUdeviceptr *)vertex_ptrs.data();
      // obi->triangleArray.numVertices = num_verts;
      // obi->triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
      // obi->triangleArray.vertexStrideInBytes = sizeof(float4);

      // obi->triangleArray.indexBuffer = index_data.device_pointer;

      obi->triangleArray.indexBuffer =
          (CUdeviceptr)scope.get()
              .cuda_mem_map[get_ptr_map((DEVICE_PTR)obi->triangleArray.indexBuffer)]
              .mem.device_pointer;

      // obi->triangleArray.numIndexTriplets = mesh->num_triangles();
      // obi->triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
      // obi->triangleArray.indexStrideInBytes = 3 * sizeof(int);
      obi->triangleArray.flags = &build_flags;
      /* The SBT does not store per primitive data since Cycles already allocates separate
       * buffers for that purpose. OptiX does not allow this to be zero though, so just pass in
       * one and rely on that having the same meaning in this case. */
      // obi->triangleArray.numSbtRecords = 1;
      // obi->triangleArray.primitiveIndexOffset = mesh->prim_offset;

      build_bvh_internal(scope, (OptixBuildOperation)operation, obi, num_motion_steps);
      dev->map_instances_buildinout[build_input_orig] = dev->tlas_handle;
    }

    if (obi->type == OPTIX_BUILD_INPUT_TYPE_INSTANCES) {
      // obi->instanceArray.instances = instances.device_pointer;
      DEVICE_PTR instance_device_pointer = get_ptr_map((DEVICE_PTR)obi->instanceArray.instances);
      obi->instanceArray.instances =
          (CUdeviceptr)scope.get().cuda_mem_map[instance_device_pointer].mem.device_pointer;

      size_t device_size = scope.get().cuda_mem_map[instance_device_pointer].mem.device_size;

      std::vector<char> instances(device_size);
      // DEVICE_PTR g_optix_instance_d;
      cu_assert(cuMemcpyDtoH(instances.data(), obi->instanceArray.instances, device_size));
      // obi->instanceArray.numInstances = num_instances;
      for (int i = 0; i < obi->instanceArray.numInstances; i++) {
        OptixInstance &instance = ((OptixInstance *)instances.data())[i];
        instance.traversableHandle = dev->map_instances_buildinout[instance.traversableHandle];
      }

      cu_assert(cuMemcpyHtoD(obi->instanceArray.instances, instances.data(), device_size));

      build_bvh_internal(scope, (OptixBuildOperation)operation, obi, num_motion_steps);
    }
  }
}
}  // namespace optix
}  // namespace kernel
}  // namespace cyclesphi

//#endif
