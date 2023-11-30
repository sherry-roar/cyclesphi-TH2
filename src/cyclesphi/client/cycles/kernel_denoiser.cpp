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

#include "kernel_denoiser.h"

#include "client_api.h"
#include "kernel_queue.h"
#include "kernel/kernel_types.h"

#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include <cuda_runtime_api.h>

//#include <assert.h>
#include <cstdio>
#include <vector>

//#if 0
//// TODO(pmours): Disable this once drivers have native support
//#  define OPTIX_DENOISER_NO_PIXEL_STRIDE 1
//
//#  define check_result_cuda(stmt) \
//    { \
//      CUresult res = stmt; \
//      if (res != CUDA_SUCCESS) { \
//        const char *name; \
//        cuGetErrorName(res, &name); \
//        printf("OptiX CUDA error %s in %s, line %d", name, #stmt, __LINE__); \
//        return; \
//      } \
//    } \
//    (void)0
//#  define check_result_cuda_ret(stmt) \
//    { \
//      CUresult res = stmt; \
//      if (res != CUDA_SUCCESS) { \
//        const char *name; \
//        cuGetErrorName(res, &name); \
//        printf("OptiX CUDA error %s in %s, line %d", name, #stmt, __LINE__); \
//        return false; \
//      } \
//    } \
//    (void)0
//
//#  define check_result_optix(stmt) \
//    { \
//      enum OptixResult res = stmt; \
//      if (res != OPTIX_SUCCESS) { \
//        const char *name = optixGetErrorName(res); \
//        printf("OptiX error %s in %s, line %d", name, #stmt, __LINE__); \
//        return; \
//      } \
//    } \
//    (void)0
//#  define check_result_optix_ret(stmt) \
//    { \
//      enum OptixResult res = stmt; \
//      if (res != OPTIX_SUCCESS) { \
//        const char *name = optixGetErrorName(res); \
//        printf("OptiX error %s in %s, line %d", name, #stmt, __LINE__); \
//        return false; \
//      } \
//    } \
//    (void)0
//
//#  define launch_filter_kernel(func_name, w, h, args) \
//    { \
//      CUfunction func; \
//      check_result_cuda_ret(cuModuleGetFunction(&func, cuFilterModule, func_name)); \
//      check_result_cuda_ret(cuFuncSetCacheConfig(func, CU_FUNC_CACHE_PREFER_L1)); \
//      int threads; \
//      check_result_cuda_ret( \
//          cuFuncGetAttribute(&threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, func)); \
//      threads = (int)sqrt((float)threads); \
//      int xblocks = ((w) + threads - 1) / threads; \
//      int yblocks = ((h) + threads - 1) / threads; \
//      check_result_cuda_ret( \
//          cuLaunchKernel(func, xblocks, yblocks, 1, threads, threads, 1, 0, 0, args, 0)); \
//    } \
//    (void)0
//
//#endif

//#include <optix_stubs.h>

/* Utility for checking return values of OptiX function calls. */
#define optix_assert(stmt) \
  { \
    OptixResult result = stmt; \
    if (result != OPTIX_SUCCESS) { \
      const char *name = optixGetErrorName(result); \
      printf("OptiX error %s in %s (%s:%d)", name, #stmt, __FILE__, __LINE__); \
    } \
  } \
  (void)0

namespace cyclesphi {
namespace kernel {

 struct DeviceImage {
  // int data_elements;
  size_t data_size;
  // size_t device_size;
  size_t data_width;
  size_t data_height;
  // size_t data_depth;

  CUdeviceptr device_pointer;

  void alloc_to_device(size_t num, bool shrink_to_fit = true)
  {
    size_t new_size = num;
    bool reallocate;

    if (shrink_to_fit) {
      reallocate = (data_size != new_size);
    }
    else {
      reallocate = (data_size < new_size);
    }

    if (reallocate) {
      cuMemFree(device_pointer);
      data_size = new_size;
      cuMemAlloc(&device_pointer, num);
    }
  }

  void free()
  {
    cuMemFree(device_pointer);
  }
};

 struct CustomOptixDenoiser {
  OptixDenoiser optix_denoiser;
  OptixDeviceContext context;
  DeviceImage state;
  int input_passes;
  size_t scratch_offset;
  size_t scratch_size;
  bool is_configured;

  int configured_size_x;
  int configured_size_y;

  CustomOptixDenoiser()
  {
    optix_denoiser = NULL;
    context = NULL;
    input_passes = 0;
    is_configured = false;

    configured_size_x = 0;
    configured_size_y = 0;
  }
};

bool device_optix_denoise_run(
    char *den_ptr,
    int task_buffer_params_width,
    int task_buffer_params_height,
    DEVICE_PTR d_input_rgb,
    CUstream queue_stream
    /*OptiXDeviceQueue *queue, const DeviceDenoiseTask &task, const device_ptr d_input_rgb*/)
{
  CustomOptixDenoiser &denoiser_ = *(CustomOptixDenoiser *)den_ptr;

  const int pixel_stride = 3 * sizeof(float);
  const int input_stride = task_buffer_params_width * pixel_stride;

  /* Set up input and output layer information. */
  OptixImage2D input_layers[3] = {};
  OptixImage2D output_layers[1] = {};

  for (int i = 0; i < 3; ++i) {
    input_layers[i].data = d_input_rgb +
                           ((size_t)task_buffer_params_width * (size_t)task_buffer_params_height *
                            (size_t)pixel_stride * (size_t)i);
    input_layers[i].width = task_buffer_params_width;
    input_layers[i].height = task_buffer_params_height;
    input_layers[i].rowStrideInBytes = input_stride;
    input_layers[i].pixelStrideInBytes = pixel_stride;
    input_layers[i].format = OPTIX_PIXEL_FORMAT_FLOAT3;
  }

  output_layers[0].data = d_input_rgb;
  output_layers[0].width = task_buffer_params_width;
  output_layers[0].height = task_buffer_params_height;
  output_layers[0].rowStrideInBytes = input_stride;
  output_layers[0].pixelStrideInBytes = pixel_stride;
  output_layers[0].format = OPTIX_PIXEL_FORMAT_FLOAT3;

  /* Finally run denonising. */
  OptixDenoiserParams params = {}; /* All parameters are disabled/zero. */
//#if OPTIX_ABI_VERSION >= 47
  OptixDenoiserLayer image_layers = {};
  image_layers.input = input_layers[0];
  image_layers.output = output_layers[0];

  OptixDenoiserGuideLayer guide_layers = {};
  guide_layers.albedo = input_layers[1];
  guide_layers.normal = input_layers[2];

  optix_assert(optixDenoiserInvoke(denoiser_.optix_denoiser,
                                   queue_stream,
                                   &params,
                                   denoiser_.state.device_pointer,
                                   denoiser_.scratch_offset,
                                   &guide_layers,
                                   &image_layers,
                                   1,
                                   0,
                                   0,
                                   denoiser_.state.device_pointer + denoiser_.scratch_offset,
                                   denoiser_.scratch_size));
//#else
//  const int input_passes = denoise_buffer_num_passes(task.params);
//
//  optix_assert(optixDenoiserInvoke(denoiser_.optix_denoiser,
//                                   queue->stream(),
//                                   &params,
//                                   denoiser_.state.device_pointer,
//                                   denoiser_.scratch_offset,
//                                   input_layers,
//                                   input_passes,
//                                   0,
//                                   0,
//                                   output_layers,
//                                   denoiser_.state.device_pointer + denoiser_.scratch_offset,
//                                   denoiser_.scratch_size));
//#endif

  return true;
}


//
//struct XY {
//  int x, y;
//};
//


bool device_optix_init()
{
  if (g_optixFunctionTable.optixDeviceContextCreate != NULL)
    return true;  // Already initialized function table

  // Need to initialize CUDA as well
  // if (!device_cuda_init())
  //  return false;

  const OptixResult result = optixInit();

  if (result == OPTIX_ERROR_UNSUPPORTED_ABI_VERSION) {
    printf("OptiX initialization failed because driver does not support ABI version\n");
    return false;
  }
  else if (result != OPTIX_SUCCESS) {
    printf("OptiX initialization failed with error code: %d \n", result);
    return false;
  }

  // Loaded OptiX successfully!
  return true;
}

char *device_optix_create()
{
#pragma omp critical
  {
    device_optix_init();
  }

  return (char *)new CustomOptixDenoiser();
}

void device_optix_destroy(char *den_ptr)
{
  CustomOptixDenoiser *cod = (CustomOptixDenoiser *)den_ptr;
  if (cod->optix_denoiser != NULL) {
    optixDenoiserDestroy(cod->optix_denoiser);
  }

  if (cod->context != NULL) {
    optixDeviceContextDestroy(cod->context);
  }

  delete cod;
}
//
///*DeviceTask &task, RenderTile &rtile*/
//bool denoise_optix(
//    char *den_ptr, cudaStream_t stream, char *input_rgb_device_pointer, int rtile_w, int rtile_h)
//{
//  CustomOptixDenoiser *cod = (CustomOptixDenoiser *)den_ptr;
//
//  int task_denoising_optix_input_passes = 1;
//  XY rect_size;
//  rect_size.x = rtile_w;
//  rect_size.y = rtile_h;
//
//  XY overlap_offset;  // = make_int2(rtile.x - rect.x, rtile.y - rect.y);
//  overlap_offset.x = 0;
//  overlap_offset.y = 0;
//
//  CUdeviceptr input_ptr = (CUdeviceptr)input_rgb_device_pointer;
//  int pixel_stride = 4 * sizeof(float);
//  int input_stride = rect_size.x * pixel_stride;
//
//  bool recreate_denoiser = false;
//
//  recreate_denoiser = (cod->denoiser == NULL) ||
//                      (task_denoising_optix_input_passes != cod->denoiser_input_passes);
//  if (recreate_denoiser) {
//    // Destroy existing handle before creating new one
//    if (cod->denoiser != NULL) {
//      optixDenoiserDestroy(cod->denoiser);
//    }
//
//    if (cod->context != NULL) {
//      optixDeviceContextDestroy(cod->context);
//    }
//
//    if (!device_optix_init())
//      return false;
//
//    OptixDeviceContextOptions options = {};
//    check_result_optix_ret(optixDeviceContextCreate(0, &options, &cod->context));
//
//    // Create OptiX cod->denoiser handle on demand when it is first used
//    OptixDenoiserOptions denoiser_options;
//    // assert(task_denoising_optix_input_passes >= 1 && task_denoising_optix_input_passes <= 3);
//    denoiser_options.inputKind = static_cast<OptixDenoiserInputKind>(
//        OPTIX_DENOISER_INPUT_RGB + (task_denoising_optix_input_passes - 1));
//    // denoiser_options.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
//#if OPTIX_ABI_VERSION < 28
//    denoiser_options.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
//#endif
//    check_result_optix_ret(optixDenoiserCreate(cod->context, &denoiser_options, &cod->denoiser));
//    check_result_optix_ret(
//        optixDenoiserSetModel(cod->denoiser, OPTIX_DENOISER_MODEL_KIND_HDR, NULL, 0));
//
//    // OptiX cod->denoiser handle was created with the requested number of input passes
//    cod->denoiser_input_passes = task_denoising_optix_input_passes;
//  }
//
//  OptixDenoiserSizes sizes = {};
//  check_result_optix_ret(
//      optixDenoiserComputeMemoryResources(cod->denoiser, rect_size.x, rect_size.y, &sizes));
//
//#if OPTIX_ABI_VERSION < 28
//  const size_t scratch_size = sizes.recommendedScratchSizeInBytes;
//#else
//  const size_t scratch_size = sizes.withOverlapScratchSizeInBytes;
//#endif
//
//  const size_t scratch_offset = sizes.stateSizeInBytes;
//
//  // Allocate cod->denoiser state if tile size has changed since last setup
//  if (recreate_denoiser || (cod->denoiser_state.data_width != rect_size.x ||
//                            cod->denoiser_state.data_height != rect_size.y)) {
//
//    cod->denoiser_state.alloc_to_device(scratch_offset + scratch_size);
//
//    // Initialize cod->denoiser state for the current tile size
//    check_result_optix_ret(optixDenoiserSetup(cod->denoiser,
//                                              (CUstream)stream,
//                                              rect_size.x,
//                                              rect_size.y,
//                                              cod->denoiser_state.device_pointer,
//                                              scratch_offset,
//                                              cod->denoiser_state.device_pointer + scratch_offset,
//                                              scratch_size));
//
//    cod->denoiser_state.data_width = rect_size.x;
//    cod->denoiser_state.data_height = rect_size.y;
//  }
//
//  // Set up input and output layer information
//  OptixImage2D input_layers[1] = {};
//  OptixImage2D output_layers[1] = {};
//
//  // for (int i = 0; i < 3; ++i) {
//  //  input_layers[i].data = input_ptr + (rect_size.x * rect_size.y * pixel_stride * i);
//  //  input_layers[i].width = rect_size.x;
//  //  input_layers[i].height = rect_size.y;
//  //  input_layers[i].rowStrideInBytes = input_stride;
//  //  input_layers[i].pixelStrideInBytes = pixel_stride;
//  //  input_layers[i].format = OPTIX_PIXEL_FORMAT_UCHAR4;
//  //}
//
//  input_layers[0].data = input_ptr;
//  input_layers[0].width = rect_size.x;
//  input_layers[0].height = rect_size.y;
//  input_layers[0].rowStrideInBytes = input_stride;
//  input_layers[0].pixelStrideInBytes = pixel_stride;
//  input_layers[0].format = OPTIX_PIXEL_FORMAT_UCHAR4;  // OPTIX_PIXEL_FORMAT_FLOAT4;
//
//  output_layers[0].data = input_ptr;
//  output_layers[0].width = rect_size.x;
//  output_layers[0].height = rect_size.y;
//  output_layers[0].rowStrideInBytes = input_stride;
//  output_layers[0].pixelStrideInBytes = pixel_stride;
//  output_layers[0].format = OPTIX_PIXEL_FORMAT_UCHAR4;  // OPTIX_PIXEL_FORMAT_FLOAT4;
//
//  // Finally run denonising
//  OptixDenoiserParams params = {};  // All parameters are disabled/zero
//  check_result_optix_ret(optixDenoiserInvoke(cod->denoiser,
//                                             (CUstream)stream,
//                                             &params,
//                                             cod->denoiser_state.device_pointer,
//                                             scratch_offset,
//                                             input_layers,
//                                             task_denoising_optix_input_passes,
//                                             overlap_offset.x,
//                                             overlap_offset.y,
//                                             output_layers,
//                                             cod->denoiser_state.device_pointer + scratch_offset,
//                                             scratch_size));
//
//  // check_result_cuda_ret(cuStreamSynchronize(stream));
//
//  return true;
//}

/* --------------------------------------------------------------------
 * Buffer denoising.
 */

/* Calculate number of passes used by the denoiser. */
static int denoise_buffer_num_passes(int params_use_pass_albedo, int params_use_pass_normal)
{
  int num_passes = 1;

  if (params_use_pass_albedo) {
    num_passes += 1;

    if (params_use_pass_normal) {
      num_passes += 1;
    }
  }

  return num_passes;
}

/* Calculate number of floats per pixel for the input buffer used by the OptiX. */
static int denoise_buffer_pass_stride(int params_use_pass_albedo, int params_use_pass_normal)
{
  return denoise_buffer_num_passes(params_use_pass_albedo, params_use_pass_normal) * 3;
}


bool device_optix_denoise_filter_convert_to_rgb(CUDAContextScope &scope,
                                                cudaStream_t cuda_stream, 
                                                CUDADeviceQueue *queue,
                                                int task_buffer_params_x,
                                                int task_buffer_params_y,
                                                int task_buffer_params_width,
                                                int task_buffer_params_height,
                                                int task_buffer_params_offset,
                                                int task_buffer_params_stride,
                                                int task_buffer_params_pass_stride,
                                                int task_num_samples,
                                                float *d_input_rgb,
                                                float *task_buffer)
{
  const int work_size = task_buffer_params_width * task_buffer_params_height;

  //const int pass_offset[3] = {task.buffer_params.get_pass_offset(PASS_DENOISING_COLOR),
  //                            task.buffer_params.get_pass_offset(PASS_DENOISING_ALBEDO),
  //                            task.buffer_params.get_pass_offset(PASS_DENOISING_NORMAL)};

  const int pass_offset[3] = {0, PASS_UNUSED, PASS_UNUSED};

  const int input_passes = denoise_buffer_num_passes(0, 0);

  const int pass_sample_count = PASS_UNUSED;  // sample_count;  // task.buffer_params.get_pass_offset(PASS_SAMPLE_COUNT);

  void *args[] = {&d_input_rgb,
                  &task_buffer,
                  const_cast<int *>(&task_buffer_params_x),
                  const_cast<int *>(&task_buffer_params_y),
                  const_cast<int *>(&task_buffer_params_width),
                  const_cast<int *>(&task_buffer_params_height),
                  const_cast<int *>(&task_buffer_params_offset),
                  const_cast<int *>(&task_buffer_params_stride),
                  const_cast<int *>(&task_buffer_params_pass_stride),
                  const_cast<int *>(pass_offset),
                  const_cast<int *>(&input_passes),
                  const_cast<int *>(&task_num_samples),
                  const_cast<int *>(&pass_sample_count)};

  return queue->enqueue(
      scope, cuda_stream, ccl::DEVICE_KERNEL_FILTER_CONVERT_TO_RGB, work_size, args);
}

bool device_optix_denoise_filter_convert_from_rgb(CUDAContextScope &scope,
                                                  cudaStream_t cuda_stream,
                                                  CUDADeviceQueue *queue,
                                                  int task_buffer_params_x,
                                                  int task_buffer_params_y,
                                                  int task_buffer_params_width,
                                                  int task_buffer_params_height,
                                                  int task_buffer_params_offset,
                                                  int task_buffer_params_stride,
                                                  int task_buffer_params_pass_stride,
                                                  int task_num_samples,
                                                  float *d_input_rgb,
                                                  char *task_buffer)
{
  const int work_size = task_buffer_params_width * task_buffer_params_height;

  const int pass_sample_count = PASS_UNUSED;  // task.buffer_params.get_pass_offset(PASS_SAMPLE_COUNT);

  void *args[] = {&d_input_rgb,
                  &task_buffer,
                  const_cast<int *>(&task_buffer_params_x),
                  const_cast<int *>(&task_buffer_params_y),
                  const_cast<int *>(&task_buffer_params_width),
                  const_cast<int *>(&task_buffer_params_height),
                  const_cast<int *>(&task_buffer_params_offset),
                  const_cast<int *>(&task_buffer_params_stride),
                  const_cast<int *>(&task_buffer_params_pass_stride),
                  const_cast<int *>(&task_num_samples),
                  const_cast<int *>(&pass_sample_count)};

  return queue->enqueue(
      scope, cuda_stream, ccl::DEVICE_KERNEL_FILTER_CONVERT_FROM_RGB, work_size, args);
}

bool device_optix_denoise_create_if_needed(char *den_ptr)
{
  CustomOptixDenoiser &denoiser_ = *(CustomOptixDenoiser *)den_ptr;

  const int input_passes = denoise_buffer_num_passes(0, 0);

  const bool recreate_denoiser = (denoiser_.optix_denoiser == nullptr) ||
                                 (input_passes != denoiser_.input_passes);
  if (!recreate_denoiser) {
    return true;
  }

  /* Destroy existing handle before creating new one. */
  if (denoiser_.optix_denoiser) {
    optixDenoiserDestroy(denoiser_.optix_denoiser);
  }

  OptixDeviceContextOptions options = {};
  optixDeviceContextCreate(0, &options, &denoiser_.context);

  /* Create OptiX denoiser handle on demand when it is first used. */
  OptixDenoiserOptions denoiser_options = {};
  denoiser_options.guideAlbedo = input_passes >= 2;
  denoiser_options.guideNormal = input_passes >= 3;
  const OptixResult result = optixDenoiserCreate(denoiser_.context,
                                                 OPTIX_DENOISER_MODEL_KIND_HDR,
                                                 &denoiser_options,
                                                 &denoiser_.optix_denoiser);

  if (result != OPTIX_SUCCESS) {
    printf("Failed to create OptiX denoiser\n");
    return false;
  }

  /* OptiX denoiser handle was created with the requested number of input passes. */
  denoiser_.input_passes = input_passes;

  /* OptiX denoiser has been created, but it needs configuration. */
  denoiser_.is_configured = false;

  return true;
}

bool device_optix_denoise_configure_if_needed(char *den_ptr,
                                              int task_buffer_params_width,
                                              int task_buffer_params_height)
{
  CustomOptixDenoiser &denoiser_ = *(CustomOptixDenoiser *)den_ptr;

  if (denoiser_.is_configured && (denoiser_.configured_size_x == task_buffer_params_width &&
                                  denoiser_.configured_size_y == task_buffer_params_height)) {
    return true;
  }

  OptixDenoiserSizes sizes = {};
  optix_assert(optixDenoiserComputeMemoryResources(
      denoiser_.optix_denoiser, task_buffer_params_width, task_buffer_params_height, &sizes));

  denoiser_.scratch_size = sizes.withOverlapScratchSizeInBytes;
  denoiser_.scratch_offset = sizes.stateSizeInBytes;

  /* Allocate denoiser state if tile size has changed since last setup. */
  denoiser_.state.alloc_to_device(denoiser_.scratch_offset + denoiser_.scratch_size);

  /* Initialize denoiser state for the current tile size. */
  const OptixResult result = optixDenoiserSetup(denoiser_.optix_denoiser,
                                                0,
                                                task_buffer_params_width,
                                                task_buffer_params_height,
                                                denoiser_.state.device_pointer,
                                                denoiser_.scratch_offset,
                                                denoiser_.state.device_pointer +
                                                    denoiser_.scratch_offset,
                                                denoiser_.scratch_size);
  if (result != OPTIX_SUCCESS) {
    printf("Failed to set up OptiX denoiser\n");
    return false;
  }

  denoiser_.is_configured = true;
  denoiser_.configured_size_x = task_buffer_params_width;
  denoiser_.configured_size_y = task_buffer_params_height;

  return true;
}

bool device_optix_denoise_ensure(char *den_ptr,
                                 int task_buffer_params_width,
                                 int task_buffer_params_height)
{
  if (!device_optix_denoise_create_if_needed(den_ptr)) {
    printf("OptiX denoiser creation has failed.\n");
    return false;
  }

  if (!device_optix_denoise_configure_if_needed(
          den_ptr, task_buffer_params_width, task_buffer_params_height)) {
    printf("OptiX denoiser configuration has failed.\n");
    return false;
  }

  return true;
}

void device_optix_denoise_buffer(char *den_ptr,
                                 CUDAContextScope &scope,
                                 cudaStream_t cuda_stream,
                                 CUDADeviceQueue *queue,
                                 char *wtile_,
                                 int pass_stride,
                                 DEVICE_PTR d_input_rgb)
{
  ccl::KernelWorkTile *wtile = (ccl::KernelWorkTile *)wtile_;

  //const CUDAContextScope scope(this);

  if (!device_optix_denoise_ensure(den_ptr, wtile->w, wtile->h)) {
    return;
  }

  const int input_pass_stride = denoise_buffer_pass_stride(0, 0);

  //device_only_memory<float> input_rgb(this, "denoiser input rgb");
  //input_rgb.alloc_to_device(task.buffer_params.width * task.buffer_params.height *
  //                          input_pass_stride);

  //OptiXDeviceQueue queue(this);

  /* Make sure input data is in [0 .. 10000] range by scaling the input buffer by the number of
   *
   * samples in the buffer. This will do (scaled) copy of the noisy image and needed passes into
   * an input buffer for the OptiX denoiser. */
  if (!device_optix_denoise_filter_convert_to_rgb(scope,
                                                  cuda_stream,
                                                  queue,
                                                  wtile->x,
                                                  wtile->y,
                                                  wtile->w,
                                                  wtile->h,
                                                  wtile->offset,
                                                  wtile->stride,
                                                  pass_stride,
                                                  wtile->start_sample + wtile->num_samples,
                                                  (float*)d_input_rgb,
                                                  wtile->buffer)) {
    printf("Error connverting denoising passes to RGB buffer.\n");
    return;
  }

  if (!device_optix_denoise_run(den_ptr, wtile->w, wtile->h,
                                d_input_rgb,
                                cuda_stream)) {
    printf("Error running OptiX denoiser.\n");
    return;
  }

  /* Store result in the combined pass of the render buffer.
   *
   * This will scale the denoiser result up to match the number of samples ans store the result in
   * the combined pass. */
  if (!device_optix_denoise_filter_convert_from_rgb(scope,
                                                    cuda_stream,
                                                    queue,
                                                    wtile->x,
                                                    wtile->y,
                                                    wtile->w,
                                                    wtile->h,
                                                    wtile->offset,
                                                    wtile->stride,
                                                    pass_stride,
                                                    wtile->start_sample + wtile->num_samples,
                                                    (float *)d_input_rgb,
                                                    (char *)wtile->pixel)) {
    printf("Error copying denoiser result to the combined pass.");
    return;
  }

  queue->synchronize(scope, cuda_stream);
}

}  // namespace kernel
}  // namespace cyclesphi