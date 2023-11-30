/*
 * Copyright 2011-2021 Blender Foundation
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
 */

#include "integrator/path_trace_work_client.h"

#include "device/cpu/kernel.h"
#include "device/device.h"

//#include "device/client/device_impl.h"

#include "kernel/integrator/path_state.h"

#include "integrator/pass_accessor_cpu.h"
#include "integrator/path_trace_display.h"

#include "scene/scene.h"
#include "session/buffers.h"

#include "util/atomic.h"
#include "util/log.h"
#include "util/tbb.h"

#ifdef WITH_CLIENT
#  include "kernel/device/client/kernel_client.h"

#  ifdef WITH_CLIENT_RENDERENGINE_SENDER
#    include "kernel/device/client/kernel_tcp.h"
#  endif

#endif

//#include <cuda_runtime.h>
//#include "device/cuda/device_impl.h"

#define DEBUG_PRINT(name, size)  //printf("%s: %lld\n", #name, size);

#define CHECK_CLIENT_ERROR \
  if (cyclesphi::kernel::client::is_error()) \
    device_->set_error(string_printf("error in %s", __FUNCTION__));

CCL_NAMESPACE_BEGIN

unsigned long long int d_pix = 0;

/* Create TBB arena for execution of path tracing and rendering tasks. */
static inline tbb::task_arena local_tbb_arena_create(const Device *device)
{
  /* TODO: limit this to number of threads of CLIENT device, it may be smaller than
   * the system number of threads when we reduce the number of CLIENT threads in
   * CPU + GPU rendering to dedicate some cores to handling the GPU device. */
  return tbb::task_arena(device->info.cpu_threads);
}

/* Get CPUKernelThreadGlobals for the current thread. */
static inline CPUKernelThreadGlobals *kernel_thread_globals_get(
    vector<CPUKernelThreadGlobals> &kernel_thread_globals)
{
  const int thread_index = tbb::this_task_arena::current_thread_index();
  DCHECK_GE(thread_index, 0);
  DCHECK_LE(thread_index, kernel_thread_globals.size());

  return &kernel_thread_globals[thread_index];
}

PathTraceWorkCLIENT::PathTraceWorkCLIENT(Device *device,
                                         Film *film,
                                         DeviceScene *device_scene,
                                         bool *cancel_requested_flag)
    : PathTraceWork(device, film, device_scene, cancel_requested_flag),
      queue_(device->gpu_queue_create()),
      kernels_(Device::get_cpu_kernels()),
      // pixels_uchar4_(device, "pixels", MEM_READ_WRITE),
      // pixels_half4_(device, "pixels_half4", MEM_READ_WRITE),
      buffer_passes_(device, "client_buffer_passes", MEM_READ_WRITE),
      // display_rgba_half_(device, "display_rgba_half", MEM_READ_WRITE),
      num_samples_pointer_(NULL)
{
  DCHECK_EQ(device->info.type, DEVICE_CLIENT);
}

void PathTraceWorkCLIENT::init_execution()
{
  /* Cache per-thread kernel globals. */
  device_->get_cpu_kernel_thread_globals(kernel_thread_globals_);
}

void PathTraceWorkCLIENT::update_progress(char *task_bin,
                                          char *tile_bin,
                                          int num_samples,
                                          int pixel_samples)
{
  if (task_bin == NULL)
    return;

   PathTraceWorkCLIENT *cl = (PathTraceWorkCLIENT *)task_bin;
   if (cl != NULL && cl->num_samples_pointer_ != NULL)
    cl->num_samples_pointer_[0] = num_samples;

  // DeviceTask *task = (DeviceTask *)task_bin;
  // RenderTile *tile = (RenderTile *)tile_bin;

  // tile->sample = pixel_samples;  // / (tile->h*tile->w);//tile->start_sample + num_samples;
  // task->update_progress(tile, tile->h * tile->w);
  //// printf("tile->sample: %d\n", tile->sample);
}

void PathTraceWorkCLIENT::tex_update(
    bool interp3d, char *kg_bin, int id, float x, float y, float z, int type, float *res)
{
}

void PathTraceWorkCLIENT::render_samples(RenderStatistics &statistics,
                                         int start_sample,
                                         int samples_num,
                                         int sample_offset,
                                         bool nodisplay,
                                         int *num_samples_pointer)
{
  num_samples_pointer_ = num_samples_pointer;

  const int64_t image_width = effective_buffer_params_.width;
  const int64_t image_height = effective_buffer_params_.height;

  //#ifdef WITH_CLIENT
  if (nodisplay) {
    // if (pixels.size() > 0) {
    //  DEBUG_PRINT(client_mem_free_pixels,pixels.size())
    //  cyclesphi::kernel::client_mem_free((unsigned long long int)&pixels[0], pixels.size());
    // CHECK_CLIENT_ERROR;
    //}
    //
    // pixels.clear();
    pixels_half4_.clear();
    pixels_uchar4_.clear();
  }
  else {
    if (pixels_half4_.size() != image_width * image_height) {
      pixels_half4_.resize(image_width * image_height);
      // pixels_half4_.zero_to_device();
    }
    if (pixels_uchar4_.size() != image_width * image_height) {
      pixels_uchar4_.resize(image_width * image_height);
      // pixels_uchar4_.zero_to_device();
    }
  }
  //#endif

  int pass_stride = buffers_->params.pass_stride;
  float *render_buffer = buffers_->buffer.data();

  // KernelWorkTile tile;
  tile.x = effective_buffer_params_.full_x;
  tile.y = effective_buffer_params_.full_y;
  tile.w = image_width;
  tile.h = image_height;
  tile.start_sample = start_sample;
  tile.num_samples = samples_num;
  tile.sample_offset = sample_offset;
  tile.offset = effective_buffer_params_.offset;
  tile.stride = effective_buffer_params_.stride;
  tile.buffer = render_buffer;

//#ifdef WITH_CLIENT_MPI
  tile.pixel = (pixels_half4_.size() > 0) ? (char *)pixels_half4_.data() : NULL;
//#else
//  tile.pixel = (pixels_uchar4_.size() > 0) ? (char *)pixels_uchar4_.data() : NULL;
//#endif
  // printf("render_samples: %d x %d\n", image_width, image_height);

#ifdef MPI_CACHE_FILE
  client_path_to_cache("");
#endif

#if defined(WITH_CLIENT) && \
    (defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET) || defined(WITH_CLIENT_SOCKET))

  if (tile.pixel != NULL) {
    DEBUG_PRINT(client_path_trace, pixels_half4_.size())
    cyclesphi::kernel::client::path_trace(
        (char *)tile.buffer,
        (char *)tile.pixel,
        //((tile.start_sample == 0) ? 0 :
        //                            tile.preview_first_step_samples +
        //                                (tile.start_sample - 1) * tile.preview_step_samples),
        //(tile.start_sample == 0) ? tile.preview_first_step_samples : tile.preview_step_samples,
        tile.start_sample,
        tile.num_samples,
        tile.sample_offset,
        tile.x,
        tile.y,
        tile.offset,

        //#  ifdef WITH_CLIENT_RENDERENGINE_SENDER
        //          custom_width,
        //          custom_height,
        //          custom_width,
        //          custom_height,
        //          custom_width,
        //#  else
        tile.stride,
        tile.h,
        tile.w,
        tile.h,
        tile.w,
        //#  endif
        pass_stride,
        // tile.use_load_balancing,
        // tile.tile_step,
        // tile.compress
        device_scene_->data.integrator.has_shadow_catcher,
        device_scene_->data.max_shaders,
        device_scene_->data.kernel_features,
        device_scene_->data.volume_stack_size);

    CHECK_CLIENT_ERROR;

#  ifndef WITH_CLIENT_RENDERENGINE_SENDER
    cyclesphi::kernel::client::receive_path_trace(
        tile.offset,
        tile.stride,
        tile.x,
        tile.y,
        tile.h,
        tile.w,
        tile.num_samples,
//#    ifdef WITH_CLIENT_MPI
        sizeof(half4),
//#    else
//        sizeof(uchar4),
//#    endif
        (char *)tile.pixel,
        (char *)this,  //(char *)&task,
        (char *)&tile,
        NULL,  //(char *)&kernel_globals,
        (void (*)(bool, char *, int, float, float, float, int, float *)) &
            PathTraceWorkCLIENT::tex_update,
        (void (*)(char *, char *, int, int)) & PathTraceWorkCLIENT::update_progress);

    CHECK_CLIENT_ERROR;
#  endif

//#  ifndef WITH_CLIENT_RENDERENGINE_SENDER
//
//    cyclesphi::kernel::client_receive_path_trace(
//        tile.offset,
//
//        //#  ifdef WITH_CLIENT_RENDERENGINE_SENDER
//        //          custom_width,
//        //#  else
//        tile.stride,
//        //#  endif
//        tile.x,
//        tile.y,
//
//        //#  ifdef WITH_CLIENT_RENDERENGINE_SENDER
//        //          custom_height,
//        //          custom_width,
//        //#  else
//        tile.h,
//        tile.w,
//        //#endif
//
//        //(tile.start_sample == 0) ? tile.preview_first_step_samples : tile.preview_step_samples,
//        tile.num_samples,
//        //#  ifdef WITH_CLIENT_RENDERENGINE_SENDER
//        //          sizeof(uchar4),
//        //#  else
//        /*(task.rgba_half) ? sizeof(half4) */
//        sizeof(uchar4),
//        //#  endif
//        // tile.compress != 0,
//        (char *)tile.pixel,
//        (char *)this,  //(char *)&task,
//        (char *)&tile,
//        NULL,  //(char *)&kernel_globals,
//        (void (*)(bool, char *, int, float, float, float, int, float *)) &
//            PathTraceWorkCLIENT::tex_update,
//        (void (*)(char *, char *, int, int)) & PathTraceWorkCLIENT::update_progress);
//# else
#  ifdef WITH_CLIENT_RENDERENGINE_SENDER
    int s = 0;
    cyclesphi::kernel::tcp::recv_data_cam((char *)&s, sizeof(int));
    update_progress((char *)this, NULL, s, 0);
#  endif
  }
  else {
    DEBUG_PRINT(client_path_trace, 0)
    cyclesphi::kernel::client::path_trace(
        (char *)tile.buffer,
        NULL,
        tile.start_sample,
        tile.num_samples,
        tile.sample_offset,
        tile.x,
        tile.y,
        tile.offset,
        //#  ifdef WITH_CLIENT_RENDERENGINE_SENDER
        //                                           custom_width,
        //                                           custom_height,
        //                                           custom_width,
        //                                           custom_height,
        //                                           custom_width,
        //#  else
        tile.stride,
        tile.h,
        tile.w,
        tile.h,
        tile.w,
        //#  endif

        pass_stride,
        // tile.use_load_balancing,
        // tile.tile_step,
        // tile.compress
        device_scene_->data.integrator.has_shadow_catcher,
        device_scene_->data.max_shaders,
        device_scene_->data.kernel_features,
        device_scene_->data.volume_stack_size);

    CHECK_CLIENT_ERROR;

    cyclesphi::kernel::client::receive_path_trace(
        tile.offset,
        //#  ifdef WITH_CLIENT_RENDERENGINE_SENDER
        //          custom_width,
        //#  else
        tile.stride,
        //#  endif
        tile.x,
        tile.y,

        //#  ifdef WITH_CLIENT_RENDERENGINE_SENDER
        //          custom_height,
        //          custom_width,
        //#  else
        tile.h,
        tile.w,
        //#  endif
        tile.num_samples,
        pass_stride * sizeof(float),
        // false,
        (char *)tile.buffer,
        NULL,  //(char *)&task,
        (char *)&tile,
        NULL,  //(char *)&kernel_globals,
        (void (*)(bool, char *, int, float, float, float, int, float *)) &
            PathTraceWorkCLIENT::tex_update,
        (void (*)(char *, char *, int, int)) & PathTraceWorkCLIENT::update_progress);

    CHECK_CLIENT_ERROR;
  }
//#  endif
#endif

#if defined(WITH_CLIENT) && defined(WITH_CLIENT_FILE_MINISCENE)
  cyclesphi::kernel::client::set_kernel_globals(
      (char *)kernel_thread_globals_get(kernel_thread_globals_));
#endif

#if defined(WITH_CLIENT) && defined(WITH_CLIENT_FILE)
  // tile_num_samples = tile.num_samples;

  if (cyclesphi::kernel::client::is_preprocessing()) {
    //printf("client_path_trace\n");
  cyclesphi::kernel::client::path_trace((char *)render_buffer,
                                        NULL,
                                        start_sample,
                                        samples_num,
                                        sample_offset,
                                        0,
                                        0,
                                        0,
                                        image_width,
                                        image_height,
                                        image_width,
                                        image_height,
                                        image_width,
                                        pass_stride,
                                        device_scene_->data.integrator.has_shadow_catcher,
                                        device_scene_->data.max_shaders,
                                        device_scene_->data.kernel_features,
                                        device_scene_->data.volume_stack_size);

  CHECK_CLIENT_ERROR;
  }

  if (cyclesphi::kernel::client::is_postprocessing()) {
    size_t offset_buffer = 0;
    size_t size_buffer = image_height * image_width * pass_stride * sizeof(float);
    printf("client_read_cycles_buffer\n");
    cyclesphi::kernel::client::read_cycles_buffer(
        &samples_num, (char *)render_buffer, offset_buffer, size_buffer);

    CHECK_CLIENT_ERROR;
  }

  // tile.sample = tile.start_sample + tile_num_samples;
  // task.update_progress(&tile, tile_num_samples * tile.w * tile.h);
#endif

  // if (device_graphics_interop_) {
  //  device_graphics_interop_->unmap();
  //}
}

//#include <omp.h>
// void PathTraceWorkCLIENT::copy_to_display(PathTraceDisplay *display,
//                                          PassMode pass_mode,
//                                          int num_samples)
//{
//  const int full_x = effective_buffer_params_.full_x;
//  const int full_y = effective_buffer_params_.full_y;
//  const int width = effective_buffer_params_.width;
//  const int height = effective_buffer_params_.height;
//  const int offset = effective_buffer_params_.offset;
//  const int stride = effective_buffer_params_.stride;
//
//  half4 *rgba_half = display->map_texture_buffer();
//  if (!rgba_half) {
//    /* TODO(sergey): Look into using copy_to_gpu_display() if mapping failed. Might be needed for
//     * some implementations of GPUDisplay which can not map memory? */
//    return;
//  }
//
//  // float *render_buffer = buffers_->buffer.data();
//
//  // tbb::task_arena local_arena = local_tbb_arena_create(device_);
//  // local_arena.execute([&]() {
//  //  tbb::parallel_for(0, height, [&](int y) {
//  //    CPUKernelThreadGlobals *kernel_globals =
//  kernel_thread_globals_get(kernel_thread_globals_);
//  //    for (int x = 0; x < width; ++x) {
//  //      kernels_.convert_to_half_float(kernel_globals,
//  //                                     reinterpret_cast<uchar4 *>(rgba_half),
//  //                                     //reinterpret_cast<float
//  //                                     *>(buffers_->buffer.device_pointer), render_buffer,
//  //                                     sample_scale,
//  //                                     full_x + x,
//  //                                     full_y + y,
//  //                                     offset,
//  //                                     stride);
//  //    }
//  //  });
//  //});
//
//#if 0 //def WITH_CLIENT
//  //double t = omp_get_wtime();
//  if (pixels.size() > 0) {
//    DEBUG_PRINT(client_rgb_to_half, pixels.size())
//    cyclesphi::kernel::client_rgb_to_half(
//        (unsigned short *)rgba_half, (unsigned char *)&pixels[0], height, width);
//    // memcpy((char *)rgba_half, &pixels[0], pixels.size());
//  }
//  //printf("client_rgb_to_half: %f\n", omp_get_wtime() - t);
//#endif
//
//  display->unmap_texture_buffer();
//}
//
// void PathTraceWorkCLIENT::destroy_gpu_resources(PathTraceDisplay * /*gpu_display*/)
//{
//}

bool PathTraceWorkCLIENT::should_use_graphics_interop()
{
  /* There are few aspects with the graphics interop when using multiple devices caused by the fact
   * that the PathTraceDisplay has a single texture:
   *
   *   CUDA will return `CUDA_ERROR_NOT_SUPPORTED` from `cuGraphicsGLRegisterBuffer()` when
   *   attempting to register OpenGL PBO which has been mapped. Which makes sense, because
   *   otherwise one would run into a conflict of where the source of truth is. */
  if (has_multiple_works()) {
    return false;
  }

  if (!interop_use_checked_) {
    // Device *device = queue_->device;
    interop_use_ = device_->should_use_graphics_interop();

    if (interop_use_) {
      VLOG(2) << "Using graphics interop GPU display update.";
    }
    else {
      VLOG(2) << "Using naive GPU display update.";
    }

    interop_use_checked_ = true;
  }

  return interop_use_;
}

void PathTraceWorkCLIENT::copy_to_display(PathTraceDisplay *display,
                                          PassMode pass_mode,
                                          int num_samples)
{
  if (device_->have_error()) {
    /* Don't attempt to update GPU display if the device has errors: the error state will make
     * wrong decisions to happen about interop, causing more chained bugs. */
    return;
  }

  if (!buffers_->buffer.device_pointer) {
    LOG(WARNING) << "Request for GPU display update without allocated render buffers.";
    return;
  }

#ifdef WITH_CLIENT_GPUJPEG
  if ((tile.pixel != NULL) && should_use_graphics_interop()) {
    if (copy_to_display_interop(display, pass_mode, num_samples)) {
      return;
    }

    /* If error happens when trying to use graphics interop fallback to the native implementation
     * and don't attempt to use interop for the further updates. */
    interop_use_ = false;
  }
#endif

  copy_to_display_naive(display, pass_mode, num_samples);
}

void PathTraceWorkCLIENT::copy_to_display_naive(PathTraceDisplay *display,
                                                PassMode pass_mode,
                                                int num_samples)
{
  const int full_x = effective_buffer_params_.full_x;
  const int full_y = effective_buffer_params_.full_y;
  const int width = effective_buffer_params_.window_width;
  const int height = effective_buffer_params_.window_height;
  const int final_width = buffers_->params.window_width;
  const int final_height = buffers_->params.window_height;

  const int texture_x = full_x - effective_big_tile_params_.full_x +
                        effective_buffer_params_.window_x - effective_big_tile_params_.window_x;
  const int texture_y = full_y - effective_big_tile_params_.full_y +
                        effective_buffer_params_.window_y - effective_big_tile_params_.window_y;

  ////////////////////////////////
  if (pixels_uchar4_.size() == 0 && pixels_half4_.size() == 0) {
    half4 *rgba_half = display->map_texture_buffer();
    if (!rgba_half) {
      /* TODO(sergey): Look into using copy_to_display() if mapping failed. Might be needed for
       * some implementations of PathTraceDisplay which can not map memory? */
      return;
    }

    const KernelFilm &kfilm = device_scene_->data.film;

    const PassAccessor::PassAccessInfo pass_access_info = get_display_pass_access_info(pass_mode);

    const PassAccessorCPU pass_accessor(pass_access_info, kfilm.exposure, num_samples);

    PassAccessor::Destination destination = get_display_destination_template(display);
    destination.pixels_half_rgba = rgba_half;

    tbb::task_arena local_arena = local_tbb_arena_create(device_);
    local_arena.execute([&]() {
      pass_accessor.get_render_tile_pixels(buffers_.get(), effective_buffer_params_, destination);
    });

    display->unmap_texture_buffer();
  }
  else {
    ///////////////////////////////

/* Re-allocate display memory if needed, and make sure the device pointer is allocated.
 *
 * NOTE: allocation happens to the final resolution so that no re-allocation happens on every
 * change of the resolution divider. However, if the display becomes smaller, shrink the
 * allocated memory as well. */
// if (display_rgba_half_.data_width != final_width ||
//    display_rgba_half_.data_height != final_height) {
//  display_rgba_half_.alloc(final_width, final_height);
//  /* TODO(sergey): There should be a way to make sure device-side memory is allocated without
//   * transferring zeroes to the device. */
//  queue_->zero_to_device(display_rgba_half_);
//}

// PassAccessor::Destination destination(film_->get_display_pass());
// destination.d_pixels_half_rgba = display_rgba_half_.device_pointer;

// get_render_tile_film_pixels(destination, pass_mode, num_samples);

// queue_->copy_from_device(display_rgba_half_);
// queue_->synchronize();

///////////
#if 0
#  if defined(WITH_CLIENT) && (defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET) || \
                               defined(WITH_CLIENT_SOCKET))

#    ifndef WITH_CLIENT_RENDERENGINE_SENDER
  cyclesphi::kernel::client::receive_path_trace(
      tile.offset,
      tile.stride,
      tile.x,
      tile.y,
      tile.h,
      tile.w,
      tile.num_samples,
      sizeof(uchar4),
      (char *)tile.pixel,
      (char *)this,  //(char *)&task,
      (char *)&tile,
      NULL,  //(char *)&kernel_globals,
      (void (*)(bool, char *, int, float, float, float, int, float *)) &
          PathTraceWorkCLIENT::tex_update,
      (void (*)(char *, char *, int, int)) & PathTraceWorkCLIENT::update_progress);

  CHECK_CLIENT_ERROR;

#    endif
#  endif

#endif
    ///////////
//#ifdef WITH_CLIENT_MPI
//
//    display->copy_pixels_to_texture(pixels_half4_.data(), texture_x, texture_y, width, height);
//
//#else
//#  ifndef WITH_CLIENT_GPUJPEG
//    cyclesphi::kernel::client::rgb_to_half((unsigned short *)(pixels_half4_.host_pointer),
//                                           (unsigned char *)pixels_uchar4_.host_pointer,
//                                           tile.h,
//                                           tile.w);
//
//    CHECK_CLIENT_ERROR;
//#  endif
//    // pixels_.copy_from_device();
//    display->copy_pixels_to_texture(pixels_half4_.data(), texture_x, texture_y, width, height);
//
//#endif
    display->copy_pixels_to_texture(pixels_half4_.data(), texture_x, texture_y, width, height);
  }
}

bool PathTraceWorkCLIENT::copy_to_display_interop(PathTraceDisplay *display,
                                                  PassMode pass_mode,
                                                  int num_samples)
{
  if (!device_graphics_interop_) {
    device_graphics_interop_ = queue_->graphics_interop_create();
  }

  const DisplayDriver::GraphicsInterop graphics_interop_dst = display->graphics_interop_get();
  device_graphics_interop_->set_display_interop(graphics_interop_dst);

  device_ptr d_rgba_half = device_graphics_interop_->map();
#if defined(WITH_CLIENT_GPUJPEG) && !defined(WITH_CLIENT_RENDERENGINE_SENDER)
  if (d_rgba_half) {
    cyclesphi::kernel::client::recv_decode((char *)d_rgba_half, tile.w, tile.h);
  }

#endif

#if 0
  ////PassAccessor::Destination destination = get_display_destination_template(display);
  ////destination.d_pixels_half_rgba = d_rgba_half;

  ////get_render_tile_film_pixels(destination, pass_mode, num_samples);
#  if defined(WITH_CLIENT) && (defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET) || \
                               defined(WITH_CLIENT_SOCKET))
#    ifndef WITH_CLIENT_RENDERENGINE_SENDER
  cyclesphi::kernel::client::receive_path_trace(
      tile.offset,
      tile.stride,
      tile.x,
      tile.y,
      tile.h,
      tile.w,
      tile.num_samples,
      sizeof(half4),
      (char *)d_rgba_half  , // tile.pixel,
      (char *)this,  //(char *)&task,
      (char *)&tile,
      NULL,  //(char *)&kernel_globals,
      (void (*)(bool, char *, int, float, float, float, int, float *)) &
          PathTraceWorkCLIENT::tex_update,
      (void (*)(char *, char *, int, int)) & PathTraceWorkCLIENT::update_progress);

  CHECK_CLIENT_ERROR;

#    endif
#  endif
#endif

  // device_->generic_copy_dtod(d_rgba_half, pixels_half4_.device_pointer,
  // pixels_half4_.memory_size());

  // cyclesphi::kernel::client::rgb_to_half((unsigned short *)(pixels_half4_.host_pointer),
  //                                       (unsigned char *)pixels_uchar4_.host_pointer,
  //                                       tile.h,
  //                                       tile.w);

  // CHECK_CLIENT_ERROR;

  // cyclesphi::kernel::client::rgb_to_half(
  //    (unsigned short *)(d_rgba_half), (unsigned char *)tile.pixel, tile.h, tile.w);
  ///////////

  device_graphics_interop_->unmap();

  return true;
}

void PathTraceWorkCLIENT::destroy_gpu_resources(PathTraceDisplay *display)
{
  if (!device_graphics_interop_) {
    return;
  }
  display->graphics_interop_activate();
  device_graphics_interop_ = nullptr;
  display->graphics_interop_deactivate();
}

bool PathTraceWorkCLIENT::copy_render_buffers_from_device()
{
  buffers_->copy_from_device();

  return true;
}

bool PathTraceWorkCLIENT::copy_render_buffers_to_device()
{
  buffers_->buffer.copy_to_device();

  return true;
}

bool PathTraceWorkCLIENT::zero_render_buffers()
{
#if 1
  buffers_->zero();
#endif

#ifdef WITH_CLIENT

  if (buffer_passes_.size() != 0 &&
      buffer_passes_.size() !=
          sizeof(client_buffer_passes) * effective_buffer_params_.passes.size()) {
    // DEBUG_PRINT(client_mem_free_buffer_passes, buffer_passes.size())
    // cyclesphi::kernel::client_mem_free((DEVICE_PTR)buffer_passes.data(), buffer_passes.size());
    // CHECK_CLIENT_ERROR;

    // buffer_passes.clear();
    buffer_passes_.free();
  }

  if (buffer_passes_.size() == 0 && effective_buffer_params_.passes.size() > 0) {
    buffer_passes_.resize(sizeof(client_buffer_passes) * effective_buffer_params_.passes.size());
    client_buffer_passes *cbp = (client_buffer_passes *)buffer_passes_.data();
    for (int i = 0; i < effective_buffer_params_.passes.size(); i++) {
      cbp[i].include_albedo = (int)effective_buffer_params_.passes[i].include_albedo;
      cbp[i].mode = (int)effective_buffer_params_.passes[i].mode;
      if (effective_buffer_params_.passes[i].name.length() > 0)
        strcpy(cbp[i].name, effective_buffer_params_.passes[i].name.c_str());
      else
        strcpy(cbp[i].name, "");

      cbp[i].offset = effective_buffer_params_.passes[i].offset;
      cbp[i].type = effective_buffer_params_.passes[i].type;

      // DEBUG_PRINT(client_mem_alloc_buffer_passes, buffer_passes.size())
      // cyclesphi::kernel::client_mem_alloc(
      //    "client_buffer_passes", (DEVICE_PTR)buffer_passes.data(), buffer_passes.size());

      // DEBUG_PRINT(client_mem_copy_to_buffer_passes, buffer_passes.size())
      // cyclesphi::kernel::client_mem_copy_to((DEVICE_PTR)buffer_passes.data(),
      // CHECK_CLIENT_ERROR;
      // buffer_passes.size(), 0);
      buffer_passes_.copy_to_device();
    }
  }
#endif

  return true;
}

int PathTraceWorkCLIENT::adaptive_sampling_converge_filter_count_active(float threshold,
                                                                        bool reset)
{
  uint num_active_pixels = 0;
  return num_active_pixels;
}

void PathTraceWorkCLIENT::cryptomatte_postproces()
{
}

PathTraceWorkCLIENT::~PathTraceWorkCLIENT()
{
  // if (buffer_passes.size() > 0) {
  //  DEBUG_PRINT(client_mem_free_buffer_passes, buffer_passes.size())
  //  cyclesphi::kernel::client_mem_free((DEVICE_PTR)buffer_passes.data(), buffer_passes.size());
  // CHECK_CLIENT_ERROR;
  //  //buffer_passes.clear();
  //}
}

CCL_NAMESPACE_END
