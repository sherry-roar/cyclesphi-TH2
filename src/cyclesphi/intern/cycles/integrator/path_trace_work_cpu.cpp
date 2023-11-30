/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2022 Blender Foundation */

#include "integrator/path_trace_work_cpu.h"

#include "device/cpu/kernel.h"
#include "device/device.h"

#include "kernel/integrator/path_state.h"

#include "integrator/pass_accessor_cpu.h"
#ifndef BLENDER_CLIENT
#include "integrator/path_trace_display.h"
#endif

#include "scene/scene.h"
#include "session/buffers.h"

#include "util/atomic.h"
#include "util/log.h"

#ifndef BLENDER_CLIENT
#include "util/tbb.h"
#endif

#ifdef BLENDER_CLIENT
#include <omp.h>
#endif

CCL_NAMESPACE_BEGIN

#ifndef BLENDER_CLIENT
/* Create TBB arena for execution of path tracing and rendering tasks. */
static inline tbb::task_arena local_tbb_arena_create(const Device *device)
{
  /* TODO: limit this to number of threads of CPU device, it may be smaller than
   * the system number of threads when we reduce the number of CPU threads in
   * CPU + GPU rendering to dedicate some cores to handling the GPU device. */
  return tbb::task_arena(device->info.cpu_threads);
}
#endif

/* Get CPUKernelThreadGlobals for the current thread. */
static inline CPUKernelThreadGlobals *kernel_thread_globals_get(
    vector<CPUKernelThreadGlobals> &kernel_thread_globals)
{
#ifdef BLENDER_CLIENT
  const int thread_index = omp_get_thread_num();
#else
  const int thread_index = tbb::this_task_arena::current_thread_index();
#endif
  DCHECK_GE(thread_index, 0);
  DCHECK_LE(thread_index, kernel_thread_globals.size());

  return &kernel_thread_globals[thread_index];
}

PathTraceWorkCPU::PathTraceWorkCPU(Device *device,
                                   Film *film,
                                   DeviceScene *device_scene,
                                   bool *cancel_requested_flag)
    : PathTraceWork(device, film, device_scene, cancel_requested_flag),
      kernels_(Device::get_cpu_kernels())
{
  DCHECK_EQ(device->info.type, DEVICE_CPU);
}

void PathTraceWorkCPU::init_execution()
{
  /* Cache per-thread kernel globals. */
  device_->get_cpu_kernel_thread_globals(kernel_thread_globals_);
}

void PathTraceWorkCPU::render_samples(RenderStatistics &statistics,
                                      int start_sample,
                                      int samples_num,
                                      int sample_offset,
                                      bool nodisplay,
                                      int *num_samples_pointer
                                      )
{
  const int64_t image_width = effective_buffer_params_.width;
  const int64_t image_height = effective_buffer_params_.height;
  const int64_t total_pixels_num = image_width * image_height;

#ifndef BLENDER_CLIENT
  if (device_->profiler.active()) {
    for (CPUKernelThreadGlobals &kernel_globals : kernel_thread_globals_) {
      kernel_globals.start_profiling();
    }
  }
#endif

#ifdef BLENDER_CLIENT
#  pragma omp parallel for schedule(dynamic, 1)
   for (int64_t work_index = 0; work_index < total_pixels_num; work_index++) {
    if (is_cancel_requested()) {
      continue;
    }
#else
  tbb::task_arena local_arena = local_tbb_arena_create(device_);
  local_arena.execute([&]() {
    parallel_for(int64_t(0), total_pixels_num, [&](int64_t work_index) {
      if (is_cancel_requested()) {
        return;
      }
#endif

    const int y = work_index / image_width;
    const int x = work_index - y * image_width;

    KernelWorkTile work_tile;
    work_tile.x = effective_buffer_params_.full_x + x;
    work_tile.y = effective_buffer_params_.full_y + y;
    work_tile.w = 1;
    work_tile.h = 1;
    work_tile.start_sample = start_sample;
    work_tile.sample_offset = sample_offset;
    // work_tile.num_samples = 1;
    work_tile.num_samples = samples_num;
    work_tile.offset = effective_buffer_params_.offset;
    work_tile.stride = effective_buffer_params_.stride;

    CPUKernelThreadGlobals *kernel_globals = kernel_thread_globals_get(kernel_thread_globals_);

    render_samples_full_pipeline(kernel_globals, work_tile, samples_num);

#ifdef BLENDER_CLIENT
  }
#else
    });
  });
  if (device_->profiler.active()) {
    for (CPUKernelThreadGlobals &kernel_globals : kernel_thread_globals_) {
      kernel_globals.stop_profiling();
    }
  }
#endif

  statistics.occupancy = 1.0f;
}

void PathTraceWorkCPU::render_samples_full_pipeline(KernelGlobalsCPU *kernel_globals,
                                                    const KernelWorkTile &work_tile,
                                                    const int samples_num)
{
  const bool has_bake = device_scene_->data.bake.use;

  IntegratorStateCPU integrator_states[2];

  IntegratorStateCPU *state = &integrator_states[0];
  IntegratorStateCPU *shadow_catcher_state = nullptr;

  if (device_scene_->data.integrator.has_shadow_catcher) {
    shadow_catcher_state = &integrator_states[1];
    path_state_init_queues(shadow_catcher_state);
  }

  KernelWorkTile sample_work_tile = work_tile;
  float *render_buffer = buffers_->buffer.data();

  for (int sample = 0; sample < samples_num; ++sample) {
    if (is_cancel_requested()) {
      break;
    }

    if (has_bake) {
      if (!kernels_.integrator_init_from_bake(
              kernel_globals, state, &sample_work_tile, render_buffer)) {
        break;
      }
    }
    else {
      if (!kernels_.integrator_init_from_camera(
              kernel_globals, state, &sample_work_tile, render_buffer)) {
        break;
      }
    }

    kernels_.integrator_megakernel(kernel_globals, state, render_buffer);

    if (shadow_catcher_state) {
      kernels_.integrator_megakernel(kernel_globals, shadow_catcher_state, render_buffer);
    }

    ++sample_work_tile.start_sample;
  }
}

#ifndef BLENDER_CLIENT
void PathTraceWorkCPU::copy_to_display(PathTraceDisplay *display,
                                       PassMode pass_mode,
                                       int num_samples)
{
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

void PathTraceWorkCPU::destroy_gpu_resources(PathTraceDisplay * /*display*/)
{
}
#endif

bool PathTraceWorkCPU::copy_render_buffers_from_device()
{
  return buffers_->copy_from_device();
}

bool PathTraceWorkCPU::copy_render_buffers_to_device()
{
  buffers_->buffer.copy_to_device();
  return true;
}

bool PathTraceWorkCPU::zero_render_buffers()
{
  buffers_->zero();
  return true;
}

int PathTraceWorkCPU::adaptive_sampling_converge_filter_count_active(float threshold, bool reset)
{
  const int full_x = effective_buffer_params_.full_x;
  const int full_y = effective_buffer_params_.full_y;
  const int width = effective_buffer_params_.width;
  const int height = effective_buffer_params_.height;
  const int offset = effective_buffer_params_.offset;
  const int stride = effective_buffer_params_.stride;

  float *render_buffer = buffers_->buffer.data();

  uint num_active_pixels = 0;

#ifdef BLENDER_CLIENT
#  pragma omp parallel for schedule(dynamic, 1)
  for (int y = full_y; y < full_y + height; y++) {
#else
  tbb::task_arena local_arena = local_tbb_arena_create(device_);

  /* Check convergency and do x-filter in a single `parallel_for`, to reduce threading overhead. */
  local_arena.execute([&]() {
  parallel_for(full_y, full_y + height, [&](int y) {
#endif      
    CPUKernelThreadGlobals *kernel_globals = &kernel_thread_globals_[0];

    bool row_converged = true;
    uint num_row_pixels_active = 0;
    for (int x = 0; x < width; ++x) {
      if (!kernels_.adaptive_sampling_convergence_check(
              kernel_globals, render_buffer, full_x + x, y, threshold, reset, offset, stride)) {
        ++num_row_pixels_active;
        row_converged = false;
      }
    }

    atomic_fetch_and_add_uint32(&num_active_pixels, num_row_pixels_active);

    if (!row_converged) {
      kernels_.adaptive_sampling_filter_x(
          kernel_globals, render_buffer, y, full_x, width, offset, stride);
    }
#ifdef BLENDER_CLIENT
  }
#else
    });
  });
#endif
  if (num_active_pixels) {
#ifdef BLENDER_CLIENT
#  pragma omp parallel for schedule(dynamic, 1)
    for (int x = full_x; x < full_x + width; x++) {
#else
    local_arena.execute([&]() {
      parallel_for(full_x, full_x + width, [&](int x) {
#endif
        CPUKernelThreadGlobals *kernel_globals = &kernel_thread_globals_[0];
        kernels_.adaptive_sampling_filter_y(
            kernel_globals, render_buffer, x, full_y, height, offset, stride);
#ifdef BLENDER_CLIENT
    }
#else
    });
    });
#endif
  }

  return num_active_pixels;
}

void PathTraceWorkCPU::cryptomatte_postproces()
{
  const int width = effective_buffer_params_.width;
  const int height = effective_buffer_params_.height;

  float *render_buffer = buffers_->buffer.data();

#ifdef BLENDER_CLIENT
#  pragma omp parallel for schedule(dynamic, 1)
  for (int y = 0; y < height; y++) {
#else
  tbb::task_arena local_arena = local_tbb_arena_create(device_);

  /* Check convergency and do x-filter in a single `parallel_for`, to reduce threading overhead. */
  local_arena.execute([&]() {
    parallel_for(0, height, [&](int y) {
#endif
    CPUKernelThreadGlobals *kernel_globals = &kernel_thread_globals_[0];
    int pixel_index = y * width;

    for (int x = 0; x < width; ++x, ++pixel_index) {
      kernels_.cryptomatte_postprocess(kernel_globals, render_buffer, pixel_index);
    }
#ifdef BLENDER_CLIENT
  }
#else
    });
  });
#endif
}

#ifdef BLENDER_CLIENT
// void PathTraceWorkCPU::set_buffer(const KernelWorkTile &wtile)
// {
//   buffers_->buffer.device_pointer = (ccl::device_ptr)wtile.buffer;
//   buffers_->buffer.host_pointer = wtile.buffer;
//   effective_buffer_params_.full_x = wtile.x;
//   effective_buffer_params_.full_y = wtile.y;
//   effective_buffer_params_.full_width = wtile.w;
//   effective_buffer_params_.full_height = wtile.h;

//   effective_buffer_params_.window_x = wtile.x;
//   effective_buffer_params_.window_y = wtile.y;
//   effective_buffer_params_.window_width = wtile.w;
//   effective_buffer_params_.window_height = wtile.h;

//   effective_buffer_params_.width = wtile.w;
//   effective_buffer_params_.height = wtile.h;
//   effective_buffer_params_.offset = wtile.offset;
//   effective_buffer_params_.stride = wtile.stride;

//   effective_buffer_params_.pass_stride = device_scene_->data.film.pass_stride;
//   effective_buffer_params_.exposure = device_scene_->data.film.exposure;
// }

void PathTraceWorkCPU::set_buffer(const KernelWorkTile &wtile,
                                  std::vector<client_buffer_passes> &passes)
{
  buffers_->buffer.device_pointer = (ccl::device_ptr)wtile.buffer;
  buffers_->buffer.host_pointer = wtile.buffer;

  if (effective_buffer_params_.passes.size() != passes.size()) {
    effective_buffer_params_.passes.resize(passes.size());
    for (int i = 0; i < passes.size(); i++) {
      effective_buffer_params_.passes[i].include_albedo = passes[i].include_albedo;
      effective_buffer_params_.passes[i].mode = (PassMode)passes[i].mode;
      effective_buffer_params_.passes[i].name = std::string(passes[i].name);
      effective_buffer_params_.passes[i].offset = passes[i].offset;
      effective_buffer_params_.passes[i].type = (PassType)passes[i].type;
    }
    effective_buffer_params_.update_passes();
    if (effective_buffer_params_.pass_stride != device_scene_->data.film.pass_stride) {
      printf(
          "ERROR: effective_buffer_params_.pass_stride != device_scene_->data.film.pass_stride\n");
      exit(-1);
    }
  }

  effective_buffer_params_.full_x = wtile.x;
  effective_buffer_params_.full_y = wtile.y;
  effective_buffer_params_.full_width = wtile.w;
  effective_buffer_params_.full_height = wtile.h;

  effective_buffer_params_.window_x = wtile.x;
  effective_buffer_params_.window_y = wtile.y;
  effective_buffer_params_.window_width = wtile.w;
  effective_buffer_params_.window_height = wtile.h;

  effective_buffer_params_.width = wtile.w;
  effective_buffer_params_.height = wtile.h;
  effective_buffer_params_.offset = wtile.offset;
  effective_buffer_params_.stride = wtile.stride;

  effective_buffer_params_.pass_stride = device_scene_->data.film.pass_stride;
  effective_buffer_params_.exposure = device_scene_->data.film.exposure;

  buffers_->params = effective_buffer_params_;
  effective_full_params_ = effective_buffer_params_;
  effective_big_tile_params_ = effective_buffer_params_;
}

void PathTraceWorkCPU::get_render_tile_film_pixels(const PassAccessor::Destination &destination,
                                                   PassMode pass_mode,
                                                   int num_samples)
{
  const KernelFilm &kfilm = device_scene_->data.film;

  const PassAccessor::PassAccessInfo pass_access_info = get_display_pass_access_info(pass_mode);
  const PassAccessorCPU pass_accessor(pass_access_info, kfilm.exposure, num_samples);

  pass_accessor.get_render_tile_pixels(buffers_.get(), effective_buffer_params_, destination);
}
#endif

CCL_NAMESPACE_END
