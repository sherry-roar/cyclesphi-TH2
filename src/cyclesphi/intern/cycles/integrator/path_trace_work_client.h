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

#pragma once

#include "kernel/integrator/state.h"

#include "device/graphics_interop.h"

#include "device/cpu/kernel_thread_globals.h"
#include "device/queue.h"

#include "integrator/path_trace_work.h"

#include "util/vector.h"

CCL_NAMESPACE_BEGIN

struct KernelWorkTile;
struct KernelGlobalsCPU;

class CPUKernels;

/* Implementation of PathTraceWork which schedules work on to queues pixel-by-pixel,
 * for CLIENT devices.
 *
 * NOTE: For the CLIENT rendering there are assumptions about TBB arena size and number of concurrent
 * queues on the render device which makes this work be only usable on CLIENT. */
class PathTraceWorkCLIENT : public PathTraceWork {
 public:
  PathTraceWorkCLIENT(Device *device, Film *film, DeviceScene *device_scene, bool *cancel_requested_flag);
  virtual ~PathTraceWorkCLIENT();


  virtual void init_execution() override;

  virtual void render_samples(RenderStatistics &statistics,
                              int start_sample,
                              int samples_num,
                              int sample_offset,
                              bool nodisplay,
                              int *num_samples_pointer) override;

  virtual void copy_to_display(PathTraceDisplay *display,
                               PassMode pass_mode,
                               int num_samples) override;
  virtual void destroy_gpu_resources(PathTraceDisplay *display) override;

  virtual bool copy_render_buffers_from_device() override;
  virtual bool copy_render_buffers_to_device() override;
  virtual bool zero_render_buffers() override;

  virtual int adaptive_sampling_converge_filter_count_active(float threshold, bool reset) override;
  virtual void cryptomatte_postproces() override;

protected:
  /* Check whether graphics interop can be used for the PathTraceDisplay update. */
  bool should_use_graphics_interop();

  /* Naive implementation of the `copy_to_display()` which performs film conversion on the
   * device, then copies pixels to the host and pushes them to the `display`. */
  void copy_to_display_naive(PathTraceDisplay *display, PassMode pass_mode, int num_samples);

  /* Implementation of `copy_to_display()` which uses driver's OpenGL/GPU interoperability
   * functionality, avoiding copy of pixels to the host. */
  bool copy_to_display_interop(PathTraceDisplay *display, PassMode pass_mode, int num_samples);

    /* Temporary buffer used by the copy_to_display() whenever graphics interoperability is not
   * available. Is allocated on-demand. */
  //device_vector<half4> display_rgba_half_;
  unique_ptr<DeviceGraphicsInterop> device_graphics_interop_;

  /* Cached result of device->should_use_graphics_interop(). */
  bool interop_use_checked_ = false;
  bool interop_use_ = false;

  /* Integrator queue. */
  unique_ptr<DeviceQueue> queue_;
  /////////////

  static void update_progress(char *task_bin,
                                                   char *tile_bin,
                                                   int num_samples,
                                                   int pixel_samples);

  static void tex_update(
      bool interp3d, char *kg_bin, int id, float x, float y, float z, int type, float *res);

  /* Core path tracing routine. Renders given work time on the given queue. */
//  void render_samples_full_pipeline(KernelGlobals *kernel_globals,
//                                    const KernelWorkTile &work_tile,
//                                    const int samples_num);

  /* CLIENT kernels. */
  const CPUKernels &kernels_;

  /* Copy of kernel globals which is suitable for concurrent access from multiple threads.
   *
   * More specifically, the `kernel_globals_` is local to each threads and nobody else is
   * accessing it, but some "localization" is required to decouple from kernel globals stored
   * on the device level. */
  vector<CPUKernelThreadGlobals> kernel_thread_globals_;

  /* Render output buffers. */
  //RenderBuffers *render_buffers_;

  /*pixels*/
  vector<half4> pixels_half4_;
  vector<uchar4> pixels_uchar4_;
  device_vector<char> buffer_passes_;

  KernelWorkTile tile;

  int *num_samples_pointer_;
};

CCL_NAMESPACE_END
