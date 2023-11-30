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
#ifndef __KERNEL_OMP_H__
#define __KERNEL_OMP_H__

#include "client_api.h"
#include <map>
#include <cstdio>
#include <string>

// CCL_NAMESPACE_BEGIN

namespace cyclesphi {
namespace kernel {

struct DevKernelData {
  //DEVICE_PTR kernel_globals_cpu;
  //ccl::CPUDevice cpuDevice;
  //std::map<DEVICE_PTR, DEVICE_PTR> ptr_map;
  std::map<DEVICE_PTR, DEVICE_PTR> ptr2ptr_map;
  std::map<std::string, DEVICE_PTR> str2ptr_map;
};

extern DevKernelData *dev_kernel_data;
}  // namespace kernel
}  // namespace cyclesphi

namespace cyclesphi {
namespace kernel {
namespace omp {

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
                     char *signal_value);

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
                       char *signal_value);   

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
                    char *signal_value);

#if defined(WITH_CLIENT_CUDA_CPU_STAT)

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
                  double* line_times);

void path_trace_gpu_stat(DEVICE_PTR kg_bin,
                             char *buffer_bin,
                             char *pixel_bin,
                             int start_sample,
                             int end_sample,
                             int tile_x,
                             int tile_y,
                             int offset,
                             int stride,
                             int tile_h,
                             int tile_w,
                             char *sample_finished_omp,
                             char *reqFinished_omp,
                             int nprocs_cpu,
                             int nprocs_gpu);
#endif
// void path_trace2(int numDevice,
//                      DEVICE_PTR kg_bin,
//                      DEVICE_PTR buffer_bin,
//                      DEVICE_PTR pixels_bin,
//                      int start_sample,
//                      int end_sample,
//                      int tile_x,
//                      int tile_y,
//                      int offset,
//                      int stride,
//                      int tile_h,
//                      int tile_w,
//                      char *sample_finished_omp,
//                      char *reqFinished_omp,
//                      int nprocs_cpu,
//                      char *signal_value);

/* Device memory */
void alloc_kg(int numDevice);
void free_kg(int numDevice, DEVICE_PTR kg);

char *host_alloc(const char *name,
                      DEVICE_PTR mem,
                      size_t memSize,
                      int type = 0);  // 1-managed, 2-pinned
void host_free(const char *name,
                    DEVICE_PTR mem,
                    char *dmem,
                    int type = 0);  // 1-managed, 2-pinned

DEVICE_PTR mem_alloc(int numDevice,
                     const char *name,
                     char *mem,
                     size_t memSize,
                     bool spec_setts = true,
                     bool alloc_host = false);
void mem_copy_to(int numDevice, char *mem, DEVICE_PTR dmem, size_t memSize, char *signal_value);
void mem_copy_from(
    int numDevice, DEVICE_PTR dmem, char *mem, size_t offset, size_t memSize, char *signal_value);
void mem_zero(const char *name, int numDevice, DEVICE_PTR dmem, size_t memSize);
void mem_free(const char *name, int numDevice, DEVICE_PTR dmem, size_t memSize);
void tex_free(int numDevice, DEVICE_PTR kg_bin, const char *name, DEVICE_PTR dmem, size_t memSize);

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
                          bool unimem_flag);

void tex_info_copy(const char *name,
                   char *mem,
                   DEVICE_PTR map_id,
                   size_t memSize,
                   int data_type,
                   bool check_uniname = true);

void const_copy(
    int numDevice, DEVICE_PTR kg, const char *name, char *host, size_t size, bool save = true);
int cmp_data(DEVICE_PTR kg_bin, char *host_bin, size_t size);
size_t get_size_data(DEVICE_PTR kg_bin);

DEVICE_PTR get_data(DEVICE_PTR kg_bin);

float *get_camera_matrix(DEVICE_PTR kg_bin);

void tex_copy(int numDevice,
              DEVICE_PTR kg_bin,
              char *name_bin,
              DEVICE_PTR dmem,
              char *mem,
              size_t data_count,
              size_t mem_size);

void film_convert_byte(DEVICE_PTR _kg,
                       char *_rgba_byte,
                       float *buffer,
                       float sample_scale,
                       int x,
                       int y,
                       int offset,
                       int stride);

void convert_to_half_float(DEVICE_PTR _kg,
                           char *_rgba,
                           float *buffer,
                           float sample_scale,
                           int x,
                           int y,
                           int offset,
                           int stride);

void load_textures(int numDevice, DEVICE_PTR kg_bin, size_t texture_info_size
                   /* std::map<DEVICE_PTR, DEVICE_PTR> &ptr_map*/);
int get_pass_stride(int numDevice, DEVICE_PTR kg);

void blender_camera(DEVICE_PTR mem, char *temp_data, size_t mem_size);

void kernel_path_trace(
    DEVICE_PTR _kg, float *buffer, int sample, int x, int y, int offset, int stride);

void anim_step(int numDevice, DEVICE_PTR kg_bin, char *data_bin, int s);
// void socket_step(int numDevice, DEVICE_PTR kg_bin, char *data_bin, float *cameratoworld,
// float w, float h);
void socket_step(int numDevice, DEVICE_PTR kg_bin, char *data_bin, char *cd);
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
                 int use_lamp_mis);

void set_device(int device, int world_rank, int world_size);

int get_cpu_threads();

bool check_unimem(const char *_name);

DEVICE_PTR get_kernel_globals_cpu();
DEVICE_PTR get_ptr_map(DEVICE_PTR id);
void set_ptr_map(const char *name, DEVICE_PTR id, DEVICE_PTR value);

void enable_stat(bool enable);



void init_execution(int has_shadow_catcher_,
                    int max_shaders_,
                    int pass_stride_,
                    unsigned int kernel_features_,
                    unsigned int volume_stack_size_,
                    bool init = true);

#ifdef WITH_CLIENT_SHOW_STAT_BVH_LOOP
int get_bvh_traversal_max_level();
#endif

void frame_info(int current_frame, int current_frame_preview, int caching_enabled);
char *get_mem_dev(int numDevice, DEVICE_PTR map_bin);

#ifdef WITH_CPU_STAT
size_t get_data_stat(std::string name, int dev, int chunk);
DEVICE_PTR get_data_mem(std::string name);
size_t get_data_size(std::string name);
#endif

// CCL_NAMESPACE_END
}}}

#endif /* __KERNEL_OMP_H__ */
