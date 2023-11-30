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

#ifndef __KERNEL_CLIENT_H__
#define __KERNEL_CLIENT_H__

//#define WITH_CLIENT_FILE

#include "client_api.h"
#include <cstdio>

// CCL_NAMESPACE_BEGIN

namespace cyclesphi {
namespace kernel {

namespace client {
// void write_data_kernelglobal(void *data, size_t size);
// void read_data_kernelglobal(void *data, size_t size);

// void receive_path_trace_buffer(int offset, int stride, int tile_x, int tile_y, int
// tile_h, int tile_w, int pass_stride, char* tile_buffer);
//
// void path_trace(char *buffer, int start_sample, int num_samples, int tile_x, int tile_y,
// int offset, int stride, int tile_h, int tile_w, int pass_stride, DEVICE_PTR host_ptr = 0);
//
// void alloc_kg();
// void free_kg();
//
// void mem_alloc(DEVICE_PTR mem, size_t memSize, DEVICE_PTR host_ptr = 0);
// void mem_copy_to(DEVICE_PTR mem, size_t memSize, size_t offset, DEVICE_PTR host_ptr = 0);
// void mem_zero(DEVICE_PTR mem, size_t memSize, size_t offset, DEVICE_PTR host_ptr = 0);
// void mem_free(DEVICE_PTR mem, size_t memSize, DEVICE_PTR host_ptr = 0);
// void tex_free(DEVICE_PTR mem, size_t memSize, DEVICE_PTR host_ptr = 0);
//
// void const_copy(const char *name, char *host, size_t size, DEVICE_PTR host_ptr = 0);
// void tex_copy(const char *name, void* mem, size_t size, DEVICE_PTR host_ptr = 0);
//
// void load_textures(size_t size);
// int count_devices();
//
bool read_cycles_buffer(int *samples, char *buffer, size_t offset, size_t size);
bool write_cycles_buffer(int *samples, char *buffer, size_t offset, size_t size);

bool is_preprocessing();
bool is_postprocessing();
//
// int get_additional_samples();
// void file_close_kernelglobal();

// void write_data_kernelglobal(void *data, size_t size);
void read_data_kernelglobal(void *data, size_t size);
void file_close_kernelglobal();

void receive_render_buffer(char *buffer_pixels, int tile_h, int tile_w, int pass_stride);

void path_trace(char *buffer,
                       char *pixels,
                       int start_sample,
                       int num_samples,
                       int sample_offset,
                       int tile_x,
                       int tile_y,
                       int offset,
                       int stride,
                       int tile_h,
                       int tile_w,
                       int tile_h2,
                       int tile_w2,
                       int pass_stride,
                       // bool use_load_balancing,
                       // int tile_step,
                       // int compress,
                       int has_shadow_catcher,
                       int max_shaders,
                       unsigned int kernel_features,
                       unsigned int volume_stack_size,
                       DEVICE_PTR buffer_host_ptr = NULL,
                       DEVICE_PTR pixels_host_ptr = NULL);

//#ifdef CLIENT_MPI_LOAD_BALANCING_SAMPLES
// void receive_path_trace_offline(int offset,
//                                       int stride,
//                                       int tile_x,
//                                       int tile_y,
//                                       int tile_h,
//                                       int tile_w,
//                                       int tile_h2,
//                                       int tile_w2,
//                                       int num_samples,
//                                       int tile_step,
//                                       size_t pass_stride_sizeof,
//                                       bool compress,
//                                       char *buffer_pixels,
//                                       char *task_bin,
//                                       char *task_pool_bin,
//                                       char *tile_bin,
//                                       void (*update_progress)(char *, char *, int, int),
//                                       bool (*update_break)(char *, char *));
//
// void receive_path_trace_interactive(int offset,
//                                           int stride,
//                                           int tile_x,
//                                           int tile_y,
//                                           int tile_h,
//                                           int tile_w,
//                                           int sample,
//                                           int num_samples,
//                                           size_t pass_stride_sizeof,
//                                           bool compress,
//                                           char *buffer_pixels,
//                                           char *task_bin,
//                                           char *tile_bin,
//                                           void (*update_progress)(char *, char *, int, int));
//
//#else
void receive_path_trace_load_balancing(int offset,
                                              int stride,
                                              int tile_x,
                                              int tile_y,
                                              int tile_h,
                                              int tile_w,
                                              int tile_h2,
                                              int tile_w2,
                                              int num_samples,
                                              int tile_step,
                                              size_t pass_stride_sizeof,
                                              // bool compress,
                                              char *buffer_pixels,
                                              char *task_bin,
                                              char *task_pool_bin,
                                              char *tile_bin,
                                              void (*update_progress)(char *, char *, int, int),
                                              bool (*update_break)(char *, char *));

void receive_path_trace(
    int offset,
    int stride,
    int tile_x,
    int tile_y,
    int tile_h,
    int tile_w,
    int num_samples,
    size_t pass_stride_sizeof,
    // bool compress,
    char *buffer_pixels,
    char *task_bin,
    char *tile_bin,
    char *kg_bin,
    void (*tex_update)(bool, char *, int, float, float, float, int, float *),
    void (*update_progress)(char *, char *, int, int));
//#endif

void recv_decode(char *dmem, int width, int height);

void alloc_kg(client_kernel_struct *_data = NULL);
void free_kg(client_kernel_struct *_data = NULL);

void mem_alloc(const char *name,
                      DEVICE_PTR mem,
                      size_t memSize,
                      DEVICE_PTR host_ptr = NULL,
                      client_kernel_struct *_data = NULL);
void mem_alloc_sub_ptr(const char *name,
                       DEVICE_PTR mem,
                              size_t offset,
                              DEVICE_PTR mem_sub,
                              client_kernel_struct *_data = NULL);

void mem_copy_to(const char *name,
                 DEVICE_PTR mem,
                        size_t memSize,
                        size_t offset,
                        DEVICE_PTR host_ptr = NULL,
                        client_kernel_struct *_data = NULL);
void mem_zero(const char *name,
              DEVICE_PTR mem,
                     size_t memSize,
                     size_t offset,
                     DEVICE_PTR host_ptr = NULL,
                     client_kernel_struct *_data = NULL);
void mem_free(const char *name,
              DEVICE_PTR mem,
                     size_t memSize,
                     DEVICE_PTR host_ptr = NULL,
                     client_kernel_struct *_data = NULL);
void tex_free(const char *name,
              DEVICE_PTR mem,
                     size_t memSize,
                     DEVICE_PTR host_ptr = NULL,
                     client_kernel_struct *_data = NULL);

void const_copy(const char *name,
                       char *host,
                       size_t size,
                       DEVICE_PTR host_ptr = NULL,
                       client_kernel_struct *_data = NULL);
void tex_copy(const char *name,
                     void *mem,
                     size_t data_size,
                     size_t mem_size,
                     DEVICE_PTR host_ptr = NULL,
                     client_kernel_struct *_data = NULL);

//void blender_camera(void *mem, size_t mem_size);

void load_textures(size_t size, client_kernel_struct *_data = NULL);
void send_to_cache(client_kernel_struct &data, void *mem = NULL, size_t size = 0);
void path_to_cache(const char *path);
// int count_devices();

///////////////////////////////////////////////////
// void denoising_non_local_means(DEVICE_PTR image_ptr,
//                                      DEVICE_PTR guide_ptr,
//                                      DEVICE_PTR variance_ptr,
//                                      DEVICE_PTR out_ptr,
//                                      denoising_task_struct *task);
//
// void denoising_construct_transform(denoising_task_struct *task);
//
// void denoising_reconstruct(DEVICE_PTR color_ptr,
//                                  DEVICE_PTR color_variance_ptr,
//                                  DEVICE_PTR output_ptr,
//                                  denoising_task_struct *task);
//
// void denoising_combine_halves(DEVICE_PTR a_ptr,
//                                     DEVICE_PTR b_ptr,
//                                     DEVICE_PTR mean_ptr,
//                                     DEVICE_PTR variance_ptr,
//                                     int r,
//                                     int *rect,
//                                     denoising_task_struct * /*task*/);
//
// void denoising_divide_shadow(DEVICE_PTR a_ptr,
//                                    DEVICE_PTR b_ptr,
//                                    DEVICE_PTR sample_variance_ptr,
//                                    DEVICE_PTR sv_variance_ptr,
//                                    DEVICE_PTR buffer_variance_ptr,
//                                    denoising_task_struct *task);
//
// void denoising_get_feature(int mean_offset,
//                                  int variance_offset,
//                                  DEVICE_PTR mean_ptr,
//                                  DEVICE_PTR variance_ptr,
//                                  denoising_task_struct *task);
//
// void denoising_detect_outliers(DEVICE_PTR image_ptr,
//                                      DEVICE_PTR variance_ptr,
//                                      DEVICE_PTR depth_ptr,
//                                      DEVICE_PTR output_ptr,
//                                      denoising_task_struct *task);

void tex_image_interp(int id, float x, float y, float *result);
void tex_image_interp3d(int id, float x, float y, float z, int type, float *result);
void tex_info(char *mem,
                     size_t size,
                     const char *name,
                     int data_type,
                     int data_elements,
                     int interpolation,
                     int extension,
                     size_t data_width,
                     size_t data_height,
                     size_t data_depth);

void rgb_to_half(unsigned short *destination,
                        unsigned char *source,
                        int tile_h,
                        int tile_w);

bool is_error();

void frame_info(int current_frame, int current_frame_preview, int caching_enabled);

#ifdef WITH_CLIENT_OPTIX
void build_optix_bvh(int operation,
                            char *build_input,
                            size_t build_size,
                            int num_motion_steps);
#endif

void set_kernel_globals(char *kg);

// CCL_NAMESPACE_END
}
}  // namespace kernel
}  // namespace cyclesphi

#endif /* __KERNEL_CLIENT_H__ */
