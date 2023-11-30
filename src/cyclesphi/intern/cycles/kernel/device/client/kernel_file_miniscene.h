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

#ifndef __KERNEL_FILE_MINISCENE_H__
#define __KERNEL_FILE_MINISCENE_H__

#include "client_api.h"
#include <cstdio>

namespace cyclesphi {
namespace kernel {
namespace miniscene {
//CCL_NAMESPACE_BEGIN

void write_data_kernelglobal(void *data, size_t size);

bool read_data_kernelglobal(void *data, size_t size);

//void receive_path_trace_buffer(int offset, int stride, int tile_x, int tile_y, int tile_h, int tile_w, size_t pass_stride_sizeof, char* tile_buffer);

void receive_render_buffer(char *buffer_pixels, int tile_h, int tile_w, int pass_stride);

void receive_path_trace(int offset,
                        int stride,
                        int tile_x,
                        int tile_y,
                        int tile_h,
                        int tile_w,
                        int num_samples,
                        size_t pass_stride_sizeof,  // bool compress,
                        char *buffer_pixels,
                        char *task_bin,
                        char *tile_bin,
                        char *kg_bin,
                        void (*tex_update)(bool, char *, int, float, float, float, int, float *),
                        void (*update_progress)(char *, char *, int, int));

//void path_trace(char *buffer,
//                     int start_sample,
//                     int num_samples,
//                     int sample_offset,
//                     int tile_x,
//                     int tile_y,
//                     int offset,
//                     int stride,
//                     int tile_h,
//                     int tile_w,
//                     int pass_stride,
//                     int has_shadow_catcher,
//                     int max_shaders,
//                     unsigned int kernel_features,
//                     unsigned int volume_stack_size, 
//                     DEVICE_PTR host_ptr = 0);

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
                int pass_stride,  // bool use_load_balancing, int tile_step, int compress,
                int has_shadow_catcher,
                int max_shaders,
                unsigned int kernel_features,
                unsigned int volume_stack_size,
                DEVICE_PTR buffer_host_ptr = NULL,
                DEVICE_PTR pixels_host_ptr = NULL);

void alloc_kg(client_kernel_struct *_client_data = NULL);
void free_kg(client_kernel_struct *_client_data = NULL);

void mem_alloc(const char *name,
               DEVICE_PTR mem,
               size_t memSize,
               DEVICE_PTR host_ptr = NULL,
               client_kernel_struct *_client_data = NULL);
void mem_alloc_sub_ptr(const char *name,
                       DEVICE_PTR mem,
                       size_t offset,
                       DEVICE_PTR mem_sub,
                       client_kernel_struct *_client_data = NULL);
void mem_copy_to(const char *name,
                 DEVICE_PTR mem,
                 size_t memSize,
                 size_t offset,
                 DEVICE_PTR host_ptr = NULL,
                 client_kernel_struct *_client_data = NULL);
void mem_zero(const char *name,
              DEVICE_PTR mem,
              size_t memSize,
              size_t offset,
              DEVICE_PTR host_ptr = NULL,
              client_kernel_struct *_client_data = NULL);
void mem_free(const char *name,
              DEVICE_PTR mem,
              size_t memSize,
              DEVICE_PTR host_ptr = NULL,
              client_kernel_struct *_client_data = NULL);
void tex_free(const char *name,
              DEVICE_PTR mem,
              size_t memSize,
              DEVICE_PTR host_ptr = NULL,
              client_kernel_struct *_client_data = NULL);

void const_copy(const char *name,
                char *host,
                size_t size,
                DEVICE_PTR host_ptr = NULL,
                client_kernel_struct *_client_data = NULL);
void tex_copy(const char *name,
              void *mem,
              size_t data_size,
              size_t mem_size,
              DEVICE_PTR host_ptr = NULL,
              client_kernel_struct *_client_data = NULL);

void tex_info(char *mem,
              char *mem_data,
              size_t size,
              const char *name,
              int data_type,
              int data_elements,
              int interpolation,
              int extension,
              size_t data_width,
              size_t data_height,
              size_t data_depth);

void convert_rgb_to_half(unsigned short *destination,
                         unsigned char *source,
                         int tile_h,
                         int tile_w);

void frame_info(int current_frame, int current_frame_preview, int caching_enabled);

void load_textures(size_t size, client_kernel_struct *_client_data = NULL);
int count_devices();

bool read_cycles_buffer(int *samples, char* buffer, size_t offset, size_t size);
bool write_cycles_buffer(int *samples, char* buffer, size_t offset, size_t size);

bool write_cycles_data(const char* filename, char* buffer, size_t offset, size_t size);
bool read_cycles_data(const char* filename, char* buffer, size_t offset, size_t size);

bool is_preprocessing();
bool is_postprocessing();

int get_additional_samples();
void close_kernelglobal();

void build_optix_bvh(int operation,
                          char *build_input,
                          size_t build_size,
                          int num_motion_steps);

void set_kernel_globals(char *kg);
//CCL_NAMESPACE_END
}
}
}

#endif /* __KERNEL_FILE_MINISCENE_H__ */
