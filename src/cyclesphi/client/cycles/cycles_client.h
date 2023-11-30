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

#ifndef __CYCLES_CLIENT_H__
#define __CYCLES_CLIENT_H__

#include "client_api.h"
#include <cstdio>

// CCL_NAMESPACE_BEGIN
namespace cyclesphi {
namespace client {

void write_data_kernelglobal(void *data, size_t size);
void read_data_kernelglobal(void *data, size_t size);
void close_kernelglobal();

//void save_buffer(client_kernel_struct &data);
//bool break_loop(client_kernel_struct &data);
//void set_device(int device, int rank);

//void render(client_kernel_struct &data);

//////

void path_trace_buffer_render(client_kernel_struct &data,
                                   int num_samples,
                                   DEVICE_PTR dev_pixels_node = 0,
                                   char *pixels_node = NULL,
                                   char *pixels = NULL,
                                   char *signal_value = NULL);

void path_trace_buffer_copy(client_kernel_struct &data,
                                 DEVICE_PTR dev_pixels_node = 0,
                                 char *pixels_node = NULL,
                                 char *pixels = NULL);

void alloc_kg(client_kernel_struct &data);

void free_kg(client_kernel_struct &data);

void mem_alloc(client_kernel_struct &data);

void mem_copy_to(client_kernel_struct &data);

void mem_zero(client_kernel_struct &data);

void mem_free(client_kernel_struct &data);

void tex_free(client_kernel_struct &data);

void const_copy(client_kernel_struct &data);

void tex_copy(client_kernel_struct &data);

void tex_info(client_kernel_struct &data);

void load_textures(client_kernel_struct &data);

void render(client_kernel_struct &data);

void save_buffer(client_kernel_struct &data);

bool break_loop(client_kernel_struct &data);

void set_device(int device, int world_rank, int world_size);

void save_bmp(int offset,
                   int stride,
                   int tile_x,
                   int tile_y,
                   int tile_h,
                   int tile_w,
                   int pass_stride,
                   int end_samples,
                   char *buffer,
                   char *pixels,
                   int step);

void build_bvh(client_kernel_struct &data);

void frame_info(client_kernel_struct &data);

void bcast(void *data, size_t size);

}  // namespace client
}  // namespace cyclesphi

#endif /* __CYCLES_CLIENT_H__ */
