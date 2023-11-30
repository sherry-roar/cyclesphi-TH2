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

#include "kernel_client.h"

#ifdef WITH_CLIENT_MPI
#  include "kernel_file.h"
#  include "kernel_mpi.h"
#  define NET_NAMESPACE mpi
#endif

#ifdef WITH_CLIENT_MPI_SOCKET
#  include "kernel_mpi.h"
#  define NET_NAMESPACE mpi
#endif

#ifdef WITH_CLIENT_SOCKET
#  include "kernel_socket.h"
#  define NET_NAMESPACE socket
#endif

#ifdef WITH_CLIENT_FILE
#  include "kernel_file.h"
#  define NET_NAMESPACE file
#endif

#ifdef WITH_CLIENT_FILE_MINISCENE
#  include "kernel_file_miniscene.h"
#  define NET_NAMESPACE miniscene
#endif

#include <pthread.h>

// CCL_NAMESPACE_BEGIN

namespace cyclesphi {
namespace kernel {
namespace client {

pthread_mutex_t *client_lock = NULL;
bool start_render = false;

#define CLIENT_LOCK \
  if (client_lock == NULL) { \
    client_lock = new pthread_mutex_t(); \
    pthread_mutex_init(client_lock, NULL); \
  } \
  pthread_mutex_lock(client_lock);

#define CLIENT_UNLOCK \
  if (client_lock != NULL) \
    pthread_mutex_unlock(client_lock);

void recv_decode(char *dmem, int width, int height)
{
  CLIENT_LOCK;

#if defined(WITH_CLIENT_SOCKET)
  NET_NAMESPACE::recv_decode(dmem, width, height);
#endif

  CLIENT_UNLOCK;
}

void receive_path_trace(int offset,
                        int stride,
                        int tile_x,
                        int tile_y,
                        int tile_h,
                        int tile_w,
                        int num_samples,
                        size_t pass_stride_sizeof,
                        /*bool compress,*/ char *buffer_pixels,
                        char *task_bin,
                        char *tile_bin,
                        char *kg_bin,
                        void (*tex_update)(bool, char *, int, float, float, float, int, float *),
                        void (*update_progress)(char *, char *, int, int))
{
  CLIENT_LOCK;

  // #if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
  //   mpi::receive_path_trace(offset,
  //                           stride,
  //                           tile_x,
  //                           tile_y,
  //                           tile_h,
  //                           tile_w,
  //                           num_samples,
  //                           pass_stride_sizeof,
  //                           /*compress,*/ buffer_pixels,
  //                           task_bin,
  //                           tile_bin,
  //                           kg_bin,
  //                           tex_update,
  //                           update_progress);
  // #endif

  // #ifdef WITH_CLIENT_FILE
  //   file::receive_path_trace_buffer(
  //       offset, stride, tile_x, tile_y, tile_h, tile_w, pass_stride_sizeof, buffer_pixels);
  // #endif

  // #if defined(WITH_CLIENT_SOCKET)
  NET_NAMESPACE::receive_path_trace(offset,
                                    stride,
                                    tile_x,
                                    tile_y,
                                    tile_h,
                                    tile_w,
                                    num_samples,
                                    pass_stride_sizeof,
                                    /*compress,*/ buffer_pixels,
                                    task_bin,
                                    tile_bin,
                                    kg_bin,
                                    tex_update,
                                    update_progress);
  //#endif
  CLIENT_UNLOCK;
}

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
                                       /*bool compress,*/ char *buffer_pixels,
                                       char *task_bin,
                                       char *task_pool_bin,
                                       char *tile_bin,
                                       void (*update_progress)(char *, char *, int, int),
                                       bool (*update_break)(char *, char *))
{
  CLIENT_LOCK;

  // #if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
  //   mpi::receive_path_trace_load_balancing(offset,
  //                                          stride,
  //                                          tile_x,
  //                                          tile_y,
  //                                          tile_h,
  //                                          tile_w,
  //                                          tile_h2,
  //                                          tile_w2,
  //                                          num_samples,
  //                                          tile_step,
  //                                          pass_stride_sizeof,
  //                                          /*compress,*/ buffer_pixels,
  //                                          task_bin,
  //                                          task_pool_bin,
  //                                          tile_bin,
  //                                          update_progress,
  //                                          update_break);
  // #endif

  // #ifdef WITH_CLIENT_FILE
  // //    file::receive_path_trace_load_balancing(offset, stride, tile_x, tile_y, tile_h,
  // //                                               tile_w, tile_h2, tile_w2, num_samples,
  // tile_step,
  // //                                               pass_stride_sizeof, compress, buffer_pixels,
  // //                                               task_bin, tile_bin, update_progress);
  // #endif

  // #if defined(WITH_CLIENT_SOCKET)
  //   socket::receive_path_trace_load_balancing(offset,
  //                                             stride,
  //                                             tile_x,
  //                                             tile_y,
  //                                             tile_h,
  //                                             tile_w,
  //                                             tile_h2,
  //                                             tile_w2,
  //                                             num_samples,
  //                                             tile_step,
  //                                             pass_stride_sizeof,
  //                                             /*compress,*/ buffer_pixels,
  //                                             task_bin,
  //                                             task_pool_bin,
  //                                             tile_bin,
  //                                             update_progress,
  //                                             update_break);
  // #endif

  CLIENT_UNLOCK;
}
//#endif
////////////////////////////////////////////////////////////////////////////

void const_copy(const char *name,
                char *host_bin,
                size_t size,
                DEVICE_PTR host_ptr,
                client_kernel_struct *_data)
{
  CLIENT_LOCK;

  // #if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
  //   mpi::const_copy(name, host_bin, size, host_ptr, _data);
  // #endif

  // #ifdef WITH_CLIENT_FILE
  //   file::const_copy(name, host_bin, size, host_ptr);
  // #endif

  // #ifdef WITH_CLIENT_SOCKET
  NET_NAMESPACE::const_copy(name, host_bin, size, host_ptr);
  //#endif

  CLIENT_UNLOCK;
}

void path_to_cache(const char *path)
{
  CLIENT_LOCK;
#ifdef WITH_CLIENT_MPI
  file::close_kernelglobal();
  mpi::path_to_cache(path);
#endif
  CLIENT_UNLOCK;
}

void send_to_cache(client_kernel_struct &data, void *mem, size_t size)
{
  CLIENT_LOCK;

#if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
#  ifdef MPI_CACHE_FILE
  NET_NAMESPACE::write_data_kernelglobal(&data, sizeof(client_kernel_struct));
  if (mem != NULL)
    NET_NAMESPACE::write_data_kernelglobal(mem, size);
#  else
  NET_NAMESPACE::send_to_cache(data, mem, size);
#  endif
#endif

#if defined(WITH_CLIENT_SOCKET)
  NET_NAMESPACE::send_to_cache(data, mem, size);
#endif

  CLIENT_UNLOCK;
}

void load_textures(size_t size, client_kernel_struct *_data)
{
  CLIENT_LOCK;

  // #if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
  //   mpi::load_textures(size, _data);
  // #endif

  // #ifdef WITH_CLIENT_FILE
  //   file::load_textures(size);
  // #endif

  // #if defined(WITH_CLIENT_SOCKET)
  NET_NAMESPACE::load_textures(size, _data);
  //#endif

  CLIENT_UNLOCK;
}

void tex_copy(const char *name,
              void *mem,
              size_t data_size,
              size_t mem_size,
              DEVICE_PTR host_ptr,
              client_kernel_struct *_data)
{
  CLIENT_LOCK;

  // #if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
  //   mpi::tex_copy(name, mem, data_size, mem_size, host_ptr, _data);
  // #endif

  // #ifdef WITH_CLIENT_FILE
  //   file::tex_copy(name, mem, data_size, mem_size, host_ptr);
  // #endif

  // #if defined(WITH_CLIENT_SOCKET)
  NET_NAMESPACE::tex_copy(name, mem, data_size, mem_size, host_ptr, _data);
  //#endif

  CLIENT_UNLOCK;
}

// void blender_camera(void *mem,
//                     size_t mem_size)
//{
//#if defined(WITH_CLIENT_SOCKET)
//  socket::blender_camera( mem,  mem_size);
//#endif
//}

// bool g_free_kg = false;

void alloc_kg(client_kernel_struct *_data)
{
  CLIENT_LOCK;

  // if (g_free_kg)
  //  return;

  // #if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
  //   mpi::alloc_kg(_data);
  // #endif

  // #ifdef WITH_CLIENT_FILE
  //   file::alloc_kg();
  // #endif

  // #if defined(WITH_CLIENT_SOCKET)
  NET_NAMESPACE::alloc_kg(_data);
  //#endif

  start_render = true;

  CLIENT_UNLOCK;
}

void free_kg(client_kernel_struct *_data)
{
  CLIENT_LOCK;

  // if (g_free_kg)
  //  return;

  // #if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
  //   mpi::free_kg(_data);
  // #endif

  // #ifdef WITH_CLIENT_FILE
  //   file::free_kg();
  // #endif

  // #if defined(WITH_CLIENT_SOCKET)
  NET_NAMESPACE::free_kg(_data);
  //#endif

  //	g_free_kg = true;

  start_render = false;

  CLIENT_UNLOCK;
}

void mem_alloc(const char *name,
               DEVICE_PTR mem,
               size_t memSize,
               DEVICE_PTR host_ptr,
               client_kernel_struct *_data)
{
  CLIENT_LOCK;

  // #if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
  //   mpi::mem_alloc(name, mem, memSize, host_ptr, _data);
  // #endif

  // #ifdef WITH_CLIENT_FILE
  //   file::mem_alloc(name, mem, memSize, host_ptr);
  // #endif

  // #if defined(WITH_CLIENT_SOCKET)
  NET_NAMESPACE::mem_alloc(name, mem, memSize, host_ptr, _data);
  //#endif

  CLIENT_UNLOCK;
}

void mem_alloc_sub_ptr(const char *name,
                       DEVICE_PTR mem,
                       size_t offset,
                       DEVICE_PTR mem_sub,
                       client_kernel_struct *_data)
{
  CLIENT_LOCK;

  // #if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
  //   mpi::mem_alloc_sub_ptr(name, mem, offset, mem_sub, _data);
  // #endif

  // #ifdef WITH_CLIENT_FILE

  // #endif

  // #if defined(WITH_CLIENT_SOCKET)
  NET_NAMESPACE::mem_alloc_sub_ptr(name, mem, offset, mem_sub, _data);
  //#endif

  CLIENT_UNLOCK;
}

void mem_copy_to(const char *name,
                 DEVICE_PTR mem,
                 size_t memSize,
                 size_t offset,
                 DEVICE_PTR host_ptr,
                 client_kernel_struct *_data)
{
  CLIENT_LOCK;

  // #if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
  //   mpi::mem_copy_to(name, mem, memSize, offset, host_ptr, _data);
  // #endif

  // #ifdef WITH_CLIENT_FILE
  //   file::mem_copy_to(name, mem, memSize, offset, host_ptr);
  // #endif

  // #if defined(WITH_CLIENT_SOCKET)
  NET_NAMESPACE::mem_copy_to(name, mem, memSize, offset, host_ptr, _data);
  //#endif

  CLIENT_UNLOCK;
}

void mem_zero(const char *name,
              DEVICE_PTR mem,
              size_t memSize,
              size_t offset,
              DEVICE_PTR host_ptr,
              client_kernel_struct *_data)
{
  CLIENT_LOCK;

  // #if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
  //   mpi::mem_zero(name, mem, memSize, offset, host_ptr, _data);
  // #endif

  // #ifdef WITH_CLIENT_FILE
  //   file::mem_zero(name, mem, memSize, offset, host_ptr);
  // #endif

  // #if defined(WITH_CLIENT_SOCKET)
  NET_NAMESPACE::mem_zero(name, mem, memSize, offset, host_ptr, _data);
  //#endif

  CLIENT_UNLOCK;
}

void mem_free(const char *name,
              DEVICE_PTR mem,
              size_t memSize,
              DEVICE_PTR host_ptr,
              client_kernel_struct *_data)
{
  CLIENT_LOCK;

  // #if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
  //   mpi::mem_free(name, mem, memSize, host_ptr, _data);
  // #endif

  // #ifdef WITH_CLIENT_FILE
  //   file::mem_free(name, mem, memSize, host_ptr);
  // #endif

  // #if defined(WITH_CLIENT_SOCKET)
  NET_NAMESPACE::mem_free(name, mem, memSize, host_ptr, _data);
  //#endif

  CLIENT_UNLOCK;
}

void tex_free(const char *name,
              DEVICE_PTR mem,
              size_t memSize,
              DEVICE_PTR host_ptr,
              client_kernel_struct *_data)
{
  CLIENT_LOCK;

  // #if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
  //   mpi::tex_free(name, mem, memSize, host_ptr, _data);
  // #endif

  // #ifdef WITH_CLIENT_FILE
  //   file::tex_free(name, mem, memSize, host_ptr);
  // #endif

  // #if defined(WITH_CLIENT_SOCKET)
  NET_NAMESPACE::tex_free(name, mem, memSize, host_ptr, _data);
  //#endif

  CLIENT_UNLOCK;
}

void receive_render_buffer(char *buffer_pixels, int tile_h, int tile_w, int pass_stride)
{
  CLIENT_LOCK;

  // #if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
  //   mpi::receive_render_buffer(buffer_pixels, tile_h, tile_w, pass_stride);
  // #endif

  // #ifdef WITH_CLIENT_FILE
  // #endif

  // #if defined(WITH_CLIENT_SOCKET)
  NET_NAMESPACE::receive_render_buffer(buffer_pixels, tile_h, tile_w, pass_stride);
  //#endif

  CLIENT_UNLOCK;
}

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
                // bool use_load_balancing, int tile_step, int compress,
                int has_shadow_catcher,
                int max_shaders,
                unsigned int kernel_features,
                unsigned int volume_stack_size,
                DEVICE_PTR buffer_host_ptr,
                DEVICE_PTR pixels_host_ptr)
{
  CLIENT_LOCK;
  // #if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
  //   mpi::path_trace(buffer,
  //                   pixels,
  //                   start_sample,
  //                   num_samples,
  //                   sample_offset,
  //                   tile_x,
  //                   tile_y,
  //                   offset,
  //                   stride,
  //                   tile_h,
  //                   tile_w,
  //                   tile_h2,
  //                   tile_w2,
  //                   pass_stride,
  //                   // use_load_balancing, tile_step, compress,
  //                   has_shadow_catcher,
  //                   max_shaders,
  //                   kernel_features,
  //                   volume_stack_size,
  //                   buffer_host_ptr,
  //                   pixels_host_ptr);
  // #endif

  // #ifdef WITH_CLIENT_FILE
  //   file::path_trace(buffer,
  //                    start_sample,
  //                    num_samples,
  //                    sample_offset,
  //                    tile_x,
  //                    tile_y,
  //                    offset,
  //                    stride,
  //                    tile_h,
  //                    tile_w,
  //                    pass_stride,
  //                    has_shadow_catcher,
  //                    max_shaders,
  //                    kernel_features,
  //                    volume_stack_size,
  //                    buffer_host_ptr);
  // #endif

  // #if defined(WITH_CLIENT_SOCKET)
  //   // socket::path_trace(buffer,
  //   //                  pixels,
  //   //                  start_sample,
  //   //                  num_samples,
  //   //                  sample_offset,
  //   //                  tile_x,
  //   //                  tile_y,
  //   //                  offset,
  //   //                  stride,
  //   //                  tile_h,
  //   //                  tile_w,
  //   //                  tile_h2,
  //   //                  tile_w2,
  //   //                  pass_stride,
  //   //                  // use_load_balancing, tile_step, compress,
  //   //                  has_shadow_catcher,
  //   //                  num_shaders,
  //   //                  kernel_features,
  //   //                  volume_stack_size,
  //   //                  buffer_host_ptr,
  //   //                  pixels_host_ptr);

  NET_NAMESPACE::path_trace(buffer,
                            pixels,
                            start_sample,
                            num_samples,
                            sample_offset,
                            tile_x,
                            tile_y,
                            offset,
                            stride,
                            tile_h,
                            tile_w,
                            tile_h2,
                            tile_w2,
                            pass_stride,
                            // use_load_balancing, tile_step, compress,
                            has_shadow_catcher,
                            max_shaders,
                            kernel_features,
                            volume_stack_size,
                            buffer_host_ptr,
                            pixels_host_ptr);
  //#endif
  CLIENT_UNLOCK;
}

bool read_cycles_buffer(int *samples, char *buffer, size_t offset, size_t size)
{
  bool res = true;

  CLIENT_LOCK;

#if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
  res = true;
#endif

#ifdef WITH_CLIENT_FILE
  res = NET_NAMESPACE::read_cycles_buffer(samples, buffer, offset, size);
#endif

#if defined(WITH_CLIENT_SOCKET)
  res = true;
#endif

  CLIENT_UNLOCK;

  return res;
}

bool write_cycles_buffer(int *samples, char *buffer, size_t offset, size_t size)
{
  bool res = true;

  CLIENT_LOCK;

#if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
  res = true;
#endif

#ifdef WITH_CLIENT_FILE
  res = NET_NAMESPACE::write_cycles_buffer(samples, buffer, offset, size);
#endif

#if defined(WITH_CLIENT_SOCKET)
  res = true;
#endif

  CLIENT_UNLOCK;

  return res;
}

bool is_preprocessing()
{
#if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
  return true;
#endif

#ifdef WITH_CLIENT_FILE
  return NET_NAMESPACE::is_preprocessing();
#endif

#if defined(WITH_CLIENT_SOCKET)
  return true;
#endif

  return true;
}

bool is_postprocessing()
{
#if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
  return true;
#endif

#ifdef WITH_CLIENT_FILE
  return NET_NAMESPACE::is_postprocessing();
#endif

#if defined(WITH_CLIENT_SOCKET)
  return true;
#endif
  return true;
}

void tex_image_interp(int id, float x, float y, float *result)
{
  CLIENT_LOCK;

#ifdef WITH_CLIENT_MPI
  NET_NAMESPACE::tex_image_interp(id, x, y, result);
#endif

  CLIENT_UNLOCK;
}

void tex_image_interp3d(int id, float x, float y, float z, int type, float *result)
{
  CLIENT_LOCK;

#ifdef WITH_CLIENT_MPI
  NET_NAMESPACE::tex_image_interp3d(id, x, y, z, type, result);
#endif

  CLIENT_UNLOCK;
}

void tex_info(char *mem,
              size_t size,
              const char *name,
              int data_type,
              int data_elements,
              int interpolation,
              int extension,
              size_t data_width,
              size_t data_height,
              size_t data_depth)
{
  CLIENT_LOCK;

  // #if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
  //   // mpi::tex_info(/*info*/ NULL, mem, size);
  //   mpi::tex_info(mem,
  //                 mem,
  //                 size,
  //                 name,
  //                 data_type,
  //                 data_elements,
  //                 interpolation,
  //                 extension,
  //                 data_width,
  //                 data_height,
  //                 data_depth);
  // #endif

  // #ifdef WITH_CLIENT_FILE
  //   file::tex_info(mem,
  //                  size,
  //                  name,
  //                  data_type,
  //                  data_elements,
  //                  interpolation,
  //                  extension,
  //                  data_width,
  //                  data_height,
  //                  data_depth);
  // #endif

  // #if defined(WITH_CLIENT_SOCKET)
  //   // mpi::tex_info(/*info*/ NULL, mem, size);
  NET_NAMESPACE::tex_info(mem,
                          mem,
                          size,
                          name,
                          data_type,
                          data_elements,
                          interpolation,
                          extension,
                          data_width,
                          data_height,
                          data_depth);
  //#endif

  CLIENT_UNLOCK;
}

#ifdef WITH_CLIENT_OPTIX
void build_optix_bvh(int operation, char *build_input, size_t build_size, int num_motion_steps)
{
  CLIENT_LOCK;

// #  if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
//   mpi::build_optix_bvh(operation, build_input, build_size, num_motion_steps);
// #  endif

// #  ifdef WITH_CLIENT_FILE
//   file::build_optix_bvh(operation, build_input, build_size, num_motion_steps);
// #  endif

// #  ifdef WITH_CLIENT_SOCKET
  NET_NAMESPACE::build_optix_bvh(operation, build_input, build_size, num_motion_steps);
//#  endif

  CLIENT_UNLOCK;
}
#endif

void rgb_to_half(unsigned short *destination, unsigned char *source, int tile_h, int tile_w)
{
  CLIENT_LOCK;

// #if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
//   mpi::convert_rgb_to_half(destination, source, tile_h, tile_w);
// #endif

// #if defined(WITH_CLIENT_SOCKET)
  NET_NAMESPACE::convert_rgb_to_half(destination, source, tile_h, tile_w);
//#endif

  CLIENT_UNLOCK;
}

bool is_error()
{
#if defined(WITH_CLIENT_SOCKET)
  return socket::is_error();
#endif

  return false;
}

void frame_info(int current_frame, int current_frame_preview, int caching_enabled)
{
  if (!start_render)
    return;

  CLIENT_LOCK;
  // #if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
  //   mpi::frame_info(current_frame, current_frame_preview, caching_enabled);
  // #endif

  // #if defined(WITH_CLIENT_SOCKET)
  NET_NAMESPACE::frame_info(current_frame, current_frame_preview, caching_enabled);
  //#endif
  CLIENT_UNLOCK;
}

void set_kernel_globals(char *kg)
{
  CLIENT_LOCK;

  NET_NAMESPACE::set_kernel_globals(kg);

  CLIENT_UNLOCK;
}

// CCL_NAMESPACE_END
}  // namespace client
}  // namespace kernel
}  // namespace cyclesphi