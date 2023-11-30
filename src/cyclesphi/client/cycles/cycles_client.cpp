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

#include "cycles_client.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#ifndef _WIN32
#  include <unistd.h>
#endif

#include "kernel/device/client/kernel_client.h"
#include "kernel_util.h"

#ifdef WITH_CLIENT_MPI
#  include "kernel/device/client/kernel_mpi.h"
#  define NET_NAMESPACE kernel::mpi
#endif

#ifdef WITH_CLIENT_FILE
#  include "kernel/device/client/kernel_file.h"
#  define NET_NAMESPACE kernel::file
#endif

#ifdef WITH_CLIENT_MPI_SOCKET
#  include "kernel/device/client/kernel_tcp.h"
#  define NET_NAMESPACE kernel::tcp
#  include "kernel/device/client/kernel_mpi.h"
#  define NET_NAMESPACE_MPI kernel::mpi
#endif

#ifdef WITH_CLIENT_NCCL_SOCKET
#  include <cuda_runtime.h>
#  include <nccl.h>
#endif

#ifdef WITH_CLIENT_SOCKET
#  include "kernel/device/client/kernel_tcp.h"
#  define NET_NAMESPACE kernel::tcp
#endif

#if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_FILE) || defined(WITH_CLIENT_MPI_SOCKET)
#  include <mpi.h>
#endif

#include "cyclesphi_data.h"

#if defined(WITH_CLIENT_RENDERENGINE) && !defined(WITH_CLIENT_ULTRAGRID)
#  include "kernel/device/client/kernel_tcp.h"
#endif

#ifdef WITH_CLIENT_ULTRAGRID
#  include "ultragrid.h"
#endif

#ifdef WITH_CLIENT_CPU
#  include "kernel_omp.h"
#  define DEV_NAMESPACE kernel::omp
#endif

#ifdef WITH_CLIENT_CUDA
#  include "kernel_cuda.h"
#  define DEV_NAMESPACE kernel::cuda
#endif

#ifdef WITH_CLIENT_OPTIX
#  include "kernel_optix.h"
#  define DEV_NAMESPACE kernel::optix
#endif

#ifdef WITH_CLIENT_HIP
#  include "kernel_hip.h"
#  define DEV_NAMESPACE kernel::hip
#endif

// CCL_NAMESPACE_BEGIN

namespace cyclesphi {
namespace client {

std::vector<char> kg_data;

////////////////////////////////////////////////////
cyclesphi_data g_cyclesphi_data;      //;[PAHTRACER_DATA_SIZE];
cyclesphi_data g_cyclesphi_data_rcv;  //[PAHTRACER_DATA_SIZE];

#define STATE_NONE 0
#define STATE_RENDERING 1
#define STATE_RENDERED 2
#define STATE_SENDING 3
#define STATE_SENDED 4

double g_previousTime[3] = {0, 0, 0};
int g_frameCount[3] = {0, 0, 0};

int g_world_rank = 0;

////////////////////////////////////////////////////

bool displayFPS(int type)
{
  double currentTime = omp_get_wtime();
  g_frameCount[type]++;

  if (currentTime - g_previousTime[type] >= 3.0) {

    //#pragma omp critical
    printf("Display %d/%d: FPS: %.2f \n",
           g_world_rank,
           type,
           (double)g_frameCount[type] / (currentTime - g_previousTime[type]));
    g_frameCount[type] = 0;
    g_previousTime[type] = omp_get_wtime();

    return true;
  }

  return false;
}

////////////////////////////////////////////////////////////
#if defined(WITH_CLIENT_MPI_FILE_1)
  void bcast(void *data, size_t size)
  {
  #if defined(WITH_CLIENT_FILE)
    if (g_world_rank == 0)
    {
      NET_NAMESPACE::read_data_kernelglobal((char *)data, size);
      const size_t unit_giga = 1024L * 1024L * 128L;
      size_t size_sended = 0;
      while (size - size_sended > unit_giga) {
        MPI_Bcast((char *)data + size_sended, unit_giga, MPI_BYTE, 0, MPI_COMM_WORLD);
        size_sended += unit_giga;
      }
      printf("[%d] bcast %zu bytes\n", g_world_rank, size);
      MPI_Bcast((char *)data + size_sended, size - size_sended, MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else
    {
      printf("[%d] bcast %zu bytes\n", g_world_rank, size);
      MPI_Bcast((char *)data, size, MPI_BYTE, 0, MPI_COMM_WORLD);
    }
  #endif
  }
#else
  void bcast(void *data, size_t size)
  {
  #if defined(WITH_CLIENT_FILE)
    NET_NAMESPACE::read_data_kernelglobal((char *)data, size);
  #endif

  #ifdef WITH_CLIENT_MPI_SOCKET
    if (g_world_rank == 0)
  #endif

  #if defined(WITH_CLIENT_SOCKET) || defined(WITH_CLIENT_MPI_SOCKET)
      NET_NAMESPACE::recv_data_cam((char *)data, size);
  #endif

  #if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
    const size_t unit_giga = 1024L * 1024L * 128L;

    size_t size_sended = 0;
    while (size - size_sended > unit_giga) {
      MPI_Bcast((char *)data + size_sended, unit_giga, MPI_BYTE, 0, MPI_COMM_WORLD);
      size_sended += unit_giga;
    }

    MPI_Bcast((char *)data + size_sended, size - size_sended, MPI_BYTE, 0, MPI_COMM_WORLD);
  #endif
  }
#endif

void write_data_kernelglobal(void *data, size_t size)
{
#ifdef WITH_CLIENT_MPI
#endif

#if defined(WITH_CLIENT_FILE)  //|| defined(WITH_CLIENT_SOCKET) || defined(WITH_CLIENT_MPI_SOCKET)
  NET_NAMESPACE::write_data_kernelglobal(data, size);
#endif
}

void read_data_kernelglobal(void *data, size_t size)
{
#if defined(WITH_CLIENT_FILE) || defined(WITH_CLIENT_SOCKET) || defined(WITH_CLIENT_MPI) || \
    defined(WITH_CLIENT_MPI_SOCKET)
  bcast(data, size);
#endif
}

void close_kernelglobal()
{
  NET_NAMESPACE::close_kernelglobal();
}

///////////////////////////////

bool break_loop(client_kernel_struct &data)
{
#if defined(WITH_CLIENT_FILE)
  return data.client_tag == CLIENT_TAG_CYCLES_path_trace;
#endif

#if defined(WITH_CLIENT_SOCKET) || defined(WITH_CLIENT_MPI_SOCKET)
  if (NET_NAMESPACE::is_error()) {
    NET_NAMESPACE::server_close();
    NET_NAMESPACE::client_close();
  }
#endif

#if defined(WITH_CLIENT_SOCKET) || defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
  return data.client_tag == CLIENT_TAG_CYCLES_close_connection;
#endif
}

void const_copy(client_kernel_struct &data)
{
  if (kg_data.size() != data.client_const_copy_data.size)
    kg_data.resize(data.client_const_copy_data.size);

  bcast(kg_data.data(), data.client_const_copy_data.size);

  DEV_NAMESPACE::const_copy(CLIENT_DEVICE_ID,
                            DEV_NAMESPACE::get_kernel_globals_cpu(),
                            data.client_const_copy_data.name,
                            kg_data.data(),
                            data.client_const_copy_data.size);
}

void tex_copy_internal(client_kernel_struct &data, bool read)
{
  bool spec_setts = true;

#if defined(WITH_CLIENT_CUDA_CPU_STAT2) && !defined(WITH_CLIENT_CUDA_CPU_STAT2v2)
  spec_setts = false;
#endif

  DEV_NAMESPACE::set_ptr_map(data.client_tex_copy_data.name,
                             data.client_tex_copy_data.mem,
                             DEV_NAMESPACE::mem_alloc(CLIENT_DEVICE_ID,
                                                      data.client_tex_copy_data.name,
                                                      NULL,
                                                      data.client_tex_copy_data.mem_size,
                                                      spec_setts));

#if defined(WITH_CUDA_STATv2) || defined(WITH_HIP_STATv2) || defined(WITH_CLIENT_CUDA_CPU_STAT2v2)
  char *cuda_mem_bin = (char *)DEV_NAMESPACE::get_ptr_map(data.client_tex_copy_data.mem);

  if (!DEV_NAMESPACE::check_unimem(data.client_tex_copy_data.name)) {
    cuda_mem_bin = DEV_NAMESPACE::host_alloc(data.client_tex_copy_data.name,
                                             data.client_tex_copy_data.mem,
                                             data.client_tex_copy_data.mem_size);
  }

#  if defined(WITH_CLIENT_CUDA_CPU_STAT2v2)
  DEV_NAMESPACE::set_ptr_map(data.client_tex_copy_data.mem,
                             DEV_NAMESPACE::mem_alloc(CLIENT_DEVICE_ID,
                                                      data.client_tex_copy_data.name,
                                                      cuda_mem_bin,
                                                      data.client_tex_copy_data.mem_size));

  char *omp_mem_bin = cuda_mem_bin;
#  endif

#else
  char *cuda_mem_bin = DEV_NAMESPACE::host_alloc(data.client_tex_copy_data.name,
                                                 data.client_tex_copy_data.mem,
                                                 data.client_tex_copy_data.mem_size);
#endif

  if (read) {
#if defined(WITH_CLIENT_CUDA_CPU_STAT2)
    bcast(&omp_mem_bin[0], data.client_tex_copy_data.mem_size);

#elif defined(WITH_CLIENT_CUDA_CPU_STAT)
    bcast(&cuda_mem_bin[0], data.client_tex_copy_data.mem_size);
    memcpy(&omp_mem_bin[0], &cuda_mem_bin[0], data.client_tex_copy_data.mem_size);
#else

    bcast(cuda_mem_bin, data.client_tex_copy_data.mem_size);
#endif
  }

  DEV_NAMESPACE::tex_copy(CLIENT_DEVICE_ID,
                          DEV_NAMESPACE::get_kernel_globals_cpu(),
                          data.client_tex_copy_data.name,
                          DEV_NAMESPACE::get_ptr_map(data.client_tex_copy_data.mem),
#if defined(WITH_CLIENT_CUDA_CPU_STAT2)
                          &omp_mem_bin[0],
#else
                          &cuda_mem_bin[0],
#endif
                          data.client_tex_copy_data.data_size,
                          data.client_tex_copy_data.mem_size);

#if defined(WITH_CUDA_STATv2) || defined(WITH_HIP_STATv2) || defined(WITH_CLIENT_CUDA_CPU_STAT2v2)
  if (!DEV_NAMESPACE::check_unimem(data.client_tex_copy_data.name))
#endif
    DEV_NAMESPACE::host_free(
        data.client_tex_copy_data.name, data.client_tex_copy_data.mem, cuda_mem_bin);
}

void tex_copy(client_kernel_struct &data)
{
  tex_copy_internal(data, true);
}

void tex_copy_internal_stat(const char *name, unsigned long long int id)
{
  client_kernel_struct _data;
  strcpy(_data.client_tex_copy_data.name, name);
  DEVICE_PTR device_ptr_max = -1;
  _data.client_tex_copy_data.mem = device_ptr_max - id;
  _data.client_tex_copy_data.data_size = 1;
  _data.client_tex_copy_data.mem_size = 1 * sizeof(unsigned int);

  tex_copy_internal(_data, false);
}

void alloc_kg(client_kernel_struct &data)
{
  // if (kernel::dev_kernel_data != NULL) {
  //  delete kernel::dev_kernel_data;
  //}

  // kernel::dev_kernel_data = new kernel::DevKernelData();

  // DEV_NAMESPACE::get_kernel_globals_cpu() =
  DEV_NAMESPACE::alloc_kg(CLIENT_DEVICE_ID);

#if defined(WITH_CPU_STAT) || defined(WITH_CUDA_STAT) || defined(WITH_HIP_STAT) || \
    defined(WITH_OPTIX_STAT)
  // KERNEL_DATA_ARRAY(uint, hits)
  tex_copy_internal_stat("hits", 1);
  // KERNEL_DATA_ARRAY(uint, bounces)
  tex_copy_internal_stat("bounces", 2);
  // KERNEL_DATA_ARRAY(uint, diffuse_bounces)
  tex_copy_internal_stat("diffuse_bounces", 3);
  // KERNEL_DATA_ARRAY(uint, glossy_bounces)
  tex_copy_internal_stat("glossy_bounces", 4);
  // KERNEL_DATA_ARRAY(uint, transmission_bounces)
  tex_copy_internal_stat("transmission_bounces", 5);
  // KERNEL_DATA_ARRAY(uint, transparent_bounces)
  tex_copy_internal_stat("transparent_bounces", 6);
  // KERNEL_DATA_ARRAY(uint, volume_bounces)
  tex_copy_internal_stat("volume_bounces", 7);
  // KERNEL_DATA_ARRAY(uint, volume_bounds_bounces)
  tex_copy_internal_stat("volume_bounds_bounces", 8);
  // KERNEL_DATA_ARRAY(uint, bvh_nodes_max_level)
  tex_copy_internal_stat("bvh_nodes_max_level", 9);
#endif
}

void free_kg(client_kernel_struct &data)
{
  DEV_NAMESPACE::free_kg(CLIENT_DEVICE_ID, DEV_NAMESPACE::get_kernel_globals_cpu());
}

void mem_alloc(client_kernel_struct &data)
{
  DEV_NAMESPACE::set_ptr_map(
      data.client_mem_data.name,
      data.client_mem_data.mem,
      DEV_NAMESPACE::mem_alloc(
          CLIENT_DEVICE_ID, data.client_mem_data.name, NULL, data.client_mem_data.memSize));
}

void mem_alloc_sub_ptr(client_kernel_struct &data)
{
  // DEV_NAMESPACE::get_ptr_map(data.client_mem_data.mem_sub] = (DEVICE_PTR)(
  //    (char *)DEV_NAMESPACE::get_ptr_map(data.client_mem_data.mem] +
  //    data.client_mem_data.offset);
}

void mem_copy_to(client_kernel_struct &data)
{
  char *temp = NULL;

  if (DEV_NAMESPACE::check_unimem(data.client_mem_data.name)) {
    temp = (char *)DEV_NAMESPACE::get_ptr_map(data.client_mem_data.mem);
  }
  else {
    temp = DEV_NAMESPACE::host_alloc(
        data.client_mem_data.name, data.client_mem_data.mem, data.client_mem_data.memSize);
  }

  bcast(&temp[0], data.client_mem_data.memSize);

  if (!DEV_NAMESPACE::check_unimem(data.client_mem_data.name)) {
    DEV_NAMESPACE::mem_copy_to(CLIENT_DEVICE_ID,
                               &temp[0],
                               DEV_NAMESPACE::get_ptr_map(data.client_mem_data.mem),
                               data.client_mem_data.memSize,
                               NULL);

    DEV_NAMESPACE::host_free(data.client_mem_data.name, data.client_mem_data.mem, temp);
  }
}

void mem_zero(client_kernel_struct &data)
{
  DEV_NAMESPACE::mem_zero(data.client_mem_data.name,
                          CLIENT_DEVICE_ID,
                          DEV_NAMESPACE::get_ptr_map(data.client_mem_data.mem),
                          data.client_mem_data.memSize);
}

void mem_free(client_kernel_struct &data)
{
  if (DEV_NAMESPACE::get_ptr_map(data.client_mem_data.mem)) {
    DEV_NAMESPACE::mem_free(data.client_mem_data.name,
                            CLIENT_DEVICE_ID,
                            DEV_NAMESPACE::get_ptr_map(data.client_mem_data.mem),
                            data.client_mem_data.memSize);
    // kernel::dev_kernel_data->ptr_map.erase(data.client_mem_data.mem);
  }
}

void tex_free(client_kernel_struct &data)
{
  if (DEV_NAMESPACE::get_ptr_map(data.client_mem_data.mem)) {
    DEV_NAMESPACE::tex_free(CLIENT_DEVICE_ID,
                            DEV_NAMESPACE::get_kernel_globals_cpu(),
                            data.client_mem_data.name,
                            DEV_NAMESPACE::get_ptr_map(data.client_mem_data.mem),
                            data.client_mem_data.memSize);
    // kernel::dev_kernel_data->ptr_map.erase(data.client_mem_data.mem);
  }
}

float *get_camera_view_matrix()
{
  return g_cyclesphi_data.cam.transform_inverse_view_matrix;
}

float *get_camera_rcv_view_matrix()
{
  return g_cyclesphi_data_rcv.cam.transform_inverse_view_matrix;
}

float get_camera_lens()
{
  return g_cyclesphi_data.cam.lens;
}

float get_camera_near_clip()
{
  return g_cyclesphi_data.cam.clip_start;
}

float get_camera_far_clip()
{
  return g_cyclesphi_data.cam.clip_end;
}

#if defined(WITH_CLIENT_RENDERENGINE_VR) || defined(WITH_CLIENT_ULTRAGRID)
float *get_camera_right_view_matrix()
{
  return g_cyclesphi_data.cam_right.transform_inverse_view_matrix;
}

float *get_camera_right_rcv_view_matrix()
{
  return g_cyclesphi_data_rcv.cam_right.transform_inverse_view_matrix;
}

float get_camera_right_lens()
{
  return g_cyclesphi_data.cam_right.lens;
}

float get_camera_right_near_clip()
{
  return g_cyclesphi_data.cam_right.clip_start;
}

float get_camera_right_far_clip()
{
  return g_cyclesphi_data.cam_right.clip_end;
}
#endif

int *get_image_size()
{
  return (int *)&g_cyclesphi_data.width;
}

int get_image_sample()
{
  return g_cyclesphi_data.step_samples;
}

#ifdef WITH_CLIENT_FILE

void socket_step_cam(client_kernel_struct &data)
{
  if (data.world_rank == 0) {
#  if defined(WITH_CLIENT_ULTRAGRID)

    //#ifndef WITH_CLIENT_ULTRAGRID_LIB
    cesnet_set_camera_data(&cyclesphi_data_rcv);
    //#endif

#  elif defined(WITH_CLIENT_RENDERENGINE_EMULATE)

    // memset((char *)&cyclesphi_data_rcv, 0, sizeof(cyclesphi_data));

    g_cyclesphi_data_rcv.step_samples = data.client_path_trace_data.num_samples;
    g_cyclesphi_data_rcv.width = data.client_path_trace_data.tile_w;
    g_cyclesphi_data_rcv.height = data.client_path_trace_data.tile_h;

    const char *env_samples = getenv("DEBUG_SAMPLES");
    if (env_samples != NULL) {
      g_cyclesphi_data_rcv.step_samples = atoi(env_samples);
    }

    const char *env_res_w = getenv("DEBUG_RES_W");
    const char *env_res_h = getenv("DEBUG_RES_H");
    if (env_res_w != NULL && env_res_h != NULL) {
      g_cyclesphi_data_rcv.width = atoi(env_res_w);
      g_cyclesphi_data_rcv.height = atoi(env_res_h);
    }

#  else
#    if defined(WITH_CLIENT_RENDERENGINE) && !defined(WITH_CLIENT_ULTRAGRID)
    kernel::tcp::recv_data_cam((char *)&g_cyclesphi_data_rcv, sizeof(cyclesphi_data));
#    else
    NET_NAMESPACE::recv_data_cam((char *)&g_cyclesphi_data_rcv, sizeof(cyclesphi_data));
#    endif
#  endif
  }
}

void socket_step_cam_ack(client_kernel_struct &data, char ack)
{
  if (data.world_rank == 0) {
#  if !defined(WITH_CLIENT_RENDERENGINE_EMULATE) && !defined(WITH_CLIENT_ULTRAGRID) && \
      !defined(WITH_CLIENT_RENDERENGINE)
    NET_NAMESPACE::send_data_data((char *)&ack, sizeof(char));
#  endif
  }
}

void tcp_exit(client_kernel_struct &data)
{
  if (data.world_rank == 0) {
#  if !defined(WITH_CLIENT_RENDERENGINE_EMULATE) && !defined(WITH_CLIENT_ULTRAGRID)

#    if defined(WITH_CLIENT_RENDERENGINE) && !defined(WITH_CLIENT_ULTRAGRID)
    kernel::tcp::recv_data_cam((char *)&g_cyclesphi_data_rcv, sizeof(cyclesphi_data));

    char ack = 0;
    kernel::tcp::send_data_data((char *)&ack, sizeof(char));
    kernel::tcp::client_close();
    kernel::tcp::server_close();
#    else
    NET_NAMESPACE::recv_data_cam((char *)&g_cyclesphi_data_rcv, sizeof(cyclesphi_data));

    char ack = 0;
    NET_NAMESPACE::send_data_data((char *)&ack, sizeof(char));
    NET_NAMESPACE::client_close();
    NET_NAMESPACE::server_close();
#    endif
#  endif
  }
}

#endif

bool cam_change(client_kernel_struct &data)
{
  if (memcmp(&g_cyclesphi_data, &g_cyclesphi_data_rcv, sizeof(cyclesphi_data))) {

    memcpy(&g_cyclesphi_data, &g_cyclesphi_data_rcv, sizeof(cyclesphi_data));

    return true;
  }

  return false;
}

bool image_size_change(client_kernel_struct &data)
{
  int *size = get_image_size();
  int num_samples = get_image_sample();
  return data.client_path_trace_data.tile_w != size[0] ||
         data.client_path_trace_data.tile_h != size[1] ||
         data.client_path_trace_data.num_samples != num_samples;
}

void set_bounces(client_kernel_struct &data)
{
  if (data.world_rank == 0) {
#if 1  // defined(WITH_CLIENT_RENDERENGINE_EMULATE)

    const char *bounces = getenv("SET_BOUNCES");
    if (bounces != NULL) {
      int min_bounce = util_get_int_from_env_array(bounces, 0);
      int max_bounce = util_get_int_from_env_array(bounces, 1);
      int max_diffuse_bounce = util_get_int_from_env_array(bounces, 2);
      int max_glossy_bounce = util_get_int_from_env_array(bounces, 3);
      int max_transmission_bounce = util_get_int_from_env_array(bounces, 4);
      int max_volume_bounce = util_get_int_from_env_array(bounces, 5);
      int max_volume_bounds_bounce = util_get_int_from_env_array(bounces, 6);

      int transparent_min_bounce = util_get_int_from_env_array(bounces, 7);
      int transparent_max_bounce = util_get_int_from_env_array(bounces, 8);
      int use_lamp_mis = util_get_int_from_env_array(bounces, 9);

      DEV_NAMESPACE::set_bounces(CLIENT_DEVICE_ID,
                                 DEV_NAMESPACE::get_kernel_globals_cpu(),
                                 &kg_data[0],
                                 min_bounce,
                                 max_bounce,
                                 max_diffuse_bounce,
                                 max_glossy_bounce,
                                 max_transmission_bounce,
                                 max_volume_bounce,
                                 max_volume_bounds_bounce,
                                 transparent_min_bounce,
                                 transparent_max_bounce,
                                 use_lamp_mis);
    }
#endif
  }
}

void socket_step_data(client_kernel_struct &data)
{

#if defined(WITH_CLIENT_RENDERENGINE) || defined(WITH_CLIENT_ULTRAGRID) || \
    defined(WITH_CLIENT_RENDERENGINE_EMULATE)
  char *cdata = (char *)&g_cyclesphi_data;
#else
  float size[2];
  size[0] = data.client_path_trace_data.tile_w;
  size[1] = data.client_path_trace_data.tile_h;
  char *cdata = (char *)&size;
#endif

  if (kg_data.size() == 0)
    kg_data.resize(DEV_NAMESPACE::get_size_data(DEV_NAMESPACE::get_kernel_globals_cpu()));

  if (data.world_rank == 0) {
    DEV_NAMESPACE::socket_step(
        CLIENT_DEVICE_ID, DEV_NAMESPACE::get_kernel_globals_cpu(), &kg_data[0], cdata);
  }

#if defined(WITH_CLIENT_MPI_FILE)
  MPI_Bcast((char *)&kg_data[0], kg_data.size(), MPI_BYTE, 0, MPI_COMM_WORLD);

  if (data.world_rank > 0) {

    DEV_NAMESPACE::const_copy(CLIENT_DEVICE_ID,
                              DEV_NAMESPACE::get_kernel_globals_cpu(),
                              "data",
                              (char *)&kg_data[0],
                              kg_data.size());
  }

#endif
}

void socket_step_image_size(client_kernel_struct &data, int w_old, int h_old)
{
  // if (data.world_rank == 0) {
  client_kernel_struct buff_data;
  strcpy(buff_data.client_mem_data.name, "RenderBuffers");

  buff_data.client_mem_data.mem = data.client_path_trace_data.buffer;
  buff_data.client_mem_data.offset = 0;
  buff_data.client_mem_data.memSize = w_old * h_old * data.client_path_trace_data.pass_stride *
                                      sizeof(float);

  mem_free(buff_data);

  buff_data.client_mem_data.memSize = data.client_path_trace_data.tile_w *
                                      data.client_path_trace_data.tile_h *
                                      data.client_path_trace_data.pass_stride * sizeof(float);

#if defined(WITH_CLIENT_RENDERENGINE_VR) || \
    (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
  buff_data.client_mem_data.memSize *= 2;
#endif

  mem_alloc(buff_data);
  mem_zero(buff_data);
  //}
}

void set_device(int device, int world_rank, int world_size)
{
  g_world_rank = world_rank;

  DEV_NAMESPACE::set_device(device, world_rank, world_size);
}

void load_textures(client_kernel_struct &data)
{
  DEV_NAMESPACE::load_textures(CLIENT_DEVICE_ID,
                               DEV_NAMESPACE::get_kernel_globals_cpu(),
                               data.client_load_textures_data.texture_info_size);
  // kernel::dev_kernel_data->ptr_map);
}

void build_bvh(client_kernel_struct &data)
{
#ifdef WITH_CLIENT_OPTIX
  char *optix_temp = kernel::optix::host_alloc("build_bvh_temp",
                                               data.client_build_bvh_data.build_input,
                                               data.client_build_bvh_data.build_size);

  bcast(&optix_temp[0], data.client_build_bvh_data.build_size);

  kernel::optix::build_bvh(data.client_build_bvh_data.operation,
                           optix_temp,
                           data.client_build_bvh_data.build_input,
                           data.client_build_bvh_data.build_size,
                           data.client_build_bvh_data.num_motion_steps);

  kernel::optix::host_free("build_bvh_temp", data.client_build_bvh_data.build_input, optix_temp);
#endif
}

#ifdef WITH_CLIENT_MPI_CACHE
#  define CACHING_DISABLED 0
#  define CACHING_RECORD 1
#  define CACHING_PREVIEW 2

int g_current_frame = 0;
int g_current_frame_preview = -1;
int g_caching_enabled = 0;
#endif

void frame_info(client_kernel_struct &data)
{
#ifdef WITH_CLIENT_MPI_CACHE
  g_current_frame = data.client_frame_info_data.current_frame;
  g_current_frame_preview = data.client_frame_info_data.current_frame_preview;
  g_caching_enabled = data.client_frame_info_data.caching_enabled;
#endif

#ifdef WITH_CLIENT_MPI_CACHE
  if ((g_caching_enabled == CACHING_PREVIEW &&
       (g_current_frame_preview % (data.world_size - 1)) == (data.world_rank - 1)) ||
      (g_caching_enabled != CACHING_PREVIEW))
#endif
  {
    // printf("current frame: %d\n", data.client_frame_info_data.current_frame);
    DEV_NAMESPACE::frame_info(data.client_frame_info_data.current_frame,
                              data.client_frame_info_data.current_frame_preview,
                              data.client_frame_info_data.caching_enabled);
  }
}

void tex_info(client_kernel_struct &data)
{
  bool spec_setts = true;

#if defined(WITH_CLIENT_CUDA_CPU_STAT2)
  spec_setts = false;
#endif

  DEV_NAMESPACE::set_ptr_map(
      data.client_tex_info_data.name,
      data.client_tex_info_data.mem,
      DEV_NAMESPACE::tex_info_alloc(CLIENT_DEVICE_ID,
                                    //(data.client_tex_info_data.size > 0) ? &cuda_temp[0] : NULL,
                                    NULL,
                                    data.client_tex_info_data.size,
                                    data.client_tex_info_data.name,
                                    data.client_tex_info_data.data_type,
                                    data.client_tex_info_data.data_elements,
                                    data.client_tex_info_data.interpolation,
                                    data.client_tex_info_data.extension,
                                    data.client_tex_info_data.data_width,
                                    data.client_tex_info_data.data_height,
                                    data.client_tex_info_data.data_depth,
                                    spec_setts));

  if (DEV_NAMESPACE::check_unimem(data.client_tex_info_data.name)) {
    // kernel::file_read_data_kernelglobal((char
    // *)kernel::cuda_mpiData->ptr_map[data.client_tex_info_data.mem],
    // data.client_tex_info_data.size);

    bcast((char *)DEV_NAMESPACE::get_ptr_map(data.client_tex_info_data.mem),
          data.client_tex_info_data.size);

    // kernel::cuda_tex_info_copy(data.client_tex_info_data.name,
    //                            NULL,
    //                            kernel::cuda_mpiData->ptr_map[data.client_tex_info_data.mem],
    //                            data.client_tex_info_data.size);
  }
  else {

    char *cuda_temp = DEV_NAMESPACE::host_alloc(data.client_tex_info_data.name,
                                                data.client_tex_info_data.mem,
                                                data.client_tex_info_data.size);

#if defined(WITH_CLIENT_CUDA_CPU_STAT2)
    // bcast(&cuda_temp[0], data.client_tex_info_data.size);
    // memcpy(&omp_temp[0], &cuda_temp[0], data.client_tex_info_data.size);
    bcast(&omp_temp[0], data.client_tex_info_data.size);
    memcpy(&cuda_temp[0], &omp_temp[0], data.client_tex_info_data.size);

#elif defined(WITH_CLIENT_CUDA_CPU_STAT)
    bcast(&cuda_temp[0], data.client_tex_info_data.size);
    memcpy(&omp_temp[0], &cuda_temp[0], data.client_tex_info_data.size);
#else

    bcast(&cuda_temp[0], data.client_tex_info_data.size);

#endif

    //#  if !defined(WITH_CLIENT_CUDA_CPU_STAT2)
    DEV_NAMESPACE::tex_info_copy(data.client_tex_info_data.name,
                                 (data.client_tex_info_data.size > 0) ? &cuda_temp[0] : NULL,
                                 DEV_NAMESPACE::get_ptr_map(data.client_tex_info_data.mem),
                                 data.client_tex_info_data.size,
                                 data.client_tex_info_data.data_type);
    //#  endif

    DEV_NAMESPACE::host_free(
        data.client_tex_info_data.name, data.client_tex_info_data.mem, cuda_temp);
  }
}

// double get_path_time_acc()
//{
//  return g_path_time_acc;
//}

#if 0
void path_trace_buffer_render(client_kernel_struct &data,
                              int num_samples,
                              DEVICE_PTR dev_pixels_node,
                              char *pixels_node,
                              char *pixels,
                              char *signal_value)
{
  size_t offsetSample = 0;
  size_t sizeSample = sizeof(int);

  int reqFinished = 0;
  int sample_finished_cpu = 0;

  int start_sample = data.client_path_trace_data.start_sample;

  int end_sample = data.client_path_trace_data.start_sample + num_samples;

#  if defined(WITH_CLIENT_CUDA_CPU_STAT)
  int pass_stride = DEV_NAMESPACE::get_pass_stride(CLIENT_DEVICE_ID,
                                                   DEV_NAMESPACE::get_kernel_globals_cpu());
#  else

  int pass_stride = DEV_NAMESPACE::get_pass_stride(CLIENT_DEVICE_ID,
                                                   DEV_NAMESPACE::get_kernel_globals_cpu());

#  endif

  int offset = data.client_path_trace_data.offset;
  int stride = data.client_path_trace_data.stride;

  int tile_x = data.client_path_trace_data.tile_x;
  int tile_w = data.client_path_trace_data.tile_w;

  int tile_y = data.client_path_trace_data.tile_y;
  int tile_h = data.client_path_trace_data.tile_h;

  ////////////////////////////one node///////////////////////////////////
  int dev_node = data.world_rank;           // - 1;
  int devices_size_node = data.world_size;  // - 1;

#  ifdef WITH_LOAD_BALANCING_COLUMN

  int tile_step_node = (int)((float)tile_w / (float)devices_size_node);

  int tile_last_node = tile_w - (devices_size_node - 1) * tile_step_node;

  int tile_x_node = tile_x + tile_step_node * dev_node;
  int tile_w_node = (devices_size_node - 1 == dev_node) ? tile_last_node : tile_step_node;

  int tile_y_node = tile_y;
  int tile_h_node = tile_h;

#  else
  int tile_step_node = (int)((float)tile_h / (float)devices_size_node);

  int tile_last_node = tile_h - (devices_size_node - 1) * tile_step_node;

  int tile_y_node = tile_y + tile_step_node * dev_node;
  int tile_h_node = (devices_size_node - 1 == dev_node) ? tile_last_node : tile_step_node;

  int tile_x_node = tile_x;
  int tile_w_node = tile_w;
#  endif

#  ifdef CLIENT_MPI_LOAD_BALANCING_LINES

#    ifdef WITH_LOAD_BALANCING_COLUMN
  std::vector<float> image_time(tile_w);
#    else
  std::vector<float> image_time(tile_h);
#    endif

  int sample_step = 1;
  const char *env_sample_step = getenv("LB_SAMPLES");
  if (env_sample_step != NULL) {
    sample_step = atoi(env_sample_step);
  }
  // for (int s = start_sample; s < end_sample; s += sample_step) {
  path_trace_buffer_time(
      data, sample_step, &image_time[0], tile_x_node, tile_w_node, tile_y_node, tile_h_node);

  double image_total_time = 0;
  for (int i = 0; i < image_time.size(); i++) {
    image_total_time += image_time[i];

    // if (data.world_rank == 0) {
    //     printf("part %d: %f [s] \n", i, image_time[i]);
    //  fflush(0);
    //}
  }

  int image_time_id = 0;
  double image_time_reminder = 0.0;

#    ifdef WITH_LOAD_BALANCING_COLUMN
  for (int id = 0; id < devices_size_node; id++) {
    tile_x_node = image_time_id;

    double dev_time = image_time_reminder;
    double avg_time = image_total_time / (double)devices_size_node;

    for (int i = tile_x_node; i < image_time.size(); i++) {
      dev_time += image_time[i];

      image_time_id = i + 1;

      if (dev_time > avg_time)
        break;
    }

    image_time_reminder = dev_time - avg_time;

    tile_w_node = (devices_size_node - 1 == dev_node) ? tile_w - tile_x_node :
                                                        image_time_id - tile_x_node;

    if (id == dev_node) {
      printf("part %d: %f - %f [s], %d - %d \n", id, dev_time, avg_time, tile_x_node, tile_w_node);
      break;
    }
  }

#    else
  for (int id = 0; id < devices_size_node; id++) {
    tile_y_node = image_time_id;

    double dev_time = image_time_reminder;
    double avg_time = image_total_time / (double)devices_size_node;

    for (int i = tile_y_node; i < image_time.size(); i++) {
      dev_time += image_time[i];

      image_time_id = i + 1;

      if (dev_time > avg_time)
        break;
    }

    image_time_reminder = dev_time - avg_time;

    tile_h_node = (devices_size_node - 1 == dev_node) ? tile_h - tile_y_node :
                                                        image_time_id - tile_y_node;

    if (id == dev_node)
      break;
  }
#    endif

  data.client_path_trace_data.start_sample = data.client_path_trace_data.start_sample +
                                             sample_step;
  //}

  // return;

  start_sample = start_sample + sample_step;

  //////////////////////
  // size_t sizeBuf_node = tile_w * tile_h *
  //                      data.client_path_trace_data.pass_stride * sizeof(float);
  // omp_mem_zero(
  //    CLIENT_DEVICE_ID, omp_mpiData->ptr_map[data.client_path_trace_data.buffer], sizeBuf_node);
  //////////////////////

#  endif

  int size_node = tile_h_node * tile_w_node;
  /////////////////////////////////////////////////////////////

  if (dev_pixels_node != NULL) {
    double t0 = omp_get_wtime();

#  if defined(WITH_CLIENT_CUDA_CPU_STAT)
    DEV_NAMESPACE::init_execution(data.client_path_trace_data.has_shadow_catcher,
                                  data.client_path_trace_data.max_shaders,
                                  data.client_path_trace_data.pass_stride,
                                  data.client_path_trace_data.kernel_features,
                                  data.client_path_trace_data.volume_stack_size);

    DEV_NAMESPACE::path_trace_stat(
        CLIENT_DEVICE_ID,
        DEV_NAMESPACE::get_kernel_globals_cpu(),
        DEV_NAMESPACE::get_kernel_globals_cpu(),
        DEV_NAMESPACE::get_ptr_map(data.client_path_trace_data.buffer],
        DEV_NAMESPACE::get_ptr_map(data.client_path_trace_data.buffer],
        (DEVICE_PTR)dev_pixels_node,
        NULL,
        start_sample,
        end_sample,
        tile_x_node,
        tile_y_node,
        offset,
        stride,
        tile_h_node,
        tile_w_node,
        (char *)&sample_finished_cpu,
        (char *)&reqFinished,
        DEV_NAMESPACE::get_cpu_threads(),
        signal_value);

//    DEV_NAMESPACE::path_trace(CLIENT_DEVICE_ID,
//                   DEV_NAMESPACE::get_kernel_globals_cpu(),
//                   DEV_NAMESPACE::get_ptr_map(data.client_path_trace_data.buffer],
//                   NULL,
//                   start_sample,
//                   end_sample,
//                   tile_x_node,
//                   tile_y_node,
//                   offset,
//                   stride,
//                   tile_h_node,
//                   tile_w_node,
//                   NULL,
//                   NULL,
//                   DEV_NAMESPACE::get_cpu_threads(),
//                   signal_value);
#  else

    DEV_NAMESPACE::init_execution(data.client_path_trace_data.has_shadow_catcher,
                                  data.client_path_trace_data.max_shaders,
                                  data.client_path_trace_data.pass_stride,
                                  data.client_path_trace_data.kernel_features,
                                  data.client_path_trace_data.volume_stack_size);

    DEV_NAMESPACE::path_trace(CLIENT_DEVICE_ID,
                              DEV_NAMESPACE::get_kernel_globals_cpu(),
                              DEV_NAMESPACE::get_ptr_map(data.client_path_trace_data.buffer),
                              (DEVICE_PTR)dev_pixels_node,
                              start_sample,
                              end_sample,
                              tile_x_node,
                              tile_y_node,
                              offset,
                              stride,
                              tile_h_node,
                              tile_w_node,
                              tile_h,
                              tile_w,
                              (char *)&sample_finished_cpu,
                              (char *)&reqFinished,
                              DEV_NAMESPACE::get_cpu_threads(),
                              signal_value);

#  endif

    //g_path_time_acc += omp_get_wtime() - t0;
  }
  else {

    size_t offsetBuf_node = (offset + tile_x_node + tile_y_node * stride) * pass_stride *
                            sizeof(float);
    size_t sizeBuf_node = size_node * pass_stride * sizeof(float);

    omp_set_nested(1);

    double t0 = omp_get_wtime();

#  if defined(WITH_CLIENT_CUDA_CPU_STAT)
    DEV_NAMESPACE::init_execution(data.client_path_trace_data.has_shadow_catcher,
                                  data.client_path_trace_data.max_shaders,
                                  data.client_path_trace_data.pass_stride,
                                  data.client_path_trace_data.kernel_features,
                                  data.client_path_trace_data.volume_stack_size);

    DEV_NAMESPACE::path_trace(CLIENT_DEVICE_ID,
                              DEV_NAMESPACE::get_kernel_globals_cpu(),
                              DEV_NAMESPACE::get_ptr_map(data.client_path_trace_data.buffer],
                              NULL,
                              start_sample,
                              end_sample,
                              tile_x_node,
                              tile_y_node,
                              offset,
                              stride,
                              tile_h_node,
                              tile_w_node,  // dev_node, devices_size_node,
                              tile_h,
                              tile_w,
                              (char *)&sample_finished_cpu,
                              (char *)&reqFinished,
                              DEV_NAMESPACE::get_cpu_threads(),
                              signal_value);

#  else

    DEV_NAMESPACE::init_execution(data.client_path_trace_data.has_shadow_catcher,
                                  data.client_path_trace_data.max_shaders,
                                  data.client_path_trace_data.pass_stride,
                                  data.client_path_trace_data.kernel_features,
                                  data.client_path_trace_data.volume_stack_size);

    DEV_NAMESPACE::path_trace(CLIENT_DEVICE_ID,
                              DEV_NAMESPACE::get_kernel_globals_cpu(),
                              DEV_NAMESPACE::get_ptr_map(data.client_path_trace_data.buffer),
                              NULL,
                              start_sample,
                              end_sample,
                              tile_x_node,
                              tile_y_node,
                              offset,
                              stride,
                              tile_h_node,
                              tile_w_node,
                              tile_h,
                              tile_w,
                              (char *)&sample_finished_cpu,
                              (char *)&reqFinished,
                              DEV_NAMESPACE::get_cpu_threads(),
                              signal_value);

#  endif

    //g_path_time_acc += omp_get_wtime() - t0;
  }
}

void path_trace(client_kernel_struct &data)
{
  size_t pix_type_size = SIZE_UCHAR4;

  if (data.world_rank == 0) {

#  if defined(WITH_CLIENT_CUDA_CPU_STAT)
    memcpy(get_camera_rcv_view_matrix(),
           (char *)DEV_NAMESPACE::get_camera_matrix(DEV_NAMESPACE::get_kernel_globals_cpu()),
           sizeof(float) * 12);
#  else

    memcpy(get_camera_rcv_view_matrix(),
           (char *)DEV_NAMESPACE::get_camera_matrix(DEV_NAMESPACE::get_kernel_globals_cpu()),
           sizeof(float) * 12);

#  endif
  }

  bool buf_new = true;
  double sender_time = -1;

  while (true) {
    if (buf_new) {
      if (data.world_rank == 0) {
        //#  pragma omp parallel num_threads(2)
        {
          // int tid = omp_get_thread_num();
          // if(tid==0) {
          if (sender_time == -1) {
            socket_step_cam(data);
            socket_step_cam_ack(data, -1);
          }
          cam_change(data);
          //}
        }

        data.client_path_trace_data.num_samples = get_image_sample();

        int *size = get_image_size();
        int w_old = data.client_path_trace_data.tile_w;
        int h_old = data.client_path_trace_data.tile_h;

        data.client_path_trace_data.tile_w = size[0];
        data.client_path_trace_data.tile_h = size[1];
        data.client_path_trace_data.stride = data.client_path_trace_data.tile_w;

        socket_step_image_size(data, w_old, h_old);
        socket_step_data(data);
        set_bounces(data);
        buf_new = false;
      }
      int num_samples = get_image_sample();

      size_t pix_size = data.client_path_trace_data.tile_w * data.client_path_trace_data.tile_h *
                        pix_type_size;

#  if defined(WITH_CLIENT_RENDERENGINE_VR) || \
      (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
      pix_size *= 2;  // left+right
#  endif

      //#if defined(WITH_CLIENT_ULTRAGRID) || defined(WITH_CLIENT_YUV)
      //      //std::vector<char> pixels_yuv(pix_size);
      //#  endif

#  ifdef WITH_NVPIPE
      std::vector<char> pixels_denoise(pix_size);
#  endif

      //#ifdef WITH_CLIENT_ULTRAGRID
      //      char *pixels_node1 = NULL;
      //      char *pixels_node2 = NULL;
      //
      //      while(true) {
      //        int s = cesnet_get_buffers(&pixels_node1, &pixels_node2);
      //        if(s > 0) {
      //          //DEV_NAMESPACE::host_register("pixels_node1", pixels_node1, s);
      //          //DEV_NAMESPACE::host_register("pixels_node2", pixels_node2, s);
      //          break;
      //        }
      //      }
      //
      //      char *pixels_node[2];
      //      pixels_node[0] = pixels_node1;
      //      pixels_node[1] = pixels_node2;
      //#else
      // double buffer

#  if defined(WITH_CUDA_BUFFER_MANAGED) || defined(WITH_HIP_BUFFER_MANAGED) || \
      defined(WITH_CLIENT_UNIMEM)
      char *pixels_node = DEV_NAMESPACE::host_alloc("pixels_node", NULL, 2 * pix_size, 1);
#  else
      char *pixels_node = DEV_NAMESPACE::host_alloc("pixels_node", NULL, 2 * pix_size, 2);
#  endif

      char *pixels_node1 = &pixels_node[0];
      char *pixels_node2 = &pixels_node[pix_size];

      char *pixels = pixels_node1;

      /////^///
      int start_sample = data.client_path_trace_data.start_sample;

      int flag_exit = 0;
      int last_state = 0;

#  if defined(WITH_CLIENT_RENDERENGINE_EMULATE)
      const char *drt = getenv("DEBUG_REPEAT_TIME");
#  endif

      int pix_state[10];
      pix_state[0] = 0;  // state buff A
      pix_state[1] = 0;  // state buff B
      pix_state[2] = 0;  // buf_reset
      pix_state[3] = data.client_path_trace_data.start_sample;
      pix_state[4] = 0;  // buf_new, change resolution
      pix_state[5] = 0;  // start render

      omp_set_nested(1);

      omp_lock_t *lock0 = (omp_lock_t *)&pix_state[6];
      omp_lock_t *lock1 = (omp_lock_t *)&pix_state[8];

      omp_init_lock(lock0);
      omp_init_lock(lock1);

      bool do_exit = false;

#  ifdef BMP_CACHE
      size_t sended = 0;
      size_t sended_max = 1000;
      std::vector<char> sended_pixels(sended_max * pix_size);
#  endif

#  if defined(WITH_CLIENT_RENDERENGINE_EMULATE)
      if (drt == NULL)
        do_exit = true;
#  endif

#  ifdef WITH_CLIENT_RENDERENGINE_EMULATE_ONE_THREAD
      {
        int tid = 1;
#  else
#    pragma omp parallel num_threads(2)

      {
        int tid = omp_get_thread_num();
#  endif
        if (tid == 1) {

          path_trace_buffer_render(
              data, num_samples, (DEVICE_PTR)&pixels_node[0], NULL, NULL, (char *)&pix_state);

#  if defined(WITH_CLIENT_RENDERENGINE_EMULATE)
          int samples = pix_state[3];

          int w = data.client_path_trace_data.tile_w;

#    if defined(WITH_CLIENT_RENDERENGINE_VR) || \
        (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
          w *= 2;  // left+right
#    endif

          util_save_bmp(data.client_path_trace_data.offset,
                        w,
                        data.client_path_trace_data.tile_x,
                        data.client_path_trace_data.tile_y,
                        data.client_path_trace_data.tile_h,
                        w,
                        data.client_path_trace_data.pass_stride,
                        samples,
                        NULL,
                        pixels_node1,
                        0);

          util_save_bmp(data.client_path_trace_data.offset,
                        w,
                        data.client_path_trace_data.tile_x,
                        data.client_path_trace_data.tile_y,
                        data.client_path_trace_data.tile_h,
                        w,
                        data.client_path_trace_data.pass_stride,
                        samples,
                        NULL,
                        pixels_node2,
                        1);
#  endif

          do_exit = true;

#  pragma omp flush
        }

        if (tid == 0) {
          if (data.world_rank == 0) {
            int used_buffer = 0;
            while (!do_exit) {

              if (used_buffer == 0) {
                omp_set_lock(lock0);

                // printf("send1: pix_state[0] = 2, %f\n", omp_get_wtime()); fflush(0);
                // pix_state[0] = 1;
                // memcpy(&pixels[0], pixels_node1, pix_size);
                pixels = pixels_node1;

#  if defined(WITH_CLIENT_RENDERENGINE_EMULATE)
                if (sender_time < 0 && pix_state[5] != 0)
                  sender_time = omp_get_wtime();
#  endif

                // pix_state[1] = 0;
                //#  pragma omp flush
              }
              else if (used_buffer == 1) {
                omp_set_lock(lock1);

                // printf("send1: pix_state[1] = 2, %f\n", omp_get_wtime()); fflush(0);
                // pix_state[1] = 1;
                // memcpy(&pixels[0], pixels_node2, pix_size);
                pixels = pixels_node2;

#  if defined(WITH_CLIENT_RENDERENGINE_EMULATE)
                if (sender_time < 0 && pix_state[5] != 0)
                  sender_time = omp_get_wtime();
#  endif

                // pix_state[0] = 0;
                //#  pragma omp flush
              }
              else {
                // omp_unset_lock(lock0);
                // omp_unset_lock(lock1);

                if (!do_exit) {
                  usleep(100);
#  pragma omp flush
                  continue;
                }
              }

              int samples = pix_state[3];

#  if defined(WITH_CLIENT_RENDERENGINE_EMULATE)
              // const char *drt = getenv("DEBUG_REPEAT_TIME");
              if (drt == NULL ||
                  ((sender_time > 0) && (omp_get_wtime() - sender_time > atof(drt)))) {
                pix_state[4] = 1;

                // util_save_bmp(data.client_path_trace_data.offset,
                //              data.client_path_trace_data.stride,
                //              data.client_path_trace_data.tile_x,
                //              data.client_path_trace_data.tile_y,
                //              data.client_path_trace_data.tile_h,
                //              data.client_path_trace_data.tile_w,
                //              data.client_path_trace_data.pass_stride,
                //              samples,
                //              NULL,
                //              pixels,
                //              0);

                // while (!do_exit) {
                //    usleep(1000);
                //}

                // exit(0);
                omp_unset_lock(lock0);
                omp_unset_lock(lock1);
                break;
              }
#  elif defined(WITH_NVPIPE)

#    ifdef WITH_CLIENT_RENDERENGINE_VRCLIENT
              // NET_NAMESPACE::send_data_data(pixels, pix_size / 2, false);
              // NET_NAMESPACE::send_data_data(pixels + pix_size / 2, pix_size / 2);

#      if defined(WITH_CUDA_BUFFER_MANAGED) || defined(WITH_HIP_BUFFER_MANAGED) || \
          defined(WITH_CLIENT_UNIMEM)
              DEVICE_PTR dev_pixels = (DEVICE_PTR)pixels;
#      else
#        if defined(WITH_CLIENT_CUDA)
              DEVICE_PTR dev_pixels = DEV_NAMESPACE::get_host_get_device_pointer(pixels);
#        endif
#        if defined(WITH_CLIENT_OPTIX)
              DEVICE_PTR dev_pixels = DEV_NAMESPACE::get_host_get_device_pointer(pixels);
#        endif
#        if defined(WITH_CLIENT_HIP)
              DEVICE_PTR dev_pixels = DEV_NAMESPACE::get_host_get_device_pointer(pixels);
#        endif
#      endif
              NET_NAMESPACE::send_nvpipe((DEVICE_PTR)dev_pixels,
                                         &pixels_denoise[0],
                                         data.client_path_trace_data.tile_w,
                                         data.client_path_trace_data.tile_h);

              NET_NAMESPACE::send_nvpipe((DEVICE_PTR)((char *)dev_pixels + pix_size / 2),
                                         &pixels_denoise[0],
                                         data.client_path_trace_data.tile_w,
                                         data.client_path_trace_data.tile_h);
#    else
              NET_NAMESPACE::send_nvpipe((DEVICE_PTR)pixels,
                                         &pixels_denoise[0],
                                         data.client_path_trace_data.tile_w,
                                         data.client_path_trace_data.tile_h);
#    endif

#  else

#    ifdef WITH_CLIENT_ULTRAGRID
            // YUV I420
            {
              //#      ifdef _WIN32
              //                // usleep(2000000);
              //#      endif

              int tile_w = data.client_path_trace_data.tile_w;
              int tile_h = data.client_path_trace_data.tile_h;

#      if defined(WITH_CLIENT_RENDERENGINE_VR) || !defined(WITH_CLIENT_RENDERENGINE)
              tile_w *= 2;
#      endif

#      if defined(WITH_RGBA_FORMAT) || defined(WITH_CLIENT_ULTRAGRID_LIB)
              // rendering buffer - results in RGBA format
              int buf_type = cesnet_set_render_buffer_rgba(
                  (unsigned char *)pixels, tile_w, tile_h);
//                if(buf_type == 0) {
//                  pix_state[0] = 0;
//                  pix_state[1] = 2;
//                }else{
//                  pix_state[0] = 2;
//                  pix_state[1] = 0;
//                }
#      else

              // convert rgb to yuv
#        if defined(WITH_CLIENT_RENDERENGINE_VR) || !defined(WITH_CLIENT_RENDERENGINE)
              util_rgb_to_yuv_i420_stereo(
                  (unsigned char *)&pixels_yuv[0], (unsigned char *)pixels, tile_h, tile_w);
#        else
              util_rgb_to_yuv_i420(
                  (unsigned char *)&pixels_yuv[0], (unsigned char *)pixels, tile_h, tile_w);
#        endif
              ////^///////

              // util_save_bmp(data.client_path_trace_data.offset,
              //    data.client_path_trace_data.stride,
              //    data.client_path_trace_data.tile_x,
              //    data.client_path_trace_data.tile_y,
              //    data.client_path_trace_data.tile_h,
              //    data.client_path_trace_data.tile_w,
              //    data.client_path_trace_data.pass_stride,
              //    pix_state[3],
              //    NULL,
              //    pixels,
              //    0);
              //^////

              unsigned char *buffer_pixels_y = (unsigned char *)&pixels_yuv[0];
              unsigned char *buffer_pixels_u = (unsigned char *)&pixels_yuv[0] + tile_h * tile_w;
              unsigned char *buffer_pixels_v = (unsigned char *)&pixels_yuv[0] + tile_h * tile_w +
                                               tile_h * tile_w / 4;

              // util_save_yuv(
              //    buffer_pixels_y, buffer_pixels_u, buffer_pixels_v, tile_w, tile_h);

              cesnet_set_render_buffer_yuv_i420(
                  buffer_pixels_y, buffer_pixels_u, buffer_pixels_v, tile_w, tile_h);
#      endif

              if (cesnet_is_required_exit()) {
                pix_state[4] = 1;
                //#      pragma omp flush
                omp_unset_lock(lock0);
                omp_unset_lock(lock1);

                break;
              }
            }
#    else

#      ifdef WITH_CLIENT_XOR_RLE
            {
              int tile_w = data.client_path_trace_data.tile_w;
              int tile_h = data.client_path_trace_data.tile_h;

#        if defined(WITH_CLIENT_RENDERENGINE_VR) || !defined(WITH_CLIENT_RENDERENGINE)
              tile_w *= 2;
#        endif
              size_t send_size = 0;
              char *send_data = util_rgb_to_xor_rle(pixels, tile_h, tile_w, send_size);
              // NET_NAMESPACE::send_data_data((char*)&send_size, sizeof(size_t));
              NET_NAMESPACE::send_data_data(send_data, send_size);
            }
//#elif defined(WITH_CLIENT_YUV_)
//              {
//                int tile_w = data.client_path_trace_data.tile_w;
//                int tile_h = data.client_path_trace_data.tile_h;
//
//#        if defined(WITH_CLIENT_RENDERENGINE_VR) || !defined(WITH_CLIENT_RENDERENGINE)
//                tile_w *= 2;
//#        endif
//
//                //convert rgb to yuv
//                double t2 = omp_get_wtime();
//#  if defined(WITH_CLIENT_RENDERENGINE_VR) || !defined(WITH_CLIENT_RENDERENGINE)
//                util_rgb_to_yuv_i420_stereo((unsigned char*)&pixels_yuv[0], (unsigned
//                char*)pixels, tile_h, tile_w);
//#else
//                util_rgb_to_yuv_i420((unsigned char*)&pixels_yuv[0], (unsigned char*)pixels,
//                tile_h, tile_w);
//#endif
//                double t3 = omp_get_wtime();
//                NET_NAMESPACE::send_data_data((unsigned char *)&pixels_yuv[0], tile_h * tile_w +
//                tile_h * tile_w / 2); double t4 = omp_get_wtime(); CLIENT_DEBUG_PRINTF3("send:
//                pix:%f, conv:%f\n", t4-t3,t3-t2);
//              }
#      elif defined(WITH_CLIENT_YUV)
            double t3 = omp_get_wtime();

            int tile_w = data.client_path_trace_data.tile_w;
            int tile_h = data.client_path_trace_data.tile_h;

#        ifdef WITH_CLIENT_RENDERENGINE_VR
            tile_w *= 2;
#        endif

            int *ps = (int *)pixels;
            ps[0] = samples;

            NET_NAMESPACE::send_data_data(pixels, tile_h * tile_w + tile_h * tile_w / 2);

            double t4 = omp_get_wtime();
            CLIENT_DEBUG_PRINTF2("send: pix:%f, \n", t4 - t3);
#      else
            double t3 = omp_get_wtime();

            //#        ifdef WITH_CLIENT_RENDERENGINE_VRCLIENT
            //              //NET_NAMESPACE::send_data_data(pixels, pix_size / 2, false);
            //              //NET_NAMESPACE::send_data_data(pixels + pix_size / 2, pix_size / 2);
            //              NET_NAMESPACE::send_data_data(pixels, pix_size);
            //
            //#        else
            int *ps = (int *)pixels;
            ps[0] = samples;

            NET_NAMESPACE::send_data_data(pixels, pix_size);

#        ifdef BMP_CACHE
            if (sended >= sended_max) {
              for (int i = 0; i < sended_max; i++) {
                //                  char outtemp[128]
                //                  strcpy(outtemp, "/lscratch/milanjaros/res");
                //
                //                  setenv("CLIENT_FILE_CYCLES_BMP", outtemp, 1);
                util_save_bmp(data.client_path_trace_data.offset,
                              data.client_path_trace_data.stride * 2,
                              data.client_path_trace_data.tile_x,
                              data.client_path_trace_data.tile_y,
                              data.client_path_trace_data.tile_h,
                              data.client_path_trace_data.tile_w * 2,
                              data.client_path_trace_data.pass_stride,
                              samples,
                              NULL,
                              &sended_pixels[i * pix_size],
                              i);
              }
              exit(0);
            }
            else if (sended < sended_max) {
              memcpy(&sended_pixels[sended * pix_size], pixels, pix_size);
              printf("sended:%d/%d\n", sended, sended_max);
              fflush(0);
            }
            sended++;
#        endif
            //#        endif
            double t4 = omp_get_wtime();
            CLIENT_DEBUG_PRINTF2("send: pix:%f, \n", t4 - t3);
#      endif

#    endif
#  endif

              //#ifndef WITH_CLIENT_ULTRAGRID
              ////              int samples = pix_state[3];
              ////              NET_NAMESPACE::send_data_data((char *)&samples, sizeof(int));
              ////#    endif
              //
              //              NET_NAMESPACE::send_data_data((char *)&samples, sizeof(int));
              //
              //              char ack = 0;
              //              NET_NAMESPACE::recv_data_cam((char *)&ack, sizeof(char));
              //
              //              if (ack != -1) {
              //                printf("data_ack != -1\n");
              //                break;
              //              }
              //#endif

              //displayFPS(0);

              socket_step_cam(data);

              bool buf_reset = cam_change(data);
              if (buf_reset) {
                socket_step_data(data);
                pix_state[2] = 1;
                //#  pragma omp flush
              }

              buf_new = image_size_change(data);
              if (buf_new) {
                pix_state[4] = 1;
                //#  pragma omp flush
                socket_step_cam_ack(data, 0);
                // NET_NAMESPACE::send_data_data(pixels, pix_size);

                omp_unset_lock(lock0);
                omp_unset_lock(lock1);

                break;
              }

              socket_step_cam_ack(data, -1);

              if (used_buffer == 0 /*pix_state[0] == 2*/) {
                // pix_state[0] = 0;
                // printf("send2: pix_state[0] = 0, %f\n", omp_get_wtime()); fflush(0);
                omp_unset_lock(lock0);
              }
              else if (used_buffer == 1 /*pix_state[1] == 2*/) {
                // pix_state[1] = 0;
                // printf("send2: pix_state[1] = 0, %f\n", omp_get_wtime()); fflush(0);
                omp_unset_lock(lock1);
              }
              used_buffer++;
              if (used_buffer > 1)
                used_buffer = 0;
            }
          }
        }
      }

#  if defined(WITH_CUDA_BUFFER_MANAGED) || defined(WITH_HIP_BUFFER_MANAGED) || \
      defined(WITH_CLIENT_UNIMEM)
      DEV_NAMESPACE::host_free("pixels_node", NULL, pixels_node, 1);
#  else
      DEV_NAMESPACE::host_free("pixels_node", NULL, pixels_node, 2);
#  endif

      // omp_destroy_lock(lock0);
      // omp_destroy_lock(lock1);

      if (!buf_new)
        break;
    }
  }
}
#endif

///////////////////////////////////////////////////////
#if defined(WITH_CLIENT_MPI)
std::vector<char> g_path_trace_buffer_received;
std::vector<char> g_path_trace_pixels_send;
DEVICE_PTR g_path_trace_buffer = NULL;
DEVICE_PTR g_path_trace_pixels = NULL;

int get_dev_node(client_kernel_struct &data)
{
#  ifdef WITH_CLIENT_MPI_CACHE
  if (g_caching_enabled != CACHING_DISABLED) {
    return 0;
  }
#  endif

  return data.world_rank - 1;
}

int get_devices_size_node(client_kernel_struct &data)
{
#  ifdef WITH_CLIENT_MPI_CACHE
  if (g_caching_enabled != CACHING_DISABLED) {
    return 1;
  }
#  endif

  return data.world_size - 1;
}

void path_trace_pixels_comm_init(client_kernel_struct &data, double &sender_time)
{
  const char *env_samples = getenv("DEBUG_SAMPLES");
  int num_samples = data.client_path_trace_data.num_samples;
  if (env_samples != NULL) {
    num_samples = atoi(env_samples);
  }

  g_cyclesphi_data.step_samples = num_samples;
  g_path_trace_buffer = data.client_path_trace_data.buffer;
}

void path_trace_pixels_comm_finish(client_kernel_struct &data,
                                   char *pixels,
                                   char *pixels_node1,
                                   char *pixels_node2,
                                   int *pix_state)
{
}

void path_trace_pixels_comm_thread(client_kernel_struct &data,
                                   char *pixels,
                                   char *pixels_node1,
                                   char *pixels_node2,
                                   char *buffer2,
                                   char *buffer3,
                                   int *pix_state,
                                   double &sender_time,
                                   const char *drt,
                                   bool &buf_new,
                                   int &used_buffer)
{
  // if (data.world_rank == 0) {

  // while (!do_exit) {

  double send_time = omp_get_wtime();

  //#  pragma omp flush

  if (data.client_tag == CLIENT_TAG_CYCLES_path_trace) {
#  if !defined(WITH_CLIENT_NCCL_SOCKET)  //&& !defined(WITH_CLIENT_MPI_REDUCE)
    // while (true) {
    if (used_buffer == 0) {
      // omp_set_lock(lock0);
      // pix_state[0] = 1;
      // memcpy(&pixels[0], pixels_node1, pix_size);
      pixels = pixels_node1;
      // pix_state[1] = 0;
    }
    else if (used_buffer == 1) {
      // omp_set_lock(lock1);
      // pix_state[1] = 1;
      // memcpy(&pixels[0], pixels_node2, pix_size);
      pixels = pixels_node2;
      // pix_state[0] = 0;
    }
#  endif
    //            else {
    //#  pragma omp flush
    //
    //              if (!do_exit)
    //                continue;
    //            }
    //            break;
    //          }

    // socket_send_data_data(&pixels[0], pixels.size());
#  if defined(WITH_CLIENT_NCCL_SOCKET) || defined(WITH_CLIENT_MPI_REDUCE)
    {
      int start_sample = data.client_path_trace_data.start_sample;
      int end_sample = data.client_path_trace_data.start_sample + num_samples;

      int pass_stride = data.client_path_trace_data.pass_stride;

      int offset = data.client_path_trace_data.offset;
      int stride = data.client_path_trace_data.stride;

      int tile_x = data.client_path_trace_data.tile_x;
      int tile_w = data.client_path_trace_data.tile_w;

#    if defined(WITH_CLIENT_RENDERENGINE_VR) || \
        (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
      tile_w *= 2;  // left+right
#    endif

      int tile_y = data.client_path_trace_data.tile_y;
      int tile_h = data.client_path_trace_data.tile_h;

      // char *buffer =
      //     (char *)DEV_NAMESPACE::get_ptr_map(data.client_path_trace_data.buffer];

      DEV_NAMESPACE::mem_copy_from(
          CLIENT_DEVICE_ID,
          DEV_NAMESPACE::get_ptr_map(data.client_path_trace_data.buffer],
          buffer2,
          0,
          tile_w * tile_h * pass_stride * sizeof(float),
          NULL);

      // memcpy(buffer2, buffer, tile_w * tile_h * pass_stride * sizeof(float));

#    ifdef WITH_CLIENT_MPI_REDUCE
      // buffer2[0] = (float)pix_state[3];
      MPI_Reduce(
          buffer2, NULL, tile_w * tile_h * pass_stride, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
#    else
      cudaSetDevice(0);
      buffer[0] = (float)pix_state[3];

      ncclResult_t res = ncclReduce(buffer,
                                    buffer2,
                                    tile_w * tile_h * pass_stride,
                                    ncclFloat,
                                    ncclSum,
                                    0,
                                    (ncclComm_t)data.comm_data,
                                    nccl_stream);

      if (res != ncclSuccess) {
        printf("ncclReduce send != ncclSuccess\n");
        exit(-1);
      }

#    endif
    }
#  else
    {
      ////////////////////////////one node///////////////////////////////////
      int dev_node = data.world_rank - 1;
      int devices_size_node = data.world_size - 1;

      // int tile_step_node = data.client_path_trace_data.tile_h / devices_size_node;
      // ceil
      int tile_step_node = (int)ceil((float)data.client_path_trace_data.tile_h /
                                     (float)devices_size_node);

      int tile_last_node = data.client_path_trace_data.tile_h -
                           (devices_size_node - 1) * tile_step_node;

      if (tile_last_node < 1) {
        tile_step_node = (int)((float)data.client_path_trace_data.tile_h /
                               (float)(devices_size_node));

        tile_last_node = data.client_path_trace_data.tile_h -
                         (devices_size_node - 1) * tile_step_node;
      }

      int tile_y_node = data.client_path_trace_data.tile_y + tile_step_node * dev_node;
      int tile_h_node = (devices_size_node - 1 == dev_node) ? tile_last_node : tile_step_node;

      size_t pix_size_node = tile_h_node * data.client_path_trace_data.tile_w * SIZE_UCHAR4;

#    if defined(WITH_CLIENT_RENDERENGINE_VR) || \
        (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
      pix_size_node *= 2;  // left+right

      size_t pix_offset = (data.client_path_trace_data.tile_x +
                           tile_y_node * data.client_path_trace_data.stride * 2) *
                          SIZE_UCHAR4;
#    else
      size_t pix_offset = (data.client_path_trace_data.tile_x +
                           tile_y_node * data.client_path_trace_data.stride) *
                          SIZE_UCHAR4;
#    endif

#    ifdef WITH_CLIENT_MPI_CACHE
      if (g_caching_enabled != CACHING_DISABLED) {
        pix_offset = 0;
        int f = (g_caching_enabled == CACHING_PREVIEW) ? g_current_frame_preview : g_current_frame;
        pix_size_node = ((f % devices_size_node) == dev_node) ?
                            data.client_path_trace_data.tile_h *
                                data.client_path_trace_data.tile_w * SIZE_UCHAR4 :
                            0;
      }

#    endif

      MPI_Gatherv((char *)&pixels[0] + pix_offset,
                  // pixels.size(),
                  pix_size_node,
                  MPI_BYTE,
                  NULL,
                  0,
                  NULL,
                  MPI_BYTE,
                  0,
                  MPI_COMM_WORLD);
#    ifndef WITH_CLIENT_MPI_VRCLIENT
      int global_min;  // = pix_state[3];
      int local_min = pix_state[3];

      MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
#    endif
    }
#  endif

    // displayFPS(1);

#  ifdef WITH_CLIENT_MPI_VRCLIENT
    bool buf_reset = mpi_cam_change(data);
    if (buf_reset) {
      mpi_socket_step_data(data);
      pix_state[2] = 1;
      //#  pragma omp flush
    }

    bool buf_new = mpi_image_size_change(data);
    if (buf_new) {
      pix_state[4] = 1;
      data.world_size = 0;

      omp_unset_lock(lock0);
      omp_unset_lock(lock1);

      break;
    }
#  endif

//////////////////////////////////////////////////
#  if !defined(WITH_CLIENT_NCCL_SOCKET)  //&& !defined(WITH_CLIENT_MPI_REDUCE)
                                         //          if (pix_state[0] == 1) {
                                         //            pix_state[0] = 0;
                                         //          }
                                         //          else if (pix_state[1] == 1) {
                                         //            pix_state[1] = 0;
                                         //          }
    if (used_buffer == 0 /*pix_state[0] == 2*/) {
      // pix_state[0] = 0;
      // printf("send2: pix_state[0] = 0, %f\n", omp_get_wtime()); fflush(0);
      // omp_unset_lock(lock0);
    }
    else if (used_buffer == 1 /*pix_state[1] == 2*/) {
      // pix_state[1] = 0;
      // printf("send2: pix_state[1] = 0, %f\n", omp_get_wtime()); fflush(0);
      // omp_unset_lock(lock1);
    }
    // used_buffer++;
    if (used_buffer > CLIENT_SWAP_BUFFERS)
      used_buffer = 0;
#  endif
  }

#  ifndef WITH_CLIENT_MPI_VRCLIENT
  else if (data.client_tag == CLIENT_TAG_CYCLES_const_copy) {
    bcast(&kg_data[0], kg_data.size());

    // if(pix_state[2] == 0)
    {
      char *gpu_data = (char *)DEV_NAMESPACE::get_data(DEV_NAMESPACE::get_kernel_globals_cpu());

      // if (memcmp(gpu_data, (char *)&mpi_kg_data[0], mpi_kg_data.size())) {
      memcpy(gpu_data, (char *)&kg_data[0], kg_data.size());

#    if defined(WITH_CLIENT_CUDA) || defined(WITH_CLIENT_HIP)
      DEV_NAMESPACE::cam_recalc(gpu_data);

      // pix_state[2] = 1;
      // printf("mpi_kg_data.size(): %lld\n", mpi_kg_data.size());
      //}

#      if defined(WITH_CLIENT_RENDERENGINE_VR) || \
          (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
      char *gpu_data_right = (char *)DEV_NAMESPACE::get_data_right(
          DEV_NAMESPACE::get_kernel_globals_cpu());
      memcpy(gpu_data_right, (char *)&mpi_kg_data[0], mpi_kg_data.size());
      DEV_NAMESPACE::cam_recalc(gpu_data_right);
#      endif
#    endif
    }

    // kernel::omp_const_copy(OMP_DEVICE_ID,
    //              kernel::omp_mpiData->kernel_globals_cpu,
    //              "data",
    //              &kg_data[0],
    //              kg_data.size());
    // displayFPS(2);
  }
  else if (data.client_tag == CLIENT_TAG_CYCLES_mem_zero &&
           data.client_mem_data.mem == g_path_trace_buffer) {
    // if(pix_state[2] == 0) {
    pix_state[2] = 1;
    //}
  }
  else if (data.client_tag == CLIENT_TAG_CYCLES_mem_copy_to &&
           data.client_mem_data.mem == g_path_trace_buffer) {

    if (g_path_trace_buffer_received.size() != data.client_mem_data.memSize)
      g_path_trace_buffer_received.resize(data.client_mem_data.memSize);

    bcast(&g_path_trace_buffer_received[0], data.client_mem_data.memSize);
    // TODO: skip
  }
  else if (data.client_tag == CLIENT_TAG_CYCLES_frame_info) {
    frame_info(data);
    pix_state[2] = 1;
  }
#  else
  else if (data.client_tag == CLIENT_TAG_CYCLES_mem_zero) {
  }
#  endif
  else {
    pix_state[4] = 1;
    data.world_size = 0;
    return;
  }

  //////////////////////////////////////////////////
  int world_rank = data.world_rank;
  int world_size = data.world_size;
  char *comm_data = data.comm_data;
  read_data_kernelglobal(&data, sizeof(client_kernel_struct));
  data.world_rank = world_rank;
  data.world_size = world_size;
  data.comm_data = comm_data;

  // displayFPS(0);

  //////////////////////////////////////////////////
  //        double send_time2 = omp_get_wtime();
  //        double *st = (double*)&pix_state[6];
  //        st[0] = send_time2 - send_time;
  //        if (pix_state[0] == 1) {
  //          pix_state[0] = 0;
  //        }
  //        else if (pix_state[1] == 1) {
  //          pix_state[1] = 0;
  //        }
}

void path_trace_buffer_comm_init(client_kernel_struct &data)
{
}

void path_trace_buffer_comm_finish(client_kernel_struct &data,
                                   char *buffer_temp,
                                   size_t buffer_size)
{
}
#endif

#if defined(WITH_CLIENT_FILE)

int get_dev_node(client_kernel_struct &data)
{
  return data.world_rank;
}

int get_devices_size_node(client_kernel_struct &data)
{
  return data.world_size;
}

void path_trace_pixels_comm_init(client_kernel_struct &data, double &sender_time)
{
  if (data.world_rank == 0) {
    //#  pragma omp parallel num_threads(2)
    {
      // int tid = omp_get_thread_num();
      // if(tid==0) {
      if (sender_time == -1) {
        socket_step_cam(data);
        socket_step_cam_ack(data, -1);
      }
      cam_change(data);
      //}
    }

    data.client_path_trace_data.num_samples = get_image_sample();

    int *size = get_image_size();
    int w_old = data.client_path_trace_data.tile_w;
    int h_old = data.client_path_trace_data.tile_h;

    data.client_path_trace_data.tile_w = size[0];
    data.client_path_trace_data.tile_h = size[1];
    data.client_path_trace_data.stride = data.client_path_trace_data.tile_w;

    socket_step_image_size(data, w_old, h_old);
    socket_step_data(data);
    set_bounces(data);
    // buf_new = false;
  }
}

void path_trace_pixels_comm_finish(client_kernel_struct &data,
                                   char *pixels,
                                   char *pixels_node1,
                                   char *pixels_node2,
                                   int *pix_state)
{
#  if defined(WITH_CLIENT_RENDERENGINE_EMULATE)
  int samples = pix_state[3];

  int w = data.client_path_trace_data.tile_w;

#    if defined(WITH_CLIENT_RENDERENGINE_VR) || \
        (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
  w *= 2;  // left+right
#    endif

  util_save_bmp(data.client_path_trace_data.offset,
                w,
                data.client_path_trace_data.tile_x,
                data.client_path_trace_data.tile_y,
                data.client_path_trace_data.tile_h,
                w,
                data.client_path_trace_data.pass_stride,
                samples,
                NULL,
                pixels_node1,
                0);

  // util_save_bmp(data.client_path_trace_data.offset,
  //               w,
  //               data.client_path_trace_data.tile_x,
  //               data.client_path_trace_data.tile_y,
  //               data.client_path_trace_data.tile_h,
  //               w,
  //               data.client_path_trace_data.pass_stride,
  //               samples,
  //               NULL,
  //               pixels_node2,
  //               1);

  DEV_NAMESPACE::print_client_tiles();

#  endif
}

void path_trace_pixels_comm_thread(client_kernel_struct &data,
                                   // bool &do_exit,
                                   // omp_lock_t *lock0,
                                   // omp_lock_t *lock1,
                                   char *pixels,
                                   char *pixels_node1,
                                   char *pixels_node2,
                                   char *buffer2,
                                   char *buffer3,
                                   int *pix_state,
                                   double &sender_time,
                                   const char *drt,
                                   bool &buf_new,
                                   int &used_buffer)
{
  //   if (pix_state[6] == 1) {
  //     pix_state[6] = 0;

  // #  pragma omp barrier
  //   }
  // if (data.world_rank == 0) {
  // int used_buffer = 0;
  // while (!do_exit) {

#  if defined(WITH_CLIENT_RENDERENGINE)

  size_t pix_size = data.client_path_trace_data.tile_h * data.client_path_trace_data.tile_w *
                    sizeof(char) * 4;
#  else
  size_t pix_size = data.client_path_trace_data.tile_h * data.client_path_trace_data.tile_w *
                    SIZE_UCHAR4;
#  endif

  if (used_buffer == 0) {
    // omp_set_lock(lock0);

    // printf("send1: pix_state[0] = 2, %f\n", omp_get_wtime()); fflush(0);
    // pix_state[0] = 1;
    // memcpy(&pixels[0], pixels_node1, pix_size);
    pixels = pixels_node1;

#  if defined(WITH_CLIENT_RENDERENGINE_EMULATE)
    if (sender_time < 0 && pix_state[5] != 0)
      sender_time = omp_get_wtime();
#  endif

    // pix_state[1] = 0;
    //#  pragma omp flush
  }
  else if (used_buffer == 1) {
    // omp_set_lock(lock1);

    // printf("send1: pix_state[1] = 2, %f\n", omp_get_wtime()); fflush(0);
    // pix_state[1] = 1;
    // memcpy(&pixels[0], pixels_node2, pix_size);
    pixels = pixels_node2;

#  if defined(WITH_CLIENT_RENDERENGINE_EMULATE)
    if (sender_time < 0 && pix_state[5] != 0)
      sender_time = omp_get_wtime();
#  endif

    // pix_state[0] = 0;
    //#  pragma omp flush
  }
  else {
    // omp_unset_lock(lock0);
    // omp_unset_lock(lock1);

    //        if (!do_exit) {
    //          usleep(100);
    //#  pragma omp flush
    //          continue;
    return;
    //}
  }

  int samples = pix_state[3];

#  if defined(WITH_CLIENT_RENDERENGINE_EMULATE)
  // const char *drt = getenv("DEBUG_REPEAT_TIME");
  if (drt == NULL || ((sender_time > 0) && (omp_get_wtime() - sender_time > atof(drt)))) {
    pix_state[4] = 1;

    // util_save_bmp(data.client_path_trace_data.offset,
    //              data.client_path_trace_data.stride,
    //              data.client_path_trace_data.tile_x,
    //              data.client_path_trace_data.tile_y,
    //              data.client_path_trace_data.tile_h,
    //              data.client_path_trace_data.tile_w,
    //              data.client_path_trace_data.pass_stride,
    //              samples,
    //              NULL,
    //              pixels,
    //              0);

    // while (!do_exit) {
    //    usleep(1000);
    //}

    // exit(0);
    // omp_unset_lock(lock0);
    // omp_unset_lock(lock1);
    return;
  }
#  elif defined(WITH_NVPIPE)

#    ifdef WITH_CLIENT_RENDERENGINE_VRCLIENT
  // kernel::socket_send_data_data(pixels, pix_size / 2, false);
  // kernel::socket_send_data_data(pixels + pix_size / 2, pix_size / 2);

#      if defined(WITH_CUDA_BUFFER_MANAGED) || defined(WITH_HIP_BUFFER_MANAGED) || \
          defined(WITH_CLIENT_UNIMEM)
  DEVICE_PTR dev_pixels = (DEVICE_PTR)pixels;
#      else
#        if defined(WITH_CLIENT_CUDA)
  DEVICE_PTR dev_pixels = DEV_NAMESPACE::get_host_get_device_pointer(pixels);
#        endif
#        if defined(WITH_CLIENT_OPTIX)
  DEVICE_PTR dev_pixels = DEV_NAMESPACE::get_host_get_device_pointer(pixels);
#        endif
#        if defined(WITH_CLIENT_HIP)
  DEVICE_PTR dev_pixels = DEV_NAMESPACE::get_host_get_device_pointer(pixels);
#        endif
#      endif
  kernel::socket_send_nvpipe((DEVICE_PTR)dev_pixels,
                             &pixels_denoise[0],
                             data.client_path_trace_data.tile_w,
                             data.client_path_trace_data.tile_h);

  kernel::socket_send_nvpipe((DEVICE_PTR)((char *)dev_pixels + pix_size / 2),
                             &pixels_denoise[0],
                             data.client_path_trace_data.tile_w,
                             data.client_path_trace_data.tile_h);
#    else
  kernel::socket_send_nvpipe((DEVICE_PTR)pixels,
                             &pixels_denoise[0],
                             data.client_path_trace_data.tile_w,
                             data.client_path_trace_data.tile_h);
#    endif

#  else

#    ifdef WITH_CLIENT_ULTRAGRID
  // YUV I420
  {
    //#      ifdef _WIN32
    //                // usleep(2000000);
    //#      endif

    int tile_w = data.client_path_trace_data.tile_w;
    int tile_h = data.client_path_trace_data.tile_h;

#      if defined(WITH_CLIENT_RENDERENGINE_VR) || !defined(WITH_CLIENT_RENDERENGINE)
    tile_w *= 2;
#      endif

#      if defined(WITH_RGBA_FORMAT) || defined(WITH_CLIENT_ULTRAGRID_LIB)
    // rendering buffer - results in RGBA format
    int buf_type = cesnet_set_render_buffer_rgba((unsigned char *)pixels, tile_w, tile_h);
//                if(buf_type == 0) {
//                  pix_state[0] = 0;
//                  pix_state[1] = 2;
//                }else{
//                  pix_state[0] = 2;
//                  pix_state[1] = 0;
//                }
#      else

    // convert rgb to yuv
#        if defined(WITH_CLIENT_RENDERENGINE_VR) || !defined(WITH_CLIENT_RENDERENGINE)
    util_rgb_to_yuv_i420_stereo(
        (unsigned char *)&pixels_yuv[0], (unsigned char *)pixels, tile_h, tile_w);
#        else
    util_rgb_to_yuv_i420((unsigned char *)&pixels_yuv[0], (unsigned char *)pixels, tile_h, tile_w);
#        endif
    ////^///////

    // util_save_bmp(data.client_path_trace_data.offset,
    //    data.client_path_trace_data.stride,
    //    data.client_path_trace_data.tile_x,
    //    data.client_path_trace_data.tile_y,
    //    data.client_path_trace_data.tile_h,
    //    data.client_path_trace_data.tile_w,
    //    data.client_path_trace_data.pass_stride,
    //    pix_state[3],
    //    NULL,
    //    pixels,
    //    0);
    //^////

    unsigned char *buffer_pixels_y = (unsigned char *)&pixels_yuv[0];
    unsigned char *buffer_pixels_u = (unsigned char *)&pixels_yuv[0] + tile_h * tile_w;
    unsigned char *buffer_pixels_v = (unsigned char *)&pixels_yuv[0] + tile_h * tile_w +
                                     tile_h * tile_w / 4;

    // util_save_yuv(
    //    buffer_pixels_y, buffer_pixels_u, buffer_pixels_v, tile_w, tile_h);

    cesnet_set_render_buffer_yuv_i420(
        buffer_pixels_y, buffer_pixels_u, buffer_pixels_v, tile_w, tile_h);
#      endif

    if (cesnet_is_required_exit()) {
      pix_state[4] = 1;
      //#      pragma omp flush
      // omp_unset_lock(lock0);
      // omp_unset_lock(lock1);

      return;
    }
  }
#    else

#      ifdef WITH_CLIENT_XOR_RLE
  {
    int tile_w = data.client_path_trace_data.tile_w;
    int tile_h = data.client_path_trace_data.tile_h;

#        if defined(WITH_CLIENT_RENDERENGINE_VR) || !defined(WITH_CLIENT_RENDERENGINE)
    tile_w *= 2;
#        endif
    size_t send_size = 0;
    char *send_data = util_rgb_to_xor_rle(pixels, tile_h, tile_w, send_size);
    // kernel::socket_send_data_data((char*)&send_size, sizeof(size_t));
    kernel::socket_send_data_data(send_data, send_size);
  }
//#elif defined(WITH_CLIENT_YUV_)
//              {
//                int tile_w = data.client_path_trace_data.tile_w;
//                int tile_h = data.client_path_trace_data.tile_h;
//
//#        if defined(WITH_CLIENT_RENDERENGINE_VR) || !defined(WITH_CLIENT_RENDERENGINE)
//                tile_w *= 2;
//#        endif
//
//                //convert rgb to yuv
//                double t2 = omp_get_wtime();
//#  if defined(WITH_CLIENT_RENDERENGINE_VR) || !defined(WITH_CLIENT_RENDERENGINE)
//                util_rgb_to_yuv_i420_stereo((unsigned char*)&pixels_yuv[0], (unsigned
//                char*)pixels, tile_h, tile_w);
//#else
//                util_rgb_to_yuv_i420((unsigned char*)&pixels_yuv[0], (unsigned char*)pixels,
//                tile_h, tile_w);
//#endif
//                double t3 = omp_get_wtime();
//                kernel::socket_send_data_data((unsigned char *)&pixels_yuv[0], tile_h * tile_w +
//                tile_h * tile_w / 2); double t4 = omp_get_wtime(); CLIENT_DEBUG_PRINTF3("send:
//                pix:%f, conv:%f\n", t4-t3,t3-t2);
//              }
#      elif defined(WITH_CLIENT_YUV)
  double t3 = omp_get_wtime();

  int tile_w = data.client_path_trace_data.tile_w;
  int tile_h = data.client_path_trace_data.tile_h;

#        ifdef WITH_CLIENT_RENDERENGINE_VR
  tile_w *= 2;
#        endif

  int *ps = (int *)pixels;
  ps[0] = samples;

  kernel::socket_send_data_data(pixels, tile_h * tile_w + tile_h * tile_w / 2);

  double t4 = omp_get_wtime();
  CLIENT_DEBUG_PRINTF2("send: pix:%f, \n", t4 - t3);
#      else
  double t3 = omp_get_wtime();

  //#        ifdef WITH_CLIENT_RENDERENGINE_VRCLIENT
  //              //kernel::socket_send_data_data(pixels, pix_size / 2, false);
  //              //kernel::socket_send_data_data(pixels + pix_size / 2, pix_size / 2);
  //              kernel::socket_send_data_data(pixels, pix_size);
  //
  //#        else
  int *ps = (int *)pixels;
  ps[0] = samples;

  NET_NAMESPACE::send_data_data(pixels, pix_size);

#        ifdef BMP_CACHE
  if (sended >= sended_max) {
    for (int i = 0; i < sended_max; i++) {
      //                  char outtemp[128]
      //                  strcpy(outtemp, "/lscratch/milanjaros/res");
      //
      //                  setenv("CLIENT_FILE_CYCLES_BMP", outtemp, 1);
      util_save_bmp(data.client_path_trace_data.offset,
                    data.client_path_trace_data.stride * 2,
                    data.client_path_trace_data.tile_x,
                    data.client_path_trace_data.tile_y,
                    data.client_path_trace_data.tile_h,
                    data.client_path_trace_data.tile_w * 2,
                    data.client_path_trace_data.pass_stride,
                    samples,
                    NULL,
                    &sended_pixels[i * pix_size],
                    i);
    }
    exit(0);
  }
  else if (sended < sended_max) {
    memcpy(&sended_pixels[sended * pix_size], pixels, pix_size);
    printf("sended:%d/%d\n", sended, sended_max);
    fflush(0);
  }
  sended++;
#        endif
  //#        endif
  double t4 = omp_get_wtime();
  CLIENT_DEBUG_PRINTF2("send: pix:%f, \n", t4 - t3);
#      endif

#    endif
#  endif

  //#ifndef WITH_CLIENT_ULTRAGRID
  ////              int samples = pix_state[3];
  ////              kernel::socket_send_data_data((char *)&samples, sizeof(int));
  ////#    endif
  //
  //              kernel::socket_send_data_data((char *)&samples, sizeof(int));
  //
  //              char ack = 0;
  //              kernel::socket_recv_data_cam((char *)&ack, sizeof(char));
  //
  //              if (ack != -1) {
  //                printf("data_ack != -1\n");
  //                break;
  //              }
  //#endif

  // displayFPS(0);

  socket_step_cam(data);

  bool buf_reset = cam_change(data);
  if (buf_reset) {
    socket_step_data(data);
    pix_state[2] = 1;
    //#  pragma omp flush
  }

  buf_new = image_size_change(data);
  if (buf_new) {
    pix_state[4] = 1;
    //#  pragma omp flush
    socket_step_cam_ack(data, 0);
    // kernel::socket_send_data_data(pixels, pix_size);

    // omp_unset_lock(lock0);
    // omp_unset_lock(lock1);

    return;
  }

  socket_step_cam_ack(data, -1);

  if (used_buffer == 0 /*pix_state[0] == 2*/) {
    // pix_state[0] = 0;
    // printf("send2: pix_state[0] = 0, %f\n", omp_get_wtime()); fflush(0);
    // omp_unset_lock(lock0);
  }
  else if (used_buffer == 1 /*pix_state[1] == 2*/) {
    // pix_state[1] = 0;
    // printf("send2: pix_state[1] = 0, %f\n", omp_get_wtime()); fflush(0);
    // omp_unset_lock(lock1);
  }
  //used_buffer++;
  //if (used_buffer > 1)
  //  used_buffer = 0;
  //}
  //}
  //}

  // return true;
}

void path_trace_buffer_comm_init(client_kernel_struct &data)
{
}

  #ifdef WITH_CLIENT_MPI_FILE_1
    std::vector<char> g_path_trace_buffer_received;
    // std::vector<char> g_path_trace_pixels_send;
    // std::vector<char> g_path_trace_pixels_received;
    DEVICE_PTR g_path_trace_buffer = NULL;
    // DEVICE_PTR g_path_trace_pixels = NULL;

    std::vector<int> g_displsPix;
    std::vector<int> g_recvcountsPix;

    void path_trace_buffer_comm_finish(client_kernel_struct &data,
                                      char *buffer_temp,
                                      size_t buffer_size)
    {

      int dev_node = data.world_rank;           //         -1;
      int devices_size_node = data.world_size;  // -1;

      // int tile_step_node = data.client_path_trace_data.tile_h / devices_size_node;
      // ceil
      int tile_step_node = (int)ceil((float)data.client_path_trace_data.tile_h /
                                    (float)devices_size_node);

      int tile_last_node = data.client_path_trace_data.tile_h -
                          (devices_size_node - 1) * tile_step_node;

      if (tile_last_node < 1) {
        tile_step_node = (int)((float)data.client_path_trace_data.tile_h / (float)(devices_size_node));

        tile_last_node = data.client_path_trace_data.tile_h - (devices_size_node - 1) * tile_step_node;
      }

      int tile_y_node = data.client_path_trace_data.tile_y + tile_step_node * dev_node;
      int tile_h_node = (devices_size_node - 1 == dev_node) ? tile_last_node : tile_step_node;

      size_t buff_size_node = tile_h_node * data.client_path_trace_data.tile_w *
                              data.client_path_trace_data.pass_stride * sizeof(float);

    #  if defined(WITH_CLIENT_RENDERENGINE_VR) || \
          (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
      buff_size_node *= 2;  // left+right

      size_t buff_offset = (data.client_path_trace_data.tile_x +
                            tile_y_node * data.client_path_trace_data.stride * 2) *
                          data.client_path_trace_data.pass_stride * sizeof(float);
    #  else
      size_t buff_offset = (data.client_path_trace_data.tile_x +
                            tile_y_node * data.client_path_trace_data.stride) *
                          data.client_path_trace_data.pass_stride * sizeof(float);
    #  endif

      if (g_displsPix.size() != devices_size_node) {
        g_displsPix.resize(devices_size_node, 0);
        g_recvcountsPix.resize(devices_size_node, 0);
      }

      for (int dev = 0; dev < devices_size_node; dev++) {
        int tile_y2 = data.client_path_trace_data.tile_y + tile_step_node * dev;
        int tile_h2 = (devices_size_node - 1 == dev) ? tile_last_node : tile_step_node;
        if (tile_h2 == 0)
          continue;

        g_displsPix[dev] = (data.client_path_trace_data.offset + data.client_path_trace_data.tile_x +
                            tile_y2 * data.client_path_trace_data.stride) *
                          data.client_path_trace_data.pass_stride * sizeof(float);
        g_recvcountsPix[dev] = data.client_path_trace_data.tile_w * tile_h2 *
                              data.client_path_trace_data.pass_stride * sizeof(float);
      }

      if (g_path_trace_buffer_received.size() != buffer_size)
        g_path_trace_buffer_received.resize(buffer_size);
      
      
      MPI_Gatherv((char *)buffer_temp + buff_offset,
                  buff_size_node,
                  MPI_BYTE,
                  g_path_trace_buffer_received.data(),
                  &g_recvcountsPix[0],
                  &g_displsPix[0],
                  MPI_BYTE,
                  0,
                  MPI_COMM_WORLD);

      // size_t pso = data.client_path_trace_data.pass_stride * sizeof(float);
      if (data.world_rank == 0)
        DEV_NAMESPACE::mem_copy_to(
          CLIENT_DEVICE_ID, 
          g_path_trace_buffer_received.data(), 
          // DEV_NAMESPACE::get_ptr_map(g_path_trace_buffer),
          DEV_NAMESPACE::get_ptr_map(data.client_path_trace_data.buffer),
          buffer_size, 
          NULL);
        cyclesphi::client::save_buffer(data);
      //   NET_NAMESPACE::send_data_data(g_path_trace_buffer_received.data(), buffer_size);
    }
    
  #else

    void path_trace_buffer_comm_finish(client_kernel_struct &data,
                                      char *buffer_temp,
                                      size_t buffer_size)
    {
    }

  #endif
#endif

#if defined(WITH_CLIENT_SOCKET)
std::vector<char> g_path_trace_buffer_received;
std::vector<char> g_path_trace_pixels_send;
DEVICE_PTR g_path_trace_buffer = NULL;
DEVICE_PTR g_path_trace_pixels = NULL;

int get_dev_node(client_kernel_struct &data)
{
  return data.world_rank;
}

int get_devices_size_node(client_kernel_struct &data)
{
  return data.world_size;
}

void path_trace_pixels_comm_init(client_kernel_struct &data, double &sender_time)
{
  const char *env_samples = getenv("DEBUG_SAMPLES");
  int num_samples = data.client_path_trace_data.num_samples;
  if (env_samples != NULL) {
    num_samples = atoi(env_samples);
  }

  g_cyclesphi_data.step_samples = num_samples;
  g_path_trace_buffer = data.client_path_trace_data.buffer;

#  if defined(WITH_CLIENT_RENDERENGINE_VR) || \
      (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
  if (data.world_rank == 0) {
    int w_old = data.client_path_trace_data.tile_w;
    int h_old = data.client_path_trace_data.tile_h;

    socket_step_image_size(data, w_old, h_old);
  }
#  endif
}

void path_trace_pixels_comm_finish(client_kernel_struct &data,
                                   char *pixels,
                                   char *pixels_node1,
                                   char *pixels_node2,
                                   int *pix_state)
{
  if (NET_NAMESPACE::is_error()) {
    NET_NAMESPACE::server_close();
    NET_NAMESPACE::client_close();
  }
}

void path_trace_pixels_comm_thread(client_kernel_struct &data,
                                   // bool &do_exit,
                                   // omp_lock_t *lock0,
                                   // omp_lock_t *lock1,
                                   char *pixels,
                                   char *pixels_node1,
                                   char *pixels_node2,
                                   char *buffer2,
                                   char *buffer3,
                                   int *pix_state,
                                   double &sender_time,
                                   const char *drt,
                                   bool &buf_new,
                                   int &used_buffer)
{
  if (NET_NAMESPACE::is_error()) {
    pix_state[4] = 1;
    return;
  }

  //  if (pix_state[6] == 1) {
  //    pix_state[6] = 0;
  //
  //#  pragma omp barrier
  //  }

  // if (data.world_rank == 0) {
  // int used_buffer = 0;
  // while (true) {

  double send_time = omp_get_wtime();
  // DEVICE_PTR path_trace_buffer = data.client_path_trace_data.buffer;

  //#  pragma omp flush

  if (data.client_tag == CLIENT_TAG_CYCLES_path_trace) {
    // while (true) {
    if (used_buffer == 0) {
      // omp_set_lock(lock0);
      // pix_state[0] = 1;
      // memcpy(&pixels[0], pixels_node1, pix_size);
      pixels = pixels_node1;
      // pix_state[1] = 0;
    }
    else if (used_buffer == 1) {
      // omp_set_lock(lock1);
      // pix_state[1] = 1;
      // memcpy(&pixels[0], pixels_node2, pix_size);
      pixels = pixels_node2;
      // pix_state[0] = 0;
    }
    //            else {
    //#  pragma omp flush
    //
    //              if (!do_exit)
    //                continue;
    //            }
    //            break;
    //          }

    int es_ = pix_state[3];
    // socket_send_data_data(&pixels[0], pixels.size());
    if (es_ > 0) {
      ////////////////////////////one node///////////////////////////////////
      int dev_node = data.world_rank;           //         -1;
      int devices_size_node = data.world_size;  // -1;

      // int tile_step_node = data.client_path_trace_data.tile_h / devices_size_node;
      // ceil
      int tile_step_node = (int)ceil((float)data.client_path_trace_data.tile_h /
                                     (float)devices_size_node);

      int tile_last_node = data.client_path_trace_data.tile_h -
                           (devices_size_node - 1) * tile_step_node;

      if (tile_last_node < 1) {
        tile_step_node = (int)((float)data.client_path_trace_data.tile_h /
                               (float)(devices_size_node));

        tile_last_node = data.client_path_trace_data.tile_h -
                         (devices_size_node - 1) * tile_step_node;
      }

      int tile_y_node = data.client_path_trace_data.tile_y + tile_step_node * dev_node;
      int tile_h_node = (devices_size_node - 1 == dev_node) ? tile_last_node : tile_step_node;

      size_t pix_size_node = tile_h_node * data.client_path_trace_data.tile_w * SIZE_UCHAR4;

      int tile_w2 = data.client_path_trace_data.tile_w;

#  if defined(WITH_CLIENT_RENDERENGINE_VR) || \
      (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
      pix_size_node *= 2;  // left+right
      tile_w2 *= 2;        // left+right

      size_t pix_offset = (data.client_path_trace_data.tile_x +
                           tile_y_node * data.client_path_trace_data.stride * 2) *
                          SIZE_UCHAR4;
#  else
      size_t pix_offset = (data.client_path_trace_data.tile_x +
                           tile_y_node * data.client_path_trace_data.stride) *
                          SIZE_UCHAR4;
#  endif

      // MPI_Gatherv((char *)&pixels[0] + pix_offset,
      //            // pixels.size(),
      //            pix_size_node,
      //            MPI_BYTE,
      //            NULL,
      //            0,
      //            NULL,
      //            MPI_BYTE,
      //            0,
      //            MPI_COMM_WORLD);

      //#  if 0
      //      int *s = (int *)((char *)&pixels[0] + pix_offset);
      //      s[0] = es_;
      //#  else
      //      NET_NAMESPACE::send_data_data((char *)&es_, sizeof(int), false);
      //#  endif

#  ifdef WITH_VRCLIENT_RECEIVER
      int s[3];
      s[0] = es_;
      s[1] = data.client_path_trace_data.tile_w;
      s[2] = tile_h_node;

      NET_NAMESPACE::send_data_data((char *)s, sizeof(int) * 3, false);
#  else
      NET_NAMESPACE::send_data_data((char *)&es_, sizeof(int), false);
#  endif

#  ifdef WITH_CLIENT_GPUJPEG
      if (g_path_trace_pixels_send.size() != pix_size_node) {
        g_path_trace_pixels_send.resize(pix_size_node);
      }

      // only send not empty data
      NET_NAMESPACE::send_gpujpeg(
          (char *)&pixels[0], g_path_trace_pixels_send.data(), tile_w2, tile_h_node);
#  else
      NET_NAMESPACE::send_data_data((char *)&pixels[0] + pix_offset, pix_size_node);
#  endif

#  ifdef WITH_VRCLIENT_RECEIVER
      NET_NAMESPACE::send_data_cam((char *)s, sizeof(int));
#  endif

      // int global_min;  // = pix_state[3];
      // int local_min = pix_state[3];

      // MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
      // kernel::socket_send_data_data((char *)&local_min, sizeof(int));
    }
    // displayFPS(1);

    //////////////////////////////////////////////////

    //          if (pix_state[0] == 1) {
    //            pix_state[0] = 0;
    //          }
    //          else if (pix_state[1] == 1) {
    //            pix_state[1] = 0;
    //          }
    if (used_buffer == 0 /*pix_state[0] == 2*/) {
      // pix_state[0] = 0;
      // printf("send2: pix_state[0] = 0, %f\n", omp_get_wtime()); fflush(0);
      // omp_unset_lock(lock0);
    }
    else if (used_buffer == 1 /*pix_state[1] == 2*/) {
      // pix_state[1] = 0;
      // printf("send2: pix_state[1] = 0, %f\n", omp_get_wtime()); fflush(0);
      // omp_unset_lock(lock1);
    }
    used_buffer++;
    if (used_buffer >= CLIENT_SWAP_BUFFERS)
      used_buffer = 0;

    if (es_ == 0)
      return;
  }
  else if (data.client_tag == CLIENT_TAG_CYCLES_const_copy) {
    bcast(&kg_data[0], kg_data.size());

    // if(pix_state[2] == 0)
    {
      char *gpu_data = (char *)DEV_NAMESPACE::get_data(DEV_NAMESPACE::get_kernel_globals_cpu());
      // if (memcmp(gpu_data, (char *)&socket_kg_data[0], socket_kg_data.size())) {
      memcpy(gpu_data, (char *)&kg_data[0], kg_data.size());

#  if defined(WITH_CLIENT_CUDA) || defined(WITH_CLIENT_OPTIX) || defined(WITH_CLIENT_HIP)
      DEV_NAMESPACE::cam_recalc(gpu_data);

      // pix_state[2] = 1;
      // printf("socket_kg_data.size(): %lld\n", socket_kg_data.size());
      //}

#    if defined(WITH_CLIENT_RENDERENGINE_VR) || \
        (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
      char *gpu_data_right = (char *)DEV_NAMESPACE::get_data_right(
          DEV_NAMESPACE::get_kernel_globals_cpu());
      memcpy(gpu_data_right, (char *)&kg_data[0], kg_data.size());

      DEV_NAMESPACE::cam_recalc(gpu_data_right);
#    endif
#  endif
    }

    // kernel::omp_const_copy(OMP_DEVICE_ID,
    //              kernel::omp_mpiData->kernel_globals_cpu,
    //              "data",
    //              &kg_data[0],
    //              kg_data.size());
    // displayFPS(2);
  }
  else if (data.client_tag == CLIENT_TAG_CYCLES_mem_zero &&
           data.client_mem_data.mem == g_path_trace_buffer) {
    pix_state[2] = 1;
  }
  else if (data.client_tag == CLIENT_TAG_CYCLES_mem_copy_to &&
           data.client_mem_data.mem == g_path_trace_buffer) {

    if (g_path_trace_buffer_received.size() != data.client_mem_data.memSize)
      g_path_trace_buffer_received.resize(data.client_mem_data.memSize);

    bcast(&g_path_trace_buffer_received[0], data.client_mem_data.memSize);
    // TODO: skip
  }
  else if (data.client_tag == CLIENT_TAG_CYCLES_frame_info) {
    frame_info(data);
    pix_state[2] = 1;
  }
  else {
    pix_state[4] = 1;
    data.world_size = 0;
    return;
  }

  //////////////////////////////////////////////////
  int world_rank = data.world_rank;
  int world_size = data.world_size;
  char *comm_data = data.comm_data;
  read_data_kernelglobal(&data, sizeof(client_kernel_struct));
  data.world_rank = world_rank;
  data.world_size = world_size;
  data.comm_data = comm_data;

  // printf("data.client_tag: %d \n", data.client_tag);

  // displayFPS(0);

  //////////////////////////////////////////////////
  //        double send_time2 = omp_get_wtime();
  //        double *st = (double*)&pix_state[6];
  //        st[0] = send_time2 - send_time;
  //        if (pix_state[0] == 1) {
  //          pix_state[0] = 0;
  //        }
  //        else if (pix_state[1] == 1) {
  //          pix_state[1] = 0;
  //        }
  //}
  //}

  // return true;
}

void path_trace_buffer_comm_init(client_kernel_struct &data)
{
}

void path_trace_buffer_comm_finish(client_kernel_struct &data,
                                   char *buffer_temp,
                                   size_t buffer_size)
{
  NET_NAMESPACE::send_data_data(
      (char *)&data.client_path_trace_data.num_samples, sizeof(int), false);
  NET_NAMESPACE::send_data_data(buffer_temp, buffer_size);
}

#endif

#if defined(WITH_CLIENT_MPI_SOCKET)
std::vector<char> g_path_trace_buffer_received;
std::vector<char> g_path_trace_pixels_send;
// std::vector<char> g_path_trace_pixels_received;
DEVICE_PTR g_path_trace_buffer = NULL;
DEVICE_PTR g_path_trace_pixels = NULL;

std::vector<int> g_displsPix;
std::vector<int> g_recvcountsPix;

int get_dev_node(client_kernel_struct &data)
{
  return data.world_rank;
}

int get_devices_size_node(client_kernel_struct &data)
{
  return data.world_size;
}

void path_trace_pixels_comm_init(client_kernel_struct &data, double &sender_time)
{
  const char *env_samples = getenv("DEBUG_SAMPLES");
  int num_samples = data.client_path_trace_data.num_samples;
  if (env_samples != NULL) {
    num_samples = atoi(env_samples);
  }

  g_cyclesphi_data.step_samples = num_samples;
  g_path_trace_buffer = data.client_path_trace_data.buffer;

#  if defined(WITH_CLIENT_NCCL_SOCKET) || defined(WITH_CLIENT_MPI_REDUCE)
  // data.client_path_trace_data.start_sample
#  endif
}

void path_trace_pixels_comm_finish(client_kernel_struct &data,
                                   char *pixels,
                                   char *pixels_node1,
                                   char *pixels_node2,
                                   int *pix_state)
{
  if (data.world_rank == 0 && NET_NAMESPACE::is_error()) {
    NET_NAMESPACE::server_close();
    NET_NAMESPACE::client_close();
  }
}

void path_trace_pixels_comm_thread(client_kernel_struct &data,
                                   // bool &do_exit,
                                   // omp_lock_t *lock0,
                                   // omp_lock_t *lock1,
                                   char *pixels,
                                   char *pixels_node1,
                                   char *pixels_node2,
                                   char *buffer2,
                                   char *buffer3,
                                   int *pix_state,
                                   double &sender_time,
                                   const char *drt,
                                   bool &buf_new,
                                   int &used_buffer)
{
  if (data.world_rank == 0 && NET_NAMESPACE::is_error()) {
    pix_state[4] = 1;
    // return;
  }

  // return;

  //  if (pix_state[6] == 1) {
  //    pix_state[6] = 0;
  //
  //#  pragma omp barrier
  //  }

  // if (data.world_rank == 0) {
  // int used_buffer = 0;
  // while (true) {

  double send_time = omp_get_wtime();
  // DEVICE_PTR path_trace_buffer = data.client_path_trace_data.buffer;

  //#  pragma omp flush

  if (data.client_tag == CLIENT_TAG_CYCLES_path_trace) {
    // while (true) {
    if (used_buffer == 0) {
      // omp_set_lock(lock0);
      // pix_state[0] = 1;
      // memcpy(&pixels[0], pixels_node1, pix_size);
      pixels = pixels_node1;
      // pix_state[1] = 0;
    }
    else if (used_buffer == 1) {
      // omp_set_lock(lock1);
      // pix_state[1] = 1;
      // memcpy(&pixels[0], pixels_node2, pix_size);
      pixels = pixels_node2;
      // pix_state[0] = 0;
    }
    //            else {
    //#  pragma omp flush
    //
    //              if (!do_exit)
    //                continue;
    //            }
    //            break;
    //          }

    int es_ = pix_state[3];
    // socket_send_data_data(&pixels[0], pixels.size());
    // if (es_ > 0)

#  if defined(WITH_CLIENT_NCCL_SOCKET) || defined(WITH_CLIENT_MPI_REDUCE)
    {
      int start_sample = data.client_path_trace_data.start_sample;
      int end_sample = data.client_path_trace_data.start_sample +
                       data.client_path_trace_data.num_samples;

      int pass_stride = data.client_path_trace_data.pass_stride;

      int offset = data.client_path_trace_data.offset;
      int stride = data.client_path_trace_data.stride;

      int tile_x = data.client_path_trace_data.tile_x;
      int tile_w = data.client_path_trace_data.tile_w;

#    if defined(WITH_CLIENT_RENDERENGINE_VR) || \
        (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
      tile_w *= 2;  // left+right
#    endif

      int tile_y = data.client_path_trace_data.tile_y;
      int tile_h = data.client_path_trace_data.tile_h;

      DEVICE_PTR _buffer = DEV_NAMESPACE::get_ptr_map(data.client_path_trace_data.buffer);

      if (g_path_trace_buffer_received.size() != tile_w * tile_h * pass_stride * sizeof(float))
        g_path_trace_buffer_received.resize(tile_w * tile_h * pass_stride * sizeof(float), 0);

#    ifdef WITH_CLIENT_UNIMEM
      float *buffer = (float *)_buffer;
#    else
      size_t pso = pass_stride * sizeof(float);
      DEV_NAMESPACE::mem_copy_from(
          CLIENT_DEVICE_ID, _buffer, buffer3, 0, tile_w * tile_h * pso, (char *)&pso);

      float *buffer = (float *)buffer3;
#    endif

      if (data.world_rank == 0) {

        // memcpy(buffer2, buffer, tile_w * tile_h * pass_stride * sizeof(float));

#    ifdef WITH_CLIENT_MPI_REDUCE
        // buffer2[0] = (float)pix_state[3]
        // memcpy(buffer3, buffer, sizeof(float) * tile_w * tile_h * pass_stride);
        buffer[0] = (float)pix_state[3];

        MPI_Reduce((char *)buffer,
                   (char *)buffer2,
                   tile_w * tile_h * pass_stride,
                   MPI_FLOAT,
                   MPI_SUM,
                   0,
                   MPI_COMM_WORLD);
#    else
        cudaSetDevice(0);

        buffer[0] = (float)pix_state[3];
        // cudaMemcpy(buffer3, buffer, sizeof(float) * tile_w * tile_h * pass_stride,
        // cudaMemcpyDefault); buffer3[0] = (float)pix_state[3];

        ncclResult_t res = ncclReduce(buffer,
                                      buffer2,
                                      tile_w * tile_h * pass_stride,
                                      ncclFloat,
                                      ncclSum,
                                      0,
                                      (ncclComm_t)data.comm_data,
                                      0);

        if (res != ncclSuccess) {
          printf("ncclReduce send != ncclSuccess\n");
          exit(-1);
        }

#    endif

        int num_samples_reduce = ((float *)buffer2)[0];

        if (num_samples_reduce > 0)
          DEV_NAMESPACE::buffer_to_pixels(0,
                                          (DEVICE_PTR)buffer2,
                                          (DEVICE_PTR)pixels_node2,
                                          0,
                                          // data.client_path_trace_data.start_sample,
                                          // data.client_path_trace_data.start_sample +
                                          //     data.client_path_trace_data.num_samples,
                                          num_samples_reduce,
                                          data.client_path_trace_data.tile_x,
                                          data.client_path_trace_data.tile_y,
                                          data.client_path_trace_data.offset,
                                          data.client_path_trace_data.stride,
                                          data.client_path_trace_data.tile_h,
                                          data.client_path_trace_data.tile_w,
                                          data.client_path_trace_data.tile_h,
                                          data.client_path_trace_data.tile_w);

        size_t pix_offset = (data.client_path_trace_data.tile_x +
                             data.client_path_trace_data.tile_y *
                                 data.client_path_trace_data.stride) *
                            SIZE_UCHAR4;

        size_t pix_size = data.client_path_trace_data.tile_h * data.client_path_trace_data.tile_w *
                          SIZE_UCHAR4;

#    if 0
        int *s = (int *)(pixels_node2);
        s[0] = es_;
#    else
        NET_NAMESPACE::send_data_data((char *)&es_, sizeof(int), false);
#    endif

#    ifdef WITH_CLIENT_GPUJPEG
        if (g_path_trace_pixels_send.size() != pix_size) {
          g_path_trace_pixels_send.resize(pix_size);
        }

        // only send not empty data
        NET_NAMESPACE::send_gpujpeg((char *)&pixels_node2[0],
                                    g_path_trace_pixels_send.data(),
                                    data.client_path_trace_data.tile_w,
                                    data.client_path_trace_data.tile_h);
#    else
        NET_NAMESPACE::send_data_data((char *)pixels_node2 + pix_offset, pix_size);
#    endif
      }
      else {
        // memcpy(buffer2, buffer, tile_w * tile_h * pass_stride * sizeof(float));

#    ifdef WITH_CLIENT_MPI_REDUCE
        // buffer2[0] = (float)pix_state[3];
        // memcpy(buffer3, buffer, sizeof(float) * tile_w * tile_h * pass_stride);
        buffer[0] = (float)pix_state[3];

        MPI_Reduce((char *)buffer,
                   (char *)buffer2,
                   tile_w * tile_h * pass_stride,
                   MPI_FLOAT,
                   MPI_SUM,
                   0,
                   MPI_COMM_WORLD);
#    else
        cudaSetDevice(0);
        // cudaMemcpy(buffer3, buffer, sizeof(float) * tile_w * tile_h * pass_stride,
        // cudaMemcpyDefault);
        buffer[0] = (float)pix_state[3];

        ncclResult_t res = ncclReduce(buffer,
                                      buffer2,
                                      tile_w * tile_h * pass_stride,
                                      ncclFloat,
                                      ncclSum,
                                      0,
                                      (ncclComm_t)data.comm_data,
                                      0);

        if (res != ncclSuccess) {
          printf("ncclReduce send != ncclSuccess\n");
          exit(-1);
        }

#    endif
      }
    }

#  else
    {
      ////////////////////////////one node///////////////////////////////////
      int dev_node = data.world_rank;           //         -1;
      int devices_size_node = data.world_size;  // -1;

      // int tile_step_node = data.client_path_trace_data.tile_h / devices_size_node;
      // ceil
      int tile_step_node = (int)ceil((float)data.client_path_trace_data.tile_h /
                                     (float)devices_size_node);

      int tile_last_node = data.client_path_trace_data.tile_h -
                           (devices_size_node - 1) * tile_step_node;

      if (tile_last_node < 1) {
        tile_step_node = (int)((float)data.client_path_trace_data.tile_h /
                               (float)(devices_size_node));

        tile_last_node = data.client_path_trace_data.tile_h -
                         (devices_size_node - 1) * tile_step_node;
      }

      int tile_y_node = data.client_path_trace_data.tile_y + tile_step_node * dev_node;
      int tile_h_node = (devices_size_node - 1 == dev_node) ? tile_last_node : tile_step_node;

#    ifdef WITH_CLIENT_GPUJPEG

      size_t pix_size_node = tile_h_node * data.client_path_trace_data.tile_w;
      size_t full_tile_size = data.client_path_trace_data.tile_h *
                              data.client_path_trace_data.tile_w;

#      if defined(WITH_CLIENT_RENDERENGINE_VR) || \
          (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
      pix_size_node *= 2;  // left+right
      full_tile_size *= 2;

      size_t pix_offset = (data.client_path_trace_data.tile_x +
                           tile_y_node * data.client_path_trace_data.stride * 2);
#      else
      size_t pix_offset = (data.client_path_trace_data.tile_x +
                           tile_y_node * data.client_path_trace_data.stride);
#      endif

      if (g_displsPix.size() != devices_size_node) {
        g_displsPix.resize(devices_size_node, 0);
        g_recvcountsPix.resize(devices_size_node, 0);
      }

      ///////////////////////// Y

      for (int dev = 0; dev < devices_size_node; dev++) {
        int tile_y2 = data.client_path_trace_data.tile_y + tile_step_node * dev;
        int tile_h2 = (devices_size_node - 1 == dev) ? tile_last_node : tile_step_node;
        if (tile_h2 == 0)
          continue;

        g_displsPix[dev] = (data.client_path_trace_data.offset +
                            data.client_path_trace_data.tile_x +
                            tile_y2 * data.client_path_trace_data.stride);

        g_recvcountsPix[dev] = data.client_path_trace_data.tile_w * tile_h2;
      }

      // y
      MPI_Gatherv((char *)&pixels[0] + pix_offset,
                  pix_size_node,
                  MPI_BYTE,
                  // g_path_trace_pixels_received.data(),
                  pixels_node2,
                  &g_recvcountsPix[0],
                  &g_displsPix[0],
                  MPI_BYTE,
                  0,
                  MPI_COMM_WORLD);

      for (int dev = 0; dev < devices_size_node; dev++) {
        g_displsPix[dev] += full_tile_size;
      }

      // u
      MPI_Gatherv((char *)&pixels[0] + full_tile_size + pix_offset,
                  pix_size_node,
                  MPI_BYTE,
                  // g_path_trace_pixels_received.data(),
                  pixels_node2,
                  &g_recvcountsPix[0],
                  &g_displsPix[0],
                  MPI_BYTE,
                  0,
                  MPI_COMM_WORLD);

      for (int dev = 0; dev < devices_size_node; dev++) {
        g_displsPix[dev] += full_tile_size;
      }

      // v
      MPI_Gatherv((char *)&pixels[0] + 2 * full_tile_size + pix_offset,
                  pix_size_node,
                  MPI_BYTE,
                  // g_path_trace_pixels_received.data(),
                  pixels_node2,
                  &g_recvcountsPix[0],
                  &g_displsPix[0],
                  MPI_BYTE,
                  0,
                  MPI_COMM_WORLD);

      //#  ifndef WITH_CLIENT_MPI_VRCLIENT
      //      int global_min;  // = pix_state[3];
      //      int local_min = pix_state[3];
      //
      //      MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
      //#  endif

#      if 0
        int *s = (int *)(pixels_node2);
        s[0] = es_;
#      else
      NET_NAMESPACE::send_data_data((char *)&es_, sizeof(int), false);
#      endif

#      ifdef WITH_VRCLIENT_RECEIVER
      s[1] = data.client_path_trace_data.tile_w;
      s[2] = tile_h_node;
#      endif

      if (g_path_trace_pixels_send.size() != pix_size_node) {
        g_path_trace_pixels_send.resize(pix_size_node);
      }

      // only send not empty data
      if (data.world_rank == 0 /* && es_ > 0*/) {
        NET_NAMESPACE::send_gpujpeg(pixels_node2,
                                    g_path_trace_pixels_send.data(),
                                    data.client_path_trace_data.tile_w,
                                    data.client_path_trace_data.tile_h);
      }
#    else

      size_t pix_size_node = tile_h_node * data.client_path_trace_data.tile_w * SIZE_UCHAR4;

#      if defined(WITH_CLIENT_RENDERENGINE_VR) || \
          (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
      pix_size_node *= 2;  // left+right

      size_t pix_offset_node = (data.client_path_trace_data.tile_x +
                                tile_y_node * data.client_path_trace_data.stride * 2) *
                               SIZE_UCHAR4;
#      else
      size_t pix_offset_node = (data.client_path_trace_data.tile_x +
                                tile_y_node * data.client_path_trace_data.stride) *
                               SIZE_UCHAR4;
#      endif
      // if (g_path_trace_pixels_received.size() != pix_size_node) {
      //  g_path_trace_pixels_received.resize(pix_size_node);
      //}

      if (g_displsPix.size() != devices_size_node) {
        g_displsPix.resize(devices_size_node, 0);
        g_recvcountsPix.resize(devices_size_node, 0);
      }

      for (int dev = 0; dev < devices_size_node; dev++) {
        int tile_y2 = data.client_path_trace_data.tile_y + tile_step_node * dev;
        int tile_h2 = (devices_size_node - 1 == dev) ? tile_last_node : tile_step_node;
        if (tile_h2 == 0)
          continue;

        g_displsPix[dev] = (data.client_path_trace_data.offset +
                            data.client_path_trace_data.tile_x +
                            tile_y2 * data.client_path_trace_data.stride) *
                           SIZE_UCHAR4;
        g_recvcountsPix[dev] = data.client_path_trace_data.tile_w * tile_h2 * SIZE_UCHAR4;
      }

      MPI_Gatherv((char *)&pixels[0] + pix_offset_node,
                  pix_size_node,
                  MPI_BYTE,
                  // g_path_trace_pixels_received.data(),
                  pixels_node2,
                  &g_recvcountsPix[0],
                  &g_displsPix[0],
                  MPI_BYTE,
                  0,
                  MPI_COMM_WORLD);

      //#  ifndef WITH_CLIENT_MPI_VRCLIENT
      //      int global_min;  // = pix_state[3];
      //      int local_min = pix_state[3];
      //
      //      MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
      //#  endif

      if (data.world_rank == 0 /*&& es_ > 0*/) {

#      if 0
        int *s = (int *)(pixels_node2);
        s[0] = es_;
#      else
        NET_NAMESPACE::send_data_data((char *)&es_, sizeof(int), false);
#      endif

#      ifdef WITH_VRCLIENT_RECEIVER
        s[1] = data.client_path_trace_data.tile_w;
        s[2] = tile_h_node;
#      endif

        size_t pix_offset = (data.client_path_trace_data.tile_x +
                             data.client_path_trace_data.tile_y *
                                 data.client_path_trace_data.stride) *
                            SIZE_UCHAR4;

        size_t pix_size = data.client_path_trace_data.tile_h * data.client_path_trace_data.tile_w *
                          SIZE_UCHAR4;

        NET_NAMESPACE::send_data_data((char *)pixels_node2 + pix_offset, pix_size);
      }
#    endif

#    ifdef WITH_VRCLIENT_RECEIVER
      NET_NAMESPACE::send_data_cam((char *)s, sizeof(int));
#    endif

      // int global_min;  // = pix_state[3];
      // int local_min = pix_state[3];

      // MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
      // kernel::socket_send_data_data((char *)&local_min, sizeof(int));
    }

#  endif
    // displayFPS(1);

    //////////////////////////////////////////////////

    //          if (pix_state[0] == 1) {
    //            pix_state[0] = 0;
    //          }
    //          else if (pix_state[1] == 1) {
    //            pix_state[1] = 0;
    //          }
    if (used_buffer == 0 /*pix_state[0] == 2*/) {
      // pix_state[0] = 0;
      // printf("send2: pix_state[0] = 0, %f\n", omp_get_wtime()); fflush(0);
      // omp_unset_lock(lock0);
    }
    else if (used_buffer == 1 /*pix_state[1] == 2*/) {
      // pix_state[1] = 0;
      // printf("send2: pix_state[1] = 0, %f\n", omp_get_wtime()); fflush(0);
      // omp_unset_lock(lock1);
    }
    used_buffer++;
    if (used_buffer >= CLIENT_SWAP_BUFFERS)
      used_buffer = 0;

    // if (es_ == 0)
    // return;
  }
  else if (data.client_tag == CLIENT_TAG_CYCLES_const_copy) {
    bcast(&kg_data[0], kg_data.size());

    // if(pix_state[2] == 0)
    {
      char *gpu_data = (char *)DEV_NAMESPACE::get_data(DEV_NAMESPACE::get_kernel_globals_cpu());
      // if (memcmp(gpu_data, (char *)&socket_kg_data[0], socket_kg_data.size())) {
      memcpy(gpu_data, (char *)&kg_data[0], kg_data.size());

#  if defined(WITH_CLIENT_CUDA) || defined(WITH_CLIENT_HIP)
      DEV_NAMESPACE::cam_recalc(gpu_data);

      // pix_state[2] = 1;
      // printf("socket_kg_data.size(): %lld\n", socket_kg_data.size());
      //}

#    if defined(WITH_CLIENT_RENDERENGINE_VR) || \
        (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
      char *gpu_data_right = (char *)DEV_NAMESPACE::get_data_right(
          DEV_NAMESPACE::get_kernel_globals_cpu());
      memcpy(gpu_data_right, (char *)&socket_kg_data[0], socket_kg_data.size());

      DEV_NAMESPACE::cam_recalc(gpu_data_right);
#    endif
#  endif
    }

    // kernel::omp_const_copy(OMP_DEVICE_ID,
    //              kernel::omp_mpiData->kernel_globals_cpu,
    //              "data",
    //              &kg_data[0],
    //              kg_data.size());
    // displayFPS(2);
  }
  else if (data.client_tag == CLIENT_TAG_CYCLES_mem_zero &&
           data.client_mem_data.mem == g_path_trace_buffer) {
    pix_state[2] = 1;
  }
  else if (data.client_tag == CLIENT_TAG_CYCLES_mem_copy_to &&
           data.client_mem_data.mem == g_path_trace_buffer) {

    if (g_path_trace_buffer_received.size() != data.client_mem_data.memSize)
      g_path_trace_buffer_received.resize(data.client_mem_data.memSize);

    bcast(&g_path_trace_buffer_received[0], data.client_mem_data.memSize);
    // TODO: skip
  }
  else {
    pix_state[4] = 1;
    data.world_size = 0;
    return;
  }

  //////////////////////////////////////////////////
  int world_rank = data.world_rank;
  int world_size = data.world_size;
  char *comm_data = data.comm_data;
  read_data_kernelglobal(&data, sizeof(client_kernel_struct));
  data.world_rank = world_rank;
  data.world_size = world_size;
  data.comm_data = comm_data;

  // printf("data.client_tag: %d \n", data.client_tag);

  // displayFPS(0);

  //////////////////////////////////////////////////
  //        double send_time2 = omp_get_wtime();
  //        double *st = (double*)&pix_state[6];
  //        st[0] = send_time2 - send_time;
  //        if (pix_state[0] == 1) {
  //          pix_state[0] = 0;
  //        }
  //        else if (pix_state[1] == 1) {
  //          pix_state[1] = 0;
  //        }
  //}
  //}

  // return true;
}

void path_trace_buffer_comm_init(client_kernel_struct &data)
{
}

void path_trace_buffer_comm_finish(client_kernel_struct &data,
                                   char *buffer_temp,
                                   size_t buffer_size)
{

  int dev_node = data.world_rank;           //         -1;
  int devices_size_node = data.world_size;  // -1;

  // int tile_step_node = data.client_path_trace_data.tile_h / devices_size_node;
  // ceil
  int tile_step_node = (int)ceil((float)data.client_path_trace_data.tile_h /
                                 (float)devices_size_node);

  int tile_last_node = data.client_path_trace_data.tile_h -
                       (devices_size_node - 1) * tile_step_node;

  if (tile_last_node < 1) {
    tile_step_node = (int)((float)data.client_path_trace_data.tile_h / (float)(devices_size_node));

    tile_last_node = data.client_path_trace_data.tile_h - (devices_size_node - 1) * tile_step_node;
  }

  int tile_y_node = data.client_path_trace_data.tile_y + tile_step_node * dev_node;
  int tile_h_node = (devices_size_node - 1 == dev_node) ? tile_last_node : tile_step_node;

  size_t buff_size_node = tile_h_node * data.client_path_trace_data.tile_w *
                          data.client_path_trace_data.pass_stride * sizeof(float);

#  if defined(WITH_CLIENT_RENDERENGINE_VR) || \
      (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
  buff_size_node *= 2;  // left+right

  size_t buff_offset = (data.client_path_trace_data.tile_x +
                        tile_y_node * data.client_path_trace_data.stride * 2) *
                       data.client_path_trace_data.pass_stride * sizeof(float);
#  else
  size_t buff_offset = (data.client_path_trace_data.tile_x +
                        tile_y_node * data.client_path_trace_data.stride) *
                       data.client_path_trace_data.pass_stride * sizeof(float);
#  endif

  if (g_displsPix.size() != devices_size_node) {
    g_displsPix.resize(devices_size_node, 0);
    g_recvcountsPix.resize(devices_size_node, 0);
  }

  for (int dev = 0; dev < devices_size_node; dev++) {
    int tile_y2 = data.client_path_trace_data.tile_y + tile_step_node * dev;
    int tile_h2 = (devices_size_node - 1 == dev) ? tile_last_node : tile_step_node;
    if (tile_h2 == 0)
      continue;

    g_displsPix[dev] = (data.client_path_trace_data.offset + data.client_path_trace_data.tile_x +
                        tile_y2 * data.client_path_trace_data.stride) *
                       data.client_path_trace_data.pass_stride * sizeof(float);
    g_recvcountsPix[dev] = data.client_path_trace_data.tile_w * tile_h2 *
                           data.client_path_trace_data.pass_stride * sizeof(float);
  }

  if (g_path_trace_buffer_received.size() != buffer_size)
    g_path_trace_buffer_received.resize(buffer_size);

  MPI_Gatherv((char *)buffer_temp + buff_offset,
              buff_size_node,
              MPI_BYTE,
              g_path_trace_buffer_received.data(),
              &g_recvcountsPix[0],
              &g_displsPix[0],
              MPI_BYTE,
              0,
              MPI_COMM_WORLD);

  if (data.world_rank == 0)
    NET_NAMESPACE::send_data_data(g_path_trace_buffer_received.data(), buffer_size);
}
#endif

////////////////////////////////////////////////////////

void path_trace_pixels(client_kernel_struct &data)
{
  bool buf_new = true;
  double sender_time = -1;

  path_trace_pixels_comm_init(data, sender_time);

  int num_samples = get_image_sample();

  size_t pix_size = data.client_path_trace_data.tile_w * data.client_path_trace_data.tile_h *
                    SIZE_UCHAR4;

#if defined(WITH_CLIENT_RENDERENGINE_VR) || \
    (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
  pix_size *= 2;  // left+right
#endif

#if defined(WITH_CLIENT_NCCL_SOCKET) || defined(WITH_CLIENT_MPI_REDUCE)
  // char *pixels_node = NULL;
  // char *pixels_node1 = NULL;
  // char *pixels_node2 = NULL;
  // char *pixels = NULL;
  char *buffer2 = DEV_NAMESPACE::host_alloc("buffer2",
                                            NULL,
                                            data.client_path_trace_data.pass_stride *
                                                data.client_path_trace_data.tile_w *
#  if defined(WITH_CLIENT_RENDERENGINE_VR) || \
      (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
                                                2 *
#  endif
                                                data.client_path_trace_data.tile_h * sizeof(float),
                                            2);

  char *buffer3 = DEV_NAMESPACE::host_alloc("buffer3",
                                            NULL,
                                            data.client_path_trace_data.pass_stride *
                                                data.client_path_trace_data.tile_w *
#  if defined(WITH_CLIENT_RENDERENGINE_VR) || \
      (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
                                                2 *
#  endif
                                                data.client_path_trace_data.tile_h * sizeof(float),
                                            2);

#  ifdef WITH_CLIENT_NCCL_SOCKET
  cudaStream_t nccl_stream;
  cudaSetDevice(0);
  cudaStreamCreateWithFlags(&nccl_stream, cudaStreamNonBlocking);
#  endif

#else
  char *buffer2 = NULL;
  char *buffer3 = NULL;
#endif

  char *pixels_node = NULL;
  char *pixels_node1 = NULL;
  char *pixels_node2 = NULL;
  char *pixels = NULL;

#if defined(WITH_CUDA_BUFFER_MANAGED) || defined(WITH_HIP_BUFFER_MANAGED) || \
    defined(WITH_CLIENT_UNIMEM)
  pixels_node = DEV_NAMESPACE::host_alloc("pixels_node", NULL, 2 * pix_size, 1);
#else
  pixels_node = DEV_NAMESPACE::host_alloc("pixels_node", NULL, 2 * pix_size, 2);
#endif

  pixels_node1 = &pixels_node[0];
  pixels_node2 = &pixels_node[pix_size];

  pixels = pixels_node1;

  /////^///
  int start_sample = data.client_path_trace_data.start_sample;
  int end_sample = data.client_path_trace_data.start_sample + num_samples;

  // receive_cam
  // file_socket_step_cam(data);
  bool buffer_init = true;
  bool buf_reset = true;

  int flag_exit = 0;
  int last_state = 0;

#if defined(WITH_CLIENT_RENDERENGINE_EMULATE)
  const char *drt = getenv("DEBUG_REPEAT_TIME");
  double drt_time_start = omp_get_wtime();
#else
  const char *drt = NULL;
  double drt_time_start = 0;
#endif

  int pix_state[10];
  pix_state[0] = 0;             // state buff A
  pix_state[1] = 0;             // state buff B
  pix_state[2] = 0;             // new data
  pix_state[3] = start_sample;  // current samples
  pix_state[4] = 0;             // exit
  pix_state[5] = 0;             // start render
  pix_state[6] = 0;             // barrier
  pix_state[7] = 0;
  //  pix_state[8] = 0; //maxtime1
  //  pix_state[9] = 0; //maxtime2
  // pix_state[2] = data.client_path_trace_data.num_samples;  // num_samples
  // client_kernel_struct data2;

  int world_rank = data.world_rank;
  int world_size = data.world_size;
  char *comm_data = data.comm_data;

  /////////
#ifdef WITH_CLIENT_MPI_VRCLIENT
  mpi_socket_step_image_size(data);
  mpi_cam_change(data);
#endif

  ////////
  size_t offsetSample = 0;
  size_t sizeSample = sizeof(int);

  int reqFinished = 0;
  int sample_finished_cpu = 0;

  // int start_sample = data.client_path_trace_data.start_sample;

  // int end_sample = data.client_path_trace_data.start_sample + num_samples;
  // int pass_stride = DEV_NAMESPACE::get_pass_stride(CLIENT_DEVICE_ID,
  //                                                 DEV_NAMESPACE::get_kernel_globals_cpu());

  int offset = data.client_path_trace_data.offset;
  int stride = data.client_path_trace_data.stride;

  int tile_x = data.client_path_trace_data.tile_x;
  int tile_w = data.client_path_trace_data.tile_w;

  int tile_y = data.client_path_trace_data.tile_y;
  int tile_h = data.client_path_trace_data.tile_h;

#if defined(WITH_CLIENT_NCCL_SOCKET) || defined(WITH_CLIENT_MPI_REDUCE)
  int tile_y_node = tile_y;
  int tile_h_node = tile_h;
#else
  ////////////////////////////one node///////////////////////////////////
  int dev_node = get_dev_node(data);
  int devices_size_node = get_devices_size_node(data);

  // int tile_step_node = data.client_path_trace_data.tile_h / devices_size_node;
  // ceil
  int tile_step_node = (int)ceil((float)data.client_path_trace_data.tile_h /
                                 (float)devices_size_node);

  int tile_last_node = data.client_path_trace_data.tile_h -
                       (devices_size_node - 1) * tile_step_node;

  if (tile_last_node < 1) {
    tile_step_node = (int)((float)data.client_path_trace_data.tile_h / (float)(devices_size_node));

    tile_last_node = data.client_path_trace_data.tile_h - (devices_size_node - 1) * tile_step_node;
  }

  int tile_y_node = data.client_path_trace_data.tile_y + tile_step_node * dev_node;
  int tile_h_node = (devices_size_node - 1 == dev_node) ? tile_last_node : tile_step_node;
#endif

  int size_node = tile_h_node * tile_w;

  DEV_NAMESPACE::init_execution(data.client_path_trace_data.has_shadow_catcher,
                                data.client_path_trace_data.max_shaders,
                                data.client_path_trace_data.pass_stride,
                                data.client_path_trace_data.kernel_features,
                                data.client_path_trace_data.volume_stack_size);

  ////////
  //#ifdef WITH_CLIENT_CPU
  omp_set_nested(1);
  //#endif

  // omp_lock_t *lock0 = (omp_lock_t *)&pix_state[6];
  // omp_lock_t *lock1 = (omp_lock_t *)&pix_state[8];

  // omp_init_lock(lock0);
  // omp_init_lock(lock1);

  // bool do_exit = false;
  int used_buffer = 0;

  //#ifdef WITH_CLIENT_CPU
  //  int n_threads = 2;
  //#else
  //  int n_threads = DEV_NAMESPACE::get_cpu_threads() + 1;
  //#endif

  // #if defined(WITH_CLIENT_RENDERENGINE_EMULATE)
  //   if (drt == NULL)
  //     do_exit = true;
  // #endif

#if defined(WITH_CLIENT_NCCL_SOCKET) || defined(WITH_CLIENT_MPI_REDUCE)
  pixels_node = 0;
#endif

  DEVICE_PTR kernel_globals_cpu = DEV_NAMESPACE::get_kernel_globals_cpu();
  DEVICE_PTR render_buffer = DEV_NAMESPACE::get_ptr_map(data.client_path_trace_data.buffer);

#ifdef WITH_CLIENT_CPU
  int cpu_threads = 1;
#else
  int cpu_threads = DEV_NAMESPACE::get_cpu_threads();
#endif

#pragma omp parallel num_threads(2)
  {
    int id = omp_get_thread_num();

    //#pragma omp flush

    if (id == 1) {

      while (true) {
        // #if defined(WITH_CLIENT_NCCL_SOCKET) || defined(WITH_CLIENT_MPI_REDUCE)
        //         if (data.world_rank == 0)
        //           break;
        // #endif

        DEV_NAMESPACE::path_trace_init(CLIENT_DEVICE_ID,
                                       kernel_globals_cpu,
                                       render_buffer,
                                       (DEVICE_PTR)pixels_node,
                                       start_sample,
                                       end_sample,
                                       tile_x,
                                       tile_y_node,
                                       offset,
                                       stride,
                                       tile_h_node,
                                       tile_w,
                                       tile_h,
                                       tile_w,
                                       (char *)&sample_finished_cpu,
                                       (char *)&reqFinished,
                                       cpu_threads,
                                       (char *)&pix_state);

#pragma omp parallel num_threads(cpu_threads)
        {
          DEV_NAMESPACE::path_trace(CLIENT_DEVICE_ID,
                                    kernel_globals_cpu,
                                    render_buffer,
                                    (DEVICE_PTR)pixels_node,
                                    start_sample,
                                    end_sample,
                                    tile_x,
                                    tile_y_node,
                                    offset,
                                    stride,
                                    tile_h_node,
                                    tile_w,
                                    tile_h,
                                    tile_w,
                                    (char *)&sample_finished_cpu,
                                    (char *)&reqFinished,
                                    cpu_threads,
                                    (char *)&pix_state);
        }

        DEV_NAMESPACE::path_trace_finish(CLIENT_DEVICE_ID,
                                         kernel_globals_cpu,
                                         render_buffer,
                                         (DEVICE_PTR)pixels_node,
                                         start_sample,
                                         end_sample,
                                         tile_x,
                                         tile_y_node,
                                         offset,
                                         stride,
                                         tile_h_node,
                                         tile_w,
                                         tile_h,
                                         tile_w,
                                         (char *)&sample_finished_cpu,
                                         (char *)&reqFinished,
                                         cpu_threads,
                                         (char *)&pix_state);

        if (pix_state[4] == 1) {
          printf("thread: %d finished\n", id);
          fflush(0);
          break;
        }

        displayFPS(id);
      }

      // do_exit = true;

      //#  pragma omp flush
    }

    if (id == 0) {
      while (true) {

        path_trace_pixels_comm_thread(data,
                                      // bool &do_exit,
                                      // lock0,
                                      // lock1,
                                      pixels,
                                      pixels_node1,
                                      pixels_node2,
                                      //(char*)render_buffer,
                                      buffer2,
                                      buffer3,
                                      pix_state,
                                      sender_time,
                                      drt,
                                      buf_new,
                                      used_buffer);

        if (pix_state[4] == 1) {
          printf("thread: %d finished\n", id);
          fflush(0);
          break;
        }

        displayFPS(id);
      }
    }
    //}

    // #if defined(ENABLE_INC_SAMPLES) && !defined(ENABLE_STEP_SAMPLES)
    //       start_sample2 = start_sample2 + num_samples;
    //       end_sample2 = start_sample2 + num_samples;
    // #endif

    //#pragma omp barrier

    // #if defined(WITH_CLIENT_RENDERENGINE_EMULATE)
    //       if (drt == NULL || omp_get_wtime() - drt_time_start > atof(drt)) {
    //         break;
    //       }
    // #endif
    //}
    //}
  }

  path_trace_pixels_comm_finish(data, pixels, pixels_node1, pixels_node2, pix_state);

#if defined(WITH_CLIENT_NCCL_SOCKET) || defined(WITH_CLIENT_MPI_REDUCE)

#  ifdef WITH_CLIENT_NCCL_SOCKET
  cudaStreamDestroy(nccl_stream);
#  endif

  DEV_NAMESPACE::host_free("buffer2", NULL, buffer2, 2);
  DEV_NAMESPACE::host_free("buffer3", NULL, buffer3, 2);
#else

#  if defined(WITH_CUDA_BUFFER_MANAGED) || defined(WITH_HIP_BUFFER_MANAGED) || \
      defined(WITH_CLIENT_UNIMEM)
  DEV_NAMESPACE::host_free("pixels_node", NULL, pixels_node, 1);
#  else
  DEV_NAMESPACE::host_free("pixels_node", NULL, pixels_node, 2);
#  endif

#endif

  // omp_destroy_lock(lock0);
  // omp_destroy_lock(lock1);
}

void path_trace_buffer(client_kernel_struct &data)
{
#if defined(WITH_CLIENT_RENDERENGINE_EMULATE) || defined(WITH_CLIENT_RENDERENGINE)
  path_trace_pixels(data);
  return;
#endif

  path_trace_buffer_comm_init(data);

  int num_samples = data.client_path_trace_data.num_samples;

  /////^///
  int start_sample = data.client_path_trace_data.start_sample;
  int end_sample = data.client_path_trace_data.start_sample + num_samples;

  size_t offsetSample = 0;
  size_t sizeSample = sizeof(int);

  int reqFinished = 0;
  int sample_finished_cpu = 0;

  int offset = data.client_path_trace_data.offset;
  int stride = data.client_path_trace_data.stride;

  int tile_x = data.client_path_trace_data.tile_x;
  int tile_w = data.client_path_trace_data.tile_w;

  int tile_y = data.client_path_trace_data.tile_y;
  int tile_h = data.client_path_trace_data.tile_h;

  ////////////////////////////one node///////////////////////////////////
  int dev_node = get_dev_node(data);
  int devices_size_node = get_devices_size_node(data);

  // int tile_step_node = data.client_path_trace_data.tile_h / devices_size_node;
  // ceil
  int tile_step_node = (int)ceil((float)data.client_path_trace_data.tile_h /
                                 (float)devices_size_node);

  int tile_last_node = data.client_path_trace_data.tile_h -
                       (devices_size_node - 1) * tile_step_node;

  if (tile_last_node < 1) {
    tile_step_node = (int)((float)data.client_path_trace_data.tile_h / (float)(devices_size_node));

    tile_last_node = data.client_path_trace_data.tile_h - (devices_size_node - 1) * tile_step_node;
  }

  int tile_y_node = data.client_path_trace_data.tile_y + tile_step_node * dev_node;
  int tile_h_node = (devices_size_node - 1 == dev_node) ? tile_last_node : tile_step_node;

  int size_node = tile_h_node * tile_w;

  DEV_NAMESPACE::init_execution(data.client_path_trace_data.has_shadow_catcher,
                                data.client_path_trace_data.max_shaders,
                                data.client_path_trace_data.pass_stride,
                                data.client_path_trace_data.kernel_features,
                                data.client_path_trace_data.volume_stack_size);

  DEVICE_PTR kernel_globals_cpu = DEV_NAMESPACE::get_kernel_globals_cpu();
  DEVICE_PTR render_buffer = DEV_NAMESPACE::get_ptr_map(data.client_path_trace_data.buffer);

  omp_set_nested(1);

#ifdef WITH_CLIENT_CPU
  int cpu_threads = 1;
#else
  int cpu_threads = DEV_NAMESPACE::get_cpu_threads();
#endif

  DEV_NAMESPACE::path_trace_init(CLIENT_DEVICE_ID,
                                 kernel_globals_cpu,
                                 render_buffer,
                                 NULL,
                                 start_sample,
                                 end_sample,
                                 tile_x,
                                 tile_y_node,
                                 offset,
                                 stride,
                                 tile_h_node,
                                 tile_w,
                                 tile_h,
                                 tile_w,
                                 (char *)&sample_finished_cpu,
                                 (char *)&reqFinished,
                                 cpu_threads,
                                 NULL);

#pragma omp parallel num_threads(cpu_threads)
  {
    DEV_NAMESPACE::path_trace(CLIENT_DEVICE_ID,
                              kernel_globals_cpu,
                              render_buffer,
                              NULL,
                              start_sample,
                              end_sample,
                              tile_x,
                              tile_y_node,
                              offset,
                              stride,
                              tile_h_node,
                              tile_w,
                              tile_h,
                              tile_w,
                              (char *)&sample_finished_cpu,
                              (char *)&reqFinished,
                              cpu_threads,
                              NULL);
  }

  DEV_NAMESPACE::path_trace_finish(CLIENT_DEVICE_ID,
                                   kernel_globals_cpu,
                                   render_buffer,
                                   NULL,
                                   start_sample,
                                   end_sample,
                                   tile_x,
                                   tile_y_node,
                                   offset,
                                   stride,
                                   tile_h_node,
                                   tile_w,
                                   tile_h,
                                   tile_w,
                                   (char *)&sample_finished_cpu,
                                   (char *)&reqFinished,
                                   cpu_threads,
                                   NULL);

  size_t pso = data.client_path_trace_data.pass_stride * sizeof(float);

  char *buffer_temp = DEV_NAMESPACE::host_alloc("buffer_temp", NULL, tile_w * tile_h * pso, 2);

  DEV_NAMESPACE::mem_copy_from(
      CLIENT_DEVICE_ID, render_buffer, buffer_temp, offset, tile_w * tile_h * pso, (char *)&pso);

  path_trace_buffer_comm_finish(data, buffer_temp, tile_w * tile_h * pso);

  DEV_NAMESPACE::host_free("buffer_temp", NULL, buffer_temp, 2);
}

void save_buffer(client_kernel_struct &data)
{
  if (data.world_rank != 0)
    return;

#if !defined(WITH_CLIENT_RENDERENGINE) && !defined(WITH_CLIENT_RENDERENGINE_EMULATE) && \
    !defined(WITH_CLIENT_ULTRAGRID)
  size_t offsetBuf = (data.client_path_trace_data.offset + data.client_path_trace_data.tile_x +
                      data.client_path_trace_data.tile_y * data.client_path_trace_data.stride) *
                     data.client_path_trace_data.pass_stride * sizeof(float);
  size_t sizeBuf = data.client_path_trace_data.tile_h * data.client_path_trace_data.tile_w *
                   data.client_path_trace_data.pass_stride * sizeof(float);

#  if defined(WITH_CLIENT_RENDERENGINE) || defined(WITH_CLIENT_ULTRAGRID)
  int num_samples = *get_image_sample();
#  else
  const char *env_samples = getenv("DEBUG_SAMPLES");
  int num_samples = data.client_path_trace_data.num_samples;
  if (env_samples != NULL) {
    num_samples = atoi(env_samples);
  }
#  endif

#  if (defined(WITH_CLIENT_MPI_FILE) || defined(WITH_CLIENT_FILE))

  std::vector<char> buffer(sizeBuf);

  size_t pass_stride_float = data.client_path_trace_data.pass_stride * sizeof(float);

#    if defined(WITH_CLIENT_CUDA_CPU_STAT)
  DEV_NAMESPACE::mem_copy_from(
      CLIENT_DEVICE_ID,
      DEV_NAMESPACE::get_ptr_map(data.client_path_trace_data.buffer),
      &buffer[0],
      offsetBuf,
      buffer.size(),
      (char *)&pass_stride_float);
#    else
//////////////
#     if !defined(WITH_CLIENT_MPI_FILE_1)
  DEV_NAMESPACE::mem_copy_from(
      CLIENT_DEVICE_ID,
      DEV_NAMESPACE::get_ptr_map(data.client_path_trace_data.buffer),
      &buffer[0],
      offsetBuf,
      buffer.size(),
      (char *)&pass_stride_float);
#     else
  DEV_NAMESPACE::mem_copy_from(
      CLIENT_DEVICE_ID,
      // g_path_trace_buffer_received.data(), // only available in CPU arch
      DEV_NAMESPACE::get_ptr_map(data.client_path_trace_data.buffer),
      &buffer[0],
      offsetBuf,
      buffer.size(),
      (char *)&pass_stride_float);
#     endif
#    endif

  NET_NAMESPACE::write_cycles_buffer(
      &num_samples,
      &buffer[0],  // omp_mpiData->ptr_map[data.client_path_trace_data.buffer],
      offsetBuf,
      sizeBuf);

#    ifdef WITH_CLIENT_BUFFER_DECODER

#      if defined(WITH_CLIENT_CPU)
  NET_NAMESPACE::write_cycles_data(
      getenv("CLIENT_FILE_CYCLES_KDATA"),
      (char *)DEV_NAMESPACE::get_data(DEV_NAMESPACE::get_kernel_globals_cpu()),
      0,
      DEV_NAMESPACE::get_size_data(DEV_NAMESPACE::get_kernel_globals_cpu()));
  NET_NAMESPACE::write_cycles_data(getenv("CLIENT_FILE_CYCLES_PATHTRACER_DATA"),
                                   (char *)&data.client_path_trace_data,
                                   0,
                                   sizeof(client_path_trace_struct));
#      endif
#      if defined(WITH_CLIENT_CUDA)
  NET_NAMESPACE::write_cycles_data(
      getenv("CLIENT_FILE_CYCLES_KDATA"),
      (char *)DEV_NAMESPACE::get_data(DEV_NAMESPACE::get_kernel_globals_cpu()),
      0,
      DEV_NAMESPACE::get_size_data(DEV_NAMESPACE::get_kernel_globals_cpu()));
  NET_NAMESPACE::write_cycles_data(getenv("CLIENT_FILE_CYCLES_PATHTRACER_DATA"),
                                   (char *)&data.client_path_trace_data,
                                   0,
                                   sizeof(client_path_trace_struct));
#      endif
#      if defined(WITH_CLIENT_OPTIX)
  NET_NAMESPACE::write_cycles_data(
      getenv("CLIENT_FILE_CYCLES_KDATA"),
      (char *)DEV_NAMESPACE::get_data(kernel::optix_mpiData->kernel_globals_cpu),
      0,
      DEV_NAMESPACE::get_size_data(kernel::optix_mpiData->kernel_globals_cpu));
  NET_NAMESPACE::write_cycles_data(getenv("CLIENT_FILE_CYCLES_PATHTRACER_DATA"),
                                   (char *)&data.client_path_trace_data,
                                   0,
                                   sizeof(client_path_trace_struct));
#      endif
#      if defined(WITH_CLIENT_HIP)
  NET_NAMESPACE::write_cycles_data(
      getenv("CLIENT_FILE_CYCLES_KDATA"),
      (char *)DEV_NAMESPACE::get_data(kernel::hip_mpiData->kernel_globals_cpu),
      0,
      DEV_NAMESPACE::get_size_data(kernel::hip_mpiData->kernel_globals_cpu));
  NET_NAMESPACE::write_cycles_data(getenv("CLIENT_FILE_CYCLES_PATHTRACER_DATA"),
                                   (char *)&data.client_path_trace_data,
                                   0,
                                   sizeof(client_path_trace_struct));
#      endif
#    endif

  util_save_bmp(data.client_path_trace_data.offset,
                data.client_path_trace_data.stride,
                data.client_path_trace_data.tile_x,
                data.client_path_trace_data.tile_y,
                data.client_path_trace_data.tile_h,
                data.client_path_trace_data.tile_w,
                data.client_path_trace_data.pass_stride,
                num_samples,
                &buffer[0],
                NULL,
                0
                //(char *)omp_mpiData->ptr_map[data.client_path_trace_data.buffer]
  );
#  endif
#endif
}
////////////////////////////////////////////////////////////////////////////////////////////////////

void render(client_kernel_struct &data)
{
  int action = data.client_tag;
  if (action == CLIENT_TAG_CYCLES_const_copy) {
    // printf("const_copy\n");
    const_copy(data);
  }
  else if (action == CLIENT_TAG_CYCLES_tex_copy) {
    // printf("tex_copy\n");
    tex_copy(data);
  }
  else if (action == CLIENT_TAG_CYCLES_path_trace) {

# if 1
// set num samples into env samples
    const char *env_samples = getenv("DEBUG_SAMPLES");
    int num_samples = data.client_path_trace_data.num_samples;
    if (env_samples != NULL) {
      num_samples = atoi(env_samples);
    }
    data.client_path_trace_data.num_samples = num_samples;
# endif

    if (data.client_path_trace_data.pixels != NULL)
      path_trace_pixels(data);
    else
      path_trace_buffer(data);
  }
  else if (action == CLIENT_TAG_CYCLES_alloc_kg) {
    // printf("alloc_kg\n");
    alloc_kg(data);
  }
  else if (action == CLIENT_TAG_CYCLES_free_kg) {
    // printf("free_kg\n");
    free_kg(data);
  }
  else if (action == CLIENT_TAG_CYCLES_mem_alloc) {
    // printf("mem_alloc\n");
    mem_alloc(data);
  }
  else if (action == CLIENT_TAG_CYCLES_mem_copy_to) {
    // printf("mem_copy_to\n");
    mem_copy_to(data);
  }
  else if (action == CLIENT_TAG_CYCLES_mem_zero) {
    // printf("mem_zero\n");
    mem_zero(data);
  }
  else if (action == CLIENT_TAG_CYCLES_mem_free) {
    // printf("mem_free\n");
    mem_free(data);
  }
  else if (action == CLIENT_TAG_CYCLES_tex_free) {
    // printf("tex_free\n");
    tex_free(data);
  }
  else if (action == CLIENT_TAG_CYCLES_load_textures) {
    // printf("load_textures\n");
    load_textures(data);
  }
  else if (action == CLIENT_TAG_CYCLES_tex_info) {
    // printf("tex_info\n");
    tex_info(data);
  }
  else if (action == CLIENT_TAG_CYCLES_build_bvh) {
    // printf("build_bvh\n");
    build_bvh(data);
  }
  else if (action == CLIENT_TAG_CYCLES_frame_info) {
    // printf("frame_info\n");
    frame_info(data);
  }
}

// CCL_NAMESPACE_END
}  // namespace client
}  // namespace cyclesphi
