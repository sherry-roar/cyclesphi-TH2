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

#include "kernel_omp.h"
#include "kernel_camera.h"
#include "kernel_util.h"

#include "device/cpu/device_impl.h"

#include "integrator/path_trace_work_cpu.h"
#include "scene/film.h"
#include "scene/pass.h"
#include "scene/scene.h"
#include "util/color.h"

#include <vector>

#if defined(WITH_CPU_SIMD)
#  define __KERNEL_SSE__
#  define __KERNEL_SSE2__
#  define __KERNEL_SSE3__
#  define __KERNEL_SSSE3__
#  define __KERNEL_SSE41__
#endif

#if defined(WITH_CPU_AVX)
#  define __KERNEL_AVX__
#endif

#if defined(WITH_CPU_AVX2)
#  define __KERNEL_AVX2__
#endif

#if defined(WITH_CLIENT_CUDA_CPU_STAT)
#  define __KERNEL_GPU_CPU__
#endif

//#define  __MIC__
#ifdef __MIC__
#  define __KERNEL_MIC__
#endif

//#include "kernel/device/cpu/kernel.h"
//#define KERNEL_ARCH cpu
//#include "kernel/device/cpu/kernel_arch_impl.h"

#include <omp.h>
#include <sstream>

#ifndef _WIN32
#  include <unistd.h>
#endif

#ifdef CRYPT_COVERAGE
//#include "coverage.h"
#  include <algorithm>
#endif

#ifdef WITH_CLIENT_SHOW_STAT_BVH_LOOP
int ccl::bvh_traversal_max_level = 1;
#endif

#ifdef WITH_CPU_STAT
// using namespace ccl;
size_t omp_chunk_size = 0;
size_t get_omp_chunk_size()
{
  if (omp_chunk_size == 0) {
    const char *CUDA_CHUNK_SIZE_MB = getenv("CUDA_CHUNK_SIZE_MB");
    if (CUDA_CHUNK_SIZE_MB != NULL) {
      omp_chunk_size = atol(CUDA_CHUNK_SIZE_MB) * 1024L * 1024L;
      printf("CUDA_CHUNK_SIZE_MB: %lld\n", omp_chunk_size);
    }
    else {
      omp_chunk_size = 2L * 1024L * 1024L;
    }
  }
  return omp_chunk_size;
}
#  define OMP_CHUNK_SIZE (get_omp_chunk_size())

//#define CRYPT_COVERAGE

int kernel_compat_cpu_enable_stat = 1;
int kernel_cpu_image_enable_stat = 1;

struct OMPMem {
  OMPMem() : data_pointer(0), data_size(0), counter_pointer(0), counter_size(0), counter_sum(NULL)
  {
  }
  std::string name;
  size_t counter_size;
  DEVICE_PTR data_pointer;
  size_t data_size;
  unsigned long long int **counter_pointer;
  size_t *counter_sum;
  // omp_lock_t *omp_locks;
  // size_t data_count;
};

std::map<std::string, OMPMem> omp_stat_map;

void tex_copy_stat(unsigned long long int ***counter,
                   double *counter_mul,
                   unsigned int *counter_size,
                   size_t data_count,
                   size_t mem_size,
                   const char *name,
                   char *mem,
                   // omp_lock_t **omp_locks,
                   int numDevices)
{
  OMPMem *omem = NULL;
  if (name != NULL) {
    omem = &omp_stat_map[std::string(name)];
    omem->name = std::string(name);
  }
  else {
    std::map<std::string, OMPMem>::iterator it_stat;
    for (it_stat = omp_stat_map.begin(); it_stat != omp_stat_map.end(); it_stat++) {
      OMPMem *o = &it_stat->second;
      if (o->data_pointer == (DEVICE_PTR)mem) {
        omem = o;
        mem_size = o->data_size;
        break;
      }
    }
  }

  counter_mul[0] = (double)(((double)mem_size / (double)data_count) / (double)OMP_CHUNK_SIZE);
  counter_size[0] = (unsigned int)ceil(data_count * counter_mul[0]);  // + 1ULL;
  if (counter_size[0] < 1)
    counter_size[0] = 1;

  int threads = numDevices;
  if (threads < 1) {
    printf("omp_tex_copy_stat: threads<1\n");
    exit(-1);
  }
  counter[0] = new unsigned long long int *[threads];
  // omp_locks[0] = new omp_lock_t[threads];

#  pragma omp parallel num_threads(threads)
  {
    int tid = omp_get_thread_num();
    counter[0][tid] = new unsigned long long int[counter_size[0]];
    memset(counter[0][tid], 0, counter_size[0] * sizeof(unsigned long long int));
    // omp_init_lock(&(omp_locks[0][tid]));
  }

  omem->counter_size = counter_size[0];
  omem->data_pointer = (DEVICE_PTR)mem;
  omem->data_size = mem_size;
  omem->counter_pointer = counter[0];
  // omem->omp_locks = omp_locks[0];
  omem->counter_sum = new size_t[omem->counter_size];
  // omem->data_count = data_count;
  memset(omem->counter_sum, 0, omem->counter_size * sizeof(size_t));
}

#  define OMP_TEX_COPY_STAT( \
      counter, counter_mul, counter_size, data_count, mem_size, name, mem, numDevices) \
    omp_tex_copy_stat( \
        counter, counter_mul, counter_size, data_count, mem_size, name, mem, numDevices);

#else

#  define OMP_TEX_COPY_STAT( \
      counter, counter_mul, counter_size, data_count, mem_size, name, mem, numDevices)

#endif

namespace ccl {

void tex_copy_internal(int numDevices,
                       KernelGlobalsCPU *kg,
                       const char *name,
                       void *mem,
                       size_t data_count,
                       size_t mem_size)
{

  // printf("CLIENT tex_copy: %s, %zu, %zu\n", name, data_count, mem_size);
  // fflush(0);

  if (0) {
  }
#define KERNEL_DATA_ARRAY(type, tname) \
  else if (strcmp(name, #tname) == 0) \
  { \
    kg->tname.data = (type *)mem; \
    kg->tname.width = data_count; \
    OMP_TEX_COPY_STAT(&kg->tname.counter, \
                      &kg->tname.counter_mul, \
                      &kg->tname.counter_size, \
                      data_count, \
                      mem_size, \
                      name, \
                      (char *)mem, \
                      numDevices) \
  }
#include "kernel/data_arrays.h"
  else {
    assert(0);
  }
}
}  // namespace ccl

namespace cyclesphi {
namespace kernel {

DevKernelData *dev_kernel_data = NULL;

namespace omp {

#ifdef WITH_CLIENT_SHOW_STAT_BVH_LOOP
int get_bvh_traversal_max_level()
{
  return ccl::bvh_traversal_max_level;
}
#endif

#ifdef WITH_CPU_STAT

size_t omp_get_data_stat(std::string name, int dev, int chunk)
{
  return omp_stat_map[name].counter_pointer[dev][chunk];
}

DEVICE_PTR omp_get_data_mem(std::string name)
{
  return omp_stat_map[name].data_pointer;
}

size_t omp_get_data_size(std::string name)
{
  return omp_stat_map[name].data_size;
}

#endif

#ifdef CRYPT_COVERAGE
//#include "coverage.h"
#  include <algorithm>

class Coverage {
 public:
  Coverage(ccl::KernelGlobalsCPU *kg_,
           char *tile_buffer_,
           int tile_stride_,
           int tile_x_,
           int tile_y_,
           int tile_w_,
           int tile_h_,
           int tile_buffers_pass_stride_)
      : kg(kg_),
        tile_buffer(tile_buffer_),
        tile_stride(tile_stride_),
        tile_x(tile_x_),
        tile_y(tile_y_),
        tile_w(tile_w_),
        tile_h(tile_h_),
        tile_buffers_pass_stride(tile_buffers_pass_stride_)
  {
  }

  void init_path_trace();
  void init_pixel(int x, int y);
  void finalize();

 private:
  vector<CoverageMap> coverage_object;
  vector<CoverageMap> coverage_material;
  vector<CoverageMap> coverage_asset;
  ccl::KernelGlobalsCPU *kg;
  // RenderTile &tile;
  char *tile_buffer;
  int tile_stride, tile_x, tile_y, tile_w, tile_h, tile_buffers_pass_stride;
  void finalize_buffer(vector<CoverageMap> &coverage, const int pass_offset);
  void flatten_buffer(vector<CoverageMap> &coverage, const int pass_offset);
  void sort_buffer(const int pass_offset);
};

static bool crypomatte_comp(const pair<float, float> &i, const pair<float, float> j)
{
  return i.first > j.first;
}

void Coverage::finalize()
{
  int pass_offset = 0;
  if (kernel_data.film.cryptomatte_passes & CRYPT_OBJECT) {
    finalize_buffer(coverage_object, pass_offset);
    pass_offset += kernel_data.film.cryptomatte_depth * 4;
  }
  if (kernel_data.film.cryptomatte_passes & CRYPT_MATERIAL) {
    finalize_buffer(coverage_material, pass_offset);
    pass_offset += kernel_data.film.cryptomatte_depth * 4;
  }
  if (kernel_data.film.cryptomatte_passes & CRYPT_ASSET) {
    finalize_buffer(coverage_asset, pass_offset);
  }
}

void Coverage::init_path_trace()
{
  kg->coverage_object = kg->coverage_material = kg->coverage_asset = NULL;

  if (kernel_data.film.cryptomatte_passes & CRYPT_ACCURATE) {
    if (kernel_data.film.cryptomatte_passes & CRYPT_OBJECT) {
      coverage_object.clear();
      coverage_object.resize(tile_w * tile_h);
    }
    if (kernel_data.film.cryptomatte_passes & CRYPT_MATERIAL) {
      coverage_material.clear();
      coverage_material.resize(tile_w * tile_h);
    }
    if (kernel_data.film.cryptomatte_passes & CRYPT_ASSET) {
      coverage_asset.clear();
      coverage_asset.resize(tile_w * tile_h);
    }
  }
}

void Coverage::init_pixel(int x, int y)
{
  if (kernel_data.film.cryptomatte_passes & CRYPT_ACCURATE) {
    const int pixel_index = tile_w * (y - tile_y) + x - tile_x;
    if (kernel_data.film.cryptomatte_passes & CRYPT_OBJECT) {
      kg->coverage_object = &coverage_object[pixel_index];
    }
    if (kernel_data.film.cryptomatte_passes & CRYPT_MATERIAL) {
      kg->coverage_material = &coverage_material[pixel_index];
    }
    if (kernel_data.film.cryptomatte_passes & CRYPT_ASSET) {
      kg->coverage_asset = &coverage_asset[pixel_index];
    }
  }
}

void Coverage::finalize_buffer(vector<CoverageMap> &coverage, const int pass_offset)
{
  if (kernel_data.film.cryptomatte_passes & CRYPT_ACCURATE) {
    flatten_buffer(coverage, pass_offset);
  }
  else {
    sort_buffer(pass_offset);
  }
}

void Coverage::flatten_buffer(vector<CoverageMap> &coverage, const int pass_offset)
{
  /* Sort the coverage map and write it to the output */
  int pixel_index = 0;
  int pass_stride = tile_buffers_pass_stride;
  for (int y = 0; y < tile_h; ++y) {
    for (int x = 0; x < tile_w; ++x) {
      const CoverageMap &pixel = coverage[pixel_index];
      if (!pixel.empty()) {
        /* buffer offset */
        int index = x + y * tile_stride;
        float *buffer = (float *)tile_buffer + index * pass_stride;

        /* sort the cryptomatte pixel */
        vector<pair<float, float>> sorted_pixel;
        for (CoverageMap::const_iterator it = pixel.begin(); it != pixel.end(); ++it) {
          sorted_pixel.push_back(std::make_pair(it->second, it->first));
        }
        std::sort(sorted_pixel.begin(), sorted_pixel.end(), crypomatte_comp);
        int num_slots = 2 * (kernel_data.film.cryptomatte_depth);
        if (sorted_pixel.size() > num_slots) {
          float leftover = 0.0f;
          for (vector<pair<float, float>>::iterator it = sorted_pixel.begin() + num_slots;
               it != sorted_pixel.end();
               ++it) {
            leftover += it->first;
          }
          sorted_pixel[num_slots - 1].first += leftover;
        }
        int limit = min(num_slots, sorted_pixel.size());
        for (int i = 0; i < limit; ++i) {
          kernel_write_id_slots(buffer + kernel_data.film.pass_cryptomatte + pass_offset,
                                2 * (kernel_data.film.cryptomatte_depth),
                                sorted_pixel[i].second,
                                sorted_pixel[i].first);
        }
      }
      ++pixel_index;
    }
  }
}

void Coverage::sort_buffer(const int pass_offset)
{
  /* Sort the coverage map and write it to the output */
  int pass_stride = tile_buffers_pass_stride;
  for (int y = 0; y < tile_h; ++y) {
    for (int x = 0; x < tile_w; ++x) {
      /* buffer offset */
      int index = x + y * tile_stride;
      float *buffer = (float *)tile_buffer + index * pass_stride;
      kernel_sort_id_slots(buffer + kernel_data.film.pass_cryptomatte + pass_offset,
                           2 * (kernel_data.film.cryptomatte_depth));
    }
  }
}

#endif

//#if defined(WITH_CLIENT_RENDERENGINE) || defined(WITH_CLIENT_ULTRAGRID) || \
//    defined(WITH_CLIENT_RENDERENGINE_EMULATE) || defined(WITH_CLIENT_MPI) || \
//    defined(WITH_CLIENT_MPI_SOCKET) || defined(WITH_CLIENT_SOCKET)
#  define OMP_INTERACTIVE
//#endif

// CCL_NAMESPACE_BEGIN

// OmpMpiData *omp_mpiData = NULL;
// DevKernelData *dev_kernel_data = NULL;
ccl::PathTraceWorkCPU *g_pathTraceWorkCPU = NULL;
ccl::Film g_film;
ccl::DeviceScene g_device_scene;
ccl::CPUDevice g_cpuDevice;

bool g_cancel_requested_flag = false;
int g_used_buffer = 0;

int g_start_sample = 0;
int g_num_samples = 1;


std::vector<client_buffer_passes> g_buffer_passes;
DEVICE_PTR g_buffer_passes_d = NULL;

// CPUKernels kernels;

void set_device(int device, int world_rank, int world_size)
{
}

/* Memory Copy */
void const_copy_internal(DEVICE_PTR kg_bin, char *host_bin, size_t size)
{
  ccl::KernelGlobalsCPU *kg = (ccl::KernelGlobalsCPU *)kg_bin;
  if (sizeof(kg->data) != size) {
    printf("sizeof(KernelData) != size");
    exit(-1);
  }
  memcpy(&kg->data, host_bin, size);
}

void const_copy(
    int numDevice, DEVICE_PTR kg_bin, const char *name, char *host_bin, size_t size, bool save)
{
  if (strcmp(name, "data") == 0) {
    const_copy_internal(kg_bin, host_bin, size);
  }
}

DEVICE_PTR get_data(DEVICE_PTR kg_bin)
{
  ccl::KernelGlobalsCPU *kg = (ccl::KernelGlobalsCPU *)kg_bin;
  return (DEVICE_PTR)&kg->data;
}

size_t get_size_data(DEVICE_PTR kg_bin)
{
  ccl::KernelGlobalsCPU *kg = (ccl::KernelGlobalsCPU *)kg_bin;
  return sizeof(kg->data);
}

float *get_camera_matrix(DEVICE_PTR kg_bin)
{
  ccl::KernelGlobalsCPU *kg = (ccl::KernelGlobalsCPU *)kg_bin;
  return (float *)&kg->data.cam.cameratoworld;
}

void anim_step(int numDevice, DEVICE_PTR kg_bin, char *data_bin, int s)
{
  const char *env_vx = getenv("DEBUG_ANIMATION_VX");
  float vx = 0.0f;
  if (env_vx != NULL) {
    vx = atof(env_vx);
  }

  const char *env_vy = getenv("DEBUG_ANIMATION_VY");
  float vy = 0.0f;
  if (env_vy != NULL) {
    vy = atof(env_vy);
  }

  const char *env_vz = getenv("DEBUG_ANIMATION_VZ");
  float vz = 0.0f;
  if (env_vz != NULL) {
    vz = atof(env_vz);
  }

  ccl::KernelGlobalsCPU *kg = (ccl::KernelGlobalsCPU *)kg_bin;
  kg->data.cam.cameratoworld.x.w += vx;
  kg->data.cam.cameratoworld.y.w += vy;
  kg->data.cam.cameratoworld.z.w += vz;

  // data_bin.resize(sizeof(kg->data));
  memcpy(data_bin, &kg->data, sizeof(kg->data));
}

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
                 int use_lamp_mis)
{
  ccl::KernelGlobalsCPU *kg = (ccl::KernelGlobalsCPU *)kg_bin;

  kg->data.integrator.min_bounce = min_bounce;
  kg->data.integrator.max_bounce = max_bounce;
  kg->data.integrator.max_diffuse_bounce = max_diffuse_bounce;
  kg->data.integrator.max_glossy_bounce = max_glossy_bounce;
  kg->data.integrator.max_transmission_bounce = max_transmission_bounce;
  kg->data.integrator.max_volume_bounce = max_volume_bounce;
  kg->data.integrator.max_volume_bounds_bounce = max_volume_bounds_bounce;
  kg->data.integrator.transparent_min_bounce = transparent_min_bounce;
  kg->data.integrator.transparent_max_bounce = transparent_max_bounce;
  kg->data.integrator.use_lamp_mis = use_lamp_mis;

  // apply data
  memcpy(data_bin, &kg->data, sizeof(kg->data));
}

void socket_step(int numDevice, DEVICE_PTR kg_bin, char *data_bin, char *cdata)
{
  ccl::KernelGlobalsCPU *kg = (ccl::KernelGlobalsCPU *)kg_bin;

#if defined(WITH_CLIENT_RENDERENGINE) || defined(WITH_CLIENT_ULTRAGRID)
  view_to_kernel_camera((char *)&kg->data, (cyclesphi_data *)cdata);
#else
  // wh,h,dx,dy
  int *size = (int *)cdata;

  float *height = &kg->data.cam.height;
  float *width = &kg->data.cam.width;
  float *dx = &kg->data.cam.dx[0];
  float *dy = &kg->data.cam.dy[0];
  float *rastertocamera = &kg->data.cam.rastertocamera.x[0];

  float old_new_height = height[0] / size[1];
  float old_new_width = width[0] / size[0];

  height[0] = size[1];
  width[0] = size[0];
  dx[0] *= old_new_width;
  dx[1] *= old_new_width;
  dx[2] *= old_new_width;
  // dx[3] *= old_new_width;

  dy[0] *= old_new_width;
  dy[1] *= old_new_width;
  dy[2] *= old_new_width;
  // dy[3] *= old_new_width;

  rastertocamera[0] *= old_new_width;
  rastertocamera[1] *= old_new_width;
  rastertocamera[2] *= old_new_width;
  // rastertocamera[3] *= old_new_width;

  rastertocamera[4] *= old_new_width;
  rastertocamera[5] *= old_new_width;
  rastertocamera[6] *= old_new_width;
  // rastertocamera[7] *= old_new_width;

  rastertocamera[8] *= old_new_width;
  rastertocamera[9] *= old_new_width;
  rastertocamera[10] *= old_new_width;
  // rastertocamera[11] *= old_new_width;
#endif

  // apply data
  memcpy(data_bin, &kg->data, sizeof(kg->data));
}

void tex_copy(int numDevice,
              DEVICE_PTR kg_bin,
              char *name_bin,
              DEVICE_PTR dmem,
              char *mem,
              size_t data_count,
              size_t mem_size)
{
  //    if (name_bin == NULL || mem == NULL)
  //        return;

  size_t nameSize = sizeof(char) * (strlen(name_bin) + 1);
  char *name = (char *)name_bin;

  if (mem != NULL)
    memcpy((char *)dmem, mem, mem_size);

  tex_copy_internal(
      numDevice, (ccl::KernelGlobalsCPU *)kg_bin, name, (char *)dmem, data_count, mem_size);
}

#ifdef OMP_INTERACTIVE

#  if defined(WITH_X264)
float4 omp_film_map(ccl::KernelGlobalsCPU *kg, float4 irradiance, float scale)
{
  float exposure = kernel_data.film.exposure;
  float4 result = irradiance * scale;

  /* conversion to srgb */
  result.x = color_linear_to_srgb(result.x * exposure);
  result.y = color_linear_to_srgb(result.y * exposure);
  result.z = color_linear_to_srgb(result.z * exposure);

  /* clamp since alpha might be > 1.0 due to russian roulette */
  result.w = saturate(result.w);

  return result;
}

uchar4 omp_film_float_to_byte(float4 color)
{
  uchar4 result;

  /* simple float to byte conversion */
  result.x = (uchar)(saturate(color.x) * 255.0f);
  result.y = (uchar)(saturate(color.y) * 255.0f);
  result.z = (uchar)(saturate(color.z) * 255.0f);
  result.w = (uchar)(saturate(color.w) * 255.0f);

  return result;
}

// i420
// void rgb_to_yuv(uchar r, uchar g, uchar b, uchar *ly, uchar *lu, uchar *lv)
//{
//  uchar y, u, v;
//
//  y = ((66 * r + 129 * g + 25 * b) >> 8) + 16;
//  u = ((-38 * r + -74 * g + 112 * b) >> 8) + 128;
//  v = ((112 * r + -94 * g + -18 * b) >> 8) + 128;
//
//  *ly = y;
//  *lu = u;
//  *lv = v;
//}

// void Bitmap2Yuv420p_calc2(uint8_t *destination, uint8_t *rgb, size_t width, size_t height)
//{
//  size_t image_size = width * height;
//  size_t upos = image_size;
//  size_t vpos = upos + upos / 4;
//  size_t i = 0;
//
//  for (size_t line = 0; line < height; ++line) {
//    if (!(line % 2)) {
//      for (size_t x = 0; x < width; x += 2) {
//        uint8_t r = rgb[3 * i];
//        uint8_t g = rgb[3 * i + 1];
//        uint8_t b = rgb[3 * i + 2];
//
//        destination[i++] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;
//
//        destination[upos++] = ((-38 * r + -74 * g + 112 * b) >> 8) + 128;
//        destination[vpos++] = ((112 * r + -94 * g + -18 * b) >> 8) + 128;
//
//        r = rgb[3 * i];
//        g = rgb[3 * i + 1];
//        b = rgb[3 * i + 2];
//
//        destination[i++] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;
//      }
//    }
//    else {
//      for (size_t x = 0; x < width; x += 1) {
//        uint8_t r = rgb[3 * i];
//        uint8_t g = rgb[3 * i + 1];
//        uint8_t b = rgb[3 * i + 2];
//
//        destination[i++] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;
//      }
//    }
//  }
//}

void rgb_to_yuv_i420(
    uchar *destination, uchar r, uchar g, uchar b, int x, int y, int tile_h, int tile_w)
{
  size_t image_size = tile_w * tile_h;
  uchar *dst_y = destination;
  uchar *dst_u = destination + image_size;
  uchar *dst_v = destination + image_size + image_size / 4;

  // Y
  int index_y = x + y * tile_w;
  dst_y[index_y] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;

  // U
  if (x % 2 == 0 && y % 2 == 0) {
    int index_u = (x / 2) + (y / 2) * (tile_w / 2);
    dst_u[index_u] = ((-38 * r + -74 * g + 112 * b) >> 8) + 128;
  }

  // V
  if (x % 2 == 0 && y % 2 == 0) {
    int index_v = (x / 2) + (y / 2) * (tile_w / 2);
    dst_v[index_v] = ((112 * r + -94 * g + -18 * b) >> 8) + 128;
  }
}

void kernel_film_convert_to_byte(ccl::KernelGlobalsCPU *kg,
                                 uchar *yuv,
                                 float *buffer,
                                 float sample_scale,
                                 int x,
                                 int y,
                                 int offset,
                                 int stride,
                                 int tile_x,
                                 int tile_y,
                                 int tile_h,
                                 int tile_w)
{
  /* buffer offset */
  int index = offset + (x + tile_x) + (y + tile_y) * stride;

  buffer += index * kernel_data.film.pass_stride;

  /* map colors */
  float4 irradiance = *((float4 *)buffer);
  float4 float_result = film_map(kg, irradiance, sample_scale);
  uchar4 byte_result = film_float_to_byte(float_result);

  // bgra[index + 0 * tile_h * tile_w] = byte_result.z;  // b
  // bgra[index + 1 * tile_h * tile_w] = byte_result.y;  // g
  // bgra[index + 2 * tile_h * tile_w] = byte_result.x;  // r
  // bgra[index + 3 * tile_h * tile_w] = byte_result.w;  // a

  // bgra[index].x = byte_result.z;  // b
  // bgra[index].y = byte_result.y;  // g
  // bgra[index].z = byte_result.x;  // r
  // bgra[index].w = byte_result.w;  // a

  // omp_rgb_to_yuv(byte_result.x,
  //               byte_result.y,
  //               byte_result.z,
  //               &yuv[index + 0 * tile_h * tile_w],
  //               &yuv[index + 1 * tile_h * tile_w],
  //               &yuv[index + 2 * tile_h * tile_w]);

  rgb_to_yuv_i420(yuv, byte_result.x, byte_result.y, byte_result.z, x, y, tile_h, tile_w);
}
#  endif

#  if 0

void path_trace_internal(DEVICE_PTR kg_bin,
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
                             int nprocs_cpu)
{
  int size = tile_h * tile_w;

  int *sample_finished = (int *)sample_finished_omp;
  int *reqFinished = (int *)reqFinished_omp;

  *sample_finished = start_sample;

#    pragma omp parallel num_threads(nprocs_cpu)
  {
    int tid = omp_get_thread_num();
    double progress = omp_get_wtime();
    double t = omp_get_wtime();

    ccl::KernelGlobalsCPU kg = *((ccl::KernelGlobalsCPU *)kg_bin);

#    pragma omp for schedule(dynamic, 16)
    for (int i = 0; i < size; i++) {
      if (*reqFinished != 0)
        continue;

      int y = i / tile_w;
      int x = i - y * tile_w;

      for (int sample = start_sample; sample < end_sample; sample++) {
        kernel_path_trace(
            &kg, (float *)buffer_bin, sample, x + tile_x, y + tile_y, offset, stride);
      }

      if (pixel_bin != NULL) {
#    if defined(WITH_X264)
        omp_kernel_film_convert_to_byte(&kg,
                                        (uchar *)pixel_bin,
                                        (float *)buffer_bin,
                                        1.0f / end_sample,
                                        x,
                                        y,
                                        offset,
                                        stride,
                                        tile_x,
                                        tile_y,
                                        tile_h,
                                        tile_w);
#    else
//#if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
//		kernel_film_convert_to_half_float(&kg,
//                                    (ccl::uchar4 *)pixel_bin,
//                                    (float *)buffer_bin,
//                                    1.0f / end_sample,
//                                    x + tile_x,
//                                    y + tile_y,
//                                    offset,
//                                    stride);
//#else

        kernel_film_convert_to_byte(&kg,
                                    (ccl::uchar4 *)pixel_bin,
                                    (float *)buffer_bin,
                                    1.0f / end_sample,
                                    x + tile_x,
                                    y + tile_y,
                                    offset,
                                    stride);
//#  endif
#    endif
      }
    }

#    pragma omp critical
    {
#    pragma omp flush
    }
  }

  *sample_finished = end_sample;
}

#  endif

void init_execution(int has_shadow_catcher_,
                        int max_shaders_,
                        int pass_stride_,
                        unsigned int kernel_features_,
                        unsigned int volume_stack_size_,
                        bool init)
{
  if (g_pathTraceWorkCPU == NULL) {
    //////////////
    const int max_srgb = 1001;
    float to_srgb[max_srgb];
    for (int i = 0; i < max_srgb; i++) {
      to_srgb[i] = ccl::color_linear_to_srgb((float)i / (float)(max_srgb - 1));
    }
    //////////////

    g_pathTraceWorkCPU = new ccl::PathTraceWorkCPU(
        &g_cpuDevice, &g_film, &g_device_scene, &g_cancel_requested_flag);
  }

  g_device_scene.data.integrator.has_shadow_catcher = has_shadow_catcher_;
  g_device_scene.data.max_shaders = max_shaders_;
  g_device_scene.data.film.pass_stride = pass_stride_;
  g_device_scene.data.film.exposure = 1.0f;
  g_device_scene.data.kernel_features = kernel_features_;
  g_device_scene.data.volume_stack_size = volume_stack_size_;

  if (init) {
    g_pathTraceWorkCPU->alloc_work_memory();
    g_pathTraceWorkCPU->init_execution();
  }
}

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
                     char *signal_value)
{
}

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
                       char *signal_value)
{
}

float g_report_time = 0;
size_t g_report_time_count = 0;
int g_path_stat_is_done = 0;

void path_trace_internal(DEVICE_PTR kg_bin,
                         char *map_buffer_bin,
                         char *map_pixel_bin,
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
                         char *signal_value)
{
  //int start_sample2 = start_sample;
  //int end_sample2 = end_sample;

  size_t pix_type_size = SIZE_UCHAR4;
  size_t pix_size = w * h * pix_type_size;
  int *pix_state = (int *)signal_value;

  //omp_lock_t *lock0 = NULL;
  //omp_lock_t *lock1 = NULL;

  //if (pix_state != NULL) {
  //  lock0 = (omp_lock_t *)&pix_state[6];
  //  lock1 = (omp_lock_t *)&pix_state[8];
  //}

#  ifdef ENABLE_STEP_SAMPLES
  int debug_step_samples = 1;
  const char *env_step_samples = getenv("DEBUG_STEP_SAMPLES");
  if (env_step_samples != NULL)
    debug_step_samples = atoi(env_step_samples);
#  endif

//#  if defined(WITH_CLIENT_RENDERENGINE_EMULATE)
//  const char *drt = getenv("DEBUG_REPEAT_TIME");
//  double drt_time_start = omp_get_wtime();
//#  endif

  //////////////////////
  char *pixels_node1 = NULL;
  char *pixels_node2 = NULL;

  if (map_pixel_bin != NULL) {
    pixels_node1 = (char *)map_pixel_bin;

    if (pix_state != NULL)
      pixels_node2 = (char *)map_pixel_bin + pix_size;
    else
      pixels_node2 = pixels_node1;
  }
  //////////////////////
  char *pixels_node = pixels_node1;
  

  //while (true) {
    if (pix_state != NULL && pix_state[4] == 1) {
      return;
    }
    else if (g_used_buffer == 0) {
      //if (lock0 != NULL)
      //  omp_set_lock(lock0);

      pixels_node = pixels_node1;
    }
    else if (g_used_buffer == 1) {
      //if (lock1 != NULL)
      //  omp_set_lock(lock1);

      pixels_node = pixels_node2;
    }
    else {
      //usleep(100);
      //continue;
      return;
    }

    if (pix_state != NULL && pix_state[2] == 1) {
    // if (1) {
      pix_state[2] = 0;
      //#  pragma omp flush
      g_start_sample = start_sample;
      g_num_samples = end_sample - start_sample;
      //end_sample = end_sample2 - start_sample2;

       size_t sizeBuf_node = w * h * g_cpuDevice.kernel_globals.data.film.pass_stride *
                            sizeof(float);
      memset((char *)map_buffer_bin, 0, sizeBuf_node);
    }

#  ifdef ENABLE_STEP_SAMPLES
    if (debug_step_samples > 0) {
      start_sample = start_sample2 + g_path_stat_is_done * debug_step_samples;
        end_sample = start_sample2 + g_path_stat_is_done + 1) * debug_step_samples;
    }
#  endif

    double t_id = omp_get_wtime();
    ///

    g_num_samples = end_sample - start_sample;

    ccl::KernelWorkTile wtile;
    wtile.buffer = (float *)map_buffer_bin;
    wtile.pixel = (char *)pixels_node;
    wtile.start_sample = g_start_sample;
    wtile.num_samples = g_num_samples;
    // wtile.num_samples = 10;
    wtile.x = tile_x;
    wtile.w = tile_w;
    wtile.y = tile_y;
    wtile.h = tile_h;
    wtile.offset = offset;
    wtile.stride = stride;

    //if (pix_state != NULL) {
    g_pathTraceWorkCPU->init_execution();
    //}

    g_pathTraceWorkCPU->set_buffer(wtile, g_buffer_passes);

    ccl::PathTraceWork::RenderStatistics rs;
    g_pathTraceWorkCPU->render_samples(rs, wtile.start_sample, wtile.num_samples, 0, true, NULL);

    if (wtile.pixel != NULL) {
      // ccl::PassAccessor::Destination destination;
      // destination.pixels_uchar_rgba = (ccl::uchar4 *)wtile.pixel;
      // destination.num_components = 4;
      // destination.offset = wtile.x + wtile.y * stride;
      // g_pathTraceWorkCPU->get_render_tile_film_pixels(
      //     destination, ccl::PassMode::DENOISED, wtile.start_sample + wtile.num_samples);
      ccl::PassAccessor::Destination destination;
      //destination.pixels_uchar_rgba = (ccl::uchar4 *)wtile.pixel;
      destination.pixels_half_rgba = (ccl::half4 *)wtile.pixel;
      destination.offset = wtile.offset + wtile.x + wtile.y * wtile.stride;
      destination.stride = wtile.stride;
      destination.num_components = 4;
      //destination.pixel_stride = g_pathTraceWorkCPU->effective_buffer_params_.pass_stride;
      g_pathTraceWorkCPU->get_render_tile_film_pixels(
          destination, ccl::PassMode::DENOISED, wtile.start_sample + wtile.num_samples);
    }

    ///
    g_report_time += (omp_get_wtime() - t_id);
    g_report_time_count++;

    if (g_path_stat_is_done == 0 || g_report_time >= 1.0) {
      printf(
          "Rendering: FPS: %.2f, MaxTime = %f [ms], AvgTime = %f [ms], Samples: %d, Tot "
          "Samples: %d\n",
          1.0f / (omp_get_wtime() - t_id),
          (omp_get_wtime() - t_id),
          (omp_get_wtime() - t_id),
          end_sample - start_sample,
          end_sample);

      fflush(0);

      g_report_time = 0;
      g_report_time_count = 0;
    }

#  if defined(WITH_CLIENT_RENDERENGINE_EMULATE)
    if (pix_state != NULL && g_path_stat_is_done > 0)
      pix_state[5] = 1;
#  endif

    ///
    if (pix_state != NULL) {
      pix_state[3] = end_sample;
    }

#  if defined(ENABLE_INC_SAMPLES) && !defined(ENABLE_STEP_SAMPLES)
    //int num_samples = end_sample - start_sample;
    g_start_sample = g_start_sample + g_num_samples;
    //end_sample = start_sample + num_samples;
#  endif

    if (pix_state == NULL) {
      return;
    }
    else {
      ///////////////////////SAMPLES////////////////////////
#  ifndef ENABLE_INC_SAMPLES
      size_t sizeBuf_node = w * h * g_cpuDevice.kernel_globals.data.film.pass_stride *
                            sizeof(float);
      memset((char *)map_buffer_bin, 0, sizeBuf_node);
#  endif
    }
    ///
    if (g_used_buffer == 0) {
      //if (lock0 != NULL)
      //  omp_unset_lock(lock0);
    }
    else if (g_used_buffer == 1) {
      //if (lock1 != NULL)
      //  omp_unset_lock(lock1);
    }

    g_used_buffer++;
    if (g_used_buffer > 1)
      g_used_buffer = 0;

    g_path_stat_is_done++;

#  if defined(ENABLE_STEP_SAMPLES)
    if (end_sample >= end_sample2)
      break;
#  endif

#  ifdef WITH_CLIENT_SHOW_STAT_BVH_LOOP
    break;
#  endif

#  if defined(WITH_CLIENT_RENDERENGINE_EMULATE_ONE_THREAD)
    // const char *drt = getenv("DEBUG_REPEAT_TIME");
    if (drt == NULL || omp_get_wtime() - drt_time_start > atof(drt)) {
      pix_state[4] = 1;
    }
#  endif

//#  if defined(WITH_CLIENT_RENDERENGINE_EMULATE) && \
//      !defined(WITH_CLIENT_RENDERENGINE_EMULATE_ONE_THREAD)
//    if (drt == NULL)
//      break;
//#  endif
//  }
}

#  define STREAM_PATH1 0
#  define STREAM_PATH2 1
#  define STREAM_PATH1_MEMCPY 2
#  define STREAM_PATH2_MEMCPY 3
#  define STREAM_COUNT 4

#  define EVENT_COUNT (2 * STREAM_COUNT)

struct OMPDevice {
  // int cuDevice;
  //// CUcontext cuContext;
  // cudaStream_t stream[STREAM_COUNT];
  // cudaEvent_t event[EVENT_COUNT];

  // CUmodule cuModule /*, cuFilterModule*/;
  // CUmodule cuModuleStat;

  // CUfunction cuPathTrace;
  // CUfunction cuPathTraceStat;

  // std::map<DEVICE_PTR, CUDAMem> cuda_mem_map;
  // std::map<std::string, CUDAMem> cuda_stat_map;
  DEVICE_PTR texture_info_dmem;
  size_t texture_info_dmem_size;
  DEVICE_PTR d_work_tiles;
  // cudaEvent_t event_start, event_stop;
  float running_time[STREAM_COUNT];
  // float time_old;
  size_t time_count;
  ccl::KernelWorkTile wtile;
  int num_threads_per_block;
  int path_stat_is_done;

  OMPDevice()
  {
    texture_info_dmem = NULL;
    texture_info_dmem_size = 0;
    d_work_tiles = NULL;
    time_count = 0;
    num_threads_per_block = 0;
    path_stat_is_done = 0;
  }
};

void path_trace_internal_gpu(int numDevice,
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
                             char *signal_value)
{

#  if 1
  printf("Rendering %d-%d: %d-%d: %d-%d:, %d-%d: thr: %d, 0 %%\n",
         start_sample,
         end_sample,
         tile_x,
         tile_y,
         tile_w,
         tile_h,
         w,
         h,
         nprocs_cpu);
#  endif

  int start_sample2 = start_sample;
  int end_sample2 = end_sample;

  float g_report_time = 0;
  size_t g_report_time_count = 0;

  int devices_size = nprocs_cpu;

  size_t pix_size = w * h * SIZE_UCHAR4;

  char *pixels_node1 = (char *)map_pixel_bin;
  char *pixels_node2 = (char *)map_pixel_bin + pix_size;
  int *pix_state = (int *)signal_value;

#  if defined(WITH_CLIENT_RENDERENGINE_VR) || \
      (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
  pixels_node2 = (char *)pixels_node2 + w * h * SIZE_UCHAR4;
  int devices_left_eye = devices_size / 2;
  if (devices_left_eye < 1)
    devices_left_eye = 1;

  // devices_left_eye = 0;

  int devices_right_eye = devices_size - devices_left_eye;
  if (devices_right_eye < 1)
    devices_right_eye = 1;
#  endif

  std::vector<double> g_image_time(h);

  std::vector<OMPDevice> omp_devices(devices_size);

  DEVICE_PTR dev_pixels_node1 = mem_alloc(CLIENT_DEVICE_ID, "dev_pixels_node1", NULL, pix_size);

  DEVICE_PTR dev_pixels_node2 = mem_alloc(CLIENT_DEVICE_ID, "dev_pixels_node2", NULL, pix_size);

#  pragma omp parallel num_threads(devices_size)
  {
    int id = omp_get_thread_num();

    omp_devices[id].wtile.start_sample = start_sample;
    omp_devices[id].wtile.num_samples = end_sample - start_sample;

#  if defined(WITH_CLIENT_RENDERENGINE_VR) || \
      (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
    int devices_size_vr = (id < devices_left_eye) ? devices_left_eye : devices_right_eye;
    int id_vr = (id < devices_left_eye) ? id : id - devices_left_eye;
#  endif

    if (omp_devices[id].d_work_tiles == 0) {

      omp_devices[id].d_work_tiles = (DEVICE_PTR)&omp_devices[id].wtile;

      omp_devices[id].wtile.x = tile_x;
      omp_devices[id].wtile.w = tile_w;

      omp_devices[id].wtile.offset = offset;
      omp_devices[id].wtile.stride = stride;
      omp_devices[id].wtile.buffer = (float *)map_buffer_bin;

      // cuda_assert(cudaMemcpy(
      //    scope.get().d_work_tiles, (char *)&wtile, sizeof(WorkTile), cudaMemcpyHostToDevice));

#  if defined(WITH_CLIENT_RENDERENGINE_VR) || \
      (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
      int tile_step_dev = (int)((float)tile_h / (float)devices_size_vr);
      int tile_last_dev = tile_h - (devices_size_vr - 1) * tile_step_dev;

      int tile_y_dev = tile_y + tile_step_dev * id_vr;
      int tile_h_dev = (devices_size_vr - 1 == id_vr) ? tile_last_dev : tile_step_dev;

      omp_devices[id].wtile.y = tile_y_dev;
      omp_devices[id].wtile.h = tile_h_dev;

      const char *env_tiles = getenv("CLIENT_TILES");
      if (env_tiles) {
        omp_devices[id].wtile.y = util_get_int_from_env_array(env_tiles, 2 * id_vr + 0);
        omp_devices[id].wtile.h = util_get_int_from_env_array(env_tiles, 2 * id_vr + 1);
      }
#  else

      int tile_step_dev = (int)((float)tile_h / (float)devices_size);
      int tile_last_dev = tile_h - (devices_size - 1) * tile_step_dev;

      int tile_y_dev = tile_y + tile_step_dev * id;
      int tile_h_dev = (devices_size - 1 == id) ? tile_last_dev : tile_step_dev;

      const char *env_tiles = getenv("CLIENT_TILES");
      if (env_tiles) {
        omp_devices[id].wtile.y = util_get_int_from_env_array(env_tiles, 2 * id + 0);
        omp_devices[id].wtile.h = util_get_int_from_env_array(env_tiles, 2 * id + 1);
      }
      else {
        omp_devices[id].wtile.y = tile_y_dev;
        omp_devices[id].wtile.h = tile_h_dev;
      }
#  endif

      // scope.get().num_threads_per_block = 64;
    }
  }

  while (true) {

    double t_id = omp_get_wtime();

    ccl::KernelGlobalsCPU *_kg = (ccl::KernelGlobalsCPU *)map_kg_bin;

#  pragma omp parallel num_threads(devices_size)  // private(kg)
    {
      int id = omp_get_thread_num();

      ccl::KernelGlobalsCPU kg = *((ccl::KernelGlobalsCPU *)map_kg_bin);

      DEVICE_PTR dev_pixels_node = (DEVICE_PTR)dev_pixels_node2;
      char *pixels_node = NULL;

      float *time_path = NULL;
      float *time_memcpy = NULL;
      int stream_memcpy_id = 0;

      // cudaHostFn_t host_fn_memcpy_status_flag;

      bool render_run = true;

      if (pix_state[0] == 0) {
        dev_pixels_node = (DEVICE_PTR)dev_pixels_node2;
        pixels_node = pixels_node2;

        time_path = &omp_devices[id].running_time[STREAM_PATH2];
        time_memcpy = &omp_devices[id].running_time[STREAM_PATH2_MEMCPY];
      }
      else if (pix_state[1] == 0) {
        dev_pixels_node = (DEVICE_PTR)dev_pixels_node1;
        pixels_node = pixels_node1;

        time_path = &omp_devices[id].running_time[STREAM_PATH1];
        time_memcpy = &omp_devices[id].running_time[STREAM_PATH1_MEMCPY];
      }
      else {
        // continue;
        render_run = false;
      }
      ///////////////////////

      ///////////////////GPU SYNC////////////////////////////
      if (render_run) {
        double event_path_start = omp_get_wtime();

        // kernel_tex_alloc(&kg, (ccl::KernelGlobalsCPU *)kg_bin);
#  if 1
        for (int y = omp_devices[id].wtile.y;
             y < omp_devices[id].wtile.h + omp_devices[id].wtile.y;
             y++) {
          double y_start = omp_get_wtime();
          for (int x = omp_devices[id].wtile.x;
               x < omp_devices[id].wtile.w + omp_devices[id].wtile.x;
               x++) {
            // for (int sample = start_sample; sample < end_sample; sample++) {
            //   kernel_omp_path_trace(&kg,
            //                         omp_devices[id].wtile.buffer,
            //                         sample,
            //                         x,
            //                         y,
            //                         omp_devices[id].wtile.offset,
            //                         omp_devices[id].wtile.stride);
            // }

            ccl::KernelWorkTile work_tile;
            work_tile.x = x;
            work_tile.y = y;
            work_tile.w = 1;
            work_tile.h = 1;
            work_tile.start_sample = start_sample;
            work_tile.num_samples = 1;
            work_tile.offset = omp_devices[id].wtile.offset;
            work_tile.stride = omp_devices[id].wtile.stride;
#    if 0
            render_samples_full_pipeline(
                &kg, work_tile, end_sample - start_sample, (char*)omp_devices[id].wtile.buffer);
#    endif

            if (dev_pixels_node != NULL) {
//#if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
//		kernel_film_convert_to_half_float(&kg,
//                                          (ccl::uchar4 *)dev_pixels_node,
//                                          omp_devices[id].wtile.buffer,
//                                          1.0f / end_sample,
//                                          x,
//                                          y,
//                                          omp_devices[id].wtile.offset,
//                                          omp_devices[id].wtile.stride);
//#else
#    if 0
		kernel_film_convert_to_byte(&kg,
                                          (ccl::uchar4 *)dev_pixels_node,
                                          omp_devices[id].wtile.buffer,
                                          1.0f / end_sample,
                                          x,
                                          y,
                                          omp_devices[id].wtile.offset,
                                          omp_devices[id].wtile.stride);
#    endif
              //#endif
            }
          }
          double y_stop = omp_get_wtime();
          g_image_time[y] = y_stop - y_start;
        }
#  endif

        double event_path_stop = omp_get_wtime();
        unsigned int total_work_size = omp_devices[id].wtile.w * omp_devices[id].wtile.h;

        double event_memcpy_start = omp_get_wtime();

        // transfer to CPU
        size_t pix_offset = (omp_devices[id].wtile.x + omp_devices[id].wtile.y * stride) *
                            SIZE_UCHAR4;

        memcpy((char *)pixels_node + pix_offset,
               (char *)dev_pixels_node + pix_offset,
               total_work_size * SIZE_UCHAR4);

        double event_memcpy_stop = omp_get_wtime();

        // cuda_assert(cudaEventElapsedTime(time_path, event_path_start, event_path_stop));
        time_path[0] = (event_path_stop - event_path_start) * 1000.0f;
        // cuda_assert(cudaEventElapsedTime(time_memcpy, event_memcpy_start, event_memcpy_stop));
        time_memcpy[0] = (event_memcpy_stop - event_memcpy_start) * 1000.0f;
      }
    }

////////////////////////////////
#  ifdef WITH_CPU_STAT
    //#    pragma omp barrier

    // if (id == 0)
    {
      std::map<std::string, OMPMem>::iterator it_stat;
      for (it_stat = omp_stat_map.begin(); it_stat != omp_stat_map.end(); it_stat++) {
        OMPMem *om = &it_stat->second;
        memset(om->counter_sum, 0, om->counter_size * sizeof(size_t));
      }
    }

#    ifdef WITH_CLIENT_SHOW_STAT
    const char *show_stat_name = getenv("SHOW_STAT_NAME");
#    endif

    std::map<std::string, OMPMem>::iterator it_stat;
    for (it_stat = omp_stat_map.begin(); it_stat != omp_stat_map.end(); it_stat++) {
      OMPMem *om = &it_stat->second;
      for (int i = 0; i < om->counter_size; i++) {

#    ifdef WITH_CLIENT_SHOW_STAT
        int id_max = 0;
        size_t value_max = 0;
#    endif

        for (int id1 = 0; id1 < devices_size; id1++) {
          om->counter_sum[i] += om->counter_pointer[id1][i];

#    ifdef WITH_CLIENT_SHOW_STAT
          if (value_max < om->counter_pointer[id1][i]) {
            value_max = om->counter_pointer[id1][i];
            id_max = id1;
          }
#    endif
        }

#    ifdef WITH_CLIENT_SHOW_STAT
        if (0) {
        }
#      define KERNEL_DATA_ARRAY(type, tname) \
        else if (show_stat_name != NULL && strcmp(show_stat_name, it_stat->first.c_str()) && \
                 strcmp(show_stat_name, #tname) == 0) \
        { \
          if (_kg->tname.counter_gpu == NULL) { \
            _kg->tname.counter_gpu = new int[_kg->tname.width]; \
          } \
          size_t index = (size_t)((double)i / (double)_kg->tname.counter_mul); \
          size_t index1 = (size_t)((double)(i + 1) / (double)_kg->tname.counter_mul); \
          if (_kg->tname.width < index1) \
            index1 = _kg->tname.width; \
          for (int j = index; j < index1; j++) \
            _kg->tname.counter_gpu[j] = id_max; \
        }
#      include "kernel/data_arrays.h"
        else {
        }
#    endif
      }
    }
    // sample_finished_omp
#  endif

    ///////////////////////////////////////////////////////////////////////
    // if (id == 0)
    {
      g_report_time += (omp_get_wtime() - t_id);
      g_report_time_count++;

#  ifdef WITH_CLIENT_SHOW_STAT
      {
#  else
      if (/*omp_devices[id].path_stat_is_done == 0 ||*/ g_report_time >= 3.0) {
#  endif
        double long_time = 0;
        for (int id1 = 0; id1 < devices_size; id1++) {

          if (pix_state[0] == 0) {
            if (long_time < omp_devices[id1].running_time[STREAM_PATH2]) {
              long_time = omp_devices[id1].running_time[STREAM_PATH2];
            }
            printf("%d: Time = %f [ms], %d-%d, %d\n",
                   id1,
                   omp_devices[id1].running_time[STREAM_PATH2],
                   omp_devices[id1].wtile.y,
                   omp_devices[id1].wtile.h,
                   end_sample);
          }
          else if (pix_state[1] == 0) {
            if (long_time < omp_devices[id1].running_time[STREAM_PATH1]) {
              long_time = omp_devices[id1].running_time[STREAM_PATH1];
            }
            printf("%d: Time = %f [ms], %d-%d, %d\n",
                   id1,
                   omp_devices[id1].running_time[STREAM_PATH1],
                   omp_devices[id1].wtile.y,
                   omp_devices[id1].wtile.h,
                   end_sample);
          }
        }

        printf("Rendering: FPS: %.2f, Time = %f [ms]\n", 1000.0f / long_time, long_time);

        printf("OMP: FPS: %.2f: Time = %f [ms]\n",
               1.0f / (omp_get_wtime() - t_id),
               (omp_get_wtime() - t_id) * 1000.0f);

#  ifndef WITH_CLIENT_SHOW_STAT
        g_report_time = 0;
        g_report_time_count = 0;
#  endif
      }

      if (pix_state[0] == 0) {
        pix_state[0] = 2;
      }
      else if (pix_state[1] == 0) {
        pix_state[1] = 2;
      }

      pix_state[3] = end_sample;
      ///////////////////////SAMPLES////////////////////////
#  ifdef ENABLE_INC_SAMPLES
      int num_samples = end_sample - start_sample;
      // if (omp_devices[id].path_stat_is_done != 0)
      {
        start_sample = start_sample + num_samples;
        end_sample = start_sample + num_samples;
      }
#  endif
      ///////////////////////LB////////////////////////
#  if defined(ENABLE_LOAD_BALANCE) || defined(ENABLE_LOAD_BALANCEv2)
#    if 1
      for (int id1 = 0; id1 < devices_size - 1; id1++) {
        float time1 = 0;
        float time2 = 0;
        if (dev_pixels_node == (DEVICE_PTR)dev_pixels_node1) {
          time1 = omp_devices[id1].running_time[STREAM_PATH1];
          time2 = omp_devices[id1 + 1].running_time[STREAM_PATH1];
        }
        else {
          time1 = omp_devices[id1].running_time[STREAM_PATH2];
          time2 = omp_devices[id1 + 1].running_time[STREAM_PATH2];
        }
        if (time1 < time2) {
          int pix_transm = (int)((float)omp_devices[id + 1].wtile.h *
                                 ((((float)time2 - (float)time1) / 2.0f) / time2));
          omp_devices[id1].wtile.h += pix_transm;
          omp_devices[id1 + 1].wtile.y += pix_transm;
          omp_devices[id1 + 1].wtile.h -= pix_transm;
        }
        else if (time1 > time2) {
          int pix_transm = (int)((float)omp_devices[id].wtile.h *
                                 ((((float)time1 - (float)time2) / 2.0f) / time1));
          omp_devices[id1].wtile.h -= pix_transm;
          omp_devices[id1 + 1].wtile.y -= pix_transm;
          omp_devices[id1 + 1].wtile.y -= pix_transm;
          omp_devices[id1 + 1].wtile.h += pix_transm;
        }
      }
#    endif

#    if 0
      {
        double g_image_total_time = 0;
        for (int i = 0; i < g_image_time.size(); i++) {
          g_image_total_time += g_image_time[i];
        }

        int g_image_time_id = 0;
        double g_image_time_reminder = 0.0;

        for (int id1 = 0; id1 < devices_size; id1++) {
          omp_devices[id1].wtile.y = g_image_time_id;

          double dev_time = g_image_time_reminder;
          double avg_time = g_image_total_time / (double)devices_size;

          for (int i = omp_devices[id1].wtile.y; i < g_image_time.size(); i++) {
            dev_time += g_image_time[i];

            g_image_time_id = i + 1;

            if (dev_time > avg_time)
              break;
          }

          g_image_time_reminder = dev_time - avg_time;

          omp_devices[id1].wtile.h = (devices_size - 1 == id1) ?
                                         tile_h - omp_devices[id1].wtile.y :
                                         g_image_time_id - omp_devices[id1].wtile.y;
        }
      }

#    endif
#  endif
    }

///////////////////////////////////////////////////////////////////

///////////////////////SAMPLES////////////////////////
#  ifndef ENABLE_INC_SAMPLES
    // cuda_assert(
    //    cudaMemset((CU_DEVICE_PTR)scope.get().cuda_mem_map[map_buffer_bin].device_pointer,
    //               0,
    //               scope.get().cuda_mem_map[map_buffer_bin].size));
    // if (id == 0)
    {
      size_t sizeBuf_node = w * h * _kg->data.film.pass_stride * sizeof(float);
      memset((char *)map_buffer_bin, 0, sizeBuf_node);
    }
#  endif

    //    omp_devices[id].path_stat_is_done++;
    //    //
    //#    pragma omp barrier
    //#    pragma omp flush

    if (pix_state[4] == 1) {
      break;
    }

    if (pix_state[2] == 1) {

      // if (id == 0) {
      pix_state[2] = 0;
      start_sample = start_sample2;
      end_sample = end_sample2;

      size_t sizeBuf_node = w * h * _kg->data.film.pass_stride * sizeof(float);
      memset((char *)map_buffer_bin, 0, sizeBuf_node);
      // }

      //#    pragma omp barrier
    }

#  ifdef WITH_CLIENT_SHOW_STAT
    if (g_report_time_count > 1)
      break;
#  endif
  }
  mem_free("dev_pixels_node1", CLIENT_DEVICE_ID, dev_pixels_node1, pix_size);
  mem_free("dev_pixels_node2", CLIENT_DEVICE_ID, dev_pixels_node2, pix_size);
}

void path_trace_internal_cpu(int numDevice,
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
                             char *signal_value)
{

#  if 1
  printf("Rendering %d-%d: %d-%d: %d-%d:, %d-%d: thr: %d, 0 %%\n",
         start_sample,
         end_sample,
         tile_x,
         tile_y,
         tile_w,
         tile_h,
         w,
         h,
         nprocs_cpu);
#  endif

  int start_sample2 = start_sample;
  int end_sample2 = end_sample;

  float g_report_time = 0;
  size_t g_report_time_count = 0;

  int devices_size = nprocs_cpu;

  size_t pix_size = w * h * SIZE_UCHAR4;

  char *pixels_node1 = (char *)map_pixel_bin;
  char *pixels_node2 = (char *)map_pixel_bin + pix_size;
  int *pix_state = (int *)signal_value;

  // omp_lock_t *lock0 = (omp_lock_t *)&pix_state[5];
  // omp_lock_t *lock1 = (omp_lock_t *)&pix_state[7];

  // std::vector<double> g_image_time(h);

  std::vector<OMPDevice> omp_devices(devices_size);

  DEVICE_PTR dev_pixels_node1 = mem_alloc(CLIENT_DEVICE_ID, "dev_pixels_node1", NULL, pix_size);

  DEVICE_PTR dev_pixels_node2 = mem_alloc(CLIENT_DEVICE_ID, "dev_pixels_node2", NULL, pix_size);

#  pragma omp parallel num_threads(devices_size)
  {
    int id = omp_get_thread_num();

    omp_devices[id].wtile.start_sample = start_sample;
    omp_devices[id].wtile.num_samples = end_sample - start_sample;

    if (omp_devices[id].d_work_tiles == 0) {

      omp_devices[id].d_work_tiles = (DEVICE_PTR)&omp_devices[id].wtile;

      omp_devices[id].wtile.x = tile_x;
      omp_devices[id].wtile.w = tile_w;

      omp_devices[id].wtile.offset = offset;
      omp_devices[id].wtile.stride = stride;
      omp_devices[id].wtile.buffer = (float *)map_buffer_bin;

      // cuda_assert(cudaMemcpy(
      //    scope.get().d_work_tiles, (char *)&wtile, sizeof(WorkTile), cudaMemcpyHostToDevice));

      // int tile_step_dev = (int)((float)tile_h / (float)devices_size);
      // int tile_last_dev = tile_h - (devices_size - 1) * tile_step_dev;

      // int tile_y_dev = tile_y + tile_step_dev * id;
      // int tile_h_dev = (devices_size - 1 == id) ? tile_last_dev : tile_step_dev;

      // const char *env_tiles = getenv("CLIENT_TILES");
      // if (env_tiles) {
      //   omp_devices[id].wtile.y = util_get_int_from_env_array(env_tiles, 2 * id + 0);
      //   omp_devices[id].wtile.h = util_get_int_from_env_array(env_tiles, 2 * id + 1);
      // }
      // else {
      //   omp_devices[id].wtile.y = tile_y_dev;
      //   omp_devices[id].wtile.h = tile_h_dev;
      // }

      omp_devices[id].wtile.y = tile_y;
      omp_devices[id].wtile.h = tile_h;

      // scope.get().num_threads_per_block = 64;
    }
  }

  int used_buffer = 0;

  while (true) {

    double t_id = omp_get_wtime();

    ccl::KernelGlobalsCPU *_kg = (ccl::KernelGlobalsCPU *)map_kg_bin;

#  pragma omp parallel num_threads(devices_size)  // private(kg)
    {
      int id = omp_get_thread_num();

      ccl::KernelGlobalsCPU kg = *((ccl::KernelGlobalsCPU *)map_kg_bin);

      DEVICE_PTR dev_pixels_node = (DEVICE_PTR)dev_pixels_node2;
      char *pixels_node = NULL;

      float *time_path = NULL;
      float *time_memcpy = NULL;
      int stream_memcpy_id = 0;

      // cudaHostFn_t host_fn_memcpy_status_flag;

      // bool render_run = true;

      if (/*(pix_state[0] == 0*/ used_buffer == 0) {

        //#  pragma omp master
        if (id == 0) {
          // omp_set_lock(lock0);
        }

        dev_pixels_node = (DEVICE_PTR)dev_pixels_node2;
        pixels_node = pixels_node2;

        time_path = &omp_devices[id].running_time[STREAM_PATH2];
        time_memcpy = &omp_devices[id].running_time[STREAM_PATH2_MEMCPY];
      }
      else if (/*pix_state[1] == 0*/ used_buffer == 1) {
        // printf("rend1: pix_state[1] = 0, %f\n", omp_get_wtime()); fflush(0);
        // pix_state[0] = 1;
        //#  pragma omp flush
        //#  pragma omp master
        if (id == 0) {
          // omp_set_lock(lock1);
        }

        dev_pixels_node = (DEVICE_PTR)dev_pixels_node1;
        pixels_node = pixels_node1;

        time_path = &omp_devices[id].running_time[STREAM_PATH1];
        time_memcpy = &omp_devices[id].running_time[STREAM_PATH1_MEMCPY];
      }
      //      else {
      //        // continue;
      //        render_run = false;
      //      }
      ///////////////////////

      ///////////////////GPU SYNC////////////////////////////
      // if (render_run)
      {
        double event_path_start = omp_get_wtime();

        // kernel_tex_alloc(&kg, (ccl::KernelGlobalsCPU *)kg_bin);

        // for (int y = omp_devices[id].wtile.y;
        //      y < omp_devices[id].wtile.h + omp_devices[id].wtile.y;
        //      y++) {
        //   double y_start = omp_get_wtime();
        //   for (int x = omp_devices[id].wtile.x;
        //        x < omp_devices[id].wtile.w + omp_devices[id].wtile.x;
        //        x++) {
        size_t size = omp_devices[id].wtile.w * omp_devices[id].wtile.h;

#  pragma omp for schedule(dynamic, 16)
        for (int i = 0; i < size; i++) {
          int y = i / omp_devices[id].wtile.w;
          int x = i - y * omp_devices[id].wtile.w;

          // for (int sample = start_sample; sample < end_sample; sample++) {
          //   kernel_omp_path_trace(&kg,
          //                         omp_devices[id].wtile.buffer,
          //                         sample,
          //                         x + omp_devices[id].wtile.x,
          //                         y + omp_devices[id].wtile.y,
          //                         omp_devices[id].wtile.offset,
          //                         omp_devices[id].wtile.stride);
          // }
          ccl::KernelWorkTile work_tile;
          work_tile.x = x + omp_devices[id].wtile.x;
          work_tile.y = y + omp_devices[id].wtile.y;
          work_tile.w = 1;
          work_tile.h = 1;
          work_tile.start_sample = start_sample;
          work_tile.num_samples = 1;
          work_tile.offset = omp_devices[id].wtile.offset;
          work_tile.stride = omp_devices[id].wtile.stride;
#  if 0
          render_samples_full_pipeline(
              &kg, work_tile, end_sample - start_sample, (char *)omp_devices[id].wtile.buffer);
#  endif
          if (dev_pixels_node != NULL) {
//#if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET)
//		kernel_film_convert_to_half_float(&kg,
//                                          (ccl::uchar4 *)dev_pixels_node,
//                                          omp_devices[id].wtile.buffer,
//                                          1.0f / end_sample,
//                                          x,
//                                          y,
//                                          omp_devices[id].wtile.offset,
//                                          omp_devices[id].wtile.stride);
//#else
#  if 0
              kernel_film_convert_to_byte(&kg,
                                          (ccl::uchar4 *)dev_pixels_node,
                                          omp_devices[id].wtile.buffer,
                                          1.0f / end_sample,
                                          x + omp_devices[id].wtile.x,
                                          y + omp_devices[id].wtile.y,
                                          omp_devices[id].wtile.offset,
                                          omp_devices[id].wtile.stride);
#  endif
            //#endif
          }
        }
        // double y_stop = omp_get_wtime();
        // g_image_time[y] = y_stop - y_start;
        //}

        double event_path_stop = omp_get_wtime();
        unsigned int total_work_size = omp_devices[id].wtile.w * omp_devices[id].wtile.h;

        double event_memcpy_start = omp_get_wtime();

        // transfer to CPU
        size_t pix_offset = (omp_devices[id].wtile.x + omp_devices[id].wtile.y * stride) *
                            SIZE_UCHAR4;

        memcpy((char *)pixels_node + pix_offset,
               (char *)dev_pixels_node + pix_offset,
               total_work_size * SIZE_UCHAR4);

        double event_memcpy_stop = omp_get_wtime();

        // cuda_assert(cudaEventElapsedTime(time_path, event_path_start, event_path_stop));
        time_path[0] = (event_path_stop - event_path_start) * 1000.0f;
        // cuda_assert(cudaEventElapsedTime(time_memcpy, event_memcpy_start, event_memcpy_stop));
        time_memcpy[0] = (event_memcpy_stop - event_memcpy_start) * 1000.0f;
      }

      {
        // if (pix_state[0] == 0) {
        //  pix_state[0] = 2;
        //  pix_state[1] = 0;
        //}
        // else //if (pix_state[1] == 0)
        //{
        //  pix_state[0] = 0;
        //  pix_state[1] = 2;
        //}

        if (/*pix_state[0] == 0*/ used_buffer == 0) {
          //#  pragma omp master
          if (id == 0) {
            // omp_unset_lock(lock0);
          }
        }
        else if (/*pix_state[1] == 0*/ used_buffer == 1) {
          //#  pragma omp master
          if (id == 0) {
            // omp_unset_lock(lock1);
          }
        }

        used_buffer++;
        if (used_buffer > 1)
          used_buffer = 0;
      }
    }

    ///////////////////////////////////////////////////////////////////////
    // if (id == 0)
    {
      g_report_time += (omp_get_wtime() - t_id);
      g_report_time_count++;

      if (/*omp_devices[id].path_stat_is_done == 0 ||*/ g_report_time >= 3.0) {
        double long_time = 0;
        for (int id1 = 0; id1 < devices_size; id1++) {

          if (/*pix_state[0] == 0*/ used_buffer == 0) {
            if (long_time < omp_devices[id1].running_time[STREAM_PATH2]) {
              long_time = omp_devices[id1].running_time[STREAM_PATH2];
            }
            // printf("%d: Time = %f [ms], %d-%d, %d\n",
            //       id1,
            //       omp_devices[id1].running_time[STREAM_PATH2],
            //       omp_devices[id1].wtile.y,
            //       omp_devices[id1].wtile.h,
            //       end_sample);
          }
          else if (/*pix_state[1] == 0*/ used_buffer == 1) {
            if (long_time < omp_devices[id1].running_time[STREAM_PATH1]) {
              long_time = omp_devices[id1].running_time[STREAM_PATH1];
            }
            // printf("%d: Time = %f [ms], %d-%d, %d\n",
            //       id1,
            //       omp_devices[id1].running_time[STREAM_PATH1],
            //       omp_devices[id1].wtile.y,
            //       omp_devices[id1].wtile.h,
            //       end_sample);
          }
        }

        printf("Rendering: FPS: %.2f, Time = %f [ms]\n", 1000.0f / long_time, long_time);

        printf("OMP: FPS: %.2f: Time = %f [ms]\n",
               1.0f / (omp_get_wtime() - t_id),
               (omp_get_wtime() - t_id) * 1000.0f);

        g_report_time = 0;
        g_report_time_count = 0;
      }

      pix_state[3] = end_sample;
      ///////////////////////SAMPLES////////////////////////
      //#  ifdef ENABLE_INC_SAMPLES
      int num_samples = end_sample - start_sample;
      // if (omp_devices[id].path_stat_is_done != 0)
      {
        start_sample = start_sample + num_samples;
        end_sample = start_sample + num_samples;
      }
      //#  endif
      ///////////////////////LB////////////////////////
    }

    ///////////////////////////////////////////////////////////////////

    ///////////////////////SAMPLES////////////////////////
    //#  ifndef ENABLE_INC_SAMPLES
    //    // cuda_assert(
    //    //    cudaMemset((CU_DEVICE_PTR)scope.get().cuda_mem_map[map_buffer_bin].device_pointer,
    //    //               0,
    //    //               scope.get().cuda_mem_map[map_buffer_bin].size));
    //    // if (id == 0)
    //    {
    //      size_t sizeBuf_node = w * h * _kg->data.film.pass_stride * sizeof(float);
    //      memset((char *)map_buffer_bin, 0, sizeBuf_node);
    //    }
    //#  endif

    //    omp_devices[id].path_stat_is_done++;
    //    //
    //#    pragma omp barrier
    //#    pragma omp flush

    if (pix_state[4] == 1) {
      break;
    }

    if (pix_state[2] == 1) {

      // if (id == 0) {
      pix_state[2] = 0;
      start_sample = start_sample2;
      end_sample = end_sample2;

      size_t sizeBuf_node = w * h * _kg->data.film.pass_stride * sizeof(float);
      memset((char *)map_buffer_bin, 0, sizeBuf_node);
      // }

      //#    pragma omp barrier
    }
  }
  mem_free("dev_pixels_node1", CLIENT_DEVICE_ID, dev_pixels_node1, pix_size);
  mem_free("dev_pixels_node2", CLIENT_DEVICE_ID, dev_pixels_node2, pix_size);
}

#  ifdef WITH_CLIENT_CUDA_CPU_STAT
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
                         int nprocs_gpu)
{
  int size = tile_h * tile_w;

  ccl::KernelGlobalsCPU *_kg = (ccl::KernelGlobalsCPU *)kg_bin;
  size_t sizeBuf_node = tile_w * tile_h * _kg->data.film.pass_stride * sizeof(float);
  memset((char *)buffer_bin, 0, sizeBuf_node);

  ccl::KernelGlobalsCPU *kgs = new ccl::KernelGlobalsCPU[nprocs_cpu];

  for (int i = 0; i < nprocs_cpu; i++) {
    ccl::KernelGlobalsCPU *kg = &kgs[i];
    memcpy(kg, (char *)kg_bin, sizeof(ccl::KernelGlobalsCPU));
#    ifdef CRYPT_COVERAGE
    Coverage coverage(
        kg, buffer_bin, stride, tile_x, tile_y, tile_w, tile_h, kg.data.film.pass_stride);
    thread_kernel_globals_init(kg, &coverage);
#    else
    // thread_kernel_globals_init(kg);
#    endif
  }

  ///////////////////
  int devices_size = nprocs_cpu;
  std::vector<OMPDevice> omp_devices(devices_size);

#    pragma omp parallel num_threads(devices_size)
  {
    int id = omp_get_thread_num();

    omp_devices[id].wtile.start_sample = start_sample;
    omp_devices[id].wtile.num_samples = end_sample - start_sample;

    omp_devices[id].d_work_tiles = (DEVICE_PTR)&omp_devices[id].wtile;

    omp_devices[id].wtile.x = tile_x;
    omp_devices[id].wtile.w = tile_w;

    omp_devices[id].wtile.offset = offset;
    omp_devices[id].wtile.stride = stride;
    omp_devices[id].wtile.buffer = (float *)buffer_bin;

    omp_devices[id].wtile.y = 0;
    omp_devices[id].wtile.h = 1;

    std::map<std::string, OMPMem>::iterator it_stat;
    for (it_stat = omp_stat_map.begin(); it_stat != omp_stat_map.end(); it_stat++) {
      OMPMem *om = &it_stat->second;

      if (om->counter_pointer != NULL)
        memset(&om->counter_pointer[id][0], 0, sizeof(unsigned long long int) * om->counter_size);
    }
  }
  //////////INIT/////////
  std::map<std::string, OMPMem> omp_stat_map_by_row;
  {
    std::map<std::string, OMPMem>::iterator it_stat;
    for (it_stat = omp_stat_map.begin(); it_stat != omp_stat_map.end(); it_stat++) {
      OMPMem *om = &it_stat->second;

      OMPMem *omp_mem_row = &omp_stat_map_by_row[it_stat->first];
      // std::string name;
      omp_mem_row->name = om->name;
      // size_t counter_size;
      omp_mem_row->counter_size = om->counter_size;
      // DEVICE_PTR data_pointer;
      omp_mem_row->data_pointer = om->data_pointer;
      // size_t data_size;
      omp_mem_row->data_size = om->data_size;
      // unsigned long long int **counter_pointer;
      omp_mem_row->counter_pointer = NULL;
      if (om->counter_pointer != NULL) {
        omp_mem_row->counter_pointer = new unsigned long long int *[tile_h];
        for (int i = 0; i < tile_h; i++) {
          omp_mem_row->counter_pointer[i] = new unsigned long long int[om->counter_size];
          memset(&omp_mem_row->counter_pointer[i][0],
                 0,
                 sizeof(unsigned long long int) * om->counter_size);
        }
      }
      // size_t *counter_sum;
      omp_mem_row->counter_sum = om->counter_sum;
    }
  }
  ///////////////////

  double wtime = omp_get_wtime();

#    pragma omp parallel num_threads(devices_size)
  {
    int id = omp_get_thread_num();

    ccl::KernelGlobalsCPU *kg = &kgs[id];

    double t_start = omp_get_wtime();

#    pragma omp for schedule(dynamic, 1)
    for (int y = tile_y; y < tile_y + tile_h; y++) {
      omp_devices[id].wtile.y = y;

      for (int x = omp_devices[id].wtile.x; x < omp_devices[id].wtile.w + omp_devices[id].wtile.x;
           x++) {
        for (int sample = start_sample; sample < end_sample; sample++) {
#    ifdef CRYPT_COVERAGE
          if (kg.data.film.cryptomatte_passes & CRYPT_ACCURATE) {
            coverage.init_pixel(x, y);
          }
#    endif
          kernel_omp_path_trace(kg,
                                omp_devices[id].wtile.buffer,
                                sample,
                                x,
                                y,
                                omp_devices[id].wtile.offset,
                                omp_devices[id].wtile.stride);
        }
      }

      std::map<std::string, OMPMem>::iterator it_stat;
      for (it_stat = omp_stat_map.begin(); it_stat != omp_stat_map.end(); it_stat++) {
        OMPMem *om = &it_stat->second;
        OMPMem *omp_mem_row = &omp_stat_map_by_row[it_stat->first];

        if (om->counter_pointer != NULL) {
          for (int i = 0; i < om->counter_size; i++) {
            omp_mem_row->counter_pointer[y][i] += om->counter_pointer[id][i];
            om->counter_pointer[id][i] = 0;
          }
        }
      }
    }

    printf("OMP %d: Time = %f [s], %d-%d, %d\n",
           id,
           omp_get_wtime() - t_start,
           omp_devices[id].wtile.y,
           omp_devices[id].wtile.h,
           end_sample);
  }

  printf("OMP Rendering %d-%d: %d-%d: %d-%d: thr: %d, 100 %%, time: %f [s]\n",
         start_sample,
         end_sample,
         tile_x,
         tile_y,
         tile_w,
         tile_h,
         nprocs_cpu,
         omp_get_wtime() - wtime);

  fflush(0);

  for (int i = 0; i < nprocs_cpu; i++) {
    ccl::KernelGlobalsCPU *kg = &kgs[i];
#    ifdef CRYPT_COVERAGE
    thread_kernel_globals_free(&kg, &coverage);
#    else
    // thread_kernel_globals_free(kg);
#    endif
  }

  // delete[] kgs;

  //////////////////////////GROUP STAT//////////////////////////////////
  devices_size = nprocs_gpu;

#    pragma omp parallel num_threads(devices_size)
  {
    int id = omp_get_thread_num();

    int tile_step_dev = (int)((float)tile_h / (float)devices_size);
    int tile_last_dev = tile_h - (devices_size - 1) * tile_step_dev;

    int tile_y_dev = tile_y + tile_step_dev * id;
    int tile_h_dev = (devices_size - 1 == id) ? tile_last_dev : tile_step_dev;

    int wtile_y = tile_y_dev;
    int wtile_h = tile_h_dev;

    const char *env_tiles = getenv("CLIENT_TILES");
    if (env_tiles) {
      wtile_y = util_get_int_from_env_array(env_tiles, 2 * id + 0);
      wtile_h = util_get_int_from_env_array(env_tiles, 2 * id + 1);
    }

    std::map<std::string, OMPMem>::iterator it_stat;
    for (it_stat = omp_stat_map.begin(); it_stat != omp_stat_map.end(); it_stat++) {
      OMPMem *om = &it_stat->second;
      OMPMem *omp_mem_row = &omp_stat_map_by_row[it_stat->first];

      if (om->counter_pointer != NULL) {
        for (int i = 0; i < om->counter_size; i++) {
          om->counter_pointer[id][i] = 0;
          for (int y = wtile_y; y < wtile_h + wtile_y; y++) {
            om->counter_pointer[id][i] += omp_mem_row->counter_pointer[y][i];
          }
        }
      }
    }
  }

  //////////CLEAN/////////
  {
    std::map<std::string, OMPMem>::iterator it_stat;
    for (it_stat = omp_stat_map_by_row.begin(); it_stat != omp_stat_map_by_row.end(); it_stat++) {
      OMPMem *omp_mem_row = &it_stat->second;
      if (omp_mem_row->counter_pointer != NULL) {
        for (int i = 0; i < tile_h; i++) {
          delete[] omp_mem_row->counter_pointer[i];
        }
        delete[] omp_mem_row->counter_pointer;
      }
    }
  }
  ///////////////////
}

#    if 0
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
                             int nprocs_cpu)
{
  int size = tile_h * tile_w;

  ccl::KernelGlobalsCPU *_kg = (ccl::KernelGlobalsCPU *)kg_bin;
  size_t sizeBuf_node = tile_w * tile_h * _kg->data.film.pass_stride * sizeof(float);
  memset((char *)buffer_bin, 0, sizeBuf_node);

  ccl::KernelGlobalsCPU *kgs = new ccl::KernelGlobalsCPU[nprocs_cpu];

  for (int i = 0; i < nprocs_cpu; i++) {
    ccl::KernelGlobalsCPU *kg = &kgs[i];
    memcpy(kg, (char *)kg_bin, sizeof(ccl::KernelGlobalsCPU));
#      ifdef CRYPT_COVERAGE
    Coverage coverage(
        kg, buffer_bin, stride, tile_x, tile_y, tile_w, tile_h, kg.data.film.pass_stride);
    thread_kernel_globals_init(kg, &coverage);
#      else
    thread_kernel_globals_init(kg);
#      endif
  }

  ///////////////////
  int devices_size = nprocs_cpu;
  std::vector<OMPDevice> omp_devices(devices_size);

#      pragma omp parallel num_threads(devices_size)
  {
    int id = omp_get_thread_num();

    omp_devices[id].wtile.start_sample = start_sample;
    omp_devices[id].wtile.num_samples = end_sample - start_sample;

    omp_devices[id].d_work_tiles = (DEVICE_PTR)&omp_devices[id].wtile;

    omp_devices[id].wtile.x = tile_x;
    omp_devices[id].wtile.w = tile_w;

    omp_devices[id].wtile.offset = offset;
    omp_devices[id].wtile.stride = stride;
    omp_devices[id].wtile.buffer = (float *)buffer_bin;

    int tile_step_dev = (int)((float)tile_h / (float)devices_size);
    int tile_last_dev = tile_h - (devices_size - 1) * tile_step_dev;

    int tile_y_dev = tile_y + tile_step_dev * id;
    int tile_h_dev = (devices_size - 1 == id) ? tile_last_dev : tile_step_dev;

    const char *env_tiles = getenv("CLIENT_TILES");
    if (env_tiles) {
      omp_devices[id].wtile.y = util_get_int_from_env_array(env_tiles, 2 * id + 0);
      omp_devices[id].wtile.h = util_get_int_from_env_array(env_tiles, 2 * id + 1);
    }
    else {
      omp_devices[id].wtile.y = tile_y_dev;
      omp_devices[id].wtile.h = tile_h_dev;
    }

    std::map<std::string, OMPMem>::iterator it_stat;
    for (it_stat = omp_stat_map.begin(); it_stat != omp_stat_map.end(); it_stat++) {
      OMPMem *om = &it_stat->second;

      if (om->counter_pointer != NULL)
        memset(&om->counter_pointer[id][0], 0, sizeof(unsigned long long int) * om->counter_size);
    }
  }
  ///////////////////

  double wtime = omp_get_wtime();

#      pragma omp parallel num_threads(devices_size)
  {
    int id = omp_get_thread_num();

    ccl::KernelGlobalsCPU *kg = &kgs[id];

    double t_start = omp_get_wtime();
    for (int y = omp_devices[id].wtile.y; y < omp_devices[id].wtile.h + omp_devices[id].wtile.y;
         y++) {

      for (int x = omp_devices[id].wtile.x; x < omp_devices[id].wtile.w + omp_devices[id].wtile.x;
           x++) {
        for (int sample = start_sample; sample < end_sample; sample++) {
#      ifdef CRYPT_COVERAGE
          if (kg.data.film.cryptomatte_passes & CRYPT_ACCURATE) {
            coverage.init_pixel(x, y);
          }
#      endif
          kernel_omp_path_trace(kg,
                                omp_devices[id].wtile.buffer,
                                sample,
                                x,
                                y,
                                omp_devices[id].wtile.offset,
                                omp_devices[id].wtile.stride);
        }
      }
    }

    printf("%d: Time = %f [s], %d-%d, %d\n",
           id,
           omp_get_wtime() - t_start,
           omp_devices[id].wtile.y,
           omp_devices[id].wtile.h,
           end_sample);
  }

  printf("Rendering %d-%d: %d-%d: %d-%d: thr: %d, 100 %%, time: %f [s]\n",
         start_sample,
         end_sample,
         tile_x,
         tile_y,
         tile_w,
         tile_h,
         nprocs_cpu,
         omp_get_wtime() - wtime);

  fflush(0);

  for (int i = 0; i < nprocs_cpu; i++) {
    ccl::KernelGlobalsCPU *kg = &kgs[i];
#      ifdef CRYPT_COVERAGE
    thread_kernel_globals_free(&kg, &coverage);
#      else
    thread_kernel_globals_free(kg);
#      endif
  }

  // delete[] kgs;
}
#    endif

///////////////////////////////////////////////////////////////////

///////////////////////SAMPLES////////////////////////
//#    ifndef ENABLE_INC_SAMPLES
//    // cuda_assert(
//    //    cudaMemset((CU_DEVICE_PTR)scope.get().cuda_mem_map[map_buffer_bin].device_pointer,
//    //               0,
//    //               scope.get().cuda_mem_map[map_buffer_bin].size));
//    // if (id == 0)
//    {
//      size_t sizeBuf_node = tile_w * tile_h * _kg->data.film.pass_stride * sizeof(float);
//      memset((char *)map_buffer_bin, 0, sizeBuf_node);
//    }
//#    endif

//    omp_devices[id].path_stat_is_done++;
//    //
//#    pragma omp barrier
//#    pragma omp flush
//}
// mem_free(CLIENT_DEVICE_ID, dev_pixels_node1, pix_size);
// mem_free(CLIENT_DEVICE_ID, dev_pixels_node2, pix_size);
//}
#  endif

#else
void path_trace_internal(DEVICE_PTR kg_bin,
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
                         int nprocs_cpu)
{
  int size = tile_h * tile_w;

  int *sample_finished = (int *)sample_finished_omp;
  int *reqFinished = (int *)reqFinished_omp;

  *sample_finished = start_sample;
  // printf("CLIENT: start init kgs\n");
  ccl::KernelGlobalsCPU *kgs = new ccl::KernelGlobalsCPU[nprocs_cpu];
  // std::vector<ccl::KernelGlobalsCPU> kgs(nprocs_cpu);

  for (int i = 0; i < nprocs_cpu; i++) {
    ccl::KernelGlobalsCPU *kg = &kgs[i];
    memcpy(kg, (char *)kg_bin, sizeof(ccl::KernelGlobalsCPU));
#  ifdef CRYPT_COVERAGE
    Coverage coverage(
        kg, buffer_bin, stride, tile_x, tile_y, tile_w, tile_h, kg.data.film.pass_stride);
    thread_kernel_globals_init(kg, &coverage);
#  else
    // thread_kernel_globals_init(kg);
#  endif
  }

  // printf("CLIENT: finish init kgs\n");

#  pragma omp parallel num_threads(nprocs_cpu)
  {
    int tid = omp_get_thread_num();
    double progress = omp_get_wtime();
    double t = omp_get_wtime();

    ccl::KernelGlobalsCPU *kg = &kgs[tid];

#  if 1
    if (tid == 0) {
      printf("Rendering %d-%d: %d-%d: %d-%d: thr: %d, 0 %%\n",
             start_sample,
             end_sample,
             tile_x,
             tile_y,
             tile_w,
             tile_h,
             nprocs_cpu);
      fflush(0);
      // printf("Rendering %d:  %d-%d: %.1f %%, %f [s], thr: %d\n", dev, start_sample,
      // end_sample, 0.0, 0.0, nprocs_cpu); fflush(0);
    }
#  endif

//#  ifdef CLIENT_MPI_LOAD_BALANCING_SAMPLES
//#    pragma omp for schedule(dynamic, 16)
//#  else
#  pragma omp for schedule(dynamic, 1)
    //#  endif
    for (int i = 0; i < size; i++) {
      if (*reqFinished != 0)
        continue;

      int y = i / tile_w;
      int x = i - y * tile_w;

      //      for (int sample = start_sample; sample < end_sample; sample++) {
      //#  ifdef CRYPT_COVERAGE
      //        if (kg.data.film.cryptomatte_passes & CRYPT_ACCURATE) {
      //          coverage.init_pixel(x + tile_x, y + tile_y);
      //        }
      //#  endif
      //        kernel_omp_path_trace(
      //            kg, (float *)buffer_bin, sample, x + tile_x, y + tile_y, offset, stride);
      //      }

      ccl::KernelWorkTile work_tile;
      work_tile.x = x + tile_x;
      work_tile.y = y + tile_y;
      work_tile.w = tile_w;
      work_tile.h = tile_h;
      work_tile.start_sample = start_sample;
      work_tile.num_samples = end_sample - start_sample;
      work_tile.offset = offset;
      work_tile.stride = stride;

      render_samples_full_pipeline(kg, work_tile, end_sample - start_sample, (char *)buffer_bin);
#  if 0
      if (pixel_bin != NULL) {
        kernel_film_convert_to_byte(kg,
                                    (ccl::uchar4 *)pixel_bin,
                                    (float *)buffer_bin,
                                    1.0f / end_sample,
                                    x + tile_x,
                                    y + tile_y,
                                    offset,
                                    stride);
      }
#  endif

#  if 0
      int progress_tmp = (int)((float)i * 100.0f / (float)size);
      if (tid == 0 && omp_get_wtime() - progress >= 60) {
        printf("Rendering %d-%d: %d-%d: %d-%d: %d %%, %f [s]\n",
               start_sample,
               end_sample,
			   tile_x,
               tile_y,
               tile_w,
               tile_h,
               progress_tmp,
               omp_get_wtime() - t);
        // fflush(0);

        progress = omp_get_wtime();
      }
#  endif
    }

#  if 1
    if (tid == 0) {
      printf("Rendering %d-%d: %d-%d: %d-%d: thr: %d, 100 %%, time: %f [s]\n",
             start_sample,
             end_sample,
             tile_x,
             tile_y,
             tile_w,
             tile_h,
             nprocs_cpu,
             omp_get_wtime() - t);

      fflush(0);
    }
#  endif

#  pragma omp critical
    {
#  pragma omp flush
    }
  }
  for (int i = 0; i < nprocs_cpu; i++) {
    ccl::KernelGlobalsCPU *kg = &kgs[i];
#  ifdef CRYPT_COVERAGE
    thread_kernel_globals_free(&kg, &coverage);
#  else
    // thread_kernel_globals_free(kg);
#  endif
  }

  // delete [] kgs;

  *sample_finished = end_sample;
}

#  ifdef WITH_POP
//////////////////////POP//////////////////////////
size_t divide_up2(size_t x, size_t y)
{
  return (x + y - 1) / y;
}

struct Tile {
  int x, y, w, h;
  Tile()
  {
  }
  Tile(int _x, int _y, int _w, int _h) : x(_x), y(_y), w(_w), h(_h)
  {
  }
};

void gen_tiles(int tile_size_x,
               int tile_size_y,
               int image_w,
               int image_h,
               // int num_logical_devices,
               std::vector<Tile> &tiles)
{
  // int slice_y = 0;
  // int slice_h = image_h;

  // int num = min(image_h, num_logical_devices);
  int tile_w = (tile_size_x >= image_w) ? 1 : divide_up2(image_w, tile_size_x);
  int tile_h = (tile_size_y >= image_h) ? 1 : divide_up2(image_h, tile_size_y);

  // int tiles_per_device = divide_up2(tile_w * tile_h, num);
  // int cur_device = 0, cur_tiles = 0;

  for (int tile_y = 0; tile_y < tile_h; tile_y++) {
    for (int tile_x = 0; tile_x < tile_w; tile_x++ /*, idx++*/) {
      int x = tile_x * tile_size_x;
      int y = tile_y * tile_size_y;
      int w = (tile_x == tile_w - 1) ? image_w - x : tile_size_x;
      int h = (tile_y == tile_h - 1) ? image_h - y : tile_size_y;

      tiles.push_back(Tile(x, y, w, h));
    }
  }
}

void path_trace_internal_pop(DEVICE_PTR kg_bin,
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
                             int nprocs_cpu)
{
  //////////////////////////////////////////////////
  int size = tile_h * tile_w;

  int *sample_finished = (int *)sample_finished_omp;
  int *reqFinished = (int *)reqFinished_omp;

  *sample_finished = start_sample;
  ccl::KernelGlobalsCPU *kgs = new ccl::KernelGlobalsCPU[nprocs_cpu];

  for (int i = 0; i < nprocs_cpu; i++) {
    ccl::KernelGlobalsCPU *kg = &kgs[i];
    memcpy(kg, (char *)kg_bin, sizeof(ccl::KernelGlobalsCPU));
    thread_kernel_globals_init(kg);
  }

#    if defined(CLIENT_PATH_TRACE_TILE)
  std::vector<Tile> tiles;
  gen_tiles(8, 8, tile_w, tile_h, tiles);
#    endif

  double t = omp_get_wtime();

#    pragma omp parallel num_threads(nprocs_cpu)
  {
    int tid = omp_get_thread_num();
    ccl::KernelGlobalsCPU *kg = &kgs[tid];

    // if (tid == 0) {
    //  printf("Rendering:  %d-%d: %d, %d, %d, %d, %.1f %%, %f [s], thr: %d\n",
    //         start_sample,
    //         end_sample,
    //         tile_x,
    //         tile_y,
    //         tile_w,
    //         tile_h,
    //         0.0,
    //         0.0,
    //         nprocs_cpu);
    //}

#    if defined(CLIENT_PATH_TRACE_ROW)
#      pragma omp for
    for (int y = 0; y < tile_h; y++)
      for (int x = 0; x < tile_w; x++) {

#    elif defined(CLIENT_PATH_TRACE_COLUMN)
#      pragma omp for
    for (int x = 0; x < tile_w; x++)
      for (int y = 0; y < tile_h; y++) {

#    elif defined(CLIENT_PATH_TRACE_TILE)
#      pragma omp for schedule(dynamic, 1)
    for (int i = 0; i < tiles.size(); i++) {
      int ty = tiles[i].y;
      int th = ty + tiles[i].h;

      int tx = tiles[i].x;
      int tw = tx + tiles[i].w;

      for (int y = ty; y < th; y++) {
        for (int x = tx; x < tw; x++) {

#    elif defined(CLIENT_PATH_TRACE_PIXEL_1)
#      pragma omp for schedule(dynamic, 1)
    for (int i = 0; i < size; i++) {
      int y = i / tile_w;
      int x = i - y * tile_w;

#    elif defined(CLIENT_PATH_TRACE_PIXEL_2)
#      pragma omp for schedule(dynamic, 2)
    for (int i = 0; i < size; i++) {
      int y = i / tile_w;
      int x = i - y * tile_w;

#    elif defined(CLIENT_PATH_TRACE_PIXEL_4)
#      pragma omp for schedule(dynamic, 4)
    for (int i = 0; i < size; i++) {
      int y = i / tile_w;
      int x = i - y * tile_w;

#    elif defined(CLIENT_PATH_TRACE_PIXEL_8)
#      pragma omp for schedule(dynamic, 8)
    for (int i = 0; i < size; i++) {
      int y = i / tile_w;
      int x = i - y * tile_w;

#    elif defined(CLIENT_PATH_TRACE_PIXEL_16)
#      pragma omp for schedule(dynamic, 16)
    for (int i = 0; i < size; i++) {
      int y = i / tile_w;
      int x = i - y * tile_w;

#    elif defined(CLIENT_PATH_TRACE_PIXEL_32)
#      pragma omp for schedule(dynamic, 32)
    for (int i = 0; i < size; i++) {
      int y = i / tile_w;
      int x = i - y * tile_w;

#    elif defined(CLIENT_PATH_TRACE_PIXEL_64)
#      pragma omp for schedule(dynamic, 64)
    for (int i = 0; i < size; i++) {
      int y = i / tile_w;
      int x = i - y * tile_w;

#    elif defined(CLIENT_PATH_TRACE_PIXEL_128)
#      pragma omp for schedule(dynamic, 128)
    for (int i = 0; i < size; i++) {
      int y = i / tile_w;
      int x = i - y * tile_w;

#    elif defined(CLIENT_PATH_TRACE_PIXEL_256)
#      pragma omp for schedule(dynamic, 256)
    for (int i = 0; i < size; i++) {
      int y = i / tile_w;
      int x = i - y * tile_w;

#    else
#      pragma omp for
    for (int i = 0; i < size; i++) {
      int y = i / tile_w;
      int x = i - y * tile_w;

#    endif

        for (int sample = start_sample; sample < end_sample; sample++) {
          kernel_omp_path_trace(
              kg, (float *)buffer_bin, sample, x + tile_x, y + tile_y, offset, stride);
        }
      }

#    ifdef CLIENT_PATH_TRACE_TILE
  }
}
#    endif

// if (tid == 0) {
//  printf("Rendering:  %d-%d: %d, %d, %d, %d, %.1f %%, %f [s], thr: %d\n",
//         start_sample,
//         end_sample,
//         tile_x,
//         tile_y,
//         tile_w,
//         tile_h,
//         100.0,
//         omp_get_wtime() - t,
//         nprocs_cpu);
//}
}

printf("Rendering:  %d-%d: %d, %d, %d, %d, %.1f %%, %f [s], thr: %d\n",
       start_sample,
       end_sample,
       tile_x,
       tile_y,
       tile_w,
       tile_h,
       100.0,
       omp_get_wtime() - t,
       nprocs_cpu);

for (int i = 0; i < nprocs_cpu; i++) {
  ccl::KernelGlobalsCPU *kg = &kgs[i];
  thread_kernel_globals_free(kg);
}

*sample_finished = end_sample;
}

#  endif

#endif

void load_textures(int numDevice,
                   DEVICE_PTR kg_bin,
                   size_t texture_info_size
                   /*std::map<DEVICE_PTR, DEVICE_PTR> &ptr_map*/)
{
  ccl::KernelGlobalsCPU *kg = (ccl::KernelGlobalsCPU *)kg_bin;

  for (int i = 0; i < texture_info_size; i++) {
    ccl::TextureInfo &info = kg->texture_info.data[i];

    if (info.data /*&& ptr_map.find(info.data) != ptr_map.end()*/) {
      info.data = get_ptr_map(info.data); //ptr_map[info.data];
      int depth = (info.depth > 0) ? info.depth : 1;

#ifdef WITH_CPU_STAT
      OMP_TEX_COPY_STAT(&info.counter,
                        &info.counter_mul,
                        &info.counter_size,
                        info.width * info.height * depth,
                        0,
                        NULL,
                        (char *)info.data,
                        numDevice);
#endif
    }
  }
}

void enable_stat(bool enable)
{
#ifdef WITH_CPU_STAT
  kernel_compat_cpu_enable_stat = (enable) ? 1 : 0;
  kernel_cpu_image_enable_stat = (enable) ? 1 : 0;
#endif
}

#if defined(WITH_CLIENT_CUDA_CPU_STAT)

void path_trace_time2(DEVICE_PTR kg_bin,
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
                      double *line_times)
{
  int size = tile_h * tile_w;

  ccl::KernelGlobalsCPU *kgs = new ccl::KernelGlobalsCPU[nprocs_cpu];

  for (int i = 0; i < nprocs_cpu; i++) {
    ccl::KernelGlobalsCPU *kg = &kgs[i];
    memcpy(kg, (char *)kg_bin, sizeof(ccl::KernelGlobalsCPU));
    // thread_kernel_globals_init(kg);
  }

#  ifdef WITH_CPU_STAT
  const char *stat_lb_by_data = getenv("STAT_LB_BY_DATA");
#  endif

#  pragma omp parallel num_threads(nprocs_cpu)
  {
    int tid = omp_get_thread_num();
    ccl::KernelGlobalsCPU *kg = &kgs[tid];

    int tile_step_dev = (int)((float)tile_h / (float)nprocs_cpu);
    int tile_last_dev = tile_h - (nprocs_cpu - 1) * tile_step_dev;

    int tile_y_dev = tile_y + tile_step_dev * tid;
    int tile_h_dev = (nprocs_cpu - 1 == tid) ? tile_last_dev : tile_step_dev;

    double t = omp_get_wtime();

    for (int y = 0; y < tile_h_dev; y++) {
      double t1 = omp_get_wtime();

      for (int x = 0; x < tile_w; x++) {

        for (int sample = start_sample; sample < end_sample; sample++) {
          kernel_omp_path_trace(
              kg, (float *)buffer_bin, sample, x + tile_x, y + tile_y_dev, offset, stride);
        }
      }
#  ifdef WITH_CPU_STAT
      if (stat_lb_by_data != NULL) {
        std::string names = std::string(stat_lb_by_data);
        size_t counter_sum = 0;

        std::map<std::string, OMPMem>::iterator it_stat;
        for (it_stat = omp_stat_map.begin(); it_stat != omp_stat_map.end(); it_stat++) {
          if (names == "all" || names.find(it_stat->first) != std::string::npos ||
              it_stat->first.find(names) != std::string::npos) {
            OMPMem *om = &it_stat->second;
            for (int i = 0; i < om->counter_size; i++) {
              counter_sum += om->counter_pointer[tid][i];
              om->counter_pointer[tid][i] = 0;
            }
          }
        }

        line_times[y + tile_y_dev] = counter_sum;
      }
      else
#  endif
        line_times[y + tile_y_dev] = omp_get_wtime() - t1;
    }

    printf("Rendering %d-%d: %d-%d: %d-%d: thr: %d, 100 %%, time: %f [s]\n",
           start_sample,
           end_sample,
           tile_x,
           tile_y_dev,
           tile_w,
           tile_h_dev,
           nprocs_cpu,
           omp_get_wtime() - t);
  }

  //#ifdef WITH_CPU_STAT
  //  if (stat_lb_by_data == NULL) {
  //    omp_enable_stat(true);
  //  }
  //#endif

  for (int i = 0; i < nprocs_cpu; i++) {
    ccl::KernelGlobalsCPU *kg = &kgs[i];
    // thread_kernel_globals_free(kg);
  }

//  float *b = (float *)buffer_bin;
//  printf("CPU: buffer_bin: %f, %f, %f\n", b[0], b[1], b[2]);
#  if 0
    util_save_bmp(offset,
                  stride,
                  tile_x,
                  tile_y,
                  tile_h,
                  tile_w,
                  ((ccl::KernelGlobalsCPU *)kg_bin)->data.film.pass_stride,
                  end_sample,
                  buffer_bin,
                  NULL,
                  0
    );
#  endif

  fflush(0);
}

#  if defined(WITH_CLIENT_CUDA_CPU_STAT)

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
                     double *line_times)
{
  int size = tile_h * tile_w;

  ccl::KernelGlobalsCPU *_kg = (ccl::KernelGlobalsCPU *)kg_bin;
  size_t sizeBuf_node = tile_w * tile_h * _kg->data.film.pass_stride * sizeof(float);
  memset((char *)buffer_bin, 0, sizeBuf_node);

  ccl::KernelGlobalsCPU *kgs = new ccl::KernelGlobalsCPU[nprocs_cpu];

  for (int i = 0; i < nprocs_cpu; i++) {
    ccl::KernelGlobalsCPU *kg = &kgs[i];
    memcpy(kg, (char *)kg_bin, sizeof(ccl::KernelGlobalsCPU));
#    ifdef CRYPT_COVERAGE
    Coverage coverage(
        kg, buffer_bin, stride, tile_x, tile_y, tile_w, tile_h, kg.data.film.pass_stride);
    thread_kernel_globals_init(kg, &coverage);
#    else
    // thread_kernel_globals_init(kg);
#    endif
  }

  ///////////////////
#    ifdef WITH_CPU_STAT
  const char *stat_lb_by_data = getenv("STAT_LB_BY_DATA");
#    endif
  ///////////////////
  int devices_size = nprocs_cpu;
  std::vector<OMPDevice> omp_devices(devices_size);

#    pragma omp parallel num_threads(devices_size)
  {
    int id = omp_get_thread_num();

    omp_devices[id].wtile.start_sample = start_sample;
    omp_devices[id].wtile.num_samples = end_sample - start_sample;

    omp_devices[id].d_work_tiles = (DEVICE_PTR)&omp_devices[id].wtile;

    omp_devices[id].wtile.x = tile_x;
    omp_devices[id].wtile.w = tile_w;

    omp_devices[id].wtile.offset = offset;
    omp_devices[id].wtile.stride = stride;
    omp_devices[id].wtile.buffer = (float *)buffer_bin;

    omp_devices[id].wtile.y = 0;
    omp_devices[id].wtile.h = 1;
#    ifdef WITH_CPU_STAT
    std::map<std::string, OMPMem>::iterator it_stat;
    for (it_stat = omp_stat_map.begin(); it_stat != omp_stat_map.end(); it_stat++) {
      OMPMem *om = &it_stat->second;

      if (om->counter_pointer != NULL)
        memset(&om->counter_pointer[id][0], 0, sizeof(unsigned long long int) * om->counter_size);
    }
#    endif
  }
  //////////INIT/////////
#    ifdef WITH_CPU_STAT
  std::map<std::string, OMPMem> omp_stat_map_by_row;
  {
    std::map<std::string, OMPMem>::iterator it_stat;
    for (it_stat = omp_stat_map.begin(); it_stat != omp_stat_map.end(); it_stat++) {
      OMPMem *om = &it_stat->second;

      OMPMem *omp_mem_row = &omp_stat_map_by_row[it_stat->first];
      // std::string name;
      omp_mem_row->name = om->name;
      // size_t counter_size;
      omp_mem_row->counter_size = om->counter_size;
      // DEVICE_PTR data_pointer;
      omp_mem_row->data_pointer = om->data_pointer;
      // size_t data_size;
      omp_mem_row->data_size = om->data_size;
      // unsigned long long int **counter_pointer;
      omp_mem_row->counter_pointer = NULL;
      if (om->counter_pointer != NULL) {
        omp_mem_row->counter_pointer = new unsigned long long int *[tile_h];
        for (int i = 0; i < tile_h; i++) {
          omp_mem_row->counter_pointer[i] = new unsigned long long int[om->counter_size];
          memset(&omp_mem_row->counter_pointer[i][0],
                 0,
                 sizeof(unsigned long long int) * om->counter_size);
        }
      }
      // size_t *counter_sum;
      omp_mem_row->counter_sum = om->counter_sum;
    }
  }
#    endif
  ///////////////////

  double wtime = omp_get_wtime();

#    pragma omp parallel num_threads(devices_size)
  {
    int id = omp_get_thread_num();

    ccl::KernelGlobalsCPU *kg = &kgs[id];

    double t_start = omp_get_wtime();

#    pragma omp for schedule(dynamic, 1)
    for (int y = tile_y; y < tile_y + tile_h; y++) {
      double t1 = omp_get_wtime();

      omp_devices[id].wtile.y = y;

      for (int x = omp_devices[id].wtile.x; x < omp_devices[id].wtile.w + omp_devices[id].wtile.x;
           x++) {
        for (int sample = start_sample; sample < end_sample; sample++) {
#    ifdef CRYPT_COVERAGE
          if (kg.data.film.cryptomatte_passes & CRYPT_ACCURATE) {
            coverage.init_pixel(x, y);
          }
#    endif
          kernel_omp_path_trace(kg,
                                omp_devices[id].wtile.buffer,
                                sample,
                                x,
                                y,
                                omp_devices[id].wtile.offset,
                                omp_devices[id].wtile.stride);
        }
      }
#    ifdef WITH_CPU_STAT
      std::map<std::string, OMPMem>::iterator it_stat;
      for (it_stat = omp_stat_map.begin(); it_stat != omp_stat_map.end(); it_stat++) {
        OMPMem *om = &it_stat->second;
        OMPMem *omp_mem_row = &omp_stat_map_by_row[it_stat->first];

        if (om->counter_pointer != NULL) {
          for (int i = 0; i < om->counter_size; i++) {
            omp_mem_row->counter_pointer[y][i] += om->counter_pointer[id][i];
            om->counter_pointer[id][i] = 0;
          }
        }
      }

      if (stat_lb_by_data != NULL) {
        std::string names = std::string(stat_lb_by_data);
        size_t counter_sum = 0;

        std::map<std::string, OMPMem>::iterator it_stat;
        for (it_stat = omp_stat_map.begin(); it_stat != omp_stat_map.end(); it_stat++) {
          if (names == "all" || names.find(it_stat->first) != std::string::npos ||
              it_stat->first.find(names) != std::string::npos) {
            OMPMem *om = &it_stat->second;
            OMPMem *omp_mem_row = &omp_stat_map_by_row[it_stat->first];
            for (int i = 0; i < om->counter_size; i++) {
              counter_sum += omp_mem_row->counter_pointer[y][i];
              // om->counter_pointer[tid][i] = 0;
            }
          }
        }

        line_times[y] = counter_sum;
      }
      else
#    endif

        line_times[y] = omp_get_wtime() - t1;
    }

    printf("OMP %d: Time = %f [s], %d-%d, %d\n",
           id,
           omp_get_wtime() - t_start,
           omp_devices[id].wtile.y,
           omp_devices[id].wtile.h,
           end_sample);
  }

  printf("OMP Rendering %d-%d: %d-%d: %d-%d: thr: %d, 100 %%, time: %f [s]\n",
         start_sample,
         end_sample,
         tile_x,
         tile_y,
         tile_w,
         tile_h,
         nprocs_cpu,
         omp_get_wtime() - wtime);

  fflush(0);

  for (int i = 0; i < nprocs_cpu; i++) {
    ccl::KernelGlobalsCPU *kg = &kgs[i];
#    ifdef CRYPT_COVERAGE
    thread_kernel_globals_free(&kg, &coverage);
#    else
    // thread_kernel_globals_free(kg);
#    endif
  }

  // delete[] kgs;

  //////////////////////////GROUP STAT//////////////////////////////////
//    devices_size = nprocs_cpu;
//
//#    pragma omp parallel num_threads(devices_size)
//    {
//        int id = omp_get_thread_num();
//
//        int tile_step_dev = (int)((float)tile_h / (float)devices_size);
//        int tile_last_dev = tile_h - (devices_size - 1) * tile_step_dev;
//
//        int tile_y_dev = tile_y + tile_step_dev * id;
//        int tile_h_dev = (devices_size - 1 == id) ? tile_last_dev : tile_step_dev;
//
//        int wtile_y = tile_y_dev;
//        int wtile_h = tile_h_dev;
//
//        const char* env_tiles = getenv("CLIENT_TILES");
//        if (env_tiles) {
//            wtile_y = util_get_int_from_env_array(env_tiles, 2 * id + 0);
//            wtile_h = util_get_int_from_env_array(env_tiles, 2 * id + 1);
//        }
//
//        std::map<std::string, OMPMem>::iterator it_stat;
//        for (it_stat = omp_stat_map.begin(); it_stat != omp_stat_map.end(); it_stat++) {
//            OMPMem* om = &it_stat->second;
//            OMPMem* omp_mem_row = &omp_stat_map_by_row[it_stat->first];
//
//            if (om->counter_pointer != NULL) {
//                for (int i = 0; i < om->counter_size; i++) {
//                    om->counter_pointer[id][i] = 0;
//                    for (int y = wtile_y; y < wtile_h + wtile_y; y++) {
//                        om->counter_pointer[id][i] += omp_mem_row->counter_pointer[y][i];
//                    }
//                }
//            }
//        }
//    }
#    ifdef WITH_CPU_STAT
  //////////CLEAN/////////
  {
    std::map<std::string, OMPMem>::iterator it_stat;
    for (it_stat = omp_stat_map_by_row.begin(); it_stat != omp_stat_map_by_row.end(); it_stat++) {
      OMPMem *omp_mem_row = &it_stat->second;
      if (omp_mem_row->counter_pointer != NULL) {
        for (int i = 0; i < tile_h; i++) {
          delete[] omp_mem_row->counter_pointer[i];
        }
        delete[] omp_mem_row->counter_pointer;
      }
    }
  }
  ///////////////////
#    endif
}

void path_trace_time1(DEVICE_PTR kg_bin,
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
                      double *line_times)
{
  int size = tile_h * tile_w;
  std::vector<double> pixel_times(size);

  int cpu_threads = get_cpu_threads();

  //#  pragma omp parallel num_threads(cpu_threads)
  {
    // int tid = omp_get_thread_num();

    ccl::KernelGlobalsCPU *kg = ((ccl::KernelGlobalsCPU *)kg_bin);
    // thread_kernel_globals_init(&kg);

    //#  pragma omp for schedule(dynamic, 1)
    //        for (int i = 0; i < size; i++) {
    //
    //            double t0 = 0, t1 = 0;
    //            for (int j = 0; j < 2; j++) {
    //                int y = i / tile_w;
    //                int x = i - y * tile_w;
    //
    //                t0 = omp_get_wtime();
    //
    //                for (int sample = start_sample; sample < end_sample; sample++) {
    //                    kernel_omp_path_trace(
    //                            &kg, (float *) buffer_bin, sample, x + tile_x, y + tile_y,
    //                            offset, stride);
    //                }
    //
    //                t1 = omp_get_wtime();
    //            }
    //            pixel_times[i] = t1 - t0;
    //        }

    for (int y = 0; y < tile_h; y++) {
      double t1 = omp_get_wtime();

#    pragma omp parallel for num_threads(cpu_threads) schedule(dynamic, 80)
      for (int x = 0; x < tile_w; x++) {

        for (int sample = start_sample; sample < end_sample; sample++) {
          kernel_omp_path_trace(
              kg, (float *)buffer_bin, sample, x + tile_x, y + tile_y, offset, stride);
        }
      }
      line_times[y + tile_y] = omp_get_wtime() - t1;
    }

    // thread_kernel_globals_free(&kg);
  }

  //#  pragma omp parallel for num_threads(cpu_threads)
  //    for (int y = 0; y < tile_h; y++) {
  //        line_times[y] = 0;
  //        for (int x = 0; x < tile_w; x++) {
  //            line_times[y] += pixel_times[x + y * tile_w];
  //        }
  //    }
}
#  endif
#endif

void path_trace(int numDevice,
                DEVICE_PTR kg_bin,
                DEVICE_PTR buffer_bin,
                DEVICE_PTR pixel_bin,
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
                char *signal_value)
{

#if 0

#  if defined(OMP_INTERACTIVE)

#    if defined(WITH_CLIENT_MPI_SOCKET) || defined(WITH_CLIENT_MPI) || \
        defined(WITH_CLIENT_MPI_FILE)
    omp_path_trace_internal_cpu(numDevice,
        kg_bin,
        buffer_bin,
        pixel_bin,
        start_sample,
        end_sample,
        tile_x,
        tile_y,
        offset,
        stride,
        tile_h,
        tile_w,
        h,
        w,
        sample_finished_omp,
        reqFinished_omp,
        nprocs_cpu,
        signal_value);
#    else

  omp_path_trace_internal_gpu(numDevice,
                              kg_bin,
                              buffer_bin,
                              pixel_bin,
                              start_sample,
                              end_sample,
                              tile_x,
                              tile_y,
                              offset,
                              stride,
                              tile_h,
                              tile_w,
                              h,
                              w,
                              sample_finished_omp,
                              reqFinished_omp,
                              nprocs_cpu,
                              signal_value);
#    endif
#  elif defined(WITH_POP)
  omp_path_trace_internal_pop(kg_bin,
                              (char *)buffer_bin,
                              (char *)pixel_bin,
                              start_sample,
                              end_sample,
                              tile_x,
                              tile_y,
                              offset,
                              stride,
                              tile_h,
                              tile_w,
                              sample_finished_omp,
                              reqFinished_omp,
                              nprocs_cpu);
#  else
  path_trace_internal(kg_bin,
                          (char *)buffer_bin,
                          (char *)pixel_bin,
                          start_sample,
                          end_sample,
                          tile_x,
                          tile_y,
                          offset,
                          stride,
                          tile_h,
                          tile_w,
                          sample_finished_omp,
                          reqFinished_omp,
                          nprocs_cpu);
#  endif

#endif

  path_trace_internal(kg_bin,
                      (char *)buffer_bin,
                      (char *)pixel_bin,
                      start_sample,
                      end_sample,
                      tile_x,
                      tile_y,
                      offset,
                      stride,
                      tile_h,
                      tile_w,
                      h,
                      w,
                      sample_finished_omp,
                      reqFinished_omp,
                      nprocs_cpu,
                      signal_value);
}

void alloc_kg(int numDevice)
{
  //DEVICE_PTR kg_bin;

  //ccl::KernelGlobalsCPU *kg = &g_cpuDevice.kernel_globals;
  //kg_bin = (DEVICE_PTR)kg;

  //return (DEVICE_PTR)kg_bin;

  if (dev_kernel_data != NULL) {
    delete dev_kernel_data;
  }

  dev_kernel_data = new DevKernelData();
  //dev_kernel_data->kernel_globals_cpu = 0;
}

void free_kg(int numDevice, DEVICE_PTR kg_bin)
{
  ccl::KernelGlobalsCPU *kg = (ccl::KernelGlobalsCPU *)kg_bin;
  //  delete kg;
}

char *host_alloc(const char *name,
                 DEVICE_PTR mem,
                 size_t memSize,
                 int type)  // 1-managed, 2-pinned
{
  return new char[memSize];
}
void host_free(const char *name, DEVICE_PTR mem, char *dmem, int type)
{
  delete[] dmem;
}

DEVICE_PTR mem_alloc(
    int numDevice, const char *name, char *mem, size_t memSize, bool spec_setts, bool alloc_host)
{
#ifdef WITH_CLIENT_CUDA_CPU_STAT2v2
  char *temp = (mem == NULL) ? (char *)malloc(memSize) : mem;
#else
  char *temp = (char *)malloc(memSize);
#endif
  // memset(temp, 0, memSize);

  DEVICE_PTR dmem = (DEVICE_PTR)temp;

#ifdef WITH_CPU_STAT
  //  const char* client_in_unimem_by_size = getenv("CLIENT_IN_UNIMEM_BY_SIZE_MB");
  //  if (client_in_unimem_by_size != NULL && std::string(name).find("tex_image") ==
  //  std::string::npos) {
  //      size_t mem_limit = atol(client_in_unimem_by_size) * 1024L * 1024L;
  //
  //      if (memSize > mem_limit) {
  //          OMPMem *omem = &omp_stat_map[std::string(name)];
  //          omem->name = std::string(name);
  //          omem->data_pointer = (DEVICE_PTR)dmem;
  //          omem->data_size = memSize;
  //      }
  //  }
  //  else
  {
    OMPMem *omem = &omp_stat_map[std::string(name)];
    omem->name = std::string(name);
    omem->data_pointer = (DEVICE_PTR)dmem;
    omem->data_size = memSize;
  }
#endif

  if (!strcmp("client_buffer_passes", name)) {
    g_buffer_passes_d = dmem;
  }

  printf(
      "mem_alloc: %s, %.3f, %zu\n", name, (float)memSize / (1024.0f * 1024.0f), (DEVICE_PTR)dmem);

  return dmem;
}

void mem_copy_to(int numDevice, char *mem, DEVICE_PTR dmem, size_t memSize, char *signal_value)
{
  if (mem != NULL)
    memcpy((char *)dmem, mem, memSize);

  if (dmem == g_buffer_passes_d) {
    g_buffer_passes.resize(memSize / sizeof(client_buffer_passes));
    memcpy(g_buffer_passes.data(), mem, memSize);
  }
}

void mem_copy_from(
    int numDevice, DEVICE_PTR dmem, char *mem, size_t offset, size_t memSize, char *signal_value)
{
  if (mem != NULL && dmem != 0 && dmem != (DEVICE_PTR)mem)
    memcpy((char *)mem, (char *)dmem, memSize);
}

void mem_zero(const char *name, int numDevice, DEVICE_PTR dmem, size_t memSize)
{
  memset((char *)dmem, 0, memSize);
}

void mem_free(const char *name, int numDevice, DEVICE_PTR dmem, size_t memSize)
{
#ifdef WITH_CPU_STAT
  std::map<std::string, OMPMem>::iterator it_stat;
  for (it_stat = omp_stat_map.begin(); it_stat != omp_stat_map.end(); it_stat++) {
    OMPMem *o = &it_stat->second;
    if (o->data_pointer == (DEVICE_PTR)dmem) {
      omp_stat_map.erase(it_stat);
      break;
    }
  }
#endif
  char *temp = (char *)dmem;
  free(temp);
}

void tex_free(
    int numDevice, DEVICE_PTR kg_bin, const char *name_bin, DEVICE_PTR dmem, size_t memSize)
{
  // delete (char *)dmem;
  mem_free(name_bin, numDevice, dmem, memSize);
}

int get_pass_stride(int numDevice, DEVICE_PTR kg)
{
  return ((ccl::KernelGlobalsCPU *)kg)->data.film.pass_stride;
}

int get_cpu_threads()
{
  return omp_get_max_threads();
}

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
                          bool unimem_flag)
{
  DEVICE_PTR dmem = mem_alloc(numDevice, name, mem, memSize);
  // mem_copy_to(numDevice, mem, dmem, memSize, NULL);

  return dmem;
}

void tex_info_copy(const char *name,
                   char *mem,
                   DEVICE_PTR map_id,
                   size_t memSize,
                   int data_type,
                   bool check_uniname)
{
  mem_copy_to(CLIENT_DEVICE_ID, mem, map_id, memSize, NULL);
}

bool check_unimem(const char* _name)
{
    return false;
}

DEVICE_PTR get_kernel_globals_cpu()
{
  return (DEVICE_PTR)&g_cpuDevice.kernel_globals;
}

bool skip_caching_by_name(const char *name)
{
#if 1  // 0
  if (!strcmp("shaders", name))
    return true;

  if (!strcmp("svm_nodes", name))
    return true;

  if (!strcmp("lookup_table", name))
    return true;

  if (!strcmp("light_distribution", name))
    return true;

  if (!strcmp("lights", name))
    return true;
#endif

  return false;
}

DEVICE_PTR get_ptr_map(DEVICE_PTR id)
{
  DEVICE_PTR offset = 0;

  //if (g_caching_enabled != CACHING_DISABLED) {
  //  offset = (DEVICE_PTR)g_current_frame * 1000000000000000LL;

  //  if (dev_kernel_data->ptr2ptr_map.find(id) == dev_kernel_data->ptr2ptr_map.end() ||
  //      dev_kernel_data->ptr2ptr_map.find(id + offset) == dev_kernel_data->ptr2ptr_map.end())
  //    offset = 0;
  //}

  DEVICE_PTR res = dev_kernel_data->ptr2ptr_map[id + offset];

  if (res == 0) {
    printf("WARNING: get_ptr_map is empty!\n");
    fflush(0);
  }

  return res;
}

void set_ptr_map(const char *name, DEVICE_PTR id, DEVICE_PTR value)
{
  DEVICE_PTR offset = 0;
  //if (g_caching_enabled != CACHING_DISABLED) {
  //  offset = (DEVICE_PTR)g_current_frame * 1000000000000000LL;

  //  ///////////////////////////////////////////////////////////////
  //  if (g_caching_enabled == CACHING_PREVIEW && skip_caching_by_name(name)) {
  //    offset = 0;

  //    // if (dev_kernel_data->str2ptr_map.find(std::string(name)) !=
  //    //    dev_kernel_data->str2ptr_map.end()) {
  //    //  DEVICE_PTR id2 = dev_kernel_data->str2ptr_map[std::string(name)];
  //    //  value = dev_kernel_data->ptr2ptr_map[id2];
  //    //}
  //    // return;
  //  }
  //  ///////////////////////////////////////////////////////////////

  //  if (dev_kernel_data->ptr2ptr_map.find(id) == dev_kernel_data->ptr2ptr_map.end())
  //    offset = 0;
  //}

  dev_kernel_data->ptr2ptr_map[id + offset] = value;
  dev_kernel_data->str2ptr_map[std::string(name)] = id + offset;
}

void frame_info(int current_frame, int current_frame_preview, int caching_enabled)
{
  //g_current_frame = current_frame;
  //g_current_frame_preview = current_frame_preview;
  //g_caching_enabled = caching_enabled;

  // caching_kpc(int dev, ccl::CUDAContextScope &scope)
}

// CCL_NAMESPACE_END
}  // namespace omp
}  // namespace kernel
}  // namespace cyclesphi