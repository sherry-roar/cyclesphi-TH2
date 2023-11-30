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

#include "kernel_cuda.h"
//#include "kernel_cuda_device.h"
#include "device/cuda/device_impl.h"
#include "kernel/device/cuda/globals.h"

#ifdef WITH_CLIENT_OPTIX
#  include "device/optix/device_impl.h"
#endif

#include "kernel_cuda_stat.h"
#include "kernel_cuda_util.h"

#include "kernel_camera.h"
#include "kernel_util.h"

#include "integrator/path_trace_work_gpu.h"
#include "scene/film.h"
#include "scene/pass.h"
#include "scene/scene.h"
#include "util/color.h"

#include <vector>

// namespace cyclesphi {
// namespace kernel {
// DevKernelData *dev_kernel_data = NULL;
//}
//}  // namespace cyclesphi

namespace cyclesphi {
namespace kernel {
namespace cuda {
///////////////////////////////////////
struct DevKernelData {
  DEVICE_PTR kernel_globals_cpu;
  std::map<DEVICE_PTR, DEVICE_PTR> ptr2ptr_map;
  std::map<std::string, DEVICE_PTR> str2ptr_map;
};
//
// extern DevKernelData *dev_kernel_data;
DevKernelData *dev_kernel_data = NULL;

//////////////////////////////////////////////////////////////////////////

int cuda_kernel_data_res[2] = {-1, -1};
std::vector<char> cuda_kernel_data;

#if defined(WITH_CLIENT_RENDERENGINE_VR) || \
    (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
std::vector<char> cuda_kernel_data_right;
#endif

std::vector<char> texture_info_mem;
std::vector<client_buffer_passes> g_buffer_passes;
DEVICE_PTR g_buffer_passes_d = NULL;

char *g_blender_camera = NULL;

int g_world_rank = 0;
int g_world_size = 1;

//////////////////////////////////////////////////////
ccl::PathTraceWorkGPU **g_pathTraceWorkGPU = NULL;
ccl::Film g_film;
ccl::DeviceScene g_device_scene;
bool g_cancel_requested_flag = false;
// ccl::PathTraceWorkGPU::SceneData g_scene_data;

// size_t g_mem_spread_cycle = 0;
#define CACHING_DISABLED 0
#define CACHING_RECORD 1
#define CACHING_PREVIEW 2

//#define CACHING_SKIP_SHADERS

int g_current_frame = 0;
int g_current_frame_preview = -1;
int g_caching_enabled = 0;
std::map<int, std::map<int, ccl::KernelParamsCUDA>> cuda_kernel_map;

/////////////////////////////////////
// ccl::CLIENTDevice::global_free, shaders: 256
// ccl::CLIENTDevice::global_free, svm_nodes: 4976
// ccl::CLIENTDevice::global_alloc, svm_nodes: 0
// ccl::CLIENTDevice::global_alloc, shaders: 0
// ccl::CLIENTDevice::global_free, lookup_table: 266240
// ccl::CLIENTDevice::global_alloc, lookup_table: 0
// ccl::CLIENTDevice::global_free, light_distribution: 32
// ccl::CLIENTDevice::global_free, lights: 192
// ccl::CLIENTDevice::global_alloc, lights: 0
// ccl::CLIENTDevice::global_alloc, light_distribution: 0

DEVICE_PTR get_kernel_globals_cpu()
{
  return dev_kernel_data->kernel_globals_cpu;
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

  if (g_caching_enabled != CACHING_DISABLED) {
    offset = (DEVICE_PTR)g_current_frame * 1000000000000000LL;

    if (dev_kernel_data->ptr2ptr_map.find(id) == dev_kernel_data->ptr2ptr_map.end() ||
        dev_kernel_data->ptr2ptr_map.find(id + offset) == dev_kernel_data->ptr2ptr_map.end())
      offset = 0;
  }

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
  if (g_caching_enabled != CACHING_DISABLED) {
    offset = (DEVICE_PTR)g_current_frame * 1000000000000000LL;

    ///////////////////////////////////////////////////////////////
    if (g_caching_enabled == CACHING_PREVIEW && skip_caching_by_name(name)) {
      offset = 0;

      // if (dev_kernel_data->str2ptr_map.find(std::string(name)) !=
      //    dev_kernel_data->str2ptr_map.end()) {
      //  DEVICE_PTR id2 = dev_kernel_data->str2ptr_map[std::string(name)];
      //  value = dev_kernel_data->ptr2ptr_map[id2];
      //}
      // return;
    }
    ///////////////////////////////////////////////////////////////

    if (dev_kernel_data->ptr2ptr_map.find(id) == dev_kernel_data->ptr2ptr_map.end())
      offset = 0;
  }

  dev_kernel_data->ptr2ptr_map[id + offset] = value;
  dev_kernel_data->str2ptr_map[std::string(name)] = id + offset;
}
/////////////////////////////////////
bool check_unimem(const char *_name)
{
  return cuda_check_unimem(_name, 0);
}

char *get_mem_dev(int numDevice, DEVICE_PTR map_bin)
{
  return (char *)ccl::cuda_devices[numDevice].cuda_mem_map[map_bin].mem.device_pointer;
}

void host_register(const char *name, char *mem, size_t memSize)
{
  // for (int id = 0; id < ccl::cuda_devices.size(); id++) {
  //  ccl::CUDAContextScope scope(id);
  cuda_assert2(cudaHostRegister(mem, memSize, cudaHostAllocPortable /*cudaHostRegisterMapped*/));
  //}
}

DEVICE_PTR get_host_get_device_pointer(char *mem)
{
  CU_DEVICE_PTR dev_p;
  cuda_assert2(cudaHostGetDevicePointer(&dev_p, mem, 0));

  return (DEVICE_PTR)dev_p;
}

char *host_alloc(const char *name,
                 DEVICE_PTR mem,
                 size_t memSize,
                 int type)  // 1-managed, 2-pinned
{
  if (type == 2) {
    void *mm = NULL;
    cuda_assert2(cudaHostAlloc(&mm, memSize, cudaHostAllocMapped));
    memset(mm, 0, memSize);
    return (char *)mm;
  }

  if (type == 1) {
    void *mm = NULL;
    cuda_assert2(cudaMallocManaged(&mm, memSize));
#if 0
    cuda_assert2(cudaMemAdvise((char *)mm, memSize, cudaMemAdviseSetAccessedBy, CU_DEVICE_CPU));
#endif
    for (int id = 0; id < ccl::cuda_devices.size(); id++) {
#ifdef _MEMADVISE
      cuda_assert2(cudaMemAdvise((char *)mm, memSize, cudaMemAdviseSetAccessedBy, id));
#endif
    }

#if 0
    cuda_assert2(
        cudaMemAdvise((char *)mm, memSize, cudaMemAdviseSetPreferredLocation, CU_DEVICE_CPU));
#endif
    //      for (int id = 0; id < ccl::cuda_devices.size(); id++) {
    //        cuda_assert2(cudaMemAdvise((char *)mm, memSize, cudaMemAdviseSetPreferredLocation,
    //        id));
    //      }

    //      advise_flag = (i <= data050_id) ? cudaMemAdviseSetReadMostly :
    //                                       cudaMemAdviseSetPreferredLocation; //

    memset(mm, 0, memSize);
    return (char *)mm;
  }

  // TEST_ALLOC
  // if (false /*check_unimem(name, CUDA_DEVICE_ID)*/) {
  //  if (mem != NULL &&
  //      dev_kernel_data->ptr2ptr_map.find(mem) != dev_kernel_data->ptr2ptr_map.end())
  //    return (char *)dev_kernel_data->ptr2ptr_map[mem];
  //  else {
  //    void *mm = NULL;
  //    cuda_assert2(cudaMallocManaged(&mm, memSize));
  //    memset(mm, 0, memSize);
  //    return (char *)mm;
  //  }
  //}
  // else
  {
    char *mm = (char *)malloc(memSize);
    memset(mm, 0, memSize);
    return mm;
  }
}

void host_free(const char *name, DEVICE_PTR mem, char *dmem, int type)  // 1-managed, 2-pinned
{
  //#  ifdef WITH_CLIENT_UNIMEM
  //  cuda_assert2(cudaFree(dmem));
  //#  else
  //  delete[] dmem;
  //#  endif
  if (type == 2) {
    cuda_assert2(cudaFreeHost(dmem));
    return;
  }

  if (type == 1) {
    cuda_assert2(cudaFree(dmem));
    return;
  }

  // TEST_ALLOC
  // if (false /*check_unimem(name, CUDA_DEVICE_ID)*/) {
  //  if (mem != NULL &&
  //      dev_kernel_data->ptr2ptr_map.find(mem) != dev_kernel_data->ptr2ptr_map.end())
  //    return;
  //  else {
  //    cuda_assert2(cudaFree(dmem));
  //  }
  //}
  // else
  {
    free(dmem);
  }
}

void frame_info(int current_frame, int current_frame_preview, int caching_enabled)
{
  g_current_frame = current_frame;
  g_current_frame_preview = current_frame_preview;
  g_caching_enabled = caching_enabled;

  // caching_kpc(int dev, ccl::CUDAContextScope &scope)
}

void set_device(int device, int world_rank, int world_size)
{
  g_world_rank = world_rank;
  g_world_size = world_size;

  const char *env_cubin = getenv("KERNEL_CUDA_CUBIN");

  if (env_cubin == NULL) {
    printf("ASSERT: KERNEL_CUDA_CUBIN is empty\n");
    exit(-1);
  }
  else {
    printf("CUBIN: %s\n", env_cubin);
  }

#ifdef WITH_CUDA_STAT
  const char *env_cubin_stat = getenv("KERNEL_CUDA_STAT_CUBIN");

  if (env_cubin_stat == NULL) {
    printf("ASSERT: KERNEL_CUDA_STAT_CUBIN is empty\n");
    exit(-1);
  }
  else {
    printf("CUBIN_STAT: %s\n", env_cubin_stat);
  }
#endif

  // cu_assert(cuInit(0));

  int dev_count = 0;

  if (cuda_error(cudaGetDeviceCount(&dev_count))) {
    exit(-1);
  }

  if (device == -1) {

    const char *MPI_CUDA_VISIBLE_DEVICES = getenv("MPI_CUDA_VISIBLE_DEVICES");
    if (MPI_CUDA_VISIBLE_DEVICES != NULL) {
      int dev_start = util_get_int_from_env_array(MPI_CUDA_VISIBLE_DEVICES, world_rank * 2);
      dev_count = util_get_int_from_env_array(MPI_CUDA_VISIBLE_DEVICES, world_rank * 2 + 1);
      printf("rank: %d, dev_start: %d, dev_count:%d\n", world_rank, dev_start, dev_count);

      ccl::cuda_devices.resize(dev_count);

      for (int id = 0; id < ccl::cuda_devices.size(); id++) {
        /* Setup device and context. */
        ccl::cuda_devices[id].cuDevice = id + dev_start;
      }
    }
    else {
#if 0  // defined(WITH_CLIENT_NCCL_SOCKET) || defined(WITH_CLIENT_MPI_REDUCE)
      ccl::cuda_devices.resize(dev_count - 1);

      for (int id = 0; id < ccl::cuda_devices.size(); id++) {
        /* Setup device and context. */
        ccl::cuda_devices[id].cuDevice = id + 1;
      }
#else
      ccl::cuda_devices.resize(dev_count);

      for (int id = 0; id < ccl::cuda_devices.size(); id++) {
        /* Setup device and context. */
        ccl::cuda_devices[id].cuDevice = id;
      }
#endif
    }
  }
  else {
    ccl::cuda_devices.resize(1);

    int id = 0;
    ccl::cuda_devices[id].cuDevice = device % dev_count;
  }

#if 0
  {
    ccl::cuda_devices.resize(2);
    for (int id = 0; id < ccl::cuda_devices.size(); id++) {
      ccl::cuda_devices[id].cuDevice = 0;
    } 
  }
#endif

  for (int id = 0; id < ccl::cuda_devices.size(); id++) {
    ccl::CUDAContextScope scope(id);

    for (int s = 0; s < STREAM_COUNT; s++) {
      // printf("dev: %d, stream:%d\n", id, s);
      cuda_assert2(cudaStreamCreate(&scope.get().stream[s]));
    }

    for (int s = 0; s < EVENT_COUNT; s++) {
      cuda_assert2(cudaEventCreate(&scope.get().event[s]));
    }

    cu_assert(cuModuleLoad(&scope.get().cuModule, env_cubin));

#ifdef WITH_CUDA_STAT
    cu_assert(cuModuleLoad(&scope.get().cuModuleStat, env_cubin_stat));
#endif

    // scope.get().cuFilterModule = 0;
    scope.get().path_stat_is_done = 0;
    scope.get().texture_info_dmem = 0;
    // scope.get().min_num_active_paths_ = 0;
    // scope.get().max_active_path_index_ = 0;
    // scope.get().texture_info_mem_size = 0;
    // scope.get().d_work_tiles = 0;
    memset(scope.get().running_time, 0, sizeof(scope.get().running_time));
    // scope.get().time_old = 0;
    scope.get().time_count = 0;
    scope.get().wtile_h = 0;
    scope.get().used_buffer = 0;
    scope.get().start_sample = 0;
    scope.get().num_samples = 1;

    // scope.get().num_threads_per_block = 0;
    // scope.get().kerneldata = 0;
  }
  printf("\n========CUDA devices: %zu\n", ccl::cuda_devices.size());
}

void cam_recalc(char *data)
{
#if defined(WITH_CLIENT_RENDERENGINE_VR) || \
    (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))

  ccl::KernelData *kdata = (ccl::KernelData *)data;
  // cyclesphi_data cdata;

  // float transform_inverse_view_matrix[12];

  // float lens;
  // float clip_start;
  // float clip_end;

  // float sensor_width;
  // float sensor_height;
  // int sensor_fit;

  // float shift_x;
  // float shift_y;

  // float interocular_distance;
  // float convergence_distance;

  // float view_camera_zoom;
  // float view_camera_offset[2];
  // int use_view_camera;

  // cyclesphi_cam cleft;
  // cyclesphi_cam cright = cleft;

  // int width, height;
  // int step_samples;

  // memcpy(cdata.cam.transform_inverse_view_matrix, &kdata->cam.cameratoworld, sizeof(float) *
  // 12);

  // cdata.width = kdata->cam.width;
  // cdata.height = kdata->cam.height;
  // cdata.step_samples = 1;

  ////cdata.cam.lens = 50;
  // kdata->cam.fov = 2.0f * atanf((0.5f * sensor_size) / bcam->lens / aspectratio);

  // cdata.cam.clip_start = kdata->cam.nearclip;
  // cdata.cam.clip_end = (kdata->cam.cliplength == FLT_MAX) ?
  //                         FLT_MAX :
  //                         kdata->cam.nearclip + kdata->cam.cliplength;

  // cdata.cam.sensor_width = kdata->cam.sensorwidth;
  // cdata.cam.sensor_height = kdata->cam.sensorheight;
  // cdata.cam.sensor_fit = 0;

  // cdata.cam.shift_x = 0;
  // cdata.cam.shift_y = 0;

  // cdata.cam.interocular_distance = 0;
  // cdata.cam.convergence_distance = 0;

  // cdata.cam.view_camera_zoom = 1;
  // cdata.cam.view_camera_offset[0] = 0;
  // cdata.cam.view_camera_offset[1] = 0;
  // cdata.cam.use_view_camera = 1;

  // cdata.cam_right = cdata.cam;

  if (cuda_kernel_data.size() > 0) {
    bcam_to_kernel_camera(
        &cuda_kernel_data[0], g_blender_camera, kdata->cam.width, kdata->cam.height);
    bcam_to_kernel_camera_right(
        &cuda_kernel_data_right[0], g_blender_camera, kdata->cam.width, kdata->cam.height);
    //  view_to_kernel_camera(&cuda_kernel_data[0], (cyclesphi_data *)&cdata);
    //  view_to_kernel_camera_right(&cuda_kernel_data_right[0], (cyclesphi_data *)&cdata);
  }

#else

#  if !defined(WITH_CLIENT_MPI) && !defined(WITH_CLIENT_MPI_SOCKET) && !defined(WITH_CLIENT_SOCKET)
  //////////////////////////////////////////
  if (cuda_kernel_data_res[0] != -1 && cuda_kernel_data_res[1] != -1) {

    ccl::KernelData *kdata = (ccl::KernelData *)data;

    //#  if defined(WITH_CLIENT_RENDERENGINE_VR) || \
//      (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
    //    if (right)
    //      kdata = (ccl::KernelData *)&cuda_kernel_data_right[0];
    //#endif

    // wh,h,dx,dy
    // int *size = (int *)cdata;

    float *height = &kdata->cam.height;
    float *width = &kdata->cam.width;
    float *dx = &kdata->cam.dx[0];
    float *dy = &kdata->cam.dy[0];
    float *rastertocamera = &kdata->cam.rastertocamera.x[0];

    float old_new_height = height[0] / cuda_kernel_data_res[1];
    float old_new_width = width[0] / cuda_kernel_data_res[0];

    height[0] = cuda_kernel_data_res[1];
    width[0] = cuda_kernel_data_res[0];
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
  }
#  endif
//////////////////////////////////////////
#endif
}

void const_copy_to_internal(ccl::CUDAContextScope &scope,
                            const char *name,
                            void *host,
                            size_t size)
{
  scope.get().const_copy_to(name, host, size);
  //#if 0
  //  CUdeviceptr cumem;
  //  size_t cubytes;
  //  uint64_t ptr;
  //
  //  cu_assert(cuModuleGetGlobal(&cumem, &cubytes, scope.get().cuModule, "kernel_params"));
  //  assert(cubytes == sizeof(ccl::KernelParamsCUDA));
  //
  //  ccl::KernelParamsCUDA *kpc = &scope.get().cuda_kernel_map[g_current_frame];
  //  cuda_assert2(cudaMemcpy((void *)kpc, (void *)cumem, cubytes, cudaMemcpyDeviceToHost));
  //
  //  ptr = (uint64_t)scope.get().cuda_mem_map[map_id].mem.device_pointer;
  //
  //  /* Update data storage pointers in launch parameters. */
  //#  define KERNEL_DATA_ARRAY(data_type, data_name) \
//    if (strcmp(name, #data_name) == 0) { \
//      void **temp_ptr = ((void **)&kpc.data_name); \
//      *temp_ptr = (void *)ptr; \
//    }
  //  KERNEL_DATA_ARRAY(ccl::KernelData, data)
  //  KERNEL_DATA_ARRAY(ccl::IntegratorStateGPU, integrator_state)
  //#  include "kernel/data_arrays.h"
  //#  undef KERNEL_DATA_ARRAY
  //
  //  cuda_assert2(cudaMemcpy((void *)cumem, (void *)kpc, cubytes, cudaMemcpyHostToDevice));
  //#else
  //  CUdeviceptr mem;
  //  size_t bytes;
  //
  //  cu_assert(cuModuleGetGlobal(&mem, &bytes, scope.get().cuModule, "kernel_params"));
  //  assert(bytes == sizeof(ccl::KernelParamsCUDA));
  //
  //  /* Update data storage pointers in launch parameters. */
  //#  define KERNEL_DATA_ARRAY(data_type, data_name) \
//    if (strcmp(name, #data_name) == 0) { \
//      cu_assert(cuMemcpyHtoD(mem + offsetof(ccl::KernelParamsCUDA, data_name), host, size)); \
//    }
  //  KERNEL_DATA_ARRAY(ccl::KernelData, data)
  //  KERNEL_DATA_ARRAY(ccl::IntegratorStateGPU, integrator_state)
  //#  include "kernel/data_arrays.h"
  //#  undef KERNEL_DATA_ARRAY
  //
  //#endif
  //
  //#ifdef WITH_CUDA_STAT
  //
  //  // cuda_assert(cuModuleGetGlobal(&mem, &bytes, cuModuleStat, name));
  //  // cuda_assert(cuMemcpyHtoD(mem, host, size));
  //
  //  cuda_assert(cuModuleGetGlobal(&mem, &bytes, cuModuleStat, "kernel_params"));
  //  assert(bytes == sizeof(KernelParamsCUDA));
  //
  //  /* Update data storage pointers in launch parameters. */
  //#  define KERNEL_DATA_ARRAY(data_type, data_name) \
//    if (strcmp(name, #data_name) == 0) { \
//      cuda_assert(cuMemcpyHtoD(mem + offsetof(KernelParamsCUDA, data_name), host, size)); \
//    }
  //  KERNEL_DATA_ARRAY(KernelData, data)
  //  KERNEL_DATA_ARRAY(IntegratorStateGPU, integrator_state)
  //#  include "kernel/data_arrays.h"
  //#  undef KERNEL_DATA_ARRAY
  //
  //  // KernelParamsCUDA kpc;
  //  // cuda_assert(cuMemcpyDtoH((void *)&kpc, mem, bytes));
  //
  //#endif
}

void caching_kpc(int dev, ccl::CUDAContextScope &scope, bool clear_cache)
{
  if (g_caching_enabled == CACHING_DISABLED)
    return;

  CUdeviceptr cumem;
  size_t cubytes;
  uint64_t ptr;

  cu_assert(cuModuleGetGlobal(&cumem, &cubytes, scope.get().cuModule, "kernel_params"));
  assert(cubytes == sizeof(ccl::KernelParamsCUDA));

  if (g_caching_enabled == CACHING_RECORD) {
    ccl::KernelParamsCUDA *kpc = &cuda_kernel_map[dev][g_current_frame];
    cuda_assert2(cudaMemcpy((void *)kpc, (void *)cumem, cubytes, cudaMemcpyDeviceToHost));
  }

  else if (clear_cache && g_caching_enabled == CACHING_PREVIEW) {
    if (cuda_kernel_map[dev].find(g_current_frame_preview) != cuda_kernel_map[dev].end()) {
      ccl::KernelParamsCUDA *kpc = &cuda_kernel_map[dev][g_current_frame_preview];
      cuda_assert2(cudaMemcpy((void *)cumem, (void *)kpc, cubytes, cudaMemcpyHostToDevice));
    }
  }
}

void const_copy(
    int numDevice, DEVICE_PTR kg_bin, const char *name, char *host_bin, size_t size, bool save)
{
  // printf("omp_const_copy: %s, %.3f, %zu\n", name, (float)size / (1024.0f * 1024.0f));
  // if (strcmp(name, "data") == 0) {
  //  cam_recalc(host_bin);
  //}

  for (int id = 0; id < ccl::cuda_devices.size(); id++) {
    ccl::CUDAContextScope scope(id);

    const_copy_to_internal(scope, name, host_bin, size);
  }

  if (strcmp(name, "data") == 0) {
    if (save) {
      cuda_kernel_data.resize(size);
      memcpy(&cuda_kernel_data[0], host_bin, size);

#if defined(WITH_CLIENT_RENDERENGINE_VR) || \
    (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
      cuda_kernel_data_right.resize(size);
      memcpy(&cuda_kernel_data_right[0], host_bin, size);
#endif

      cam_recalc(host_bin);
    }
  }
}

void tex_copy(int numDevice,
              DEVICE_PTR kg_bin,
              char *name_bin,
              DEVICE_PTR map_id,
              char *mem,
              size_t data_count,
              size_t mem_size)
{
  double t = omp_get_wtime();
  //    if (name_bin == NULL || mem == NULL)
  //        return;

  size_t nameSize = sizeof(char) * (strlen(name_bin) + 1);
  char *name = (char *)name_bin;

  // printf("omp_tex_copy: %s, %.3f [MB], %lld\n",
  //       name,
  //       (float)mem_size / (1024.0f * 1024.0f),
  //       data_count);

#ifndef WITH_CUDA_STATv2

#  if defined(WITH_CLIENT_CUDA_CPU_STAT2)
  if (!check_mem_advise_name(name, false))
#  endif
    generic_copy_to(mem, map_id, mem_size);

#endif

#if 0
  set_unimem_flag(map_id);
#endif

  for (int id = 0; id < ccl::cuda_devices.size(); id++) {
    ccl::CUDAContextScope scope(id);

    // CUdeviceptr cumem;
    // size_t cubytes;
    // uint64_t ptr;

    // cu_assert(cuModuleGetGlobal(&cumem, &cubytes, scope.get().cuModule, name));
    // ptr = (uint64_t)scope.get().cuda_mem_map[map_id].mem.device_pointer;
    // cuda_assert2(cudaMemcpy((char *)cumem, (void *)&ptr, cubytes, cudaMemcpyHostToDevice));

    CUdeviceptr cumem;
    size_t cubytes;
    uint64_t ptr;

    cu_assert(cuModuleGetGlobal(&cumem, &cubytes, scope.get().cuModule, "kernel_params"));
    assert(cubytes == sizeof(ccl::KernelParamsCUDA));

    ccl::KernelParamsCUDA kpc;  //  = &cuda_kernel_map[id][g_current_frame];
    cuda_assert2(cudaMemcpy((void *)&kpc, (void *)cumem, cubytes, cudaMemcpyDeviceToHost));

    ptr = (uint64_t)scope.get().cuda_mem_map[map_id].mem.device_pointer;

    /* Update data storage pointers in launch parameters. */
#define KERNEL_DATA_ARRAY(data_type, data_name) \
  if (strcmp(name, #data_name) == 0) { \
    void **temp_ptr = ((void **)&kpc.data_name); \
    *temp_ptr = (void *)ptr; \
  }
#include "kernel/data_arrays.h"
#undef KERNEL_DATA_ARRAY

    cuda_assert2(cudaMemcpy((void *)cumem, (void *)&kpc, cubytes, cudaMemcpyHostToDevice));

#ifdef WITH_CUDA_STAT
    // cu_assert(cuModuleGetGlobal(&cumem, &cubytes, scope.get().cuModuleStat, name));
    // ptr = (uint64_t)scope.get().cuda_mem_map[map_id].mem.device_pointer;
    // cuda_assert2(cudaMemcpy((char *)cumem, (void *)&ptr, cubytes, cudaMemcpyHostToDevice));

    cu_assert(cuModuleGetGlobal(&cumem, &cubytes, scope.get().cuModuleStat, "kernel_params"));
    assert(cubytes == sizeof(ccl::KernelParamsCUDA));

    // ccl::KernelParamsCUDA kpc;
    cuda_assert2(cudaMemcpy((void *)&kpc, (void *)cumem, cubytes, cudaMemcpyDeviceToHost));

    ptr = (uint64_t)scope.get().cuda_mem_map[map_id].mem.device_pointer;

    /* Update data storage pointers in launch parameters. */
#  define KERNEL_DATA_ARRAY(data_type, data_name) \
    if (strcmp(name, #data_name) == 0) { \
      void **temp_ptr = ((void **)&kpc.data_name); \
      *temp_ptr = (void *)ptr; \
    }
#  include "kernel/data_arrays.h"
#  undef KERNEL_DATA_ARRAY

    cuda_assert2(cudaMemcpy((void *)cumem, (void *)&kpc, cubytes, cudaMemcpyHostToDevice));

    if (scope.get().cuda_mem_map[map_id].host_pointer != NULL) {
      memcpy(&scope.get().cuda_mem_map[map_id].host_pointer[0], mem, mem_size);
    }
#endif

    if (!strcmp(name, "texture_info")) {
      scope.get().texture_info_dmem = map_id;
      scope.get().texture_info_mem.resize(mem_size);
      memcpy(&scope.get().texture_info_mem[0], mem, mem_size);
      // scope.get().texture_info_dmem_size = mem_size;
      if (id == 0) {
        texture_info_mem.resize(mem_size);
        memcpy(&texture_info_mem[0], mem, mem_size);
      }
    }

    cuda_tex_copy_stat(scope, data_count, mem_size, name, map_id, NULL);

    scope.get().cuda_mem_map[map_id].mem.data_size = data_count;
  }

  printf("cuda_tex_copy: TIME: %f\n", omp_get_wtime() - t);
}

void load_textures_internal(ccl::CUDAContextScope &scope,
                            // std::map<DEVICE_PTR, DEVICE_PTR> &ptr_map,
                            bool use_stat)
{
  size_t texture_info_size1 = scope.get().texture_info_mem.size() / sizeof(ccl::TextureInfo);
  for (int i = 0; i < texture_info_size1; i++) {
    ccl::TextureInfo &info = ((ccl::TextureInfo *)&scope.get().texture_info_mem[0])[i];
    // printf("client: omp_load_textures, remote tex: %d\n", info.node_id);

    if (info.data /* && ptr_map.find(info.data) != ptr_map.end()*/) {
      info.data = get_ptr_map(info.data);
      if (info.data &&
          scope.get().cuda_mem_map.find(info.data) != scope.get().cuda_mem_map.end()) {
        int depth = (info.depth > 0) ? info.depth : 1;
        scope.get().cuda_mem_map[info.data].mem.data_size = info.width * info.height * depth;

        if (use_stat) {
          cuda_tex_copy_stat(scope,
                             info.width * info.height *
                                 depth,  // scope.get().cuda_mem_map[info.data].image_data_count
                             scope.get().cuda_mem_map[info.data].mem.device_size,
                             scope.get().cuda_mem_map[info.data].mem.name.c_str(),
                             info.data,
                             (char *)&info);
        }

#ifdef WITH_CUDA_CPUIMAGE
        info.data = scope.get().cuda_mem_map[info.data].mem.device_pointer;
#else
        if (info.data_type != 8 && info.data_type != 9) {
          info.data = scope.get().cuda_mem_map[info.data].texobject;
        }
        else {
          info.data = scope.get().cuda_mem_map[info.data].mem.device_pointer;
        }
#endif
      }
    }
  }

  //#  if !defined(WITH_CLIENT_CUDA_CPU_STAT2)
  // always on each GPU
  cuda_assert2(cudaMemcpy(
      (CU_DEVICE_PTR)scope.get().cuda_mem_map[scope.get().texture_info_dmem].mem.device_pointer,
      &scope.get().texture_info_mem[0],
      scope.get().texture_info_mem.size(),
      cudaMemcpyHostToDevice));

  //#  endif
}

void blender_camera(DEVICE_PTR mem, char *temp_data, size_t mem_size)
{
  g_blender_camera = temp_data;
}

void load_textures(int numDevice, DEVICE_PTR kg_bin, size_t texture_info_size
                   /* std::map<DEVICE_PTR, DEVICE_PTR> &ptr_map*/)
{
  for (int id = 0; id < ccl::cuda_devices.size(); id++) {
    ccl::CUDAContextScope scope(id);

    if (scope.get().texture_info_mem.size() == 0)
      break;

    load_textures_internal(scope, true);

    if (scope.get().cuda_mem_map[scope.get().texture_info_dmem].uni_mem)
      break;
  }
}

//#ifdef WITH_OPTIX_DENOISER
//#  define WITH_OPTIX_DENOISER2
//#endif

double cuda_previousTime[3] = {0, 0, 0};
int cuda_frameCount[3] = {0, 0, 0};
bool displayFPS(int type = 0)
{
  double currentTime = omp_get_wtime();
  cuda_frameCount[type]++;

  if (currentTime - cuda_previousTime[type] >= 1.0) {

#pragma omp critical
    printf("Rendering: FPS: %.2f \n",
           // type,
           (double)cuda_frameCount[type] / (currentTime - cuda_previousTime[type]));
    cuda_frameCount[type] = 0;
    cuda_previousTime[type] = omp_get_wtime();
    return true;
  }

  return false;
}

void init_execution(int has_shadow_catcher_,
                    int max_shaders_,
                    int pass_stride_,
                    unsigned int kernel_features_,
                    unsigned int volume_stack_size_,
                    bool init)
{
  if (g_pathTraceWorkGPU == NULL) {
    //////////////
    const int max_srgb = 1001;
    float to_srgb[max_srgb];
    for (int i = 0; i < max_srgb; i++) {
      to_srgb[i] = ccl::color_linear_to_srgb((float)i / (float)(max_srgb - 1));
    }
    //////////////
    CUdeviceptr cumem;
    size_t cubytes;
    uint64_t ptr;

    g_pathTraceWorkGPU = new ccl::PathTraceWorkGPU *[ccl::cuda_devices.size()];
    for (int id = 0; id < ccl::cuda_devices.size(); id++) {
      ccl::CUDAContextScope scope(id);
      // const_copy_to_internal(scope, "linear_to_srgb_table", to_srgb, sizeof(float) * max_srgb);
      cu_assert(cuModuleGetGlobal(&cumem, &cubytes, scope.get().cuModule, "linear_to_srgb_table"));
      cuda_assert2(cudaMemcpy((void *)cumem, (void *)to_srgb, cubytes, cudaMemcpyHostToDevice));

      g_pathTraceWorkGPU[id] = new ccl::PathTraceWorkGPU(
          &scope.get(), &g_film, &g_device_scene, &g_cancel_requested_flag);

      // g_pathTraceWorkGPU[id] = ccl::PathTraceWork::create(
      //  &scope.get(), &g_film, &g_device_scene, &g_cancel_requested_flag);
    }
  }

  // cuda_pathTraceWorkGPU->init_execution(
  //    has_shadow_catcher, num_shaders, pass_stride, kernel_features);

  // g_device_scene.data.max_work_tiles = 2048;
  g_device_scene.data.integrator.has_shadow_catcher = has_shadow_catcher_;
  g_device_scene.data.max_shaders = max_shaders_;
  g_device_scene.data.film.pass_stride = pass_stride_;
  g_device_scene.data.film.exposure = 1.0f;
  g_device_scene.data.kernel_features = kernel_features_;
  g_device_scene.data.volume_stack_size = volume_stack_size_;

  if (init) {
    for (int id = 0; id < ccl::cuda_devices.size(); id++) {
      ccl::CUDAContextScope scope(id);
      g_pathTraceWorkGPU[id]->alloc_work_memory();
      g_pathTraceWorkGPU[id]->init_execution();

      scope.get().path_stat_is_done = 0;
    }
  }
}

ccl::PathTraceWorkGPU *g_buffer_to_pixels_worker = NULL;

void buffer_to_pixels(int numDevice,
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
                      int w)
{
  int id = numDevice;
  ccl::CUDAContextScope scope(id);
  if (g_buffer_to_pixels_worker == NULL) {
    g_buffer_to_pixels_worker = new ccl::PathTraceWorkGPU(
        &scope.get(),
        &g_film,
        &g_device_scene,
        &g_cancel_requested_flag);  // = g_pathTraceWorkGPU[id];

    g_buffer_to_pixels_worker->alloc_work_memory();
    g_buffer_to_pixels_worker->init_execution();
  }

  ccl::KernelWorkTile wtile;  // = scope.get().wtile;
  wtile.start_sample = start_sample;
  wtile.num_samples = end_sample - start_sample;

  if (!scope.get().kernels.is_loaded()) {
    // scope.get().kernels.load(&scope.get());
    return;
  }

  wtile.x = tile_x;
  wtile.w = tile_w;
  wtile.offset = offset;
  wtile.stride = stride;
#if defined(WITH_CLIENT_RENDERENGINE_VR) || \
    (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
  wtile.stride *= 2;
#endif
  wtile.buffer = (float *)map_buffer_bin;
  wtile.pixel = (char *)map_pixel_bin;
  wtile.y = tile_y;
  wtile.h = tile_h;

  g_buffer_to_pixels_worker->set_buffer(wtile, g_buffer_passes, w, h);

  ccl::PassAccessor::Destination destination;
#ifdef WITH_CLIENT_GPUJPEG
  destination.d_pixels_uchar_yuv = (ccl::device_ptr)wtile.pixel;
  destination.offset = wtile.offset + wtile.x + wtile.y * wtile.stride;
  destination.stride = wtile.stride;
  destination.num_components = 4;
#else
  // destination.d_pixels_uchar_rgba = (ccl::device_ptr)wtile.pixel;
  destination.d_pixels_half_rgba = (ccl::device_ptr)wtile.pixel;
  destination.offset = wtile.offset + wtile.x + wtile.y * wtile.stride;
  destination.stride = wtile.stride;
  destination.num_components = 4;
#endif
  destination.pixel_stride = g_buffer_to_pixels_worker->effective_buffer_params_.pass_stride;
  g_buffer_to_pixels_worker->get_render_tile_film_pixels(
      destination, ccl::PassMode::DENOISED, wtile.start_sample + wtile.num_samples);
}

float g_report_time = 0;
size_t g_report_time_count = 0;
// int g_used_buffer = 0;

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
  //#pragma omp master
  {
    cuda_kernel_data_res[0] = w;
    cuda_kernel_data_res[1] = h;
  }
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
  int *pix_state = (int *)signal_value;

  //#ifdef WITH_CUDA_STAT
  //  int WITH_CUDA_STAT_LOOP = 0;
  //#endif

  int devices_size = nprocs_cpu;

  int id = 0;
  // omp_get_thread_num();
  // printf("start: %d\n", id);
  ccl::CUDAContextScope scope(id);

  // DEVICE_PTR dev_pixels_node = (DEVICE_PTR)dev_pixels_node2;

  // char *pixels_node = NULL;
  // cudaStream_t stream_path;
  // cudaStream_t stream_memcpy;

  // cudaEvent_t event_path_start;
  // cudaEvent_t event_path_stop;

  // cudaEvent_t event_memcpy_start;
  // cudaEvent_t event_memcpy_stop;

  // float *time_path = NULL;
  // float *time_memcpy = NULL;
  int stream_memcpy_id = 0;

  if (/*(pix_state[0] == 0*/ scope.get().used_buffer == 0) {

    // dev_pixels_node = (DEVICE_PTR)dev_pixels_node1;

    // pixels_node = pixels_node1;
    // stream_path = scope.get().stream[STREAM_PATH1];
    // stream_memcpy = scope.get().stream[STREAM_PATH1_MEMCPY];
    stream_memcpy_id = STREAM_PATH1_MEMCPY;
    // // host_fn_memcpy_status_flag = memcpy_status_flag1;

    // event_path_start = scope.get().event[STREAM_PATH1];
    // event_path_stop = scope.get().event[STREAM_PATH1 + STREAM_COUNT];

    // event_memcpy_start = scope.get().event[STREAM_PATH1_MEMCPY];
    // event_memcpy_stop = scope.get().event[STREAM_PATH1_MEMCPY + STREAM_COUNT];

    // time_path = &scope.get().running_time[STREAM_PATH1];
    // time_memcpy = &scope.get().running_time[STREAM_PATH1_MEMCPY];
  }
  else if (/*pix_state[1] == 0*/ scope.get().used_buffer == 1) {

    // dev_pixels_node = (DEVICE_PTR)dev_pixels_node2;

    // pixels_node = pixels_node2;
    // stream_path = scope.get().stream[STREAM_PATH2];
    // stream_memcpy = scope.get().stream[STREAM_PATH2_MEMCPY];
    stream_memcpy_id = STREAM_PATH2_MEMCPY;
    // // host_fn_memcpy_status_flag = memcpy_status_flag2;

    // event_path_start = scope.get().event[STREAM_PATH2];
    // event_path_stop = scope.get().event[STREAM_PATH2 + STREAM_COUNT];

    // event_memcpy_start = scope.get().event[STREAM_PATH2_MEMCPY];
    // event_memcpy_stop = scope.get().event[STREAM_PATH2_MEMCPY + STREAM_COUNT];

    // time_path = &scope.get().running_time[STREAM_PATH2];
    // time_memcpy = &scope.get().running_time[STREAM_PATH2_MEMCPY];
  }

  ///////////////////////////////////////////////////////////////////////
  // if (id == 0)
  //#pragma omp master
  {
    // cuda_displayFPS(0);
    // g_report_time += (omp_get_wtime() - t_id);
    g_report_time_count++;

#if defined(ENABLE_LOAD_BALANCE) || defined(ENABLE_LOAD_BALANCEv2) || \
    !defined(WITH_CLIENT_PATHTRACER2)
    if (scope.get().path_stat_is_done == 0 || omp_get_wtime() - g_report_time >= 3.0) {
#else
    if (scope.get().path_stat_is_done > 0) {
#endif
      double max_time = 0;
      double min_time = DBL_MAX;
      double avg_time = 0;
      for (int id1 = 0; id1 < devices_size; id1++) {

        if (scope.get().used_buffer == 0) {
          if (max_time < ccl::cuda_devices[id1].running_time[STREAM_PATH1]) {
            max_time = ccl::cuda_devices[id1].running_time[STREAM_PATH1];
          }
          if (min_time > ccl::cuda_devices[id1].running_time[STREAM_PATH1]) {
            min_time = ccl::cuda_devices[id1].running_time[STREAM_PATH1];
          }
          avg_time += ccl::cuda_devices[id1].running_time[STREAM_PATH1];
#if 1
          printf("%d: Time = %f [ms], %d-%d, %d\n",
                 id1,
                 ccl::cuda_devices[id1].running_time[STREAM_PATH1],
                 ccl::cuda_devices[id1].wtile.y,
                 ccl::cuda_devices[id1].wtile.h,
                 ccl::cuda_devices[id1].start_sample + ccl::cuda_devices[id1].num_samples);
#endif
        }
        else {
          if (max_time < ccl::cuda_devices[id1].running_time[STREAM_PATH2]) {
            max_time = ccl::cuda_devices[id1].running_time[STREAM_PATH2];
          }
          if (min_time > ccl::cuda_devices[id1].running_time[STREAM_PATH2]) {
            min_time = ccl::cuda_devices[id1].running_time[STREAM_PATH2];
          }
          avg_time += ccl::cuda_devices[id1].running_time[STREAM_PATH2];
#if 1
          printf("%d: Time = %f [ms], %d-%d, %d\n",
                 id1,
                 ccl::cuda_devices[id1].running_time[STREAM_PATH2],
                 ccl::cuda_devices[id1].wtile.y,
                 ccl::cuda_devices[id1].wtile.h,
                 ccl::cuda_devices[id1].start_sample + ccl::cuda_devices[id1].num_samples);
#endif
        }
      }

      // printf(
      //     "Rendering: FPS: %.2f, MaxTime = %f [ms], AvgTime = %f [ms], Samples: %d, buf: %d\n",
      //     1000.0f / max_time,
      //     max_time,
      //     avg_time / (double)devices_size,
      //     end_sample,
      //     (dev_pixels_node == (DEVICE_PTR)dev_pixels_node1) ? 0 : 1);

#if 0
          printf("OMP: FPS: %.2f: Time = %f [ms], %f [ms]\n",
                 1.0f / (omp_get_wtime() - t_id),
                 (omp_get_wtime() - t_id) * 1000.0f, (t_id2 - t_id) * 1000.0f);
#endif
      if (scope.get().path_stat_is_done == 0 || omp_get_wtime() - g_report_time >= 1.0) {
        printf(
            "Rendering: FPS: %.2f, MaxTime = %f [ms], AvgTime = %f [ms], Samples: %d, Tot "
            "Samples: %d\n",
            1000.0f / max_time,
            max_time,
            avg_time / (double)devices_size,
            scope.get().num_samples,
            scope.get().start_sample + scope.get().num_samples);

        // printf("OMP: FPS: %.2f: Time = %f [ms]\n",
        //        1.0f / (omp_get_wtime() - t_id),
        //        (omp_get_wtime() - t_id) * 1000.0f);

        // fflush(0);

        g_report_time = omp_get_wtime();
        g_report_time_count = 0;
      }

#ifdef ENABLE_LOAD_BALANCE_EXIT
      // 1.0% err
      if (max_time - min_time < max_time * 1.0 / 100.0) {
        pix_state[4] = 1;
      }
#endif
    }
    if (pix_state != NULL) {
      pix_state[3] = scope.get().start_sample + scope.get().num_samples;
    }

    // pix_state[3] = end_sample;
    //#  pragma omp flush

    ///////////////////////SAMPLES////////////////////////
    //#if defined(ENABLE_INC_SAMPLES) && !defined(ENABLE_STEP_SAMPLES)
    //        int num_samples = end_sample - start_sample;
    //        if (scope.get().path_stat_is_done != 0) {
    //          start_sample = start_sample + num_samples;
    //          end_sample = start_sample + num_samples;
    //        }
    //#endif

    ///////////////////////LB////////////////////////
#if defined(ENABLE_LOAD_BALANCE) || defined(ENABLE_LOAD_BALANCEv2)
    if (scope.get().path_stat_is_done > 0) {
#  ifdef ENABLE_LOAD_BALANCEv3
      int sp = 0;
      int id1_start = 0;
      int id1_step = 1;
      float coef = devices_size;
      if (scope.get().used_buffer == 0) {
        sp = STREAM_PATH1;
        id1_start = 0;
      }
      else {
        sp = STREAM_PATH2;
        id1_start = 0;
      }
      for (int id1 = id1_start; id1 < devices_size - 1; id1 += id1_step) {
#    if defined(WITH_CLIENT_RENDERENGINE_VR) || \
        (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))

        int devices_left_eye = devices_size / 2;
        if (devices_left_eye < 1)
          devices_left_eye = 1;

        if (id1 == devices_left_eye - 1)
          continue;
#    endif

        float time1 = ccl::cuda_devices[id1].running_time[sp];
        float time2 = ccl::cuda_devices[id1 + 1].running_time[sp];

        if (time1 < time2) {
          int pix_transm = (int)ceil((float)ccl::cuda_devices[id + 1].wtile.h *
                                     ((((float)time2 - (float)time1) / coef) / time2));
          if (ccl::cuda_devices[id1 + 1].wtile.h > pix_transm && pix_transm > 0) {
            ccl::cuda_devices[id1].wtile.h += pix_transm;
            ccl::cuda_devices[id1 + 1].wtile.y += pix_transm;
            ccl::cuda_devices[id1 + 1].wtile.h -= pix_transm;

            // if (scope.get().used_buffer == 0) {
            //   ccl::cuda_devices[id1 + 1].running_time[STREAM_PATH1] -= time2 *
            //   ((((float)time2 - (float)time1) / 2.0f) / time2);
            // }
            // else {
            ccl::cuda_devices[id1].running_time[sp] += time2 *
                                                       ((((float)time2 - (float)time1) / coef) /
                                                        time2);
            ccl::cuda_devices[id1 + 1].running_time[sp] -=
                time2 * ((((float)time2 - (float)time1) / coef) / time2);
            // } //  //
          }
        }
        else if (time1 > time2) {
          int pix_transm = (int)ceil((float)ccl::cuda_devices[id].wtile.h *
                                     ((((float)time1 - (float)time2) / coef) / time1));
          if (ccl::cuda_devices[id1].wtile.h > pix_transm && pix_transm > 0) {
            ccl::cuda_devices[id1].wtile.h -= pix_transm;
            ccl::cuda_devices[id1 + 1].wtile.y -= pix_transm;
            ccl::cuda_devices[id1 + 1].wtile.h += pix_transm;

            // if (scope.get().used_buffer == 0) {
            //   ccl::cuda_devices[id1 + 1].running_time[STREAM_PATH1] += time1 *
            //   ((((float)time1 - (float)time2) / 2.0f) / time1);
            // }
            // else {
            ccl::cuda_devices[id1].running_time[sp] -= time1 *
                                                       ((((float)time1 - (float)time2) / coef) /
                                                        time1);
            ccl::cuda_devices[id1 + 1].running_time[sp] +=
                time1 * ((((float)time1 - (float)time2) / coef) / time1);
            // }////
          }
        }
      }
#  else

#    if 1
      for (int id1 = 0; id1 < devices_size - 1; id1++) {
#      if defined(WITH_CLIENT_RENDERENGINE_VR) || \
          (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
        if (id1 == devices_left_eye - 1)
          continue;
#      endif
        float time1 = 0;
        float time2 = 0;
        if (scope.get().used_buffer == 0) {
          time1 = ccl::cuda_devices[id1].running_time[STREAM_PATH1];
          time2 = ccl::cuda_devices[id1 + 1].running_time[STREAM_PATH1];
        }
        else {
          time1 = ccl::cuda_devices[id1].running_time[STREAM_PATH2];
          time2 = ccl::cuda_devices[id1 + 1].running_time[STREAM_PATH2];
        }

        if (time1 < time2 && ccl::cuda_devices[id1 + 1].wtile.h > 2) {
          ccl::cuda_devices[id1].wtile.h++;
          ccl::cuda_devices[id1 + 1].wtile.y++;
          ccl::cuda_devices[id1 + 1].wtile.h--;
        }
        else if (time1 > time2 && ccl::cuda_devices[id1].wtile.h > 2) {
          ccl::cuda_devices[id1].wtile.h--;
          ccl::cuda_devices[id1 + 1].wtile.y--;
          ccl::cuda_devices[id1 + 1].wtile.h++;
        }
      }
#    else
      ////////////////////////////////////////////////////
      {
        double max_time = 0;
        double min_time = DBL_MAX;
        int max_h = 0;
        if (scope.get().used_buffer == 0) {
          for (int id1 = 0; id1 < devices_size; id1++) {
            if (max_time < ccl::cuda_devices[id1].running_time[STREAM_PATH1]) {
              max_time = ccl::cuda_devices[id1].running_time[STREAM_PATH1];
            }
            if (min_time > ccl::cuda_devices[id1].running_time[STREAM_PATH1]) {
              min_time = ccl::cuda_devices[id1].running_time[STREAM_PATH1];
            }
            if (max_h <
                abs((int)ccl::cuda_devices[id1].wtile_h - (int)ccl::cuda_devices[id1].wtile.h)) {
              max_h = abs((int)ccl::cuda_devices[id1].wtile_h -
                          (int)ccl::cuda_devices[id1].wtile.h);
            }
          }
        }
        else {
          for (int id1 = 0; id1 < devices_size; id1++) {
            if (max_time < ccl::cuda_devices[id1].running_time[STREAM_PATH2]) {
              max_time = ccl::cuda_devices[id1].running_time[STREAM_PATH2];
            }
            if (min_time > ccl::cuda_devices[id1].running_time[STREAM_PATH2]) {
              min_time = ccl::cuda_devices[id1].running_time[STREAM_PATH2];
            }
            if (max_h <
                abs((int)ccl::cuda_devices[id1].wtile_h - (int)ccl::cuda_devices[id1].wtile.h)) {
              max_h = abs((int)ccl::cuda_devices[id1].wtile_h -
                          (int)ccl::cuda_devices[id1].wtile.h);
            }
          }
        }

        // printf("min/max: %f, %f, %d\n", min_time, max_time, max_h);

        if (max_h > 2) {
          double total_time = 0;
          double avg_time = 0;
          for (int id1 = 0; id1 < devices_size; id1++) {
            ccl::cuda_devices[id1].wtile_h = ccl::cuda_devices[id1].wtile.h;

            if (scope.get().used_buffer == 0)
              total_time += ccl::cuda_devices[id1].running_time[STREAM_PATH1];
            else
              total_time += ccl::cuda_devices[id1].running_time[STREAM_PATH2];
          }
          avg_time = total_time / (double)devices_size;
          // printf("avg_time: %f\n", avg_time);

#      ifndef ENABLE_LOAD_BALANCEv2
          int _tile_h = tile_h;
#      else
          int _tile_h = tile_w * tile_h;
#      endif
          int total_h = 0;

          for (int id1 = 0; id1 < devices_size; id1++) {
            float time1 = ccl::cuda_devices[id1].running_time[STREAM_PATH1];
            if (dev_pixels_node != (DEVICE_PTR)dev_pixels_node1)
              time1 = ccl::cuda_devices[id1].running_time[STREAM_PATH2];

            // printf("time1: %f\n", time1);
            ccl::cuda_devices[id1].wtile.h = (int)ceil((double)ccl::cuda_devices[id1].wtile.h *
                                                       (double)avg_time / (double)time1);
            total_h += ccl::cuda_devices[id1].wtile.h;
          }
          // for (int id1 = 0; id1 < devices_size; id1++) {
          //   printf("LB1: %d, %d\n", ccl::cuda_devices[id1].wtile.y,
          //   ccl::cuda_devices[id1].wtile.h);
          // }
          // printf("total_h: %d, %d\n", total_h, _tile_h);

          if (total_h > _tile_h) {
            int c = (int)ceil((double)(total_h - _tile_h) / (double)devices_size);
            // printf("c: %d\n", c);
            for (int id1 = 0; id1 < devices_size; id1++) {
              // if (ccl::cuda_devices[id1].wtile.h > c) {
              ccl::cuda_devices[id1].wtile.h = ccl::cuda_devices[id1].wtile.h - c;
              //} else {
              //  ccl::cuda_devices[id1].wtile.h = 1;
              //}
            }
          }
          // for (int id1 = 0; id1 < devices_size; id1++) {
          //   printf("LB2: %d, %d\n", ccl::cuda_devices[id1].wtile.y,
          //   ccl::cuda_devices[id1].wtile.h);
          // }
          int total_y = tile_y;
          ccl::cuda_devices[0].wtile.y = total_y;
          for (int id1 = 1; id1 < devices_size; id1++) {
            total_y += ccl::cuda_devices[id1 - 1].wtile.h;
            ccl::cuda_devices[id1].wtile.y = total_y;
          }
          ccl::cuda_devices[devices_size - 1].wtile.h =
              _tile_h - ccl::cuda_devices[devices_size - 1].wtile.y;

          // for (int id1 = 0; id1 < devices_size; id1++) {
          //   printf("LB3: %d, %d\n", ccl::cuda_devices[id1].wtile.y,
          //   ccl::cuda_devices[id1].wtile.h);
          // }
        }
      }
      ////////////////////////////////////////////////////

#    endif

#  endif
    }
#endif

#ifdef WITH_CUDA_STAT
    if (scope.get().path_stat_is_done == 1) {
      // set_mem_advise_by_stat2(devices_size, stream_memcpy_id);
      // set_mem_advise_by_stat3(devices_size, stream_memcpy_id);
      set_mem_advise_by_stat3_credits(devices_size, stream_memcpy_id, false);
      // cuda_print_stat_gpu(devices_size, stream_memcpy_id);
    }
#endif
#if defined(WITH_CLIENT_RENDERENGINE_EMULATE)
    if (pix_state != NULL && scope.get().path_stat_is_done > 0)
      pix_state[5] = 1;
#endif

    if (pix_state != NULL && scope.get().path_stat_is_done == 1 && pix_state[2] == 1)
      pix_state[2] = 0;
  }

#if defined(WITH_CUDA_STAT) && !defined(WITH_CUDA_STATv2)

  //#  pragma omp barrier

  // apply to global var
  if (scope.get().path_stat_is_done == 1) {
    double stat3_copy_time_start = omp_get_wtime();
    double total_size = 0;

    std::map<std::string, ccl::CUDADevice::CUDAMem>::iterator it_stat;
    std::vector<std::string> data_names;
    // size_t max_data_size = 0;

    if (texture_info_mem.size() > 0) {
      memcpy(&scope.get().texture_info_mem[0], &texture_info_mem[0], texture_info_mem.size());
      cuda_load_textures_internal(scope, dev_kernel_data->ptr_map, false);
    }

    for (it_stat = scope.get().cuda_stat_map.begin(); it_stat != scope.get().cuda_stat_map.end();
         it_stat++) {

      std::string data_name = it_stat->first;
      ccl::CUDADevice::CUDAMem &cm = it_stat->second;

      if (!check_mem_advise_name(data_name.c_str(), false))
        continue;

      if (data_name.find("tex_image") == std::string::npos) {
        CUdeviceptr cumem;
        size_t cubytes;
        uint64_t ptr;

        cu_assert(cuModuleGetGlobal(&cumem, &cubytes, scope.get().cuModule, data_name.c_str()));
        // printf("%d: cuModuleGetGlobal: %s, %lld\n", id, data_name.c_str(), cumem);
        ptr = (uint64_t)scope.get()
                  .cuda_mem_map[cm.device_pointers[CUDA_DEVICE_POINTER_MAP_ID]]
                  .device_pointer;
        cuda_assert2(cudaMemcpy((char *)cumem, (void *)&ptr, cubytes, cudaMemcpyHostToDevice));
      }

      if (id == 0 || !cm.uni_mem) {
        cuda_assert2(cudaMemcpy(
            (CU_DEVICE_PTR)scope.get()
                .cuda_mem_map[cm.device_pointers[CUDA_DEVICE_POINTER_MAP_ID]]
                .device_pointer,
            &scope.get()
                 .cuda_mem_map[cm.device_pointers[CUDA_DEVICE_POINTER_MAP_ID]]
                 .host_pointer[0],
            scope.get().cuda_mem_map[cm.device_pointers[CUDA_DEVICE_POINTER_MAP_ID]].size,
            cudaMemcpyHostToDevice));

        total_size +=
            scope.get().cuda_mem_map[cm.device_pointers[CUDA_DEVICE_POINTER_MAP_ID]].size;
      }
    }
    printf("V3: data copy from CPU to GPU (cudastat): %f [s], %f [GB], %f [GB/s]\n",
           omp_get_wtime() - stat3_copy_time_start,
           total_size,
           total_size / (omp_get_wtime() - stat3_copy_time_start));
  }
#endif

  ///////////////////////////////////////////////////////////////////
  //       if (pix_state == NULL) {
  //         //#if defined(WITH_CUDA_STAT)
  //         //        if (scope.get().path_stat_is_done > 1)
  //         //          break;
  //         //#endif

  // #if !defined(WITH_CUDA_STAT) && !defined(ENABLE_STEP_SAMPLES)
  //         // break;
  // #endif
  //       }
  //       else {
  //         ///////////////////////SAMPLES////////////////////////
  // #ifndef ENABLE_INC_SAMPLES
  //         cuda_assert2(cudaMemset((CU_DEVICE_PTR)dev_buffer, 0, dev_buffer_size));
  // #endif
  //       }
}

void path_trace_internal(int numDevice,
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

  //#ifdef WITH_CUDA_STAT
  //  int WITH_CUDA_STAT_LOOP = 0;
  //#endif

  // int start_sample2 = start_sample;
  // int end_sample2 = end_sample;

  int devices_size = nprocs_cpu;

#ifdef WITH_CLIENT_ULTRAGRID
  const char *env_libug_dev = getenv("LIBUG_DEV");
  if (env_libug_dev != NULL && devices_size > 1)
    devices_size = devices_size - 1;
#endif

#ifdef ENABLE_STEP_SAMPLES
  int debug_step_samples = 1;
  const char *env_step_samples = getenv("DEBUG_STEP_SAMPLES");
  if (env_step_samples != NULL)
    debug_step_samples = atoi(env_step_samples);
#endif

#ifdef UNITEST3_PENALTY
  devices_size = 1;  // rendering on only first device
#endif

#if 0  // def WITH_OPTIX_DENOISER
  size_t pix_type_size = SIZE_FLOAT4;
  size_t pix_size = w * h * pix_type_size;

  size_t pix_type_uchar_size = SIZE_UCHAR4;
  size_t pix_uchar_size = w * h * pix_type_uchar_size;
#else
  size_t pix_type_size = SIZE_UCHAR4;
  size_t pix_size = w * h * pix_type_size;
#endif

  int *pix_state = (int *)signal_value;

  // omp_lock_t *lock0 = NULL;
  // omp_lock_t *lock1 = NULL;

  // if (pix_state != NULL) {
  //  lock0 = (omp_lock_t *)&pix_state[6];
  //  lock1 = (omp_lock_t *)&pix_state[8];
  //}

  //#if defined(WITH_CLIENT_ULTRAGRID)
  //  //pixels_node2 = (char *)pixels_node2 + pix_size;
  //  stride *= 2;
  //#endif

#if defined(WITH_CLIENT_RENDERENGINE_VR) || \
    (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
  // pixels_node2 = (char *)pixels_node2 + pix_size;

  int devices_left_eye = devices_size / 2;
  if (devices_left_eye < 1)
    devices_left_eye = 1;

  // devices_left_eye = 0;

  int devices_right_eye = devices_size - devices_left_eye;
  if (devices_right_eye < 1)
    devices_right_eye = 1;
#endif

#if 0  // def WITH_OPTIX_DENOISER

  size_t pix_size_denoiser = w * h * sizeof(float) * 3;  // 1-pass, rgb

  //#  if defined(WITH_CLIENT_RENDERENGINE_VR) || \
//      (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
  //  pix_size_denoiser *= 2;
  //#  endif

#  ifdef WITH_CUDA_BUFFER_MANAGED
  char *pixels_node_denoiser = cuda_host_alloc(
      "pixels_node_denoiser", NULL, 4 * pix_size_denoiser, 1);
#  else
  char *pixels_node_denoiser = cuda_host_alloc(
      "pixels_node_denoiser", NULL, 4 * pix_size_denoiser, 2);
#  endif
#endif

  //#if defined(WITH_CLIENT_RENDERENGINE_EMULATE)
  //  const char *drt = getenv("DEBUG_REPEAT_TIME");
  //  double drt_time_start = omp_get_wtime();
  //#endif

  // float long_time_path = 0;
  // for schedule(static, 1)// reduction(max : long_time_path)
  // for (int id = 0; id < devices_size; id++)
  //#ifdef ENABLE_STEP_SAMPLES
  //#  pragma omp parallel num_threads(devices_size) private(start_sample, end_sample)
  //#else
  //#  pragma omp parallel num_threads(devices_size)
  //#endif
  {
    int id = omp_get_thread_num();
    // printf("start: %d\n", id);
    ccl::CUDAContextScope scope(id);

    ccl::PathTraceWorkGPU *cuda_pathTraceWorkGPU = g_pathTraceWorkGPU[id];

    //#ifdef WITH_CLIENT_ULTRAGRID_
    //    CU_DEVICE_PTR dev_pixels_node1;  // omp_mem_alloc(DEVICE_ID, "pixels1", NULL, pix_size);
    //    cuda_assert2(cudaHostGetDevicePointer(&dev_pixels_node1, pixels_node1, 0));
    //
    //    CU_DEVICE_PTR dev_pixels_node2;  // omp_mem_alloc(DEVICE_ID, "pixels2", NULL, pix_size);
    //    cuda_assert2(cudaHostGetDevicePointer(&dev_pixels_node2, pixels_node2, 0));
    //#else

    char *pixels_node1 = NULL;
    char *pixels_node2 = NULL;

    if (map_pixel_bin != NULL) {
      pixels_node1 = (char *)map_pixel_bin;

      if (pix_state != NULL)
        pixels_node2 = (char *)map_pixel_bin + pix_size;
      else
        pixels_node2 = pixels_node1;
    }

#ifdef WITH_CUDA_BUFFER_MANAGED
    CU_DEVICE_PTR dev_pixels_node1 = (CU_DEVICE_PTR)pixels_node1;
    CU_DEVICE_PTR dev_pixels_node2 = (CU_DEVICE_PTR)pixels_node2;
#else
    CU_DEVICE_PTR dev_pixels_node1 = NULL;  // omp_mem_alloc(DEVICE_ID, "pixels1", NULL, pix_size);
    CU_DEVICE_PTR dev_pixels_node2 = NULL;

    if (pixels_node1 != NULL) {
      if (pix_state != NULL) {
        // cuda_assert2(cudaHostGetDevicePointer(&dev_pixels_node1, pixels_node1, 0));
        dev_pixels_node1 = pixels_node1;
        dev_pixels_node2 = (CU_DEVICE_PTR)((char *)dev_pixels_node1 + pix_size);
      }
      else {
        dev_pixels_node1 =
            (CU_DEVICE_PTR)scope.get().cuda_mem_map[map_pixel_bin].mem.device_pointer;
        dev_pixels_node2 = dev_pixels_node1;
      }
    }
#endif

#if defined(WITH_CLIENT_RENDERENGINE_VR) || \
    (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
    pixels_node2 = (char *)pixels_node2 + pix_size;
    dev_pixels_node2 = (char *)dev_pixels_node2 + pix_size;
#endif

    char *den_ptr = NULL;

#if 0  // def WITH_OPTIX_DENOISER
    CU_DEVICE_PTR dev_pixels_node1_denoised = NULL;
    CU_DEVICE_PTR dev_pixels_node2_denoised = NULL;

    cuda_assert2(cudaHostGetDevicePointer(&dev_pixels_node1_denoised, pixels_node_denoiser, 0));
    dev_pixels_node2_denoised = (CU_DEVICE_PTR)((char *)dev_pixels_node1_denoised +
                                                pix_size_denoiser);

#  if defined(WITH_CLIENT_RENDERENGINE_VR) || \
      (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
    dev_pixels_node2_denoised = (char *)dev_pixels_node2_denoised + pix_size_denoiser;
#  endif

    // CUDADeviceQueue queue_denoiser;
    den_ptr = device_optix_create();
#endif

#if defined(WITH_CLIENT_RENDERENGINE_VR) || \
    (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
    int devices_size_vr = (id < devices_left_eye) ? devices_left_eye : devices_right_eye;
    int id_vr = (id < devices_left_eye) ? id : id - devices_left_eye;
#endif

    if (!scope.get().kernels.is_loaded()) {
      /* Get kernel function. */
      // cu_assert(cuModuleGetFunction(
      //    &scope.get().cuPathTrace, scope.get().cuModule, "kernel_cuda_path_trace"));
      // cu_assert(cuFuncSetCacheConfig(scope.get().cuPathTrace, CU_FUNC_CACHE_PREFER_L1));
      // cu_assert(cuFuncSetAttribute(
      //    scope.get().cuPathTrace, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 0));
      // cuda_assert2(cuFuncSetCacheConfig(scope.get().cuPathTrace, CU_FUNC_CACHE_PREFER_NONE));
      scope.get().kernels.load(&scope.get());

#if 0  // def WITH_OPTIX_DENOISER
      cu_assert(cuModuleGetFunction(
          &scope.get().cuFloatToUchar, scope.get().cuModule, "kernel_cuda_float_to_uchar"));
#endif

#if 0  // WITH_CUDA_STAT
      cu_assert(cuModuleGetFunction(
          &scope.get().cuPathTraceStat, scope.get().cuModuleStat, "kernel_cuda_path_trace"));
      cu_assert(cuFuncSetCacheConfig(scope.get().cuPathTraceStat, CU_FUNC_CACHE_PREFER_L1));
      cu_assert(cuFuncSetAttribute(
          scope.get().cuPathTraceStat, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 0));
#endif
    }

    ccl::KernelWorkTile &wtile = scope.get().wtile;

#ifdef WITH_CUDA_STAT
    if (scope.get().path_stat_is_done == 0) {
      scope.get().start_sample = 0;
      scope.get().num_samples = 1;
      //}
#else
    if (scope.get().path_stat_is_done == 0) {
      scope.get().start_sample = start_sample;
      scope.get().num_samples = end_sample - start_sample;
      //}
#endif

      // wtile.pixel = (char *)dev_pixels_node;
      wtile.start_sample = scope.get().start_sample;
      wtile.num_samples = scope.get().num_samples;

      // scope.get().path_stat_is_done = 0;

      // cuda_assert2(cudaMalloc(&scope.get().d_work_tiles, sizeof(ccl::KernelWorkTile)));
      // log_alloc(id, sizeof(ccl::KernelWorkTile), "d_work_tiles");

      wtile.x = tile_x;
      wtile.w = tile_w;
      wtile.offset = offset;
      wtile.stride = stride;
#if defined(WITH_CLIENT_RENDERENGINE_VR) || \
    (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
      wtile.stride *= 2;
      if (id < devices_left_eye) {
      }
      else {
        wtile.offset = wtile.w;
      }
#endif
      wtile.buffer = (map_buffer_bin != 0) ?
                         (float *)scope.get().cuda_mem_map[map_buffer_bin].mem.device_pointer :
                         0;

      // cuda_assert2(cudaMemcpy(scope.get().d_work_tiles,
      //                       (char *)&wtile,
      //                       sizeof(ccl::KernelWorkTile),
      //                       cudaMemcpyHostToDevice));

#ifdef WITH_CLIENT_CUDA_GPU_TILES
      // wtile.y = tile_y;
      // wtile.h = tile_h;
#  if defined(WITH_CLIENT_NCCL_SOCKET) || defined(WITH_CLIENT_MPI_REDUCE)
      wtile.gpu_id = g_world_rank * devices_size + id;
      wtile.num_gpus = g_world_size * devices_size;
#  else
      wtile.gpu_id = id;
      wtile.num_gpus = devices_size;
#  endif
#endif

#if defined(WITH_CLIENT_RENDERENGINE_VR) || \
    (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
      int tile_step_dev = (int)((float)tile_h / (float)devices_size_vr);
      int tile_last_dev = tile_h - (devices_size_vr - 1) * tile_step_dev;

      int tile_y_dev = tile_y + tile_step_dev * id_vr;
      int tile_h_dev = (devices_size_vr - 1 == id_vr) ? tile_last_dev : tile_step_dev;

      wtile.y = tile_y_dev;
      wtile.h = tile_h_dev;

      const char *env_tiles = getenv("CLIENT_TILES");
      if (env_tiles) {
        wtile.y = util_get_int_from_env_array(env_tiles, 2 * id_vr + 0);
        wtile.h = util_get_int_from_env_array(env_tiles, 2 * id_vr + 1);
      }
#else

      int tile_step_dev = (int)((float)tile_h / (float)devices_size);
      int tile_last_dev = tile_h - (devices_size - 1) * tile_step_dev;

      int tile_y_dev = tile_y + tile_step_dev * id;
      int tile_h_dev = (devices_size - 1 == id) ? tile_last_dev : tile_step_dev;

#  if 1
      const char *env_tiles = getenv("CLIENT_TILES");
      const char *env_tiles_offset = getenv("CLIENT_TILES_OFFSET");
      if (env_tiles) {
        int offset = 0;
        if (env_tiles_offset) {
          offset = atoi(env_tiles_offset);
        }
        wtile.y = util_get_int_from_env_array(env_tiles, 2 * (id + offset) + 0);
        wtile.h = util_get_int_from_env_array(env_tiles, 2 * (id + offset) + 1);
      }
#  else
      const char *env_tiles = getenv("CLIENT_TILES");
      if (env_tiles) {
        wtile.y = util_get_int_from_env_array(env_tiles, 2 * id + 0);
        wtile.h = util_get_int_from_env_array(env_tiles, 2 * id + 1);
      }
#  endif
      else {
#  ifndef ENABLE_LOAD_BALANCEv2
        wtile.y = tile_y_dev;
        wtile.h = tile_h_dev;
#  else
        wtile.y = tile_x + tile_y_dev * tile_w;
        wtile.h = tile_w * tile_h_dev;
#  endif
      }

#endif
      //#endif
    }

#if 0
    if (scope.get().path_stat_is_done > 300)
       scope.get().path_stat_is_done = 0;
#endif

    // scope.get().num_threads_per_block = 64;
    //{
    //  int min_blocks, num_threads_per_block;
    //  cu_assert(cuOccupancyMaxPotentialBlockSize(
    //      &min_blocks, &num_threads_per_block, scope.get().cuPathTrace, NULL, 0, 0));
    //  scope.get().num_threads_per_block = num_threads_per_block;
    //  printf("%d: min_blocks: %d, num_threads_per_block %d\n",
    //         id,
    //         min_blocks,
    //         num_threads_per_block);
    //}

    DEVICE_PTR dev_pixels_node = (DEVICE_PTR)dev_pixels_node2;
#if 0  // def WITH_OPTIX_DENOISER
    DEVICE_PTR dev_pixels_node_denoised = (DEVICE_PTR)dev_pixels_node2_denoised;
#endif
    char *pixels_node = NULL;
    cudaStream_t stream_path;
    cudaStream_t stream_memcpy;

    cudaEvent_t event_path_start;
    cudaEvent_t event_path_stop;

    cudaEvent_t event_memcpy_start;
    cudaEvent_t event_memcpy_stop;

    float *time_path = NULL;
    float *time_memcpy = NULL;
    int stream_memcpy_id = 0;

    // cudaHostFn_t host_fn_memcpy_status_flag;

    ///////////////////////////////////////////////
    float *dev_buffer = NULL;
    size_t dev_buffer_size = 0;

    caching_kpc(id, scope, (pix_state != NULL) ? pix_state[2] == 1 : false);

#if defined(WITH_CLIENT_RENDERENGINE_VR) || \
    (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
    if (id < devices_left_eye) {
      dev_buffer = (map_buffer_bin != 0) ?
                       (float *)scope.get().cuda_mem_map[map_buffer_bin].mem.device_pointer :
                       0;
      //#ifdef WITH_CLIENT_ULTRAGRID
      dev_buffer_size = scope.get().cuda_mem_map[map_buffer_bin].mem.device_size;
      //#else
      //      dev_buffer_size = scope.get().cuda_mem_map[map_buffer_bin].size / 2;
      //#endif
      // const_copy_to_internal(scope, "data", &cuda_kernel_data[0], cuda_kernel_data.size());
    }
    else {

      //      dev_buffer_size = scope.get().cuda_mem_map[map_buffer_bin].size / 2;
      //#ifdef WITH_CLIENT_ULTRAGRID
      dev_buffer = (map_buffer_bin != 0) ?
                       (float *)scope.get().cuda_mem_map[map_buffer_bin].mem.device_pointer :
                       0;

      dev_buffer_size = scope.get().cuda_mem_map[map_buffer_bin].mem.device_size;
      //#else
      //      dev_buffer = (float *)((char
      //      *)scope.get().cuda_mem_map[map_buffer_bin].mem.device_pointer +
      //                             scope.get().cuda_mem_map[map_buffer_bin].size / 2);
      //
      //      dev_buffer_size = scope.get().cuda_mem_map[map_buffer_bin].size / 2;
      //#endif
      // dev_buffer = (map_buffer_bin != 0) ?
      //                 (float *)scope.get().cuda_mem_map[map_buffer_bin].mem.device_pointer :
      //                 0;

      // dev_pixels_node = (DEVICE_PTR)((char *)dev_pixels_node + pix_offset +
      //                               total_work_size * pix_type_size);

      // const_copy_to_internal(
      //    scope, "data", &cuda_kernel_data_right[0], cuda_kernel_data_right.size());
    }
#else
    dev_buffer = (map_buffer_bin != 0) ?
                     (float *)scope.get().cuda_mem_map[map_buffer_bin].mem.device_pointer :
                     0;

    dev_buffer_size = scope.get().cuda_mem_map[map_buffer_bin].mem.device_size;

// blurring
#  ifdef WITH_CLIENT_CACHE
    const_copy_to_internal(scope, "data", &cuda_kernel_data[0], cuda_kernel_data.size());
#  endif

#endif
    ///////////////////////////////////////////////
    // int used_buffer = 0;

    // while (true)
    {

      // if (pix_state != NULL && pix_state[4] == 1) {
      //   {
      //     size_t mem_free, mem_tot;
      //     cuda_assert2(cudaMemGetInfo(&mem_free, &mem_tot));
      //     printf("%d: Mem3 = %lld, %lld, used: %f [GB]\n",
      //            id,
      //            mem_free,
      //            mem_tot,
      //            (mem_tot - mem_free) / (1024.0 * 1024.0 * 1024.0));
      //     fflush(0);
      //   }

      //   // break;
      //   return;
      // }
      // else
      if (/*(pix_state[0] == 0*/ scope.get().used_buffer == 0) {

        //        if (id == 0) {
        //#if !defined(WITH_CLIENT_NCCL_SOCKET)  //&& !defined(WITH_CLIENT_MPI_REDUCE)
        //          if (lock0 != NULL)
        //            omp_set_lock(lock0);
        //#endif
        //        }

        //#pragma omp barrier

        dev_pixels_node = (DEVICE_PTR)dev_pixels_node1;
#if 0  // def WITH_OPTIX_DENOISER
        dev_pixels_node_denoised = (DEVICE_PTR)dev_pixels_node1_denoised;
#endif
        pixels_node = pixels_node1;
        stream_path = scope.get().stream[STREAM_PATH1];
        stream_memcpy = scope.get().stream[STREAM_PATH1_MEMCPY];
        stream_memcpy_id = STREAM_PATH1_MEMCPY;
        // host_fn_memcpy_status_flag = memcpy_status_flag1;

        event_path_start = scope.get().event[STREAM_PATH1];
        event_path_stop = scope.get().event[STREAM_PATH1 + STREAM_COUNT];

        event_memcpy_start = scope.get().event[STREAM_PATH1_MEMCPY];
        event_memcpy_stop = scope.get().event[STREAM_PATH1_MEMCPY + STREAM_COUNT];

        time_path = &scope.get().running_time[STREAM_PATH1];
        time_memcpy = &scope.get().running_time[STREAM_PATH1_MEMCPY];
      }
      else if (/*pix_state[1] == 0*/ scope.get().used_buffer == 1) {
        // printf("rend1: pix_state[1] = 0, %f\n", omp_get_wtime()); fflush(0);
        // pix_state[0] = 1;
        //#  pragma omp flush
        //        if (id == 0) {
        //#if !defined(WITH_CLIENT_NCCL_SOCKET)  //&& !defined(WITH_CLIENT_MPI_REDUCE)
        //          if (lock1 != NULL)
        //            omp_set_lock(lock1);
        //#endif
        //        }

        //#pragma omp barrier

        dev_pixels_node = (DEVICE_PTR)dev_pixels_node2;
#if 0  // def WITH_OPTIX_DENOISER
        dev_pixels_node_denoised = (DEVICE_PTR)dev_pixels_node2_denoised;
#endif
        pixels_node = pixels_node2;
        stream_path = scope.get().stream[STREAM_PATH2];
        stream_memcpy = scope.get().stream[STREAM_PATH2_MEMCPY];
        stream_memcpy_id = STREAM_PATH2_MEMCPY;
        // host_fn_memcpy_status_flag = memcpy_status_flag2;

        event_path_start = scope.get().event[STREAM_PATH2];
        event_path_stop = scope.get().event[STREAM_PATH2 + STREAM_COUNT];

        event_memcpy_start = scope.get().event[STREAM_PATH2_MEMCPY];
        event_memcpy_stop = scope.get().event[STREAM_PATH2_MEMCPY + STREAM_COUNT];

        time_path = &scope.get().running_time[STREAM_PATH2];
        time_memcpy = &scope.get().running_time[STREAM_PATH2_MEMCPY];
      }
      //      else {
      //        usleep(100);
      //
      //#pragma omp flush
      //        continue;
      //      }

      double t_id = omp_get_wtime();

      ///////////////////////
#ifdef ENABLE_INC_SAMPLES
      if (pix_state != NULL && pix_state[2] == 1)
#endif
      {
        // caching_kpc(id, scope);

#if defined(WITH_CLIENT_RENDERENGINE_VR) || \
    (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
        if (id < devices_left_eye) {
          // cuda_assert2(cudaMemcpy((CU_DEVICE_PTR)scope.get().kerneldata,
          //                        &cuda_kernel_data[0],
          //                        cuda_kernel_data.size(),
          //                        cudaMemcpyHostToDevice));
          const_copy_to_internal(scope, "data", &cuda_kernel_data[0], cuda_kernel_data.size());

          if (!scope.get().cuda_mem_map[map_buffer_bin].uni_mem || id == 0)
            cuda_assert2(cudaMemset((CU_DEVICE_PTR)dev_buffer, 0, dev_buffer_size));
        }
        else {
          // cuda_assert2(cudaMemcpy((CU_DEVICE_PTR)scope.get().kerneldata,
          //                        &cuda_kernel_data_right[0],
          //                        cuda_kernel_data_right.size(),
          //                        cudaMemcpyHostToDevice));
          const_copy_to_internal(
              scope, "data", &cuda_kernel_data_right[0], cuda_kernel_data_right.size());

          if (!scope.get().cuda_mem_map[map_buffer_bin].uni_mem || id == 0)
            cuda_assert2(cudaMemset((CU_DEVICE_PTR)dev_buffer, 0, dev_buffer_size));
        }
#else
        // cuda_assert2(cudaMemcpy((CU_DEVICE_PTR)scope.get().kerneldata,
        //                        &cuda_kernel_data[0],
        //                        cuda_kernel_data.size(),
        //                        cudaMemcpyHostToDevice));
        const_copy_to_internal(scope, "data", &cuda_kernel_data[0], cuda_kernel_data.size());

        // if (!scope.get().cuda_mem_map[map_buffer_bin].uni_mem || id == 0)
        //  cuda_assert2(cudaMemset((CU_DEVICE_PTR)dev_buffer, 0, dev_buffer_size));
        //     cuda_assert2(cudaMemcpyAsync((char *)pixels_node + pix_offset,
        //                                  (char *)dev_pixels_node + pix_offset,
        //                                  total_work_size * pix_type_size,
        int pass_stride = ((ccl::KernelData *)&cuda_kernel_data[0])->film.pass_stride;

        cuda_assert2(cudaMemset((char *)dev_buffer +
                                    (wtile.x + wtile.y * stride) * pass_stride * sizeof(float),
                                0,
                                wtile.w * wtile.h * pass_stride * sizeof(float)));
#endif
        // if (id == 0)
        // #pragma omp master
        //         {
        //           pix_state[2] = 0;
        //           //#  pragma omp flush
        //           //start_sample = 0;
        //           //end_sample = end_sample2 - start_sample2;
        //         }

        scope.get().start_sample = start_sample;
        scope.get().num_samples = end_sample - start_sample;

        scope.get().path_stat_is_done = 0;

        // return;
        //#pragma omp flush
        //#pragma omp barrier
      }

      //////////////////////
      // cuda_assert2(cudaStreamSynchronize(stream_memcpy));

      //#  ifdef WITH_CLIENT_PATHTRACER2
      //
      //#    pragma omp critical
#ifdef WITH_CLIENT_FILE
      if (scope.get().path_stat_is_done == 0) {
        size_t mem_free, mem_tot;
        cuda_assert2(cudaMemGetInfo(&mem_free, &mem_tot));
        printf("%d: Mem1 = %lld, %lld, used: %f [GB]\n",
               id,
               mem_free,
               mem_tot,
               (mem_tot - mem_free) / (1024.0 * 1024.0 * 1024.0));
        // fflush(0);
      }
#endif

      ///////////////////GPU SYNC////////////////////////////
      // cuda_assert2(cudaDeviceSynchronize());
      ///////////////////GPU SYNC////////////////////////////
      ///////////////////////////////////////////////////////////////

      // long_time_path = 0;
#ifndef ENABLE_LOAD_BALANCEv2
      unsigned int total_work_size = wtile.w * wtile.h;
#else
      unsigned int total_work_size = wtile.h;
#endif
      unsigned int num_blocks =
          0;  // ccl::divide_up(total_work_size, scope.get().num_threads_per_block);

#ifdef ENABLE_STEP_SAMPLES
      // if (id == 0)
      {
        if (debug_step_samples > 0) {
          scope.get().start_sample = start_sample +
                                     scope.get().path_stat_is_done * debug_step_samples;
          scope.get().num_samples = debug_step_samples;
        }
      }
//#pragma omp flush
//#pragma omp barrier
#endif

      wtile.buffer = (float *)dev_buffer;
      wtile.pixel = (char *)dev_pixels_node;
      wtile.start_sample = scope.get().start_sample;
      wtile.num_samples = scope.get().num_samples;
      // wtile.h = h;

#ifdef WITH_ORIG_TILES
      wtile.x2 = tile_x;
      wtile.y2 = tile_y;
      wtile.w2 = tile_w;
      wtile.h2 = tile_h;
#endif

      cuda_pathTraceWorkGPU->queue_set_stream(stream_path);
      cuda_assert2(cudaEventRecord(event_path_start, stream_path));

#ifdef WITH_CLIENT_CUDA_GPU_TILES
      int old_wty = wtile.y;
      int old_wth = wtile.h;
      wtile.y = tile_y;
      wtile.h = tile_h;
#endif

#ifdef WITH_CUDA_STAT
      if (scope.get().path_stat_is_done == 0) {
        cuda_pathTraceWorkGPU->kernels_use_stat(true);
        // cuda_pathTraceWorkGPU->queue_.use_stat = true;
        //
        // cu_assert(cuLaunchKernel(scope.get().cuPathTraceStat,
        //                         num_blocks,
        //                         1,
        //                         1,
        //                         scope.get().num_threads_per_block,
        //                         1,
        //                         1,
        //                         0,
        //                         stream_path,
        //                         args,
        //                         0));

        cuda_pathTraceWorkGPU->set_buffer(wtile, g_buffer_passes, w, h);

        ccl::PathTraceWork::RenderStatistics rs;
        cuda_pathTraceWorkGPU->render_samples(
            rs, wtile.start_sample, wtile.num_samples, 0, true, NULL);
      }
      else {
        // cuda_pathTraceWorkGPU->queue_.use_stat = false;
        cuda_pathTraceWorkGPU->kernels_use_stat(false);
#endif
        // cu_assert(cuLaunchKernel(scope.get().cuPathTrace,
        //                         num_blocks,
        //                         1,
        //                         1,
        //                         scope.get().num_threads_per_block,
        //                         1,
        //                         1,
        //                         0,
        //                         stream_path,
        //                         args,
        //                         0));
        cuda_pathTraceWorkGPU->set_buffer(wtile, g_buffer_passes, w, h);

        ccl::PathTraceWork::RenderStatistics rs;
        cuda_pathTraceWorkGPU->render_samples(
            rs, wtile.start_sample, wtile.num_samples, 0, true, NULL);

#ifdef WITH_CUDA_STAT
      }
#endif

#ifdef WITH_CLIENT_CUDA_GPU_TILES
      wtile.y = old_wty;
      wtile.h = old_wth;
      cuda_pathTraceWorkGPU->set_buffer(wtile, g_buffer_passes, w, h);
#endif

      cuda_assert2(cudaEventRecord(event_path_stop, stream_path));
      // cuda_assert2(cudaDeviceSynchronize());
      cuda_assert2(cudaEventSynchronize(event_path_stop));
      cuda_assert2(cudaEventElapsedTime(time_path, event_path_start, event_path_stop));

#ifdef WITH_OPTIX_DENOISER
      ccl::DenoiseParams denoise_params;
      denoise_params.use = true;
      denoise_params.type = ccl::DENOISER_OPTIX;
      denoise_params.start_sample = 0;

#  if 1
      denoise_params.use_pass_albedo = false;
      denoise_params.use_pass_normal = false;
      denoise_params.prefilter = ccl::DENOISER_PREFILTER_FAST;
#  else
      denoise_params.use_pass_albedo = true;
      denoise_params.use_pass_normal = true;
      denoise_params.prefilter = ccl::DENOISER_PREFILTER_ACCURATE;
#  endif

      ccl::DeviceDenoiseTask denoise_task;
      denoise_task.params = denoise_params;
      denoise_task.num_samples = wtile.start_sample + wtile.num_samples;
      denoise_task.buffer_params = cuda_pathTraceWorkGPU->effective_buffer_params_;
      denoise_task.allow_inplace_modification = true;  // TODO
      denoise_task.render_buffers = cuda_pathTraceWorkGPU->buffers_.get();

      const bool denoise_result = cuda_pathTraceWorkGPU->device_->denoise_buffer(denoise_task);
#endif

#if 1
      if (wtile.pixel != NULL) {
#  if 0  // def WITH_OPTIX_DENOISER
          cuda_pathTraceWorkGPU->run_denoiser(
              den_ptr, scope, stream_path, (char *)&wtile, dev_pixels_node_denoised);
#  else
        // ccl::KernelWorkTile wtile_film = scope.get().wtile;
        // wtile_film.x = tile_x;
        // wtile_film.y = tile_y_dev;
        // wtile_film.w = tile_w;
        // wtile_film.h = tile_h_dev;

        // cuda_pathTraceWorkGPU->run_film_convert(scope, stream_path, wtile);
        ccl::PassAccessor::Destination destination;
#    ifdef WITH_CLIENT_GPUJPEG
        destination.d_pixels_uchar_yuv = (ccl::device_ptr)wtile.pixel;
        destination.offset = wtile.offset + wtile.x + wtile.y * wtile.stride;
        destination.stride = wtile.stride;
        destination.num_components = 4;
#    else

#      if defined(WITH_CLIENT_RENDERENGINE)
        destination.d_pixels_uchar_rgba = (ccl::device_ptr)wtile.pixel;
#      else
        destination.d_pixels_half_rgba = (ccl::device_ptr)wtile.pixel;
#      endif

        destination.offset = wtile.offset + wtile.x + wtile.y * wtile.stride;
        destination.stride = wtile.stride;
        destination.num_components = 4;
#    endif
        destination.pixel_stride = cuda_pathTraceWorkGPU->effective_buffer_params_.pass_stride;
        cuda_pathTraceWorkGPU->get_render_tile_film_pixels(
            destination, ccl::PassMode::DENOISED, wtile.start_sample + wtile.num_samples);
#  endif
        // queue_.synchronize(scope, cuda_stream);
      }
#endif

      double t_id2 = omp_get_wtime();

#ifdef WITH_CUDA_STAT
      if (scope.get().path_stat_is_done == 0) {
        std::map<std::string, ccl::CUDADevice::CUDAMem>::iterator it_stat;
        for (it_stat = scope.get().cuda_stat_map.begin();
             it_stat != scope.get().cuda_stat_map.end();
             it_stat++) {
          ccl::CUDADevice::CUDAMem *cm = &it_stat->second;

          cuda_assert2(cudaMemcpyAsync((char *)&cm->map_host_pointer[0],
                                       (CU_DEVICE_PTR)cm->mem.device_pointer,
                                       cm->mem.device_size,
                                       cudaMemcpyDeviceToHost,
                                       stream_path));

          cuda_assert2(cudaMemsetAsync(
              (CU_DEVICE_PTR)cm->mem.device_pointer, 0, cm->mem.device_size, stream_path));
        }
      }

#endif

#if 0  // def WITH_OPTIX_DENOISER2
      unsigned char *up = (unsigned char *)((char *)pixels_node);
      float *fp = (float *)((char *)pixels_node_denoiser);
      for (int y1 = wtile.y; y1 < wtile.y + wtile.h; y1++) {
        for (int x1 = wtile.x; x1 < wtile.x + wtile.w; x1++) {
          up[(x1 + y1 * tile_w) * 4 + 0] =
              (unsigned char)(ccl::saturate(fp[(x1 + y1 * tile_w) * 4 + 0]) * 255.0f);
          up[(x1 + y1 * tile_w) * 4 + 1] =
              (unsigned char)(ccl::saturate(fp[(x1 + y1 * tile_w) * 4 + 1]) * 255.0f);
          up[(x1 + y1 * tile_w) * 4 + 2] =
              (unsigned char)(ccl::saturate(fp[(x1 + y1 * tile_w) * 4 + 2]) * 255.0f);
          up[(x1 + y1 * tile_w) * 4 + 3] =
              (unsigned char)(ccl::saturate(fp[(x1 + y1 * tile_w) * 4 + 3]) * 255.0f);
        }
      }

#endif

      //#  ifdef WITH_CLIENT_PATHTRACER2

      //#    pragma omp critical
#ifdef WITH_CLIENT_FILE
      if (scope.get().path_stat_is_done == 0) {
        size_t mem_free, mem_tot;
        cuda_assert2(cudaMemGetInfo(&mem_free, &mem_tot));
        printf("%d: Mem2 = %lld, %lld, used: %f [GB]\n",
               id,
               mem_free,
               mem_tot,
               (mem_tot - mem_free) / (1024.0 * 1024.0 * 1024.0));
        // fflush(0);
      }
#endif

///////////////////////////////////
#ifdef WITH_CLIENT_SHOW_STAT
      {
        SHOW_STAT_COLOR_PICKER;
        unsigned char *up = (unsigned char *)((char *)pixels_node);
        for (int y1 = wtile.y; y1 < wtile.y + wtile.h; y1++) {
          for (int x1 = wtile.x + (int)((float)wtile.w * 0.99f); x1 < wtile.x + wtile.w; x1++) {
            up[(x1 + y1 * tile_w) * 4 + 0] = (unsigned char)(color_picker[0 + id * 3] * 255.0f);
            up[(x1 + y1 * tile_w) * 4 + 1] = (unsigned char)(color_picker[1 + id * 3] * 255.0f);
            up[(x1 + y1 * tile_w) * 4 + 2] = (unsigned char)(color_picker[2 + id * 3] * 255.0f);
            up[(x1 + y1 * tile_w) * 4 + 3] = (unsigned char)(1.0f * 255.0f);
          }
        }
      }
#endif
      ///////////////////////////////////

#ifdef WITH_CUDA_STATv2_LB
      if (pix_state == NULL) {
        break;
      }
#endif

      // #pragma omp master
      //       {
      //         pix_state[6] = 1;
      //       }

      // if(pix_state[4] == 1)
      // {
      //   return;
      // }

      //#pragma omp barrier

      /*
      #  pragma omp critical
            if (g_report_time_count == 0) {
              size_t mem_free, mem_tot;
              cuda_assert2(cudaMemGetInfo(&mem_free, &mem_tot));
              //printf("%d: Mem2 = %lld, %lld\n", id, mem_free, mem_tot);
              printf("%d: Mem2 = %lld, %lld, used: %f [GB]\n", id, mem_free, mem_tot, (mem_tot -
      mem_free) / (1024.0*1024.0*1024.0)); fflush(0);
            }
      */
      scope.get().path_stat_is_done++;
      scope.get().start_sample = scope.get().start_sample + scope.get().num_samples;

      //#pragma omp master
      {
        if (/*pix_state[0] == 0*/ scope.get().used_buffer == 0) {
          //          pix_state[0] = 2;
          //          pix_state[1] = 0;
          //          //printf("rend2: pix_state[0] = 2, %f\n", omp_get_wtime()); fflush(0);
          //#  pragma omp flush
          //#pragma omp master
          //          if (id == 0) {
          //#if !defined(WITH_CLIENT_NCCL_SOCKET)  //&& !defined(WITH_CLIENT_MPI_REDUCE)
          //            if (lock0 != NULL)
          //              omp_unset_lock(lock0);
          //              // omp_unset_nest_lock(lock0);
          //#endif
          //          }
        }
        else if (/*pix_state[1] == 0*/ scope.get().used_buffer == 1) {
          //          pix_state[1] = 2;
          //          pix_state[0] = 0;
          //          //printf("rend2: pix_state[1] = 2, %f\n", omp_get_wtime()); fflush(0);
          //#  pragma omp flush
          //          if (id == 0) {
          //#if !defined(WITH_CLIENT_NCCL_SOCKET)  //&& !defined(WITH_CLIENT_MPI_REDUCE)
          //            if (lock1 != NULL)
          //              omp_unset_lock(lock1);
          //#endif
          //          }
        }

        scope.get().used_buffer++;
        if (scope.get().used_buffer >= CLIENT_SWAP_BUFFERS)
          scope.get().used_buffer = 0;
      }

      // #pragma omp master
      //       {
      //         pix_state[6] = 1;
      //       }

      // if(pix_state[4] == 1)
      // {
      //   return;
      // }
      //#pragma omp barrier

// #if defined(ENABLE_STEP_SAMPLES)
//       if (scope.get().start_sample + scope.get().num_samples >= end_sample)
//         break;
// #endif

// #ifdef WITH_CLIENT_SHOW_STAT_BVH_LOOP
//       break;
// #endif

      //#if defined(WITH_CLIENT_RENDERENGINE_EMULATE_ONE_THREAD)
      //      // const char *drt = getenv("DEBUG_REPEAT_TIME");
      //      if (drt == NULL || omp_get_wtime() - drt_time_start > atof(drt)) {
      //        pix_state[4] = 1;
      //      }
      //#endif

      //#if defined(WITH_CLIENT_RENDERENGINE_EMULATE) && \
//    !defined(WITH_CLIENT_RENDERENGINE_EMULATE_ONE_THREAD)
      //      //if (drt == NULL)
      //      //  break;
      //#endif
      //      if (scope.get().path_stat_is_done > 10)
      //           break;
      /*
      #  if defined(ENABLE_LOAD_BALANCE) || defined(ENABLE_LOAD_BALANCEv2)
            if (scope.get().path_stat_is_done > 60)
              break;
      #  elif defined(WITH_CLIENT_PATHTRACER2)
            if (scope.get().path_stat_is_done > 11)
              break;
      #  endif
      */

      ///////////////////////////////////////////////////////////////////

      // cuda_assert2(
      //    cudaLaunchHostFunc(stream_memcpy, host_fn_memcpy_status_flag, (void *)pix_state));
    }

    // cuda_assert2(cudaFree(scope.get().d_work_tiles));
    // scope.get().d_work_tiles = 0;

#if 0  // def WITH_OPTIX_DENOISER
    device_optix_destroy(den_ptr);
#endif

    // omp_mem_free(DEVICE_ID, dev_pixels_node1, pix_size);
    // omp_mem_free(DEVICE_ID, dev_pixels_node2, pix_size);
    //#ifndef WITH_CUDA_BUFFER_MANAGED
    //    //cuda_assert2(cudaFree((CU_DEVICE_PTR)dev_pixels_node1));
    //    //log_free(id, pix_size, "dev_pixels_node1");
    //    //cuda_assert2(cudaFree((CU_DEVICE_PTR)dev_pixels_node2));
    //    //log_free(id, pix_size, "dev_pixels_node2");
    //#endif
  }

#if 0  // def WITH_OPTIX_DENOISER
  // delete[] pixels_node_denoiser;
#  ifdef WITH_CUDA_BUFFER_MANAGED
  cuda_host_free("pixels_node_denoiser", NULL, pixels_node_denoiser, 1);
#  else
  cuda_host_free("pixels_node_denoiser", NULL, pixels_node_denoiser, 2);
#  endif
#endif
}  // namespace kernel

void path_trace_lb(int numDevice,
                   DEVICE_PTR map_buffer_bin,
                   int start_sample,
                   int end_sample,
                   int tile_x,
                   int tile_y,
                   int offset,
                   int stride,
                   int tile_h,
                   int tile_w,
                   int nprocs_cpu)
{
  //////////////////////
  printf("Rendering LB: ss: %d, se: %d: x: %d, y:%d: w: %d, h: %d, devs: %d\n",
         start_sample,
         end_sample,
         tile_x,
         tile_y,
         tile_w,
         tile_h,
         nprocs_cpu);
  //////////////////////

  // int start_sample2 = start_sample;
  // int end_sample2 = end_sample;

  // float g_report_time = 0;
  // size_t g_report_time_count = 0;

  int devices_size = nprocs_cpu;

  size_t pix_type_size = SIZE_UCHAR4;
  size_t pix_size = tile_w * tile_h * pix_type_size;

  // char *pixels_node1 = (char *)map_pixel_bin;
  // char *pixels_node2 = (char *)map_pixel_bin + tile_w * tile_h * SIZE_UCHAR4;
  // int *pix_state = (int *)signal_value;

  // float long_time_path = 0;
  // for schedule(static, 1)// reduction(max : long_time_path)
  // for (int id = 0; id < devices_size; id++)
#pragma omp parallel num_threads(devices_size)
  //#  pragma omp parallel num_threads(1)
  {
    int id = omp_get_thread_num();
    ccl::CUDAContextScope scope(id);

    ccl::PathTraceWorkGPU *cuda_pathTraceWorkGPU = g_pathTraceWorkGPU[id];

    // CU_DEVICE_PTR dev_pixels_node1;  // omp_mem_alloc(DEVICE_ID, "pixels1", NULL, pix_size);
    // cuda_assert2(cudaMalloc(&dev_pixels_node1, pix_size));
    // log_alloc(id, pix_size, "dev_pixels_node1");

    // CU_DEVICE_PTR dev_pixels_node2;  // omp_mem_alloc(DEVICE_ID, "pixels2", NULL, pix_size);
    // cuda_assert2(cudaMalloc(&dev_pixels_node2, pix_size));
    // log_alloc(id, pix_size, "dev_pixels_node2");

    ccl::KernelWorkTile &wtile = scope.get().wtile;
    // wtile.pixel = (char *)dev_pixels_node;
    wtile.start_sample = start_sample;
    wtile.num_samples = end_sample - start_sample;
    scope.get().path_stat_is_done = 0;

    if (!scope.get().kernels.is_loaded()) {
      /* Get kernel function. */
      // cu_assert(cuModuleGetFunction(
      //    &scope.get().cuPathTrace, scope.get().cuModule, "kernel_cuda_path_trace"));
      // cu_assert(cuFuncSetCacheConfig(scope.get().cuPathTrace, CU_FUNC_CACHE_PREFER_L1));
      // cu_assert(cuFuncSetAttribute(
      //    scope.get().cuPathTrace, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 0));
      // cuda_assert2(cuFuncSetCacheConfig(scope.get().cuPathTrace, CU_FUNC_CACHE_PREFER_NONE));
      scope.get().kernels.load(&scope.get());

      // cuda_assert2(cudaMalloc(&scope.get().d_work_tiles, sizeof(ccl::KernelWorkTile)));
      // log_alloc(id, sizeof(ccl::KernelWorkTile), "d_work_tiles");

      wtile.x = tile_x;
      wtile.w = tile_w;

      wtile.offset = offset;
      wtile.stride = stride;
      wtile.buffer = (map_buffer_bin != 0) ?
                         (float *)scope.get().cuda_mem_map[map_buffer_bin].mem.device_pointer :
                         0;

      // cuda_assert2(cudaMemcpy(scope.get().d_work_tiles,
      //                       (char *)&wtile,
      //                       sizeof(ccl::KernelWorkTile),
      //                       cudaMemcpyHostToDevice));

      int tile_step_dev = (int)((float)tile_h / (float)devices_size);
      int tile_last_dev = tile_h - (devices_size - 1) * tile_step_dev;

      int tile_y_dev = tile_y + tile_step_dev * id;
      int tile_h_dev = (devices_size - 1 == id) ? tile_last_dev : tile_step_dev;

      const char *env_tiles = getenv("CLIENT_TILES");
      if (env_tiles) {
        wtile.y = util_get_int_from_env_array(env_tiles, 2 * id + 0);
        wtile.h = util_get_int_from_env_array(env_tiles, 2 * id + 1);
      }
      else {
#ifndef ENABLE_LOAD_BALANCEv2
        wtile.y = tile_y_dev;
        wtile.h = tile_h_dev;
#else
        wtile.y = tile_x + tile_y_dev * tile_w;
        wtile.h = tile_w * tile_h_dev;
#endif
      }

      //{
      //  int min_blocks, num_threads_per_block;
      //  cu_assert(cuOccupancyMaxPotentialBlockSize(
      //      &min_blocks, &num_threads_per_block, scope.get().cuPathTrace, NULL, 0, 0));
      //  scope.get().num_threads_per_block = num_threads_per_block;
      //  printf("%d: min_blocks: %d, num_threads_per_block %d\n",
      //         id,
      //         min_blocks,
      //         num_threads_per_block);
      //}
    }

    DEVICE_PTR dev_pixels_node = 0;
    // char *pixels_node = NULL;
    cudaStream_t stream_path;
    cudaStream_t stream_memcpy;

    cudaEvent_t event_path_start;
    cudaEvent_t event_path_stop;

    cudaEvent_t event_memcpy_start;
    cudaEvent_t event_memcpy_stop;

    float *time_path = NULL;
    float *time_memcpy = NULL;
    int stream_memcpy_id = 0;

    // cudaHostFn_t host_fn_memcpy_status_flag;

    stream_path = scope.get().stream[STREAM_PATH1];
    stream_memcpy = scope.get().stream[STREAM_PATH1_MEMCPY];
    stream_memcpy_id = STREAM_PATH1_MEMCPY;
    // host_fn_memcpy_status_flag = memcpy_status_flag1;

    event_path_start = scope.get().event[STREAM_PATH1];
    event_path_stop = scope.get().event[STREAM_PATH1 + STREAM_COUNT];

    event_memcpy_start = scope.get().event[STREAM_PATH1_MEMCPY];
    event_memcpy_stop = scope.get().event[STREAM_PATH1_MEMCPY + STREAM_COUNT];

    time_path = &scope.get().running_time[STREAM_PATH1];
    time_memcpy = &scope.get().running_time[STREAM_PATH1_MEMCPY];

    bool do_render = true;

    while (do_render) {

      double t_id = omp_get_wtime();
      // long_time_path = 0;
#ifndef ENABLE_LOAD_BALANCEv2
      unsigned int total_work_size = wtile.w * wtile.h;
#else
      unsigned int total_work_size = wtile.h;
#endif
      unsigned int num_blocks =
          0;  // ccl::divide_up(total_work_size, scope.get().num_threads_per_block);

      /* Launch kernel. */
      // void *args[] = {&scope.get().d_work_tiles,
      //                &total_work_size,
      //                &wtile.y,
      //                &wtile.h,
      //                &dev_pixels_node,
      //                &start_sample,
      //                &end_sample};

      wtile.pixel = (char *)dev_pixels_node;
      wtile.start_sample = start_sample;
      wtile.num_samples = end_sample - start_sample;

      cuda_assert2(cudaEventRecord(event_path_start, stream_path));

      // cu_assert(cuLaunchKernel(scope.get().cuPathTrace,
      //                         num_blocks,
      //                         1,
      //                         1,
      //                         scope.get().num_threads_per_block,
      //                         1,
      //                         1,
      //                         0,
      //                         stream_path,
      //                         args,
      //                         0));

      cuda_pathTraceWorkGPU->set_buffer(wtile, g_buffer_passes, tile_w, tile_h);

      ccl::PathTraceWork::RenderStatistics rs;
      cuda_pathTraceWorkGPU->render_samples(
          rs, wtile.start_sample, wtile.num_samples, 0, true, NULL);

      cuda_assert2(cudaEventRecord(event_path_stop, stream_path));
      cuda_assert2(cudaEventSynchronize(event_path_stop));

      // transfer to CPU
      // size_t pix_offset = (wtile.x + wtile.y * stride) * pix_type_size;
      // cuda_assert2(cudaEventRecord(event_memcpy_start, stream_path));

      // cuda_assert2(cudaMemcpyAsync((char *)pixels_node + pix_offset,
      //                            (char *)dev_pixels_node + pix_offset,
      //                            total_work_size * pix_type_size,
      //                            cudaMemcpyDeviceToHost,
      //                            stream_path));

      // cuda_assert2(cudaStreamSynchronize(stream_path));

      // cuda_assert2(cudaEventRecord(event_memcpy_stop, stream_path));
      // cuda_assert2(cudaDeviceSynchronize());
      // cuda_assert2(cudaEventSynchronize(event_memcpy_stop));

      cuda_assert2(cudaEventElapsedTime(time_path, event_path_start, event_path_stop));
      // cuda_assert2(cudaEventElapsedTime(time_memcpy, event_memcpy_start, event_memcpy_stop));

#pragma omp flush
#pragma omp barrier
      ///////////////////////////////////////////////////////////////////////
      //#if 1
      double max_time = 0;
      double min_time = DBL_MAX;
      int max_h = 0;
      for (int id1 = 0; id1 < devices_size; id1++) {
        if (max_time < ccl::cuda_devices[id1].running_time[STREAM_PATH1]) {
          max_time = ccl::cuda_devices[id1].running_time[STREAM_PATH1];
        }
        if (min_time > ccl::cuda_devices[id1].running_time[STREAM_PATH1]) {
          min_time = ccl::cuda_devices[id1].running_time[STREAM_PATH1];
        }
        if (max_h <
            abs((int)ccl::cuda_devices[id1].wtile_h - (int)ccl::cuda_devices[id1].wtile.h)) {
          max_h = abs((int)ccl::cuda_devices[id1].wtile_h - (int)ccl::cuda_devices[id1].wtile.h);
        }
      }

      // 1.0% err
      double err_time = 1.0;
      if (max_time - min_time < max_time * err_time / 100.0 ||
          scope.get().path_stat_is_done > 100) {
        do_render = false;
      }
      //#endif

      //#if 0

      //      for (int id1 = 0; id1 < devices_size; id1++) {
      //        if (max_h < abs((int)ccl::cuda_devices[id1].wtile_h -
      //        (int)ccl::cuda_devices[id1].wtile.h)) {
      //          max_h = abs((int)ccl::cuda_devices[id1].wtile_h -
      //          (int)ccl::cuda_devices[id1].wtile.h);
      //        }
      //      }
      //
      //      // 1.0 err
      //      int err_h = 1;
      //      if (err_h > max_h) {
      //        do_render = false;
      //      }

      //#endif

#pragma omp barrier

      ///////////////////////////////////////////////////////////////////////
      // if (id == 0)
      if (id == 0) {
        std::string client_tiles = "";
        double total_time = 0;
        double avg_time = 0;
        for (int id1 = 0; id1 < devices_size; id1++) {
          ccl::cuda_devices[id1].wtile_h = ccl::cuda_devices[id1].wtile.h;

          client_tiles += std::to_string(ccl::cuda_devices[id1].wtile.y) + std::string(";") +
                          std::to_string(ccl::cuda_devices[id1].wtile.h) + std::string(";");

          total_time += ccl::cuda_devices[id1].running_time[STREAM_PATH1];
        }
        avg_time = total_time / (double)devices_size;
        //#if 0
        //          printf("LB: %s, STEPS: %d, max: %d\n", client_tiles.c_str(),
        //          scope.get().path_stat_is_done, max_h);
        //
        //
        //        if (err_h > max_h) {
        //          setenv("CLIENT_TILES", client_tiles.c_str(), 1);
        //        }
        //#endif
        //#if 1
        printf("LB: %s, STEPS: %d, diff time: %f, diff h: %d, num_blocks: %d\n",
               client_tiles.c_str(),
               scope.get().path_stat_is_done,
               max_time - min_time,
               max_h,
               num_blocks);

        if (max_time - min_time < max_time * err_time / 100.0 ||
            scope.get().path_stat_is_done > 100) {
          setenv("CLIENT_TILES", client_tiles.c_str(), 1);

          const char *lb_cuda_file = getenv("LOAD_BALANCE_CUDA_FILE");
          if (lb_cuda_file != NULL) {
            FILE *f = fopen(lb_cuda_file, "wt+");
            fputs(client_tiles.c_str(), f);
            fclose(f);
          }
        }
        //#endif
        else {

          if (max_h > 2) {
            ///////////////////////LB////////////////////////
//          for (int id1 = 0; id1 < devices_size - 1; id1++) {
//            float time1 = 0;
//            float time2 = 0;
//            time1 = ccl::cuda_devices[id1].running_time[STREAM_PATH1];
//            time2 = ccl::cuda_devices[id1 + 1].running_time[STREAM_PATH1];
//
//            if (time1 < time2) {
//              int pix_transm = (int)((float)ccl::cuda_devices[id + 1].wtile.h *
//                                     ((((float)time2 - (float)time1) / 2.0f) / time2));
//              ccl::cuda_devices[id1].wtile.h += pix_transm;
//              ccl::cuda_devices[id1 + 1].wtile.y += pix_transm;
//              ccl::cuda_devices[id1 + 1].wtile.h -= pix_transm;
//            }
//            else if (time1 > time2) {
//              int pix_transm = (int)((float)ccl::cuda_devices[id].wtile.h *
//                                     ((((float)time1 - (float)time2) / 2.0f) / time1));
//              ccl::cuda_devices[id1].wtile.h -= pix_transm;
//              ccl::cuda_devices[id1 + 1].wtile.y -= pix_transm;
//              ccl::cuda_devices[id1 + 1].wtile.h += pix_transm;
//            }
//          }
#ifndef ENABLE_LOAD_BALANCEv2
            int _tile_h = tile_h;
#else
            int _tile_h = tile_w * tile_h;
#endif
            int total_h = 0;

            for (int id1 = 0; id1 < devices_size; id1++) {
              float time1 = ccl::cuda_devices[id1].running_time[STREAM_PATH1];
              ccl::cuda_devices[id1].wtile.h = (int)ceil((double)ccl::cuda_devices[id1].wtile.h *
                                                         (double)avg_time / (double)time1);
              total_h += ccl::cuda_devices[id1].wtile.h;
            }
            if (total_h > _tile_h) {
              for (int id1 = 0; id1 < devices_size; id1++) {
                ccl::cuda_devices[id1].wtile.h = ccl::cuda_devices[id1].wtile.h -
                                                 (int)ceil((double)(total_h - _tile_h) /
                                                           (double)devices_size);
              }
            }
            int total_y = tile_y;
            ccl::cuda_devices[0].wtile.y = total_y;
            for (int id1 = 1; id1 < devices_size; id1++) {
              total_y += ccl::cuda_devices[id1 - 1].wtile.h;
              ccl::cuda_devices[id1].wtile.y = total_y;
            }
            ccl::cuda_devices[devices_size - 1].wtile.h =
                _tile_h - ccl::cuda_devices[devices_size - 1].wtile.y;
          }
          else {
            for (int id1 = 0; id1 < devices_size - 1; id1++) {
              float time1 = 0;
              float time2 = 0;
              time1 = ccl::cuda_devices[id1].running_time[STREAM_PATH1];
              time2 = ccl::cuda_devices[id1 + 1].running_time[STREAM_PATH1];

              if (time1 < time2) {
                ccl::cuda_devices[id1].wtile.h++;
                ccl::cuda_devices[id1 + 1].wtile.y++;
                ccl::cuda_devices[id1 + 1].wtile.h--;
              }
              else if (time1 > time2) {
                ccl::cuda_devices[id1].wtile.h--;
                ccl::cuda_devices[id1 + 1].wtile.y--;
                ccl::cuda_devices[id1 + 1].wtile.h++;
              }
            }
          }
        }
      }

      scope.get().path_stat_is_done++;
      ///////////////////////////////////////////////////////////////////
#pragma omp barrier
    }
    ////////////////////////////FREE MEM///////////////////////////////////
    scope.get().path_stat_is_done = 0;

    // cuda_assert2(cudaFree(scope.get().d_work_tiles));
    // scope.get().d_work_tiles = 0;

    if (!scope.get().cuda_mem_map[map_buffer_bin].uni_mem || id == 0) {
      //////////////////////////////clear buffer/////////////////////////////////////
      cuda_assert2(
          cudaMemset((CU_DEVICE_PTR)scope.get().cuda_mem_map[map_buffer_bin].mem.device_pointer,
                     0,
                     scope.get().cuda_mem_map[map_buffer_bin].mem.device_size));
    }
  }
}

//#  ifdef WITH_CLIENT_CUDA_CPU_STAT
//
// void path_trace_internal_cpu_stat(int numDevice,
//                                 DEVICE_PTR map_kg_bin,
//                                 DEVICE_PTR map_buffer_bin,
//                                 DEVICE_PTR map_pixel_bin,
//                                 int start_sample,
//                                 int end_sample,
//                                 int tile_x,
//                                 int tile_y,
//                                 int offset,
//                                 int stride,
//                                 int tile_h,
//                                 int tile_w,
//                                 char *sample_finished_omp,
//                                 char *reqFinished_omp,
//                                 int nprocs_cpu,
//                                 char *signal_value)
//{
//  int start_sample2 = start_sample;
//  int end_sample2 = end_sample;
//
//  float g_report_time = 0;
//  size_t g_report_time_count = 0;
//
//  int devices_size = nprocs_cpu;
//
//  size_t pix_size = tile_w * tile_h * SIZE_UCHAR4;
//
//  char *pixels_node1 = (char *)map_pixel_bin;
//  char *pixels_node2 = (char *)map_pixel_bin + pix_size;
//  int *pix_state = (int *)signal_value;
//
//  std::vector<double> g_image_time(tile_h);
//
//  std::vector<CUDADevice> omp_devices(devices_size);
//
//  //DEVICE_PTR dev_pixels_node1 = omp_mem_alloc(OMP_DEVICE_ID, "dev_pixels_node1", NULL,
//  pix_size);
//
//  //DEVICE_PTR dev_pixels_node2 = omp_mem_alloc(OMP_DEVICE_ID, "dev_pixels_node2", NULL,
//  pix_size);
//
//#    pragma omp parallel num_threads(devices_size)
//  {
//    int id = omp_get_thread_num();
//
//    omp_devices[id].wtile.start_sample = start_sample;
//    omp_devices[id].wtile.num_samples = end_sample - start_sample;
//
//    if (omp_devices[id].d_work_tiles == 0) {
//
//      omp_devices[id].d_work_tiles = (CU_DEVICE_PTR)&omp_devices[id].wtile;
//
//      omp_devices[id].wtile.x = tile_x;
//      omp_devices[id].wtile.w = tile_w;
//
//      omp_devices[id].wtile.offset = offset;
//      omp_devices[id].wtile.stride = stride;
//      omp_devices[id].wtile.buffer = (float *)map_buffer_bin;
//
//      // cuda_assert2(cudaMemcpy(
//      //    scope.get().d_work_tiles, (char *)&wtile, sizeof(WorkTile),
//      cudaMemcpyHostToDevice));
//
//      int tile_step_dev = (int)((float)tile_h / (float)devices_size);
//      int tile_last_dev = tile_h - (devices_size - 1) * tile_step_dev;
//
//      int tile_y_dev = tile_y + tile_step_dev * id;
//      int tile_h_dev = (devices_size - 1 == id) ? tile_last_dev : tile_step_dev;
//
//      const char *env_tiles = getenv("CLIENT_TILES");
//      if (env_tiles) {
//        omp_devices[id].wtile.y = util_get_int_from_env_array(env_tiles, 2 * id + 0);
//        omp_devices[id].wtile.h = util_get_int_from_env_array(env_tiles, 2 * id + 1);
//      }
//      else {
//        omp_devices[id].wtile.y = tile_y_dev;
//        omp_devices[id].wtile.h = tile_h_dev;
//      }
//
//      // scope.get().num_threads_per_block = 64;
//    }
//  }
//
//  while (true) {
//
//    double t_id = omp_get_wtime();
//
//    ccl::KernelGlobals *_kg = (ccl::KernelGlobals *)map_kg_bin;
//
//    if (pix_state[2] == 1) {
//
//      // if (id == 0) {
//      pix_state[2] = 0;
//      start_sample = start_sample2;
//      end_sample = end_sample2;
//
//      size_t sizeBuf_node = tile_w * tile_h * _kg->data.film.pass_stride * sizeof(float);
//      memset((char *)map_buffer_bin, 0, sizeBuf_node);
//      // }
//
//      //#    pragma omp barrier
//    }
//
//#    pragma omp parallel num_threads(devices_size)  // private(kg)
//    {
//      int id = omp_get_thread_num();
//
//      ccl::KernelGlobals kg = *((ccl::KernelGlobals *)map_kg_bin);
//
//      //DEVICE_PTR dev_pixels_node = (DEVICE_PTR)dev_pixels_node2;
//      //char *pixels_node = NULL;
//
//      float *time_path = NULL;
//      float *time_memcpy = NULL;
//      int stream_memcpy_id = 0;
//
//      // cudaHostFn_t host_fn_memcpy_status_flag;
//
//      bool render_run = true;
//
//      if (pix_state[0] == 0) {
//        //dev_pixels_node = (DEVICE_PTR)dev_pixels_node2;
//        //pixels_node = pixels_node2;
//
//        time_path = &omp_devices[id].running_time[STREAM_PATH2];
//        time_memcpy = &omp_devices[id].running_time[STREAM_PATH2_MEMCPY];
//      }
//      else if (pix_state[1] == 0) {
//        //dev_pixels_node = (DEVICE_PTR)dev_pixels_node1;
//        //pixels_node = pixels_node1;
//
//        time_path = &omp_devices[id].running_time[STREAM_PATH1];
//        time_memcpy = &omp_devices[id].running_time[STREAM_PATH1_MEMCPY];
//      }
//      else {
//        // continue;
//        render_run = false;
//      }
//      ///////////////////////
//
//      ///////////////////GPU SYNC////////////////////////////
//      if (render_run) {
//        double event_path_start = omp_get_wtime();
//
//        // kernel_tex_alloc(&kg, (KernelGlobals *)kg_bin);
//#    if 1
//        for (int y = omp_devices[id].wtile.y;
//             y < omp_devices[id].wtile.h + omp_devices[id].wtile.y;
//             y++) {
//          double y_start = omp_get_wtime();
//          for (int x = omp_devices[id].wtile.x;
//               x < omp_devices[id].wtile.w + omp_devices[id].wtile.x;
//               x++) {
//            for (int sample = start_sample; sample < end_sample; sample++) {
//              kernel_omp_path_trace(&kg,
//                                    omp_devices[id].wtile.buffer,
//                                    sample,
//                                    x,
//                                    y,
//                                    omp_devices[id].wtile.offset,
//                                    omp_devices[id].wtile.stride);
//            }
//            if (dev_pixels_node != NULL) {
//              kernel_film_convert_to_byte(&kg,
//                                          (uchar4 *)dev_pixels_node,
//                                          omp_devices[id].wtile.buffer,
//                                          1.0f / end_sample,
//                                          x,
//                                          y,
//                                          omp_devices[id].wtile.offset,
//                                          omp_devices[id].wtile.stride);
//            }
//          }
//          double y_stop = omp_get_wtime();
//          g_image_time[y] = y_stop - y_start;
//        }
//#    endif
//
//        double event_path_stop = omp_get_wtime();
//        unsigned int total_work_size = omp_devices[id].wtile.w * omp_devices[id].wtile.h;
//
//        double event_memcpy_start = omp_get_wtime();
//
//        // transfer to CPU
//        size_t pix_offset = (omp_devices[id].wtile.x + omp_devices[id].wtile.y * stride) *
//                            SIZE_UCHAR4;
//
//        memcpy((char *)pixels_node + pix_offset,
//               (char *)dev_pixels_node + pix_offset,
//               total_work_size * SIZE_UCHAR4);
//
//        double event_memcpy_stop = omp_get_wtime();
//
//        // cuda_assert2(cudaEventElapsedTime(time_path, event_path_start, event_path_stop));
//        time_path[0] = (event_path_stop - event_path_start) * 1000.0f;
//        // cuda_assert2(cudaEventElapsedTime(time_memcpy, event_memcpy_start,
//        event_memcpy_stop)); time_memcpy[0] = (event_memcpy_stop - event_memcpy_start) *
//        1000.0f;
//      }
//    }
//
//////////////////////////////////
////#    pragma omp barrier
//#    ifdef WITH_CPU_STAT
//
//    // if (id == 0)
//    {
//      std::map<std::string, OMPMem>::iterator it_stat;
//      for (it_stat = omp_stat_map.begin(); it_stat != omp_stat_map.end(); it_stat++) {
//        OMPMem *om = &it_stat->second;
//        memset(om->counter_sum, 0, om->counter_size * sizeof(size_t));
//      }
//    }
//    //#      pragma omp barrier
//    //#      pragma omp critical
//    for (int id1 = 0; id1 < devices_size; id1++) {
//      std::map<std::string, OMPMem>::iterator it_stat;
//      for (it_stat = omp_stat_map.begin(); it_stat != omp_stat_map.end(); it_stat++) {
//        OMPMem *om = &it_stat->second;
//
//        for (int i = 0; i < om->counter_size; i++)
//          om->counter_sum[i] += om->counter_pointer[id1][i];
//
//        // cuda_assert2(cudaMemcpyAsync((char *)&cm->map_host_pointer[0],
//        //                            (CU_DEVICE_PTR)cm->device_pointer,
//        //                            cm->size * sizeof(unsigned long long int),
//        //                            cudaMemcpyDeviceToHost,
//        //                            stream_path));
//
//        // cuda_assert2(cudaMemsetAsync((CU_DEVICE_PTR)cm->device_pointer,
//        //                            0,
//        //                            cm->size * sizeof(unsigned long long int),
//        //                            stream_path));
//      }
//    }
//    // sample_finished_omp
//#    endif
//
//    ///////////////////////////////////////////////////////////////////////
//    // if (id == 0)
//    {
//      g_report_time += (omp_get_wtime() - t_id);
//      g_report_time_count++;
//
//      if (/*omp_devices[id].path_stat_is_done == 0 ||*/ g_report_time >= 3.0) {
//        double long_time = 0;
//        for (int id1 = 0; id1 < devices_size; id1++) {
//
//          if (pix_state[0] == 0) {
//            if (long_time < omp_devices[id1].running_time[STREAM_PATH1]) {
//              long_time = omp_devices[id1].running_time[STREAM_PATH1];
//            }
//            printf("%d: Time = %f [ms], %d-%d, %d\n",
//                   id1,
//                   omp_devices[id1].running_time[STREAM_PATH1],
//                   omp_devices[id1].wtile.y,
//                   omp_devices[id1].wtile.h,
//                   end_sample);
//          }
//          else if (pix_state[1] == 0) {
//            if (long_time < omp_devices[id1].running_time[STREAM_PATH2]) {
//              long_time = omp_devices[id1].running_time[STREAM_PATH2];
//            }
//            printf("%d: Time = %f [ms], %d-%d, %d\n",
//                   id1,
//                   omp_devices[id1].running_time[STREAM_PATH2],
//                   omp_devices[id1].wtile.y,
//                   omp_devices[id1].wtile.h,
//                   end_sample);
//          }
//        }
//
//        printf("Rendering: FPS: %.2f, Time = %f [ms]\n", 1000.0f / long_time, long_time);
//
//        printf("OMP: FPS: %.2f: Time = %f [ms]\n",
//               1.0f / (omp_get_wtime() - t_id),
//               (omp_get_wtime() - t_id) * 1000.0f);
//
//        g_report_time = 0;
//        g_report_time_count = 0;
//      }
//
//      if (pix_state[0] == 0) {
//        pix_state[0] = 2;
//      }
//      else if (pix_state[1] == 0) {
//        pix_state[1] = 2;
//      }
//
//      pix_state[3] = end_sample;
//      ///////////////////////SAMPLES////////////////////////
//#    ifdef ENABLE_INC_SAMPLES
//      int num_samples = end_sample - start_sample;
//      // if (omp_devices[id].path_stat_is_done != 0)
//      {
//        start_sample = start_sample + num_samples;
//        end_sample = start_sample + num_samples;
//      }
//#    endif
//      ///////////////////////LB////////////////////////
//#    if defined(ENABLE_LOAD_BALANCE) || defined(ENABLE_LOAD_BALANCEv2)
//#      if 0
//        for (int id1 = 0; id1 < devices_size - 1; id1++) {
//          float time1 = 0;
//          float time2 = 0;
//          if (dev_pixels_node == (DEVICE_PTR)dev_pixels_node1) {
//            time1 = omp_devices[id1].running_time[STREAM_PATH1];
//            time2 = omp_devices[id1 + 1].running_time[STREAM_PATH1];
//          }
//          else {
//            time1 = omp_devices[id1].running_time[STREAM_PATH2];
//            time2 = omp_devices[id1 + 1].running_time[STREAM_PATH2];
//          }
//          if (time1 < time2) {
//            int pix_transm = (int)((float)omp_devices[id + 1].wtile.h *
//                                   ((((float)time2 - (float)time1) / 2.0f) / time2));
//            omp_devices[id1].wtile.h += pix_transm;
//            omp_devices[id1 + 1].wtile.y += pix_transm;
//            omp_devices[id1 + 1].wtile.h -= pix_transm;
//          }
//          else if (time1 > time2) {
//            int pix_transm = (int)((float)omp_devices[id].wtile.h *
//                                   ((((float)time1 - (float)time2) / 2.0f) / time1));
//            omp_devices[id1].wtile.h -= pix_transm;
//            omp_devices[id1 + 1].wtile.y -= pix_transm;
//            omp_devices[id1 + 1].wtile.y -= pix_transm;
//            omp_devices[id1 + 1].wtile.h += pix_transm;
//          }
//        }
//#      endif
//
//#      if 1
//      {
//        double g_image_total_time = 0;
//        for (int i = 0; i < g_image_time.size(); i++) {
//          g_image_total_time += g_image_time[i];
//        }
//
//        int g_image_time_id = 0;
//        double g_image_time_reminder = 0.0;
//
//        for (int id1 = 0; id1 < devices_size; id1++) {
//          omp_devices[id1].wtile.y = g_image_time_id;
//
//          double dev_time = g_image_time_reminder;
//          double avg_time = g_image_total_time / (double)devices_size;
//
//          for (int i = omp_devices[id1].wtile.y; i < g_image_time.size(); i++) {
//            dev_time += g_image_time[i];
//
//            g_image_time_id = i + 1;
//
//            if (dev_time > avg_time)
//              break;
//          }
//
//          g_image_time_reminder = dev_time - avg_time;
//
//          omp_devices[id1].wtile.h = (devices_size - 1 == id1) ?
//                                         tile_h - omp_devices[id1].wtile.y :
//                                         g_image_time_id - omp_devices[id1].wtile.y;
//        }
//      }
//
//#      endif
//#    endif
//    }
//
/////////////////////////////////////////////////////////////////////
//
/////////////////////////SAMPLES////////////////////////
//#    ifndef ENABLE_INC_SAMPLES
//    // cuda_assert2(
//    // cudaMemset((CU_DEVICE_PTR)scope.get().cuda_mem_map[map_buffer_bin].mem.device_pointer,
//    //               0,
//    //               scope.get().cuda_mem_map[map_buffer_bin].size));
//    // if (id == 0)
//    {
//      size_t sizeBuf_node = tile_w * tile_h * _kg->data.film.pass_stride * sizeof(float);
//      memset((char *)map_buffer_bin, 0, sizeBuf_node);
//    }
//#    endif
//
//    //    omp_devices[id].path_stat_is_done++;
//    //    //
//    //#    pragma omp barrier
//    //#    pragma omp flush
//  }
//  //omp_mem_free(OMP_DEVICE_ID, dev_pixels_node1, pix_size);
//  //omp_mem_free(OMP_DEVICE_ID, dev_pixels_node2, pix_size);
//}
//#  endif  // WITH_CLIENT_CUDA_CPU_STAT

void path_trace_time(DEVICE_PTR kg_bin,
                     DEVICE_PTR map_buffer_bin,
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
  int devices_size = nprocs_cpu;
  int pass_stride = ccl::cuda_devices[0].cuda_mem_map[map_buffer_bin].mem.device_size /
                    (tile_w * tile_h * sizeof(float));

  std::vector<char> buffer_bin(tile_w * tile_h * pass_stride * sizeof(float));
  memset(&buffer_bin[0], 0, buffer_bin.size());

#pragma omp parallel num_threads(devices_size)
  {
    int id = omp_get_thread_num();
    ccl::CUDAContextScope scope(id);

    ccl::PathTraceWorkGPU *cuda_pathTraceWorkGPU = g_pathTraceWorkGPU[id];

    ccl::KernelWorkTile &wtile = scope.get().wtile;
    wtile.start_sample = start_sample;
    wtile.num_samples = end_sample - start_sample;

    // if (scope.get().d_work_tiles == 0) {
    // cu_assert(cuModuleGetFunction(
    //    &scope.get().cuPathTrace, scope.get().cuModule, "kernel_cuda_path_trace"));
    // cu_assert(cuFuncSetCacheConfig(scope.get().cuPathTrace, CU_FUNC_CACHE_PREFER_L1));
    // cu_assert(cuFuncSetAttribute(
    //    scope.get().cuPathTrace, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 0));
    scope.get().kernels.load(&scope.get());

    // cuda_assert2(cudaMalloc(&scope.get().d_work_tiles, sizeof(ccl::KernelWorkTile)));
    // log_alloc(id, sizeof(ccl::KernelWorkTile), "d_work_tiles");

    wtile.x = tile_x;
    wtile.w = tile_w;

    wtile.offset = offset;
    wtile.stride = stride;
    wtile.buffer = (map_buffer_bin != 0) ?
                       (float *)scope.get().cuda_mem_map[map_buffer_bin].mem.device_pointer :
                       0;

    // cuda_assert2(cudaMemcpy(scope.get().d_work_tiles,
    //                       (char *)&wtile,
    //                       sizeof(ccl::KernelWorkTile),
    //                       cudaMemcpyHostToDevice));

    int tile_step_dev = (int)((float)tile_h / (float)devices_size);
    int tile_last_dev = tile_h - (devices_size - 1) * tile_step_dev;

    int tile_y_dev = tile_y + tile_step_dev * id;
    int tile_h_dev = (devices_size - 1 == id) ? tile_last_dev : tile_step_dev;

    wtile.y = tile_y_dev;
    wtile.h = 1;

    // scope.get().num_threads_per_block = 64;
    //{
    //  int min_blocks, num_threads_per_block;
    //  cu_assert(cuOccupancyMaxPotentialBlockSize(
    //      &min_blocks, &num_threads_per_block, scope.get().cuPathTrace, NULL, 0, 0));
    //  scope.get().num_threads_per_block = num_threads_per_block;
    //  printf(
    //      "%d: min_blocks: %d, num_threads_per_block %d\n", id, min_blocks,
    //      num_threads_per_block);
    //}
    //}
    /////////////////////////////////
    unsigned int total_work_size = wtile.w * wtile.h;
    unsigned int num_blocks =
        0;  // ccl::divide_up(total_work_size, scope.get().num_threads_per_block);

    DEVICE_PTR dev_pixels_node = 0;

    cudaStream_t stream_path = scope.get().stream[STREAM_PATH1];

    cudaEvent_t event_path_start = scope.get().event[STREAM_PATH1];
    cudaEvent_t event_path_stop = scope.get().event[STREAM_PATH1 + STREAM_COUNT];

    float *time_path = &scope.get().running_time[STREAM_PATH1];

    double t = omp_get_wtime();

    // int wtile_h = 1;

    for (unsigned int y = tile_y_dev; y < tile_y_dev + tile_h_dev; y++) {
      /* Launch kernel. */
      // void *args[] = {&scope.get().d_work_tiles,
      //                &total_work_size,
      //                &y,
      //                &wtile.h,
      //                &dev_pixels_node,
      //                &start_sample,
      //                &end_sample};

      wtile.y = y;
      wtile.pixel = (char *)dev_pixels_node;
      wtile.start_sample = start_sample;
      wtile.num_samples = end_sample - start_sample;

      cuda_assert2(cudaEventRecord(event_path_start, stream_path));

      // cu_assert(cuLaunchKernel(scope.get().cuPathTrace,
      //                         num_blocks,
      //                         1,
      //                         1,
      //                         scope.get().num_threads_per_block,
      //                         1,
      //                         1,
      //                         0,
      //                         stream_path,
      //                         args,
      //                         0));

      cuda_pathTraceWorkGPU->set_buffer(wtile, g_buffer_passes, tile_w, tile_h);

      ccl::PathTraceWork::RenderStatistics rs;
      cuda_pathTraceWorkGPU->render_samples(
          rs, wtile.start_sample, wtile.num_samples, 0, true, NULL);

      cuda_assert2(cudaEventRecord(event_path_stop, stream_path));
      cuda_assert2(cudaEventSynchronize(event_path_stop));
      cuda_assert2(cudaEventElapsedTime(time_path, event_path_start, event_path_stop));

      line_times[y] = time_path[0] / 1000.0;
    }

    size_t sizeBuf_offset = (offset + tile_x + tile_y_dev * stride) * pass_stride * sizeof(float);
    size_t sizeBuf_size = tile_h_dev * tile_w * pass_stride * sizeof(float);

    cuda_assert2(cudaMemcpy((char *)&buffer_bin[0] + sizeBuf_offset,
                            (char *)scope.get().cuda_mem_map[map_buffer_bin].mem.device_pointer +
                                sizeBuf_offset,
                            sizeBuf_size,
                            cudaMemcpyDeviceToHost));

    /////////////////////////////

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
  //  float *b = (float *)&buffer_bin[0];
  //  printf("GPU: buffer_bin: %f, %f, %f\n", b[0], b[1], b[2]);

  util_save_bmp(offset,
                stride,
                tile_x,
                tile_y,
                tile_h,
                tile_w,
                pass_stride,
                end_sample,
                &buffer_bin[0],
                NULL,
                0);

  fflush(0);
}

#if defined(WITH_CLIENT_CUDA_CPU_STAT2) || defined(WITH_CLIENT_CUDA_CPU_STAT3)

#  ifdef WITH_CLIENT_SHOW_STAT_BVH_LOOP
int cuda_set_mem_advise_by_stat3_credits_done = 0;
#  endif

void path_trace_stat(int numDevice,
                     DEVICE_PTR kg_bin,
                     DEVICE_PTR kg_bin_cpu,
                     DEVICE_PTR buffer_bin,
                     DEVICE_PTR buffer_bin_cpu,
                     DEVICE_PTR pixels_bin,
                     DEVICE_PTR pixels_bin_cpu,
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
                     char *signal_value)
{

  ///////////////////////////////////////////////////////////////////////
  // TESTLB
  const char *env_sample_step = getenv("LB_SAMPLES");
  int sample_step = 1;
  if (env_sample_step != NULL) {
    sample_step = atoi(env_sample_step);
  }

  int sstart = start_sample;
  int send = start_sample + sample_step;

#  ifdef WITH_CLIENT_SHOW_STAT_BVH_LOOP
  if (cuda_set_mem_advise_by_stat3_credits_done == 0) {
#  endif

#  if defined(WITH_CLIENT_CUDA_CPU_STAT2_LB)

    std::vector<double> image_time(tile_h);
    double image_total_time = 0;

    //#      if 0
    //  for (int i = 0; i < 10; i++) {
    //    printf("%d: image_time_sample\n", i);
    //
    //    const char *env_sample_step = getenv("LB_SAMPLES");
    //    if (env_sample_step != NULL) {
    //      sample_step = atoi(env_sample_step);
    //    }
    //
    //    int sstart = start_sample;
    //    int send = start_sample + sample_step;
    //
    //    sstart = i;
    //    send = i + 1;
    //
    //    ////////////////////
    //    omp_path_trace_time(kg_bin_cpu,
    //                        buffer_bin_cpu,
    //                        sstart,
    //                        send,
    //                        tile_x,
    //                        tile_y,
    //                        offset,
    //                        stride,
    //                        tile_h,
    //                        tile_w,
    //                        1,  // ccl::cuda_devices.size(), //omp_get_cpu_threads(),
    //                        &image_time[0]);
    //
    //
    //    for (int i = 0; i < image_time.size(); i++) {
    //      image_total_time += image_time[i];
    //      printf("CPU %d: image_time[i]: %f\n", i, image_time[i]);
    //    }
    //  }
    //  fflush(0);
    //  return;
    //#      endif

#    if 1
    const char *stat_lb_by_data = getenv("STAT_LB_BY_DATA");
    if (stat_lb_by_data == NULL)
      omp_enable_stat(false);

    ////////////////////
    omp_path_trace_time(kg_bin_cpu,
                        buffer_bin_cpu,
                        sstart,
                        send,
                        tile_x,
                        tile_y,
                        offset,
                        stride,
                        tile_h,
                        tile_w,
                        omp_get_cpu_threads(),  // ccl::cuda_devices.size(),
                        &image_time[0]);

    // LB time require second run for accuracy
    if (stat_lb_by_data == NULL) {
      omp_path_trace_time(kg_bin_cpu,
                          buffer_bin_cpu,
                          sstart,
                          send,
                          tile_x,
                          tile_y,
                          offset,
                          stride,
                          tile_h,
                          tile_w,
                          omp_get_cpu_threads(),  // ccl::cuda_devices.size(),
                          &image_time[0]);
    }
    ////////////////////

    for (int i = 0; i < image_time.size(); i++) {
      image_total_time += image_time[i];
      // printf("%d: image_time[i]: %f\n", i, image_time[i]);
    }

#    endif

    int image_time_id = 0;
    double image_time_reminder = 0.0;

    std::string client_tiles = "";

    for (int id = 0; id < ccl::cuda_devices.size(); id++) {
      int tile_y_dev = image_time_id;

      double dev_time = image_time_reminder;
      double avg_time = image_total_time / (double)ccl::cuda_devices.size();

      for (int i = tile_y_dev; i < image_time.size(); i++) {
        dev_time += image_time[i];

        image_time_id = i + 1;

        if (dev_time > avg_time)
          break;
      }

      printf("%d: image_total_time: %f, dev_time: %f, avg_time: %f, stat_lb_by_data: %s\n",
             id,
             image_total_time,
             dev_time,
             avg_time,
             (stat_lb_by_data != NULL) ? stat_lb_by_data : "");

      image_time_reminder = dev_time - avg_time;

      int tile_h_dev = (ccl::cuda_devices.size() - 1 == id) ? tile_h - tile_y_dev :
                                                              image_time_id - tile_y_dev;

      client_tiles += std::to_string(tile_y_dev) + std::string(";") + std::to_string(tile_h_dev) +
                      std::string(";");
    }
    printf("LB: %s\n", client_tiles.c_str());
    setenv("CLIENT_TILES", client_tiles.c_str(), 1);

    const char *lb_cuda_file = getenv("LOAD_BALANCE_CUDA_FILE");
    if (lb_cuda_file != NULL) {
      FILE *f = fopen(lb_cuda_file, "wt+");
      fputs(client_tiles.c_str(), f);
      fclose(f);

      return;
    }

    //  // cpu test
    //  omp_enable_stat(false);
    //  int pix_state[5];
    //  pix_state[0] = 0;  // state buff A
    //  pix_state[1] = 0;  // state buff B
    //  pix_state[2] = 0;  // buf_reset
    //  pix_state[3] = sstart;
    //  pix_state[4] = 1;  // buf_new, change resolution
    //
    //  omp_path_trace(OMP_DEVICE_ID,
    //                 kg_bin_cpu,
    //                 buffer_bin_cpu,
    //                 pixels_bin_cpu,
    //                 sstart,
    //                 send,
    //                 tile_x,
    //                 tile_y,
    //                 offset,
    //                 stride,
    //                 tile_h,
    //                 tile_w,
    //                 sample_finished_omp,
    //                 reqFinished_omp,
    //                 ccl::cuda_devices.size(),
    //                 (char*)pix_state);

    omp_enable_stat(true);

#  endif

#  if defined(ENABLE_LOAD_BALANCE_CUDA) || defined(WITH_CLIENT_CUDA_CPU_STAT3)
    cuda_path_trace_lb(numDevice,
                       buffer_bin,
                       sstart,
                       send,
                       tile_x,
                       tile_y,
                       offset,
                       stride,
                       tile_h,
                       tile_w,
                       ccl::cuda_devices.size());
#  endif

    ///////////////////////////////////////////////////////////////////////

    omp_path_trace_gpu_stat(kg_bin_cpu,
                            (char *)buffer_bin_cpu,
                            (char *)pixels_bin_cpu,
                            sstart,
                            send,
                            tile_x,
                            tile_y,
                            offset,
                            stride,
                            tile_h,
                            tile_w,
                            sample_finished_omp,
                            reqFinished_omp,
                            omp_get_cpu_threads(),
                            ccl::cuda_devices.size());

    // cuda_print_stat_cpu(ccl::cuda_devices.size());
    // set_mem_advise_by_stat_cpu_credits(ccl::cuda_devices.size());
    set_mem_advise_by_stat3_credits(ccl::cuda_devices.size(), 0, true);

#  ifdef WITH_CLIENT_SHOW_STAT_BVH_LOOP
    cuda_set_mem_advise_by_stat3_credits_done = 1;
  }
#  endif

  cuda_path_trace_internal(numDevice,
                           kg_bin,
                           buffer_bin,
                           pixels_bin,
                           start_sample,
                           end_sample,
                           tile_x,
                           tile_y,
                           offset,
                           stride,
                           tile_h,
                           tile_w,
                           tile_h,
                           tile_w,
                           sample_finished_omp,
                           reqFinished_omp,
                           ccl::cuda_devices.size(),
                           signal_value);
}

#elif defined(WITH_CLIENT_CUDA_CPU_STAT)
void path_trace_stat(int numDevice,
                     DEVICE_PTR kg_bin,
                     DEVICE_PTR kg_bin_cpu,
                     DEVICE_PTR buffer_bin,
                     DEVICE_PTR buffer_bin_cpu,
                     DEVICE_PTR pixels_bin,
                     DEVICE_PTR pixels_bin_cpu,
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
                     char *signal_value)
{

#  if defined(WITH_CLIENT_CUDA_CPU_STAT_LB)

  std::vector<double> image_time(tile_h);
  double image_total_time = 0;
  int sample_step = 1;

  //#      if 0
  //    for (int i = 0; i < 10; i++) {
  //      printf("%d: image_time: %f\n", i);
  //
  //      const char *env_sample_step = getenv("LB_SAMPLES");
  //      if (env_sample_step != NULL) {
  //        sample_step = atoi(env_sample_step);
  //      }
  //
  //      int sstart = start_sample;
  //      int send = start_sample + sample_step;
  //
  //      sstart = i;
  //      send = i + 1;
  //
  //      ////////////////////
  //      cuda_path_trace_time(kg_bin,
  //                          buffer_bin,
  //                          sstart,
  //                          send,
  //                          tile_x,
  //                          tile_y,
  //                          offset,
  //                          stride,
  //                          tile_h,
  //                          tile_w,
  //                          1,  // ccl::cuda_devices.size(), //omp_get_cpu_threads(),
  //                          &image_time[0]);
  //
  //      for (int i = 0; i < image_time.size(); i++) {
  //        image_total_time += image_time[i];
  //        printf("GPU %d: image_time[i]: %f\n", i, image_time[i]);
  //      }
  //    }
  //    fflush(0);
  //    //exit(0);
  //    return;
  //#      endif

#    if 1
  const char *env_sample_step = getenv("LB_SAMPLES");
  if (env_sample_step != NULL) {
    sample_step = atoi(env_sample_step);
  }

  int sstart = start_sample;
  int send = start_sample + sample_step;

  ////////////////////
  cuda_path_trace_time(kg_bin,
                       buffer_bin,
                       sstart,
                       send,
                       tile_x,
                       tile_y,
                       offset,
                       stride,
                       tile_h,
                       tile_w,
                       1,  // ccl::cuda_devices.size(), //omp_get_cpu_threads(),
                       &image_time[0]);
  ////////////////////

  for (int i = 0; i < image_time.size(); i++) {
    image_total_time += image_time[i];
    // printf("%d: image_time[i]: %f\n", i, image_time[i]);
  }

#    endif

  int image_time_id = 0;
  double image_time_reminder = 0.0;

  std::string client_tiles = "";

  for (int id = 0; id < ccl::cuda_devices.size(); id++) {
    int tile_y_dev = image_time_id;

    double dev_time = image_time_reminder;
    double avg_time = image_total_time / (double)ccl::cuda_devices.size();

    for (int i = tile_y_dev; i < image_time.size(); i++) {
      dev_time += image_time[i];

      image_time_id = i + 1;

      if (dev_time > avg_time)
        break;
    }

    printf("%d: image_total_time: %f, dev_time: %f, avg_time: %f\n",
           id,
           image_total_time,
           dev_time,
           avg_time);

    image_time_reminder = dev_time - avg_time;

    int tile_h_dev = (ccl::cuda_devices.size() - 1 == id) ? tile_h - tile_y_dev :
                                                            image_time_id - tile_y_dev;

    client_tiles += std::to_string(tile_y_dev) + std::string(";") + std::to_string(tile_h_dev) +
                    std::string(";");
  }
  printf("LB: %s\n", client_tiles.c_str());
  setenv("CLIENT_TILES", client_tiles.c_str(), 1);

#  endif

#  ifdef ENABLE_LOAD_BALANCE_CUDA
  const char *env_sample_step = getenv("LB_SAMPLES");
  int sample_step = 1;
  if (env_sample_step != NULL) {
    sample_step = atoi(env_sample_step);
  }

  int sstart = start_sample;
  int send = start_sample + sample_step;

  cuda_path_trace_lb(numDevice,
                     buffer_bin,
                     sstart,
                     send,
                     tile_x,
                     tile_y,
                     offset,
                     stride,
                     tile_h,
                     tile_w,
                     ccl::cuda_devices.size());
#  endif

  bool req_exit = false;

#  pragma omp parallel num_threads(2)
  {

    int tid = omp_get_thread_num();
    if (tid == 0) {
      while (!req_exit) {

        omp_path_trace_gpu_stat(kg_bin_cpu,
                                (char *)buffer_bin_cpu,
                                (char *)pixels_bin_cpu,
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
                                ccl::cuda_devices.size());
        if (req_exit)
          break;

        set_mem_advise_by_stat3_credits(ccl::cuda_devices.size(), 0, true);
        // cuda_print_stat_cpu(ccl::cuda_devices.size());

#  pragma omp flush
      }
    }
    if (tid == 1) {

      cuda_path_trace_internal(numDevice,
                               kg_bin,
                               buffer_bin,
                               pixels_bin,
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
                               ccl::cuda_devices.size(),
                               signal_value);

      req_exit = true;

#  pragma omp flush
    }
  }
}
#endif

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
                char *signal_value)
{
#ifdef ENABLE_LOAD_BALANCE_CUDA
  const char *env_sample_step = getenv("LB_SAMPLES");
  int sample_step = 1;
  if (env_sample_step != NULL) {
    sample_step = atoi(env_sample_step);
  }

  int sstart = start_sample;
  int send = start_sample + sample_step;
  cuda_path_trace_lb(numDevice,
                     buffer_bin,
                     sstart,
                     send,
                     tile_x,
                     tile_y,
                     offset,
                     stride,
                     tile_h,
                     tile_w,
                     ccl::cuda_devices.size());
#endif

#ifdef WITH_CUDA_STATv2_LB
  const char *env_sample_step = getenv("LB_SAMPLES");
  int sample_step = 1;
  if (env_sample_step != NULL) {
    sample_step = atoi(env_sample_step);
  }

  int sstart = start_sample;
  int send = start_sample + sample_step;

  std::string client_tiles = "";

  std::vector<double> image_time(tile_h);
  double image_total_time = 0;

  const char *stat_lb_by_data = getenv("STAT_LB_BY_DATA");
  if (stat_lb_by_data != NULL) {

    std::string names = std::string(stat_lb_by_data);

    int row = 0;
    while (row < tile_h) {

      client_tiles = "";

      for (int id = 0; id < ccl::cuda_devices.size(); id++) {
        int tile_y_dev = row + id;
        int tile_h_dev = (row + id < tile_h) ? 1 : 0;

        client_tiles += std::to_string(tile_y_dev) + std::string(";") +
                        std::to_string(tile_h_dev) + std::string(";");
      }

      // printf("LB: %s\n", client_tiles.c_str());
      setenv("CLIENT_TILES", client_tiles.c_str(), 1);

      cuda_path_trace_internal(numDevice,
                               kg_bin,
                               buffer_bin,
                               pixels_bin,
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
                               ccl::cuda_devices.size(),
                               NULL);

      ////////////////////

      for (int id = 0; id < ccl::cuda_devices.size(); id++) {
        int tile_y_dev = row + id;

        if (tile_y_dev < tile_h) {
          std::map<std::string, ccl::CUDADevice::CUDAMem>::iterator it_stat;
          for (it_stat = ccl::cuda_devices[id].cuda_stat_map.begin();
               it_stat != ccl::cuda_devices[id].cuda_stat_map.end();
               it_stat++) {
            if (names == "all" || names.find(it_stat->first) != std::string::npos ||
                it_stat->first.find(names) != std::string::npos) {
              ccl::CUDADevice::CUDAMem *cm = &it_stat->second;
              unsigned long long int *c_data = (unsigned long long int *)&cm->map_host_pointer[0];

              for (size_t c = 0; c < cm->mem.data_size; c++) {
                unsigned long long int cd = c_data[c];
                image_time[tile_y_dev] += cd;
              }
            }
          }
        }

        ccl::CUDAContextScope scope(id);
        std::map<std::string, ccl::CUDADevice::CUDAMem>::iterator it_stat;
        for (it_stat = scope.get().cuda_stat_map.begin();
             it_stat != scope.get().cuda_stat_map.end();
             it_stat++) {
          ccl::CUDADevice::CUDAMem *cm = &it_stat->second;
          cuda_assert2(cudaMemset((CU_DEVICE_PTR)cm->device_pointer, 0, cm->mem.device_size));
        }
      }
      row += ccl::cuda_devices.size();
    }
  }

  ////////////////////
  for (int i = 0; i < image_time.size(); i++) {
    image_total_time += image_time[i];
  }

  client_tiles = "";

  int image_time_id = 0;
  double image_time_reminder = 0.0;

  for (int id = 0; id < ccl::cuda_devices.size(); id++) {
    int tile_y_dev = image_time_id;

    double dev_time = image_time_reminder;
    double avg_time = image_total_time / (double)ccl::cuda_devices.size();

    for (int i = tile_y_dev; i < image_time.size(); i++) {
      dev_time += image_time[i];

      image_time_id = i + 1;

      if (dev_time > avg_time)
        break;
    }

    printf("%d: image_total_time: %f, dev_time: %f, avg_time: %f, stat_lb_by_data: %s\n",
           id,
           image_total_time,
           dev_time,
           avg_time,
           (stat_lb_by_data != NULL) ? stat_lb_by_data : "");

    image_time_reminder = dev_time - avg_time;

    int tile_h_dev = (ccl::cuda_devices.size() - 1 == id) ? tile_h - tile_y_dev :
                                                            image_time_id - tile_y_dev;

    client_tiles += std::to_string(tile_y_dev) + std::string(";") + std::to_string(tile_h_dev) +
                    std::string(";");
  }
  printf("LB: %s\n", client_tiles.c_str());
  setenv("CLIENT_TILES", client_tiles.c_str(), 1);

  const char *lb_cuda_file = getenv("LOAD_BALANCE_CUDA_FILE");
  if (lb_cuda_file != NULL) {
    FILE *f = fopen(lb_cuda_file, "wt+");
    fputs(client_tiles.c_str(), f);
    fclose(f);

    return;
  }

#endif
  path_trace_internal(numDevice,
                      kg_bin,
                      buffer_bin,
                      pixels_bin,
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
                      ccl::cuda_devices.size(),
                      signal_value);
}

void alloc_kg(int numDevice)
{

  // DEVICE_PTR kg_bin = 0;

#if 1
  const char *fill_mem = getenv("FILL_MEMORY_MB");
  if (fill_mem != NULL) {
    size_t fill_mem_size = atol(fill_mem) * 1024L * 1024L;
    DEVICE_PTR map_id = mem_alloc(-1, "FILL_MEMORY", 0, fill_mem_size);
    mem_zero("FILL_MEMORY_MB", -1, map_id, fill_mem_size);
  }
#endif

  if (dev_kernel_data != NULL) {
    delete dev_kernel_data;
  }

  dev_kernel_data = new DevKernelData();
  dev_kernel_data->kernel_globals_cpu = 0;

  // return (DEVICE_PTR)kg_bin;
}

void free_kg(int numDevice, DEVICE_PTR kg_bin)
{
}

DEVICE_PTR mem_alloc(
    int numDevice, const char *name, char *mem, size_t memSize, bool spec_setts, bool alloc_host)
{
  DEVICE_PTR dmem = 0;

  if (g_caching_enabled == CACHING_PREVIEW && skip_caching_by_name(name)) {
    if (dev_kernel_data->str2ptr_map.find(std::string(name)) !=
        dev_kernel_data->str2ptr_map.end()) {
      DEVICE_PTR id2 = dev_kernel_data->str2ptr_map[std::string(name)];
      return dev_kernel_data->ptr2ptr_map[id2];
    }
  }

  dmem = (DEVICE_PTR)generic_alloc(name, memSize, 0, spec_setts, alloc_host);

  if (!strcmp("client_buffer_passes", name)) {
    g_buffer_passes_d = dmem;
  }

  return dmem;
}

void set_buffer_passes(char *mem, size_t memSize)
{
  g_buffer_passes.resize(memSize / sizeof(client_buffer_passes));
  memcpy(g_buffer_passes.data(), mem, memSize);
}

void mem_copy_to(int numDevice, char *mem, DEVICE_PTR map_id, size_t memSize, char *signal_value)
{
  // printf("cuda_mem_copy_to: %lld, %lld\n", map_id, memSize);

  generic_copy_to(mem, map_id, memSize);

  if (map_id == g_buffer_passes_d) {
    g_buffer_passes.resize(memSize / sizeof(client_buffer_passes));
    memcpy(g_buffer_passes.data(), mem, memSize);
  }
}

void mem_copy_from(
    int numDevice, DEVICE_PTR map_id, char *mem, size_t offset, size_t memSize, char *signal_value)
{
  size_t pix_type_size = SIZE_UCHAR4;

  for (int id = 0; id < ccl::cuda_devices.size(); id++) {
    ccl::CUDAContextScope scope(id);

    if (scope.get().cuda_mem_map[map_id].uni_mem) {
      cuda_assert2(cudaMemcpy(
          mem + offset,
          (CU_DEVICE_PTR)((char *)scope.get().cuda_mem_map[map_id].mem.device_pointer + offset),
          memSize,
          cudaMemcpyDeviceToHost));
      break;
    }
    else {
      ccl::KernelWorkTile &wtile = scope.get().wtile;
      // cuda_assert2(cuMemcpyDtoH((char *)&wtile, scope.get().d_work_tiles,
      // sizeof(ccl::KernelWorkTile)));

#ifndef ENABLE_LOAD_BALANCEv2
      size_t offsetPix_dev = (wtile.x + wtile.y * wtile.stride) * pix_type_size;
      size_t sizePix_dev = wtile.h * wtile.w * pix_type_size;

      if (signal_value != NULL) {
        size_t pass_stride_float = *((size_t *)signal_value);
        offsetPix_dev = (wtile.x + wtile.y * wtile.stride) * pass_stride_float;
        sizePix_dev = wtile.h * wtile.w * pass_stride_float;
      }
#else
      size_t offsetPix_dev = wtile.y * pix_type_size;
      size_t sizePix_dev = wtile.h * pix_type_size;

      if (signal_value != NULL) {
        size_t pass_stride_float = *((size_t *)signal_value);
        offsetPix_dev = wtile.y * pass_stride_float;
        sizePix_dev = wtile.h * pass_stride_float;
      }
#endif

      cuda_assert2(
          cudaMemcpy(mem + offset + offsetPix_dev,
                     (CU_DEVICE_PTR)((char *)scope.get().cuda_mem_map[map_id].mem.device_pointer +
                                     offsetPix_dev),
                     sizePix_dev,
                     cudaMemcpyDeviceToHost));
    }
  }
}

void mem_zero(const char *name, int numDevice, DEVICE_PTR map_id, size_t memSize)
{
  // printf("cuda_mem_zero: %lld, %lld\n", map_id, memSize);

  for (int id = 0; id < ccl::cuda_devices.size(); id++) {
    ccl::CUDAContextScope scope(id);
    cuda_assert2(cudaMemset(
        (CU_DEVICE_PTR)scope.get().cuda_mem_map[map_id].mem.device_pointer, 0, memSize));

    if (scope.get().cuda_mem_map[map_id].mem.host_pointer != NULL) {
      memset(scope.get().cuda_mem_map[map_id].mem.host_pointer, 0, memSize);
    }

    if (scope.get().cuda_mem_map[map_id].uni_mem)
      break;
  }
}

void mem_free(const char *name, int numDevice, DEVICE_PTR map_id, size_t memSize)
{
  if (/*!skip_caching_by_name(name) &&*/ g_caching_enabled != CACHING_DISABLED)
    return;

  generic_free(map_id, memSize);

  // dev_kernel_data->ptr2ptr_map.erase(map_id);
}

void tex_free(
    int numDevice, DEVICE_PTR kg_bin, const char *name_bin, DEVICE_PTR map_id, size_t memSize)
{
  if (/* !skip_caching_by_name(name_bin) &&*/ g_caching_enabled != CACHING_DISABLED)
    return;

  const ccl::CUDADevice::CUDAMem &cmem = ccl::cuda_devices[0].cuda_mem_map[map_id];

  if (cmem.texobject) {
    /* Free bindless texture. */
    for (int id = 0; id < ccl::cuda_devices.size(); id++) {
      ccl::CUDAContextScope scope(id);

      // always on each GPU
      cuTexObjectDestroy(scope.get().cuda_mem_map[map_id].texobject);

      // if (scope.get().cuda_mem_map[map_id].uni_mem)
      //  break;
    }
  }

  if (cmem.array) {
    /* Free array. */
    for (int id = 0; id < ccl::cuda_devices.size(); id++) {
      ccl::CUDAContextScope scope(id);
      cuArrayDestroy(scope.get().cuda_mem_map[map_id].array);
      scope.get().cuda_mem_map.erase(scope.get().cuda_mem_map.find(map_id));
    }
  }
  else {
    generic_free(map_id, memSize);
  }

  // dev_kernel_data->ptr2ptr_map.erase(map_id);
}

int get_pass_stride(int numDevice, DEVICE_PTR kg)
{
  return 0;  // ((KernelGlobals*)kg)->data.film.pass_stride;
}

enum CudaDataType {
  TYPE_UNKNOWN,
  TYPE_UCHAR,
  TYPE_UINT16,
  TYPE_UINT,
  TYPE_INT,
  TYPE_FLOAT,
  TYPE_HALF,
  TYPE_UINT64,
};

static size_t cuda_datatype_size(CudaDataType datatype)
{
  switch (datatype) {
    case TYPE_UNKNOWN:
      return 1;
    case TYPE_UCHAR:
      return sizeof(uint8_t);
    case TYPE_FLOAT:
      return sizeof(float);
    case TYPE_UINT:
      return sizeof(uint32_t);
    case TYPE_UINT16:
      return sizeof(uint16_t);
    case TYPE_INT:
      return sizeof(int);
    case TYPE_HALF:
      return sizeof(unsigned short);
    case TYPE_UINT64:
      return sizeof(uint64_t);
    default:
      return 0;
  }
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
  // printf("omp_tex_info: %s, %.3f\n", name, (float)memSize / (1024.0f * 1024.0f));

  DEVICE_PTR map_id = 0;

#ifdef WITH_CUDA_CPUIMAGE

  map_id = generic_alloc(name, memSize, 0, unimem_flag);
  // generic_copy_to(mem, map_id, memSize);

#else

  /* Image Texture Storage */
  CUarray_format_enum format;
  switch (data_type) {
    case TYPE_UCHAR:
      format = CU_AD_FORMAT_UNSIGNED_INT8;
      break;
    case TYPE_UINT16:
      format = CU_AD_FORMAT_UNSIGNED_INT16;
      break;
    case TYPE_UINT:
      format = CU_AD_FORMAT_UNSIGNED_INT32;
      break;
    case TYPE_INT:
      format = CU_AD_FORMAT_SIGNED_INT32;
      break;
    case TYPE_FLOAT:
      format = CU_AD_FORMAT_FLOAT;
      break;
    case TYPE_HALF:
      format = CU_AD_FORMAT_HALF;
      break;
    default:
      printf("assert: CUarray_format_enum\n");
      return 0;
  }

  // ccl::CUDADevice::CUDAMem *cmem = NULL;
  // CUarray array_3d = NULL;
  size_t dsize = cuda_datatype_size((CudaDataType)data_type);
  size_t src_pitch = data_width * dsize * data_elements;
  size_t dst_pitch = src_pitch;

  if (data_depth > 1) {
    /* 3D texture using array, there is no API for linear memory. */
    CUDA_ARRAY3D_DESCRIPTOR desc;

    desc.Width = data_width;
    desc.Height = data_height;
    desc.Depth = data_depth;
    desc.Format = format;
    desc.NumChannels = data_elements;
    desc.Flags = 0;

    for (int id = 0; id < ccl::cuda_devices.size(); id++) {

      ccl::CUDAContextScope scope(id);

      CUarray array_3d = NULL;
      // TODO UNIMEM
      // printf("cuArray3DCreate: %s, %.3f, %zu\n", name, (float)memSize / (1024.0f * 1024.0f));
      cu_assert(cuArray3DCreate(&array_3d, &desc));

      if (id == 0)
        map_id = (DEVICE_PTR)array_3d;

      ccl::CUDADevice::CUDAMem *cmem = &scope.get().cuda_mem_map[map_id];
      // CUDA_MEMCPY3D param;
      CUDA_MEMCPY3D *param = &cmem->param3D;
      memset(param, 0, sizeof(CUDA_MEMCPY3D));
      param->dstMemoryType = CU_MEMORYTYPE_ARRAY;
      param->srcMemoryType = CU_MEMORYTYPE_HOST;
      param->srcHost = mem;
      param->srcPitch = src_pitch;
      param->WidthInBytes = param->srcPitch;
      param->Height = data_height;
      param->Depth = data_depth;
      param->dstArray = array_3d;

      // cu_assert(cuMemcpy3D(&param));

      cmem->texobject = 0;
      cmem->array = array_3d;
      cmem->tex_type = 3;

      if (cmem->uni_mem)
        break;
    }
  }
  else if (data_height > 0) {
    /* 2D texture, using pitch aligned linear memory. */
    size_t pitch_padding = 0;
    {
      ccl::CUDAContextScope scope(0);
      CU_DEVICE_PTR device_pointer = 0;

      int alignment = 0;
      cuda_assert2(cudaDeviceGetAttribute(
          &alignment, cudaDevAttrTexturePitchAlignment, scope.get().cuDevice));

      // cu_assert(cuDeviceGetAttribute(
      //    &alignment, CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT, scope.get().cuDevice));

      dst_pitch = cuda_align_up(src_pitch, alignment);
      size_t dst_size = dst_pitch * data_height;

      pitch_padding = dst_size - memSize;
    }

    map_id = generic_alloc(name, memSize, pitch_padding, unimem_flag);

    for (int id = 0; id < ccl::cuda_devices.size(); id++) {
      ccl::CUDAContextScope scope(id);

      ccl::CUDADevice::CUDAMem *cmem = &scope.get().cuda_mem_map[map_id];

      scope.get().cuda_mem_map[map_id].tex_type = 2;

      CUDA_MEMCPY2D *param = &cmem->param2D;  // new CUDA_MEMCPY2D();
      memset(param, 0, sizeof(CUDA_MEMCPY2D));
      param->dstMemoryType = CU_MEMORYTYPE_DEVICE;
      param->dstPitch = dst_pitch;
      param->srcMemoryType = CU_MEMORYTYPE_HOST;
      // param->srcHost = mem;
      param->srcPitch = src_pitch;
      param->WidthInBytes = param->srcPitch;
      param->Height = data_height;
      param->dstDevice = scope.get().cuda_mem_map[map_id].mem.device_pointer;

      // cuda_memcpy2Ds.push_back(param);
      // cmem->param2D = param;
      // cmem->image_data_count = data_width * data_height * data_depth;
      // cu_assert(cuMemcpy2DUnaligned(param));

      // if (scope.get().cuda_mem_map[map_id].uni_mem)
      //  break;
    }
  }
  else {
    /* 1D texture, using linear memory. */
    map_id = generic_alloc(name, memSize, 0, unimem_flag);

    // generic_copy_to(mem, map_id, memSize);
  }

  for (int id = 0; id < ccl::cuda_devices.size(); id++) {
    ccl::CUDAContextScope scope(id);

    ccl::CUDADevice::CUDAMem *cmem = &scope.get().cuda_mem_map[map_id];

    CUDA_RESOURCE_DESC *resDesc = &cmem->resDesc;
    memset(resDesc, 0, sizeof(CUDA_RESOURCE_DESC));

    if (data_depth > 1) {
      resDesc->resType = CU_RESOURCE_TYPE_ARRAY;
      // resDesc.res.array.hArray = array_3d;
      resDesc->flags = 0;
    }
    else if (data_height > 0) {
      resDesc->resType = CU_RESOURCE_TYPE_PITCH2D;
      // resDesc.res.pitch2D.devPtr = device_pointer;
      resDesc->res.pitch2D.format = format;
      resDesc->res.pitch2D.numChannels = data_elements;
      resDesc->res.pitch2D.height = data_height;
      resDesc->res.pitch2D.width = data_width;
      resDesc->res.pitch2D.pitchInBytes = dst_pitch;
    }
    else {
      resDesc->resType = CU_RESOURCE_TYPE_LINEAR;
      // resDesc.res.linear.devPtr = device_pointer;
      resDesc->res.linear.format = format;
      resDesc->res.linear.numChannels = data_elements;
      resDesc->res.linear.sizeInBytes = memSize;
    }

    CUaddress_mode address_mode = CU_TR_ADDRESS_MODE_WRAP;
    switch (extension) {
      case ccl::EXTENSION_REPEAT:
        address_mode = CU_TR_ADDRESS_MODE_WRAP;
        break;
      case ccl::EXTENSION_EXTEND:
        address_mode = CU_TR_ADDRESS_MODE_CLAMP;
        break;
      case ccl::EXTENSION_CLIP:
        address_mode = CU_TR_ADDRESS_MODE_BORDER;
        break;
      default:
        printf("assert: CUaddress_mode\n");
        break;
    }

    CUfilter_mode filter_mode;
    if (interpolation == ccl::INTERPOLATION_CLOSEST) {
      filter_mode = CU_TR_FILTER_MODE_POINT;
    }
    else {
      filter_mode = CU_TR_FILTER_MODE_LINEAR;
    }

    CUDA_TEXTURE_DESC *texDesc = &cmem->texDesc;
    memset(texDesc, 0, sizeof(CUDA_TEXTURE_DESC));
    texDesc->addressMode[0] = address_mode;
    texDesc->addressMode[1] = address_mode;
    texDesc->addressMode[2] = address_mode;
    texDesc->filterMode = filter_mode;
    texDesc->flags = CU_TRSF_NORMALIZED_COORDINATES;

    if (data_depth > 1) {
      resDesc->res.array.hArray = scope.get().cuda_mem_map[map_id].array;
    }
    else if (data_height > 0) {
      resDesc->res.pitch2D.devPtr = scope.get().cuda_mem_map[map_id].mem.device_pointer;
    }
    else {
      resDesc->res.linear.devPtr = scope.get().cuda_mem_map[map_id].mem.device_pointer;
    }

    // cuTexObjectCreate always on each GPU
    // if (data_type != 8 /*IMAGE_DATA_TYPE_NANOVDB_FLOAT*/ &&
    //    data_type != 9 /*IMAGE_DATA_TYPE_NANOVDB_FLOAT3*/) {
    //  cu_assert(cuTexObjectCreate(
    //      &scope.get().cuda_mem_map[map_id].texobject, &resDesc, &texDesc, NULL));
    //}
    // if (scope.get().cuda_mem_map[map_id].uni_mem)
    //  break;
  }

#endif

  return map_id;
}

void tex_info_copy(const char *name,
                   char *mem,
                   DEVICE_PTR map_id,
                   size_t memSize,
                   int data_type,
                   bool check_unimem)
{

#if defined(WITH_CLIENT_CUDA_CPU_STAT2)
  if (check_unimem && check_mem_advise_name(name, false))
    return;
#endif

#ifdef WITH_CUDA_CPUIMAGE

  // map_id = generic_alloc(name, memSize);
  generic_copy_to(mem, map_id, memSize);

#else
  int tex_type = ccl::cuda_devices[0].cuda_mem_map[map_id].tex_type;

  if (tex_type == 3) {
    // 3D texture using array, there is no API for linear memory.
    for (int id = 0; id < ccl::cuda_devices.size(); id++) {
      ccl::CUDAContextScope scope(id);

      ccl::CUDADevice::CUDAMem *cmem = &scope.get().cuda_mem_map[map_id];
      cmem->param3D.srcHost = mem;
      // cu_assert(cuMemcpy2DUnaligned(&cmem->param2D));
      cu_assert(cuMemcpy3D(&cmem->param3D));

#  ifdef WITH_CUDA_STAT
      if (scope.get().cuda_mem_map[map_id].host_pointer != NULL) {
        memcpy(&scope.get().cuda_mem_map[map_id].host_pointer[0], mem, memSize);
      }
#  endif

      // cuTexObjectCreate always on each GPU
      // if (data_type != 8 /*IMAGE_DATA_TYPE_NANOVDB_FLOAT*/ &&
      //     data_type != 9 /*IMAGE_DATA_TYPE_NANOVDB_FLOAT3*/) {
      //   cu_assert(cuTexObjectCreate(
      //       &scope.get().cuda_mem_map[map_id].texobject, &cmem->resDesc, &cmem->texDesc, NULL));
      // }

      if (scope.get().cuda_mem_map[map_id].uni_mem)
        break;
    }
  }
  else if (tex_type == 2) {
    // 2D texture, using pitch aligned linear memory.
    for (int id = 0; id < ccl::cuda_devices.size(); id++) {
      ccl::CUDAContextScope scope(id);

      ccl::CUDADevice::CUDAMem *cmem = &scope.get().cuda_mem_map[map_id];
      cmem->param2D.srcHost = mem;
      cu_assert(cuMemcpy2DUnaligned(&cmem->param2D));

#  ifdef WITH_CUDA_STAT
      if (scope.get().cuda_mem_map[map_id].host_pointer != NULL) {
        memcpy(&scope.get().cuda_mem_map[map_id].host_pointer[0], mem, memSize);
      }
#  endif

      // cuTexObjectCreate always on each GPU
      if (data_type != 8 /*IMAGE_DATA_TYPE_NANOVDB_FLOAT*/ &&
          data_type != 9 /*IMAGE_DATA_TYPE_NANOVDB_FLOAT3*/) {
        cu_assert(cuTexObjectCreate(
            &scope.get().cuda_mem_map[map_id].texobject, &cmem->resDesc, &cmem->texDesc, NULL));
      }

      if (scope.get().cuda_mem_map[map_id].uni_mem)
        break;
    }
  }
  else {
    /* 1D texture, using linear memory. */
    generic_copy_to(mem, map_id, memSize);
  }

#endif
}

void tex_info_copy_async(
    const char *name, char *mem, DEVICE_PTR map_id, size_t memSize, bool check_unimem)
{

#if defined(WITH_CLIENT_CUDA_CPU_STAT2)
  if (check_unimem && check_mem_advise_name(name, false))
    return;
#endif

  // map_id = generic_alloc(name, memSize);
  generic_copy_to_async(mem, map_id, memSize);
}

DEVICE_PTR get_data(DEVICE_PTR kg_bin)
{
  return (DEVICE_PTR)&cuda_kernel_data[0];
}

#if defined(WITH_CLIENT_RENDERENGINE_VR) || \
    (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
DEVICE_PTR get_data_right(DEVICE_PTR kg_bin)
{
  return (DEVICE_PTR)&cuda_kernel_data_right[0];
}
#endif

size_t get_size_data(DEVICE_PTR kg_bin)
{
  return cuda_kernel_data.size();
}

float *get_camera_matrix(DEVICE_PTR kg_bin)
{
  return (float *)&cuda_kernel_data[48];
}

float *get_camera_rastertocamera(DEVICE_PTR kg_bin)
{
  return (float *)&cuda_kernel_data[96];
}

float *get_camera_dx(DEVICE_PTR kg_bin)
{
  return (float *)&cuda_kernel_data[160];
}

float *get_camera_dy(DEVICE_PTR kg_bin)
{
  return (float *)&cuda_kernel_data[176];
}

float *get_camera_width(DEVICE_PTR kg_bin)
{
  return (float *)&cuda_kernel_data[236];
}

float *get_camera_height(DEVICE_PTR kg_bin)
{
  return (float *)&cuda_kernel_data[240];
}

void anim_step(int numDevice, DEVICE_PTR kg_bin, char *data_bin, int s)
{
}

int get_cpu_threads()
{
  return ccl::cuda_devices.size();
}

//// CCL_NAMESPACE_BEGIN
// void camera_compute_auto_viewplane(CameraType &type,
//                                        BoundBox2D &viewplane,
//                                        int width,
//                                        int height)
//{
//  if (type == CAMERA_PANORAMA) {
//    viewplane.left = 0.0f;
//    viewplane.right = 1.0f;
//    viewplane.bottom = 0.0f;
//    viewplane.top = 1.0f;
//  }
//  else {
//    float aspect = (float)width / (float)height;
//    if (width >= height) {
//      viewplane.left = -aspect;
//      viewplane.right = aspect;
//      viewplane.bottom = -1.0f;
//      viewplane.top = 1.0f;
//    }
//    else {
//      viewplane.left = -1.0f;
//      viewplane.right = 1.0f;
//      viewplane.bottom = -1.0f / aspect;
//      viewplane.top = 1.0f / aspect;
//    }
//  }
//}
//
// void camera_update(CameraType type,
//                        float fov,
//                        float nearclip,
//                        float farclip,
//                        Transform &matrix,
//                        int width,
//                        int height,
//                        KernelCamera *kcam)
//{
//  // in
//  // CameraType type = CAMERA_PERSPECTIVE;
//  // float fov, nearclip, farclip;
//  // Transform matrix;
//  // int width, height;
//  // out
//  /*ProjectionTransform kcam_rastertocamera;
//  Transform kcam_cameratoworld;
//  float4 kcam_dx, kcam_dy;*/
//
//  // temp
//  ProjectionTransform rastertocamera, cameratoraster, screentoworld, rastertoworld, ndctoworld;
//
//  Transform worldtocamera, cameratoworld;
//  ProjectionTransform worldtoscreen, worldtondc, worldtoraster;
//
//  _float3 dx, dy;
//  BoundBox2D viewplane, viewport_camera_border;
//
//  viewport_camera_border.right = 1.0f;
//  viewport_camera_border.top = 1.0f;
//  cuda_camera_compute_auto_viewplane(type, viewplane, width, height);
//
//  // Scene::MotionType need_motion = scene->need_motion();
//
//  // if (previous_need_motion != need_motion) {
//  //  /* scene's motion model could have been changed since previous device
//  //   * camera update this could happen for example in case when one render
//  //   * layer has got motion pass and another not */
//  //  need_device_update = true;
//  //}
//
//  // if (!need_update)
//  //  return;
//
//  /* Full viewport to camera border in the viewport. */
//  Transform fulltoborder = transform_from_viewplane(viewport_camera_border);
//  Transform bordertofull = transform_inverse(fulltoborder);
//
//  /* ndc to raster */
//  Transform ndctoraster = transform_scale(width, height, 1.0f) * bordertofull;
//  // Transform full_ndctoraster = transform_scale(full_width, full_height, 1.0f) *
//  bordertofull;
//
//  /* raster to screen */
//  Transform screentondc = fulltoborder * transform_from_viewplane(viewplane);
//
//  Transform screentoraster = ndctoraster * screentondc;
//  Transform rastertoscreen = transform_inverse(screentoraster);
//  // Transform full_screentoraster = full_ndctoraster * screentondc;
//  // Transform full_rastertoscreen = transform_inverse(full_screentoraster);
//
//  /* screen to camera */
//  ProjectionTransform cameratoscreen;
//  if (type == CAMERA_PERSPECTIVE)
//    cameratoscreen = projection_perspective(fov, nearclip, farclip);
//  else if (type == CAMERA_ORTHOGRAPHIC)
//    cameratoscreen = projection_orthographic(nearclip, farclip);
//  else
//    cameratoscreen = projection_identity();
//
//  ProjectionTransform screentocamera = projection_inverse(cameratoscreen);
//
//  rastertocamera = screentocamera * rastertoscreen;
//  // full_rastertocamera = screentocamera * full_rastertoscreen;
//  cameratoraster = screentoraster * cameratoscreen;
//
//  cameratoworld = matrix;
//  screentoworld = cameratoworld * screentocamera;
//  rastertoworld = cameratoworld * rastertocamera;
//  ndctoworld = rastertoworld * ndctoraster;
//
//  /* note we recompose matrices instead of taking inverses of the above, this
//   * is needed to avoid inverting near degenerate matrices that happen due to
//   * precision issues with large scenes */
//  worldtocamera = transform_inverse(matrix);
//  worldtoscreen = cameratoscreen * worldtocamera;
//  worldtondc = screentondc * worldtoscreen;
//  worldtoraster = ndctoraster * worldtondc;
//
//  /* differentials */
//  if (type == CAMERA_ORTHOGRAPHIC) {
//    dx = transform_perspective_direction(&rastertocamera, make_float3(1, 0, 0));
//    dy = (&rastertocamera, make_float3(0, 1, 0));
//    // full_dx = transform_perspective_direction(&full_rastertocamera, make_float3(1, 0, 0));
//    // full_dy = transform_perspective_direction(&full_rastertocamera, make_float3(0, 1, 0));
//  }
//  else if (type == CAMERA_PERSPECTIVE) {
//    dx = transform_perspective(&rastertocamera, make_float3(1, 0, 0)) -
//         transform_perspective(&rastertocamera, make_float3(0, 0, 0));
//    dy = transform_perspective(&rastertocamera, make_float3(0, 1, 0)) -
//         transform_perspective(&rastertocamera, make_float3(0, 0, 0));
//    // full_dx = transform_perspective(&full_rastertocamera, make_float3(1, 0, 0)) -
//    //          transform_perspective(&full_rastertocamera, make_float3(0, 0, 0));
//    // full_dy = transform_perspective(&full_rastertocamera, make_float3(0, 1, 0)) -
//    //          transform_perspective(&full_rastertocamera, make_float3(0, 0, 0));
//  }
//  else {
//    dx = make_float3(0.0f, 0.0f, 0.0f);
//    dy = make_float3(0.0f, 0.0f, 0.0f);
//  }
//
//  dx = transform_direction(&cameratoworld, dx);
//  dy = transform_direction(&cameratoworld, dy);
//  // full_dx = transform_direction(&cameratoworld, full_dx);
//  // full_dy = transform_direction(&cameratoworld, full_dy);
//
//  // if (type == CAMERA_PERSPECTIVE) {
//  //  float3 v = transform_perspective(&full_rastertocamera,
//  //                                   make_float3(full_width, full_height, 1.0f));
//
//  //  frustum_right_normal = normalize(make_float3(v.z, 0.0f, -v.x));
//  //  frustum_top_normal = normalize(make_float3(0.0f, v.z, -v.y));
//  //}
//
//  ///* Compute kernel camera data. */
//  // KernelCamera *kcam = &kernel_camera;
//
//  /* store matrices */
//  kcam->screentoworld = screentoworld;
//  kcam->rastertoworld = rastertoworld;
//  kcam->rastertocamera = rastertocamera;
//  kcam->cameratoworld = cameratoworld;
//  kcam->worldtocamera = worldtocamera;
//  kcam->worldtoscreen = worldtoscreen;
//  kcam->worldtoraster = worldtoraster;
//  kcam->worldtondc = worldtondc;
//  kcam->ndctoworld = ndctoworld;
//
//  ///* camera motion */
//  // kcam->num_motion_steps = 0;
//  // kcam->have_perspective_motion = 0;
//  // kernel_camera_motion.clear();
//
//  ///* Test if any of the transforms are actually different. */
//  // bool have_motion = false;
//  // for (size_t i = 0; i < motion.size(); i++) {
//  //  have_motion = have_motion || motion[i] != matrix;
//  //}
//
//  // if (need_motion == Scene::MOTION_PASS) {
//  //  /* TODO(sergey): Support perspective (zoom, fov) motion. */
//  //  if (type == CAMERA_PANORAMA) {
//  //    if (have_motion) {
//  //      kcam->motion_pass_pre = transform_inverse(motion[0]);
//  //      kcam->motion_pass_post = transform_inverse(motion[motion.size() - 1]);
//  //    }
//  //    else {
//  //      kcam->motion_pass_pre = kcam->worldtocamera;
//  //      kcam->motion_pass_post = kcam->worldtocamera;
//  //    }
//  //  }
//  //  else {
//  //    if (have_motion) {
//  //      kcam->perspective_pre = cameratoraster * transform_inverse(motion[0]);
//  //      kcam->perspective_post = cameratoraster * transform_inverse(motion[motion.size() -
//  1]);
//  //    }
//  //    else {
//  //      kcam->perspective_pre = worldtoraster;
//  //      kcam->perspective_post = worldtoraster;
//  //    }
//  //  }
//  //}
//  // else if (need_motion == Scene::MOTION_BLUR) {
//  //  if (have_motion) {
//  //    kernel_camera_motion.resize(motion.size());
//  //    transform_motion_decompose(kernel_camera_motion.data(), motion.data(), motion.size());
//  //    kcam->num_motion_steps = motion.size();
//  //  }
//
//  //  /* TODO(sergey): Support other types of camera. */
//  //  if (use_perspective_motion && type == CAMERA_PERSPECTIVE) {
//  //    /* TODO(sergey): Move to an utility function and de-duplicate with
//  //     * calculation above.
//  //     */
//  //    ProjectionTransform screentocamera_pre = projection_inverse(
//  //        projection_perspective(fov_pre, nearclip, farclip));
//  //    ProjectionTransform screentocamera_post = projection_inverse(
//  //        projection_perspective(fov_post, nearclip, farclip));
//
//  //    kcam->perspective_pre = screentocamera_pre * rastertoscreen;
//  //    kcam->perspective_post = screentocamera_post * rastertoscreen;
//  //    kcam->have_perspective_motion = 1;
//  //  }
//  //}
//
//  ///* depth of field */
//  // kcam->aperturesize = aperturesize;
//  // kcam->focaldistance = focaldistance;
//  // kcam->blades = (blades < 3) ? 0.0f : blades;
//  // kcam->bladesrotation = bladesrotation;
//
//  ///* motion blur */
//  // kcam->shuttertime = (need_motion == Scene::MOTION_BLUR) ? shuttertime : -1.0f;
//
//  /* type */
//  kcam->type = type;
//
//  ///* anamorphic lens bokeh */
//  // kcam->inv_aperture_ratio = 1.0f / aperture_ratio;
//
//  ///* panorama */
//  // kcam->panorama_type = panorama_type;
//  // kcam->fisheye_fov = fisheye_fov;
//  // kcam->fisheye_lens = fisheye_lens;
//  // kcam->equirectangular_range = make_float4(longitude_min - longitude_max,
//  //                                          -longitude_min,
//  //                                          latitude_min - latitude_max,
//  //                                          -latitude_min + M_PI_2_F);
//
//  // switch (stereo_eye) {
//  //  case STEREO_LEFT:
//  //    kcam->interocular_offset = -interocular_distance * 0.5f;
//  //    break;
//  //  case STEREO_RIGHT:
//  //    kcam->interocular_offset = interocular_distance * 0.5f;
//  //    break;
//  //  case STEREO_NONE:
//  //  default:
//  //    kcam->interocular_offset = 0.0f;
//  //    break;
//  //}
//
//  // kcam->convergence_distance = convergence_distance;
//  // if (use_pole_merge) {
//  //  kcam->pole_merge_angle_from = pole_merge_angle_from;
//  //  kcam->pole_merge_angle_to = pole_merge_angle_to;
//  //}
//  // else {
//  //  kcam->pole_merge_angle_from = -1.0f;
//  //  kcam->pole_merge_angle_to = -1.0f;
//  //}
//
//  ///* sensor size */
//  // kcam->sensorwidth = sensorwidth;
//  // kcam->sensorheight = sensorheight;
//
//  ///* render size */
//  kcam->width = width;
//  kcam->height = height;
//  // kcam->resolution = resolution;
//
//  /* store differentials */
//  kcam->dx = float3_to_float4(dx);
//  kcam->dy = float3_to_float4(dy);
//
//  /* clipping */
//  kcam->nearclip = nearclip;
//  kcam->cliplength = (farclip == FLT_MAX) ? FLT_MAX : farclip - nearclip;
//
//  ///* Camera in volume. */
//  // kcam->is_inside_volume = 0;
//
//  ///* Rolling shutter effect */
//  // kcam->rolling_shutter_type = rolling_shutter_type;
//  // kcam->rolling_shutter_duration = rolling_shutter_duration;
//
//  ///* Set further update flags */
//  // need_update = false;
//  // need_device_update = true;
//  // need_flags_update = true;
//  // previous_need_motion = need_motion;
//}

// CCL_NAMESPACE_END

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
  ccl::KernelData *kdata = (ccl::KernelData *)&cuda_kernel_data[0];

  kdata->integrator.min_bounce = min_bounce;
  kdata->integrator.max_bounce = max_bounce;
  kdata->integrator.max_diffuse_bounce = max_diffuse_bounce;
  kdata->integrator.max_glossy_bounce = max_glossy_bounce;
  kdata->integrator.max_transmission_bounce = max_transmission_bounce;
  kdata->integrator.max_volume_bounce = max_volume_bounce;
  kdata->integrator.max_volume_bounds_bounce = max_volume_bounds_bounce;
  kdata->integrator.transparent_min_bounce = transparent_min_bounce;
  kdata->integrator.transparent_max_bounce = transparent_max_bounce;
  kdata->integrator.use_lamp_mis = use_lamp_mis;

  // apply data
  memcpy(data_bin, &cuda_kernel_data[0], get_size_data(kg_bin));
  const_copy(numDevice, kg_bin, "data", data_bin, get_size_data(kg_bin));
}

void socket_step(int numDevice, DEVICE_PTR kg_bin, char *data_bin, char *cdata)
{

  // view_to_kernel_camera(
  //    &cuda_kernel_data[0], nearclip, farclip, lens, cameratoworld, width, height);

#if defined(WITH_CLIENT_RENDERENGINE) || defined(WITH_CLIENT_ULTRAGRID) || \
    defined(WITH_CLIENT_MPI_VRCLIENT)
  view_to_kernel_camera(&cuda_kernel_data[0], (cyclesphi_data *)cdata);

#  if defined(WITH_CLIENT_RENDERENGINE_VR) || \
      (defined(WITH_CLIENT_ULTRAGRID) && !defined(WITH_CLIENT_RENDERENGINE))
  view_to_kernel_camera_right(&cuda_kernel_data_right[0], (cyclesphi_data *)cdata);

  ccl::KernelData *kdata1 = (ccl::KernelData *)&cuda_kernel_data[0];
  ccl::KernelData *kdata2 = (ccl::KernelData *)&cuda_kernel_data_right[0];

  kdata2->integrator.seed = util_hash_uint2(10000, kdata1->integrator.seed + 1);

#  endif

#else
  ccl::KernelData *kdata = (ccl::KernelData *)&cuda_kernel_data[0];

  // memcpy((char *)&kg->data.cam, (char *)cam, sizeof(KernelCamera));

  // cameratoworld
  // memcpy((char *)&kg->data.cam.cameratoworld, (char *)cameratoworld, sizeof(float) * 12);
  // memcpy((char *)cuda_get_camera_matrix(kg_bin), (char *)cameratoworld, sizeof(float) * 12);

  // cuda_camera_update((CameraType)kdata->cam.type,
  //                   fov,
  //                   nearclip,
  //                   farclip,
  //                   kdata->cam.cameratoworld,
  //                   width,
  //                   height,
  //                   &kdata->cam);

  // wh,h,dx,dy
  int *size = (int *)cdata;

  float *height = &kdata->cam.height;
  float *width = &kdata->cam.width;
  float *dx = &kdata->cam.dx[0];
  float *dy = &kdata->cam.dy[0];
  float *rastertocamera = &kdata->cam.rastertocamera.x[0];

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

  const_copy(numDevice, kg_bin, "data", &cuda_kernel_data[0], get_size_data(kg_bin), false);

#endif

  // apply data
  memcpy(data_bin, &cuda_kernel_data[0], get_size_data(kg_bin));
}

// void socket_step(
//    int numDevice, DEVICE_PTR kg_bin, char *data_bin, float *cameratoworld, float _w, float _h)
//{
//  // cameratoworld
//  memcpy((char *)cuda_get_camera_matrix(kg_bin), (char *)cameratoworld, sizeof(float) * 12);
//
//  // wh,h,dx,dy
//  float *height = cuda_get_camera_height(kg_bin);
//  float *width = cuda_get_camera_width(kg_bin);
//  float *dx = cuda_get_camera_dx(kg_bin);
//  float *dy = cuda_get_camera_dy(kg_bin);
//  float *rastertocamera = cuda_get_camera_rastertocamera(kg_bin);
//
//  float old_new_height = height[0] / _h;
//  float old_new_width = width[0] / _w;
//
//  height[0] = _h;
//  width[0] = _w;
//  dx[0] *= old_new_width;
//  dx[1] *= old_new_width;
//  dx[2] *= old_new_width;
//  // dx[3] *= old_new_width;
//
//  dy[0] *= old_new_width;
//  dy[1] *= old_new_width;
//  dy[2] *= old_new_width;
//  // dy[3] *= old_new_width;
//
//  rastertocamera[0] *= old_new_width;
//  rastertocamera[1] *= old_new_width;
//  rastertocamera[2] *= old_new_width;
//  // rastertocamera[3] *= old_new_width;
//
//  rastertocamera[4] *= old_new_width;
//  rastertocamera[5] *= old_new_width;
//  rastertocamera[6] *= old_new_width;
//  // rastertocamera[7] *= old_new_width;
//
//  rastertocamera[8] *= old_new_width;
//  rastertocamera[9] *= old_new_width;
//  rastertocamera[10] *= old_new_width;
//  // rastertocamera[11] *= old_new_width;
//
//  // apply data
//  memcpy(data_bin, &cuda_kernel_data[0], cuda_get_size_data(kg_bin));
//  cuda_const_copy(numDevice, kg_bin, "data", data_bin, cuda_get_size_data(kg_bin));
//}

void print_client_tiles()
{
  // #ifdef WITH_CUDA_PROFILE
  //       cudaProfilerStop();
  // #endif
  std::string client_tiles = "";
  for (int id1 = 0; id1 < ccl::cuda_devices.size(); id1++) {
    // cuda_devices[id1].wtile_h = cuda_devices[id1].wtile.h;

    client_tiles += std::to_string(ccl::cuda_devices[id1].wtile.y) + std::string(";") +
                    std::to_string(ccl::cuda_devices[id1].wtile.h) + std::string(";");
  }

  printf("LB: %s\n", client_tiles.c_str());
  setenv("CLIENT_TILES", client_tiles.c_str(), 1);

  const char *lb_cuda_file = getenv("LOAD_BALANCE_CUDA_FILE");
  if (lb_cuda_file != NULL) {
    FILE *f = fopen(lb_cuda_file, "wt+");
    fputs(client_tiles.c_str(), f);
    fclose(f);
  }
}

// CCL_NAMESPACE_END
}  // namespace cuda
}  // namespace kernel
}  // namespace cyclesphi

//#endif
