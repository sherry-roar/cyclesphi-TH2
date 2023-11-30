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

#include "kernel_cuda_util.h"
#include "kernel_cuda_stat.h"

#include "kernel_camera.h"
#include "kernel_cuda.h"
#include "kernel_util.h"

//#  include "kernel/kernel.h"
//#  include "kernel/kernel_types.h"
// CCL_NAMESPACE_BEGIN
//#  include "integrator/tile.h"
//#include "kernel/device/cuda/globals.h"
//#include "kernel/integrator/integrator_state.h"
//#  include "integrator/work_tile_scheduler.h"
#include "integrator/work_tile_scheduler.h"
// CCL_NAMESPACE_END

#include "device/cuda/device_impl.h"
#include "device/cuda/util.h"

#ifdef WITH_CLIENT_OPTIX
#  include "device/optix/device_impl.h"
#endif

namespace cyclesphi {
namespace kernel {

void check_exit()
{
  fflush(0);
  assert(false);
  exit(-1);
}

bool cuda_error_(cudaError_t result, const std::string &stmt)
{
  if (result == cudaSuccess)
    return false;

  char err[1024];
  sprintf(err,
          "CUDA error at %s: %s: %s",
          stmt.c_str(),
          cudaGetErrorName(result),
          cudaGetErrorString(result));
  std::string message(err);
  // string message = string_printf("CUDA error at %s: %s", stmt.c_str(), cuewErrorString(result));
  fprintf(stderr, "%s\n", message.c_str());
  return true;
}

void cuda_error_message(const std::string &message)
{
  fprintf(stderr, "%s\n", message.c_str());
}

/////////////////////////////////////
size_t cuda_align_up(size_t offset, size_t alignment)
{
  return (offset + alignment - 1) & ~(alignment - 1);
}

// size_t divide_up(size_t x, size_t y)
//{
//  return (x + y - 1) / y;
//}
// void cwassert(const char *_Message, const char *_File, unsigned _Line)
//{
//  printf("ASSERT: %s, %s, %d\n", _Message, _File, _Line);
//}

bool cuda_check_writable_mem(const char *_name, int dev)
{
  if (!strcmp(_name, "RenderBuffers")) {
    return true;
  }

  if (!strcmp(_name, "pixels")) {
    return true;
  }

  if (!strcmp(_name, "pixels1")) {
    return true;
  }

  if (!strcmp(_name, "pixels2")) {
    return true;
  }

  if (!strcmp(_name, "pixels3")) {
    return true;
  }

  return false;
}

bool cuda_check_unimem(const char *_name, int dev)
{
  if (_name == NULL) {
    printf("WARNING: name is NULL!\n");
    return false;
  }

  std::string name(_name);

  //#  ifdef WITH_CLIENT_UNIMEM
  // always on each GPU
  //  if (!strcmp(_name, "texture_info")) {
  //    return false;
  //  }
  if (name.find("texture_info") != std::string::npos) {
    return false;
  }

  if (name.find("integrator_") != std::string::npos) {
    return false;
  }

  if (name.find("queued_paths") != std::string::npos) {
    return false;
  }

  if (name.find("work_tiles") != std::string::npos) {
    return false;
  }

  if (name.find("client_buffer_passes") != std::string::npos) {
    return false;
  }

  //#  endif

  if (name.find("FILL_MEMORY") != std::string::npos) {
    return false;
  }

  if (strlen(_name) == 0) {
    printf("WARNING: name is empty!\n");
    return false;
  }

  //  if (!strcmp(_name, "FILL_MEMORY")) {
  //    return false;
  //  }

  //#  endif

#ifdef WITH_CLIENT_UNIMEM

  if (name.find("RenderBuffers") != std::string::npos) {
    return true;
  }

  const char *env_names_out = getenv("CLIENT_NOT_IN_UNIMEM");
  if (env_names_out != NULL && !strcmp(env_names_out, "none"))
    return true;

  if (env_names_out != NULL && !strcmp(env_names_out, "all"))
    return false;

  if (env_names_out != NULL && strlen(env_names_out) > 0) {
    std::string names(env_names_out);
    if (names.find(name) != std::string::npos) {
      return false;
    }

    if (name.find(names) != std::string::npos) {
      return false;
    }

    if (names.find("tex_image") != std::string::npos) {
      if (name.find("tex_image") != std::string::npos) {
        return false;
      }
    }

    if (dev == 0)
      printf("In UNIMEM: %s\n", _name);

    return true;
  }

  const char *env_names_in = getenv("CLIENT_IN_UNIMEM");
  if (env_names_in != NULL && !strcmp(env_names_in, "none"))
    return false;

  if (env_names_in != NULL && !strcmp(env_names_in, "all"))
    return true;

  if (env_names_in != NULL && strlen(env_names_in) > 0) {
    std::string names(env_names_in);
    if (names.find(name) != std::string::npos) {
      if (dev == 0)
        printf("In UNIMEM: %s\n", _name);

      return true;
    }

    if (name.find(names) != std::string::npos) {
      if (dev == 0)
        printf("In UNIMEM: %s\n", _name);

      return true;
    }

    if (names.find("tex_image") != std::string::npos) {
      if (name.find("tex_image") != std::string::npos) {
        if (dev == 0)
          printf("In UNIMEM: %s\n", _name);

        return true;
      }
    }

    return false;
  }

  if (dev == 0)
    printf("In UNIMEM: %s\n", _name);

  return true;
#else
  return false;
#endif
}

/////////////////////////////////////

/////////////////////////////////////
int cuda_set_unimem_flag_round_robin_dev = 0;
#if defined(UNITEST1_CREDITS_FILE)
size_t g_total_size = 0;
size_t g_read_mostly = 0;
size_t g_pref_loc = 0;
size_t g_pref_loc_cpu = 0;
#endif
void cuda_set_unimem_flag(DEVICE_PTR map_id)
{
  // TODO
#ifdef UNITEST1_ROUND_ROBIN2
  cuda_set_unimem_flag_round_robin_dev = 0;
#endif

  ccl::CUDAContextScope scope(0);

  ccl::CUDADevice::CUDAMem *cmem = &scope.get().cuda_mem_map[map_id];

  if (cmem->uni_mem) {
    int devices_size = ccl::cuda_devices.size();
    DEVICE_PTR device_pointer = cmem->mem.device_pointer;
    size_t size = cmem->mem.device_size;

    for (int id = 0; id < devices_size; id++) {
#ifdef _MEMADVISE
      cuda_assert2(cudaMemAdvise((char *)device_pointer, size, cudaMemAdviseSetAccessedBy, id));
#endif
    }

    // no flag
    if (cuda_check_writable_mem(cmem->mem.name.c_str(), 0))
      return;

#ifdef WITH_CLIENT_CUDA_CPU_STAT2v2
#  ifdef _MEMADVISE
    cuda_assert(cudaMemAdvise(
        (char *)device_pointer, size, cudaMemAdviseSetPreferredLocation, CU_DEVICE_CPU));
#  endif
    return;
#endif

    printf(
        "set_unimem_flag: cudaMemAdviseSetAccessedBy: %s: "
        "cuda_set_unimem_flag_round_robin_dev:%d\n",
        cmem->mem.name.c_str(),
        cuda_set_unimem_flag_round_robin_dev);

    // std::string name_(cmem->mem.name);
    // if(name_.find("RenderBuffers") != std::string::npos)
    //   return;

    // initialize random seed
#if defined(UNITEST1_RANDOM)
    srand(time(NULL));
#endif

    size_t advise_size_step = devices_size;
    size_t csize = size / devices_size;

    // continuos distribution
#if defined(UNITEST1_COUNTINOUS)
    csize = size / devices_size;
    advise_size_step = devices_size;
#endif

// chunks: round robin / random
#if defined(UNITEST1_ROUND_ROBIN) || defined(UNITEST1_RANDOM) || defined(UNITEST1_CPU) || \
    defined(UNITEST1_CREDITS_FILE)
    csize = CUDA_CHUNK_SIZE;
    advise_size_step = (size_t)ceil((double)size / (double)csize);
    if (advise_size_step < 1)
      advise_size_step = 1;

#endif

      // int cuda_set_unimem_flag_round_robin_dev = 0; move to global

#ifdef UNITEST3_PENALTY
    cuda_set_unimem_flag_round_robin_dev = 1;  // skip first
#endif

    //#  if defined(UNITEST1_ROUND_ROBIN)
    //      const char *ADVISE_MUL = std::getenv("PERCENT_READ_MOSTLY");
    //      if (ADVISE_MUL != NULL) {
    //        printf("ADVISE_MUL: %s\n", ADVISE_MUL);
    //      }
    //
    //      double advise_mul = 0.0;
    //      size_t data050_id = 0;
    //
    //      if (ADVISE_MUL)
    //        advise_mul = atof(ADVISE_MUL) / 100.0;
    //
    //      if (advise_mul < 0.0)
    //        data050_id = -1;
    //      else if (advise_mul > 1.0)
    //        data050_id = advise_size_step;
    //      else
    //        data050_id = (size_t)(advise_mul * (double)(advise_size_step - 1));
    //
    //#  endif

// file
#if defined(UNITEST1_CREDITS_FILE)
    g_total_size += size;

    const char *crfp = std::getenv("CREDITS_READ_FROM_PATH");
    std::vector<int> prefered_devices;
    // FILE *file = NULL;
    if (crfp != NULL) {
      printf("CREDITS_READ_FROM_PATH: %s\n", crfp);
      char credit_filename[1024];

      // sprintf(credit_filename, "%s/%s_%lld_%lld", cwtp, data_name.c_str(), CUDA_CHUNK_SIZE,
      // adv_size_step);
      sprintf(credit_filename,
              "%s/%s_%lld_%lld",
              crfp,
              cmem->name.c_str(),
              CUDA_CHUNK_SIZE,
              advise_size_step);
      FILE *file = fopen(credit_filename, "rb");
      if (file != NULL) {
        prefered_devices.resize(advise_size_step);
        fread(&prefered_devices[0], sizeof(int), advise_size_step, file);
        fclose(file);
      }
    }
    else {
      printf("CREDITS_READ_FROM_PATH: File could not find for %s\n", cmem->name.c_str());
    }
#endif

    //#pragma omp parallel for
    for (int i = 0; i < advise_size_step; i++) {

      size_t advise_offset = i * csize;
      size_t advise_size = (i == advise_size_step - 1) ? size - (i * csize) : csize;

      if (advise_size == 0)
        continue;

// file
#if defined(UNITEST1_CREDITS_FILE)
      {
        int prefered_device = CU_DEVICE_CPU;
        if (prefered_devices.size() > 0) {
          // fread(&prefered_device, sizeof(int), 1, file);
          prefered_device = prefered_devices[i];
        }
#  ifdef _MEMADVISE
        // cudaMemoryAdvise advise_flag = cudaMemAdviseSetPreferredLocation;
        if (prefered_device == 16) {
          g_read_mostly += advise_size;
          // advise_flag = cudaMemAdviseSetReadMostly;
          // prefered_device = 0;
          CUDAContextScope scope(0);
          cuda_assert(cudaMemAdvise(
              (char *)device_pointer + advise_offset, advise_size, cudaMemAdviseSetReadMostly, 0));

#    if 0              
          for (int id = 0; id < devices_size; id++) {
            CUDAContextScope scope(id);
            cudaStream_t stream_memcpy = cuda_devices[id].stream[STREAM_PATH1_MEMCPY];            
            cuda_assert(cudaMemPrefetchAsync((char *)device_pointer + advise_offset, advise_size, id, stream_memcpy));            
          }
#    endif
        }
        else if (prefered_device >= 0 && prefered_device < devices_size) {
          g_pref_loc += advise_size;
          ccl::CUDAContextScope scope(prefered_device);
          cuda_assert(cudaMemAdvise((char *)device_pointer + advise_offset,
                                    advise_size,
                                    cudaMemAdviseSetPreferredLocation,
                                    prefered_device));
#    if 0
          cudaStream_t stream_memcpy = cuda_devices[prefered_device].stream[STREAM_PATH1_MEMCPY];           
          cuda_assert(cudaMemPrefetchAsync((char *)device_pointer + advise_offset,
                                           advise_size,
                                           prefered_device,
                                           stream_memcpy));
#    endif
        }
        else if (prefered_device == CU_DEVICE_CPU) {
          g_pref_loc_cpu += advise_size;
          cuda_assert(cudaMemAdvise((char *)device_pointer + advise_offset,
                                    advise_size,
                                    cudaMemAdviseSetPreferredLocation,
                                    prefered_device));

#    if 0
          cudaStream_t stream_memcpy = cuda_devices[prefered_device].stream[STREAM_PATH1_MEMCPY];           
          cuda_assert(cudaMemPrefetchAsync((char *)device_pointer + advise_offset,
                                           advise_size,
                                           prefered_device,
                                           stream_memcpy));
#    endif
        }
        else {
          printf("WARNING: prefered_device >= devices_size\n");
        }
#  endif
      }
#endif

// random
#if defined(UNITEST1_RANDOM) || defined(UNITEST1_CPU)
      int prefered_device = rand() % devices_size;

#  ifdef UNITEST3_PENALTY
      prefered_device = (rand() % (devices_size - 1)) + 1;  // skip first
#  endif

#  ifdef UNITEST3_PENALTYv2
      prefered_device = 0;
#  endif

#  ifdef UNITEST1_CPU
      prefered_device = CU_DEVICE_CPU;
#  endif

#  ifdef _MEMADVISE
      ccl::CUDAContextScope scope(prefered_device);
      cuda_assert2(cudaMemAdvise((char *)device_pointer + advise_offset,
                                 advise_size,
                                 cudaMemAdviseSetPreferredLocation,
                                 prefered_device));
#  endif
#  ifdef _MEMADVISE
      if (prefered_device != CU_DEVICE_CPU) {
        cudaStream_t stream_memcpy =
            ccl::cuda_devices[prefered_device].stream[STREAM_PATH1_MEMCPY];
        cuda_assert2(cudaMemPrefetchAsync(
            (char *)device_pointer + advise_offset, advise_size, prefered_device, stream_memcpy));
      }
#  endif
#endif

// round robin
#if defined(UNITEST1_ROUND_ROBIN) || defined(UNITEST1_COUNTINOUS)
      int prefered_device = cuda_set_unimem_flag_round_robin_dev;

      cudaMemoryAdvise advise_flag = cudaMemAdviseSetPreferredLocation;

#  ifdef UNITEST1_ROUND_ROBIN
//#if 0
//    if (advise_mul < 0.0)
//      advise_flag = cudaMemAdviseSetPreferredLocation;
//    else if (advise_mul > 1.0)
//      advise_flag = cudaMemAdviseSetReadMostly;
//    else
//      advise_flag = (i <= data050_id) ? cudaMemAdviseSetReadMostly :
//                                       cudaMemAdviseSetPreferredLocation;
//#endif
#  endif

#  ifdef UNITEST3_PENALTYv2
      prefered_device = 0;
#  endif

#  ifdef _MEMADVISE
      ccl::CUDAContextScope scope(prefered_device);
      cuda_assert2(cudaMemAdvise(
          (char *)device_pointer + advise_offset, advise_size, advise_flag, prefered_device));
#  endif
      cudaStream_t stream_memcpy = ccl::cuda_devices[prefered_device].stream[STREAM_PATH1_MEMCPY];
#  ifdef _MEMADVISE
      cuda_assert2(cudaMemPrefetchAsync(
          (char *)device_pointer + advise_offset, advise_size, prefered_device, stream_memcpy));
#  endif
#endif

      cuda_set_unimem_flag_round_robin_dev++;
      if (cuda_set_unimem_flag_round_robin_dev > devices_size - 1) {
        cuda_set_unimem_flag_round_robin_dev = 0;

#ifdef UNITEST3_PENALTY
        cuda_set_unimem_flag_round_robin_dev = 1;  // skip first
#endif
      }
    }

    // file
    // #  if defined(UNITEST1_CREDITS_FILE)
    //     if (prefered_devices.size() > 0) {
    //       //fclose(file);
    //     }
    // #  endif

    for (int id = 0; id < devices_size; id++) {
      ccl::CUDAContextScope scope(id);
      cudaStream_t stream_memcpy = ccl::cuda_devices[id].stream[STREAM_PATH1_MEMCPY];
      cuda_assert2(cudaStreamSynchronize(stream_memcpy));
    }

#if defined(UNITEST1_CREDITS_FILE)
    printf("g_total_size: %lld, g_read_mostly: %lld, g_pref_loc: %lld, g_pref_loc_cpu: %lld\n",
           g_total_size,
           g_read_mostly,
           g_pref_loc,
           g_pref_loc_cpu);
#endif
  }
}

DEVICE_PTR generic_alloc(
    const char *name, size_t memory_size, size_t pitch_padding, bool unimem_flag, bool alloc_host)
{
  mem_sum += memory_size;
  double t = omp_get_wtime();

  DEVICE_PTR map_id = 0;
  size_t size_padd = memory_size + pitch_padding;

  int devices_size = ccl::cuda_devices.size();
  bool uni_mem = false;
  uni_mem = cuda_check_unimem(name, 0);

  if (uni_mem) {
    CU_DEVICE_PTR device_pointer = 0;
    cudaSetDevice(0);
    cuda_assert2(cudaMallocManaged(&device_pointer, size_padd));
    // printf("cudaMallocManaged: %f\n", omp_get_wtime() - t);
    map_id = (DEVICE_PTR)device_pointer;

    for (int id = 0; id < devices_size; id++) {
      ccl::CUDAContextScope scope(id);

      ccl::CUDADevice::CUDAMem *cmem = &scope.get().cuda_mem_map[map_id];
      // cmem->free_map_host = false;
      cmem->mem.device_pointer = (DEVICE_PTR)device_pointer;
      cmem->uni_mem = uni_mem;
      cmem->mem.name = name;
      cmem->mem.device_size = size_padd;

      if (id == 0 && uni_mem) {
#if defined(WITH_CUDA_STAT) && !defined(WITH_CUDA_STATv2)
        cmem->host_pointer = cuda_host_alloc(NULL, NULL, cmem->size);
#else
        char *host_mem = NULL;
        if (alloc_host) {
          host_mem = cuda::host_alloc(NULL, NULL, size_padd);
          memset(host_mem, 0, size_padd);
        }

        cmem->mem.host_pointer = host_mem;
#endif
      }
    }
#if 1  // def _MEMADVISE
    if (unimem_flag)
      cuda_set_unimem_flag(map_id);
#endif
  }
  else {
    for (int id = 0; id < devices_size; id++) {
      ccl::CUDAContextScope scope(id);

      CU_DEVICE_PTR device_pointer = 0;

      cuda_assert2(cudaMalloc(&device_pointer, size_padd));
      // printf("cudaMalloc: %f\n", omp_get_wtime() - t);

      if (id == 0)
        map_id = (DEVICE_PTR)device_pointer;

      ccl::CUDADevice::CUDAMem *cmem = &scope.get().cuda_mem_map[map_id];
      // cmem->free_map_host = false;
      cmem->mem.device_pointer = (DEVICE_PTR)device_pointer;
      cmem->uni_mem = uni_mem;
      cmem->mem.name = name;
      cmem->mem.device_size = size_padd;

      char *host_mem = NULL;
      if (alloc_host) {
        host_mem = cuda::host_alloc(NULL, NULL, size_padd);
        memset(host_mem, 0, size_padd);
      }

      cmem->mem.host_pointer = host_mem;
    }
  }

  printf("generic_alloc: %s, %.3f, %zu, %.3f, time: %f\n",
         name,
         (float)memory_size / (1024.0f * 1024.0f),
         map_id,
         mem_sum / (1024.0f * 1024.0f),
         omp_get_wtime() - t);
  // printf("'%s' ", name);

  return map_id;
}

void generic_copy_to(char *host_pointer, DEVICE_PTR map_id, size_t memory_size)
{
  double t = omp_get_wtime();

  // if (!ccl::cuda_devices[0].cuda_mem_map[map_id].uni_mem) {
  //#  pragma omp parallel for
  for (int id = 0; id < ccl::cuda_devices.size(); id++) {
    ccl::CUDAContextScope scope(id);

    if (scope.get().cuda_mem_map[map_id].uni_mem && id != 0)
      continue;

    cuda_assert2(cudaMemcpy((CU_DEVICE_PTR)scope.get().cuda_mem_map[map_id].mem.device_pointer,
                            host_pointer,
                            memory_size,
                            cudaMemcpyHostToDevice));

    if (scope.get().cuda_mem_map[map_id].mem.host_pointer != NULL) {
      memcpy(scope.get().cuda_mem_map[map_id].mem.host_pointer, host_pointer, memory_size);
    }
  }

  printf("generic_copy_to: %f, %lld, %lld\n", omp_get_wtime() - t, map_id, memory_size);
}

void generic_copy_to_async(char *host_pointer, DEVICE_PTR map_id, size_t memory_size)
{
  // initialize random seed
  // srand(time(NULL));
  int id = rand() % ccl::cuda_devices.size();
  ccl::CUDAContextScope scope(id);
  cudaStream_t stream_memcpy = scope.get().stream[STREAM_PATH1_MEMCPY];

  cuda_assert2(cudaMemcpyAsync((CU_DEVICE_PTR)scope.get().cuda_mem_map[map_id].mem.device_pointer,
                               host_pointer,
                               memory_size,
                               cudaMemcpyHostToDevice,
                               stream_memcpy));

  if (scope.get().cuda_mem_map[map_id].mem.host_pointer != NULL) {
    memcpy(scope.get().cuda_mem_map[map_id].mem.host_pointer, host_pointer, memory_size);
  }
}

void generic_free(DEVICE_PTR map_id, size_t memory_size)
{
  mem_sum -= memory_size;

  printf("generic_free: %zu, %f, %f\n",
         map_id,
         (float)memory_size / (1024.0f * 1024.0f),
         (float)mem_sum / (1024.0f * 1024.0f));

  for (int id = 0; id < ccl::cuda_devices.size(); id++) {
    ccl::CUDAContextScope scope(id);

    cuda_assert2(cudaFree((CU_DEVICE_PTR)scope.get().cuda_mem_map[map_id].mem.device_pointer));

    if (scope.get().cuda_mem_map[map_id].mem.host_pointer != NULL) {
      cuda::host_free(NULL, NULL, (char *)scope.get().cuda_mem_map[map_id].mem.host_pointer);
      scope.get().cuda_mem_map[map_id].mem.host_pointer = NULL;
    }

#if defined(WITH_CUDA_STAT) || defined(WITH_CLIENT_CUDA_CPU_STAT)
    {
      std::map<std::string, ccl::CUDADevice::CUDAMem>::iterator it_stat;

      for (it_stat = scope.get().cuda_stat_map.begin(); it_stat != scope.get().cuda_stat_map.end();
           it_stat++) {
        ccl::CUDADevice::CUDAMem *cmem = &it_stat->second;
        if (cmem->dpointers[CUDA_DEVICE_POINTER_MAP_ID].device_pointer == (DEVICE_PTR)map_id) {
          printf("generic_free: %s\n", it_stat->first.c_str());
          scope.get().cuda_stat_map.erase(it_stat);
          break;
        }
      }
    }
#endif

    bool uni_mem = scope.get().cuda_mem_map[map_id].uni_mem;

    scope.get().cuda_mem_map.erase(scope.get().cuda_mem_map.find(map_id));

    if (uni_mem)
      break;
  }
}

}  // namespace kernel
}  // namespace cyclesphi
