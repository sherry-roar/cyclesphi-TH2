#pragma once

//#include "kernel_cuda_context_scope.h"
#include "device/cuda/util.h"
#include "kernel_cuda_util.h"

namespace cyclesphi {
namespace kernel {

extern size_t cuda_chunk_size;
size_t cuda_get_chunk_size();

#define CUDA_CHUNK_SIZE (cuda_get_chunk_size())

#ifndef _WIN32
#  define _MEMADVISE
#  define _PREFETCH
#endif

void log_alloc(int dev, size_t memory_size, const char *name);
void log_free(int dev, size_t memory_size, const char *name);

void cuda_tex_copy_stat(ccl::CUDAContextScope &scope,
                        size_t data_count,
                        size_t mem_size,
                        const char *name,
                        DEVICE_PTR map_id,
                        char *texture_info);

void cuda_print_stat_gpu(int devices_size, int stream_memcpy_id);

// bool g_alloc = true;
extern size_t mem_sum;

struct MemSpread {
  size_t offset;
  size_t size;
  cudaMemoryAdvise advise_flag;
  int dev;
};

struct stat_gpu {
  int gpu_id;
  unsigned long long value;
};

struct stat_sum {
  size_t sum_gpu;
  std::vector<stat_gpu *> stat_gpus;
  // stat_gpu **stat_gpus;
  int data_id;
  size_t data_chunk_id;
  int preffered_device;
};

bool compare_stat_gpu(stat_gpu *i, stat_gpu *j);
bool compare_stat_sum(stat_sum *i, stat_sum *j);
//#define DEBUG_LOG
void set_mem_advise_by_stat3_credits(int devices_size, int stream_memcpy_id, bool cpu_stat);

bool check_mem_advise_name(const char *_name, bool log = true);

bool compare_longlong(unsigned long long int i, unsigned long long int j);

void cuda_set_show_stat_max_bvh_level(unsigned int bvh_level);

#if defined(WITH_CLIENT_CUDA_CPU_STAT)

void cuda_print_stat_cpu(int devices_size);

#endif

}  // namespace kernel
}  // namespace cyclesphi
