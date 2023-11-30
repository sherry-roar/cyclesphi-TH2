#include "kernel_cuda_stat.h"
//# include "kernel_cuda_util.h"

#include "kernel/types.h"
//# include "kernel_cuda_device.h"
#include "device/cuda/device_impl.h"
#include "device/cuda/util.h"

#ifdef WITH_CLIENT_OPTIX
#  include "device/optix/device_impl.h"
#endif

namespace cyclesphi {
namespace kernel {

size_t mem_sum = 0;
size_t cuda_chunk_size = 0;
size_t cuda_get_chunk_size()
{
  if (cuda_chunk_size == 0) {
    const char *CUDA_CHUNK_SIZE_MB = getenv("CUDA_CHUNK_SIZE_MB");
    if (CUDA_CHUNK_SIZE_MB != NULL) {
      cuda_chunk_size = atol(CUDA_CHUNK_SIZE_MB) * 1024L * 1024L;
      printf("CUDA_CHUNK_SIZE_MB: %lld\n", cuda_chunk_size);
    }
    else {
      cuda_chunk_size = 2L * 1024L * 1024L;
    }
  }
  return cuda_chunk_size;
}

bool check_mem_advise_name(const char *_name, bool log)
{
  const char *env_names_in = getenv("CLIENT_IN_ADVISE");

  if (env_names_in != NULL) {
    std::string names(env_names_in);
    std::string name(_name);
    if (names.find(name) != std::string::npos) {
      if (log)
        printf("In ADVISE: %s\n", _name);
      return true;
    }
    return false;
  }

#if 1
  if (!cuda_check_unimem(_name, 1))
    return false;
#endif

  if (log)
    printf("In ADVISE: %s\n", _name);
  return true;
}

bool compare_longlong(unsigned long long int i, unsigned long long int j)
{
  return (i > j);
}

void cuda_set_show_stat_max_bvh_level(unsigned int bvh_level)
{
  unsigned int ishow_stat_max_bvh_level = bvh_level;
  std::string ssmax_bvh_level_name = "__bvh_nodes_max_level";

  for (int id = 0; id < ccl::cuda_devices.size(); id++) {
    ccl::CUDAContextScope scope(id);

    CU_DEVICE_PTR dev_max_level;
    cuda_assert2(cudaMalloc(&dev_max_level, sizeof(unsigned int)));
    cuda_assert2(cudaMemcpy(dev_max_level,
                            (void *)&ishow_stat_max_bvh_level,
                            sizeof(unsigned int),
                            cudaMemcpyHostToDevice));

    CUdeviceptr cumem;
    size_t cubytes;
    uint64_t ptr;

    cu_assert(
        cuModuleGetGlobal(&cumem, &cubytes, scope.get().cuModule, ssmax_bvh_level_name.c_str()));
    ptr = (uint64_t)dev_max_level;
    cuda_assert2(cudaMemcpy((char *)cumem, (void *)&ptr, cubytes, cudaMemcpyHostToDevice));
  }
}

#if defined(WITH_CLIENT_CUDA_CPU_STAT)

void cuda_print_stat_cpu(int devices_size)
{
  printf("cuda_print_stat_cpu\n");

  cudaSetDevice(ccl::cuda_devices[0].cuDevice);

  std::map<std::string, ccl::CUDADevice::CUDAMem>::iterator it_stat;
  std::vector<std::string> data_names;

  for (it_stat = ccl::cuda_devices[0].cuda_stat_map.begin();
       it_stat != ccl::cuda_devices[0].cuda_stat_map.end();
       it_stat++) {

    std::string data_name = it_stat->first;

    if (!check_mem_advise_name(data_name.c_str(), false))
      continue;

    data_names.push_back(data_name);
  }

  for (int id = 0; id < devices_size; id++) {
    char temp_filename[1024];
    sprintf(temp_filename, "CPU_DATA_STATISTICS_SAMPLE_%d.txt", id);
    FILE *f = fopen(temp_filename, "w+");

    for (int n = 0; n < data_names.size(); n++) {
      std::string data_name = data_names[n];

      fprintf(f, "%s:\t", data_name.c_str());

      ccl::CUDADevice::CUDAMem *cm = &ccl::cuda_devices[id].cuda_stat_map[data_name];
      std::vector<unsigned long long int> temp_sum(cm->mem.data_size);

      for (size_t c = 0; c < cm->mem.data_size; c++) {
        temp_sum[c] = omp_get_data_stat(data_name, id, c);
      }
      //      memcpy(&temp_sum[0],
      //             &cm->map_host_pointer[0],
      //             sizeof(unsigned long long int) * temp_sum.size());

      // std::sort(temp_sum.begin(), temp_sum.end(), compare_longlong);

      size_t total_size = 0;
      for (int j = 0; j < cm->mem.data_size; j++) {
        fprintf(f, "%lld\t", temp_sum[j]);
        total_size += temp_sum[j];
      }
      // fprintf(f, "%lld\t", total_size);

      fprintf(f, "\n");
    }
    fclose(f);
  }
  exit(0);
}

// void set_mem_advise_by_stat_cpu_credits(int devices_size)
//{
//  printf("set_mem_advise_by_stat_cpu_credits\n");
//
//  cudaSetDevice(ccl::cuda_devices[0].cuDevice);
//  // cudaSetDevice(ccl::cuda_devices[0].cuDevice);
//
//  std::map<std::string, ccl::CUDADevice::CUDAMem>::iterator it_stat;
//  std::vector<std::string> data_names;
//
//  for (it_stat = ccl::cuda_devices[0].cuda_stat_map.begin();
//       it_stat != ccl::cuda_devices[0].cuda_stat_map.end();
//       it_stat++) {
//
//    std::string data_name = it_stat->first;
//
//    if (!check_mem_advise_name(data_name.c_str(), false))
//      continue;
//
//    data_names.push_back(data_name);
//  }
//
//  ///////////////////////////////////////////////////////////////////////////
//
//  std::vector<stat_sum *> data_sum_sort;
//
//  for (int n = 0; n < data_names.size(); n++) {
//    std::string data_name = data_names[n];
//
//    ccl::CUDADevice::CUDAMem *cm_bvh_nodes0 = &ccl::cuda_devices[0].cuda_stat_map[data_name];
//    CU_DEVICE_PTR device_ptr0 =
//        (CU_DEVICE_PTR)ccl::cuda_devices[0]
//            .cuda_mem_map[cm_bvh_nodes0->device_pointers[CUDA_DEVICE_POINTER_MAP_ID]]
//            .device_pointer;
//    size_t device_ptr_size0 =
//        ccl::cuda_devices[0]
//            .cuda_mem_map[cm_bvh_nodes0->device_pointers[CUDA_DEVICE_POINTER_MAP_ID]]
//            .size;
//
//#    if 1
//    //    for (int id = 0; id < devices_size; id++) {
//    //      cuda_assert2(cudaMemAdvise(device_ptr0, device_ptr_size0,
//    cudaMemAdviseUnsetAccessedBy,
//    //      id)); cuda_assert2(cudaMemAdvise(device_ptr0, device_ptr_size0,
//    //      cudaMemAdviseUnsetReadMostly, id)); cuda_assert2(
//    //          cudaMemAdvise(device_ptr0, device_ptr_size0, cudaMemAdviseUnsetPreferredLocation,
//    //          id));
//    //    }
//
//    for (int id = 0; id < devices_size; id++) {
//      cuda_assert2(cudaMemAdvise(device_ptr0, device_ptr_size0, cudaMemAdviseSetAccessedBy, id));
//    }
//#    endif
//
//    //#    if 0
//    //#      ifdef WITH_CLIENT_UNIMEM
//    //    // cuda_assert2(cudaFree(device_ptr0));
//    //    cuda_assert2(cudaMallocManaged(&device_ptr0, device_ptr_size0));
//    //    for (int id = 0; id < devices_size; id++) {
//    //      ccl::cuda_devices[id]
//    //          .cuda_mem_map[cm_bvh_nodes0->device_pointers[CUDA_DEVICE_POINTER_MAP_ID]]
//    //          .device_pointer = (DEVICE_PTR)device_ptr0;
//    //    }
//    //#      endif
//    //#    endif
//
//    std::vector<stat_sum *> temp_sum(cm_bvh_nodes0->size);
//
//    // unsigned long long int *c_data0 =
//    //    (unsigned long long int *)&cm_bvh_nodes0->map_host_pointer[0];
//
//    for (size_t c = 0; c < cm_bvh_nodes0->size; c++) {
//      stat_sum *ss = new stat_sum();
//      ss->data_chunk_id = c;
//      ss->data_id = n;
//      ss->sum_gpu = 0;
//      ss->stat_gpus.resize(devices_size);
//      // ss->stat_gpus = new [devices_size];
//
//      for (int id1 = 0; id1 < devices_size; id1++) {
//        ccl::CUDADevice::CUDAMem *cm_bvh_nodes =
//        &ccl::cuda_devices[id1].cuda_stat_map[data_name];
//        // unsigned long long int *c_data =
//        //    (unsigned long long int *)&cm_bvh_nodes->map_host_pointer[0];
//        // ss->sum_gpu += c_data[c];
//        unsigned long long int c_data = omp_get_data_stat(data_name, id1, c);
//        ss->sum_gpu += c_data;
//        ss->stat_gpus[id1] = new stat_gpu();
//        ss->stat_gpus[id1]->gpu_id = id1;
//        // ss->stat_gpus[id1]->value = c_data[c];
//        ss->stat_gpus[id1]->value = c_data;
//      }
//
//      std::sort(ss->stat_gpus.begin(), ss->stat_gpus.end(), compare_stat_gpu);
//      temp_sum[c] = ss;
//    }
//    data_sum_sort.insert(data_sum_sort.end(), temp_sum.begin(), temp_sum.end());
//
//    printf("cudaMemAdviseSetAccessedBy: %s, temp_sum: %lld\n", data_name.c_str(),
//    temp_sum.size());
//  }
//
//  std::sort(data_sum_sort.begin(), data_sum_sort.end(), compare_stat_sum);
//
//  ///////////////////////////////////////////////////////////////////////////
//
//  ///////////////////////////////////////////////////////////////////////////
//
//  const char *ADVISE_MUL = std::getenv("PERCENT_READ_MOSTLY");
//  double advise_mul = 0.0;
//  size_t data050_id = 0;
//
//  if (ADVISE_MUL)
//    advise_mul = atof(ADVISE_MUL) / 100.0;
//
//  if (advise_mul <= 0.0)
//    data050_id = 0;
//  else if (advise_mul >= 1.0)
//    data050_id = data_sum_sort.size() - 1;
//  else
//    data050_id = (size_t)(advise_mul * (double)(data_sum_sort.size() - 1ULL));
//
//  // printf("data050_id: %lld\n", data050_id);
//  ////////////////////////////////////////////////////////////////////////////
//  size_t total_credits = ((data_sum_sort.size() - data050_id) / devices_size) + 1;
//
//// enable max by credits
//#    if 1
//  total_credits = (20L * 1024L * 1024L * 1024L / (size_t)CUDA_CHUNK_SIZE) - data050_id;
//#    endif
//
//  std::vector<size_t> credits_gpus(devices_size, total_credits);
//
//  //////////////////////////////////////////////////////////////////////////////
//  std::vector<size_t> read_mostly_size(devices_size, 0);
//  std::vector<size_t> read_mostly_count(devices_size, 0);
//  std::vector<size_t> preffered_location_size(devices_size, 0);
//  std::vector<size_t> preffered_location_count(devices_size, 0);
//
//  // initialize random seed
//  srand(time(NULL));
//  ////////////////////////////////////////////////////////////////////////////
//#    ifdef WITH_CLIENT_UNIMEM
//  for (size_t s = 0; s < data_sum_sort.size(); s++) {
//    stat_sum *ss = data_sum_sort[s];
//
//    std::string data_name = data_names[ss->data_id];
//
//    ccl::CUDADevice::CUDAMem *cm_bvh_nodes0 = &ccl::cuda_devices[0].cuda_stat_map[data_name];
//    CU_DEVICE_PTR device_ptr0 =
//        (CU_DEVICE_PTR)ccl::cuda_devices[0]
//            .cuda_mem_map[cm_bvh_nodes0->device_pointers[CUDA_DEVICE_POINTER_MAP_ID]]
//            .device_pointer;
//    size_t device_ptr_size0 =
//        ccl::cuda_devices[0]
//            .cuda_mem_map[cm_bvh_nodes0->device_pointers[CUDA_DEVICE_POINTER_MAP_ID]]
//            .size;
//
//    size_t chunk_offset = ss->data_chunk_id * CUDA_CHUNK_SIZE;
//    size_t chunk_size = ((ss->data_chunk_id + 1) * CUDA_CHUNK_SIZE < device_ptr_size0) ?
//                            CUDA_CHUNK_SIZE :
//                            device_ptr_size0 - ss->data_chunk_id * CUDA_CHUNK_SIZE;
//
//    if (chunk_size < 1) {
//      printf("WARN: stat: %s, %lld\n", data_name.c_str(), s);
//      printf("WARN: hunk_offset: %lld, chunk_size: %lld, device_ptr_size0: %lld\n",
//             chunk_offset,
//             chunk_size,
//             device_ptr_size0);
//
//      continue;
//    }
//
//    cudaMemoryAdvise advise_flag;
//
//    if (advise_mul < 0.0)
//      advise_flag = cudaMemAdviseSetPreferredLocation;
//    else if (advise_mul > 1.0)
//      advise_flag = cudaMemAdviseSetReadMostly;
//    else
//      advise_flag = (s <= data050_id) ? cudaMemAdviseSetReadMostly :
//                                        cudaMemAdviseSetPreferredLocation;
//
//      // cudaMemAdviseSetReadMostly
//#      ifdef _MEMADVISE
//    if (advise_flag == cudaMemAdviseSetReadMostly) {
//#        if 1
//      cuda_assert2(cudaMemAdvise((char *)device_ptr0 + chunk_offset,
//                                chunk_size,
//                                cudaMemAdviseSetReadMostly,
//                                ss->stat_gpus[0]->gpu_id));
//
//      read_mostly_size[ss->stat_gpus[0]->gpu_id] += chunk_size;
//      read_mostly_count[ss->stat_gpus[0]->gpu_id]++;
//
//      for (int id = 0; id < devices_size; id++) {
//        cuda_assert2(cudaMemPrefetchAsync((char *)device_ptr0 + chunk_offset, chunk_size, id));
//      }
//#        endif
//
//#        if 0
//      for (int id = 0; id < devices_size; id++) {
//
//	      cuda_assert2(cudaMemAdvise((char *)device_ptr0 + chunk_offset,
//		                        chunk_size,
//		                        cudaMemAdviseSetReadMostly,
//		                        id));
//
//	      read_mostly_size[id] += chunk_size;
//	      read_mostly_count[id]++;
//
//	      cuda_assert2(cudaMemPrefetchAsync((char *)device_ptr0 + chunk_offset, chunk_size, id));
//      }
//
//#        endif
//    }
//    else {
//      // default max
//      int preffered_device = ss->stat_gpus[0]->gpu_id;
//
//// enable random
//#        if 1
//      preffered_device = rand() % devices_size;
//#        endif
//
//// enable credits
//#        if 0
//      //bool force_exit = true;
//      for (int id = 0; id < ss->stat_gpus.size(); id++) {
//        int gpu_id = ss->stat_gpus[id]->gpu_id;
//        if (credits_gpus[gpu_id] > 0) {
//          preffered_device = gpu_id;
//          credits_gpus[gpu_id]--;
//          //force_exit = false;
//          break;
//        }
//      }
//
////      if (force_exit) {
////        printf("credits_gpus is empty\n");
////        exit(0);
////      }
//#        endif
//
//      cuda_assert2(cudaMemAdvise((char *)device_ptr0 + chunk_offset,
//                                chunk_size,
//                                cudaMemAdviseSetPreferredLocation,
//                                preffered_device));
//
//      preffered_location_size[preffered_device] += chunk_size;
//      preffered_location_count[preffered_device]++;
//
//      cuda_assert2(
//          cudaMemPrefetchAsync((char *)device_ptr0 + chunk_offset, chunk_size,
//          preffered_device));
//    }
//#      endif
//  }
//#    endif
//
//  ///////////////////////////////////////////////////////
//#    if 0
//  for (int id = 0; id < devices_size; id++) {
//    printf(
//        "%d: preffered_location_size: %lld, read_mostly_size: %lld, preffered_location_count: "
//        "%lld, read_mostly_count: %lld, credits_gpus: %lld, total_credits: %lld, total_chunks: "
//        "%lld\n",
//        id,
//        preffered_location_size[id],
//        read_mostly_size[id],
//        preffered_location_count[id],
//        read_mostly_count[id],
//        credits_gpus[id],
//        total_credits,
//        data_sum_sort.size());
//
//    fflush(0);
//  }
//#    endif
//  ///////////////////////////////////////////////////////
//
//  for (size_t s = 0; s < data_sum_sort.size(); s++) {
//    for (size_t g = 0; g < data_sum_sort[s]->stat_gpus.size(); g++) {
//      stat_gpu *sg = data_sum_sort[s]->stat_gpus[g];
//      delete sg;
//    }
//    stat_sum *ss = data_sum_sort[s];
//    delete ss;
//  }
//  //////////////////////////////////////////////////
//
//#    if defined(WITH_CLIENT_CUDA_CPU_STAT2)
//
//  for (int n = 0; n < data_names.size(); n++) {
//    std::string data_name = data_names[n];
//
//    ccl::CUDADevice::CUDAMem *cm_bvh_nodes0 = &ccl::cuda_devices[0].cuda_stat_map[data_name];
//    CU_DEVICE_PTR device_ptr0 =
//        (CU_DEVICE_PTR)ccl::cuda_devices[0]
//            .cuda_mem_map[cm_bvh_nodes0->device_pointers[CUDA_DEVICE_POINTER_MAP_ID]]
//            .device_pointer;
//    size_t device_ptr_size0 =
//        ccl::cuda_devices[0]
//            .cuda_mem_map[cm_bvh_nodes0->device_pointers[CUDA_DEVICE_POINTER_MAP_ID]]
//            .size;
//
//    DEVICE_PTR cpu_mem = omp_get_data_mem(data_name);
//    DEVICE_PTR cpu_size = omp_get_data_size(data_name);
//
//    if (data_name == "texture_info")
//      cpu_mem = (DEVICE_PTR)&ccl::cuda_devices[0].texture_info_mem[0];
//    // cuda_assert2(
//    //  cudaMemcpy(device_ptr0, (void *)cpu_mem, device_ptr_size0, cudaMemcpyHostToDevice));
//
//    printf(
//        "cuda_tex_info_copy: %s, cpu_size: %lld, device_ptr_size0: %lld, device_ptr0: %lld, "
//        "CUDA_DEVICE_POINTER_MAP_ID: %lld, cpu_mem: %lld\n",
//        data_name.c_str(),
//        cpu_size,
//        device_ptr_size0,
//        (DEVICE_PTR)device_ptr0,
//        (DEVICE_PTR)cm_bvh_nodes0->device_pointers[CUDA_DEVICE_POINTER_MAP_ID],
//        (DEVICE_PTR)cpu_mem);
//
//    // memcpy(device_ptr0, (void *)cpu_mem, (cpu_size < device_ptr_size0) ? cpu_size :
//    // device_ptr_size0);
//    cuda_tex_info_copy(
//        (char *)cpu_mem, cm_bvh_nodes0->device_pointers[CUDA_DEVICE_POINTER_MAP_ID], cpu_size);
//  }
//
//  // cuda_tex_info_copy()
//
//#    endif
//}
#endif

void log_alloc(int dev, size_t memory_size, const char *name)
{
  if (dev == 0) {
    mem_sum += memory_size;
    printf("log_alloc: %s, %lld, %.3f [MB]\n",
           name,
           memory_size,
           (float)mem_sum / (1024.0f * 1024.0f));
  }
}

void log_free(int dev, size_t memory_size, const char *name)
{
  if (dev == 0) {
    mem_sum -= memory_size;
    printf("log_free: %s, %lld, %.3f [MB]\n",
           name,
           memory_size,
           (float)mem_sum / (1024.0f * 1024.0f));
  }
}

void cuda_tex_copy_stat(ccl::CUDAContextScope &scope,
                        size_t data_count,
                        size_t mem_size,
                        const char *name,
                        DEVICE_PTR map_id,
                        char *texture_info_)
{
  ccl::TextureInfo *texture_info = (ccl::TextureInfo *)texture_info_;

#ifdef WITH_CUDA_STAT
  std::string counter_name(name);
  counter_name = counter_name + "_counter";

  std::string counter_mul_name(name);
  counter_mul_name = counter_mul_name + "_counter_mul";

  std::string counter_size_name(name);
  counter_size_name = counter_size_name + "_counter_size";

  // for (int id = 0; id < ccl::cuda_devices.size(); id++)
  {
    //  ccl::CUDAContextScope scope(id);

    double counter_mul = (double)(((double)mem_size / (double)data_count) /
                                  (double)CUDA_CHUNK_SIZE);
    unsigned int counter_size = (unsigned int)ceil(data_count * counter_mul);  // + 1ULL;
    if (counter_size < 1)
      counter_size = 1;

    size_t counter_mem_size = counter_size * sizeof(unsigned long long int);
    size_t counter_pitch = 0;

    CU_DEVICE_PTR dev_counter;

    if (texture_info != NULL) {
      texture_info->counter_mul = counter_mul;
      texture_info->counter_size = counter_size;

      //      uint64_t ptr = (uint64_t)dev_counter;
      //
      //      CU_DEVICE_PTR dev_counter0;
      //      cuda_assert2(cudaMalloc(&dev_counter0, sizeof(uint64_t)));
      //      cuda_assert2(
      //          cudaMemcpy(dev_counter0, (void *)&ptr, sizeof(uint64_t),
      //          cudaMemcpyHostToDevice));
      //      log_alloc(id, sizeof(uint64_t), "dev_counter0");
      cuda_assert2(cudaMalloc(&dev_counter, counter_mem_size));
      cuda_assert2(cudaMemset(dev_counter, 0, counter_mem_size));
      log_alloc(scope.get().cuDevId, counter_mem_size, "dev_counter");

      texture_info->counter = (unsigned long long int **)dev_counter;  // 0;
    }
    else {
      cuda_assert2(cudaMalloc(&dev_counter, counter_mem_size));
      cuda_assert2(cudaMemset(dev_counter, 0, counter_mem_size));
      log_alloc(scope.get().cuDevId, counter_mem_size, "dev_counter");

      CU_DEVICE_PTR dev_counter_mul;
      cuda_assert2(cudaMalloc(&dev_counter_mul, sizeof(double)));
      cuda_assert2(
          cudaMemcpy(dev_counter_mul, &counter_mul, sizeof(double), cudaMemcpyHostToDevice));
      log_alloc(scope.get().cuDevId, sizeof(double), "dev_counter_mul");

      CU_DEVICE_PTR dev_counter_size;
      cuda_assert2(cudaMalloc(&dev_counter_size, sizeof(unsigned int)));
      cuda_assert2(cudaMemcpy(
          dev_counter_size, &counter_size, sizeof(unsigned int), cudaMemcpyHostToDevice));
      log_alloc(scope.get().cuDevId, sizeof(unsigned int), "dev_counter_size");

      CUdeviceptr cumem;
      size_t cubytes;
      uint64_t ptr;

      CUdeviceptr cumem_mul;
      size_t cubytes_mul;
      uint64_t ptr_mul;

      CUdeviceptr cumem_size;
      size_t cubytes_size;
      uint64_t ptr_size;

      cu_assert(
          cuModuleGetGlobal(&cumem, &cubytes, scope.get().cuModuleStat, counter_name.c_str()));
      ptr = (uint64_t)dev_counter;
      cuda_assert2(cudaMemcpy((char *)cumem, (void *)&ptr, cubytes, cudaMemcpyHostToDevice));

      cu_assert(cuModuleGetGlobal(
          &cumem_mul, &cubytes_mul, scope.get().cuModuleStat, counter_mul_name.c_str()));
      ptr_mul = (uint64_t)dev_counter_mul;
      cuda_assert2(
          cudaMemcpy((char *)cumem_mul, (void *)&ptr_mul, cubytes_mul, cudaMemcpyHostToDevice));

      cu_assert(cuModuleGetGlobal(
          &cumem_size, &cubytes_size, scope.get().cuModuleStat, counter_size_name.c_str()));
      ptr_size = (uint64_t)dev_counter_size;
      cuda_assert2(
          cudaMemcpy((char *)cumem_size, (void *)&ptr_size, cubytes_size, cudaMemcpyHostToDevice));
    }

    ccl::CUDADevice::CUDAMem *cmem = &scope.get().cuda_stat_map[std::string(name)];
    cmem->mem.data_size = counter_size;
    cmem->mem.device_size = counter_mem_size;
    cmem->counter_pitch = counter_pitch;
    cmem->map_host_pointer = new char[counter_mem_size];
    cmem->free_map_host = false;
    cmem->mem.device_pointer = (DEVICE_PTR)dev_counter;
    cmem->dpointers[CUDA_DEVICE_POINTER_MAP_ID].device_pointer = (DEVICE_PTR)map_id;
    // cmem->device_pointers[CUDA_DEVICE_POINTER_SORT] = (DEVICE_PTR)dev_sort;
    cmem->uni_mem = true;
    cmem->mem.name = name;
  }

#  ifdef WITH_CLIENT_UNIMEM

#  endif

#endif

#ifdef WITH_CLIENT_CUDA_CPU_STAT
  std::string counter_name(name);
  counter_name = counter_name + "_counter";

  std::string counter_mul_name(name);
  counter_mul_name = counter_mul_name + "_counter_mul";

  std::string counter_size_name(name);
  counter_size_name = counter_size_name + "_counter_size";

  // for (int id = 0; id < ccl::cuda_devices.size(); id++)
  {
    // ccl::CUDAContextScope scope(id);

    double counter_mul = (double)(((double)mem_size / (double)data_count) /
                                  (double)CUDA_CHUNK_SIZE);
    unsigned int counter_size = (unsigned int)ceil(data_count * counter_mul);  // + 1ULL;
    if (counter_size < 1)
      counter_size = 1;

    size_t counter_mem_size = counter_size * sizeof(unsigned long long int);

    ccl::CUDADevice::CUDAMem *cmem = &scope.get().cuda_stat_map[std::string(name)];
    cmem->size = counter_size;
    // cmem->map_host_pointer.resize(counter_mem_size);
    cmem->free_map_host = false;
    // cmem->device_pointer = (DEVICE_PTR)dev_counter;
    cmem->device_pointers[CUDA_DEVICE_POINTER_MAP_ID] = (DEVICE_PTR)map_id;
    // cmem->device_pointers[CUDA_DEVICE_POINTER_SORT] = (DEVICE_PTR)dev_sort;
    cmem->uni_mem = true;
    cmem->name = std::string(name);
  }

#  ifdef WITH_CLIENT_UNIMEM

#  endif

#endif
}

void cuda_print_stat_gpu(int devices_size, int stream_memcpy_id)
{
  printf("cuda_print_stat_gpu\n");

#if defined(WITH_CLIENT_CUDA_CPU_STAT) || defined(WITH_CUDA_STAT)

  std::map<std::string, ccl::CUDADevice::CUDAMem>::iterator it_stat;
  std::vector<std::string> data_names;

  for (it_stat = ccl::cuda_devices[0].cuda_stat_map.begin();
       it_stat != ccl::cuda_devices[0].cuda_stat_map.end();
       it_stat++) {

    std::string data_name = it_stat->first;

    if (!check_mem_advise_name(data_name.c_str()))
      continue;

    data_names.push_back(data_name);
  }

  for (int id = 0; id < devices_size; id++) {
    char temp_filename[1024];
    sprintf(temp_filename, "GPU_DATA_STATISTICS_SAMPLE_%d.txt", id);
    FILE *f = fopen(temp_filename, "w+");

    for (int n = 0; n < data_names.size(); n++) {
      std::string data_name = data_names[n];

      fprintf(f, "%s:\t", data_name.c_str());

      ccl::CUDADevice::CUDAMem *cm = &ccl::cuda_devices[id].cuda_stat_map[data_name];
      std::vector<unsigned long long int> temp_sum(cm->mem.data_size);
      memcpy(&temp_sum[0],
             &cm->map_host_pointer[0],
             sizeof(unsigned long long int) * temp_sum.size());

      // std::sort(temp_sum.begin(), temp_sum.end(), compare_longlong);

      size_t total_size = 0;
      for (int j = 0; j < cm->mem.data_size; j++) {
        fprintf(f, "%lld\t", temp_sum[j]);
        total_size += temp_sum[j];
      }
      // fprintf(f, "%lld\t", total_size);

      fprintf(f, "\n");
    }
    fclose(f);
  }
  exit(0);
#endif
}

bool compare_stat_gpu(stat_gpu *i, stat_gpu *j)
{
  return (i->value > j->value);
}

bool compare_stat_sum(stat_sum *i, stat_sum *j)
{
  return (i->sum_gpu > j->sum_gpu);
}

#define DEBUG_LOG
void set_mem_advise_by_stat3_credits(int devices_size, int stream_memcpy_id, bool cpu_stat)
{
  // usleep(10L * 1000L * 1000L);

  printf("set_mem_advise_by_stat3_credits\n");

#if defined(WITH_CLIENT_CUDA_CPU_STAT) || defined(WITH_CUDA_STAT)

  double stat3_time_start = omp_get_wtime();

  if (cpu_stat)
    cudaSetDevice(ccl::cuda_devices[0].cuDevice);

  std::map<std::string, ccl::CUDADevice::CUDAMem>::iterator it_stat;
  std::vector<std::string> data_names;

  for (it_stat = ccl::cuda_devices[0].cuda_stat_map.begin();
       it_stat != ccl::cuda_devices[0].cuda_stat_map.end();
       it_stat++) {

    std::string data_name = it_stat->first;

    if (!check_mem_advise_name(data_name.c_str(), false))
      continue;

    data_names.push_back(data_name);
  }

  ///////////////////////////////////////////////////////////////////////////

  std::vector<stat_sum *> data_sum_sort;
  size_t data_sum_memory = 0;
  for (int n = 0; n < data_names.size(); n++) {
    std::string data_name = data_names[n];

    ccl::CUDADevice::CUDAMem *cm_bvh_nodes0 = &ccl::cuda_devices[0].cuda_stat_map[data_name];
    CU_DEVICE_PTR device_ptr0 =
        (CU_DEVICE_PTR)ccl::cuda_devices[0]
            .cuda_mem_map[cm_bvh_nodes0->dpointers[CUDA_DEVICE_POINTER_MAP_ID].device_pointer]
            .mem.device_pointer;
    size_t device_ptr_size0 =
        ccl::cuda_devices[0]
            .cuda_mem_map[cm_bvh_nodes0->dpointers[CUDA_DEVICE_POINTER_MAP_ID].device_pointer]
            .mem.device_size;

    data_sum_memory += device_ptr_size0;

#  if defined(WITH_CLIENT_UNIMEM) && !defined(WITH_CUDA_STATv2)
#    ifdef WITH_CLIENT_CUDA_CPU_STAT3
    if (true) {
#    else
    if (!cpu_stat) {
#    endif
      cuda_assert2(cudaFree(device_ptr0));
      cuda_assert2(cudaMallocManaged(&device_ptr0, device_ptr_size0));
      for (int id = 0; id < devices_size; id++) {
        ccl::cuda_devices[id]
            .cuda_mem_map[cm_bvh_nodes0->device_pointers[CUDA_DEVICE_POINTER_MAP_ID]]
            .device_pointer = (DEVICE_PTR)device_ptr0;
      }
    }

#    ifdef _MEMADVISE

#      if !defined(WITH_CUDA_STATv2) && !defined(WITH_CLIENT_CUDA_CPU_STAT2v2)
    for (int id = 0; id < devices_size; id++) {
      cuda_assert2(cudaMemAdvise(device_ptr0, device_ptr_size0, cudaMemAdviseSetAccessedBy, id));
    }
#        ifdef DEBUG_LOG
    printf("cudaMemAdviseSetAccessedBy: %s\n", data_name.c_str());
#        endif

#      endif

#    endif

#  endif

    std::vector<stat_sum *> temp_sum(cm_bvh_nodes0->mem.data_size);

    unsigned long long int *c_data0 =
        (unsigned long long int *)&cm_bvh_nodes0->map_host_pointer[0];

    for (size_t c = 0; c < cm_bvh_nodes0->mem.data_size; c++) {
      stat_sum *ss = new stat_sum();
      ss->data_chunk_id = c;
      ss->data_id = n;
      ss->sum_gpu = 0;
      ss->stat_gpus.resize(devices_size);

      for (int id1 = 0; id1 < devices_size; id1++) {
        ccl::CUDADevice::CUDAMem *cm_bvh_nodes = &ccl::cuda_devices[id1].cuda_stat_map[data_name];
        unsigned long long int cd = 0;

        if (cpu_stat) {
#  if defined(WITH_CLIENT_CUDA_CPU_STAT)
          cd = omp_get_data_stat(data_name, id1, c);
#  endif
        }
        else {
          unsigned long long int *c_data =
              (unsigned long long int *)&cm_bvh_nodes->map_host_pointer[0];
          cd = c_data[c];
        }

        ss->sum_gpu += cd;
        ss->stat_gpus[id1] = new stat_gpu();
        ss->stat_gpus[id1]->gpu_id = id1;
        ss->stat_gpus[id1]->value = cd;
      }

      std::sort(ss->stat_gpus.begin(), ss->stat_gpus.end(), compare_stat_gpu);
      temp_sum[c] = ss;
    }
    data_sum_sort.insert(data_sum_sort.end(), temp_sum.begin(), temp_sum.end());
#  ifdef DEBUG_LOG
    printf("data_sum_sort: %s, temp_sum: %lld\n", data_name.c_str(), temp_sum.size());
#  endif
  }

  std::sort(data_sum_sort.begin(), data_sum_sort.end(), compare_stat_sum);

  printf("V3: algorithm runtime: %f [s]\n", omp_get_wtime() - stat3_time_start);
  ////////////////////////////
  double stat3_mem_advise_time_start = omp_get_wtime();

  const char *ADVISE_MUL = std::getenv("PERCENT_READ_MOSTLY");
  if (ADVISE_MUL != NULL) {
    printf("ADVISE_MUL: %s\n", ADVISE_MUL);
  }

  // #  ifdef UNITEST2_CREDITS_CPU
  //   const char *ADVISE_MUL_CPU = std::getenv("PERCENT_READ_MOSTLY_CPU");
  //   if (ADVISE_MUL_CPU != NULL) {
  //     printf("ADVISE_MUL_CPU: %s\n", ADVISE_MUL_CPU);
  //   }
  // #  endif

  double advise_mul = 0.0;
  size_t data050_id = 0;

  if (ADVISE_MUL)
    advise_mul = atof(ADVISE_MUL) / 100.0;

  // #  ifdef UNITEST2_CREDITS_CPU
  //   double advise_mul_cpu = 1.1;
  //   size_t data050_id_cpu = 0;

  //   if (ADVISE_MUL_CPU)
  //     advise_mul_cpu = atof(ADVISE_MUL_CPU) / 100.0;
  // #  endif

  size_t mem_free, mem_tot;
  cuda_assert2(cudaMemGetInfo(&mem_free, &mem_tot));
  double mem_gpu = (double)mem_tot - 3.5 * 1024.0 * 1024.0 * 1024.0;  // kernel

  //#  ifdef UNITEST2_CREDITS_CPU
  const char *CREDITS_GPU_MAX_MEM = std::getenv("CREDITS_GPU_MAX_MEM");
  if (CREDITS_GPU_MAX_MEM != NULL) {
    printf("CREDITS_GPU_MAX_MEM: %s\n", CREDITS_GPU_MAX_MEM);
    mem_gpu = atof(CREDITS_GPU_MAX_MEM) * 1024.0 * 1024.0 * 1024.0;
  }
  //#  endif

  // AUTO
  if (advise_mul > 10.0) {
    double gpu_count = (double)devices_size;
    double scene_size = (mem_gpu - (double)mem_free) * gpu_count;

    advise_mul = (mem_gpu - scene_size / gpu_count) / (scene_size - scene_size / gpu_count);
    printf("AUTO ADVISE_MUL: %f, %f, %f, %f, %lld, %lld\n",
           advise_mul * 100.0,
           mem_gpu,
           scene_size,
           gpu_count,
           mem_free,
           mem_tot);

    // 39.872048, 30868504576.000000, 70750568448.000000, 16.000000
    // 39.872048, 30868504576.000000, 70750568448.000000, 16.000000, 26446594048, 34089730048
    // 88.278052, 30868504576.000000, 34679554048.000000, 16.000000
    // 88.278052, 30868504576.000000, 34679554048.000000, 16.000000, 28701032448, 34089730048

    fflush(0);
  }

  if (advise_mul < 0.0)
    data050_id = 0;
  else if (advise_mul > 1.0)
    data050_id = data_sum_memory;
  else
    data050_id = (size_t)(advise_mul * (double)(data_sum_memory));

    // #  ifdef UNITEST2_CREDITS_CPU
    //   if (advise_mul_cpu < 0.0)
    //     data050_id_cpu = 0;
    //   else if (advise_mul_cpu > 1.0)
    //     data050_id_cpu = data_sum_memory;
    //   else
    //     data050_id_cpu = (size_t)(advise_mul_cpu * (double)(data_sum_memory));

    //   printf("data050_id_cpu: %lld\n", data050_id_cpu);
    // #  endif

#  ifdef DEBUG_LOG
  printf("data050_id: %lld\n", data050_id);
#  endif
  ////////////////////////////////////////////////////////////////////////////

  size_t data050_memory = 0;
  size_t total_credits = (size_t)ceil(((double)data_sum_memory - (double)data050_id) /
                                      (double)devices_size) +
                         data050_id;

#  ifdef UNITEST2_CREDITS_CPU
  total_credits += CUDA_CHUNK_SIZE;

  if (total_credits > mem_gpu)
    total_credits = mem_gpu;
#  endif

#  ifdef UNITEST2_CREDITS_CPU2
  total_credits = mem_gpu;
#  endif

#  ifdef DEBUG_LOG
  printf("total_credits: %lld, data_sum_size: %lld\n", total_credits, data_sum_sort.size());
#  endif
  std::vector<long long int> credits_gpus(devices_size, total_credits);

  //////////////////////////////////////////////////////////////////////////////
  // std::vector<size_t> read_mostly_size(devices_size, 0);
  std::vector<size_t> read_mostly_count(devices_size, 0);
  // std::vector<size_t> preffered_location_size(devices_size, 0);
  std::vector<size_t> preffered_location_count(devices_size, 0);
  size_t preffered_location_cpu_count = 0;
  std::vector<size_t> preffered_location_zero_chunk(devices_size, 0);
#  if 0  
  std::vector<size_t> preffered_extreme_count(devices_size, 0);
#  endif
  // initialize random seed
  srand(time(NULL));
  ////////////////////////////////////////////////////////////////////////////
#  ifdef WITH_CLIENT_SHOW_STAT
  const char *show_stat_name = getenv("SHOW_STAT_NAME");
  std::string ssname = std::string(show_stat_name);

  std::vector<int> counter_gpu;
  if (show_stat_name != NULL) {
    printf("SHOW_STAT_NAME: %s\n", show_stat_name);
    std::string counter_gpu_name = ssname + "_counter_gpu";

    ccl::CUDADevice::CUDAMem *cm_bvh_nodes0 = &ccl::cuda_devices[0].cuda_stat_map[ssname];
    size_t data_count0 =
        ccl::cuda_devices[0]
            .cuda_mem_map[cm_bvh_nodes0->device_pointers[CUDA_DEVICE_POINTER_MAP_ID]]
            .data_count;

    counter_gpu.resize(data_count0);
  }
#  endif
  ////////////////////////////////////////////////////////////////////////////
#  ifdef WITH_CLIENT_UNIMEM
  int last_preffered_device = 0;

#    if 0
  const char *cgev = std::getenv("CREDIT_GPU_EXTREME_VALUE");
  if (cgev != NULL) {
    printf("CREDIT_GPU_EXTREME_VALUE: %s\n", cgev);
  }
#    endif

  for (size_t s = 0; s < data_sum_sort.size(); s++) {
    stat_sum *ss = data_sum_sort[s];

    std::string data_name = data_names[ss->data_id];

    ccl::CUDADevice::CUDAMem *cm_bvh_nodes0 = &ccl::cuda_devices[0].cuda_stat_map[data_name];
    CU_DEVICE_PTR device_ptr0 =
        (CU_DEVICE_PTR)ccl::cuda_devices[0]
            .cuda_mem_map[cm_bvh_nodes0->dpointers[CUDA_DEVICE_POINTER_MAP_ID].device_pointer]
            .mem.device_pointer;
    size_t device_ptr_size0 =
        ccl::cuda_devices[0]
            .cuda_mem_map[cm_bvh_nodes0->dpointers[CUDA_DEVICE_POINTER_MAP_ID].device_pointer]
            .mem.device_size;
    // size_t data_count0 =
    //     ccl::cuda_devices[0]
    //         .cuda_mem_map[cm_bvh_nodes0->dpointers[CUDA_DEVICE_POINTER_MAP_ID].device_pointer]
    //         .data_count;

    size_t chunk_offset = ss->data_chunk_id * CUDA_CHUNK_SIZE;
    size_t chunk_size = ((ss->data_chunk_id + 1) * CUDA_CHUNK_SIZE < device_ptr_size0) ?
                            CUDA_CHUNK_SIZE :
                            device_ptr_size0 - ss->data_chunk_id * CUDA_CHUNK_SIZE;

    if (chunk_size < 1) {
      printf("WARN: stat: %s, %lld\n", data_name.c_str(), s);
      printf("WARN: chunk_offset: %lld, chunk_size: %lld, device_ptr_size0: %lld\n",
             chunk_offset,
             chunk_size,
             device_ptr_size0);

      continue;
    }

    data050_memory += chunk_size;

    cudaMemoryAdvise advise_flag;

    if (advise_mul < 0.0)
      advise_flag = cudaMemAdviseSetPreferredLocation;
    else if (advise_mul > 1.0)
      advise_flag = cudaMemAdviseSetReadMostly;
    else
      advise_flag = (data050_memory < data050_id) ? cudaMemAdviseSetReadMostly :
                                                    cudaMemAdviseSetPreferredLocation;

    CUdevice preffered_device = 0;

#    if 0
    if (cgev != NULL) {
      float credit_gpu_extreme_value = atof(cgev) / 100.0f;

      bool find_preffered_device = true;
      for (int id = 1; id < ss->stat_gpus.size(); id++) {
        if (credit_gpu_extreme_value * ss->stat_gpus[0]->value <= ss->stat_gpus[id]->value) {
          find_preffered_device = false;
          break;
        }
      }
      if (find_preffered_device) {
        // for (int id = 0; id < ss->stat_gpus.size(); id++) {
        int gpu_id = ss->stat_gpus[0]->gpu_id;
        if (credits_gpus[gpu_id] > chunk_size) {
          preffered_device = gpu_id;
          credits_gpus[preffered_device] -= chunk_size;
          preffered_extreme_count[preffered_device]++;
          // break;
          data050_id++;
        }
        //}
      }
    }
#    endif

    // printf("V3: algorithm runtime: %f [s]\n", omp_get_wtime() - stat3_time_start);
    ////////////////////////////
    // double stat3_mem_advise_time_start = omp_get_wtime();

#    if defined(UNITEST2_CREDITS_CPU)

    if (advise_flag == cudaMemAdviseSetReadMostly) {
      bool test_free_mem = true;
      for (int id = 0; id < devices_size; id++) {
        if (credits_gpus[id] < chunk_size)
          test_free_mem = false;
      }

      if (/*data050_memory > data050_id_cpu ||*/ data050_memory > mem_gpu || !test_free_mem) {
#      ifdef DEBUG_LOG
        printf("preffered_device = CU_DEVICE_CPU, mem: %f\n",
               (double)data050_memory / (double)data_sum_memory);
#      endif
        advise_flag = cudaMemAdviseSetPreferredLocation;
      }
    }
#    endif

    // cudaMemAdviseSetReadMostly
    if (advise_flag == cudaMemAdviseSetReadMostly) {

#    ifdef _MEMADVISE
#      if 0
      cuda_assert2(cudaMemAdvise((char *)device_ptr0 + chunk_offset,
                                chunk_size,
                                cudaMemAdviseUnsetPreferredLocation,
                                ss->stat_gpus[0]->gpu_id));
#      endif
      cuda_assert2(cudaMemAdvise((char *)device_ptr0 + chunk_offset,
                                 chunk_size,
                                 cudaMemAdviseSetReadMostly,
                                 ss->stat_gpus[0]->gpu_id));
#    endif

      for (int id = 0; id < devices_size; id++) {
        ccl::CUDAContextScope scope(id);
        // read_mostly_size[id] += chunk_size;
        read_mostly_count[id] += chunk_size;
        credits_gpus[id] -= chunk_size;

#    ifdef _PREFETCH
        // if (!cpu_stat)
        {
          cudaStream_t stream_memcpy = ccl::cuda_devices[id].stream[STREAM_PATH1_MEMCPY];
          cuda_assert2(cudaMemPrefetchAsync(
              (char *)device_ptr0 + chunk_offset, chunk_size, id, stream_memcpy));
        }
#    endif
      }

      ss->preffered_device = 16;

#    ifdef WITH_CLIENT_SHOW_STAT
      if (ssname == data_name) {
        // all gpu
        double counter_mul = (double)(((double)device_ptr_size0 / (double)data_count0) /
                                      (double)CUDA_CHUNK_SIZE);
        size_t counter_gpu_index = (size_t)((double)ss->data_chunk_id / (double)counter_mul);
        size_t counter_gpu_index1 = (size_t)((double)(ss->data_chunk_id + 1) /
                                             (double)counter_mul);

        if (data_count0 < counter_gpu_index1)
          counter_gpu_index1 = data_count0;

        for (int j = counter_gpu_index; j < counter_gpu_index1; j++)
          counter_gpu[j] = 16;
      }
#    endif
    }
    else {

      // if (preffered_device == -1) {

// enable random
#    if defined(UNITEST2_RANDOM)
      preffered_device = rand() % devices_size;
#    endif

// enable round robin
#    if defined(UNITEST2_ROUND_ROBIN)
      last_preffered_device++;

      if (last_preffered_device > devices_size - 1) {
        last_preffered_device = 0;
      }

      preffered_device = last_preffered_device;
#    endif

// enable credits
#    if defined(UNITEST2_CREDITS) || defined(UNITEST2_CREDITS_CPU)
#      if 1
      bool force_exit = true;
#      endif

#      if defined(UNITEST2_CREDITS_CPU)
#        ifdef DEBUG_LOG
      // printf("data050_memory: %f >= data050_id_cpu: %f || data050_memory: %f > mem_gpu *
      // devices_size: %f\n", (double)data050_memory, (double)data050_id_cpu,
      // (double)data050_memory, (double)mem_gpu * (double)devices_size);
#        endif
      if (/*data050_memory > data050_id_cpu ||*/ data050_memory > mem_gpu * (double)devices_size) {
        //#  ifdef DEBUG_LOG
        //        printf("preffered_device = CU_DEVICE_CPU\n");
        //#  endif
        preffered_device = CU_DEVICE_CPU;
        force_exit = false;
      }
      else {
#      endif
        if (ss->sum_gpu > 0) {
          for (int id = 0; id < ss->stat_gpus.size(); id++) {
            int gpu_id = ss->stat_gpus[id]->gpu_id;
            if (credits_gpus[gpu_id] > chunk_size) {
              preffered_device = gpu_id;
              credits_gpus[preffered_device] -= chunk_size;
#      if 1
              force_exit = false;
#      endif
              break;
            }
          }
        }
        else {
          for (int id = 0; id < devices_size; id++) {
            last_preffered_device++;

            if (last_preffered_device > devices_size - 1) {
              last_preffered_device = 0;
            }

            if (credits_gpus[last_preffered_device] > chunk_size) {
              preffered_device = last_preffered_device;
              credits_gpus[preffered_device] -= chunk_size;
#      if 1
              force_exit = false;
#      endif
              break;
            }
          }

          preffered_location_zero_chunk[preffered_device] += chunk_size;
        }
#      if defined(UNITEST2_CREDITS_CPU)
      }
#      endif

#      if 1
      if (force_exit) {
        printf("credits_gpus is empty: %d / %d\n", s + 1, data_sum_sort.size());
#        if defined(UNITEST2_CREDITS_CPU)
        preffered_device = CU_DEVICE_CPU;
#        else
        exit(0);
#        endif
      }
#      endif

#    endif
      //}

#    ifdef _MEMADVISE
      cuda_assert2(cudaMemAdvise((char *)device_ptr0 + chunk_offset,
                                 chunk_size,
                                 cudaMemAdviseSetPreferredLocation,
                                 preffered_device));
#    endif

      //#    ifdef UNITEST2_CREDITS_CPU
      if (preffered_device == CU_DEVICE_CPU)
        preffered_location_cpu_count += chunk_size;
      else
        //#    endif
        preffered_location_count[preffered_device] += chunk_size;

#    ifdef _PREFETCH
      // if (!cpu_stat)
      if (preffered_device != CU_DEVICE_CPU) {
        ccl::CUDAContextScope scope(preffered_device);
        cudaStream_t stream_memcpy =
            ccl::cuda_devices[preffered_device].stream[STREAM_PATH1_MEMCPY];
        cuda_assert2(cudaMemPrefetchAsync(
            (char *)device_ptr0 + chunk_offset, chunk_size, preffered_device, stream_memcpy));
      }
#    endif

      ss->preffered_device = preffered_device;

#    ifdef WITH_CLIENT_SHOW_STAT
      if (ssname == data_name) {
        // preffered_device gpu
        double counter_mul = (double)(((double)device_ptr_size0 / (double)data_count0) /
                                      (double)CUDA_CHUNK_SIZE);
        size_t counter_gpu_index = (size_t)((double)ss->data_chunk_id / (double)counter_mul);
        size_t counter_gpu_index1 = (size_t)((double)(ss->data_chunk_id + 1) /
                                             (double)counter_mul);

        if (data_count0 < counter_gpu_index1)
          counter_gpu_index1 = data_count0;

        for (int j = counter_gpu_index; j < counter_gpu_index1; j++)
          counter_gpu[j] = preffered_device;
      }
#    endif
    }
  }
//#    ifdef DEBUG_LOG
//  printf("data050_id after extreme: %lld\n", data050_id);
//#    endif
#    ifdef WITH_CLIENT_SHOW_STAT
  if (show_stat_name != NULL) {
    std::string ssname = std::string(show_stat_name);
    std::string counter_gpu_name = ssname + "_counter_gpu";

    for (int id = 0; id < ccl::cuda_devices.size(); id++) {
      ccl::CUDAContextScope scope(id);

      ccl::CUDADevice::CUDAMem *cm_bvh_nodes0 = &scope.get().cuda_stat_map[ssname];
      size_t data_count0 =
          ccl::cuda_devices[0]
              .cuda_mem_map[cm_bvh_nodes0->device_pointers[CUDA_DEVICE_POINTER_MAP_ID]]
              .data_count;

      CU_DEVICE_PTR dev_counter_gpu;
      cuda_assert2(cudaMalloc(&dev_counter_gpu, sizeof(int) * data_count0));
      cuda_assert2(cudaMemcpy(dev_counter_gpu,
                              (void *)&counter_gpu[0],
                              sizeof(int) * data_count0,
                              cudaMemcpyHostToDevice));

      CUdeviceptr cumem;
      size_t cubytes;
      uint64_t ptr;

      cu_assert(
          cuModuleGetGlobal(&cumem, &cubytes, scope.get().cuModule, counter_gpu_name.c_str()));
      ptr = (uint64_t)dev_counter_gpu;
      cuda_assert2(cudaMemcpy((char *)cumem, (void *)&ptr, cubytes, cudaMemcpyHostToDevice));
    }
  }
#      ifndef WITH_CLIENT_SHOW_STAT_BVH_LOOP
  const char *show_stat_max_bvh_level = getenv("SHOW_STAT_MAX_BVH_LEVEL");
  if (show_stat_max_bvh_level != NULL) {
    printf("SHOW_STAT_MAX_BVH_LEVEL: %s\n", show_stat_max_bvh_level);
    unsigned int ishow_stat_max_bvh_level = atoi(show_stat_max_bvh_level);
    cuda_set_show_stat_max_bvh_level(ishow_stat_max_bvh_level);
  }
#      endif
#    endif

#  endif
  printf("V3: use stat - memadvise: %f [s]\n", omp_get_wtime() - stat3_mem_advise_time_start);
  ///////////////////////////////////////////////////////
  if (cpu_stat) {
    double stat3_copy_time_start = omp_get_wtime();
    double total_size = 0;

#  if (defined(WITH_CLIENT_CUDA_CPU_STAT2) || defined(WITH_CLIENT_CUDA_CPU_STAT3)) && \
      !defined(WITH_CLIENT_CUDA_CPU_STAT2v2)

    for (int n = 0; n < data_names.size(); n++) {
      std::string data_name = data_names[n];

      ccl::CUDADevice::CUDAMem *cm_bvh_nodes0 = &ccl::cuda_devices[0].cuda_stat_map[data_name];
      CU_DEVICE_PTR device_ptr0 =
          (CU_DEVICE_PTR)ccl::cuda_devices[0]
              .cuda_mem_map[cm_bvh_nodes0->device_pointers[CUDA_DEVICE_POINTER_MAP_ID]]
              .device_pointer;
      size_t device_ptr_size0 =
          ccl::cuda_devices[0]
              .cuda_mem_map[cm_bvh_nodes0->device_pointers[CUDA_DEVICE_POINTER_MAP_ID]]
              .size;

      DEVICE_PTR cpu_mem = omp_get_data_mem(data_name);
      DEVICE_PTR cpu_size = omp_get_data_size(data_name);

      if (data_name == "texture_info")
        cpu_mem = (DEVICE_PTR)&ccl::cuda_devices[0].texture_info_mem[0];
        // cuda_assert2(
        //  cudaMemcpy(device_ptr0, (void *)cpu_mem, device_ptr_size0, cudaMemcpyHostToDevice));
#    ifdef DEBUG_LOG
      printf("cuda_tex_info_copy: %s, cpu_size: %lld, device_ptr_size0: %lld\n",
             data_name.c_str(),
             cpu_size,
             device_ptr_size0);
#    endif
      // memcpy(device_ptr0, (void *)cpu_mem, (cpu_size < device_ptr_size0) ? cpu_size :
      // device_ptr_size0);
      cuda_tex_info_copy(data_name.c_str(),
                         (char *)cpu_mem,
                         cm_bvh_nodes0->device_pointers[CUDA_DEVICE_POINTER_MAP_ID],
                         cpu_size,
                         false);

      total_size += cpu_size / 1024.0 / 1024.0 / 1024.0;
    }

    // cuda_tex_info_copy()

#  endif

    printf("V3: data copy from CPU to GPU: %f [s], %f [GB], %f [GB/s]\n",
           omp_get_wtime() - stat3_copy_time_start,
           total_size,
           total_size / (omp_get_wtime() - stat3_copy_time_start));
  }

  //////SYNC//////
  double stat3_sync_time_start = omp_get_wtime();
  for (int id = 0; id < devices_size; id++) {
    ccl::CUDAContextScope scope(id);
    cudaStream_t stream_memcpy = ccl::cuda_devices[id].stream[STREAM_PATH1_MEMCPY];
    cuda_assert2(cudaStreamSynchronize(stream_memcpy));
  }
  printf("V3: sync GPUs: %f [s]\n", omp_get_wtime() - stat3_sync_time_start);
// exit(0);
///////////////////////////PRINT////////////////////////////
#  if 1
  const char *cwtp = std::getenv("CREDITS_WRITE_TO_PATH");
  if (cwtp != NULL) {
    printf("CREDITS_WRITE_TO_PATH: %s\n", cwtp);
    char credit_filename[1024];
    for (int n = 0; n < data_names.size(); n++) {
      std::string data_name = data_names[n];

      ccl::CUDADevice::CUDAMem *cm_bvh_nodes0 = &ccl::cuda_devices[0].cuda_stat_map[data_name];
      size_t adv_size_step = cm_bvh_nodes0->mem.data_size;

      sprintf(credit_filename,
              "%s/%s_%lld_%lld",
              cwtp,
              data_name.c_str(),
              CUDA_CHUNK_SIZE,
              adv_size_step);
      FILE *f = fopen(credit_filename, "wb+");
      printf("CREDITS_WRITE_TO_PATH: %s\n", credit_filename);

      // size_t device_ptr_size0 =
      //   cuda_devices[0]
      //       .cuda_mem_map[cm_bvh_nodes0->device_pointers[CUDA_DEVICE_POINTER_MAP_ID]]
      //       .size;

      // if (adv_size_step < 1)
      //   adv_size_step = 1;

      std::vector<int> chunks_per_device(adv_size_step, CU_DEVICE_CPU);
      // size_t chunks_per_device_count = 0;
      printf("adv_size_step: %d\n", adv_size_step);

      for (size_t s = 0; s < data_sum_sort.size(); s++) {
        stat_sum *ss = data_sum_sort[s];
        if (n == ss->data_id) {
          // printf("ss->data_chunk_id: %d - %d\n", ss->data_chunk_id, ss->preffered_device);
          chunks_per_device[ss->data_chunk_id] = ss->preffered_device;
        }
      }

      fwrite(&chunks_per_device[0], sizeof(int), adv_size_step, f);
      fclose(f);
    }
    // exit(0);
  }
#  endif
///////////////////////////PRINT////////////////////////////
#  ifdef DEBUG_LOG
  ///////////////////////////////////////////////////////
  for (int id = 0; id < devices_size; id++) {
    // read_mostly_size: %lld, preffered_location_size: %lld,

    printf(
        "%d: "
        "preffered_location_count: %lld, "
        "preffered_location_cpu_count: %lld, "
        "read_mostly_count: %lld, "
        "preffered_location_zero_chunk: %lld, "
        //"preffered_extreme_count: %lld, "
        "left_credits_gpus: %lld, "
        "total_credits: %lld, "
        "total_chunks: %lld\n",
        id,
        preffered_location_count[id],
        preffered_location_cpu_count,
        read_mostly_count[id],
        preffered_location_zero_chunk[id],
        // preffered_extreme_count[id],
        credits_gpus[id],
        total_credits,
        data_sum_sort.size());

    // fflush(0);
  }

#    if 1
  for (size_t s = 0; s < data_sum_sort.size(); s++) {
    stat_sum *ss = data_sum_sort[s];
    printf("sorted chunk count per data: %d: dev: %d, %s: %lld, chunk: %lld\n",
           s,
           ss->preffered_device,
           data_names[ss->data_id].c_str(),
           ss->sum_gpu,
           ss->data_chunk_id);
  }
#    endif

#    if 1
  for (size_t s = 0; s < data_sum_sort.size(); s++) {
    stat_sum *ss = data_sum_sort[s];
    printf("sorted chunk count per gpu: %d: dev: %d, %s: %lld, chunk: %lld, gpus: ",
           s,
           ss->preffered_device,
           data_names[ss->data_id].c_str(),
           ss->sum_gpu,
           ss->data_chunk_id);

    std::vector<size_t> gpu_stat(devices_size, 0);
    for (int d = 0; d < ss->stat_gpus.size(); d++) {
      gpu_stat[ss->stat_gpus[d]->gpu_id] = ss->stat_gpus[d]->value;
    }
    for (int d = 0; d < gpu_stat.size(); d++) {
      printf("%lld, ", gpu_stat[d]);
    }
    printf("\n");
  }
#    endif

  for (int n = 0; n < data_names.size(); n++) {
    std::vector<size_t> chunk_count(devices_size + 1, 0);
    for (size_t s = 0; s < data_sum_sort.size(); s++) {
      stat_sum *ss = data_sum_sort[s];
      if (n == ss->data_id && ss->preffered_device >= 0 &&
          ss->preffered_device < chunk_count.size()) {
        chunk_count[ss->preffered_device] = chunk_count[ss->preffered_device] + 1;
      }
    }
    printf("data chunk count per device + all: %s:", data_names[n].c_str());
    for (int c = 0; c < chunk_count.size(); c++) {
      printf("%lld\t", chunk_count[c]);
    }
    printf("\n");
  }

  for (int n = 0; n < data_names.size(); n++) {
    std::vector<size_t> sum_stat(devices_size, 0);
    for (size_t s = 0; s < data_sum_sort.size(); s++) {
      stat_sum *ss = data_sum_sort[s];
      if (n == ss->data_id) {
        for (int d = 0; d < ss->stat_gpus.size(); d++) {
          sum_stat[ss->stat_gpus[d]->gpu_id] = sum_stat[ss->stat_gpus[d]->gpu_id] +
                                               ss->stat_gpus[d]->value;
        }
      }
    }
    printf("sum stat per device: %s:", data_names[n].c_str());
    for (int c = 0; c < sum_stat.size(); c++) {
      printf("%lld\t", sum_stat[c]);
    }
    printf("\n");
  }
#  endif
  ////////////////////////FREE///////////////////////////////

  for (size_t s = 0; s < data_sum_sort.size(); s++) {
    for (size_t g = 0; g < data_sum_sort[s]->stat_gpus.size(); g++) {
      stat_gpu *sg = data_sum_sort[s]->stat_gpus[g];
      delete sg;
    }
    stat_sum *ss = data_sum_sort[s];
    delete ss;
  }

  ///////////////////////////////////////////////////////
#endif
}

}  // namespace kernel
}  // namespace cyclesphi