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

#include "client_api.h"
#include "cycles_client.h"
#include "kernel_util.h"
#include <omp.h>

#if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET) || defined(WITH_CLIENT_MPI_FILE)
#  include <mpi.h>
#endif

#if defined(_WIN32)  //&& defined(_DEBUG)
#  include <windows.h>
#else
#  include <stdlib.h>
#endif

#ifdef WITH_NVML
#  include <nvml.h>
#endif

#ifdef WITH_CLIENT_NCCL_SOCKET
#  include <nccl.h>
#endif

int main(int argc, char **argv)
{
  /////////////////////////
  // setenv("DEBUG_SAMPLES", "2", 1);
  // setenv("DEBUG_STEP_SAMPLES", "10", 1);
  setenv("DEBUG_RES_W", "960", 1);
  setenv("DEBUG_RES_H", "540", 1);
#if 0
  setenv("KERNEL_CUDA_CUBIN",
         "/mnt/proj3/open-18-15/blender/pop2/src/cyclesx/client/build/cycles_cuda/kernel_sm_80.cubin",
         1);

  setenv("KERNEL_CUDA_STAT_CUBIN",
         "/mnt/proj3/open-18-15/blender/pop2/src/cyclesx/client/build/cycles_cuda/kernel_sm_80_stat.cubin",
         1);

  setenv("CLIENT_FILE_KERNEL_GLOBAL",
         "/mnt/proj3/open-18-15/blender/pop2/src/cyclesx/client/../../../data/new.kg",
         1);
  setenv("CLIENT_FILE_CYCLES_BUFFER", "/mnt/proj3/open-18-15/blender/pop2/src/cyclesx/client/../../../data/new2.bf", 1);
  setenv("CLIENT_FILE_CYCLES_BMP", "/mnt/proj3/open-18-15/blender/pop2/src/cyclesx/client/../../../data/new2.bmp", 1);
  setenv("DEBUG_SAMPLES", "64", 1);
  setenv("DEBUG_RES_W", "1920", 1);
  setenv("DEBUG_RES_H", "1080", 1);
  setenv("SOCKET_SERVER_PORT_CAM", "7000", 1);
  setenv("SOCKET_SERVER_PORT_DATA", "7001", 1);

  setenv("SOCKET_SERVER_NAME_CAM", "localhost", 1);
  setenv("SOCKET_SERVER_NAME_DATA", "localhost", 1);
  setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7", 1);
  setenv("DEBUG_REPEAT_TIME", "30", 1);
#endif

#if 0
  setenv("KERNEL_CUDA_CUBIN",
         "f:\\work\\blender\\blender_client_build\\cycles_cuda\\kernel_sm_75.cubin",
         1);

  setenv("KERNEL_OPTIX_PTX",
         "f:\\work\\blender\\blender_client_build\\cycles_optix\\kernel_shader_raytrace_sm_75.ptx",
         1);

  setenv("KERNEL_CUDA_STAT_CUBIN",
         "f:\\work\\blender\\blender_client_build\\cycles_cuda\\kernel_sm_75_stat.cubin",
         1);

  setenv("CLIENT_FILE_KERNEL_GLOBAL", "f:\\temp\\test.kg", 1);

  // setenv("DEBUG_SAMPLES", "1", 1);
  // setenv("DEBUG_RES_W", "1920", 1);
  // setenv("DEBUG_RES_H", "1080", 1);

  //setenv("DEBUG_REPEAT_TIME", "10", 1);

  setenv("SOCKET_SERVER_PORT_CAM", "6004", 1);
  setenv("SOCKET_SERVER_PORT_DATA", "6005", 1);

  setenv("SOCKET_SERVER_NAME_CAM", "localhost", 1);
  setenv("SOCKET_SERVER_NAME_DATA", "localhost", 1);
#endif

#if 0
  setenv(
      "KERNEL_CUDA_CUBIN",
      "c:\\Users\\jar091\\projects\\work\\blender_client_build\\cycles_cuda\\kernel_sm_75.cubin",
      1);

  setenv("KERNEL_CUDA_STAT_CUBIN",
         "c:\\Users\\jar091\\projects\\work\\blender_client_build\\cycles_cuda\\kernel_sm_75_stat.cubin",
         1);

  setenv("CLIENT_FILE_KERNEL_GLOBAL",
         "d:\\data\\jar091\\models\\new.kg",
         1);

  //setenv("DEBUG_SAMPLES", "1", 1);
  //setenv("DEBUG_RES_W", "1920", 1);
  //("DEBUG_RES_H", "1080", 1);
  //setenv("CUDA_VISIBLE_DEVICES", "0", 1);

  //setenv("DEBUG_REPEAT_TIME", "10", 1);

  setenv("SOCKET_SERVER_PORT_CAM", "6004", 1);
  setenv("SOCKET_SERVER_PORT_DATA", "6005", 1);

  setenv("SOCKET_SERVER_NAME_CAM", "localhost", 1);
  setenv("SOCKET_SERVER_NAME_DATA", "localhost", 1);
#endif
  /////////////////////////

  client_kernel_struct data;
  int world_rank = 0;
  int world_size = 1;

#if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET) || defined(WITH_CLIENT_MPI_FILE)
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  //MPI_Init(&argc, &argv);

  // Get the number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  // world_size = 2;

  // Get the rank of the process
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  // world_rank = 0;

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  if (world_rank < world_size) {
    // Print off a hello world message
    const char *cfkg = getenv("CLIENT_FILE_KERNEL_GLOBAL");
    printf(
        "Start from processor %s, rank %d"
        " out of %d processors, CLIENT_FILE_KERNEL_GLOBAL: %s\n",
        processor_name,
        world_rank,
        world_size,
        (cfkg != NULL) ? cfkg : "");
    fflush(0);

    // mpi_print_memory(world_rank);
  }

#  if 0  // defined(_WIN32) //&& defined(_DEBUG)
  
    if (world_rank == 0) {
      int stop = 0;
      printf("attach\n");
      fflush(0);
      while (stop) {
        Sleep(1000);
      }
    }
  
    MPI_Barrier(MPI_COMM_WORLD);
#  endif

#endif

{

        if (world_rank == 0)
        {
          int i=1;
          while (i == 0)
            sleep(5);
        }
}

#ifdef WITH_CLIENT_NCCL_SOCKET
  ncclUniqueId ncclId;  // = (ncclUniqueId *)data.nccl_id;
  ncclResult_t res;
  ncclComm_t comm_data;
  if (world_rank == 0) {
    res = ncclGetUniqueId(&ncclId);
    if (res != ncclSuccess) {
      printf("ncclGetUniqueId != ncclSuccess\n");
      exit(-1);
    }
  }

  MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  int gpuArray = 0;
  // res = ncclCommInitAll(&comm_data, 1, &gpuArray);
  ncclGroupStart();
  cudaSetDevice(0);
  res = ncclCommInitRank(&comm_data, world_size, ncclId, world_rank);
  ncclGroupEnd();

  if (comm_data == 0 || res != ncclSuccess) {
    printf("ncclCommInitAll != ncclSuccess\n");
    exit(-1);
  }

  printf("gpuArray: %d\n", gpuArray);

  data.comm_data = (char *)comm_data;
#endif

#ifdef WITH_NVML
  nvmlDevice_t deviceHandle;
  unsigned long long energyStart, energyEnd;

  nvmlInit();
  nvmlDeviceGetHandleByIndex(0, &deviceHandle);
#endif

#if defined(WITH_CLIENT_CUDA) || defined(WITH_CLIENT_HIP) || defined(WITH_CLIENT_OPTIX) || defined(WITH_CLIENT_MPI_FILE_1)//|| defined(WITH_CLIENT_MPI_SOCKET)

  // #if defined(WITH_CLIENT_NCCL_SOCKET) || defined(WITH_CLIENT_MPI_REDUCE)
  //         if (world_rank == 0)
  //           cyclesphi::client::set_device(0, 0,  0,  world_size);
  //         else
  //           cyclesphi::client::set_device(-1, world_rank,  world_rank,  world_size);
  // #else

#  if defined(WITH_CLIENT_CUDA_GPU_TILES) && (defined(WITH_CLIENT_NCCL_SOCKET) || defined(WITH_CLIENT_MPI_REDUCE)) // WITH_CLIENT_GPU_PER_MPI_RANK
  cyclesphi::client::set_device(world_rank, world_rank, world_size);
#  else
  cyclesphi::client::set_device(-1, world_rank, world_size);
#  endif

  //#endif

#endif

  data.world_rank = world_rank;
  data.world_size = world_size;

  double t_start = omp_get_wtime();
  double t_start0 = omp_get_wtime();

  while (true) {

    if (data.world_size != 0)
      cyclesphi::client::read_data_kernelglobal(&data, sizeof(client_kernel_struct));

    data.world_rank = world_rank;
    data.world_size = world_size;

#ifdef WITH_CLIENT_NCCL_SOCKET
    data.comm_data = (char *)comm_data;
#endif

    if (CLIENT_TAG_CYCLES_start <= data.client_tag && data.client_tag <= CLIENT_TAG_CYCLES_end) {
#ifdef WITH_NVML
      if (data.client_tag == CLIENT_TAG_CYCLES_path_trace)
        nvmlDeviceGetTotalEnergyConsumption(deviceHandle, &energyStart);
#endif

      cyclesphi::client::render(data);

#ifdef WITH_NVML
      if (data.client_tag == CLIENT_TAG_CYCLES_path_trace)
        nvmlDeviceGetTotalEnergyConsumption(deviceHandle, &energyEnd);
#endif
# ifndef WITH_CLIENT_MPI_FILE_1
      if (data.client_tag == CLIENT_TAG_CYCLES_path_trace) {
        cyclesphi::client::save_buffer(data);
      }
# endif
      if (cyclesphi::client::break_loop(data))
        break;
    }
    else {
      printf("BAD TAG: %d, %d\n", data.world_rank, data.client_tag);
      break;
    }
#if 0
     printf("TAG: %d, TIME: %f, %f\n",
           data.client_tag,
           omp_get_wtime() - t_start,
           omp_get_wtime() - t_start0);
#endif
    t_start = omp_get_wtime();
  }

#ifdef WITH_NVML
  printf("Energy consumption: %llu mJ\n", energyEnd - energyStart);
  fflush(0);
  nvmlShutdown();
#endif

#ifdef WITH_CLIENT_NCCL_SOCKET
  ncclCommDestroy(comm_data);
#endif

#if defined(WITH_CLIENT_MPI) || defined(WITH_CLIENT_MPI_SOCKET) || defined(WITH_CLIENT_MPI_FILE)

#  if defined(WITH_CLIENT_MPI_SOCKET)
  if (world_rank == 0) {
    cyclesphi::client::close_kernelglobal();
    cyclesphi::client::write_data_kernelglobal(&data, sizeof(client_kernel_struct));
  }

#  endif

#  if defined(WITH_CLIENT_MPI_FILE) || defined(WITH_CLIENT_FILE)
  cyclesphi::client::close_kernelglobal();
#  endif

  MPI_Barrier(MPI_COMM_WORLD);

  // MPI_Barrier(MPI_COMM_WORLD);

  // Finalize the MPI environment.
  MPI_Finalize();
#endif

  if (world_rank < world_size) {
    printf(
        "End from processor, rank %d"
        " out of %d processors, total time: %f\n",
        world_rank,
        world_size,
        // omp_get_wtime() - t_start,
        omp_get_wtime() - t_start0);

    fflush(0);
  }

  return 0;
}
