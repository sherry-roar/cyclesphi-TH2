#!/bin/bash

# cd /mnt/projX/open-XX-XX/project-cyclesphi

# modules for Karolina cluster
# ml Mesa/22.2.4-GCCcore-12.2.0
# ml CUDA/12.0.0
# ml GCC/12.2.0
# ml intel/2022b

ROOT=$PWD
CACHE_DIR=$ROOT/data

export CLIENT_FILE_KERNEL_GLOBAL=${CACHE_DIR}/cache.kg
export CLIENT_FILE_CYCLES_BUFFER=${CACHE_DIR}/cache.bf
export CLIENT_FILE_CYCLES_BMP=${CACHE_DIR}/cache.bmp
export DEBUG_SAMPLES=1
export DEBUG_STEP_SAMPLES=1

mpirun --allow-run-as-root -np 4 $ROOT/install/cyclesphi33_cpu/bin/blender_client
# mpirun -env DEBUG_SAMPLES=1024 -env DEBUG_STEP_SAMPLES=1 $ROOT/install/cyclesphi33_gpu_unimem_rr/bin/blender_client
# i have set void unmap(char *mem_buffer)
# {
#   if (munmap(mem_buffer, g_client_kernel_global_size)) {  ## origin is !munmap, now cannot catch the errno 34
#     perror("Could not unmap file");
#     exit(1);
#   }